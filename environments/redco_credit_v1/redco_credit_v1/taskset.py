from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import replace
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field, field_validator

from redco.contracts import SeedNamespace
from redco.env.tasks.credit_probes import credit_probe_by_name, standard_credit_probes
from redco.integrations.stage_c_prompts import (
    ROUTES,
    stage_c_branch_prompt,
    stage_c_root_prompt,
)
from redco.integrations.stage_c_roles import stage_c_branch_roles

ACTION_PATTERN = re.compile(r"<action>\s*([^<\r\n]+?)\s*</action>", re.IGNORECASE)
SELF_TAG_PATTERN = re.compile(
    r"<([A-Za-z0-9_.:-]+)>\s*</\1>",
    re.IGNORECASE,
)
ROUTE_PATTERN = re.compile(r"<route>\s*([^<\r\n]+?)\s*</route>", re.IGNORECASE)
ROUTE_REWARD = {"alpha": -0.25, "beta": 0.0, "gamma": 0.25, "delta": 0.5}
BRANCH_SEED_MASTER = "redco-stage-c-branch-v1"
CONTEXT_SEED_MASTER = "redco-stage-c-context-v1"
CONFUSION_PROBES = {
    "confusion_irrelevant",
    "confusion_redundant",
    "confusion_lucky",
}


class RedcoCreditData(vf.TaskData):
    probe_name: str
    actions: tuple[str, ...]
    action_map: tuple[tuple[str, str], ...]
    exogenous_seed: int
    context_route: str | None = None
    episode_luck: int = 0


def parse_action(reply: str | None, actions: tuple[str, ...]) -> str | None:
    if reply is None:
        return None
    exact = reply.strip()
    if exact in actions:
        return exact
    candidates = [match.strip() for match in ACTION_PATTERN.findall(reply)]
    candidates.extend(match.strip() for match in SELF_TAG_PATTERN.findall(reply))
    if not candidates or any(candidate not in actions for candidate in candidates):
        return None
    unique = set(candidates)
    return next(iter(unique)) if len(unique) == 1 else None


def branch_sampling(
    base: vf.SamplingConfig,
    *,
    seed: int,
    cache_salt_suffix: str,
    temperature: float,
) -> vf.SamplingConfig:
    """Return an exact one-token behavior-policy config for one branch role."""
    raw = base.model_dump(exclude_none=True)
    raw.pop("allowed_token_ids", None)
    extra_body = dict(raw.pop("extra_body", None) or {})
    parent_salt = extra_body.get("cache_salt")
    extra_body["cache_salt"] = (
        f"{parent_salt}:redco:{cache_salt_suffix}"
        if parent_salt is not None
        else f"redco:{cache_salt_suffix}"
    )
    raw["seed"] = seed
    raw["temperature"] = temperature
    raw["max_tokens"] = 1
    raw["extra_body"] = extra_body
    return vf.SamplingConfig(**raw)


def context_sampling(
    base: vf.SamplingConfig,
    *,
    seed: int,
    cache_salt_suffix: str,
    temperature: float,
) -> vf.SamplingConfig:
    """Freeze root randomness without changing its completion-length contract."""
    raw = base.model_dump(exclude_none=True)
    extra_body = dict(raw.pop("extra_body", None) or {})
    parent_salt = extra_body.get("cache_salt")
    extra_body["cache_salt"] = (
        f"{parent_salt}:redco:{cache_salt_suffix}"
        if parent_salt is not None
        else f"redco:{cache_salt_suffix}"
    )
    raw["seed"] = seed
    raw["temperature"] = temperature
    raw["extra_body"] = extra_body
    return vf.SamplingConfig(**raw)


def constrain_root_routes(sampling: vf.SamplingConfig) -> vf.SamplingConfig:
    """Define the root policy over exactly the four canonical route strings."""
    raw = sampling.model_dump(exclude_none=True)
    extra_body = dict(raw.pop("extra_body", None) or {})
    extra_body["structured_outputs"] = {
        "choice": [f"<route>{route}</route>" for route in ROUTES]
    }
    raw["extra_body"] = extra_body
    return vf.SamplingConfig(**raw)


def _branch_seed_namespace(context_id: str, target_node_id: str) -> SeedNamespace:
    return SeedNamespace(
        master_seed=BRANCH_SEED_MASTER,
        rollout_id=context_id,
        target_id=target_node_id,
        replicate=1,
    )


def _context_seed(task_name: str, episode_address: str) -> int:
    if not episode_address:
        raise ValueError("context sampling requires a stable episode address")
    digest = hashlib.sha256(
        f"{CONTEXT_SEED_MASTER}:{task_name}:{episode_address}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def confusion_reward(
    probe_name: str,
    *,
    canonical_action: str | None,
    context_route: str | None,
    episode_luck: int,
) -> float:
    """Compose a fixed-world reward for one live credit-confusion branch."""
    if episode_luck not in {-1, 1}:
        raise ValueError("episode_luck must be -1 or 1")
    route_reward = ROUTE_REWARD.get(context_route or "", -0.5)
    target_success = canonical_action == "5"
    if probe_name == "confusion_irrelevant":
        return route_reward
    if probe_name == "confusion_redundant":
        context_success = context_route == "delta"
        return float(context_success or target_success)
    if probe_name == "confusion_lucky":
        return float(target_success) + route_reward + float(episode_luck)
    raise ValueError(f"not a credit-confusion probe: {probe_name}")


def replay_reward(
    data: RedcoCreditData,
    canonical_action: str | None,
    *,
    mode: Literal["full_suffix", "sliced"],
) -> float:
    probe = credit_probe_by_name(data.probe_name)
    if data.probe_name in CONFUSION_PROBES:
        return confusion_reward(
            data.probe_name,
            canonical_action=canonical_action,
            context_route=data.context_route,
            episode_luck=data.episode_luck,
        )
    return probe.replay_reward(
        canonical_action,
        data.exogenous_seed,
        mode=mode,
    ) + ROUTE_REWARD.get(data.context_route or "", -0.5)


def parse_route(reply: str | None) -> str | None:
    if reply is None:
        return None
    matches = ROUTE_PATTERN.findall(reply)
    if not matches:
        return None
    route = matches[-1].strip()
    return route if route in ROUTES else None


class RedcoSeedTask(vf.Task[RedcoCreditData]):
    """The untargeted root decision, deliberately irrelevant to probe reward."""


class RedcoCreditTask(vf.Task[RedcoCreditData]):
    @vf.reward(weight=1.0)
    async def deterministic_reward(self, trace: vf.Trace) -> float:
        displayed_action = parse_action(trace.last_reply, self.data.actions)
        action = dict(self.data.action_map).get(displayed_action)
        return replay_reward(
            self.data,
            action,
            mode="full_suffix",
        )

    @vf.metric
    async def valid_action(self, trace: vf.Trace) -> float:
        return float(parse_action(trace.last_reply, self.data.actions) is not None)

    @vf.metric
    async def target_success(self, trace: vf.Trace) -> float:
        displayed_action = parse_action(trace.last_reply, self.data.actions)
        canonical_action = (
            dict(self.data.action_map).get(displayed_action)
            if displayed_action is not None
            else None
        )
        return float(canonical_action == "5")


class RedcoCreditTasksetConfig(vf.TasksetConfig):
    repeats_per_probe: int = Field(16, ge=1)
    exogenous_seed_offset: int = Field(0, ge=0)
    probe_names: tuple[str, ...] | None = None

    @field_validator("probe_names")
    @classmethod
    def validate_probe_names(cls, probe_names: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if probe_names is None:
            return None
        if not probe_names:
            raise ValueError("probe_names must be non-empty when provided")
        if len(set(probe_names)) != len(probe_names):
            raise ValueError("probe_names must be unique")
        for probe_name in probe_names:
            credit_probe_by_name(probe_name)
        return probe_names


class RedcoCreditTaskset(vf.Taskset[RedcoSeedTask, RedcoCreditTasksetConfig]):
    def load(self) -> list[RedcoSeedTask]:
        tasks: list[RedcoSeedTask] = []
        index = 0
        probes = (
            tuple(credit_probe_by_name(probe_name) for probe_name in self.config.probe_names)
            if self.config.probe_names is not None
            else standard_credit_probes()
        )
        for repeat in range(self.config.repeats_per_probe):
            for probe in probes:
                action_map = tuple(
                    (str(action_index), action) for action_index, action in enumerate(probe.actions)
                )
                prompt = stage_c_root_prompt()
                data = RedcoCreditData(
                    idx=index,
                    name=f"{probe.name}-{repeat:04d}",
                    prompt=prompt,
                    probe_name=probe.name,
                    actions=tuple(alias for alias, _ in action_map),
                    action_map=action_map,
                    exogenous_seed=self.config.exogenous_seed_offset + repeat,
                )
                tasks.append(RedcoSeedTask(data, self.config.task))
                index += 1
        return tasks


class RedcoCreditEnvConfig(vf.EnvConfig):
    context: vf.AgentConfig = vf.AgentConfig(max_turns=1)
    original: vf.AgentConfig = vf.AgentConfig(max_turns=1)
    alternative_1: vf.AgentConfig = vf.AgentConfig(max_turns=1)
    alternative_2: vf.AgentConfig = vf.AgentConfig(max_turns=1)
    alternative_3: vf.AgentConfig = vf.AgentConfig(max_turns=1)
    alternative_4: vf.AgentConfig = vf.AgentConfig(max_turns=1)
    alternative_5: vf.AgentConfig = vf.AgentConfig(max_turns=1)
    alternative_6: vf.AgentConfig = vf.AgentConfig(max_turns=1)
    alternative_7: vf.AgentConfig = vf.AgentConfig(max_turns=1)
    alternative_8: vf.AgentConfig = vf.AgentConfig(max_turns=1)
    alternative_9: vf.AgentConfig = vf.AgentConfig(max_turns=1)
    alternative_10: vf.AgentConfig = vf.AgentConfig(max_turns=1)
    branching_enabled: bool = True
    branch_group_size: int = Field(4, ge=2, le=11)
    replay_mode: Literal["full_suffix", "sliced"] = "sliced"
    branch_temperature: float = Field(1.0, ge=0, le=2.0)
    context_temperature: float = Field(0.7, ge=0, le=2.0)
    constrained_root_routes: bool = False
    forced_integration_smoke: bool = False


def _episode_index(episode_address: str) -> int:
    marker = ":episode:"
    if marker not in episode_address:
        if episode_address.startswith("episode:"):
            value = episode_address.removeprefix("episode:")
        else:
            raise ValueError("episode address is missing an episode index")
    else:
        value = episode_address.rsplit(marker, maxsplit=1)[1]
    if not value.isdigit():
        raise ValueError("episode address has an invalid episode index")
    return int(value)


def _forced_smoke_choices(episode_address: str) -> tuple[str, str]:
    """Cover every route/action plus all reward regions without sampling."""
    choices = (
        ("<route>gamma</route>", "5"),
        ("<route>delta</route>", "1"),
        ("<route>gamma</route>", "0"),
        ("<route>alpha</route>", "2"),
        ("<route>beta</route>", "3"),
        ("<route>gamma</route>", "4"),
        ("<route>alpha</route>", "6"),
        ("<route>beta</route>", "7"),
    )
    return choices[_episode_index(episode_address) % len(choices)]


def _force_single_choice(
    sampling: vf.SamplingConfig,
    choice: str,
) -> vf.SamplingConfig:
    raw = sampling.model_dump(exclude_none=True)
    extra_body = dict(raw.pop("extra_body", None) or {})
    extra_body["structured_outputs"] = {"choice": [choice]}
    raw["extra_body"] = extra_body
    return vf.SamplingConfig(**raw)


class RedcoCreditEnv(vf.Env[RedcoCreditEnvConfig]):
    async def setup(self, agents: vf.Agents) -> None:
        roles = stage_c_branch_roles(self.config.branch_group_size)
        for role in ("context", *roles):
            getattr(agents, role).trainable = True

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task, RedcoSeedTask):
            raise TypeError("ReDCO environment requires a RedcoSeedTask")
        data = task.data
        parent_extra = agents.context.ctx.sampling.extra_body or {}
        episode_address = str(parent_extra.get("cache_salt", ""))
        forced_context: str | None = None
        forced_action: str | None = None
        if self.config.forced_integration_smoke:
            if self.config.branching_enabled:
                raise ValueError("forced integration smoke requires broadcast mode")
            forced_context, forced_action = _forced_smoke_choices(episode_address)
        context_config = context_sampling(
            agents.context.ctx.sampling,
            seed=_context_seed(data.name, episode_address),
            cache_salt_suffix=(f"{data.name}:{episode_address}:context"),
            temperature=self.config.context_temperature,
        )
        if self.config.constrained_root_routes:
            context_config = constrain_root_routes(context_config)
        if forced_context is not None:
            context_config = _force_single_choice(
                context_config,
                forced_context,
            )
        agents.context.ctx = replace(
            agents.context.ctx,
            sampling=context_config,
        )
        context = await agents.context.run(task)
        context.record_metric(
            "redco_context_token_budget_ok",
            float(context_config.max_tokens is not None and context_config.max_tokens >= 8),
        )
        context_route = parse_route(context.last_reply)
        episode_luck = 1 if data.exogenous_seed % 2 == 0 else -1
        target_node_id = f"{data.name}:depth-one-subcall"
        branch_prompt = stage_c_branch_prompt(
            context.last_reply,
            data.actions,
        )
        branch_task = RedcoCreditTask(
            data.model_copy(
                update={
                    "prompt": branch_prompt,
                    "context_route": context_route,
                    "episode_luck": episode_luck,
                }
            ),
            task.config,
        )
        seed_namespace = _branch_seed_namespace(context.id, target_node_id)
        roles = stage_c_branch_roles(self.config.branch_group_size)
        sampled_roles = roles if self.config.branching_enabled else ("original",)
        for branch_index, role in enumerate(sampled_roles):
            agent = getattr(agents, role)
            seed = seed_namespace.action_seed(branch_index + 1)
            branch_config = branch_sampling(
                agent.ctx.sampling,
                seed=seed,
                cache_salt_suffix=f"{context.id}:{role}",
                temperature=self.config.branch_temperature,
            )
            if forced_action is not None:
                branch_config = _force_single_choice(
                    branch_config,
                    forced_action,
                )
            agent.ctx = replace(
                agent.ctx,
                sampling=branch_config,
            )
        if not self.config.branching_enabled:
            await agents.original.run(branch_task)
            return
        async with asyncio.TaskGroup() as task_group:
            for role in roles:
                task_group.create_task(getattr(agents, role).run(branch_task))

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        del task
        if not self.config.branching_enabled:
            if len(episode.traces) != 2:
                raise ValueError("broadcast control requires context plus one trace")
            by_role = {trace.agent_name: trace for trace in episode.traces}
            if set(by_role) != {"context", "original"}:
                raise ValueError("broadcast control has an unexpected trace role")
            reward = by_role["original"].reward
            by_role["context"].record_reward("trajectory_reward", reward)
            for trace in episode.traces:
                trace.info["redco_control"] = {
                    "schema_version": 1,
                    "arm": "broadcast",
                    "branch_evaluations": 0,
                }
            by_role["context"].record_metric(
                "redco_valid_route",
                float(parse_route(by_role["context"].last_reply) is not None),
            )
            return
        roles = stage_c_branch_roles(self.config.branch_group_size)
        if len(episode.traces) != 1 + len(roles):
            raise ValueError("clean Stage C requires context plus the configured branch traces")
        by_role = {trace.agent_name: trace for trace in episode.traces}
        if set(by_role) != {"context", *roles}:
            raise ValueError("episode is missing a declared ReDCO branch role")

        target_node_id: str | None = None
        context = by_role["context"]
        for branch_index, role in enumerate(roles):
            trace = by_role[role]
            data = trace.task.data
            if not isinstance(data, RedcoCreditData):
                raise TypeError("ReDCO environment received an incompatible task")
            displayed_action = parse_action(trace.last_reply, data.actions)
            action = (
                dict(data.action_map).get(displayed_action)
                if displayed_action is not None
                else None
            )
            full_reward = replay_reward(
                data,
                action,
                mode="full_suffix",
            )
            sliced_reward = replay_reward(
                data,
                action,
                mode="sliced",
            )
            equivalent = full_reward == sliced_reward
            if not equivalent:
                raise RuntimeError("sliced and full-suffix replay disagree in-loop")
            expected_reward = (
                full_reward if self.config.replay_mode == "full_suffix" else sliced_reward
            )
            if trace.rewards.get("deterministic_reward") != expected_reward:
                raise RuntimeError("task reward differs from replayed branch reward")
            target_node_id = f"{data.name}:depth-one-subcall"
            expected_seed = _branch_seed_namespace(
                context.id,
                target_node_id,
            ).action_seed(branch_index + 1)
            sampling = trace.agent.sampling if trace.agent is not None else None
            sampling_payload = (
                sampling.model_dump(exclude_none=True) if sampling is not None else {}
            )
            if sampling_payload.get("seed") != expected_seed:
                raise RuntimeError("branch trace did not preserve its structural seed")
            if sampling_payload.get("max_tokens") != 1:
                raise RuntimeError("branch trace did not preserve its one-token action space")
            if sampling_payload.get("allowed_token_ids") is not None:
                raise RuntimeError("branch trace unexpectedly constrained token support")
            if sampling_payload.get("temperature") != self.config.branch_temperature:
                raise RuntimeError("branch trace did not preserve its behavior temperature")
            sampled_action_token_ids = [
                token_id
                for node in trace.nodes
                for token_id, trainable in zip(node.token_ids, node.mask, strict=True)
                if trainable
            ]
            if len(sampled_action_token_ids) != 1:
                raise RuntimeError("branch action must contain exactly one sampled token")
            extra_body = sampling_payload.get("extra_body")
            if not isinstance(extra_body, dict) or not isinstance(
                extra_body.get("cache_salt"), str
            ):
                raise RuntimeError("branch trace did not preserve its cache salt")
            trace.info["redco"] = {
                "schema_version": 1,
                "record_kind": "branch",
                "branch_index": branch_index,
                "action_source": "original" if branch_index == 0 else "sampled",
                "target_node_id": target_node_id,
                "selected_pre_action": True,
                "selection_features": {
                    "node_kind": "subcall_output",
                    "depth": 1,
                    "turn_index": 0,
                    "task_metadata": [["probe_name", data.probe_name]],
                    "predicted_replay_cost": 1.0,
                },
                "replay_mode": self.config.replay_mode,
                "replay_equivalent": equivalent,
                "full_suffix_reward": full_reward,
                "sliced_reward": sliced_reward,
                "parsed_action": displayed_action,
                "canonical_action": action,
                "action_token_id": sampled_action_token_ids[0],
                "action_seed": expected_seed,
                "branch_temperature": self.config.branch_temperature,
                "branch_cache_salt": extra_body["cache_salt"],
                "context_route": data.context_route,
                "episode_luck": data.episode_luck,
                "outer_weight": 1.0,
                "checkpoint_contract": "episode-policy-version",
                "logical_deployment_cost": {"model_calls": 1},
                "actual_evaluation_cost": {
                    "branch_evaluations": 1,
                    "environment_events": 2,
                    "equivalence_checks": 1,
                    "judge_calls": 0,
                },
                "logical_replay_work": {
                    "full_suffix_environment_events": 6,
                    "sliced_environment_events": 2,
                    "sliced_work_fraction": 1.0 / 3.0,
                    "note": "graph work model, not measured wall time",
                },
            }
            trace.record_metric("redco_replay_equivalent", float(equivalent))
            trace.record_metric("redco_valid_action", float(action is not None))
            trace.record_metric(
                "redco_environment_events",
                2.0,
            )

        original_reward = by_role["original"].reward
        context.record_reward("trajectory_reward", original_reward)
        context.info["redco"] = {
            "schema_version": 1,
            "record_kind": "context",
            "target_node_id": target_node_id,
            "selected_pre_action": True,
            "replay_mode": self.config.replay_mode,
            "replay_equivalent": True,
            "outer_weight": 1.0,
            "parsed_route": parse_route(context.last_reply),
        }
        context.record_metric(
            "redco_valid_route",
            float(parse_route(context.last_reply) is not None),
        )
