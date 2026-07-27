from __future__ import annotations

import asyncio
import re
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

from redco.env.tasks.credit_probes import credit_probe_by_name, standard_credit_probes

ACTION_PATTERN = re.compile(r"<action>\s*([^<\r\n]+?)\s*</action>", re.IGNORECASE)
ROUTE_PATTERN = re.compile(r"<route>\s*([^<\r\n]+?)\s*</route>", re.IGNORECASE)
ROLE_ORDER = ("original", "alternative_1", "alternative_2", "alternative_3")
ROUTES = ("alpha", "beta", "gamma", "delta")
ROUTE_REWARD = {"alpha": -0.25, "beta": 0.0, "gamma": 0.25, "delta": 0.5}


class RedcoCreditData(vf.TaskData):
    probe_name: str
    actions: tuple[str, ...]
    exogenous_seed: int
    context_route: str | None = None


def parse_action(reply: str | None, actions: tuple[str, ...]) -> str | None:
    if reply is None:
        return None
    matches = ACTION_PATTERN.findall(reply)
    if not matches:
        return None
    action = matches[-1].strip()
    return action if action in actions else None


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
        probe = credit_probe_by_name(self.data.probe_name)
        action = parse_action(trace.last_reply, self.data.actions)
        return probe.replay_reward(
            action,
            self.data.exogenous_seed,
            mode="full_suffix",
        ) + ROUTE_REWARD.get(self.data.context_route or "", -0.5)

    @vf.metric
    async def valid_action(self, trace: vf.Trace) -> float:
        return float(parse_action(trace.last_reply, self.data.actions) is not None)


class RedcoCreditTasksetConfig(vf.TasksetConfig):
    repeats_per_probe: int = Field(16, ge=1)
    exogenous_seed_offset: int = Field(0, ge=0)


class RedcoCreditTaskset(
    vf.Taskset[RedcoSeedTask, RedcoCreditTasksetConfig]
):
    def load(self) -> list[RedcoSeedTask]:
        tasks: list[RedcoSeedTask] = []
        index = 0
        for repeat in range(self.config.repeats_per_probe):
            for probe in standard_credit_probes():
                prompt = (
                    "Choose a routing label for a later sub-call. The route adds a "
                    "deterministic background contribution to the final reward, "
                    "while the sub-call action supplies the targeted contribution. "
                    "Your final reply must end with exactly "
                    f"<route>VALUE</route>, replacing VALUE with one of: "
                    f"{', '.join(ROUTES)}."
                )
                data = RedcoCreditData(
                    idx=index,
                    name=f"{probe.name}-{repeat:04d}",
                    prompt=prompt,
                    probe_name=probe.name,
                    actions=probe.actions,
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
    branching_enabled: bool = True
    replay_mode: Literal["full_suffix", "sliced"] = "sliced"


class RedcoCreditEnv(vf.Env[RedcoCreditEnvConfig]):
    async def setup(self, agents: vf.Agents) -> None:
        for role in ("context", *ROLE_ORDER):
            getattr(agents, role).trainable = True

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task, RedcoSeedTask):
            raise TypeError("ReDCO environment requires a RedcoSeedTask")
        context = await agents.context.run(task)
        data = task.data
        context_route = parse_route(context.last_reply)
        branch_prompt = (
            "You are the trainable depth-one sub-call in a deterministic "
            "credit-assignment probe. The already-sampled root context is quoted "
            f"below; do not modify it:\n<context>{context.last_reply}</context>\n"
            "Choose one allowed action. Your final reply must end with exactly "
            "<action>VALUE</action>, replacing VALUE with one of: "
            f"{', '.join(data.actions)}. Invalid values are executed and score "
            "zero; they are never resampled."
        )
        branch_task = RedcoCreditTask(
            data.model_copy(
                update={"prompt": branch_prompt, "context_route": context_route}
            ),
            task.config,
        )
        if not self.config.branching_enabled:
            await agents.original.run(branch_task)
            return
        async with asyncio.TaskGroup() as task_group:
            for role in ROLE_ORDER:
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
            return
        if len(episode.traces) != 5:
            raise ValueError("clean Stage C requires context plus four branch traces")
        by_role = {trace.agent_name: trace for trace in episode.traces}
        if set(by_role) != {"context", *ROLE_ORDER}:
            raise ValueError("episode is missing a declared ReDCO branch role")

        target_node_id: str | None = None
        for branch_index, role in enumerate(ROLE_ORDER):
            trace = by_role[role]
            data = trace.task.data
            if not isinstance(data, RedcoCreditData):
                raise TypeError("ReDCO environment received an incompatible task")
            probe = credit_probe_by_name(data.probe_name)
            action = parse_action(trace.last_reply, data.actions)
            full_reward = probe.replay_reward(
                action,
                data.exogenous_seed,
                mode="full_suffix",
            ) + ROUTE_REWARD.get(data.context_route or "", -0.5)
            sliced_reward = probe.replay_reward(
                action,
                data.exogenous_seed,
                mode="sliced",
            ) + ROUTE_REWARD.get(data.context_route or "", -0.5)
            equivalent = full_reward == sliced_reward
            if not equivalent:
                raise RuntimeError("sliced and full-suffix replay disagree in-loop")
            expected_reward = (
                full_reward
                if self.config.replay_mode == "full_suffix"
                else sliced_reward
            )
            if trace.rewards.get("deterministic_reward") != expected_reward:
                raise RuntimeError("task reward differs from replayed branch reward")
            target_node_id = f"{data.name}:depth-one-subcall"
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
                "parsed_action": action,
                "context_route": data.context_route,
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

        context = by_role["context"]
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
