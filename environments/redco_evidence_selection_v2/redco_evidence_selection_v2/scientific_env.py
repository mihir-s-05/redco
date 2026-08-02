"""Single-episode live Stage-D reconstruction and scientific replay environment."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import verifiers.v1 as vf

from redco.analysis.stage_d_dynamic_taint import DynamicCausalTaintTracker
from redco.analysis.stage_d_exact_action import BehaviorAction
from redco.analysis.stage_d_receipt_ledger import (
    ExecutionAttempt,
    StageDReceiptLedger,
)
from redco.analysis.stage_d_replay_controller import (
    SeedOracleLike,
    StageDReconstructionQAController,
    StageDReplayCallController,
)
from redco.analysis.stage_d_runtime_isolation import (
    StageDIsolatedRuntimeContract,
    build_pre_action_runtime_snapshot,
)
from redco.analysis.stage_d_scientific_branch_group import OutcomeKind
from redco.analysis.stage_d_source_contracts import SourceRollout
from redco.analysis.stage_d_spawn_provenance import CausalProvenanceGraph, PolicyEventAddress
from redco.contracts import ActualEvaluationCost, canonical_json
from redco.integrations.verifiers_trace_v2 import RecordedRLMProvenanceV2
from redco_evidence_selection_v2.source_env import (
    StageDSourceData,
    StageDSourceEnvConfig,
    StageDSourceTask,
    _resolved_agent_sampling_law_sha256,
    _resolved_train_client_sha256,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class StageDScientificEpisodeBinding:
    """Already-verified immutable inputs for one QA or one scientific arm replay."""

    mode: Literal["qa", "execution"]
    task: StageDSourceTask
    source: SourceRollout
    source_records: tuple[RecordedRLMProvenanceV2, ...]
    source_graph: CausalProvenanceGraph
    target: PolicyEventAddress
    expected_runtime_snapshot: bytes
    expected_terminal_reply: object
    ledger: StageDReceiptLedger
    candidate_action: BehaviorAction | None = None
    seed_oracle: SeedOracleLike | None = None
    execution_attempt: ExecutionAttempt | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"qa", "execution"}:
            raise ValueError("scientific episode mode is invalid")
        if type(self.task) is not StageDSourceTask or type(self.source) is not SourceRollout:
            raise ValueError("scientific episode requires verified task and source types")
        if not self.source_records or type(self.target) is not PolicyEventAddress:
            raise ValueError("scientific episode lacks its source topology or target")
        if type(self.expected_runtime_snapshot) is not bytes or not self.expected_runtime_snapshot:
            raise ValueError("scientific episode lacks its frozen runtime snapshot")
        if self.source.evidence_class != "live" or not self.source.branch_eligible:
            raise ValueError("scientific episode requires one eligible verified live source")
        source_addresses = {decision.event_address for decision in self.source.decisions}
        record_addresses = {record.scientific_address for record in self.source_records}
        if (
            any(record.trace_id != self.source.rollout_id for record in self.source_records)
            or source_addresses != record_addresses
            or self.target not in source_addresses
        ):
            raise ValueError("scientific episode source topology is not bijective")
        execution_values = (
            self.candidate_action,
            self.seed_oracle,
            self.execution_attempt,
        )
        if (self.mode == "execution") != all(value is not None for value in execution_values):
            raise ValueError("execution binding fields must be supplied together")

    @property
    def source_actions(self) -> dict[PolicyEventAddress, BehaviorAction]:
        return {decision.event_address: decision.action for decision in self.source.decisions}

    @property
    def recorded_action(self) -> BehaviorAction:
        matching = [
            decision.action
            for decision in self.source.decisions
            if decision.event_address == self.target
        ]
        if len(matching) != 1:
            raise ValueError("scientific target does not biject one source action")
        return matching[0]

    @property
    def episode_identity(self) -> str:
        attempt = self.execution_attempt
        payload = {
            "schema_version": 1,
            "domain": "redco-stage-d-scientific-episode-identity-v1",
            "mode": self.mode,
            "group_id": self.source.group_id,
            "target_id": self._target_id(),
            "arm_id": None if attempt is None else attempt.arm_id,
            "continuation_replicate": (
                None if attempt is None else attempt.continuation_replicate
            ),
            "attempt_ordinal": None if attempt is None else attempt.attempt_ordinal,
        }
        return _sha256(canonical_json(payload))

    def _target_id(self) -> str:
        matching = [
            decision.target_id
            for decision in self.source.decisions
            if decision.event_address == self.target
        ]
        if len(matching) != 1 or not matching[0]:
            raise ValueError("scientific target lacks one target ID")
        return matching[0]


class _OneTaskset:
    def __init__(self, task: StageDSourceTask) -> None:
        self._task = task

    def load(self) -> list[StageDSourceTask]:
        return [self._task]

    def select(self, num_tasks: int, shuffle: bool) -> list[StageDSourceTask]:
        del shuffle
        if num_tasks not in {-1, 1}:
            raise ValueError("scientific replay requires exactly one task")
        return self.load()

    def tool_servers(self) -> list[object]:
        return []


class StageDScientificReplayEnv(vf.Env[StageDSourceEnvConfig]):
    """Run one isolated replay with shared sampling and response control."""

    def __init__(
        self,
        config: StageDSourceEnvConfig,
        *,
        binding: StageDScientificEpisodeBinding,
    ) -> None:
        super().__init__(config)
        self._validate_policy_identity(config, binding)
        self.taskset = _OneTaskset(binding.task)  # type: ignore[assignment]
        self._task_cls = StageDSourceTask
        self._binding = binding
        self._controllers: dict[
            str, StageDReconstructionQAController | StageDReplayCallController
        ] = {}
        self._runtime_verified: set[str] = set()
        self._terminal: set[str] = set()
        self._run_lock = asyncio.Lock()
        self.receipt: bytes | None = None

    @staticmethod
    def _validate_policy_identity(
        config: StageDSourceEnvConfig,
        binding: StageDScientificEpisodeBinding,
    ) -> None:
        manifests = (
            (
                config.base_model_manifest_path,
                config.base_model_manifest_sha256,
                "base model",
            ),
            (
                config.tokenizer_manifest_path,
                config.tokenizer_manifest_sha256,
                "tokenizer",
            ),
            (
                config.renderer_manifest_path,
                config.renderer_manifest_sha256,
                "renderer",
            ),
            (
                config.sampler_conformance_manifest_path,
                config.sampler_conformance_manifest_sha256,
                "sampler conformance",
            ),
        )
        for path, expected, name in manifests:
            if _sha256(path.read_bytes()) != expected:
                raise ValueError(f"scientific replay {name} manifest bytes changed")
        if config.adapter_manifest_path is not None:
            assert config.adapter_manifest_sha256 is not None
            if _sha256(config.adapter_manifest_path.read_bytes()) != (
                config.adapter_manifest_sha256
            ):
                raise ValueError("scientific replay adapter manifest bytes changed")
        expected_identity = (
            config.checkpoint_id,
            config.base_model_manifest_sha256,
            config.adapter_manifest_sha256,
            config.tokenizer_manifest_sha256,
            config.renderer_manifest_sha256,
            config.sampler_conformance_manifest_sha256,
        )
        if binding.task.data.policy_checkpoint_id != config.checkpoint_id:
            raise ValueError("scientific replay task checkpoint differs from its config")
        for action in binding.source_actions.values():
            key = action.key
            observed_identity = (
                key.checkpoint_id,
                key.base_model_manifest_sha256,
                key.adapter_manifest_sha256,
                key.tokenizer_manifest_sha256,
                key.renderer_manifest_sha256,
                key.sampler_conformance_manifest_sha256,
            )
            if observed_identity != expected_identity:
                raise ValueError("scientific replay source policy identity changed")

    async def setup(self, agents: vf.Agents) -> None:
        agents.agent.trainable = False

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        await agents.agent.run(task)

    def prepared_call_observer(
        self,
        task: vf.Task,
        trace: vf.Trace,
        agent_config: vf.AgentConfig,
        client: vf.Client,
    ) -> StageDReconstructionQAController | StageDReplayCallController:
        from verifiers.v1.clients.train import TrainClient

        if trace.id in self._controllers or trace.id in self._terminal:
            raise ValueError("scientific trace registered more than once")
        if task is not self._binding.task or task.tool_servers():
            raise ValueError("scientific replay task or tool-server roster changed")
        if not isinstance(task.data, StageDSourceData) or not isinstance(client, TrainClient):
            raise TypeError("scientific replay requires the frozen task and TrainClient")
        if client.openai.max_retries != 0 or agent_config.retries.max_retries != 0:
            raise ValueError("scientific replay forbids transport retries")
        if agent_config.model != self.config.checkpoint_id or agent_config.sampling is None:
            raise ValueError("scientific replay agent identity changed")
        if (
            _resolved_agent_sampling_law_sha256(agent_config.sampling)
            != self.config.resolved_agent_sampling_law_sha256
            or _resolved_train_client_sha256(client)
            != self.config.resolved_train_client_sha256
        ):
            raise ValueError("scientific replay sampling law or client changed")
        if self._runtime_snapshot(task, agent_config) != self._binding.expected_runtime_snapshot:
            raise ValueError("scientific replay runtime/workspace snapshot changed")
        self._runtime_verified.add(trace.id)
        if self._binding.mode == "qa":
            controller: StageDReconstructionQAController | StageDReplayCallController = (
                StageDReconstructionQAController(
                    source_records=self._binding.source_records,
                    source_actions=self._binding.source_actions,
                    pre_forward_guard=lambda: self._require_runtime_preflight(trace),
                )
            )
        else:
            assert self._binding.candidate_action is not None
            assert self._binding.seed_oracle is not None
            assert self._binding.execution_attempt is not None
            controller = StageDReplayCallController(
                tracker=DynamicCausalTaintTracker(
                    target=self._binding.target,
                    source_records=self._binding.source_records,
                    source_graph=self._binding.source_graph,
                ),
                source_actions=self._binding.source_actions,
                target=self._binding.target,
                candidate_action=self._binding.candidate_action,
                seed_oracle=self._binding.seed_oracle,
                ledger=self._binding.ledger,
                attempt=self._binding.execution_attempt,
                pre_forward_guard=lambda: self._require_runtime_preflight(trace),
            )
        self._controllers[trace.id] = controller
        return controller

    def prepared_sampling_director(
        self,
        task: vf.Task,
        trace: vf.Trace,
        agent_config: vf.AgentConfig,
        client: vf.Client,
    ) -> StageDReconstructionQAController | StageDReplayCallController:
        del task, agent_config, client
        try:
            return self._controllers[trace.id]
        except KeyError as error:
            raise RuntimeError("sampling director was requested before its observer") from error

    async def run_episode(
        self,
        task: vf.Task,
        ctx: vf.ModelContext,
        **kwargs: Any,
    ) -> vf.Episode:
        async with self._run_lock:
            if self.receipt is not None:
                raise RuntimeError("scientific environment already completed its one episode")
            episode = await super().run_episode(task, ctx, **kwargs)
            if len(episode.traces) != 1:
                raise RuntimeError("scientific replay episode did not complete exactly once")
            trace = episode.traces[0]
            controller = self._controllers.get(trace.id)
            if controller is None or trace.id not in self._runtime_verified:
                raise RuntimeError("scientific replay bypassed its shared controller")
            allow_terminal_truncation = bool(
                self._binding.mode == "execution"
                and isinstance(controller, StageDReplayCallController)
                and _is_clean_terminal_truncation(episode, trace)
            )
            if isinstance(controller, StageDReplayCallController):
                controller.finalize(
                    allow_terminal_truncation=allow_terminal_truncation
                )
            else:
                controller.finalize()
            if self._binding.mode == "qa":
                if not episode.ok or not trace.ok or episode.errors or trace.errors:
                    raise RuntimeError("scientific replay QA did not complete cleanly")
                if trace.last_reply != self._binding.expected_terminal_reply:
                    raise RuntimeError("scientific replay changed the terminal reply")
                outcome_kind = None
            else:
                assert isinstance(controller, StageDReplayCallController)
                assert self._binding.candidate_action is not None
                outcome_kind = _classify_execution_outcome(
                    episode=episode,
                    trace=trace,
                    action=self._binding.candidate_action,
                    controller=controller,
                )
            score_evidence = canonical_json(
                {
                    "schema_version": 1,
                    "domain": "redco-stage-d-live-replay-score-v1",
                    "mode": self._binding.mode,
                    "source_sha256": self._binding.source.source_sha256,
                    "trace_reward": trace.reward,
                    "source_reward": self._binding.source.reward,
                    "outcome_kind": (
                        outcome_kind.value if outcome_kind is not None else None
                    ),
                    "stop_condition": trace.stop_condition,
                    "episode_ok": episode.ok,
                    "trace_ok": trace.ok,
                    "terminal_reply_exact": (
                        True if self._binding.mode == "qa" else None
                    ),
                    "runtime_snapshot_sha256": _sha256(
                        self._binding.expected_runtime_snapshot
                    ),
                }
            )
            evidence_sha256 = self._binding.ledger.put_evidence(score_evidence)
            if self._binding.mode == "qa":
                if trace.reward != self._binding.source.reward:
                    raise RuntimeError("reconstruction QA changed the terminal reward")
                self.receipt = self._binding.ledger.record_reconstruction_qa(
                    group_id=self._binding.source.group_id,
                    target_id=self._target_id(),
                    recorded_action=self._binding.recorded_action,
                    passed=True,
                    report_sha256=evidence_sha256,
                    actual_cost=ActualEvaluationCost(
                        generated_tokens=0,
                        judge_calls=0,
                        cpu_seconds=0.0,
                        gpu_seconds=0.0,
                        wall_seconds=trace.timing.generation.duration,
                        storage_bytes=0,
                    ),
                )
            else:
                assert isinstance(controller, StageDReplayCallController)
                assert self._binding.execution_attempt is not None
                assert outcome_kind is not None
                self.receipt = self._binding.ledger.finish_execution(
                    self._binding.execution_attempt,
                    outcome_kind=outcome_kind,
                    scored_reward=trace.reward,
                    scorer_evidence_sha256=evidence_sha256,
                    latency_seconds=trace.timing.generation.duration,
                    dollars=0.0,
                    judge_calls=0,
                    cpu_seconds=0.0,
                    gpu_seconds=0.0,
                    wall_seconds=trace.timing.generation.duration,
                    storage_bytes=0,
                )
            self._terminal.add(trace.id)
            return episode

    def _target_id(self) -> str:
        matches = [
            decision.target_id
            for decision in self._binding.source.decisions
            if decision.event_address == self._binding.target
        ]
        if len(matches) != 1 or not matches[0]:
            raise ValueError("scientific target lacks one target ID")
        return matches[0]

    def _runtime_snapshot(self, task: vf.Task, agent_config: vf.AgentConfig) -> bytes:
        image = self.config.taskset.isolated_runtime_image
        if image is None:
            raise ValueError("scientific replay requires isolated Docker")
        if not isinstance(task.data, StageDSourceData):
            raise TypeError("scientific runtime snapshot requires source task data")
        manifest_path = self.config.frozen_workspace_manifest_path
        manifest_sha256 = self.config.frozen_workspace_manifest_sha256
        assert manifest_path is not None and manifest_sha256 is not None
        manifest_bytes = manifest_path.read_bytes()
        if _sha256(manifest_bytes) != manifest_sha256:
            raise ValueError("frozen workspace manifest bytes changed")
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError("frozen workspace manifest must be an object")
        runtime = agent_config.harness.runtime.model_dump(
            mode="json",
            exclude_none=False,
        )
        return build_pre_action_runtime_snapshot(
            contract=StageDIsolatedRuntimeContract(image),
            runtime_config=runtime,
            task_data=task.data.model_dump(mode="json", exclude_none=False),
            task_config=task.config.model_dump(mode="json", exclude_none=False),
            paper=task.data.paper.encode("utf-8"),
            frozen_workspace_manifest=manifest,
        )

    @staticmethod
    def _require_runtime_preflight(trace: vf.Trace) -> None:
        preflight = trace.info.get("stage_d_isolated_runtime_preflight")
        if not isinstance(preflight, dict) or preflight.get("domain") != (
            "redco-stage-d-isolated-runtime-preflight-v1"
        ):
            raise RuntimeError("scientific replay bypassed its post-cut runtime preflight")


def _classify_execution_outcome(
    *,
    episode: vf.Episode,
    trace: vf.Trace,
    action: BehaviorAction,
    controller: StageDReplayCallController,
) -> OutcomeKind:
    """Map every post-injection workflow termination onto the frozen vocabulary."""
    if not controller.target_injection_delivered:
        raise RuntimeError("scientific target injection was not durably delivered")
    if action.parse_status == "malformed":
        return OutcomeKind.MALFORMED_ACTION
    if episode.errors or trace.errors or not episode.ok or not trace.ok:
        return OutcomeKind.RUNTIME_EXCEPTION
    if trace.stop_condition == "harness_timeout":
        return OutcomeKind.TIMEOUT
    resource_stops = {
        "max_turns",
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
        "context_length",
    }
    last_successful_call = next(
        (call for call in reversed(trace.calls) if call.error is None),
        None,
    )
    if trace.stop_condition in resource_stops or (
        last_successful_call is not None
        and last_successful_call.finish_reason == "length"
    ):
        return OutcomeKind.RESOURCE_LIMIT
    if trace.stop_condition != "agent_completed":
        raise RuntimeError("scientific execution ended with an unknown clean stop")
    return (
        OutcomeKind.SUCCESS
        if controller.logical_downstream_observed
        else OutcomeKind.TERMINAL_WITHOUT_DOWNSTREAM
    )


def _is_clean_terminal_truncation(episode: vf.Episode, trace: vf.Trace) -> bool:
    if episode.errors or trace.errors or not episode.ok or not trace.ok:
        return False
    if trace.stop_condition in {
        "harness_timeout",
        "max_turns",
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
        "context_length",
    }:
        return True
    last_successful_call = next(
        (call for call in reversed(trace.calls) if call.error is None),
        None,
    )
    return bool(
        last_successful_call is not None
        and last_successful_call.finish_reason == "length"
    )


async def run_bound_scientific_episode(
    *,
    binding: StageDScientificEpisodeBinding,
    env_config: StageDSourceEnvConfig,
    eval_config: Any,
) -> bytes:
    """Production entry point: run exactly one bound episode through Verifiers."""
    from verifiers.v1.cli.eval.runner import run_eval
    from verifiers.v1.configs.eval import EvalConfig

    if not isinstance(eval_config, EvalConfig):
        raise TypeError("scientific episode requires a validated EvalConfig")
    if eval_config.num_tasks != 1 or eval_config.num_rollouts != 1:
        raise ValueError("scientific episode EvalConfig must contain exactly one rollout")
    if eval_config.max_concurrent != 1 or eval_config.resume is not None:
        raise ValueError("scientific episode forbids concurrency and resume")
    env = StageDScientificReplayEnv(env_config, binding=binding)
    episodes = await run_eval(env, eval_config)
    if len(episodes) != 1 or env.receipt is None:
        raise RuntimeError("scientific episode did not produce one durable receipt")
    return env.receipt


__all__ = [
    "StageDScientificEpisodeBinding",
    "StageDScientificReplayEnv",
    "run_bound_scientific_episode",
]
