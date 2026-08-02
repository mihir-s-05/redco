"""Production assembly for one complete Stage-D scientific campaign."""

from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from redco.analysis.stage_d_dynamic_taint import build_source_causal_graph
from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.analysis.stage_d_receipt_ledger import StageDReceiptLedger
from redco.analysis.stage_d_replay_controller import preload_replay_runtime_types
from redco.analysis.stage_d_scientific_branch_group import (
    BranchGroupArtifact,
    BranchGroupSpec,
    CandidateSubmission,
    ZeroCallInfrastructureFailure,
)
from redco.analysis.stage_d_scientific_campaign import (
    ScientificCampaignResult,
    ScientificGroupRun,
    run_scientific_campaign,
)
from redco.analysis.stage_d_source_contracts import SourceRollout
from redco.contracts import canonical_json
from redco.integrations.verifiers_trace_v2 import extract_v2_rlm_provenance
from redco_evidence_selection_v2.scientific_env import (
    StageDScientificEpisodeBinding,
)
from redco_evidence_selection_v2.source_env import StageDSourceData, StageDSourceTask


@dataclass(frozen=True, slots=True)
class CandidateEngineResult:
    action: BehaviorAction
    response_evidence: bytes


class CandidateEngine(Protocol):
    def __call__(
        self,
        *,
        reference_key: ExactActionKey,
        action_seed: int,
        before_post: Callable[[bytes], None],
        after_post: Callable[[bytes], None],
    ) -> Awaitable[CandidateEngineResult]: ...


class BoundEpisodeRunner(Protocol):
    def __call__(self, binding: StageDScientificEpisodeBinding) -> Awaitable[bytes]: ...


@dataclass(frozen=True, slots=True)
class LiveScientificGroup:
    spec: BranchGroupSpec
    source: SourceRollout
    task: StageDSourceTask
    source_trace: Mapping[str, Any]
    expected_runtime_snapshot: bytes
    candidate_engine: CandidateEngine
    decode_action: Callable[[bytes], BehaviorAction]
    run_episode: BoundEpisodeRunner
    artifact_path: Path

    def __post_init__(self) -> None:
        if self.source.evidence_class != "live":
            raise ValueError("scientific driver accepts only verified live sources")
        if self.spec.commitment.rollout_id != self.source.rollout_id:
            raise ValueError("scientific group crosses source rollouts")
        if self.spec.commitment.group_id != self.source.group_id:
            raise ValueError("scientific group crosses source deployment groups")
        if type(self.task) is not StageDSourceTask:
            raise ValueError("scientific group requires the exact source task type")
        if not self.expected_runtime_snapshot:
            raise ValueError("scientific group lacks its frozen runtime snapshot")


def run_live_scientific_campaign(
    groups: Sequence[LiveScientificGroup],
    *,
    ledger: StageDReceiptLedger,
    event_loop: asyncio.AbstractEventLoop,
) -> ScientificCampaignResult:
    """Assemble real QA, candidate, arm, artifact, and campaign transactions."""
    if event_loop.is_closed() or event_loop.is_running():
        raise ValueError("scientific campaign requires one idle owned event loop")
    preload_replay_runtime_types()
    live_groups = tuple(groups)
    if not live_groups:
        raise ValueError("live scientific campaign requires target groups")
    _verify_artifact_roster_before_calls(live_groups, ledger)
    scientific_runs = tuple(
        _group_run(group, ledger=ledger, event_loop=event_loop)
        for group in live_groups
    )
    return run_scientific_campaign(
        scientific_runs,
        ledger=ledger,
        verifier=ledger,
    )


def _verify_artifact_roster_before_calls(
    groups: tuple[LiveScientificGroup, ...],
    ledger: StageDReceiptLedger,
) -> None:
    paths = tuple(group.artifact_path.resolve() for group in groups)
    if len(paths) != len(set(paths)):
        raise ValueError("scientific groups reuse an artifact path")
    parents = {path.parent for path in paths}
    if len(parents) != 1:
        raise ValueError("scientific artifacts must share one exact output root")
    expected_existing: set[Path] = set()
    for group, path in zip(groups, paths, strict=True):
        completed = ledger.completed_branch_artifact_sha256(
            group_id=group.spec.commitment.group_id,
            target_id=group.spec.commitment.target_id,
        )
        if completed is None:
            if path.exists():
                raise RuntimeError("uncommitted branch artifact exists before model calls")
            continue
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != completed:
            raise RuntimeError("completed branch artifact is absent or differs before model calls")
        expected_existing.add(path)
    root = next(iter(parents))
    if root.exists():
        actual = {path.resolve() for path in root.iterdir()}
        if actual != expected_existing:
            raise RuntimeError("scientific artifact output contains a stale or unknown member")


def _group_run(
    group: LiveScientificGroup,
    *,
    ledger: StageDReceiptLedger,
    event_loop: asyncio.AbstractEventLoop,
) -> ScientificGroupRun:
    records = extract_v2_rlm_provenance(dict(group.source_trace))
    graph = build_source_causal_graph(records)
    expected_reply = _terminal_reply(group.source_trace)
    target = group.spec.commitment.target_address

    def qa(_spec: BranchGroupSpec) -> bytes:
        binding = StageDScientificEpisodeBinding(
            mode="qa",
            task=group.task,
            source=group.source,
            source_records=records,
            source_graph=graph,
            target=target,
            expected_runtime_snapshot=group.expected_runtime_snapshot,
            expected_terminal_reply=expected_reply,
            ledger=ledger,
        )
        return event_loop.run_until_complete(group.run_episode(binding))

    def sample_candidate(
        *,
        action_slot: int,
        action_seed: int,
        reference_key: ExactActionKey,
    ) -> CandidateSubmission:
        recovered = ledger.completed_candidate_evidence(
            group_id=group.spec.commitment.group_id,
            target_id=group.spec.commitment.target_id,
            action_slot=action_slot,
        )
        if recovered is not None:
            action_bytes, receipt = recovered
            action = group.decode_action(action_bytes)
            if action.key.sampler.seed != action_seed:
                raise RuntimeError("recovered candidate used a different frozen seed")
            return CandidateSubmission(action, receipt)
        attempt = ledger.begin_candidate_attempt(
            group_id=group.spec.commitment.group_id,
            target_id=group.spec.commitment.target_id,
            action_slot=action_slot,
        )
        started = False
        response_sha256: str | None = None

        def before_post(request: bytes) -> None:
            nonlocal started
            if started:
                raise RuntimeError("candidate engine attempted more than one POST")
            request_sha256 = ledger.put_evidence(request)
            ledger.mark_candidate_model_call_started(
                attempt,
                request_sha256=request_sha256,
            )
            started = True

        def after_post(response: bytes) -> None:
            nonlocal response_sha256
            if not started or response_sha256 is not None:
                raise RuntimeError("candidate response witness has invalid ordering")
            response_sha256 = ledger.put_evidence(response)
            ledger.mark_candidate_response_observed(
                attempt,
                response_sha256=response_sha256,
            )

        try:
            result = event_loop.run_until_complete(
                group.candidate_engine(
                    reference_key=reference_key,
                    action_seed=action_seed,
                    before_post=before_post,
                    after_post=after_post,
                )
            )
        except BaseException as error:
            if started:
                raise
            failure_evidence = canonical_json(
                {
                    "schema_version": 1,
                    "domain": "redco-stage-d-zero-call-candidate-failure-v1",
                    "error_type": type(error).__qualname__,
                    "error_message": str(error),
                }
            )
            receipt = ledger.record_zero_call_candidate_failure(
                attempt,
                reason=f"{type(error).__qualname__}: {error}",
                supervisor_evidence_sha256=ledger.put_evidence(failure_evidence),
            )
            raise ZeroCallInfrastructureFailure(receipt) from error
        if not started:
            missing_marker_error = RuntimeError(
                "candidate engine returned without its durable POST marker"
            )
            failure_evidence = canonical_json(
                {
                    "schema_version": 1,
                    "domain": "redco-stage-d-zero-call-candidate-failure-v1",
                    "error_type": type(missing_marker_error).__qualname__,
                    "error_message": str(missing_marker_error),
                }
            )
            receipt = ledger.record_zero_call_candidate_failure(
                attempt,
                reason=str(missing_marker_error),
                supervisor_evidence_sha256=ledger.put_evidence(failure_evidence),
            )
            raise ZeroCallInfrastructureFailure(receipt) from missing_marker_error
        if response_sha256 is None:
            raise RuntimeError("candidate engine omitted its durable raw response witness")
        if hashlib.sha256(result.response_evidence).hexdigest() != response_sha256:
            raise RuntimeError("candidate result changed its witnessed raw response")
        receipt = ledger.complete_candidate_call(
            attempt,
            action=result.action,
            response_sha256=response_sha256,
        )
        return CandidateSubmission(result.action, receipt)

    def execute_arm(
        *,
        arm_id: str,
        action: BehaviorAction,
        continuation_replicate: int,
        seed_oracle: Any,
    ) -> bytes:
        recovered = ledger.completed_execution_receipt(
            group_id=group.spec.commitment.group_id,
            target_id=group.spec.commitment.target_id,
            arm_id=arm_id,
            continuation_replicate=continuation_replicate,
        )
        if recovered is not None:
            return recovered
        attempt = ledger.begin_execution(
            group_id=group.spec.commitment.group_id,
            target_id=group.spec.commitment.target_id,
            arm_id=arm_id,
            action=action,
            continuation_replicate=continuation_replicate,
        )
        binding = StageDScientificEpisodeBinding(
            mode="execution",
            task=group.task,
            source=group.source,
            source_records=records,
            source_graph=graph,
            target=target,
            expected_runtime_snapshot=group.expected_runtime_snapshot,
            expected_terminal_reply=expected_reply,
            ledger=ledger,
            candidate_action=action,
            seed_oracle=seed_oracle,
            execution_attempt=attempt,
        )
        context = canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-scientific-dispatch-v1",
                "source_sha256": group.source.source_sha256,
                "target": target.as_payload(),
                "arm_id": arm_id,
                "action_digest": action.digest,
                "continuation_replicate": continuation_replicate,
                "episode_identity": binding.episode_identity,
                "runtime_snapshot_sha256": hashlib.sha256(
                    group.expected_runtime_snapshot
                ).hexdigest(),
            }
        )
        context_sha256 = ledger.put_evidence(context)
        ledger.bind_execution_context(attempt, context_sha256=context_sha256)
        ledger.mark_execution_dispatched(attempt)
        try:
            return event_loop.run_until_complete(group.run_episode(binding))
        except BaseException as error:
            failure_evidence = canonical_json(
                {
                    "schema_version": 1,
                    "domain": "redco-stage-d-zero-call-execution-failure-v1",
                    "error_type": type(error).__qualname__,
                    "error_message": str(error),
                }
            )
            try:
                receipt = ledger.record_zero_call_execution_failure(
                    attempt,
                    reason=f"{type(error).__qualname__}: {error}",
                    supervisor_evidence_sha256=ledger.put_evidence(failure_evidence),
                )
            except BaseException as record_error:
                raise error from record_error
            raise ZeroCallInfrastructureFailure(receipt) from error

    def prepare_artifact(artifact: BranchGroupArtifact) -> None:
        value = artifact.to_bytes()
        artifact_sha256 = ledger.put_evidence(value)
        if group.artifact_path.exists():
            if group.artifact_path.read_bytes() != value:
                raise RuntimeError("existing branch artifact differs from prepared bytes")
        else:
            _exclusive_write(group.artifact_path, value)
        completed = ledger.completed_branch_artifact_sha256(
            group_id=group.spec.commitment.group_id,
            target_id=group.spec.commitment.target_id,
        )
        if completed is None:
            ledger.record_branch_group_artifact_completed(
                group_id=group.spec.commitment.group_id,
                target_id=group.spec.commitment.target_id,
                artifact_sha256=artifact_sha256,
                training_batch_identity=artifact.training_batch_identity,
            )
        elif completed != artifact_sha256:
            raise RuntimeError("completed branch artifact differs from rebuilt bytes")

    return ScientificGroupRun(
        group.spec,
        qa,
        sample_candidate,
        execute_arm,
        prepare_artifact,
    )


def _terminal_reply(trace: Mapping[str, Any]) -> object:
    nodes = trace.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("source trace lacks message nodes")
    assistants: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict) or node.get("sampled") is not True:
            continue
        message = node.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            assistants.append(message)
    if not assistants:
        raise ValueError("source trace lacks a sampled terminal assistant reply")
    content = assistants[-1].get("content")
    return content.strip() if isinstance(content, str) else ""


def source_task_from_trace(trace: Mapping[str, Any], task_config: Any) -> StageDSourceTask:
    """Rebuild the exact task row persisted in the verified source trace."""
    task = trace.get("task")
    if not isinstance(task, dict) or not isinstance(task.get("data"), dict):
        raise ValueError("source trace lacks its exact task data")
    data = StageDSourceData.model_validate(task["data"])
    return StageDSourceTask(data, task_config)


def _exclusive_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    with temporary.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
