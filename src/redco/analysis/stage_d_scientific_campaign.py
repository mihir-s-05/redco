"""Whole-roster Stage D execution with a global reconstruction-QA barrier."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from redco.analysis.stage_d_exact_action import BehaviorAction
from redco.analysis.stage_d_receipt_ledger import StageDReceiptLedger
from redco.analysis.stage_d_scientific_branch_group import (
    ArmExecutor,
    BranchGroupArtifact,
    BranchGroupSpec,
    CandidateSampler,
    PreActionTargetCommitment,
    ReceiptVerifier,
    ReconstructionQAResult,
    RepairableInfrastructureAbort,
    run_scientific_branch_group,
)
from redco.contracts import canonical_json
from redco.integrations.verifiers_trace_v2 import parse_v2_rlm_provenance_payload


def runtime_snapshot_from_pre_action_evidence(
    value: bytes,
    *,
    commitment: PreActionTargetCommitment,
    recorded_action: BehaviorAction,
) -> bytes:
    """Extract the runtime subobject only after binding the full prepared snapshot."""
    try:
        payload = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("pre-action snapshot evidence is not JSON") from error
    expected_fields = {
        "schema_version",
        "domain",
        "trace_id",
        "event_address",
        "application_request",
        "engine_endpoint",
        "engine_request",
        "engine_headers",
        "observer_context",
        "frozen_runtime_snapshot",
    }
    key = recorded_action.key
    context = payload.get("observer_context") if isinstance(payload, dict) else None
    rlm = context.get("rlm") if isinstance(context, dict) else None
    provenance = (
        parse_v2_rlm_provenance_payload(
            trace_id=commitment.rollout_id,
            payload=rlm,
        )
        if isinstance(rlm, dict)
        else None
    )
    if (
        not isinstance(payload, dict)
        or canonical_json(payload) != value
        or hashlib.sha256(value).hexdigest() != commitment.pre_action_snapshot_sha256
        or set(payload) != expected_fields
        or payload.get("schema_version") != 1
        or payload.get("domain") != "redco-stage-d-pre-action-prepared-snapshot-v1"
        or payload.get("trace_id") != commitment.rollout_id
        or payload.get("event_address") != commitment.target_address.as_payload()
        or payload.get("application_request") != json.loads(key.request)
        or key.prepared_engine_request is None
        or payload.get("engine_request") != json.loads(key.prepared_engine_request)
        or not isinstance(payload.get("engine_endpoint"), str)
        or not payload["engine_endpoint"].endswith("/inference/v1/generate")
        or payload.get("engine_headers") != {"X-Session-ID": commitment.rollout_id}
        or not isinstance(context, dict)
        or context.get("trace_id") != commitment.rollout_id
        or provenance is None
        or provenance.scientific_address != commitment.target_address
        or not isinstance(payload.get("frozen_runtime_snapshot"), dict)
    ):
        raise ValueError("pre-action snapshot differs from its target action or runtime")
    return canonical_json(payload["frozen_runtime_snapshot"])


ReconstructionQARunner = Callable[[BranchGroupSpec], bytes]


@dataclass(frozen=True, slots=True)
class ScientificGroupRun:
    spec: BranchGroupSpec
    run_reconstruction_qa: ReconstructionQARunner
    sample_candidate: CandidateSampler
    execute_arm: ArmExecutor
    prepare_artifact: Callable[[BranchGroupArtifact], None]


@dataclass(frozen=True, slots=True)
class ScientificCampaignResult:
    reconstruction_qa_barrier_receipt: bytes
    artifacts: tuple[BranchGroupArtifact, ...]


def run_scientific_campaign(
    groups: Sequence[ScientificGroupRun],
    *,
    ledger: StageDReceiptLedger,
    verifier: ReceiptVerifier,
) -> ScientificCampaignResult:
    """Run model-free QA for the complete roster before any scientific call."""

    frozen_groups = tuple(groups)
    if not frozen_groups:
        raise ValueError("scientific campaign requires at least one target group")
    keys = tuple(
        (group.spec.commitment.group_id, group.spec.commitment.target_id) for group in frozen_groups
    )
    if len(set(keys)) != len(keys):
        raise ValueError("scientific campaign target groups must be unique")
    if tuple(sorted(keys)) != ledger.branch_target_keys:
        raise ValueError("scientific campaign groups differ from the frozen ledger roster")

    qa_receipts: list[bytes] = []
    for group in frozen_groups:
        group_id = group.spec.commitment.group_id
        target_id = group.spec.commitment.target_id
        try:
            receipt = ledger.reconstruction_qa_receipt(group_id, target_id)
            if receipt is None:
                receipt = group.run_reconstruction_qa(group.spec)
            qa = ReconstructionQAResult.from_receipt(
                receipt,
                verifier=verifier,
                commitment=group.spec.commitment,
                recorded_action=group.spec.recorded_action,
            )
        except BaseException as error:
            raise RepairableInfrastructureAbort(
                "whole-roster reconstruction QA did not produce a valid receipt"
            ) from error
        if not qa.passed:
            raise RepairableInfrastructureAbort(
                "whole-roster reconstruction QA failed before scientific execution"
            )
        qa_receipts.append(receipt)

    barrier = ledger.reconstruction_qa_barrier_receipt()
    if barrier is None:
        try:
            barrier = ledger.seal_reconstruction_qa_barrier()
        except BaseException as error:
            raise RepairableInfrastructureAbort(
                "whole-roster reconstruction QA could not be sealed"
            ) from error

    artifacts: list[BranchGroupArtifact] = []
    for group, qa_receipt in zip(frozen_groups, qa_receipts, strict=True):
        try:
            artifact = run_scientific_branch_group(
                group.spec,
                verifier=verifier,
                sample_candidate=group.sample_candidate,
                run_reconstruction_qa=None,
                execute_arm=group.execute_arm,
                prepare_artifact=group.prepare_artifact,
                reconstruction_qa_receipt=qa_receipt,
            )
        except RepairableInfrastructureAbort:
            raise
        artifacts.append(artifact)
    return ScientificCampaignResult(barrier, tuple(artifacts))
