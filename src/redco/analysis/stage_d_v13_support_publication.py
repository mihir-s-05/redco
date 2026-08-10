"""Read-only support artifact authentication owned outside the materializer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from redco.analysis import stage_d_v13_support_contract as contract
from redco.analysis.stage_d_dependency_stack import live_owner_patch_payload
from redco.analysis.stage_d_v13_draft import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
)


def _canonical_json_object(raw: bytes, subject: str, suffix: str = "") -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{subject} is not JSON{suffix}") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{subject} is not canonical{suffix}")
    return value


def _authenticated_json(
    root: Path,
    relative: str,
    expected_sha256: str,
) -> dict[str, object]:
    raw = contract.read_authenticated(root, relative, expected_sha256)
    return _canonical_json_object(raw, "authenticated repository input", f": {relative}")


def _authenticate_repository_provenance(
    root: Path,
    *,
    selection_evidence: object,
    dependency_stack: object,
    sampling_contract: object,
) -> None:
    """Authenticate tracked predecessors without opening source or dependency trees."""

    if not isinstance(selection_evidence, dict):
        raise ValueError("published selection evidence is malformed")
    if selection_evidence.get("upstream_v12_hashes") != contract.UPSTREAM_EVIDENCE_SHA256:
        raise ValueError("published upstream evidence map differs")
    for relative, expected_sha256 in contract.UPSTREAM_EVIDENCE_SHA256.items():
        path = root / relative
        if relative in contract.SOURCE_FREE_OPTIONAL_EVIDENCE and not (
            path.exists() or path.is_symlink()
        ):
            continue
        contract.read_authenticated(root, relative, expected_sha256)

    contract.read_authenticated(
        root,
        contract.SELECTION_RECEIPT_RELATIVE,
        contract.SELECTION_RECEIPT_SHA256,
    )
    contract.read_authenticated(
        root,
        contract.SELECTION_MANIFEST_RELATIVE,
        contract.SELECTION_MANIFEST_SHA256,
    )
    mirror_raw = contract.read_authenticated(
        root, contract.SELECTION_CLAIM_RELATIVE, contract.SELECTION_CLAIM_SHA256
    )
    original_claim = root / contract.SELECTION_ORIGINAL_CLAIM_RELATIVE
    if original_claim.exists() or original_claim.is_symlink():
        original_raw = contract.read_authenticated(
            root,
            contract.SELECTION_ORIGINAL_CLAIM_RELATIVE,
            contract.SELECTION_CLAIM_SHA256,
        )
        if original_raw != mirror_raw:
            raise ValueError("retained selection claim differs from its tracked mirror")

    if dependency_stack != live_owner_patch_payload(root):
        raise ValueError("published dependency stack differs from repository patch bindings")

    action_closure = _authenticated_json(
        root,
        contract.ACTION_CLOSURE_RELATIVE,
        contract.ACTION_CLOSURE_SHA256,
    )
    action_closure_audit = _authenticated_json(
        root,
        contract.ACTION_CLOSURE_AUDIT_RELATIVE,
        contract.ACTION_CLOSURE_AUDIT_SHA256,
    )
    if (
        not isinstance(sampling_contract, dict)
        or action_closure.get("sampling_contract") != sampling_contract
        or action_closure_audit.get("sampling_contract") != sampling_contract
        or action_closure.get("dependency_stack") != dependency_stack
        or action_closure_audit.get("dependency_stack") != dependency_stack
    ):
        raise ValueError("published sampling contract differs from frozen action closure")

    launch_authorization = _authenticated_json(
        root,
        contract.LAUNCH_AUTHORIZATION_RELATIVE,
        contract.LAUNCH_AUTHORIZATION_SHA256,
    )
    frozen_bindings = launch_authorization.get("frozen_bindings")
    current_sampling_contract = contract.sampling_contract_binding(root)
    if (
        not isinstance(frozen_bindings, dict)
        or any(
            frozen_bindings.get(relative) != expected_sha256
            for relative, expected_sha256 in contract.LAUNCH_PREDECESSOR_BINDINGS.items()
        )
        or frozen_bindings.get(contract.SAMPLING_CONTRACT_SOURCE_RELATIVE)
        != current_sampling_contract["producer_source_sha256"]
        or frozen_bindings.get("sampling_contract_sha256") != sampling_contract.get("sha256")
        or any(
            current_sampling_contract[field] != sampling_contract.get(field)
            for field in ("version", "sha256", "producer_source_path")
        )
    ):
        raise ValueError("current sampling owner lacks an authenticated successor binding")


def authenticate_protocol_artifact_bytes(
    root: Path,
    artifacts: Mapping[str, bytes],
) -> dict[str, str]:
    """Authenticate reviewed bytes, semantics, and repository provenance."""

    artifacts = _stable_artifacts(artifacts)
    if set(artifacts) != set(contract.REVIEWED_PROTOCOL_ARTIFACT_SHA256):
        raise ValueError("reviewed support artifact set differs")
    hashes = {relative: sha256_bytes(raw) for relative, raw in artifacts.items()}
    if hashes != contract.REVIEWED_PROTOCOL_ARTIFACT_SHA256:
        raise ValueError("published support artifact differs from the reviewed byte set")
    parsed: dict[str, dict[str, object]] = {}
    for relative, raw in artifacts.items():
        parsed[relative] = _canonical_json_object(raw, "published support artifact")
    candidate = parsed[contract.CANDIDATE_RELATIVE]
    composition = parsed[contract.COMPOSITION_RELATIVE]
    protocol = parsed[contract.PROTOCOL_RELATIVE]
    audit = parsed[contract.PROTOCOL_AUDIT_RELATIVE]
    candidate_source = candidate.get("source")
    instrumentation = candidate.get("instrumentation")
    candidate_authority = candidate.get("authority")
    composition_candidate = composition.get("candidate")
    composition_cohort = composition.get("support_cohort")
    composition_authorization = composition.get("authorization")
    protocol_source = protocol.get("source")
    protocol_authorization = protocol.get("authorization")
    protocol_attempt_policy = protocol.get("attempt_policy")
    protocol_selection = protocol.get("selection_evidence")
    audit_selection = audit.get("selection_evidence")
    dependency_stack = protocol.get("dependency_stack")
    sampling_contract = candidate.get("sampling_contract")
    _authenticate_repository_provenance(
        root,
        selection_evidence=protocol_selection,
        dependency_stack=dependency_stack,
        sampling_contract=sampling_contract,
    )
    runtime = candidate_source.get("runtime") if isinstance(candidate_source, dict) else None
    expected_protocol_source = contract.protocol_source_binding()
    if (
        candidate.get("domain") != "redco-stage-d1-support-v13-candidate-ordinal-180-v1"
        or not isinstance(candidate_source, dict)
        or candidate_source.get("ordinal") != contract.CANDIDATE_SOURCE_ORDINAL
        or candidate_source.get("paper_id") != contract.CANDIDATE_PAPER_ID
        or candidate_source.get("example_id") != contract.CANDIDATE_EXAMPLE_ID
        or candidate_source.get("question_index") != contract.CANDIDATE_QUESTION_INDEX
        or candidate_source.get("row_sha256") != contract.CANDIDATE_ROW_SHA256
        or candidate_source.get("selection_claim_sha256") != contract.SELECTION_CLAIM_SHA256
        or candidate_source.get("selection_receipt_sha256")
        != contract.SELECTION_RECEIPT_SHA256
        or candidate_source.get("selection_evidence_manifest_sha256")
        != contract.SELECTION_MANIFEST_SHA256
        or not isinstance(protocol_selection, dict)
        or candidate_source.get("upstream_evidence_hashes")
        != protocol_selection.get("upstream_v12_hashes")
        or candidate_source.get("frozen_decision_rule_sha256")
        != protocol_selection.get("frozen_decision_rule_sha256")
        or not isinstance(instrumentation, dict)
        or instrumentation.get("post_180_requested") is not False
        or instrumentation.get("post_180_materialized") is not False
        or instrumentation.get("post_180_canonicalized") is not False
        or instrumentation.get("post_180_evaluated") is not False
        or instrumentation.get("materialized_ordinals") != [contract.CANDIDATE_SOURCE_ORDINAL]
        or instrumentation.get("canonicalized_ordinals") != [contract.CANDIDATE_SOURCE_ORDINAL]
        or instrumentation.get("evaluated_ordinals") != [contract.CANDIDATE_SOURCE_ORDINAL]
        or runtime
        != {
            "python": contract.SUPPORTED_PYTHON,
            "pyarrow": contract.SUPPORTED_PYARROW,
            "datasets": contract.SUPPORTED_DATASETS,
            "supported": True,
        }
        or candidate_authority != contract.CANDIDATE_AUTHORITY
        or protocol_source != expected_protocol_source
        or not isinstance(dependency_stack, dict)
        or audit.get("dependency_stack") != dependency_stack
        or not isinstance(sampling_contract, dict)
        or composition.get("sampling_contract") != sampling_contract
        or protocol.get("sampling_contract") != sampling_contract
        or audit.get("sampling_contract") != sampling_contract
    ):
        raise ValueError("published candidate does not bind the authenticated materialization")
    if (
        not isinstance(composition_candidate, dict)
        or composition_candidate.get("sha256") != hashes[contract.CANDIDATE_RELATIVE]
        or composition_candidate.get("ordinal") != contract.CANDIDATE_SOURCE_ORDINAL
        or composition_candidate.get("paper_id") != contract.CANDIDATE_PAPER_ID
        or composition_candidate.get("example_id") != contract.CANDIDATE_EXAMPLE_ID
        or composition_candidate.get("row_sha256") != contract.CANDIDATE_ROW_SHA256
        or composition_cohort != contract.SUPPORT_COHORT
        or composition_authorization != contract.COMPOSITION_AUTHORIZATION
        or audit.get("candidate_sha256") != hashes[contract.CANDIDATE_RELATIVE]
        or audit.get("composition_sha256") != sha256_json(composition)
        or audit.get("protocol_sha256") != sha256_json(protocol)
        or protocol_selection.get("manifest_sha256") != contract.SELECTION_MANIFEST_SHA256
        or protocol_selection.get("receipt_sha256") != contract.SELECTION_RECEIPT_SHA256
        or protocol_selection.get("claim_sha256") != contract.SELECTION_CLAIM_SHA256
        or not isinstance(audit_selection, dict)
        or audit_selection.get("upstream_v12_hashes")
        != protocol_selection.get("upstream_v12_hashes")
        or not isinstance(protocol_attempt_policy, dict)
        or protocol_attempt_policy.get("maximum_live_support_attempts_global") != 1
        or protocol_attempt_policy.get("outcome_bearing_cohorts") != 1
        or protocol_attempt_policy.get("second_outcome_bearing_attempt")
        != "forbidden_unconditionally"
        or protocol_authorization != contract.PROTOCOL_AUTHORIZATION
        or protocol.get("support_pass_transition")
        != "user_checkpoint_required_before_any_support_spend_or_science_transition"
        or audit.get("domain") != "redco-stage-d1-support-v13-protocol-audit-v1"
        or audit.get("ready_for_live_support") is not False
        or audit.get("live_activity_performed") is not False
        or audit.get("runtime") != runtime
    ):
        raise ValueError("published support artifact cross-references differ")
    return hashes


def _stable_artifacts(artifacts: Mapping[str, bytes]) -> dict[str, bytes]:
    """Copy one exact immutable byte mapping before any trust decision."""

    stable: dict[str, bytes] = {}
    for relative, raw in artifacts.items():
        if type(relative) is not str or type(raw) is not bytes or relative in stable:
            raise ValueError("reviewed support artifact representation differs")
        stable[relative] = raw
    return stable


def check_protocol_artifacts(root: Path, output_root: Path) -> dict[str, str]:
    """Authenticate a published set without opening or decoding QASPER."""

    if not output_root.is_dir() or output_root.is_symlink():
        raise ValueError("published support output root must be a non-symlink directory")
    paths = set(contract.REVIEWED_PROTOCOL_ARTIFACT_SHA256)
    for entry in output_root.rglob("*"):
        relative = entry.relative_to(output_root).as_posix()
        if relative not in paths and (entry.is_file() or entry.is_symlink()):
            raise ValueError(f"published support output contains an unexpected file: {relative}")
    artifacts: dict[str, bytes] = {}
    for relative in paths:
        path = output_root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"published support artifact is missing: {relative}")
        artifacts[relative] = path.read_bytes()
    return authenticate_protocol_artifact_bytes(root, artifacts)


__all__ = ["authenticate_protocol_artifact_bytes", "check_protocol_artifacts"]
