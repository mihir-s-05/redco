"""Dormant live-v13 returning-root correspondence contract.

Verifiers' strict V2 reader owns provenance parsing.  This module only applies
the returning-root law after that reader has authenticated the trace.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any, Final, Literal, cast

from redco.analysis.stage_d_receipt_ledger import StageDReceiptLedger, inspect_ledger
from redco.analysis.stage_d_returning_root_contract import (
    CAUSAL_PROVENANCE_SCHEMA,
    CAUSAL_PROVENANCE_SCHEMA_BYTES,
    CAUSAL_PROVENANCE_SCHEMA_SHA256,
    CONTRACT_PAYLOAD,
    E2_ACTION_SHA256,
    E2_COMMITMENT_SHA256,
    E2_SOURCE_SHA256,
    E2_TRACE_SHA256,
    JSON,
    LIVE_CORRESPONDENCE_RECEIPT_VERSION,
    PROVENANCE_FIELDS,
    RAW_TRACE_FIELDS,
    RETURNING_ROOT_CONTRACT_BYTES,
    RETURNING_ROOT_CONTRACT_SHA256,
    RETURNING_ROOT_CONTRACT_VERSION,
    TERMINAL_OWNER_IDENTITY,
)
from redco.analysis.stage_d_spawn_provenance import (
    PolicyEventAddress,
)
from redco.contracts import canonical_json
from redco.integrations.verifiers_trace_v2 import (
    RecordedRLMProvenanceV2,
    extract_v2_rlm_provenance,
)

type Trace = dict[str, Any]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


_SNAPSHOT_FIELDS: Final = frozenset(
    {"completed_predecessor_spawn_ordinals", "completed_episode_spawn_ordinals"}
)


class CorrespondenceDisposition(StrEnum):
    ELIGIBLE_MATCH = "eligible_match"
    TERMINAL_WITHOUT_DOWNSTREAM = "terminal_without_downstream"
    MISSING_TARGET = "missing_target"
    NO_VALID_RETURN = "no_valid_return"
    AMBIGUOUS_CONFLICTING_MINIMA = "ambiguous_conflicting_minima"
    PROVENANCE_DIVERGENCE = "provenance_divergence"
    TOPOLOGY_DIVERGENCE = "topology_divergence"


def _valid_sha(value: object) -> bool:
    if type(value) is not str:
        return False
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True)
class AuthenticatedTerminationEvidence:
    """Terminal binding authenticated by the source-side receipt owner."""

    schema_version: int
    state: Literal["clean_terminated_before_recurrence"]
    recurrence_observed: bool
    source_completed: bool
    terminal_call_ordinal: int
    terminal_policy_turn: int
    terminal_policy_lineage: str
    source_id: str
    source_sha256: str
    trace_id: str
    trace_sha256: str
    group_id: str
    rollout_id: str
    terminal_owner_identity: str
    terminal_disposition: Literal["terminal_without_downstream"]
    terminal_reason: Literal["no_downstream_model_call"]
    ledger_id: str
    source_completion_receipt: bytes
    source_completion_receipt_sha256: str
    terminal_trace_evidence_sha256: str
    target_id: str
    target_ordinal: int
    target_address: PolicyEventAddress
    target_commitment_receipt: bytes
    target_commitment_receipt_sha256: str
    verifier: StageDReceiptLedger

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.state != "clean_terminated_before_recurrence"
            or self.recurrence_observed is not False
            or self.source_completed is not False
            or type(self.terminal_call_ordinal) is not int
            or self.terminal_call_ordinal < 0
            or type(self.terminal_policy_turn) is not int
            or self.terminal_policy_turn < 0
            or not self.terminal_policy_lineage
            or self.terminal_owner_identity != TERMINAL_OWNER_IDENTITY
            or self.terminal_disposition != "terminal_without_downstream"
            or self.terminal_reason != "no_downstream_model_call"
            or not self.target_id
            or type(self.target_ordinal) is not int
            or self.target_ordinal < 0
            or type(self.target_address) is not PolicyEventAddress
            or type(self.source_completion_receipt) is not bytes
            or type(self.target_commitment_receipt) is not bytes
            or not _valid_sha(self.source_sha256)
            or not _valid_sha(self.trace_sha256)
            or not _valid_sha(self.source_completion_receipt_sha256)
            or not _valid_sha(self.target_commitment_receipt_sha256)
            or not _valid_sha(self.terminal_trace_evidence_sha256)
            or _sha256(self.source_completion_receipt)
            != self.source_completion_receipt_sha256
            or _sha256(self.target_commitment_receipt)
            != self.target_commitment_receipt_sha256
            or type(self.verifier) is not StageDReceiptLedger
        ):
            raise ValueError("authenticated termination evidence is invalid")

    def to_payload(self) -> JSON:
        return {
            "schema_version": self.schema_version,
            "domain": "redco-stage-d-live-returning-root-source-terminal-binding-v1",
            "state": self.state,
            "recurrence_observed": self.recurrence_observed,
            "source_completed": self.source_completed,
            "terminal_call_ordinal": self.terminal_call_ordinal,
            "terminal_policy_turn": self.terminal_policy_turn,
            "terminal_policy_lineage": self.terminal_policy_lineage,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "trace_id": self.trace_id,
            "trace_sha256": self.trace_sha256,
            "group_id": self.group_id,
            "rollout_id": self.rollout_id,
            "terminal_owner_identity": self.terminal_owner_identity,
            "terminal_disposition": self.terminal_disposition,
            "terminal_reason": self.terminal_reason,
            "target_id": self.target_id,
            "target_ordinal": self.target_ordinal,
            "target_address": _address_payload(self.target_address),
            "ledger_id": self.ledger_id,
            "target_commitment_receipt_kind": "pre_action_group_commitment",
            "target_commitment_receipt_b64": base64.b64encode(
                self.target_commitment_receipt
            ).decode("ascii"),
            "target_commitment_receipt_sha256": self.target_commitment_receipt_sha256,
            "source_completion_receipt_kind": "source_rollout_completed",
            "source_completion_receipt_b64": base64.b64encode(
                self.source_completion_receipt
            ).decode("ascii"),
            "source_completion_receipt_sha256": self.source_completion_receipt_sha256,
            "terminal_trace_evidence_sha256": self.terminal_trace_evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class ReturningRootCorrespondenceInput:
    source_id: str
    source_sha256: str
    trace_id: str
    trace_sha256: str
    trace: Trace
    group_id: str
    rollout_id: str
    causal_graph_schema_sha256: str
    target_ordinal: int
    target_id: str
    target_address: PolicyEventAddress
    spawn_ordinal: int
    parent_lineage: str
    commitment_receipt_sha256: str
    recorded_action_digest: str
    termination_evidence: AuthenticatedTerminationEvidence | None = None


@dataclass(frozen=True, slots=True)
class ReturningRootCorrespondenceResult:
    disposition: CorrespondenceDisposition
    target_ordinal: int
    target_id: str
    matched_address: PolicyEventAddress | None = None

    def __post_init__(self) -> None:
        if self.disposition is CorrespondenceDisposition.ELIGIBLE_MATCH:
            if self.matched_address is None:
                raise ValueError("eligible correspondence requires one matched address")
        elif self.matched_address is not None:
            raise ValueError("non-eligible correspondence cannot carry a match")

    def to_payload(self) -> JSON:
        return {
            "disposition": self.disposition.value,
            "target_ordinal": self.target_ordinal,
            "target_id": self.target_id,
            "matched_address": _address_payload(self.matched_address),
        }


def _address_payload(address: PolicyEventAddress | None) -> JSON | None:
    if address is None:
        return None
    return {
        "depth": address.depth,
        "lineage": address.lineage,
        "session_call_ordinal": address.session_call_ordinal,
        "turn": address.turn,
        "call_kind": address.call_kind,
    }


def _record_payload(record: RecordedRLMProvenanceV2) -> JSON:
    payload: JSON = {"provenance_version": 2}
    for field in PROVENANCE_FIELDS[1:]:
        value = getattr(record, field)
        payload[field] = (
            list(value)
            if field in _SNAPSHOT_FIELDS and value is not None
            else value
        )
    return payload


def _record_projection(record: RecordedRLMProvenanceV2) -> tuple[object, ...]:
    payload = _record_payload(record)
    return tuple(payload[field] for field in PROVENANCE_FIELDS)


def _record_sort_key(record: RecordedRLMProvenanceV2) -> tuple[object, ...]:
    return (
        record.depth,
        record.lineage,
        record.session_call_ordinal,
        record.turn,
        record.call_kind,
    )


def _validate_returning_projection(
    records: tuple[RecordedRLMProvenanceV2, ...],
) -> None:
    """Apply only returning-root constraints not owned by strict V2 parsing."""
    by_lineage: dict[str, list[RecordedRLMProvenanceV2]] = {}
    for record in records:
        by_lineage.setdefault(record.lineage, []).append(record)
        if record.depth > 0 and any(
            field is None
            for field in (
                record.parent_session_id,
                record.parent_turn,
                record.parent_tool_call_id,
                record.invocation_id,
                record.parent_lineage,
                record.parent_call_ordinal,
                record.parent_tool_call_slot,
                record.spawn_ordinal,
                record.episode_spawn_ordinal,
                record.completed_predecessor_spawn_ordinals,
            )
        ):
            raise ValueError("returning-root child provenance is incomplete")
    for lineage_records in by_lineage.values():
        ordered = sorted(lineage_records, key=lambda record: record.session_call_ordinal)
        if [record.turn for record in ordered] != list(range(len(ordered))):
            raise ValueError("returning-root turn recurrence is invalid")


def _strict_records(
    trace: Trace,
    trace_sha256: str,
) -> tuple[tuple[RecordedRLMProvenanceV2, ...], JSON | None]:
    if set(trace) != set(RAW_TRACE_FIELDS):
        raise ValueError("returning-root trace envelope fields differ")
    if not _valid_sha(trace_sha256) or _sha256(canonical_json(trace)) != trace_sha256:
        raise ValueError("returning-root trace bytes do not match trace_sha256")
    records = extract_v2_rlm_provenance(trace)
    _validate_returning_projection(records)
    if (
        trace.get("ok") is not True
        or trace.get("is_completed") is not True
        or trace.get("errors") != []
        or not isinstance(trace.get("stop_condition"), str)
        or not trace["stop_condition"]
    ):
        return records, None
    if not records:
        raise ValueError("completed source trace has no authenticated policy calls")
    final = records[-1]
    return records, {
        "schema_version": 1,
        "state": "authenticated_completed_trace",
        "terminal_call_ordinal": final.call_index,
        "terminal_policy_turn": final.turn,
        "terminal_policy_lineage": final.lineage,
    }


def _source_identity(group_id: str, rollout_id: str) -> str:
    return f"{group_id}/{rollout_id}"


def _read_owner_evidence(owner: StageDReceiptLedger, digest: str) -> bytes:
    """Read evidence through the active source-finalization ledger owner."""
    scan = inspect_ledger(owner._root, allow_source_inflight=True)
    if scan.status != "active-clean":
        raise RuntimeError("source ledger is not active-clean")
    if not _valid_sha(digest) or digest not in scan.evidence_refs:
        raise ValueError("trace evidence is not anchored in the source ledger")
    value = cast(bytes, (owner._root / "evidence" / digest).read_bytes())
    if _sha256(value) != digest:
        raise ValueError("source trace evidence digest differs")
    return value


def authenticate_returning_root_terminal_evidence(
    trace: Trace,
    *,
    trace_sha256: str,
    source_id: str,
    source_sha256: str,
    trace_id: str,
    group_id: str,
    rollout_id: str,
    target_id: str,
    target_ordinal: int,
    target_address: PolicyEventAddress,
    source_completion_receipt: bytes,
    target_commitment_receipt: bytes,
    terminal_owner: StageDReceiptLedger,
) -> AuthenticatedTerminationEvidence:
    """Authenticate source terminal evidence before correspondence or QA."""
    if type(terminal_owner) is not StageDReceiptLedger:
        raise ValueError("terminal evidence requires the source Stage-D ledger owner")
    _, completion = _strict_records(trace, trace_sha256)
    if completion is None:
        raise ValueError("trace has no authenticated completed source trace")
    if (
        type(source_id) is not str
        or source_id != _source_identity(group_id, rollout_id)
        or trace_id != trace.get("id")
        or not _valid_sha(source_sha256)
    ):
        raise ValueError("terminal source identity differs from its authenticated rollout")
    source_payload = terminal_owner(
        source_completion_receipt,
        receipt_kind="source_rollout_completed",
    )
    commitment_payload = terminal_owner(
        target_commitment_receipt,
        receipt_kind="pre_action_group_commitment",
    )
    target_roster = commitment_payload.get("target_roster")
    if (
        source_payload.get("source_sha256") != source_sha256
        or source_payload.get("group_id") != group_id
        or source_payload.get("rollout_id") != rollout_id
        or source_payload.get("ledger_id") != commitment_payload.get("ledger_id")
        or commitment_payload.get("group_id") != group_id
        or commitment_payload.get("rollout_id") != rollout_id
        or commitment_payload.get("target_id") != target_id
        or commitment_payload.get("target_ordinal") != target_ordinal
        or commitment_payload.get("target_address") != _address_payload(target_address)
        or not isinstance(target_roster, list)
        or any(type(item) is not str or not item for item in target_roster)
        or len(target_roster) != len(set(target_roster))
        or target_ordinal >= len(target_roster)
        or target_roster[target_ordinal] != target_id
    ):
        raise ValueError("source terminal target commitment does not bind the same rollout")
    trace_evidence_sha256 = source_payload.get("trace_sha256")
    if not _valid_sha(trace_evidence_sha256):
        raise ValueError("source completion lacks an authenticated trace evidence digest")
    trace_evidence_digest = cast(str, trace_evidence_sha256)
    raw_episode = _read_owner_evidence(terminal_owner, trace_evidence_digest)
    try:
        episode = json.loads(raw_episode)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("durable trace evidence is not canonical JSON") from error
    if not isinstance(episode, dict) or canonical_json(episode) != raw_episode:
        raise ValueError("durable trace evidence is not canonical JSON")
    if set(episode) != {"id", "env", "ok", "errors", "traces"}:
        raise ValueError("durable trace evidence envelope differs")
    traces = episode.get("traces")
    if not isinstance(traces, list) or len(traces) != 1 or traces[0] != trace:
        raise ValueError("durable trace evidence does not match the evaluator trace")
    if _sha256(canonical_json(trace)) != trace_sha256:
        raise ValueError("durable trace evidence trace digest differs")
    return AuthenticatedTerminationEvidence(
        schema_version=1,
        state="clean_terminated_before_recurrence",
        recurrence_observed=False,
        source_completed=False,
            terminal_call_ordinal=cast(int, completion["terminal_call_ordinal"]),
            terminal_policy_turn=cast(int, completion["terminal_policy_turn"]),
            terminal_policy_lineage=cast(str, completion["terminal_policy_lineage"]),
        source_id=source_id,
        source_sha256=source_sha256,
        trace_id=trace_id,
        trace_sha256=trace_sha256,
        group_id=group_id,
        rollout_id=rollout_id,
        terminal_owner_identity=TERMINAL_OWNER_IDENTITY,
        terminal_disposition="terminal_without_downstream",
        terminal_reason="no_downstream_model_call",
        ledger_id=cast(str, source_payload["ledger_id"]),
        source_completion_receipt=source_completion_receipt,
        source_completion_receipt_sha256=_sha256(source_completion_receipt),
        terminal_trace_evidence_sha256=trace_evidence_digest,
        target_id=target_id,
        target_ordinal=target_ordinal,
        target_address=target_address,
        target_commitment_receipt=target_commitment_receipt,
        target_commitment_receipt_sha256=_sha256(target_commitment_receipt),
        verifier=terminal_owner,
    )


def extract_authenticated_termination_evidence(
    trace: Trace,
    *,
    trace_sha256: str,
    source_id: str | None = None,
    source_sha256: str | None = None,
    trace_id: str | None = None,
    group_id: str | None = None,
    rollout_id: str | None = None,
    target_id: str | None = None,
    target_ordinal: int | None = None,
    target_address: PolicyEventAddress | None = None,
    source_completion_receipt: bytes | None = None,
    target_commitment_receipt: bytes | None = None,
    terminal_owner: StageDReceiptLedger | None = None,
) -> AuthenticatedTerminationEvidence | None:
    """Extract terminal evidence only through the existing sealed receipt owner."""
    if _strict_records(trace, trace_sha256)[1] is None:
        return None
    if any(
        item is None
        for item in (
            source_id,
            source_sha256,
            trace_id,
            group_id,
            rollout_id,
            target_id,
            target_ordinal,
            target_address,
            source_completion_receipt,
            target_commitment_receipt,
            terminal_owner,
        )
    ):
        raise ValueError("durable terminal owner and receipts are required")
    assert source_id is not None
    assert source_sha256 is not None
    assert trace_id is not None
    assert group_id is not None
    assert rollout_id is not None
    assert target_id is not None
    assert target_ordinal is not None
    assert target_address is not None
    assert source_completion_receipt is not None
    assert target_commitment_receipt is not None
    assert terminal_owner is not None
    return authenticate_returning_root_terminal_evidence(
        trace,
        trace_sha256=trace_sha256,
        source_id=source_id,
        source_sha256=source_sha256,
        trace_id=trace_id,
        group_id=group_id,
        rollout_id=rollout_id,
        target_id=target_id,
        target_ordinal=target_ordinal,
        target_address=target_address,
        source_completion_receipt=source_completion_receipt,
        target_commitment_receipt=target_commitment_receipt,
        terminal_owner=terminal_owner,
    )


def _input_records(
    value: ReturningRootCorrespondenceInput,
) -> tuple[tuple[RecordedRLMProvenanceV2, ...], AuthenticatedTerminationEvidence | None]:
    if (
        not value.source_id
        or not _valid_sha(value.source_sha256)
        or not value.trace_id
        or not _valid_sha(value.trace_sha256)
        or value.causal_graph_schema_sha256 != CAUSAL_PROVENANCE_SCHEMA_SHA256
        or not _valid_sha(value.commitment_receipt_sha256)
        or not _valid_sha(value.recorded_action_digest)
        or type(value.target_ordinal) is not int
        or value.target_ordinal < 0
        or not value.target_id
        or type(value.spawn_ordinal) is not int
        or value.spawn_ordinal < 0
        or not value.parent_lineage
        or not value.group_id
        or not value.rollout_id
        or value.source_id != _source_identity(value.group_id, value.rollout_id)
        or value.trace_id != value.trace.get("id")
    ):
        raise ValueError("returning-root evaluator input binding is invalid")
    records, completion = _strict_records(value.trace, value.trace_sha256)
    if completion is None:
        if value.termination_evidence is not None:
            raise ValueError("terminal binding exists without an authenticated completion")
        return records, None
    evidence = value.termination_evidence
    if evidence is None:
        return records, None
    if evidence.to_payload()["terminal_call_ordinal"] != completion["terminal_call_ordinal"]:
        raise ValueError("durable terminal binding is absent or differs")
    if (
        evidence.source_id != value.source_id
        or evidence.source_sha256 != value.source_sha256
        or evidence.trace_id != value.trace_id
        or evidence.trace_sha256 != value.trace_sha256
        or evidence.group_id != value.group_id
        or evidence.rollout_id != value.rollout_id
        or evidence.target_id != value.target_id
        or evidence.target_ordinal != value.target_ordinal
        or evidence.target_address != value.target_address
        or value.commitment_receipt_sha256
        != evidence.target_commitment_receipt_sha256
        or evidence.terminal_policy_turn != completion["terminal_policy_turn"]
        or evidence.terminal_policy_lineage != completion["terminal_policy_lineage"]
    ):
        raise ValueError("durable terminal binding does not match evaluator input")
    authenticated = authenticate_returning_root_terminal_evidence(
        value.trace,
        trace_sha256=value.trace_sha256,
        source_id=value.source_id,
        source_sha256=value.source_sha256,
        trace_id=value.trace_id,
        group_id=value.group_id,
        rollout_id=value.rollout_id,
        target_id=value.target_id,
        target_ordinal=value.target_ordinal,
        target_address=value.target_address,
        source_completion_receipt=evidence.source_completion_receipt,
        target_commitment_receipt=evidence.target_commitment_receipt,
        terminal_owner=evidence.verifier,
    )
    if evidence != authenticated:
        raise ValueError("durable terminal binding differs from sealed owner output")
    return records, evidence


def _result(
    value: ReturningRootCorrespondenceInput,
    disposition: CorrespondenceDisposition,
    matched: PolicyEventAddress | None = None,
) -> ReturningRootCorrespondenceResult:
    return ReturningRootCorrespondenceResult(
        disposition,
        value.target_ordinal,
        value.target_id,
        matched,
    )


def _reachable(
    records: tuple[RecordedRLMProvenanceV2, ...],
    target: RecordedRLMProvenanceV2,
    candidate: RecordedRLMProvenanceV2,
) -> bool:
    by_event = {
        (record.lineage, record.session_call_ordinal): record
        for record in records
    }
    edges: dict[PolicyEventAddress, set[PolicyEventAddress]] = {
        record.scientific_address: set() for record in records
    }
    by_lineage: dict[str, list[RecordedRLMProvenanceV2]] = {}
    for record in records:
        by_lineage.setdefault(record.lineage, []).append(record)
    for group in by_lineage.values():
        for earlier, later in pairwise(
            sorted(group, key=lambda record: record.session_call_ordinal)
        ):
            edges[earlier.scientific_address].add(later.scientific_address)
    for record in records:
        if record.depth == 0:
            continue
        assert record.parent_lineage is not None
        assert record.parent_call_ordinal is not None
        parent = by_event.get((record.parent_lineage, record.parent_call_ordinal))
        if parent is None:
            return False
        edges[parent.scientific_address].add(record.scientific_address)
        for root in records:
            if (
                root.depth == 0
                and root.call_kind == "policy"
                and root.lineage == parent.lineage
                and root.session_id == parent.session_id
                and root.session_call_ordinal > parent.session_call_ordinal
                and record.episode_spawn_ordinal is not None
                and record.episode_spawn_ordinal
                in root.completed_episode_spawn_ordinals
            ):
                edges[record.scientific_address].add(root.scientific_address)
    pending = [target.scientific_address]
    visited = set(pending)
    while pending:
        current = pending.pop(0)
        for successor in edges[current]:
            if successor == candidate.scientific_address:
                return True
            if successor not in visited:
                visited.add(successor)
                pending.append(successor)
    return False


def derive_returning_root_correspondence(
    value: ReturningRootCorrespondenceInput,
) -> ReturningRootCorrespondenceResult:
    """Apply the returning-root law without content, reward, or arrival fields."""
    try:
        records, termination = _input_records(value)
    except (TypeError, ValueError, RuntimeError):
        return _result(value, CorrespondenceDisposition.PROVENANCE_DIVERGENCE)
    targets = [
        record for record in records if record.scientific_address == value.target_address
    ]
    if not targets:
        return _result(value, CorrespondenceDisposition.MISSING_TARGET)
    if len(targets) != 1:
        return _result(value, CorrespondenceDisposition.AMBIGUOUS_CONFLICTING_MINIMA)
    target = targets[0]
    if (
        target.depth != 1
        or target.call_kind != "policy"
        or target.parent_lineage != value.parent_lineage
        or target.spawn_ordinal != value.spawn_ordinal
        or target.episode_spawn_ordinal != value.spawn_ordinal
        or target.parent_session_id is None
        or target.parent_turn is None
        or target.parent_call_ordinal is None
        or target.parent_tool_call_id is None
        or target.parent_tool_call_slot is None
        or value.spawn_ordinal in target.completed_episode_spawn_ordinals
    ):
        return _result(value, CorrespondenceDisposition.PROVENANCE_DIVERGENCE)
    by_event = {
        (record.lineage, record.session_call_ordinal): record
        for record in records
    }
    parent = by_event.get((value.parent_lineage, target.parent_call_ordinal))
    if (
        parent is None
        or parent.depth != 0
        or parent.call_kind != "policy"
        or parent.session_id != target.parent_session_id
        or parent.turn != target.parent_turn
    ):
        return _result(value, CorrespondenceDisposition.TOPOLOGY_DIVERGENCE)
    candidates = [
        record
        for record in records
        if (
            record.depth == 0
            and record.call_kind == "policy"
            and record.lineage == value.parent_lineage
            and record.session_id == parent.session_id
            and record.session_call_ordinal > parent.session_call_ordinal
            and record.turn > parent.turn
            and value.spawn_ordinal in record.completed_episode_spawn_ordinals
            and _reachable(records, target, record)
        )
    ]
    if not candidates:
        if termination is not None:
            return _result(value, CorrespondenceDisposition.TERMINAL_WITHOUT_DOWNSTREAM)
        return _result(value, CorrespondenceDisposition.NO_VALID_RETURN)
    ordered = sorted(
        candidates,
        key=lambda record: (record.session_call_ordinal, record.turn),
    )
    minimum = ordered[0].session_call_ordinal, ordered[0].turn
    if sum(
        (record.session_call_ordinal, record.turn) == minimum for record in ordered
    ) != 1:
        return _result(value, CorrespondenceDisposition.AMBIGUOUS_CONFLICTING_MINIMA)
    return _result(
        value,
        CorrespondenceDisposition.ELIGIBLE_MATCH,
        ordered[0].scientific_address,
    )


def evaluate_returning_root_targets(
    values: Iterable[ReturningRootCorrespondenceInput],
) -> tuple[ReturningRootCorrespondenceResult, ...]:
    """Evaluate targets in target ordinal then target ID order."""
    ordered = tuple(sorted(values, key=lambda value: (value.target_ordinal, value.target_id)))
    keys = [(value.target_ordinal, value.target_id) for value in ordered]
    if len(set(keys)) != len(keys):
        raise ValueError("returning-root target order is not bijective")
    return tuple(derive_returning_root_correspondence(value) for value in ordered)


def _evaluator_input_payload(
    value: ReturningRootCorrespondenceInput,
    records: tuple[RecordedRLMProvenanceV2, ...],
    evidence: AuthenticatedTerminationEvidence | None,
) -> JSON:
    return {
        "source_id": value.source_id,
        "source_sha256": value.source_sha256,
        "trace_id": value.trace_id,
        "trace_sha256": value.trace_sha256,
        "trace": {
            "id": value.trace["id"],
            "provenance_records": [
                _record_payload(record)
                for record in sorted(records, key=_record_sort_key)
            ],
        },
        "group_id": value.group_id,
        "rollout_id": value.rollout_id,
        "causal_graph_schema_sha256": value.causal_graph_schema_sha256,
        "target_ordinal": value.target_ordinal,
        "target_id": value.target_id,
        "target_address": _address_payload(value.target_address),
        "spawn_ordinal": value.spawn_ordinal,
        "parent_lineage": value.parent_lineage,
        "commitment_receipt_sha256": value.commitment_receipt_sha256,
        "termination_evidence": None if evidence is None else evidence.to_payload(),
    }


def evaluator_input_payload(value: ReturningRootCorrespondenceInput) -> JSON:
    """Return the authenticated canonical projection used by live receipts."""
    records, evidence = _input_records(value)
    return _evaluator_input_payload(value, records, evidence)


def validate_replay_pair(
    *,
    source_input: ReturningRootCorrespondenceInput,
    replay_input: ReturningRootCorrespondenceInput,
) -> None:
    """Require exact source/replay provenance, including parent tool IDs."""
    source_records, _ = _input_records(source_input)
    replay_records, _ = _input_records(replay_input)
    source_result = derive_returning_root_correspondence(source_input)
    replay_result = derive_returning_root_correspondence(replay_input)
    if (
        source_result.disposition is not CorrespondenceDisposition.ELIGIBLE_MATCH
        or replay_result.disposition is not CorrespondenceDisposition.ELIGIBLE_MATCH
    ):
        raise ValueError("replay correspondence is not eligible")
    if (
        source_input.source_id != replay_input.source_id
        or source_input.source_sha256 != replay_input.source_sha256
        or source_input.trace_id != replay_input.trace_id
        or source_input.trace_sha256 != replay_input.trace_sha256
        or source_input.group_id != replay_input.group_id
        or source_input.rollout_id != replay_input.rollout_id
        or source_input.target_id != replay_input.target_id
        or source_input.target_ordinal != replay_input.target_ordinal
        or source_input.target_address != replay_input.target_address
        or source_input.spawn_ordinal != replay_input.spawn_ordinal
        or source_input.parent_lineage != replay_input.parent_lineage
        or source_result.matched_address != replay_result.matched_address
    ):
        raise ValueError("replay structural or target binding differs")
    source_projection = tuple(
        _record_projection(record)
        for record in sorted(source_records, key=_record_sort_key)
    )
    replay_projection = tuple(
        _record_projection(record)
        for record in sorted(replay_records, key=_record_sort_key)
    )
    if source_projection != replay_projection:
        raise ValueError("replay provenance projection differs")


def validate_live_correspondence_receipt(
    payload: object,
    *,
    evaluator_input: ReturningRootCorrespondenceInput,
) -> ReturningRootCorrespondenceResult:
    """Recompute exact evaluator bytes; legacy receipts cannot auto-upgrade."""
    fields = {
        "schema_version",
        "domain",
        "contract_version",
        "contract_sha256",
        "causal_graph_schema_sha256",
        "evaluator_input",
        "evaluator_input_sha256",
        "evaluator_result",
        "evaluator_result_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("legacy or malformed correspondence receipt is not live-v1")
    if (
        payload["schema_version"] != 1
        or payload["domain"] != LIVE_CORRESPONDENCE_RECEIPT_VERSION
        or payload["contract_version"] != RETURNING_ROOT_CONTRACT_VERSION
        or payload["contract_sha256"] != RETURNING_ROOT_CONTRACT_SHA256
        or payload["causal_graph_schema_sha256"] != CAUSAL_PROVENANCE_SCHEMA_SHA256
    ):
        raise ValueError("correspondence receipt contract binding differs")
    result = derive_returning_root_correspondence(evaluator_input)
    if result.disposition in {
        CorrespondenceDisposition.MISSING_TARGET,
        CorrespondenceDisposition.AMBIGUOUS_CONFLICTING_MINIMA,
        CorrespondenceDisposition.PROVENANCE_DIVERGENCE,
        CorrespondenceDisposition.TOPOLOGY_DIVERGENCE,
    }:
        raise ValueError("correspondence receipt input is not semantically valid")
    records, evidence = _input_records(evaluator_input)
    expected_input = _evaluator_input_payload(evaluator_input, records, evidence)
    expected_result = result.to_payload()
    input_digest = _sha256(canonical_json(expected_input))
    result_digest = _sha256(canonical_json(expected_result))
    if (
        payload["evaluator_input"] != expected_input
        or payload["evaluator_result"] != expected_result
        or payload["evaluator_input_sha256"] != input_digest
        or payload["evaluator_result_sha256"] != result_digest
        or not _valid_sha(payload["evaluator_input_sha256"])
        or not _valid_sha(payload["evaluator_result_sha256"])
    ):
        raise ValueError("correspondence receipt evaluator bytes or digests differ")
    return result


def contract_artifact_payload() -> JSON:
    """Return the candidate-independent Phase-1 contract audit payload."""
    return {
        "schema_version": 2,
        "domain": RETURNING_ROOT_CONTRACT_VERSION,
        "state": "phase-1-dormant",
        "contract_version": RETURNING_ROOT_CONTRACT_VERSION,
        "contract_sha256": RETURNING_ROOT_CONTRACT_SHA256,
        "causal_graph_schema_sha256": CAUSAL_PROVENANCE_SCHEMA_SHA256,
        "selection_order": ["target_ordinal", "target_id"],
        "legacy_partition": {
            "live_receipt_version": LIVE_CORRESPONDENCE_RECEIPT_VERSION,
            "legacy_receipts_accepted": False,
            "auto_upgrade": False,
        },
        "activation": {
            "live_receipt_production": False,
            "phase_2_required": True,
            "candidate": None,
            "source_access": False,
        },
        "recorded_action_digest": CONTRACT_PAYLOAD["recorded_action_digest"],
        "schemas": CONTRACT_PAYLOAD["schemas"],
        "schema_sha256s": CONTRACT_PAYLOAD["schema_sha256s"],
        "e2_regression_vector": {
            "source_sha256": E2_SOURCE_SHA256,
            "trace_sha256": E2_TRACE_SHA256,
            "target_ordinal": 0,
            "target_id": "e2-target-0",
            "target_address": {
                "depth": 1,
                "lineage": "root/child",
                "session_call_ordinal": 0,
                "turn": 0,
                "call_kind": "policy",
            },
            "spawn_ordinal": 0,
            "parent_lineage": "root",
            "commitment_receipt_sha256": E2_COMMITMENT_SHA256,
            "recorded_action_digest": E2_ACTION_SHA256,
            "disposition": CorrespondenceDisposition.ELIGIBLE_MATCH.value,
            "matched_address": {
                "depth": 0,
                "lineage": "root",
                "session_call_ordinal": 2,
                "turn": 2,
                "call_kind": "policy",
            },
        },
    }


def contract_artifact_bytes() -> bytes:
    return cast(bytes, canonical_json(contract_artifact_payload()))


def contract_artifact_sha256() -> str:
    return _sha256(contract_artifact_bytes())


__all__ = [
    "CAUSAL_PROVENANCE_SCHEMA",
    "CAUSAL_PROVENANCE_SCHEMA_BYTES",
    "CAUSAL_PROVENANCE_SCHEMA_SHA256",
    "LIVE_CORRESPONDENCE_RECEIPT_VERSION",
    "RETURNING_ROOT_CONTRACT_BYTES",
    "RETURNING_ROOT_CONTRACT_SHA256",
    "RETURNING_ROOT_CONTRACT_VERSION",
    "AuthenticatedTerminationEvidence",
    "CorrespondenceDisposition",
    "ReturningRootCorrespondenceInput",
    "ReturningRootCorrespondenceResult",
    "contract_artifact_bytes",
    "contract_artifact_payload",
    "contract_artifact_sha256",
    "derive_returning_root_correspondence",
    "evaluate_returning_root_targets",
    "evaluator_input_payload",
    "extract_authenticated_termination_evidence",
    "validate_live_correspondence_receipt",
    "validate_replay_pair",
]
