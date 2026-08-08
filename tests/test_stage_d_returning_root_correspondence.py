from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from redco.analysis.stage_d_receipt_ledger import StageDReceiptLedger, inspect_ledger
from redco.analysis.stage_d_returning_root_correspondence import (
    CAUSAL_PROVENANCE_SCHEMA_SHA256,
    LIVE_CORRESPONDENCE_RECEIPT_VERSION,
    RETURNING_ROOT_CONTRACT_SHA256,
    RETURNING_ROOT_CONTRACT_VERSION,
    AuthenticatedTerminationEvidence,
    CorrespondenceDisposition,
    ReturningRootCorrespondenceInput,
    authenticate_returning_root_terminal_evidence,
    contract_artifact_bytes,
    contract_artifact_payload,
    contract_artifact_sha256,
    derive_returning_root_correspondence,
    evaluate_returning_root_targets,
    evaluator_input_payload,
    extract_authenticated_termination_evidence,
    validate_live_correspondence_receipt,
    validate_replay_pair,
)
from redco.analysis.stage_d_spawn_provenance import (
    PolicyEventAddress,
    SpawnScope,
    derive_child_lineage,
)
from redco.contracts import canonical_json


def _sha256(value: bytes) -> str:
    return str(hashlib.sha256(value).hexdigest())


def _address(
    depth: int,
    lineage: str,
    ordinal: int,
    turn: int,
    call_kind: str = "policy",
) -> PolicyEventAddress:
    return PolicyEventAddress(depth, lineage, ordinal, turn, call_kind)


def _target_lineage() -> str:
    return cast(
        str,
        derive_child_lineage(
            SpawnScope(1, "root", 0, 0, 0),
            spawn_ordinal=0,
        ),
    )


def _root_rlm(
    ordinal: int,
    turn: int,
    *,
    completed: tuple[int, ...] = (),
) -> dict[str, Any]:
    return {
        "provenance_version": 2,
        "depth": 0,
        "session_id": "root-session",
        "turn": turn,
        "call_kind": "policy",
        "parent_session_id": None,
        "parent_turn": None,
        "parent_tool_call_id": None,
        "invocation_id": None,
        "lineage": "root",
        "parent_lineage": None,
        "session_call_ordinal": ordinal,
        "parent_call_ordinal": None,
        "parent_tool_call_slot": None,
        "spawn_ordinal": None,
        "episode_spawn_ordinal": None,
        "completed_predecessor_spawn_ordinals": None,
        "completed_episode_spawn_ordinals": list(completed),
    }


def _child_rlm(
    *,
    parent_ordinal: int = 0,
    parent_turn: int = 0,
    spawn_ordinal: int = 0,
    episode_spawn_ordinal: int = 0,
    parent_tool_call_id: str = "call-0",
) -> dict[str, Any]:
    lineage = derive_child_lineage(
        SpawnScope(1, "root", parent_ordinal, 0, parent_turn),
        spawn_ordinal=spawn_ordinal,
    )
    return {
        "provenance_version": 2,
        "depth": 1,
        "session_id": "child-session" if spawn_ordinal == 0 else "dynamic-session",
        "turn": 0,
        "call_kind": "policy",
        "parent_session_id": "root-session",
        "parent_turn": parent_turn,
        "parent_tool_call_id": parent_tool_call_id,
        "invocation_id": "child-invocation",
        "lineage": lineage,
        "parent_lineage": "root",
        "session_call_ordinal": 0,
        "parent_call_ordinal": parent_ordinal,
        "parent_tool_call_slot": 0,
        "spawn_ordinal": spawn_ordinal,
        "episode_spawn_ordinal": episode_spawn_ordinal,
        "completed_predecessor_spawn_ordinals": [],
        "completed_episode_spawn_ordinals": [],
    }


def _trace(
    *,
    include_candidate: bool = True,
    clean_terminated: bool = False,
    child: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rlms = [
        _root_rlm(0, 0),
        _child_rlm() if child is None else child,
        _root_rlm(1, 1),
    ]
    if include_candidate:
        rlms.append(_root_rlm(2, 2, completed=(0,)))
    calls = [
        {
            "endpoint": "/chat/completions",
            "error": None,
            "finish_reason": "stop",
            "model": "model@commit",
            "node": index,
            "rlm": rlm,
            "sampling": {
                "temperature": 0.7,
                "top_p": 1.0,
                "reasoning_effort": None,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "seed": 100 + index,
                "max_tokens": 2,
                "n": 1,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
            },
            "time": {"start": float(index), "end": float(index + 1)},
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_input_tokens": None,
                "reasoning_tokens": None,
                "cost": None,
            },
        }
        for index, rlm in enumerate(rlms)
    ]
    return {
        "id": "trace",
        "task": {"type": "EvidenceSelectionTask", "data": {}},
        "runtime": None,
        "version": 1,
        "verifiers": {"version": "pinned"},
        "run": {"type": "train", "id": "run-1"},
        "agent": {
            "model": "model@commit",
            "sampling": {"temperature": 0.7},
        },
        "calls": calls,
        "nodes": [
            {
                "message": {},
                "sampled": False,
                "timestamp": 0.0,
                "token_ids": [],
                "mask": [],
                "is_content": [],
                "logprobs": [],
            }
            for _ in calls
        ],
        "tools": [],
        "rewards": {"exact_span_f1": 0.0},
        "metrics": {},
        "info": {"checkpoint_id": "model@commit"},
        "extra_usage": [],
        "is_completed": clean_terminated,
        "ok": True,
        "stop_condition": "final_answer" if clean_terminated else None,
        "errors": [],
        "timing": {},
    }


def _value(
    *,
    trace: dict[str, Any] | None = None,
    termination_evidence: object = ...,
    target_address: PolicyEventAddress | None = None,
) -> ReturningRootCorrespondenceInput:
    actual_trace = _trace() if trace is None else trace
    trace_sha256 = _sha256(canonical_json(actual_trace))
    try:
        extracted = extract_authenticated_termination_evidence(
            actual_trace,
            trace_sha256=trace_sha256,
        )
    except ValueError:
        extracted = None
    evidence: AuthenticatedTerminationEvidence | None
    if termination_evidence is ...:
        evidence = extracted
    else:
        evidence = cast(
            AuthenticatedTerminationEvidence | None,
            termination_evidence,
        )
    return ReturningRootCorrespondenceInput(
        source_id="group-1/rollout-1",
        source_sha256=_sha256(b"source"),
        trace_id=actual_trace["id"],
        trace_sha256=trace_sha256,
        trace=actual_trace,
        group_id="group-1",
        rollout_id="rollout-1",
        causal_graph_schema_sha256=CAUSAL_PROVENANCE_SCHEMA_SHA256,
        target_ordinal=0,
        target_id="target-0",
        target_address=(
            _address(1, _target_lineage(), 0, 0)
            if target_address is None
            else target_address
        ),
        spawn_ordinal=0,
        parent_lineage="root",
        commitment_receipt_sha256=_sha256(b"commitment"),
        recorded_action_digest=_sha256(b"action"),
        termination_evidence=evidence,
    )


def _without_candidate(*, clean: bool) -> ReturningRootCorrespondenceInput:
    try:
        return _value(trace=_trace(include_candidate=False, clean_terminated=clean))
    except ValueError:
        # A source terminal is intentionally unusable without the durable owner.
        return _value(
            trace=_trace(include_candidate=False, clean_terminated=clean),
            termination_evidence=None,
        )


def _rehashed(
    value: ReturningRootCorrespondenceInput,
    trace: dict[str, Any],
) -> ReturningRootCorrespondenceInput:
    trace_sha256 = _sha256(canonical_json(trace))
    evidence: AuthenticatedTerminationEvidence | None
    try:
        evidence = extract_authenticated_termination_evidence(
            trace,
            trace_sha256=trace_sha256,
        )
    except (TypeError, ValueError):
        evidence = value.termination_evidence
    return replace(
        value,
        trace=trace,
        trace_sha256=trace_sha256,
        termination_evidence=evidence,
    )


def _receipt(value: ReturningRootCorrespondenceInput) -> dict[str, Any]:
    result = derive_returning_root_correspondence(value)
    evaluator = evaluator_input_payload(value)
    result_payload = result.to_payload()
    return {
        "schema_version": 1,
        "domain": LIVE_CORRESPONDENCE_RECEIPT_VERSION,
        "contract_version": RETURNING_ROOT_CONTRACT_VERSION,
        "contract_sha256": RETURNING_ROOT_CONTRACT_SHA256,
        "causal_graph_schema_sha256": CAUSAL_PROVENANCE_SCHEMA_SHA256,
        "evaluator_input": evaluator,
        "evaluator_input_sha256": _sha256(canonical_json(evaluator)),
        "evaluator_result": result_payload,
        "evaluator_result_sha256": _sha256(canonical_json(result_payload)),
    }


def _durable_terminal_owner(
    tmp_path: Path,
    trace: dict[str, Any],
) -> tuple[StageDReceiptLedger, bytes, bytes, bytes]:
    """Build source-side evidence before correspondence or scientific QA."""
    import test_stage_d_receipt_ledger as ledger_tests

    source_sha256 = _sha256(b"source")
    ledger_root = tmp_path / "ledger"
    writer = ledger_tests._create(ledger_root)
    root_key = ledger_tests._key(11)
    root_request = writer.put_evidence(root_key.request)
    root_reservation = writer.reserve_source_policy_call(
        group_id="group-1",
        rollout_id="rollout-1",
        decision_id="root-turn-0",
        node_kind="root",
        target_id=None,
        target_ordinal=None,
        target_address=_address(0, "root", 0, 0),
        recorded_action_key=root_key,
        request_sha256=root_request,
        branch_selected=False,
        raw_response_required=True,
    )
    root_action = ledger_tests._materialize(root_key)
    root_raw_response = writer.put_evidence(b"root-raw-response")
    writer.mark_source_policy_response_observed(
        root_reservation,
        response_sha256=root_raw_response,
    )
    root_response = writer.put_evidence(root_action.to_bytes())
    root_completion = writer.complete_source_policy_call(
        root_reservation,
        action=root_action,
        response_sha256=root_response,
    )

    target_address = _address(1, _target_lineage(), 0, 0)
    recorded_key = ledger_tests._key(12)
    snapshot = writer.put_evidence(b"pre-action-snapshot")
    recorded_reservation = writer.commit_pre_action_and_reserve(
        group_id="group-1",
        rollout_id="rollout-1",
        target_roster=("target-0",),
        target_ordinal=0,
        target_id="target-0",
        target_address=target_address,
        pre_action_snapshot_sha256=snapshot,
        recorded_action_key=recorded_key,
        branch_count=2,
        continuation_replicates=1,
        failure_reward=-1.0,
    )
    recorded_action = ledger_tests._materialize(recorded_key)
    recorded_request = writer.put_evidence(recorded_key.request)
    writer.mark_recorded_action_model_call_started(
        recorded_reservation,
        request_sha256=recorded_request,
    )
    recorded_response = writer.put_evidence(recorded_action.to_bytes())
    writer.complete_recorded_action(
        recorded_reservation,
        action=recorded_action,
        response_sha256=recorded_response,
    )
    foreign_key = ledger_tests._key(13)
    foreign_address = _address(1, "root/foreign", 0, 0)
    foreign_snapshot = writer.put_evidence(b"foreign-pre-action-snapshot")
    foreign_reservation = writer.commit_pre_action_and_reserve(
        group_id="group-1",
        rollout_id="rollout-2",
        target_roster=("target-foreign",),
        target_ordinal=0,
        target_id="target-foreign",
        target_address=foreign_address,
        pre_action_snapshot_sha256=foreign_snapshot,
        recorded_action_key=foreign_key,
        branch_count=2,
        continuation_replicates=1,
        failure_reward=-1.0,
    )
    foreign_action = ledger_tests._materialize(foreign_key)
    foreign_request = writer.put_evidence(foreign_key.request)
    writer.mark_recorded_action_model_call_started(
        foreign_reservation,
        request_sha256=foreign_request,
    )
    foreign_response = writer.put_evidence(foreign_action.to_bytes())
    writer.complete_recorded_action(
        foreign_reservation,
        action=foreign_action,
        response_sha256=foreign_response,
    )

    child_key = recorded_key
    child_request = writer.put_evidence(child_key.request)
    child_reservation = writer.reserve_source_policy_call(
        group_id="group-1",
        rollout_id="rollout-1",
        decision_id="child-0",
        node_kind="child",
        target_id="target-0",
        target_ordinal=0,
        target_address=target_address,
        recorded_action_key=child_key,
        request_sha256=child_request,
        branch_selected=True,
        raw_response_required=True,
        recorded_action_reservation=recorded_reservation,
    )
    child_action = ledger_tests._materialize(child_key)
    child_raw_response = writer.put_evidence(b"child-raw-response")
    writer.mark_source_policy_response_observed(
        child_reservation,
        response_sha256=child_raw_response,
    )
    child_response = writer.put_evidence(child_action.to_bytes())
    child_completion = writer.complete_source_policy_call(
        child_reservation,
        action=child_action,
        response_sha256=child_response,
    )

    raw_episode = canonical_json(
        {
            "id": "episode-1",
            "env": "returning-root-test",
            "ok": True,
            "errors": [],
            "traces": [trace],
        }
    )
    trace_evidence = writer.put_evidence(raw_episode)
    reward_evidence = writer.put_evidence(b"reward")
    stock_evidence = writer.put_evidence(b"stock")
    source_completion = writer.record_source_rollout_completed(
        group_id="group-1",
        rollout_id="rollout-1",
        source_sha256=source_sha256,
        trace_sha256=trace_evidence,
        reward_evidence_sha256=reward_evidence,
        stock_sequences_evidence_sha256=stock_evidence,
        base_model_manifest_sha256="6" * 64,
        decision_ids=("root-turn-0", "child-0"),
        decision_completion_receipt_sha256s=(
            _sha256(root_completion),
            _sha256(child_completion),
        ),
    )
    return (
        writer,
        source_completion.receipt,
        recorded_reservation.commitment_receipt,
        foreign_reservation.commitment_receipt,
    )


def test_e2_vector_and_contract_artifact_are_exact_and_dormant() -> None:
    result = derive_returning_root_correspondence(_value())
    assert result.disposition is CorrespondenceDisposition.ELIGIBLE_MATCH
    assert result.matched_address == _address(0, "root", 2, 2)
    artifact_path = (
        Path(__file__).parents[1]
        / "reports/stage-d1-support-v13-returning-root-correspondence-contract-v1.json"
    )
    assert artifact_path.read_bytes() == contract_artifact_bytes()
    assert artifact_path.read_bytes().endswith(b"\n") is False
    assert contract_artifact_sha256() == _sha256(artifact_path.read_bytes())
    artifact = contract_artifact_payload()
    assert artifact["contract_sha256"] == RETURNING_ROOT_CONTRACT_SHA256
    assert artifact["contract_version"] == RETURNING_ROOT_CONTRACT_VERSION
    assert artifact["state"] == "phase-1-dormant"
    assert artifact["activation"] == {
        "live_receipt_production": False,
        "phase_2_required": True,
        "candidate": None,
        "source_access": False,
    }
    assert artifact["schemas"]["input"]["additionalProperties"] is False  # type: ignore[index]


def test_completed_trace_with_returning_root_is_not_terminal_evidence() -> None:
    value = _value(trace=_trace(clean_terminated=True))
    assert value.termination_evidence is None
    result = derive_returning_root_correspondence(value)
    assert result.disposition is CorrespondenceDisposition.ELIGIBLE_MATCH
    assert result.matched_address == _address(0, "root", 2, 2)


def test_target_processing_is_ordinal_then_id_and_not_arrival_order() -> None:
    first = replace(_value(), target_id="b", target_ordinal=1)
    second = replace(_value(), target_id="a", target_ordinal=0)
    results = evaluate_returning_root_targets((first, second))
    assert [(item.target_ordinal, item.target_id) for item in results] == [(0, "a"), (1, "b")]
    assert results == evaluate_returning_root_targets((second, first))


def test_public_evaluator_calls_strict_v2_owner_and_rejects_reduced_parser_exploits() -> None:
    value = _value()
    for field, replacement in (
        ("turn", 99),
        ("completed_episode_spawn_ordinals", [0, 0]),
        ("provenance_version", 1),
    ):
        trace = copy.deepcopy(value.trace)
        trace["calls"][3]["rlm"][field] = replacement
        changed = _rehashed(value, trace)
        assert (
            derive_returning_root_correspondence(changed).disposition
            is CorrespondenceDisposition.PROVENANCE_DIVERGENCE
        )
    trace = copy.deepcopy(value.trace)
    trace["calls"][1]["rlm"]["lineage"] = "root/not-derived"
    assert (
        derive_returning_root_correspondence(_rehashed(value, trace)).disposition
        is CorrespondenceDisposition.PROVENANCE_DIVERGENCE
    )


def test_one_field_provenance_matrix_is_fail_closed() -> None:
    fields: tuple[str, ...] = (
        "depth",
        "turn",
        "call_kind",
        "parent_session_id",
        "parent_turn",
        "invocation_id",
        "parent_lineage",
        "parent_call_ordinal",
        "parent_tool_call_slot",
        "spawn_ordinal",
        "episode_spawn_ordinal",
        "completed_predecessor_spawn_ordinals",
        "completed_episode_spawn_ordinals",
    )
    replacements: dict[str, object] = {
        "depth": 2,
        "session_id": "other-session",
        "turn": 99,
        "call_kind": "compaction",
        "parent_session_id": "other-root",
        "parent_turn": 9,
        "parent_tool_call_id": "other-call",
        "invocation_id": None,
        "parent_lineage": "other",
        "parent_call_ordinal": 9,
        "parent_tool_call_slot": 9,
        "spawn_ordinal": 9,
        "episode_spawn_ordinal": 9,
        "completed_predecessor_spawn_ordinals": [0],
        "completed_episode_spawn_ordinals": [0, 0],
    }
    value = _value()
    for field in fields:
        trace = copy.deepcopy(value.trace)
        trace["calls"][1]["rlm"][field] = replacements[field]
        result = derive_returning_root_correspondence(_rehashed(value, trace))
        assert result.disposition is not CorrespondenceDisposition.ELIGIBLE_MATCH, field


def test_same_lineage_session_identity_is_stable() -> None:
    value = _value()
    trace = copy.deepcopy(value.trace)
    continuation = _child_rlm()
    continuation["turn"] = 1
    continuation["session_call_ordinal"] = 1
    trace["calls"].insert(2, {"node": 4, "rlm": continuation})
    trace["nodes"].insert(2, {})
    trace["calls"][1]["rlm"]["session_id"] = "mutated-session"
    assert (
        derive_returning_root_correspondence(_rehashed(value, trace)).disposition
        is CorrespondenceDisposition.PROVENANCE_DIVERGENCE
    )


def test_target_address_and_parent_lineage_mutations_fail_closed() -> None:
    value = _value()
    for address in (
        _address(0, "root", 0, 0),
        _address(1, "other", 0, 0),
        _address(1, _target_lineage(), 9, 0),
        _address(1, _target_lineage(), 0, 99),
        _address(1, _target_lineage(), 0, 0, "compaction"),
    ):
        result = derive_returning_root_correspondence(
            replace(value, target_address=address),
        )
        assert result.disposition is not CorrespondenceDisposition.ELIGIBLE_MATCH
    assert (
        derive_returning_root_correspondence(
            replace(value, parent_lineage="other"),
        ).disposition
        is CorrespondenceDisposition.PROVENANCE_DIVERGENCE
    )


def test_upstream_target_self_and_missing_spawn_never_match() -> None:
    value = _value()
    no_candidate = _without_candidate(clean=False)
    assert (
        derive_returning_root_correspondence(no_candidate).disposition
        is CorrespondenceDisposition.NO_VALID_RETURN
    )
    trace = copy.deepcopy(value.trace)
    trace["calls"][1]["rlm"]["completed_episode_spawn_ordinals"] = [0]
    assert (
        derive_returning_root_correspondence(_rehashed(value, trace)).disposition
        is CorrespondenceDisposition.PROVENANCE_DIVERGENCE
    )
    trace = copy.deepcopy(value.trace)
    trace["calls"][3]["rlm"]["completed_episode_spawn_ordinals"] = []
    assert (
        derive_returning_root_correspondence(_rehashed(value, trace)).disposition
        is CorrespondenceDisposition.NO_VALID_RETURN
    )


def test_earliest_returning_root_wins_and_later_root_is_exogenous() -> None:
    value = _value()
    trace = copy.deepcopy(value.trace)
    trace["calls"].append({"node": 4, "rlm": _root_rlm(3, 3, completed=(0,))})
    trace["nodes"].append({})
    result = derive_returning_root_correspondence(_rehashed(value, trace))
    assert result.disposition is CorrespondenceDisposition.ELIGIBLE_MATCH
    assert result.matched_address == _address(0, "root", 2, 2)


def test_dynamic_child_and_arrival_order_cannot_change_selection() -> None:
    value = _value()
    dynamic = _child_rlm(
        parent_ordinal=1,
        parent_turn=1,
        spawn_ordinal=0,
        episode_spawn_ordinal=1,
        parent_tool_call_id="call-1",
    )
    trace = copy.deepcopy(value.trace)
    dynamic_call = copy.deepcopy(trace["calls"][0])
    dynamic_call["node"] = 4
    dynamic_call["rlm"] = dynamic
    trace["calls"].insert(3, dynamic_call)
    trace["nodes"].insert(3, copy.deepcopy(trace["nodes"][0]))
    trace["calls"][4]["rlm"]["completed_episode_spawn_ordinals"] = [0, 1]
    dynamic_result = derive_returning_root_correspondence(_rehashed(value, trace))
    assert dynamic_result.matched_address == _address(0, "root", 2, 2)
    reversed_trace = copy.deepcopy(trace)
    reversed_trace["calls"] = list(reversed(reversed_trace["calls"]))
    reversed_result = derive_returning_root_correspondence(
        _rehashed(value, reversed_trace),
    )
    assert reversed_result == dynamic_result


def test_authenticated_terminal_evidence_replaces_caller_boolean() -> None:
    value = _without_candidate(clean=True)
    assert (
        derive_returning_root_correspondence(value).disposition
        is CorrespondenceDisposition.NO_VALID_RETURN
    )
    forged = replace(value, termination_evidence=None)
    assert (
        derive_returning_root_correspondence(forged).disposition
        is CorrespondenceDisposition.NO_VALID_RETURN
    )


def test_terminal_self_hash_without_durable_owner_is_rejected() -> None:
    value = _without_candidate(clean=True)
    forged_receipt = canonical_json(
        {
            "receipt_kind": "source_rollout_completed",
            "source_sha256": value.source_sha256,
            "trace_sha256": value.trace_sha256,
        }
    )
    assert _sha256(forged_receipt) != "0" * 64
    assert (
        derive_returning_root_correspondence(value).disposition
        is CorrespondenceDisposition.NO_VALID_RETURN
    )


def test_terminal_disposition_requires_source_ledger_owner(tmp_path: Path) -> None:
    from jsonschema import Draft202012Validator

    from redco.analysis.stage_d_returning_root_contract import TERMINAL_BINDING_SCHEMA

    value = _without_candidate(clean=True)
    owner, source_receipt, commitment_receipt, _ = _durable_terminal_owner(
        tmp_path,
        value.trace,
    )
    evidence = authenticate_returning_root_terminal_evidence(
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
        source_completion_receipt=source_receipt,
        target_commitment_receipt=commitment_receipt,
        terminal_owner=owner,
    )
    Draft202012Validator(TERMINAL_BINDING_SCHEMA).validate(evidence.to_payload())
    authenticated = replace(
        value,
        commitment_receipt_sha256=_sha256(commitment_receipt),
        termination_evidence=evidence,
    )
    assert (
        derive_returning_root_correspondence(authenticated).disposition
        is CorrespondenceDisposition.TERMINAL_WITHOUT_DOWNSTREAM
    )
    mutated_commitment_digest = replace(
        authenticated,
        commitment_receipt_sha256="f" * 64,
    )
    assert (
        derive_returning_root_correspondence(mutated_commitment_digest).disposition
        is CorrespondenceDisposition.PROVENANCE_DIVERGENCE
    )
    assert owner.branch_target_roster_sha256 is None
    assert all(
        kind not in {
            "branch_target_roster",
            "seed_correspondence_map",
            "reconstruction_qa",
            "scientific_arm_execution",
        }
        for kind, _ in inspect_ledger(owner._root, allow_source_inflight=True).receipts
    )
    forged = replace(
        authenticated,
        termination_evidence=replace(
            evidence,
            terminal_call_ordinal=evidence.terminal_call_ordinal + 1,
        ),
    )
    assert (
        derive_returning_root_correspondence(forged).disposition
        is CorrespondenceDisposition.PROVENANCE_DIVERGENCE
    )
    owner.close()


def test_source_terminal_rejects_same_group_foreign_rollout(tmp_path: Path) -> None:
    value = _without_candidate(clean=True)
    owner, source_receipt, _, foreign_commitment = _durable_terminal_owner(
        tmp_path,
        value.trace,
    )
    with pytest.raises(ValueError, match="same rollout"):
        authenticate_returning_root_terminal_evidence(
            value.trace,
            trace_sha256=value.trace_sha256,
            source_id=value.source_id,
            source_sha256=value.source_sha256,
            trace_id=value.trace_id,
            group_id=value.group_id,
            rollout_id=value.rollout_id,
            target_id="target-foreign",
            target_ordinal=0,
            target_address=_address(1, "root/foreign", 0, 0),
            source_completion_receipt=source_receipt,
            target_commitment_receipt=foreign_commitment,
            terminal_owner=owner,
        )
    owner.close()


def test_terminal_mutations_fail_before_typed_terminal_disposition() -> None:
    value = _without_candidate(clean=True)
    for field, replacement in (("is_completed", False), ("stop_condition", None)):
        trace = copy.deepcopy(value.trace)
        trace[field] = replacement
        result = derive_returning_root_correspondence(_rehashed(value, trace))
        assert (
            result.disposition
            is not CorrespondenceDisposition.TERMINAL_WITHOUT_DOWNSTREAM
        ), field


def test_replay_compares_exact_provenance_including_parent_tool_call_id() -> None:
    value = _value()
    validate_replay_pair(source_input=value, replay_input=value)
    replay_trace = copy.deepcopy(value.trace)
    replay_trace["calls"][1]["rlm"]["parent_tool_call_id"] = "call-mutated"
    replay = _rehashed(value, replay_trace)
    with pytest.raises(ValueError, match="replay structural"):
        validate_replay_pair(source_input=value, replay_input=replay)


def test_live_receipt_recomputes_input_and_result_bytes() -> None:
    value = _value()
    payload: dict[str, Any] = _receipt(value)
    result = validate_live_correspondence_receipt(
        payload,
        evaluator_input=value,
    )
    assert result.disposition is CorrespondenceDisposition.ELIGIBLE_MATCH
    for field, replacement in (
        ("schema_version", 2),
        ("domain", "legacy"),
        ("contract_version", "legacy"),
        ("contract_sha256", "0" * 64),
        ("causal_graph_schema_sha256", "0" * 64),
        ("evaluator_input_sha256", "0" * 64),
        ("evaluator_result_sha256", "0" * 64),
    ):
        mutated = copy.deepcopy(payload)
        mutated[field] = replacement
        with pytest.raises(ValueError):
            validate_live_correspondence_receipt(
                mutated,
                evaluator_input=value,
            )
    input_mutations: tuple[tuple[str, object], ...] = (
        ("target_id", "mutated"),
        ("parent_lineage", "mutated"),
        ("target_address", None),
    )
    for input_field, input_replacement in input_mutations:
        input_mutated: dict[str, Any] = copy.deepcopy(payload)
        evaluator = cast(dict[str, Any], input_mutated["evaluator_input"])
        evaluator[input_field] = input_replacement
        with pytest.raises(ValueError):
            validate_live_correspondence_receipt(
                input_mutated,
                evaluator_input=value,
            )
    result_mutations: tuple[tuple[str, object], ...] = (
        ("disposition", CorrespondenceDisposition.NO_VALID_RETURN.value),
        ("matched_address", None),
        ("target_ordinal", 9),
    )
    for result_field, result_replacement in result_mutations:
        result_mutated: dict[str, Any] = copy.deepcopy(payload)
        evaluator_result = cast(dict[str, Any], result_mutated["evaluator_result"])
        evaluator_result[result_field] = result_replacement
        with pytest.raises(ValueError):
            validate_live_correspondence_receipt(
                result_mutated,
                evaluator_input=value,
            )
    with pytest.raises(ValueError, match="legacy or malformed"):
        validate_live_correspondence_receipt(
            {"schema_version": 1, "receipt_kind": "seed_correspondence_map"},
            evaluator_input=value,
        )
    with pytest.raises(ValueError, match="legacy or malformed"):
        validate_live_correspondence_receipt(
            {**payload, "unknown": None},
            evaluator_input=value,
        )


def test_contract_freezes_exact_schema_and_no_unknown_receipt_fields() -> None:
    payload = contract_artifact_payload()
    schemas = cast(dict[str, dict[str, Any]], payload["schemas"])
    assert set(schemas) == {
        "input",
        "authenticated_termination_evidence",
        "terminal_binding",
        "result_output",
        "replay_validation",
        "live_receipt",
        "evaluator_input_projection",
    }
    for schema in schemas.values():
        assert schema["additionalProperties"] is False
    assert set(cast(dict[str, str], payload["schema_sha256s"])) == set(schemas)
    assert payload["recorded_action_digest"] == {
        "phase": "phase-2-pending",
        "authenticated_in_phase_1": False,
        "used_by_returning_root_selection": False,
        "evaluator_projection_includes_field": False,
    }


def test_contract_schemas_are_resolvable_and_reject_mutations() -> None:
    from jsonschema import Draft202012Validator, ValidationError

    value = _value()
    projected = evaluator_input_payload(value)
    assert "recorded_action_digest" not in projected
    result = derive_returning_root_correspondence(value).to_payload()
    receipt = _receipt(value)
    schemas = cast(dict[str, dict[str, Any]], contract_artifact_payload()["schemas"])
    Draft202012Validator.check_schema(schemas["input"])
    Draft202012Validator.check_schema(schemas["evaluator_input_projection"])
    Draft202012Validator.check_schema(schemas["replay_validation"])
    Draft202012Validator.check_schema(schemas["live_receipt"])
    raw_input = {
        "source_id": value.source_id,
        "source_sha256": value.source_sha256,
        "trace_id": value.trace_id,
        "trace_sha256": value.trace_sha256,
        "trace": value.trace,
        "group_id": value.group_id,
        "rollout_id": value.rollout_id,
        "causal_graph_schema_sha256": value.causal_graph_schema_sha256,
        "target_ordinal": value.target_ordinal,
        "target_id": value.target_id,
        "target_address": {
            "depth": value.target_address.depth,
            "lineage": value.target_address.lineage,
            "session_call_ordinal": value.target_address.session_call_ordinal,
            "turn": value.target_address.turn,
            "call_kind": value.target_address.call_kind,
        },
        "spawn_ordinal": value.spawn_ordinal,
        "parent_lineage": value.parent_lineage,
        "commitment_receipt_sha256": value.commitment_receipt_sha256,
        "recorded_action_digest": value.recorded_action_digest,
        "termination_evidence": None,
    }
    Draft202012Validator(schemas["input"]).validate(raw_input)
    Draft202012Validator(schemas["evaluator_input_projection"]).validate(projected)
    Draft202012Validator(schemas["replay_validation"]).validate(
        {"source_input": projected, "replay_input": projected}
    )
    Draft202012Validator(schemas["live_receipt"]).validate(receipt)
    for schema_name, valid in (
        ("evaluator_input_projection", projected),
        ("result_output", result),
    ):
        mutated = copy.deepcopy(valid)
        mutated["unknown"] = None
        with pytest.raises(ValidationError):
            Draft202012Validator(schemas[schema_name]).validate(mutated)
