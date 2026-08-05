"""Durable action, call, and downstream invariant checks for the v12 audit."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_exact_action import BehaviorAction
from redco.analysis.stage_d_source_producer import (
    _CALL_FIELDS,
    _TRACE_FIELDS,
    _checkpoint_claims,
    _float_list,
    _integer_list,
    _node_fields_are_pinned,
    _normalize_openai_message,
    _path_to_node,
)
from redco.analysis.stage_d_v12_audit_common import (
    _ABSENT,
    _METHOD,
    _STATUS,
    _bounded,
    _canonical,
    _mapping,
    _message_audit,
    _normalize_tools_for_audit,
    _status,
    sha256_bytes,
)
from redco.analysis.stage_d_v12_audit_inputs import _receipt_records


def _address_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        value.get("depth"),
        value.get("lineage"),
        value.get("session_call_ordinal"),
        value.get("turn"),
    )


def _action_by_address(
    root: Path,
) -> tuple[
    dict[tuple[Any, ...], tuple[dict[str, Any], dict[str, Any], BehaviorAction]],
    list[dict[str, Any]],
]:
    """Verify durable action/evidence hashes and bind actions to addresses."""
    records = _receipt_records(root)
    reservations: dict[str, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    for record in records:
        if record.get("record_kind") != "receipt":
            continue
        receipt = _mapping(record.get("body", {}).get("receipt"), "receipt")
        kind = receipt.get("receipt_kind")
        decision_id = receipt.get("decision_id")
        if kind == "source_policy_call_reserved" and isinstance(decision_id, str):
            reservations[decision_id] = receipt
        elif kind == "source_policy_call_completed":
            completed.append(receipt)
    if len(reservations) != 4 or len(completed) != 4:
        raise ValueError("terminal ledger does not contain four reserved/completed calls")

    result: dict[tuple[Any, ...], tuple[dict[str, Any], dict[str, Any], BehaviorAction]] = {}
    hash_checks: list[dict[str, Any]] = []
    for receipt in completed:
        decision_id = receipt.get("decision_id")
        if not isinstance(decision_id, str):
            raise ValueError("completed action lacks a decision identity")
        reservation = reservations.get(decision_id)
        if reservation is None:
            raise ValueError("completed action lacks its durable reservation")
        address = _mapping(reservation.get("target_address"), "target_address")
        response_digest = receipt.get("response_sha256")
        raw_response_digest = receipt.get("raw_response_sha256")
        if not isinstance(response_digest, str) or not isinstance(raw_response_digest, str):
            raise ValueError("completed action lacks durable response hashes")
        action_path = root / "ledger" / "evidence" / response_digest
        raw_path = root / "ledger" / "evidence" / raw_response_digest
        if not action_path.is_file() or not raw_path.is_file():
            raise ValueError("completed action references missing evidence")
        action_bytes = action_path.read_bytes()
        raw_response_bytes = raw_path.read_bytes()
        if sha256_bytes(action_bytes) != response_digest:
            raise ValueError("action response evidence filename/hash disagrees")
        if sha256_bytes(raw_response_bytes) != raw_response_digest:
            raise ValueError("raw response evidence filename/hash disagrees")
        envelope = _mapping(json.loads(action_bytes), "behavior action envelope")
        action_payload = _mapping(envelope.get("action"), "behavior action")
        key_payload = _mapping(action_payload.get("key"), "exact action key")
        prompt = key_payload.get("prompt_token_ids")
        if not isinstance(prompt, list) or any(type(item) is not int for item in prompt):
            raise ValueError("behavior action lacks valid prompt token IDs")

        def _render_prompt(
            _request: Mapping[str, Any], prompt_tokens: tuple[int, ...] = tuple(prompt)
        ) -> list[int]:
            return list(prompt_tokens)

        action = BehaviorAction.from_bytes(
            action_bytes,
            validate_action=lambda _request, _message, _tokens: None,
            render_prompt=_render_prompt,
        )
        if action.to_bytes() != action_bytes:
            raise ValueError("behavior action bytes are not a stable canonical round-trip")
        if envelope.get("digest") != action.digest:
            raise ValueError("behavior action digest disagrees with its payload")
        if receipt.get("action_digest") != action.digest:
            raise ValueError("completion receipt action digest disagrees with evidence")
        if receipt.get("request_sha256") is not None and receipt.get(
            "request_sha256"
        ) != reservation.get("request_sha256"):
            raise ValueError("completion/reservation request hashes disagree")
        if action.key.request_sha256 != reservation.get("request_sha256"):
            raise ValueError("exact action key request hash disagrees with reservation")
        if receipt.get("evidence_refs") is not None:
            record_body = _mapping(record.get("body"), "record body")
            refs = record_body.get("evidence_refs", [])
            if (
                not isinstance(refs, list)
                or response_digest not in refs
                or raw_response_digest not in refs
            ):
                raise ValueError("completion record omits an action evidence reference")
        action_message_hash = sha256_bytes(_canonical(action.message))
        if action_message_hash != action.raw_transport_message_sha256:
            raise ValueError("action raw transport message hash disagrees")
        identity_receipt = dict(receipt)
        identity_receipt["target_address"] = address
        identity_receipt["target_id"] = reservation.get("target_id")
        identity_receipt["target_ordinal"] = reservation.get("target_ordinal")
        identity_receipt["reservation_receipt"] = _canonical(reservation)
        identity_receipt["completion_receipt"] = _canonical(receipt)
        identity_receipt["request_sha256"] = reservation.get("request_sha256")
        key = _address_key(address)
        if key in result:
            raise ValueError("two completed actions share one durable address")
        result[key] = (identity_receipt, envelope, action)
        hash_checks.append(
            {
                "decision_id": decision_id,
                "address": address,
                "action_evidence_sha256": response_digest,
                "raw_response_evidence_sha256": raw_response_digest,
                "action_digest": action.digest,
                "action_raw_transport_message_sha256": action.raw_transport_message_sha256,
                "status": "pass",
            }
        )
    return result, hash_checks


def _checkpoint_claims_for_audit(trace: Mapping[str, Any]) -> set[str]:
    return set(_checkpoint_claims(trace))


def _call_audit(
    trace: Mapping[str, Any],
    call: Mapping[str, Any],
    *,
    call_index: int,
    nodes: list[dict[str, Any]],
    action_entry: tuple[dict[str, Any], dict[str, Any], BehaviorAction],
) -> dict[str, Any]:
    receipt, action_envelope, _behavior_action = action_entry
    action = _mapping(action_envelope.get("action"), "behavior action")
    key = _mapping(action.get("key"), "exact action key")
    rlm = _mapping(call.get("rlm"), "trace call rlm")
    node_index = call.get("node")
    if not isinstance(node_index, int) or not 0 <= node_index < len(nodes):
        raise ValueError("trace call names an absent node")
    node = nodes[node_index]
    transport_message = action.get("raw_transport_message", _ABSENT)
    trace_message = node.get("message", _ABSENT)
    message_result = _message_audit(transport_message, trace_message)

    path: list[int] = []
    path_error: str | None = None
    try:
        path = _path_to_node(nodes, node_index)
    except Exception as error:
        path_error = f"{type(error).__name__}: {error}"
    prompt_tokens: list[int] = []
    action_tokens_from_trace: list[int] = []
    action_logprobs_from_trace: list[float] = []
    try:
        if not path:
            raise ValueError("sampled node path is empty")
        node_tokens = _integer_list(node.get("token_ids"), "sampled node token_ids")
        node_mask = node.get("mask")
        if not isinstance(node_mask, list) or any(type(item) is not bool for item in node_mask):
            raise ValueError("sampled node mask is invalid")
        first_sampled = next(
            (index for index, selected in enumerate(node_mask) if selected),
            len(node_mask),
        )
        prompt_tokens = [
            token
            for prior in path[:-1]
            for token in _integer_list(nodes[prior].get("token_ids"), "prompt token_ids")
        ] + list(node_tokens[:first_sampled])
        action_tokens_from_trace = list(node_tokens[first_sampled:])
        action_logprobs_from_trace = list(
            _float_list(node.get("logprobs"), "sampled node logprobs")
        )
    except Exception as error:
        path_error = f"{type(error).__name__}: {error}"

    request = _mapping(key.get("request"), "exact request")
    request_messages = request.get("messages", _ABSENT)
    request_tools = request.get("tools", _ABSENT)
    graph_messages = [nodes[index].get("message", _ABSENT) for index in path[:-1]]
    request_message_error: str | None = None
    graph_message_error: str | None = None
    normalized_request_messages: list[dict[str, Any]] = []
    normalized_graph_messages: list[dict[str, Any]] = []
    try:
        if not isinstance(request_messages, list):
            raise ValueError("request messages are not a list")
        normalized_request_messages = [_normalize_openai_message(item) for item in request_messages]
    except Exception as error:
        request_message_error = f"{type(error).__name__}: {error}"
    try:
        normalized_graph_messages = [_normalize_openai_message(item) for item in graph_messages]
    except Exception as error:
        graph_message_error = f"{type(error).__name__}: {error}"
    normalized_request_tools, request_tools_error = _normalize_tools_for_audit(request_tools)
    normalized_trace_tools, trace_tools_error = _normalize_tools_for_audit(trace.get("tools"))

    sampler_config = _mapping(key.get("sampler_config"), "sampler config")
    expected_sampling = {
        "temperature": sampler_config.get("temperature"),
        "top_p": sampler_config.get("top_p"),
        "reasoning_effort": None,
        "max_tokens": sampler_config.get("max_tokens"),
        "parallel_tool_calls": False,
        "seed": sampler_config.get("seed"),
        "tool_choice": sampler_config.get("tool_choice"),
    }
    sampling = call.get("sampling", _ABSENT)
    usage = call.get("usage", _ABSENT)
    expected_usage = {
        "prompt_tokens": action.get("prompt_tokens"),
        "completion_tokens": action.get("completion_tokens"),
        "cached_input_tokens": None,
        "reasoning_tokens": None,
        "cost": None,
    }
    call_fields_status: _STATUS
    call_fields_detail: str | None = None
    if set(call) == _CALL_FIELDS:
        call_fields_status = "pass"
    elif set(call) == _CALL_FIELDS - {"error"}:
        call_fields_status = "not_observable_from_persisted_schema"
        call_fields_detail = (
            "nullable error field was omitted by the persisted Verifiers call schema"
        )
    else:
        call_fields_status = "fail"
        call_fields_detail = "persisted call fields differ beyond the known nullable omission"

    def _nullable_mapping_status(
        observed: object,
        expected: dict[str, Any],
        name: str,
    ) -> tuple[_STATUS, str | None]:
        if not isinstance(observed, dict):
            return "fail", f"{name} is absent or not an object"
        if observed == expected:
            return "pass", None
        nullable_fields = {field for field, value in expected.items() if value is None}
        if set(observed) == set(expected) - nullable_fields and all(
            observed.get(field) == expected[field] for field in observed
        ):
            return (
                "not_observable_from_persisted_schema",
                f"nullable {name} fields were omitted by the persisted Verifiers schema",
            )
        return "fail", f"{name} differs from the exact action evidence"

    sampler_status, sampler_detail = _nullable_mapping_status(
        sampling, expected_sampling, "sampler"
    )
    usage_status, usage_detail = _nullable_mapping_status(usage, expected_usage, "usage")
    error_status: _STATUS
    error_detail: str | None
    if "error" not in call:
        error_status, error_detail = (
            "not_observable_from_persisted_schema",
            "nullable error field was omitted by the persisted Verifiers call schema",
        )
    elif call.get("error") is None:
        error_status, error_detail = "pass", None
    else:
        error_status, error_detail = "fail", "persisted call contains an error"

    request_sha = key.get("request_sha256")
    request_hash_status: _STATUS = (
        "pass"
        if isinstance(request_sha, str)
        and request_sha == receipt.get("request_sha256")
        and request_sha == sha256_bytes(_canonical(request))
        else "fail"
    )
    max_tokens = sampler_config.get("max_tokens")
    completion_tokens = action.get("completion_tokens")
    finish_reason = action.get("finish_reason")
    token_cap_status: _STATUS = (
        "pass"
        if type(max_tokens) is int
        and type(completion_tokens) is int
        and max_tokens == 768
        and completion_tokens <= 768
        and (finish_reason != "length" or completion_tokens == max_tokens)
        else "fail"
    )
    call_fields_method: _METHOD = (
        "not_observable_from_persisted_schema"
        if call_fields_status == "not_observable_from_persisted_schema"
        else "directly_verified_from_archive"
    )
    error_method: _METHOD = (
        "not_observable_from_persisted_schema"
        if error_status == "not_observable_from_persisted_schema"
        else "directly_verified_from_archive"
    )
    invariant_results = [
        _status(
            "call_fields_exact", call_fields_status, call_fields_method, detail=call_fields_detail
        ),
        _status(
            "structural_address",
            "pass"
            if _address_key(rlm) == _address_key(receipt.get("target_address", {}))
            else "fail",
            "directly_verified_from_archive",
        ),
        _status(
            "node_index_and_sampled_node",
            "pass" if _node_fields_are_pinned(node) and node.get("sampled") is True else "fail",
            "directly_verified_from_archive",
        ),
        _status(
            "prompt_action_logprob_streams",
            "pass"
            if path_error is None
            and prompt_tokens == list(key.get("prompt_token_ids", []))
            and action_tokens_from_trace == list(action.get("action_token_ids", []))
            and action_logprobs_from_trace == list(action.get("behavior_logprobs", []))
            else "fail",
            "directly_verified_from_archive",
            detail=path_error,
        ),
        _status(
            "message_comparison",
            "pass" if message_result["canonical_equal_under_current_finalizer"] else "fail",
            "directly_verified_from_archive",
        ),
        _status(
            "request_object_and_messages",
            "pass" if request_message_error is None else "fail",
            "directly_verified_from_archive",
            detail=request_message_error,
        ),
        _status(
            "request_context_messages",
            "pass"
            if request_message_error is None
            and graph_message_error is None
            and normalized_request_messages == normalized_graph_messages
            else "fail",
            "directly_verified_from_archive",
            detail=graph_message_error,
        ),
        _status(
            "request_context_tools",
            "pass"
            if request_tools_error is None
            and trace_tools_error is None
            and normalized_request_tools == normalized_trace_tools
            else "fail",
            "directly_verified_from_archive",
            detail=request_tools_error or trace_tools_error,
        ),
        _status("request_hash_binding", request_hash_status, "directly_verified_from_archive"),
        _status(
            "model_identity",
            "pass" if call.get("model") == key.get("checkpoint_id") else "fail",
            "directly_verified_from_archive",
        ),
        _status(
            "checkpoint_claims",
            "pass"
            if not _checkpoint_claims_for_audit(trace)
            or _checkpoint_claims_for_audit(trace) == {key.get("checkpoint_id")}
            else "fail",
            "directly_verified_from_archive",
        ),
        _status(
            "sampler",
            sampler_status,
            "not_observable_from_persisted_schema"
            if sampler_status == "not_observable_from_persisted_schema"
            else "directly_verified_from_archive",
            detail=sampler_detail,
        ),
        _status(
            "usage",
            usage_status,
            "not_observable_from_persisted_schema"
            if usage_status == "not_observable_from_persisted_schema"
            else "directly_verified_from_archive",
            detail=usage_detail,
        ),
        _status("successful_call_error", error_status, error_method, detail=error_detail),
        _status(
            "finish_reason",
            "pass" if call.get("finish_reason") == action.get("finish_reason") else "fail",
            "directly_verified_from_archive",
        ),
        _status("token_cap", token_cap_status, "directly_verified_from_archive"),
        _status(
            "endpoint",
            "pass" if call.get("endpoint") == "/chat/completions" else "fail",
            "directly_verified_from_archive",
        ),
    ]
    return {
        "call_index": call_index,
        "request_sequence": receipt.get("request_sequence"),
        "decision_id": receipt.get("decision_id"),
        "lineage": rlm.get("lineage"),
        "depth": rlm.get("depth"),
        "node": node_index,
        "evidence_sha256": receipt.get("response_sha256"),
        "evidence_identity": {
            "action_raw_message_sha256": action.get("raw_transport_message_sha256"),
            "decision_id": receipt.get("decision_id"),
            "request_sha256": receipt.get("request_sha256"),
        },
        "finish_reason": action.get("finish_reason"),
        "completion_tokens": action.get("completion_tokens"),
        "termination_kind": action.get("termination_kind"),
        "message": message_result,
        "invariants": invariant_results,
        "invariant_summary": {
            "pass_count": sum(item["status"] == "pass" for item in invariant_results),
            "fail_count": sum(item["status"] == "fail" for item in invariant_results),
            "not_observable_count": sum(
                item["status"] == "not_observable_from_persisted_schema"
                for item in invariant_results
            ),
        },
        "bounded_subvalue_hashes": {
            "transport_message": _bounded(transport_message),
            "trace_message": _bounded(trace_message),
            "request": _bounded(request),
            "sampling": _bounded(sampling),
            "usage": _bounded(usage),
        },
    }


def _post_call_invariants(
    trace: Mapping[str, Any],
    calls: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    action_map: Mapping[tuple[Any, ...], tuple[dict[str, Any], dict[str, Any], BehaviorAction]],
    ledger_status: str,
    ledger_reason: str | None,
    abort_receipts: list[dict[str, Any]],
    source_counts: tuple[int, int],
    trace_artifact_hash_status: _STATUS,
    abort_error_status: _STATUS,
) -> list[dict[str, Any]]:
    call_nodes: list[int] = []
    for call in calls:
        node = call.get("node")
        if type(node) is not int:
            raise ValueError("trace call has a non-integer node address")
        call_nodes.append(node)
    sampled_nodes = [index for index, node in enumerate(nodes) if node.get("sampled") is True]
    address_keys = [_address_key(_mapping(call.get("rlm"), "trace call rlm")) for call in calls]
    return [
        _status(
            "trace_fields_exact",
            "pass" if set(trace) == _TRACE_FIELDS else "fail",
            "directly_verified_from_archive",
        ),
        _status(
            "trace_artifact_hash_binding",
            trace_artifact_hash_status,
            "directly_verified_from_archive",
        ),
        _status(
            "trace_success_state",
            "pass"
            if trace.get("is_completed") is True
            and trace.get("ok") is True
            and trace.get("stop_condition") == "max_total_tokens"
            and trace.get("errors") == []
            else "fail",
            "directly_verified_from_archive",
        ),
        _status(
            "sampled_nodes_biject_calls",
            "pass"
            if sorted(call_nodes) == sampled_nodes and len(set(call_nodes)) == len(call_nodes)
            else "fail",
            "directly_verified_from_archive",
        ),
        _status(
            "parent_graph_paths",
            "pass" if all(_path_to_node(nodes, node) for node in call_nodes) else "fail",
            "reconstructed_on_disposable_copy",
        ),
        _status(
            "durable_address_mapping",
            "pass"
            if len(action_map) == len(calls) and len(set(address_keys)) == len(calls)
            else "fail",
            "directly_verified_from_archive",
        ),
        _status(
            "trace_reward_and_metadata",
            "pass"
            if isinstance(trace.get("rewards"), dict)
            and isinstance(trace.get("info"), dict)
            and isinstance(trace.get("agent"), dict)
            else "fail",
            "directly_verified_from_archive",
        ),
        _status(
            "ledger_terminal_poison",
            "pass"
            if ledger_status == "poisoned"
            and ledger_reason == "ledger records an aborted source rollout finalization"
            else "fail",
            "directly_verified_from_archive",
        ),
        _status(
            "exactly_one_finalization_abort",
            "pass" if len(abort_receipts) == 1 else "fail",
            "directly_verified_from_archive",
        ),
        _status(
            "finalization_abort_error_digest", abort_error_status, "directly_verified_from_archive"
        ),
        _status(
            "source_artifacts_absent",
            "pass" if source_counts == (0, 0) else "fail",
            "directly_verified_from_archive",
        ),
        _status(
            "source_eligibility_not_recovered",
            "pass" if source_counts == (0, 0) else "fail",
            "directly_verified_from_archive",
            detail=(
                "no committed or pending source artifact exists in the immutable archive"
                if source_counts == (0, 0)
                else "source artifact evidence exists in the immutable archive"
            ),
        ),
    ]
