"""Versioned Stage D precommit for structurally addressed child policy calls."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json
from redco.env.tracer import EventNodeKind
from redco.integrations.signed_subprocess import sign_payload, verify_signed_payload
from redco.integrations.verifiers_provenance import import_trace_file
from redco.integrations.verifiers_trace import audit_trace_file, load_trace_records


def _sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _candidate_fields(call: Any, node_id: str) -> dict[str, Any]:
    """Return fields fixed before any counterfactual candidate is sampled."""
    return {
        "structural_event_address": node_id,
        "native_call_index_diagnostic_only": call.call_index,
        "agent_depth": call.agent_depth,
        "session_id": call.session_id,
        "turn_index": call.turn_index,
        "call_kind": call.call_kind,
        "parent_session_id": call.parent_session_id,
        "parent_turn_index": call.parent_turn_index,
        "parent_tool_call_id": call.parent_tool_call_id,
        "invocation_id": call.invocation_id,
        "prompt_token_ids": list(call.prompt_token_ids),
        "checkpoint_id": call.checkpoint_id,
        "decoding_config_hash": call.decoding_config_hash,
        "event_seed": call.event_seed,
    }


def precommit_all_depth_one_policy_targets(trace_path: Path) -> dict[str, Any]:
    """Commit every trainable depth-one policy action, excluding compaction."""
    native = load_trace_records(trace_path)
    audit = audit_trace_file(trace_path)
    provenance = import_trace_file(trace_path)
    if len(native) != 1 or len(provenance.traces) != 1:
        raise ValueError("v2 precommit requires exactly one trace")
    trace = provenance.traces[0]
    node_ids: dict[int, str] = {}
    for node in trace.graph.nodes.values():
        if node.kind is EventNodeKind.POLICY:
            call_index = node.metadata.get("call_index")
            if type(call_index) is int:
                node_ids[call_index] = node.node_id
    eligible = [
        call for call in audit.calls if call.agent_depth == 1 and call.call_kind == "policy"
    ]
    if any(call.call_index not in node_ids for call in eligible):
        raise ValueError("eligible child policy call lacks a policy graph node")
    candidates = [_candidate_fields(call, node_ids[call.call_index]) for call in eligible]
    for candidate in candidates:
        candidate["pre_action_rank_sha256"] = _sha256(
            {
                key: value
                for key, value in candidate.items()
                if key != "native_call_index_diagnostic_only"
            }
        )
    candidates.sort(key=lambda row: row["pre_action_rank_sha256"])
    count = len(candidates)
    for candidate in candidates:
        candidate["decision_unit_weight"] = {
            "numerator": 1,
            "denominator": count,
        }
    raw_trace = native[0]
    task = (raw_trace.get("task") or {}).get("data") or {}
    source_trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    return sign_payload(
        {
            "schema_version": 2,
            "analysis": "stage-d-all-child-policy-precommit-v2",
            "trace_id": raw_trace.get("id"),
            "source_trace_sha256": source_trace_sha256,
            "paper_id": task.get("paper_id"),
            "selector": "all-depth-one-policy-precommitted-v2",
            "selection_time": ("after recorded rollout, before any counterfactual candidate"),
            "excluded_call_kinds": ["compaction"],
            "candidate_count": count,
            "candidate_set_sha256": _sha256(candidates),
            "outer_decision_unit_weight_sum": {
                "numerator": count,
                "denominator": count if count else 1,
            },
            "candidates": candidates,
        }
    )


def verify_canonical_precommit_v2(trace_path: Path, committed: dict[str, Any]) -> dict[str, Any]:
    """Verify exact equality with the complete versioned policy target set."""
    verify_signed_payload(committed)
    expected = precommit_all_depth_one_policy_targets(trace_path)
    if canonical_json(committed) != canonical_json(expected):
        raise ValueError("precommit is not the canonical v2 child policy set")
    if int(committed["candidate_count"]) < 1:
        raise ValueError("v2 precommit contains no child policy target")
    return expected
