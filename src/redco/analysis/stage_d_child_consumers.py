"""Diagnose how recorded Stage D child outputs surface in root tool turns."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import sign_payload
from redco.integrations.verifiers_trace import (
    RecordedPolicyCall,
    audit_trace_file,
    load_trace_records,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _message(nodes: list[dict[str, Any]], node_index: int) -> dict[str, Any]:
    message = nodes[node_index].get("message")
    if not isinstance(message, dict):
        raise TypeError(f"node {node_index} has no message object")
    return message


def _content(nodes: list[dict[str, Any]], node_index: int) -> str:
    content = _message(nodes, node_index).get("content")
    if not isinstance(content, str):
        raise TypeError(f"node {node_index} has no string message content")
    return content


def classify_child_consumption(
    *,
    child_text: str,
    child_call_index: int,
    duplicate_call_indices: tuple[int, ...],
    parent_tool_content: str,
    other_tool_contents: tuple[str, ...],
) -> dict[str, Any]:
    """Classify exact, escaped, hidden, and duplicate child-result surfaces."""
    escaped = repr(child_text)[1:-1]
    escaped_differs = escaped != child_text
    parent_exact = parent_tool_content.count(child_text)
    parent_escaped = (
        parent_tool_content.count(escaped) if escaped_differs else 0
    )
    other_exact = [content.count(child_text) for content in other_tool_contents]
    other_escaped = [
        content.count(escaped) if escaped_differs else 0
        for content in other_tool_contents
    ]
    duplicated = len(duplicate_call_indices) > 1

    if parent_exact:
        classification = (
            "exact_surface_ambiguous_duplicate"
            if duplicated
            else "exact_surface"
        )
    elif parent_escaped:
        classification = (
            "escaped_surface_ambiguous_duplicate"
            if duplicated
            else "escaped_surface"
        )
    elif sum(other_exact) + sum(other_escaped):
        classification = (
            "duplicate_alias_elsewhere"
            if duplicated
            else "surfaced_only_outside_parent_turn"
        )
    else:
        classification = "no_serialized_surface_observed"

    return {
        "call_index": child_call_index,
        "classification": classification,
        "duplicate_action_call_indices": list(duplicate_call_indices),
        "action_text_sha256": _sha256_text(child_text),
        "action_text_characters": len(child_text),
        "action_text_line_breaks": child_text.count("\n"),
        "parent_tool_exact_count": parent_exact,
        "parent_tool_escaped_repr_count": parent_escaped,
        "other_root_tool_exact_counts": other_exact,
        "other_root_tool_escaped_repr_counts": other_escaped,
    }


def _root_tool_nodes(
    *,
    calls: tuple[RecordedPolicyCall, ...],
    nodes: list[dict[str, Any]],
    required_parent_keys: set[tuple[str, int]],
) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    for call in calls:
        if call.agent_depth != 0:
            continue
        if call.session_id is None or call.turn_index is None:
            raise ValueError("root call lacks session or turn provenance")
        key = (call.session_id, call.turn_index)
        if key not in required_parent_keys:
            continue
        candidates = [
            index
            for index, node in enumerate(nodes)
            if node.get("parent") == call.node_index
            and _message(nodes, index).get("role") == "tool"
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"root call {call.call_index} has {len(candidates)} tool responses"
            )
        result[key] = candidates[0]
    return result


def analyze(trace_path: Path) -> dict[str, Any]:
    """Return a signed, action-addressed consumer diagnostic for one trace."""
    records = load_trace_records(trace_path)
    if len(records) != 1:
        raise ValueError("consumer diagnostic requires exactly one trace")
    trace = records[0]
    nodes_value = trace.get("nodes")
    if not isinstance(nodes_value, list) or not all(
        isinstance(node, dict) for node in nodes_value
    ):
        raise TypeError("trace.nodes must be a list of objects")
    nodes: list[dict[str, Any]] = nodes_value
    audit = audit_trace_file(trace_path)
    calls = tuple(sorted(audit.calls, key=lambda call: call.call_index))
    child_calls = tuple(call for call in calls if call.agent_depth == 1)
    if not child_calls:
        raise ValueError("trace has no depth-one child calls")
    required_parent_keys = {
        (call.parent_session_id, call.parent_turn_index)
        for call in child_calls
        if call.parent_session_id is not None
        and call.parent_turn_index is not None
    }
    tool_nodes = _root_tool_nodes(
        calls=calls,
        nodes=nodes,
        required_parent_keys=required_parent_keys,
    )
    all_tool_node_indices = tuple(tool_nodes.values())
    final_tool_node = tool_nodes[max(tool_nodes, key=lambda key: key[1])]

    calls_by_action: dict[str, list[int]] = defaultdict(list)
    action_by_call: dict[int, str] = {}
    for call in child_calls:
        action = _content(nodes, call.node_index)
        action_by_call[call.call_index] = action
        calls_by_action[_sha256_text(action)].append(call.call_index)

    reports: list[dict[str, Any]] = []
    for call in child_calls:
        if call.parent_session_id is None or call.parent_turn_index is None:
            raise ValueError(f"child call {call.call_index} lacks parent provenance")
        parent_key = (call.parent_session_id, call.parent_turn_index)
        if parent_key not in tool_nodes:
            raise ValueError(
                f"child call {call.call_index} has no addressed root tool response"
            )
        parent_tool_node = tool_nodes[parent_key]
        other_nodes = tuple(
            node for node in all_tool_node_indices if node != parent_tool_node
        )
        action = action_by_call[call.call_index]
        report = classify_child_consumption(
            child_text=action,
            child_call_index=call.call_index,
            duplicate_call_indices=tuple(
                calls_by_action[_sha256_text(action)]
            ),
            parent_tool_content=_content(nodes, parent_tool_node),
            other_tool_contents=tuple(_content(nodes, node) for node in other_nodes),
        )
        report.update(
            {
                "session_id": call.session_id,
                "parent_session_id": call.parent_session_id,
                "parent_turn_index": call.parent_turn_index,
                "parent_tool_node_index": parent_tool_node,
                "other_root_tool_node_indices": list(other_nodes),
                "final_root_tool_exact_count": _content(
                    nodes, final_tool_node
                ).count(action),
            }
        )
        reports.append(report)

    counts: dict[str, int] = defaultdict(int)
    for report in reports:
        counts[str(report["classification"])] += 1
    replay_order = [call.call_index for call in child_calls]
    first_replace_unique_failure = next(
        (
            int(report["call_index"])
            for report in reports
            if int(report["final_root_tool_exact_count"]) != 1
        ),
        None,
    )
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-child-consumer-provenance-v1",
            "source_trace": trace_path.as_posix(),
            "source_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "trace_id": str(trace.get("id")),
            "root_tool_turn_count": len(tool_nodes),
            "child_call_count": len(child_calls),
            "native_child_replay_order": replay_order,
            "first_replace_unique_failure_in_native_order": (
                first_replace_unique_failure
            ),
            "classification_counts": dict(sorted(counts.items())),
            "child_consumers": reports,
            "interpretation": {
                "current_adapter_contract": (
                    "Each child action must occur exactly once as raw text in the "
                    "final root tool-response message."
                ),
                "observed_violation": (
                    "The multi-turn IPython trace contains absent serialized "
                    "surfaces, escaped values, and duplicate/aliased child results."
                ),
                "disposition": (
                    "Use structured child-result provenance or event/state-aware "
                    "replay; do not relax replace_unique or drop committed targets."
                ),
            },
        }
    )
