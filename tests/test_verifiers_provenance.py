from __future__ import annotations

import json
from pathlib import Path

from redco.analysis.recorded_raf import build_recorded_raf_projection
from redco.analysis.verifiers_provenance import build_provenance_report
from redco.contracts import PolicyNodeKind
from redco.env.tracer import EdgeKind
from redco.integrations.verifiers_provenance import (
    import_trace,
    import_trace_file,
)


def _node(
    parent: int | None,
    tokens: list[int],
    mask: list[bool],
    *,
    timestamp: float,
    role: str,
) -> dict[str, object]:
    return {
        "parent": parent,
        "message": {"role": role, "content": role},
        "sampled": any(mask),
        "timestamp": timestamp,
        "token_ids": tokens,
        "mask": mask,
        "logprobs": [0.0 for sampled in mask if sampled],
    }


def _trace() -> dict[str, object]:
    return {
        "id": "recursive-trace",
        "ok": True,
        "errors": [],
        "agent": {"model": "model-a"},
        "nodes": [
            _node(None, [1], [False], timestamp=0.0, role="system"),
            _node(0, [2], [False], timestamp=0.1, role="user"),
            _node(1, [3, 10], [False, True], timestamp=1.0, role="assistant"),
            _node(None, [4], [False], timestamp=2.5, role="system"),
            _node(3, [5], [False], timestamp=2.6, role="user"),
            _node(4, [6, 20], [False, True], timestamp=4.0, role="assistant"),
            _node(2, [7], [False], timestamp=4.5, role="tool"),
            _node(6, [8, 30], [False, True], timestamp=5.0, role="assistant"),
        ],
        "calls": [
            {
                "node": 2,
                "model": "model-a",
                "sampling": {"temperature": 0.7, "seed": 7},
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                "time": {"start": 1.0, "end": 2.0},
            },
            {
                "node": 5,
                "model": "model-a",
                "sampling": {"temperature": 0.7, "seed": 7},
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "time": {"start": 3.0, "end": 4.0},
            },
            {
                "node": 7,
                "model": "model-a",
                "sampling": {"temperature": 0.7, "seed": 7},
                "usage": {"prompt_tokens": 6, "completion_tokens": 3},
                "time": {"start": 5.0, "end": 6.0},
            },
        ],
        "rewards": {"reward": 1.0},
        "timing": {"generation": {"start": 0.5, "end": 6.5}},
    }


def test_import_maps_exact_prompts_and_marks_cross_component_fallback() -> None:
    result = import_trace(_trace(), source_bytes=123)

    assert result.exact_prompt_provenance_coverage == 1.0
    assert result.rendered_prompt_tokens == 12
    assert result.mapped_prompt_tokens == 12
    assert result.connected_components == 2
    assert result.recursive_components == 1
    assert result.exact_cross_component_links == 0
    assert result.structurally_identified_model_calls == 0
    assert result.recursive_model_calls == 0
    assert result.exactly_parented_recursive_model_calls == 0
    assert result.cross_component_fallbacks == 1
    assert result.cross_component_fallback_rate == 1.0
    assert result.unresolved_cross_component_links == 0
    assert result.cost.prompt_tokens == 12
    assert result.cost.generated_tokens == 6
    assert result.cost.model_call_wall_seconds == 3.0
    assert result.cost.trace_generation_wall_seconds == 6.0
    assert result.cost.storage_bytes == 123
    assert [record.kind for record in result.policy_records] == [
        PolicyNodeKind.ROOT_TURN,
        PolicyNodeKind.SUBCALL_OUTPUT,
        PolicyNodeKind.ROOT_TURN,
    ]

    edges = {
        (edge.source, edge.target, edge.kind, edge.via)
        for edge in result.graph.edges
    }
    assert (
        "recursive-trace:policy:0",
        "recursive-trace:message:3",
        EdgeKind.CALL,
        "cross_component_all_prior_temporal_fallback",
    ) in edges
    assert (
        "recursive-trace:message:5",
        "recursive-trace:policy:2",
        EdgeKind.CALL,
        "cross_component_all_later_temporal_fallback",
    ) in edges
    result.graph.assert_acyclic()


def test_file_report_is_diagnostic_until_explicit_call_links_exist(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "episode",
                "ok": True,
                "errors": [],
                "traces": [_trace()],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    imported = import_trace_file(path)
    report = build_provenance_report(path)

    assert imported.exact_prompt_provenance_coverage == 1.0
    assert not imported.ready_for_representative_raf
    assert report["completed"]
    assert not report["ready_for_representative_raf"]
    assert report["blocking_finding"] is not None


def test_structural_session_turns_remove_cross_component_fallback() -> None:
    trace = _trace()
    calls = trace["calls"]
    assert isinstance(calls, list)
    structures = [
        {
            "depth": 0,
            "session_id": "root",
            "turn": 0,
            "call_kind": "policy",
        },
        {
            "depth": 1,
            "session_id": "sub-child",
            "turn": 0,
            "call_kind": "policy",
            "parent_session_id": "root",
            "parent_turn": 0,
        },
        {
            "depth": 0,
            "session_id": "root",
            "turn": 1,
            "call_kind": "policy",
        },
    ]
    for call, structure in zip(calls, structures, strict=True):
        assert isinstance(call, dict)
        call["rlm"] = structure
    nodes = trace["nodes"]
    assert isinstance(nodes, list)
    tool_message = nodes[6]["message"]
    assert isinstance(tool_message, dict)
    tool_message["tool_call_id"] = "call_0"
    structures[1]["parent_tool_call_id"] = "call_0"
    structures[1]["invocation_id"] = "shard-0"

    result = import_trace(trace)
    edges = {
        (edge.source, edge.target, edge.kind, edge.via)
        for edge in result.graph.edges
    }

    assert result.exact_cross_component_links == 1
    assert result.structurally_identified_model_calls == 3
    assert result.recursive_model_calls == 1
    assert result.exactly_parented_recursive_model_calls == 1
    assert result.cross_component_fallbacks == 0
    assert result.cross_component_fallback_rate == 0.0
    assert result.unresolved_cross_component_links == 0
    assert (
        "recursive-trace:policy:0",
        "recursive-trace:message:3",
        EdgeKind.CALL,
        "cross_component_parent_invocation",
    ) in edges
    assert (
        "recursive-trace:message:5",
        "recursive-trace:message:6",
        EdgeKind.CALL,
        "cross_component_return_parent_tool_call",
    ) in edges


def test_session_turn_without_invocation_address_is_not_exact() -> None:
    trace = _trace()
    calls = trace["calls"]
    assert isinstance(calls, list)
    structures = [
        {
            "depth": 0,
            "session_id": "root",
            "turn": 0,
            "call_kind": "policy",
        },
        {
            "depth": 1,
            "session_id": "sub-child",
            "turn": 0,
            "call_kind": "policy",
            "parent_session_id": "root",
            "parent_turn": 0,
        },
        {
            "depth": 0,
            "session_id": "root",
            "turn": 1,
            "call_kind": "policy",
        },
    ]
    for call, structure in zip(calls, structures, strict=True):
        assert isinstance(call, dict)
        call["rlm"] = structure

    result = import_trace(trace)

    assert result.exact_cross_component_links == 0
    assert result.exactly_parented_recursive_model_calls == 0
    assert result.cross_component_fallbacks == 1


def test_structural_file_report_is_ready_for_representative_raf(
    tmp_path: Path,
) -> None:
    trace = _trace()
    calls = trace["calls"]
    assert isinstance(calls, list)
    structures = [
        {
            "depth": 0,
            "session_id": "root",
            "turn": 0,
            "call_kind": "policy",
        },
        {
            "depth": 1,
            "session_id": "sub-child",
            "turn": 0,
            "call_kind": "policy",
            "parent_session_id": "root",
            "parent_turn": 0,
        },
        {
            "depth": 0,
            "session_id": "root",
            "turn": 1,
            "call_kind": "policy",
        },
    ]
    for call, structure in zip(calls, structures, strict=True):
        assert isinstance(call, dict)
        call["rlm"] = structure
    nodes = trace["nodes"]
    assert isinstance(nodes, list)
    tool_message = nodes[6]["message"]
    assert isinstance(tool_message, dict)
    tool_message["tool_call_id"] = "call_0"
    structures[1]["parent_tool_call_id"] = "call_0"
    structures[1]["invocation_id"] = "shard-0"
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "episode",
                "ok": True,
                "errors": [],
                "traces": [trace],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    imported = import_trace_file(path)
    report = build_provenance_report(path)

    assert imported.ready_for_representative_raf
    assert imported.exact_cross_component_links == 1
    assert imported.cross_component_fallbacks == 0
    assert report["ready_for_representative_raf"]
    assert report["blocking_finding"] is None

    projection = build_recorded_raf_projection(
        path,
        alternatives_per_target=3,
    )
    assert projection.target_count == 1
    assert projection.requires_broader_trace
    assert projection.empirical_branch_replay_status == "not_run_projection_only"
    (target,) = projection.targets
    assert target.target_depth == 1
    assert target.full_suffix_policy_events == 1
    assert target.sliced_affected_policy_events == 1
    assert target.exact_key_reusable_policy_events == 0
    assert target.conservative_no_cache_generated_token_work_fraction == 1.0
    assert target.modeled_sliced_policy_token_raf == (
        target.modeled_exact_key_full_policy_token_raf
    )
    assert target.modeled_sliced_policy_token_raf == (
        target.modeled_no_cache_full_policy_token_raf
    )
