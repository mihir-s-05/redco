from __future__ import annotations

import json
from pathlib import Path

import pytest

from redco.integrations.verifiers_trace import (
    audit_trace_file,
    build_policy_cache,
    extract_policy_calls,
    load_trace_records,
)


def _trace() -> dict[str, object]:
    return {
        "id": "trace-1",
        "agent": {"model": "model-a"},
        "ok": True,
        "errors": [],
        "info": {"redco_trace_audit": {"model_calls": 3}},
        "nodes": [
            {
                "parent": None,
                "token_ids": [10, 11],
                "mask": [False, False],
            },
            {
                "parent": 0,
                "token_ids": [12, 20, 21],
                "mask": [False, True, True],
            },
            {
                "parent": 1,
                "token_ids": [30],
                "mask": [False],
            },
            {
                "parent": 2,
                "token_ids": [31, 22],
                "mask": [False, True],
            },
            {
                "parent": None,
                "token_ids": [40],
                "mask": [False],
            },
            {
                "parent": 4,
                "token_ids": [41, 23],
                "mask": [False, True],
            },
        ],
        "calls": [
            {
                "node": 1,
                "model": "model-a",
                "sampling": {"temperature": 0.7, "seed": 7},
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "time": {"start": 1.0, "end": 1.5},
            },
            {
                "node": 3,
                "model": "model-a",
                "sampling": {"temperature": 0.7, "seed": 8},
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
                "time": {"start": 2.0, "end": 2.25},
            },
            {
                "node": 5,
                "model": "model-a",
                "sampling": {"temperature": 0.7, "seed": 9},
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                "time": {"start": 3.0, "end": 3.75},
            },
        ],
    }


def test_episode_reader_and_exact_prompt_reconstruction(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "episode-1",
                "ok": True,
                "errors": [],
                "traces": [_trace()],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    traces = load_trace_records(path)
    calls = extract_policy_calls(traces[0])

    assert len(traces) == 1
    assert calls[0].prompt_token_ids == (10, 11, 12)
    assert calls[0].action_token_ids == (20, 21)
    assert calls[1].prompt_token_ids == (10, 11, 12, 20, 21, 30, 31)
    assert calls[1].action_token_ids == (22,)
    assert calls[2].component_root_node == 4
    assert calls[2].wall_seconds == 0.75

    report = audit_trace_file(path)
    assert report.episode_count == 1
    assert report.successful_episode_count == 1
    assert report.episode_error_count == 0
    assert report.trace_count == 1
    assert report.successful_trace_count == 1
    assert report.trace_error_count == 0
    assert report.model_call_count == 3
    assert report.failed_model_call_count == 0
    assert report.exact_prompt_action_coverage == 1.0
    assert report.seed_coverage == 1.0
    assert report.usage_coverage == 1.0
    assert report.connected_components == 2
    assert report.message_graph_leaves == 2
    assert report.recursive_trace_count == 1
    assert report.native_audit_trace_count == 1
    assert report.ready_for_exact_key_replay

    cache = build_policy_cache(report.calls)
    reused = cache.resolve(
        calls[0].prompt_token_ids,
        checkpoint_id=calls[0].checkpoint_id,
        decoding_config_hash=calls[0].decoding_config_hash,
        event_seed=7,
        sampler=lambda _prompt, _seed: (999,),
    )
    assert reused.reused
    assert reused.action_token_ids == (20, 21)


def test_incomplete_seed_blocks_exact_replay_cache() -> None:
    trace = _trace()
    calls = trace["calls"]
    assert isinstance(calls, list)
    first = calls[0]
    assert isinstance(first, dict)
    first["sampling"] = {"temperature": 0.7}

    extracted = extract_policy_calls(trace)

    assert not extracted[0].exact_key_complete
    with pytest.raises(ValueError, match="lacks an exact replay key"):
        build_policy_cache(extracted)


def test_non_suffix_sample_mask_fails_closed() -> None:
    trace = _trace()
    nodes = trace["nodes"]
    assert isinstance(nodes, list)
    node = nodes[1]
    assert isinstance(node, dict)
    node["mask"] = [False, True, False]

    with pytest.raises(ValueError, match="non-suffix sampled-token mask"):
        extract_policy_calls(trace)
