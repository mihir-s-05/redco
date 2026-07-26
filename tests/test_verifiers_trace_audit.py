from __future__ import annotations

import json
from pathlib import Path

from redco.analysis.verifiers_trace_audit import evaluate_trace_gate


def test_trace_gate_signs_passing_recursive_audit(tmp_path: Path) -> None:
    trace = {
        "id": "trace-gate",
        "ok": True,
        "errors": [],
        "agent": {"model": "model-a"},
        "nodes": [
            {"parent": None, "token_ids": [1], "mask": [False]},
            {"parent": 0, "token_ids": [2, 3], "mask": [False, True]},
            {"parent": None, "token_ids": [4], "mask": [False]},
            {"parent": 2, "token_ids": [5, 6], "mask": [False, True]},
        ],
        "calls": [
            {
                "node": 1,
                "model": "model-a",
                "sampling": {"temperature": 0.7, "seed": 71},
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                "time": {"start": 1.0, "end": 2.0},
            },
            {
                "node": 3,
                "model": "model-a",
                "sampling": {"temperature": 0.7, "seed": 72},
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                "time": {"start": 2.0, "end": 3.0},
            },
        ],
        "info": {"redco_trace_audit": {"model_calls": 2}},
    }
    input_path = tmp_path / "traces.jsonl"
    input_path.write_text(
        json.dumps(
            {"id": "episode", "ok": True, "errors": [], "traces": [trace]}
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_trace_gate(input_path, require_recursive=True)

    assert result["passed"] is True
    assert result["checks"]["recursive_model_call_observed"] is True
    assert len(result["source_sha256"]) == 64
    assert len(result["report_sha256"]) == 64


def test_trace_gate_fails_without_recursive_component(tmp_path: Path) -> None:
    trace = {
        "id": "trace-gate",
        "ok": True,
        "errors": [],
        "agent": {"model": "model-a"},
        "nodes": [
            {"parent": None, "token_ids": [1], "mask": [False]},
            {"parent": 0, "token_ids": [2, 3], "mask": [False, True]},
        ],
        "calls": [
            {
                "node": 1,
                "model": "model-a",
                "sampling": {"seed": 71},
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
        ],
        "info": {"redco_trace_audit": {"model_calls": 1}},
    }
    input_path = tmp_path / "traces.jsonl"
    input_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")

    result = evaluate_trace_gate(input_path, require_recursive=True)

    assert result["passed"] is False
    assert result["checks"]["recursive_model_call_observed"] is False
