from __future__ import annotations

import json
import math
from pathlib import Path

from redco.analysis.stage_c5_smoke import verify_trace_file


def _root_scores() -> dict:
    probabilities = {"alpha": 0.1, "beta": 0.2, "gamma": 0.3, "delta": 0.4}
    token_ids = {"alpha": 11, "beta": 12, "gamma": 13, "delta": 14}
    return {
        "temperature_2": {
            "token_details": {
                route: [
                    {"token_id": 1, "temperature_2_logprob": -0.01},
                    {
                        "token_id": token_ids[route],
                        "temperature_2_logprob": math.log(probability),
                    },
                    {"token_id": 2, "temperature_2_logprob": -0.02},
                ]
                for route, probability in probabilities.items()
            }
        }
    }


def test_verify_trace_file_accepts_exact_constrained_choices(tmp_path: Path) -> None:
    root_scores = _root_scores()
    probabilities = {"alpha": 0.1, "beta": 0.2, "gamma": 0.3, "delta": 0.4}
    token_ids = {"alpha": 11, "beta": 12, "gamma": 13, "delta": 14}
    traces = tmp_path / "traces.jsonl"
    rows = []
    for route in ("alpha", "delta"):
        rows.append(
            {
                "agent": {
                    "name": "context",
                    "sampling": {
                        "extra_body": {
                            "structured_outputs": {
                                "choice": [
                                    "<route>alpha</route>",
                                    "<route>beta</route>",
                                    "<route>gamma</route>",
                                    "<route>delta</route>",
                                ]
                            }
                        }
                    },
                },
                "nodes": [
                    {
                        "sampled": True,
                        "message": {"content": f"<route>{route}</route>"},
                        "token_ids": [100, 1, token_ids[route], 2],
                        "is_content": [False, True, True, True],
                        "logprobs": [-0.01, math.log(probabilities[route]), -0.02],
                    }
                ],
            }
        )
    traces.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = verify_trace_file(traces, root_scores, expected_context_traces=2)

    assert result["passed"]
    assert all(result["checks"].values())
