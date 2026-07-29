import json
import math
from pathlib import Path
from typing import Any

from redco.analysis.stage_c3_power import analyze_power, prepare_root_cases


def _root_trace(route: str, completion_token: int) -> dict[str, Any]:
    return {
        "agent": {"name": "context"},
        "nodes": [
            {
                "token_ids": [10, 11],
                "mask": [False, False],
            },
            {
                "message": {"content": f"<route>{route}</route>"},
                "token_ids": [12, completion_token],
                "mask": [False, True],
            },
        ],
    }


def test_prepare_root_cases_covers_every_route(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    traces.write_text(
        "".join(
            json.dumps(_root_trace(route, 20 + index)) + "\n"
            for index, route in enumerate(("alpha", "beta", "gamma", "delta"))
        ),
        encoding="utf-8",
    )

    result = prepare_root_cases(traces)

    assert [case["route"] for case in result["cases"]] == [
        "alpha",
        "beta",
        "gamma",
        "delta",
    ]
    assert all(case["prefix_token_ids"] == [10, 11, 12] for case in result["cases"])


def test_exact_power_gate_passes_well_exposed_design() -> None:
    digit_probabilities = {
        str(index): (0.10 if index == 5 else 0.90 / 7.0)
        for index in range(8)
    }
    action_scores = {
        "source": {"cases_sha256": "actions"},
        "models": [
            {
                "name": "warmstart",
                "temperatures": {
                    "2.0": [
                        {
                            "context_route": route,
                            "action_probabilities": digit_probabilities,
                        }
                        for route in ("alpha", "beta", "gamma", "delta")
                    ]
                },
            }
        ],
    }
    route_scores = {
        "source": {"cases_sha256": "routes"},
        "temperature_2": {
            "route_sequence_probabilities": {
                "alpha": 0.24,
                "beta": 0.24,
                "gamma": 0.24,
                "delta": 0.24,
            }
        },
    }

    result = analyze_power(action_scores, route_scores)

    assert result["status"] == "passed"
    assert result["measurements"]["sampling_false_abort_probability"] == 0.0
    assert (
        result["measurements"][
            "expected_target_informative_groups_per_sliced_step"
        ]
        >= 5.0
    )


def test_exact_power_gate_rejects_root_collapse() -> None:
    digit_probabilities = {
        str(index): (0.08 if index == 5 else 0.92 / 7.0)
        for index in range(8)
    }
    action_scores = {
        "source": {"cases_sha256": "actions"},
        "models": [
            {
                "name": "warmstart",
                "temperatures": {
                    "2.0": [
                        {
                            "context_route": route,
                            "action_probabilities": digit_probabilities,
                        }
                        for route in ("alpha", "beta", "gamma", "delta")
                    ]
                },
            }
        ],
    }
    route_scores = {
        "source": {"cases_sha256": "routes"},
        "temperature_2": {
            "route_sequence_probabilities": {
                "alpha": 0.005,
                "beta": 0.005,
                "gamma": 0.98,
                "delta": 0.005,
            }
        },
    }

    result = analyze_power(action_scores, route_scores)

    assert result["status"] == "failed"
    assert (
        result["checks"][
            "irrelevant_root_group_informative_probability_at_least_0_50"
        ]
        is False
    )
    assert math.isfinite(
        result["measurements"]["root_group_informative_probability"]
    )
