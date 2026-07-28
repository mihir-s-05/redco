from __future__ import annotations

import json
from pathlib import Path

import pytest

from redco.analysis.stage_c_policy_audit import (
    aggregate_policy_scores,
    prepare_policy_cases,
)


def _trace(action: str, token_id: int, *, probe: str = "planted_needle") -> dict:
    return {
        "agent": {"name": "original"},
        "task": {
            "data": {
                "probe_name": probe,
                "context_route": "alpha",
                "actions": ["0", "5"],
                "prompt": "prompt",
            }
        },
        "nodes": [
            {
                "sampled": False,
                "token_ids": [10, 11],
                "mask": [False, False],
            },
            {
                "sampled": True,
                "message": {"content": action},
                "token_ids": [12, token_id],
                "mask": [False, True],
            },
        ],
    }


def test_prepare_policy_cases_extracts_prefix_and_action_tokens(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in (_trace("0", 20), _trace("5", 25)))
        + "\n"
    )

    report = prepare_policy_cases([path])

    assert report["case_count"] == 1
    assert report["action_token_ids"] == {"0": 20, "5": 25}
    assert report["cases"][0]["prefix_token_ids"] == [10, 11, 12]


def test_prepare_policy_cases_rejects_inconsistent_tokenization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (_trace("0", 20), _trace("0", 21), _trace("5", 25))
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="multiple token ids"):
        prepare_policy_cases([path])


def _score_row(case_id: str, *, needle_mass: float, greedy: str) -> dict:
    return {
        "case_id": case_id,
        "probe_name": "planted_needle",
        "context_route": "alpha",
        "greedy_allowed_action": greedy,
        "full_vocab_action_probabilities_t1": {"0": 0.2, "5": needle_mass},
        "full_vocab_action_probabilities_t2": {"0": 0.2, "5": needle_mass},
        "conditional_action_probabilities_t1": {"0": 0.5, "5": 0.5},
        "conditional_action_probabilities_t2": {"0": 0.5, "5": 0.5},
    }


def test_aggregate_policy_scores_reports_movement_and_needle_mass() -> None:
    base = _score_row("planted_needle:alpha", needle_mass=0.05, greedy="0")
    tuned = _score_row("planted_needle:alpha", needle_mass=0.15, greedy="5")
    tuned["full_vocab_kl_from_base_t1"] = 0.01
    tuned["allowed_action_kl_from_base_t1"] = 0.02

    report = aggregate_policy_scores(
        {
            "source": {"model": "model"},
            "models": [
                {"name": "base", "cases": [base]},
                {"name": "tuned", "cases": [tuned]},
            ],
        }
    )

    assert report["summaries"]["base"][
        "needle_action_5_full_vocab_mass_t2"
    ]["mean"] == pytest.approx(0.05)
    assert report["summaries"]["tuned"]["needle_action_5_greedy_rate"] == 1.0
    assert report["summaries"]["tuned"]["mean_full_vocab_kl_from_base_t1"] == 0.01
