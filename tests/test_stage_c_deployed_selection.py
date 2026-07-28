from __future__ import annotations

import pytest

from redco.analysis.stage_c_deployed_selection import select_deployed_warmstart


def _score(name: str, mass: float, greedy: str = "3") -> dict:
    return {
        "backend": "vllm",
        "source": "merged",
        "temperature_semantics": "full vocabulary",
        "models": [
            {
                "name": name,
                "temperatures": {
                    "2.0": [
                        {
                            "probe_name": "planted_needle",
                            "action_probabilities": {"5": mass},
                            "greedy_allowed_action": greedy,
                        }
                    ]
                },
            }
        ],
    }


def _select(scored: list[tuple[int, dict]]) -> dict:
    return select_deployed_warmstart(
        scored,
        start_step=21,
        minimum_needle_mass_t2=0.08,
        maximum_needle_mass_t2=0.12,
        maximum_needle_greedy_rate=0.0,
        branch_count=11,
        groups_per_step=8,
        minimum_expected_informative_groups=4.75,
    )


def test_selects_earliest_passing_deployed_checkpoint() -> None:
    report = _select(
        [
            (21, _score("merged", 0.075)),
            (22, _score("merged_step_22", 0.085)),
            (23, _score("merged_step_23", 0.095)),
        ]
    )

    assert report["status"] == "pass"
    assert report["selected"]["step"] == 22
    assert report["analysis"] == "stage-c-deployed-warmstart-selection"
    assert [record["step"] for record in report["source_records"]] == [21, 22, 23]


def test_fails_closed_below_support_floor() -> None:
    report = _select([(21, _score("merged", 0.075))])

    assert report["status"] == "fail"
    assert report["selected"] is None


def test_rejects_noncontiguous_candidate_observation() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        _select(
            [
                (21, _score("merged", 0.075)),
                (23, _score("merged_step_23", 0.095)),
            ]
        )
