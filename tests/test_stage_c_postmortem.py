from __future__ import annotations

import pytest

from redco.analysis.stage_c_postmortem import (
    binary_informative_group_probability,
    enumerate_probe,
    exact_policy_gradient,
    informative_group_probability,
    minimum_branch_count,
)
from redco.env.tasks.credit_probes import planted_needle, spurious_correlation


def test_binary_informativeness_matches_mixed_group_probability() -> None:
    probability = binary_informative_group_probability(0.05, branch_count=4)

    assert probability == pytest.approx(1 - 0.05**4 - 0.95**4)
    assert probability == pytest.approx(0.1854875)


def test_multiclass_informativeness_uses_reward_classes_not_actions() -> None:
    assert informative_group_probability(
        (0.5, 0.25, 0.25),
        branch_count=2,
    ) == pytest.approx(0.625)


def test_minimum_branch_count_meets_target() -> None:
    count = minimum_branch_count((0.05, 0.95), target_probability=0.625)

    assert count == 20


def test_exact_policy_gradient_is_centered_and_points_to_the_best_action() -> None:
    gradient = exact_policy_gradient((0.2, 0.3, 0.5), (0.0, 1.0, 0.0))

    assert sum(gradient) == pytest.approx(0.0)
    assert gradient == pytest.approx((-0.06, 0.21, -0.15))


def test_enumeration_marks_action_independent_states() -> None:
    report = enumerate_probe(spurious_correlation(), exogenous_seeds=(1, 2))

    assert report["action_dependent_states"] == 0
    assert report["states"][0]["best_reward"] == 0.0
    assert report["states"][1]["best_reward"] == 1.0


def test_enumeration_finds_the_planted_action() -> None:
    report = enumerate_probe(
        planted_needle(chunk_count=8, needle_chunk=5),
        exogenous_seeds=(9000,),
    )

    assert report["states"][0]["best_actions"] == ["5"]
