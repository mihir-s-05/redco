from __future__ import annotations

import random

import pytest

from redco.analysis.stage_c_oracle import (
    OracleConfig,
    expected_regret,
    run_oracle_diagnostic,
    sampled_loo_gradient,
    softmax,
)


def test_softmax_is_stable_and_normalized() -> None:
    probabilities = softmax([1_000.0, 1_001.0])

    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[1] > probabilities[0]


def test_expected_regret_is_zero_for_optimal_point_mass() -> None:
    assert expected_regret([0.0, 1.0], [0.0, 1.0]) == 0.0


def test_flat_sampled_group_has_zero_loo_gradient() -> None:
    gradient, informative = sampled_loo_gradient(
        [1.0, 0.0],
        [1.0, 0.0],
        group_size=4,
        rng=random.Random(1),
    )

    assert informative is False
    assert gradient == [0.0, 0.0]


def test_oracle_diagnostic_is_labeled_and_reproducible() -> None:
    config = OracleConfig(updates=4, trials=8, seed=17)

    first = run_oracle_diagnostic(config)
    second = run_oracle_diagnostic(config)

    assert first == second
    assert "not ReDCO" in first["label"]
    assert first["states"]["action_dependent"] > 0
    assert first["oracle"]["final_mean_regret"] < first["oracle"][
        "initial_mean_regret"
    ]
    assert len(first["signed_payload_sha256"]) == 64
