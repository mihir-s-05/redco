from __future__ import annotations

import pytest

from redco.analysis.stage_c_efficiency import (
    ACTIONS,
    BranchObservation,
    _advantages,
    _oracle_route_gradient,
    _softmax,
    branch_group_power_analysis,
)


def test_matched_credit_relabeling_changes_only_advantages() -> None:
    observations = (
        BranchObservation("e1", "alpha", "5", 1.0),
        BranchObservation("e1", "alpha", "0", 0.0),
        BranchObservation("e2", "delta", "5", 1.0),
        BranchObservation("e2", "delta", "0", 1.0),
    )

    assert _advantages(observations, method="local_loo") == pytest.approx(
        (1.0, -1.0, 0.0, 0.0)
    )
    assert _advantages(observations, method="matched_broadcast") == pytest.approx(
        (0.25, -0.75, 0.25, 0.25)
    )


def test_exact_oracle_is_zero_for_an_irrelevant_target() -> None:
    probabilities = _softmax((0.0,) * len(ACTIONS))
    gradient = _oracle_route_gradient(
        "confusion_irrelevant",
        "gamma",
        probabilities,
    )

    assert gradient == pytest.approx((0.0,) * len(ACTIONS))


def test_power_analysis_distinguishes_fixed_states_from_fixed_calls() -> None:
    probabilities = {
        (probe, route): (0.1125, 0.1125, 0.1125, 0.1125, 0.1125, 0.1, 0.1125, 0.1125, 0.0)
        for probe in (
            "confusion_irrelevant",
            "confusion_redundant",
            "confusion_lucky",
        )
        for route in ("alpha", "beta", "gamma", "delta")
    }
    power = branch_group_power_analysis(
        probabilities,
        {"alpha": 0.3, "beta": 0.2, "gamma": 0.4, "delta": 0.1},
    )

    assert power["smallest_group_size_with_8_episodes"] > 2
    assert power["smallest_group_size_at_96_calls"] == 2
