from __future__ import annotations

import pytest

from redco.analysis.proposal_factorial import (
    build_proposal_diagnostic,
    exact_estimator_moments,
    exact_policy_gradient,
    importance_weighted_loo_gradient,
    mixture_distribution,
)


def test_mixture_preserves_policy_support_and_ratio_bound() -> None:
    policy = (0.9, 0.1)
    sampling = mixture_distribution(policy, (0.2, 0.8), epsilon=0.25)

    assert sum(sampling) == pytest.approx(1.0)
    assert all(value > 0 for value in sampling)
    assert max(policy[index] / sampling[index] for index in range(2)) <= 1.0 / 0.75 + 1e-12


def test_importance_weighted_loo_is_exactly_unbiased_by_enumeration() -> None:
    policy = (0.8, 0.2)
    sampling = mixture_distribution(policy, (0.25, 0.75), epsilon=0.4)
    moments = exact_estimator_moments(
        policy=policy,
        sampling=sampling,
        rewards=(0.0, 1.0),
        group_size=4,
    )

    assert moments["target_gradient"] == pytest.approx(exact_policy_gradient(policy, (0.0, 1.0)))
    assert moments["estimator_mean"] == pytest.approx(moments["target_gradient"], abs=1e-14)
    assert moments["maximum_absolute_bias"] < 1e-14
    assert moments["enumerated_probability"] == pytest.approx(1.0)


def test_flat_group_can_have_zero_gradient_without_biasing_expectation() -> None:
    estimate = importance_weighted_loo_gradient(
        (0, 0, 0),
        policy=(0.9, 0.1),
        sampling=(0.7, 0.3),
        rewards=(0.0, 1.0),
    )

    assert estimate == pytest.approx((0.0, 0.0))


def test_diagnostic_is_reproducible_and_crosses_factors() -> None:
    first = build_proposal_diagnostic(
        success_masses=(0.02,),
        proposal_families={"teacher": 0.5},
        epsilons=(0.0, 0.25),
        group_size=4,
        groups_per_step=2,
    )
    second = build_proposal_diagnostic(
        success_masses=(0.02,),
        proposal_families={"teacher": 0.5},
        epsilons=(0.0, 0.25),
        group_size=4,
        groups_per_step=2,
    )

    assert first == second
    assert all(first["mandatory_checks"].values())
    assert "not ReDCO" in first["label"]
    assert len(first["future_factorial"]["credit_factor"]) == 3
    assert len(first["signed_payload_sha256"]) == 64
