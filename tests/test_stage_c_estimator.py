from __future__ import annotations

import pytest

from redco.analysis.stage_c_estimator import audit_probe_estimator
from redco.env.tasks.credit_probes import (
    spurious_correlation,
    standard_credit_probes,
)


def test_finite_estimator_recovers_exhaustive_policy_gradient() -> None:
    results = [
        audit_probe_estimator(
            probe,
            samples=10_000,
            master_seed=72026,
            exogenous_seed_count=300,
            noise_floor=0.01,
        )
        for probe in standard_credit_probes()
    ]

    informative = [
        result
        for result in results
        if any(value != 0.0 for value in result.true_policy_gradient)
    ]
    assert min(result.gradient_cosine for result in informative) > 0.99
    assert min(result.advantage_rank_correlation for result in informative) > 0.99
    assert min(result.sign_accuracy_above_noise for result in informative) == 1.0
    assert max(result.advantage_rmse for result in informative) < 0.03
    assert all(min(result.action_counts) > 0 for result in results)


def test_zero_signal_probe_has_zero_exact_gradient() -> None:
    result = audit_probe_estimator(
        spurious_correlation(),
        samples=1_000,
        master_seed=9,
        exogenous_seed_count=100,
        noise_floor=0.1,
    )

    assert result.true_policy_gradient == pytest.approx((0.0, 0.0))
    assert result.sign_comparisons == 0
