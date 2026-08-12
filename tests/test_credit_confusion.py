from __future__ import annotations

from math import fsum
from random import Random

import pytest

from redco.analysis.credit_confusion import (
    ConfusionProbe,
    DiagnosticConfig,
    _loss_gradient,
    _redco_records,
    _trajectory_records,
    build_credit_confusion_diagnostic,
    exact_target_gradient,
    logit,
    redco_target_moments,
    standard_confusion_probes,
    trajectory_target_moments,
)


def _probe(name: str) -> ConfusionProbe:
    return next(probe for probe in standard_confusion_probes() if probe.name == name)


def test_irrelevant_target_has_zero_exact_and_branch_gradient() -> None:
    probe = _probe("irrelevant_target")
    redco = redco_target_moments(
        probe,
        branch_group_size=4,
        episodes_per_step=2,
    )
    trajectory = trajectory_target_moments(probe, group_size=4)

    assert exact_target_gradient(probe) == pytest.approx(0.0)
    assert redco["mean"] == pytest.approx(0.0)
    assert redco["variance"] == pytest.approx(0.0)
    assert trajectory["mean"] == pytest.approx(0.0)
    assert trajectory["variance"] > 0


@pytest.mark.parametrize("name", ["redundant_target", "lucky_target"])
def test_exact_moments_are_unbiased(name: str) -> None:
    probe = _probe(name)
    exact = exact_target_gradient(probe)
    trajectory = trajectory_target_moments(probe, group_size=4)
    redco = redco_target_moments(
        probe,
        branch_group_size=4,
        episodes_per_step=2,
    )

    assert trajectory["enumerated_probability"] == pytest.approx(1.0)
    assert redco["enumerated_probability"] == pytest.approx(1.0)
    assert trajectory["mean"] == pytest.approx(exact, abs=1e-14)
    assert redco["mean"] == pytest.approx(exact, abs=1e-14)


def test_small_diagnostic_is_reproducible() -> None:
    config = DiagnosticConfig(
        trajectory_group_size=4,
        branch_group_size=3,
        branch_episodes_per_step=2,
        policy_call_budget=64,
        learning_trials=20,
        seed=17,
    )

    first = build_credit_confusion_diagnostic(config)
    second = build_credit_confusion_diagnostic(config)

    assert first == second
    assert first["config"]["calls_per_update"] == 8
    assert first["mandatory_checks"]["all_exact_mean_biases_at_most_1e_12"]
    assert first["mandatory_checks"]["all_enumerated_probabilities_sum_to_one"]
    assert first["mandatory_checks"]["irrelevant_redco_variance_at_most_1pct_trajectory"]
    assert first["mandatory_checks"]["both_relevant_tasks_have_positive_exact_gradient"]
    assert first["schema_version"] == 2
    assert set(first["probes"]["lucky_target"]["learning"]) == {
        "trajectory_loo",
        "redco",
    }
    irrelevant = first["probes"]["irrelevant_target"]["learning"]
    assert irrelevant["redco"]["mean_absolute_probability_drift"] == 0.0
    assert irrelevant["trajectory_loo"]["mean_absolute_probability_drift"] > 0.0
    comparison = first["probes"]["irrelevant_target"][
        "learning_comparison_redco_minus_trajectory"
    ]
    assert comparison["mean_absolute_drift_delta"] < 0.0
    assert len(first["signed_payload_sha256"]) == 64


def test_learning_config_rejects_unequal_sampling_batches() -> None:
    with pytest.raises(ValueError, match="same calls per update"):
        DiagnosticConfig(
            trajectory_group_size=4,
            branch_group_size=4,
            branch_episodes_per_step=2,
        )


def test_real_loss_gradient_matches_the_record_score_estimator() -> None:
    probe = _probe("lucky_target")
    policy_logit = logit(0.3)
    record_groups = (
        _trajectory_records(
            probe,
            target_logit=policy_logit,
            group_size=8,
            rng=Random(7),
        ),
        _redco_records(
            probe,
            target_logit=policy_logit,
            branch_group_size=7,
            episodes_per_step=2,
            rng=Random(7),
        ),
    )
    for records in record_groups:
        decision_units = fsum(record.decision_unit_normalizer for record in records)
        expected = -fsum(
            record.advantages[0]
            * record.rl_weights[0]
            * (record.sequence.token_ids[0] - 0.3)
            for record in records
        ) / decision_units
        observed = _loss_gradient(
            records,
            policy_logit=policy_logit,
            epsilon=1e-5,
        )
        assert observed == pytest.approx(expected, abs=1e-10)
