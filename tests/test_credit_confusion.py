from __future__ import annotations

import pytest

from redco.analysis.credit_confusion import (
    ConfusionProbe,
    DiagnosticConfig,
    branch_target_moments,
    broadcast_target_moments,
    build_credit_confusion_diagnostic,
    exact_target_gradient,
    standard_confusion_probes,
)


def _probe(name: str) -> ConfusionProbe:
    return next(probe for probe in standard_confusion_probes() if probe.name == name)


def test_irrelevant_target_has_zero_exact_and_branch_gradient() -> None:
    probe = _probe("irrelevant_target")
    branch = branch_target_moments(
        probe,
        branch_group_size=4,
        episodes_per_step=2,
    )
    broadcast = broadcast_target_moments(probe, group_size=4)

    assert exact_target_gradient(probe) == pytest.approx(0.0)
    assert branch["mean"] == pytest.approx(0.0)
    assert branch["variance"] == pytest.approx(0.0)
    assert broadcast["mean"] == pytest.approx(0.0)
    assert broadcast["variance"] > 0


@pytest.mark.parametrize("name", ["redundant_target", "lucky_target"])
def test_exact_moments_are_unbiased(name: str) -> None:
    probe = _probe(name)
    exact = exact_target_gradient(probe)
    broadcast = broadcast_target_moments(probe, group_size=4)
    branch = branch_target_moments(
        probe,
        branch_group_size=4,
        episodes_per_step=2,
    )

    assert broadcast["enumerated_probability"] == pytest.approx(1.0)
    assert branch["enumerated_probability"] == pytest.approx(1.0)
    assert broadcast["mean"] == pytest.approx(exact, abs=1e-14)
    assert branch["mean"] == pytest.approx(exact, abs=1e-14)


def test_small_diagnostic_is_reproducible() -> None:
    config = DiagnosticConfig(
        trajectory_group_size=4,
        branch_group_size=4,
        branch_episodes_per_step=2,
        policy_call_budget=64,
        learning_trials=20,
        seed=17,
    )

    first = build_credit_confusion_diagnostic(config)
    second = build_credit_confusion_diagnostic(config)

    assert first == second
    assert first["mandatory_checks"]["all_exact_mean_biases_at_most_1e_12"]
    assert first["mandatory_checks"]["all_enumerated_probabilities_sum_to_one"]
    assert first["mandatory_checks"]["irrelevant_node_loo_variance_at_most_1pct_broadcast"]
    assert len(first["signed_payload_sha256"]) == 64
