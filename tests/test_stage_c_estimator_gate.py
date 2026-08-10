from __future__ import annotations

import hashlib

from redco.analysis.stage_c_estimator_gate import evaluate_estimator_gate
from redco.contracts import canonical_json


def test_estimator_gate_is_signed_and_passes_high_fidelity_campaign() -> None:
    report = evaluate_estimator_gate(
        samples_per_probe=10_000,
        master_seed=72026,
        exogenous_seed_count=300,
        noise_floor=0.01,
        minimum_gradient_cosine=0.99,
        minimum_rank_correlation=0.99,
        minimum_sign_accuracy=1.0,
        maximum_advantage_rmse=0.03,
    )

    signature = report.pop("report_sha256")
    assert signature == hashlib.sha256(canonical_json(report)).hexdigest()
    assert report["passed"] is True
    assert report["checks"]["all_actions_observed"] is True

    headline = report["headline"]
    assert headline["minimum_informative_gradient_cosine"] > 0.99
    assert headline["minimum_informative_rank_correlation"] > 0.99
    assert headline["minimum_informative_sign_accuracy"] == 1.0
    assert headline["maximum_informative_advantage_rmse"] < 0.03

    spurious = [
        probe for probe in report["probes"] if probe["probe_name"] == "spurious_correlation"
    ]
    assert len(spurious) == 1
    assert tuple(spurious[0]["true_policy_gradient"]) == (0.0, 0.0)
    assert spurious[0]["sign_comparisons"] == 0
