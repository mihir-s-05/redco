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
