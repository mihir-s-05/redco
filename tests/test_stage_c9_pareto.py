from __future__ import annotations

import json
from pathlib import Path

import pytest

from redco.analysis.stage_c9_pareto import evaluate, render_svg


def test_pareto_analysis_keeps_frozen_failure_and_corrects_claim(
    tmp_path: Path,
) -> None:
    runs = {}
    values = {
        "local-e1": [(0.30, 0.08), (0.31, 0.09), (0.29, 0.07)],
        "local-e2": [(0.60, 0.20), (0.62, 0.21), (0.61, 0.19)],
        "branch-global-e2": [
            (0.49, 0.14),
            (0.50, 0.15),
            (0.48, 0.13),
        ],
        "stock": [(0.90, 0.51), (0.89, 0.50), (0.91, 0.52)],
    }
    for arm, arm_values in values.items():
        for seed, (gain, drift) in zip(
            (10031, 10032, 10033), arm_values, strict=True
        ):
            runs[f"{arm}--s{seed}"] = {
                "final_causal_mass_gain": gain,
                "exact_final": {
                    "delta_nuisance_js_from_initial": drift,
                },
            }
    partial = tmp_path / "partial.json"
    partial.write_text(
        json.dumps(
            {
                "signed_payload_sha256": "source-signature",
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    trigger = tmp_path / "trigger.json"
    trigger.write_text(
        json.dumps({"default": "do_not_run"}), encoding="utf-8"
    )

    result = evaluate(partial, trigger)

    status = result["claim_status"]
    assert status["frozen_endpoint_credit_gate"] == "failed_as_preregistered"
    assert (
        status["practical_local_vs_branch_global_increment"]
        == "indeterminate"
    )
    assert status["branching_vs_stock"] == "descriptively_supported"
    assert result["oracle_floor_trigger"]["default"] == "do_not_run"
    assert result["arm_summaries"]["local-e2"][
        "drift_per_unit_mean_causal_gain"
    ] == pytest.approx(0.2 / 0.61)
    assert (
        result["descriptive_findings"][
            "stock_to_mean_branch_drift_per_gain_ratio"
        ]
        > 1.5
    )
    assert "<svg" in render_svg(result)
    assert "Stage C9 final endpoint Pareto view" in render_svg(result)
