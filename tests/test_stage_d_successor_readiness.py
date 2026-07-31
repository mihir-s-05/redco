from __future__ import annotations

import json
from pathlib import Path

import pytest

from redco.analysis.stage_d_successor_readiness import evaluate


def test_successor_power_math_and_budget_block(tmp_path: Path) -> None:
    source = (
        Path(__file__).parents[1]
        / "configs"
        / "stage-d"
        / "stage-d0-scaffold-successor-design-v2.json"
    )
    design = json.loads(source.read_text(encoding="utf-8"))
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design), encoding="utf-8")

    report = evaluate(path)

    power = report["power_derivation"]
    assert power["probability_at_least_five_of_eight"] >= 0.95
    assert power["two_sided_95pct_wilson_lower"] >= 0.808
    budget = report["minimum_budget_lower_bound"]
    assert budget["scenarios"]["2.00"]["fits_wallet_after_reserve"] is False
    assert budget["scenarios"]["1.50"]["fits_wallet_after_reserve"] is True
    assert budget["maximum_hourly_rate_to_fit_lower_bound_usd"] == pytest.approx(
        1.592174500,
        rel=1e-4,
    )
    assert not report["passes"]
    assert report["checks"]["full_campaign_budget_currently_proven"] is False
