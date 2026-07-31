from __future__ import annotations

from pathlib import Path

from redco.analysis.stage_d_successor_funding import evaluate


def test_added_funding_clears_conservative_envelope() -> None:
    amendment = (
        Path(__file__).parents[1]
        / "configs"
        / "stage-d"
        / "stage-d0-scaffold-successor-funding-amendment-v2-1.json"
    )
    report = evaluate(amendment)

    assert report["passes"]
    assert report["campaign"]["episode_equivalents"] == 1728
    assert report["campaign"]["cost_ceiling_usd"] < 32.0
    assert report["total_envelope_with_reserve_usd"] < 43.5
    assert report["headroom_after_full_envelope_usd"] > 3.5
    assert report["checks"]["resource_not_selected_or_reserved"]
