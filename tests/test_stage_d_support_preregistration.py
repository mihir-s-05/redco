from __future__ import annotations

from pathlib import Path

from redco.analysis.stage_d_support_preregistration import evaluate


def test_closed_v3_protocol_detects_successor_source_drift() -> None:
    report = evaluate(
        Path(
            "configs/stage-d/"
            "stage-d0-scaffold-support-preregistration-v3.json"
        )
    )
    assert not report["passes"]
    assert report["decision"] == "blocked"
    assert not report["checks"]["all_source_hashes_match"]
    assert Path(
        "reports/stage-d0-scaffold-support-v3-preobservation-closure.json"
    ).is_file()
    assert report["power"]["wilson_lower_58_of_64"] >= 0.808
    assert (
        report["power"]["probability_at_least_5_of_8_at_p_0808"] >= 0.95
    )
    assert report["budget"]["headroom_usd"] > 0
