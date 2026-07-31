from __future__ import annotations

from pathlib import Path

from redco.analysis.stage_d_support_preregistration import evaluate


def test_frozen_stage_d_support_protocol_passes() -> None:
    report = evaluate(
        Path(
            "configs/stage-d/"
            "stage-d0-scaffold-support-preregistration-v3.json"
        )
    )
    assert report["passes"]
    assert report["power"]["wilson_lower_58_of_64"] >= 0.808
    assert (
        report["power"]["probability_at_least_5_of_8_at_p_0808"] >= 0.95
    )
    assert report["budget"]["headroom_usd"] > 0
