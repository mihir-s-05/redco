from pathlib import Path

from redco.analysis.stage_c6_v3_repair import audit


def test_stage_c6_v3_outcome_independent_repair_passes_audit() -> None:
    amendment = Path(
        "configs/stage-c6/credit-confusion-repair-amendment-v3-1.json"
    )
    if not amendment.exists():
        return
    result = audit(
        Path(
            "configs/stage-c6/credit-confusion-live-preregistration-v3.json"
        ),
        amendment,
    )
    assert result["passed"], result
    assert all(result["checks"].values())
