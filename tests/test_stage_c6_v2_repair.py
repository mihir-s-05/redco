from pathlib import Path

from redco.analysis.stage_c6_v2_repair import audit


def test_stage_c6_v2_bounded_repair_passes_machine_audit() -> None:
    amendment = Path(
        "configs/stage-c6/credit-confusion-repair-amendment-v2-1.json"
    )
    if not amendment.exists():
        return
    result = audit(
        Path(
            "configs/stage-c6/credit-confusion-live-preregistration-v2.json"
        ),
        amendment,
    )
    assert result["passed"], result
    assert all(result["checks"].values())
