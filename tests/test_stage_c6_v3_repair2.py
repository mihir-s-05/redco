from pathlib import Path

from redco.analysis.stage_c6_v3_repair2 import audit


def test_stage_c6_v3_in_place_parser_repair_passes_audit() -> None:
    second = Path(
        "configs/stage-c6/credit-confusion-repair-amendment-v3-2.json"
    )
    if not second.exists():
        return
    result = audit(
        Path(
            "configs/stage-c6/credit-confusion-live-preregistration-v3.json"
        ),
        Path(
            "configs/stage-c6/credit-confusion-repair-amendment-v3-1.json"
        ),
        second,
    )
    assert result["passed"], result
