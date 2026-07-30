from pathlib import Path

from redco.analysis.stage_c6_v3_repair import audit
from redco.analysis.stage_c6_v3_repair2 import audit as audit_repair2


def test_stage_c6_v3_outcome_independent_repair_passes_audit() -> None:
    amendment = Path(
        "configs/stage-c6/credit-confusion-repair-amendment-v3-1.json"
    )
    if not amendment.exists():
        return
    second_amendment = Path(
        "configs/stage-c6/credit-confusion-repair-amendment-v3-2.json"
    )
    if second_amendment.exists():
        result = audit_repair2(
            Path(
                "configs/stage-c6/credit-confusion-live-preregistration-v3.json"
            ),
            amendment,
            second_amendment,
        )
        assert result["passed"], result
        return
    result = audit(
        Path(
            "configs/stage-c6/credit-confusion-live-preregistration-v3.json"
        ),
        amendment,
    )
    assert result["passed"], result
    assert all(result["checks"].values())
