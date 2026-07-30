from pathlib import Path

from redco.analysis.stage_c6_v2_preregistration import audit
from redco.analysis.stage_c6_v2_repair import audit as audit_repair


def test_frozen_stage_c6_v2_protocol_passes_machine_audit() -> None:
    path = Path(
        "configs/stage-c6/credit-confusion-live-preregistration-v2.json"
    )
    if not path.exists():
        return
    amendment = Path(
        "configs/stage-c6/credit-confusion-repair-amendment-v2-1.json"
    )
    if amendment.exists():
        result = audit_repair(path, amendment)
        assert result["passed"], result
        return
    result = audit(path)
    assert result["passed"], result
    assert all(result["checks"].values())
