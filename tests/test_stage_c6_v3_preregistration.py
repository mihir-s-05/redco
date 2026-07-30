from pathlib import Path

from redco.analysis.stage_c6_v3_preregistration import audit
from redco.analysis.stage_c6_v3_repair import audit as audit_repair
from redco.analysis.stage_c6_v3_repair2 import audit as audit_repair2


def test_stage_c6_v3_protocol_passes_machine_audit() -> None:
    protocol = Path(
        "configs/stage-c6/credit-confusion-live-preregistration-v3.json"
    )
    if not protocol.exists():
        return
    second_amendment = Path(
        "configs/stage-c6/credit-confusion-repair-amendment-v3-2.json"
    )
    amendment = Path(
        "configs/stage-c6/credit-confusion-repair-amendment-v3-1.json"
    )
    if second_amendment.exists():
        result = audit_repair2(protocol, amendment, second_amendment)
        assert result["passed"], result
        return
    if amendment.exists():
        result = audit_repair(protocol, amendment)
        assert result["passed"], result
        return
    result = audit(protocol)
    assert result["passed"], result
    assert all(result["checks"].values())
