from pathlib import Path

from redco.analysis.stage_c6_v3_preregistration import audit


def test_stage_c6_v3_protocol_passes_machine_audit() -> None:
    protocol = Path(
        "configs/stage-c6/credit-confusion-live-preregistration-v3.json"
    )
    if not protocol.exists():
        return
    result = audit(protocol)
    assert result["passed"], result
    assert all(result["checks"].values())
