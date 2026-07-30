from pathlib import Path

from redco.analysis.stage_c6_v2_preregistration import audit


def test_frozen_stage_c6_v2_protocol_passes_machine_audit() -> None:
    path = Path(
        "configs/stage-c6/credit-confusion-live-preregistration-v2.json"
    )
    if not path.exists():
        return
    result = audit(path)
    assert result["passed"], result
    assert all(result["checks"].values())
