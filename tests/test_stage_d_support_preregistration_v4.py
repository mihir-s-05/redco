from __future__ import annotations

import importlib.util
from pathlib import Path


def test_stage_d_v4_preregistration_audit_passes() -> None:
    root = Path(__file__).parents[1]
    script = (
        root
        / "scripts"
        / "audit_stage_d0_scaffold_support_preregistration_v4.py"
    )
    spec = importlib.util.spec_from_file_location("stage_d_v4_audit", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = (
        root
        / "configs"
        / "stage-d"
        / "stage-d0-scaffold-support-preregistration-v4.json"
    )
    report = module.audit(root, protocol)
    assert report["passes"], {
        name: value
        for name, value in report["checks"].items()
        if not value
    }
