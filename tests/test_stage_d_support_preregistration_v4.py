from __future__ import annotations

import importlib.util
from pathlib import Path


def test_stage_d_v4_1_preregistration_audit_passes() -> None:
    root = Path(__file__).parents[1]
    script = (
        root
        / "scripts"
        / "audit_stage_d0_scaffold_support_amendment_v4_1.py"
    )
    spec = importlib.util.spec_from_file_location("stage_d_v4_1_audit", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = (
        root
        / "configs"
        / "stage-d"
        / "stage-d0-scaffold-support-preregistration-v4.json"
    )
    amendment = (
        root
        / "configs"
        / "stage-d"
        / "stage-d0-scaffold-support-amendment-v4-1.json"
    )
    base_audit = (
        root
        / "reports"
        / "stage-d0-scaffold-support-preregistration-audit-v4.json"
    )
    report = module.audit(root, protocol, amendment, base_audit)
    assert report["passes"], {
        name: value
        for name, value in report["checks"].items()
        if not value
    }
