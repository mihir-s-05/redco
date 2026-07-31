from __future__ import annotations

import importlib.util
from pathlib import Path


def test_stage_d_v4_5_preregistration_audit_passes() -> None:
    root = Path(__file__).parents[1]
    script = (
        root
        / "scripts"
        / "audit_stage_d0_scaffold_support_amendment_v4_5.py"
    )
    spec = importlib.util.spec_from_file_location("stage_d_v4_5_audit", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = (
        root
        / "configs"
        / "stage-d"
        / "stage-d0-scaffold-support-preregistration-v4.json"
    )
    amendment_v4_1 = (
        root
        / "configs"
        / "stage-d"
        / "stage-d0-scaffold-support-amendment-v4-1.json"
    )
    amendment_v4_2 = (
        root
        / "configs"
        / "stage-d"
        / "stage-d0-scaffold-support-amendment-v4-2.json"
    )
    amendment_v4_5 = (
        root
        / "configs"
        / "stage-d"
        / "stage-d0-scaffold-support-amendment-v4-5.json"
    )
    audit_v4_2 = (
        root
        / "reports"
        / "stage-d0-scaffold-support-preregistration-audit-v4-2.json"
    )
    patch_audit = (
        root
        / "reports"
        / "stage-d-prime-sft-runtime-patch-v2-audit.json"
    )
    report = module.audit(
        root,
        protocol,
        amendment_v4_1,
        amendment_v4_2,
        amendment_v4_5,
        audit_v4_2,
        patch_audit,
    )
    assert report["passes"], {
        name: value
        for name, value in report["checks"].items()
        if not value
    }
