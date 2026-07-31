from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eager_config_changes_exactly_one_field() -> None:
    module = _load(
        "stage_d_v4_10_config_audit",
        ROOT / "scripts" / "audit_stage_d_v4_10_eager_config.py",
    )
    report = module.audit(
        ROOT / "configs/stage-d/stage-d0-scaffold-inference-sft-v4.toml",
        ROOT / "configs/stage-d/stage-d0-scaffold-inference-sft-v4-eager.toml",
    )
    assert report["passes"]
    assert all(report["checks"].values())


def test_eager_tail_gate_precedes_byte_identical_request_tail(tmp_path: Path) -> None:
    module = _load(
        "stage_d_v4_10_eager_tail",
        ROOT / "scripts/generate_stage_d_v4_10_eager_tail.py",
    )
    output = tmp_path / "eager-tail.sh"
    report = module.generate(
        ROOT / "scripts/run_stage_d0_scaffold_support_v4_6.sh", output
    )
    generated = output.read_text(encoding="utf-8")
    assert report["passes"]
    assert generated.index("EAGER_RUNTIME_PREFLIGHT_PASSED") < generated.index(
        "run_eval() {"
    )
    assert 'grep -F "enforce_eager=True"' in generated
    assert 'grep -Fq "Profiling CUDA graph memory"' in generated


def test_eager_tail_rejects_parent_contract_drift(tmp_path: Path) -> None:
    module = _load(
        "stage_d_v4_10_eager_tail_drift",
        ROOT / "scripts/generate_stage_d_v4_10_eager_tail.py",
    )
    parent = tmp_path / "parent.sh"
    parent.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    with pytest.raises(ValueError, match="health-return block"):
        module.generate(parent, tmp_path / "generated.sh")


def test_inner_runner_has_only_named_successor_changes(tmp_path: Path) -> None:
    module = _load(
        "stage_d_v4_10_inner",
        ROOT / "scripts/generate_stage_d_v4_10_inner_runner.py",
    )
    output = tmp_path / "inner.sh"
    report = module.generate(
        ROOT / "scripts/run_stage_d0_scaffold_support_v4_7.sh", output
    )
    assert report["passes"]
    assert all(report["checks"].values())
    assert sum(line.startswith("@@") for line in report["unified_diff"]) == 3


def test_v4_10_preregistration_audit_passes() -> None:
    module = _load(
        "stage_d_v4_10_prereg_audit",
        ROOT
        / "scripts/audit_stage_d0_scaffold_support_preregistration_v4_10.py",
    )
    protocol = (
        ROOT
        / "configs/stage-d/stage-d0-scaffold-support-preregistration-v4-10.json"
    )
    report = module.audit(ROOT, protocol)
    assert report["passes"], {
        name: value for name, value in report["checks"].items() if not value
    }
