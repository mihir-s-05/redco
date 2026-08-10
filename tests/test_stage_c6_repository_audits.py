from __future__ import annotations

from pathlib import Path

import pytest

from redco.analysis.stage_c6_v2_preregistration import audit as audit_v2
from redco.analysis.stage_c6_v3_preregistration import audit as audit_v3
from redco.analysis.stage_c6_v3_repair import audit as audit_v3_repair

ROOT = Path(__file__).resolve().parents[1]
C6 = ROOT / "configs/stage-c6"
ARCHIVE = (
    ROOT / "runs/stage-c6/credit-confusion-live-v3-control/stage-c6-v3-attempt1-evidence.tar.gz"
)


def _failed_checks(result: dict[str, object]) -> set[str]:
    assert result["passed"] is False, result
    checks = result["checks"]
    assert isinstance(checks, dict)
    assert all(isinstance(name, str) and isinstance(value, bool) for name, value in checks.items())
    return {name for name, value in checks.items() if value is False}


def test_repaired_trees_fail_only_obsolete_source_hash_checks() -> None:
    expected = {"all_source_hashes_match"}
    v2 = C6 / "credit-confusion-live-preregistration-v2.json"
    v3 = C6 / "credit-confusion-live-preregistration-v3.json"
    assert _failed_checks(audit_v2(v2, root=ROOT)) == expected
    assert _failed_checks(audit_v3(v3, root=ROOT)) == expected


def test_v3_tree_fails_only_intermediate_repair_hash_checks() -> None:
    if not ARCHIVE.is_file():
        pytest.skip(f"retained ignored Stage-C6 evidence is absent: {ARCHIVE.relative_to(ROOT)}")
    protocol = C6 / "credit-confusion-live-preregistration-v3.json"
    amendment = C6 / "credit-confusion-repair-amendment-v3-1.json"
    expected = {"replacement_hashes_match", "unchanged_source_hashes_match"}
    assert _failed_checks(audit_v3_repair(protocol, amendment, root=ROOT)) == expected
