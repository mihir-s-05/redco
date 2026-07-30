"""Audit the bounded Stage-C6 v2 infrastructure repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(
    protocol_path: Path,
    amendment_path: Path,
    *,
    root: Path = Path("."),
) -> dict[str, Any]:
    protocol = _load(protocol_path)
    amendment = _load(amendment_path)
    attempt_root = root / amendment["attempt_1"]["evidence_root"]
    changed = amendment["repair_source_changes"]
    unchanged_source_checks = {
        path: _sha256(root / path) == expected
        for path, expected in protocol["source"]["sha256"].items()
        if path not in changed
    }
    changed_source_checks = {
        path: {
            "old_matches_protocol": (
                values["old_sha256"]
                == protocol["source"]["sha256"].get(path)
            ),
            "new_matches_workspace": (
                values["new_sha256"] == _sha256(root / path)
            ),
        }
        for path, values in changed.items()
    }
    evidence_checks = {
        path: _sha256(attempt_root / path) == expected
        for path, expected in amendment["attempt_1"][
            "evidence_sha256"
        ].items()
    }
    identity = _load(attempt_root / "initialization/model-identity.json")
    reproducibility = _load(
        attempt_root / "initialization/canonical/reproducibility.json"
    )
    canonical = _load(
        attempt_root
        / "initialization/canonical/replicate_1/support-verification.json"
    )
    runtime = _load(
        attempt_root / "initialization/runtime/runtime-support.json"
    )
    invariant = _load(attempt_root / "smoke/structural-invariant.json")
    control = (
        attempt_root / "smoke/structural-broadcast-s9910.control.log"
    ).read_text(encoding="utf-8")
    driver = (root / "scripts/run_stage_c6_campaign_v2.sh").read_text(
        encoding="utf-8"
    )
    scientific_dirs = {
        "confusion_irrelevant",
        "confusion_redundant",
        "confusion_lucky",
    }
    checks = {
        "parent_protocol_exact": (
            amendment["parent_protocol_sha256"] == _sha256(protocol_path)
        ),
        "parent_audit_exact": (
            amendment["parent_audit_sha256"]
            == _sha256(root / amendment["parent_audit"])
        ),
        "repair_commit_exact": (
            amendment["repair_commit"] == "991395b"
        ),
        "only_declared_frozen_sources_changed": (
            all(unchanged_source_checks.values())
            and all(
                all(values.values())
                for values in changed_source_checks.values()
            )
        ),
        "all_attempt_evidence_hashes_match": all(evidence_checks.values()),
        "model_identity_passed": identity["status"] == "passed",
        "canonical_reproducibility_passed": (
            reproducibility["status"] == "passed"
            and len(set(reproducibility["action_signatures"])) == 1
            and len(set(reproducibility["root_signatures"])) == 1
        ),
        "canonical_support_passed": canonical["status"] == "passed",
        "runtime_power_passed": runtime["status"] == "passed",
        "smoke_was_pre_model_exit_127": (
            invariant["passed"] is False
            and invariant["launcher_exit_code"] == 127
            and invariant["checks"]["first_training_row_before_timeout"]
            is False
            and "failed to run command" in control
            and "uv" in control
            and "No such file or directory" in control
        ),
        "no_scientific_arm_directory_exists": not any(
            (attempt_root / name).exists() for name in scientific_dirs
        ),
        "path_repair_precedes_nested_launcher": (
            driver.index('export PATH="$(dirname "$uv_binary"):$PATH"')
            < driver.index("run_supervised()")
        ),
        "resume_reuses_completed_measurements": (
            'cp -a "$resume_evidence_root/initialization/canonical/."'
            in driver
            and 'cp -a "$resume_evidence_root/initialization/runtime/."'
            in driver
            and 'if test -z "$resume_evidence_root"; then' in driver
        ),
        "redeployment_budget_exact": (
            amendment["redeployment"]["attempt_number"] == 2
            and amendment["redeployment"]["remaining_after_this"] == 0
        ),
        "scientific_design_unchanged": (
            amendment["scientific_changes"] is False
        ),
    }
    return {
        "schema_version": 1,
        "analysis": "stage-c6-v2-bounded-repair-audit-v2-1",
        "passed": all(checks.values()),
        "checks": checks,
        "unchanged_source_checks": unchanged_source_checks,
        "changed_source_checks": changed_source_checks,
        "evidence_checks": evidence_checks,
    }
