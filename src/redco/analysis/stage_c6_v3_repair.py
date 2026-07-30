"""Audit the one allowed outcome-independent Stage-C6 v3 redeployment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    protocol_path: Path,
    amendment_path: Path,
    *,
    root: Path = Path("."),
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    replacements = amendment["source_replacements"]
    unchanged_source_checks = {
        path: _sha256(root / path) == expected
        for path, expected in protocol["source"]["sha256"].items()
        if path not in replacements
    }
    replacement_checks = {
        path: (
            protocol["source"]["sha256"][path] == values["old_sha256"]
            and _sha256(root / path) == values["new_sha256"]
        )
        for path, values in replacements.items()
    }
    rendered_checks = {
        path: _sha256(root / path) == expected
        for path, expected in protocol["rendered_configs"]["sha256"].items()
    }
    checks = {
        "base_protocol_bytes_match": (
            _sha256(protocol_path)
            == amendment["base_protocol_sha256"]
        ),
        "attempt_evidence_bytes_match": (
            _sha256(root / amendment["attempt_1"]["evidence_archive"])
            == amendment["attempt_1"]["evidence_archive_sha256"]
        ),
        "failure_is_outcome_independent": (
            amendment["attempt_1"]["classification"]
            == "outcome-independent launcher path defect"
            and amendment["attempt_1"]["signed_interface_result_exists"]
            is False
            and amendment["attempt_1"]["scientific_policy_calls"] == 0
            and amendment["attempt_1"]["scientific_optimizer_steps"] == 0
        ),
        "single_redeployment_consumed": (
            amendment["redeployment"]["deployment_number"] == 2
            and amendment["redeployment"][
                "remaining_redeployments_after_launch"
            ]
            == 0
        ),
        "scientific_fields_unchanged": (
            amendment["scientific_changes"] == []
            and amendment["rendered_config_changes"] == []
            and amendment["decision_rule_changes"] == []
        ),
        "replacement_hashes_match": all(replacement_checks.values()),
        "unchanged_source_hashes_match": all(
            unchanged_source_checks.values()
        ),
        "rendered_hashes_match": all(rendered_checks.values()),
    }
    return {
        "schema_version": 1,
        "analysis": "stage-c6-v3-outcome-independent-repair-audit",
        "passed": all(checks.values()),
        "checks": checks,
        "replacement_checks": replacement_checks,
        "unchanged_source_checks": unchanged_source_checks,
        "rendered_checks": rendered_checks,
    }
