"""Audit the in-place Stage-C6 v3 trace-parser repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    protocol_path: Path,
    first_amendment_path: Path,
    second_amendment_path: Path,
    *,
    root: Path = Path("."),
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    first = json.loads(first_amendment_path.read_text(encoding="utf-8"))
    second = json.loads(second_amendment_path.read_text(encoding="utf-8"))
    expected_sources = dict(protocol["source"]["sha256"])
    for amendment in (first, second):
        for path, replacement in amendment["source_replacements"].items():
            expected_sources[path] = replacement["new_sha256"]
    source_checks = {
        path: _sha256(root / path) == expected
        for path, expected in expected_sources.items()
    }
    rendered_checks = {
        path: _sha256(root / path) == expected
        for path, expected in protocol["rendered_configs"]["sha256"].items()
    }
    smoke = second["completed_smoke"]
    checks = {
        "base_and_first_amendment_bytes_match": (
            _sha256(protocol_path) == second["base_protocol_sha256"]
            and _sha256(first_amendment_path)
            == second["first_amendment_sha256"]
        ),
        "smoke_evidence_bytes_match": (
            _sha256(root / smoke["evidence_archive"])
            == smoke["evidence_archive_sha256"]
        ),
        "completed_smoke_never_rerun": (
            second["continuation"]["rerun_smoke"] is False
            and second["continuation"]["reuse_recorded_smoke_bytes"] is True
        ),
        "failure_is_verifier_only": (
            smoke["signed_interface_result_exists"] is False
            and smoke["scientific_policy_calls"] == 0
            and smoke["scientific_optimizer_steps"] == 0
            and second["classification"]
            == "outcome-independent verifier parser defect"
        ),
        "same_deployment_continuation": (
            second["continuation"]["new_deployment"] is False
            and second["continuation"]["deployment_number"] == 2
            and second["continuation"]["remaining_redeployments"] == 0
        ),
        "scientific_fields_unchanged": (
            second["scientific_changes"] == []
            and second["rendered_config_changes"] == []
            and second["decision_rule_changes"] == []
        ),
        "all_chained_source_hashes_match": all(source_checks.values()),
        "all_rendered_hashes_match": all(rendered_checks.values()),
    }
    return {
        "schema_version": 1,
        "analysis": "stage-c6-v3-in-place-parser-repair-audit",
        "passed": all(checks.values()),
        "checks": checks,
        "source_checks": source_checks,
        "rendered_checks": rendered_checks,
    }
