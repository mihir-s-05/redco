"""Audit the distinct Stage-C4 renderer-aligned warm-start selection v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from redco.analysis.stage_c4_warmstart import SELECTION_THRESHOLDS

V2_TERMINAL_COMMIT = "555db13"
V2_TERMINAL_BUNDLE_SHA256 = (
    "8186b730c0529f0c92b54036d7fa4caad5b92a1cf66bfd16b3cda638797178d9"
)
ALIGNMENT_SIGNATURE = (
    "f4fa5fa4b45bb307455bef31e4c721f43d53d4319579b0089e0a7efcb363b9af"
)
MISMATCH_SIGNATURE = (
    "7bd7d8209158dfb57f855f7cba604ec91fbea2e2b254bd5043158865627261f9"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit(
    v2_path: Path,
    v3_path: Path,
    *,
    root: Path = Path(),
) -> dict[str, Any]:
    """Verify that v3 changes only the diagnosed renderer defect and freshness."""
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    source_checks = {
        path: _sha256(root / path) == expected
        for path, expected in v3["source"]["sha256"].items()
    }
    checks = {
        "status_is_frozen_before_v3_model_calls": (
            v3["status"]
            == "frozen_before_any_stage_c4_v3_model_load_or_optimizer_step"
        ),
        "v2_is_terminal_and_scientifically_empty": (
            v3["v2_terminal_record"]["terminal_report_commit"]
            == V2_TERMINAL_COMMIT
            and v3["v2_terminal_record"]["bundle_sha256"]
            == V2_TERMINAL_BUNDLE_SHA256
            and v3["v2_terminal_record"]["selected_adapter"] is False
            and v3["v2_terminal_record"]["scientific_arms_started"] == 0
        ),
        "renderer_defect_is_exactly_recorded": (
            v3["renderer_defect"]["v2_mismatch_signature"] == MISMATCH_SIGNATURE
            and v3["renderer_defect"]["v3_alignment_signature"]
            == ALIGNMENT_SIGNATURE
            and v3["renderer_defect"]["v2_renderer"] == "qwen3"
            and v3["renderer_defect"]["v3_renderer"] == "prime-qwen3"
            and v3["renderer_defect"]["mismatched_rows"] == 40
        ),
        "fresh_sft_seed": (
            v3["sft"]["seed"] == 7203003
            and v3["sft"]["seed"] != v2["sft"]["seed"]
        ),
        "sft_design_changes_are_declared": (
            v3["sft"]["changes_from_v2"]
            == [
                "fresh seed 7203003",
                "renderer qwen3 -> prime-qwen3",
                "versioned input/output paths",
            ]
        ),
        "renderer_matches_campaign": (
            v3["sft"]["renderer"] == v3["campaign_renderer"] == "prime-qwen3"
        ),
        "selection_thresholds_unchanged": (
            v3["candidate_selection"]["buffered_thresholds"]
            == v2["candidate_selection"]["buffered_thresholds"]
            == SELECTION_THRESHOLDS
        ),
        "selection_rule_unchanged": (
            v3["candidate_selection"]["rule"]
            == v2["candidate_selection"]["rule"]
            and v3["candidate_selection"]["no_passing_candidate"]
            == v2["candidate_selection"]["no_passing_candidate"]
        ),
        "alignment_gate_precedes_optimizer": (
            v3["execution"]["renderer_alignment_gate"]["position"]
            == "after inherited-model merge and before the first v3 SFT optimizer step"
            and v3["execution"]["renderer_alignment_gate"]["required_rows_exact"]
            == 40
        ),
        "scientific_work_still_separated": (
            v3["separation"]["scientific_reward_calls"] == 0
            and v3["separation"]["rl_optimizer_steps"] == 0
        ),
        "hardware_is_eligible_nonspot": (
            v3["hardware"]["gpu"] == "2x A40 48GB"
            and v3["hardware"]["spot"] is False
            and v3["hardware"]["hourly_rate_usd"] <= 2.0
            and "A100" in v3["hardware"]["forbidden"]
            and "H100" in v3["hardware"]["forbidden"]
        ),
        "no_persistent_storage": v3["hardware"]["persistent_storage"] is False,
        "all_source_hashes_match": all(source_checks.values()),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c4-warmstart-selection-v3-preregistration-audit",
        "passed": all(checks.values()),
        "checks": checks,
        "source_checks": source_checks,
        "v2_seed": v2["sft"]["seed"],
        "v3_seed": v3["sft"]["seed"],
    }
    result["signed_payload_sha256"] = _canonical_sha256(result)
    return result
