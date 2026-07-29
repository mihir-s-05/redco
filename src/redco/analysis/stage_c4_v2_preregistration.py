"""Audit the distinct Stage-C4 warm-start selection v2 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from redco.analysis.stage_c4_warmstart import SELECTION_THRESHOLDS

V1_TERMINAL_BUNDLE_SHA256 = "ace8f9b6f853965ad2a54adbca33cc69dfb1df26906683f0bab847a673a422ef"
V1_TERMINAL_COMMIT = "d6c5206"


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
    v1_path: Path,
    v2_path: Path,
    *,
    root: Path = Path(),
) -> dict[str, Any]:
    """Audit v2 against the terminal v1 protocol and current frozen files."""
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v1_seed = v1["sft"]["seed"]
    v2_seed = v2["sft"]["seed"]
    source_checks = {
        path: _sha256(root / path) == expected for path, expected in v2["source"]["sha256"].items()
    }
    checks = {
        "status_is_frozen_before_v2_model_calls": (
            v2["status"] == "frozen_before_any_stage_c4_v2_model_load_or_optimizer_step"
        ),
        "v1_is_explicitly_terminal": (
            v2["v1_terminal_record"]["terminal_report_commit"] == V1_TERMINAL_COMMIT
            and v2["v1_terminal_record"]["bundle_sha256"] == V1_TERMINAL_BUNDLE_SHA256
            and v2["v1_terminal_record"]["selected_adapter"] is False
            and v2["v1_terminal_record"]["scientific_arms_started"] == 0
        ),
        "v2_is_declared_distinct_not_retry": (
            v2["v1_terminal_record"]["disposition"]
            == "v1 remains closed; v2 is a distinct selection with fresh SFT randomness"
        ),
        "fresh_sft_seed": (isinstance(v2_seed, int) and v2_seed != v1_seed and v2_seed == 7203002),
        "sft_design_unchanged_except_seed_and_paths": (
            {key: value for key, value in v2["sft"].items() if key not in {"config", "seed"}}
            == {key: value for key, value in v1["sft"].items() if key not in {"config", "seed"}}
        ),
        "selection_thresholds_match_v1_and_code": (
            v2["candidate_selection"]["buffered_thresholds"]
            == v1["candidate_selection"]["buffered_thresholds"]
            == SELECTION_THRESHOLDS
        ),
        "selection_rule_unchanged": (
            v2["candidate_selection"]["rule"] == v1["candidate_selection"]["rule"]
            and v2["candidate_selection"]["no_passing_candidate"]
            == v1["candidate_selection"]["no_passing_candidate"]
        ),
        "lifecycle_gate_precedes_sft": (
            v2["execution"]["scorer_lifecycle_gate"]["position"]
            == "after inherited-model merge and before the first v2 SFT optimizer step"
        ),
        "lifecycle_gate_runs_both_exact_scorers": (
            v2["execution"]["scorer_lifecycle_gate"]["scorers"] == ["action", "root-route"]
        ),
        "lifecycle_requires_zero_and_parent_signature_verification": (
            v2["execution"]["scorer_lifecycle_gate"]["child_exit_code_required"] == 0
            and v2["execution"]["scorer_lifecycle_gate"]["parent_signature_verification_required"]
            is True
        ),
        "scientific_work_still_separated": (
            v2["separation"]["scientific_reward_calls"] == 0
            and v2["separation"]["rl_optimizer_steps"] == 0
        ),
        "hardware_is_eligible_nonspot": (
            v2["hardware"]["gpu"] == "2x A6000 48GB"
            and v2["hardware"]["spot"] is False
            and v2["hardware"]["hourly_rate_usd"] <= 2.0
            and "A100" in v2["hardware"]["forbidden"]
            and "H100" in v2["hardware"]["forbidden"]
        ),
        "no_persistent_storage": v2["hardware"]["persistent_storage"] is False,
        "all_source_hashes_match": all(source_checks.values()),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c4-warmstart-selection-v2-preregistration-audit",
        "passed": all(checks.values()),
        "checks": checks,
        "source_checks": source_checks,
        "v1_seed": v1_seed,
        "v2_seed": v2_seed,
    }
    result["signed_payload_sha256"] = _canonical_sha256(result)
    return result
