"""Audit the zero-information Stage-C5 v2 constrained successor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

V1_TERMINAL_COMMIT = "c43a3d1"
V1_BUNDLE_SHA256 = (
    "c2c77790cceb89192464369b5df3f11ca174dcc9604a7562599dd96ad6578886"
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
    v1_path: Path,
    v2_path: Path,
    *,
    root: Path = Path(),
) -> dict[str, Any]:
    """Verify v2 changes only outcome-blind execution identity and source."""
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    source_checks = {
        path: _sha256(root / path) == expected
        for path, expected in v2["source"]["sha256"].items()
    }
    terminal = v2["v1_terminal_record"]
    checks = {
        "status_is_frozen_before_v2_model_calls": (
            v2["status"]
            == "frozen_before_any_stage_c5_v2_live_model_call_or_optimizer_step"
        ),
        "v1_is_terminal_and_zero_information": (
            terminal["terminal_report_commit"] == V1_TERMINAL_COMMIT
            and terminal["bundle_sha256"] == V1_BUNDLE_SHA256
            and terminal["context_samples"] == 0
            and terminal["rewards"] == 0
            and terminal["serialized_training_batches"] == 0
            and terminal["rl_optimizer_steps"] == 0
            and terminal["sft_optimizer_steps"] == 0
            and terminal["candidate_evaluations"] == 0
            and terminal["scientific_arms_started"] == 0
        ),
        "failure_is_exactly_launcher_syntax": (
            terminal["launcher_exit_code"] == 1
            and terminal["supervisor_exit_code"] == 42
            and terminal["first_training_row_before_timeout"] is False
            and terminal["root_cause"] == "invalid multiline Bash parameter expansion"
        ),
        "runtime_regression_is_frozen": (
            v2["execution"]["launcher_runtime_regression"]
            == (
                "Invoke scripts/run_stage_c2_campaign_arm.sh through Bash, "
                "require a nonzero missing-arguments exit, and reject any "
                "'bad substitution' output before provisioning."
            )
        ),
        "fresh_seeds_and_versioned_paths": (
            v2["sft"]["seed"] == 7203006
            and v2["sft"]["seed"] != v1["sft"]["seed"]
            and v2["constrained_interface_smoke"]["inference_seed"] == 9701
            and v2["execution"]["campaign_version"] == "v2"
        ),
        "experimental_design_is_unchanged": (
            v2["constrained_policy"] == v1["constrained_policy"]
            and v2["candidate_selection"] == v1["candidate_selection"]
            and v2["terminal_stopping_rule"] == v1["terminal_stopping_rule"]
            and v2["scientific_campaign_if_selected"]
            == v1["scientific_campaign_if_selected"]
            and v2["factorized_dataset"] == v1["factorized_dataset"]
        ),
        "declared_changes_are_exhaustive": (
            v2["changes_from_v1"]
            == [
                "corrected campaign-launcher parameter expansion",
                "fresh SFT seed 7203006",
                "fresh smoke seed 9701 and exogenous offsets",
                "versioned v2 paths",
                "frozen runtime launcher regression",
            ]
        ),
        "one_run_rule_is_preserved": (
            v2["sft"]["runs"] == 1
            and v2["sft"]["maximum_steps"] == 32
            and v2["sft"]["candidate_steps"] == v1["sft"]["candidate_steps"]
            and v2["sft"]["one_run_rule"] == v1["sft"]["one_run_rule"]
        ),
        "hardware_is_unselected_and_conservative": (
            v2["hardware"]["resource_id"] is None
            and v2["hardware"]["gpu_count"] == 2
            and v2["hardware"]["minimum_memory_per_gpu_gb"] == 48
            and v2["hardware"]["spot"] is False
            and v2["hardware"]["maximum_hourly_rate_usd"] <= 2.0
            and v2["hardware"]["persistent_storage"] is False
        ),
        "hardware_amendment_is_required": (
            v2["hardware"]["resource_pin_rule"] == v1["hardware"]["resource_pin_rule"]
        ),
        "uv_only": v2["execution"]["package_manager"] == "uv only; pip is forbidden",
        "all_source_hashes_match": all(source_checks.values()),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c5-constrained-successor-v2-preregistration-audit",
        "passed": all(checks.values()),
        "checks": checks,
        "source_checks": source_checks,
        "v1_sft_seed": v1["sft"]["seed"],
        "v2_sft_seed": v2["sft"]["seed"],
        "v2_smoke_seed": v2["constrained_interface_smoke"]["inference_seed"],
    }
    result["signed_payload_sha256"] = _canonical_sha256(result)
    return result
