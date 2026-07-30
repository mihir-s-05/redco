"""Audit the bounded infrastructure-repair Stage-C5 v3 successor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

V2_TERMINAL_COMMIT = "e28d8a9"
V2_BUNDLE_SHA256 = (
    "e1fe3bc633a5774c43836ac0ce09a6109349ed8b3779cd728aa1dbfe31189731"
)
V2_CONSTRAINT_SIGNATURE = (
    "e767efee087bbf89041a85dc524360f19662ad496963f0115ab2015eef63a83d"
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


def _without(value: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in keys}


def audit(
    v2_path: Path,
    v3_path: Path,
    *,
    root: Path = Path(),
) -> dict[str, Any]:
    """Verify that v3 changes repair policy and execution, not science."""
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    source_checks = {
        path: _sha256(root / path) == expected
        for path, expected in v3["source"]["sha256"].items()
    }
    terminal = v3["v2_terminal_record"]
    policy = v3["repair_and_redeploy_policy"]
    checks = {
        "status_is_frozen_before_v3_model_calls": (
            v3["status"]
            == "frozen_before_any_stage_c5_v3_live_model_call_or_optimizer_step"
        ),
        "v2_is_terminal_without_decision_bearing_observations": (
            terminal["terminal_report_commit"] == V2_TERMINAL_COMMIT
            and terminal["bundle_sha256"] == V2_BUNDLE_SHA256
            and terminal["constrained_interface_signature"]
            == V2_CONSTRAINT_SIGNATURE
            and terminal["constrained_interface_passed"] is True
            and terminal["sft_optimizer_steps"] == 32
            and terminal["candidate_model_calls"] == 0
            and terminal["candidate_evaluations"] == 0
            and terminal["selected_adapters"] == 0
            and terminal["scientific_arms_started"] == 0
        ),
        "failure_is_outcome_independent_launcher_expansion": (
            terminal["root_cause"]
            == "post-smoke continuation omitted signed case variables"
            and terminal["action_scorer_started"] is False
            and terminal["root_scorer_started"] is False
        ),
        "repair_boundary_is_explicit": (
            policy["maximum_redeployments_after_initial"] == 1
            and policy["eligibility_boundary"]
            == "before any candidate score payload or scientific-arm outcome"
            and policy["candidate_score_payload_is_decision_bearing"] is True
            and policy["scientific_arm_data_is_decision_bearing"] is True
            and policy["sft_loss_is_selection_ineligible"] is True
            and policy["no_silent_retries"] is True
            and policy["no_outcome_informed_repairs"] is True
        ),
        "repair_requires_fixed_science": (
            policy["immutable_fields"]
            == [
                "factorized corpus and renderer",
                "optimizer and 32-step horizon",
                "checkpoint cadence and candidate order",
                "support thresholds and selection rule",
                "constrained policy semantics",
                "scientific outcomes and decision rules",
            ]
        ),
        "fresh_seed_and_versioned_paths": (
            v3["sft"]["seed"] == 7203007
            and v3["sft"]["seed"] != v2["sft"]["seed"]
            and v3["constrained_interface_smoke"]["inference_seed"] == 9801
            and v3["execution"]["campaign_version"] == "v3"
        ),
        "scientific_design_is_unchanged": (
            v3["constrained_policy"] == v2["constrained_policy"]
            and v3["factorized_dataset"] == v2["factorized_dataset"]
            and v3["candidate_selection"] == v2["candidate_selection"]
            and v3["scientific_campaign_if_selected"]
            == v2["scientific_campaign_if_selected"]
            and _without(
                v3["sft"],
                "config",
                "seed",
                "one_run_rule",
            )
            == _without(
                v2["sft"],
                "config",
                "seed",
                "one_run_rule",
            )
        ),
        "constraint_supervisor_is_outcome_independent": (
            v3["constrained_interface_smoke"]["supervisor_mode"]
            == "constraint"
            and v3["constrained_interface_smoke"][
                "supervisor_decision_inputs"
            ]
            == [
                "rollout error fraction",
                "root completion budget contract",
                "route parseability",
            ]
            and v3["constrained_interface_smoke"][
                "supervisor_ignored_outcomes"
            ]
            == ["reward variance", "trainable fraction"]
        ),
        "runtime_regressions_are_frozen": (
            v3["execution"]["runtime_regressions"]
            == [
                "Bash syntax validation for shared and v3 drivers.",
                "Execute the real candidate command builder under bash -u and require both signed case arguments.",
                "Run a constant-reward, zero-trainable, parseable-route row through constraint supervisor mode and require pass.",
            ]
        ),
        "hardware_is_unselected_and_conservative": (
            v3["hardware"]["resource_id"] is None
            and v3["hardware"]["gpu_count"] == 2
            and v3["hardware"]["minimum_memory_per_gpu_gb"] == 48
            and v3["hardware"]["spot"] is False
            and v3["hardware"]["maximum_hourly_rate_usd"] <= 2.0
            and v3["hardware"]["persistent_storage"] is False
        ),
        "uv_only": (
            v3["execution"]["package_manager"] == "uv only; pip is forbidden"
        ),
        "all_source_hashes_match": all(source_checks.values()),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c5-constrained-successor-v3-preregistration-audit",
        "passed": all(checks.values()),
        "checks": checks,
        "source_checks": source_checks,
        "v2_sft_seed": v2["sft"]["seed"],
        "v3_sft_seed": v3["sft"]["seed"],
        "v3_smoke_seed": v3["constrained_interface_smoke"]["inference_seed"],
    }
    result["signed_payload_sha256"] = _canonical_sha256(result)
    return result
