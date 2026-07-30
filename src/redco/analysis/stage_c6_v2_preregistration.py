"""Machine audit for the Stage-C6 v2 scientific preregistration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DECISION_SHA256 = (
    "73453e9424f79b85542cba46f0656561c789b4a3d00d01755b04ab608f8c2d3b"
)
STAGE_C2_ADAPTER_SHA256 = (
    "28fba5d421ea611db2e0d9cd411e40a0fc2035a9a45eb0bb3be24c84947e0ab6"
)
STAGE_C5_ADAPTER_SHA256 = (
    "e1d56f45485eef065bae42980427ee3c88176a5c864cbb350fa8494d0370e623"
)
REPLICATE_RULE = (
    "All three signed action payloads and all three signed root payloads "
    "must be byte-identical."
)
CANONICAL_FAILURE_RULE = (
    "Terminate v2 before scientific arms; do not change initialization, "
    "measurement, or thresholds."
)
OBSERVED_ARM_RULE = (
    "Never rerun an observed arm. Only exact continuation of previously "
    "unobserved frozen arms from retained byte-identical initialization is "
    "allowed."
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(protocol_path: Path, *, root: Path = Path(".")) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    design = protocol["design"]
    measurement = protocol["measurement_protocol"]
    repair = protocol["repair_and_redeploy_policy"]
    source_checks = {
        path: _sha256(root / path) == expected
        for path, expected in protocol["source"]["sha256"].items()
    }
    rendered_checks = {
        path: _sha256(root / path) == expected
        for path, expected in protocol["rendered_configs"]["sha256"].items()
    }
    rendered_semantics = {}
    for path in protocol["rendered_configs"]["sha256"]:
        text = (root / path).read_text(encoding="utf-8")
        rendered_semantics[path] = {
            "constraint_in_train_and_eval": (
                text.count("env.constrained_root_routes = true") == 2
            ),
            "selected_initialization_path": (
                text.count(
                    'name = "runs/stage-c6/selected-initialization-merged"'
                )
                == 1
            ),
        }
    decision_sha = hashlib.sha256(
        _canonical(protocol["frozen_metrics_and_decision"])
    ).hexdigest()
    seeds = {int(run["seed"]) for run in design["runs"]}
    canonical = measurement["canonical_scorer"]
    checks = {
        "protocol_frozen_before_v2_model_call": (
            protocol["status"]
            == "frozen_before_any_stage_c6_v2_model_call"
        ),
        "v1_terminal_zero_science_recorded": (
            protocol["stage_c6_v1_terminal"]["scientific_policy_calls"] == 0
            and protocol["stage_c6_v1_terminal"]["scientific_optimizer_steps"]
            == 0
            and protocol["stage_c6_v1_terminal"]["status"]
            == "terminal_support_failure"
        ),
        "original_scientific_decision_rule_unchanged": (
            decision_sha == DECISION_SHA256
        ),
        "fresh_seed_block_exact": seeds == {9911, 9912, 9913, 9914},
        "eight_runs_exact": design["total_runs"] == 8,
        "matched_policy_calls_exact": (
            design["matched_policy_calls_per_run"] == 576
            and design["total_policy_calls"] == 4608
        ),
        "selected_adapters_unchanged": (
            protocol["initialization"]["stage_c2_adapter_sha256"]
            == STAGE_C2_ADAPTER_SHA256
            and protocol["initialization"]["stage_c5_adapter_sha256"]
            == STAGE_C5_ADAPTER_SHA256
        ),
        "merged_model_byte_identity_required": (
            protocol["initialization"]["byte_identity_gate"]["required"]
            is True
        ),
        "canonical_replicates_exact": (
            canonical["replicates"] == 3
            and canonical["replicate_rule"] == REPLICATE_RULE
            and canonical["batch_size"] == 1
            and canonical["model_dtype"] == "float32"
            and canonical["normalizer"] == "python-math-fsum-float64"
        ),
        "factorization_thresholds_unchanged": (
            measurement["factorization"]["joint_tv_maximum"] == 0.05
            and measurement["factorization"][
                "mutual_information_nats_maximum"
            ]
            == 0.01
            and measurement["factorization"]["authority"]
            == "canonical_transformers_scorer"
        ),
        "runtime_power_is_separate": (
            measurement["runtime_power"]["authority"]
            == "same-deployment-vllm"
            and measurement["runtime_power"][
                "may_decide_factorization"
            ]
            is False
        ),
        "final_endpoints_are_canonical": (
            measurement["final_policy_endpoints"]["authority"]
            == "canonical_transformers_scorer"
        ),
        "same_constraint_in_both_arms_everywhere": (
            protocol["constrained_policy"]["broadcast_arm"] is True
            and protocol["constrained_policy"]["sliced_arm"] is True
            and protocol["constrained_policy"]["training"] is True
            and protocol["constrained_policy"]["evaluation"] is True
            and all(
                all(values.values()) for values in rendered_semantics.values()
            )
        ),
        "bounded_repair_is_outcome_independent": (
            repair["maximum_redeployments_after_initial"] == 1
            and repair["no_outcome_informed_repairs"] is True
            and repair["no_silent_retries"] is True
        ),
        "canonical_failure_stops_before_science": (
            repair["canonical_support_failure"] == CANONICAL_FAILURE_RULE
        ),
        "observed_arm_never_rerun": (
            repair["after_first_scientific_arm_outcome"]
            == OBSERVED_ARM_RULE
        ),
        "all_source_hashes_match": all(source_checks.values()),
        "all_rendered_hashes_match": all(rendered_checks.values()),
    }
    return {
        "schema_version": 1,
        "analysis": "stage-c6-v2-scientific-preregistration-audit",
        "passed": all(checks.values()),
        "checks": checks,
        "decision_rule_canonical_sha256": decision_sha,
        "source_checks": source_checks,
        "rendered_config_checks": rendered_checks,
        "rendered_semantics": rendered_semantics,
    }
