"""Machine audit for the Stage-C6 scientific campaign preregistration."""

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


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(protocol_path: Path, *, root: Path = Path(".")) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    design = protocol["design"]
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
    seeds = {int(run["seed"]) for run in design["runs"]}
    decision_sha = hashlib.sha256(
        _canonical(protocol["frozen_metrics_and_decision"])
    ).hexdigest()
    checks = {
        "protocol_frozen_before_scientific_model_call": (
            protocol["status"]
            == "frozen_before_any_stage_c6_scientific_model_call"
        ),
        "original_scientific_decision_rule_unchanged": (
            decision_sha == DECISION_SHA256
        ),
        "fresh_seed_block_exact": seeds == {9901, 9902, 9903, 9904},
        "eight_runs_exact": design["total_runs"] == 8,
        "matched_policy_calls_exact": (
            design["matched_policy_calls_per_run"] == 576
            and design["total_policy_calls"] == 4608
        ),
        "both_primary_outcomes_reported_separately": (
            protocol["reporting"]["composite_score"] is False
            and len(protocol["reporting"]["primary_outcomes"]) == 2
        ),
        "selected_stage_c2_adapter_frozen": (
            protocol["initialization"]["stage_c2_adapter_sha256"]
            == STAGE_C2_ADAPTER_SHA256
        ),
        "selected_stage_c5_adapter_frozen": (
            protocol["initialization"]["stage_c5_adapter_sha256"]
            == STAGE_C5_ADAPTER_SHA256
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
            and repair["scientific_arm_outcome_is_decision_bearing"] is True
        ),
        "observed_arm_never_rerun": (
            repair["after_first_scientific_arm_outcome"]
            == "Never rerun an observed arm. Only exact continuation of previously unobserved frozen arms from retained byte-identical initialization is allowed."
        ),
        "all_source_hashes_match": all(source_checks.values()),
        "all_rendered_hashes_match": all(rendered_checks.values()),
    }
    return {
        "schema_version": 1,
        "analysis": "stage-c6-scientific-preregistration-audit-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "decision_rule_canonical_sha256": decision_sha,
        "source_checks": source_checks,
        "rendered_config_checks": rendered_checks,
        "rendered_semantics": rendered_semantics,
    }
