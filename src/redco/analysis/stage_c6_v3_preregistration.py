"""Machine audit for the exact-likelihood Stage-C6 v3 successor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

V2_DECISION_SHA256 = (
    "73453e9424f79b85542cba46f0656561c789b4a3d00d01755b04ab608f8c2d3b"
)
ROUTE_TOKEN_IDS = [7141, 19127, 32214, 20255]
STAGE_C2_ADAPTER_SHA256 = (
    "28fba5d421ea611db2e0d9cd411e40a0fc2035a9a45eb0bb3be24c84947e0ab6"
)
STAGE_C5_ADAPTER_SHA256 = (
    "e1d56f45485eef065bae42980427ee3c88176a5c864cbb350fa8494d0370e623"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def audit(protocol_path: Path, *, root: Path = Path(".")) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    predecessor_path = root / protocol["predecessor"]["protocol"]
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    predecessor_decision_sha = hashlib.sha256(
        _canonical(predecessor["frozen_metrics_and_decision"])
    ).hexdigest()
    source_checks = {
        path: _sha256(root / path) == expected
        for path, expected in protocol["source"]["sha256"].items()
    }
    rendered_checks = {
        path: _sha256(root / path) == expected
        for path, expected in protocol["rendered_configs"]["sha256"].items()
    }
    semantics: dict[str, dict[str, bool]] = {}
    for path in protocol["rendered_configs"]["sha256"]:
        text = (root / path).read_text(encoding="utf-8")
        is_smoke = "structural-" in path
        semantics[path] = {
            "constraint_train_and_eval": (
                text.count("env.constrained_root_routes = true") == 2
            ),
            "exact_group_once": (
                text.count("[trainer.exact_categorical]") == 1
                and text.count(
                    "token_groups = [[7141,19127,32214,20255]]"
                )
                == 1
            ),
            "full_logits_enabled": (
                text.count(
                    'fused_lm_head_token_chunk_size = "disabled"'
                )
                == 1
            ),
            "token_export_smoke_only": (
                text.count("enable_token_export = true")
                == (1 if is_smoke else 0)
            ),
        }
    seeds = {int(run["seed"]) for run in protocol["design"]["runs"]}
    exact = protocol["exact_training_policy"]
    repair = protocol["repair_and_redeploy_policy"]
    checks = {
        "frozen_before_v3_model_call": (
            protocol["status"]
            == "frozen_before_any_stage_c6_v3_model_call"
        ),
        "predecessor_had_zero_scientific_calls": (
            protocol["predecessor"]["scientific_policy_calls"] == 0
            and protocol["predecessor"]["scientific_optimizer_steps"] == 0
        ),
        "decision_rule_is_exact_v2_rule": (
            predecessor_decision_sha == V2_DECISION_SHA256
            and protocol["decision_rule"]["canonical_sha256"]
            == V2_DECISION_SHA256
            and protocol["decision_rule"]["changes_from_v2"] == []
        ),
        "fresh_seed_block_exact": seeds == {9921, 9922, 9923, 9924},
        "eight_runs_and_matched_calls_exact": (
            protocol["design"]["total_runs"] == 8
            and protocol["design"]["matched_policy_calls_per_run"] == 576
            and protocol["design"]["total_policy_calls"] == 4608
        ),
        "same_adapters_exact": (
            protocol["initialization"]["stage_c2_adapter_sha256"]
            == STAGE_C2_ADAPTER_SHA256
            and protocol["initialization"]["stage_c5_adapter_sha256"]
            == STAGE_C5_ADAPTER_SHA256
        ),
        "exact_training_group_frozen": (
            exact["route_token_ids"] == ROUTE_TOKEN_IDS
            and exact["normalization"]
            == "selected_logit_minus_logsumexp_of_four_allowed_logits"
            and exact["gradient_includes_log_normalizer"] is True
            and exact["applies_to_broadcast_and_sliced"] is True
        ),
        "live_smoke_checks_transport_and_trainer": (
            protocol["execution"]["smoke"][
                "expected_context_trace_count"
            ]
            == 8
            and protocol["execution"]["smoke"][
                "static_reference_is_decision_bearing"
            ]
            is False
            and protocol["execution"]["smoke"][
                "token_export_required"
            ]
            is True
        ),
        "reuse_never_rescores_initialization": (
            protocol["execution"]["inherited_initialization_evidence"][
                "rescore"
            ]
            is False
        ),
        "bounded_outcome_independent_repair": (
            repair["maximum_redeployments_after_initial"] == 1
            and repair["no_outcome_informed_repairs"] is True
            and repair["never_rerun_observed_scientific_arm"] is True
        ),
        "all_rendered_semantics_match": all(
            all(values.values()) for values in semantics.values()
        ),
        "all_source_hashes_match": all(source_checks.values()),
        "all_rendered_hashes_match": all(rendered_checks.values()),
    }
    return {
        "schema_version": 1,
        "analysis": "stage-c6-v3-scientific-preregistration-audit",
        "passed": all(checks.values()),
        "checks": checks,
        "predecessor_decision_rule_sha256": predecessor_decision_sha,
        "source_checks": source_checks,
        "rendered_config_checks": rendered_checks,
        "rendered_semantics": semantics,
    }
