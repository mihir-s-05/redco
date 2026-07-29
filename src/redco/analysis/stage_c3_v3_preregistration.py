"""Machine audit for the Stage-C3 v3 credit-confusion protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

V2_BUNDLE_SHA256 = (
    "550caf8c1128eaa0694920755d2faa039fb6be555fefa50280de558f9fac561b"
)
DECISION_SHA256 = (
    "73453e9424f79b85542cba46f0656561c789b4a3d00d01755b04ab608f8c2d3b"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    v1_path: Path,
    v2_path: Path,
    v3_path: Path,
    *,
    root: Path = Path("."),
) -> dict[str, Any]:
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    rules = [
        _canonical(protocol["frozen_metrics_and_decision"])
        for protocol in (v1, v2, v3)
    ]
    v1_seeds = {int(run["seed"]) for run in v1["design"]["runs"]}
    v2_seeds = {int(run["seed"]) for run in v2["design"]["runs"]}
    v3_seeds = {int(run["seed"]) for run in v3["design"]["runs"]}
    source_checks = {
        name: _sha256(root / name) == expected
        for name, expected in v3["source"]["sha256"].items()
    }
    rendered_checks = {
        name: _sha256(root / name) == expected
        for name, expected in v3["rendered_configs"]["sha256"].items()
    }
    power = v3["execution"]["exact_power_gate"]
    abort = v3["execution"]["scientific_early_abort"]
    checks = {
        "decision_rules_byte_identical_to_v1_and_v2": (
            rules[0] == rules[1] == rules[2]
        ),
        "decision_rule_sha_is_canonical_original": (
            all(hashlib.sha256(rule).hexdigest() == DECISION_SHA256 for rule in rules)
        ),
        "fresh_seed_block_exact": v3_seeds == {9501, 9502, 9503, 9504},
        "fresh_seed_block_disjoint_from_v1_and_v2": (
            v3_seeds.isdisjoint(v1_seeds | v2_seeds)
        ),
        "v2_terminal_bundle_frozen": (
            v3["v2_terminal_record"]["bundle_sha256"] == V2_BUNDLE_SHA256
        ),
        "v2_scientific_gate_not_evaluated": (
            v3["v2_terminal_record"]["scientific_gate_evaluated"] is False
            and v3["v2_terminal_record"]["scientific_arms_started"] == 0
        ),
        "forced_smoke_is_deterministic": (
            v3["execution"]["forced_integration_smoke"][
                "sampled_pass_conditions"
            ]
            == 0
        ),
        "exact_power_floor_is_five_groups": (
            power["expected_target_informative_groups_per_sliced_step_minimum"]
            == 5.0
        ),
        "exact_power_precedes_scientific_runs": (
            power["position"] == "before_every_scientific_arm"
        ),
        "scientific_abort_has_no_sampled_outcome_rules": (
            abort["sampling_dependent_rules"] == 0
            and abort["computed_sampling_false_abort_probability"] == 0.0
        ),
        "root_seed_address_is_task_and_episode": (
            v3["execution"]["root_seed_contract"]
            == "sha256(master_seed, task_name, stable_episode_address)"
        ),
        "within_episode_crn_is_preserved": (
            v3["execution"]["within_episode_common_random_numbers"] is True
        ),
        "all_source_hashes_match": all(source_checks.values()),
        "all_rendered_config_hashes_match": all(rendered_checks.values()),
    }
    return {
        "schema_version": 1,
        "analysis": "stage-c3-v3-preregistration-audit",
        "passed": all(checks.values()),
        "checks": checks,
        "source_checks": source_checks,
        "rendered_config_checks": rendered_checks,
    }
