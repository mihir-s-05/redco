"""Audit the distinct Stage-C4 extended factorized selection v4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from redco.analysis.stage_c4_warmstart import SELECTION_THRESHOLDS

V3_TERMINAL_COMMIT = "5a32341"
V3_TERMINAL_BUNDLE_SHA256 = (
    "df1d1f04ad58be73a739c95c5835c9029d5822f831c68283688c62a2a01ceffa"
)
DESIGN_SIGNATURE = (
    "bebe296ff8e21f1b817e46a24ea3d3b1ddb468307e2d48523fb924b2f4c7b2cb"
)
V4_CANDIDATE_STEPS = list(range(2, 33, 2))


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
    v3_path: Path,
    v4_path: Path,
    *,
    root: Path = Path(),
) -> dict[str, Any]:
    """Verify v4 extends only the reward-blind marginal-shaping horizon."""
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    v4 = json.loads(v4_path.read_text(encoding="utf-8"))
    source_checks = {
        path: _sha256(root / path) == expected
        for path, expected in v4["source"]["sha256"].items()
    }
    changes = v4["sft"]["changes_from_v3"]
    checks = {
        "status_is_frozen_before_v4_model_calls": (
            v4["status"]
            == "frozen_before_any_stage_c4_v4_model_load_or_optimizer_step"
        ),
        "v3_is_terminal_and_scientifically_empty": (
            v4["v3_terminal_record"]["terminal_report_commit"]
            == V3_TERMINAL_COMMIT
            and v4["v3_terminal_record"]["bundle_sha256"]
            == V3_TERMINAL_BUNDLE_SHA256
            and v4["v3_terminal_record"]["selected_adapter"] is False
            and v4["v3_terminal_record"]["scientific_arms_started"] == 0
            and v4["v3_terminal_record"]["scientific_reward_calls"] == 0
            and v4["v3_terminal_record"]["rl_optimizer_steps"] == 0
        ),
        "cpu_design_analysis_is_pinned": (
            v4["design_analysis"]["signed_payload_sha256"]
            == DESIGN_SIGNATURE
            and v4["design_analysis"]["root_only_disposition"] == "rejected"
        ),
        "fresh_sft_seed": (
            v4["sft"]["seed"] == 7203004
            and v4["sft"]["seed"] != v3["sft"]["seed"]
        ),
        "declared_design_changes_only": (
            changes
            == [
                "fresh seed 7203004",
                "maximum steps 16 -> 32",
                "checkpoint interval 1 -> 2",
                "candidate steps 1..16 -> even steps 2..32",
                "versioned input/output paths",
            ]
        ),
        "same_renderer_corpus_and_optimizer": (
            v4["sft"]["renderer"] == v3["sft"]["renderer"] == "prime-qwen3"
            and v4["factorized_dataset"] == v3["factorized_dataset"]
            and v4["sft"]["learning_rate"] == v3["sft"]["learning_rate"]
            and v4["sft"]["lora_rank"] == v3["sft"]["lora_rank"]
            and v4["sft"]["lora_alpha"] == v3["sft"]["lora_alpha"]
            and v4["sft"]["optimizer"] == v3["sft"]["optimizer"]
            and v4["sft"]["scheduler"] == v3["sft"]["scheduler"]
        ),
        "extended_even_candidate_horizon_is_exact": (
            v4["sft"]["maximum_steps"] == 32
            and v4["sft"]["checkpoint_interval"] == 2
            and v4["sft"]["candidate_steps"] == V4_CANDIDATE_STEPS
            and v4["candidate_selection"]["candidate_steps"]
            == V4_CANDIDATE_STEPS
        ),
        "selection_thresholds_unchanged": (
            v4["candidate_selection"]["buffered_thresholds"]
            == v3["candidate_selection"]["buffered_thresholds"]
            == SELECTION_THRESHOLDS
        ),
        "selection_and_terminal_rules_unchanged": (
            v4["candidate_selection"]["rule"]
            == v3["candidate_selection"]["rule"]
            and v4["candidate_selection"]["no_passing_candidate"]
            == v3["candidate_selection"]["no_passing_candidate"]
        ),
        "scientific_work_still_separated": (
            v4["separation"]["scientific_reward_calls"] == 0
            and v4["separation"]["rl_optimizer_steps"] == 0
        ),
        "hardware_rule_is_conservative": (
            v4["hardware"]["resource_id"] is None
            and v4["hardware"]["spot"] is False
            and v4["hardware"]["maximum_hourly_rate_usd"] <= 2.0
            and v4["hardware"]["gpu_count"] == 2
            and v4["hardware"]["minimum_memory_per_gpu_gb"] == 48
            and "A100" in v4["hardware"]["forbidden"]
            and "H100" in v4["hardware"]["forbidden"]
        ),
        "hardware_requires_preprovision_amendment": (
            v4["hardware"]["resource_pin_rule"]
            == (
                "Before provisioning, commit a hardware-only amendment naming "
                "the exact eligible resource ID, provider, location, and rate."
            )
        ),
        "no_persistent_storage": v4["hardware"]["persistent_storage"] is False,
        "all_source_hashes_match": all(source_checks.values()),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c4-warmstart-selection-v4-preregistration-audit",
        "passed": all(checks.values()),
        "checks": checks,
        "source_checks": source_checks,
        "v3_seed": v3["sft"]["seed"],
        "v4_seed": v4["sft"]["seed"],
        "v4_candidate_steps": V4_CANDIDATE_STEPS,
    }
    result["signed_payload_sha256"] = _canonical_sha256(result)
    return result
