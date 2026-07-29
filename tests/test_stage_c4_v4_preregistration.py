from __future__ import annotations

import hashlib
import json

from redco.analysis.stage_c4_v4_preregistration import (
    DESIGN_SIGNATURE,
    V3_TERMINAL_BUNDLE_SHA256,
    V3_TERMINAL_COMMIT,
    V4_CANDIDATE_STEPS,
    audit,
)
from redco.analysis.stage_c4_warmstart import SELECTION_THRESHOLDS


def test_v4_audit_pins_terminal_v3_and_unchanged_scientific_gate(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("frozen", encoding="utf-8")
    factorized_dataset = {"path": "same", "examples": 40}
    candidate_selection = {
        "candidate_steps": V4_CANDIDATE_STEPS,
        "buffered_thresholds": SELECTION_THRESHOLDS,
        "rule": "ascending",
        "no_passing_candidate": "terminal",
    }
    v3 = {
        "factorized_dataset": factorized_dataset,
        "sft": {
            "seed": 7203003,
            "renderer": "prime-qwen3",
            "learning_rate": 5e-6,
            "lora_rank": 32,
            "lora_alpha": 64,
            "optimizer": "adamw",
            "scheduler": "constant",
        },
        "candidate_selection": {
            **candidate_selection,
            "candidate_steps": list(range(1, 17)),
        },
    }
    v4 = {
        "status": "frozen_before_any_stage_c4_v4_model_load_or_optimizer_step",
        "v3_terminal_record": {
            "terminal_report_commit": V3_TERMINAL_COMMIT,
            "bundle_sha256": V3_TERMINAL_BUNDLE_SHA256,
            "selected_adapter": False,
            "scientific_arms_started": 0,
            "scientific_reward_calls": 0,
            "rl_optimizer_steps": 0,
        },
        "design_analysis": {
            "signed_payload_sha256": DESIGN_SIGNATURE,
            "root_only_disposition": "rejected",
        },
        "factorized_dataset": factorized_dataset,
        "sft": {
            "seed": 7203004,
            "renderer": "prime-qwen3",
            "maximum_steps": 32,
            "checkpoint_interval": 2,
            "candidate_steps": V4_CANDIDATE_STEPS,
            "learning_rate": 5e-6,
            "lora_rank": 32,
            "lora_alpha": 64,
            "optimizer": "adamw",
            "scheduler": "constant",
            "changes_from_v3": [
                "fresh seed 7203004",
                "maximum steps 16 -> 32",
                "checkpoint interval 1 -> 2",
                "candidate steps 1..16 -> even steps 2..32",
                "versioned input/output paths",
            ],
        },
        "candidate_selection": candidate_selection,
        "separation": {"scientific_reward_calls": 0, "rl_optimizer_steps": 0},
        "hardware": {
            "resource_id": None,
            "spot": False,
            "maximum_hourly_rate_usd": 2.0,
            "gpu_count": 2,
            "minimum_memory_per_gpu_gb": 48,
            "forbidden": ["A100", "H100"],
            "resource_pin_rule": (
                "Before provisioning, commit a hardware-only amendment naming "
                "the exact eligible resource ID, provider, location, and rate."
            ),
            "persistent_storage": False,
        },
        "source": {
            "sha256": {
                "source.txt": hashlib.sha256(b"frozen").hexdigest(),
            }
        },
    }
    v3_path = tmp_path / "v3.json"
    v4_path = tmp_path / "v4.json"
    v3_path.write_text(json.dumps(v3), encoding="utf-8")
    v4_path.write_text(json.dumps(v4), encoding="utf-8")
    result = audit(v3_path, v4_path, root=tmp_path)
    assert result["passed"] is True
    assert all(result["checks"].values())
