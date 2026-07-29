from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from redco.analysis.stage_c4_v2_preregistration import (
    V1_TERMINAL_BUNDLE_SHA256,
    V1_TERMINAL_COMMIT,
    audit,
)
from redco.analysis.stage_c4_warmstart import SELECTION_THRESHOLDS


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _protocols(
    source_name: str,
    source_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    common_sft = {
        "runs": 1,
        "maximum_steps": 16,
        "checkpoint_interval": 1,
        "candidate_steps": list(range(1, 17)),
        "batch_size": 40,
        "micro_batch_size": 4,
        "shuffle": False,
        "learning_rate": 0.000005,
        "lora_rank": 32,
        "lora_alpha": 64,
        "optimizer": "adamw",
        "scheduler": "constant",
        "checkpoint_contents": "adapter-only safetensors",
        "one_run_rule": "one frozen run",
    }
    selection = {
        "rule": "earliest",
        "no_passing_candidate": "terminal",
        "buffered_thresholds": SELECTION_THRESHOLDS,
    }
    v1 = {
        "sft": {**common_sft, "config": "v1.toml", "seed": 7203001},
        "candidate_selection": selection,
    }
    v2 = {
        "status": "frozen_before_any_stage_c4_v2_model_load_or_optimizer_step",
        "v1_terminal_record": {
            "terminal_report_commit": V1_TERMINAL_COMMIT,
            "bundle_sha256": V1_TERMINAL_BUNDLE_SHA256,
            "selected_adapter": False,
            "scientific_arms_started": 0,
            "disposition": (
                "v1 remains closed; v2 is a distinct selection with fresh SFT randomness"
            ),
        },
        "sft": {**common_sft, "config": "v2.toml", "seed": 7203002},
        "candidate_selection": selection,
        "execution": {
            "scorer_lifecycle_gate": {
                "position": (
                    "after inherited-model merge and before the first v2 SFT optimizer step"
                ),
                "scorers": ["action", "root-route"],
                "child_exit_code_required": 0,
                "parent_signature_verification_required": True,
            }
        },
        "separation": {"scientific_reward_calls": 0, "rl_optimizer_steps": 0},
        "hardware": {
            "gpu": "2x A6000 48GB",
            "spot": False,
            "hourly_rate_usd": 1.08,
            "forbidden": ["A100", "H100"],
            "persistent_storage": False,
        },
        "source": {"sha256": {source_name: source_sha256}},
    }
    return v1, v2


def test_v2_audit_accepts_fresh_seed_and_crash_proof_gate(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("source\n", encoding="utf-8")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    v1, v2 = _protocols(source.name, source_sha256)
    v1_path = tmp_path / "v1.json"
    v2_path = tmp_path / "v2.json"
    _write(v1_path, v1)
    _write(v2_path, v2)

    result = audit(v1_path, v2_path, root=tmp_path)

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_v2_audit_rejects_reused_seed_or_relaxed_threshold(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("source\n", encoding="utf-8")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    v1, v2 = _protocols(source.name, source_sha256)
    v2["sft"]["seed"] = 7203001
    v2["candidate_selection"]["buffered_thresholds"] = {
        **SELECTION_THRESHOLDS,
        "minimum_route_mass": 0.01,
    }
    v1_path = tmp_path / "v1.json"
    v2_path = tmp_path / "v2.json"
    _write(v1_path, v1)
    _write(v2_path, v2)

    result = audit(v1_path, v2_path, root=tmp_path)

    assert result["passed"] is False
    assert result["checks"]["fresh_sft_seed"] is False
    assert result["checks"]["selection_thresholds_match_v1_and_code"] is False
