from __future__ import annotations

import hashlib
import json

from redco.analysis.stage_c4_v3_preregistration import (
    ALIGNMENT_SIGNATURE,
    MISMATCH_SIGNATURE,
    V2_TERMINAL_BUNDLE_SHA256,
    V2_TERMINAL_COMMIT,
    audit,
)
from redco.analysis.stage_c4_warmstart import SELECTION_THRESHOLDS


def test_v3_audit_pins_renderer_fix_and_unchanged_thresholds(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("frozen", encoding="utf-8")
    v2 = {
        "sft": {"seed": 7203002},
        "candidate_selection": {
            "buffered_thresholds": SELECTION_THRESHOLDS,
            "rule": "ascending",
            "no_passing_candidate": "terminal",
        },
    }
    v3 = {
        "status": "frozen_before_any_stage_c4_v3_model_load_or_optimizer_step",
        "v2_terminal_record": {
            "terminal_report_commit": V2_TERMINAL_COMMIT,
            "bundle_sha256": V2_TERMINAL_BUNDLE_SHA256,
            "selected_adapter": False,
            "scientific_arms_started": 0,
        },
        "renderer_defect": {
            "v2_mismatch_signature": MISMATCH_SIGNATURE,
            "v3_alignment_signature": ALIGNMENT_SIGNATURE,
            "v2_renderer": "qwen3",
            "v3_renderer": "prime-qwen3",
            "mismatched_rows": 40,
        },
        "sft": {
            "seed": 7203003,
            "renderer": "prime-qwen3",
            "changes_from_v2": [
                "fresh seed 7203003",
                "renderer qwen3 -> prime-qwen3",
                "versioned input/output paths",
            ],
        },
        "campaign_renderer": "prime-qwen3",
        "candidate_selection": {
            "buffered_thresholds": SELECTION_THRESHOLDS,
            "rule": "ascending",
            "no_passing_candidate": "terminal",
        },
        "execution": {
            "renderer_alignment_gate": {
                "position": (
                    "after inherited-model merge and before the first v3 "
                    "SFT optimizer step"
                ),
                "required_rows_exact": 40,
            }
        },
        "separation": {"scientific_reward_calls": 0, "rl_optimizer_steps": 0},
        "hardware": {
            "gpu": "2x A40 48GB",
            "spot": False,
            "hourly_rate_usd": 0.8968,
            "persistent_storage": False,
            "forbidden": ["A100", "H100"],
        },
        "source": {
            "sha256": {
                "source.txt": hashlib.sha256(b"frozen").hexdigest(),
            }
        },
    }
    v2_path = tmp_path / "v2.json"
    v3_path = tmp_path / "v3.json"
    v2_path.write_text(json.dumps(v2), encoding="utf-8")
    v3_path.write_text(json.dumps(v3), encoding="utf-8")
    result = audit(v2_path, v3_path, root=tmp_path)
    assert result["passed"] is True
    assert all(result["checks"].values())
