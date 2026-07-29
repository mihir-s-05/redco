"""Audit the bounded Stage-C5 constrained-interface successor protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from redco.analysis.stage_c4_warmstart import SELECTION_THRESHOLDS
from redco.analysis.stage_c5_constrained import ROUTE_CHOICES, SEMANTICS

V4_TERMINAL_COMMIT = "7723749"
V4_TERMINAL_BUNDLE_SHA256 = (
    "e0983c490bfdf8283ff7ad27dcc99e85923dfc7f0528cb4710926812244b0b59"
)
V4_RESCORE_SIGNATURE = (
    "d424cc6037c66754816a431ee706cfd8987a3f33d2848d21f27ee0842d1e4281"
)
CANDIDATE_STEPS = list(range(2, 33, 2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit(protocol_path: Path, *, root: Path = Path()) -> dict[str, Any]:
    """Verify C5 is bounded, symmetric, estimator-explicit, and byte frozen."""
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_checks = {
        path: _sha256(root / path) == expected
        for path, expected in protocol["source"]["sha256"].items()
    }
    smoke = protocol["constrained_interface_smoke"]
    selection = protocol["candidate_selection"]
    stop = protocol["terminal_stopping_rule"]
    scientific = protocol["scientific_campaign_if_selected"]
    checks = {
        "status_is_frozen_before_model_calls": (
            protocol["status"]
            == "frozen_before_any_stage_c5_live_model_call_or_optimizer_step"
        ),
        "v4_is_terminal_and_scientifically_empty": (
            protocol["v4_terminal_record"]["terminal_report_commit"]
            == V4_TERMINAL_COMMIT
            and protocol["v4_terminal_record"]["bundle_sha256"]
            == V4_TERMINAL_BUNDLE_SHA256
            and protocol["v4_terminal_record"]["selected_adapter"] is False
            and protocol["v4_terminal_record"]["scientific_arms_started"] == 0
            and protocol["v4_terminal_record"]["scientific_reward_calls"] == 0
            and protocol["v4_terminal_record"]["rl_optimizer_steps"] == 0
        ),
        "zero_cost_rescore_is_pinned": (
            protocol["constrained_rescore"]["signed_payload_sha256"]
            == V4_RESCORE_SIGNATURE
            and protocol["constrained_rescore"]["earliest_passing_step"] == 18
            and protocol["constrained_rescore"]["passing_steps"]
            == [18, 20, 22, 24, 26, 28, 30, 32]
            and protocol["constrained_rescore"]["interpretation"]
            == "design calibration only; not a scientific result"
        ),
        "constraint_is_exact_and_symmetric": (
            protocol["constrained_policy"]["semantics"] == SEMANTICS
            and protocol["constrained_policy"]["choices"] == list(ROUTE_CHOICES)
            and protocol["constrained_policy"]["rollout"] is True
            and protocol["constrained_policy"]["branch_sampling"] is True
            and protocol["constrained_policy"]["evaluation"] is True
            and protocol["constrained_policy"]["broadcast_arm"] is True
            and protocol["constrained_policy"]["sliced_arm"] is True
        ),
        "behavior_logprob_contract_is_explicit": (
            protocol["constrained_policy"]["behavior_logprob"]
            == "exact renormalized four-choice categorical at the unique divergent token"
            and protocol["constrained_policy"]["trainer_logprob_policy"]
            == (
                "The live smoke must show both trace and packed-batch behavior "
                "logprobs agree with the exact constrained categorical within "
                "0.02 nats; otherwise C5 terminates before SFT."
            )
        ),
        "smoke_precedes_sft_and_is_terminal": (
            smoke["position"] == "before the C5 SFT optimizer"
            and smoke["maximum_rl_optimizer_steps"] == 1
            and smoke["scientific_arm"] is False
            and smoke["failure_handling"]
            == "Terminate C5 without SFT, scientific arms, retry, or fallback."
        ),
        "fresh_seed_and_single_run": (
            protocol["sft"]["seed"] == 7203005
            and protocol["sft"]["runs"] == 1
            and protocol["sft"]["maximum_steps"] == 32
            and protocol["sft"]["candidate_steps"] == CANDIDATE_STEPS
            and protocol["sft"]["one_run_rule"]
            == (
                "After the first C5 live model call, do not rerun the smoke; "
                "after the first SFT optimizer step, do not rerun SFT or train "
                "replacement candidates."
            )
        ),
        "v4_thresholds_are_unchanged": (
            selection["candidate_steps"] == CANDIDATE_STEPS
            and selection["buffered_thresholds"] == SELECTION_THRESHOLDS
            and selection["rule"]
            == (
                "Select the earliest ascending optimizer checkpoint satisfying "
                "every unchanged v4 threshold under constrained root semantics; "
                "later checkpoints are not consulted."
            )
        ),
        "initialization_engineering_has_hard_stop": (
            stop["trigger"] == "No C5 candidate satisfies every unchanged threshold."
            and stop["initialization_engineering_ends"] is True
            and stop["forbidden_next_step"]
            == "No C6 or other replacement initialization design may be launched."
            and stop["allowed_dispositions"]
            == [
                "Run one separately preregistered, explicitly underpowered "
                "exploratory campaign at the measured support.",
                "Stop live experiments and write the systems and CPU results.",
            ]
        ),
        "scientific_outcomes_are_separate": (
            scientific["launch_before_selected_adapter"] is False
            and scientific["primary_outcomes"]
            == [
                "known-noncausal-action Jensen-Shannon drift",
                "policy calls to the frozen causal-action threshold",
            ]
            and scientific["composite_score"] is False
            and scientific["final_reward_role"] == "sanity check only"
            and scientific["branch_cost_ledger"] is True
        ),
        "hardware_is_conservative": (
            protocol["hardware"]["resource_id"] is None
            and protocol["hardware"]["gpu_count"] == 2
            and protocol["hardware"]["minimum_memory_per_gpu_gb"] == 48
            and protocol["hardware"]["spot"] is False
            and protocol["hardware"]["maximum_hourly_rate_usd"] <= 2.0
            and protocol["hardware"]["persistent_storage"] is False
            and "A100" in protocol["hardware"]["forbidden"]
            and "H100" in protocol["hardware"]["forbidden"]
        ),
        "hardware_requires_committed_amendment": (
            protocol["hardware"]["resource_pin_rule"]
            == (
                "Before provisioning, commit a hardware-only amendment naming "
                "the exact eligible resource ID, provider, location, and rate."
            )
        ),
        "uv_only": protocol["execution"]["package_manager"] == "uv only; pip is forbidden",
        "compact_artifact_policy": (
            protocol["execution"]["retain_intermediate_adapters"] is False
            and protocol["execution"]["retain_merged_4b_models_locally"] is False
            and protocol["execution"]["retain_optimizer_state"] is False
            and protocol["execution"]["retain_cuda_or_uv_caches"] is False
        ),
        "all_source_hashes_match": all(source_checks.values()),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c5-constrained-successor-preregistration-audit",
        "passed": all(checks.values()),
        "checks": checks,
        "source_checks": source_checks,
        "candidate_steps": CANDIDATE_STEPS,
        "constraint_semantics": SEMANTICS,
    }
    result["signed_payload_sha256"] = _canonical_sha256(result)
    return result
