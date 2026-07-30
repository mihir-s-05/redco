"""Analyze the preregistered final-only subset surviving Stage C9 retention."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from redco.analysis.stage_c9_efficiency import (
    ARMS,
    SEEDS,
    _models,
    _policy_point,
    _practical_diagnostics,
    _redundant_rows,
    _reuse_contract,
    _usage,
)
from redco.integrations.signed_subprocess import atomic_write_json, sign_payload

FINAL_CALLS = 576


def evaluate(run_root: Path, scores_path: Path) -> dict[str, Any]:
    models = _models(scores_path)
    initial = _redundant_rows(models["warmstart"])
    initial_point = {
        "policy_calls": 0,
        **_policy_point(initial, initial),
    }
    runs: dict[str, Any] = {}
    for seed in SEEDS:
        for arm in ARMS:
            score_name = f"{arm}--s{seed}--final"
            current = _redundant_rows(models[score_name])
            final_point = {
                "policy_calls": FINAL_CALLS,
                **_policy_point(initial, current),
            }
            final_gain = (
                final_point["causal_non_delta_target_mass"]
                - initial_point["causal_non_delta_target_mass"]
            )
            updates = (
                36 if arm == "stock" else (12 if arm.endswith("e2") else 6)
            )
            run_dir = (
                run_root
                / "confusion_redundant"
                / f"{arm}-s{seed}"
            )
            result: dict[str, Any] = {
                "exact_initial": initial_point,
                "exact_final": {
                    **final_point,
                    "optimizer_step": updates,
                },
                "final_causal_mass_gain": final_gain,
                "optimizer_updates": updates,
                "final_gain_per_update_descriptive": final_gain / updates,
                "usage": _usage(run_dir),
                "exact_checkpoint_auc": {
                    "status": "unevaluable",
                    "reason": (
                        "Prime-RL retention deleted four of six frozen "
                        "checkpoint adapters before postprocessing."
                    ),
                },
            }
            if arm != "stock":
                result["practical_loss"] = _practical_diagnostics(
                    run_dir, updates
                )
            if arm.endswith("e2"):
                result["reuse_contract"] = _reuse_contract(run_dir)
            runs[f"{arm}--s{seed}"] = result

    e1_mean_gain = fmean(
        runs[f"local-e1--s{seed}"]["final_causal_mass_gain"]
        for seed in SEEDS
    )
    e2_mean_gain = fmean(
        runs[f"local-e2--s{seed}"]["final_causal_mass_gain"]
        for seed in SEEDS
    )
    if e1_mean_gain == 0.0:
        gain_ratio = math.inf if e2_mean_gain > 0.0 else math.nan
    else:
        gain_ratio = e2_mean_gain / e1_mean_gain
    local_drift = [
        runs[f"local-e2--s{seed}"]["exact_final"][
            "delta_nuisance_js_from_initial"
        ]
        for seed in SEEDS
    ]
    global_drift = [
        runs[f"branch-global-e2--s{seed}"]["exact_final"][
            "delta_nuisance_js_from_initial"
        ]
        for seed in SEEDS
    ]
    checks: dict[str, bool | None] = {
        "all_reuse_contracts_pass": all(
            runs[f"{arm}--s{seed}"]["reuse_contract"]["all_pairs_passed"]
            and runs[f"{arm}--s{seed}"]["reuse_contract"][
                "fresh_example_stream_between_collections"
            ]
            for arm in ("local-e2", "branch-global-e2")
            for seed in SEEDS
        ),
        "each_run_has_exactly_576_training_calls": all(
            run["usage"]["policy_calls"] == FINAL_CALLS
            for run in runs.values()
        ),
        "local_e2_auc_exceeds_e1_in_at_least_two_seeds": None,
        "local_e2_mean_final_gain_at_least_1_5x_e1": (
            math.isfinite(gain_ratio) and gain_ratio >= 1.5
        ),
        "global_e2_nuisance_exposure_floor_met": (
            fmean(global_drift) >= 0.002
        ),
        "local_e2_mean_delta_js_at_most_75pct_global": (
            fmean(local_drift) <= 0.75 * fmean(global_drift)
        ),
        "each_local_e2_delta_js_no_more_than_global_plus_0_002": all(
            local <= global_value + 0.002
            for local, global_value in zip(
                local_drift, global_drift, strict=True
            )
        ),
    }
    engineering_pass = bool(
        checks["all_reuse_contracts_pass"]
        and checks["each_run_has_exactly_576_training_calls"]
    )
    credit_pass = bool(
        checks["global_e2_nuisance_exposure_floor_met"]
        and checks["local_e2_mean_delta_js_at_most_75pct_global"]
        and checks[
            "each_local_e2_delta_js_no_more_than_global_plus_0_002"
        ]
    )
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-c9-final-only-partial-recovery",
            "status": (
                "terminal_postprocessing_failure_with_"
                "partial_preregistered_evidence"
            ),
            "engineering_reuse_and_ledger_pass": engineering_pass,
            "reuse_efficiency_gate": {
                "status": "indeterminate",
                "reason": (
                    "The conjunctive frozen gate requires exact six-point "
                    "checkpoint AUC. Deleted adapter bytes make that component "
                    "irrecoverable; endpoint gain alone cannot pass the gate."
                ),
                "endpoint_gain_component_pass": checks[
                    "local_e2_mean_final_gain_at_least_1_5x_e1"
                ],
            },
            "matched_data_credit_pass": credit_pass,
            "checks": checks,
            "summary": {
                "local_e1_mean_final_gain": e1_mean_gain,
                "local_e2_mean_final_gain": e2_mean_gain,
                "local_e2_to_e1_mean_final_gain_ratio": gain_ratio,
                "local_e2_mean_delta_js": fmean(local_drift),
                "branch_global_e2_mean_delta_js": fmean(global_drift),
            },
            "runs": runs,
            "claim_limits": [
                "No exact checkpoint-AUC or calls-to-threshold claim.",
                "No reuse-efficiency pass or fail verdict.",
                "Stock is a final-endpoint Pareto comparator only.",
                "Sampled live-evaluation curves are non-decision-bearing.",
                "No deleted checkpoint may be reconstructed by retraining.",
            ],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(args.output, evaluate(args.run_root, args.scores))


if __name__ == "__main__":
    main()
