"""Audit the CPU-only readiness and budget of the Stage D scaffold successor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload


def _binomial_tail(*, trials: int, successes_at_least: int, p: float) -> float:
    return math.fsum(
        math.comb(trials, successes) * p**successes
        * (1.0 - p) ** (trials - successes)
        for successes in range(successes_at_least, trials + 1)
    )


def _wilson_lower(*, successes: int, trials: int, z: float) -> float:
    observed = successes / trials
    denominator = 1.0 + z**2 / trials
    center = observed + z**2 / (2.0 * trials)
    radius = z * math.sqrt(
        observed * (1.0 - observed) / trials
        + z**2 / (4.0 * trials**2)
    )
    return (center - radius) / denominator


def evaluate(design_path: Path) -> dict[str, Any]:
    design = json.loads(design_path.read_text(encoding="utf-8"))
    power = design["eligible_target_power"]
    budget = design["budget"]
    interface = design["interface_contract"]

    file_hashes = {}
    for key, hash_key in (
        ("shared_scaffold_prompt", "shared_scaffold_prompt_sha256"),
        ("taskset_source", "taskset_source_sha256"),
        ("feasibility_runner_source", "feasibility_runner_source_sha256"),
    ):
        if key == "taskset_source":
            path = Path(
                "environments/redco_evidence_selection_v2/"
                "redco_evidence_selection_v2/taskset.py"
            )
        elif key == "feasibility_runner_source":
            path = Path(
                "environments/redco_evidence_selection_v2/"
                "redco_evidence_selection_v2/run_feasibility.py"
            )
        else:
            path = Path(interface[key])
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        file_hashes[key] = {
            "path": str(path),
            "expected_sha256": interface[hash_key],
            "observed_sha256": observed,
            "passes": observed == interface[hash_key],
        }

    target_probability = float(power["minimum_underlying_probability"])
    group_probability = _binomial_tail(
        trials=int(power["groups_per_update"]),
        successes_at_least=int(power["minimum_groups_per_update"]),
        p=target_probability,
    )
    wilson_lower = _wilson_lower(
        successes=int(power["support_successes_required"]),
        trials=int(power["support_rollouts"]),
        z=float(power["two_sided_95pct_z"]),
    )

    setup_seconds = float(budget["setup_and_download_seconds"])
    p95_seconds = float(budget["observed_root_episode_p95_seconds"])
    root_rollouts_per_arm = int(budget["root_rollouts_per_arm"])
    training_seconds_per_arm = float(budget["training_seconds_per_arm"])
    contingency = float(budget["contingency_multiplier"])
    arms = int(budget["scientific_arms"])
    minimum_three_arm_seconds = contingency * (
        setup_seconds
        + arms * root_rollouts_per_arm * p95_seconds
        + arms * training_seconds_per_arm
    )
    spendable = float(budget["wallet_usd"]) - float(
        budget["reserve_usd"]
    )
    maximum_rate_for_minimum = (
        spendable * 3600.0 / minimum_three_arm_seconds
    )
    scenarios = {}
    for rate in budget["hourly_rate_scenarios_usd"]:
        numeric_rate = float(rate)
        scenarios[f"{numeric_rate:.2f}"] = {
            "minimum_three_arm_cost_usd": (
                minimum_three_arm_seconds / 3600.0 * numeric_rate
            ),
            "fits_wallet_after_reserve": (
                minimum_three_arm_seconds / 3600.0 * numeric_rate
                <= spendable
            ),
        }

    checks = {
        "five_of_eight_probability_at_least_95pct": (
            group_probability >= 0.95
        ),
        "support_wilson_lower_at_least_target_probability": (
            wilson_lower >= target_probability
        ),
        "fewshot_is_shared_scaffold_intervention": (
            design["fairness"]["classification"]
            == "shared_scaffold_intervention"
        ),
        "all_three_arms_required": (
            design["fairness"]["required_arms"]
            == ["stock", "branch-global", "local"]
        ),
        "density_and_informativeness_are_separate": (
            power["primary_support_metrics"]
            == [
                "selector_eligible_restorable_target_probability",
                "informative_branch_group_probability_conditional_on_eligible",
                "joint_eligible_and_informative_probability",
            ]
        ),
        "frozen_interface_file_hashes_match": all(
            record["passes"] for record in file_hashes.values()
        ),
        "full_campaign_budget_currently_proven": False,
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-scaffold-successor-cpu-readiness",
            "design_status": design["status"],
            "power_derivation": {
                "minimum_underlying_probability": target_probability,
                "probability_at_least_five_of_eight": group_probability,
                "support_rollouts": power["support_rollouts"],
                "support_successes_required": power[
                    "support_successes_required"
                ],
                "two_sided_95pct_wilson_lower": wilson_lower,
            },
            "interface_file_hashes": file_hashes,
            "minimum_budget_lower_bound": {
                "scope": (
                    "Three stock-equivalent root-rollout arms only. This "
                    "excludes alternative branch continuations, density and "
                    "informativeness support blocks, few-shot prompt-token "
                    "overhead, and any SFT, so it is a lower bound rather than "
                    "a complete campaign projection."
                ),
                "seconds": minimum_three_arm_seconds,
                "spendable_after_reserve_usd": spendable,
                "maximum_hourly_rate_to_fit_lower_bound_usd": (
                    maximum_rate_for_minimum
                ),
                "scenarios": scenarios,
            },
            "checks": checks,
            "passes": all(checks.values()),
            "decision": (
                "CPU interface work may continue. No paid few-shot support, "
                "SFT, branch-power audit, or scientific arm is authorized "
                "until a measured full-campaign ledger includes branch "
                "continuations and fits the wallet after reserve."
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(args.output, evaluate(args.design))


if __name__ == "__main__":
    main()
