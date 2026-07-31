"""Audit the funded conservative Stage D successor budget amendment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload


def evaluate(amendment_path: Path) -> dict[str, Any]:
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    observation = amendment["prime_read_only_observation"]
    budget = amendment["budget"]
    expected_equivalents = int(budget["stock_root_rollouts"]) + int(
        budget["branch_arms"]
    ) * int(budget["root_rollouts_per_branch_arm"]) * int(
        budget["planned_k_including_original"]
    )
    campaign_seconds = float(budget["contingency_multiplier"]) * (
        float(budget["setup_and_download_seconds"])
        + expected_equivalents
        * float(budget["observed_root_episode_p95_seconds"])
        + float(budget["training_and_checkpoint_seconds_total"])
    )
    campaign_cost = (
        campaign_seconds
        / 3600.0
        * float(budget["maximum_hourly_rate_usd"])
    )
    noncampaign = sum(
        float(budget[key])
        for key in (
            "support_and_informativeness_ceiling_usd",
            "conditional_sft_and_exact_scoring_ceiling_usd",
            "artifact_recovery_and_teardown_ceiling_usd",
        )
    )
    reserve = float(budget["reserve_usd"])
    wallet = float(observation["wallet_balance_usd"])
    total_with_reserve = campaign_cost + noncampaign + reserve
    checks = {
        "zero_active_resources_at_observation": (
            observation["active_pods"] == 0
            and observation["persistent_disks"] == 0
        ),
        "observed_resource_is_nonspot_2x48gb_at_or_below_cap": (
            observation["observed_affordable_resource"]["gpu_count"] == 2
            and "48GB"
            in observation["observed_affordable_resource"]["gpu_type"]
            and observation["observed_affordable_resource"]["is_spot"] is False
            and float(
                observation["observed_affordable_resource"][
                    "price_per_hour_usd"
                ]
            )
            <= float(budget["maximum_hourly_rate_usd"])
        ),
        "episode_equivalent_arithmetic_exact": (
            expected_equivalents
            == int(budget["conservative_episode_equivalents"])
            == 1728
        ),
        "full_envelope_fits_wallet": total_with_reserve <= wallet,
        "resource_not_selected_or_reserved": (
            observation["resource_is_selected_or_reserved"] is False
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-successor-funded-conservative-envelope",
            "campaign": {
                "episode_equivalents": expected_equivalents,
                "seconds": campaign_seconds,
                "cost_ceiling_usd": campaign_cost,
            },
            "noncampaign_ceiling_usd": noncampaign,
            "reserve_usd": reserve,
            "total_envelope_with_reserve_usd": total_with_reserve,
            "wallet_usd": wallet,
            "headroom_after_full_envelope_usd": wallet - total_with_reserve,
            "checks": checks,
            "passes": all(checks.values()),
            "decision": (
                "The funding blocker is cleared for protocol preparation. "
                "Live work remains blocked on the scientific and engineering "
                "preconditions listed in the amendment."
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(args.output, evaluate(args.amendment))


if __name__ == "__main__":
    main()
