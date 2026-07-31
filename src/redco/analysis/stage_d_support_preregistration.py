"""Audit the frozen Stage D scaffold support preregistration."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import sign_payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binomial_tail(
    *, trials: int, minimum_successes: int, probability: float
) -> float:
    return math.fsum(
        math.comb(trials, successes)
        * probability**successes
        * (1 - probability) ** (trials - successes)
        for successes in range(minimum_successes, trials + 1)
    )


def _wilson_lower(
    *, successes: int, trials: int, z: float = 1.959963984540054
) -> float:
    observed = successes / trials
    denominator = 1 + z**2 / trials
    center = observed + z**2 / (2 * trials)
    radius = z * math.sqrt(
        observed * (1 - observed) / trials
        + z**2 / (4 * trials**2)
    )
    return (center - radius) / denominator


def evaluate(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    hashes = {
        name: {
            "expected": expected,
            "observed": _sha256(Path(name)),
            "passes": expected == _sha256(Path(name)),
        }
        for name, expected in protocol["source_hashes"].items()
    }
    partition = json.loads(
        Path(
            "datasets/stage-d/qasper-scaffold-successor-manifest-v2.json"
        ).read_text(encoding="utf-8")
    )
    sft_audit = json.loads(
        Path("reports/stage-d0-scaffold-sft-audit-v2.json").read_text(
            encoding="utf-8"
        )
    )
    funding = json.loads(
        Path(
            "configs/stage-d/"
            "stage-d0-scaffold-successor-funding-amendment-v2-2.json"
        ).read_text(encoding="utf-8")
    )
    with Path(
        "configs/stage-d/stage-d0-scaffold-sft-v2.toml"
    ).open("rb") as handle:
        sft = tomllib.load(handle)
    runner = Path(
        "scripts/run_stage_d0_scaffold_support_v3.sh"
    ).read_text(encoding="utf-8")

    wilson = _wilson_lower(successes=58, trials=64)
    group_probability = _binomial_tail(
        trials=8,
        minimum_successes=5,
        probability=0.808,
    )
    partitions = partition["partitions"]
    paper_sets = {
        name: set(record["paper_ids"])
        for name, record in partitions.items()
    }
    pairwise_disjoint = all(
        not paper_sets[left] & paper_sets[right]
        for index, left in enumerate(paper_sets)
        for right in list(paper_sets)[index + 1 :]
    )
    checks = {
        "status_is_frozen": (
            protocol["status"] == "frozen_before_any_v3_model_call"
        ),
        "all_source_hashes_match": all(
            record["passes"] for record in hashes.values()
        ),
        "partition_manifest_passes": partition["passes"] is True,
        "four_paper_sets_pairwise_disjoint": pairwise_disjoint,
        "support_blocks_cover_all_answer_types": all(
            set(partitions[name]["answer_types"])
            == {"abstractive", "extractive", "yes_no"}
            for name in ("fewshot_support", "power_audit")
        ),
        "sft_audit_passes": sft_audit["passes"] is True,
        "sft_is_fixed_eight_steps": sft["max_steps"] == 8,
        "sft_is_nonadaptive": (
            sft["data"]["shuffle"] is False
            and sft["data"]["seed"] == 7402001
        ),
        "checkpoint_retention_is_2_4_6_8": (
            sft["ckpt"]["interval"] == 2
            and sft["ckpt"]["keep_last"] == 4
            and sft["ckpt"]["weights_only"] is True
        ),
        "support_wilson_lower_exceeds_0808": wilson >= 0.808,
        "five_of_eight_probability_at_least_095": (
            group_probability >= 0.95
        ),
        "three_future_arms_required": (
            protocol["future_science_fairness"]["required_arms"]
            == [
                "shared-scaffold broadcast baseline",
                "shared-scaffold branch-global",
                "shared-scaffold local",
            ]
        ),
        "full_envelope_fits_wallet": (
            funding["full_envelope_with_reserve_usd"]
            <= funding["wallet_balance_usd"]
            and funding["headroom_usd"] > 0
        ),
        "runner_uses_one_structural_target": (
            runner.count("--maximum-targets 1") == 2
        ),
        "runner_uses_three_alternatives": (
            runner.count("--alternatives-per-target 3") == 2
        ),
        "runner_allows_iid_duplicate_branches": (
            runner.count(
                "--minimum-distinct-candidate-fraction "
                "0.3333333333333333"
            )
            == 2
        ),
        "runner_never_invokes_pip": "pip " not in runner,
        "runner_has_no_wandb_upload": "wandb online" not in runner.lower(),
        "science_is_separately_preregistered": (
            protocol["future_science_fairness"][
                "scientific_campaign_requires_separate_preregistration"
            ]
            is True
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d0-scaffold-support-preregistration-audit-v3",
            "protocol": path.as_posix(),
            "protocol_sha256": _sha256(path),
            "source_hashes": hashes,
            "power": {
                "wilson_lower_58_of_64": wilson,
                "probability_at_least_5_of_8_at_p_0808": (
                    group_probability
                ),
            },
            "budget": {
                "wallet_usd": funding["wallet_balance_usd"],
                "full_envelope_with_reserve_usd": funding[
                    "full_envelope_with_reserve_usd"
                ],
                "headroom_usd": funding["headroom_usd"],
            },
            "checks": checks,
            "passes": all(checks.values()),
            "decision": (
                "live_support_gate_ready_for_final_adversarial_review"
                if all(checks.values())
                else "blocked"
            ),
        }
    )
