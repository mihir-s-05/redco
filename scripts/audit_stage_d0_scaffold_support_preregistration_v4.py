"""Audit the frozen Stage D0 scaffold-support v4 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wilson_lower(successes: int, total: int, z: float = 1.95996398454) -> float:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = proportion + z * z / (2 * total)
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total
        + z * z / (4 * total * total)
    )
    return (center - radius) / denominator


def audit(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_results = {}
    for relative, expected in protocol["source_sha256"].items():
        path = root / relative
        actual = _sha256(path) if path.is_file() else None
        source_results[relative] = {
            "expected": expected,
            "actual": actual,
            "passes": actual == expected,
        }

    partition = protocol["paper_partition"]
    rows = [
        json.loads(line)
        for line in (
            root / partition["dataset"]
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_split = Counter(str(row["split"]) for row in rows)
    paper_sets = {
        split: {
            str(row["paper_id"])
            for row in rows
            if row["split"] == split
        }
        for split in (
            "fewshot_support",
            "power_audit",
            "science_train",
            "science_eval",
        )
    }
    all_papers = set().union(*paper_sets.values())
    answer_types = {
        split: {
            str(row["answer_type"])
            for row in rows
            if row["split"] == split
        }
        for split in paper_sets
    }

    budget = protocol["budget"]
    recomputed_envelope = (
        budget["support_and_informativeness_ceiling_usd"]
        + budget["conditional_sft_and_scoring_ceiling_usd"]
        + budget["artifact_recovery_ceiling_usd"]
        + budget["scientific_training_reserved_ceiling_usd"]
        + budget["science_evaluation_reserved_usd"]
        + budget["untouchable_reserve_usd"]
    )
    runner = (
        root / "scripts/run_stage_d0_scaffold_support_v4.sh"
    ).read_text(encoding="utf-8")
    power = protocol["frozen_cascade"][
        "step_4_independent_power_audit"
    ]
    checks = {
        "frozen_before_output": (
            protocol["status"] == "frozen_before_any_v4_model_call"
        ),
        "v3_closed_without_output": (
            protocol["history"]["v3_model_calls"] == 0
            and protocol["history"]["v3_optimizer_steps"] == 0
        ),
        "all_source_hashes_exact": (
            bool(source_results)
            and all(row["passes"] for row in source_results.values())
        ),
        "partition_counts_exact": by_split
        == {
            "fewshot_support": 8,
            "power_audit": 64,
            "science_train": 16,
            "science_eval": 32,
        },
        "all_120_papers_unique_and_disjoint": (
            len(all_papers) == 120
            and sum(len(value) for value in paper_sets.values()) == 120
        ),
        "answer_strata_present": all(
            value == {"abstractive", "extractive", "yes_no"}
            for value in answer_types.values()
        ),
        "paper_is_power_unit": (
            partition["power_audit"]["papers"] == 64
            and partition["power_audit"]["replicates_per_paper"] == 1
        ),
        "threshold_unchanged": (
            power["joint_pass_rule"]
            == (
                "at least 58 of 64 unique papers are both eligible "
                "and informative"
            )
            and power["minimum_f1_range"] == 0.05
        ),
        "wilson_lower_exceeds_0_808": _wilson_lower(58, 64) > 0.808,
        "full_budget_arithmetic_exact": (
            math.isclose(
                recomputed_envelope,
                budget["full_envelope_usd"],
                abs_tol=1e-9,
            )
            and math.isclose(
                budget["wallet_verified_usd"]
                - budget["full_envelope_usd"],
                budget["headroom_usd"],
                abs_tol=1e-9,
            )
        ),
        "one_gpu_support_rate_bound": (
            protocol["support_hardware"][
                "maximum_rate_usd_per_hour"
            ]
            == 1.0
            and all(
                "1x48GB" in item
                for item in protocol["support_hardware"]["allowed"]
            )
        ),
        "hard_six_hour_timeout_executable": (
            'timeout --signal=TERM 21600 bash "$0"' in runner
        ),
        "all_64_power_slots_executed": (
            '"$run_root/power-audit" 64 1' in runner
        ),
        "summary_driven_denominator": (
            "--summary \"$run_root/power-audit/run-summary.json\""
            in runner
            and "materialize_stage_d_power_records.py" in runner
        ),
        "real_sft_preflight_before_inference": (
            runner.index("audit_stage_d_sft_renderer_v4.py")
            < runner.index('start_inference "$base_config"')
            and "sft @ \"$sft_train_config\" --dry-run" in runner
        ),
        "ephemeral_merged_model": (
            "/tmp/redco-stage-d0-scaffold-support-v4-sft-merged"
            in runner
            and 'rm -rf "$sft_dir" "$sft_reloaded" "$sft_merged"'
            in runner
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d0-scaffold-support-preregistration-audit-v4",
            "protocol": protocol_path.as_posix(),
            "protocol_sha256": _sha256(protocol_path),
            "wilson_lower_58_of_64": _wilson_lower(58, 64),
            "recomputed_budget_envelope_usd": recomputed_envelope,
            "split_counts": dict(by_split),
            "answer_types": {
                key: sorted(value) for key, value in answer_types.items()
            },
            "source_results": source_results,
            "checks": checks,
            "passes": all(checks.values()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "configs/stage-d/"
            "stage-d0-scaffold-support-preregistration-v4.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, args.protocol)
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
