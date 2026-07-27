"""Run and aggregate the preregistered CPU portion of Gate GC."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from redco.analysis.stage_c_estimator import audit_probe_estimator
from redco.contracts import canonical_json
from redco.env.tasks.credit_probes import standard_credit_probes


def evaluate_estimator_gate(
    *,
    samples_per_probe: int,
    master_seed: int,
    exogenous_seed_count: int,
    noise_floor: float,
    minimum_gradient_cosine: float,
    minimum_rank_correlation: float,
    minimum_sign_accuracy: float,
    maximum_advantage_rmse: float,
) -> dict[str, Any]:
    results = tuple(
        audit_probe_estimator(
            probe,
            samples=samples_per_probe,
            master_seed=master_seed,
            exogenous_seed_count=exogenous_seed_count,
            noise_floor=noise_floor,
        )
        for probe in standard_credit_probes()
    )
    informative = tuple(
        result
        for result in results
        if any(value != 0.0 for value in result.true_policy_gradient)
    )
    if not informative:
        raise RuntimeError("the probe suite contains no informative gradients")

    all_actions_observed = all(min(result.action_counts) > 0 for result in results)
    minimum_cosine = min(result.gradient_cosine for result in informative)
    minimum_rank = min(
        result.advantage_rank_correlation for result in informative
    )
    sign_outcomes = [
        result.sign_accuracy_above_noise
        for result in informative
        if result.sign_comparisons
    ]
    minimum_sign = min(sign_outcomes) if sign_outcomes else 1.0
    maximum_rmse = max(result.advantage_rmse for result in informative)
    checks = {
        "all_actions_observed": all_actions_observed,
        "gradient_cosine": minimum_cosine >= minimum_gradient_cosine,
        "advantage_rank_correlation": minimum_rank >= minimum_rank_correlation,
        "sign_accuracy_above_noise": minimum_sign >= minimum_sign_accuracy,
        "advantage_rmse": maximum_rmse <= maximum_advantage_rmse,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate": "GC-estimator-cpu",
        "passed": all(checks.values()),
        "settings": {
            "branch_group_size": 4,
            "samples_per_probe": samples_per_probe,
            "master_seed": master_seed,
            "exogenous_seed_count": exogenous_seed_count,
            "noise_floor": noise_floor,
        },
        "thresholds": {
            "minimum_gradient_cosine": minimum_gradient_cosine,
            "minimum_rank_correlation": minimum_rank_correlation,
            "minimum_sign_accuracy": minimum_sign_accuracy,
            "maximum_advantage_rmse": maximum_advantage_rmse,
        },
        "checks": checks,
        "headline": {
            "minimum_informative_gradient_cosine": minimum_cosine,
            "minimum_informative_rank_correlation": minimum_rank,
            "minimum_informative_sign_accuracy": minimum_sign,
            "maximum_informative_advantage_rmse": maximum_rmse,
        },
        "probes": [asdict(result) for result in results],
        "scope": (
            "Exact finite categorical policies over the restricted deterministic "
            "Stage-C probe suite. This clears estimator math only, not learning."
        ),
    }
    payload["report_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-probe", type=int, required=True)
    parser.add_argument("--master-seed", type=int, required=True)
    parser.add_argument("--exogenous-seed-count", type=int, required=True)
    parser.add_argument("--noise-floor", type=float, required=True)
    parser.add_argument("--minimum-gradient-cosine", type=float, required=True)
    parser.add_argument("--minimum-rank-correlation", type=float, required=True)
    parser.add_argument("--minimum-sign-accuracy", type=float, required=True)
    parser.add_argument("--maximum-advantage-rmse", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_estimator_gate(
        samples_per_probe=args.samples_per_probe,
        master_seed=args.master_seed,
        exogenous_seed_count=args.exogenous_seed_count,
        noise_floor=args.noise_floor,
        minimum_gradient_cosine=args.minimum_gradient_cosine,
        minimum_rank_correlation=args.minimum_rank_correlation,
        minimum_sign_accuracy=args.minimum_sign_accuracy,
        maximum_advantage_rmse=args.maximum_advantage_rmse,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
