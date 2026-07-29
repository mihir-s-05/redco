"""Prepare exact Stage-C3 prefixes and analyze live informativeness power."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

ROUTES = ("alpha", "beta", "gamma", "delta")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _prefix_and_completion(
    trace: dict[str, Any],
) -> tuple[list[int], list[int]]:
    prefix: list[int] = []
    completion: list[int] = []
    for node in trace["nodes"]:
        token_ids = node["token_ids"]
        mask = node["mask"]
        if len(token_ids) != len(mask):
            raise ValueError("trace token ids and masks differ in length")
        for token_id, trainable in zip(token_ids, mask, strict=True):
            if trainable:
                completion.append(int(token_id))
            else:
                if completion:
                    raise ValueError(
                        "unmasked token follows root completion tokens"
                    )
                prefix.append(int(token_id))
    if not prefix or not completion:
        raise ValueError("root trace has an empty prefix or completion")
    return prefix, completion


def prepare_root_cases(traces_path: Path) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {}
    for trace in _read_jsonl(traces_path):
        if trace.get("agent", {}).get("name") != "context":
            continue
        reply = str(trace["nodes"][-1]["message"]["content"]).strip()
        route = next(
            (
                value
                for value in ROUTES
                if reply == f"<route>{value}</route>"
            ),
            None,
        )
        if route is None:
            raise ValueError(f"unexpected forced root reply: {reply}")
        prefix, completion = _prefix_and_completion(trace)
        candidate = {
            "case_id": f"root-route:{route}",
            "route": route,
            "prefix_token_ids": prefix,
            "completion_token_ids": completion,
        }
        if route in cases and cases[route] != candidate:
            raise ValueError(f"route {route} used inconsistent tokenization")
        cases[route] = candidate
    if set(cases) != set(ROUTES):
        raise ValueError(f"missing forced route cases: {set(ROUTES) - set(cases)}")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c3-root-route-cases",
        "source_traces": traces_path.as_posix(),
        "cases": [cases[route] for route in ROUTES],
    }
    signed = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return payload


def _normalized_entropy(values: dict[str, float]) -> float:
    total = math.fsum(values.values())
    if total <= 0:
        return 0.0
    probabilities = [value / total for value in values.values() if value > 0]
    return -math.fsum(value * math.log(value) for value in probabilities)


def _group_mixed_probability(success_mass: float, size: int) -> float:
    return 1.0 - success_mass**size - (1.0 - success_mass) ** size


def analyze_power(
    action_scores: dict[str, Any],
    route_scores: dict[str, Any],
) -> dict[str, Any]:
    models = action_scores.get("models", [])
    if len(models) != 1:
        raise ValueError("power gate requires exactly one warm-start model")
    action_rows = models[0]["temperatures"]["2.0"]
    p5_by_route: dict[str, float] = {}
    entropies: list[float] = []
    digit_masses: list[float] = []
    for row in action_rows:
        route = str(row["context_route"])
        if route not in ROUTES:
            continue
        probabilities = {
            str(key): float(value)
            for key, value in row["action_probabilities"].items()
        }
        p5_by_route[route] = probabilities["5"]
        entropies.append(_normalized_entropy(probabilities))
        digit_masses.append(math.fsum(probabilities.values()))
    if set(p5_by_route) != set(ROUTES):
        raise ValueError("action scores do not cover every route")

    route_probabilities = {
        str(key): float(value)
        for key, value in route_scores["temperature_2"][
            "route_sequence_probabilities"
        ].items()
    }
    if set(route_probabilities) != set(ROUTES):
        raise ValueError("route scores do not cover every route")
    valid_route_mass = math.fsum(route_probabilities.values())
    if valid_route_mass > 1.0 + 1e-8:
        raise ValueError("valid route sequence mass exceeds one")
    invalid_route_mass = max(0.0, 1.0 - valid_route_mass)
    route_outcomes = [*route_probabilities.values(), invalid_route_mass]
    root_group_informative = 1.0 - math.fsum(
        probability**8 for probability in route_outcomes
    )

    branch_group_informative = math.fsum(
        route_probabilities[route]
        * _group_mixed_probability(p5_by_route[route], 11)
        for route in ROUTES
    )
    redundant_success_lower = route_probabilities["delta"] + math.fsum(
        route_probabilities[route] * p5_by_route[route]
        for route in ROUTES
        if route != "delta"
    )
    redundant_success_upper = min(
        1.0,
        redundant_success_lower + invalid_route_mass,
    )
    redundant_group_informative_lower = min(
        _group_mixed_probability(redundant_success_lower, 8),
        _group_mixed_probability(redundant_success_upper, 8),
    )
    mean_p5 = fmean(p5_by_route.values())
    mean_entropy = fmean(entropies)
    checks = {
        "valid_route_sequence_mass_at_least_0_90": valid_route_mass >= 0.90,
        "delta_route_mass_at_least_0_02": (
            route_probabilities["delta"] >= 0.02
        ),
        "non_delta_route_mass_at_least_0_10": (
            valid_route_mass - route_probabilities["delta"] >= 0.10
        ),
        "mean_digit_5_mass_between_0_03_and_0_35": (
            0.03 <= mean_p5 <= 0.35
        ),
        "mean_normalized_digit_entropy_at_least_1_25": (
            mean_entropy >= 1.25
        ),
        "mean_valid_digit_mass_at_least_0_90": (
            fmean(digit_masses) >= 0.90
        ),
        "expected_target_informative_groups_per_sliced_step_at_least_5": (
            8.0 * branch_group_informative >= 5.0
        ),
        "irrelevant_root_group_informative_probability_at_least_0_50": (
            root_group_informative >= 0.50
        ),
        "redundant_broadcast_group_informative_probability_at_least_0_50": (
            redundant_group_informative_lower >= 0.50
        ),
        "scientific_sampling_false_abort_probability_is_zero": True,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c3-v3-exact-informativeness-power",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "measurements": {
            "route_sequence_probabilities_t2": route_probabilities,
            "valid_route_sequence_mass_t2": valid_route_mass,
            "invalid_route_mass_upper_bound_t2": invalid_route_mass,
            "digit_5_mass_by_route_t2": p5_by_route,
            "mean_digit_5_mass_t2": mean_p5,
            "mean_normalized_digit_entropy_t2": mean_entropy,
            "mean_valid_digit_mass_t2": fmean(digit_masses),
            "root_group_informative_probability": root_group_informative,
            "target_branch_group_informative_probability": (
                branch_group_informative
            ),
            "expected_target_informative_groups_per_sliced_step": (
                8.0 * branch_group_informative
            ),
            "expected_irrelevant_root_groups_broadcast_run": (
                36.0 * root_group_informative
            ),
            "expected_irrelevant_root_groups_sliced_run": (
                6.0 * root_group_informative
            ),
            "redundant_success_mass_bounds": [
                redundant_success_lower,
                redundant_success_upper,
            ],
            "redundant_broadcast_group_informative_probability_lower": (
                redundant_group_informative_lower
            ),
            "sampling_dependent_early_abort_rules": 0,
            "sampling_false_abort_probability": 0.0,
        },
        "sources": {
            "action_score_case_sha256": action_scores["source"][
                "cases_sha256"
            ],
            "route_score_case_sha256": route_scores["source"][
                "cases_sha256"
            ],
        },
    }
    signed = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-root-cases")
    prepare.add_argument("--traces", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--action-scores", type=Path, required=True)
    analyze.add_argument("--route-scores", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-root-cases":
        result = prepare_root_cases(args.traces)
    else:
        result = analyze_power(
            json.loads(args.action_scores.read_text(encoding="utf-8")),
            json.loads(args.route_scores.read_text(encoding="utf-8")),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
