"""Build and select the factorized Stage-C3 v4 shared warm start."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from redco.analysis.stage_c3_power import ROUTES, analyze_power
from redco.integrations.stage_c_prompts import (
    stage_c_branch_prompt,
    stage_c_root_prompt,
)

DIGITS = tuple(str(value) for value in range(8))
ROOT_REPEATS_PER_ROUTE = 2
SELECTION_THRESHOLDS: dict[str, float] = {
    "valid_route_sequence_mass_minimum": 0.92,
    "minimum_route_mass": 0.05,
    "delta_route_mass_minimum": 0.10,
    "delta_route_mass_maximum": 0.35,
    "maximum_route_mass": 0.55,
    "normalized_root_entropy_minimum": 1.10,
    "mean_digit_5_mass_minimum": 0.05,
    "mean_digit_5_mass_maximum": 0.25,
    "mean_digit_entropy_minimum": 1.40,
    "mean_valid_digit_mass_minimum": 0.92,
    "expected_target_informative_groups_minimum": 5.5,
    "root_group_informative_probability_minimum": 0.65,
    "redundant_group_informative_probability_minimum": 0.65,
    "route_digit_joint_tv_maximum": 0.05,
    "route_digit_mutual_information_nats_maximum": 0.01,
}


def _signed(payload: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        **payload,
        "signed_payload_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def build_factorized_dataset() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build format/diversity supervision with exact route-digit independence."""
    root_examples: list[dict[str, Any]] = []
    for _repeat in range(ROOT_REPEATS_PER_ROUTE):
        for route in ROUTES:
            root_examples.append(
                {
                    "messages": [
                        {"role": "user", "content": stage_c_root_prompt()},
                        {
                            "role": "assistant",
                            "content": f"<route>{route}</route>",
                        },
                    ],
                    "example_kind": "root_format",
                    "route_label": route,
                    "digit_label": None,
                    "generator_index": len(root_examples),
                }
            )

    target_examples: list[dict[str, Any]] = []
    for route in ROUTES:
        for digit in DIGITS:
            target_examples.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": stage_c_branch_prompt(
                                f"<route>{route}</route>",
                                DIGITS,
                            ),
                        },
                        {"role": "assistant", "content": digit},
                    ],
                    "example_kind": "target_format",
                    "route_label": route,
                    "digit_label": digit,
                    "generator_index": len(target_examples),
                }
            )

    examples = [*root_examples, *target_examples]
    manifest = audit_factorized_dataset(examples)
    return examples, manifest


def audit_factorized_dataset(
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify the SFT corpus is marginal-only and reward-label free."""
    root = [row for row in examples if row.get("example_kind") == "root_format"]
    target = [row for row in examples if row.get("example_kind") == "target_format"]
    unknown = [
        row for row in examples if row.get("example_kind") not in {"root_format", "target_format"}
    ]
    root_counts = Counter(str(row.get("route_label")) for row in root)
    target_route_counts = Counter(str(row.get("route_label")) for row in target)
    target_digit_counts = Counter(str(row.get("digit_label")) for row in target)
    joint_counts = Counter(
        (str(row.get("route_label")), str(row.get("digit_label"))) for row in target
    )

    target_count = len(target)
    empirical_tv = 0.0
    empirical_mi = 0.0
    if target_count:
        for route in ROUTES:
            p_route = target_route_counts[route] / target_count
            for digit in DIGITS:
                p_digit = target_digit_counts[digit] / target_count
                p_joint = joint_counts[(route, digit)] / target_count
                empirical_tv += abs(p_joint - p_route * p_digit)
                if p_joint > 0 and p_route > 0 and p_digit > 0:
                    empirical_mi += p_joint * math.log(p_joint / (p_route * p_digit))
        empirical_tv *= 0.5

    forbidden_supervision_keys = {
        "reward",
        "success",
        "causal",
        "advantage",
        "return",
    }
    reward_supervision_fields = sum(
        any(str(key).lower() in forbidden_supervision_keys for key in row) for row in examples
    )
    complete_episode_examples = sum(
        row.get("digit_label") is not None and row.get("example_kind") == "root_format"
        for row in examples
    )
    checks = {
        "exact_example_count_40": len(examples) == 40,
        "exact_root_example_count_8": len(root) == 8,
        "exact_target_example_count_32": len(target) == 32,
        "no_unknown_example_kind": not unknown,
        "root_routes_exactly_balanced": all(
            root_counts[route] == ROOT_REPEATS_PER_ROUTE for route in ROUTES
        ),
        "target_routes_exactly_balanced": all(
            target_route_counts[route] == len(DIGITS) for route in ROUTES
        ),
        "target_digits_exactly_balanced": all(
            target_digit_counts[digit] == len(ROUTES) for digit in DIGITS
        ),
        "every_route_digit_pair_occurs_once": all(
            joint_counts[(route, digit)] == 1 for route in ROUTES for digit in DIGITS
        ),
        "empirical_joint_equals_product_exactly": (empirical_tv == 0.0 and empirical_mi == 0.0),
        "no_reward_or_causality_supervision_fields": (reward_supervision_fields == 0),
        "no_joint_complete_episode_supervision": complete_episode_examples == 0,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "stage-c4-factorized-format-warmstart",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "counts": {
            "examples": len(examples),
            "root_examples": len(root),
            "target_examples": len(target),
            "root_route_counts": dict(sorted(root_counts.items())),
            "target_route_counts": dict(sorted(target_route_counts.items())),
            "target_digit_counts": dict(sorted(target_digit_counts.items())),
            "target_joint_counts": {
                f"{route}:{digit}": joint_counts[(route, digit)]
                for route in ROUTES
                for digit in DIGITS
            },
        },
        "factorization": {
            "empirical_total_variation": empirical_tv,
            "empirical_mutual_information_nats": empirical_mi,
        },
        "supervision": {
            "purpose": "format and marginal diversity only",
            "root_and_target_labels_in_same_example": 0,
            "reward_or_causality_fields": reward_supervision_fields,
            "label_specific_reward_information": False,
            "task_interface_mentions_generic_reward": True,
            "note": (
                "The exact campaign prompt mentions reward generically, but "
                "the generator contains no route/digit reward mapping or "
                "outcome-derived label."
            ),
        },
    }
    return _signed(payload)


def _normalized_entropy(probabilities: dict[str, float]) -> float:
    return -math.fsum(value * math.log(value) for value in probabilities.values() if value > 0)


def _model_factorization(
    action_scores: dict[str, Any],
    root_scores: dict[str, Any],
) -> dict[str, Any]:
    models = action_scores.get("models", [])
    if len(models) != 1:
        raise ValueError("candidate scoring requires exactly one deployed model")
    rows = models[0]["temperatures"]["2.0"]
    conditional: dict[str, dict[str, float]] = {}
    for row in rows:
        route = str(row["context_route"])
        raw = {str(key): float(value) for key, value in row["action_probabilities"].items()}
        selected_mass = math.fsum(raw.values())
        conditional[route] = {digit: raw[digit] / selected_mass for digit in DIGITS}
    if set(conditional) != set(ROUTES):
        raise ValueError("candidate action scores do not cover every route")

    raw_routes = {
        str(key): float(value)
        for key, value in root_scores["temperature_2"]["route_sequence_probabilities"].items()
    }
    valid_route_mass = math.fsum(raw_routes.values())
    route_distribution = {route: raw_routes[route] / valid_route_mass for route in ROUTES}
    digit_marginal = {
        digit: math.fsum(route_distribution[route] * conditional[route][digit] for route in ROUTES)
        for digit in DIGITS
    }
    joint_tv = 0.5 * math.fsum(
        route_distribution[route] * abs(conditional[route][digit] - digit_marginal[digit])
        for route in ROUTES
        for digit in DIGITS
    )
    mutual_information = math.fsum(
        route_distribution[route]
        * conditional[route][digit]
        * math.log(conditional[route][digit] / digit_marginal[digit])
        for route in ROUTES
        for digit in DIGITS
        if conditional[route][digit] > 0
    )
    return {
        "normalized_valid_route_distribution": route_distribution,
        "normalized_root_entropy_nats": _normalized_entropy(route_distribution),
        "normalized_digit_distribution_by_route": conditional,
        "normalized_digit_marginal": digit_marginal,
        "route_digit_joint_total_variation": joint_tv,
        "route_digit_mutual_information_nats": mutual_information,
    }


def evaluate_candidate(
    *,
    step: int,
    action_scores: dict[str, Any],
    root_scores: dict[str, Any],
    dataset_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one deployed merged candidate against the frozen envelope."""
    if step < 1:
        raise ValueError("candidate step must be positive")
    campaign_power = analyze_power(action_scores, root_scores)
    factorization = _model_factorization(action_scores, root_scores)
    measurements = campaign_power["measurements"]
    raw_routes = measurements["route_sequence_probabilities_t2"]
    checks = {
        "factorized_dataset_audit_passed": (dataset_manifest.get("status") == "passed"),
        "all_unchanged_v3_power_checks_pass": (campaign_power["status"] == "passed"),
        "valid_route_mass_has_selection_buffer": (
            measurements["valid_route_sequence_mass_t2"]
            >= SELECTION_THRESHOLDS["valid_route_sequence_mass_minimum"]
        ),
        "every_route_mass_at_least_0_05": (
            min(raw_routes.values()) >= SELECTION_THRESHOLDS["minimum_route_mass"]
        ),
        "delta_route_mass_in_0_10_to_0_35": (
            SELECTION_THRESHOLDS["delta_route_mass_minimum"]
            <= raw_routes["delta"]
            <= SELECTION_THRESHOLDS["delta_route_mass_maximum"]
        ),
        "maximum_route_mass_at_most_0_55": (
            max(raw_routes.values()) <= SELECTION_THRESHOLDS["maximum_route_mass"]
        ),
        "normalized_root_entropy_at_least_1_10": (
            factorization["normalized_root_entropy_nats"]
            >= SELECTION_THRESHOLDS["normalized_root_entropy_minimum"]
        ),
        "mean_digit_5_mass_in_0_05_to_0_25": (
            SELECTION_THRESHOLDS["mean_digit_5_mass_minimum"]
            <= measurements["mean_digit_5_mass_t2"]
            <= SELECTION_THRESHOLDS["mean_digit_5_mass_maximum"]
        ),
        "mean_digit_entropy_at_least_1_40": (
            measurements["mean_normalized_digit_entropy_t2"]
            >= SELECTION_THRESHOLDS["mean_digit_entropy_minimum"]
        ),
        "mean_valid_digit_mass_at_least_0_92": (
            measurements["mean_valid_digit_mass_t2"]
            >= SELECTION_THRESHOLDS["mean_valid_digit_mass_minimum"]
        ),
        "expected_target_groups_at_least_5_5": (
            measurements["expected_target_informative_groups_per_sliced_step"]
            >= SELECTION_THRESHOLDS["expected_target_informative_groups_minimum"]
        ),
        "root_group_informativeness_at_least_0_65": (
            measurements["root_group_informative_probability"]
            >= SELECTION_THRESHOLDS["root_group_informative_probability_minimum"]
        ),
        "redundant_group_informativeness_at_least_0_65": (
            measurements["redundant_broadcast_group_informative_probability_lower"]
            >= SELECTION_THRESHOLDS["redundant_group_informative_probability_minimum"]
        ),
        "route_digit_joint_tv_at_most_0_05": (
            factorization["route_digit_joint_total_variation"]
            <= SELECTION_THRESHOLDS["route_digit_joint_tv_maximum"]
        ),
        "route_digit_mutual_information_at_most_0_01_nats": (
            factorization["route_digit_mutual_information_nats"]
            <= SELECTION_THRESHOLDS["route_digit_mutual_information_nats_maximum"]
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c4-factorized-warmstart-candidate",
        "step": step,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "selection_thresholds": SELECTION_THRESHOLDS,
        "campaign_power": campaign_power,
        "model_factorization": factorization,
        "dataset_manifest_sha256": dataset_manifest.get("signed_payload_sha256"),
        "sources": {
            "action_scores_sha256": action_scores.get("signed_payload_sha256"),
            "root_scores_sha256": root_scores.get("signed_payload_sha256"),
        },
    }
    return _signed(payload)


def select_earliest_candidate(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the earliest evaluated candidate satisfying every frozen check."""
    if not reports:
        raise ValueError("no candidate reports supplied")
    ordered = sorted(reports, key=lambda report: int(report["step"]))
    if len({int(report["step"]) for report in ordered}) != len(ordered):
        raise ValueError("candidate steps must be unique")
    passing = [report for report in ordered if report["status"] == "passed"]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c4-factorized-warmstart-selection",
        "status": "passed" if passing else "failed",
        "selection_rule": (
            "Earliest optimizer checkpoint satisfying every frozen deployed-"
            "model support, power, concentration, and factorization check."
        ),
        "selected_step": int(passing[0]["step"]) if passing else None,
        "evaluated_steps": [int(report["step"]) for report in ordered],
        "candidate_statuses": {str(report["step"]): report["status"] for report in ordered},
        "candidate_signed_payloads": {
            str(report["step"]): report["signed_payload_sha256"] for report in ordered
        },
        "selection_thresholds": SELECTION_THRESHOLDS,
    }
    return _signed(payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--output-jsonl", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--step", type=int, required=True)
    evaluate.add_argument("--action-scores", type=Path, required=True)
    evaluate.add_argument("--root-scores", type=Path, required=True)
    evaluate.add_argument("--dataset-manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    select = subparsers.add_parser("select")
    select.add_argument(
        "--candidate-report",
        action="append",
        type=Path,
        required=True,
    )
    select.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "build":
        examples, manifest = build_factorized_dataset()
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text(
            "".join(json.dumps(example, sort_keys=True) + "\n" for example in examples),
            encoding="utf-8",
        )
        _write_json(args.manifest, manifest)
    elif args.command == "audit":
        examples = [
            json.loads(line)
            for line in args.dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _write_json(args.output, audit_factorized_dataset(examples))
    elif args.command == "evaluate":
        _write_json(
            args.output,
            evaluate_candidate(
                step=args.step,
                action_scores=json.loads(args.action_scores.read_text(encoding="utf-8")),
                root_scores=json.loads(args.root_scores.read_text(encoding="utf-8")),
                dataset_manifest=json.loads(args.dataset_manifest.read_text(encoding="utf-8")),
            ),
        )
    else:
        _write_json(
            args.output,
            select_earliest_candidate(
                [json.loads(path.read_text(encoding="utf-8")) for path in args.candidate_report]
            ),
        )


if __name__ == "__main__":
    main()
