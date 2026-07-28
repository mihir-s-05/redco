"""Build and select the shared Stage-C exploration warm start."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from redco.analysis.stage_c_postmortem import (
    binary_informative_group_probability,
)
from redco.env.tasks.credit_probes import credit_probe_by_name


def build_warmstart_dataset(
    policy_cases: dict[str, Any],
    *,
    exogenous_seeds: range,
    probe_names: tuple[str, ...] = ("planted_needle",),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create exact-action demonstrations without using held-out outcomes."""
    cases = policy_cases.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("policy cases are missing")
    examples: list[dict[str, Any]] = []
    skipped_action_independent = 0
    for case in sorted(cases, key=lambda item: item["case_id"]):
        if case["probe_name"] not in probe_names:
            continue
        probe = credit_probe_by_name(case["probe_name"])
        aliases = list(case["actions"])
        if len(aliases) != len(probe.actions):
            raise ValueError(f"action count mismatch for {probe.name}")
        prompt = case.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"case {case['case_id']} is missing its exact prompt")
        for seed in exogenous_seeds:
            rewards = [
                float(probe.reward_function(action, seed))
                for action in probe.actions
            ]
            if len(set(rewards)) == 1:
                skipped_action_independent += 1
                continue
            best_reward = max(rewards)
            best_indices = [
                index for index, reward in enumerate(rewards) if reward == best_reward
            ]
            choice = best_indices[(seed + len(examples)) % len(best_indices)]
            examples.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": aliases[choice]},
                    ],
                    "probe_name": probe.name,
                    "context_route": case["context_route"],
                    "calibration_seed": seed,
                    "teacher_action": aliases[choice],
                }
            )
    if not examples:
        raise ValueError("warm-start construction produced no examples")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "stage-c-shared-warmstart",
        "source_policy_cases_sha256": policy_cases.get("signed_payload_sha256"),
        "included_probes": list(probe_names),
        "seed_start": exogenous_seeds.start,
        "seed_stop_exclusive": exogenous_seeds.stop,
        "examples": len(examples),
        "skipped_action_independent_states": skipped_action_independent,
        "heldout_seed_overlap": bool(
            set(exogenous_seeds).intersection(range(9000, 9032))
        ),
        "examples_by_probe": {
            probe: sum(example["probe_name"] == probe for example in examples)
            for probe in sorted({example["probe_name"] for example in examples})
        },
        "teacher_actions": {
            action: sum(example["teacher_action"] == action for example in examples)
            for action in sorted({example["teacher_action"] for example in examples})
        },
    }
    signed = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return examples, manifest


def select_warmstart_checkpoint(
    raw_policy_scores: dict[str, Any],
    *,
    minimum_needle_mass_t2: float,
    maximum_needle_mass_t2: float,
    maximum_needle_greedy_rate: float,
    branch_count: int,
    groups_per_step: int,
    minimum_expected_informative_groups: float,
) -> dict[str, Any]:
    """Select the earliest SFT checkpoint satisfying frozen support bounds."""
    if not (
        0
        < minimum_needle_mass_t2
        <= maximum_needle_mass_t2
        < 1
    ):
        raise ValueError("needle mass bounds must be ordered inside (0, 1)")
    if not 0 <= maximum_needle_greedy_rate <= 1:
        raise ValueError("maximum greedy rate must lie in [0, 1]")
    models = raw_policy_scores.get("models")
    if not isinstance(models, list):
        raise ValueError("raw policy scores are missing models")
    candidates: list[dict[str, Any]] = []
    for model in models:
        name = model.get("name") if isinstance(model, dict) else None
        if not isinstance(name, str) or not name.startswith("sft_step_"):
            continue
        try:
            step = int(name.removeprefix("sft_step_"))
        except ValueError as error:
            raise ValueError(f"invalid SFT checkpoint name: {name}") from error
        if isinstance(model.get("temperatures"), dict):
            scored_rows = model["temperatures"].get("2.0", [])
            mass_field = "action_probabilities"
        else:
            scored_rows = model.get("cases", [])
            mass_field = "full_vocab_action_probabilities_t2"
        rows = [
            row
            for row in scored_rows
            if row.get("probe_name") == "planted_needle"
        ]
        if not rows:
            raise ValueError(f"{name} has no planted-needle scoring cases")
        masses = [float(row[mass_field]["5"]) for row in rows]
        greedy_rate = sum(
            row.get("greedy_allowed_action") == "5" for row in rows
        ) / len(rows)
        minimum_mass = min(masses)
        expected_groups = groups_per_step * binary_informative_group_probability(
            minimum_mass,
            branch_count=branch_count,
        )
        candidates.append(
            {
                "name": name,
                "step": step,
                "minimum_needle_mass_t2": minimum_mass,
                "mean_needle_mass_t2": math.fsum(masses) / len(masses),
                "maximum_needle_mass_t2": max(masses),
                "needle_greedy_rate": greedy_rate,
                "expected_informative_groups_per_step_at_minimum_mass": (
                    expected_groups
                ),
                "passes": (
                    minimum_mass >= minimum_needle_mass_t2
                    and max(masses) <= maximum_needle_mass_t2
                    and greedy_rate <= maximum_needle_greedy_rate
                    and expected_groups >= minimum_expected_informative_groups
                ),
            }
        )
    if not candidates:
        raise ValueError("no SFT checkpoints were scored")
    candidates.sort(key=lambda candidate: candidate["step"])
    passing = [candidate for candidate in candidates if candidate["passes"]]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c-warmstart-selection",
        "status": "pass" if passing else "fail",
        "selected": passing[0] if passing else None,
        "candidates": candidates,
        "thresholds": {
            "minimum_needle_mass_t2": minimum_needle_mass_t2,
            "maximum_needle_mass_t2": maximum_needle_mass_t2,
            "maximum_needle_greedy_rate": maximum_needle_greedy_rate,
            "branch_count": branch_count,
            "groups_per_step": groups_per_step,
            "minimum_expected_informative_groups": (
                minimum_expected_informative_groups
            ),
        },
        "selection_rule": (
            "Earliest checkpoint meeting every frozen bound; no held-out reward "
            "is consulted."
        ),
    }
    signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--cases", type=Path, required=True)
    build.add_argument("--output-jsonl", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--seed-start", type=int, default=20_000)
    build.add_argument("--seed-count", type=int, default=32)
    build.add_argument(
        "--probe",
        action="append",
        default=[],
        help="Probe to demonstrate (default: planted_needle)",
    )
    select = subparsers.add_parser("select")
    select.add_argument("--raw-scores", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--minimum-needle-mass-t2", type=float, default=0.15)
    select.add_argument("--maximum-needle-mass-t2", type=float, default=0.25)
    select.add_argument("--maximum-needle-greedy-rate", type=float, default=0.5)
    select.add_argument("--branch-count", type=int, default=6)
    select.add_argument("--groups-per-step", type=int, default=8)
    select.add_argument(
        "--minimum-expected-informative-groups",
        type=float,
        default=4.75,
    )
    args = parser.parse_args()
    if args.command == "build":
        cases = json.loads(args.cases.read_text())
        examples, manifest = build_warmstart_dataset(
            cases,
            exogenous_seeds=range(
                args.seed_start,
                args.seed_start + args.seed_count,
            ),
            probe_names=tuple(args.probe or ["planted_needle"]),
        )
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text(
            "".join(json.dumps(example, sort_keys=True) + "\n" for example in examples)
        )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    else:
        report = select_warmstart_checkpoint(
            json.loads(args.raw_scores.read_text()),
            minimum_needle_mass_t2=args.minimum_needle_mass_t2,
            maximum_needle_mass_t2=args.maximum_needle_mass_t2,
            maximum_needle_greedy_rate=args.maximum_needle_greedy_rate,
            branch_count=args.branch_count,
            groups_per_step=args.groups_per_step,
            minimum_expected_informative_groups=(
                args.minimum_expected_informative_groups
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
