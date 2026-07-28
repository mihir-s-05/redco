"""Verify the frozen Stage-C3 multi-decision live campaign."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from statistics import fmean
from typing import Any

PROBES = (
    "confusion_irrelevant",
    "confusion_redundant",
    "confusion_lucky",
)
RUNS = {
    "confusion_irrelevant": (9401, 9402),
    "confusion_redundant": (9403,),
    "confusion_lucky": (9404,),
}
ARMS = ("broadcast", "sliced")
TARGET_THRESHOLD = 0.5


def _normalize(values: dict[str, float]) -> dict[str, float]:
    total = math.fsum(values.values())
    if total <= 0:
        raise ValueError("selected action probabilities have zero mass")
    return {key: value / total for key, value in values.items()}


def _js_divergence(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    if set(left) != set(right):
        raise ValueError("action distributions have different support")
    left = _normalize(left)
    right = _normalize(right)
    midpoint = {key: (left[key] + right[key]) / 2.0 for key in left}

    def kl(first: dict[str, float], second: dict[str, float]) -> float:
        return math.fsum(
            probability * math.log(probability / second[key])
            for key, probability in first.items()
            if probability > 0
        )

    return 0.5 * (kl(left, midpoint) + kl(right, midpoint))


def _models(score_path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, Any]]] = {}
    for model in payload["models"]:
        name = str(model["name"])
        if name in result:
            raise ValueError(f"duplicate scored model: {name}")
        result[name] = list(model["temperatures"]["2.0"])
    return result


def _row_map(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["case_id"]): row for row in rows}
    if not result:
        raise ValueError("policy score contains no cases")
    return result


def _policy_metrics(
    initial: list[dict[str, Any]],
    final: list[dict[str, Any]],
) -> dict[str, float]:
    initial_by_case = _row_map(initial)
    final_by_case = _row_map(final)
    if set(initial_by_case) != set(final_by_case):
        raise ValueError("initial and final policy cases differ")
    divergences: list[float] = []
    target_mass: list[float] = []
    initial_target_mass: list[float] = []
    for case_id in sorted(initial_by_case):
        before = initial_by_case[case_id]["action_probabilities"]
        after = final_by_case[case_id]["action_probabilities"]
        divergences.append(_js_divergence(before, after))
        target_mass.append(float(after["5"]))
        initial_target_mass.append(float(before["5"]))
    return {
        "mean_selected_action_js_from_initial": fmean(divergences),
        "mean_target_action_mass": fmean(target_mass),
        "mean_initial_target_action_mass": fmean(initial_target_mass),
    }


def _find_eval_rows(run_dir: Path) -> list[dict[str, Any]]:
    key = "eval/redco-credit-eval/all/metrics/target_success/mean"
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if key in row:
                rows.append(row)
    if not rows:
        raise ValueError(f"no target-success evaluations under {run_dir}")
    unique = {int(row["step"]): row for row in rows}
    return [unique[step] for step in sorted(unique)]


def _learning_curve(run_dir: Path, arm: str) -> dict[str, Any]:
    calls_per_step = 16 if arm == "broadcast" else 96
    key = "eval/redco-credit-eval/all/metrics/target_success/mean"
    points = [
        {
            "step": int(row["step"]),
            "policy_calls": int(row["step"]) * calls_per_step,
            "target_success_rate": float(row[key]),
        }
        for row in _find_eval_rows(run_dir)
    ]
    aligned = [
        point
        for point in points
        if point["policy_calls"] % 96 == 0
        and point["policy_calls"] <= 576
    ]
    threshold = next(
        (
            int(point["policy_calls"])
            for point in aligned
            if point["target_success_rate"] >= TARGET_THRESHOLD
        ),
        None,
    )
    return {
        "points": points,
        "aligned_points": aligned,
        "calls_to_target_success_at_least_0_5": threshold,
    }


def verify_campaign(run_root: Path, score_path: Path) -> dict[str, Any]:
    models = _models(score_path)
    if "warmstart" not in models:
        raise ValueError("scores must include a warmstart model")
    initial = models["warmstart"]
    runs: dict[str, Any] = {}
    for probe, seeds in RUNS.items():
        for seed in seeds:
            for arm in ARMS:
                name = f"{probe}--{arm}--s{seed}"
                if name not in models:
                    raise ValueError(f"missing final policy score: {name}")
                run_dir = run_root / probe / f"{arm}-s{seed}"
                if not run_dir.is_dir():
                    raise ValueError(f"missing run directory: {run_dir}")
                runs[name] = {
                    **_policy_metrics(initial, models[name]),
                    "learning_curve": _learning_curve(run_dir, arm),
                }

    initial_target_mass = fmean(
        float(row["action_probabilities"]["5"]) for row in initial
    )
    initial_entropies = []
    for row in initial:
        normalized = _normalize(row["action_probabilities"])
        initial_entropies.append(
            -math.fsum(
                probability * math.log(probability)
                for probability in normalized.values()
                if probability > 0
            )
        )
    support_guard = {
        "target_mass_between_0_03_and_0_35": 0.03 <= initial_target_mass <= 0.35,
        "mean_selected_action_entropy_at_least_1_25": (
            fmean(initial_entropies) >= 1.25
        ),
    }
    irrelevant = {
        arm: [
            float(
                runs[f"confusion_irrelevant--{arm}--s{seed}"][
                    "mean_selected_action_js_from_initial"
                ]
            )
            for seed in RUNS["confusion_irrelevant"]
        ]
        for arm in ARMS
    }
    means = {arm: fmean(values) for arm, values in irrelevant.items()}
    exposure = means["broadcast"] >= 0.002
    primary = {
        "broadcast_exposure_floor_met": exposure,
        "mean_sliced_js_at_most_75pct_broadcast": (
            means["sliced"] <= 0.75 * means["broadcast"]
        ),
        "each_sliced_pair_no_more_than_broadcast_plus_0_002": all(
            sliced <= broadcast + 0.002
            for broadcast, sliced in zip(
                irrelevant["broadcast"],
                irrelevant["sliced"],
                strict=True,
            )
        ),
    }
    causal_sanity: dict[str, bool] = {}
    for probe, seed in (
        ("confusion_redundant", 9403),
        ("confusion_lucky", 9404),
    ):
        improvements = [
            float(runs[f"{probe}--{arm}--s{seed}"]["mean_target_action_mass"])
            - initial_target_mass
            for arm in ARMS
        ]
        causal_sanity[f"{probe}_at_least_one_arm_improves_target_mass_by_0_05"] = (
            max(improvements) >= 0.05
        )
    integration = {
        "all_expected_runs_scored": len(runs) == 8,
        "all_runs_have_aligned_evaluations": all(
            bool(run["learning_curve"]["aligned_points"])
            for run in runs.values()
        ),
    }
    mandatory = {
        **support_guard,
        **primary,
        **causal_sanity,
        **integration,
    }
    if not all(support_guard.values()) or not exposure:
        status = "underpowered"
    elif all(mandatory.values()):
        status = "passed"
    else:
        status = "failed"
    saturated = any(
        float(run["mean_target_action_mass"]) >= 0.98 for run in runs.values()
    )
    return {
        "schema_version": 1,
        "analysis": "stage-c3-credit-confusion-live",
        "status": status,
        "initial": {
            "mean_target_action_mass": initial_target_mass,
            "mean_selected_action_entropy": fmean(initial_entropies),
        },
        "irrelevant_target_js_by_arm": irrelevant,
        "irrelevant_target_mean_js_by_arm": means,
        "runs": runs,
        "mandatory_checks": mandatory,
        "saturation_guard_triggered": saturated,
        "saturation_interpretation": (
            "Use frozen calls-to-threshold curves; do not apply an endpoint "
            "margin when any final target mass is at least 0.98."
            if saturated
            else "No final policy reached the 0.98 saturation guard."
        ),
        "scope": (
            "Small mechanistic GPU battery. Passing supports reduced nuisance "
            "drift in these tasks; it is not a broad population claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_campaign(args.run_root, args.scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
