"""Post-hoc Pareto analysis of the retained Stage C9 endpoint scores."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload

BRANCH_ARMS = ("local-e1", "local-e2", "branch-global-e2")
E2_ARMS = ("local-e2", "branch-global-e2")
ALL_ARMS = (*BRANCH_ARMS, "stock")
ARM_COLORS = {
    "local-e1": "#2563eb",
    "local-e2": "#7c3aed",
    "branch-global-e2": "#059669",
    "stock": "#dc2626",
}


def _arm(run_name: str) -> str:
    return run_name.split("--s", maxsplit=1)[0]


def _seed(run_name: str) -> int:
    return int(run_name.rsplit("--s", maxsplit=1)[1])


def _linear_fit(points: list[dict[str, Any]]) -> dict[str, float]:
    xs = [float(point["causal_gain"]) for point in points]
    ys = [float(point["nuisance_drift"]) for point in points]
    mean_x = fmean(xs)
    mean_y = fmean(ys)
    covariance = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)
    )
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    slope = covariance / variance_x
    intercept = mean_y - slope * mean_x
    correlation = covariance / math.sqrt(variance_x * variance_y)
    return {
        "slope": slope,
        "intercept": intercept,
        "pearson_r": correlation,
        "r_squared": correlation**2,
    }


def evaluate(partial_recovery_path: Path, trigger_path: Path) -> dict[str, Any]:
    partial = json.loads(partial_recovery_path.read_text(encoding="utf-8"))
    trigger = json.loads(trigger_path.read_text(encoding="utf-8"))
    points: list[dict[str, Any]] = []
    for run_name, run in partial["runs"].items():
        arm = _arm(run_name)
        gain = float(run["final_causal_mass_gain"])
        drift = float(run["exact_final"]["delta_nuisance_js_from_initial"])
        points.append(
            {
                "run": run_name,
                "arm": arm,
                "seed": _seed(run_name),
                "causal_gain": gain,
                "nuisance_drift": drift,
                "drift_per_causal_gain": drift / gain,
            }
        )
    points.sort(key=lambda point: (str(point["arm"]), int(point["seed"])))

    arm_summaries: dict[str, Any] = {}
    for arm in ALL_ARMS:
        arm_points = [point for point in points if point["arm"] == arm]
        mean_gain = fmean(float(point["causal_gain"]) for point in arm_points)
        mean_drift = fmean(
            float(point["nuisance_drift"]) for point in arm_points
        )
        arm_summaries[arm] = {
            "seeds": [int(point["seed"]) for point in arm_points],
            "mean_causal_gain": mean_gain,
            "mean_nuisance_drift": mean_drift,
            "drift_per_unit_mean_causal_gain": mean_drift / mean_gain,
        }

    branch_ratios = [
        float(
            arm_summaries[arm]["drift_per_unit_mean_causal_gain"]
        )
        for arm in BRANCH_ARMS
    ]
    stock_ratio = float(
        arm_summaries["stock"]["drift_per_unit_mean_causal_gain"]
    )
    e2_points = [point for point in points if point["arm"] in E2_ARMS]
    e2_fit = _linear_fit(e2_points)
    local_e2_gain = float(
        arm_summaries["local-e2"]["mean_causal_gain"]
    )
    global_e2_gain = float(
        arm_summaries["branch-global-e2"]["mean_causal_gain"]
    )

    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-c9-final-endpoint-pareto",
            "source_partial_recovery_sha256": partial[
                "signed_payload_sha256"
            ],
            "points": points,
            "arm_summaries": arm_summaries,
            "descriptive_findings": {
                "branch_arm_drift_per_gain_range": [
                    min(branch_ratios),
                    max(branch_ratios),
                ],
                "mean_branch_arm_drift_per_gain": fmean(branch_ratios),
                "stock_drift_per_gain": stock_ratio,
                "stock_to_mean_branch_drift_per_gain_ratio": (
                    stock_ratio / fmean(branch_ratios)
                ),
                "branch_global_e2_mean_gain_fraction_below_local_e2": (
                    1.0 - global_e2_gain / local_e2_gain
                ),
                "pooled_e2_gain_drift_linear_fit": e2_fit,
            },
            "claim_status": {
                "frozen_endpoint_credit_gate": "failed_as_preregistered",
                "practical_local_vs_branch_global_increment": "indeterminate",
                "reason": (
                    "Local E2 learned more and drifted more. Across the six E2 "
                    "arm-seed endpoints, nuisance drift is tightly coupled to "
                    "causal gain, so the endpoint comparison does not isolate "
                    "advantage-label quality under shared-parameter reuse."
                ),
                "branching_vs_stock": "descriptively_supported",
                "branching_reason": (
                    "All three branch-based arms cluster at substantially lower "
                    "nuisance drift per unit causal gain than stock broadcast. "
                    "This is a descriptive endpoint result, not a randomized "
                    "test of a composite ratio."
                ),
                "mechanistic_local_increment": "supported_by_stage_c7_only",
            },
            "oracle_floor_trigger": trigger,
            "limits": [
                "The drift-per-gain ratio was not a frozen Stage C9 decision rule.",
                "Only three seeds per arm and final endpoints are available.",
                "The pooled regression is descriptive and does not remove "
                "shared-parameter confounding.",
                "No deleted checkpoint trajectory or AUC is reconstructed.",
                "A faithful oracle parameter-sharing floor requires actual 4B LoRA updates.",
            ],
        }
    )


def render_svg(report: dict[str, Any]) -> str:
    width, height = 760, 520
    left, right, top, bottom = 80, 30, 45, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    points = report["points"]
    max_x = max(float(point["causal_gain"]) for point in points) * 1.08
    max_y = max(float(point["nuisance_drift"]) for point in points) * 1.08

    def x_pos(value: float) -> float:
        return left + value / max_x * plot_width

    def y_pos(value: float) -> float:
        return top + plot_height - value / max_y * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="25" text-anchor="middle" '
        'font-family="sans-serif" font-size="18">Stage C9 final endpoint Pareto view</text>',
    ]
    for index in range(6):
        x_value = max_x * index / 5
        y_value = max_y * index / 5
        x = x_pos(x_value)
        y = y_pos(y_value)
        elements.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
                f'y2="{top + plot_height}" stroke="#e5e7eb"/>',
                f'<text x="{x:.2f}" y="{top + plot_height + 22}" '
                f'text-anchor="middle" font-family="sans-serif" '
                f'font-size="11">{x_value:.2f}</text>',
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" '
                f'y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="11">{y_value:.2f}</text>',
            ]
        )
    elements.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top + plot_height}" stroke="#111827"/>',
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 22}" '
            'text-anchor="middle" font-family="sans-serif" font-size="13">'
            "Causal mass gain</text>",
            f'<text x="20" y="{top + plot_height / 2:.2f}" '
            'text-anchor="middle" font-family="sans-serif" font-size="13" '
            f'transform="rotate(-90 20 {top + plot_height / 2:.2f})">'
            "Nuisance JS drift</text>",
        ]
    )
    for point in points:
        arm = str(point["arm"])
        x = x_pos(float(point["causal_gain"]))
        y = y_pos(float(point["nuisance_drift"]))
        color = ARM_COLORS[arm]
        label = html.escape(f"{arm} s{point['seed']}")
        elements.extend(
            [
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" '
                f'fill="{color}"><title>{label}</title></circle>',
                f'<text x="{x + 7:.2f}" y="{y - 7:.2f}" '
                f'font-family="sans-serif" font-size="9" fill="{color}">'
                f"{point['seed']}</text>",
            ]
        )
    legend_y = top + 10
    for index, arm in enumerate(ALL_ARMS):
        x = left + index * 155
        color = ARM_COLORS[arm]
        elements.extend(
            [
                f'<circle cx="{x}" cy="{legend_y}" r="5" fill="{color}"/>',
                f'<text x="{x + 10}" y="{legend_y + 4}" '
                f'font-family="sans-serif" font-size="11">{html.escape(arm)}</text>',
            ]
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial-recovery", type=Path, required=True)
    parser.add_argument("--oracle-trigger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    report = evaluate(args.partial_recovery, args.oracle_trigger)
    atomic_write_json(args.output, report)
    if args.svg is not None:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        args.svg.write_text(render_svg(report), encoding="utf-8")


if __name__ == "__main__":
    main()
