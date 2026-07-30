"""Verify the frozen-batch practical-loss reuse microcampaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload


def _metrics(path: Path) -> dict[int, dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        step = int(row["step"])
        by_step.setdefault(step, {}).update(
            {
                str(key): float(value)
                for key, value in row.items()
                if key not in {"step", "time"} and isinstance(value, int | float)
            }
        )
    return by_step


def _mean_action_5(model: dict[str, Any]) -> float:
    rows = model["temperatures"]["2.0"]
    return math.fsum(
        float(row["action_probabilities"]["5"]) for row in rows
    ) / len(rows)


def evaluate(run_root: Path, scores_path: Path) -> dict[str, Any]:
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    masses = {
        str(model["name"]): _mean_action_5(model)
        for model in scores["models"]
    }
    arms: dict[str, Any] = {}
    for updates in (1, 2, 3):
        name = f"reuse-{updates}"
        arm_root = run_root / name
        metrics = _metrics(arm_root / "metrics.jsonl")
        expected_steps = set(range(1, updates + 1))
        if set(metrics) != expected_steps:
            raise ValueError(f"{name} has unexpected optimizer steps")
        for step, row in metrics.items():
            required = (
                "optim/grad_norm",
                "redco_node_ratio/mean",
                "redco_node_clipped/mean",
                "redco_node_kl/mean",
            )
            if not all(key in row and math.isfinite(row[key]) for key in required):
                raise ValueError(f"{name} step {step} is missing finite node metrics")
        adapter = (
            arm_root
            / "run_default"
            / "broadcasts"
            / f"step_{updates}"
            / "adapter_model.safetensors"
        )
        if not adapter.is_file():
            raise ValueError(f"{name} final adapter is missing")
        arms[name] = {
            "optimizer_updates": updates,
            "mean_action_5_mass_t2": masses[name],
            "steps": metrics,
        }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-c7-frozen-batch-practical-loss-reuse",
            "status": "passed",
            "warmstart_mean_action_5_mass_t2": masses["warmstart"],
            "arms": arms,
            "interpretation": (
                "One, two, and three optimizer updates consumed byte-identical copies "
                "of one previously observed frozen batch. This tests practical loss "
                "and reuse mechanics, not fresh-rollout generalization."
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(args.output, evaluate(args.run_root, args.scores))


if __name__ == "__main__":
    main()
