"""Structural early-abort invariants for Stage-C3 live runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

Mode = Literal["smoke", "arm", "constraint"]


def first_training_row(metrics_path: Path) -> dict[str, Any] | None:
    if not metrics_path.is_file():
        return None
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "train/agg/all/reward/mean" in row:
            return cast(dict[str, Any], row)
    return None


def check_first_training_row(
    row: dict[str, Any],
    *,
    mode: Mode,
) -> dict[str, Any]:
    checks = {
        "no_rollout_errors": row.get("train/agg/all/has_error/mean") == 0.0,
        "root_completion_budget_contract": (
            row.get(
                "train/agg/all/metrics/redco_context_token_budget_ok/mean"
            )
            == 1.0
        ),
    }
    if mode in {"smoke", "constraint"}:
        checks.update(
            {
                "every_root_route_parseable": (
                    row.get(
                        "train/agg/all/metrics/redco_valid_route/mean"
                    )
                    == 1.0
                )
            }
        )
    if mode == "smoke":
        checks.update(
            {
                "nonzero_trainable_fraction": (
                    float(
                        row.get(
                            "train/agg/all/is_trainable/mean",
                            0.0,
                        )
                    )
                    > 0.0
                ),
                "nonconstant_reward_exposure": (
                    float(row.get("train/agg/all/reward/max", 0.0))
                    > float(row.get("train/agg/all/reward/min", 0.0))
                ),
            }
        )
    return {
        "schema_version": 1,
        "mode": mode,
        "step": int(row["step"]),
        "checks": checks,
        "passed": all(checks.values()),
        "observed": {
            "reward_mean": row.get("train/agg/all/reward/mean"),
            "reward_min": row.get("train/agg/all/reward/min"),
            "reward_max": row.get("train/agg/all/reward/max"),
            "trainable_fraction": row.get("train/agg/all/is_trainable/mean"),
            "truncation_fraction": row.get("train/agg/all/is_truncated/mean"),
            "error_fraction": row.get("train/agg/all/has_error/mean"),
            "valid_route_fraction": row.get(
                "train/agg/all/metrics/redco_valid_route/mean"
            ),
            "context_token_budget_contract_fraction": row.get(
                "train/agg/all/metrics/redco_context_token_budget_ok/mean"
            ),
        },
    }
