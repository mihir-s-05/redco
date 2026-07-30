"""Verify the bounded Stage-C9 practical-efficiency bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from statistics import fmean
from typing import Any

import msgspec
from prime_rl.transport import TrainingBatch

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload

ARMS = ("local-e1", "local-e2", "branch-global-e2", "stock")
SEEDS = (10031, 10032, 10033)
COLLECTIONS = 6
CALLS_PER_COLLECTION = 96


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    total = math.fsum(values.values())
    if total <= 0.0:
        raise ValueError("categorical probabilities have zero mass")
    return {key: float(value) / total for key, value in values.items()}


def _kl(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    p = _normalize(left)
    q = _normalize(right)
    return math.fsum(
        value * math.log(value / q[key])
        for key, value in p.items()
        if value > 0.0
    )


def _js(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    p = _normalize(left)
    q = _normalize(right)
    midpoint = {key: (p[key] + q[key]) / 2.0 for key in p}
    return 0.5 * (_kl(p, midpoint) + _kl(q, midpoint))


def _models(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(model["name"]): list(model["temperatures"]["2.0"])
        for model in payload["models"]
    }


def _redundant_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        if row["probe_name"] != "confusion_redundant":
            continue
        result[str(row["case_id"])] = {
            "route": str(row["context_route"]),
            "probabilities": {
                str(key): float(value)
                for key, value in row["action_probabilities"].items()
            },
        }
    if not result:
        raise ValueError("no confusion_redundant exact-score rows")
    return result


def _policy_point(
    initial: Mapping[str, dict[str, Any]],
    current: Mapping[str, dict[str, Any]],
) -> dict[str, float]:
    if set(initial) != set(current):
        raise ValueError("exact-score cases differ")
    causal_mass = []
    delta_js = []
    for case_id in sorted(initial):
        before = initial[case_id]
        after = current[case_id]
        if before["route"] != after["route"]:
            raise ValueError("route label changed across exact scores")
        if before["route"] == "delta":
            delta_js.append(
                _js(before["probabilities"], after["probabilities"])
            )
        else:
            causal_mass.append(float(after["probabilities"]["5"]))
    return {
        "causal_non_delta_target_mass": fmean(causal_mass),
        "delta_nuisance_js_from_initial": fmean(delta_js),
    }


def _auc(points: Sequence[Mapping[str, float]]) -> float:
    area = 0.0
    for left, right in pairwise(points):
        width = float(right["policy_calls"]) - float(left["policy_calls"])
        area += width * (
            float(left["causal_non_delta_target_mass"])
            + float(right["causal_non_delta_target_mass"])
        ) / 2.0
    return area / float(points[-1]["policy_calls"])


def _examples_hash(path: Path) -> tuple[int, str]:
    batch = msgspec.msgpack.decode(path.read_bytes(), type=TrainingBatch)
    return batch.step, hashlib.sha256(
        msgspec.msgpack.encode(batch.examples)
    ).hexdigest()


def _trace_policy_versions(path: Path) -> set[int]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return {
        int(
            row["policy_version"]
            if "policy_version" in row
            else row["info"]["policy_version"]
        )
        for row in rows
    }


def _reuse_contract(run_dir: Path) -> dict[str, Any]:
    pairs = []
    prior_hash: str | None = None
    fresh_collections = True
    for collection in range(1, COLLECTIONS + 1):
        first_step = collection * 2 - 1
        second_step = collection * 2
        first_path = (
            run_dir
            / "run_default"
            / "rollouts"
            / f"step_{first_step}"
            / "train_rollouts.bin"
        )
        second_path = (
            run_dir
            / "run_default"
            / "rollouts"
            / f"step_{second_step}"
            / "train_rollouts.bin"
        )
        first_transport, first_hash = _examples_hash(first_path)
        second_transport, second_hash = _examples_hash(second_path)
        trace = (
            first_path.parent / "train" / "all" / "traces.jsonl"
        )
        even_trace = (
            second_path.parent / "train" / "all" / "traces.jsonl"
        )
        versions = _trace_policy_versions(trace)
        expected_version = first_step - 1
        pair_pass = (
            first_transport == first_step
            and second_transport == second_step
            and first_hash == second_hash
            and versions == {expected_version}
            and not even_trace.exists()
        )
        if prior_hash is not None and first_hash == prior_hash:
            fresh_collections = False
        prior_hash = first_hash
        pairs.append(
            {
                "collection": collection,
                "trainer_steps": [first_step, second_step],
                "examples_sha256": first_hash,
                "behavior_policy_versions": sorted(versions),
                "no_even_step_rollout_trace": not even_trace.exists(),
                "passed": pair_pass,
            }
        )
    return {
        "pairs": pairs,
        "all_pairs_passed": all(pair["passed"] for pair in pairs),
        "fresh_example_stream_between_collections": fresh_collections,
    }


def _numeric_metrics(path: Path) -> dict[int, dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "step" not in row:
            continue
        step = int(row["step"])
        by_step.setdefault(step, {}).update(
            {
                str(key): float(value)
                for key, value in row.items()
                if isinstance(value, int | float) and key not in {"step", "time"}
            }
        )
    return by_step


def _practical_diagnostics(run_dir: Path, updates: int) -> dict[str, Any]:
    metrics = _numeric_metrics(run_dir / "metrics.jsonl")
    rows = []
    required = (
        "optim/grad_norm",
        "redco_node_ratio/mean",
        "redco_node_ratio/min",
        "redco_node_ratio/max",
        "redco_node_clipped/mean",
        "redco_node_sampled_kl/mean",
        "redco_node_squared_log_ratio/mean",
    )
    for step in range(1, updates + 1):
        row = metrics.get(step, {})
        if not all(key in row and math.isfinite(row[key]) for key in required):
            raise ValueError(f"step {step} lacks finite practical metrics")
        rows.append(
            {
                "step": step,
                "epoch_within_collection": 1 if updates == 6 else (1 + (step + 1) % 2),
                **{key: row[key] for key in required},
            }
        )
    second = [
        row for row in rows if row["epoch_within_collection"] == 2
    ]
    return {
        "steps": rows,
        "second_epoch_mean_clipped_fraction": (
            None
            if not second
            else fmean(row["redco_node_clipped/mean"] for row in second)
        ),
        "second_epoch_max_clipped_fraction": (
            None
            if not second
            else max(row["redco_node_clipped/mean"] for row in second)
        ),
    }


def _usage(run_dir: Path) -> dict[str, float | int]:
    calls = []
    # Prime-RL writes the same accepted rollout calls under both ``all`` and
    # ``effective``.  The latter is a filtered reporting view, not a second
    # set of policy calls, so the canonical ledger must read exactly one view.
    for path in sorted(
        (run_dir / "run_default" / "rollouts").glob(
            "step_*/train/all/traces.jsonl"
        )
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            calls.extend(row.get("calls", ()))
    prompt = sum(int(call["usage"]["prompt_tokens"]) for call in calls)
    completion = sum(
        int(call["usage"]["completion_tokens"]) for call in calls
    )
    return {
        "policy_calls": len(calls),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "service_seconds": math.fsum(
            float(call["time"]["end"]) - float(call["time"]["start"])
            for call in calls
        ),
    }


def evaluate(run_root: Path, scores_path: Path) -> dict[str, Any]:
    models = _models(scores_path)
    initial = _redundant_rows(models["warmstart"])
    initial_point = {
        "collection": 0,
        "policy_calls": 0,
        **_policy_point(initial, initial),
    }
    runs: dict[str, Any] = {}
    for seed in SEEDS:
        for arm in ARMS:
            run_name = f"{arm}-s{seed}"
            run_dir = run_root / "confusion_redundant" / run_name
            points = [initial_point]
            prior_rows = initial
            transition_divergence = []
            for collection in range(1, COLLECTIONS + 1):
                name = f"{arm}--s{seed}--c{collection}"
                current = _redundant_rows(models[name])
                point = {
                    "collection": collection,
                    "policy_calls": collection * CALLS_PER_COLLECTION,
                    **_policy_point(initial, current),
                }
                points.append(point)
                transition_divergence.append(
                    {
                        "collection": collection,
                        "mean_behavior_to_updated_exact_kl": fmean(
                            _kl(
                                prior_rows[case_id]["probabilities"],
                                current[case_id]["probabilities"],
                            )
                            for case_id in prior_rows
                        ),
                        "mean_behavior_to_updated_exact_js": fmean(
                            _js(
                                prior_rows[case_id]["probabilities"],
                                current[case_id]["probabilities"],
                            )
                            for case_id in prior_rows
                        ),
                    }
                )
                prior_rows = current
            final_gain = (
                points[-1]["causal_non_delta_target_mass"]
                - points[0]["causal_non_delta_target_mass"]
            )
            updates = 36 if arm == "stock" else (12 if arm.endswith("e2") else 6)
            updates_per_collection = updates // COLLECTIONS
            points = [
                {
                    **point,
                    "optimizer_step": int(point["collection"])
                    * updates_per_collection,
                }
                for point in points
            ]
            result: dict[str, Any] = {
                "points": points,
                "causal_mass_auc_per_576_calls": _auc(points),
                "final_causal_mass_gain": final_gain,
                "calls_to_causal_target_mass_at_least_0_98": next(
                    (
                        int(point["policy_calls"])
                        for point in points
                        if point["causal_non_delta_target_mass"] >= 0.98
                    ),
                    None,
                ),
                "optimizer_updates": updates,
                "final_gain_per_update_descriptive": final_gain / updates,
                "exact_transition_divergence": transition_divergence,
                "usage": _usage(run_dir),
            }
            if arm != "stock":
                result["practical_loss"] = _practical_diagnostics(
                    run_dir, updates
                )
            if arm.endswith("e2"):
                result["reuse_contract"] = _reuse_contract(run_dir)
            runs[f"{arm}--s{seed}"] = result

    e2_auc_wins = sum(
        runs[f"local-e2--s{seed}"]["causal_mass_auc_per_576_calls"]
        > runs[f"local-e1--s{seed}"]["causal_mass_auc_per_576_calls"]
        for seed in SEEDS
    )
    e1_mean_gain = fmean(
        runs[f"local-e1--s{seed}"]["final_causal_mass_gain"]
        for seed in SEEDS
    )
    e2_mean_gain = fmean(
        runs[f"local-e2--s{seed}"]["final_causal_mass_gain"]
        for seed in SEEDS
    )
    gain_ratio = (
        math.inf if e1_mean_gain == 0.0 and e2_mean_gain > 0.0
        else e2_mean_gain / e1_mean_gain
    )
    local_drift = [
        runs[f"local-e2--s{seed}"]["points"][-1][
            "delta_nuisance_js_from_initial"
        ]
        for seed in SEEDS
    ]
    global_drift = [
        runs[f"branch-global-e2--s{seed}"]["points"][-1][
            "delta_nuisance_js_from_initial"
        ]
        for seed in SEEDS
    ]
    checks = {
        "all_reuse_contracts_pass": all(
            runs[f"{arm}--s{seed}"]["reuse_contract"]["all_pairs_passed"]
            and runs[f"{arm}--s{seed}"]["reuse_contract"][
                "fresh_example_stream_between_collections"
            ]
            for arm in ("local-e2", "branch-global-e2")
            for seed in SEEDS
        ),
        "each_run_has_exactly_576_training_calls": all(
            run["usage"]["policy_calls"] == 576 for run in runs.values()
        ),
        "local_e2_auc_exceeds_e1_in_at_least_two_seeds": e2_auc_wins >= 2,
        "local_e2_mean_final_gain_at_least_1_5x_e1": gain_ratio >= 1.5,
        "global_e2_nuisance_exposure_floor_met": fmean(global_drift) >= 0.002,
        "local_e2_mean_delta_js_at_most_75pct_global": (
            fmean(local_drift) <= 0.75 * fmean(global_drift)
        ),
        "each_local_e2_delta_js_no_more_than_global_plus_0_002": all(
            local <= global_value + 0.002
            for local, global_value in zip(
                local_drift, global_drift, strict=True
            )
        ),
    }
    engineering_pass = (
        checks["all_reuse_contracts_pass"]
        and checks["each_run_has_exactly_576_training_calls"]
    )
    efficiency_pass = (
        checks["local_e2_auc_exceeds_e1_in_at_least_two_seeds"]
        and checks["local_e2_mean_final_gain_at_least_1_5x_e1"]
    )
    credit_pass = (
        checks["global_e2_nuisance_exposure_floor_met"]
        and checks["local_e2_mean_delta_js_at_most_75pct_global"]
        and checks[
            "each_local_e2_delta_js_no_more_than_global_plus_0_002"
        ]
    )
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-c9-practical-efficiency",
            "status": (
                "passed"
                if engineering_pass and efficiency_pass and credit_pass
                else "completed_without_full_success"
            ),
            "engineering_pass": engineering_pass,
            "reuse_efficiency_pass": efficiency_pass,
            "matched_data_credit_pass": credit_pass,
            "checks": checks,
            "summary": {
                "local_e2_auc_seed_wins": e2_auc_wins,
                "local_e1_mean_final_gain": e1_mean_gain,
                "local_e2_mean_final_gain": e2_mean_gain,
                "local_e2_to_e1_mean_final_gain_ratio": gain_ratio,
                "local_e2_mean_delta_js": fmean(local_drift),
                "branch_global_e2_mean_delta_js": fmean(global_drift),
            },
            "runs": runs,
            "interpretation": (
                "This bounded bridge tests whether a second clipped update "
                "extracts more causal learning from the same collected branch "
                "data while preserving the local-credit nuisance-drift benefit. "
                "Stock GRPO is an economic Pareto comparator, not a required win. "
                "The second update is a known-ratio PPO-style surrogate and does "
                "not validate long macro-action reuse."
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(args.output, evaluate(args.run_root, args.scores))


if __name__ == "__main__":
    main()
