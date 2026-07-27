from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    lower: float
    upper: float


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object JSON line")
                records.append(value)
    return records


def _step(path: Path) -> int:
    for parent in path.parents:
        if parent.name.startswith("step_"):
            return int(parent.name.removeprefix("step_"))
    raise ValueError(f"cannot infer a step from {path}")


def _one_per_step(run_dir: Path, pattern: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(run_dir.glob(pattern)):
        step = _step(path)
        if step in result:
            raise ValueError(f"multiple artifacts for step {step}: {pattern}")
        result[step] = path
    return result


def _eval_rewards(path: Path) -> dict[tuple[str, int], float]:
    rewards: dict[tuple[str, int], float] = {}
    for trace in _read_jsonl(path):
        agent = trace.get("agent")
        if not isinstance(agent, dict) or agent.get("name") != "original":
            continue
        task = trace.get("task")
        reward_parts = trace.get("rewards")
        if not isinstance(task, dict) or not isinstance(reward_parts, dict):
            raise ValueError("eval trace is missing task or reward data")
        data = task.get("data")
        if not isinstance(data, dict):
            raise ValueError("eval trace is missing task data")
        probe = data.get("probe_name")
        seed = data.get("exogenous_seed")
        reward = reward_parts.get("deterministic_reward")
        if not isinstance(probe, str) or not isinstance(seed, int):
            raise ValueError("eval trace is missing its probe key")
        if not isinstance(reward, int | float) or not math.isfinite(float(reward)):
            raise ValueError("eval trace has a non-finite deterministic reward")
        key = (probe, seed)
        if key in rewards:
            raise ValueError(f"duplicate eval key: {key}")
        rewards[key] = float(reward)
    if len(rewards) != 32:
        raise ValueError(f"expected 32 held-out original traces, found {len(rewards)}")
    return rewards


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sample")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _paired_bootstrap(
    candidate: dict[tuple[str, int], float],
    reference: dict[tuple[str, int], float],
    *,
    confidence: float,
    samples: int,
    seed: int,
) -> Interval:
    if set(candidate) != set(reference):
        raise ValueError("paired eval arms do not contain identical held-out tasks")
    keys = sorted(candidate)
    differences = [candidate[key] - reference[key] for key in keys]
    rng = random.Random(seed)
    boot = [
        fmean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(samples)
    ]
    alpha = (1 - confidence) / 2
    return Interval(
        estimate=fmean(differences),
        lower=_quantile(boot, alpha),
        upper=_quantile(boot, 1 - alpha),
    )


def _summarize_arm(
    run_dir: Path,
    *,
    name: str,
    max_steps: int,
    branching: bool,
) -> dict[str, Any]:
    train_paths = _one_per_step(
        run_dir, "**/rollouts/step_*/train/all/traces.jsonl"
    )
    if set(train_paths) != set(range(1, max_steps + 1)):
        raise ValueError(f"{name} does not contain every train step")
    expected_per_step = 40 if branching else 16
    branch_records = 0
    replay_mismatches = 0
    invalid_snapshots = 0
    errors = 0
    total_calls = 0
    for step, path in train_paths.items():
        traces = _read_jsonl(path)
        if len(traces) != expected_per_step:
            raise ValueError(
                f"{name} step {step} has {len(traces)} traces; "
                f"expected {expected_per_step}"
            )
        total_calls += len(traces)
        for trace in traces:
            info = trace.get("info")
            if not isinstance(info, dict):
                raise ValueError("training trace is missing info")
            if info.get("policy_version") != step - 1:
                invalid_snapshots += 1
            if trace.get("ok") is not True or trace.get("errors") not in ([], None):
                errors += 1
            if branching:
                record = info.get("redco")
                if not isinstance(record, dict):
                    raise ValueError("branching trace is missing its redco record")
                if record.get("record_kind") == "branch":
                    branch_records += 1
                    if record.get("replay_equivalent") is not True:
                        replay_mismatches += 1
                    if record.get("selected_pre_action") is not True:
                        raise ValueError("a Stage C target was not precommitted")
            elif not isinstance(info.get("redco_control"), dict):
                raise ValueError("broadcast trace is missing control metadata")

    metrics_paths = sorted(run_dir.glob("**/metrics.jsonl"))
    if len(metrics_paths) != 1:
        raise ValueError(f"{name} must contain exactly one metrics.jsonl")
    metric_rows = _read_jsonl(metrics_paths[0])
    grad_norms = [
        float(row["optim/grad_norm"])
        for row in metric_rows
        if "optim/grad_norm" in row
    ]
    if len(grad_norms) != max_steps:
        raise ValueError(f"{name} has {len(grad_norms)} optimizer steps")
    if not all(math.isfinite(value) for value in grad_norms):
        raise ValueError(f"{name} contains a non-finite gradient")
    if not any(value > 0 for value in grad_norms):
        raise ValueError(f"{name} never produced a positive gradient")

    eval_paths = _one_per_step(
        run_dir, "**/rollouts/step_*/eval/all/traces.jsonl"
    )
    if max_steps not in eval_paths:
        raise ValueError(f"{name} is missing its final evaluation")
    curves = {
        step: fmean(_eval_rewards(path).values())
        for step, path in eval_paths.items()
    }
    final_rewards = _eval_rewards(eval_paths[max_steps])
    adapter_matches = sorted(
        run_dir.glob(
            f"**/broadcasts/step_{max_steps}/adapter_model.safetensors"
        )
    )
    if len(adapter_matches) != 1 or adapter_matches[0].stat().st_size == 0:
        raise ValueError(f"{name} is missing its final LoRA adapter")

    return {
        "name": name,
        "max_steps": max_steps,
        "train_policy_calls": total_calls,
        "branch_records": branch_records,
        "replay_mismatches": replay_mismatches,
        "invalid_snapshot_records": invalid_snapshots,
        "errored_traces": errors,
        "positive_grad_steps": sum(value > 0 for value in grad_norms),
        "eval_curve": {str(step): value for step, value in curves.items()},
        "final_eval_mean": fmean(final_rewards.values()),
        "final_eval_rewards": {
            f"{probe}:{seed}": value
            for (probe, seed), value in sorted(final_rewards.items())
        },
    }


def evaluate_learning_gate(
    *,
    broadcast_dir: Path,
    full_dir: Path,
    sliced_dir: Path,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
    minimum_branch_improvement: float,
    mode_point_margin: float,
    mode_interval_margin: float,
) -> dict[str, Any]:
    arms = {
        "broadcast": _summarize_arm(
            broadcast_dir, name="broadcast", max_steps=75, branching=False
        ),
        "full_suffix": _summarize_arm(
            full_dir, name="full_suffix", max_steps=30, branching=True
        ),
        "sliced": _summarize_arm(
            sliced_dir, name="sliced", max_steps=30, branching=True
        ),
    }

    def final_rewards(arm: str) -> dict[tuple[str, int], float]:
        values = arms[arm]["final_eval_rewards"]
        return {
            (key.rsplit(":", 1)[0], int(key.rsplit(":", 1)[1])): value
            for key, value in values.items()
        }

    full_broadcast = _paired_bootstrap(
        final_rewards("full_suffix"),
        final_rewards("broadcast"),
        confidence=confidence,
        samples=bootstrap_samples,
        seed=seed,
    )
    sliced_broadcast = _paired_bootstrap(
        final_rewards("sliced"),
        final_rewards("broadcast"),
        confidence=confidence,
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    sliced_full = _paired_bootstrap(
        final_rewards("sliced"),
        final_rewards("full_suffix"),
        confidence=confidence,
        samples=bootstrap_samples,
        seed=seed + 2,
    )

    integration_pass = all(
        arm["invalid_snapshot_records"] == 0
        and arm["errored_traces"] == 0
        and arm["replay_mismatches"] == 0
        and arm["train_policy_calls"] == 1200
        for arm in arms.values()
    )
    learning_pass = (
        full_broadcast.estimate >= minimum_branch_improvement
        and full_broadcast.lower > 0
        and sliced_broadcast.estimate >= minimum_branch_improvement
        and sliced_broadcast.lower > 0
    )
    mode_equivalence_pass = (
        abs(sliced_full.estimate) <= mode_point_margin
        and sliced_full.lower >= -mode_interval_margin
        and sliced_full.upper <= mode_interval_margin
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "gate": "gate-gc-live-learning",
        "status": (
            "pass"
            if integration_pass and learning_pass and mode_equivalence_pass
            else "fail"
        ),
        "arms": arms,
        "comparisons": {
            "full_suffix_minus_broadcast": asdict(full_broadcast),
            "sliced_minus_broadcast": asdict(sliced_broadcast),
            "sliced_minus_full_suffix": asdict(sliced_full),
        },
        "thresholds": {
            "confidence": confidence,
            "bootstrap_samples": bootstrap_samples,
            "minimum_branch_improvement": minimum_branch_improvement,
            "mode_point_margin": mode_point_margin,
            "mode_interval_margin": mode_interval_margin,
        },
        "checks": {
            "integration_pass": integration_pass,
            "learning_pass": learning_pass,
            "mode_equivalence_pass": mode_equivalence_pass,
            "matched_policy_calls_per_arm": 1200,
            "accelerator_savings_claimed": False,
            "note": (
                "The restricted deterministic probe executes both replay reward "
                "paths in-loop, so this mini gate tests learning and exact replay "
                "agreement but cannot establish physical GPU savings."
            ),
        },
    }
    signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the Stage C live gate")
    parser.add_argument("--broadcast-dir", type=Path, required=True)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--sliced-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7202603)
    parser.add_argument("--minimum-branch-improvement", type=float, default=0.02)
    parser.add_argument("--mode-point-margin", type=float, default=0.05)
    parser.add_argument("--mode-interval-margin", type=float, default=0.15)
    args = parser.parse_args()
    report = evaluate_learning_gate(
        broadcast_dir=args.broadcast_dir,
        full_dir=args.full_dir,
        sliced_dir=args.sliced_dir,
        confidence=args.confidence,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        minimum_branch_improvement=args.minimum_branch_improvement,
        mode_point_margin=args.mode_point_margin,
        mode_interval_margin=args.mode_interval_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
