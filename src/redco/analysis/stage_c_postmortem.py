"""Postmortem and power diagnostics for the completed Stage-C campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

from redco.env.tasks.credit_probes import (
    FiniteCreditProbe,
    credit_probe_by_name,
    standard_credit_probes,
)


@dataclass(frozen=True, slots=True)
class ArmSpec:
    name: str
    run_dir: Path
    final_step: int
    branching: bool


@dataclass(frozen=True, slots=True)
class GradientSummary:
    count: int
    minimum: float
    median: float
    mean: float
    maximum: float


def informative_group_probability(
    reward_class_masses: Sequence[float],
    *,
    branch_count: int,
) -> float:
    """Return the chance that independent branches span multiple reward classes."""
    if branch_count < 2:
        raise ValueError("branch_count must be at least two")
    if not reward_class_masses:
        raise ValueError("at least one reward-class mass is required")
    if any(not math.isfinite(value) or value < 0 for value in reward_class_masses):
        raise ValueError("reward-class masses must be finite and non-negative")
    total = math.fsum(reward_class_masses)
    if not math.isclose(total, 1.0, abs_tol=1e-12):
        raise ValueError("reward-class masses must sum to one")
    return 1.0 - math.fsum(value**branch_count for value in reward_class_masses)


def binary_informative_group_probability(
    successful_action_mass: float,
    *,
    branch_count: int,
) -> float:
    """Specialize informativeness to one successful and one failed reward class."""
    if not 0.0 <= successful_action_mass <= 1.0:
        raise ValueError("successful_action_mass must lie in [0, 1]")
    return informative_group_probability(
        (successful_action_mass, 1.0 - successful_action_mass),
        branch_count=branch_count,
    )


def minimum_branch_count(
    reward_class_masses: Sequence[float],
    *,
    target_probability: float,
    maximum: int = 10_000,
) -> int:
    """Return the smallest branch count meeting an informativeness target."""
    if not 0.0 < target_probability < 1.0:
        raise ValueError("target_probability must lie strictly between zero and one")
    for branch_count in range(2, maximum + 1):
        if (
            informative_group_probability(
                reward_class_masses,
                branch_count=branch_count,
            )
            >= target_probability
        ):
            return branch_count
    raise ValueError("target probability is not attainable below the maximum")


def exact_policy_gradient(
    probabilities: Sequence[float],
    q_values: Sequence[float],
) -> tuple[float, ...]:
    """Return the exact categorical logit gradient from enumerated Q values."""
    if len(probabilities) != len(q_values) or not probabilities:
        raise ValueError("probabilities and q_values must have equal nonzero length")
    if any(not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("probabilities must be finite and non-negative")
    if any(not math.isfinite(value) for value in q_values):
        raise ValueError("q_values must be finite")
    if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-12):
        raise ValueError("probabilities must sum to one")
    value = math.fsum(
        probability * q
        for probability, q in zip(probabilities, q_values, strict=True)
    )
    return tuple(
        probability * (q - value)
        for probability, q in zip(probabilities, q_values, strict=True)
    )


def enumerate_probe(
    probe: FiniteCreditProbe,
    *,
    exogenous_seeds: Iterable[int],
) -> dict[str, Any]:
    """Enumerate exact rewards and optimal actions for finite probe states."""
    states: list[dict[str, Any]] = []
    for seed in exogenous_seeds:
        rewards = {
            action: probe.reward_function(action, seed) for action in probe.actions
        }
        best_reward = max(rewards.values())
        states.append(
            {
                "exogenous_seed": seed,
                "rewards": rewards,
                "best_reward": best_reward,
                "best_actions": sorted(
                    action for action, reward in rewards.items() if reward == best_reward
                ),
                "action_dependent": len(set(rewards.values())) > 1,
            }
        )
    return {
        "probe_name": probe.name,
        "actions": list(probe.actions),
        "states": states,
        "action_dependent_states": sum(state["action_dependent"] for state in states),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        values = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"{path} contains a non-object JSON line")
    return values


def _step(path: Path) -> int:
    match = re.search(r"[\\/]step_(\d+)[\\/]", str(path))
    if match is None:
        raise ValueError(f"cannot infer step from {path}")
    return int(match.group(1))


def _sampled_reply(trace: dict[str, Any]) -> tuple[str, tuple[float, ...]]:
    nodes = trace.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("trace is missing nodes")
    sampled = [node for node in nodes if isinstance(node, dict) and node.get("sampled")]
    if len(sampled) != 1:
        raise ValueError("expected exactly one sampled node")
    message = sampled[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("sampled node is missing its reply")
    logprobs = sampled[0].get("logprobs")
    if not isinstance(logprobs, list):
        raise ValueError("sampled node has invalid logprobs")
    return message["content"], tuple(float(value) for value in logprobs)


def _original_eval_records(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for trace in _read_jsonl(path):
        agent = trace.get("agent")
        rewards = trace.get("rewards")
        if (
            not isinstance(agent, dict)
            or agent.get("name") != "original"
            or not isinstance(rewards, dict)
            or "deterministic_reward" not in rewards
        ):
            continue
        task = trace.get("task")
        if not isinstance(task, dict) or not isinstance(task.get("data"), dict):
            raise ValueError("eval trace is missing task data")
        data = task["data"]
        probe = data.get("probe_name")
        seed = data.get("exogenous_seed")
        if not isinstance(probe, str) or not isinstance(seed, int):
            raise ValueError("eval trace is missing its paired key")
        reply, logprobs = _sampled_reply(trace)
        records[(probe, seed)] = {
            "reply": reply,
            "reward": float(rewards["deterministic_reward"]),
            "logprobs": logprobs,
            "task_data": data,
        }
    return records


def _best_achievable_reward(record: dict[str, Any]) -> float:
    data = record["task_data"]
    probe = credit_probe_by_name(data["probe_name"])
    aliases = dict(data["action_map"])
    canonical = aliases.get(record["reply"])
    observed_local = (
        probe.reward_function(canonical, data["exogenous_seed"])
        if canonical is not None
        else 0.0
    )
    background = record["reward"] - observed_local
    best_local = max(
        probe.reward_function(action, data["exogenous_seed"])
        for action in probe.actions
    )
    return float(background + best_local)


def _gradient_summary(run_dir: Path) -> GradientSummary:
    matches = sorted(run_dir.glob("metrics.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"expected one root metrics.jsonl in {run_dir}")
    values = [
        float(row["optim/grad_norm"])
        for row in _read_jsonl(matches[0])
        if "optim/grad_norm" in row
    ]
    if not values:
        raise ValueError(f"{run_dir} contains no gradient norms")
    return GradientSummary(
        count=len(values),
        minimum=min(values),
        median=median(values),
        mean=fmean(values),
        maximum=max(values),
    )


def _adapter_sha256(run_dir: Path, final_step: int) -> str:
    matches = sorted(
        run_dir.glob(
            f"run_default/broadcasts/step_{final_step}/adapter_model.safetensors"
        )
    )
    if len(matches) != 1:
        raise ValueError(f"missing final adapter for {run_dir}")
    digest = hashlib.sha256()
    with matches[0].open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_arm(spec: ArmSpec) -> dict[str, Any]:
    """Summarize movement, effective information, and held-out regret for one arm."""
    train_paths = sorted(
        spec.run_dir.glob(
            "run_default/rollouts/step_*/train/effective/traces.jsonl"
        ),
        key=_step,
    )
    if len(train_paths) != spec.final_step:
        raise ValueError(f"{spec.name} has {len(train_paths)} train steps")

    action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    planted_logprobs: list[float] = []
    groups: dict[tuple[int, str], list[float]] = defaultdict(list)
    nonzero_advantages = 0
    branch_records = 0
    informative_steps: set[int] = set()
    training_policy_calls = 0
    for path in train_paths:
        step = _step(path)
        for trace in _read_jsonl(path):
            calls = trace.get("calls")
            if isinstance(calls, list):
                training_policy_calls += len(calls)
            task = trace.get("task")
            info = trace.get("info")
            rewards = trace.get("rewards")
            if (
                not isinstance(task, dict)
                or not isinstance(task.get("data"), dict)
                or not isinstance(info, dict)
                or not isinstance(rewards, dict)
                or "deterministic_reward" not in rewards
            ):
                continue
            data = task["data"]
            probe = data.get("probe_name")
            if not isinstance(probe, str):
                continue
            record = info.get("redco")
            if spec.branching and isinstance(record, dict):
                if record.get("record_kind") != "branch":
                    continue
                branch_records += 1
                action = record.get("parsed_action")
                if isinstance(action, str):
                    action_counts[probe][action] += 1
                episode = info.get("episode_id")
                if not isinstance(episode, str):
                    raise ValueError("branch trace is missing episode_id")
                groups[(step, episode)].append(float(rewards["deterministic_reward"]))
                advantage = trace.get("metrics", {}).get("redco_branch_advantage")
                if not isinstance(advantage, int | float):
                    raise ValueError("effective branch trace is missing its advantage")
                if float(advantage) != 0.0:
                    nonzero_advantages += 1
                    informative_steps.add(step)
                if probe == "planted_needle" and action == "5":
                    _, logprobs = _sampled_reply(trace)
                    if logprobs:
                        planted_logprobs.append(logprobs[-1])
            elif not spec.branching and isinstance(info.get("redco_control"), dict):
                reply, _ = _sampled_reply(trace)
                action_counts[probe][reply] += 1

    informative_groups = sum(len(set(rewards)) > 1 for rewards in groups.values())
    eval_paths = {
        _step(path): path
        for path in spec.run_dir.glob(
            "run_default/rollouts/step_*/eval/all/traces.jsonl"
        )
    }
    first_step = min(eval_paths)
    first = _original_eval_records(eval_paths[first_step])
    final = _original_eval_records(eval_paths[spec.final_step])
    if set(first) != set(final):
        raise ValueError("initial and final evaluation keys differ")

    changed_actions = 0
    improved = 0
    regressed = 0
    reward_neutral_changes = 0
    final_regret = 0.0
    avoidable_failures = 0
    for key in first:
        if first[key]["reply"] != final[key]["reply"]:
            changed_actions += 1
            delta = final[key]["reward"] - first[key]["reward"]
            improved += delta > 0
            regressed += delta < 0
            reward_neutral_changes += delta == 0
        regret = _best_achievable_reward(final[key]) - final[key]["reward"]
        final_regret += regret
        avoidable_failures += regret > 0

    planted_count = action_counts["planted_needle"]["5"]
    return {
        "name": spec.name,
        "training": {
            "optimizer_steps": spec.final_step,
            "policy_calls": training_policy_calls,
            "gradient_norm": asdict(_gradient_summary(spec.run_dir)),
            "branch_records": branch_records,
            "branch_groups": len(groups),
            "informative_branch_groups": informative_groups,
            "informative_branch_group_rate": (
                informative_groups / len(groups) if groups else None
            ),
            "nonzero_branch_advantage_records": nonzero_advantages,
            "informative_branch_steps": len(informative_steps),
            "actions_by_probe": {
                probe: dict(sorted(counts.items()))
                for probe, counts in sorted(action_counts.items())
            },
            "planted_needle_action_5_samples": planted_count,
            "planted_needle_action_5_selected_logprobs": planted_logprobs,
        },
        "movement": {
            "first_eval_step": first_step,
            "final_eval_step": spec.final_step,
            "greedy_action_changes": changed_actions,
            "reward_improvements": improved,
            "reward_regressions": regressed,
            "reward_neutral_action_changes": reward_neutral_changes,
            "eval_action_logprobs_recorded": any(
                record["logprobs"] for record in final.values()
            ),
            "kl_from_initialization_recorded": False,
            "final_adapter_sha256": _adapter_sha256(
                spec.run_dir,
                spec.final_step,
            ),
        },
        "heldout": {
            "examples": len(final),
            "final_mean_reward": fmean(record["reward"] for record in final.values()),
            "best_achievable_mean_reward": fmean(
                _best_achievable_reward(record) for record in final.values()
            ),
            "total_action_regret": final_regret,
            "avoidable_failures": avoidable_failures,
        },
    }


def power_grid(
    *,
    successful_action_masses: Sequence[float],
    branch_counts: Sequence[int],
    groups_per_step: int,
) -> list[dict[str, Any]]:
    if groups_per_step < 1:
        raise ValueError("groups_per_step must be positive")
    rows: list[dict[str, Any]] = []
    for mass in successful_action_masses:
        for branch_count in branch_counts:
            probability = binary_informative_group_probability(
                mass,
                branch_count=branch_count,
            )
            rows.append(
                {
                    "successful_action_mass": mass,
                    "branch_count": branch_count,
                    "informative_group_probability": probability,
                    "expected_informative_groups_per_step": (
                        probability * groups_per_step
                    ),
                }
            )
    return rows


def build_postmortem(specs: Sequence[ArmSpec]) -> dict[str, Any]:
    arms = {spec.name: summarize_arm(spec) for spec in specs}
    enumeration = [
        enumerate_probe(probe, exogenous_seeds=range(9000, 9004))
        for probe in standard_credit_probes()
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c-postmortem",
        "status": "exploratory-after-frozen-gate",
        "arms": arms,
        "heldout_enumeration": enumeration,
        "power_grid": power_grid(
            successful_action_masses=(0.01, 0.02, 0.035, 0.05, 0.1, 0.2, 0.3),
            branch_counts=(4, 8, 16, 20, 32, 64),
            groups_per_step=8,
        ),
        "limitations": [
            "The completed evaluation traces omit action logprobs and KL from init.",
            "Selected-action logprobs do not recover the complete categorical policy.",
            "The oracle enumeration is a diagnostic, not the sampled ReDCO estimator.",
        ],
    }
    signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sliced-dir", type=Path, required=True)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--broadcast-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_postmortem(
        (
            ArmSpec("sliced", args.sliced_dir, 30, True),
            ArmSpec("full_suffix", args.full_dir, 30, True),
            ArmSpec("broadcast", args.broadcast_dir, 75, False),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
