"""CPU diagnostics for the Stage-C credit-quality and efficiency follow-up."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from redco.algo.branching import (
    CommitmentStatus,
    RandomizedSelectiveTargetSelector,
    inclusive_group_mean_advantages,
    leave_one_out_advantages,
)
from redco.contracts import PolicyNodeKind, PrefixFeatures

ACTIONS = (*tuple(str(index) for index in range(8)), "invalid")
ROUTES = ("alpha", "beta", "gamma", "delta")
PROBES = ("confusion_irrelevant", "confusion_redundant", "confusion_lucky")
LOOP_DURATION = re.compile(
    r"Orchestrator step loop done in (?:(?P<minutes>\d+)m )?(?P<seconds>\d+)s"
)


@dataclass(frozen=True, slots=True)
class BranchObservation:
    episode_id: str
    route: str
    action: str
    reward: float


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _mean(values: Iterable[float]) -> float:
    items = tuple(values)
    if not items:
        raise ValueError("cannot take the mean of an empty sequence")
    return math.fsum(items) / len(items)


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in vector))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    denominator = _norm(left) * _norm(right)
    return None if denominator == 0.0 else _dot(left, right) / denominator


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    offset = max(logits)
    unnormalized = tuple(math.exp(value - offset) for value in logits)
    denominator = math.fsum(unnormalized)
    return tuple(value / denominator for value in unnormalized)


def _score_gradient(
    action: str,
    probabilities: Sequence[float],
    advantage: float,
    *,
    ratio: float = 1.0,
) -> tuple[float, ...]:
    action_index = ACTIONS.index(action)
    return tuple(
        advantage * ratio * ((1.0 if index == action_index else 0.0) - probability)
        for index, probability in enumerate(probabilities)
    )


def _add_vectors(vectors: Iterable[Sequence[float]]) -> tuple[float, ...]:
    items = tuple(vectors)
    if not items:
        return (0.0,) * len(ACTIONS)
    return tuple(math.fsum(vector[index] for vector in items) for index in range(len(ACTIONS)))


def _scale(vector: Sequence[float], factor: float) -> tuple[float, ...]:
    return tuple(value * factor for value in vector)


def _orthogonal_residual(
    estimate: Sequence[float],
    reference: Sequence[float],
) -> tuple[float, ...]:
    denominator = _dot(reference, reference)
    if denominator == 0.0:
        return tuple(estimate)
    projection_scale = _dot(estimate, reference) / denominator
    return tuple(
        value - projection_scale * target
        for value, target in zip(estimate, reference, strict=True)
    )


def _target_probability_derivative(
    probabilities: Sequence[float],
    gradient: Sequence[float],
) -> float:
    target_index = ACTIONS.index("5")
    expected = _dot(probabilities, gradient)
    return probabilities[target_index] * (gradient[target_index] - expected)


def _reward(probe: str, route: str, action: str, *, luck: int = 0) -> float:
    route_reward = {"alpha": -0.25, "beta": 0.0, "gamma": 0.25, "delta": 0.5}[route]
    target_success = action == "5"
    if probe == "confusion_irrelevant":
        return route_reward
    if probe == "confusion_redundant":
        return float(route == "delta" or target_success)
    if probe == "confusion_lucky":
        return float(target_success) + route_reward + float(luck)
    raise ValueError(f"unknown probe: {probe}")


def _probability_rows(score_payload: Mapping[str, Any]) -> dict[tuple[str, str], tuple[float, ...]]:
    warmstart = next(model for model in score_payload["models"] if model["name"] == "warmstart")
    by_route: dict[str, tuple[float, ...]] = {}
    for row in warmstart["temperatures"]["2.0"]:
        valid = tuple(float(row["action_probabilities"][action]) for action in ACTIONS[:-1])
        invalid = max(0.0, 1.0 - math.fsum(valid))
        route = str(row["context_route"])
        probabilities = (*valid, invalid)
        if route in by_route and by_route[route] != probabilities:
            raise ValueError("shared target prompt has inconsistent probability rows")
        by_route[route] = probabilities
    if set(by_route) != set(ROUTES):
        raise ValueError("warm-start scores must cover all four context routes")
    return {
        (probe, route): by_route[route]
        for probe in PROBES
        for route in ROUTES
    }


def _branch_observations(path: Path) -> tuple[BranchObservation, ...]:
    observations: list[BranchObservation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        metadata = row.get("info", {}).get("redco", {})
        if metadata.get("record_kind") != "branch":
            continue
        action = metadata.get("canonical_action")
        observations.append(
            BranchObservation(
                episode_id=str(row["info"]["episode_id"]),
                route=str(metadata["context_route"]),
                action=str(action) if action in ACTIONS[:-1] else "invalid",
                reward=float(metadata["sliced_reward"]),
            )
        )
    return tuple(observations)


def _advantages(
    observations: Sequence[BranchObservation],
    *,
    method: str,
) -> tuple[float, ...]:
    if method == "matched_broadcast":
        return inclusive_group_mean_advantages(tuple(row.reward for row in observations))
    if method != "local_loo":
        raise ValueError(f"unknown credit method: {method}")
    by_episode: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(observations):
        by_episode[row.episode_id].append(index)
    result = [0.0] * len(observations)
    for indices in by_episode.values():
        episode_advantages = leave_one_out_advantages(
            tuple(observations[index].reward for index in indices)
        )
        for index, advantage in zip(indices, episode_advantages, strict=True):
            result[index] = advantage
    return tuple(result)


def _estimated_route_gradients(
    observations: Sequence[BranchObservation],
    advantages: Sequence[float],
    probabilities: Mapping[tuple[str, str], Sequence[float]],
    probe: str,
) -> dict[str, tuple[float, ...]]:
    route_vectors: dict[str, list[tuple[float, ...]]] = defaultdict(list)
    for row, advantage in zip(observations, advantages, strict=True):
        route_vectors[row.route].append(
            _score_gradient(row.action, probabilities[(probe, row.route)], advantage)
        )
    return {
        route: _scale(_add_vectors(vectors), 1.0 / len(vectors))
        for route, vectors in route_vectors.items()
    }


def _oracle_route_gradient(
    probe: str,
    route: str,
    probabilities: Sequence[float],
) -> tuple[float, ...]:
    rewards = tuple(_reward(probe, route, action) for action in ACTIONS)
    baseline = _dot(probabilities, rewards)
    return tuple(
        probability * (reward - baseline)
        for probability, reward in zip(probabilities, rewards, strict=True)
    )


def _aggregate_by_empirical_route(
    route_vectors: Mapping[str, Sequence[float]],
    observations: Sequence[BranchObservation],
) -> tuple[float, ...]:
    episode_routes: dict[str, str] = {}
    for row in observations:
        episode_routes[row.episode_id] = row.route
    counts: dict[str, int] = defaultdict(int)
    for route in episode_routes.values():
        counts[route] += 1
    total = math.fsum(counts.values())
    return _add_vectors(
        _scale(route_vectors.get(route, (0.0,) * len(ACTIONS)), count / total)
        for route, count in counts.items()
    )


def matched_gradient_diagnostic(
    evidence_root: Path,
    probabilities: Mapping[tuple[str, str], Sequence[float]],
) -> dict[str, Any]:
    runs = (
        ("confusion_irrelevant", "sliced-s9921"),
        ("confusion_irrelevant", "sliced-s9922"),
        ("confusion_redundant", "sliced-s9923"),
        ("confusion_lucky", "sliced-s9924"),
    )
    results: dict[str, Any] = {}
    for probe, run_name in runs:
        trace_path = (
            evidence_root
            / probe
            / run_name
            / "run_default"
            / "rollouts"
            / "step_1"
            / "train"
            / "all"
            / "traces.jsonl"
        )
        observations = _branch_observations(trace_path)
        oracle_by_route = {
            route: _oracle_route_gradient(probe, route, probabilities[(probe, route)])
            for route in ROUTES
        }
        oracle = _aggregate_by_empirical_route(oracle_by_route, observations)
        methods: dict[str, Any] = {}
        for method in ("local_loo", "matched_broadcast"):
            advantages = _advantages(observations, method=method)
            estimated_by_route = _estimated_route_gradients(
                observations,
                advantages,
                probabilities,
                probe,
            )
            estimate = _aggregate_by_empirical_route(estimated_by_route, observations)
            derivative_by_route = {
                route: _target_probability_derivative(
                    probabilities[(probe, route)],
                    estimated_by_route.get(route, (0.0,) * len(ACTIONS)),
                )
                for route in ROUTES
            }
            episode_routes = {
                row.episode_id: row.route
                for row in observations
            }
            target_derivative = _mean(
                derivative_by_route[route] for route in episode_routes.values()
            )
            methods[method] = {
                "advantage_zero_fraction": sum(value == 0.0 for value in advantages)
                / len(advantages),
                "cosine_to_exact_oracle": _cosine(estimate, oracle),
                "gradient": dict(zip(ACTIONS, estimate, strict=True)),
                "gradient_norm": _norm(estimate),
                "nuisance_residual_norm": _norm(_orthogonal_residual(estimate, oracle)),
                "target_mass_first_order_derivative": target_derivative,
            }
        results[f"{probe}--{run_name}"] = {
            "branch_records": len(observations),
            "episodes": len({row.episode_id for row in observations}),
            "exact_oracle_gradient": dict(zip(ACTIONS, oracle, strict=True)),
            "exact_oracle_norm": _norm(oracle),
            "methods": methods,
        }
    return {
        "analysis": "stage-c7-matched-frozen-batch-policy-space-gradient",
        "scope": (
            "Exact categorical action-logit gradients on each sliced arm's frozen step-1 "
            "branch batch; this is not a 4B parameter-gradient comparison."
        ),
        "matched_data": True,
        "matched_optimizer_updates": True,
        "results": results,
    }


def _load_trace_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    return rows


def _usage(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    calls = [call for row in rows for call in row.get("calls", ())]
    prompt_tokens = sum(int(call["usage"]["prompt_tokens"]) for call in calls)
    completion_tokens = sum(int(call["usage"]["completion_tokens"]) for call in calls)
    model_seconds = math.fsum(
        float(call["time"]["end"]) - float(call["time"]["start"])
        for call in calls
    )
    return {
        "policy_calls": len(calls),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "summed_model_service_seconds": model_seconds,
    }


def _loop_seconds(path: Path) -> int:
    match = LOOP_DURATION.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"orchestrator duration is missing from {path}")
    return int(match.group("minutes") or 0) * 60 + int(match.group("seconds"))


def campaign_ledger(evidence_root: Path, decision: Mapping[str, Any]) -> dict[str, Any]:
    ledgers: dict[str, Any] = {}
    for probe in PROBES:
        for run_dir in sorted((evidence_root / probe).iterdir()):
            if not run_dir.is_dir() or not (
                run_dir.name.startswith("broadcast-")
                or run_dir.name.startswith("sliced-")
            ):
                continue
            arm, seed_text = run_dir.name.split("-s", maxsplit=1)
            train_paths = sorted(
                run_dir.glob("run_default/rollouts/step_*/train/all/traces.jsonl")
            )
            eval_paths = sorted(
                run_dir.glob("run_default/rollouts/step_*/eval/all/traces.jsonl")
            )
            train_rows = _load_trace_rows(train_paths)
            eval_rows = _load_trace_rows(eval_paths)
            key = f"{probe}--{arm}--s{seed_text}"
            frozen_run = decision["runs"][key]
            initial_mass = float(frozen_run["mean_initial_target_action_mass"])
            final_mass = float(frozen_run["mean_target_action_mass"])
            ledgers[key] = {
                "probe": probe,
                "arm": arm,
                "seed": int(seed_text),
                "optimizer_updates": len(train_paths),
                "orchestrator_loop_seconds": _loop_seconds(run_dir / "logs" / "orchestrator.log"),
                "train": _usage(train_rows),
                "evaluation": _usage(eval_rows),
                "initial_target_action_mass": initial_mass,
                "final_target_action_mass": final_mass,
                "causal_target_mass_gain": (
                    final_mass - initial_mass
                    if probe in {"confusion_redundant", "confusion_lucky"}
                    else None
                ),
                "irrelevant_target_js_drift": (
                    float(frozen_run["mean_selected_action_js_from_initial"])
                    if probe == "confusion_irrelevant"
                    else None
                ),
            }
    for ledger in ledgers.values():
        updates = int(ledger["optimizer_updates"])
        calls = int(ledger["train"]["policy_calls"])
        tokens = int(ledger["train"]["generated_tokens"])
        seconds = int(ledger["orchestrator_loop_seconds"])
        gain = ledger["causal_target_mass_gain"]
        drift = ledger["irrelevant_target_js_drift"]
        ledger["normalized"] = {
            "causal_gain_per_update": None if gain is None else gain / updates,
            "causal_gain_per_policy_call": None if gain is None else gain / calls,
            "causal_gain_per_generated_token": None if gain is None else gain / tokens,
            "causal_gain_per_wall_second": None if gain is None else gain / seconds,
            "irrelevant_drift_per_update": None if drift is None else drift / updates,
            "irrelevant_drift_per_policy_call": None if drift is None else drift / calls,
            "irrelevant_drift_per_generated_token": None if drift is None else drift / tokens,
            "irrelevant_drift_per_wall_second": None if drift is None else drift / seconds,
        }
    return {
        "analysis": "stage-c7-dual-efficiency-ledger",
        "wall_clock_definition": (
            "orchestrator step-loop duration, excluding startup and final save"
        ),
        "runs": ledgers,
    }


def _root_distribution(root_scores: Mapping[str, Any]) -> dict[str, float]:
    raw = root_scores["temperature_2"]["route_sequence_probabilities"]
    denominator = math.fsum(float(raw[route]) for route in ROUTES)
    return {route: float(raw[route]) / denominator for route in ROUTES}


def branch_group_power_analysis(
    probabilities: Mapping[tuple[str, str], Sequence[float]],
    root_distribution: Mapping[str, float],
    *,
    fixed_episode_groups: int = 8,
    fixed_call_budget: int = 96,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for group_size in range(2, 12):
        target_informative_by_route = {
            route: (
                1.0
                - probabilities[("confusion_redundant", route)][ACTIONS.index("5")]
                ** group_size
                - (
                    1.0
                    - probabilities[("confusion_redundant", route)][ACTIONS.index("5")]
                )
                ** group_size
            )
            for route in ROUTES
        }
        mean_target_informative = math.fsum(
            root_distribution[route] * target_informative_by_route[route]
            for route in ROUTES
        )
        mean_redundant_reward_informative = math.fsum(
            root_distribution[route]
            * (target_informative_by_route[route] if route != "delta" else 0.0)
            for route in ROUTES
        )
        fixed_groups_expected = fixed_episode_groups * mean_target_informative
        groups_at_fixed_calls = fixed_call_budget // (group_size + 1)
        rows.append(
            {
                "branch_group_size": group_size,
                "calls_per_episode": group_size + 1,
                "target_action_informative_probability_per_episode": (
                    mean_target_informative
                ),
                "redundant_reward_informative_probability_per_episode": (
                    mean_redundant_reward_informative
                ),
                "fixed_8_episode_calls_per_update": fixed_episode_groups
                * (group_size + 1),
                "fixed_8_episode_expected_informative_groups": fixed_groups_expected,
                "fixed_8_episode_expected_redundant_reward_informative_groups": (
                    fixed_episode_groups * mean_redundant_reward_informative
                ),
                "episode_groups_at_96_calls": groups_at_fixed_calls,
                "expected_informative_groups_at_96_calls": groups_at_fixed_calls
                * mean_target_informative,
                "expected_redundant_reward_informative_groups_at_96_calls": (
                    groups_at_fixed_calls * mean_redundant_reward_informative
                ),
                "minimum_episode_groups_for_5_reward_informative": math.ceil(
                    5.0 / mean_redundant_reward_informative
                ),
                "minimum_calls_for_5_reward_informative": (
                    math.ceil(5.0 / mean_redundant_reward_informative)
                    * (group_size + 1)
                ),
            }
        )
    passing_fixed_groups = [
        row["branch_group_size"]
        for row in rows
        if row["fixed_8_episode_expected_informative_groups"] >= 5.0
    ]
    passing_fixed_calls = [
        row["branch_group_size"]
        for row in rows
        if row["expected_informative_groups_at_96_calls"] >= 5.0
    ]
    minimum_powered_calls = min(
        row["minimum_calls_for_5_reward_informative"] for row in rows
    )
    cheapest_reward_powered_group_sizes = [
        row["branch_group_size"]
        for row in rows
        if row["minimum_calls_for_5_reward_informative"] == minimum_powered_calls
    ]
    return {
        "analysis": "stage-c7-branch-group-power",
        "root_distribution": dict(root_distribution),
        "frozen_informative_group_floor": 5.0,
        "smallest_group_size_with_8_episodes": min(passing_fixed_groups),
        "smallest_group_size_at_96_calls": min(passing_fixed_calls),
        "minimum_calls_for_5_reward_informative": minimum_powered_calls,
        "cheapest_reward_powered_group_sizes": cheapest_reward_powered_group_sizes,
        "rows": rows,
        "interpretation": (
            "Shrinking the branch group while holding eight episode states fixed loses power. "
            "Reallocating calls to more independent states can preserve the floor, but does not "
            "materially lower calls per powered update. The frozen action-diversity definition "
            "is reported separately from stricter reward informativeness, which is zero on "
            "redundant episodes whose root route is delta."
        ),
    }


def _ppo_epoch_gradient(
    observations: Sequence[BranchObservation],
    advantages: Sequence[float],
    old_probabilities: Mapping[str, Sequence[float]],
    logits: Mapping[str, Sequence[float]],
    *,
    clip_epsilon: float,
) -> tuple[dict[str, tuple[float, ...]], float]:
    gradients: dict[str, list[tuple[float, ...]]] = defaultdict(list)
    clipped = 0
    for row, advantage in zip(observations, advantages, strict=True):
        current = _softmax(logits[row.route])
        action_index = ACTIONS.index(row.action)
        ratio = current[action_index] / old_probabilities[row.route][action_index]
        clip_active = (advantage >= 0.0 and ratio > 1.0 + clip_epsilon) or (
            advantage < 0.0 and ratio < 1.0 - clip_epsilon
        )
        if clip_active:
            clipped += 1
            gradients[row.route].append((0.0,) * len(ACTIONS))
        else:
            gradients[row.route].append(
                _score_gradient(row.action, current, advantage, ratio=ratio)
            )
    return (
        {
            route: _scale(_add_vectors(vectors), 1.0 / len(vectors))
            for route, vectors in gradients.items()
        },
        clipped / len(observations),
    )


def mini_epoch_simulation(
    evidence_root: Path,
    probabilities: Mapping[tuple[str, str], Sequence[float]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for probe, run_name in (
        ("confusion_redundant", "sliced-s9923"),
        ("confusion_lucky", "sliced-s9924"),
    ):
        trace_path = (
            evidence_root
            / probe
            / run_name
            / "run_default"
            / "rollouts"
            / "step_1"
            / "train"
            / "all"
            / "traces.jsonl"
        )
        observations = _branch_observations(trace_path)
        advantages = _advantages(observations, method="local_loo")
        old = {
            route: tuple(probabilities[(probe, route)])
            for route in ROUTES
        }
        episode_routes = {row.episode_id: row.route for row in observations}
        cases: list[dict[str, Any]] = []
        for learning_rate in (0.1, 0.5, 1.0):
            for mini_epochs in (1, 2, 3):
                logits = {
                    route: tuple(math.log(max(value, 1e-300)) for value in old[route])
                    for route in ROUTES
                }
                clip_fractions: list[float] = []
                for _ in range(mini_epochs):
                    gradients, clip_fraction = _ppo_epoch_gradient(
                        observations,
                        advantages,
                        old,
                        logits,
                        clip_epsilon=0.2,
                    )
                    clip_fractions.append(clip_fraction)
                    logits = {
                        route: tuple(
                            value + learning_rate * gradient
                            for value, gradient in zip(
                                logits[route],
                                gradients.get(route, (0.0,) * len(ACTIONS)),
                                strict=True,
                            )
                        )
                        for route in ROUTES
                    }
                initial_mass = _mean(
                    old[route][ACTIONS.index("5")]
                    for route in episode_routes.values()
                )
                final_mass = _mean(
                    _softmax(logits[route])[ACTIONS.index("5")]
                    for route in episode_routes.values()
                )
                cases.append(
                    {
                        "logit_learning_rate": learning_rate,
                        "mini_epochs": mini_epochs,
                        "target_mass_change": final_mass - initial_mass,
                        "clip_fraction_by_epoch": clip_fractions,
                    }
                )
        results[probe] = cases
    return {
        "analysis": "stage-c7-tabular-clipped-mini-epoch-reuse",
        "clip_epsilon": 0.2,
        "scope": (
            "Exact categorical PPO simulation on frozen step-1 action batches. It tests "
            "reuse mechanics and clipping, not the unknown 4B parameter-space Jacobian."
        ),
        "results": results,
    }


def selective_targeting_audit(
    *,
    randomized_replay_fraction: float = 0.1,
    trials: int = 100_000,
) -> dict[str, Any]:
    """Exercise priority and randomized paths using pre-action features only."""
    probe_scores = {
        "confusion_irrelevant": 1.0,
        "confusion_redundant": 1.0,
        "confusion_lucky": 0.0,
    }
    selected = 0
    for seed in range(trials):
        selector = RandomizedSelectiveTargetSelector(
            seed=seed,
            randomized_replay_fraction=randomized_replay_fraction,
            priority_threshold=0.5,
        )
        result = selector.consider(
            "synthetic:depth-one-subcall",
            PrefixFeatures(
                node_kind=PolicyNodeKind.SUBCALL_OUTPUT,
                depth=1,
                turn_index=0,
                task_metadata=(("probe_name", "confusion_lucky"),),
                predicted_replay_cost=1.0,
            ),
            priority_score=probe_scores["confusion_lucky"],
        )
        selected += result.status is CommitmentStatus.COMMITTED
    mean_selection_probability = _mean(
        1.0 if score >= 0.5 else randomized_replay_fraction
        for score in probe_scores.values()
    )
    current_calls_per_episode = 1 + 11
    selective_calls_per_episode = (
        1
        + mean_selection_probability * 11
        + (1.0 - mean_selection_probability) * 1
    )
    return {
        "analysis": "stage-c7-pre-action-selective-targeting-audit",
        "priority_scores": probe_scores,
        "priority_threshold": 0.5,
        "randomized_replay_fraction": randomized_replay_fraction,
        "randomized_trials": trials,
        "observed_low_priority_selection_fraction": selected / trials,
        "known_low_priority_selection_propensity": randomized_replay_fraction,
        "pre_action_features_only": True,
        "current_calls_per_episode": current_calls_per_episode,
        "expected_selective_calls_per_episode": selective_calls_per_episode,
        "expected_call_reduction_fraction": (
            1.0 - selective_calls_per_episode / current_calls_per_episode
        ),
        "scope": (
            "Synthetic-probe targeting policy: irrelevant/redundant nodes are priority "
            "targets; lucky nodes retain the permanent randomized replay floor. A skipped "
            "episode still pays for its root and ordinary target rollout."
        ),
    }


def _write_ledger_csv(path: Path, ledger: Mapping[str, Any]) -> None:
    fields = (
        "run",
        "probe",
        "arm",
        "seed",
        "optimizer_updates",
        "train_policy_calls",
        "train_generated_tokens",
        "train_prompt_tokens",
        "wall_seconds",
        "causal_target_mass_gain",
        "irrelevant_target_js_drift",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, row in ledger["runs"].items():
            writer.writerow(
                {
                    "run": key,
                    "probe": row["probe"],
                    "arm": row["arm"],
                    "seed": row["seed"],
                    "optimizer_updates": row["optimizer_updates"],
                    "train_policy_calls": row["train"]["policy_calls"],
                    "train_generated_tokens": row["train"]["generated_tokens"],
                    "train_prompt_tokens": row["train"]["prompt_tokens"],
                    "wall_seconds": row["orchestrator_loop_seconds"],
                    "causal_target_mass_gain": row["causal_target_mass_gain"],
                    "irrelevant_target_js_drift": row["irrelevant_target_js_drift"],
                }
            )


def _svg_bar_chart(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    width, height = 920, 440
    left, top, bottom = 90, 55, 110
    chart_width = width - left - 30
    chart_height = height - top - bottom
    maximum = max(values) if values else 1.0
    bar_width = chart_width / max(1, len(values))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" '
        'font-family="sans-serif" font-size="18" fill="#202020">'
        f"{title}</text>",
        f'<line x1="{left}" y1="{top + chart_height}" x2="{width - 30}" '
        f'y2="{top + chart_height}" stroke="#444"/>',
    ]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = left + index * bar_width + bar_width * 0.15
        rendered_width = bar_width * 0.7
        rendered_height = 0.0 if maximum == 0.0 else chart_height * value / maximum
        y = top + chart_height - rendered_height
        color = "#367a70" if "sliced" in label else "#b65f3c"
        elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{rendered_width:.2f}" '
            f'height="{rendered_height:.2f}" fill="{color}"/>'
        )
        elements.append(
            f'<text x="{x + rendered_width / 2:.2f}" y="{y - 6:.2f}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="11">{value:.4g}</text>'
        )
        elements.append(
            f'<text x="{x + rendered_width / 2:.2f}" y="{top + chart_height + 18}" '
            'text-anchor="end" transform="rotate(-38 '
            f'{x + rendered_width / 2:.2f} {top + chart_height + 18})" '
            f'font-family="sans-serif" font-size="10">{label}</text>'
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def run(evidence_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_payload = _read_json(evidence_root / "final-policy-scores.json")
    decision = _read_json(evidence_root / "frozen-decision.json")
    root_scores = _read_json(evidence_root / "initialization" / "runtime" / "root-scores.json")
    probabilities = _probability_rows(score_payload)

    gradient = matched_gradient_diagnostic(evidence_root, probabilities)
    ledger = campaign_ledger(evidence_root, decision)
    power = branch_group_power_analysis(probabilities, _root_distribution(root_scores))
    mini_epochs = mini_epoch_simulation(evidence_root, probabilities)
    targeting = selective_targeting_audit()
    _write_json(output_dir / "matched-gradient.json", gradient)
    _write_json(output_dir / "efficiency-ledger.json", ledger)
    _write_ledger_csv(output_dir / "efficiency-ledger.csv", ledger)
    _write_json(output_dir / "branch-group-power.json", power)
    _write_json(output_dir / "mini-epoch-simulation.json", mini_epochs)
    _write_json(output_dir / "selective-targeting-audit.json", targeting)

    causal_rows = [
        (key, float(row["causal_target_mass_gain"]))
        for key, row in ledger["runs"].items()
        if row["causal_target_mass_gain"] is not None
    ]
    drift_rows = [
        (key, float(row["irrelevant_target_js_drift"]))
        for key, row in ledger["runs"].items()
        if row["irrelevant_target_js_drift"] is not None
    ]
    _svg_bar_chart(
        output_dir / "causal-gain-by-run.svg",
        "Causal target-mass gain at 576 train calls",
        [key for key, _ in causal_rows],
        [value for _, value in causal_rows],
    )
    _svg_bar_chart(
        output_dir / "irrelevant-drift-by-run.svg",
        "Irrelevant target JS drift at 576 train calls",
        [key for key, _ in drift_rows],
        [value for _, value in drift_rows],
    )

    summary = {
        "analysis": "stage-c7-efficiency-and-targeting-cpu-phase",
        "artifacts": {
            "matched_gradient": "matched-gradient.json",
            "efficiency_ledger": "efficiency-ledger.json",
            "branch_group_power": "branch-group-power.json",
            "mini_epoch_simulation": "mini-epoch-simulation.json",
            "selective_targeting_audit": "selective-targeting-audit.json",
        },
        "gpu_used": False,
        "scientific_scope": (
            "Items 1-4 are frozen-artifact or tabular-policy diagnostics. Item 5 is "
            "implemented as a pre-action selector with a known permanent replay propensity."
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    run(args.evidence_root, args.output_dir)


if __name__ == "__main__":
    main()
