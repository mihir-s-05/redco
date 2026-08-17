"""Cheap adaptive-routing falsification test for the ReDCO-Routing hypothesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from redco.algo.branching import leave_one_out_advantages
from redco.contracts import canonical_json


class RouteAction(StrEnum):
    ARTIFACT_ONLY = "artifact_only"
    CONTEXT_ONLY = "context_only"
    BOTH = "both"


class TaskMode(StrEnum):
    ARTIFACT_WITH_CONTEXT_SHORTCUT = "artifact_with_context_shortcut"
    CONTEXT_NEEDED = "context_needed"
    EITHER_SUFFICIENT = "either_sufficient"
    BOTH_NEEDED = "both_needed"


class Method(StrEnum):
    TRAJECTORY = "trajectory_route_loo"
    REDCO_LITE = "redco_lite_route_branching"
    CONTEXT_DROPOUT = "context_dropout"
    CONTEXT_CORRUPTION = "context_corruption"
    TYPED_INTERCHANGE = "typed_interchange"
    FIXED_ARTIFACT = "fixed_artifact_only"


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    seeds: int = 64
    updates_per_mode: int = 160
    samples_per_update: int = 3
    learning_rate: float = 0.25
    dropout_probability: float = 0.5
    corruption_probability: float = 0.5
    interchange_penalty: float = 1.0
    minimum_typed_gain_over_simple_baseline: float = 0.02
    maximum_normal_reward_regression: float = 0.02

    def __post_init__(self) -> None:
        if self.seeds < 2 or self.updates_per_mode < 1 or self.samples_per_update < 2:
            raise ValueError("probe counts are too small")
        if self.learning_rate <= 0.0 or self.interchange_penalty < 0.0:
            raise ValueError("learning parameters are invalid")
        probabilities = (self.dropout_probability, self.corruption_probability)
        if any(not 0.0 < value < 1.0 for value in probabilities):
            raise ValueError("augmentation probabilities must lie strictly between zero and one")


@dataclass(frozen=True, slots=True)
class PolicyResult:
    method: Method
    seed: int
    normal_reward: float
    ambient_failure_reward: float
    robust_route_accuracy: float
    logical_reward_calls: int
    probabilities: dict[str, dict[str, float]]


def _normal_reward(mode: TaskMode, action: RouteAction) -> float:
    if mode is TaskMode.ARTIFACT_WITH_CONTEXT_SHORTCUT:
        return 1.0
    if mode is TaskMode.CONTEXT_NEEDED:
        return float(action is not RouteAction.ARTIFACT_ONLY)
    if mode is TaskMode.EITHER_SUFFICIENT:
        return 1.0
    if mode is TaskMode.BOTH_NEEDED:
        return float(action is RouteAction.BOTH)
    raise ValueError(f"unsupported task mode: {mode}")


def _ambient_failure_reward(mode: TaskMode, action: RouteAction) -> float:
    if mode is TaskMode.ARTIFACT_WITH_CONTEXT_SHORTCUT:
        return float(action is RouteAction.ARTIFACT_ONLY)
    return _normal_reward(mode, action)


def _context_dropout_reward(mode: TaskMode, action: RouteAction) -> float:
    if mode in {
        TaskMode.ARTIFACT_WITH_CONTEXT_SHORTCUT,
        TaskMode.EITHER_SUFFICIENT,
    }:
        return float(action is not RouteAction.CONTEXT_ONLY)
    return 0.0


def _context_corruption_reward(mode: TaskMode, action: RouteAction) -> float:
    if mode in {
        TaskMode.ARTIFACT_WITH_CONTEXT_SHORTCUT,
        TaskMode.EITHER_SUFFICIENT,
    }:
        return float(action is RouteAction.ARTIFACT_ONLY)
    return 0.0


def _softmax(logits: list[float]) -> tuple[float, ...]:
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = math.fsum(weights)
    return tuple(value / total for value in weights)


def _sample_action(probabilities: tuple[float, ...], rng: random.Random) -> RouteAction:
    draw = rng.random()
    cumulative = 0.0
    actions = tuple(RouteAction)
    for action, probability in zip(actions, probabilities, strict=True):
        cumulative += probability
        if draw <= cumulative:
            return action
    return actions[-1]


def _reward_for_training(
    method: Method,
    mode: TaskMode,
    action: RouteAction,
    augmentation_rng: random.Random,
    config: ProbeConfig,
) -> tuple[float, int]:
    normal = _normal_reward(mode, action)
    if method in {Method.TRAJECTORY, Method.REDCO_LITE}:
        return normal, 1
    if method is Method.CONTEXT_DROPOUT:
        if augmentation_rng.random() < config.dropout_probability:
            return _context_dropout_reward(mode, action), 1
        return normal, 1
    if method is Method.CONTEXT_CORRUPTION:
        if augmentation_rng.random() < config.corruption_probability:
            return _context_corruption_reward(mode, action), 1
        return normal, 1
    if method is Method.TYPED_INTERCHANGE:
        shifted = _ambient_failure_reward(mode, action)
        fragile_context_effect = max(0.0, normal - shifted)
        return normal - config.interchange_penalty * fragile_context_effect, 2
    raise ValueError(f"method does not train: {method}")


def _update(
    logits: list[float],
    actions: tuple[RouteAction, ...],
    rewards: tuple[float, ...],
    learning_rate: float,
) -> None:
    advantages = leave_one_out_advantages(rewards)
    probabilities = _softmax(logits)
    action_order = tuple(RouteAction)
    scale = learning_rate / len(actions)
    for selected, advantage in zip(actions, advantages, strict=True):
        selected_index = action_order.index(selected)
        for index, probability in enumerate(probabilities):
            score = float(index == selected_index) - probability
            logits[index] += scale * advantage * score


def _expected_reward(
    policy: dict[TaskMode, list[float]],
    reward: object,
) -> float:
    if not callable(reward):
        raise TypeError("reward must be callable")
    values: list[float] = []
    for mode in TaskMode:
        probabilities = _softmax(policy[mode])
        values.append(
            math.fsum(
                probability * reward(mode, action)
                for action, probability in zip(RouteAction, probabilities, strict=True)
            )
        )
    return math.fsum(values) / len(values)


def _robust_route_accuracy(policy: dict[TaskMode, list[float]]) -> float:
    acceptable = {
        TaskMode.ARTIFACT_WITH_CONTEXT_SHORTCUT: {RouteAction.ARTIFACT_ONLY},
        TaskMode.CONTEXT_NEEDED: {RouteAction.CONTEXT_ONLY, RouteAction.BOTH},
        TaskMode.EITHER_SUFFICIENT: set(RouteAction),
        TaskMode.BOTH_NEEDED: {RouteAction.BOTH},
    }
    values = []
    for mode in TaskMode:
        probabilities = _softmax(policy[mode])
        values.append(
            math.fsum(
                probability
                for action, probability in zip(RouteAction, probabilities, strict=True)
                if action in acceptable[mode]
            )
        )
    return math.fsum(values) / len(values)


def _policy_probabilities(
    policy: dict[TaskMode, list[float]],
) -> dict[str, dict[str, float]]:
    return {
        mode.value: {
            action.value: probability
            for action, probability in zip(
                RouteAction,
                _softmax(policy[mode]),
                strict=True,
            )
        }
        for mode in TaskMode
    }


def train(seed: int, method: Method, config: ProbeConfig) -> PolicyResult:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    policy = {mode: [0.0] * len(RouteAction) for mode in TaskMode}
    if method is Method.FIXED_ARTIFACT:
        policy = {
            mode: [30.0 if action is RouteAction.ARTIFACT_ONLY else -30.0 for action in RouteAction]
            for mode in TaskMode
        }
        calls = 0
    else:
        schedule = [mode for mode in TaskMode for _ in range(config.updates_per_mode)]
        schedule_rng = random.Random(f"schedule:{seed}")
        schedule_rng.shuffle(schedule)
        action_rng = random.Random(f"actions:{seed}")
        augmentation_rng = random.Random(f"augmentation:{seed}:{method.value}")
        calls = 0
        for mode in schedule:
            probabilities = _softmax(policy[mode])
            actions = tuple(
                _sample_action(probabilities, action_rng) for _ in range(config.samples_per_update)
            )
            reward_records = tuple(
                _reward_for_training(method, mode, action, augmentation_rng, config)
                for action in actions
            )
            rewards = tuple(record[0] for record in reward_records)
            calls += sum(record[1] for record in reward_records)
            _update(policy[mode], actions, rewards, config.learning_rate)

    return PolicyResult(
        method=method,
        seed=seed,
        normal_reward=_expected_reward(policy, _normal_reward),
        ambient_failure_reward=_expected_reward(policy, _ambient_failure_reward),
        robust_route_accuracy=_robust_route_accuracy(policy),
        logical_reward_calls=calls,
        probabilities=_policy_probabilities(policy),
    )


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_stdev(values: list[float]) -> float:
    mean = _mean(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))


def build_report(config: ProbeConfig | None = None) -> dict[str, object]:
    if config is None:
        config = ProbeConfig()
    results = [train(seed, method, config) for seed in range(config.seeds) for method in Method]
    summaries: dict[str, dict[str, float | int]] = {}
    for method in Method:
        rows = [result for result in results if result.method is method]
        summaries[method.value] = {
            "ambient_failure_reward_mean": _mean([row.ambient_failure_reward for row in rows]),
            "logical_reward_calls_per_seed": rows[0].logical_reward_calls,
            "normal_reward_mean": _mean([row.normal_reward for row in rows]),
            "robust_route_accuracy_mean": _mean([row.robust_route_accuracy for row in rows]),
        }

    trajectory = summaries[Method.TRAJECTORY.value]
    typed = summaries[Method.TYPED_INTERCHANGE.value]
    simple_names = (Method.CONTEXT_DROPOUT.value, Method.CONTEXT_CORRUPTION.value)
    best_simple_name = max(
        simple_names,
        key=lambda name: summaries[name]["ambient_failure_reward_mean"],
    )
    best_simple_shift = float(summaries[best_simple_name]["ambient_failure_reward_mean"])
    typed_gain = float(typed["ambient_failure_reward_mean"]) - best_simple_shift
    paired_typed_minus_simple = [
        next(
            row.ambient_failure_reward
            for row in results
            if row.seed == seed and row.method is Method.TYPED_INTERCHANGE
        )
        - next(
            row.ambient_failure_reward
            for row in results
            if row.seed == seed and row.method.value == best_simple_name
        )
        for seed in range(config.seeds)
    ]
    paired_stdev = _sample_stdev(paired_typed_minus_simple)
    paired_ci_half_width = 1.96 * paired_stdev / math.sqrt(config.seeds)
    paired_ci_lower = typed_gain - paired_ci_half_width
    paired_ci_upper = typed_gain + paired_ci_half_width
    typed_normal_regression = float(trajectory["normal_reward_mean"]) - float(
        typed["normal_reward_mean"]
    )
    single_decision_equivalence = all(
        left.normal_reward == right.normal_reward
        and left.ambient_failure_reward == right.ambient_failure_reward
        and left.probabilities == right.probabilities
        for left, right in zip(
            (row for row in results if row.method is Method.TRAJECTORY),
            (row for row in results if row.method is Method.REDCO_LITE),
            strict=True,
        )
    )
    go_to_llm = (
        paired_ci_lower >= config.minimum_typed_gain_over_simple_baseline
        and typed_normal_regression <= config.maximum_normal_reward_regression
    )
    payload: dict[str, object] = {
        "config": asdict(config),
        "decision": {
            "go_to_llm": go_to_llm,
            "reason": (
                "typed interchange clears the predeclared robustness gain"
                if go_to_llm
                else (
                    "typed interchange does not beat the strongest simple "
                    "augmentation by the predeclared margin"
                )
            ),
        },
        "environment": {
            "actions": [action.value for action in RouteAction],
            "ambient_failure": (
                "corrupt the shortcut only in artifact_with_context_shortcut tasks; "
                "genuinely useful context remains available"
            ),
            "modes": [mode.value for mode in TaskMode],
        },
        "paired_findings": {
            "best_simple_augmentation": best_simple_name,
            "paired_gain_95pct_normal_ci": [paired_ci_lower, paired_ci_upper],
            "paired_gain_by_seed": paired_typed_minus_simple,
            "redco_lite_equals_trajectory_on_one_decision": single_decision_equivalence,
            "typed_gain_over_best_simple_augmentation": typed_gain,
            "typed_normal_reward_regression": typed_normal_regression,
        },
        "seed_results": [
            {
                "ambient_failure_reward": row.ambient_failure_reward,
                "logical_reward_calls": row.logical_reward_calls,
                "method": row.method.value,
                "normal_reward": row.normal_reward,
                "robust_route_accuracy": row.robust_route_accuracy,
                "seed": row.seed,
            }
            for row in results
        ],
        "summaries": summaries,
        "training_backend": "dependency-free tabular softmax policy",
    }
    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
        "schema_version": 1,
    }


def compact_result(
    report: dict[str, object],
    *,
    source_path: str,
    source_raw_sha256: str,
) -> dict[str, object]:
    payload = report["payload"]
    if type(payload) is not dict:
        raise ValueError("routing report has the wrong schema")
    compact_payload = {key: value for key, value in payload.items() if key != "seed_results"}
    compact_payload["source_report_path"] = source_path
    compact_payload["source_report_raw_sha256"] = source_raw_sha256
    return {
        "payload": compact_payload,
        "payload_sha256": hashlib.sha256(canonical_json(compact_payload)).hexdigest(),
        "schema_version": 1,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--compact-output", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.check and (args.output is not None or args.compact_output is not None):
        raise ValueError("--check cannot write outputs")
    if args.compact_output is not None and args.output is None:
        raise ValueError("--compact-output requires --output")
    report = build_report()
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
        if args.compact_output is not None:
            compact = compact_result(
                report,
                source_path=args.output.as_posix(),
                source_raw_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            )
            compact_rendered = (
                json.dumps(compact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            )
            args.compact_output.parent.mkdir(parents=True, exist_ok=True)
            with args.compact_output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(compact_rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
