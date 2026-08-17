"""Cost-matched CPU controls for declared-versus-ambient route learning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import time
import tracemalloc
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from redco.algo.branching import leave_one_out_advantages
from redco.analysis.channel_interchange import InterchangeEffects
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
    UNIFORM_CONTEXT_DROPOUT = "uniform_context_dropout"
    UNIFORM_CONTEXT_CORRUPTION = "uniform_context_corruption"
    NONORACLE_RISK_CORRUPTION = "nonoracle_risk_corruption"
    ORACLE_TARGETED_CORRUPTION = "oracle_mode_targeted_corruption"
    ORACLE_SHIFT_PENALTY = "oracle_heldout_shift_penalty"
    FOUR_CELL_MEAN = "four_cell_mean"
    FOUR_CELL_MINIMUM = "four_cell_minimum"
    FOUR_CELL_SOFT_MIN = "four_cell_soft_minimum"
    TYPED_ALLOCATION = "typed_interaction_allocation"
    SHUFFLED_TYPED = "shuffled_typed_allocation_placebo"
    FIXED_ARTIFACT = "fixed_artifact_only"


_FOUR_CELL_METHODS = frozenset(
    {
        Method.FOUR_CELL_MEAN,
        Method.FOUR_CELL_MINIMUM,
        Method.FOUR_CELL_SOFT_MIN,
        Method.TYPED_ALLOCATION,
        Method.SHUFFLED_TYPED,
    }
)
_ORACLE_METHODS = frozenset({Method.ORACLE_TARGETED_CORRUPTION, Method.ORACLE_SHIFT_PENALTY})


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    seeds: int = 64
    updates_per_mode: int = 160
    learning_rate: float = 0.25
    augmentation_probability: float = 0.5
    interchange_penalty: float = 1.0
    soft_minimum_temperature: float = 0.2
    minimum_gain_for_sequential_benchmark: float = 0.02
    maximum_normal_reward_regression: float = 0.02

    def __post_init__(self) -> None:
        if self.seeds < 2 or self.updates_per_mode < 1:
            raise ValueError("probe counts are too small")
        if self.learning_rate <= 0.0 or self.interchange_penalty < 0.0:
            raise ValueError("learning parameters are invalid")
        if not 0.0 < self.augmentation_probability < 1.0:
            raise ValueError("augmentation probability must lie between zero and one")
        if self.soft_minimum_temperature <= 0.0:
            raise ValueError("soft-minimum temperature must be positive")


@dataclass(frozen=True, slots=True)
class PolicyResult:
    method: Method
    seed: int
    updates_per_mode: int
    normal_reward: float
    ambient_failure_reward: float
    robust_route_accuracy: float
    logical_reward_calls: int
    per_mode: dict[str, dict[str, float]]
    diagnostics: dict[str, float]


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


def _normal_cells(mode: TaskMode) -> InterchangeEffects:
    if mode is TaskMode.ARTIFACT_WITH_CONTEXT_SHORTCUT:
        return InterchangeEffects(0.0, 1.0, 1.0, 1.0)
    if mode is TaskMode.CONTEXT_NEEDED:
        return InterchangeEffects(0.0, 0.0, 1.0, 1.0)
    if mode is TaskMode.EITHER_SUFFICIENT:
        return InterchangeEffects(0.0, 1.0, 1.0, 1.0)
    if mode is TaskMode.BOTH_NEEDED:
        return InterchangeEffects(0.0, 0.0, 0.0, 1.0)
    raise ValueError(f"unsupported task mode: {mode}")


def _soft_minimum(values: tuple[float, ...], temperature: float) -> float:
    minimum = min(values)
    shifted = math.fsum(math.exp(-(value - minimum) / temperature) for value in values)
    return minimum - temperature * math.log(shifted / len(values))


def _four_cell_utilities(
    method: Method,
    effects: InterchangeEffects,
    config: ProbeConfig,
) -> tuple[float, float, float]:
    if method is Method.FOUR_CELL_MEAN:
        value = math.fsum(effects.cells) / len(effects.cells)
        return (value, value, value)
    if method is Method.FOUR_CELL_MINIMUM:
        value = min(effects.cells)
        return (value, value, value)
    if method is Method.FOUR_CELL_SOFT_MIN:
        value = _soft_minimum(effects.cells, config.soft_minimum_temperature)
        return (value, value, value)
    typed = (
        effects.artifact_allocation,
        effects.context_allocation,
        effects.total + effects.interaction,
    )
    if method is Method.TYPED_ALLOCATION:
        return typed
    if method is Method.SHUFFLED_TYPED:
        return typed[1], typed[2], typed[0]
    raise ValueError(f"method does not consume four-cell effects: {method}")


def _softmax(logits: list[float]) -> tuple[float, ...]:
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = math.fsum(weights)
    return tuple(value / total for value in weights)


def _training_rewards(
    method: Method,
    mode: TaskMode,
    rng: random.Random,
    config: ProbeConfig,
) -> tuple[tuple[float, ...], int]:
    actions = tuple(RouteAction)
    normal = tuple(_normal_reward(mode, action) for action in actions)
    if method in {Method.TRAJECTORY, Method.REDCO_LITE}:
        return normal, len(actions)
    if method is Method.UNIFORM_CONTEXT_DROPOUT:
        reward = (
            _context_dropout_reward
            if rng.random() < config.augmentation_probability
            else _normal_reward
        )
        return tuple(reward(mode, action) for action in actions), len(actions)
    if method is Method.UNIFORM_CONTEXT_CORRUPTION:
        reward = (
            _context_corruption_reward
            if rng.random() < config.augmentation_probability
            else _normal_reward
        )
        return tuple(reward(mode, action) for action in actions), len(actions)
    if method is Method.NONORACLE_RISK_CORRUPTION:
        # The augmenter sees a noisy state-derived risk proxy, never the mode label.
        alpha, beta = (4.0, 1.0) if mode is TaskMode.ARTIFACT_WITH_CONTEXT_SHORTCUT else (1.0, 4.0)
        risk_observation = rng.betavariate(alpha, beta)
        reward = _context_corruption_reward if risk_observation >= 0.5 else _normal_reward
        return tuple(reward(mode, action) for action in actions), len(actions)
    if method is Method.ORACLE_TARGETED_CORRUPTION:
        targeted = (
            mode is TaskMode.ARTIFACT_WITH_CONTEXT_SHORTCUT
            and rng.random() < config.augmentation_probability
        )
        reward = _context_corruption_reward if targeted else _normal_reward
        return tuple(reward(mode, action) for action in actions), len(actions)
    if method is Method.ORACLE_SHIFT_PENALTY:
        shifted = tuple(_ambient_failure_reward(mode, action) for action in actions)
        return (
            tuple(
                base - config.interchange_penalty * max(0.0, base - failure)
                for base, failure in zip(normal, shifted, strict=True)
            ),
            2 * len(actions),
        )
    if method in _FOUR_CELL_METHODS:
        return _four_cell_utilities(method, _normal_cells(mode), config), 4
    raise ValueError(f"method does not train: {method}")


def _update(logits: list[float], rewards: tuple[float, ...], learning_rate: float) -> None:
    advantages = leave_one_out_advantages(rewards)
    probabilities = _softmax(logits)
    scale = learning_rate / len(RouteAction)
    for selected_index, advantage in enumerate(advantages):
        for index, probability in enumerate(probabilities):
            score = float(index == selected_index) - probability
            logits[index] += scale * advantage * score


def _entropy(probabilities: tuple[float, ...]) -> float:
    return -math.fsum(value * math.log(value) for value in probabilities if value > 0.0)


def _per_mode_metrics(policy: dict[TaskMode, list[float]]) -> dict[str, dict[str, float]]:
    acceptable = {
        TaskMode.ARTIFACT_WITH_CONTEXT_SHORTCUT: {RouteAction.ARTIFACT_ONLY},
        TaskMode.CONTEXT_NEEDED: {RouteAction.CONTEXT_ONLY, RouteAction.BOTH},
        TaskMode.EITHER_SUFFICIENT: set(RouteAction),
        TaskMode.BOTH_NEEDED: {RouteAction.BOTH},
    }
    records: dict[str, dict[str, float]] = {}
    for mode in TaskMode:
        probabilities = _softmax(policy[mode])
        normal = math.fsum(
            probability * _normal_reward(mode, action)
            for action, probability in zip(RouteAction, probabilities, strict=True)
        )
        shifted = math.fsum(
            probability * _ambient_failure_reward(mode, action)
            for action, probability in zip(RouteAction, probabilities, strict=True)
        )
        route_accuracy = math.fsum(
            probability
            for action, probability in zip(RouteAction, probabilities, strict=True)
            if action in acceptable[mode]
        )
        records[mode.value] = {
            **{
                f"probability_{action.value}": probability
                for action, probability in zip(RouteAction, probabilities, strict=True)
            },
            "ambient_failure_reward": shifted,
            "entropy": _entropy(probabilities),
            "normal_reward": normal,
            "oracle_regret": 1.0 - shifted,
            "robust_route_accuracy": route_accuracy,
        }
    return records


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def train(
    seed: int,
    method: Method,
    config: ProbeConfig,
    *,
    updates_per_mode: int | None = None,
) -> PolicyResult:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    updates = config.updates_per_mode if updates_per_mode is None else updates_per_mode
    if updates < 1:
        raise ValueError("updates_per_mode must be positive")
    policy = {mode: [0.0] * len(RouteAction) for mode in TaskMode}
    if method is Method.FIXED_ARTIFACT:
        policy = {
            mode: [30.0 if action is RouteAction.ARTIFACT_ONLY else -30.0 for action in RouteAction]
            for mode in TaskMode
        }
        logical_calls = groups = all_equal = zero_advantage = 0
    else:
        schedule = [mode for mode in TaskMode for _ in range(updates)]
        random.Random(f"schedule:{seed}:{updates}").shuffle(schedule)
        reward_rng = random.Random(f"reward:{seed}:{method.value}")
        logical_calls = all_equal = zero_advantage = 0
        groups = len(schedule)
        for mode in schedule:
            rewards, calls = _training_rewards(method, mode, reward_rng, config)
            logical_calls += calls
            all_equal += int(len(set(rewards)) == 1)
            zero_advantage += int(all(value == 0.0 for value in leave_one_out_advantages(rewards)))
            _update(policy[mode], rewards, config.learning_rate)

    per_mode = _per_mode_metrics(policy)
    return PolicyResult(
        method=method,
        seed=seed,
        updates_per_mode=updates,
        normal_reward=_mean([record["normal_reward"] for record in per_mode.values()]),
        ambient_failure_reward=_mean(
            [record["ambient_failure_reward"] for record in per_mode.values()]
        ),
        robust_route_accuracy=_mean(
            [record["robust_route_accuracy"] for record in per_mode.values()]
        ),
        logical_reward_calls=logical_calls,
        per_mode=per_mode,
        diagnostics={
            "all_equal_reward_group_fraction": all_equal / groups if groups else 0.0,
            "enumerated_unique_route_group_fraction": 1.0 if groups else 0.0,
            "zero_advantage_update_fraction": zero_advantage / groups if groups else 0.0,
        },
    )


def _sample_stdev(values: list[float]) -> float:
    mean = _mean(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _paired_comparison(
    left: list[PolicyResult],
    right: list[PolicyResult],
) -> dict[str, object]:
    if len(left) != len(right) or any(a.seed != b.seed for a, b in zip(left, right, strict=True)):
        raise ValueError("paired result sets must have identical seeds")
    differences = [
        a.ambient_failure_reward - b.ambient_failure_reward
        for a, b in zip(left, right, strict=True)
    ]
    mean = _mean(differences)
    half_width = 1.96 * _sample_stdev(differences) / math.sqrt(len(differences))
    return {
        "ambient_failure_difference_by_seed": differences,
        "ambient_failure_difference_mean": mean,
        "ambient_failure_difference_normal_95pct_ci": [mean - half_width, mean + half_width],
        "left_calls_per_seed": left[0].logical_reward_calls,
        "left_method": left[0].method.value,
        "left_updates_per_mode": left[0].updates_per_mode,
        "right_calls_per_seed": right[0].logical_reward_calls,
        "right_method": right[0].method.value,
        "right_updates_per_mode": right[0].updates_per_mode,
    }


def _summarize(rows: list[PolicyResult]) -> dict[str, object]:
    per_mode: dict[str, dict[str, float]] = {}
    for mode in TaskMode:
        per_mode[mode.value] = {
            key: _mean([row.per_mode[mode.value][key] for row in rows])
            for key in rows[0].per_mode[mode.value]
        }
    return {
        "ambient_failure_reward_mean": _mean([row.ambient_failure_reward for row in rows]),
        "diagnostics": {
            key: _mean([row.diagnostics[key] for row in rows]) for key in rows[0].diagnostics
        },
        "logical_reward_calls_per_seed": rows[0].logical_reward_calls,
        "normal_reward_mean": _mean([row.normal_reward for row in rows]),
        "per_mode": per_mode,
        "robust_route_accuracy_mean": _mean([row.robust_route_accuracy for row in rows]),
        "updates_per_mode": rows[0].updates_per_mode,
    }


def _summary_float(
    summaries: dict[str, dict[str, object]],
    method: Method,
    key: str,
) -> float:
    value = summaries[method.value][key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"summary {method.value}.{key} is not numeric")
    return float(value)


def _ci_lower(comparison: dict[str, object]) -> float:
    interval = comparison["ambient_failure_difference_normal_95pct_ci"]
    if type(interval) is not list or len(interval) != 2:
        raise TypeError("paired comparison interval has the wrong schema")
    value = interval[0]
    if type(value) not in {int, float}:
        raise TypeError("paired comparison lower bound is not numeric")
    return float(value)


def _git_output(*arguments: str) -> bytes:
    return subprocess.run(("git", *arguments), check=True, capture_output=True).stdout


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance(cli_args: dict[str, object]) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[3]
    diff = _git_output("diff", "--binary", "HEAD", "--", ".", ":(exclude)external/prime-rl")
    return {
        "campaign_seed_range": [0, 63],
        "cli_args": cli_args,
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "git_commit": _git_output("rev-parse", "HEAD").decode("ascii").strip(),
        "python_version": platform.python_version(),
        "source_files": {
            "src/redco/analysis/channel_interchange.py": _sha256_path(
                repository / "src/redco/analysis/channel_interchange.py"
            ),
            "src/redco/analysis/routing_probe.py": _sha256_path(Path(__file__)),
        },
        "untracked_files_in_dirty_diff": False,
        "uv_lock_sha256": _sha256_path(repository / "uv.lock"),
    }


def build_report(
    config: ProbeConfig | None = None,
    *,
    provenance: dict[str, object] | None = None,
    execution_work: dict[str, object] | None = None,
) -> dict[str, object]:
    if config is None:
        config = ProbeConfig()
    standard = {
        method: [train(seed, method, config) for seed in range(config.seeds)] for method in Method
    }
    oracle_half = [
        train(
            seed,
            Method.ORACLE_SHIFT_PENALTY,
            config,
            updates_per_mode=max(1, config.updates_per_mode // 2),
        )
        for seed in range(config.seeds)
    ]
    uniform_double = [
        train(
            seed,
            Method.UNIFORM_CONTEXT_CORRUPTION,
            config,
            updates_per_mode=2 * config.updates_per_mode,
        )
        for seed in range(config.seeds)
    ]
    summaries = {method.value: _summarize(rows) for method, rows in standard.items()}

    typed_rows = standard[Method.TYPED_ALLOCATION]
    eligible_baselines = [
        method
        for method in Method
        if method not in _ORACLE_METHODS
        and method not in {Method.TYPED_ALLOCATION, Method.FIXED_ARTIFACT}
        and standard[method][0].logical_reward_calls
        <= standard[Method.TYPED_ALLOCATION][0].logical_reward_calls
    ]
    best_nonoracle = max(
        eligible_baselines,
        key=lambda method: _summary_float(
            summaries,
            method,
            "ambient_failure_reward_mean",
        ),
    )
    typed_vs_best = _paired_comparison(typed_rows, standard[best_nonoracle])
    typed_vs_shuffled = _paired_comparison(typed_rows, standard[Method.SHUFFLED_TYPED])
    typed_regression = _summary_float(
        summaries,
        Method.TRAJECTORY,
        "normal_reward_mean",
    ) - _summary_float(
        summaries,
        Method.TYPED_ALLOCATION,
        "normal_reward_mean",
    )
    sequential_go = (
        _ci_lower(typed_vs_best) >= config.minimum_gain_for_sequential_benchmark
        and _ci_lower(typed_vs_shuffled) >= config.minimum_gain_for_sequential_benchmark
        and typed_regression <= config.maximum_normal_reward_regression
    )

    payload: dict[str, object] = {
        "config": asdict(config),
        "decision": {
            "build_sequential_heldout_shift_benchmark": sequential_go,
            "reason": (
                "typed allocation clears equal-information and placebo gates"
                if sequential_go
                else "typed allocation does not clear the declared CPU evidence gate"
            ),
        },
        "effect_contract": {
            "four_cell_effects_by_mode": {
                mode.value: _normal_cells(mode).as_dict() for mode in TaskMode
            },
            "typed_route_utility": {
                "artifact_only": "phi_A",
                "both": "total + interaction",
                "context_only": "phi_C",
                "interaction_is_intentionally_reused": True,
                "objective_kind": "deliberately_modified_route_robustness_objective",
                "ordinary_task_advantage_replaced": True,
            },
        },
        "environment": {
            "actions": [action.value for action in RouteAction],
            "ambient_failure": (
                "context shortcut becomes contradictory only in the held-out shortcut mode"
            ),
            "content_rule": (
                "one latent Boolean packet is routed through artifact, context, or both"
            ),
            "modes": [mode.value for mode in TaskMode],
            "route_actions_enumerated_without_replacement": True,
        },
        "execution_work": execution_work
        or {
            "actual_cpu_wall_seconds": None,
            "judge_calls": 0,
            "peak_traced_memory_bytes": None,
            "reexecuted_environment_events": 0,
            "regenerated_policy_tokens": 0,
            "reused_policy_events": 0,
            "scope_note": "dependency-free tabular probe; graph and token work are not applicable",
        },
        "findings": {
            "best_nonoracle_baseline": best_nonoracle.value,
            "legacy_equal_call_controls": {
                "half_oracle_vs_standard_uniform": _paired_comparison(
                    oracle_half,
                    standard[Method.UNIFORM_CONTEXT_CORRUPTION],
                ),
                "standard_oracle_vs_double_uniform": _paired_comparison(
                    standard[Method.ORACLE_SHIFT_PENALTY],
                    uniform_double,
                ),
            },
            "old_typed_name_corrected_to_oracle_shift_penalty": True,
            "single_decision_redco_lite_equals_trajectory": all(
                left.per_mode == right.per_mode
                for left, right in zip(
                    standard[Method.TRAJECTORY],
                    standard[Method.REDCO_LITE],
                    strict=True,
                )
            ),
            "typed_normal_reward_regression": typed_regression,
            "typed_vs_best_nonoracle": typed_vs_best,
            "typed_vs_shuffled_placebo": typed_vs_shuffled,
        },
        "method_disclosures": {
            Method.NONORACLE_RISK_CORRUPTION.value: (
                "uses a noisy observable correlated with fragility, never the planted mode label"
            ),
            Method.ORACLE_SHIFT_PENALTY.value: (
                "the historical typed_interchange condition; directly evaluates the held-out shift"
            ),
            Method.ORACLE_TARGETED_CORRUPTION.value: (
                "uses the planted fragile-mode label and is a privileged upper baseline"
            ),
            Method.UNIFORM_CONTEXT_CORRUPTION.value: (
                "applies corruption independently of task mode"
            ),
        },
        "provenance": provenance,
        "seed_results": [
            {
                "ambient_failure_reward": row.ambient_failure_reward,
                "diagnostics": row.diagnostics,
                "logical_reward_calls": row.logical_reward_calls,
                "method": row.method.value,
                "normal_reward": row.normal_reward,
                "per_mode": row.per_mode,
                "robust_route_accuracy": row.robust_route_accuracy,
                "seed": row.seed,
                "updates_per_mode": row.updates_per_mode,
            }
            for rows in standard.values()
            for row in rows
        ],
        "summaries": summaries,
        "training_backend": "dependency-free tabular softmax policy",
    }
    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
        "schema_version": 2,
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
        "schema_version": 2,
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

    cli_args = {
        "check": args.check,
        "compact_output": args.compact_output.as_posix() if args.compact_output else None,
        "output": args.output.as_posix() if args.output else None,
    }
    tracemalloc.start()
    started = time.perf_counter()
    report = build_report(provenance=_provenance(cli_args))
    wall_seconds = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    payload = report["payload"]
    if type(payload) is not dict:
        raise AssertionError("internal payload is not an object")
    payload["execution_work"] = {
        "actual_cpu_wall_seconds": wall_seconds,
        "judge_calls": 0,
        "peak_traced_memory_bytes": peak_bytes,
        "reexecuted_environment_events": 0,
        "regenerated_policy_tokens": 0,
        "reused_policy_events": 0,
        "scope_note": "dependency-free tabular probe; graph and token work are not applicable",
    }
    report["payload_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
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
            compact_rendered = json.dumps(compact, indent=2, sort_keys=True) + "\n"
            args.compact_output.parent.mkdir(parents=True, exist_ok=True)
            with args.compact_output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(compact_rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
