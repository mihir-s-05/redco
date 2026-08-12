"""Exact multi-decision credit-confusion moments and seeded learning diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

ProbeName = Literal["irrelevant_target", "redundant_target", "lucky_target"]
RewardFunction = Callable[[int, int, int], float]


@dataclass(frozen=True, slots=True)
class ConfusionProbe:
    name: ProbeName
    context_probability: float
    target_probability: float
    luck_probability: float
    reward: RewardFunction


@dataclass(frozen=True, slots=True)
class DiagnosticConfig:
    trajectory_group_size: int = 8
    branch_group_size: int = 11
    branch_episodes_per_step: int = 8
    policy_call_budget: int = 1_152
    learning_rate: float = 0.5
    learning_trials: int = 10_000
    seed: int = 920_280_1
    target_threshold: float = 0.8

    def __post_init__(self) -> None:
        if self.trajectory_group_size < 2:
            raise ValueError("trajectory_group_size must be at least two")
        if self.branch_group_size < 2:
            raise ValueError("branch_group_size must be at least two")
        if self.branch_episodes_per_step < 1:
            raise ValueError("branch_episodes_per_step must be positive")
        if self.policy_call_budget < 1:
            raise ValueError("policy_call_budget must be positive")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if self.learning_trials < 1:
            raise ValueError("learning_trials must be positive")
        if not 0 < self.target_threshold < 1:
            raise ValueError("target_threshold must lie strictly between zero and one")


def standard_confusion_probes() -> tuple[ConfusionProbe, ...]:
    """Return fixed two-decision tasks with known target-node causal structure."""

    def irrelevant(context: int, target: int, luck: int) -> float:
        del target, luck
        return float(context)

    def redundant(context: int, target: int, luck: int) -> float:
        del luck
        return float(bool(context or target))

    def lucky(context: int, target: int, luck: int) -> float:
        del context
        return float(target) + (1.0 if luck else -1.0)

    return (
        ConfusionProbe("irrelevant_target", 0.35, 0.5, 0.0, irrelevant),
        ConfusionProbe("redundant_target", 0.5, 0.2, 0.0, redundant),
        ConfusionProbe("lucky_target", 0.5, 0.2, 0.5, lucky),
    )


def sigmoid(logit: float) -> float:
    if logit >= 0:
        inverse = math.exp(-logit)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(logit)
    return exponent / (1.0 + exponent)


def logit(probability: float) -> float:
    if not 0 < probability < 1:
        raise ValueError("probability must lie strictly between zero and one")
    return math.log(probability / (1.0 - probability))


def _bernoulli_probability(value: int, probability: float) -> float:
    return probability if value else 1.0 - probability


def _categories(
    probe: ConfusionProbe,
    *,
    target_probability: float | None = None,
) -> tuple[tuple[float, int, float, int, int], ...]:
    target_mass = probe.target_probability if target_probability is None else target_probability
    values: list[tuple[float, int, float, int, int]] = []
    luck_values = (0, 1) if 0 < probe.luck_probability < 1 else (0,)
    for context in (0, 1):
        for target in (0, 1):
            for luck in luck_values:
                probability = (
                    _bernoulli_probability(context, probe.context_probability)
                    * _bernoulli_probability(target, target_mass)
                    * _bernoulli_probability(luck, probe.luck_probability)
                )
                if probability == 0:
                    continue
                values.append(
                    (
                        probability,
                        target,
                        probe.reward(context, target, luck),
                        context,
                        luck,
                    )
                )
    return tuple(values)


def _count_vectors(total: int, categories: int) -> Iterator[tuple[int, ...]]:
    if categories == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in _count_vectors(total - first, categories - 1):
            yield (first, *tail)


def _multinomial_probability(
    counts: Sequence[int],
    probabilities: Sequence[float],
) -> float:
    total = sum(counts)
    coefficient = math.factorial(total)
    for count in counts:
        coefficient //= math.factorial(count)
    return coefficient * math.prod(
        probability**count for count, probability in zip(counts, probabilities, strict=True)
    )


def _loo_gradient_from_counts(
    counts: Sequence[int],
    rewards: Sequence[float],
    scores: Sequence[float],
) -> float:
    group_size = sum(counts)
    if group_size < 2:
        raise ValueError("LOO requires at least two records")
    total_reward = math.fsum(count * reward for count, reward in zip(counts, rewards, strict=True))
    return math.fsum(
        count * (reward - (total_reward - reward) / (group_size - 1)) * score / group_size
        for count, reward, score in zip(counts, rewards, scores, strict=True)
    )


def _moments(weighted_values: Sequence[tuple[float, float]]) -> dict[str, float]:
    probability = math.fsum(weight for weight, _ in weighted_values)
    mean = math.fsum(weight * value for weight, value in weighted_values)
    second = math.fsum(weight * value * value for weight, value in weighted_values)
    return {
        "enumerated_probability": probability,
        "mean": mean,
        "variance": max(0.0, second - mean * mean),
    }


def broadcast_target_moments(
    probe: ConfusionProbe,
    *,
    group_size: int,
    target_probability: float | None = None,
) -> dict[str, float]:
    """Enumerate trajectory-LOO gradient moments at the target node."""
    target_mass = probe.target_probability if target_probability is None else target_probability
    categories = _categories(probe, target_probability=target_mass)
    probabilities = [category[0] for category in categories]
    rewards = [category[2] for category in categories]
    scores = [category[1] - target_mass for category in categories]
    values = []
    for counts in _count_vectors(group_size, len(categories)):
        weight = _multinomial_probability(counts, probabilities)
        values.append(
            (
                weight,
                _loo_gradient_from_counts(counts, rewards, scores),
            )
        )
    return _moments(values)


def branch_target_moments(
    probe: ConfusionProbe,
    *,
    branch_group_size: int,
    episodes_per_step: int,
    target_probability: float | None = None,
) -> dict[str, float]:
    """Enumerate CRN branch-LOO moments, averaged across independent episodes."""
    target_mass = probe.target_probability if target_probability is None else target_probability
    luck_values = (0, 1) if 0 < probe.luck_probability < 1 else (0,)
    values: list[tuple[float, float]] = []
    for context in (0, 1):
        for luck in luck_values:
            world_probability = _bernoulli_probability(
                context, probe.context_probability
            ) * _bernoulli_probability(luck, probe.luck_probability)
            if world_probability == 0:
                continue
            rewards = (
                probe.reward(context, 0, luck),
                probe.reward(context, 1, luck),
            )
            scores = (-target_mass, 1.0 - target_mass)
            for successes in range(branch_group_size + 1):
                counts = (branch_group_size - successes, successes)
                branch_probability = (
                    math.comb(branch_group_size, successes)
                    * target_mass**successes
                    * (1.0 - target_mass) ** (branch_group_size - successes)
                )
                values.append(
                    (
                        world_probability * branch_probability,
                        _loo_gradient_from_counts(counts, rewards, scores),
                    )
                )
    single = _moments(values)
    return {
        "enumerated_probability": single["enumerated_probability"],
        "mean": single["mean"],
        "variance": single["variance"] / episodes_per_step,
        "single_episode_variance": single["variance"],
    }


def exact_target_gradient(
    probe: ConfusionProbe,
    *,
    target_probability: float | None = None,
) -> float:
    target_mass = probe.target_probability if target_probability is None else target_probability
    return math.fsum(
        probability * reward * (target - target_mass)
        for probability, target, reward, _, _ in _categories(
            probe,
            target_probability=target_mass,
        )
    )


def _draw_bernoulli(probability: float, rng: random.Random) -> int:
    return int(rng.random() < probability)


def _sample_broadcast_gradient(
    probe: ConfusionProbe,
    *,
    target_probability: float,
    group_size: int,
    rng: random.Random,
) -> float:
    rewards: list[float] = []
    scores: list[float] = []
    for _ in range(group_size):
        context = _draw_bernoulli(probe.context_probability, rng)
        target = _draw_bernoulli(target_probability, rng)
        luck = _draw_bernoulli(probe.luck_probability, rng)
        rewards.append(probe.reward(context, target, luck))
        scores.append(target - target_probability)
    return _loo_gradient_from_counts(
        tuple(1 for _ in rewards),
        rewards,
        scores,
    )


def _sample_branch_gradient(
    probe: ConfusionProbe,
    *,
    target_probability: float,
    branch_group_size: int,
    episodes_per_step: int,
    rng: random.Random,
) -> float:
    gradients: list[float] = []
    for _ in range(episodes_per_step):
        context = _draw_bernoulli(probe.context_probability, rng)
        luck = _draw_bernoulli(probe.luck_probability, rng)
        targets = [_draw_bernoulli(target_probability, rng) for _ in range(branch_group_size)]
        rewards = [probe.reward(context, target, luck) for target in targets]
        gradients.append(
            _loo_gradient_from_counts(
                tuple(1 for _ in targets),
                rewards,
                [target - target_probability for target in targets],
            )
        )
    return fmean(gradients)


def _learning_trial(
    probe: ConfusionProbe,
    config: DiagnosticConfig,
    *,
    arm: Literal["broadcast", "node_loo"],
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    target_logit = logit(probe.target_probability)
    calls_per_step = (
        2 * config.trajectory_group_size
        if arm == "broadcast"
        else config.branch_episodes_per_step * (1 + config.branch_group_size)
    )
    maximum_steps = config.policy_call_budget // calls_per_step
    threshold_calls: int | None = None
    initial_probability = probe.target_probability
    for step in range(1, maximum_steps + 1):
        target_probability = sigmoid(target_logit)
        gradient = (
            _sample_broadcast_gradient(
                probe,
                target_probability=target_probability,
                group_size=config.trajectory_group_size,
                rng=rng,
            )
            if arm == "broadcast"
            else _sample_branch_gradient(
                probe,
                target_probability=target_probability,
                branch_group_size=config.branch_group_size,
                episodes_per_step=config.branch_episodes_per_step,
                rng=rng,
            )
        )
        target_logit += config.learning_rate * gradient
        if threshold_calls is None and sigmoid(target_logit) >= config.target_threshold:
            threshold_calls = step * calls_per_step
    final_probability = sigmoid(target_logit)
    return {
        "initial_probability": initial_probability,
        "final_probability": final_probability,
        "absolute_probability_drift": abs(final_probability - initial_probability),
        "threshold_calls": threshold_calls,
        "reached_threshold": threshold_calls is not None,
        "steps": maximum_steps,
        "used_policy_calls": maximum_steps * calls_per_step,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def learning_diagnostic(
    probe: ConfusionProbe,
    config: DiagnosticConfig,
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    arm_names: tuple[Literal["broadcast", "node_loo"], ...] = (
        "broadcast",
        "node_loo",
    )
    for arm_index, arm in enumerate(arm_names):
        trials = [
            _learning_trial(
                probe,
                config,
                arm=arm,
                seed=config.seed + arm_index * config.learning_trials + trial,
            )
            for trial in range(config.learning_trials)
        ]
        final = [float(trial["final_probability"]) for trial in trials]
        drift = [float(trial["absolute_probability_drift"]) for trial in trials]
        threshold = [
            int(trial["threshold_calls"])
            for trial in trials
            if trial["threshold_calls"] is not None
        ]
        arms[arm] = {
            "trials": len(trials),
            "steps": trials[0]["steps"],
            "used_policy_calls": trials[0]["used_policy_calls"],
            "mean_final_probability": fmean(final),
            "final_probability_interval_90": [
                _percentile(final, 0.05),
                _percentile(final, 0.95),
            ],
            "mean_absolute_probability_drift": fmean(drift),
            "absolute_probability_drift_p90": _percentile(drift, 0.9),
            "threshold_reach_rate": len(threshold) / len(trials),
            "mean_calls_to_threshold_among_reached": (fmean(threshold) if threshold else None),
        }
    return arms


def build_credit_confusion_diagnostic(
    config: DiagnosticConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = DiagnosticConfig()
    probes: dict[str, Any] = {}
    for probe in standard_confusion_probes():
        exact = exact_target_gradient(probe)
        broadcast = broadcast_target_moments(
            probe,
            group_size=config.trajectory_group_size,
        )
        node = branch_target_moments(
            probe,
            branch_group_size=config.branch_group_size,
            episodes_per_step=config.branch_episodes_per_step,
        )
        probes[probe.name] = {
            "initial": {
                "context_probability": probe.context_probability,
                "target_probability": probe.target_probability,
                "luck_probability": probe.luck_probability,
            },
            "exact_target_gradient": exact,
            "broadcast_moments": broadcast,
            "node_loo_moments": node,
            "mean_bias": {
                "broadcast": broadcast["mean"] - exact,
                "node_loo": node["mean"] - exact,
            },
            "variance_ratio_node_to_broadcast": (
                node["variance"] / broadcast["variance"] if broadcast["variance"] else 0.0
            ),
            "learning": learning_diagnostic(probe, config),
        }

    irrelevant = probes["irrelevant_target"]
    lucky = probes["lucky_target"]
    redundant = probes["redundant_target"]
    mandatory = {
        "all_exact_mean_biases_at_most_1e_12": all(
            abs(float(value)) <= 1e-12
            for probe in probes.values()
            for value in probe["mean_bias"].values()
        ),
        "all_enumerated_probabilities_sum_to_one": all(
            math.isclose(
                float(moments["enumerated_probability"]),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for probe in probes.values()
            for moments in (
                probe["broadcast_moments"],
                probe["node_loo_moments"],
            )
        ),
        "irrelevant_node_loo_variance_at_most_1pct_broadcast": (
            irrelevant["variance_ratio_node_to_broadcast"] <= 0.01
        ),
        "lucky_node_loo_variance_at_most_50pct_broadcast": (
            lucky["variance_ratio_node_to_broadcast"] <= 0.5
        ),
        "redundant_node_loo_variance_at_most_75pct_broadcast": (
            redundant["variance_ratio_node_to_broadcast"] <= 0.75
        ),
        "irrelevant_node_loo_drift_at_most_25pct_broadcast": (
            irrelevant["learning"]["node_loo"]["mean_absolute_probability_drift"]
            <= 0.25 * irrelevant["learning"]["broadcast"]["mean_absolute_probability_drift"]
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "redco-multi-decision-credit-confusion",
        "label": (
            "Exact estimator moments plus seeded tabular learning; this is a "
            "power/design gate, not a language-model result"
        ),
        "config": asdict(config),
        "probes": probes,
        "mandatory_checks": mandatory,
        "cpu_gate_passed": all(mandatory.values()),
        "gpu_authorization_if_passed": (
            "A separate saturation-safe GPU preregistration may be written; "
            "this CPU result does not itself authorize provisioning."
        ),
        "limitations": [
            "Tabular node logits remove representation sharing and language-model "
            "optimization effects.",
            "The exact comparison tests equal expected gradients and different "
            "variance, not systematic bias in broadcast policy gradients.",
            "Calls-to-threshold is exploratory in this CPU gate; the GPU threshold "
            "and saturation switch must be frozen before model calls.",
        ],
    }
    signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_credit_confusion_diagnostic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
