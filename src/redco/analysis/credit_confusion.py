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

from redco.algo import (
    BranchActionExample,
    PolicyDecision,
    ReDCOTrainerRecord,
    SequenceExample,
    TokenSpan,
    compile_redco_records,
    decision_normalized_loss,
    leave_one_out_advantages,
)

ProbeName = Literal["irrelevant_target", "redundant_target", "lucky_target"]
RewardFunction = Callable[[int, int, int], float]
LearningArm = Literal["trajectory_loo", "redco"]


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
    branch_group_size: int = 7
    branch_episodes_per_step: int = 2
    policy_call_budget: int = 1_152
    learning_rate: float = 0.5
    learning_trials: int = 1_000
    seed: int = 920_280_1
    target_threshold: float = 0.8
    gradient_epsilon: float = 1e-5

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
        if self.gradient_epsilon <= 0 or not math.isfinite(self.gradient_epsilon):
            raise ValueError("gradient_epsilon must be finite and positive")
        trajectory_calls = 2 * self.trajectory_group_size
        redco_calls = self.branch_episodes_per_step * (1 + self.branch_group_size)
        if trajectory_calls != redco_calls:
            raise ValueError("both learning arms must use the same calls per update")
        if self.policy_call_budget < trajectory_calls:
            raise ValueError("policy_call_budget must fund at least one update")

    @property
    def calls_per_update(self) -> int:
        return 2 * self.trajectory_group_size


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


def trajectory_target_moments(
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


def redco_target_moments(
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


def _bernoulli_logprob(action: int, policy_logit: float) -> float:
    if action not in (0, 1):
        raise ValueError("Bernoulli action must be zero or one")
    if action:
        return -math.log1p(math.exp(-policy_logit)) if policy_logit >= 0 else (
            policy_logit - math.log1p(math.exp(policy_logit))
        )
    return -math.log1p(math.exp(policy_logit)) if policy_logit <= 0 else (
        -policy_logit - math.log1p(math.exp(-policy_logit))
    )


def _action_sequence(action: int, behavior_logit: float) -> SequenceExample:
    return SequenceExample(
        token_ids=(action,),
        trainable_mask=(True,),
        behavior_logprobs=(_bernoulli_logprob(action, behavior_logit),),
        env_name="credit-confusion",
    )


def _trajectory_records(
    probe: ConfusionProbe,
    *,
    target_logit: float,
    group_size: int,
    rng: random.Random,
) -> tuple[ReDCOTrainerRecord, ...]:
    target_probability = sigmoid(target_logit)
    actions: list[int] = []
    rewards: list[float] = []
    for _ in range(group_size):
        context = _draw_bernoulli(probe.context_probability, rng)
        target = _draw_bernoulli(target_probability, rng)
        luck = _draw_bernoulli(probe.luck_probability, rng)
        actions.append(target)
        rewards.append(probe.reward(context, target, luck))
    advantages = leave_one_out_advantages(rewards)
    return tuple(
        ReDCOTrainerRecord(
            sequence=_action_sequence(action, target_logit),
            advantages=(advantage,),
            rl_weights=(1.0,),
            decision_unit_normalizer=1.0,
            record_kind="incumbent",
            target_node_id=None,
            branch_index=None,
        )
        for action, advantage in zip(actions, advantages, strict=True)
    )


def _redco_records(
    probe: ConfusionProbe,
    *,
    target_logit: float,
    branch_group_size: int,
    episodes_per_step: int,
    rng: random.Random,
) -> tuple[ReDCOTrainerRecord, ...]:
    target_probability = sigmoid(target_logit)
    records: list[ReDCOTrainerRecord] = []
    for _ in range(episodes_per_step):
        context = _draw_bernoulli(probe.context_probability, rng)
        luck = _draw_bernoulli(probe.luck_probability, rng)
        targets = [_draw_bernoulli(target_probability, rng) for _ in range(branch_group_size)]
        rewards = [probe.reward(context, target, luck) for target in targets]
        incumbent = _action_sequence(targets[0], target_logit)
        branches = tuple(
            BranchActionExample(
                sequence=_action_sequence(target, target_logit),
                action_span=TokenSpan(0, 1),
                reward=reward,
                action_source="original" if index == 0 else "sampled",
            )
            for index, (target, reward) in enumerate(
                zip(targets, rewards, strict=True)
            )
        )
        records.extend(
            compile_redco_records(
                incumbent=incumbent,
                decisions=(PolicyDecision("target", TokenSpan(0, 1)),),
                trajectory_advantage=0.0,
                target_node_id="target",
                branches=branches,
            )
        )
    return tuple(records)


def _loss_gradient(
    records: tuple[ReDCOTrainerRecord, ...],
    *,
    policy_logit: float,
    epsilon: float,
) -> float:
    def loss(candidate: float) -> float:
        current_logprobs = tuple(
            (_bernoulli_logprob(record.sequence.token_ids[0], candidate),)
            for record in records
        )
        return decision_normalized_loss(records, current_logprobs).loss

    return (loss(policy_logit + epsilon) - loss(policy_logit - epsilon)) / (2 * epsilon)


def _learning_trial(
    probe: ConfusionProbe,
    config: DiagnosticConfig,
    *,
    arm: LearningArm,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    target_logit = logit(probe.target_probability)
    calls_per_step = config.calls_per_update
    maximum_steps = config.policy_call_budget // calls_per_step
    threshold_calls: int | None = None
    initial_probability = probe.target_probability
    for step in range(1, maximum_steps + 1):
        records = (
            _trajectory_records(
                probe,
                target_logit=target_logit,
                group_size=config.trajectory_group_size,
                rng=rng,
            )
            if arm == "trajectory_loo"
            else _redco_records(
                probe,
                target_logit=target_logit,
                branch_group_size=config.branch_group_size,
                episodes_per_step=config.branch_episodes_per_step,
                rng=rng,
            )
        )
        loss_gradient = _loss_gradient(
            records,
            policy_logit=target_logit,
            epsilon=config.gradient_epsilon,
        )
        target_logit -= config.learning_rate * loss_gradient
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
    arm_names: tuple[LearningArm, ...] = (
        "trajectory_loo",
        "redco",
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


def _learning_comparison(arms: dict[str, Any]) -> dict[str, float | None]:
    trajectory = arms["trajectory_loo"]
    redco = arms["redco"]
    trajectory_calls = trajectory["mean_calls_to_threshold_among_reached"]
    redco_calls = redco["mean_calls_to_threshold_among_reached"]
    return {
        "mean_final_probability_delta": (
            redco["mean_final_probability"] - trajectory["mean_final_probability"]
        ),
        "mean_absolute_drift_delta": (
            redco["mean_absolute_probability_drift"]
            - trajectory["mean_absolute_probability_drift"]
        ),
        "threshold_reach_rate_delta": (
            redco["threshold_reach_rate"] - trajectory["threshold_reach_rate"]
        ),
        "mean_calls_to_threshold_delta": (
            redco_calls - trajectory_calls
            if redco_calls is not None and trajectory_calls is not None
            else None
        ),
    }


def build_credit_confusion_diagnostic(
    config: DiagnosticConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = DiagnosticConfig()
    probes: dict[str, Any] = {}
    for probe in standard_confusion_probes():
        exact = exact_target_gradient(probe)
        trajectory = trajectory_target_moments(
            probe,
            group_size=config.trajectory_group_size,
        )
        redco = redco_target_moments(
            probe,
            branch_group_size=config.branch_group_size,
            episodes_per_step=config.branch_episodes_per_step,
        )
        learning = learning_diagnostic(probe, config)
        probes[probe.name] = {
            "initial": {
                "context_probability": probe.context_probability,
                "target_probability": probe.target_probability,
                "luck_probability": probe.luck_probability,
            },
            "exact_target_gradient": exact,
            "trajectory_loo_moments": trajectory,
            "redco_moments": redco,
            "mean_bias": {
                "trajectory_loo": trajectory["mean"] - exact,
                "redco": redco["mean"] - exact,
            },
            "variance_ratio_redco_to_trajectory": (
                redco["variance"] / trajectory["variance"]
                if trajectory["variance"]
                else 0.0
            ),
            "learning": learning,
            "learning_comparison_redco_minus_trajectory": _learning_comparison(
                learning
            ),
        }

    irrelevant = probes["irrelevant_target"]
    lucky = probes["lucky_target"]
    redundant = probes["redundant_target"]
    lucky_calls_delta = lucky["learning_comparison_redco_minus_trajectory"][
        "mean_calls_to_threshold_delta"
    ]
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
                probe["trajectory_loo_moments"],
                probe["redco_moments"],
            )
        ),
        "irrelevant_redco_variance_at_most_1pct_trajectory": (
            irrelevant["variance_ratio_redco_to_trajectory"] <= 0.01
        ),
        "lucky_redco_variance_at_most_50pct_trajectory": (
            lucky["variance_ratio_redco_to_trajectory"] <= 0.5
        ),
        "both_relevant_tasks_have_positive_exact_gradient": (
            lucky["exact_target_gradient"] > 0
            and redundant["exact_target_gradient"] > 0
        ),
        "irrelevant_redco_drift_at_most_25pct_trajectory": (
            irrelevant["learning"]["redco"]["mean_absolute_probability_drift"]
            <= 0.25
            * irrelevant["learning"]["trajectory_loo"]["mean_absolute_probability_drift"]
        ),
        "irrelevant_redco_drift_is_zero": (
            irrelevant["learning"]["redco"]["mean_absolute_probability_drift"] == 0.0
        ),
        "both_relevant_redco_policies_reach_mean_threshold": (
            lucky["learning"]["redco"]["mean_final_probability"]
            >= config.target_threshold
            and redundant["learning"]["redco"]["mean_final_probability"]
            >= config.target_threshold
        ),
        "lucky_redco_reaches_threshold_no_later_on_average": (
            lucky_calls_delta is not None and lucky_calls_delta <= 0
        ),
        "redundant_redco_mean_policy_within_1pct_trajectory": (
            redundant["learning_comparison_redco_minus_trajectory"][
                "mean_final_probability_delta"
            ]
            >= -0.01
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 2,
        "analysis": "redco-multi-decision-credit-confusion",
        "label": (
            "Exact estimator moments plus seeded optimizer updates through the "
            "maintained ReDCO loss; this is not a language-model result"
        ),
        "config": {**asdict(config), "calls_per_update": config.calls_per_update},
        "probes": probes,
        "mandatory_checks": mandatory,
        "cpu_gate_passed": all(mandatory.values()),
        "next_experiment_if_passed": (
            "Compare trajectory LOO and ReDCO on a small language-model task "
            "under the same rollout budget."
        ),
        "limitations": [
            "Tabular node logits remove representation sharing and language-model "
            "optimization effects.",
            "The exact comparison tests equal expected gradients and different "
            "variance, not systematic bias in trajectory-level gradients.",
            "Calls-to-threshold is exploratory; model-scale stopping rules must "
            "be fixed before the next experiment.",
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
