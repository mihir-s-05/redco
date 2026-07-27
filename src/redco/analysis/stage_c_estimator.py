"""Exact finite-policy audit for the Stage-C all-branch LOO estimator."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Final

from redco.algo.branching import leave_one_out_advantages
from redco.contracts import canonical_json
from redco.env.tasks.credit_probes import FiniteCreditProbe

BRANCH_COUNT: Final = 4


@dataclass(frozen=True, slots=True)
class ProbeEstimatorResult:
    probe_name: str
    samples: int
    action_counts: tuple[int, ...]
    true_q_values: tuple[float, ...]
    estimated_q_values: tuple[float, ...]
    true_advantages: tuple[float, ...]
    estimated_advantages: tuple[float, ...]
    true_policy_gradient: tuple[float, ...]
    estimated_policy_gradient: tuple[float, ...]
    gradient_cosine: float
    advantage_rank_correlation: float
    sign_accuracy_above_noise: float
    sign_comparisons: int
    advantage_rmse: float


def _seed(*parts: object) -> int:
    payload = canonical_json(list(parts))
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _softmax(logits: tuple[float, ...]) -> tuple[float, ...]:
    maximum = max(logits)
    exponentials = tuple(math.exp(value - maximum) for value in logits)
    total = math.fsum(exponentials)
    return tuple(value / total for value in exponentials)


def _policy_probabilities(probe: FiniteCreditProbe) -> tuple[float, ...]:
    """Return a fixed, nonuniform policy without consulting probe rewards."""
    logits = tuple(
        ((_seed("stage-c-policy", probe.name, action) % 2001) - 1000) / 2000.0
        for action in probe.actions
    )
    return _softmax(logits)


def _sample_index(probabilities: tuple[float, ...], rng: random.Random) -> int:
    draw = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index
    return len(probabilities) - 1


def _center(values: tuple[float, ...], weights: tuple[float, ...]) -> tuple[float, ...]:
    mean = math.fsum(
        value * weight for value, weight in zip(values, weights, strict=True)
    )
    return tuple(value - mean for value in values)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    numerator = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    ranks = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        average = (start + 1 + stop) / 2.0
        for index in ordered[start:stop]:
            ranks[index] = average
        start = stop
    return tuple(ranks)


def _pearson(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    centered_left = tuple(value - left_mean for value in left)
    centered_right = tuple(value - right_mean for value in right)
    return _cosine(centered_left, centered_right)


def _rank_correlation(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _sign_accuracy(
    *,
    truth: tuple[float, ...],
    estimate: tuple[float, ...],
    noise_floor: float,
) -> tuple[float, int]:
    outcomes: list[bool] = []
    for left, right in combinations(range(len(truth)), 2):
        true_delta = truth[left] - truth[right]
        if abs(true_delta) <= noise_floor:
            continue
        estimated_delta = estimate[left] - estimate[right]
        outcomes.append(
            (true_delta > 0 and estimated_delta > 0)
            or (true_delta < 0 and estimated_delta < 0)
        )
    if not outcomes:
        return 1.0, 0
    return sum(outcomes) / len(outcomes), len(outcomes)


def audit_probe_estimator(
    probe: FiniteCreditProbe,
    *,
    samples: int,
    master_seed: int,
    exogenous_seed_count: int,
    noise_floor: float,
) -> ProbeEstimatorResult:
    """Compare sampled n=4 LOO credit with exact enumerated policy quantities."""
    if samples < 1:
        raise ValueError("samples must be positive")
    if exogenous_seed_count < 1:
        raise ValueError("exogenous_seed_count must be positive")
    if noise_floor < 0:
        raise ValueError("noise_floor must be non-negative")

    probabilities = _policy_probabilities(probe)
    exogenous_seeds = tuple(range(exogenous_seed_count))
    true_q = tuple(probe.q_values(exogenous_seeds)[action] for action in probe.actions)
    true_advantage = _center(true_q, probabilities)
    true_gradient = tuple(
        probability * advantage
        for probability, advantage in zip(
            probabilities,
            true_advantage,
            strict=True,
        )
    )

    reward_totals = [0.0] * len(probe.actions)
    action_counts = [0] * len(probe.actions)
    gradient_total = [0.0] * len(probe.actions)
    for sample_index in range(samples):
        rng = random.Random(_seed(master_seed, probe.name, sample_index))
        action_indices = tuple(
            _sample_index(probabilities, rng) for _ in range(BRANCH_COUNT)
        )
        exogenous_seed = exogenous_seeds[
            _seed(master_seed, probe.name, "environment", sample_index)
            % exogenous_seed_count
        ]
        rewards = tuple(
            probe.reward_function(probe.actions[index], exogenous_seed)
            for index in action_indices
        )
        advantages = leave_one_out_advantages(rewards)
        for action_index, reward in zip(action_indices, rewards, strict=True):
            action_counts[action_index] += 1
            reward_totals[action_index] += reward
        for action_index, advantage in zip(
            action_indices,
            advantages,
            strict=True,
        ):
            for logit_index, probability in enumerate(probabilities):
                score = float(action_index == logit_index) - probability
                gradient_total[logit_index] += advantage * score / BRANCH_COUNT

    estimated_q = tuple(
        reward_totals[index] / count if count else 0.0
        for index, count in enumerate(action_counts)
    )
    estimated_advantage = _center(estimated_q, probabilities)
    estimated_gradient = tuple(value / samples for value in gradient_total)
    sign_accuracy, sign_comparisons = _sign_accuracy(
        truth=true_q,
        estimate=estimated_q,
        noise_floor=noise_floor,
    )
    return ProbeEstimatorResult(
        probe_name=probe.name,
        samples=samples,
        action_counts=tuple(action_counts),
        true_q_values=true_q,
        estimated_q_values=estimated_q,
        true_advantages=true_advantage,
        estimated_advantages=estimated_advantage,
        true_policy_gradient=true_gradient,
        estimated_policy_gradient=estimated_gradient,
        gradient_cosine=_cosine(true_gradient, estimated_gradient),
        advantage_rank_correlation=_rank_correlation(
            true_advantage,
            estimated_advantage,
        ),
        sign_accuracy_above_noise=sign_accuracy,
        sign_comparisons=sign_comparisons,
        advantage_rmse=math.sqrt(
            math.fsum(
                (truth - estimate) ** 2
                for truth, estimate in zip(
                    true_advantage,
                    estimated_advantage,
                    strict=True,
                )
            )
            / len(true_advantage)
        ),
    )
