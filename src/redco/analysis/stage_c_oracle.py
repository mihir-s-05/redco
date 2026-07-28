"""CPU-only full-information diagnostic for the Stage-C finite probes.

This module deliberately does not implement ReDCO.  It compares an exact
categorical policy-gradient update, computed by enumerating every action, with
a sampled leave-one-out REINFORCE update.  The comparison removes model
inference and replay from the question and measures only the information lost
when the behavior policy does not sample outcome-diverse actions.
"""

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

from redco.analysis.stage_c_postmortem import exact_policy_gradient
from redco.env.tasks.credit_probes import standard_credit_probes


@dataclass(frozen=True, slots=True)
class OracleConfig:
    updates: int = 30
    learning_rate: float = 1.0
    sampled_group_size: int = 4
    trials: int = 1_000
    seed: int = 7202801
    exogenous_seeds: tuple[int, ...] = tuple(range(9000, 9032))

    def __post_init__(self) -> None:
        if self.updates < 1:
            raise ValueError("updates must be positive")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if self.sampled_group_size < 2:
            raise ValueError("sampled_group_size must be at least two")
        if self.trials < 1:
            raise ValueError("trials must be positive")
        if not self.exogenous_seeds:
            raise ValueError("at least one exogenous seed is required")


def softmax(logits: list[float]) -> list[float]:
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def expected_regret(probabilities: list[float], rewards: list[float]) -> float:
    best = max(rewards)
    expected = math.fsum(
        probability * reward
        for probability, reward in zip(probabilities, rewards, strict=True)
    )
    return best - expected


def _sample_index(probabilities: list[float], rng: random.Random) -> int:
    draw = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index
    return len(probabilities) - 1


def sampled_loo_gradient(
    probabilities: list[float],
    rewards: list[float],
    *,
    group_size: int,
    rng: random.Random,
) -> tuple[list[float], bool]:
    """Estimate the categorical logit gradient with all-sample LOO baselines."""
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    samples = [_sample_index(probabilities, rng) for _ in range(group_size)]
    outcomes = [rewards[index] for index in samples]
    total_reward = math.fsum(outcomes)
    gradient = [0.0] * len(probabilities)
    for sample, reward in zip(samples, outcomes, strict=True):
        baseline = (total_reward - reward) / (group_size - 1)
        advantage = reward - baseline
        for action in range(len(probabilities)):
            score = float(action == sample) - probabilities[action]
            gradient[action] += advantage * score / group_size
    return gradient, len(set(outcomes)) > 1


def _finite_states(exogenous_seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for probe in standard_credit_probes():
        for exogenous_seed in exogenous_seeds:
            rewards = [
                float(probe.reward_function(action, exogenous_seed))
                for action in probe.actions
            ]
            if len(set(rewards)) == 1:
                continue
            states.append(
                {
                    "probe": probe.name,
                    "exogenous_seed": exogenous_seed,
                    "actions": list(probe.actions),
                    "rewards": rewards,
                }
            )
    return states


def _run_oracle(
    states: list[dict[str, Any]],
    config: OracleConfig,
) -> dict[str, Any]:
    logits = [[0.0] * len(state["actions"]) for state in states]
    curve: list[float] = []
    for _ in range(config.updates + 1):
        curve.append(
            fmean(
                expected_regret(softmax(state_logits), state["rewards"])
                for state_logits, state in zip(logits, states, strict=True)
            )
        )
        if len(curve) > config.updates:
            break
        for state_logits, state in zip(logits, states, strict=True):
            gradient = exact_policy_gradient(
                softmax(state_logits),
                state["rewards"],
            )
            for index, value in enumerate(gradient):
                state_logits[index] += config.learning_rate * value
    return {
        "mean_regret_curve": curve,
        "initial_mean_regret": curve[0],
        "final_mean_regret": curve[-1],
        "action_evaluations_per_update": sum(
            len(state["actions"]) for state in states
        ),
    }


def _run_sampled_trial(
    states: list[dict[str, Any]],
    config: OracleConfig,
    *,
    seed: int,
) -> tuple[list[float], int]:
    rng = random.Random(seed)
    logits = [[0.0] * len(state["actions"]) for state in states]
    curve: list[float] = []
    informative_groups = 0
    for _ in range(config.updates + 1):
        curve.append(
            fmean(
                expected_regret(softmax(state_logits), state["rewards"])
                for state_logits, state in zip(logits, states, strict=True)
            )
        )
        if len(curve) > config.updates:
            break
        for state_logits, state in zip(logits, states, strict=True):
            gradient, informative = sampled_loo_gradient(
                softmax(state_logits),
                state["rewards"],
                group_size=config.sampled_group_size,
                rng=rng,
            )
            informative_groups += int(informative)
            for index, value in enumerate(gradient):
                state_logits[index] += config.learning_rate * value
    return curve, informative_groups


def run_oracle_diagnostic(config: OracleConfig | None = None) -> dict[str, Any]:
    if config is None:
        config = OracleConfig()
    states = _finite_states(config.exogenous_seeds)
    oracle = _run_oracle(states, config)
    sampled_trials = [
        _run_sampled_trial(states, config, seed=config.seed + trial)
        for trial in range(config.trials)
    ]
    sampled_curves = [curve for curve, _ in sampled_trials]
    final_regrets = [curve[-1] for curve in sampled_curves]
    mean_curve = [
        fmean(curve[step] for curve in sampled_curves)
        for step in range(config.updates + 1)
    ]
    sorted_final = sorted(final_regrets)

    def percentile(probability: float) -> float:
        position = probability * (len(sorted_final) - 1)
        lower = int(position)
        upper = min(lower + 1, len(sorted_final) - 1)
        fraction = position - lower
        return (
            sorted_final[lower] * (1 - fraction)
            + sorted_final[upper] * fraction
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c-exact-action-oracle",
        "label": (
            "full-information exact policy-gradient oracle; this is not ReDCO"
        ),
        "config": asdict(config),
        "states": {
            "action_dependent": len(states),
            "by_probe": {
                probe: sum(state["probe"] == probe for state in states)
                for probe in sorted({state["probe"] for state in states})
            },
        },
        "oracle": oracle,
        "sampled_loo_reinforce": {
            "mean_regret_curve": mean_curve,
            "initial_mean_regret": mean_curve[0],
            "final_mean_regret": fmean(final_regrets),
            "final_regret_interval_90": [
                percentile(0.05),
                percentile(0.95),
            ],
            "mean_informative_group_rate": (
                fmean(informative for _, informative in sampled_trials)
                / (len(states) * config.updates)
            ),
            "action_evaluations_per_update": (
                len(states) * config.sampled_group_size
            ),
        },
        "comparison": {
            "oracle_minus_sampled_final_regret": (
                oracle["final_mean_regret"] - fmean(final_regrets)
            ),
            "oracle_has_lower_final_regret": (
                oracle["final_mean_regret"] < fmean(final_regrets)
            ),
            "interpretation": (
                "Enumeration supplies a deterministic exact gradient. Any advantage "
                "over sampled LOO measures information/variance from action coverage, "
                "not a live ReDCO learning effect."
            ),
        },
    }
    signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--sampled-group-size", type=int, default=4)
    parser.add_argument("--trials", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7202801)
    args = parser.parse_args()
    report = run_oracle_diagnostic(
        OracleConfig(
            updates=args.updates,
            learning_rate=args.learning_rate,
            sampled_group_size=args.sampled_group_size,
            trials=args.trials,
            seed=args.seed,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
