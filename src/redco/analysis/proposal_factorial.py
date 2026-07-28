"""Known-propensity proposal diagnostics for a future exploration factor.

This module does not change ReDCO. It validates an importance-weighted
leave-one-out estimator when actions are sampled from a declared mixture
proposal q = (1 - epsilon) * pi + epsilon * r. The target gradient remains the
gradient of pi, and proposal support is guaranteed whenever epsilon < 1.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def mixture_distribution(
    policy: Sequence[float],
    proposal: Sequence[float],
    *,
    epsilon: float,
) -> tuple[float, ...]:
    """Return a normalized policy/proposal mixture with target-policy support."""
    if len(policy) != len(proposal) or not policy:
        raise ValueError("policy and proposal must have equal nonzero length")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must lie in [0, 1)")
    for name, values in (("policy", policy), ("proposal", proposal)):
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"{name} probabilities must be finite and non-negative")
        if not math.isclose(math.fsum(values), 1.0, abs_tol=1e-12):
            raise ValueError(f"{name} probabilities must sum to one")
    return tuple(
        (1.0 - epsilon) * target + epsilon * offered
        for target, offered in zip(policy, proposal, strict=True)
    )


def exact_policy_gradient(
    policy: Sequence[float],
    rewards: Sequence[float],
) -> tuple[float, ...]:
    """Return the exact categorical-logit policy gradient."""
    if len(policy) != len(rewards) or not policy:
        raise ValueError("policy and rewards must have equal nonzero length")
    value = math.fsum(
        probability * reward for probability, reward in zip(policy, rewards, strict=True)
    )
    return tuple(
        probability * (reward - value) for probability, reward in zip(policy, rewards, strict=True)
    )


def importance_weighted_loo_gradient(
    actions: Sequence[int],
    *,
    policy: Sequence[float],
    sampling: Sequence[float],
    rewards: Sequence[float],
) -> tuple[float, ...]:
    """Estimate the target-policy gradient from known-propensity proposal draws.

    Each term uses an unnormalized importance ratio pi(a)/q(a). Its baseline is
    the mean of the other samples' importance-weighted rewards, so it is
    independent of the current action. Self-normalized ratios are intentionally
    not used because they would introduce finite-sample bias.
    """
    if len(actions) < 2:
        raise ValueError("at least two actions are required for LOO")
    if len(policy) != len(sampling) or len(policy) != len(rewards):
        raise ValueError("policy, sampling, and rewards must have equal length")
    action_count = len(policy)
    weighted_rewards: list[float] = []
    ratios: list[float] = []
    for action in actions:
        if not 0 <= action < action_count:
            raise ValueError("action index is out of range")
        if sampling[action] <= 0 and policy[action] > 0:
            raise ValueError("sampling distribution does not cover the policy")
        ratio = policy[action] / sampling[action] if sampling[action] else 0.0
        ratios.append(ratio)
        weighted_rewards.append(ratio * rewards[action])

    total_weighted_reward = math.fsum(weighted_rewards)
    gradient = [0.0] * action_count
    group_size = len(actions)
    for position, action in enumerate(actions):
        baseline = (total_weighted_reward - weighted_rewards[position]) / (group_size - 1)
        scaled_advantage = ratios[position] * (rewards[action] - baseline)
        for logit in range(action_count):
            score = float(logit == action) - policy[logit]
            gradient[logit] += scaled_advantage * score / group_size
    return tuple(gradient)


def exact_estimator_moments(
    *,
    policy: Sequence[float],
    sampling: Sequence[float],
    rewards: Sequence[float],
    group_size: int,
) -> dict[str, Any]:
    """Enumerate all proposal groups and return exact bias and variance."""
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    target = exact_policy_gradient(policy, rewards)
    mean = [0.0] * len(policy)
    second = [0.0] * len(policy)
    informative_probability = 0.0
    total_probability = 0.0
    for actions in itertools.product(range(len(policy)), repeat=group_size):
        probability = math.prod(sampling[action] for action in actions)
        total_probability += probability
        estimate = importance_weighted_loo_gradient(
            actions,
            policy=policy,
            sampling=sampling,
            rewards=rewards,
        )
        informative_probability += probability * (len({rewards[action] for action in actions}) > 1)
        for index, value in enumerate(estimate):
            mean[index] += probability * value
            second[index] += probability * value * value
    variance = tuple(second[index] - mean[index] * mean[index] for index in range(len(policy)))
    bias = tuple(mean[index] - target[index] for index in range(len(policy)))
    return {
        "target_gradient": target,
        "estimator_mean": tuple(mean),
        "bias": bias,
        "maximum_absolute_bias": max(abs(value) for value in bias),
        "coordinate_variance": variance,
        "variance_trace": math.fsum(variance),
        "informative_group_probability": informative_probability,
        "enumerated_probability": total_probability,
    }


def _binary_row(
    *,
    success_mass: float,
    proposal_success_mass: float,
    epsilon: float,
    group_size: int,
    groups_per_step: int,
) -> dict[str, Any]:
    policy = (1.0 - success_mass, success_mass)
    proposal = (1.0 - proposal_success_mass, proposal_success_mass)
    sampling = mixture_distribution(policy, proposal, epsilon=epsilon)
    moments = exact_estimator_moments(
        policy=policy,
        sampling=sampling,
        rewards=(0.0, 1.0),
        group_size=group_size,
    )
    ratios = [policy[index] / sampling[index] if sampling[index] else 0.0 for index in range(2)]
    second_weight_moment = math.fsum(sampling[index] * ratios[index] ** 2 for index in range(2))
    return {
        "target_success_mass": success_mass,
        "proposal_success_mass": proposal_success_mass,
        "epsilon": epsilon,
        "sampling_success_mass": sampling[1],
        "maximum_importance_ratio": max(ratios),
        "importance_effective_sample_fraction": 1.0 / second_weight_moment,
        "group_size": group_size,
        "groups_per_step": groups_per_step,
        "informative_group_probability": moments["informative_group_probability"],
        "expected_informative_groups_per_step": (
            groups_per_step * moments["informative_group_probability"]
        ),
        "target_gradient": moments["target_gradient"],
        "estimator_mean": moments["estimator_mean"],
        "maximum_absolute_bias": moments["maximum_absolute_bias"],
        "variance_trace": moments["variance_trace"],
        "enumerated_probability": moments["enumerated_probability"],
    }


def build_proposal_diagnostic(
    *,
    success_masses: Sequence[float],
    proposal_families: dict[str, float],
    epsilons: Sequence[float],
    group_size: int,
    groups_per_step: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for success_mass in success_masses:
        on_policy = _binary_row(
            success_mass=success_mass,
            proposal_success_mass=success_mass,
            epsilon=0.0,
            group_size=group_size,
            groups_per_step=groups_per_step,
        )
        for family, proposal_mass in proposal_families.items():
            for epsilon in epsilons:
                row = _binary_row(
                    success_mass=success_mass,
                    proposal_success_mass=proposal_mass,
                    epsilon=epsilon,
                    group_size=group_size,
                    groups_per_step=groups_per_step,
                )
                row["proposal_family"] = family
                row["variance_ratio_to_on_policy"] = (
                    row["variance_trace"] / on_policy["variance_trace"]
                    if on_policy["variance_trace"]
                    else None
                )
                rows.append(row)
    maximum_bias = max(row["maximum_absolute_bias"] for row in rows)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c2-known-propensity-proposal-diagnostic",
        "label": (
            "CPU estimator/design diagnostic; proposal sampling is a separate "
            "exploration factor and is not ReDCO"
        ),
        "config": {
            "success_masses": list(success_masses),
            "proposal_families": proposal_families,
            "epsilons": list(epsilons),
            "group_size": group_size,
            "groups_per_step": groups_per_step,
            "rewards": [0.0, 1.0],
        },
        "estimator_contract": {
            "sampling": "q=(1-epsilon)*pi+epsilon*r",
            "target": "categorical policy gradient under pi",
            "ratio": "unnormalized pi(a)/q(a)",
            "baseline": (
                "mean importance-weighted reward of other group members; "
                "independent of the current action"
            ),
            "support": "epsilon<1 guarantees q(a)>0 whenever pi(a)>0",
            "self_normalization": "prohibited in the unbiased diagnostic",
        },
        "rows": rows,
        "mandatory_checks": {
            "all_enumerated_probabilities_sum_to_one": all(
                math.isclose(
                    row["enumerated_probability"],
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for row in rows
            ),
            "maximum_absolute_gradient_bias_at_most_1e_12": (maximum_bias <= 1e-12),
            "all_importance_ratios_finite": all(
                math.isfinite(row["maximum_importance_ratio"]) for row in rows
            ),
        },
        "maximum_absolute_gradient_bias": maximum_bias,
        "future_factorial": {
            "exploration_factor": [
                "shared warm-start on-policy sampling",
                "shared declared mixture proposal with known propensities",
            ],
            "credit_factor": [
                "broadcast trajectory credit",
                "ReDCO full-suffix LOO credit",
                "ReDCO sliced LOO credit",
            ],
            "crossing_rule": (
                "Every exploration level is crossed with every credit level. "
                "No teacher, uniform, forced, or diversity proposal may appear "
                "only in a ReDCO arm."
            ),
            "broadcast_equivalent": (
                "At a proposal exploration level, broadcast rollouts sample the "
                "same target decision from q and use the same known pi/q "
                "correction; ReDCO differs only by counterfactual credit."
            ),
        },
        "limitations": [
            "The binary reduction validates reward-class coverage and estimator "
            "moments, not language-model sequence propensities.",
            "Sequence proposals require exact proposal and target logprobs for "
            "the complete macro-action; tokenwise approximations are a separate "
            "surrogate.",
            "This diagnostic does not authorize another GPU campaign after the "
            "negative Stage-C2 credit result.",
        ],
    }
    signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_proposal_diagnostic(
        success_masses=(0.01, 0.02, 0.05, 0.08143392780108703),
        proposal_families={
            "uniform_over_eight_actions": 0.125,
            "teacher_half_mass_on_success": 0.5,
        },
        epsilons=(0.0, 0.1, 0.25, 0.5),
        group_size=11,
        groups_per_step=8,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
