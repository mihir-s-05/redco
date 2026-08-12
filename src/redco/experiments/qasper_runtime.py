"""Framework-neutral rollout batches for the QASPER evidence pilot."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from redco.algo.branching import leave_one_out_advantages
from redco.experiments.qasper_evidence import (
    EvidenceTask,
    PilotBudget,
    build_span_options,
    stage_one_prompt,
    stage_two_prompt,
)


class ChoicePolicy(Protocol):
    def sample(self, prompts: Sequence[str]) -> tuple[tuple[int, float], ...]: ...


@dataclass(frozen=True, slots=True)
class Decision:
    prompt: str
    action: int
    behavior_logprob: float
    advantage: float
    outer_weight: float
    decision_units: float


@dataclass(frozen=True, slots=True)
class Episode:
    paragraph: Decision
    span: Decision
    reward: float
    paragraph_correct: bool


def sample_episode(policy: ChoicePolicy, task: EvidenceTask) -> Episode:
    paragraph_prompt = stage_one_prompt(task)
    ((paragraph_action, paragraph_logprob),) = policy.sample((paragraph_prompt,))
    span_prompt = stage_two_prompt(task, paragraph_action)
    ((span_action, span_logprob),) = policy.sample((span_prompt,))
    _, gold_span = build_span_options(task, paragraph_action)
    paragraph_correct = paragraph_action == task.gold_paragraph_index
    reward = float(paragraph_correct and span_action == gold_span)
    return Episode(
        Decision(paragraph_prompt, paragraph_action, paragraph_logprob, 0.0, 1.0, 1.0),
        Decision(span_prompt, span_action, span_logprob, 0.0, 1.0, 1.0),
        reward,
        paragraph_correct,
    )


def _with_credit(decision: Decision, advantage: float, weight: float, units: float) -> Decision:
    return Decision(
        decision.prompt,
        decision.action,
        decision.behavior_logprob,
        advantage,
        weight,
        units,
    )


def trajectory_batch(
    policy: ChoicePolicy,
    task: EvidenceTask,
    budget: PilotBudget,
) -> tuple[list[Decision], float]:
    episodes = [sample_episode(policy, task) for _ in range(budget.baseline_episodes_per_update)]
    advantages = leave_one_out_advantages(tuple(episode.reward for episode in episodes))
    decisions = [
        _with_credit(decision, advantage, 1.0, 1.0)
        for episode, advantage in zip(episodes, advantages, strict=True)
        for decision in (episode.paragraph, episode.span)
    ]
    return decisions, sum(episode.reward for episode in episodes) / len(episodes)


def redco_batch(policy: ChoicePolicy, task: EvidenceTask) -> tuple[list[Decision], float]:
    primary = [sample_episode(policy, task) for _ in range(2)]
    paragraph_advantages = leave_one_out_advantages(tuple(episode.reward for episode in primary))
    paragraph_records = [
        _with_credit(episode.paragraph, advantage, 0.5, 0.5)
        for episode, advantage in zip(primary, paragraph_advantages, strict=True)
    ]

    original = primary[0].span
    alternatives = policy.sample((original.prompt,) * 6)
    span_records = [original]
    span_rewards = [primary[0].reward]
    _, gold_span = build_span_options(task, primary[0].paragraph.action)
    for action, logprob in alternatives:
        span_records.append(Decision(original.prompt, action, logprob, 0.0, 1.0, 1.0))
        span_rewards.append(float(primary[0].paragraph_correct and action == gold_span))
    span_advantages = leave_one_out_advantages(tuple(span_rewards))
    weight = 1.0 / len(span_records)
    credited_spans = [
        _with_credit(record, advantage, weight, weight)
        for record, advantage in zip(span_records, span_advantages, strict=True)
    ]
    rewards = [episode.reward for episode in primary] + span_rewards[1:]
    return [*paragraph_records, *credited_spans], sum(rewards) / len(rewards)
