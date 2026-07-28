"""Pure Stage-C compilation from decision spans to trainer records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from redco.algo.branching import TokenSpan, leave_one_out_advantages


@dataclass(frozen=True, slots=True)
class SequenceExample:
    """One tokenized policy sequence before credit assignment."""

    token_ids: tuple[int, ...]
    trainable_mask: tuple[bool, ...]
    behavior_logprobs: tuple[float, ...]
    env_name: str

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("sequence must contain tokens")
        if not self.env_name:
            raise ValueError("env_name must be non-empty")
        if not (
            len(self.token_ids)
            == len(self.trainable_mask)
            == len(self.behavior_logprobs)
        ):
            raise ValueError("sequence fields must have identical lengths")


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """A trainable macro-action and its clean-objective outer weight."""

    node_id: str
    token_span: TokenSpan
    outer_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("decision node_id must be non-empty")
        if self.outer_weight <= 0:
            raise ValueError("decision outer_weight must be positive")


@dataclass(frozen=True, slots=True)
class BranchActionExample:
    """One behavior-policy action evaluated from the committed target state."""

    sequence: SequenceExample
    action_span: TokenSpan
    reward: float
    action_source: Literal["original", "sampled"]


@dataclass(frozen=True, slots=True)
class StageCTrainerRecord:
    """A trainer-bound sequence with explicit clean-loss normalization."""

    sequence: SequenceExample
    advantages: tuple[float, ...]
    rl_weights: tuple[float, ...]
    decision_unit_normalizer: float
    record_kind: Literal["incumbent", "branch"]
    target_node_id: str | None
    branch_index: int | None

    def __post_init__(self) -> None:
        length = len(self.sequence.token_ids)
        if len(self.advantages) != length or len(self.rl_weights) != length:
            raise ValueError("credit streams must align with sequence tokens")
        if self.decision_unit_normalizer <= 0:
            raise ValueError("decision_unit_normalizer must be positive")
        if any(
            weight != 0.0 and not trainable
            for weight, trainable in zip(
                self.rl_weights,
                self.sequence.trainable_mask,
                strict=True,
            )
        ):
            raise ValueError("RL weights may only select trainable tokens")


def _validate_decisions(
    sequence: SequenceExample,
    decisions: tuple[PolicyDecision, ...],
) -> None:
    if not decisions:
        raise ValueError("at least one policy decision is required")
    node_ids = [decision.node_id for decision in decisions]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("decision node_ids must be unique")

    owner: list[str | None] = [None] * len(sequence.token_ids)
    for decision in decisions:
        span = decision.token_span
        if span.stop > len(owner):
            raise ValueError(f"decision {decision.node_id} exceeds the sequence")
        for index in range(span.start, span.stop):
            if not sequence.trainable_mask[index]:
                raise ValueError(
                    f"decision {decision.node_id} includes a non-trainable token"
                )
            if owner[index] is not None:
                raise ValueError(
                    f"decision spans overlap at token {index}: "
                    f"{owner[index]} and {decision.node_id}"
                )
            owner[index] = decision.node_id

    uncovered = [
        index
        for index, trainable in enumerate(sequence.trainable_mask)
        if trainable and owner[index] is None
    ]
    if uncovered:
        raise ValueError(f"trainable tokens lack a decision span: {uncovered}")


def _validate_branch_action(branch: BranchActionExample) -> None:
    span = branch.action_span
    sequence = branch.sequence
    if span.stop > len(sequence.token_ids):
        raise ValueError("branch action span exceeds its sequence")
    selected = {
        index
        for index in range(span.start, span.stop)
    }
    trainable = {
        index
        for index, value in enumerate(sequence.trainable_mask)
        if value
    }
    if selected != trainable:
        raise ValueError("a branch record must train exactly its action span")


def compile_stage_c_records(
    *,
    incumbent: SequenceExample,
    decisions: tuple[PolicyDecision, ...],
    trajectory_advantage: float,
    target_node_id: str | None,
    branches: tuple[BranchActionExample, ...] = (),
) -> tuple[StageCTrainerRecord, ...]:
    """Apply the clean Stage-C replacement and decision-unit weighting rules."""
    _validate_decisions(incumbent, decisions)
    by_node = {decision.node_id: decision for decision in decisions}

    if target_node_id is None:
        if branches:
            raise ValueError("branch records require a committed target")
        advantages = [0.0] * len(incumbent.token_ids)
        weights = [0.0] * len(incumbent.token_ids)
        for decision in decisions:
            for index in range(
                decision.token_span.start,
                decision.token_span.stop,
            ):
                advantages[index] = float(trajectory_advantage)
                weights[index] = decision.outer_weight
        return (
            StageCTrainerRecord(
                incumbent,
                tuple(advantages),
                tuple(weights),
                float(len(decisions)),
                "incumbent",
                None,
                None,
            ),
        )

    try:
        target = by_node[target_node_id]
    except KeyError as error:
        raise ValueError("target_node_id must identify a decision span") from error
    if len(branches) < 2:
        raise ValueError(
            "clean Stage C requires an original plus at least one alternative"
        )
    if branches[0].action_source != "original":
        raise ValueError("the first branch must be the precommitted original action")
    if any(branch.action_source != "sampled" for branch in branches[1:]):
        raise ValueError("alternative branches must be behavior-policy samples")
    for branch in branches:
        _validate_branch_action(branch)

    original_tokens = incumbent.token_ids[
        target.token_span.start : target.token_span.stop
    ]
    first = branches[0]
    first_tokens = first.sequence.token_ids[
        first.action_span.start : first.action_span.stop
    ]
    if first_tokens != original_tokens:
        raise ValueError("the original branch action must match the incumbent target")

    records: list[StageCTrainerRecord] = []
    untargeted = tuple(
        decision for decision in decisions if decision.node_id != target_node_id
    )
    if untargeted:
        advantages = [0.0] * len(incumbent.token_ids)
        weights = [0.0] * len(incumbent.token_ids)
        for decision in untargeted:
            for index in range(
                decision.token_span.start,
                decision.token_span.stop,
            ):
                advantages[index] = float(trajectory_advantage)
                weights[index] = decision.outer_weight
        records.append(
            StageCTrainerRecord(
                incumbent,
                tuple(advantages),
                tuple(weights),
                float(len(untargeted)),
                "incumbent",
                target_node_id,
                None,
            )
        )

    branch_advantages = leave_one_out_advantages(
        tuple(branch.reward for branch in branches)
    )
    branch_weight = target.outer_weight / len(branches)
    branch_normalizer = 1.0 / len(branches)
    for index, (branch, advantage) in enumerate(
        zip(branches, branch_advantages, strict=True)
    ):
        advantages = [0.0] * len(branch.sequence.token_ids)
        weights = [0.0] * len(branch.sequence.token_ids)
        for token_index in range(
            branch.action_span.start,
            branch.action_span.stop,
        ):
            advantages[token_index] = float(advantage)
            weights[token_index] = branch_weight
        records.append(
            StageCTrainerRecord(
                branch.sequence,
                tuple(advantages),
                tuple(weights),
                branch_normalizer,
                "branch",
                target_node_id,
                index,
            )
        )
    return tuple(records)
