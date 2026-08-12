"""Pure ReDCO compilation and decision-normalized training objective."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite
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
        if any(not isfinite(value) for value in self.behavior_logprobs):
            raise ValueError("behavior log-probabilities must be finite")


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
class ReDCOTrainerRecord:
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
        if not isfinite(self.decision_unit_normalizer):
            raise ValueError("decision_unit_normalizer must be finite")
        if any(not isfinite(value) for value in self.advantages):
            raise ValueError("advantages must be finite")
        if any(not isfinite(value) or value < 0 for value in self.rl_weights):
            raise ValueError("RL weights must be finite and non-negative")
        if any(
            weight != 0.0 and not trainable
            for weight, trainable in zip(
                self.rl_weights,
                self.sequence.trainable_mask,
                strict=True,
            )
        ):
            raise ValueError("RL weights may only select trainable tokens")


@dataclass(frozen=True, slots=True)
class DecisionLoss:
    """Framework-neutral ReDCO loss and its auditable components."""

    loss: float
    policy_gradient: float
    behavior_drift_penalty: float
    decision_units: float
    selected_tokens: int
    records: int


def decision_normalized_loss(
    records: Sequence[ReDCOTrainerRecord],
    current_logprobs: Sequence[Sequence[float]],
    *,
    behavior_drift_weight: float = 0.0,
) -> DecisionLoss:
    """Evaluate the clean ReDCO objective without a training framework.

    The numerator is the action-token REINFORCE objective used by the Prime RL
    bridge. An optional squared log-ratio penalty discourages movement away from
    the behavior policy that produced the replay. The denominator counts policy
    decisions rather than tokens. The selected token log-probabilities still sum
    to the log-probability of each action.
    """
    if not records:
        raise ValueError("at least one trainer record is required")
    if len(records) != len(current_logprobs):
        raise ValueError("current log-probabilities must align with records")
    if not isfinite(behavior_drift_weight) or behavior_drift_weight < 0:
        raise ValueError("behavior_drift_weight must be finite and non-negative")

    policy_terms: list[float] = []
    drift_terms: list[float] = []
    decision_units: list[float] = []
    selected_tokens = 0
    for record, observed in zip(records, current_logprobs, strict=True):
        current = tuple(float(value) for value in observed)
        if len(current) != len(record.sequence.token_ids):
            raise ValueError("current log-probabilities must align with record tokens")
        if any(not isfinite(value) for value in current):
            raise ValueError("current log-probabilities must be finite")
        decision_units.append(record.decision_unit_normalizer)
        for advantage, weight, trainer_logprob, behavior_logprob in zip(
            record.advantages,
            record.rl_weights,
            current,
            record.sequence.behavior_logprobs,
            strict=True,
        ):
            if weight == 0.0:
                continue
            selected_tokens += 1
            policy_term = -advantage * trainer_logprob * weight
            drift_term = (trainer_logprob - behavior_logprob) ** 2 * weight
            if not isfinite(policy_term) or not isfinite(drift_term):
                raise ValueError("loss terms must be finite")
            policy_terms.append(policy_term)
            drift_terms.append(drift_term)

    try:
        normalizer = fsum(decision_units)
        policy_gradient = fsum(policy_terms) / normalizer
        behavior_drift_penalty = fsum(drift_terms) / normalizer
    except OverflowError as error:
        raise ValueError("aggregated loss terms must be finite") from error
    loss = policy_gradient + behavior_drift_weight * behavior_drift_penalty
    if not all(isfinite(value) for value in (policy_gradient, behavior_drift_penalty, loss)):
        raise ValueError("aggregated loss terms must be finite")
    return DecisionLoss(
        loss=loss,
        policy_gradient=policy_gradient,
        behavior_drift_penalty=behavior_drift_penalty,
        decision_units=normalizer,
        selected_tokens=selected_tokens,
        records=len(records),
    )


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
    selected = set(range(span.start, span.stop))
    trainable = {
        index
        for index, value in enumerate(sequence.trainable_mask)
        if value
    }
    if selected != trainable:
        raise ValueError("a branch record must train exactly its action span")


def compile_redco_records(
    *,
    incumbent: SequenceExample,
    decisions: tuple[PolicyDecision, ...],
    trajectory_advantage: float,
    target_node_id: str | None,
    branches: tuple[BranchActionExample, ...] = (),
    branch_advantages: tuple[float, ...] | None = None,
) -> tuple[ReDCOTrainerRecord, ...]:
    """Apply ReDCO's branch replacement and decision-unit weighting rules."""
    _validate_decisions(incumbent, decisions)
    by_node = {decision.node_id: decision for decision in decisions}

    if target_node_id is None:
        if branches:
            raise ValueError("branch records require a committed target")
        if branch_advantages is not None:
            raise ValueError("branch advantages require branch records")
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
            ReDCOTrainerRecord(
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
            "ReDCO requires an original plus at least one alternative"
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

    records: list[ReDCOTrainerRecord] = []
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
            ReDCOTrainerRecord(
                incumbent,
                tuple(advantages),
                tuple(weights),
                float(len(untargeted)),
                "incumbent",
                target_node_id,
                None,
            )
        )

    if branch_advantages is None:
        resolved_branch_advantages = leave_one_out_advantages(
            tuple(branch.reward for branch in branches)
        )
    else:
        if len(branch_advantages) != len(branches):
            raise ValueError("branch advantages must align with branch records")
        resolved_branch_advantages = tuple(float(value) for value in branch_advantages)
    branch_weight = target.outer_weight / len(branches)
    branch_normalizer = 1.0 / len(branches)
    for index, (branch, advantage) in enumerate(
        zip(branches, resolved_branch_advantages, strict=True)
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
            ReDCOTrainerRecord(
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
