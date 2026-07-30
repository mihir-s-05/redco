"""Online target selection and branch-group estimators."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import fsum
from random import Random

from redco.contracts import PolicyNodeKind, PrefixFeatures


class CommitmentStatus(StrEnum):
    COMMITTED = "committed"
    INELIGIBLE = "ineligible"
    ALREADY_COMMITTED = "already_committed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class TargetCommitment:
    node_id: str | None
    status: CommitmentStatus
    features: PrefixFeatures | None
    selection_probability: float | None = None
    selection_mode: str | None = None
    priority_score: float | None = None


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """Half-open token interval for one committed policy action."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("token span start must be non-negative")
        if self.stop <= self.start:
            raise ValueError("token span must be non-empty")


@dataclass(frozen=True, slots=True)
class BranchRecordCredit:
    """Credit and decision-unit weight for one emitted branch action."""

    advantage: float
    record_weight: float


@dataclass(frozen=True, slots=True)
class StageCCreditAssignment:
    """Trainer-facing credit for one rollout and its optional branch group."""

    incumbent_token_advantages: tuple[float, ...]
    branch_records: tuple[BranchRecordCredit, ...]
    target_replaced: bool


class OnlineTargetSelector:
    """Commit the first eligible node before its action is sampled."""

    def __init__(
        self,
        *,
        eligible_kinds: frozenset[PolicyNodeKind] = frozenset(
            {PolicyNodeKind.SUBCALL_OUTPUT}
        ),
    ) -> None:
        self._eligible_kinds = eligible_kinds
        self._commitment: TargetCommitment | None = None

    @property
    def commitment(self) -> TargetCommitment | None:
        return self._commitment

    def consider(self, node_id: str, features: PrefixFeatures) -> TargetCommitment:
        """Consider a node using a signature that cannot accept post-action data."""
        if not node_id:
            raise ValueError("node_id must be non-empty")
        if self._commitment is not None:
            return TargetCommitment(None, CommitmentStatus.ALREADY_COMMITTED, features)
        if features.node_kind not in self._eligible_kinds:
            return TargetCommitment(None, CommitmentStatus.INELIGIBLE, features)
        self._commitment = TargetCommitment(node_id, CommitmentStatus.COMMITTED, features)
        return self._commitment

    def finalize(self) -> TargetCommitment:
        """Return a logged skip when the rollout had no eligible target."""
        if self._commitment is not None:
            return self._commitment
        self._commitment = TargetCommitment(None, CommitmentStatus.SKIPPED, None)
        return self._commitment


class RandomizedSelectiveTargetSelector:
    """Target high-priority nodes while preserving randomized replay support.

    Every reached eligible node is selected with probability at least
    ``randomized_replay_fraction``. Priority nodes are selected with probability
    one. The decision consumes only ``PrefixFeatures`` and is therefore made
    before the action being targeted exists.
    """

    def __init__(
        self,
        *,
        seed: int,
        randomized_replay_fraction: float,
        priority_threshold: float,
        eligible_kinds: frozenset[PolicyNodeKind] = frozenset(
            {PolicyNodeKind.SUBCALL_OUTPUT}
        ),
    ) -> None:
        if not 0.0 < randomized_replay_fraction <= 1.0:
            raise ValueError("randomized_replay_fraction must be in (0, 1]")
        if not 0.0 <= priority_threshold <= 1.0:
            raise ValueError("priority_threshold must be in [0, 1]")
        self._random = Random(seed)
        self._randomized_replay_fraction = randomized_replay_fraction
        self._priority_threshold = priority_threshold
        self._eligible_kinds = eligible_kinds
        self._commitment: TargetCommitment | None = None
        self._last_decision: TargetCommitment | None = None

    @property
    def commitment(self) -> TargetCommitment | None:
        return self._commitment

    def consider(
        self,
        node_id: str,
        features: PrefixFeatures,
        *,
        priority_score: float,
    ) -> TargetCommitment:
        """Make a logged priority-or-randomized decision before action sampling."""
        if not node_id:
            raise ValueError("node_id must be non-empty")
        if not 0.0 <= priority_score <= 1.0:
            raise ValueError("priority_score must be in [0, 1]")
        if self._commitment is not None:
            return TargetCommitment(
                None,
                CommitmentStatus.ALREADY_COMMITTED,
                features,
                priority_score=priority_score,
            )
        if features.node_kind not in self._eligible_kinds:
            self._last_decision = TargetCommitment(
                None,
                CommitmentStatus.INELIGIBLE,
                features,
                selection_probability=0.0,
                priority_score=priority_score,
            )
            return self._last_decision

        priority = priority_score >= self._priority_threshold
        selection_probability = (
            1.0 if priority else self._randomized_replay_fraction
        )
        randomized = self._random.random() < self._randomized_replay_fraction
        if priority or randomized:
            self._commitment = TargetCommitment(
                node_id,
                CommitmentStatus.COMMITTED,
                features,
                selection_probability=selection_probability,
                selection_mode="priority" if priority else "randomized_replay",
                priority_score=priority_score,
            )
            self._last_decision = self._commitment
            return self._commitment
        self._last_decision = TargetCommitment(
            None,
            CommitmentStatus.SKIPPED,
            features,
            selection_probability=selection_probability,
            selection_mode="not_selected",
            priority_score=priority_score,
        )
        return self._last_decision

    def finalize(self) -> TargetCommitment:
        if self._commitment is not None:
            return self._commitment
        if self._last_decision is not None:
            return self._last_decision
        self._commitment = TargetCommitment(
            None,
            CommitmentStatus.SKIPPED,
            None,
            selection_probability=0.0,
            selection_mode="no_eligible_target",
        )
        return self._commitment


def leave_one_out_advantages(rewards: Sequence[float]) -> tuple[float, ...]:
    """Compute symmetric all-branch LOO advantages in raw reward units."""
    if len(rewards) < 2:
        raise ValueError("LOO requires at least two rewards")
    total = fsum(rewards)
    denominator = len(rewards) - 1
    return tuple(reward - ((total - reward) / denominator) for reward in rewards)


def trajectory_rloo(rewards: Sequence[float]) -> tuple[float, ...]:
    """Alias that makes the ReDCO trajectory-scaling contract explicit."""
    return leave_one_out_advantages(rewards)


def inclusive_group_mean_advantages(rewards: Sequence[float]) -> tuple[float, ...]:
    """Exact stock-GRPO scaling, isolated to the incumbent arm."""
    if not rewards:
        raise ValueError("at least one reward is required")
    mean = fsum(rewards) / len(rewards)
    return tuple(reward - mean for reward in rewards)


def mean_branch_gradient_weight(
    advantages: Iterable[float],
    *,
    outer_weight: float,
) -> tuple[float, ...]:
    """Return explicit per-record coefficients for one branch decision unit."""
    values = tuple(advantages)
    if not values:
        raise ValueError("at least one branch is required")
    if outer_weight <= 0:
        raise ValueError("outer_weight must be positive")
    scale = outer_weight / len(values)
    return tuple(scale * advantage for advantage in values)


def assemble_stage_c_credit(
    *,
    trainable_mask: Sequence[bool],
    trajectory_advantage: float,
    target_span: TokenSpan | None,
    branch_rewards: Sequence[float] | None,
    outer_weight: float,
) -> StageCCreditAssignment:
    """Compile the clean Stage-C replacement rule for one rollout.

    Untargeted action tokens retain trajectory RLOO credit. A committed target
    is zeroed in the incumbent record and emitted once per branch with
    all-branch LOO credit and equal decision-unit weight.
    """
    if not trainable_mask:
        raise ValueError("trainable_mask must be non-empty")
    if outer_weight <= 0:
        raise ValueError("outer_weight must be positive")

    incumbent = [
        float(trajectory_advantage) if trainable else 0.0
        for trainable in trainable_mask
    ]
    if target_span is None:
        if branch_rewards is not None:
            raise ValueError("branch rewards require a committed target")
        return StageCCreditAssignment(tuple(incumbent), (), False)

    if branch_rewards is None:
        raise ValueError("a committed target requires branch rewards")
    if len(branch_rewards) != 4:
        raise ValueError("clean Stage C requires exactly four branch rewards")
    if target_span.stop > len(trainable_mask):
        raise ValueError("target span exceeds the token stream")
    if not all(trainable_mask[target_span.start : target_span.stop]):
        raise ValueError("target span must contain only trainable action tokens")

    for index in range(target_span.start, target_span.stop):
        incumbent[index] = 0.0

    branch_advantages = leave_one_out_advantages(branch_rewards)
    record_weight = outer_weight / len(branch_advantages)
    branch_records = tuple(
        BranchRecordCredit(float(advantage), record_weight)
        for advantage in branch_advantages
    )
    return StageCCreditAssignment(tuple(incumbent), branch_records, True)
