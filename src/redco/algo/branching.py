"""Online target selection and branch-group estimators."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import fsum

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
