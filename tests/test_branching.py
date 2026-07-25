from __future__ import annotations

import pytest

from redco.algo.branching import (
    CommitmentStatus,
    OnlineTargetSelector,
    inclusive_group_mean_advantages,
    leave_one_out_advantages,
    trajectory_rloo,
)
from redco.contracts import PolicyNodeKind, PrefixFeatures


def features(kind: PolicyNodeKind) -> PrefixFeatures:
    return PrefixFeatures(node_kind=kind, depth=1, turn_index=2)


def test_selector_commits_before_action_to_first_eligible_subcall() -> None:
    selector = OnlineTargetSelector()
    root = selector.consider("root-1", features(PolicyNodeKind.ROOT_TURN))
    child = selector.consider("child-1", features(PolicyNodeKind.SUBCALL_OUTPUT))
    second = selector.consider("child-2", features(PolicyNodeKind.SUBCALL_OUTPUT))

    assert root.status is CommitmentStatus.INELIGIBLE
    assert child.status is CommitmentStatus.COMMITTED
    assert child.node_id == "child-1"
    assert second.status is CommitmentStatus.ALREADY_COMMITTED


def test_selector_logs_skip_without_root_fallback() -> None:
    selector = OnlineTargetSelector()
    selector.consider("root-1", features(PolicyNodeKind.ROOT_TURN))

    assert selector.finalize().status is CommitmentStatus.SKIPPED


def test_loo_and_stock_scaling_are_explicitly_distinct() -> None:
    rewards = (1.0, 2.0, 4.0)
    loo = leave_one_out_advantages(rewards)
    stock = inclusive_group_mean_advantages(rewards)

    assert trajectory_rloo(rewards) == loo
    assert loo == pytest.approx((-2.0, -0.5, 2.5))
    assert stock == pytest.approx(tuple((len(rewards) - 1) / len(rewards) * x for x in loo))
    assert sum(loo) == pytest.approx(0.0)

