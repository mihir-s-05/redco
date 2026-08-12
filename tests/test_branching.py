from __future__ import annotations

import pytest

from redco.algo.branching import (
    CommitmentStatus,
    OnlineTargetSelector,
    RandomizedSelectiveTargetSelector,
    TokenSpan,
    assemble_redco_credit,
    inclusive_group_mean_advantages,
    leave_one_out_advantages,
    trajectory_rloo,
)
from redco.contracts import (
    PolicyNodeKind,
    PracticalSnapshotLifecycle,
    PrefixFeatures,
    SnapshotPhase,
)


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


def test_selective_selector_preserves_a_seeded_randomized_replay_floor() -> None:
    priority = RandomizedSelectiveTargetSelector(
        seed=7,
        randomized_replay_fraction=0.1,
        priority_threshold=0.75,
    )
    priority_result = priority.consider(
        "child-priority",
        features(PolicyNodeKind.SUBCALL_OUTPUT),
        priority_score=0.9,
    )
    assert priority_result.status is CommitmentStatus.COMMITTED
    assert priority_result.selection_probability == 1.0
    assert priority_result.selection_mode == "priority"

    randomized = RandomizedSelectiveTargetSelector(
        seed=31,
        randomized_replay_fraction=0.1,
        priority_threshold=0.75,
    )
    randomized_result = randomized.consider(
        "child-randomized",
        features(PolicyNodeKind.SUBCALL_OUTPUT),
        priority_score=0.2,
    )
    assert randomized_result.status is CommitmentStatus.COMMITTED
    assert randomized_result.selection_probability == 0.1
    assert randomized_result.selection_mode == "randomized_replay"

    skipped = RandomizedSelectiveTargetSelector(
        seed=7,
        randomized_replay_fraction=0.1,
        priority_threshold=0.75,
    )
    skipped_result = skipped.consider(
        "child-skipped",
        features(PolicyNodeKind.SUBCALL_OUTPUT),
        priority_score=0.2,
    )
    assert skipped_result.status is CommitmentStatus.SKIPPED
    assert skipped_result.selection_probability == 0.1
    assert skipped_result.selection_mode == "not_selected"


@pytest.mark.parametrize("mini_epochs", [2, 3])
def test_practical_snapshot_allows_a_frozen_fixed_mini_epoch_budget(
    mini_epochs: int,
) -> None:
    lifecycle = PracticalSnapshotLifecycle("theta-0", mini_epochs)
    lifecycle.begin_collection(
        rollout_checkpoint="theta-0",
        branch_checkpoint="theta-0",
    )
    lifecycle.finish_collection()

    for step in range(mini_epochs):
        lifecycle.record_optimizer_step()
        expected = (
            SnapshotPhase.UPDATED
            if step + 1 == mini_epochs
            else SnapshotPhase.COLLECTED
        )
        assert lifecycle.phase is expected

    with pytest.raises(RuntimeError, match="cannot update"):
        lifecycle.record_optimizer_step()


def test_loo_and_stock_scaling_are_explicitly_distinct() -> None:
    rewards = (1.0, 2.0, 4.0)
    loo = leave_one_out_advantages(rewards)
    stock = inclusive_group_mean_advantages(rewards)

    assert trajectory_rloo(rewards) == loo
    assert loo == pytest.approx((-2.0, -0.5, 2.5))
    assert stock == pytest.approx(tuple((len(rewards) - 1) / len(rewards) * x for x in loo))
    assert sum(loo) == pytest.approx(0.0)


def test_redco_credit_replaces_target_with_four_branch_records() -> None:
    assignment = assemble_redco_credit(
        trainable_mask=(False, True, True, False, True, True),
        trajectory_advantage=2.0,
        target_span=TokenSpan(4, 6),
        branch_rewards=(1.0, 0.0, 0.5, -1.0),
        outer_weight=0.5,
    )

    expected = leave_one_out_advantages((1.0, 0.0, 0.5, -1.0))
    assert assignment.incumbent_token_advantages == (0.0, 2.0, 2.0, 0.0, 0.0, 0.0)
    assert tuple(record.advantage for record in assignment.branch_records) == expected
    assert tuple(record.record_weight for record in assignment.branch_records) == (
        0.125,
        0.125,
        0.125,
        0.125,
    )
    assert assignment.target_replaced


def test_redco_credit_keeps_trajectory_credit_when_target_is_skipped() -> None:
    assignment = assemble_redco_credit(
        trainable_mask=(False, True, True),
        trajectory_advantage=-0.75,
        target_span=None,
        branch_rewards=None,
        outer_weight=1.0,
    )

    assert assignment.incumbent_token_advantages == (0.0, -0.75, -0.75)
    assert assignment.branch_records == ()
    assert not assignment.target_replaced


@pytest.mark.parametrize(
    ("target_span", "branch_rewards", "message"),
    [
        (TokenSpan(1, 2), None, "requires branch rewards"),
        (None, (1.0, 0.0, 0.5, -1.0), "require a committed target"),
        (TokenSpan(1, 2), (1.0, 0.0, 0.5), "exactly four"),
        (TokenSpan(0, 1), (1.0, 0.0, 0.5, -1.0), "only trainable"),
    ],
)
def test_redco_credit_rejects_incomplete_or_misaligned_inputs(
    target_span: TokenSpan | None,
    branch_rewards: tuple[float, ...] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        assemble_redco_credit(
            trainable_mask=(False, True),
            trajectory_advantage=1.0,
            target_span=target_span,
            branch_rewards=branch_rewards,
            outer_weight=1.0,
        )
