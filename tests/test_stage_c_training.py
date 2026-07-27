from __future__ import annotations

import pytest

from redco.algo.branching import TokenSpan, leave_one_out_advantages
from redco.algo.training import (
    BranchActionExample,
    PolicyDecision,
    SequenceExample,
    compile_stage_c_records,
)


def sequence(
    token_ids: tuple[int, ...],
    mask: tuple[bool, ...],
) -> SequenceExample:
    return SequenceExample(
        token_ids,
        mask,
        tuple(-0.5 if trainable else 0.0 for trainable in mask),
        "credit-probe",
    )


def test_compile_targeted_rollout_emits_one_weighted_branch_group() -> None:
    incumbent = sequence(
        (10, 11, 12, 13, 14, 15, 16),
        (False, True, True, False, True, True, True),
    )
    decisions = (
        PolicyDecision("root", TokenSpan(1, 3), 1.0),
        PolicyDecision("child", TokenSpan(4, 7), 0.5),
    )
    rewards = (1.0, 0.0, 0.5, -1.0)
    branches = tuple(
        BranchActionExample(
            sequence(
                (20, 21, *action),
                (False, False, *(True for _ in action)),
            ),
            TokenSpan(2, 2 + len(action)),
            reward,
            "original" if index == 0 else "sampled",
        )
        for index, (action, reward) in enumerate(
            zip(
                (
                    (14, 15, 16),
                    (30,),
                    (31, 32),
                    (33,),
                ),
                rewards,
                strict=True,
            )
        )
    )

    records = compile_stage_c_records(
        incumbent=incumbent,
        decisions=decisions,
        trajectory_advantage=2.0,
        target_node_id="child",
        branches=branches,
    )

    assert len(records) == 5
    original, *branch_records = records
    assert original.record_kind == "incumbent"
    assert original.advantages == (0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0)
    assert original.rl_weights == (0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert original.decision_unit_normalizer == 1.0

    expected = leave_one_out_advantages(rewards)
    assert tuple(record.advantages[-1] for record in branch_records) == expected
    assert all(
        {
            weight
            for weight in record.rl_weights
            if weight != 0.0
        }
        == {0.125}
        for record in branch_records
    )
    assert sum(record.decision_unit_normalizer for record in branch_records) == 1.0


def test_compile_skipped_rollout_keeps_all_trajectory_decisions() -> None:
    incumbent = sequence(
        (10, 11, 12, 13),
        (False, True, False, True),
    )
    records = compile_stage_c_records(
        incumbent=incumbent,
        decisions=(
            PolicyDecision("one", TokenSpan(1, 2)),
            PolicyDecision("two", TokenSpan(3, 4), 0.5),
        ),
        trajectory_advantage=-0.75,
        target_node_id=None,
    )

    assert len(records) == 1
    assert records[0].advantages == (0.0, -0.75, 0.0, -0.75)
    assert records[0].rl_weights == (0.0, 1.0, 0.0, 0.5)
    assert records[0].decision_unit_normalizer == 2.0


def test_compile_rejects_post_commitment_and_alignment_violations() -> None:
    incumbent = sequence((10, 11), (False, True))
    decisions = (PolicyDecision("child", TokenSpan(1, 2)),)
    original = BranchActionExample(
        sequence((20, 99), (False, True)),
        TokenSpan(1, 2),
        1.0,
        "original",
    )
    alternative = BranchActionExample(
        sequence((20, 30), (False, True)),
        TokenSpan(1, 2),
        0.0,
        "sampled",
    )

    with pytest.raises(ValueError, match="must match"):
        compile_stage_c_records(
            incumbent=incumbent,
            decisions=decisions,
            trajectory_advantage=0.0,
            target_node_id="child",
            branches=(original, alternative, alternative, alternative),
        )
