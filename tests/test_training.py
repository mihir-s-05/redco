from __future__ import annotations

import pytest

from redco.algo import (
    BranchActionExample,
    PolicyDecision,
    ReDCOTrainerRecord,
    SequenceExample,
    TokenSpan,
    compile_redco_records,
    decision_normalized_loss,
)


def _sequence(
    token_ids: tuple[int, ...],
    trainable_mask: tuple[bool, ...],
    logprobs: tuple[float, ...],
) -> SequenceExample:
    return SequenceExample(token_ids, trainable_mask, logprobs, "credit-probe")


def test_decision_loss_matches_the_prime_rl_objective() -> None:
    sequence = _sequence((10, 20, 21), (False, True, True), (0.0, -0.3, -0.4))
    record = ReDCOTrainerRecord(
        sequence=sequence,
        advantages=(0.0, 2.0, 2.0),
        rl_weights=(0.0, 0.25, 0.25),
        decision_unit_normalizer=1.0,
        record_kind="branch",
        target_node_id="decision",
        branch_index=0,
    )

    result = decision_normalized_loss((record,), ((-0.2, -0.3, -0.4),))

    assert result.loss == pytest.approx(0.35)
    assert result.policy_gradient == pytest.approx(0.35)
    assert result.behavior_drift_penalty == 0.0
    assert result.decision_units == 1.0
    assert result.selected_tokens == 2


def test_branch_group_is_one_decision_unit_and_replaces_incumbent_credit() -> None:
    incumbent = _sequence(
        (1, 10, 11, 20),
        (False, True, True, True),
        (0.0, -0.2, -0.3, -0.4),
    )
    decisions = (
        PolicyDecision("context", TokenSpan(1, 3)),
        PolicyDecision("target", TokenSpan(3, 4)),
    )
    branches = tuple(
        BranchActionExample(
            _sequence((1, token), (False, True), (0.0, -0.5)),
            TokenSpan(1, 2),
            reward,
            "original" if index == 0 else "sampled",
        )
        for index, (token, reward) in enumerate(
            ((20, 1.0), (21, 0.0), (22, 0.5), (23, -1.0))
        )
    )
    records = compile_redco_records(
        incumbent=incumbent,
        decisions=decisions,
        trajectory_advantage=2.0,
        target_node_id="target",
        branches=branches,
    )

    result = decision_normalized_loss(
        records,
        tuple(record.sequence.behavior_logprobs for record in records),
    )

    assert len(records) == 5
    assert records[0].advantages == (0.0, 2.0, 2.0, 0.0)
    assert result.decision_units == pytest.approx(2.0)
    assert result.selected_tokens == 6
    assert result.records == 5


def test_equal_action_logprob_has_equal_loss_across_tokenizations() -> None:
    short = ReDCOTrainerRecord(
        _sequence((1,), (True,), (-0.5,)),
        (2.0,),
        (1.0,),
        1.0,
        "incumbent",
        None,
        None,
    )
    long = ReDCOTrainerRecord(
        _sequence((1, 2), (True, True), (-0.25, -0.25)),
        (2.0, 2.0),
        (1.0, 1.0),
        1.0,
        "incumbent",
        None,
        None,
    )

    short_loss = decision_normalized_loss((short,), ((-0.5,),))
    long_loss = decision_normalized_loss((long,), ((-0.25, -0.25),))

    assert short_loss.loss == long_loss.loss == pytest.approx(1.0)


def test_behavior_drift_penalty_uses_rollout_logprobs() -> None:
    record = ReDCOTrainerRecord(
        _sequence((1, 2), (True, True), (-0.4, -0.6)),
        (0.0, 0.0),
        (0.5, 0.5),
        1.0,
        "branch",
        "target",
        0,
    )

    result = decision_normalized_loss(
        (record,),
        ((-0.2, -0.8),),
        behavior_drift_weight=0.25,
    )

    assert result.policy_gradient == 0.0
    assert result.behavior_drift_penalty == pytest.approx(0.04)
    assert result.loss == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("current", "weight", "message"),
    [
        ((), 0.0, "align with record tokens"),
        ((float("nan"),), 0.0, "must be finite"),
        ((-0.2,), -1.0, "finite and non-negative"),
    ],
)
def test_decision_loss_rejects_invalid_inputs(
    current: tuple[float, ...],
    weight: float,
    message: str,
) -> None:
    record = ReDCOTrainerRecord(
        _sequence((1,), (True,), (-0.2,)),
        (1.0,),
        (1.0,),
        1.0,
        "incumbent",
        None,
        None,
    )
    with pytest.raises(ValueError, match=message):
        decision_normalized_loss((record,), (current,), behavior_drift_weight=weight)


def test_trainer_record_rejects_nonfinite_credit() -> None:
    sequence = _sequence((1,), (True,), (-0.2,))
    with pytest.raises(ValueError, match="advantages must be finite"):
        ReDCOTrainerRecord(
            sequence,
            (float("nan"),),
            (1.0,),
            1.0,
            "incumbent",
            None,
            None,
        )


def test_sequence_rejects_nonfinite_behavior_logprob() -> None:
    with pytest.raises(ValueError, match="behavior log-probabilities must be finite"):
        _sequence((1,), (True,), (float("nan"),))


def test_decision_loss_rejects_aggregate_overflow() -> None:
    record = ReDCOTrainerRecord(
        _sequence((1,), (True,), (-2.0,)),
        (1e308,),
        (1.0,),
        1.0,
        "incumbent",
        None,
        None,
    )
    with pytest.raises(ValueError, match="loss terms must be finite"):
        decision_normalized_loss(
            (record,),
            ((-1.0,),),
            behavior_drift_weight=1e308,
        )
