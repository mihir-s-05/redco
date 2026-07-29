from redco.analysis.stage_c3_invariants import check_first_training_row


def _row() -> dict[str, float]:
    return {
        "step": 1.0,
        "train/agg/all/reward/mean": 0.5,
        "train/agg/all/reward/min": 0.0,
        "train/agg/all/reward/max": 1.0,
        "train/agg/all/is_trainable/mean": 0.5,
        "train/agg/all/is_truncated/mean": 0.5,
        "train/agg/all/has_error/mean": 0.0,
        "train/agg/all/metrics/redco_valid_route/mean": 1.0,
        "train/agg/all/metrics/redco_context_token_budget_ok/mean": 1.0,
    }


def test_smoke_invariants_accept_healthy_multidecision_batch() -> None:
    result = check_first_training_row(_row(), mode="smoke")

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_smoke_invariants_reproduce_v1_terminal_failure() -> None:
    row = _row()
    row.update(
        {
            "train/agg/all/reward/mean": -0.5,
            "train/agg/all/reward/min": -0.5,
            "train/agg/all/reward/max": -0.5,
            "train/agg/all/is_trainable/mean": 0.0,
            "train/agg/all/is_truncated/mean": 1.0,
            "train/agg/all/metrics/redco_valid_route/mean": 0.0,
        }
    )

    result = check_first_training_row(row, mode="smoke")

    assert result["passed"] is False
    assert result["checks"] == {
        "no_rollout_errors": True,
        "root_completion_budget_contract": True,
        "every_root_route_parseable": False,
        "nonzero_trainable_fraction": False,
        "nonconstant_reward_exposure": False,
    }


def test_arm_gate_ignores_sampled_outcomes() -> None:
    row = _row()
    row.update(
        {
            "train/agg/all/reward/min": 0.5,
            "train/agg/all/reward/max": 0.5,
            "train/agg/all/is_trainable/mean": 0.0,
        }
    )

    result = check_first_training_row(row, mode="arm")

    assert result["passed"] is True
    assert result["checks"] == {
        "no_rollout_errors": True,
        "root_completion_budget_contract": True,
    }


def test_arm_gate_rejects_context_budget_regression() -> None:
    row = _row()
    row["train/agg/all/metrics/redco_context_token_budget_ok/mean"] = 0.0

    result = check_first_training_row(row, mode="arm")

    assert result["passed"] is False
    assert result["checks"]["root_completion_budget_contract"] is False
