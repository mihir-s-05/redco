from __future__ import annotations

import pytest

from redco.analysis.channel_interchange import InterchangeEffects
from redco.analysis.routing_probe import (
    Method,
    ProbeConfig,
    RouteAction,
    TaskMode,
    _ambient_failure_reward,
    _context_corruption_reward,
    _context_dropout_reward,
    _four_cell_utilities,
    _normal_cells,
    _normal_reward,
    build_report,
    compact_result,
    train,
)


def test_reward_table_requires_adaptive_routing() -> None:
    assert all(
        _normal_reward(TaskMode.ARTIFACT_WITH_CONTEXT_SHORTCUT, action) == 1.0
        for action in RouteAction
    )
    assert (
        _ambient_failure_reward(
            TaskMode.ARTIFACT_WITH_CONTEXT_SHORTCUT,
            RouteAction.ARTIFACT_ONLY,
        )
        == 1.0
    )
    assert (
        _ambient_failure_reward(
            TaskMode.ARTIFACT_WITH_CONTEXT_SHORTCUT,
            RouteAction.BOTH,
        )
        == 0.0
    )
    assert _normal_reward(TaskMode.CONTEXT_NEEDED, RouteAction.ARTIFACT_ONLY) == 0.0
    assert _normal_reward(TaskMode.BOTH_NEEDED, RouteAction.BOTH) == 1.0


def test_dropout_and_corruption_are_distinct_uniform_baselines() -> None:
    mode = TaskMode.ARTIFACT_WITH_CONTEXT_SHORTCUT
    assert _context_dropout_reward(mode, RouteAction.BOTH) == 1.0
    assert _context_corruption_reward(mode, RouteAction.BOTH) == 0.0


def test_normal_four_cells_do_not_reveal_which_redundant_channel_will_fail() -> None:
    shortcut = _normal_cells(TaskMode.ARTIFACT_WITH_CONTEXT_SHORTCUT)
    either = _normal_cells(TaskMode.EITHER_SUFFICIENT)
    assert shortcut == either == InterchangeEffects(0.0, 1.0, 1.0, 1.0)
    typed = _four_cell_utilities(Method.TYPED_ALLOCATION, shortcut, ProbeConfig())
    assert typed[0] == typed[1]
    assert typed[2] < typed[0]


def test_same_four_cells_give_scalar_ablations_no_route_gradient() -> None:
    effects = InterchangeEffects(0.0, 1.0, 1.0, 1.0)
    config = ProbeConfig()
    for method in (
        Method.FOUR_CELL_MEAN,
        Method.FOUR_CELL_MINIMUM,
        Method.FOUR_CELL_SOFT_MIN,
    ):
        utilities = _four_cell_utilities(method, effects, config)
        assert len(set(utilities)) == 1


def test_one_decision_redco_lite_and_trajectory_are_exactly_equivalent() -> None:
    config = ProbeConfig(seeds=2, updates_per_mode=20)
    for seed in range(2):
        trajectory = train(seed, Method.TRAJECTORY, config)
        redco_lite = train(seed, Method.REDCO_LITE, config)
        assert trajectory.normal_reward == redco_lite.normal_reward
        assert trajectory.ambient_failure_reward == redco_lite.ambient_failure_reward
        assert trajectory.per_mode == redco_lite.per_mode


def test_equal_call_schedules_are_exact() -> None:
    config = ProbeConfig(seeds=2, updates_per_mode=160)
    uniform_160 = train(0, Method.UNIFORM_CONTEXT_CORRUPTION, config)
    oracle_80 = train(0, Method.ORACLE_SHIFT_PENALTY, config, updates_per_mode=80)
    uniform_320 = train(
        0,
        Method.UNIFORM_CONTEXT_CORRUPTION,
        config,
        updates_per_mode=320,
    )
    oracle_160 = train(0, Method.ORACLE_SHIFT_PENALTY, config)
    assert uniform_160.logical_reward_calls == oracle_80.logical_reward_calls == 1920
    assert uniform_320.logical_reward_calls == oracle_160.logical_reward_calls == 3840


def test_route_actions_are_enumerated_and_informative_updates_are_reported() -> None:
    result = train(0, Method.TRAJECTORY, ProbeConfig(seeds=2, updates_per_mode=10))
    assert result.diagnostics["enumerated_unique_route_group_fraction"] == 1.0
    assert 0.0 <= result.diagnostics["all_equal_reward_group_fraction"] <= 1.0
    assert (
        result.diagnostics["all_equal_reward_group_fraction"]
        == result.diagnostics["zero_advantage_update_fraction"]
    )


def test_fixed_artifact_route_cannot_solve_the_mixed_environment() -> None:
    result = train(0, Method.FIXED_ARTIFACT, ProbeConfig(seeds=2, updates_per_mode=1))
    assert result.normal_reward == pytest.approx(0.5)
    assert result.ambient_failure_reward == pytest.approx(0.5)


def test_report_answers_cost_and_decomposition_questions_without_llm() -> None:
    report = build_report(ProbeConfig(seeds=8, updates_per_mode=40))
    payload = report["payload"]
    assert payload["findings"]["old_typed_name_corrected_to_oracle_shift_penalty"] is True
    assert payload["findings"]["single_decision_redco_lite_equals_trajectory"] is True
    assert payload["environment"]["route_actions_enumerated_without_replacement"] is True
    assert payload["execution_work"]["judge_calls"] == 0
    for comparison in payload["findings"]["legacy_equal_call_controls"].values():
        assert comparison["left_calls_per_seed"] == comparison["right_calls_per_seed"]
    assert payload["decision"]["build_sequential_heldout_shift_benchmark"] is False

    compact = compact_result(
        report,
        source_path="runs/routing-controls-v2/report.json",
        source_raw_sha256="b" * 64,
    )
    assert compact["schema_version"] == 2
    assert "seed_results" not in compact["payload"]
