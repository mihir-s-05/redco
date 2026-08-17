from __future__ import annotations

import pytest

from redco.analysis.routing_probe import (
    Method,
    ProbeConfig,
    RouteAction,
    TaskMode,
    _ambient_failure_reward,
    _context_corruption_reward,
    _context_dropout_reward,
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
            RouteAction.CONTEXT_ONLY,
        )
        == 0.0
    )
    assert _normal_reward(TaskMode.CONTEXT_NEEDED, RouteAction.ARTIFACT_ONLY) == 0.0
    assert _normal_reward(TaskMode.CONTEXT_NEEDED, RouteAction.CONTEXT_ONLY) == 1.0
    assert _normal_reward(TaskMode.BOTH_NEEDED, RouteAction.BOTH) == 1.0
    assert _normal_reward(TaskMode.BOTH_NEEDED, RouteAction.CONTEXT_ONLY) == 0.0


def test_dropout_and_corruption_are_distinct_simple_baselines() -> None:
    mode = TaskMode.ARTIFACT_WITH_CONTEXT_SHORTCUT
    assert _context_dropout_reward(mode, RouteAction.BOTH) == 1.0
    assert _context_corruption_reward(mode, RouteAction.BOTH) == 0.0


def test_one_decision_redco_lite_and_trajectory_are_exactly_equivalent() -> None:
    config = ProbeConfig(seeds=2, updates_per_mode=20)
    for seed in range(2):
        trajectory = train(seed, Method.TRAJECTORY, config)
        redco_lite = train(seed, Method.REDCO_LITE, config)
        assert trajectory.normal_reward == redco_lite.normal_reward
        assert trajectory.ambient_failure_reward == redco_lite.ambient_failure_reward
        assert trajectory.probabilities == redco_lite.probabilities


def test_fixed_artifact_route_cannot_solve_the_mixed_environment() -> None:
    result = train(0, Method.FIXED_ARTIFACT, ProbeConfig(seeds=2, updates_per_mode=1))
    assert result.normal_reward == pytest.approx(0.5)
    assert result.ambient_failure_reward == pytest.approx(0.5)


def test_report_applies_the_predeclared_simple_baseline_kill_gate() -> None:
    report = build_report(ProbeConfig(seeds=8, updates_per_mode=80))
    payload = report["payload"]
    assert payload["paired_findings"]["redco_lite_equals_trajectory_on_one_decision"] is True
    summaries = payload["summaries"]
    typed = summaries[Method.TYPED_INTERCHANGE.value]["ambient_failure_reward_mean"]
    corruption = summaries[Method.CONTEXT_CORRUPTION.value]["ambient_failure_reward_mean"]
    assert typed > summaries[Method.TRAJECTORY.value]["ambient_failure_reward_mean"]
    assert abs(typed - corruption) < 0.08
    best_simple = max(
        summaries[Method.CONTEXT_DROPOUT.value]["ambient_failure_reward_mean"],
        corruption,
    )
    normal_regression = (
        summaries[Method.TRAJECTORY.value]["normal_reward_mean"]
        - summaries[Method.TYPED_INTERCHANGE.value]["normal_reward_mean"]
    )
    expected = (
        payload["paired_findings"]["paired_gain_95pct_normal_ci"][0]
        >= payload["config"]["minimum_typed_gain_over_simple_baseline"]
        and normal_regression <= payload["config"]["maximum_normal_reward_regression"]
    )
    assert payload["decision"]["go_to_llm"] is expected
    assert typed - best_simple == pytest.approx(
        payload["paired_findings"]["typed_gain_over_best_simple_augmentation"]
    )

    compact = compact_result(
        report,
        source_path="runs/routing-probe-v1/report.json",
        source_raw_sha256="b" * 64,
    )
    assert "seed_results" not in compact["payload"]
