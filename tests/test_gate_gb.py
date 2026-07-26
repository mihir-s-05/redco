from __future__ import annotations

from redco.analysis.gate_gb import run_gate


def test_cpu_gate_exercises_all_dependencies_and_passes() -> None:
    report = run_gate(
        seed=19,
        programs=100,
        interventions_per_program=10,
        events=12,
    )

    assert report.passed_cpu_gate
    assert report.interventions == 1_000
    assert report.deterministic_failures == 0
    assert all(count > 0 for count in report.dependency_edge_counts.values())
    assert report.event_raf > 1.0
