from __future__ import annotations

from redco.analysis.replay_equivalence import run_randomized_equivalence


def test_one_thousand_randomized_interventions_are_exact() -> None:
    report = run_randomized_equivalence(
        seed=7,
        program_count=100,
        interventions_per_program=10,
        events_per_program=10,
    )

    assert report.passed
    assert report.interventions == 1_000
    assert report.failures == ()
    assert report.sliced_suffix_events < report.full_suffix_events
    assert 0.0 <= report.mean_sliced_work_fraction < 1.0

