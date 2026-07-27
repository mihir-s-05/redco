from __future__ import annotations

import pytest

from redco.analysis.stochastic_replay_equivalence import (
    _equivalence_interval,
    run_stochastic_replay_equivalence,
)


def test_equivalence_interval_uses_distribution_free_bound() -> None:
    interval = _equivalence_interval(
        [1] * 1000,
        [1] * 1000,
        confidence=0.9,
        margin=0.2,
    )

    assert interval.mean_difference == 0
    assert interval.lower == -interval.half_width
    assert interval.upper == interval.half_width
    assert interval.passed


def test_stochastic_replay_campaign_is_reproducible() -> None:
    first = run_stochastic_replay_equivalence(
        master_seed="test-stochastic-replay",
        program_seed=20260727,
        programs=20,
        alternatives_per_program=4,
        confidence=0.9,
        overall_margin=0.9,
        route_margin=0.9,
    )
    second = run_stochastic_replay_equivalence(
        master_seed="test-stochastic-replay",
        program_seed=20260727,
        programs=20,
        alternatives_per_program=4,
        confidence=0.9,
        overall_margin=0.9,
        route_margin=0.9,
    )

    assert first.unsigned_dict() | {"generated_at_utc": ""} == (
        second.unsigned_dict() | {"generated_at_utc": ""}
    )
    assert first.deterministic_state_mismatches == 0
    assert first.topology_divergences > 0


def test_stochastic_replay_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="need two programs"):
        run_stochastic_replay_equivalence(
            master_seed="seed",
            program_seed=1,
            programs=1,
            alternatives_per_program=1,
            confidence=0.9,
            overall_margin=0.1,
            route_margin=0.2,
        )
