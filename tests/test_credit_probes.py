from __future__ import annotations

import pytest

from redco.env.tasks.credit_probes import (
    credit_probe_by_name,
    integration_planted_needle,
    planted_needle,
    redundancy,
    spurious_correlation,
    standard_credit_probes,
)


def test_planted_needle_has_enumerable_ground_truth() -> None:
    probe = planted_needle(chunk_count=4, needle_chunk=2)

    assert probe.q_values([0]) == {"0": 0.0, "1": 0.0, "2": 1.0, "3": 0.0}
    assert probe.advantages([0])["2"] == pytest.approx(0.75)


def test_integration_needle_is_signal_rich_but_not_in_learning_suite() -> None:
    probe = integration_planted_needle()

    assert probe.name == "integration_planted_needle"
    assert probe.q_values([0])["1"] == 1.0
    assert sum(probe.q_values([0]).values()) == 1.0
    assert probe.name not in {item.name for item in standard_credit_probes()}
    resolved = credit_probe_by_name(probe.name)
    assert resolved.actions == probe.actions
    assert resolved.q_values([0]) == probe.q_values([0])


def test_redundancy_and_spurious_probes_separate_causation() -> None:
    redundant = redundancy().q_values([0])
    spurious = spurious_correlation().advantages(range(100))

    assert redundant == {"none": 0.0, "left": 1.0, "right": 1.0, "both": 1.0}
    assert spurious == pytest.approx(
        {"spurious_absent": 0.0, "spurious_present": 0.0}
    )


def test_standard_probe_suite_covers_dependency_traps() -> None:
    probes = standard_credit_probes()
    names = {probe.name for probe in probes}

    assert len(probes) == 8
    assert {
        "control_flow_trap",
        "aliasing_trap",
        "observation_trap",
        "side_effect_ordering_trap",
        "resource_dependency_trap",
    } <= names
    assert all(probe.q_values(range(1, 101)) for probe in probes)


def test_restricted_probe_replay_is_exact_and_invalid_actions_are_scored() -> None:
    for probe in standard_credit_probes():
        for action in (*probe.actions, None, "invalid-action"):
            full = probe.replay_reward(action, 7, mode="full_suffix")
            sliced = probe.replay_reward(action, 7, mode="sliced")
            assert sliced == full
        assert probe.replay_reward(None, 7, mode="sliced") == 0.0
        assert credit_probe_by_name(probe.name).actions == probe.actions
