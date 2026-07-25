from __future__ import annotations

import pytest

from redco.env.tasks.credit_probes import planted_needle, redundancy, spurious_correlation


def test_planted_needle_has_enumerable_ground_truth() -> None:
    probe = planted_needle(chunk_count=4, needle_chunk=2)

    assert probe.q_values([0]) == {"0": 0.0, "1": 0.0, "2": 1.0, "3": 0.0}
    assert probe.advantages([0])["2"] == pytest.approx(0.75)


def test_redundancy_and_spurious_probes_separate_causation() -> None:
    redundant = redundancy().q_values([0])
    spurious = spurious_correlation().advantages(range(100))

    assert redundant == {"none": 0.0, "left": 1.0, "right": 1.0, "both": 1.0}
    assert spurious == pytest.approx(
        {"spurious_absent": 0.0, "spurious_present": 0.0}
    )

