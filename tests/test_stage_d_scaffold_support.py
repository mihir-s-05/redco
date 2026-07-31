from __future__ import annotations

from redco.analysis.stage_d_scaffold_support import _percentile


def test_nearest_rank_percentile_is_conservative() -> None:
    assert _percentile([1] * 60 + [2] * 3 + [3], 0.95) == 2
    assert _percentile([1] * 60 + [3] * 4, 0.95) == 3
