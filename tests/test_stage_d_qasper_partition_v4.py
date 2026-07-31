from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.build_stage_d_qasper_partition_v4 import (
    PARTITION_SIZES,
    partition,
)


def _row(index: int, split: str) -> dict:
    return {
        "paper_id": f"paper-{index}",
        "example_id": f"example-{index}",
        "answer_type": (
            "extractive",
            "abstractive",
            "yes_no",
        )[index % 3],
        "split": split,
    }


def test_v4_uses_64_unique_power_papers() -> None:
    rows = [_row(index, "train") for index in range(88)] + [
        _row(index, "validation") for index in range(88, 120)
    ]
    output = partition(rows)
    assert output == partition(json.loads(json.dumps(rows)))
    assert {
        name: sum(row["split"] == name for row in output)
        for name in PARTITION_SIZES
    } == PARTITION_SIZES
    power = [
        row["paper_id"]
        for row in output
        if row["split"] == "power_audit"
    ]
    assert len(power) == len(set(power)) == 64
