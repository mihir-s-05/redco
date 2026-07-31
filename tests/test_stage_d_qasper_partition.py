from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.build_stage_d_qasper_partition_v2 import PARTITION_SIZES, partition


def _row(index: int, split: str, answer_type: str) -> dict:
    return {
        "example_id": f"example-{index}",
        "paper_id": f"paper-{index}",
        "split": split,
        "answer_type": answer_type,
    }


def test_partition_is_deterministic_and_disjoint() -> None:
    kinds = ("abstractive", "yes_no", "extractive")
    rows = [
        _row(index, "train", kinds[index % len(kinds)])
        for index in range(32)
    ] + [
        _row(index, "validation", kinds[index % len(kinds)])
        for index in range(32, 64)
    ]
    first = partition(rows)
    second = partition(json.loads(json.dumps(rows)))
    assert first == second
    assert {
        name: sum(row["split"] == name for row in first)
        for name in PARTITION_SIZES
    } == PARTITION_SIZES
    partitions = {
        name: {
            row["paper_id"] for row in first if row["split"] == name
        }
        for name in PARTITION_SIZES
    }
    assert all(
        not partitions[left] & partitions[right]
        for index, left in enumerate(PARTITION_SIZES)
        for right in list(PARTITION_SIZES)[index + 1 :]
    )
