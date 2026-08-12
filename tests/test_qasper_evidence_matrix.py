import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))
from run_qasper_allocation_sweep import _load_sweep_config
from run_qasper_evidence_matrix import _load_matrix_config, _paired_summary


def _arm(name: str, before: int, after: int) -> dict[str, object]:
    return {
        "arm": name,
        "evaluation_after": {"exact_evidence": after},
        "evaluation_before": {"exact_evidence": before},
    }


def test_matrix_config_freezes_the_exact_seed_block(tmp_path: Path) -> None:
    path = Path("configs/qasper-evidence-matrix-v1.json")
    assert _load_matrix_config(path)["seeds"] == list(range(20260812, 20260817))

    changed = json.loads(path.read_bytes())
    changed["seeds"][-1] += 1
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="exact reviewed five-seed block"):
        _load_matrix_config(changed_path)


def test_paired_summary_uses_five_seed_level_differences() -> None:
    post_differences = (-1, 0, 1, 2, 3)
    runs = []
    for index, difference in enumerate(post_differences):
        runs.append(
            {
                "arms": [
                    _arm("trajectory_loo", 10, 11),
                    _arm("redco", 10, 11 + difference),
                ],
                "seed": 20260812 + index,
            }
        )

    summary = _paired_summary(runs)
    assert summary["inferential_unit"] == "seed"
    assert summary["n"] == 5
    assert summary["paired_post_exact_evidence"] == {
        "differences": list(post_differences),
        "mean": 1.0,
        "median": 1,
    }
    assert summary["paired_change_from_baseline"] == {
        "differences": list(post_differences),
        "mean": 1.0,
        "median": 1,
    }


def test_allocation_sweep_config_freezes_the_frontier(tmp_path: Path) -> None:
    path = Path("configs/qasper-allocation-sweep-v1.json")
    config = _load_sweep_config(path)
    assert config["arms"] == [
        "trajectory_loo",
        "branch_4_2",
        "branch_3_4",
        "branch_2_6",
    ]

    changed = json.loads(path.read_bytes())
    changed["arms"][1:3] = reversed(changed["arms"][1:3])
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="exact reviewed allocation frontier"):
        _load_sweep_config(changed_path)
