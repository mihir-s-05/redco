from __future__ import annotations

import json
from pathlib import Path

from redco.analysis.stage_c_learning_gate import _eval_rewards, _paired_bootstrap


def test_paired_bootstrap_preserves_task_pairing() -> None:
    reference = {(f"probe-{index}", index): 0.0 for index in range(32)}
    candidate = {key: 0.25 for key in reference}

    interval = _paired_bootstrap(
        candidate,
        reference,
        confidence=0.9,
        samples=1_000,
        seed=42,
    )

    assert interval.estimate == 0.25
    assert interval.lower == 0.25
    assert interval.upper == 0.25


def test_eval_rewards_uses_only_original_role(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for index in range(32):
        task = {
            "data": {
                "probe_name": f"probe-{index}",
                "exogenous_seed": index,
            }
        }
        rows.extend(
            [
                {
                    "agent": {"name": "context"},
                    "task": task,
                    "rewards": {"trajectory_reward": -99.0},
                    "info": {"policy_version": 4},
                    "calls": [{}],
                },
                {
                    "agent": {"name": "original"},
                    "task": task,
                    "rewards": {"deterministic_reward": float(index)},
                    "info": {"policy_version": 4},
                    "calls": [{}],
                },
            ]
        )
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    rewards = _eval_rewards(path, expected_policy_version=4)

    assert len(rewards) == 32
    assert rewards[("probe-7", 7)] == 7.0
