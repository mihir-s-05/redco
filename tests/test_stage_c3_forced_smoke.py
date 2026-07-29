import json
from pathlib import Path
from typing import Any

from redco.analysis.stage_c3_forced_smoke import verify


def _trace(
    *,
    episode_id: str,
    episode_index: int,
    role: str,
    reply: str,
    reward: float | None = None,
) -> dict[str, Any]:
    rewards = {} if reward is None else {"deterministic_reward": reward}
    return {
        "info": {"episode_id": episode_id},
        "agent": {
            "name": role,
            "sampling": {
                "extra_body": {
                    "cache_salt": f"0:episode:{episode_index}:redco:test"
                }
            },
        },
        "nodes": [
            {
                "message": {"content": reply},
                "token_ids": [1, 2, 3],
            }
        ],
        "rewards": rewards,
    }


def test_forced_smoke_verifier_accepts_exact_three_regions(
    tmp_path: Path,
) -> None:
    expected = (
        ("<route>gamma</route>", "5", 1.0),
        ("<route>delta</route>", "1", 1.0),
        ("<route>gamma</route>", "0", 0.0),
        ("<route>alpha</route>", "2", 0.0),
        ("<route>beta</route>", "3", 0.0),
        ("<route>gamma</route>", "4", 0.0),
        ("<route>alpha</route>", "6", 0.0),
        ("<route>beta</route>", "7", 0.0),
    )
    traces = []
    for index, (route, action, reward) in enumerate(expected):
        episode_id = f"episode-{index}"
        traces.append(
            _trace(
                episode_id=episode_id,
                episode_index=index,
                role="context",
                reply=route,
            )
        )
        traces.append(
            _trace(
                episode_id=episode_id,
                episode_index=index,
                role="original",
                reply=action,
                reward=reward,
            )
        )
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_text(
        "".join(json.dumps(trace) + "\n" for trace in traces),
        encoding="utf-8",
    )
    invariant_path = tmp_path / "invariant.json"
    invariant_path.write_text(
        json.dumps(
            {
                "passed": True,
                "observed": {
                    "reward_min": 0.0,
                    "reward_max": 1.0,
                    "trainable_fraction": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    result = verify(traces_path, invariant_path)

    assert result["passed"] is True
    assert result["checks"]["forced_outputs_and_rewards_exact"] is True
