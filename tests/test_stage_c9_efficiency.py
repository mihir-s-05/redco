from __future__ import annotations

import json
from pathlib import Path

import msgspec
from prime_rl.transport import TrainingBatch, TrainingSample

from redco.analysis.stage_c9_efficiency import (
    _auc,
    _policy_point,
    _reuse_contract,
)


def _sample(token: int) -> TrainingSample:
    return TrainingSample(
        token_ids=[1, token],
        mask=[False, True],
        logprobs=[0.0, -0.5],
        temperatures=[1.0, 2.0],
        env_name="redco-credit",
        advantages=[0.0, 1.0],
    )


def test_policy_point_separates_delta_nuisance_from_causal_routes() -> None:
    initial = {
        "delta": {
            "route": "delta",
            "probabilities": {"4": 0.5, "5": 0.5},
        },
        "alpha": {
            "route": "alpha",
            "probabilities": {"4": 0.8, "5": 0.2},
        },
    }
    current = {
        "delta": {
            "route": "delta",
            "probabilities": {"4": 0.5, "5": 0.5},
        },
        "alpha": {
            "route": "alpha",
            "probabilities": {"4": 0.4, "5": 0.6},
        },
    }

    point = _policy_point(initial, current)

    assert point["causal_non_delta_target_mass"] == 0.6
    assert point["delta_nuisance_js_from_initial"] == 0.0


def test_auc_is_normalized_over_policy_calls() -> None:
    points = [
        {"policy_calls": 0, "causal_non_delta_target_mass": 0.0},
        {"policy_calls": 96, "causal_non_delta_target_mass": 1.0},
        {"policy_calls": 192, "causal_non_delta_target_mass": 1.0},
    ]
    assert _auc(points) == 0.75


def test_reuse_contract_checks_pair_identity_and_snapshot_progression(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    for collection in range(1, 7):
        first = collection * 2 - 1
        second = collection * 2
        examples = [_sample(collection)]
        for step in (first, second):
            step_dir = (
                run_dir / "run_default" / "rollouts" / f"step_{step}"
            )
            step_dir.mkdir(parents=True)
            (step_dir / "train_rollouts.bin").write_bytes(
                msgspec.msgpack.encode(
                    TrainingBatch(examples=examples, step=step)
                )
            )
        trace = (
            run_dir
            / "run_default"
            / "rollouts"
            / f"step_{first}"
            / "train"
            / "all"
            / "traces.jsonl"
        )
        trace.parent.mkdir(parents=True)
        trace.write_text(
            json.dumps({"policy_version": first - 1}) + "\n",
            encoding="utf-8",
        )

    contract = _reuse_contract(run_dir)

    assert contract["all_pairs_passed"]
    assert contract["fresh_example_stream_between_collections"]
