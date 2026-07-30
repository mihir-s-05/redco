from __future__ import annotations

import pytest

from redco.analysis.stage_d_ledger import (
    ResourceMeters,
    build_stage_d_ledger,
)


def _call(depth: int, prompt: int, completion: int) -> dict:
    return {
        "rlm": {"depth": depth},
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
        },
    }


def test_stage_d_ledger_separates_every_policy_and_judge_role() -> None:
    train = [
        {
            "traces": [
                {
                    "calls": [_call(0, 10, 3), _call(1, 8, 5)],
                    "extra_usage": [
                        {"prompt_tokens": 20, "completion_tokens": 2}
                    ],
                },
                {
                    "info": {"record_kind": "branch"},
                    "calls": [_call(0, 7, 4)],
                },
            ]
        }
    ]
    evaluation = [{"calls": [_call(0, 11, 6), _call(1, 9, 2)]}]
    ledger = build_stage_d_ledger(
        train,
        evaluation,
        ResourceMeters(
            optimizer_updates=2,
            service_seconds=8,
            wall_seconds=10,
            gpu_seconds=20,
            storage_bytes=100,
        ),
    )
    assert ledger.usage["root"].completion_tokens == 3
    assert ledger.usage["child"].completion_tokens == 5
    assert ledger.usage["branch_continuation"].completion_tokens == 4
    assert ledger.usage["evaluation"].completion_tokens == 8
    assert ledger.usage["judge"].completion_tokens == 2
    assert ledger.training_generated_tokens == 12
    assert ledger.to_dict()["primary"]["gpu_hours"] == pytest.approx(20 / 3600)


def test_stage_d_ledger_fails_closed_on_missing_usage() -> None:
    with pytest.raises(ValueError, match="must record usage"):
        build_stage_d_ledger(
            [{"calls": [{"rlm": {"depth": 0}}]}],
            [],
            ResourceMeters(0, 0, 0, 0, 0),
        )


def test_stage_d_resource_meters_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        ResourceMeters(
            optimizer_updates=0,
            service_seconds=0,
            wall_seconds=-1,
            gpu_seconds=0,
            storage_bytes=0,
        )
