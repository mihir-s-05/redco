from __future__ import annotations

from pathlib import Path

import pytest

from redco.analysis.stage_d_target_eligibility import (
    _snapshot,
    aggregate_support,
)
from redco.integrations.verifiers_trace import RecordedPolicyCall


def _call(
    index: int,
    *,
    depth: int,
    action: tuple[int, ...] = (90,),
) -> RecordedPolicyCall:
    return RecordedPolicyCall(
        trace_id="trace-1",
        call_index=index,
        node_index=index,
        component_root_node=0,
        prompt_token_ids=(10 + index,),
        action_token_ids=action,
        checkpoint_id="checkpoint",
        decoding_config_hash="decode",
        event_seed=100 + index,
        prompt_tokens_reported=1,
        completion_tokens_reported=len(action),
        cost_reported=None,
        wall_seconds=0.0,
        agent_depth=depth,
        session_id=f"session-{index}",
        turn_index=0,
        call_kind="policy",
        parent_session_id="root" if depth == 1 else None,
        parent_turn_index=0 if depth == 1 else None,
    )


def test_snapshot_excludes_target_action_and_reward(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text('{"trace":1}\n', encoding="utf-8")
    prefix = _call(0, depth=0, action=(1, 2))
    target = _call(1, depth=1, action=(7, 8, 9))
    raw_trace = {
        "task": {
            "data": {
                "paper": "paper bytes",
                "paper_id": "paper-1",
                "example_id": "example-1",
                "snapshot_sha256": "a" * 64,
            }
        }
    }

    snapshot = _snapshot(
        trace_path=trace_path,
        raw_trace=raw_trace,
        target=target,
        target_node_id="policy:target",
        calls=(prefix, target),
    )

    payload = snapshot["payload"]
    assert payload["selector"]["maximum_targets_per_rollout"] == 1
    assert payload["target"]["native_call_index"] == 1
    assert "action_token_ids" not in payload["target"]
    assert payload["exact_prefix_policy_cache"][0]["action_token_ids"] == [1, 2]
    assert snapshot["bytes"] > 0
    assert len(snapshot["sha256"]) == 64


def test_support_aggregate_requires_58_of_64() -> None:
    passing = [
        {
            "trace_id": f"trace-{index}",
            "eligible": index < 60,
            "informative": index < 58,
            "joint_eligible_and_informative": index < 58,
        }
        for index in range(64)
    ]
    report = aggregate_support(passing)
    assert report["eligible"] == 60
    assert report["joint_eligible_and_informative"] == 58
    assert report["passes"]

    failing = [dict(row) for row in passing]
    failing[57]["informative"] = False
    failing[57]["joint_eligible_and_informative"] = False
    assert not aggregate_support(failing)["passes"]


def test_support_aggregate_rejects_duplicate_or_partial_blocks() -> None:
    with pytest.raises(ValueError, match="exactly 64"):
        aggregate_support([])
    duplicate = [
        {
            "trace_id": "same",
            "eligible": True,
            "informative": True,
            "joint_eligible_and_informative": True,
        }
        for _ in range(64)
    ]
    with pytest.raises(ValueError, match="unique"):
        aggregate_support(duplicate)
