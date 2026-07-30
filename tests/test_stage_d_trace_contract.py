from __future__ import annotations

from redco.analysis.stage_d_trace_contract import audit_rlm_trace


def _node(sampled_tokens: int) -> dict:
    return {
        "sampled": True,
        "token_ids": list(range(sampled_tokens + 2)),
        "mask": [False, False] + [True] * sampled_tokens,
        "logprobs": [-0.1] * sampled_tokens,
    }


def _call(
    node: int,
    depth: int,
    session: str,
    *,
    parent: str | None = None,
) -> dict:
    structure = {
        "depth": depth,
        "session_id": session,
        "turn": 0,
        "call_kind": "policy",
    }
    if parent is not None:
        structure["parent_session_id"] = parent
        structure["parent_turn"] = 0
    return {
        "node": node,
        "sampling": {"seed": 1234},
        "rlm": structure,
    }


def test_real_rlm_contract_requires_linked_children_and_checkpoint() -> None:
    trace = {
        "nodes": [_node(3), _node(2), _node(2), _node(1)],
        "calls": [
            _call(0, 0, "root"),
            _call(1, 1, "child-a", parent="root"),
            _call(2, 1, "child-b", parent="root"),
            _call(3, 0, "root"),
        ],
        "info": {"policy_version": 7},
    }
    result = audit_rlm_trace(trace)
    assert result.trace_contract_passes
    assert result.stage_d_science_ready
    assert result.child_calls == 2
    assert result.linked_child_calls == 2


def test_eval_trace_without_checkpoint_is_only_partial_pass() -> None:
    trace = {
        "nodes": [_node(3), _node(2), _node(2), _node(1)],
        "calls": [
            _call(0, 0, "root"),
            _call(1, 1, "child-a", parent="root"),
            _call(2, 1, "child-b", parent="root"),
            _call(3, 0, "root"),
        ],
        "info": {},
    }
    result = audit_rlm_trace(trace)
    assert result.trace_contract_passes
    assert not result.stage_d_science_ready
