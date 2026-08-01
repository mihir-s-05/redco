from __future__ import annotations

from types import SimpleNamespace

from redco.analysis import stage_d_all_child_support_v2 as support_v2
from redco.env.tracer import EventNodeKind


def _call(call_index: int, call_kind: str) -> SimpleNamespace:
    return SimpleNamespace(
        call_index=call_index,
        agent_depth=1,
        session_id=f"session-{call_index}",
        turn_index=call_index,
        call_kind=call_kind,
        parent_session_id="root",
        parent_turn_index=0,
        parent_tool_call_id="call_0",
        invocation_id=f"child-{call_index}",
        prompt_token_ids=(1, call_index),
        checkpoint_id="model",
        decoding_config_hash="a" * 64,
        event_seed=call_index + 1,
    )


def test_v2_precommit_excludes_depth_one_compaction(monkeypatch, tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    calls = (_call(0, "policy"), _call(1, "compaction"))
    graph = SimpleNamespace(
        nodes={
            "policy": SimpleNamespace(
                kind=EventNodeKind.POLICY,
                node_id="trace:policy:0",
                metadata={"call_index": 0},
            ),
            "compaction": SimpleNamespace(
                kind=EventNodeKind.POLICY,
                node_id="trace:compaction:1",
                metadata={"call_index": 1},
            ),
        }
    )
    monkeypatch.setattr(
        support_v2,
        "load_trace_records",
        lambda _: [{"id": "trace", "task": {"data": {"paper_id": "paper"}}}],
    )
    monkeypatch.setattr(support_v2, "audit_trace_file", lambda _: SimpleNamespace(calls=calls))
    monkeypatch.setattr(
        support_v2,
        "import_trace_file",
        lambda _: SimpleNamespace(traces=(SimpleNamespace(graph=graph),)),
    )

    report = support_v2.precommit_all_depth_one_policy_targets(trace)

    assert report["candidate_count"] == 1
    assert report["candidates"][0]["call_kind"] == "policy"
    assert report["excluded_call_kinds"] == ["compaction"]
