from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from redco.analysis.rlm_episode_replay import (
    CounterfactualCompletionRouter,
    EpisodeReplayIneligibility,
    RLMEventAddress,
    ScriptedCompletionRouter,
    ScriptedEvent,
    _child_lineage,
    _counterfactual_seed,
    _request_projection_sha256,
    inject_child_answer,
    inject_committed_child_answer,
    load_scripted_events,
    recorded_event_addresses,
    trace_to_scripted_events,
)
from redco.analysis.stage_d_child_consumers import analyze
from redco.integrations.signed_subprocess import sign_payload


def _headers(invocation_id: str) -> dict[str, str]:
    return {
        "X-RLM-Depth": "1",
        "X-RLM-Turn": "0",
        "X-RLM-Call-Kind": "policy",
        "X-RLM-Session-ID": f"session-{invocation_id}",
        "X-RLM-Parent-Session-ID": "root-session",
        "X-RLM-Parent-Turn": "3",
        "X-RLM-Parent-Tool-Call-ID": "call_0",
        "X-RLM-Invocation-ID": invocation_id,
    }


def test_recursive_address_requires_explicit_invocation_identity() -> None:
    with pytest.raises(EpisodeReplayIneligibility, match="invocation ID"):
        RLMEventAddress(depth=1, turn=0, call_kind="policy", parent_turn=0)


def test_request_headers_require_explicit_call_kind() -> None:
    with pytest.raises(EpisodeReplayIneligibility, match="Call-Kind"):
        RLMEventAddress.from_headers(
            {
                "X-RLM-Depth": "0",
                "X-RLM-Turn": "0",
                "X-RLM-Session-ID": "root",
            }
        )


def test_router_uses_invocation_id_not_duplicate_answer_text() -> None:
    child_events = tuple(
        ScriptedEvent(
            address=RLMEventAddress(
                depth=1,
                turn=0,
                call_kind="policy",
                parent_lineage="root",
                parent_turn=3,
                parent_tool_call_id="call_0",
                invocation_id=invocation_id,
            ),
            message={"role": "assistant", "content": "duplicate"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )
        for invocation_id in ("shard-0", "shard-1")
    )
    root = ScriptedEvent(
        address=RLMEventAddress(depth=0, turn=0, call_kind="policy"),
        message={"role": "assistant", "content": "root"},
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
    )
    events = (root, *child_events)
    router = ScriptedCompletionRouter(events)
    router.respond(
        headers={
            "X-RLM-Depth": "0",
            "X-RLM-Turn": "0",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Session-ID": "root-session",
        },
        request={"model": "m", "messages": []},
    )

    first = router.respond(headers=_headers("shard-1"), request={"model": "m", "messages": []})
    second = router.respond(headers=_headers("shard-0"), request={"model": "m", "messages": []})

    assert first["choices"][0]["message"]["content"] == "duplicate"
    assert second["choices"][0]["message"]["content"] == "duplicate"
    assert router.audit()["seen_addresses"] == sorted(event.address.key() for event in events)


def test_router_rejects_changed_retry_body() -> None:
    event = ScriptedEvent(
        address=RLMEventAddress(depth=0, turn=0, call_kind="policy"),
        message={"role": "assistant", "content": "done"},
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
    )
    router = ScriptedCompletionRouter((event,))
    headers = {
        "X-RLM-Depth": "0",
        "X-RLM-Turn": "0",
        "X-RLM-Call-Kind": "policy",
        "X-RLM-Session-ID": "root",
    }
    router.respond(headers=headers, request={"model": "one", "messages": []})
    with pytest.raises(EpisodeReplayIneligibility, match="different retry"):
        router.respond(headers=headers, request={"model": "two", "messages": []})


def test_router_rejects_an_identical_duplicate_request() -> None:
    event = ScriptedEvent(
        address=RLMEventAddress(depth=0, turn=0, call_kind="policy"),
        message={"role": "assistant", "content": "done"},
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
    )
    router = ScriptedCompletionRouter((event,))
    headers = {
        "X-RLM-Depth": "0",
        "X-RLM-Turn": "0",
        "X-RLM-Call-Kind": "policy",
        "X-RLM-Session-ID": "root",
    }
    request = {"model": "one", "messages": []}
    router.respond(headers=headers, request=request)
    with pytest.raises(EpisodeReplayIneligibility, match="more than once"):
        router.respond(headers=headers, request=request)


def test_wire_equivalent_tool_messages_have_one_request_key() -> None:
    trace_form = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "name": "ipython",
                        "arguments": '{"code":"1+1"}',
                    }
                ],
            },
            {
                "role": "tool",
                "name": "ipython",
                "tool_call_id": "call_0",
                "content": "2",
            },
        ],
    }
    client_form = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "ipython",
                            "arguments": '{"code":"1+1"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": "2"},
        ],
    }
    assert _request_projection_sha256(trace_form) == _request_projection_sha256(client_form)


def test_system_paths_are_only_normalized_for_engineering_transport_audit() -> None:
    first = {
        "model": "m",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Working directory: /tmp/one\nConversation log: /tmp/one/messages.jsonl\n"
                ),
            }
        ],
    }
    second = {
        "model": "m",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Working directory: /tmp/two\nConversation log: /tmp/two/messages.jsonl\n"
                ),
            }
        ],
    }
    assert _request_projection_sha256(first) != _request_projection_sha256(second)
    assert _request_projection_sha256(
        first, normalize_transport_paths=True
    ) == _request_projection_sha256(second, normalize_transport_paths=True)


def test_continuation_seed_ignores_generated_ids_but_binds_trace_and_target() -> None:
    first = RLMEventAddress(
        depth=1,
        turn=2,
        call_kind="policy",
        parent_lineage="root",
        parent_turn=0,
        parent_tool_call_id="generated-a",
        invocation_id="generated-x",
    )
    second = RLMEventAddress(
        depth=1,
        turn=2,
        call_kind="policy",
        parent_lineage="root",
        parent_turn=9,
        parent_tool_call_id="generated-b",
        invocation_id="generated-y",
    )
    kwargs = {"trace_id": "trace-a", "target_id": "target-a"}
    seed = _counterfactual_seed("master", address=first, **kwargs)
    assert seed == _counterfactual_seed("master", address=second, **kwargs)
    assert seed != _counterfactual_seed(
        "master", address=first, trace_id="trace-b", target_id="target-a"
    )
    assert seed != _counterfactual_seed(
        "master", address=first, trace_id="trace-a", target_id="target-b"
    )


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("seed", 8),
        ("temperature", 0.8),
        ("top_p", 0.9),
        ("max_tokens", 33),
        ("stop", ["END"]),
        ("presence_penalty", 0.1),
        ("response_format", {"type": "json_object"}),
        ("future_behavior_option", {"enabled": True}),
    ],
)
def test_request_key_binds_policy_relevant_decoding_fields(field: str, changed: object) -> None:
    base = {
        "model": "m",
        "messages": [{"role": "user", "content": "q"}],
        "seed": 7,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 32,
        "stop": None,
        "presence_penalty": 0.0,
        "response_format": {"type": "text"},
    }
    modified = {**base, field: changed}
    assert _request_projection_sha256(base) != _request_projection_sha256(modified)


def test_router_rejects_a_second_root_session() -> None:
    event = ScriptedEvent(
        address=RLMEventAddress(depth=0, turn=0, call_kind="policy"),
        message={"role": "assistant", "content": "done"},
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
    )
    router = ScriptedCompletionRouter((event,))
    base = {
        "X-RLM-Depth": "0",
        "X-RLM-Turn": "0",
        "X-RLM-Call-Kind": "policy",
    }
    router.respond(
        headers={**base, "X-RLM-Session-ID": "root-one"},
        request={"model": "one", "messages": []},
    )
    with pytest.raises(EpisodeReplayIneligibility, match="multiple root sessions"):
        router.respond(
            headers={**base, "X-RLM-Session-ID": "root-two"},
            request={"model": "one", "messages": []},
        )


def test_child_session_scope_includes_parent_event() -> None:
    events = []
    for parent_turn, tool_call_id in ((0, "call_0"), (1, "call_1")):
        events.append(
            ScriptedEvent(
                address=RLMEventAddress(
                    depth=1,
                    turn=0,
                    call_kind="policy",
                    parent_lineage="root",
                    parent_turn=parent_turn,
                    parent_tool_call_id=tool_call_id,
                    invocation_id="midpoint-shard-0",
                ),
                message={"role": "assistant", "content": "child"},
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )
        )
    roots = tuple(
        ScriptedEvent(
            address=RLMEventAddress(depth=0, turn=turn, call_kind="policy"),
            message={"role": "assistant", "content": "root"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )
        for turn in (0, 1)
    )
    router = ScriptedCompletionRouter((*roots, *events))
    root_headers = {
        "X-RLM-Depth": "0",
        "X-RLM-Call-Kind": "policy",
        "X-RLM-Session-ID": "root-session",
    }
    for turn in (0, 1):
        router.respond(
            headers={**root_headers, "X-RLM-Turn": str(turn)},
            request={"model": "m", "messages": []},
        )
        router.respond(
            headers={
                "X-RLM-Depth": "1",
                "X-RLM-Turn": "0",
                "X-RLM-Call-Kind": "policy",
                "X-RLM-Session-ID": f"child-{turn}",
                "X-RLM-Parent-Session-ID": "root-session",
                "X-RLM-Parent-Turn": str(turn),
                "X-RLM-Parent-Tool-Call-ID": f"call_{turn}",
                "X-RLM-Invocation-ID": "midpoint-shard-0",
            },
            request={"model": "m", "messages": []},
        )
    assert router.audit()["complete"] is True


def test_child_injection_changes_only_the_structural_target() -> None:
    root = Path(__file__).parents[1]
    events = load_scripted_events(str(root / "tests/fixtures/stage_d_rlm_replay_cassette_v1.json"))
    target = next(event.address for event in events if event.address.invocation_id == "shard-0")

    changed = inject_child_answer(events, target=target, answer="candidate")

    differing = [
        (before, after) for before, after in zip(events, changed, strict=True) if before != after
    ]
    assert len(differing) == 1
    assert differing[0][1].address == target
    assert differing[0][1].message["content"] == "candidate"


def test_golden_cassette_covers_retry_hidden_state_and_reverse_completion() -> None:
    root = Path(__file__).parents[1]
    events = load_scripted_events(str(root / "tests/fixtures/stage_d_rlm_replay_cassette_v1.json"))
    addresses = {event.address.key() for event in events}

    assert "depth:0:root:policy:2" in addresses
    assert "depth:1:parent:root:child:0:call_0:shard-0:policy:0" in addresses
    shard_0 = next(event for event in events if event.address.invocation_id == "shard-0")
    shard_1 = next(event for event in events if event.address.invocation_id == "shard-1")
    assert shard_0.delay_seconds > shard_1.delay_seconds
    assert "deliberate retry fixture" in json.dumps(events[0].message)


def test_recovered_trace_is_parented_but_not_silently_migrated() -> None:
    root = Path(__file__).parents[1]
    trace = root / (
        "runs/stage-d0/all-child-support-v1/work/fixture-000/traces/000-9a6df158322f.jsonl"
    )
    if not trace.exists():
        pytest.skip("local terminal evidence bundle is not present")
    diagnostic = analyze(trace)
    call_two = next(row for row in diagnostic["child_consumers"] if row["call_index"] == 2)

    assert call_two["parent_turn_index"] == 0
    assert call_two["parent_tool_node_index"] == 9
    with pytest.raises(EpisodeReplayIneligibility, match="invocation ID"):
        recorded_event_addresses(trace)
    with pytest.raises(EpisodeReplayIneligibility, match="invocation ID"):
        trace_to_scripted_events(
            trace,
            expected_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
            signed_precommit=sign_payload(
                {
                    "source_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                    "candidates": [],
                }
            ),
        )


def test_trace_cassette_is_hash_bound_and_requires_signed_precommit(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "traces.jsonl"
    row = {
        "id": "trace-one",
        "ok": True,
        "errors": [],
        "nodes": [
            {
                "parent": None,
                "message": {"role": "assistant", "content": "done"},
                "token_ids": [1, 2],
                "mask": [False, True],
            }
        ],
        "calls": [
            {
                "node": 0,
                "model": "fixture",
                "sampling": {"seed": 7},
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "time": {"start": 0.0, "end": 1.0},
                "rlm": {
                    "depth": 0,
                    "session_id": "root",
                    "turn": 0,
                    "call_kind": "policy",
                },
            }
        ],
    }
    trace.write_text(json.dumps(row) + "\n", encoding="utf-8")
    digest = hashlib.sha256(trace.read_bytes()).hexdigest()
    events = trace_to_scripted_events(
        trace,
        expected_sha256=digest,
        signed_precommit=sign_payload({"source_trace_sha256": digest, "candidates": []}),
    )
    assert events[0].message == {"role": "assistant", "content": "done"}
    with pytest.raises(EpisodeReplayIneligibility, match="SHA-256"):
        trace_to_scripted_events(
            trace,
            expected_sha256="0" * 64,
            signed_precommit=sign_payload({"source_trace_sha256": digest, "candidates": []}),
        )


def test_committed_injection_uses_candidate_rank_and_address() -> None:
    event = ScriptedEvent(
        address=RLMEventAddress(
            depth=1,
            turn=0,
            call_kind="policy",
            parent_lineage="root",
            parent_turn=0,
            parent_tool_call_id="call_0",
            invocation_id="shard-0",
        ),
        message={"role": "assistant", "content": "before"},
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
    )
    precommit = sign_payload(
        {
            "candidates": [
                {
                    "agent_depth": 1,
                    "turn_index": 0,
                    "call_kind": "policy",
                    "parent_turn_index": 0,
                    "parent_tool_call_id": "call_0",
                    "invocation_id": "shard-0",
                    "decision_unit_weight": {"numerator": 1, "denominator": 1},
                }
            ]
        }
    )
    changed, target, candidate = inject_committed_child_answer(
        (event,),
        signed_precommit=precommit,
        candidate_rank=0,
        answer="after",
    )
    assert target == event.address
    assert candidate["decision_unit_weight"]["numerator"] == 1
    assert changed[0].message["content"] == "after"


def test_counterfactual_router_resamples_changed_downstream_request() -> None:
    root_0_request = {"model": "m", "messages": [{"role": "user", "content": "q"}]}
    child_request = {"model": "m", "messages": [{"role": "user", "content": "child"}]}
    recorded_root_1 = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "tool", "tool_call_id": "call_0", "content": "old"},
        ],
    }
    events = (
        ScriptedEvent(
            address=RLMEventAddress(depth=0, turn=0, call_kind="policy"),
            message={"role": "assistant", "content": "root-0"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            expected_request_sha256=_request_projection_sha256(root_0_request),
        ),
        ScriptedEvent(
            address=RLMEventAddress(
                depth=1,
                turn=0,
                call_kind="policy",
                parent_lineage="root",
                parent_turn=0,
                parent_tool_call_id="call_0",
                invocation_id="shard-0",
            ),
            message={"role": "assistant", "content": "old"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            expected_request_sha256=_request_projection_sha256(child_request),
        ),
        ScriptedEvent(
            address=RLMEventAddress(depth=0, turn=1, call_kind="policy"),
            message={"role": "assistant", "content": "old-final"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            expected_request_sha256=_request_projection_sha256(recorded_root_1),
        ),
    )
    forwarded: list[tuple[RLMEventAddress, int]] = []

    def generate(
        request: dict[str, object], address: RLMEventAddress, seed: int
    ) -> dict[str, object]:
        assert request["seed"] == seed
        forwarded.append((address, seed))
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "fresh-final"},
                    "finish_reason": "stop",
                }
            ]
        }

    target = events[1].address
    router = CounterfactualCompletionRouter(
        events,
        target=target,
        candidate_message={"role": "assistant", "content": "new"},
        candidate_finish_reason="stop",
        candidate_prompt_tokens=1,
        candidate_completion_tokens=1,
        master_seed="master",
        trace_id="trace-one",
        target_id="target-one",
        generator=generate,
    )
    root_headers = {
        "X-RLM-Depth": "0",
        "X-RLM-Call-Kind": "policy",
        "X-RLM-Session-ID": "root",
    }
    router.respond(headers={**root_headers, "X-RLM-Turn": "0"}, request=root_0_request)
    child_response = router.respond(
        headers={
            "X-RLM-Depth": "1",
            "X-RLM-Turn": "0",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Session-ID": "child",
            "X-RLM-Parent-Session-ID": "root",
            "X-RLM-Parent-Turn": "0",
            "X-RLM-Parent-Tool-Call-ID": "call_0",
            "X-RLM-Invocation-ID": "shard-0",
        },
        request=child_request,
    )
    changed_root_1 = {
        **recorded_root_1,
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "tool", "tool_call_id": "call_0", "content": "new"},
        ],
    }
    final = router.respond(headers={**root_headers, "X-RLM-Turn": "1"}, request=changed_root_1)
    assert child_response["choices"][0]["message"]["content"] == "new"
    assert final["choices"][0]["message"]["content"] == "fresh-final"
    assert forwarded[0][0].turn == 1
    assert router.audit()["valid_counterfactual"] is True


def test_terminal_intervention_is_valid_without_downstream_resampling() -> None:
    request = {"model": "m", "messages": [{"role": "user", "content": "q"}]}
    target = RLMEventAddress(
        depth=1,
        turn=0,
        call_kind="policy",
        parent_lineage="root",
        parent_turn=0,
        parent_tool_call_id="call_0",
        invocation_id="shard-0",
    )
    event = ScriptedEvent(
        address=target,
        message={"role": "assistant", "content": "old"},
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
        expected_request_sha256=_request_projection_sha256(request),
    )
    root_request = {
        "model": "m",
        "messages": [{"role": "user", "content": "root"}],
    }
    root_event = ScriptedEvent(
        address=RLMEventAddress(depth=0, turn=0, call_kind="policy"),
        message={"role": "assistant", "content": "call child"},
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
        expected_request_sha256=_request_projection_sha256(root_request),
    )

    def no_generation(
        request: dict[str, object], address: RLMEventAddress, seed: int
    ) -> dict[str, object]:
        del request, address, seed
        raise AssertionError("terminal intervention must not generate")

    router = CounterfactualCompletionRouter(
        (root_event, event),
        target=target,
        candidate_message={"role": "assistant", "content": "new"},
        candidate_finish_reason="stop",
        candidate_prompt_tokens=1,
        candidate_completion_tokens=1,
        master_seed="master",
        trace_id="trace",
        target_id="target",
        generator=no_generation,
    )
    router.respond(
        headers={
            "X-RLM-Depth": "0",
            "X-RLM-Turn": "0",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Session-ID": "root",
        },
        request=root_request,
    )
    response = router.respond(
        headers={
            "X-RLM-Depth": "1",
            "X-RLM-Turn": "0",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Session-ID": "child",
            "X-RLM-Parent-Session-ID": "root",
            "X-RLM-Parent-Turn": "0",
            "X-RLM-Parent-Tool-Call-ID": "call_0",
            "X-RLM-Invocation-ID": "shard-0",
        },
        request=request,
    )
    assert response["choices"][0]["message"]["content"] == "new"
    assert router.audit()["valid_counterfactual"] is True
    assert router.audit()["resampled_policy_calls"] == 0


def test_duplicate_target_action_still_regenerates_causal_continuation() -> None:
    root_0_request = {
        "model": "m",
        "messages": [{"role": "user", "content": "q"}],
    }
    child_request = {
        "model": "m",
        "messages": [{"role": "user", "content": "child"}],
    }
    root_1_request = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "tool", "tool_call_id": "call_0", "content": "same"},
        ],
    }
    target = RLMEventAddress(
        depth=1,
        turn=0,
        call_kind="policy",
        parent_lineage="root",
        parent_turn=0,
        parent_tool_call_id="call_0",
        invocation_id="shard-0",
    )
    events = (
        ScriptedEvent(
            address=RLMEventAddress(depth=0, turn=0, call_kind="policy"),
            message={"role": "assistant", "content": "call child"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            expected_request_sha256=_request_projection_sha256(root_0_request),
        ),
        ScriptedEvent(
            address=target,
            message={"role": "assistant", "content": "same"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            expected_request_sha256=_request_projection_sha256(child_request),
        ),
        ScriptedEvent(
            address=RLMEventAddress(depth=0, turn=1, call_kind="policy"),
            message={"role": "assistant", "content": "recorded"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            expected_request_sha256=_request_projection_sha256(root_1_request),
        ),
    )
    generated: list[int] = []

    def generate(
        request: dict[str, object], address: RLMEventAddress, seed: int
    ) -> dict[str, object]:
        del request, address
        generated.append(seed)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "fresh"},
                    "finish_reason": "stop",
                }
            ]
        }

    router = CounterfactualCompletionRouter(
        events,
        target=target,
        candidate_message={"role": "assistant", "content": "same"},
        candidate_finish_reason="stop",
        candidate_prompt_tokens=1,
        candidate_completion_tokens=1,
        master_seed="master",
        trace_id="trace",
        target_id="target",
        generator=generate,
    )
    root_headers = {
        "X-RLM-Depth": "0",
        "X-RLM-Call-Kind": "policy",
        "X-RLM-Session-ID": "root",
    }
    router.respond(headers={**root_headers, "X-RLM-Turn": "0"}, request=root_0_request)
    router.respond(
        headers={
            "X-RLM-Depth": "1",
            "X-RLM-Turn": "0",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Session-ID": "child",
            "X-RLM-Parent-Session-ID": "root",
            "X-RLM-Parent-Turn": "0",
            "X-RLM-Parent-Tool-Call-ID": "call_0",
            "X-RLM-Invocation-ID": "shard-0",
        },
        request=child_request,
    )
    final = router.respond(headers={**root_headers, "X-RLM-Turn": "1"}, request=root_1_request)
    assert final["choices"][0]["message"]["content"] == "fresh"
    assert len(generated) == 1
    assert router.audit()["resampled_policy_calls"] == 1


def _session_taint_router(
    post_address: RLMEventAddress,
    post_request: dict[str, object],
) -> tuple[
    CounterfactualCompletionRouter,
    dict[str, object],
    dict[str, object],
    list[RLMEventAddress],
]:
    root_request: dict[str, object] = {
        "model": "m",
        "messages": [{"role": "user", "content": "root"}],
    }
    target_request: dict[str, object] = {
        "model": "m",
        "messages": [{"role": "user", "content": "target child"}],
    }
    target = RLMEventAddress(
        depth=1,
        turn=0,
        call_kind="policy",
        parent_lineage="root",
        parent_turn=0,
        parent_tool_call_id="call-target",
        invocation_id="target",
    )
    events = (
        ScriptedEvent(
            address=RLMEventAddress(depth=0, turn=0, call_kind="policy"),
            message={"role": "assistant", "content": "call children"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            expected_request_sha256=_request_projection_sha256(root_request),
        ),
        ScriptedEvent(
            address=target,
            message={"role": "assistant", "content": "same"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            expected_request_sha256=_request_projection_sha256(target_request),
        ),
        ScriptedEvent(
            address=post_address,
            message={"role": "assistant", "content": "recorded-post"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            expected_request_sha256=_request_projection_sha256(post_request),
        ),
    )
    generated: list[RLMEventAddress] = []

    def generate(
        request: dict[str, object], address: RLMEventAddress, seed: int
    ) -> dict[str, object]:
        assert request["seed"] == seed
        generated.append(address)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "fresh-post"},
                    "finish_reason": "stop",
                }
            ]
        }

    router = CounterfactualCompletionRouter(
        events,
        target=target,
        candidate_message={"role": "assistant", "content": "same"},
        candidate_finish_reason="stop",
        candidate_prompt_tokens=1,
        candidate_completion_tokens=1,
        master_seed="master",
        trace_id="trace",
        target_id="target",
        generator=generate,
    )
    router.respond(
        headers={
            "X-RLM-Depth": "0",
            "X-RLM-Turn": "0",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Session-ID": "root",
        },
        request=root_request,
    )
    router.respond(
        headers={
            "X-RLM-Depth": "1",
            "X-RLM-Turn": "0",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Session-ID": "target-child",
            "X-RLM-Parent-Session-ID": "root",
            "X-RLM-Parent-Turn": "0",
            "X-RLM-Parent-Tool-Call-ID": "call-target",
            "X-RLM-Invocation-ID": "target",
        },
        request=target_request,
    )
    return router, root_request, target_request, generated


def test_same_child_session_continuation_is_resampled_after_intervention() -> None:
    post_request: dict[str, object] = {
        "model": "m",
        "messages": [{"role": "user", "content": "unchanged later child turn"}],
    }
    post_address = RLMEventAddress(
        depth=1,
        turn=1,
        call_kind="policy",
        parent_lineage="root",
        parent_turn=0,
        parent_tool_call_id="call-target",
        invocation_id="target",
    )
    router, _, _, generated = _session_taint_router(post_address, post_request)
    response = router.respond(
        headers={
            "X-RLM-Depth": "1",
            "X-RLM-Turn": "1",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Session-ID": "target-child",
            "X-RLM-Parent-Session-ID": "root",
            "X-RLM-Parent-Turn": "0",
            "X-RLM-Parent-Tool-Call-ID": "call-target",
            "X-RLM-Invocation-ID": "target",
        },
        request=post_request,
    )
    assert response["choices"][0]["message"]["content"] == "fresh-post"
    assert generated == [post_address]


def test_grandchild_of_target_session_is_resampled_after_intervention() -> None:
    post_request: dict[str, object] = {
        "model": "m",
        "messages": [{"role": "user", "content": "unchanged grandchild"}],
    }
    post_address = RLMEventAddress(
        depth=2,
        turn=0,
        call_kind="policy",
        parent_lineage=_child_lineage(
            parent_lineage="root",
            depth=1,
            parent_turn=0,
            parent_tool_call_id="call-target",
            invocation_id="target",
        ),
        parent_turn=0,
        parent_tool_call_id="grandchild-call",
        invocation_id="grandchild",
    )
    router, _, _, generated = _session_taint_router(post_address, post_request)
    response = router.respond(
        headers={
            "X-RLM-Depth": "2",
            "X-RLM-Turn": "0",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Session-ID": "grandchild",
            "X-RLM-Parent-Session-ID": "target-child",
            "X-RLM-Parent-Turn": "0",
            "X-RLM-Parent-Tool-Call-ID": "grandchild-call",
            "X-RLM-Invocation-ID": "grandchild",
        },
        request=post_request,
    )
    assert response["choices"][0]["message"]["content"] == "fresh-post"
    assert generated == [post_address]


def test_independent_sibling_session_remains_eligible_for_exact_reuse() -> None:
    post_request: dict[str, object] = {
        "model": "m",
        "messages": [{"role": "user", "content": "unchanged sibling"}],
    }
    post_address = RLMEventAddress(
        depth=1,
        turn=0,
        call_kind="policy",
        parent_lineage="root",
        parent_turn=0,
        parent_tool_call_id="call-sibling",
        invocation_id="sibling",
    )
    router, _, _, generated = _session_taint_router(post_address, post_request)
    response = router.respond(
        headers={
            "X-RLM-Depth": "1",
            "X-RLM-Turn": "0",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Session-ID": "sibling-child",
            "X-RLM-Parent-Session-ID": "root",
            "X-RLM-Parent-Turn": "0",
            "X-RLM-Parent-Tool-Call-ID": "call-sibling",
            "X-RLM-Invocation-ID": "sibling",
        },
        request=post_request,
    )
    assert response["choices"][0]["message"]["content"] == "recorded-post"
    assert generated == []


def test_independent_sibling_later_turn_and_grandchild_remain_reusable() -> None:
    root_request = {"model": "m", "messages": [{"role": "user", "content": "root"}]}
    target_request = {
        "model": "m",
        "messages": [{"role": "user", "content": "target"}],
    }
    sibling_0_request = {
        "model": "m",
        "messages": [{"role": "user", "content": "sibling zero"}],
    }
    sibling_1_request = {
        "model": "m",
        "messages": [{"role": "user", "content": "sibling one"}],
    }
    grandchild_request = {
        "model": "m",
        "messages": [{"role": "user", "content": "sibling grandchild"}],
    }
    target = RLMEventAddress(
        depth=1,
        turn=0,
        call_kind="policy",
        parent_lineage="root",
        parent_turn=0,
        parent_tool_call_id="call-target",
        invocation_id="target",
    )
    sibling_lineage = _child_lineage(
        parent_lineage="root",
        depth=1,
        parent_turn=0,
        parent_tool_call_id="call-sibling",
        invocation_id="sibling",
    )
    events = (
        ScriptedEvent(
            address=RLMEventAddress(depth=0, turn=0, call_kind="policy"),
            message={"role": "assistant", "content": "root"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            expected_request_sha256=_request_projection_sha256(root_request),
        ),
        ScriptedEvent(
            address=target,
            message={"role": "assistant", "content": "target-old"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            expected_request_sha256=_request_projection_sha256(target_request),
        ),
        ScriptedEvent(
            address=RLMEventAddress(
                depth=1,
                turn=0,
                call_kind="policy",
                parent_lineage="root",
                parent_turn=0,
                parent_tool_call_id="call-sibling",
                invocation_id="sibling",
            ),
            message={"role": "assistant", "content": "sibling-zero-recorded"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            expected_request_sha256=_request_projection_sha256(sibling_0_request),
        ),
        ScriptedEvent(
            address=RLMEventAddress(
                depth=1,
                turn=1,
                call_kind="policy",
                parent_lineage="root",
                parent_turn=0,
                parent_tool_call_id="call-sibling",
                invocation_id="sibling",
            ),
            message={"role": "assistant", "content": "sibling-one-recorded"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            expected_request_sha256=_request_projection_sha256(sibling_1_request),
        ),
        ScriptedEvent(
            address=RLMEventAddress(
                depth=2,
                turn=0,
                call_kind="policy",
                parent_lineage=sibling_lineage,
                parent_turn=1,
                parent_tool_call_id="call-nested",
                invocation_id="nested",
            ),
            message={"role": "assistant", "content": "grandchild-recorded"},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            expected_request_sha256=_request_projection_sha256(grandchild_request),
        ),
    )

    def no_generation(
        request: dict[str, object], address: RLMEventAddress, seed: int
    ) -> dict[str, object]:
        del request, address, seed
        raise AssertionError("independent sibling lineage must remain reusable")

    router = CounterfactualCompletionRouter(
        events,
        target=target,
        candidate_message={"role": "assistant", "content": "target-new"},
        candidate_finish_reason="stop",
        candidate_prompt_tokens=1,
        candidate_completion_tokens=1,
        master_seed="master",
        trace_id="trace",
        target_id="target",
        generator=no_generation,
    )
    router.respond(
        headers={
            "X-RLM-Depth": "0",
            "X-RLM-Turn": "0",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Session-ID": "root",
        },
        request=root_request,
    )
    router.respond(
        headers={
            "X-RLM-Depth": "1",
            "X-RLM-Turn": "0",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Session-ID": "target-child",
            "X-RLM-Parent-Session-ID": "root",
            "X-RLM-Parent-Turn": "0",
            "X-RLM-Parent-Tool-Call-ID": "call-target",
            "X-RLM-Invocation-ID": "target",
        },
        request=target_request,
    )
    sibling_headers = {
        "X-RLM-Depth": "1",
        "X-RLM-Call-Kind": "policy",
        "X-RLM-Session-ID": "sibling-child",
        "X-RLM-Parent-Session-ID": "root",
        "X-RLM-Parent-Turn": "0",
        "X-RLM-Parent-Tool-Call-ID": "call-sibling",
        "X-RLM-Invocation-ID": "sibling",
    }
    sibling_0 = router.respond(
        headers={**sibling_headers, "X-RLM-Turn": "0"},
        request=sibling_0_request,
    )
    sibling_1 = router.respond(
        headers={**sibling_headers, "X-RLM-Turn": "1"},
        request=sibling_1_request,
    )
    grandchild = router.respond(
        headers={
            "X-RLM-Depth": "2",
            "X-RLM-Turn": "0",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Session-ID": "sibling-grandchild",
            "X-RLM-Parent-Session-ID": "sibling-child",
            "X-RLM-Parent-Turn": "1",
            "X-RLM-Parent-Tool-Call-ID": "call-nested",
            "X-RLM-Invocation-ID": "nested",
        },
        request=grandchild_request,
    )
    assert sibling_0["choices"][0]["message"]["content"] == "sibling-zero-recorded"
    assert sibling_1["choices"][0]["message"]["content"] == "sibling-one-recorded"
    assert grandchild["choices"][0]["message"]["content"] == "grandchild-recorded"
    assert router.audit()["resampled_policy_calls"] == 0


def test_nested_transport_keys_distinguish_identical_local_ids_by_lineage() -> None:
    parent_a_lineage = _child_lineage(
        parent_lineage="root",
        depth=1,
        parent_turn=0,
        parent_tool_call_id="call-parent-a",
        invocation_id="parent-a",
    )
    parent_b_lineage = _child_lineage(
        parent_lineage="root",
        depth=1,
        parent_turn=0,
        parent_tool_call_id="call-parent-b",
        invocation_id="parent-b",
    )
    common = {
        "depth": 2,
        "turn": 0,
        "call_kind": "policy",
        "parent_turn": 1,
        "parent_tool_call_id": "same-call",
        "invocation_id": "same-invocation",
    }
    nested_a = RLMEventAddress(parent_lineage=parent_a_lineage, **common)
    nested_b = RLMEventAddress(parent_lineage=parent_b_lineage, **common)
    assert nested_a.key() != nested_b.key()
    assert nested_a.key().startswith("depth:2:")

    def event(address: RLMEventAddress, content: str) -> ScriptedEvent:
        return ScriptedEvent(
            address=address,
            message={"role": "assistant", "content": content},
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )

    parent_a = RLMEventAddress(
        depth=1,
        turn=0,
        call_kind="policy",
        parent_lineage="root",
        parent_turn=0,
        parent_tool_call_id="call-parent-a",
        invocation_id="parent-a",
    )
    parent_b = RLMEventAddress(
        depth=1,
        turn=0,
        call_kind="policy",
        parent_lineage="root",
        parent_turn=0,
        parent_tool_call_id="call-parent-b",
        invocation_id="parent-b",
    )
    router = ScriptedCompletionRouter(
        (
            event(RLMEventAddress(depth=0, turn=0, call_kind="policy"), "root"),
            event(parent_a, "parent-a"),
            event(parent_b, "parent-b"),
            event(nested_a, "nested-a"),
            event(nested_b, "nested-b"),
        )
    )
    router.respond(
        headers={
            "X-RLM-Depth": "0",
            "X-RLM-Turn": "0",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Session-ID": "root",
        },
        request={"messages": []},
    )
    for session, tool, invocation in (
        ("parent-a-session", "call-parent-a", "parent-a"),
        ("parent-b-session", "call-parent-b", "parent-b"),
    ):
        router.respond(
            headers={
                "X-RLM-Depth": "1",
                "X-RLM-Turn": "0",
                "X-RLM-Call-Kind": "policy",
                "X-RLM-Session-ID": session,
                "X-RLM-Parent-Session-ID": "root",
                "X-RLM-Parent-Turn": "0",
                "X-RLM-Parent-Tool-Call-ID": tool,
                "X-RLM-Invocation-ID": invocation,
            },
            request={"messages": []},
        )
    nested_results = []
    for parent_session, session in (
        ("parent-a-session", "nested-a-session"),
        ("parent-b-session", "nested-b-session"),
    ):
        nested_results.append(
            router.respond(
                headers={
                    "X-RLM-Depth": "2",
                    "X-RLM-Turn": "0",
                    "X-RLM-Call-Kind": "policy",
                    "X-RLM-Session-ID": session,
                    "X-RLM-Parent-Session-ID": parent_session,
                    "X-RLM-Parent-Turn": "1",
                    "X-RLM-Parent-Tool-Call-ID": "same-call",
                    "X-RLM-Invocation-ID": "same-invocation",
                },
                request={"messages": []},
            )
        )
    assert [row["choices"][0]["message"]["content"] for row in nested_results] == [
        "nested-a",
        "nested-b",
    ]
