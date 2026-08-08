from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from redco.analysis.stage_d_live_observer import StageDForwardDirectiveObserver
from redco.analysis.stage_d_receipt_ledger import inspect_ledger
from redco.analysis.stage_d_source_artifacts import StageDSourceArtifactStore
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress
from redco.contracts import canonical_json


def test_real_four_call_source_finalization_regression_is_green_after_directional_repair(
    tmp_path: Path,
) -> None:
    """Pre-fix characterization: the real path must fail until the directional repair."""
    import multidict
    import verifiers.v1 as vf
    from renderers.base import (
        ParsedResponse,
        ParsedToolCall,
        RenderedTokens,
        ToolCallParseStatus,
    )
    from test_stage_d_live_observer import (
        _child_rlm,
        _observer,
        _root_rlm,
    )
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.clients.train import TrainClient
    from verifiers.v1.dialects.chat import ChatDialect
    from verifiers.v1.interception.server import InterceptionServer
    from verifiers.v1.session import RolloutSession

    environment_root = Path(__file__).parents[1] / "environments" / "redco_evidence_selection_v2"
    if str(environment_root) not in sys.path:
        sys.path.insert(0, str(environment_root))
    from redco_evidence_selection_v2.source_env import (
        _canonical_source_episode,
    )

    compact_tool = {
        "name": "ipython",
        "description": "Execute code.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    }
    wrapped_tool = {"type": "function", "function": compact_tool}

    class Renderer:
        supports_tools = True

        @staticmethod
        def _rendered(token_ids: list[int]) -> RenderedTokens:
            return RenderedTokens(
                token_ids=token_ids,
                message_indices=[0] * len(token_ids),
                sampled_mask=[False] * len(token_ids),
                is_content=[False] * len(token_ids),
                message_roles=["user"] * len(token_ids),
            )

        def render(
            self,
            messages: list[dict[str, object]],
            *,
            tools: Any = None,
            add_generation_prompt: bool = False,
        ) -> Any:
            del tools
            assert add_generation_prompt is True
            return self._rendered(self._token_ids(messages))

        @staticmethod
        def _token_ids(
            messages: list[dict[str, object]], *, default: list[int] | None = None
        ) -> list[int]:
            markers = {
                "child-provenance-0": [10, 11, 20, 2, 30, 40],
                "child-provenance-1": [10, 11, 20, 2, 30, 41],
            }
            for message in messages:
                marker = message.get("content")
                if isinstance(marker, str) and marker in markers:
                    return markers[marker]
            if default is not None:
                return default
            return [10, 11] if len(messages) == 1 else [10, 11, 20, 2, 30]

        def bridge_to_next_turn(
            self,
            _prompt_ids: list[int],
            _completion_ids: list[int],
            messages: list[dict[str, object]],
            *,
            tools: Any = None,
        ) -> Any:
            del tools
            return self._rendered(self._token_ids(messages, default=[10, 11, 20, 2, 30]))

        def get_stop_token_ids(self) -> list[int]:
            return [2]

        def parse_response(self, completion_ids: list[int], *, tools: Any = None) -> Any:
            del tools
            root_tool_ids: dict[tuple[int, ...], str] = {
                (20, 2): "transport-call-0-0",
                (21, 2): "transport-call-1-0",
            }
            if tuple(completion_ids) in root_tool_ids:
                return ParsedResponse(
                    content="",
                    tool_calls=[
                        ParsedToolCall(
                            raw='{"name":"ipython","arguments":{}}',
                            name="ipython",
                            arguments={},
                            status=ToolCallParseStatus.OK,
                            id=root_tool_ids[tuple(completion_ids)],
                        )
                    ],
                )
            return ParsedResponse(content="" if len(completion_ids) == 768 else "normal child")

    child_token_ids = {
        (10, 11, 20, 2, 30, 40): "child-session-1-0",
        (10, 11, 20, 2, 30, 41): "child-session-1-1",
    }
    child_inflight: set[str] = set()
    both_children_inflight = asyncio.Event()
    release_children = asyncio.Event()
    post_observations: list[dict[str, object]] = []

    async def post(*_args: Any, **kwargs: Any) -> Any:
        options = kwargs.get("options") or {}
        raw_headers = options.get("headers") or {}
        headers = {str(key).lower(): str(value) for key, value in dict(raw_headers).items()}
        session_id = headers.get("x-session-id")
        if session_id is None:
            raise AssertionError("TrainClient did not forward the durable session identity")
        body = kwargs.get("body")
        if not isinstance(body, dict):
            raise AssertionError("TrainClient did not send a prepared engine request")
        token_ids = tuple(body.get("token_ids", ()))
        provenance = child_token_ids.get(token_ids)
        if provenance is not None:
            child_inflight.add(provenance)
            if set(child_inflight) == set(child_token_ids.values()):
                both_children_inflight.set()
            await both_children_inflight.wait()
            await release_children.wait()
            capped = provenance == "child-session-1-0"
        else:
            if token_ids == (10, 11):
                root_turn = 0
            elif token_ids == (10, 11, 20, 2, 30):
                root_turn = 1
            else:
                raise AssertionError(f"unexpected prompt token identity: {token_ids}")
            capped = False
            provenance = f"root-turn-{root_turn}"
        post_observations.append(
            {
                "session_id": session_id,
                "provenance": provenance,
                "capped": capped,
                "token_count": 768 if capped else 2,
            }
        )
        if capped:
            completion_ids = [30] + ([31] * 767)
        elif provenance == "root-turn-0":
            completion_ids = [20, 2]
        elif provenance == "root-turn-1":
            completion_ids = [21, 2]
        else:
            completion_ids = [30, 2]
        return SimpleNamespace(
            content=canonical_json(
                {
                    "request_id": f"fixture-{provenance}",
                    "choices": [
                        {
                            "token_ids": completion_ids,
                            "logprobs": {"content": [{"logprob": -0.1}] * len(completion_ids)},
                            "finish_reason": "length" if capped else "stop",
                        }
                    ],
                }
            )
        )

    openai = SimpleNamespace(
        base_url="http://engine/v1",
        max_retries=0,
        post=AsyncMock(side_effect=post),
        close=AsyncMock(),
    )
    client = TrainClient(openai)
    client._pool = Renderer()
    parent_event = PolicyEventAddress(0, "root", 1, 1)
    observer, _ledger, producer = _observer(
        tmp_path / "ledger",
        rollout_id="four-call-regression",
        maximum_captured_session_call_count=8,
        child_parent_event=parent_event,
    )
    trace = vf.Trace(
        id="four-call-regression",
        task=vf.TraceTask(type="ObservedTask", data=vf.TaskData(prompt="q")),
    )
    sampling = vf.Sampling(
        temperature=0.7,
        top_p=1.0,
        reasoning_effort=None,
        min_p=0.0,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=1,
        max_tokens=768,
        parallel_tool_calls=False,
        seed=81,
        tool_choice="auto",
        extra_body={"cache_salt": "integration"},
    )
    session = RolloutSession(
        ModelContext("model@commit", client, sampling),
        trace,
        observer=StageDForwardDirectiveObserver(observer),
    )
    server = InterceptionServer()
    server.sessions["secret"] = session

    root_zero = _root_rlm(0)
    root_one = _root_rlm(1)
    root_one["completed_episode_spawn_ordinals"] = []
    child_zero = _child_rlm(
        0,
        "midpoint-shard-0",
        parent_turn=1,
        parent_call_ordinal=1,
    )
    child_one = _child_rlm(
        1,
        "midpoint-shard-1",
        parent_turn=1,
        parent_call_ordinal=1,
    )
    for child in (child_zero, child_one):
        child["parent_tool_call_id"] = "transport-call-1-0"
    rlm_calls = [root_zero, root_one, child_zero, child_one]

    def rlm_headers(rlm: dict[str, object]) -> Any:
        names = {
            "provenance_version": "X-RLM-Provenance-Version",
            "depth": "X-RLM-Depth",
            "session_id": "X-RLM-Session-ID",
            "turn": "X-RLM-Turn",
            "call_kind": "X-RLM-Call-Kind",
            "lineage": "X-RLM-Lineage",
            "session_call_ordinal": "X-RLM-Session-Call-Ordinal",
            "parent_session_id": "X-RLM-Parent-Session-ID",
            "parent_turn": "X-RLM-Parent-Turn",
            "parent_tool_call_id": "X-RLM-Parent-Tool-Call-ID",
            "invocation_id": "X-RLM-Invocation-ID",
            "parent_lineage": "X-RLM-Parent-Lineage",
            "parent_call_ordinal": "X-RLM-Parent-Call-Ordinal",
            "parent_tool_call_slot": "X-RLM-Parent-Tool-Call-Slot",
            "spawn_ordinal": "X-RLM-Spawn-Ordinal",
            "episode_spawn_ordinal": "X-RLM-Episode-Spawn-Ordinal",
            "completed_predecessor_spawn_ordinals": ("X-RLM-Completed-Predecessor-Spawn-Ordinals"),
            "completed_episode_spawn_ordinals": ("X-RLM-Completed-Episode-Spawn-Ordinals"),
        }
        headers = {"Authorization": "Bearer secret"}
        for key, value in rlm.items():
            if key not in names:
                continue
            headers[names[key]] = (
                ",".join(str(item) for item in value) if isinstance(value, list) else str(value)
            )
        return multidict.CIMultiDict(headers)

    async def invoke(call_index: int, messages: list[dict[str, object]]) -> dict[str, object]:
        body = canonical_json({"model": "ignored", "messages": messages, "tools": [wrapped_tool]})
        request = SimpleNamespace(
            headers=rlm_headers(rlm_calls[call_index]),
            path="/v1/chat/completions",
            read=AsyncMock(return_value=body),
            _read_bytes=body,
        )
        response = await server.handle_request(request, ChatDialect())
        assert response.status == 200, response.body.decode()
        return cast(dict[str, object], json.loads(response.body)["choices"][0]["message"])

    async def scenario() -> None:
        root_zero_message = await invoke(0, [{"role": "user", "content": "q"}])
        root_one_message = await invoke(
            1,
            [
                {"role": "user", "content": "q"},
                root_zero_message,
                {"role": "tool", "tool_call_id": "transport-call-0-0", "content": "computed"},
            ],
        )
        children = [
            asyncio.create_task(
                invoke(
                    2,
                    [
                        {"role": "user", "content": "q"},
                        root_one_message,
                        {
                            "role": "tool",
                            "tool_call_id": "transport-call-1-0",
                            "content": "computed",
                        },
                        {"role": "user", "content": "child-provenance-0"},
                    ],
                )
            ),
            asyncio.create_task(
                invoke(
                    3,
                    [
                        {"role": "user", "content": "q"},
                        root_one_message,
                        {
                            "role": "tool",
                            "tool_call_id": "transport-call-1-0",
                            "content": "computed",
                        },
                        {"role": "user", "content": "child-provenance-1"},
                    ],
                )
            ),
        ]
        await both_children_inflight.wait()
        assert child_inflight == set(child_token_ids.values())
        release_children.set()
        await asyncio.gather(*children)
        await client.close()

    asyncio.run(scenario())
    assert openai.post.await_count == 4
    assert len(post_observations) == 4
    assert {
        observation["session_id"]
        for observation in post_observations
        if observation["provenance"] in child_token_ids.values()
    } == {"four-call-regression"}
    assert [
        observation["provenance"] for observation in post_observations if observation["capped"]
    ] == ["child-session-1-0"]
    token_counts = [observation["token_count"] for observation in post_observations]
    assert all(type(token_count) is int for token_count in token_counts)
    assert sorted(cast(list[int], token_counts)) == [
        2,
        2,
        2,
        768,
    ]
    assert len(producer._completed) == 4
    assert sorted(decision.event_address.depth for decision in producer._completed.values()) == [
        0,
        0,
        1,
        1,
    ]
    assert sorted(
        decision.action.completion_tokens for decision in producer._completed.values()
    ) == [2, 2, 2, 768]
    assert (
        sum(
            decision.action.termination_kind == "max_tokens"
            for decision in producer._completed.values()
        )
        == 1
    )

    trace.rewards = {"evidence": 1.0}
    trace.info = {"checkpoint_id": "model@commit"}
    trace.agent = vf.AgentInfo(
        model="model@commit",
        sampling=vf.Sampling(temperature=0.7, max_tokens=768),
    )
    trace.is_completed = True
    trace.ok = True
    trace.stop_condition = "max_total_tokens"
    trace.errors = []
    episode = vf.Episode(
        id="four-call-regression",
        env="redco_evidence_selection_v2",
        ok=True,
        errors=[],
        traces=[trace],
    )
    episode_bytes = _canonical_source_episode(episode)
    episode_payload = json.loads(episode_bytes)
    persisted_trace = episode_payload["traces"][0]

    def address_key(value: dict[str, object]) -> tuple[object, ...]:
        return (
            value["depth"],
            value["lineage"],
            value["session_call_ordinal"],
            value["turn"],
        )

    trace_call_by_address = {address_key(call["rlm"]): call for call in persisted_trace["calls"]}
    completed_by_address = {
        (
            decision.event_address.depth,
            decision.event_address.lineage,
            decision.event_address.session_call_ordinal,
            decision.event_address.turn,
        ): decision
        for decision in producer._completed.values()
    }
    assert set(trace_call_by_address) == set(completed_by_address)
    trace_nodes = {call["node"] for call in trace_call_by_address.values()}
    assert len(trace_nodes) == 4
    assert all(isinstance(node, int) for node in trace_nodes)
    assert {address[0] for address in trace_call_by_address} == {0, 1}
    assert len(trace_call_by_address) == 4
    for address, decision in completed_by_address.items():
        call = trace_call_by_address[address]
        assert call["model"] == "model@commit"
        assert call["sampling"] == {
            "temperature": 0.7,
            "top_p": 1.0,
            "reasoning_effort": None,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "seed": 81,
            "max_tokens": 768,
            "n": 1,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
        }
        assert call["finish_reason"] == decision.action.finish_reason
        assert call["usage"]["completion_tokens"] == decision.action.completion_tokens
        assert call["usage"]["prompt_tokens"] == decision.action.prompt_tokens
        assert isinstance(decision.action.key.request, bytes)
        request = json.loads(decision.action.key.request)
        assert request["model"] == "model@commit"
        assert isinstance(request["messages"], list)
        assert isinstance(request["tools"], list)
        assert (
            decision.action.raw_transport_message_sha256
            == hashlib.sha256(canonical_json(decision.action.message)).hexdigest()
        )
    root_decisions = [
        decision for decision in completed_by_address.values() if decision.event_address.depth == 0
    ]
    assert {decision.action.message["tool_calls"][0]["id"] for decision in root_decisions} == {
        "transport-call-0-0",
        "transport-call-1-0",
    }
    assert all(
        "tool_calls" in persisted_trace["nodes"][trace_call_by_address[address]["node"]]["message"]
        for address, decision in completed_by_address.items()
        if decision.event_address.depth == 0
    )
    capped_address = next(
        address
        for address, decision in completed_by_address.items()
        if decision.action.termination_kind == "max_tokens"
    )
    assert capped_address[1] == child_zero["lineage"]
    capped_decision = completed_by_address[capped_address]
    capped_call = trace_call_by_address[capped_address]
    assert capped_decision.action.completion_tokens == 768
    assert capped_decision.action.finish_reason == "length"
    assert capped_decision.action.message == {"role": "assistant", "content": None}
    assert "content" not in persisted_trace["nodes"][capped_call["node"]]["message"]

    artifact_store = StageDSourceArtifactStore(tmp_path / "source-artifacts")
    artifact_store.assert_pristine()

    def prepare_source_rollout(payload: bytes) -> None:
        artifact_store.prepare(payload)

    try:
        source = producer.finalize_episode(
            episode_bytes,
            prepare_source_rollout=prepare_source_rollout,
        )
    except ValueError as error:
        assert str(error) == "captured transport message differs from the Verifiers trace"
        first_abort = producer.abort_finalization(error)
        assert first_abort is not None
        assert producer.abort_finalization(error) is None
        assert inspect_ledger(tmp_path / "ledger").status == "poisoned"
        assert artifact_store.source_paths() == ()
        artifact_store.assert_no_pending()
        ledger_records = [
            json.loads(path.read_bytes())
            for path in sorted((tmp_path / "ledger" / "records").glob("*.json"))
        ]
        aborts = [
            record["body"].get("receipt")
            for record in ledger_records
            if record["body"].get("receipt", {}).get("receipt_kind")
            == "source_rollout_finalization_aborted"
        ]
        assert len(aborts) == 1
        assert aborts[0]["phase"] == "source_finalization"
        assert set(aborts[0]["decision_ids"]) == set(producer._completed)
        error_bytes = (tmp_path / "ledger" / "evidence" / aborts[0]["error_sha256"]).read_bytes()
        error_payload = json.loads(error_bytes)
        assert error_payload["error_message"] == str(error)
        raise AssertionError(
            "pre-fix source finalization did not complete; the directional repair is still required"
        ) from error
    artifact_store.commit(source)
    assert len(artifact_store.source_paths()) == 1
    assert inspect_ledger(tmp_path / "ledger").status == "active-clean"
