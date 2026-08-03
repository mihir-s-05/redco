from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from redco.analysis.stage_d_branch_artifacts import StageDBranchTargetRoster
from redco.analysis.stage_d_live_observer import (
    StageDForwardDirectiveObserver,
    StageDObserverIdentity,
    StageDObserverProtocol,
    StageDPreparedCallObserver,
    require_zero_retry_configuration,
)
from redco.analysis.stage_d_receipt_ledger import (
    GenesisBinding,
    LedgerError,
    StageDReceiptLedger,
    inspect_ledger,
)
from redco.analysis.stage_d_source_producer import StageDSourceRolloutProducer
from redco.analysis.stage_d_spawn_provenance import (
    PolicyEventAddress,
    SpawnScope,
    derive_child_lineage,
)
from redco.contracts import canonical_json
from redco.integrations.signed_subprocess import sign_payload

MASTER_SEED = "stage-d-live-observer-test"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _conformance() -> bytes:
    return canonical_json(
        sign_payload(
            {
                "schema_version": 1,
                "analysis": "served-stack-categorical-logprob-conformance-v1",
                "passes": True,
                "logprob_semantics": "served_chosen_token_post_transform",
                "categorical_case_count": 3,
                "served_stack_sha256": "a" * 64,
                "tool_call_termination_includes_all_generated_tokens": True,
                "eos_is_included_in_action_tokens_and_logprobs": True,
            }
        )
    )


def _ledger(root: Path) -> StageDReceiptLedger:
    return StageDReceiptLedger.create(
        root,
        binding=GenesisBinding(
            preregistration_sha256="1" * 64,
            source_sha256="2" * 64,
            runtime_sha256="3" * 64,
            config_sha256="4" * 64,
            protocol_manifest_sha256="5" * 64,
            master_seed_sha256=_sha256(MASTER_SEED.encode()),
            support_rules_sha256="6" * 64,
        ),
        master_seed=MASTER_SEED,
    )


def _observer(
    root: Path,
    *,
    rollout_id: str = "trace-1",
    maximum_captured_session_call_count: int = 8,
    ledger: StageDReceiptLedger | None = None,
) -> tuple[
    StageDPreparedCallObserver,
    StageDReceiptLedger,
    StageDSourceRolloutProducer,
]:
    ledger = _ledger(root) if ledger is None else ledger
    parent = PolicyEventAddress(0, "root", 0, 0)
    producer = StageDSourceRolloutProducer(
        ledger=ledger,
        group_id="group-1",
        rollout_id=rollout_id,
        child_parent_event=parent,
        child_parent_tool_call_slot=0,
        root_policy_turn_count=2,
        base_model_manifest_sha256="b" * 64,
    )
    observer = StageDPreparedCallObserver(
        producer=producer,
        trace_id=rollout_id,
        identity=StageDObserverIdentity(
            checkpoint_id="model@commit",
            base_model_manifest=b"base",
            adapter_manifest=b"adapter",
            tokenizer_manifest=b"tokenizer",
            renderer_manifest=b"renderer",
            sampler_conformance_manifest=_conformance(),
            eos_token_id=2,
        ),
        protocol=StageDObserverProtocol(
            branch_count=4,
            continuation_replicates=1,
            failure_reward=-1.0,
            root_policy_turn_count=2,
            maximum_captured_session_call_count=(
                maximum_captured_session_call_count
            ),
        ),
        runtime_snapshot=canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-test-runtime-snapshot-v1",
            }
        ),
        validate_action=lambda _request, _message, _action_token_ids: None,
    )
    return observer, ledger, producer


def _request(seed: int, label: str) -> dict[str, object]:
    return {
        "model": "model@commit",
        "messages": [{"role": "user", "content": label}],
        "tools": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": None,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "logit_bias": {},
        "seed": seed,
        "max_tokens": 2,
        "stop": None,
        "n": 1,
        "best_of": None,
        "use_beam_search": False,
        "logprobs": True,
        "top_logprobs": 0,
        "ignore_eos": False,
        "min_tokens": 0,
        "extra_body": {"cache_salt": f"seed-{seed}"},
    }


def _root_rlm(
    turn: int = 0,
    *,
    call_ordinal: int | None = None,
    call_kind: str = "policy",
) -> dict[str, object]:
    call_ordinal = turn if call_ordinal is None else call_ordinal
    return {
        "provenance_version": 2,
        "depth": 0,
        "session_id": "root-session",
        "turn": turn,
        "call_kind": call_kind,
        "lineage": "root",
        "session_call_ordinal": call_ordinal,
        "completed_episode_spawn_ordinals": [] if turn == 0 else [0, 1],
    }


def _child_rlm(
    spawn: int,
    invocation: str,
    *,
    turn: int = 0,
    parent_turn: int = 0,
    parent_call_ordinal: int | None = None,
    episode_spawn: int | None = None,
) -> dict[str, object]:
    parent_call_ordinal = (
        parent_turn if parent_call_ordinal is None else parent_call_ordinal
    )
    lineage = derive_child_lineage(
        SpawnScope(1, "root", parent_call_ordinal, 0, parent_turn),
        spawn_ordinal=spawn,
    )
    episode_spawn = spawn if episode_spawn is None else episode_spawn
    return {
        "provenance_version": 2,
        "depth": 1,
        "session_id": f"child-session-{parent_turn}-{spawn}",
        "turn": turn,
        "call_kind": "policy",
        "lineage": lineage,
        "session_call_ordinal": turn,
        "parent_session_id": "root-session",
        "parent_turn": parent_turn,
        "parent_tool_call_id": f"transport-call-{parent_turn}-{spawn}",
        "invocation_id": invocation,
        "parent_lineage": "root",
        "parent_call_ordinal": parent_call_ordinal,
        "parent_tool_call_slot": 0,
        "spawn_ordinal": spawn,
        "episode_spawn_ordinal": episode_spawn,
        "completed_predecessor_spawn_ordinals": [],
        "completed_episode_spawn_ordinals": [],
    }


def _prepared(
    seed: int,
    label: str,
    rlm: dict[str, object],
    *,
    routed_experts_prompt_start: int | None = None,
    trace_id: str = "trace-1",
) -> SimpleNamespace:
    request = _request(seed, label)
    engine = {
        "model": "model@commit",
        "token_ids": [10, 11],
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 1.0,
            "seed": seed,
            "max_tokens": 2,
            "stop_token_ids": [2],
            "logprobs": 1,
            "skip_special_tokens": False,
        },
        "cache_salt": f"seed-{seed}",
    }
    if routed_experts_prompt_start is not None:
        engine["sampling_params"]["routed_experts_prompt_start"] = (  # type: ignore[index]
            routed_experts_prompt_start
        )
    return SimpleNamespace(
        application_request=canonical_json(request),
        engine_endpoint="http://engine/inference/v1/generate",
        engine_request=canonical_json(engine),
        engine_headers=canonical_json({"X-Session-ID": trace_id}),
        observer_context=canonical_json({"trace_id": trace_id, "rlm": rlm}),
        prompt_token_ids=(10, 11),
    )


def test_observer_accepts_bridge_boundary_on_returning_sessions(tmp_path: Path) -> None:
    async def scenario() -> None:
        observer, ledger, producer = _observer(tmp_path / "ledger")
        first = await observer.before_forward(_prepared(17, "root", _root_rlm()))
        await _deliver_response(observer, first, _response(finish_reason="tool_calls"))
        for spawn in (0, 1):
            child = await observer.before_forward(
                _prepared(
                    18 + spawn,
                    f"child-{spawn}",
                    _child_rlm(spawn, f"midpoint-shard-{spawn}"),
                )
            )
            await _deliver_response(observer, child, _response())
        child_return = await observer.before_forward(
            _prepared(
                20,
                "child-return",
                _child_rlm(0, "midpoint-shard-0", turn=1),
                routed_experts_prompt_start=1,
            )
        )
        await _deliver_response(observer, child_return, _response())
        returned = await observer.before_forward(
            _prepared(
                21,
                "return",
                _root_rlm(1),
                routed_experts_prompt_start=1,
            )
        )
        await _deliver_response(observer, returned, _response(finish_reason="tool_calls"))
        later_child = await observer.before_forward(
            _prepared(
                22,
                "later-child",
                _child_rlm(
                    0,
                    "late-shard",
                    parent_turn=1,
                    episode_spawn=2,
                ),
            )
        )
        await _deliver_response(observer, later_child, _response())
        child_decisions = tuple(
            decision
            for decision in producer._completed.values()
            if decision.node_kind == "child"
        )
        assert sum(decision.provenance.branch_selected for decision in child_decisions) == 2
        continuation = next(
            decision
            for decision in child_decisions
            if decision.event_address.session_call_ordinal == 1
        )
        assert continuation.target_ordinal == 0
        assert continuation.provenance.branch_selected is False
        later = next(
            decision
            for decision in child_decisions
            if decision.event_address.lineage
            == _child_rlm(0, "late-shard", parent_turn=1, episode_spawn=2)["lineage"]
        )
        assert later.target_ordinal == 2
        assert later.provenance.branch_selected is False
        ledger.close()

    asyncio.run(scenario())

    async def invalid_first_turn() -> None:
        observer, ledger, _producer = _observer(tmp_path / "bad-ledger")
        with pytest.raises(ValueError, match="only valid on a returning session"):
            await observer.before_forward(
                _prepared(
                    17,
                    "root",
                    _root_rlm(),
                    routed_experts_prompt_start=1,
                )
            )
        ledger.close()

    asyncio.run(invalid_first_turn())


def test_observer_tracks_policy_parent_across_compaction(tmp_path: Path) -> None:
    async def scenario() -> None:
        observer, ledger, producer = _observer(tmp_path / "ledger")
        first = await observer.before_forward(_prepared(31, "root", _root_rlm()))
        await _deliver_response(observer, first, _response(finish_reason="tool_calls"))
        compaction = await observer.before_forward(
            _prepared(
                32,
                "compact",
                _root_rlm(0, call_ordinal=1, call_kind="compaction"),
            )
        )
        await _deliver_response(observer, compaction, _response())
        returned = await observer.before_forward(
            _prepared(
                33,
                "returned",
                _root_rlm(1, call_ordinal=2),
                routed_experts_prompt_start=1,
            )
        )
        await _deliver_response(observer, returned, _response(finish_reason="tool_calls"))
        child = await observer.before_forward(
            _prepared(
                34,
                "child",
                _child_rlm(
                    0,
                    "post-compaction",
                    parent_turn=1,
                    parent_call_ordinal=2,
                    episode_spawn=0,
                ),
            )
        )
        await _deliver_response(observer, child, _response())
        decision = next(
            value
            for value in producer._completed.values()
            if value.node_kind == "child"
        )
        assert decision.target_ordinal == 2
        assert decision.provenance.branch_selected is False
        ledger.close()

    asyncio.run(scenario())


def _response(*, finish_reason: str = "stop") -> SimpleNamespace:
    message: dict[str, object] = {"role": "assistant", "content": "answer"}
    if finish_reason == "tool_calls":
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "parent-call",
                    "type": "function",
                    "function": {"name": "ipython", "arguments": "{}"},
                }
            ],
        }
    return SimpleNamespace(
        tokens=SimpleNamespace(
            prompt_ids=[10, 11],
            completion_ids=[20, 21] if finish_reason == "length" else [20, 2],
            completion_logprobs=[-0.2, -0.1],
        ),
        raw={"id": "fixture-request", "choices": [{"message": message}]},
        usage=SimpleNamespace(input_tokens=2, completion_tokens=2),
        finish_reason=finish_reason,
    )


async def _deliver_response(
    observer: StageDPreparedCallObserver,
    ticket: object,
    response: SimpleNamespace,
) -> None:
    await observer.after_raw_response(
        ticket,
        canonical_json({"fixture": "exact-provider-response"}),
    )
    await observer.after_response(ticket, response)


def test_actual_interception_train_renderer_path_observes_bytes_once(
    tmp_path: Path,
) -> None:
    multidict = pytest.importorskip("multidict")
    vf = pytest.importorskip("verifiers.v1")
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.clients.train import TrainClient
    from verifiers.v1.dialects.chat import ChatDialect
    from verifiers.v1.interception.server import InterceptionServer
    from verifiers.v1.session import RolloutSession

    if "observer" not in inspect.signature(RolloutSession).parameters:
        pytest.skip("prepared-observer patch is not applied to the local verifier stack")

    class Renderer:
        supports_tools = True

        def render(self, messages, *, tools=None, add_generation_prompt=False):
            del messages, tools
            assert add_generation_prompt is True
            return SimpleNamespace(
                token_ids=[10, 11],
                multi_modal_data=None,
                message_token_spans=lambda: None,
                is_content=[False, False],
            )

        def get_stop_token_ids(self):
            return [2]

        def parse_response(self, completion_ids, *, tools=None):
            del tools
            assert completion_ids == [20, 2]
            return SimpleNamespace(
                content="answer",
                reasoning_content=None,
                tool_calls=[],
            )

    raw_response = canonical_json(
        {
            "request_id": "fixture",
            "choices": [
                {
                    "token_ids": [20, 2],
                    "logprobs": {
                        "content": [{"logprob": -0.2}, {"logprob": -0.1}]
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    )
    openai = SimpleNamespace(
        base_url="http://engine/v1",
        max_retries=0,
        post=AsyncMock(return_value=SimpleNamespace(content=raw_response)),
        close=AsyncMock(),
    )
    client = TrainClient(openai)
    client._pool = Renderer()
    observer, ledger, producer = _observer(tmp_path / "ledger")
    task_data = vf.TaskData(prompt="hello")
    trace = vf.Trace(
        id="trace-1",
        task=vf.TraceTask(type="ObservedTask", data=task_data),
    )
    sampling = vf.Sampling(
        temperature=0.7,
        top_p=1.0,
        top_k=None,
        min_p=0.0,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logit_bias={},
        seed=17,
        max_tokens=2,
        stop=None,
        n=1,
        best_of=None,
        use_beam_search=False,
        logprobs=True,
        top_logprobs=0,
        ignore_eos=False,
        min_tokens=0,
        tool_choice="auto",
        parallel_tool_calls=False,
        extra_body={"cache_salt": "seed-17"},
    )
    session = RolloutSession(
        ModelContext("model@commit", client, sampling),
        trace,
        observer=StageDForwardDirectiveObserver(observer),
    )
    server = InterceptionServer()
    server.sessions["secret"] = session
    body = canonical_json(
        {
            "model": "ignored",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    headers = multidict.CIMultiDict(
        {
            "Authorization": "Bearer secret",
            "X-RLM-Provenance-Version": "2",
            "X-RLM-Depth": "0",
            "X-RLM-Session-ID": "root-session",
            "X-RLM-Lineage": "root",
            "X-RLM-Session-Call-Ordinal": "0",
            "X-RLM-Turn": "0",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Completed-Episode-Spawn-Ordinals": "",
        }
    )
    request = SimpleNamespace(
        headers=headers,
        path="/v1/chat/completions",
        read=AsyncMock(return_value=body),
        _read_bytes=body,
    )

    async def scenario() -> None:
        response = await server.handle_request(request, ChatDialect())
        assert response.status == 200
        await client.close()
        ledger.close()

    asyncio.run(scenario())
    openai.post.assert_awaited_once()
    assert len(producer._completed) == 1
    (completed,) = producer._completed.values()
    assert completed.action.key.prompt_token_ids == (10, 11)
    assert completed.action.action_token_ids == (20, 2)
    receipt_kinds = []
    for record_path in sorted((tmp_path / "ledger" / "records").glob("*.json")):
        record = json.loads(record_path.read_bytes())
        if record["record_kind"] == "receipt":
            receipt_kinds.append(record["body"]["receipt"]["receipt_kind"])
    assert receipt_kinds == [
        "source_policy_call_reserved",
        "source_policy_response_observed",
        "source_policy_call_completed",
    ]
    assert (tmp_path / "ledger" / "evidence" / _sha256(raw_response)).read_bytes() == raw_response


def test_actual_two_turn_child_finalizes_as_excluded_without_replay(
    tmp_path: Path,
) -> None:
    multidict = pytest.importorskip("multidict")
    vf = pytest.importorskip("verifiers.v1")
    from renderers.base import (
        ParsedResponse,
        ParsedToolCall,
        RenderedTokens,
        ToolCallParseStatus,
    )
    env_root = Path(__file__).parents[1] / "environments" / "redco_evidence_selection_v2"
    sys.path.insert(0, str(env_root))
    from redco_evidence_selection_v2.source_env import _canonical_source_episode
    from test_stage_d_source_producer import _two_turn_child_episode
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.clients.train import TrainClient
    from verifiers.v1.dialects.chat import ChatDialect
    from verifiers.v1.interception.server import InterceptionServer
    from verifiers.v1.session import RolloutSession

    if "observer" not in inspect.signature(RolloutSession).parameters:
        pytest.skip("prepared-observer patch is not applied to the verifier stack")

    episode = json.loads(_two_turn_child_episode())
    trace_payload = episode["traces"][0]
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
    trace_payload["tools"] = [compact_tool]
    nodes = trace_payload["nodes"]
    calls = trace_payload["calls"]
    calls[1]["finish_reason"] = "length"
    nodes[3]["token_ids"] = [20, 21]

    child_tool = nodes[8]
    child_continuation = nodes[9]
    root_tool = json.loads(json.dumps(child_tool))
    root_return = nodes[7]
    child_tool["parent"] = 5
    child_continuation["parent"] = 6
    root_tool["parent"] = 1
    root_return["parent"] = 8
    trace_payload["nodes"] = [
        *nodes[:6],
        child_tool,
        child_continuation,
        root_tool,
        root_return,
        *nodes[10:],
    ]
    calls[3]["node"] = 7
    calls[4]["node"] = 9
    calls[4]["usage"]["prompt_tokens"] = 5
    for call in calls:
        call["sampling"]["seed"] = 81
    for node in trace_payload["nodes"]:
        message = node["message"]
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            continue
        message["tool_calls"] = [
            {
                "id": item["id"],
                "name": item["function"]["name"],
                "arguments": item["function"]["arguments"],
            }
            for item in tool_calls
        ]
    episode_bytes = _canonical_source_episode(vf.WireEpisode.model_validate(episode))
    serialized_tools = json.loads(episode_bytes)["traces"][0]["tools"]
    assert serialized_tools[0]["strict"] is None

    class Renderer:
        supports_tools = True

        def __init__(self) -> None:
            self.parse_index = 0

        @staticmethod
        def _rendered(token_ids: list[int]) -> RenderedTokens:
            return RenderedTokens(
                token_ids=token_ids,
                message_indices=[0] * len(token_ids),
                sampled_mask=[False] * len(token_ids),
                is_content=[False] * len(token_ids),
                message_roles=["user"],
            )

        def render(self, messages, *, tools=None, add_generation_prompt=False):
            del tools
            assert add_generation_prompt is True
            return self._rendered(
                [10, 11]
                if len(messages) == 1
                else [10, 11, 20, 2, 30]
            )

        def bridge_to_next_turn(self, *args, **kwargs):
            del args, kwargs
            return self._rendered([10, 11, 20, 2, 30])

        def get_stop_token_ids(self):
            return [2]

        def parse_response(self, completion_ids, *, tools=None):
            del tools
            assert completion_ids in ([20, 2], [20, 21])
            call_index = self.parse_index
            self.parse_index += 1
            tool_calls = []
            if call_index in {0, 2, 4}:
                tool_calls = [
                    ParsedToolCall(
                        raw='{"name":"ipython","arguments":{}}',
                        name="ipython",
                        arguments={},
                        status=ToolCallParseStatus.OK,
                        id="call_0",
                    )
                ]
            return ParsedResponse(
                content="" if tool_calls else "duplicate",
                tool_calls=tool_calls,
            )

    post_index = 0

    async def post(*_args, **_kwargs):
        nonlocal post_index
        call_index = post_index
        request_id = f"fixture-{call_index}"
        post_index += 1
        max_tokens = call_index == 1
        return SimpleNamespace(
            content=canonical_json(
                {
                    "request_id": request_id,
                    "choices": [
                        {
                            "token_ids": [20, 21] if max_tokens else [20, 2],
                            "logprobs": {
                                "content": [
                                    {"logprob": -0.2},
                                    {"logprob": -0.1},
                                ]
                            },
                            "finish_reason": "length" if max_tokens else "stop",
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
    observer, ledger, producer = _observer(
        tmp_path / "ledger",
        rollout_id="rollout-strict",
        maximum_captured_session_call_count=16,
    )
    task_data = vf.TaskData(prompt="q")
    trace = vf.Trace(
        id="rollout-strict",
        task=vf.TraceTask(type="ObservedTask", data=task_data),
    )
    sampling = vf.Sampling(
        temperature=0.7,
        top_p=1.0,
        top_k=None,
        min_p=0.0,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logit_bias={},
        seed=81,
        max_tokens=2,
        stop=None,
        n=1,
        best_of=None,
        use_beam_search=False,
        logprobs=True,
        top_logprobs=0,
        ignore_eos=False,
        min_tokens=0,
        tool_choice="auto",
        parallel_tool_calls=False,
        extra_body={"cache_salt": "integration"},
    )
    session = RolloutSession(
        ModelContext("model@commit", client, sampling),
        trace,
        observer=StageDForwardDirectiveObserver(observer),
    )
    server = InterceptionServer()
    server.sessions["secret"] = session

    def rlm_headers(rlm: dict[str, object]):
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
            "completed_predecessor_spawn_ordinals": (
                "X-RLM-Completed-Predecessor-Spawn-Ordinals"
            ),
            "completed_episode_spawn_ordinals": (
                "X-RLM-Completed-Episode-Spawn-Ordinals"
            ),
        }
        headers = {"Authorization": "Bearer secret"}
        for key, value in rlm.items():
            if key not in names:
                continue
            headers[names[key]] = (
                ",".join(str(item) for item in value)
                if isinstance(value, list)
                else str(value)
            )
        return multidict.CIMultiDict(headers)

    async def invoke(call_index: int, messages: list[dict[str, object]]):
        body = canonical_json(
            {"model": "ignored", "messages": messages, "tools": [wrapped_tool]}
        )
        request = SimpleNamespace(
            headers=rlm_headers(calls[call_index]["rlm"]),
            path="/v1/chat/completions",
            read=AsyncMock(return_value=body),
            _read_bytes=body,
        )
        response = await server.handle_request(request, ChatDialect())
        assert response.status == 200, response.body.decode()
        return json.loads(response.body)["choices"][0]["message"]

    async def scenario() -> None:
        root_tool_message = await invoke(0, [{"role": "user", "content": "q"}])
        _, child_tool_message = await asyncio.gather(
            invoke(1, [{"role": "user", "content": "q"}]),
            invoke(2, [{"role": "user", "content": "q"}]),
        )
        await invoke(
            3,
            [
                {"role": "user", "content": "q"},
                child_tool_message,
                {"role": "tool", "tool_call_id": "call_0", "content": "computed"},
            ],
        )
        await invoke(
            4,
            [
                {"role": "user", "content": "q"},
                root_tool_message,
                {"role": "tool", "tool_call_id": "call_0", "content": "computed"},
            ],
        )
        await invoke(5, [{"role": "user", "content": "q"}])
        await client.close()

    asyncio.run(scenario())

    assert openai.post.await_count == 6
    assert len(producer._completed) == 6
    max_token_decisions = [
        decision
        for decision in producer._completed.values()
        if decision.action.finish_reason == "length"
    ]
    assert len(max_token_decisions) == 1
    assert max_token_decisions[0].action.termination_kind == "max_tokens"
    assert max_token_decisions[0].action.action_token_ids == (20, 21)
    child_return_request = openai.post.await_args_list[3].kwargs["body"]
    assert child_return_request["sampling_params"][
        "routed_experts_prompt_start"
    ] == 3
    root_return_decision = next(
        decision
        for decision in producer._completed.values()
        if decision.event_address.depth == 0 and decision.event_address.turn == 1
    )
    assert root_return_decision.action.key.prompt_token_ids == (10, 11, 20, 2, 30)
    assert root_return_decision.action.action_token_ids == (20, 2)
    assert root_return_decision.action.behavior_logprobs == (-0.2, -0.1)
    source = producer.finalize_episode(episode_bytes)
    assert source.branch_eligible is False
    assert len(source.child_target_roster) == 3
    roster = StageDBranchTargetRoster.from_sources(
        (source,),
        planned_source_count=1,
        minimum_eligible_sources=1,
    )
    assert roster.targets == ()
    assert len(roster.excluded_targets) == 2
    ledger.record_branch_target_roster(roster.to_bytes())
    for item in roster.excluded_targets:
        with pytest.raises(LedgerError, match="absent from the frozen branch target roster"):
            ledger.begin_candidate_attempt(
                group_id=source.group_id,
                target_id=item.target.target_id,
                action_slot=1,
            )
    assert inspect_ledger(tmp_path / "ledger").status == "active-clean"
    ledger.close()


def test_raw_response_is_witnessed_once_before_parse_failure(tmp_path: Path) -> None:
    async def scenario() -> None:
        observer, ledger, _producer = _observer(tmp_path / "ledger")
        ticket = await observer.before_forward(_prepared(17, "root", _root_rlm()))
        raw_response = b"malformed-provider-response"
        with pytest.raises(ValueError, match="wrong type"):
            await observer.after_raw_response(object(), raw_response)
        with pytest.raises(ValueError, match="nonempty bytes"):
            await observer.after_raw_response(ticket, b"")
        await observer.after_raw_response(ticket, raw_response)
        with pytest.raises(ValueError, match="observed twice"):
            await observer.after_raw_response(ticket, raw_response)
        await observer.abort(ticket, "response_received", ValueError("malformed response"))
        ledger.close()
        assert (
            tmp_path / "ledger" / "evidence" / _sha256(raw_response)
        ).read_bytes() == raw_response

    asyncio.run(scenario())


def test_selected_typed_completion_cannot_mutate_before_raw_witness(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        observer, ledger, _producer = _observer(tmp_path / "ledger")
        root = await observer.before_forward(_prepared(17, "root", _root_rlm()))
        await _deliver_response(observer, root, _response(finish_reason="tool_calls"))
        child = await observer.before_forward(
            _prepared(18, "child-zero", _child_rlm(0, "diagnostic"))
        )
        with pytest.raises(ValueError, match="lacks its raw response witness"):
            await observer.after_response(child, _response())
        record_kinds = [
            json.loads(path.read_bytes())["record_kind"]
            for path in sorted((tmp_path / "ledger" / "records").glob("*.json"))
        ]
        assert "recorded_action_materialized" not in record_kinds
        assert "model_call_completed" not in record_kinds
        await observer.abort(child, "post_unknown", RuntimeError("fixture stop"))
        ledger.close()

    asyncio.run(scenario())


def test_observer_reserves_before_forward_and_completes_children_out_of_order(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        observer, ledger, _producer = _observer(tmp_path / "ledger")
        root = await observer.before_forward(_prepared(17, "root", _root_rlm()))
        await _deliver_response(observer, root, _response(finish_reason="tool_calls"))
        first = await observer.before_forward(
            _prepared(18, "child-zero", _child_rlm(0, "misleading-b"))
        )
        second = await observer.before_forward(
            _prepared(19, "child-one", _child_rlm(1, "misleading-a"))
        )
        await _deliver_response(observer, second, _response())
        await _deliver_response(observer, first, _response())
        returning = await observer.before_forward(_prepared(20, "returning-root", _root_rlm(1)))
        await _deliver_response(observer, returning, _response())
        assert inspect_ledger(tmp_path / "ledger").status == "active-clean"
        ledger.close()

    asyncio.run(scenario())


def test_observer_duplicate_and_out_of_scaffold_calls_fail_before_forward(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        observer, ledger, _producer = _observer(tmp_path / "ledger")
        prepared = _prepared(17, "root", _root_rlm())
        ticket = await observer.before_forward(prepared)
        await _deliver_response(observer, ticket, _response(finish_reason="tool_calls"))
        with pytest.raises(ValueError, match="outside the deployed session bounds"):
            await observer.before_forward(prepared)
        bad = _child_rlm(0, "diagnostic", turn=1)
        with pytest.raises(ValueError, match="contiguous stable session"):
            await observer.before_forward(_prepared(18, "bad", bad))
        ledger.close()

    asyncio.run(scenario())


def test_child_commit_then_reservation_failure_is_terminal_before_post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        observer, ledger, producer = _observer(tmp_path / "ledger")
        root = await observer.before_forward(_prepared(17, "root", _root_rlm()))
        await _deliver_response(observer, root, _response(finish_reason="tool_calls"))

        def fail_after_commit(**_kwargs: object) -> object:
            raise RuntimeError("injected reservation failure")

        monkeypatch.setattr(producer, "reserve_policy_call", fail_after_commit)
        with pytest.raises(RuntimeError, match="injected reservation"):
            await observer.before_forward(_prepared(18, "child-zero", _child_rlm(0, "diagnostic")))
        assert inspect_ledger(tmp_path / "ledger").status == "poisoned"
        ledger.close()

    asyncio.run(scenario())


def test_observer_abort_is_durable_and_terminal(tmp_path: Path) -> None:
    async def scenario() -> None:
        observer, ledger, _producer = _observer(tmp_path / "ledger")
        ticket = await observer.before_forward(_prepared(17, "root", _root_rlm()))
        await observer.abort(ticket, "post_unknown", TimeoutError("unknown outcome"))
        scan = inspect_ledger(tmp_path / "ledger")
        assert scan.status == "poisoned"
        assert scan.reason == "ledger records an aborted source policy call"
        ledger.close()

    asyncio.run(scenario())


def test_observer_accepts_max_token_action_and_next_rollout(tmp_path: Path) -> None:
    async def scenario() -> None:
        observer, ledger, producer = _observer(tmp_path / "ledger")
        ticket = await observer.before_forward(_prepared(17, "root", _root_rlm()))
        await _deliver_response(observer, ticket, _response(finish_reason="length"))
        (decision,) = producer._completed.values()
        assert decision.action.finish_reason == "length"
        assert decision.action.termination_kind == "max_tokens"
        assert decision.action.eos_token_id is None
        assert decision.action.action_token_ids == (20, 21)
        assert not producer._pending
        assert inspect_ledger(tmp_path / "ledger").status == "active-clean"

        next_observer, returned_ledger, next_producer = _observer(
            tmp_path / "unused",
            rollout_id="trace-2",
            ledger=ledger,
        )
        assert returned_ledger is ledger
        next_ticket = await next_observer.before_forward(
            _prepared(19, "root-next", _root_rlm(), trace_id="trace-2")
        )
        await _deliver_response(next_observer, next_ticket, _response())
        assert len(next_producer._completed) == 1
        assert inspect_ledger(tmp_path / "ledger").status == "active-clean"
        ledger.close()

    asyncio.run(scenario())


def test_observer_rejects_tool_finish_without_a_tool_call(tmp_path: Path) -> None:
    async def scenario() -> None:
        observer, ledger, producer = _observer(tmp_path / "ledger")
        ticket = await observer.before_forward(_prepared(17, "root", _root_rlm()))
        response = _response(finish_reason="tool_calls")
        response.raw["choices"][0]["message"] = {
            "role": "assistant",
            "content": "I cannot call that tool.",
        }
        await observer.after_raw_response(
            ticket,
            canonical_json({"fixture": "content-only-tool-finish"}),
        )
        with pytest.raises(ValueError, match="requires a nonempty tool-call"):
            await observer.after_response(ticket, response)
        assert not producer._completed
        await observer.abort(
            ticket,
            "typed_response",
            ValueError("tool finish without a tool call"),
        )
        ledger.close()

    asyncio.run(scenario())


def test_observer_accepts_textual_refusal_as_ordinary_content(tmp_path: Path) -> None:
    async def scenario() -> None:
        observer, ledger, producer = _observer(tmp_path / "ledger")
        ticket = await observer.before_forward(_prepared(17, "root", _root_rlm()))
        response = _response()
        response.raw["choices"][0]["message"]["content"] = (
            "I cannot answer that request."
        )
        await _deliver_response(observer, ticket, response)
        (decision,) = producer._completed.values()
        assert decision.action.finish_reason == "stop"
        assert decision.action.termination_kind == "eos"
        assert not producer._pending
        assert inspect_ledger(tmp_path / "ledger").status == "active-clean"
        ledger.close()

    asyncio.run(scenario())


def test_observer_rejects_noncanonical_or_mismatched_prepared_evidence(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        observer, ledger, _producer = _observer(tmp_path / "ledger")
        prepared = _prepared(17, "root", _root_rlm())
        prepared.application_request = json.dumps(
            json.loads(prepared.application_request), indent=2
        ).encode()
        with pytest.raises(ValueError, match="canonical"):
            await observer.before_forward(prepared)
        ledger.close()

    asyncio.run(scenario())


def test_zero_retry_guard_rejects_every_retry_layer() -> None:
    require_zero_retry_configuration(agent_max_retries=0, client_max_retries=0)
    with pytest.raises(ValueError, match="zero retries"):
        require_zero_retry_configuration(agent_max_retries=1, client_max_retries=0)
    with pytest.raises(ValueError, match="zero retries"):
        require_zero_retry_configuration(agent_max_retries=0, client_max_retries=1)
