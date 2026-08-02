from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from redco.analysis.stage_d_live_observer import (
    StageDForwardDirectiveObserver,
    StageDObserverIdentity,
    StageDObserverProtocol,
    StageDPreparedCallObserver,
    require_zero_retry_configuration,
)
from redco.analysis.stage_d_receipt_ledger import (
    GenesisBinding,
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
        ),
        master_seed=MASTER_SEED,
    )


def _observer(
    root: Path,
) -> tuple[
    StageDPreparedCallObserver,
    StageDReceiptLedger,
    StageDSourceRolloutProducer,
]:
    ledger = _ledger(root)
    parent = PolicyEventAddress(0, "root", 0, 0)
    producer = StageDSourceRolloutProducer(
        ledger=ledger,
        group_id="group-1",
        rollout_id="trace-1",
        child_parent_event=parent,
        child_parent_tool_call_slot=0,
        root_policy_turn_count=2,
        base_model_manifest_sha256="b" * 64,
    )
    observer = StageDPreparedCallObserver(
        producer=producer,
        trace_id="trace-1",
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
        ),
        runtime_snapshot=canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-test-runtime-snapshot-v1",
            }
        ),
        encode_action=lambda _request, _message, _prompt_token_ids: (20, 2),
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


def _root_rlm(turn: int = 0) -> dict[str, object]:
    return {
        "provenance_version": 2,
        "depth": 0,
        "session_id": "root-session",
        "turn": turn,
        "call_kind": "policy",
        "lineage": "root",
        "session_call_ordinal": turn,
        "completed_episode_spawn_ordinals": [] if turn == 0 else [0, 1],
    }


def _child_rlm(spawn: int, invocation: str) -> dict[str, object]:
    lineage = derive_child_lineage(
        SpawnScope(1, "root", 0, 0, 0),
        spawn_ordinal=spawn,
    )
    return {
        "provenance_version": 2,
        "depth": 1,
        "session_id": f"child-session-{spawn}",
        "turn": 0,
        "call_kind": "policy",
        "lineage": lineage,
        "session_call_ordinal": 0,
        "parent_session_id": "root-session",
        "parent_turn": 0,
        "parent_tool_call_id": f"transport-call-{spawn}",
        "invocation_id": invocation,
        "parent_lineage": "root",
        "parent_call_ordinal": 0,
        "parent_tool_call_slot": 0,
        "spawn_ordinal": spawn,
        "episode_spawn_ordinal": spawn,
        "completed_predecessor_spawn_ordinals": [],
        "completed_episode_spawn_ordinals": [],
    }


def _prepared(
    seed: int,
    label: str,
    rlm: dict[str, object],
    *,
    routed_experts_prompt_start: int | None = None,
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
        engine_headers=canonical_json({"X-Session-ID": "trace-1"}),
        observer_context=canonical_json({"trace_id": "trace-1", "rlm": rlm}),
        prompt_token_ids=(10, 11),
    )


def test_observer_accepts_bridge_boundary_only_on_returning_root(tmp_path: Path) -> None:
    async def scenario() -> None:
        observer, _ledger, _producer = _observer(tmp_path / "ledger")
        first = await observer.before_forward(_prepared(17, "root", _root_rlm()))
        await observer.after_response(first, _response(finish_reason="tool_calls"))
        for spawn in (0, 1):
            child = await observer.before_forward(
                _prepared(
                    18 + spawn,
                    f"child-{spawn}",
                    _child_rlm(spawn, f"midpoint-shard-{spawn}"),
                )
            )
            await observer.after_response(child, _response())
        returned = await observer.before_forward(
            _prepared(
                20,
                "return",
                _root_rlm(1),
                routed_experts_prompt_start=1,
            )
        )
        await observer.after_response(returned, _response())

    asyncio.run(scenario())

    async def invalid_first_turn() -> None:
        observer, _ledger, _producer = _observer(tmp_path / "bad-ledger")
        with pytest.raises(ValueError, match="only valid on a returning root"):
            await observer.before_forward(
                _prepared(
                    17,
                    "root",
                    _root_rlm(),
                    routed_experts_prompt_start=1,
                )
            )

    asyncio.run(invalid_first_turn())


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
            completion_ids=[20, 2],
            completion_logprobs=[-0.2, -0.1],
        ),
        raw={"id": "fixture-request", "choices": [{"message": message}]},
        usage=SimpleNamespace(input_tokens=2, completion_tokens=2),
        finish_reason=finish_reason,
    )


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


def test_observer_reserves_before_forward_and_completes_children_out_of_order(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        observer, ledger, _producer = _observer(tmp_path / "ledger")
        root = await observer.before_forward(_prepared(17, "root", _root_rlm()))
        await observer.after_response(root, _response(finish_reason="tool_calls"))
        first = await observer.before_forward(
            _prepared(18, "child-zero", _child_rlm(0, "misleading-b"))
        )
        second = await observer.before_forward(
            _prepared(19, "child-one", _child_rlm(1, "misleading-a"))
        )
        await observer.after_response(second, _response())
        await observer.after_response(first, _response())
        returning = await observer.before_forward(_prepared(20, "returning-root", _root_rlm(1)))
        await observer.after_response(returning, _response())
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
        await observer.after_response(ticket, _response(finish_reason="tool_calls"))
        with pytest.raises(ValueError, match="outside the frozen root"):
            await observer.before_forward(prepared)
        bad = _child_rlm(0, "diagnostic")
        bad["session_call_ordinal"] = 1
        with pytest.raises(ValueError, match="outside the frozen"):
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
        await observer.after_response(root, _response(finish_reason="tool_calls"))

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


def test_observer_refuses_max_token_truncation(tmp_path: Path) -> None:
    async def scenario() -> None:
        observer, ledger, _producer = _observer(tmp_path / "ledger")
        ticket = await observer.before_forward(_prepared(17, "root", _root_rlm()))
        with pytest.raises(ValueError, match="refuses truncated"):
            await observer.after_response(ticket, _response(finish_reason="length"))
        await observer.abort(ticket, "typed_response", ValueError("truncated"))
        assert inspect_ledger(tmp_path / "ledger").status == "poisoned"
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
