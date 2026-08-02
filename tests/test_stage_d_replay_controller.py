from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from redco.analysis.stage_d_dynamic_taint import DynamicCausalTaintTracker
from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.analysis.stage_d_receipt_ledger import (
    ExecutionAttempt,
    ReplayOverrideTicket,
)
from redco.analysis.stage_d_replay_controller import (
    ExactSamplingDirector,
    SamplingOverride,
    StageDReconstructionQAController,
    StageDReplayCallController,
    frozen_engine_response_content,
)
from redco.analysis.stage_d_spawn_provenance import (
    CausalProvenanceGraph,
    EventCompletionSnapshot,
    PolicyEventAddress,
)
from redco.contracts import canonical_json
from redco.integrations.signed_subprocess import sign_payload
from redco.integrations.verifiers_trace_v2 import parse_v2_rlm_provenance_payload


class _Sampling:
    def __init__(self, payload: dict[str, object], *, corrupt_copy: bool = False) -> None:
        self.payload = copy.deepcopy(payload)
        self.corrupt_copy = corrupt_copy

    def model_dump(self, *, mode: str, exclude_none: bool) -> dict[str, object]:
        assert mode == "json"
        assert exclude_none is False
        return copy.deepcopy(self.payload)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any],
        deep: bool,
    ) -> _Sampling:
        assert deep is True
        payload = copy.deepcopy(self.payload)
        payload.update(copy.deepcopy(dict(update)))
        if self.corrupt_copy:
            payload["temperature"] = 0.9
        return _Sampling(payload)


def _context() -> dict[str, object]:
    return {
        "trace_id": "trace-1",
        "rlm": {
            "provenance_version": 2,
            "depth": 0,
            "session_id": "root-session",
            "turn": 0,
            "call_kind": "policy",
            "lineage": "root",
            "session_call_ordinal": 0,
            "completed_episode_spawn_ordinals": [],
        },
    }


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


def _request() -> dict[str, object]:
    return {
        "model": "model@commit",
        "messages": [{"role": "user", "content": "q"}],
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
        "seed": 17,
        "max_tokens": 2,
        "stop": None,
        "n": 1,
        "best_of": None,
        "use_beam_search": False,
        "logprobs": True,
        "top_logprobs": 0,
        "ignore_eos": False,
        "min_tokens": 0,
        "extra_body": {"cache_salt": "exact"},
    }


def _action(*, tool_call: bool = False) -> BehaviorAction:
    key = ExactActionKey.build(
        checkpoint_id="model@commit",
        base_model_manifest=b"base",
        adapter_manifest=b"adapter",
        tokenizer_manifest=b"tokenizer",
        renderer_manifest=b"renderer",
        sampler_conformance_manifest=_conformance(),
        action_selection_policy="direct_single_sample",
        transport_retry_policy="fail_before_action_no_resample",
        request=_request(),
        prompt_token_ids=(10, 11),
        render_prompt=lambda _: (10, 11),
    )
    message: dict[str, object] = {"role": "assistant", "content": "ok"}
    if tool_call:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "ipython", "arguments": "{}"},
                }
            ],
        }
    return BehaviorAction.build(
        key=key,
        action_token_ids=(20, 2),
        behavior_logprobs=(-0.2, -0.1),
        raw_transport_message=message,
        finish_reason="tool_calls" if tool_call else "stop",
        prompt_tokens=2,
        completion_tokens=2,
        termination_kind="tool_calls" if tool_call else "eos",
        eos_token_id=None if tool_call else 2,
        encode_action=lambda _request, _message: (20, 2),
        request_id="source-request-id",
    )


def _prepared_action() -> BehaviorAction:
    engine = {
        "model": "model@commit",
        "token_ids": [10, 11],
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 1.0,
            "seed": 17,
            "max_tokens": 2,
            "stop_token_ids": [2],
            "logprobs": 1,
            "skip_special_tokens": False,
            "parallel_tool_calls": False,
        },
        "cache_salt": "exact",
    }
    key = ExactActionKey.build_prepared(
        checkpoint_id="model@commit",
        base_model_manifest=b"base",
        adapter_manifest=b"adapter",
        tokenizer_manifest=b"tokenizer",
        renderer_manifest=b"renderer",
        sampler_conformance_manifest=_conformance(),
        action_selection_policy="direct_single_sample",
        transport_retry_policy="fail_before_action_no_resample",
        request=_request(),
        prompt_token_ids=(10, 11),
        prepared_engine_request=engine,
    )
    return BehaviorAction.build(
        key=key,
        action_token_ids=(20, 2),
        behavior_logprobs=(-0.2, -0.1),
        raw_transport_message={"role": "assistant", "content": "ok"},
        finish_reason="stop",
        prompt_tokens=2,
        completion_tokens=2,
        termination_kind="eos",
        eos_token_id=2,
        encode_action=lambda _request, _message: (20, 2),
        request_id="source-request-id",
    )


def test_director_changes_only_seed_and_cache_salt() -> None:
    address = PolicyEventAddress(0, "root", 0, 0)
    override = SamplingOverride(address, 9001, "directed-salt")
    director = ExactSamplingDirector(lambda _: override)
    base = _Sampling(
        {
            "temperature": 0.2,
            "top_p": 1.0,
            "seed": 7,
            "extra_body": {"cache_salt": "base-salt"},
        }
    )

    directed = director.direct_sampling(_context(), base)

    assert directed.model_dump(mode="json", exclude_none=False) == {
        "temperature": 0.2,
        "top_p": 1.0,
        "seed": 9001,
        "extra_body": {"cache_salt": "directed-salt"},
    }
    assert base.payload["seed"] == 7
    assert director.consume_override(address) == override


def test_director_rejects_hidden_copy_mutation_and_duplicate_address() -> None:
    address = PolicyEventAddress(0, "root", 0, 0)
    override = SamplingOverride(address, 9001, "directed-salt")
    corrupt = ExactSamplingDirector(lambda _: override)
    with pytest.raises(ValueError, match="forbidden fields"):
        corrupt.direct_sampling(
            _context(),
            _Sampling(
                {"temperature": 0.2, "seed": 7, "extra_body": {}},
                corrupt_copy=True,
            ),
        )

    director = ExactSamplingDirector(lambda _: override)
    base = _Sampling({"temperature": 0.2, "seed": 7, "extra_body": {}})
    director.direct_sampling(_context(), base)
    with pytest.raises(ValueError, match="direct a policy address twice"):
        director.direct_sampling(_context(), base)


def test_reconstruction_qa_replays_complete_source_with_zero_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class Return:
        ticket: object
        response_content: bytes

    renderers = ModuleType("renderers")
    client_module = ModuleType("renderers.client")
    client_module.PreparedGenerateReturn = Return  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "renderers", renderers)
    monkeypatch.setitem(sys.modules, "renderers.client", client_module)
    record = parse_v2_rlm_provenance_payload(
        trace_id="source-trace",
        payload=cast(dict[str, Any], _context()["rlm"]),
    )
    action = _prepared_action()
    controller = StageDReconstructionQAController(
        source_records=(record,),
        source_actions={record.scientific_address: action},
    )
    controller.direct_sampling(
        _context(),
        _Sampling(
            {
                "temperature": 0.7,
                "top_p": 1.0,
                "seed": 17,
                "extra_body": {"cache_salt": "exact"},
            }
        ),
    )
    prepared = SimpleNamespace(
        application_request=action.key.request,
        engine_endpoint="http://engine/inference/v1/generate",
        engine_request=action.key.prepared_engine_request,
        engine_headers=canonical_json({"X-Session-ID": "qa-trace"}),
        observer_context=canonical_json(_context()),
        prompt_token_ids=action.key.prompt_token_ids,
    )
    directive = __import__("asyncio").run(controller.before_forward(prepared))
    assert isinstance(directive, Return)
    response = SimpleNamespace(
        id=action.request_id,
        finish_reason=action.finish_reason,
        tokens=SimpleNamespace(
            prompt_ids=list(action.key.prompt_token_ids),
            completion_ids=list(action.action_token_ids),
            completion_logprobs=list(action.behavior_logprobs),
            routed_experts=None,
            kept_tokens=None,
        ),
        usage=SimpleNamespace(
            input_tokens=action.prompt_tokens,
            completion_tokens=action.completion_tokens,
        ),
        raw={"choices": [{"message": action.message}]},
    )
    __import__("asyncio").run(controller.after_response(directive.ticket, response))
    controller.finalize()


def test_replay_guard_runs_at_sampling_boundary_before_any_plan() -> None:
    record = parse_v2_rlm_provenance_payload(
        trace_id="source-trace",
        payload=cast(dict[str, Any], _context()["rlm"]),
    )
    action = _prepared_action()
    ready = False

    def require_ready() -> None:
        if not ready:
            raise RuntimeError("post-cut preflight is absent")

    controller = StageDReconstructionQAController(
        source_records=(record,),
        source_actions={record.scientific_address: action},
        pre_forward_guard=require_ready,
    )
    sampling = _Sampling(
        {
            "temperature": 0.7,
            "top_p": 1.0,
            "seed": 17,
            "extra_body": {"cache_salt": "exact"},
        }
    )

    with pytest.raises(RuntimeError, match="preflight"):
        controller.direct_sampling(_context(), sampling)
    ready = True
    directed = controller.direct_sampling(_context(), sampling)
    assert directed.model_dump(mode="json", exclude_none=False)["seed"] == 17


@pytest.mark.parametrize("tool_call", [False, True])
def test_frozen_response_retains_ids_tokens_logprobs_and_engine_finish(tool_call: bool) -> None:
    action = _action(tool_call=tool_call)
    payload = json.loads(frozen_engine_response_content(action))

    assert payload["request_id"] == "source-request-id"
    choice = payload["choices"][0]
    assert choice["token_ids"] == [20, 2]
    assert [row["logprob"] for row in choice["logprobs"]["content"]] == [-0.2, -0.1]
    assert choice["finish_reason"] == "stop"
    assert "routed_experts" not in choice
    assert "kept_tokens" not in choice


def test_shared_controller_commits_zero_call_override_before_return_and_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class Return:
        ticket: object
        response_content: bytes

    @dataclass(frozen=True)
    class Forward:
        ticket: object

    renderers = ModuleType("renderers")
    client_module = ModuleType("renderers.client")
    client_module.PreparedGenerateReturn = Return  # type: ignore[attr-defined]
    client_module.PreparedGenerateForward = Forward  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "renderers", renderers)
    monkeypatch.setitem(sys.modules, "renderers.client", client_module)

    class Ledger:
        def __init__(self) -> None:
            self.events: list[str] = []

        def put_evidence(self, value: bytes) -> str:
            self.events.append("evidence")
            return hashlib.sha256(value).hexdigest()

        def commit_execution_override(
            self,
            attempt: ExecutionAttempt,
            **kwargs: Any,
        ) -> ReplayOverrideTicket:
            self.events.append("override_committed")
            return ReplayOverrideTicket(
                attempt.attempt_id,
                "override-1",
                kwargs["address"],
                kwargs["action_digest"],
                kwargs["disposition"],
                kwargs["response_content_sha256"],
                kwargs["prompt_tokens"],
                kwargs["completion_tokens"],
                kwargs["counts_toward_logical_cost"],
            )

        def mark_execution_override_delivered(
            self,
            attempt: ExecutionAttempt,
            ticket: ReplayOverrideTicket,
            **kwargs: Any,
        ) -> None:
            assert attempt.attempt_id == "attempt-1"
            assert ticket.override_id == "override-1"
            assert kwargs["typed_response_sha256"]
            self.events.append("override_delivered")

    address = PolicyEventAddress(0, "root", 0, 0)
    record = parse_v2_rlm_provenance_payload(
        trace_id="trace-1",
        payload=cast(dict[str, Any], _context()["rlm"]),
    )
    graph = CausalProvenanceGraph(
        events=(address,),
        spawns=(),
        completion_snapshots=(EventCompletionSnapshot(address, ()),),
    )
    tracker = DynamicCausalTaintTracker(
        target=address,
        source_records=(record,),
        source_graph=graph,
    )
    action = _prepared_action()
    ledger = Ledger()
    attempt = ExecutionAttempt(
        "ledger",
        "group",
        "target",
        "arm-0",
        action.digest,
        1,
        0,
        "attempt-1",
    )

    class UnusedOracle:
        def seed_for(self, address: PolicyEventAddress) -> Any:
            raise AssertionError(address)

    controller = StageDReplayCallController(
        tracker=tracker,
        source_actions={address: action},
        target=address,
        candidate_action=action,
        seed_oracle=UnusedOracle(),
        ledger=ledger,  # type: ignore[arg-type]
        attempt=attempt,
    )
    controller.direct_sampling(
        _context(),
        _Sampling(
            {
                "temperature": 0.7,
                "top_p": 1.0,
                "seed": 17,
                "extra_body": {"cache_salt": "exact"},
            }
        ),
    )
    prepared = SimpleNamespace(
        application_request=action.key.request,
        engine_endpoint="http://engine/inference/v1/generate",
        engine_request=action.key.prepared_engine_request,
        engine_headers=canonical_json({"X-Session-ID": "trace-1"}),
        observer_context=canonical_json(_context()),
        prompt_token_ids=action.key.prompt_token_ids,
    )

    directive = __import__("asyncio").run(controller.before_forward(prepared))

    assert isinstance(directive, Return)
    assert ledger.events[-1] == "override_committed"
    __import__("asyncio").run(
        controller.after_raw_response(directive.ticket, directive.response_content)
    )
    response = SimpleNamespace(
        id=action.request_id,
        finish_reason=action.finish_reason,
        tokens=SimpleNamespace(
            prompt_ids=list(action.key.prompt_token_ids),
            completion_ids=list(action.action_token_ids),
            completion_logprobs=list(action.behavior_logprobs),
            routed_experts=None,
            kept_tokens=None,
        ),
        usage=SimpleNamespace(
            input_tokens=action.prompt_tokens,
            completion_tokens=action.completion_tokens,
        ),
        raw={"choices": [{"message": action.message}]},
    )
    __import__("asyncio").run(controller.after_response(directive.ticket, response))
    assert ledger.events[-1] == "override_delivered"
