"""Prepared-request observer binding live Verifiers calls to Stage-D receipts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.analysis.stage_d_source_producer import (
    PendingSourcePolicyCall,
    StageDSourceRolloutProducer,
    structural_child_target_id,
)
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress
from redco.contracts import canonical_json
from redco.integrations.verifiers_trace_v2 import (
    RecordedRLMProvenanceV2,
    parse_v2_rlm_provenance_payload,
)

AbortPhase = Literal[
    "post_unknown",
    "response_received",
    "response_parsed",
    "typed_response",
]


class PreparedRequestLike(Protocol):
    application_request: bytes
    engine_endpoint: str
    engine_request: bytes
    engine_headers: bytes
    observer_context: bytes
    prompt_token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StageDObserverIdentity:
    checkpoint_id: str
    base_model_manifest: bytes
    adapter_manifest: bytes | None
    tokenizer_manifest: bytes
    renderer_manifest: bytes
    sampler_conformance_manifest: bytes
    eos_token_id: int

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("observer checkpoint ID must be nonempty")
        for value, name in (
            (self.base_model_manifest, "base_model_manifest"),
            (self.tokenizer_manifest, "tokenizer_manifest"),
            (self.renderer_manifest, "renderer_manifest"),
            (self.sampler_conformance_manifest, "sampler_conformance_manifest"),
        ):
            if type(value) is not bytes or not value:
                raise ValueError(f"{name} must be nonempty immutable bytes")
        if self.adapter_manifest is not None and (
            type(self.adapter_manifest) is not bytes or not self.adapter_manifest
        ):
            raise ValueError("adapter_manifest must be bytes or None")
        if type(self.eos_token_id) is not int or self.eos_token_id < 0:
            raise ValueError("eos_token_id must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class StageDObserverProtocol:
    branch_count: int
    continuation_replicates: int
    failure_reward: float
    root_policy_turn_count: int
    maximum_observed_root_policy_turn_count: int = 4
    child_parent_event: PolicyEventAddress = field(
        default_factory=lambda: PolicyEventAddress(0, "root", 0, 0)
    )
    parent_tool_call_slot: int = 0

    def __post_init__(self) -> None:
        if type(self.branch_count) is not int or self.branch_count < 2:
            raise ValueError("branch_count must be an integer >= 2")
        if type(self.continuation_replicates) is not int or self.continuation_replicates < 1:
            raise ValueError("continuation_replicates must be an integer >= 1")
        if type(self.failure_reward) is not float:
            raise ValueError("failure_reward must be an explicit float")
        if type(self.root_policy_turn_count) is not int or self.root_policy_turn_count < 2:
            raise ValueError("observer protocol requires at least two root policy turns")
        if (
            type(self.maximum_observed_root_policy_turn_count) is not int
            or self.maximum_observed_root_policy_turn_count < self.root_policy_turn_count
        ):
            raise ValueError("observer protocol root-call ceiling is invalid")
        if self.child_parent_event.call_kind != "policy":
            raise ValueError("child parent event must be a policy call")
        if self.child_parent_event.depth != 0:
            raise ValueError("child parent event must be in the root session")
        if type(self.parent_tool_call_slot) is not int or self.parent_tool_call_slot < 0:
            raise ValueError("parent tool-call slot must be nonnegative")


@dataclass(frozen=True, slots=True)
class StageDPreparedTicket:
    pending: PendingSourcePolicyCall
    action_key: ExactActionKey


class StageDPreparedCallObserver:
    """Fail-closed adapter at the final renderer-to-engine POST boundary."""

    def __init__(
        self,
        *,
        producer: StageDSourceRolloutProducer,
        trace_id: str,
        identity: StageDObserverIdentity,
        protocol: StageDObserverProtocol,
        encode_action: Callable[
            [Mapping[str, Any], Mapping[str, Any], tuple[int, ...]],
            tuple[int, ...],
        ],
    ) -> None:
        if not trace_id:
            raise ValueError("observer trace ID must be nonempty")
        self._producer = producer
        self._trace_id = trace_id
        self._identity = identity
        self._protocol = protocol
        self._encode_action = encode_action
        self._root_turns: set[int] = set()

    async def before_forward(self, prepared: PreparedRequestLike) -> object:
        request = _canonical_object(prepared.application_request, "application request")
        engine = _canonical_object(prepared.engine_request, "engine request")
        headers = _canonical_object(prepared.engine_headers, "engine headers")
        context = _canonical_object(prepared.observer_context, "observer context")
        if context.get("trace_id") != self._trace_id:
            raise ValueError("prepared observer context changed trace ID")
        provenance_payload = context.get("rlm")
        if not isinstance(provenance_payload, dict):
            raise ValueError("prepared observer context lacks RLM provenance")
        provenance = parse_v2_rlm_provenance_payload(
            trace_id=self._trace_id,
            payload=provenance_payload,
        )
        sampling_params = engine.get("sampling_params")
        if not isinstance(sampling_params, dict):
            raise ValueError("prepared engine request lacks sampling parameters")
        routing_start = sampling_params.get("routed_experts_prompt_start")
        if routing_start is not None and not (
            provenance.depth == 0 and provenance.session_call_ordinal > 0
        ):
            raise ValueError("routed-expert boundary is only valid on a returning root")
        if not prepared.engine_endpoint.endswith("/inference/v1/generate"):
            raise ValueError("prepared engine endpoint is not the pinned generate route")
        if set(headers) != {"X-Session-ID"}:
            raise ValueError("prepared engine request contains unsupported headers")
        if headers.get("X-Session-ID") != self._trace_id:
            raise ValueError("prepared engine session header differs from trace ID")
        action_key = ExactActionKey.build_prepared(
            checkpoint_id=self._identity.checkpoint_id,
            base_model_manifest=self._identity.base_model_manifest,
            adapter_manifest=self._identity.adapter_manifest,
            tokenizer_manifest=self._identity.tokenizer_manifest,
            renderer_manifest=self._identity.renderer_manifest,
            sampler_conformance_manifest=self._identity.sampler_conformance_manifest,
            action_selection_policy="direct_single_sample",
            transport_retry_policy="fail_before_action_no_resample",
            request=request,
            prompt_token_ids=prepared.prompt_token_ids,
            prepared_engine_request=engine,
        )
        node_kind, target_id = self._classify(provenance)
        branch_selected = (
            node_kind == "child"
            and target_id is not None
            and self._producer.is_predeclared_child_target(target_id)
        )
        if branch_selected:
            assert target_id is not None
            snapshot = canonical_json(
                {
                    "schema_version": 1,
                    "domain": "redco-stage-d-pre-action-prepared-snapshot-v1",
                    "trace_id": self._trace_id,
                    "event_address": provenance.scientific_address.as_payload(),
                    "application_request": request,
                    "engine_endpoint": prepared.engine_endpoint,
                    "engine_request": engine,
                    "engine_headers": headers,
                    "observer_context": context,
                }
            )
            pending = self._producer.reserve_selected_child_policy_call(
                event_address=provenance.scientific_address,
                target_id=target_id,
                action_key=action_key,
                pre_action_snapshot=snapshot,
                branch_count=self._protocol.branch_count,
                continuation_replicates=self._protocol.continuation_replicates,
                failure_reward=self._protocol.failure_reward,
            )
        else:
            pending = self._producer.reserve_policy_call(
                event_address=provenance.scientific_address,
                action_key=action_key,
                node_kind=node_kind,
                target_id=target_id,
                branch_selected=False,
            )
        if node_kind == "root":
            self._root_turns.add(provenance.session_call_ordinal)
        return StageDPreparedTicket(pending, action_key)

    async def after_response(self, ticket: object, response: object) -> None:
        if type(ticket) is not StageDPreparedTicket:
            raise ValueError("prepared response ticket has the wrong type")
        tokens = getattr(response, "tokens", None)
        raw = getattr(response, "raw", None)
        usage = getattr(response, "usage", None)
        finish_reason = getattr(response, "finish_reason", None)
        if tokens is None or usage is None or not isinstance(raw, dict):
            raise ValueError("typed prepared response lacks tokens, usage, or raw bytes")
        action_token_ids = _integer_tuple(
            getattr(tokens, "completion_ids", None), "completion token IDs"
        )
        behavior_logprobs = _float_tuple(
            getattr(tokens, "completion_logprobs", None), "completion logprobs"
        )
        if len(action_token_ids) != len(behavior_logprobs):
            raise ValueError("typed prepared response token/logprob lengths differ")
        if not isinstance(finish_reason, str):
            raise ValueError("typed prepared response lacks a finish reason")
        if finish_reason == "length":
            raise ValueError("Stage-D source collection refuses truncated actions")
        prompt_tokens = _exact_int(getattr(usage, "input_tokens", None), "prompt usage")
        completion_tokens = _exact_int(
            getattr(usage, "completion_tokens", None), "completion usage"
        )
        message = _raw_message(raw)
        termination_kind, eos_token_id = self._termination(
            finish_reason,
            action_token_ids,
        )
        action = BehaviorAction.build(
            key=ticket.action_key,
            action_token_ids=action_token_ids,
            behavior_logprobs=behavior_logprobs,
            raw_transport_message=message,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            termination_kind=termination_kind,
            eos_token_id=eos_token_id,
            encode_action=lambda request, typed_message: self._encode_action(
                request,
                typed_message,
                ticket.action_key.prompt_token_ids,
            ),
        )
        self._producer.complete_policy_call(ticket.pending, action=action)

    async def abort(
        self,
        ticket: object,
        phase: AbortPhase,
        error: BaseException,
    ) -> None:
        if type(ticket) is not StageDPreparedTicket:
            raise ValueError("prepared abort ticket has the wrong type")
        self._producer.abort_policy_call(ticket.pending, phase=phase, error=error)

    def _classify(
        self,
        provenance: RecordedRLMProvenanceV2,
    ) -> tuple[Literal["root", "child"], str | None]:
        if provenance.depth == 0:
            expected_ordinal = len(self._root_turns)
            if (
                provenance.lineage != self._protocol.child_parent_event.lineage
                or provenance.call_kind != "policy"
                or provenance.turn != provenance.session_call_ordinal
                or provenance.session_call_ordinal != expected_ordinal
                or expected_ordinal
                >= self._protocol.maximum_observed_root_policy_turn_count
            ):
                raise ValueError("prepared root call is outside the frozen root session")
            return "root", None
        if (
            provenance.depth != 1
            or provenance.call_kind != "policy"
            or provenance.turn != 0
            or provenance.session_call_ordinal != 0
            or provenance.parent_lineage != self._protocol.child_parent_event.lineage
            or provenance.parent_call_ordinal
            != self._protocol.child_parent_event.session_call_ordinal
            or provenance.parent_turn != self._protocol.child_parent_event.turn
            or provenance.parent_tool_call_slot != self._protocol.parent_tool_call_slot
            or provenance.spawn_ordinal not in {0, 1, 2, 3}
            or self._protocol.child_parent_event.session_call_ordinal not in self._root_turns
        ):
            raise ValueError("prepared call is outside the frozen two-child scaffold")
        assert provenance.spawn_ordinal is not None
        return (
            "child",
            structural_child_target_id(
                self._protocol.child_parent_event,
                rollout_id=self._trace_id,
                parent_tool_call_slot=self._protocol.parent_tool_call_slot,
                spawn_ordinal=provenance.spawn_ordinal,
            ),
        )

    def _termination(
        self,
        finish_reason: object,
        action_token_ids: tuple[int, ...],
    ) -> tuple[Literal["eos", "max_tokens", "tool_calls"], int | None]:
        if finish_reason == "tool_calls":
            return "tool_calls", None
        if finish_reason == "length":
            return "max_tokens", None
        if finish_reason != "stop" or action_token_ids[-1] != self._identity.eos_token_id:
            raise ValueError("prepared response has an unsupported termination")
        return "eos", self._identity.eos_token_id


def require_zero_retry_configuration(
    *,
    agent_max_retries: int,
    client_max_retries: int,
) -> None:
    """Reject every retry layer before a Stage-D observer is installed."""
    if agent_max_retries != 0 or client_max_retries != 0:
        raise ValueError("Stage-D observed rollouts require zero retries at every layer")


def _canonical_object(value: bytes, name: str) -> dict[str, Any]:
    if type(value) is not bytes:
        raise ValueError(f"prepared {name} must be immutable bytes")
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"prepared {name} must be JSON") from error
    if not isinstance(parsed, dict) or canonical_json(parsed) != value:
        raise ValueError(f"prepared {name} must be a canonical JSON object")
    return parsed


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"{name} must be a nonempty sequence")
    result = tuple(value)
    if any(type(item) is not int or item < 0 for item in result):
        raise ValueError(f"{name} must contain nonnegative integers")
    return result


def _exact_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _float_tuple(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"{name} must be a nonempty sequence")
    result = tuple(value)
    if any(type(item) is not float for item in result):
        raise ValueError(f"{name} must contain explicit floats")
    return result


def _raw_message(raw: Mapping[str, Any]) -> dict[str, Any]:
    choices = raw.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or not isinstance(choices[0].get("message"), dict)
    ):
        raise ValueError("typed prepared response lacks one raw transport message")
    return dict(choices[0]["message"])
