"""Exact Stage-D replay sampling and zero-call engine-response contracts."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar, cast

from redco.analysis.stage_d_action_closure import ActionClosureWatchdog
from redco.analysis.stage_d_dynamic_taint import (
    DynamicCausalTaintTracker,
    ReplayDisposition,
)
from redco.analysis.stage_d_exact_action import BehaviorAction
from redco.analysis.stage_d_receipt_ledger import (
    ExecutionAttempt,
    ModelCallAttempt,
    ReplayOverrideTicket,
    StageDReceiptLedger,
)
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress, ScheduledSeed
from redco.contracts import canonical_json
from redco.integrations.verifiers_trace_v2 import (
    RecordedRLMProvenanceV2,
    parse_v2_rlm_provenance_payload,
)


class SamplingConfigLike(Protocol):
    """The immutable Pydantic sampling surface used by pinned Verifiers."""

    def model_dump(self, *, mode: str, exclude_none: bool) -> dict[str, Any]: ...

    def model_copy(
        self,
        *,
        update: Mapping[str, Any],
        deep: bool,
    ) -> SamplingConfigLike: ...


class PreparedRequestLike(Protocol):
    application_request: bytes
    engine_endpoint: str
    engine_request: bytes
    engine_headers: bytes
    observer_context: bytes
    prompt_token_ids: tuple[int, ...]


class SeedOracleLike(Protocol):
    def seed_for(self, address: PolicyEventAddress) -> ScheduledSeed: ...


_T = TypeVar("_T")


async def _discard_provider_operation(operation: Awaitable[Any]) -> None:
    """Close or cancel a forbidden QA provider operation without awaiting it."""

    if isinstance(operation, asyncio.Future):
        operation.cancel()
        try:
            await operation
        except BaseException:
            return
        return
    close = getattr(operation, "close", None)
    if not callable(close):
        raise TypeError("QA provider operation is not safely closable")
    close()


@dataclass(frozen=True, slots=True)
class _ScientificReplayWatchdog:
    """Compose one campaign watchdog with the authenticated replay mode."""

    watchdog: ActionClosureWatchdog
    mode: Literal["qa", "execution"]

    def __post_init__(self) -> None:
        if type(self.watchdog) is not ActionClosureWatchdog:
            raise TypeError("scientific replay requires the exact watchdog owner")
        if self.mode not in {"qa", "execution"}:
            raise ValueError("scientific replay watchdog mode is invalid")

    async def run_provider_call(self, operation: Awaitable[_T]) -> _T:
        if self.mode == "qa":
            await _discard_provider_operation(operation)
            raise RuntimeError("reconstruction QA forbids provider calls")
        return cast(_T, await self.watchdog.run_provider_call(operation))

    async def run_concurrent_children(self, operation: Awaitable[_T]) -> _T:
        return cast(_T, await self.watchdog.run_concurrent_children(operation))


def preload_replay_runtime_types() -> None:
    """Fail before ledger mutation if the pinned renderer interception API is absent."""
    from renderers.client import (
        PreparedGenerateForward,
        PreparedGenerateReturn,
    )

    if not isinstance(PreparedGenerateForward, type) or not isinstance(
        PreparedGenerateReturn, type
    ):
        raise TypeError("pinned renderer prepared-response types are invalid")


@dataclass(frozen=True, slots=True)
class SamplingOverride:
    """The only two request fields a Stage-D replay call may redirect."""

    address: PolicyEventAddress
    seed: int
    cache_salt: str

    def __post_init__(self) -> None:
        if type(self.address) is not PolicyEventAddress:
            raise ValueError("sampling override requires a structural event address")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("sampling override seed must be a nonnegative integer")
        if not isinstance(self.cache_salt, str) or not self.cache_salt:
            raise ValueError("sampling override cache salt must be nonempty")

    @classmethod
    def from_scheduled_seed(cls, scheduled: ScheduledSeed) -> SamplingOverride:
        if type(scheduled) is not ScheduledSeed:
            raise ValueError("scheduled seed must be exact")
        return cls(scheduled.address, scheduled.seed, scheduled.cache_salt)

    @classmethod
    def from_action(
        cls,
        address: PolicyEventAddress,
        action: BehaviorAction,
    ) -> SamplingOverride:
        if type(action) is not BehaviorAction:
            raise ValueError("action sampling override requires an exact behavior action")
        request = json.loads(action.key.request)
        if not isinstance(request, dict):
            raise ValueError("behavior action request is not an object")
        extra_body = request.get("extra_body")
        if not isinstance(extra_body, dict) or set(extra_body) != {"cache_salt"}:
            raise ValueError("behavior action lacks one exact cache salt")
        salt = extra_body["cache_salt"]
        if not isinstance(salt, str) or not salt:
            raise ValueError("behavior action cache salt is invalid")
        return cls(address, action.key.sampler.seed, salt)


class ExactSamplingDirector:
    """Apply a structural seed plan before both application and engine preparation."""

    def __init__(
        self,
        plan: Callable[[RecordedRLMProvenanceV2], SamplingOverride],
    ) -> None:
        self._plan = plan
        self._lock = threading.Lock()
        self._directed: dict[PolicyEventAddress, SamplingOverride] = {}

    def direct_sampling(
        self,
        observer_context: Mapping[str, Any],
        sampling: SamplingConfigLike,
    ) -> SamplingConfigLike:
        context = dict(observer_context)
        trace_id = context.get("trace_id")
        payload = context.get("rlm")
        if not isinstance(trace_id, str) or not isinstance(payload, dict):
            raise ValueError("sampling direction requires exact RLM provenance")
        provenance = parse_v2_rlm_provenance_payload(
            trace_id=trace_id,
            payload=payload,
        )
        override = self._plan(provenance)
        if (
            type(override) is not SamplingOverride
            or override.address.as_payload()
            != provenance.scientific_address.as_payload()
        ):
            raise ValueError("sampling plan returned a different structural address")
        before = _sampling_payload(sampling)
        extra_body = before.get("extra_body")
        if not isinstance(extra_body, dict) or set(extra_body) - {"cache_salt"}:
            raise ValueError("base sampling extra_body is outside the frozen allowlist")
        directed = sampling.model_copy(
            update={
                "seed": override.seed,
                "extra_body": {"cache_salt": override.cache_salt},
            },
            deep=True,
        )
        if _sampling_payload(sampling) != before:
            raise ValueError("sampling direction mutated the base config")
        after = _sampling_payload(directed)
        changed = _changed_paths(before, after)
        if not changed <= {("seed",), ("extra_body", "cache_salt")}:
            raise ValueError(f"sampling direction changed forbidden fields: {sorted(changed)}")
        if after.get("seed") != override.seed or after.get("extra_body") != {
            "cache_salt": override.cache_salt
        }:
            raise ValueError("sampling direction did not install the exact seed and salt")
        with self._lock:
            if provenance.scientific_address in self._directed:
                raise ValueError("one replay attempted to direct a policy address twice")
            self._directed[provenance.scientific_address] = override
        return directed

    def consume_override(self, address: PolicyEventAddress) -> SamplingOverride:
        """Bind the later prepared-request decision to the prior sampling direction."""
        with self._lock:
            try:
                return self._directed.pop(address)
            except KeyError as error:
                raise ValueError("prepared request lacks a unique sampling direction") from error

    def assert_drained(self) -> None:
        with self._lock:
            if self._directed:
                raise ValueError("sampling directions remain unconsumed")


@dataclass(frozen=True, slots=True)
class _ReplayPlan:
    address: PolicyEventAddress
    disposition: ReplayDisposition
    override: SamplingOverride
    action: BehaviorAction | None
    scheduled_seed: ScheduledSeed | None
    counts_toward_logical_continuation: bool


@dataclass(frozen=True, slots=True)
class _GeneratedTicket:
    call: ModelCallAttempt


@dataclass(frozen=True, slots=True)
class _ReconstructionQATicket:
    address: PolicyEventAddress
    action: BehaviorAction


class StageDReconstructionQAController:
    """Replay every frozen source policy event through the parser with zero POSTs."""

    def __init__(
        self,
        *,
        source_records: tuple[RecordedRLMProvenanceV2, ...],
        source_actions: Mapping[PolicyEventAddress, BehaviorAction],
        watchdog: _ScientificReplayWatchdog,
        pre_forward_guard: Callable[[], None] | None = None,
    ) -> None:
        records = {record.scientific_address: record for record in source_records}
        actions = dict(source_actions)
        if not records or set(records) != set(actions):
            raise ValueError("reconstruction QA requires one action for every source event")
        if len(records) != len(source_records):
            raise ValueError("reconstruction QA source addresses must be unique")
        self._records = records
        self._actions = actions
        self._pending: dict[PolicyEventAddress, BehaviorAction] = {}
        self._raw_observed: set[PolicyEventAddress] = set()
        self._consumed: set[PolicyEventAddress] = set()
        self._lock = threading.Lock()
        if watchdog.mode != "qa":
            raise ValueError("reconstruction QA requires the QA watchdog mode")
        self._watchdog = watchdog
        self._pre_forward_guard = pre_forward_guard
        self._director = ExactSamplingDirector(self._plan)

    def direct_sampling(
        self,
        observer_context: Mapping[str, Any],
        sampling: SamplingConfigLike,
    ) -> SamplingConfigLike:
        self._guard()
        return self._director.direct_sampling(observer_context, sampling)

    async def run_provider_call(self, operation: Awaitable[_T]) -> _T:
        return await self._watchdog.run_provider_call(operation)

    async def run_concurrent_children(self, operation: Awaitable[_T]) -> _T:
        return await self._watchdog.run_concurrent_children(operation)

    async def before_forward(self, prepared: PreparedRequestLike) -> object:
        self._guard()
        provenance = _prepared_provenance(prepared)
        address = provenance.scientific_address
        override = self._director.consume_override(address)
        with self._lock:
            try:
                action = self._pending.pop(address)
            except KeyError as error:
                raise ValueError("QA prepared request lacks one frozen replay plan") from error
        _require_prepared_sampling(prepared, override)
        _require_prepared_action(prepared, action)
        from renderers.client import PreparedGenerateReturn

        return PreparedGenerateReturn(
            ticket=_ReconstructionQATicket(address, action),
            response_content=frozen_engine_response_content(action),
        )

    async def after_response(self, ticket: object, response: object) -> None:
        if type(ticket) is not _ReconstructionQATicket:
            raise ValueError("QA response ticket has the wrong type")
        with self._lock:
            if ticket.address not in self._raw_observed:
                raise ValueError("QA typed response predates its exact raw response")
        _require_typed_action_response(response, ticket.action)
        with self._lock:
            if ticket.address in self._consumed:
                raise ValueError("QA source event was delivered twice")
            self._raw_observed.remove(ticket.address)
            self._consumed.add(ticket.address)

    async def after_raw_response(self, ticket: object, response_content: bytes) -> None:
        if type(ticket) is not _ReconstructionQATicket:
            raise ValueError("QA raw-response ticket has the wrong type")
        if response_content != frozen_engine_response_content(ticket.action):
            raise ValueError("QA frozen response bytes changed before parsing")
        with self._lock:
            if ticket.address in self._raw_observed or ticket.address in self._consumed:
                raise ValueError("QA raw response was delivered twice")
            self._raw_observed.add(ticket.address)

    async def abort(self, ticket: object, phase: str, error: BaseException) -> None:
        if type(ticket) is not _ReconstructionQATicket:
            raise ValueError("QA abort ticket has the wrong type")
        if not phase or not isinstance(error, BaseException):
            raise ValueError("QA abort lacks an exact terminal failure")

    def finalize(self) -> None:
        self._director.assert_drained()
        with self._lock:
            if self._pending or self._raw_observed or self._consumed != set(self._records):
                raise ValueError("QA did not consume the complete frozen source trace")

    def _plan(self, provenance: RecordedRLMProvenanceV2) -> SamplingOverride:
        address = provenance.scientific_address
        source = self._records.get(address)
        action = self._actions.get(address)
        if source is None or action is None:
            raise ValueError("QA replay produced an unknown source policy event")
        _require_same_qa_provenance(source, provenance)
        with self._lock:
            if address in self._pending or address in self._consumed:
                raise ValueError("QA replay attempted one source event twice")
            self._pending[address] = action
        return SamplingOverride.from_action(address, action)

    def _guard(self) -> None:
        if self._pre_forward_guard is not None:
            self._pre_forward_guard()


class StageDReplayCallController:
    """One shared sampling director and renderer observer for an execution."""

    def __init__(
        self,
        *,
        tracker: DynamicCausalTaintTracker,
        source_actions: Mapping[PolicyEventAddress, BehaviorAction],
        target: PolicyEventAddress,
        candidate_action: BehaviorAction,
        seed_oracle: SeedOracleLike,
        ledger: StageDReceiptLedger,
        attempt: ExecutionAttempt,
        watchdog: _ScientificReplayWatchdog,
        pre_forward_guard: Callable[[], None] | None = None,
    ) -> None:
        actions = dict(source_actions)
        if target not in actions:
            raise ValueError("replay controller target lacks a source action")
        if type(candidate_action) is not BehaviorAction:
            raise ValueError("replay controller candidate action must be exact")
        self._tracker = tracker
        self._source_actions = actions
        self._target = target
        self._candidate_action = candidate_action
        self._seed_oracle = seed_oracle
        self._ledger = ledger
        self._attempt = attempt
        self._lock = threading.Lock()
        if watchdog.mode != "execution":
            raise ValueError("scientific replay requires the execution watchdog mode")
        self._watchdog = watchdog
        self._plans: dict[PolicyEventAddress, _ReplayPlan] = {}
        self._logical_downstream_observed = False
        self._target_injection_delivered = False
        self._pre_forward_guard = pre_forward_guard
        self._director = ExactSamplingDirector(self._plan)

    def direct_sampling(
        self,
        observer_context: Mapping[str, Any],
        sampling: SamplingConfigLike,
    ) -> SamplingConfigLike:
        self._guard()
        return self._director.direct_sampling(observer_context, sampling)

    async def run_provider_call(self, operation: Awaitable[_T]) -> _T:
        return await self._watchdog.run_provider_call(operation)

    async def run_concurrent_children(self, operation: Awaitable[_T]) -> _T:
        return await self._watchdog.run_concurrent_children(operation)

    async def before_forward(self, prepared: PreparedRequestLike) -> object:
        self._guard()
        provenance = _prepared_provenance(prepared)
        address = provenance.scientific_address
        override = self._director.consume_override(address)
        with self._lock:
            try:
                plan = self._plans.pop(address)
            except KeyError as error:
                raise ValueError("prepared replay request lacks one causal plan") from error
        if plan.override != override:
            raise ValueError("prepared replay request changed its sampling plan")
        _require_prepared_sampling(prepared, override)
        request_evidence = _prepared_evidence(prepared)
        request_sha256 = self._ledger.put_evidence(request_evidence)
        if plan.disposition is ReplayDisposition.GENERATE:
            if plan.scheduled_seed is None or plan.action is not None:
                raise ValueError("generated replay plan is internally inconsistent")
            call = self._ledger.mark_execution_model_call_started(
                self._attempt,
                address=address,
                scheduled_seed=plan.scheduled_seed,
                request_sha256=request_sha256,
            )
            from renderers.client import (
                PreparedGenerateForward,
            )

            return PreparedGenerateForward(_GeneratedTicket(call))
        if plan.action is None or plan.scheduled_seed is not None:
            raise ValueError("zero-call replay plan is internally inconsistent")
        _require_prepared_action(prepared, plan.action)
        content = frozen_engine_response_content(plan.action)
        content_sha256 = self._ledger.put_evidence(content)
        ticket = self._ledger.commit_execution_override(
            self._attempt,
            address=address,
            action_digest=plan.action.digest,
            disposition=cast(Literal["reuse", "inject"], plan.disposition.value),
            request_sha256=request_sha256,
            response_content_sha256=content_sha256,
            prompt_tokens=plan.action.prompt_tokens,
            completion_tokens=plan.action.completion_tokens,
            counts_toward_logical_cost=plan.counts_toward_logical_continuation,
        )
        from renderers.client import PreparedGenerateReturn

        return PreparedGenerateReturn(ticket=ticket, response_content=content)

    async def after_response(self, ticket: object, response: object) -> None:
        evidence = _typed_response_evidence(response)
        response_sha256 = self._ledger.put_evidence(evidence)
        if type(ticket) is _GeneratedTicket:
            usage = getattr(response, "usage", None)
            if usage is None:
                raise ValueError("generated replay response lacks exact usage")
            self._ledger.complete_execution_model_call(
                self._attempt,
                ticket.call,
                prompt_tokens=_exact_nonnegative_int(
                    getattr(usage, "input_tokens", None),
                    "prompt tokens",
                ),
                completion_tokens=_exact_nonnegative_int(
                    getattr(usage, "completion_tokens", None),
                    "completion tokens",
                ),
                response_sha256=response_sha256,
            )
            return
        if type(ticket) is ReplayOverrideTicket:
            override_ticket = cast(ReplayOverrideTicket, ticket)
            action = (
                self._candidate_action
                if override_ticket.disposition == "inject"
                else self._source_actions.get(override_ticket.address)
            )
            if action is None:
                raise ValueError("delivered replay override lacks its exact action")
            _require_typed_action_response(response, action)
            self._ledger.mark_execution_override_delivered(
                self._attempt,
                override_ticket,
                typed_response_sha256=response_sha256,
            )
            if override_ticket.disposition == "inject":
                with self._lock:
                    if self._target_injection_delivered:
                        raise ValueError("target injection was delivered twice")
                    self._target_injection_delivered = True
            return
        raise ValueError("prepared replay response ticket has the wrong type")

    async def after_raw_response(self, ticket: object, response_content: bytes) -> None:
        """Persist exact provider bytes before the renderer parses them."""
        if type(response_content) is not bytes or not response_content:
            raise ValueError("prepared replay raw response must be nonempty bytes")
        response_sha256 = self._ledger.put_evidence(response_content)
        if type(ticket) is _GeneratedTicket:
            self._ledger.mark_execution_response_observed(
                self._attempt,
                ticket.call,
                response_sha256=response_sha256,
            )
            return
        if type(ticket) is ReplayOverrideTicket:
            override_ticket = cast(ReplayOverrideTicket, ticket)
            if response_sha256 != override_ticket.response_content_sha256:
                raise ValueError("prepared replay override response bytes changed")
            return
        raise ValueError("prepared replay raw response ticket has the wrong type")

    async def abort(
        self,
        ticket: object,
        phase: str,
        error: BaseException,
    ) -> None:
        """Leave the durable call/override unresolved so restart is terminal."""
        if type(ticket) not in {_GeneratedTicket, ReplayOverrideTicket}:
            raise ValueError("prepared replay abort ticket has the wrong type")
        if phase not in {
            "post_unknown",
            "response_received",
            "response_parsed",
            "typed_response",
        }:
            raise ValueError("prepared replay abort phase is unknown")
        if not isinstance(error, BaseException):
            raise ValueError("prepared replay abort lacks its failure")

    def finalize(self, *, allow_terminal_truncation: bool = False) -> None:
        """Close topology coverage only after every prepared call completed."""
        if allow_terminal_truncation:
            if not self.target_injection_delivered:
                raise ValueError("terminal truncation predates target delivery")
            self._tracker.finalize_terminal_truncation()
        else:
            self._tracker.finalize()
        self._director.assert_drained()
        with self._lock:
            if self._plans:
                raise ValueError("causal replay plans remain unconsumed")

    @property
    def logical_downstream_observed(self) -> bool:
        with self._lock:
            return self._logical_downstream_observed

    @property
    def target_injection_delivered(self) -> bool:
        """Whether the frozen target action completed the ordinary typed response path."""
        with self._lock:
            return self._target_injection_delivered

    def _plan(self, provenance: RecordedRLMProvenanceV2) -> SamplingOverride:
        decision = self._tracker.observe(provenance)
        if decision.disposition is ReplayDisposition.REUSE:
            action = self._source_actions.get(decision.address)
            if action is None:
                raise ValueError("reused replay event lacks its source action")
            override = SamplingOverride.from_action(decision.address, action)
            scheduled = None
        elif decision.disposition is ReplayDisposition.INJECT:
            if decision.address != self._target:
                raise ValueError("causal tracker injected a non-target event")
            action = self._candidate_action
            override = SamplingOverride.from_action(decision.address, action)
            scheduled = None
        else:
            action = None
            scheduled = self._seed_oracle.seed_for(decision.address)
            override = SamplingOverride.from_scheduled_seed(scheduled)
        plan = _ReplayPlan(
            decision.address,
            decision.disposition,
            override,
            action,
            scheduled,
            decision.counts_toward_logical_continuation,
        )
        with self._lock:
            if decision.address in self._plans:
                raise ValueError("one replay policy event received two causal plans")
            self._plans[decision.address] = plan
            if (
                decision.disposition is ReplayDisposition.GENERATE
                or decision.counts_toward_logical_continuation
            ):
                self._logical_downstream_observed = True
        return override

    def _guard(self) -> None:
        if self._pre_forward_guard is not None:
            self._pre_forward_guard()


def frozen_engine_response_content(action: BehaviorAction) -> bytes:
    """Build the canonical raw engine response used by a zero-POST replay return."""
    if type(action) is not BehaviorAction:
        raise ValueError("frozen response requires an exact behavior action")
    finish_reason = "stop" if action.finish_reason == "tool_calls" else action.finish_reason
    payload = {
        "request_id": action.request_id,
        "choices": [
            {
                "index": 0,
                "token_ids": list(action.action_token_ids),
                "logprobs": {
                    "content": [
                        {
                            "token": f"token_id:{token_id}",
                            "logprob": logprob,
                        }
                        for token_id, logprob in zip(
                            action.action_token_ids,
                            action.behavior_logprobs,
                            strict=True,
                        )
                    ]
                },
                "finish_reason": finish_reason,
            }
        ],
    }
    return cast(bytes, canonical_json(payload))


def _require_same_qa_provenance(
    source: RecordedRLMProvenanceV2,
    replay: RecordedRLMProvenanceV2,
) -> None:
    """Allow a fresh trace ID while requiring the exact frozen causal structure."""
    fields = (
        "depth",
        "turn",
        "call_kind",
        "lineage",
        "session_call_ordinal",
        "parent_turn",
        "parent_lineage",
        "parent_call_ordinal",
        "parent_tool_call_slot",
        "spawn_ordinal",
        "episode_spawn_ordinal",
        "completed_predecessor_spawn_ordinals",
        "completed_episode_spawn_ordinals",
    )
    if any(getattr(source, name) != getattr(replay, name) for name in fields):
        raise ValueError("QA replay changed the frozen source causal structure")


def _prepared_provenance(prepared: PreparedRequestLike) -> RecordedRLMProvenanceV2:
    context = _canonical_object(prepared.observer_context, "observer context")
    trace_id = context.get("trace_id")
    payload = context.get("rlm")
    if not isinstance(trace_id, str) or not isinstance(payload, dict):
        raise ValueError("prepared replay request lacks exact RLM provenance")
    return parse_v2_rlm_provenance_payload(trace_id=trace_id, payload=payload)


def _require_prepared_sampling(
    prepared: PreparedRequestLike,
    override: SamplingOverride,
) -> None:
    application = _canonical_object(prepared.application_request, "application request")
    engine = _canonical_object(prepared.engine_request, "engine request")
    extra_body = application.get("extra_body")
    sampling = engine.get("sampling_params")
    if (
        application.get("seed") != override.seed
        or not isinstance(extra_body, dict)
        or extra_body.get("cache_salt") != override.cache_salt
        or not isinstance(sampling, dict)
        or sampling.get("seed") != override.seed
        or engine.get("cache_salt") != override.cache_salt
    ):
        raise ValueError("application and engine requests disagree with directed sampling")


def _require_prepared_action(
    prepared: PreparedRequestLike,
    action: BehaviorAction,
) -> None:
    if (
        prepared.application_request != action.key.request
        or prepared.prompt_token_ids != action.key.prompt_token_ids
        or action.key.prepared_engine_request is None
        or prepared.engine_request != action.key.prepared_engine_request
    ):
        raise ValueError("zero-call replay request differs from the exact behavior action")


def _prepared_evidence(prepared: PreparedRequestLike) -> bytes:
    return cast(
        bytes,
        canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-replay-prepared-request-v1",
                "application_request": _canonical_object(
                    prepared.application_request,
                    "application request",
                ),
                "engine_endpoint": prepared.engine_endpoint,
                "engine_request": _canonical_object(
                    prepared.engine_request,
                    "engine request",
                ),
                "engine_headers": _canonical_object(
                    prepared.engine_headers,
                    "engine headers",
                ),
                "observer_context": _canonical_object(
                    prepared.observer_context,
                    "observer context",
                ),
                "prompt_token_ids": list(prepared.prompt_token_ids),
            }
        ),
    )


def _typed_response_evidence(response: object) -> bytes:
    tokens = getattr(response, "tokens", None)
    usage = getattr(response, "usage", None)
    raw = getattr(response, "raw", None)
    if tokens is None or usage is None or not isinstance(raw, dict):
        raise ValueError("typed replay response lacks tokens, usage, or raw response")
    if (
        getattr(tokens, "routed_experts", None) is not None
        or getattr(tokens, "kept_tokens", None) is not None
    ):
        raise ValueError("Stage-D replay forbids token sidecars")
    return cast(
        bytes,
        canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-replay-typed-response-v1",
                "request_id": getattr(response, "id", None),
                "finish_reason": getattr(response, "finish_reason", None),
                "message": _response_message(raw),
                "prompt_token_ids": _integer_list(
                    getattr(tokens, "prompt_ids", None),
                    "prompt token IDs",
                ),
                "completion_token_ids": _integer_list(
                    getattr(tokens, "completion_ids", None),
                    "completion token IDs",
                ),
                "completion_logprobs": _float_list(
                    getattr(tokens, "completion_logprobs", None),
                    "completion logprobs",
                ),
                "usage": {
                    "prompt_tokens": _exact_nonnegative_int(
                        getattr(usage, "input_tokens", None),
                        "prompt tokens",
                    ),
                    "completion_tokens": _exact_nonnegative_int(
                        getattr(usage, "completion_tokens", None),
                        "completion tokens",
                    ),
                },
            }
        ),
    )


def _require_typed_action_response(response: object, action: BehaviorAction) -> None:
    evidence = json.loads(_typed_response_evidence(response))
    if (
        evidence.get("request_id") != action.request_id
        or evidence.get("finish_reason") != action.finish_reason
        or tuple(evidence.get("completion_token_ids", ())) != action.action_token_ids
        or tuple(evidence.get("completion_logprobs", ())) != action.behavior_logprobs
        or evidence.get("message") != action.message
        or evidence.get("usage", {}).get("prompt_tokens") != action.prompt_tokens
        or evidence.get("usage", {}).get("completion_tokens")
        != action.completion_tokens
    ):
        raise ValueError("delivered replay response differs from its exact behavior action")


def _response_message(raw: Mapping[str, Any]) -> dict[str, Any]:
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("typed replay response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("typed replay response lacks its assistant message")
    return dict(choice["message"])


def _canonical_object(value: bytes, name: str) -> dict[str, Any]:
    if type(value) is not bytes:
        raise ValueError(f"{name} must be immutable bytes")
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or canonical_json(parsed) != value:
        raise ValueError(f"{name} must be canonical JSON")
    return parsed


def _integer_list(value: object, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or any(
        type(item) is not int or item < 0 for item in value
    ):
        raise ValueError(f"{name} must be nonnegative integers")
    return list(value)


def _float_list(value: object, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or any(
        type(item) is not float for item in value
    ):
        raise ValueError(f"{name} must be explicit floats")
    return list(value)


def _exact_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _sampling_payload(sampling: SamplingConfigLike) -> dict[str, Any]:
    value = sampling.model_dump(mode="json", exclude_none=False)
    if not isinstance(value, dict):
        raise ValueError("sampling config did not serialize to an object")
    return value


def _changed_paths(
    before: object,
    after: object,
    prefix: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if isinstance(before, dict) and isinstance(after, dict):
        changed: set[tuple[str, ...]] = set()
        for key in set(before) | set(after):
            name = str(key)
            if key not in before or key not in after:
                changed.add((*prefix, name))
            else:
                changed.update(_changed_paths(before[key], after[key], (*prefix, name)))
        return changed
    return set() if before == after else {prefix}
