"""Trusted transcript-replaying model port for the Stage-D runtime worker."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any, Protocol, cast

from redco.analysis.stage_d_evaluation_capabilities import (
    EvaluationClientSession,
    EvaluationTaskAttempt,
)
from redco.analysis.stage_d_evaluation_codec import canonical_object
from redco.analysis.stage_d_evaluation_http import (
    dispatch_reserved_local_http_once,
)
from redco.analysis.stage_d_evaluation_ledger import StageDEvaluationLedger
from redco.analysis.stage_d_evaluation_transport import build_local_post_transport
from redco.analysis.stage_d_openai_response import (
    ParsedOpenAIResponse,
    parse_openai_response,
)
from redco.contracts import EventAddress, canonical_json


class RequestSerializer(Protocol):
    def __call__(
        self,
        payload: dict[str, Any],
        *,
        seed: int,
        cache_salt: str,
    ) -> bytes: ...


@dataclass(frozen=True, slots=True)
class EvaluationCallSpec:
    address: EventAddress
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, dict) or canonical_json(self.payload) == b"{}":
            raise ValueError("evaluation call payload must be a nonempty JSON object")


class EvaluationModelPort:
    def __init__(
        self,
        *,
        ledger: StageDEvaluationLedger,
        task: EvaluationTaskAttempt,
        session: EvaluationClientSession,
        serialize_request: RequestSerializer,
        timeout_seconds: float,
        max_calls: int,
        max_completion_tokens: int,
    ) -> None:
        if max_calls < 1 or max_completion_tokens < 1:
            raise ValueError("evaluation model-port budgets must be positive")
        self._ledger = ledger
        self._task = task
        self._session = session
        self._serialize = serialize_request
        self._timeout_seconds = timeout_seconds
        self._max_calls = max_calls
        self._max_completion_tokens = max_completion_tokens
        self._ordinal = 0
        self._completion_tokens = 0

    def call(self, spec: EvaluationCallSpec) -> ParsedOpenAIResponse:
        if self._ordinal >= self._max_calls:
            raise RuntimeError("evaluation task exceeded its frozen call budget")
        address_bytes = canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-heldout-event-address-v1",
                "task_attempt_id": self._task.task_attempt_id,
                "call_ordinal": self._ordinal,
                "address": spec.address.as_payload(),
            }
        )
        seed, cache_salt = _sampling_identity(self._task, address_bytes)
        request = self._serialize(spec.payload, seed=seed, cache_salt=cache_salt)
        request_payload = canonical_object(request, "evaluation serialized request")
        extra = request_payload.get("extra_body")
        if (
            request_payload.get("seed") != seed
            or not isinstance(extra, dict)
            or extra.get("cache_salt") != cache_salt
        ):
            raise ValueError("evaluation request serializer changed seed or cache salt")
        endpoint = self._ledger.manifest.program(self._task.unit.arm, "server").endpoint
        transport, _ = build_local_post_transport(
            endpoint=endpoint,
            request_body=request,
            timeout_seconds=self._timeout_seconds,
        )
        call = self._ledger.reserve_call(
            self._task,
            session=self._session,
            event_address_bytes=address_bytes,
            seed=seed,
            cache_salt=cache_salt,
            request_body_bytes=request,
            transport_bytes=transport,
            call_ordinal=self._ordinal,
        )
        state = self._ledger.call_state(call)
        if state.outcome_sha256 is not None:
            stored = self._ledger.finalized_response_bytes(call)
            if stored is None:
                raise RuntimeError("finalized evaluation call lacks parsed response")
            parsed = ParsedOpenAIResponse.from_bytes(stored)
        elif state.response_envelope_sha256 is not None:
            envelope = canonical_object(
                self._ledger.evidence.get(state.response_envelope_sha256),
                "evaluation response envelope",
            )
            raw = self._ledger.evidence.get(cast(str, state.raw_response_sha256))
            parsed = parse_openai_response(raw, status_code=envelope["status_code"])
            self._ledger.finalize_call(
                call,
                session=self._session,
                parsed_response_bytes=parsed.to_bytes(),
                prompt_tokens=parsed.prompt_tokens,
                completion_tokens=parsed.completion_tokens,
                wall_seconds=envelope["wall_seconds"],
                gpu_seconds=0.0,
                finish_kind=parsed.finish_kind,
            )
        elif state.dispatch_receipt_sha256 is not None:
            raise RuntimeError("evaluation call has an ambiguous dispatched outcome")
        else:
            result = dispatch_reserved_local_http_once(
                ledger=self._ledger,
                call=call,
                session=self._session,
                timeout_seconds=self._timeout_seconds,
            )
            parsed = parse_openai_response(
                result.raw_response,
                status_code=result.status_code,
            )
            envelope = canonical_object(result.response_envelope, "evaluation response envelope")
            self._ledger.finalize_call(
                call,
                session=self._session,
                parsed_response_bytes=parsed.to_bytes(),
                prompt_tokens=parsed.prompt_tokens,
                completion_tokens=parsed.completion_tokens,
                wall_seconds=envelope["wall_seconds"],
                gpu_seconds=0.0,
                finish_kind=parsed.finish_kind,
            )
        self._completion_tokens += parsed.completion_tokens
        if self._completion_tokens > self._max_completion_tokens:
            raise RuntimeError("evaluation task exceeded its completion-token budget")
        self._ordinal += 1
        return parsed


def _sampling_identity(
    task: EvaluationTaskAttempt,
    address_bytes: bytes,
) -> tuple[int, str]:
    key = f"stage-d-heldout:{task.unit.seed}".encode()
    digest = hmac.new(key, address_bytes, hashlib.sha256).digest()
    seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    salt = "stage-d-heldout-" + hashlib.sha256(digest + b"cache").hexdigest()
    return seed, salt


__all__ = ["EvaluationCallSpec", "EvaluationModelPort", "RequestSerializer"]
