"""Exactly-once local HTTP dispatch for the Stage-D held-out evaluator."""

from __future__ import annotations

import http.client
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from redco.analysis.stage_d_evaluation_capabilities import (
    EvaluationCallAuthorization,
    EvaluationClientSession,
    EvaluationDispatchAuthorization,
    EvaluationTaskAttempt,
)
from redco.analysis.stage_d_evaluation_ledger import (
    StageDEvaluationLedger,
)
from redco.analysis.stage_d_evaluation_transport import (
    build_local_post_transport,
    canonical_response_headers,
    verify_transport_request,
)

DispatchFaultHook = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class EvaluationHttpResult:
    call: EvaluationCallAuthorization
    dispatch: EvaluationDispatchAuthorization
    response_envelope: bytes
    status_code: int
    response_headers: tuple[tuple[str, str], ...]
    raw_response: bytes


def dispatch_local_http_once(
    *,
    ledger: StageDEvaluationLedger,
    task: EvaluationTaskAttempt,
    session: EvaluationClientSession,
    event_address_bytes: bytes,
    seed: int,
    cache_salt: str,
    request_body_bytes: bytes,
    timeout_seconds: float,
    fault_hook: DispatchFaultHook | None = None,
) -> EvaluationHttpResult:
    """Make one non-retrying POST whose intent is durable before network I/O."""
    endpoint = ledger.manifest.program(task.unit.arm, "server").endpoint
    parsed = urlsplit(endpoint)
    if parsed.hostname != "127.0.0.1" or parsed.port is None:
        raise ValueError("evaluation HTTP endpoint is not bound to loopback")
    transport, _ = build_local_post_transport(
        endpoint=endpoint,
        request_body=request_body_bytes,
        timeout_seconds=timeout_seconds,
    )
    call = ledger.reserve_call(
        task,
        session=session,
        event_address_bytes=event_address_bytes,
        seed=seed,
        cache_salt=cache_salt,
        request_body_bytes=request_body_bytes,
        transport_bytes=transport,
    )
    _fault(fault_hook, "after-call-reserved")
    return dispatch_reserved_local_http_once(
        ledger=ledger,
        call=call,
        session=session,
        timeout_seconds=timeout_seconds,
        fault_hook=fault_hook,
    )


def dispatch_reserved_local_http_once(
    *,
    ledger: StageDEvaluationLedger,
    call: EvaluationCallAuthorization,
    session: EvaluationClientSession,
    timeout_seconds: float,
    fault_hook: DispatchFaultHook | None = None,
) -> EvaluationHttpResult:
    """Dispatch one already-reserved call without creating a second reservation."""
    endpoint = ledger.manifest.program(session.arm, "server").endpoint
    parsed = urlsplit(endpoint)
    if parsed.hostname != "127.0.0.1" or parsed.port is None:
        raise ValueError("evaluation HTTP endpoint is not bound to loopback")
    request_body_bytes = ledger.evidence.get(call.request_sha256)
    transport_bytes = ledger.evidence.get(call.transport_sha256)
    _, request_headers = build_local_post_transport(
        endpoint=endpoint,
        request_body=request_body_bytes,
        timeout_seconds=timeout_seconds,
    )
    verify_transport_request(
        transport_bytes,
        expected_endpoint=endpoint,
        expected_body_sha256=call.request_sha256,
    )
    dispatch = ledger.authorize_dispatch(call, session=session)
    _fault(fault_hook, "after-dispatch-durable")
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_seconds)
    started = time.monotonic()
    try:
        connection.putrequest(
            "POST",
            "/v1/chat/completions",
            skip_host=True,
            skip_accept_encoding=True,
        )
        for name, value in request_headers:
            connection.putheader(name, value)
        connection.endheaders(request_body_bytes)
        response = connection.getresponse()
        raw_response = response.read()
        response_headers = canonical_response_headers(response.getheaders())
        status_code = response.status
    finally:
        connection.close()
    wall_seconds = time.monotonic() - started
    _fault(fault_hook, "after-response-read")
    envelope = ledger.record_response(
        dispatch,
        session=session,
        status_code=status_code,
        headers=response_headers,
        raw_response_bytes=raw_response,
        wall_seconds=wall_seconds,
    )
    _fault(fault_hook, "after-response-witnessed")
    return EvaluationHttpResult(
        call,
        dispatch,
        envelope,
        status_code,
        response_headers,
        raw_response,
    )


def _fault(hook: DispatchFaultHook | None, stage: str) -> None:
    if hook is not None:
        hook(stage)


__all__ = [
    "EvaluationHttpResult",
    "dispatch_local_http_once",
    "dispatch_reserved_local_http_once",
]
