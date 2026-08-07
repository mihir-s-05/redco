"""Small, deterministic CPU contracts for Stage-D action closure.

The historical corpus is intentionally evidence-addressed: this module never
creates a provider response or infers one from a terminal report.  It supplies
the pure case/mutation contract and the single watchdog seam used by live
source collection.  Historical byte auditing remains in the existing audit
script, which owns ledger parsing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeVar, cast

from redco.analysis.stage_d_exact_action import BehaviorAction
from redco.contracts import canonical_json
from redco.integrations.write_once import write_once

T = TypeVar("T")

ACCEPTED_TERMINATION_KINDS: Final = frozenset(
    {
        "eos",
        "max_tokens",
        "tool_calls",
        "empty_content",
        "textual_refusal",
        "multi_turn_child",
        "concurrent_child_completion",
    }
)
ABORT_DISPOSITIONS: Final = frozenset(
    {
        "cancellation",
        "transport_error",
        "malformed_provider_refusal",
        "schema_failure",
        "zero_token_response",
    }
)
WATCHDOG_PHASES: Final = (
    "provider_call",
    "episode",
    "concurrent_children",
    "finalizer",
    "campaign",
    "pod_lifetime",
)
RUNTIME_TERM_TIMEOUT_SECONDS: Final = 2.0


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_terminal_record_writer(path: Path) -> Callable[[bytes], None]:
    """Return the one-shot writer owned by the collection launcher."""

    resolved = path.resolve(strict=False)

    def write_terminal_record(value: bytes) -> None:
        write_once(
            resolved,
            value,
            allow_existing_same=False,
            error_type=RuntimeError,
        )

    return write_terminal_record


@dataclass(frozen=True, slots=True)
class ActionClosureCase:
    """One bounded closure vector; payloads contain no generated model text."""

    case_id: str
    disposition: str
    termination_kind: str | None
    mutation_field: str | None = None

    @property
    def owner(self) -> str:
        """Name the production owner that must accept or reject this vector."""

        if self.disposition == "accept":
            return "train_client_observer_action_source_finalizer_ledger"
        if self.disposition == "abort":
            return "observer_abort_policy_call_ledger"
        return "exact_action_or_source_finalizer_ledger"

    @property
    def failure_origin(self) -> str | None:
        """Stable failure owner for negative vectors; never a generic ValueError."""

        if self.disposition == "accept":
            return None
        if self.disposition == "abort":
            return "stage_d_source_producer.abort_policy_call"
        return {
            "finish_reason": "stage_d_exact_action.BehaviorAction.build",
            "usage": "stage_d_exact_action.BehaviorAction.build",
            "tool_argument_bytes": "stage_d_source_producer._verify_trace_call",
            "address": "stage_d_source_producer.StageDSourceRolloutProducer.finalize_episode",
        }[self.mutation_field or ""]

    def to_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "disposition": self.disposition,
            "failure_origin": self.failure_origin,
            "owner": self.owner,
            "mutation_field": self.mutation_field,
            "termination_kind": self.termination_kind,
        }


def build_action_closure_cases() -> tuple[ActionClosureCase, ...]:
    """Return the fixed, bounded closure matrix in canonical order."""

    accepted = (
        ("eos", "eos"),
        ("exact-cap", "max_tokens"),
        ("tool-calls", "tool_calls"),
        ("empty-string-content", "empty_content"),
        ("textual-refusal", "textual_refusal"),
        ("multi-turn-child", "multi_turn_child"),
        ("concurrent-child-order-a", "concurrent_child_completion"),
        ("concurrent-child-order-b", "concurrent_child_completion"),
    )
    rejected = (
        "cancellation",
        "transport-error",
        "malformed-provider-refusal",
        "schema-failure",
        "zero-token-response",
    )
    mutations = ("finish_reason", "usage", "tool_argument_bytes", "address")
    return tuple(
        [
            ActionClosureCase(case_id=name, disposition="accept", termination_kind=kind)
            for name, kind in accepted
        ]
        + [
            ActionClosureCase(case_id=name, disposition="abort", termination_kind=None)
            for name in rejected
        ]
        + [
            ActionClosureCase(
                case_id=f"one-field-mutation-{field}",
                disposition="reject",
                termination_kind=None,
                mutation_field=field,
            )
            for field in mutations
        ]
    )


def action_closure_case_manifest() -> dict[str, Any]:
    """Build the deterministic case manifest and its independent hash."""

    cases = [case.to_payload() for case in build_action_closure_cases()]
    payload = {
        "schema_version": 1,
        "domain": "redco-stage-d-action-closure-case-manifest-v1",
        "generator": "fixed_bounded_case_table",
        "cases": cases,
        "accepted_termination_kinds": sorted(ACCEPTED_TERMINATION_KINDS),
        "abort_dispositions": sorted(ABORT_DISPOSITIONS),
        "retry": False,
        "seed": 0,
    }
    raw = canonical_json(payload)
    return {**payload, "manifest_sha256": sha256_bytes(raw)}


def audit_raw_response_fixtures(
    repository: Path, fixtures: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Authenticate retained response bytes without printing their contents."""

    audited: list[dict[str, Any]] = []
    for fixture in fixtures:
        relative = fixture.get("path")
        expected_sha = fixture.get("sha256")
        expected_bytes = fixture.get("bytes")
        if (
            not isinstance(relative, str)
            or not isinstance(expected_sha, str)
            or not isinstance(expected_bytes, int)
        ):
            raise ValueError("response fixture binding is malformed")
        path = repository / relative
        raw = path.read_bytes()
        actual_sha = sha256_bytes(raw)
        if len(raw) != expected_bytes or actual_sha != expected_sha:
            raise ValueError(f"response fixture changed: {relative}")
        audited.append(
            {
                "bytes": len(raw),
                "path": relative,
                "sha256": actual_sha,
                "version": fixture.get("version"),
                "record": fixture.get("record"),
                "replay": fixture.get("replay"),
            }
        )
    if len(audited) != 14:
        raise ValueError("the retained raw response corpus must contain exactly 14 fixtures")
    return {
        "fixture_count": len(audited),
        "fixtures": audited,
        "raw_fixture_manifest_sha256": sha256_bytes(canonical_json(audited)),
        "total_bytes": sum(item["bytes"] for item in audited),
    }


def mutate_one_field(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    """Apply a deterministic invalid mutation to a copied action-like mapping."""

    mutated = dict(payload)
    if field == "finish_reason":
        mutated["finish_reason"] = "length"
    elif field == "usage":
        mutated["completion_tokens"] = int(mutated.get("completion_tokens", 0)) + 1
    elif field == "tool_argument_bytes":
        mutated["tool_arguments"] = str(mutated.get("tool_arguments", "")) + "!"
    elif field == "address":
        mutated["address"] = str(mutated.get("address", "")) + ":mutated"
    else:
        raise ValueError(f"unknown action-closure mutation: {field}")
    return mutated


def validate_termination_kind(kind: object) -> str:
    if not isinstance(kind, str) or kind not in ACCEPTED_TERMINATION_KINDS:
        raise ValueError("action termination is outside the frozen closure")
    return kind


def reload_completed_action(
    raw: bytes,
    *,
    render_prompt: Callable[[Mapping[str, Any]], tuple[int, ...]],
) -> BehaviorAction:
    """Reload one retained completed action without semantic text replay."""

    try:
        envelope = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("completed action evidence is not JSON") from error
    if not isinstance(envelope, dict):
        raise ValueError("completed action evidence is not an object")
    action = BehaviorAction.from_bytes(
        raw,
        validate_action=lambda _request, _message, _tokens: None,
        render_prompt=render_prompt,
    )
    validate_termination_kind(action.termination_kind)
    if action.to_bytes() != raw:
        raise ValueError("completed action did not round-trip byte-for-byte")
    return action


@dataclass(frozen=True, slots=True)
class WatchdogDeadlines:
    """Finite ownership deadlines for every Stage-D live phase."""

    provider_call: float = 900.0
    episode: float = 900.0
    concurrent_children: float = 900.0
    finalizer: float = 120.0
    campaign: float = 21_600.0
    pod_lifetime: float = 21_600.0

    def for_phase(self, phase: str) -> float:
        if phase not in WATCHDOG_PHASES:
            raise ValueError(f"unknown Stage-D watchdog phase: {phase}")
        value = getattr(self, phase)
        if type(value) not in (int, float) or value <= 0:
            raise ValueError(f"watchdog deadline for {phase} is not positive")
        return float(value)

    def to_payload(self) -> dict[str, float]:
        return {phase: self.for_phase(phase) for phase in WATCHDOG_PHASES}


@dataclass(slots=True)
class ActionClosureWatchdog:
    """Reusable phase deadline owner with one durable terminal disposition.

    A successful phase is not a campaign terminal event.  Callers may therefore
    supervise provider, gather, episode, and finalizer phases in sequence.  A
    timeout, cancellation, or error is terminal immediately; successful
    campaigns call :meth:`complete` exactly once after the outer campaign phase
    has completed.  ``run`` remains as a compatibility alias for ``run_phase``.
    """

    deadlines: WatchdogDeadlines = WatchdogDeadlines()
    terminal_callback: Callable[[str, str], None] | None = None
    terminal_record_callback: Callable[[bytes], None] | None = None
    runtime_term_timeout: float = RUNTIME_TERM_TIMEOUT_SECONDS
    _closed: bool = False
    _terminal: bool = False
    _terminal_phase: str | None = None
    _terminal_disposition: str | None = None
    _phase_count: int = 0
    _owned_runtime: Any | None = None

    def _record_terminal(self, phase: str, disposition: str) -> None:
        if self._terminal:
            raise RuntimeError("Stage-D watchdog received a second terminal disposition")
        if self._closed and disposition == "completed":
            raise RuntimeError("Stage-D watchdog is already closed")
        # Closing is irreversible and deliberately precedes durable publication.
        # A launcher may still publish one replacement failure disposition after
        # a failed callback, but no phase operation may start again.
        self._closed = True
        payload = canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-action-closure-terminal-v1",
                "phase": phase,
                "disposition": disposition,
                "phase_count": self._phase_count,
            }
        )
        if self.terminal_record_callback is not None:
            self.terminal_record_callback(payload)
        if self.terminal_callback is not None:
            self.terminal_callback(phase, disposition)
        # Publish the durable terminal state only after the callback succeeds.
        # The closed state above remains in force if publication fails.
        self._terminal = True
        self._terminal_phase = phase
        self._terminal_disposition = disposition

    async def _record_failure_and_stop(
        self,
        phase: str,
        disposition: str,
        primary_error: BaseException,
    ) -> None:
        """Always tear down owned runtime while preserving the phase error."""

        terminal_error: BaseException | None = None
        try:
            if not self._terminal:
                self._record_terminal(phase, disposition)
        except BaseException as error:
            terminal_error = error
        try:
            await self._stop_owned_runtime()
        except BaseException as error:
            if terminal_error is None:
                terminal_error = error
        if terminal_error is not None:
            raise primary_error.with_traceback(primary_error.__traceback__) from terminal_error

    async def run_phase(self, operation: Awaitable[T], *, phase: str) -> T:
        """Run one owned phase and await cancellation before returning/raising."""

        if self._closed:
            close = getattr(operation, "close", None)
            if callable(close):
                close()
            raise RuntimeError(
                "Stage-D watchdog forbids work after closure/terminal disposition"
            )
        timeout = self.deadlines.for_phase(phase)
        self._phase_count += 1
        try:
            result = await asyncio.wait_for(operation, timeout=timeout)
        except TimeoutError as error:
            await self._record_failure_and_stop(phase, "timeout", error)
            raise
        except asyncio.CancelledError as error:
            await self._record_failure_and_stop(phase, "cancellation", error)
            raise
        except BaseException as error:
            await self._record_failure_and_stop(phase, "error", error)
            raise
        return result

    async def run(self, operation: Awaitable[T], *, phase: str) -> T:
        """Compatibility alias for the reusable phase runner."""

        return await self.run_phase(operation, phase=phase)

    async def run_provider_call(self, operation: Awaitable[T]) -> T:
        return await self.run_phase(operation, phase="provider_call")

    async def run_concurrent_children(self, operation: Awaitable[T]) -> T:
        return await self.run_phase(operation, phase="concurrent_children")

    async def run_episode(self, operation: Awaitable[T]) -> T:
        return await self.run_phase(operation, phase="episode")

    async def run_finalizer(self, operation: Awaitable[T]) -> T:
        return await self.run_phase(operation, phase="finalizer")

    async def run_campaign(self, operation: Awaitable[T]) -> T:
        return await self.run_phase(operation, phase="campaign")

    async def run_pod_lifetime(self, operation: Awaitable[T]) -> T:
        return await self.run_phase(operation, phase="pod_lifetime")

    def bind_runtime(self, runtime: Any) -> None:
        """Bind the currently executing Verifiers runtime to this watchdog."""

        if self._closed:
            raise RuntimeError("Stage-D watchdog cannot bind a runtime after termination")
        if self._owned_runtime is not None and self._owned_runtime is not runtime:
            raise RuntimeError("Stage-D watchdog runtime ownership changed mid-episode")
        self._owned_runtime = runtime

    def release_runtime(self, runtime: Any) -> None:
        if self._owned_runtime is runtime:
            self._owned_runtime = None

    async def _stop_owned_runtime(self) -> None:
        runtime = self._owned_runtime
        if runtime is None:
            return
        self._owned_runtime = None
        await stop_owned_runtime(runtime, term_timeout=self.runtime_term_timeout)

    def fail(self, phase: str = "campaign", disposition: str = "error") -> bytes:
        """Record a launcher-owned failure after publication or teardown fails."""

        if disposition == "completed":
            raise ValueError("launcher failure disposition cannot be completed")
        self._record_terminal(phase, disposition)
        return cast(
            bytes,
            canonical_json(
                {
                    "schema_version": 1,
                    "domain": "redco-stage-d-action-closure-terminal-v1",
                    "phase": phase,
                    "disposition": disposition,
                    "phase_count": self._phase_count,
                }
            ),
        )

    def complete(self) -> bytes:
        """Commit the one successful campaign terminal record."""

        if self._terminal:
            raise RuntimeError("Stage-D watchdog received a second terminal disposition")
        self._record_terminal("campaign", "completed")
        if self._terminal_phase != "campaign" or self._terminal_disposition != "completed":
            raise AssertionError("watchdog completion record changed")
        return cast(
            bytes,
            canonical_json(
                {
                    "schema_version": 1,
                    "domain": "redco-stage-d-action-closure-terminal-v1",
                    "phase": "campaign",
                    "disposition": "completed",
                    "phase_count": self._phase_count,
                }
            ),
        )

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def phase_count(self) -> int:
        return self._phase_count


async def terminate_process_then_kill(
    process: Any,
    *,
    term_timeout: float = 2.0,
) -> str:
    """Bounded launcher ownership: TERM first, then KILL if the process lingers."""

    if type(term_timeout) not in (int, float) or term_timeout <= 0:
        raise ValueError("process TERM timeout must be positive")
    terminate = getattr(process, "terminate", None)
    wait = getattr(process, "wait", None)
    kill = getattr(process, "kill", None)
    if not callable(terminate) or not callable(wait) or not callable(kill):
        raise TypeError("process must expose terminate, wait, and kill")
    terminate()
    try:
        result = wait()
        if hasattr(result, "__await__"):
            await asyncio.wait_for(result, timeout=float(term_timeout))
        return "term"
    except TimeoutError:
        kill()
        result = wait()
        if hasattr(result, "__await__"):
            await result
        return "kill"


async def stop_owned_runtime(runtime: Any, *, term_timeout: float = 2.0) -> None:
    """Stop only subprocesses owned by the supplied launcher runtime.

    The runtime's private background collection is the ownership boundary; no
    process lookup or broad OS kill is performed here.
    """

    owned = getattr(runtime, "_background", ())
    if owned is None:
        owned = ()
    if not isinstance(owned, (tuple, list)):
        raise TypeError("launcher runtime background ownership is not inspectable")
    for process in tuple(owned):
        if getattr(process, "returncode", None) is None:
            await terminate_process_then_kill(process, term_timeout=term_timeout)
    stop = getattr(runtime, "stop", None)
    if not callable(stop):
        raise TypeError("launcher runtime lacks its owned stop operation")
    result = stop()
    if hasattr(result, "__await__"):
        await result


__all__ = [
    "ABORT_DISPOSITIONS",
    "ACCEPTED_TERMINATION_KINDS",
    "RUNTIME_TERM_TIMEOUT_SECONDS",
    "WATCHDOG_PHASES",
    "ActionClosureCase",
    "ActionClosureWatchdog",
    "WatchdogDeadlines",
    "action_closure_case_manifest",
    "atomic_terminal_record_writer",
    "audit_raw_response_fixtures",
    "build_action_closure_cases",
    "mutate_one_field",
    "reload_completed_action",
    "sha256_bytes",
    "stop_owned_runtime",
    "terminate_process_then_kill",
    "validate_termination_kind",
]
