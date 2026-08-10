"""Trusted held-out driver joining isolated task logic to the evaluation ledger."""

from __future__ import annotations

import hashlib
import inspect
import math
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
from typing import Any, Protocol

from redco.analysis.stage_d_evaluation_codec import canonical_object
from redco.analysis.stage_d_evaluation_contracts import EvaluationRuntimeEntrypoint
from redco.analysis.stage_d_evaluation_ledger import StageDEvaluationLedger
from redco.analysis.stage_d_evaluation_model_port import (
    EvaluationModelPort,
    RequestSerializer,
)
from redco.analysis.stage_d_evaluation_worker import DockerEvaluationRuntime
from redco.analysis.stage_d_objective_binding import ArmName

_ENTRYPOINT_SCHEMAS = {
    "task_runner": "redco-stage-d-worker-ipc-v1",
    "scorer": "redco-stage-d-scorer-v1",
    "request_serializer": "redco-stage-d-request-serializer-v1",
}
_SERIALIZER = "evaluation request serializer"
_SCORER = "evaluation scorer"
_SERIALIZER_SIGNATURE = (("payload",), frozenset(("seed", "cache_salt")))
_SCORER_SIGNATURE = (
    (),
    frozenset(
        ("task_attempt_id", "task_id", "seed", "terminal_output_bytes", "task_evidence_bytes")
    ),
)


class _TaskScorer(Protocol):
    def __call__(
        self,
        *,
        task_attempt_id: str,
        task_id: str,
        seed: int,
        terminal_output_bytes: bytes,
        task_evidence_bytes: bytes,
    ) -> bytes: ...


@dataclass(frozen=True, slots=True)
class _RequestSerializerAdapter:
    entrypoint: Callable[..., object]

    def __post_init__(self) -> None:
        _require_signature(self.entrypoint, _SERIALIZER, _SERIALIZER_SIGNATURE)

    def __call__(
        self,
        payload: dict[str, Any],
        *,
        seed: int,
        cache_salt: str,
    ) -> bytes:
        return _require_immutable_bytes(
            self.entrypoint(payload, seed=seed, cache_salt=cache_salt),
            _SERIALIZER,
        )


@dataclass(frozen=True, slots=True)
class _TaskScorerAdapter:
    entrypoint: Callable[..., object]

    def __post_init__(self) -> None:
        _require_signature(self.entrypoint, _SCORER, _SCORER_SIGNATURE)

    def __call__(
        self,
        *,
        task_attempt_id: str,
        task_id: str,
        seed: int,
        terminal_output_bytes: bytes,
        task_evidence_bytes: bytes,
    ) -> bytes:
        return _require_immutable_bytes(
            self.entrypoint(
                task_attempt_id=task_attempt_id,
                task_id=task_id,
                seed=seed,
                terminal_output_bytes=terminal_output_bytes,
                task_evidence_bytes=task_evidence_bytes,
            ),
            _SCORER,
        )


def _require_immutable_bytes(value: object, name: str) -> bytes:
    if type(value) is not bytes:
        raise TypeError(f"{name} must return immutable bytes")
    return value


def _require_signature(
    entrypoint: Callable[..., object],
    name: str,
    expected: tuple[tuple[str, ...], frozenset[str]],
) -> None:
    if type(entrypoint) is not FunctionType:
        raise TypeError(f"{name} signature differs from the frozen API")
    code = entrypoint.__code__
    if code.co_flags & (inspect.CO_COROUTINE | inspect.CO_GENERATOR | inspect.CO_ASYNC_GENERATOR):
        raise TypeError(f"{name} must be synchronous")
    positional_count = code.co_argcount
    keyword_count = code.co_kwonlyargcount
    observed = (
        code.co_varnames[:positional_count],
        frozenset(code.co_varnames[positional_count : positional_count + keyword_count]),
    )
    if (
        observed != expected
        or code.co_posonlyargcount
        or code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS)
        or entrypoint.__defaults__ is not None
        or entrypoint.__kwdefaults__ is not None
    ):
        raise TypeError(f"{name} signature differs from the frozen API")


@dataclass(frozen=True, slots=True)
class EvaluationDriverLimits:
    call_timeout_seconds: float
    task_timeout_seconds: float
    max_calls_per_task: int
    max_completion_tokens_per_task: int

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.call_timeout_seconds)
            or not math.isfinite(self.task_timeout_seconds)
            or self.call_timeout_seconds <= 0
            or self.task_timeout_seconds <= 0
            or type(self.max_calls_per_task) is not int
            or type(self.max_completion_tokens_per_task) is not int
            or self.max_calls_per_task < 1
            or self.max_completion_tokens_per_task < 1
        ):
            raise ValueError("evaluation driver limits are invalid")


def run_evaluation_arm(
    *,
    ledger: StageDEvaluationLedger,
    arm: ArmName,
    docker_executable: Path,
    docker_executable_sha256: str,
    limits: EvaluationDriverLimits,
) -> bytes:
    session = ledger.resume_current_client_session(arm)
    runtime_path = Path(ledger.manifest.runtime_bundle_path)
    serializer: RequestSerializer = _RequestSerializerAdapter(
        _load_callable(runtime_path, _entrypoint(ledger, "request_serializer"))
    )
    scorer: _TaskScorer = _TaskScorerAdapter(
        _load_callable(runtime_path, _entrypoint(ledger, "scorer"))
    )
    runtime = DockerEvaluationRuntime(
        manifest=ledger.manifest,
        docker_executable=docker_executable,
        docker_executable_sha256=docker_executable_sha256,
        task_timeout_seconds=limits.task_timeout_seconds,
    )
    while True:
        snapshot = ledger.inspect()
        current = snapshot.current_task
        if current is None:
            if len(snapshot.tasks) >= len(ledger.manifest.schedule):
                break
            if ledger.manifest.schedule[len(snapshot.tasks)].arm != arm:
                break
            task = ledger.reserve_next_task(session=session)
        else:
            if current.unit.arm != arm:
                raise RuntimeError("evaluation client encountered another arm's open task")
            task = ledger.resume_open_task(session=session)
        started = time.monotonic()
        port = EvaluationModelPort(
            ledger=ledger,
            task=task,
            session=session,
            serialize_request=serializer,
            timeout_seconds=limits.call_timeout_seconds,
            max_calls=limits.max_calls_per_task,
            max_completion_tokens=limits.max_completion_tokens_per_task,
        )
        output = runtime.run_task(
            task_id=task.unit.task_id,
            seed=task.unit.seed,
            task_attempt_id=task.task_attempt_id,
            model_port=port,
        )
        score_bytes = scorer(
            task_attempt_id=task.task_attempt_id,
            task_id=task.unit.task_id,
            seed=task.unit.seed,
            terminal_output_bytes=output.terminal_output_bytes,
            task_evidence_bytes=output.task_evidence_bytes,
        )
        reward = _verify_score(
            score_bytes,
            task_attempt_id=task.task_attempt_id,
            task_id=task.unit.task_id,
            seed=task.unit.seed,
            terminal_output_bytes=output.terminal_output_bytes,
            task_evidence_bytes=output.task_evidence_bytes,
        )
        elapsed = time.monotonic() - started
        call_wall = _current_task_call_wall_seconds(ledger)
        ledger.complete_task(
            task,
            session=session,
            terminal_result_bytes=output.terminal_output_bytes,
            scorer_evidence_bytes=score_bytes,
            reward=reward,
            overhead_wall_seconds=max(0.0, elapsed - call_wall),
            overhead_gpu_seconds=0.0,
        )
    final_snapshot = ledger.inspect()
    remaining = [
        unit for unit in ledger.manifest.schedule[len(final_snapshot.tasks) :] if unit.arm == arm
    ]
    if remaining:
        raise RuntimeError("evaluation arm schedule is not a contiguous block")
    return ledger.complete_arm(arm)


def _entrypoint(
    ledger: StageDEvaluationLedger,
    role: str,
) -> EvaluationRuntimeEntrypoint:
    matches = [item for item in ledger.manifest.runtime_entrypoints if item.role == role]
    if len(matches) != 1 or matches[0].api_schema != _ENTRYPOINT_SCHEMAS[role]:
        raise ValueError("evaluation runtime entrypoint schema differs")
    return matches[0]


def _load_callable(
    runtime_path: Path,
    entrypoint: EvaluationRuntimeEntrypoint,
) -> Callable[..., object]:
    with zipfile.ZipFile(runtime_path, "r") as archive:
        source = archive.read(entrypoint.member_path)
    if hashlib.sha256(source).hexdigest() != entrypoint.source_sha256:
        raise ValueError("evaluation runtime entrypoint changed before load")
    filename = f"{runtime_path}!/{entrypoint.member_path}"
    namespace: dict[str, Any] = {
        "__file__": filename,
        "__name__": f"_redco_frozen_{entrypoint.role}",
    }
    exec(compile(source, filename, "exec"), namespace)
    result: object = namespace.get(entrypoint.callable_name)
    if not callable(result):
        raise ValueError("evaluation runtime entrypoint callable is absent")
    return result


def _verify_score(
    value: bytes,
    *,
    task_attempt_id: str,
    task_id: str,
    seed: int,
    terminal_output_bytes: bytes,
    task_evidence_bytes: bytes,
) -> float:
    payload = canonical_object(value, "evaluation scorer evidence")
    expected = {
        "schema_version": 1,
        "domain": "redco-stage-d-heldout-score-v1",
        "task_attempt_id": task_attempt_id,
        "task_id": task_id,
        "seed": seed,
        "terminal_output_sha256": hashlib.sha256(terminal_output_bytes).hexdigest(),
        "task_evidence_sha256": hashlib.sha256(task_evidence_bytes).hexdigest(),
    }
    if set(payload) != {*expected, "reward", "details"} or any(
        payload[name] != expected_value for name, expected_value in expected.items()
    ):
        raise ValueError("evaluation scorer evidence binding differs")
    reward = payload["reward"]
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise ValueError("evaluation scorer reward is invalid")
    numeric = float(reward)
    if not math.isfinite(numeric):
        raise ValueError("evaluation scorer reward is non-finite")
    return numeric


def _current_task_call_wall_seconds(ledger: StageDEvaluationLedger) -> float:
    current = ledger.inspect().current_task
    if current is None:
        raise RuntimeError("evaluation task vanished before accounting")
    total = 0.0
    for call in current.calls:
        if call.outcome_sha256 is None:
            raise RuntimeError("evaluation task accounting found an unfinished call")
        outcome = canonical_object(
            ledger.evidence.get(call.outcome_sha256),
            "evaluation call outcome",
        )
        total += outcome["wall_seconds"]
    return total


__all__ = ["EvaluationDriverLimits", "run_evaluation_arm"]
