"""CPU-only toy subprocess boundary for Stage D scientific replay.

This module validates lifecycle and trust-boundary behavior on Windows, where
workers are contained by a kill-on-close Job Object. It does not provide an OS
security sandbox and must not be used as evidence of live RLM readiness.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.analysis.stage_d_receipt_ledger import (
    ExecutionAttempt,
    LedgerError,
    StageDReceiptLedger,
)
from redco.analysis.stage_d_scientific_branch_group import (
    BranchGroupSpec,
    BranchSeedOracle,
    CandidateSubmission,
    OutcomeKind,
)
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress, ScheduledSeed
from redco.contracts import canonical_json

_DOMAIN = "redco-stage-d-toy-subprocess-executor-v1"
_MAX_CONTROL_BYTES = 1 << 20
_MAX_CAS_OBJECT_BYTES = 64 << 20
_MAX_MANIFEST_ENTRIES = 1 << 16
_CLEANUP_GRACE_SECONDS = 5.0


class _ResourceLimit(RuntimeError):
    pass


class _UnquiescedCallback(RuntimeError):
    pass


class InferenceGateway(Protocol):
    """Supervisor-owned policy gateway; workers never receive this object."""

    @property
    def runtime_sha256(self) -> str: ...

    @property
    def config_sha256(self) -> str: ...

    def sample_action(
        self,
        *,
        reference_key: ExactActionKey,
        action_seed: int,
        exact_request: Mapping[str, Any],
        dispatch_id: str,
    ) -> GatewayActionResponse: ...

    def continue_policy(
        self,
        semantic_request: Mapping[str, Any],
        *,
        scheduled_seed: ScheduledSeed,
        dispatch_id: str,
    ) -> GatewayContinuationResponse: ...


class ProvenanceResolver(Protocol):
    """Trusted adapter that derives a B1 event address from worker provenance."""

    def __call__(self, provenance: Mapping[str, Any]) -> PolicyEventAddress: ...


class DeterministicScorer(Protocol):
    """Supervisor-owned deterministic reward function."""

    @property
    def artifact_sha256(self) -> str: ...

    def __call__(
        self,
        *,
        action: BehaviorAction,
        continuations: Sequence[GatewayContinuationResponse],
        worker_result: Mapping[str, Any],
    ) -> ScoredResult: ...


@dataclass(frozen=True, slots=True)
class GatewayActionResponse:
    action: BehaviorAction
    raw_response: bytes

    def __post_init__(self) -> None:
        if type(self.raw_response) is not bytes or not self.raw_response:
            raise ValueError("candidate raw response must be nonempty bytes")


@dataclass(frozen=True, slots=True)
class GatewayContinuationResponse:
    request_sha256: str
    scheduled_seed: ScheduledSeed
    dispatch_id: str
    prompt_tokens: int
    completion_tokens: int
    raw_response: bytes

    def __post_init__(self) -> None:
        _require_sha256(self.request_sha256, "continuation request sha256")
        if type(self.scheduled_seed) is not ScheduledSeed:
            raise ValueError("continuation response requires the scheduled seed")
        if type(self.dispatch_id) is not str or not self.dispatch_id:
            raise ValueError("continuation response requires a dispatch id")
        _exact_int(self.prompt_tokens, "prompt_tokens")
        _exact_int(self.completion_tokens, "completion_tokens")
        if type(self.raw_response) is not bytes or not self.raw_response:
            raise ValueError("continuation raw response must be nonempty bytes")


@dataclass(frozen=True, slots=True)
class ScoredResult:
    reward: float
    evidence: bytes

    def __post_init__(self) -> None:
        _finite_float(self.reward, "reward")
        if type(self.evidence) is not bytes or not self.evidence:
            raise ValueError("scorer evidence must be nonempty bytes")


@dataclass(frozen=True, slots=True)
class ReplayResourceLimits:
    wall_seconds: float
    policy_calls: int
    prompt_tokens: int
    completion_tokens: int
    worker_output_bytes: int
    workspace_snapshot_bytes: int
    workspace_entries: int

    def __post_init__(self) -> None:
        if _finite_float(self.wall_seconds, "wall_seconds") <= 0:
            raise ValueError("wall_seconds must be positive")
        for name in (
            "policy_calls",
            "prompt_tokens",
            "completion_tokens",
            "worker_output_bytes",
            "workspace_snapshot_bytes",
            "workspace_entries",
        ):
            _exact_int(getattr(self, name), name, minimum=1)

    def to_payload(self) -> dict[str, int | float]:
        return {
            "wall_seconds": self.wall_seconds,
            "policy_calls": self.policy_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "worker_output_bytes": self.worker_output_bytes,
            "workspace_snapshot_bytes": self.workspace_snapshot_bytes,
            "workspace_entries": self.workspace_entries,
        }


@dataclass(frozen=True, slots=True)
class WorkspaceFile:
    path: str
    sha256: str
    mode: int

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)
        _require_sha256(self.sha256, "workspace file sha256")
        if type(self.mode) is not int or self.mode not in {0o600, 0o644, 0o700, 0o755}:
            raise ValueError("workspace mode is not allowlisted")

    def to_payload(self) -> dict[str, str | int]:
        return {"path": self.path, "sha256": self.sha256, "mode": self.mode}


@dataclass(frozen=True, slots=True)
class WorkspaceManifest:
    files: tuple[WorkspaceFile, ...]
    digest: str

    @classmethod
    def build(cls, files: Sequence[WorkspaceFile]) -> WorkspaceManifest:
        ordered = tuple(sorted(files, key=lambda item: item.path))
        if not ordered or len({item.path for item in ordered}) != len(ordered):
            raise ValueError("workspace manifest needs unique nonempty file paths")
        payload = {
            "schema_version": 1,
            "domain": "redco-stage-d-workspace-manifest-v1",
            "files": [item.to_payload() for item in ordered],
        }
        return cls(ordered, _sha256(canonical_json(payload)))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "domain": "redco-stage-d-workspace-manifest-v1",
            "files": [item.to_payload() for item in self.files],
            "manifest_sha256": self.digest,
        }


class ContentAddressedStore:
    """Small immutable CAS used by the toy executor."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        if not _is_real_directory(root):
            raise ValueError("CAS root must be a real directory")
        self.root = root

    def put(self, value: bytes) -> str:
        if type(value) is not bytes:
            raise ValueError("CAS values must be immutable bytes")
        digest = _sha256(value)
        path = self.root / digest
        if path.exists():
            if self.read_verified(digest) != value:
                raise ValueError("CAS object is not an exact immutable file")
            return digest
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return digest

    def read_verified(
        self,
        digest: str,
        *,
        max_bytes: int = _MAX_CAS_OBJECT_BYTES,
    ) -> bytes:
        _require_sha256(digest, "CAS digest")
        path = self.root / digest
        try:
            value = _read_regular_limited(path, max_bytes)
        except _ResourceLimit as error:
            raise ValueError("CAS object exceeds the immutable object limit") from error
        if _sha256(value) != digest:
            raise ValueError("CAS object digest mismatch")
        return value

    def materialize(
        self,
        manifest: WorkspaceManifest,
        destination: Path,
        *,
        max_bytes: int,
        max_entries: int,
        deadline: float,
    ) -> int:
        destination.mkdir(parents=True, exist_ok=False)
        if len(manifest.files) > max_entries:
            raise _ResourceLimit("workspace manifest exceeded frozen entry limit")
        total = 0
        for item in manifest.files:
            _check_deadline(deadline)
            remaining = max_bytes - total
            if remaining < 0:
                raise _ResourceLimit("workspace exceeded frozen byte limit")
            value = self.read_verified(item.sha256, max_bytes=remaining)
            target = destination.joinpath(*PurePosixPath(item.path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise ValueError("workspace target already exists")
            target.write_bytes(value)
            target.chmod(item.mode)
            total += len(value)
        verify_workspace(
            destination,
            manifest,
            max_bytes=max_bytes,
            max_entries=max_entries,
            deadline=deadline,
        )
        return total


def workspace_manifest(root: Path, cas: ContentAddressedStore) -> WorkspaceManifest:
    """Capture regular files only; symlinks and special files fail closed."""
    if not _is_real_directory(root):
        raise ValueError("workspace root must be a real directory")
    files: list[WorkspaceFile] = []
    for path in _bounded_workspace_files(
        root,
        max_entries=_MAX_MANIFEST_ENTRIES,
        deadline=None,
    ):
        relative = path.relative_to(root).as_posix()
        mode = path.stat().st_mode & 0o777
        normalized_mode = 0o755 if mode & 0o111 else 0o644
        files.append(WorkspaceFile(relative, cas.put(path.read_bytes()), normalized_mode))
    return WorkspaceManifest.build(files)


def toy_worker_runtime_manifest(
    executable: Path,
    worker: Path,
    *,
    cas: ContentAddressedStore,
) -> bytes:
    """Freeze the exact executable and toy worker bytes used by this boundary."""
    artifacts = []
    for role, path in (("executable", executable), ("worker", worker)):
        if not path.is_absolute():
            raise ValueError("toy runtime paths must be absolute")
        value = _read_regular_limited(path, _MAX_CAS_OBJECT_BYTES)
        artifacts.append({"role": role, "path": str(path), "sha256": cas.put(value)})
    return canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-toy-worker-runtime-v1",
            "artifacts": artifacts,
        }
    )


def verify_workspace(
    root: Path,
    expected: WorkspaceManifest,
    *,
    max_bytes: int = _MAX_CAS_OBJECT_BYTES,
    max_entries: int = 4096,
    deadline: float | None = None,
) -> int:
    """Verify a materialized workspace without admitting new bytes into the CAS."""
    if not _is_real_directory(root):
        raise ValueError("workspace root must be a real directory")
    paths = _bounded_workspace_files(
        root,
        max_entries=max_entries,
        deadline=deadline,
    )
    files: list[WorkspaceFile] = []
    total = 0
    for path in paths:
        if deadline is not None:
            _check_deadline(deadline)
        remaining = max_bytes - total
        if remaining < 0:
            raise _ResourceLimit("workspace exceeded frozen byte limit")
        value = _read_regular_limited(path, remaining)
        total += len(value)
        relative = path.relative_to(root).as_posix()
        mode = path.stat().st_mode & 0o777
        normalized_mode = 0o755 if mode & 0o111 else 0o644
        files.append(WorkspaceFile(relative, _sha256(value), normalized_mode))
    if WorkspaceManifest.build(files).digest != expected.digest:
        raise ValueError("materialized workspace differs from its frozen manifest")
    return total


def _bounded_workspace_files(
    root: Path,
    *,
    max_entries: int,
    deadline: float | None,
) -> list[Path]:
    """Walk an untrusted workspace with a hard entry bound before sorting."""
    _exact_int(max_entries, "max_entries", minimum=1)
    files: list[Path] = []
    pending = [root]
    entries = 0
    while pending:
        if deadline is not None:
            _check_deadline(deadline)
        directory = pending.pop()
        with os.scandir(directory) as iterator:
            for entry in iterator:
                entries += 1
                if entries > max_entries:
                    raise _ResourceLimit("workspace exceeded frozen entry limit")
                if deadline is not None:
                    _check_deadline(deadline)
                metadata = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or _stat_is_reparse(metadata):
                    raise ValueError("workspace symlinks and reparse points are forbidden")
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files.append(path)
                else:
                    raise ValueError("workspace special files are forbidden")
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    return files


@dataclass(frozen=True, slots=True)
class ToyExecutionContext:
    spec: BranchGroupSpec
    runtime_sha256: str
    config_sha256: str
    scorer_sha256: str
    workspace: WorkspaceManifest
    limits: ReplayResourceLimits

    def __post_init__(self) -> None:
        if type(self.spec) is not BranchGroupSpec:
            raise ValueError("execution context requires a validated BranchGroupSpec")
        if type(self.workspace) is not WorkspaceManifest:
            raise ValueError("execution context requires a validated workspace manifest")
        if type(self.limits) is not ReplayResourceLimits:
            raise ValueError("execution context requires validated resource limits")
        for name in (
            "runtime_sha256",
            "config_sha256",
            "scorer_sha256",
        ):
            _require_sha256(getattr(self, name), name)


class DurableCandidateSampler:
    """Exactly-once candidate gateway backed by D1a write-ahead receipts."""

    def __init__(
        self,
        writer: StageDReceiptLedger,
        gateway: InferenceGateway,
        *,
        group_id: str,
        target_id: str,
        wall_seconds: float,
    ) -> None:
        self._writer = writer
        self._gateway = gateway
        self._group_id = group_id
        self._target_id = target_id
        self._wall_seconds = _finite_float(wall_seconds, "wall_seconds")
        if self._wall_seconds <= 0:
            raise ValueError("candidate wall_seconds must be positive")
        binding = writer.genesis_binding
        self._runtime_sha256 = binding.runtime_sha256
        self._config_sha256 = binding.config_sha256
        self._verify_gateway_binding()

    def _verify_gateway_binding(self) -> None:
        if (
            self._gateway.runtime_sha256 != self._runtime_sha256
            or self._gateway.config_sha256 != self._config_sha256
        ):
            raise ValueError("candidate gateway differs from the ledger genesis binding")

    def __call__(
        self,
        *,
        action_slot: int,
        action_seed: int,
        reference_key: ExactActionKey,
    ) -> CandidateSubmission:
        self._verify_gateway_binding()
        attempt = self._writer.begin_candidate_attempt(
            group_id=self._group_id,
            target_id=self._target_id,
            action_slot=action_slot,
        )
        request = _request_with_seed(reference_key, action_seed)
        request_sha256 = self._writer.put_evidence(canonical_json(request))
        dispatch_id = self._writer.mark_candidate_model_call_started(
            attempt,
            request_sha256=request_sha256,
        )
        deadline = time.monotonic() + self._wall_seconds
        try:
            response = _call_with_deadline(
                lambda: self._sample_candidate(
                    reference_key,
                    action_seed,
                    request,
                    dispatch_id,
                ),
                deadline,
            )
        except _UnquiescedCallback:
            self._writer.close()
            raise
        if response.action.key.request != canonical_json(request):
            raise ValueError("candidate gateway did not execute the ledgered exact request")
        response_sha256 = self._writer.put_evidence(response.raw_response)
        self._writer.mark_candidate_response_observed(
            attempt,
            response_sha256=response_sha256,
        )
        receipt = self._writer.complete_candidate_call(
            attempt,
            action=response.action,
            response_sha256=response_sha256,
        )
        return CandidateSubmission(response.action, receipt)

    def _sample_candidate(
        self,
        reference_key: ExactActionKey,
        action_seed: int,
        request: Mapping[str, Any],
        dispatch_id: str,
    ) -> GatewayActionResponse:
        self._verify_gateway_binding()
        return self._gateway.sample_action(
            reference_key=reference_key,
            action_seed=action_seed,
            exact_request=request,
            dispatch_id=dispatch_id,
        )


class ToySubprocessArmExecutor:
    """Windows-only toy executor implementing the C1 ArmExecutor protocol."""

    def __init__(
        self,
        *,
        writer: StageDReceiptLedger,
        gateway: InferenceGateway,
        provenance_resolver: ProvenanceResolver,
        scorer: DeterministicScorer,
        cas: ContentAddressedStore,
        context: ToyExecutionContext,
        worker_command: Sequence[str],
        scratch_root: Path,
    ) -> None:
        if os.name != "nt":
            raise RuntimeError("toy executor requires Windows Job Object process-tree containment")
        if len(worker_command) < 2 or any(
            type(item) is not str or not item for item in worker_command
        ):
            raise ValueError("worker command needs nonempty executable and worker argv")
        for item in worker_command[:2]:
            path = Path(item)
            if not path.is_absolute() or path != path.resolve(strict=True):
                raise ValueError("worker executable and entrypoint must be exact absolute paths")
        if Path(worker_command[0]) != Path(getattr(sys, "_base_executable", sys.executable)):
            raise ValueError("Windows toy workers must use the non-launcher base Python")
        if not _is_real_directory(scratch_root):
            raise ValueError("scratch_root must be an existing real directory")
        if context.spec.commitment.ledger_id != writer.ledger_id:
            raise ValueError("execution context is bound to a different durable ledger")
        binding = writer.genesis_binding
        if (
            context.runtime_sha256 != binding.runtime_sha256
            or context.config_sha256 != binding.config_sha256
        ):
            raise ValueError("execution context differs from the ledger genesis binding")
        if (
            gateway.runtime_sha256 != context.runtime_sha256
            or gateway.config_sha256 != context.config_sha256
        ):
            raise ValueError("inference gateway differs from the frozen execution context")
        if scorer.artifact_sha256 != context.scorer_sha256:
            raise ValueError("scorer callable differs from its frozen artifact")
        self._writer = writer
        self._gateway = gateway
        self._provenance_resolver = provenance_resolver
        self._scorer = scorer
        self._cas = cas
        self._context = context
        self._worker_command = tuple(worker_command)
        self._scratch_root = scratch_root
        self._executable_sha256, self._worker_sha256 = self._verify_frozen_artifacts()
        self._verify_callable_bindings()

    def __call__(
        self,
        *,
        arm_id: str,
        action: BehaviorAction,
        continuation_replicate: int,
        seed_oracle: BranchSeedOracle,
    ) -> bytes:
        self._verify_callable_bindings()
        self._verify_frozen_artifacts()
        attempt = self._writer.begin_execution(
            group_id=self._context.spec.commitment.group_id,
            target_id=self._context.spec.commitment.target_id,
            arm_id=arm_id,
            action=action,
            continuation_replicate=continuation_replicate,
        )
        started = time.monotonic()
        deadline = started + self._context.limits.wall_seconds
        storage_bytes = 0
        request = self._request_payload(
            attempt_id=attempt.attempt_id,
            arm_id=arm_id,
            action=action,
            continuation_replicate=continuation_replicate,
        )
        request_bytes = canonical_json(request)
        request_evidence_sha256 = self._writer.put_evidence(request_bytes)
        self._writer.bind_execution_context(
            attempt,
            context_sha256=request_evidence_sha256,
        )
        self._writer.mark_execution_dispatched(attempt)
        if self._writer.branch_target_roster_sha256 is not None:
            injected_response_sha256 = self._writer.put_evidence(action.to_bytes())
            injected_ticket = self._writer.commit_execution_override(
                attempt,
                address=self._context.spec.commitment.target_address,
                action_digest=action.digest,
                disposition="inject",
                request_sha256=self._writer.put_evidence(b"target-injection-request"),
                response_content_sha256=injected_response_sha256,
                prompt_tokens=action.prompt_tokens,
                completion_tokens=action.completion_tokens,
                counts_toward_logical_cost=False,
            )
            self._writer.mark_execution_override_delivered(
                attempt,
                injected_ticket,
                typed_response_sha256=injected_response_sha256,
            )
        if action.parse_status == "malformed":
            return self._finish_failure(
                attempt,
                OutcomeKind.MALFORMED_ACTION,
                started,
                b"malformed action rejected before toy worker launch",
                storage_bytes=storage_bytes,
            )
        with _scratch_directory(self._scratch_root) as temporary_name:
            temporary = Path(temporary_name)
            workspace = temporary / "workspace"

            def finish_with_cleanup(kind: OutcomeKind, evidence: bytes) -> bytes:
                try:
                    _remove_scratch(
                        temporary,
                        max(deadline, time.monotonic() + _CLEANUP_GRACE_SECONDS),
                    )
                except (OSError, _ResourceLimit) as error:
                    kind = OutcomeKind.RUNTIME_EXCEPTION
                    evidence = canonical_json(
                        {
                            "kind": "scratch_cleanup_failure",
                            "error_type": type(error).__name__,
                        }
                    )
                return self._finish_failure(
                    attempt,
                    kind,
                    started,
                    evidence,
                    storage_bytes=storage_bytes,
                )

            try:
                storage_bytes = self._cas.materialize(
                    self._context.workspace,
                    workspace,
                    max_bytes=self._context.limits.workspace_snapshot_bytes,
                    max_entries=self._context.limits.workspace_entries,
                    deadline=deadline,
                )
                request_path = temporary / "request.json"
                output_path = temporary / "output.json"
                runtime_path = temporary / "runtime-worker.py"
                _durable_write(request_path, request_bytes)
                _durable_write(
                    runtime_path,
                    self._cas.read_verified(self._worker_sha256),
                )
                request_digest = _sha256(request_bytes)
                with (
                    _locked_verified_windows_file(
                        Path(self._worker_command[0]),
                        self._executable_sha256,
                    ),
                    _locked_verified_windows_file(runtime_path, self._worker_sha256),
                ):
                    returncode = _run_worker(
                        [
                            self._worker_command[0],
                            str(runtime_path),
                            *self._worker_command[2:],
                            str(request_path),
                            str(output_path),
                        ],
                        cwd=workspace,
                        env=_worker_environment(),
                        timeout=_remaining(deadline),
                    )
                _check_deadline(deadline)
                if returncode != 0:
                    evidence = canonical_json(
                        {
                            "kind": "worker_nonzero_exit",
                            "returncode": returncode,
                        }
                    )
                    return finish_with_cleanup(
                        OutcomeKind.RUNTIME_EXCEPTION,
                        evidence,
                    )
                worker_output = _read_regular_limited(
                    output_path,
                    self._context.limits.worker_output_bytes,
                )
                parsed = _parse_worker_output(worker_output, request_digest=request_digest)
                downstream_events = parsed["downstream_events"]
                if len(downstream_events) > self._context.limits.policy_calls:
                    return finish_with_cleanup(
                        OutcomeKind.RESOURCE_LIMIT,
                        b"worker requested too many downstream policy calls",
                    )
                resolved_values = []
                for event in downstream_events:
                    provenance = event["provenance"]

                    def resolve(
                        provenance: Mapping[str, Any] = provenance,
                    ) -> PolicyEventAddress:
                        return self._provenance_resolver(provenance)

                    resolved_values.append(
                        (
                            _call_with_deadline(resolve, deadline),
                            event["request"],
                        )
                    )
                    _check_deadline(deadline)
                resolved = tuple(resolved_values)
                continuations: list[GatewayContinuationResponse] = []
                prompt_tokens = 0
                completion_tokens = 0
                for address, downstream_request in resolved:
                    scheduled = seed_oracle.seed_for(address)
                    semantic_request = _validated_semantic_request(downstream_request)
                    call_request_sha256 = self._writer.put_evidence(
                        canonical_json(
                            {
                                "semantic_request": semantic_request,
                                "scheduled_seed": {
                                    "seed": scheduled.seed,
                                    "coupling_mode": scheduled.coupling_mode.value,
                                    "address": {
                                        **scheduled.address.as_payload(),
                                        "turn": scheduled.address.turn,
                                    },
                                },
                            }
                        )
                    )
                    call = self._writer.mark_execution_model_call_started(
                        attempt,
                        address=address,
                        scheduled_seed=scheduled,
                        request_sha256=call_request_sha256,
                    )

                    def continue_call(
                        semantic_request: Mapping[str, Any] = semantic_request,
                        scheduled: ScheduledSeed = scheduled,
                        dispatch_id: str = call.call_id,
                    ) -> GatewayContinuationResponse:
                        self._verify_callable_bindings()
                        return self._gateway.continue_policy(
                            semantic_request,
                            scheduled_seed=scheduled,
                            dispatch_id=dispatch_id,
                        )

                    continuation = _call_with_deadline(
                        continue_call,
                        deadline,
                    )
                    if (
                        continuation.request_sha256 != call_request_sha256
                        or continuation.scheduled_seed != scheduled
                        or continuation.dispatch_id != call.call_id
                    ):
                        raise ValueError(
                            "continuation gateway did not execute the ledgered schedule"
                        )
                    response_sha256 = self._writer.put_evidence(continuation.raw_response)
                    self._writer.mark_execution_response_observed(
                        attempt,
                        call,
                        response_sha256=response_sha256,
                    )
                    self._writer.complete_execution_model_call(
                        attempt,
                        call,
                        prompt_tokens=continuation.prompt_tokens,
                        completion_tokens=continuation.completion_tokens,
                        response_sha256=response_sha256,
                    )
                    continuations.append(continuation)
                    _check_deadline(deadline)
                    prompt_tokens += continuation.prompt_tokens
                    completion_tokens += continuation.completion_tokens
                    if (
                        prompt_tokens > self._context.limits.prompt_tokens
                        or completion_tokens > self._context.limits.completion_tokens
                    ):
                        return finish_with_cleanup(
                            OutcomeKind.RESOURCE_LIMIT,
                            b"downstream policy token budget exceeded",
                        )
                terminal = parsed["terminal_without_downstream"]
                if terminal and continuations:
                    return finish_with_cleanup(
                        OutcomeKind.RUNTIME_EXCEPTION,
                        b"worker claimed terminal status after downstream calls",
                    )
                if not terminal and not continuations:
                    return finish_with_cleanup(
                        OutcomeKind.RUNTIME_EXCEPTION,
                        b"worker produced neither terminal outcome nor downstream call",
                    )
                score = _call_with_deadline(
                    lambda: self._score(
                        action,
                        continuations,
                        parsed["worker_result"],
                    ),
                    deadline,
                )
                _check_deadline(deadline)
                score_sha256 = self._writer.put_evidence(score.evidence)
                final_bytes = verify_workspace(
                    workspace,
                    self._context.workspace,
                    max_bytes=self._context.limits.workspace_snapshot_bytes,
                    max_entries=self._context.limits.workspace_entries,
                    deadline=deadline,
                )
                _check_deadline(deadline)
                storage_bytes = max(storage_bytes, final_bytes)
                if storage_bytes > self._context.limits.workspace_snapshot_bytes:
                    return finish_with_cleanup(
                        OutcomeKind.RESOURCE_LIMIT,
                        b"final workspace exceeded frozen byte limit",
                    )
                try:
                    _remove_scratch(
                        temporary,
                        max(deadline, time.monotonic() + _CLEANUP_GRACE_SECONDS),
                    )
                except (OSError, _ResourceLimit) as error:
                    return self._finish_failure(
                        attempt,
                        OutcomeKind.RUNTIME_EXCEPTION,
                        started,
                        canonical_json(
                            {
                                "kind": "scratch_cleanup_failure",
                                "error_type": type(error).__name__,
                            }
                        ),
                        storage_bytes=storage_bytes,
                    )
                wall = time.monotonic() - started
                return self._writer.finish_execution(
                    attempt,
                    outcome_kind=(
                        OutcomeKind.TERMINAL_WITHOUT_DOWNSTREAM if terminal else OutcomeKind.SUCCESS
                    ),
                    scored_reward=score.reward,
                    scorer_evidence_sha256=score_sha256,
                    latency_seconds=wall,
                    dollars=0.0,
                    judge_calls=0,
                    cpu_seconds=wall,
                    gpu_seconds=0.0,
                    wall_seconds=wall,
                    storage_bytes=storage_bytes,
                )
            except subprocess.TimeoutExpired:
                return finish_with_cleanup(
                    OutcomeKind.TIMEOUT,
                    b"toy worker exceeded frozen wall timeout",
                )
            except _UnquiescedCallback:
                self._writer.close()
                raise
            except _ResourceLimit:
                return finish_with_cleanup(
                    OutcomeKind.RESOURCE_LIMIT,
                    b"toy execution exceeded the frozen end-to-end wall limit",
                )
            except LedgerError:
                raise
            except Exception as error:
                evidence = canonical_json(
                    {"kind": "worker_contract_failure", "error_type": type(error).__name__}
                )
                return finish_with_cleanup(
                    OutcomeKind.RUNTIME_EXCEPTION,
                    evidence,
                )

    def _request_payload(
        self,
        *,
        attempt_id: str,
        arm_id: str,
        action: BehaviorAction,
        continuation_replicate: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "domain": _DOMAIN,
            "toy_only_no_isolation_claim": True,
            "callback_timeout_contract": "caller_control_only_poison_and_stop",
            "storage_contract": "initial_and_final_workspace_snapshots_no_peak_quota",
            "ledger_id": self._writer.ledger_id,
            "group_id": self._context.spec.commitment.group_id,
            "target_id": self._context.spec.commitment.target_id,
            "execution_attempt_id": attempt_id,
            "arm_id": arm_id,
            "continuation_replicate": continuation_replicate,
            "pre_action_snapshot_sha256": (
                self._context.spec.commitment.pre_action_snapshot_sha256
            ),
            "workspace_manifest_sha256": self._context.workspace.digest,
            "runtime_sha256": self._context.runtime_sha256,
            "worker_command_sha256": _sha256(canonical_json(list(self._worker_command))),
            "config_sha256": self._context.config_sha256,
            "correspondence_sha256": self._context.spec.correspondence.receipt_sha256,
            "scorer_sha256": self._context.scorer_sha256,
            "action": json.loads(action.to_bytes()),
            "action_sha256": action.digest,
            "resource_limits": self._context.limits.to_payload(),
            "workspace_interface": "worker_process_cwd",
        }

    def _finish_failure(
        self,
        attempt: ExecutionAttempt,
        kind: OutcomeKind,
        started: float,
        evidence: bytes,
        *,
        storage_bytes: int,
    ) -> bytes:
        evidence_sha256 = self._writer.put_evidence(evidence)
        wall = time.monotonic() - started
        return self._writer.finish_execution(
            attempt,
            outcome_kind=kind,
            scored_reward=0.0,
            scorer_evidence_sha256=evidence_sha256,
            latency_seconds=wall,
            dollars=0.0,
            judge_calls=0,
            cpu_seconds=wall,
            gpu_seconds=0.0,
            wall_seconds=wall,
            storage_bytes=storage_bytes,
        )

    def _verify_frozen_artifacts(self) -> tuple[str, str]:
        runtime_manifest = self._cas.read_verified(self._context.runtime_sha256)
        artifacts = _verify_toy_worker_runtime(runtime_manifest, self._worker_command)
        for digest in artifacts:
            self._cas.read_verified(digest)
        self._cas.read_verified(self._context.config_sha256)
        self._cas.read_verified(self._context.scorer_sha256)
        return artifacts

    def _verify_callable_bindings(self) -> None:
        if (
            self._gateway.runtime_sha256 != self._context.runtime_sha256
            or self._gateway.config_sha256 != self._context.config_sha256
        ):
            raise ValueError("inference gateway changed after frozen binding")
        if self._scorer.artifact_sha256 != self._context.scorer_sha256:
            raise ValueError("scorer changed after frozen binding")

    def _score(
        self,
        action: BehaviorAction,
        continuations: Sequence[GatewayContinuationResponse],
        worker_result: Mapping[str, Any],
    ) -> ScoredResult:
        self._verify_callable_bindings()
        return self._scorer(
            action=action,
            continuations=continuations,
            worker_result=worker_result,
        )


def _parse_worker_output(value: bytes, *, request_digest: str) -> dict[str, Any]:
    if not value or len(value) > _MAX_CONTROL_BYTES:
        raise ValueError("worker output is empty or oversized")
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or canonical_json(parsed) != value:
        raise ValueError("worker output must be canonical JSON")
    expected = {
        "schema_version",
        "request_sha256",
        "downstream_events",
        "terminal_without_downstream",
        "worker_result",
    }
    if set(parsed) != expected or parsed["schema_version"] != 1:
        raise ValueError("worker output fields differ from the toy protocol")
    if parsed["request_sha256"] != request_digest:
        raise ValueError("worker output is bound to a different execution request")
    if type(parsed["terminal_without_downstream"]) is not bool:
        raise ValueError("terminal_without_downstream must be bool")
    worker_result = parsed["worker_result"]
    if (
        not isinstance(worker_result, dict)
        or set(worker_result) != {"answer"}
        or not isinstance(worker_result["answer"], str)
        or len(worker_result["answer"].encode("utf-8")) > 64 * 1024
    ):
        raise ValueError("worker_result must be the typed semantic answer schema")
    events = parsed["downstream_events"]
    if not isinstance(events, list):
        raise ValueError("downstream_events must be a list")
    for event in events:
        if (
            not isinstance(event, dict)
            or set(event) != {"provenance", "request"}
            or not isinstance(event["provenance"], dict)
            or not isinstance(event["request"], dict)
        ):
            raise ValueError("worker event fields differ from the toy protocol")
    return parsed


def _verify_toy_worker_runtime(
    value: bytes,
    command: Sequence[str],
) -> tuple[str, str]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or canonical_json(parsed) != value:
        raise ValueError("toy worker runtime manifest must be canonical JSON")
    if set(parsed) != {"schema_version", "domain", "artifacts"} or (
        parsed["schema_version"] != 1 or parsed["domain"] != "redco-stage-d-toy-worker-runtime-v1"
    ):
        raise ValueError("toy worker runtime manifest envelope is invalid")
    artifacts = parsed["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 2 or len(command) < 2:
        raise ValueError("toy runtime must bind an executable and worker")
    digests: list[str] = []
    for index, role in enumerate(("executable", "worker")):
        artifact = artifacts[index]
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"role", "path", "sha256"}
            or artifact["role"] != role
            or artifact["path"] != command[index]
        ):
            raise ValueError("launched command differs from the frozen toy runtime")
        digests.append(_require_sha256(artifact["sha256"], "runtime artifact sha256"))
    return digests[0], digests[1]


def _request_with_seed(key: ExactActionKey, seed: int) -> dict[str, Any]:
    request = json.loads(key.request)
    if not isinstance(request, dict):
        raise ValueError("exact action request must be an object")
    request["seed"] = seed
    extra = request.get("extra_body")
    if not isinstance(extra, dict):
        raise ValueError("exact action extra_body must be an object")
    request["extra_body"] = {**extra, "cache_salt": f"candidate-{seed}"}
    return request


def _validated_semantic_request(request: Mapping[str, Any]) -> dict[str, Any]:
    reserved = {"seed", "coupling_mode", "call_id", "model", "sampling_config"}
    if reserved.intersection(request):
        raise ValueError("worker may not supply trusted transport fields")
    extra_body = request.get("extra_body")
    if isinstance(extra_body, Mapping) and reserved.intersection(extra_body):
        raise ValueError("worker extra_body may not override trusted transport fields")
    return dict(request)


def _worker_environment() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "TZ": "UTC",
        "LC_ALL": "C",
        "LANG": "C",
    }


def _run_worker(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
) -> int:
    if os.name != "nt":
        raise RuntimeError("toy worker execution requires Windows Job Objects")
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000004
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        start_new_session=False,
    )
    windows_job: int | None = None
    try:
        windows_job = _create_windows_kill_job(process)
        _resume_windows_process(process)
        returncode = process.wait(timeout=timeout)
    finally:
        if windows_job is not None:
            _close_windows_handle(windows_job)
        if process.poll() is None:
            process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10.0)
    return returncode


@contextmanager
def _locked_verified_windows_file(path: Path, expected_sha256: str) -> Iterator[None]:
    """Deny replacement/writes while a verified runtime file is being loaded."""
    import ctypes
    from ctypes import wintypes

    if os.name != "nt":
        raise RuntimeError("runtime file locking requires Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    handle = kernel32.CreateFileW(
        str(path),
        0x80000000,
        0x00000001,
        None,
        3,
        0x00000080,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_last_error(), "CreateFileW runtime lock failed")
    handle_value = int(handle)
    try:
        actual = _sha256(_read_regular_limited(path, _MAX_CAS_OBJECT_BYTES))
        if actual != expected_sha256:
            raise ValueError("locked runtime artifact differs from its frozen bytes")
        yield
    finally:
        _close_windows_handle(handle_value)


def _create_windows_kill_job(process: subprocess.Popen[bytes]) -> int:
    """Assign a suspended worker to a kill-on-close Windows Job Object."""
    import ctypes
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    job_value = int(job)
    information = ExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = 0x00002000
    if not kernel32.SetInformationJobObject(
        job,
        9,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        error = ctypes.get_last_error()
        _close_windows_handle(job_value)
        raise OSError(error, "SetInformationJobObject failed")
    process_handle = wintypes.HANDLE(cast(Any, process)._handle)
    if not kernel32.AssignProcessToJobObject(job, process_handle):
        error = ctypes.get_last_error()
        _close_windows_handle(job_value)
        raise OSError(error, "AssignProcessToJobObject failed")
    return job_value


def _resume_windows_process(process: subprocess.Popen[bytes]) -> None:
    import ctypes
    from ctypes import wintypes

    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    status = int(ntdll.NtResumeProcess(wintypes.HANDLE(cast(Any, process)._handle)))
    if status != 0:
        raise OSError(status, "NtResumeProcess failed")


def _close_windows_handle(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    if not kernel32.CloseHandle(wintypes.HANDLE(handle)):
        raise OSError(ctypes.get_last_error(), "CloseHandle failed")


def _durable_write(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _scratch_directory(root: Path) -> Iterator[str]:
    path = Path(tempfile.mkdtemp(dir=root, prefix="redco-stage-d-toy-"))
    try:
        yield str(path)
    except BaseException:
        with suppress(OSError):
            shutil.rmtree(path)
        raise


def _remove_scratch(path: Path, deadline: float) -> None:
    last_error: OSError | None = None
    for _ in range(20):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as error:
            last_error = error
            remaining = _remaining(deadline)
            time.sleep(min(0.025, remaining))
    assert last_error is not None
    raise last_error


def _read_regular_limited(path: Path, limit: int) -> bytes:
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("worker output must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("worker output changed to a non-regular file")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("worker output changed while it was opened")
        if opened.st_size > limit:
            raise _ResourceLimit("worker output exceeded frozen byte limit")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
        if len(value) > limit or os.read(descriptor, 1):
            raise _ResourceLimit("worker output exceeded frozen byte limit")
        return value
    finally:
        os.close(descriptor)


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _ResourceLimit("wall-time budget exhausted")
    return remaining


def _check_deadline(deadline: float) -> None:
    _remaining(deadline)


def _call_with_deadline[T](call: Callable[[], T], deadline: float) -> T:
    _remaining(deadline)
    completed = threading.Event()
    result: deque[tuple[bool, object]] = deque(maxlen=1)

    def invoke() -> None:
        try:
            result.append((True, call()))
        except BaseException as error:
            result.append((False, error))
        finally:
            completed.set()

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not completed.wait(remaining):
        raise _UnquiescedCallback(
            "trusted callback exceeded caller wall limit; run is poisoned and stopped"
        )
    succeeded, value = result[0]
    if not succeeded:
        assert isinstance(value, BaseException)
        raise value
    return cast(T, value)


def _validate_relative_path(value: str) -> None:
    if not value or "\\" in value:
        raise ValueError("workspace path must be nonempty canonical POSIX")
    path = PurePosixPath(value)
    if len(value.encode("utf-8")) > 4096 or len(path.parts) > 128:
        raise ValueError("workspace path exceeds the frozen structural limit")
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("workspace path must stay beneath /workspace")
    if path.as_posix() != value:
        raise ValueError("workspace path is not canonical")


def _stat_is_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x00000400))


def _is_real_directory(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(metadata.st_mode) and not _stat_is_reparse(metadata)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not (-float("inf") < value < float("inf")):
        raise ValueError(f"{name} must be a finite float")
    return value
