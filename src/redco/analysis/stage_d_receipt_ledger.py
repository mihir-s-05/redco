"""Durable, fail-closed receipt ledger for Stage D scientific execution."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import secrets
import shutil
import stat
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Literal, cast

from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.analysis.stage_d_ledger_contracts import (
    BatchAlreadyClaimed,
    CandidateAttempt,
    ExecutionAttempt,
    LedgerError,
    LedgerPoisoned,
    ModelCallAttempt,
    ReplayOverrideTicket,
)
from redco.analysis.stage_d_ledger_validation import (
    validate_state_machine as _validate_state_machine,
)
from redco.analysis.stage_d_scientific_branch_group import (
    OutcomeKind,
    behavior_law_digest,
)
from redco.analysis.stage_d_spawn_provenance import (
    CouplingMode,
    EventSeedScheduler,
    PolicyEventAddress,
    ScheduledSeed,
)
from redco.contracts import ActualEvaluationCost, LogicalDeploymentCost, canonical_json

_DOMAIN = "redco-stage-d-receipt-ledger-v2"
_GENESIS_PRIOR = "0" * 64
_MAX_RECEIPT_BYTES = 1 << 20
_MAX_RECORD_BYTES = 2 << 20
_MAX_FINALIZATION_REQUEST_BYTES = 8 << 20
# The cleanup request carries the canonical finalization request as hex.  It
# therefore has a deliberately separate envelope bound instead of reusing the
# request bound (which silently rejected valid large requests).
_MAX_FINALIZATION_RESULT_BYTES = 16 << 20
_MAX_FINALIZATION_CLEANUP_BYTES = (2 * _MAX_FINALIZATION_REQUEST_BYTES) + (1 << 20)
_MAX_FINALIZATION_CLEANUP_RESULT_BYTES = _MAX_FINALIZATION_RESULT_BYTES + (1 << 20)
_RECORD_KEYS = {
    "schema_version",
    "domain",
    "ledger_id",
    "offset",
    "prior_record_sha256",
    "record_kind",
    "body",
}

RecoveryStatus = Literal[
    "active-clean",
    "active-repairable-zero-call",
    "sealed-valid",
    "poisoned",
]
_FINALIZATION_TRANSACTION_DOMAIN = "redco-stage-d-finalization-transaction-v2"
_FINALIZATION_TRANSACTION_SCHEMA_VERSION = 2
_FINALIZATION_REQUEST_DOMAIN = "redco-stage-d-finalization-request-v2"
_FINALIZATION_RESULT_DOMAIN = "redco-stage-d-finalization-result-v2"
_FINALIZATION_CLEANUP_DOMAIN = "redco-stage-d-finalization-cleanup-v2"
_FINALIZATION_TERM_GRACE_SECONDS = 0.1
_FINALIZATION_KILL_WAIT_SECONDS = 2.0
_FINALIZATION_SOURCE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_FINALIZATION_REQUEST_KEYS = {
    "schema_version",
    "domain",
    "transaction_id",
    "root_path",
    "root_binding_sha256",
    "master_seed",
    "master_seed_sha256",
    "ledger_id",
    "genesis_sha256",
    "base_head_sha256",
    "base_record_count",
    "operation",
    "evidence_hex",
    "evidence_sha256",
    "group_id",
    "target_id",
    "recorded_action_hex",
    "passed",
    "actual_cost",
    "execution_attempt",
    "outcome_kind",
    "scored_reward",
    "latency_seconds",
    "dollars",
    "judge_calls",
    "cpu_seconds",
    "gpu_seconds",
    "wall_seconds",
    "storage_bytes",
}
_FINALIZATION_RESULT_KEYS = {
    "schema_version",
    "domain",
    "transaction_id",
    "request_sha256",
    "result_sha256",
    "receipt_hex",
    "record_patches",
    "evidence_patches",
}
_FINALIZATION_PATCH_KEYS = {
    "relative_path",
    "temporary_relative_path",
    "sha256",
    "bytes",
    "content_hex",
}
_FINALIZATION_MANIFEST_KEYS = {
    "schema_version",
    "domain",
    "transaction_id",
    "root_path",
    "root_binding_sha256",
    "ledger_id",
    "genesis_sha256",
    "base_head_sha256",
    "operation",
    "state",
    "request_sha256",
    "result_sha256",
    "result",
    "baseline_paths",
    "baseline_sha256",
    "request_hex",
}
_FINALIZATION_CLEANUP_KEYS = {
    "schema_version",
    "domain",
    "transaction_id",
    "root_path",
    "root_binding_sha256",
    "request_sha256",
    "action",
    "request_hex",
}
_FINALIZATION_CLEANUP_RESULT_KEYS = {
    "schema_version",
    "domain",
    "transaction_id",
    "request_sha256",
    "state",
    "result",
}


@dataclass(frozen=True, slots=True)
class _ActionDigestProxy:
    digest: str


@dataclass(frozen=True, slots=True)
class _FinalizationTransactionSpec:
    """Internal, parent-authored request for one isolated ledger finalization."""

    operation: Literal["qa", "execution"]
    root: Path
    master_seed: str
    transaction_id: str
    ledger_id: str
    genesis_sha256: str
    base_head_sha256: str
    base_record_count: int
    evidence: bytes
    group_id: str
    target_id: str
    recorded_action: BehaviorAction | _ActionDigestProxy | None
    recorded_action_bytes: bytes | None
    passed: bool | None
    actual_cost: ActualEvaluationCost | None
    execution_attempt: ExecutionAttempt | None
    outcome_kind: OutcomeKind | None
    scored_reward: float
    latency_seconds: float
    dollars: float
    judge_calls: int
    cpu_seconds: float
    gpu_seconds: float
    wall_seconds: float
    storage_bytes: int


def _writer_transaction[**P, R](method: Callable[P, R]) -> Callable[P, R]:
    """Serialize a complete public writer transition, including its state checks."""

    @wraps(method)
    def locked(*args: P.args, **kwargs: P.kwargs) -> R:
        owner = cast(Any, args[0])
        with owner._state_lock:
            return method(*args, **kwargs)

    return locked


@dataclass(frozen=True, slots=True)
class GenesisBinding:
    preregistration_sha256: str
    source_sha256: str
    runtime_sha256: str
    config_sha256: str
    protocol_manifest_sha256: str
    master_seed_sha256: str
    support_rules_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "preregistration_sha256",
            "source_sha256",
            "runtime_sha256",
            "config_sha256",
            "protocol_manifest_sha256",
            "master_seed_sha256",
            "support_rules_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    def to_payload(self) -> dict[str, str]:
        return {
            "preregistration_sha256": self.preregistration_sha256,
            "source_sha256": self.source_sha256,
            "runtime_sha256": self.runtime_sha256,
            "config_sha256": self.config_sha256,
            "protocol_manifest_sha256": self.protocol_manifest_sha256,
            "master_seed_sha256": self.master_seed_sha256,
            "support_rules_sha256": self.support_rules_sha256,
        }


@dataclass(frozen=True, slots=True)
class LedgerSeal:
    ledger_id: str
    genesis_sha256: str
    head_sha256: str
    record_count: int
    receipt_count: int

    def __post_init__(self) -> None:
        if not self.ledger_id:
            raise ValueError("ledger_id must be nonempty")
        _require_sha256(self.genesis_sha256, "genesis_sha256")
        _require_sha256(self.head_sha256, "head_sha256")
        _exact_int(self.record_count, "record_count", minimum=2)
        _exact_int(self.receipt_count, "receipt_count")

    def to_payload(self) -> dict[str, str | int]:
        return {
            "ledger_id": self.ledger_id,
            "genesis_sha256": self.genesis_sha256,
            "head_sha256": self.head_sha256,
            "record_count": self.record_count,
            "receipt_count": self.receipt_count,
        }

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-ledger-seal-v1",
                "seal": self.to_payload(),
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> LedgerSeal:
        parsed = _strict_canonical_object(value, "Stage D ledger seal")
        if set(parsed) != {"schema_version", "domain", "seal"} or (
            parsed["schema_version"],
            parsed["domain"],
        ) != (1, "redco-stage-d-ledger-seal-v1"):
            raise ValueError("unsupported Stage D ledger seal")
        seal = parsed["seal"]
        if not isinstance(seal, dict) or set(seal) != {
            "ledger_id",
            "genesis_sha256",
            "head_sha256",
            "record_count",
            "receipt_count",
        }:
            raise ValueError("Stage D ledger seal fields differ")
        return cls(
            seal["ledger_id"],
            seal["genesis_sha256"],
            seal["head_sha256"],
            seal["record_count"],
            seal["receipt_count"],
        )


@dataclass(frozen=True, slots=True)
class RecordedActionReservation:
    ledger_id: str
    group_id: str
    target_id: str
    reservation_id: str
    action_seed: int
    exact_action_key_digest: str
    request_sha256: str
    commitment_receipt: bytes


@dataclass(frozen=True, slots=True)
class SourcePolicyCallReservation:
    ledger_id: str
    group_id: str
    rollout_id: str
    decision_id: str
    receipt: bytes
    exact_action_key_digest: str
    request_sequence: int


@dataclass(frozen=True, slots=True)
class SourceRolloutCompletion:
    ledger_id: str
    group_id: str
    rollout_id: str
    source_sha256: str
    receipt: bytes


@dataclass(frozen=True, slots=True)
class StageDTrainingBatchAuthorization:
    ledger_id: str
    arm: Literal["stock", "branch-global", "local"]
    training_batch_identity: str
    sealed_batch_sha256: str
    receipt: bytes


@dataclass(frozen=True, slots=True)
class _ScanResult:
    status: RecoveryStatus
    reason: str | None
    records: tuple[dict[str, Any], ...]
    record_sha256s: tuple[str, ...]
    receipts: Mapping[tuple[str, str], dict[str, Any]]
    evidence_refs: frozenset[str]
    seal: LedgerSeal | None
    repairable_attempt: Mapping[str, Any] | None = None


class SealedReceiptVerifier:
    """Read-only verifier authorized by an out-of-band expected terminal seal."""

    def __init__(self, root: Path, expected: LedgerSeal) -> None:
        scan = inspect_ledger(root)
        if scan.status != "sealed-valid" or scan.seal != expected:
            raise LedgerError("ledger does not match the out-of-band expected seal")
        self._root = root
        self._expected = expected

    def __call__(self, receipt: bytes, *, receipt_kind: str) -> Mapping[str, Any]:
        scan = inspect_ledger(self._root)
        if scan.status != "sealed-valid" or scan.seal != self._expected:
            raise LedgerError("sealed ledger changed after verifier construction")
        return _lookup_receipt(scan, receipt, receipt_kind)


class StageDReceiptLedger:
    """Exclusive trusted writer and live receipt verifier for one Stage D run."""

    _branch_target_roster_sha256: str | None
    _branch_target_keys: set[tuple[str, str]]
    _reconstruction_qa_barrier_sha256: str | None

    def __init__(
        self,
        root: Path,
        *,
        master_seed: str,
        _allow_repairable_zero_call: bool = False,
    ) -> None:
        _validate_root_ancestors(root)
        self._root = root
        self._records_dir = root / "records"
        self._evidence_dir = root / "evidence"
        self._lock_path = root / "writer.lock"
        self._state_lock = threading.RLock()
        self._closed = False
        self._poisoned = False
        self._finalization_active = False
        self._lock_descriptor: int | None = _acquire_writer_lock(self._lock_path)
        try:
            _resolve_stale_finalization_manifests(root)
            scan = (
                _scan_ledger(root, allow_repairable_zero_call=True)
                if _allow_repairable_zero_call
                else inspect_ledger(root)
            )
            accepted = {"active-clean"}
            if _allow_repairable_zero_call:
                accepted.add("active-repairable-zero-call")
            if scan.status not in accepted:
                raise LedgerError(f"writer requires active-clean ledger, got {scan.status}")
            genesis = scan.records[0]
            body = _body(genesis)
            if body["master_seed_sha256"] != _sha256(master_seed.encode("utf-8")):
                raise LedgerError("master seed does not match the ledger genesis")
            self._master_seed = master_seed
            self._ledger_id = str(genesis["ledger_id"])
            self._records = list(scan.records)
            self._record_sha256s = list(scan.record_sha256s)
            self._receipts = dict(scan.receipts)
            self._evidence_refs = set(scan.evidence_refs)
            self._rebuild_state()
            self._repairable_attempt = scan.repairable_attempt
        except BaseException:
            self._release_lock()
            raise

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        binding: GenesisBinding,
        master_seed: str,
    ) -> StageDReceiptLedger:
        if not master_seed:
            raise ValueError("master_seed must be nonempty")
        if binding.master_seed_sha256 != _sha256(master_seed.encode("utf-8")):
            raise ValueError("binding does not match master_seed")
        _validate_root_ancestors(root)
        root.mkdir(parents=True, exist_ok=False)
        (root / "records").mkdir()
        (root / "evidence").mkdir()
        if os.name != "nt":
            _fsync_directory(root)
            _fsync_directory(root.parent)
        ledger_id = secrets.token_hex(16)
        genesis = {
            "schema_version": 1,
            "domain": _DOMAIN,
            "ledger_id": ledger_id,
            "offset": 0,
            "prior_record_sha256": _GENESIS_PRIOR,
            "record_kind": "genesis",
            "body": binding.to_payload(),
        }
        _atomic_record_write(root / "records" / _record_name(0), canonical_json(genesis))
        return cls(root, master_seed=master_seed)

    @classmethod
    def recover_zero_call_failure(
        cls,
        root: Path,
        *,
        master_seed: str,
        reason: str,
        supervisor_evidence: bytes,
    ) -> StageDReceiptLedger:
        """Close one hard-killed, provably zero-activity attempt and reopen safely."""
        if not reason or not supervisor_evidence:
            raise ValueError("zero-call recovery requires reason and supervisor evidence")
        ledger = cls(
            root,
            master_seed=master_seed,
            _allow_repairable_zero_call=True,
        )
        try:
            repair = ledger._repairable_attempt
            if repair is None:
                raise LedgerError("ledger has no repairable zero-call attempt")
            evidence_sha256 = ledger.put_evidence(supervisor_evidence)
            if repair["attempt_kind"] == "candidate":
                attempt = CandidateAttempt(
                    ledger.ledger_id,
                    repair["group_id"],
                    repair["target_id"],
                    repair["action_slot"],
                    repair["action_seed"],
                    repair["attempt_ordinal"],
                    repair["attempt_id"],
                )
                ledger.record_zero_call_candidate_failure(
                    attempt,
                    reason=reason,
                    supervisor_evidence_sha256=evidence_sha256,
                )
            else:
                attempt = ExecutionAttempt(
                    ledger.ledger_id,
                    repair["group_id"],
                    repair["target_id"],
                    repair["arm_id"],
                    repair["action_digest"],
                    repair["continuation_replicate"],
                    repair["attempt_ordinal"],
                    repair["attempt_id"],
                )
                ledger.record_zero_call_execution_failure(
                    attempt,
                    reason=reason,
                    supervisor_evidence_sha256=evidence_sha256,
                )
            ledger._repairable_attempt = None
            return ledger
        except BaseException:
            ledger.close()
            raise

    @property
    def ledger_id(self) -> str:
        return self._ledger_id

    @property
    def genesis_binding(self) -> GenesisBinding:
        """Return the immutable run binding anchored by the genesis record."""
        body = _body(self._records[0])
        return GenesisBinding(
            preregistration_sha256=str(body["preregistration_sha256"]),
            source_sha256=str(body["source_sha256"]),
            runtime_sha256=str(body["runtime_sha256"]),
            config_sha256=str(body["config_sha256"]),
            protocol_manifest_sha256=str(body["protocol_manifest_sha256"]),
            master_seed_sha256=str(body["master_seed_sha256"]),
            support_rules_sha256=str(body["support_rules_sha256"]),
        )

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def head_sha256(self) -> str:
        return self._record_sha256s[-1]

    @property
    def completed_source_sha256s(self) -> tuple[str, ...]:
        """Return the exact completed-source roster anchored in this live writer."""
        return tuple(
            sorted(
                completion.source_sha256 for completion in self._source_rollout_completed.values()
            )
        )

    @property
    def completed_branch_artifact_sha256s(self) -> tuple[str, ...]:
        """Return the exact completed scientific-artifact roster."""
        return tuple(sorted(self._branch_artifacts.values()))

    @property
    def branch_target_roster_sha256(self) -> str | None:
        """Return the single frozen target-roster identity, if one was recorded."""
        return self._branch_target_roster_sha256

    @property
    def branch_target_keys(self) -> tuple[tuple[str, str], ...]:
        """Return the exact frozen scientific target roster."""
        return tuple(sorted(self._branch_target_keys))

    @property
    def reconstruction_qa_barrier_sha256(self) -> str | None:
        """Return the whole-roster zero-call QA barrier receipt hash, if sealed."""
        return self._reconstruction_qa_barrier_sha256

    def reconstruction_qa_receipt(
        self,
        group_id: str,
        target_id: str,
    ) -> bytes | None:
        """Return already-anchored QA bytes so recovery never repeats QA work."""
        key = _group_key(group_id, target_id)
        digest = self._qa_receipt_sha256s.get(key)
        if digest is None:
            return None
        payload = self._receipts.get(("reconstruction_qa", digest))
        if payload is None:
            raise LedgerPoisoned("QA state lacks its anchored receipt bytes")
        return canonical_json(payload)

    def reconstruction_qa_barrier_receipt(self) -> bytes | None:
        """Return the existing barrier bytes instead of creating a duplicate."""
        digest = self._reconstruction_qa_barrier_sha256
        if digest is None:
            return None
        payload = self._receipts.get(("reconstruction_qa_barrier", digest))
        if payload is None:
            raise LedgerPoisoned("QA barrier state lacks its anchored receipt bytes")
        return canonical_json(payload)

    def require_reconstruction_qa_ready(
        self,
        *,
        group_id: str,
        target_id: str,
        recorded_action: BehaviorAction,
        actual_cost: ActualEvaluationCost,
    ) -> None:
        """Check the shared model-free QA gate without changing ledger state.

        The guard is deliberately shared by the pre-``run_eval`` entry point and
        the durable receipt writer.  A caller cannot use a correctly shaped QA
        receipt to bypass the frozen target roster, correspondence, recorded
        action, or the no-scientific-activity barrier.
        """

        with self._state_lock:
            self._require_reconstruction_qa_ready(
                group_id=group_id,
                target_id=target_id,
                recorded_action=recorded_action,
                actual_cost=actual_cost,
            )

    async def record_reconstruction_qa_transaction(
        self,
        *,
        group_id: str,
        target_id: str,
        recorded_action: BehaviorAction,
        passed: bool,
        report: bytes,
        actual_cost: ActualEvaluationCost,
    ) -> bytes:
        """Commit evidence and its QA receipt as one bounded transaction.

        The existing receipt methods remain the sole schema/state owner. This
        method only moves their execution behind an isolated child-process
        transaction so a synchronous fsync cannot pin the event loop. The live
        writer is closed while the child owns the lock; an uncommitted child
        transaction is removed and the writer stays closed fail-closed.
        """

        with self._state_lock:
            self._require_reconstruction_qa_ready(
                group_id=group_id,
                target_id=target_id,
                recorded_action=recorded_action,
                actual_cost=actual_cost,
            )
            if type(passed) is not bool:
                raise ValueError("passed must be bool")
            if type(report) is not bytes or not report:
                raise ValueError("report must be nonempty immutable bytes")
            spec = self._new_finalization_spec(
                operation="qa",
                group_id=group_id,
                target_id=target_id,
                evidence=report,
                recorded_action=recorded_action,
                passed=passed,
                actual_cost=actual_cost,
                execution_attempt=None,
                outcome_kind=None,
            )
        return await self._run_finalization_transaction(spec)

    async def finish_execution_transaction(
        self,
        attempt: ExecutionAttempt,
        *,
        outcome_kind: OutcomeKind,
        scored_reward: float,
        scorer_evidence: bytes,
        latency_seconds: float,
        dollars: float,
        judge_calls: int,
        cpu_seconds: float,
        gpu_seconds: float,
        wall_seconds: float,
        storage_bytes: int,
    ) -> bytes:
        """Commit execution evidence and receipt through the same owner."""

        with self._state_lock:
            self._require_writable()
            if type(scorer_evidence) is not bytes or not scorer_evidence:
                raise ValueError("scorer evidence must be nonempty immutable bytes")
            if not isinstance(outcome_kind, OutcomeKind):
                raise ValueError("outcome_kind must be OutcomeKind")
            spec = self._new_finalization_spec(
                operation="execution",
                group_id=attempt.group_id,
                target_id=attempt.target_id,
                evidence=scorer_evidence,
                recorded_action=None,
                passed=None,
                actual_cost=None,
                execution_attempt=attempt,
                outcome_kind=outcome_kind,
                scored_reward=scored_reward,
                latency_seconds=latency_seconds,
                dollars=dollars,
                judge_calls=judge_calls,
                cpu_seconds=cpu_seconds,
                gpu_seconds=gpu_seconds,
                wall_seconds=wall_seconds,
                storage_bytes=storage_bytes,
            )
        return await self._run_finalization_transaction(spec)

    def _new_finalization_spec(self, **kwargs: Any) -> _FinalizationTransactionSpec:
        self._require_writable()
        if self._finalization_active:
            raise LedgerError("another finalization transaction is active")
        self._finalization_active = True
        recorded_action = kwargs.pop("recorded_action")
        recorded_action_bytes = kwargs.pop("recorded_action_bytes", None)
        if recorded_action is not None and recorded_action_bytes is None:
            recorded_action_bytes = recorded_action.to_bytes()
        return _FinalizationTransactionSpec(
            root=self._root,
            master_seed=self._master_seed,
            transaction_id=secrets.token_hex(16),
            ledger_id=self._ledger_id,
            genesis_sha256=self._record_sha256s[0],
            base_head_sha256=self.head_sha256,
            base_record_count=len(self._records),
            scored_reward=kwargs.pop("scored_reward", 0.0),
            latency_seconds=kwargs.pop("latency_seconds", 0.0),
            dollars=kwargs.pop("dollars", 0.0),
            judge_calls=kwargs.pop("judge_calls", 0),
            cpu_seconds=kwargs.pop("cpu_seconds", 0.0),
            gpu_seconds=kwargs.pop("gpu_seconds", 0.0),
            wall_seconds=kwargs.pop("wall_seconds", 0.0),
            storage_bytes=kwargs.pop("storage_bytes", 0),
            recorded_action=recorded_action,
            recorded_action_bytes=recorded_action_bytes,
            **kwargs,
        )

    async def _run_finalization_transaction(
        self,
        spec: _FinalizationTransactionSpec,
    ) -> bytes:
        process: asyncio.subprocess.Process | None = None
        try:
            request = _finalization_request_bytes(spec)
            process = await asyncio.create_subprocess_exec(
                *_finalization_child_command("--finalize-transaction-stdin", spec.root),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await process.communicate(request)
            except asyncio.CancelledError:
                # The watchdog cancellation is the deadline signal.  Clear
                # it while the bounded child termination/rollback owner runs;
                # otherwise Python 3.12 can cancel the cleanup await itself
                # and leave the event-loop transaction unresolved.
                current = asyncio.current_task()
                if current is not None:
                    while current.cancelling():
                        current.uncancel()
                await _terminate_finalization_process(process)
                resolution = await _run_finalization_cleanup_owner(spec, action="resolve")
                if resolution.get("state") == "committed":
                    result = _validate_finalization_result(
                        spec, resolution["result"], spec.base_record_count
                    )
                    await _run_finalization_cleanup_owner(spec, action="ack")
                    self._apply_finalization_result(spec, result)
                    return cast(bytes, result["receipt"])
                raise
            if len(stdout) > _MAX_FINALIZATION_RESULT_BYTES:
                raise LedgerPoisoned("finalization result exceeds its bound")
            if process.returncode != 0:
                message = stderr.decode("utf-8", errors="replace").strip()
                resolution = await _run_finalization_cleanup_owner(spec, action="resolve")
                if resolution.get("state") == "committed":
                    result = _validate_finalization_result(
                        spec, resolution["result"], spec.base_record_count
                    )
                    await _run_finalization_cleanup_owner(spec, action="ack")
                    self._apply_finalization_result(spec, result)
                    return cast(bytes, result["receipt"])
                raise LedgerError(
                    "isolated finalization transaction failed"
                    + (f": {message}" if message else "")
                )
            # The child result is only a transport hint.  The authenticated
            # committed manifest is the authority, so ack re-reads and
            # validates it in the bounded owner before the parent changes any
            # in-memory state.
            del stdout
            resolution = await _run_finalization_cleanup_owner(spec, action="ack")
            result = _validate_finalization_result(
                spec, resolution["result"], spec.base_record_count
            )
            self._apply_finalization_result(spec, result)
            return cast(bytes, result["receipt"])
        except BaseException:
            try:
                if process is not None and process.returncode is None:
                    await _terminate_finalization_process(process)
            except BaseException:
                # Preserve the first operational exception.  The writer is
                # nevertheless poisoned even if child teardown reports a
                # secondary failure.
                pass
            finally:
                self._fail_closed_after_finalization()
            raise
        finally:
            with self._state_lock:
                self._finalization_active = False

    def _fail_closed_after_finalization(self) -> None:
        """Close and poison after any unresolved transaction failure."""

        with self._state_lock:
            self._poisoned = True
            self._closed = True
            self._finalization_active = False
            descriptor = self._lock_descriptor
            self._lock_descriptor = None
        if descriptor is not None:
            with suppress(BaseException):
                _release_os_lock(descriptor)

    def _apply_finalization_result(
        self,
        spec: _FinalizationTransactionSpec,
        result: Mapping[str, Any],
    ) -> None:
        """Apply a child-owned commit to the already-authenticated memory view.

        The parent deliberately does not rescan the filesystem here.  The
        child has authenticated the committed manifest while holding this
        writer's exclusive lock; only the canonical patch bytes are folded
        into the parent state after that proof.
        """

        with self._state_lock:
            if (
                len(self._records) != spec.base_record_count
                or self.head_sha256 != spec.base_head_sha256
            ):
                self._poisoned = True
                raise LedgerPoisoned("ledger changed during finalization")
            for patch in cast(Sequence[Mapping[str, Any]], result["evidence_patches"]):
                encoded = cast(bytes, patch["content"])
                if _sha256(encoded) != patch["sha256"]:
                    self._poisoned = True
                    raise LedgerPoisoned("finalization evidence patch changed")
            for patch in cast(Sequence[Mapping[str, Any]], result["record_patches"]):
                encoded = cast(bytes, patch["content"])
                record = _strict_canonical_object(encoded, "finalization record")
                expected_offset = len(self._records)
                expected_prior = self.head_sha256
                if (
                    record.get("ledger_id") != self._ledger_id
                    or record.get("offset") != expected_offset
                    or record.get("prior_record_sha256") != expected_prior
                ):
                    self._poisoned = True
                    raise LedgerPoisoned("finalization record chain changed")
                self._records.append(record)
                self._record_sha256s.append(_sha256(encoded))
                body = _body(record)
                refs = body.get("evidence_refs", [])
                if isinstance(refs, list):
                    self._evidence_refs.update(cast(str, ref) for ref in refs)
                if record.get("record_kind") == "receipt":
                    receipt = body.get("receipt")
                    receipt_sha256 = body.get("receipt_sha256")
                    kind = body.get("receipt_kind")
                    if (
                        not isinstance(receipt, dict)
                        or not isinstance(kind, str)
                        or not _is_sha256(receipt_sha256)
                    ):
                        self._poisoned = True
                        raise LedgerPoisoned("finalization receipt is malformed")
                    self._receipts[(kind, cast(str, receipt_sha256))] = receipt
            self._rebuild_state()
            self._repairable_attempt = None

    def __enter__(self) -> StageDReceiptLedger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @_writer_transaction
    def close(self) -> None:
        if self._finalization_active:
            raise LedgerError("cannot close while finalization is active")
        if not self._closed:
            self._closed = True
            self._release_lock()

    @_writer_transaction
    def __call__(self, receipt: bytes, *, receipt_kind: str) -> Mapping[str, Any]:
        self._require_writable()
        scan = inspect_ledger(self._root, allow_source_inflight=True)
        if scan.status != "active-clean":
            self._poisoned = True
            raise LedgerPoisoned(
                "live ledger is no longer active-clean"
                + (f": {scan.reason}" if scan.reason else "")
            )
        return _lookup_receipt(scan, receipt, receipt_kind)

    @_writer_transaction
    def put_evidence(self, evidence: bytes) -> str:
        self._require_writable()
        if type(evidence) is not bytes or not evidence:
            raise ValueError("evidence must be nonempty immutable bytes")
        digest = _sha256(evidence)
        path = self._evidence_dir / digest
        if path.exists():
            if path.read_bytes() != evidence:
                raise LedgerPoisoned("content-addressed evidence collision")
            self._evidence_refs.add(digest)
            return digest
        _atomic_blob_write(path, evidence)
        self._evidence_refs.add(digest)
        return digest

    @_writer_transaction
    def completed_candidate_evidence(
        self,
        *,
        group_id: str,
        target_id: str,
        action_slot: int,
    ) -> tuple[bytes, bytes] | None:
        """Return immutable action/receipt bytes for an already-paid candidate slot."""
        self._require_writable()
        key = (group_id, target_id, action_slot)
        receipt = self._candidate_receipts.get(key)
        action_sha256 = self._candidate_action_sha256s.get(key)
        if receipt is None and action_sha256 is None:
            return None
        if receipt is None or action_sha256 is None:
            raise LedgerPoisoned("candidate recovery evidence is incomplete")
        action = (self._evidence_dir / action_sha256).read_bytes()
        if _sha256(action) != action_sha256:
            raise LedgerPoisoned("candidate action evidence changed")
        return action, receipt

    @_writer_transaction
    def completed_execution_receipt(
        self,
        *,
        group_id: str,
        target_id: str,
        arm_id: str,
        continuation_replicate: int,
    ) -> bytes | None:
        """Return one already-completed arm receipt without re-executing it."""
        self._require_writable()
        return self._execution_receipts.get(
            (group_id, target_id, arm_id, continuation_replicate)
        )

    @_writer_transaction
    def completed_branch_artifact_sha256(
        self,
        *,
        group_id: str,
        target_id: str,
    ) -> str | None:
        """Return the committed artifact digest for one completed target."""
        self._require_writable()
        return self._branch_artifacts.get((group_id, target_id))

    @_writer_transaction
    def reserve_source_policy_call(
        self,
        *,
        group_id: str,
        rollout_id: str,
        decision_id: str,
        node_kind: Literal["root", "child"],
        target_id: str | None,
        target_ordinal: int | None,
        target_address: PolicyEventAddress,
        recorded_action_key: ExactActionKey,
        request_sha256: str,
        branch_selected: bool,
        raw_response_required: bool = False,
        recorded_action_reservation: RecordedActionReservation | None = None,
    ) -> SourcePolicyCallReservation:
        """Durably reserve one source-rollout policy call before forwarding it."""
        self._require_writable()
        if not group_id or not rollout_id or not decision_id:
            raise ValueError("source policy identifiers must be nonempty")
        if node_kind not in {"root", "child"}:
            raise ValueError("source policy node kind must be root or child")
        if (node_kind == "root" and (target_id is not None or target_ordinal is not None)) or (
            node_kind == "child"
            and (not target_id or type(target_ordinal) is not int or target_ordinal < 0)
        ):
            raise ValueError("source policy target fields disagree with node kind")
        if type(branch_selected) is not bool:
            raise ValueError("branch_selected must be bool")
        if type(raw_response_required) is not bool:
            raise ValueError("raw_response_required must be bool")
        if branch_selected and node_kind != "child":
            raise ValueError("only child decisions can be selected for branching")
        self._require_evidence(request_sha256)
        if request_sha256 != recorded_action_key.request_sha256:
            raise ValueError("source policy request evidence differs from exact action key")
        key = (rollout_id, decision_id)
        if key in self._source_policy_pending or key in self._source_policy_completed:
            raise LedgerError("source policy decision is already reserved")
        commitment_sha256: str | None = None
        recorded_action_reservation_id: str | None = None
        if branch_selected:
            assert target_id is not None
            commitment = self._commitments.get((group_id, target_id))
            if commitment is None:
                raise LedgerError("selected source policy call lacks pre-action commitment")
            if (
                commitment["rollout_id"] != rollout_id
                or commitment["target_address"] != _address_payload(target_address)
                or commitment["target_ordinal"] != target_ordinal
            ):
                raise LedgerError("selected source policy call differs from its commitment")
            commitment_sha256 = _sha256(canonical_json(commitment))
            state = self._recorded_reservations.get((group_id, target_id))
            if (
                recorded_action_reservation is None
                or state is None
                or state["reservation"] != recorded_action_reservation
                or recorded_action_reservation.ledger_id != self._ledger_id
                or recorded_action_reservation.exact_action_key_digest != recorded_action_key.digest
                or recorded_action_reservation.request_sha256 != request_sha256
            ):
                raise LedgerError(
                    "selected source call lacks its same-ledger recorded-action reservation"
                )
            recorded_action_reservation_id = recorded_action_reservation.reservation_id
        elif recorded_action_reservation is not None:
            raise ValueError("unselected source calls cannot name recorded-action reservations")
        request_sequence = len(self._records)
        receipt = self._append_receipt(
            "source_policy_call_reserved",
            {
                "ledger_id": self._ledger_id,
                "ledger_offset": request_sequence,
                "prior_chain_sha256": self.head_sha256,
                "group_id": group_id,
                "rollout_id": rollout_id,
                "decision_id": decision_id,
                "node_kind": node_kind,
                "target_id": target_id,
                "target_ordinal": target_ordinal,
                "target_address": _address_payload(target_address),
                "exact_action_key_digest": recorded_action_key.digest,
                "request_sha256": request_sha256,
                "branch_selected": branch_selected,
                "raw_response_required": raw_response_required,
                "target_commitment_receipt_sha256": commitment_sha256,
                "recorded_action_reservation_id": recorded_action_reservation_id,
                "request_sequence": request_sequence,
            },
            evidence_refs=(request_sha256,),
        )
        reservation = SourcePolicyCallReservation(
            self._ledger_id,
            group_id,
            rollout_id,
            decision_id,
            receipt,
            recorded_action_key.digest,
            request_sequence,
        )
        self._source_policy_pending[key] = reservation
        self._source_policy_reservations[key] = json.loads(receipt)
        return reservation

    @_writer_transaction
    def complete_source_policy_call(
        self,
        reservation: SourcePolicyCallReservation,
        *,
        action: BehaviorAction,
        response_sha256: str,
    ) -> bytes:
        """Anchor the exact response for a previously reserved source policy call."""
        self._require_writable()
        key = (reservation.rollout_id, reservation.decision_id)
        if self._source_policy_pending.get(key) != reservation:
            raise LedgerError("source policy reservation is not pending")
        if (
            reservation.ledger_id != self._ledger_id
            or action.key.digest != reservation.exact_action_key_digest
        ):
            raise LedgerError("source policy completion differs from its reservation")
        raw_response_sha256 = self._source_policy_responses.get(key)
        raw_response_required = self._source_policy_reservations[key].get(
            "raw_response_required",
            False,
        )
        if raw_response_required and raw_response_sha256 is None:
            raise LedgerError("source policy completion lacks its durable raw response witness")
        self._require_evidence(response_sha256)
        if response_sha256 != _sha256(action.to_bytes()):
            raise ValueError("source policy response evidence differs from exact action")
        completion_sequence = len(self._records)
        receipt = self._append_receipt(
            "source_policy_call_completed",
            {
                "ledger_id": self._ledger_id,
                "ledger_offset": completion_sequence,
                "prior_chain_sha256": self.head_sha256,
                "group_id": reservation.group_id,
                "rollout_id": reservation.rollout_id,
                "decision_id": reservation.decision_id,
                "request_receipt_sha256": _sha256(reservation.receipt),
                "exact_action_key_digest": reservation.exact_action_key_digest,
                "action_digest": action.digest,
                "response_sha256": response_sha256,
                "raw_response_sha256": raw_response_sha256,
                "request_sequence": reservation.request_sequence,
                "completion_sequence": completion_sequence,
            },
            evidence_refs=tuple(
                sorted(
                    {response_sha256}
                    | ({raw_response_sha256} if raw_response_sha256 is not None else set())
                )
            ),
        )
        del self._source_policy_pending[key]
        self._source_policy_completed[key] = _sha256(receipt)
        self._source_policy_action_digests[key] = action.digest
        return receipt

    @_writer_transaction
    def mark_source_policy_response_observed(
        self,
        reservation: SourcePolicyCallReservation,
        *,
        response_sha256: str,
    ) -> bytes:
        """Anchor exact provider bytes before source-response parsing."""
        self._require_writable()
        key = (reservation.rollout_id, reservation.decision_id)
        if self._source_policy_pending.get(key) != reservation:
            raise LedgerError("source policy response does not match the pending reservation")
        if reservation.ledger_id != self._ledger_id:
            raise LedgerError("source policy response belongs to another ledger")
        if key in self._source_policy_responses:
            raise LedgerError("source policy response was observed twice")
        self._require_evidence(response_sha256)
        receipt = self._append_receipt(
            "source_policy_response_observed",
            {
                "ledger_id": self._ledger_id,
                "ledger_offset": len(self._records),
                "prior_chain_sha256": self.head_sha256,
                "group_id": reservation.group_id,
                "rollout_id": reservation.rollout_id,
                "decision_id": reservation.decision_id,
                "request_receipt_sha256": _sha256(reservation.receipt),
                "exact_action_key_digest": reservation.exact_action_key_digest,
                "raw_response_sha256": response_sha256,
                "request_sequence": reservation.request_sequence,
            },
            evidence_refs=(response_sha256,),
        )
        self._source_policy_responses[key] = response_sha256
        return receipt

    @_writer_transaction
    def abort_source_policy_call(
        self,
        reservation: SourcePolicyCallReservation,
        *,
        phase: Literal[
            "post_unknown",
            "response_received",
            "response_parsed",
            "typed_response",
        ],
        error_sha256: str,
    ) -> bytes:
        """Durably terminate an observed source call that cannot be completed.

        Any such abort makes the scientific ledger terminal.  The receipt exists
        so a transport ambiguity cannot be mistaken for an unused reservation.
        """
        self._require_writable()
        key = (reservation.rollout_id, reservation.decision_id)
        if self._source_policy_pending.get(key) != reservation:
            raise LedgerError("source policy reservation is not pending")
        if reservation.ledger_id != self._ledger_id:
            raise LedgerError("source policy abort belongs to another ledger")
        if phase not in {
            "post_unknown",
            "response_received",
            "response_parsed",
            "typed_response",
        }:
            raise ValueError("source policy abort phase is invalid")
        self._require_evidence(error_sha256)
        abort_sequence = len(self._records)
        receipt = self._append_receipt(
            "source_policy_call_aborted",
            {
                "ledger_id": self._ledger_id,
                "ledger_offset": abort_sequence,
                "prior_chain_sha256": self.head_sha256,
                "group_id": reservation.group_id,
                "rollout_id": reservation.rollout_id,
                "decision_id": reservation.decision_id,
                "request_receipt_sha256": _sha256(reservation.receipt),
                "exact_action_key_digest": reservation.exact_action_key_digest,
                "error_sha256": error_sha256,
                "phase": phase,
                "request_sequence": reservation.request_sequence,
                "abort_sequence": abort_sequence,
            },
            evidence_refs=(error_sha256,),
        )
        del self._source_policy_pending[key]
        self._poisoned = True
        return receipt

    @_writer_transaction
    def abort_source_child_before_post(
        self,
        reservation: RecordedActionReservation,
        *,
        rollout_id: str,
        error_sha256: str,
    ) -> bytes:
        """Durably poison a committed child that never obtained a POST ticket."""
        self._require_writable()
        state = self._recorded_action_state(reservation)
        if state["started"] or state["materialized"]:
            raise LedgerError("pre-POST abort follows a recorded model call")
        if not rollout_id:
            raise ValueError("pre-POST abort requires a rollout ID")
        self._require_evidence(error_sha256)
        abort_sequence = len(self._records)
        receipt = self._append_receipt(
            "source_child_pre_post_aborted",
            {
                "ledger_id": self._ledger_id,
                "ledger_offset": abort_sequence,
                "prior_chain_sha256": self.head_sha256,
                "group_id": reservation.group_id,
                "rollout_id": rollout_id,
                "target_id": reservation.target_id,
                "reservation_id": reservation.reservation_id,
                "exact_action_key_digest": reservation.exact_action_key_digest,
                "error_sha256": error_sha256,
                "phase": "before_post",
                "abort_sequence": abort_sequence,
            },
            evidence_refs=(error_sha256,),
        )
        self._poisoned = True
        return receipt

    @_writer_transaction
    def abort_source_rollout_finalization(
        self,
        *,
        group_id: str,
        rollout_id: str,
        error_sha256: str,
    ) -> bytes:
        """Durably terminate a rollout whose observed calls cannot form a source."""
        self._require_writable()
        if not group_id or not rollout_id:
            raise ValueError("source finalization abort requires rollout identifiers")
        if (group_id, rollout_id) in self._source_rollout_completed:
            raise LedgerError("completed source rollout cannot be aborted")
        completed = {
            decision_id
            for completed_rollout, decision_id in self._source_policy_completed
            if completed_rollout == rollout_id
        }
        if not completed:
            raise LedgerError("source finalization abort requires completed policy calls")
        if any(pending_rollout == rollout_id for pending_rollout, _ in self._source_policy_pending):
            raise LedgerError("source finalization abort cannot hide a pending policy call")
        self._require_evidence(error_sha256)
        abort_sequence = len(self._records)
        receipt = self._append_receipt(
            "source_rollout_finalization_aborted",
            {
                "ledger_id": self._ledger_id,
                "ledger_offset": abort_sequence,
                "prior_chain_sha256": self.head_sha256,
                "group_id": group_id,
                "rollout_id": rollout_id,
                "decision_ids": sorted(completed),
                "error_sha256": error_sha256,
                "phase": "source_finalization",
                "abort_sequence": abort_sequence,
            },
            evidence_refs=(error_sha256,),
        )
        self._poisoned = True
        return receipt

    @_writer_transaction
    def record_source_rollout_completed(
        self,
        *,
        group_id: str,
        rollout_id: str,
        source_sha256: str,
        trace_sha256: str,
        reward_evidence_sha256: str,
        stock_sequences_evidence_sha256: str,
        base_model_manifest_sha256: str,
        decision_ids: Sequence[str],
        decision_completion_receipt_sha256s: Sequence[str],
    ) -> SourceRolloutCompletion:
        """Seal one complete source rollout after all of its policy calls."""
        self._require_writable()
        if not group_id or not rollout_id:
            raise ValueError("source rollout identifiers must be nonempty")
        for value, name in (
            (source_sha256, "source_sha256"),
            (trace_sha256, "trace_sha256"),
            (reward_evidence_sha256, "reward_evidence_sha256"),
            (stock_sequences_evidence_sha256, "stock_sequences_evidence_sha256"),
            (base_model_manifest_sha256, "base_model_manifest_sha256"),
        ):
            _require_sha256(value, name)
        evidence_refs = tuple(
            sorted(
                {
                    trace_sha256,
                    reward_evidence_sha256,
                    stock_sequences_evidence_sha256,
                }
            )
        )
        for digest in evidence_refs:
            self._require_evidence(digest)
        ids = tuple(decision_ids)
        receipt_hashes = tuple(decision_completion_receipt_sha256s)
        if not ids or len(ids) != len(receipt_hashes) or len(set(ids)) != len(ids):
            raise ValueError("source rollout decision roster must be nonempty and unique")
        if any(not decision_id for decision_id in ids):
            raise ValueError("source rollout decision IDs must be nonempty")
        if any(
            _require_sha256(digest, "decision completion receipt sha256") != digest
            for digest in receipt_hashes
        ):
            raise AssertionError("unreachable")
        expected = {
            decision_id: self._source_policy_completed.get((rollout_id, decision_id))
            for decision_id in ids
        }
        if any(
            expected[decision_id] != digest
            for decision_id, digest in zip(ids, receipt_hashes, strict=True)
        ):
            raise LedgerError("source rollout names an unanchored policy completion")
        completed_for_rollout = {
            decision_id
            for (completed_rollout, decision_id) in self._source_policy_completed
            if completed_rollout == rollout_id
        }
        if completed_for_rollout != set(ids):
            raise LedgerError("source rollout decision roster is incomplete")
        completed_receipts = [
            self._receipts.get(("source_policy_call_completed", digest))
            for digest in receipt_hashes
        ]
        if any(
            receipt is None or receipt.get("group_id") != group_id for receipt in completed_receipts
        ):
            raise LedgerError("source rollout policy completions cross groups")
        request_sequences = [
            cast(dict[str, Any], receipt)["request_sequence"] for receipt in completed_receipts
        ]
        if request_sequences != sorted(request_sequences):
            raise LedgerError("source rollout decision roster changed request order")
        key = (group_id, rollout_id)
        if key in self._source_rollout_completed:
            raise LedgerError("source rollout is already completed")
        completion_sequence = len(self._records)
        receipt = self._append_receipt(
            "source_rollout_completed",
            {
                "ledger_id": self._ledger_id,
                "ledger_offset": completion_sequence,
                "prior_chain_sha256": self.head_sha256,
                "group_id": group_id,
                "rollout_id": rollout_id,
                "source_sha256": source_sha256,
                "trace_sha256": trace_sha256,
                "reward_evidence_sha256": reward_evidence_sha256,
                "stock_sequences_evidence_sha256": stock_sequences_evidence_sha256,
                "base_model_manifest_sha256": base_model_manifest_sha256,
                "decision_ids": list(ids),
                "decision_completion_receipt_sha256s": list(receipt_hashes),
                "completion_sequence": completion_sequence,
            },
            evidence_refs=evidence_refs,
        )
        completion = SourceRolloutCompletion(
            self._ledger_id,
            group_id,
            rollout_id,
            source_sha256,
            receipt,
        )
        self._source_rollout_completed[key] = completion
        return completion

    @_writer_transaction
    def record_branch_target_roster(self, roster_bytes: bytes) -> bytes:
        """Freeze the complete source denominator and eligible target set before replay."""
        self._require_writable()
        roster = _strict_canonical_object(roster_bytes, "Stage D branch target roster")
        expected_fields = {
            "schema_version",
            "domain",
            "planned_source_count",
            "completed_source_count",
            "eligible_source_count",
            "ineligible_source_count",
            "minimum_eligible_sources",
            "eligibility_passed",
            "source_sha256s",
            "targets",
            "excluded_targets",
        }
        if (
            set(roster) != expected_fields
            or roster.get("schema_version") != 2
            or roster.get("domain") != "redco-stage-d-branch-target-roster-v2"
        ):
            raise ValueError("unsupported Stage D branch target roster")
        if self._branch_target_roster_sha256 is not None:
            raise LedgerError("branch target roster is already frozen")
        if (
            self._candidate_slots
            or self._pending_candidates
            or self._executions
            or self._pending_executions
            or self._branch_artifacts
        ):
            raise LedgerError("branch target roster must precede scientific activity")
        planned = roster.get("planned_source_count")
        completed = roster.get("completed_source_count")
        eligible = roster.get("eligible_source_count")
        ineligible = roster.get("ineligible_source_count")
        minimum = roster.get("minimum_eligible_sources")
        passed = roster.get("eligibility_passed")
        if (
            type(planned) is not int
            or planned < 1
            or type(completed) is not int
            or completed != planned
            or completed != len(self._source_rollout_completed)
            or type(eligible) is not int
            or eligible < 0
            or type(ineligible) is not int
            or ineligible != completed - eligible
            or type(minimum) is not int
            or minimum < 1
            or minimum > planned
            or type(passed) is not bool
            or passed is not (eligible >= minimum)
        ):
            raise ValueError("branch target roster denominator is invalid")
        source_sha256s = roster.get("source_sha256s")
        expected_sources = self.completed_source_sha256s
        if (
            not isinstance(source_sha256s, list)
            or source_sha256s != sorted(set(source_sha256s))
            or tuple(source_sha256s) != expected_sources
        ):
            raise ValueError("branch target roster differs from completed sources")
        targets = roster.get("targets")
        excluded_targets = roster.get("excluded_targets")
        if not isinstance(targets, list) or not isinstance(excluded_targets, list):
            raise ValueError("branch target roster target lists must be lists")
        normalized_targets = tuple(self._validate_roster_target(item) for item in targets)
        normalized_excluded = tuple(
            self._validate_roster_target(item, excluded=True)
            for item in excluded_targets
        )
        target_keys = {(item["group_id"], item["target_id"]) for item in normalized_targets}
        excluded_keys = {
            (item["group_id"], item["target_id"]) for item in normalized_excluded
        }
        if (
            len(target_keys) != len(normalized_targets)
            or len(excluded_keys) != len(normalized_excluded)
            or target_keys & excluded_keys
            or target_keys | excluded_keys != set(self._commitments)
        ):
            raise ValueError("branch target roster differs from committed targets")
        active_source_sha256s = {
            item["source_sha256"] for item in normalized_targets
        }
        excluded_source_sha256s = {
            item["source_sha256"] for item in normalized_excluded
        }
        # Eligibility and target selection are distinct; the support gate enforces
        # target presence for every source counted as a scientific success.
        if (
            len(active_source_sha256s) > eligible
            or active_source_sha256s & excluded_source_sha256s
            or len(excluded_source_sha256s) > ineligible
        ):
            raise ValueError("eligible-source count differs from target roster")
        digest = self.put_evidence(roster_bytes)
        offset = len(self._records)
        receipt = self._append_receipt(
            "branch_target_roster",
            {
                "ledger_id": self._ledger_id,
                "ledger_offset": offset,
                "prior_chain_sha256": self.head_sha256,
                "roster_sha256": digest,
                "planned_source_count": planned,
                "completed_source_count": completed,
                "eligible_source_count": eligible,
                "ineligible_source_count": ineligible,
                "minimum_eligible_sources": minimum,
                "eligibility_passed": passed,
                "source_sha256s": source_sha256s,
                "targets": list(normalized_targets),
                "excluded_targets": list(normalized_excluded),
                "roster_sequence": offset,
            },
            evidence_refs=(digest,),
        )
        self._branch_target_roster_sha256 = digest
        self._branch_target_keys = target_keys
        return receipt

    @_writer_transaction
    def commit_pre_action_and_reserve(
        self,
        *,
        group_id: str,
        rollout_id: str,
        target_roster: Sequence[str],
        target_ordinal: int,
        target_id: str,
        target_address: PolicyEventAddress,
        pre_action_snapshot_sha256: str,
        recorded_action_key: ExactActionKey,
        branch_count: int,
        continuation_replicates: int,
        failure_reward: float,
    ) -> RecordedActionReservation:
        self._require_writable()
        key = _group_key(group_id, target_id)
        if key in self._commitments:
            raise LedgerError("group target already has a commitment")
        roster = tuple(target_roster)
        if not roster or target_ordinal < 0 or target_ordinal >= len(roster):
            raise ValueError("target roster and ordinal disagree")
        if roster[target_ordinal] != target_id or len(set(roster)) != len(roster):
            raise ValueError("target roster must be unique and contain target_id at ordinal")
        _require_sha256(pre_action_snapshot_sha256, "pre_action_snapshot_sha256")
        self._require_evidence(pre_action_snapshot_sha256)
        _exact_int(branch_count, "branch_count", minimum=2)
        _exact_int(continuation_replicates, "continuation_replicates", minimum=1)
        failure_reward = _finite_float(failure_reward, "failure_reward")
        commitment_offset = len(self._records)
        reservation_offset = commitment_offset + 1
        receipt = self._append_receipt(
            "pre_action_group_commitment",
            {
                "ledger_id": self._ledger_id,
                "ledger_offset": commitment_offset,
                "prior_chain_sha256": self.head_sha256,
                "phase": "pre_action",
                "group_id": group_id,
                "rollout_id": rollout_id,
                "target_roster": list(roster),
                "target_ordinal": target_ordinal,
                "target_id": target_id,
                "target_address": _address_payload(target_address),
                "pre_action_snapshot_sha256": pre_action_snapshot_sha256,
                "behavior_law_sha256": behavior_law_digest(recorded_action_key),
                "recorded_action_seed": recorded_action_key.sampler.seed,
                "branch_count": branch_count,
                "continuation_replicates": continuation_replicates,
                "failure_reward": failure_reward,
                "master_seed_sha256": _sha256(self._master_seed.encode("utf-8")),
                "commitment_sequence": commitment_offset,
                "action_reservation_sequence": reservation_offset,
            },
            evidence_refs=(pre_action_snapshot_sha256,),
        )
        reservation_id = self._fresh_id("recorded-action")
        try:
            self._append_event(
                "action_reservation",
                {
                    "group_id": group_id,
                    "target_id": target_id,
                    "commitment_receipt_sha256": _sha256(receipt),
                    "recorded_action_seed": recorded_action_key.sampler.seed,
                    "exact_action_key_digest": recorded_action_key.digest,
                    "request_sha256": recorded_action_key.request_sha256,
                    "reservation_id": reservation_id,
                },
                evidence_refs=(),
            )
        except BaseException:
            self._poisoned = True
            raise
        self._commitments[key] = json.loads(receipt)
        reservation = RecordedActionReservation(
            self._ledger_id,
            group_id,
            target_id,
            reservation_id,
            recorded_action_key.sampler.seed,
            recorded_action_key.digest,
            recorded_action_key.request_sha256,
            receipt,
        )
        self._recorded_reservations[key] = {
            "reservation": reservation,
            "started": False,
            "materialized": False,
        }
        return reservation

    @_writer_transaction
    def resume_recorded_action_reservation(
        self,
        *,
        group_id: str,
        target_id: str,
    ) -> RecordedActionReservation:
        key = self._require_committed(group_id, target_id)
        state = self._recorded_reservations[key]
        if state["started"] or state["materialized"]:
            raise LedgerError("recorded action reservation is no longer resumable")
        return cast(RecordedActionReservation, state["reservation"])

    @_writer_transaction
    def mark_recorded_action_model_call_started(
        self,
        reservation: RecordedActionReservation,
        *,
        request_sha256: str,
    ) -> str:
        state = self._recorded_action_state(reservation)
        self._require_evidence(request_sha256)
        if request_sha256 != reservation.request_sha256:
            raise ValueError("recorded action request differs from its exact pre-action key")
        if state["started"]:
            raise LedgerError("recorded action model call already started")
        call_id = self._fresh_id("recorded-action-call")
        self._append_event(
            "model_call_started",
            {
                "attempt_kind": "recorded_action",
                "attempt_id": reservation.reservation_id,
                "call_id": call_id,
                "group_id": reservation.group_id,
                "target_id": reservation.target_id,
            },
            evidence_refs=(request_sha256,),
        )
        state["started"] = True
        state["call_id"] = call_id
        return call_id

    @_writer_transaction
    def complete_recorded_action(
        self,
        reservation: RecordedActionReservation,
        *,
        action: BehaviorAction,
        response_sha256: str,
    ) -> None:
        state = self._recorded_action_state(reservation)
        self._require_evidence(response_sha256)
        if not state["started"] or state["materialized"]:
            raise LedgerError("recorded action is not exactly once and in flight")
        if (
            action.key.digest != reservation.exact_action_key_digest
            or action.key.sampler.seed != reservation.action_seed
        ):
            raise ValueError("recorded action differs from its pre-action reservation")
        self._append_event(
            "model_call_completed",
            {
                "attempt_kind": "recorded_action",
                "attempt_id": reservation.reservation_id,
                "call_id": state["call_id"],
                "action_digest": action.digest,
                "prompt_tokens": action.prompt_tokens,
                "completion_tokens": action.completion_tokens,
            },
            evidence_refs=(response_sha256,),
        )
        try:
            self._append_event(
                "recorded_action_materialized",
                {
                    "group_id": reservation.group_id,
                    "target_id": reservation.target_id,
                    "reservation_id": reservation.reservation_id,
                    "call_id": state["call_id"],
                    "action_digest": action.digest,
                    "exact_action_key_digest": action.key.digest,
                },
                evidence_refs=(response_sha256,),
            )
        except BaseException:
            self._poisoned = True
            raise
        key = (reservation.group_id, reservation.target_id)
        state["materialized"] = True
        self._recorded_action_digests[key] = action.digest

    @_writer_transaction
    def freeze_correspondence(
        self,
        *,
        group_id: str,
        target_id: str,
        recorded_action: BehaviorAction,
        matched_addresses: Sequence[PolicyEventAddress],
        evidence_sha256: str,
    ) -> bytes:
        self._require_evidence(evidence_sha256)
        key = self._require_committed(group_id, target_id)
        if key in self._correspondence:
            raise LedgerError("correspondence is already frozen")
        if self._candidate_slots_for(key) or self._execution_keys_for(key):
            raise LedgerError("correspondence must be frozen before candidate activity")
        reservation_state = self._recorded_reservations[key]
        if not reservation_state["materialized"]:
            raise LedgerError("recorded action must materialize after its reservation")
        if self._recorded_action_digests[key] != recorded_action.digest:
            raise ValueError("recorded action changed after commitment")
        commitment = self._commitments[key]
        ordered_addresses = sorted(
            matched_addresses,
            key=lambda item: canonical_json(_address_payload(item)),
        )
        address_keys = [canonical_json(_address_payload(item)) for item in ordered_addresses]
        if len(set(address_keys)) != len(address_keys):
            raise ValueError("matched correspondence addresses must be unique")
        receipt = self._append_receipt(
            "seed_correspondence_map",
            {
                "group_id": group_id,
                "target_id": target_id,
                "pre_action_snapshot_sha256": commitment["pre_action_snapshot_sha256"],
                "recorded_action_digest": recorded_action.digest,
                "matched_addresses": [_address_payload(address) for address in ordered_addresses],
            },
            evidence_refs=(evidence_sha256,),
        )
        self._correspondence.add(key)
        return receipt

    @_writer_transaction
    def record_reconstruction_qa(
        self,
        *,
        group_id: str,
        target_id: str,
        recorded_action: BehaviorAction,
        passed: bool,
        report_sha256: str,
        actual_cost: ActualEvaluationCost,
    ) -> bytes:
        self._require_reconstruction_qa_ready(
            group_id=group_id,
            target_id=target_id,
            recorded_action=recorded_action,
            actual_cost=actual_cost,
        )
        key = (group_id, target_id)
        self._require_evidence(report_sha256)
        if type(passed) is not bool:
            raise ValueError("passed must be bool")
        commitment = self._commitments[key]
        receipt = self._append_receipt(
            "reconstruction_qa",
            {
                "group_id": group_id,
                "target_id": target_id,
                "pre_action_snapshot_sha256": commitment["pre_action_snapshot_sha256"],
                "recorded_action_digest": recorded_action.digest,
                "passed": passed,
                "report_sha256": report_sha256,
                "actual_cost": _actual_cost_payload(actual_cost),
            },
            evidence_refs=(report_sha256,),
        )
        self._qa.add(key)
        self._qa_passed[key] = passed
        self._qa_receipt_sha256s[key] = _sha256(receipt)
        return receipt

    @_writer_transaction
    def seal_reconstruction_qa_barrier(self) -> bytes:
        """Prove every frozen target passed model-free QA before any science begins."""
        self._require_writable()
        if self._branch_target_roster_sha256 is None or not self._branch_target_keys:
            raise LedgerError("reconstruction QA barrier requires a frozen target roster")
        if self._reconstruction_qa_barrier_sha256 is not None:
            raise LedgerError("reconstruction QA barrier is already sealed")
        if self._candidate_slots or self._pending_candidates or self._executions or (
            self._pending_executions
        ):
            raise LedgerError("reconstruction QA barrier must precede scientific activity")
        if set(self._qa_passed) != self._branch_target_keys or not all(
            self._qa_passed.values()
        ):
            raise LedgerError("every frozen target must pass reconstruction QA")
        qa_receipts = [
            {
                "group_id": group_id,
                "target_id": target_id,
                "qa_receipt_sha256": self._qa_receipt_sha256s[(group_id, target_id)],
            }
            for group_id, target_id in sorted(self._branch_target_keys)
        ]
        offset = len(self._records)
        receipt = self._append_receipt(
            "reconstruction_qa_barrier",
            {
                "ledger_id": self._ledger_id,
                "ledger_offset": offset,
                "prior_chain_sha256": self.head_sha256,
                "branch_target_roster_sha256": self._branch_target_roster_sha256,
                "qa_receipts": qa_receipts,
                "target_count": len(qa_receipts),
                "all_passed": True,
                "scientific_model_calls_before_barrier": 0,
                "barrier_sequence": offset,
            },
            evidence_refs=(),
        )
        self._reconstruction_qa_barrier_sha256 = _sha256(receipt)
        return receipt

    @_writer_transaction
    def begin_candidate_attempt(
        self,
        *,
        group_id: str,
        target_id: str,
        action_slot: int,
    ) -> CandidateAttempt:
        key = self._require_scientific_ready(group_id, target_id)
        commitment = self._commitments[key]
        _exact_int(action_slot, "action_slot", minimum=1)
        if action_slot >= commitment["branch_count"]:
            raise ValueError("candidate slot is outside frozen K")
        slot_key = (*key, action_slot)
        if slot_key in self._candidate_slots or slot_key in self._pending_candidates:
            raise LedgerError("candidate slot already has an attempt or outcome")
        attempt_ordinal = self._candidate_zero_call_failures.get(slot_key, 0)
        if attempt_ordinal > 1:
            raise LedgerError("candidate slot exhausted its one bounded repair")
        action_seed = EventSeedScheduler(
            self._master_seed,
            commitment["rollout_id"],
            target_id,
            1,
        ).action_seed(action_slot=action_slot)
        attempt_id = self._fresh_id("candidate")
        attempt = CandidateAttempt(
            self._ledger_id,
            group_id,
            target_id,
            action_slot,
            action_seed,
            attempt_ordinal,
            attempt_id,
        )
        self._append_event(
            "candidate_attempt",
            {
                "group_id": group_id,
                "target_id": target_id,
                "action_slot": action_slot,
                "action_seed": action_seed,
                "attempt_ordinal": attempt_ordinal,
                "attempt_id": attempt_id,
            },
            evidence_refs=(),
        )
        self._pending_candidates[slot_key] = {"attempt": attempt, "started": False}
        return attempt

    @_writer_transaction
    def mark_candidate_model_call_started(
        self,
        attempt: CandidateAttempt,
        *,
        request_sha256: str,
    ) -> str:
        state = self._candidate_state(attempt)
        self._require_evidence(request_sha256)
        if state["started"]:
            raise LedgerError("candidate model call already started")
        call_id = self._fresh_id("candidate-call")
        self._append_event(
            "model_call_started",
            {
                "attempt_kind": "candidate",
                "attempt_id": attempt.attempt_id,
                "call_id": call_id,
                "group_id": attempt.group_id,
                "target_id": attempt.target_id,
                "action_slot": attempt.action_slot,
            },
            evidence_refs=(request_sha256,),
        )
        state["started"] = True
        state["call_id"] = call_id
        return call_id

    @_writer_transaction
    def complete_candidate_call(
        self,
        attempt: CandidateAttempt,
        *,
        action: BehaviorAction,
        response_sha256: str,
    ) -> bytes:
        state = self._candidate_state(attempt)
        self._require_evidence(response_sha256)
        if not state["started"] or "completed" in state:
            raise LedgerError("candidate call is not exactly once and in flight")
        if state.get("response_sha256") != response_sha256:
            raise LedgerError("candidate completion lacks its durable raw response witness")
        if action.key.sampler.seed != attempt.action_seed:
            raise ValueError("candidate action seed differs from reserved seed")
        commitment = self._commitments[(attempt.group_id, attempt.target_id)]
        if behavior_law_digest(action.key) != commitment["behavior_law_sha256"]:
            raise ValueError("candidate action changed the frozen behavior law")
        action_evidence_sha256 = self.put_evidence(action.to_bytes())
        try:
            receipt = self._append_receipt(
                "candidate_action_inference",
                {
                    "group_id": attempt.group_id,
                    "target_id": attempt.target_id,
                    "action_slot": attempt.action_slot,
                    "action_seed": attempt.action_seed,
                    "action_digest": action.digest,
                    "action_evidence_sha256": action_evidence_sha256,
                    "behavior_law_sha256": behavior_law_digest(action.key),
                    "selection_policy": "direct_single_sample",
                    "sample_attempts": 1,
                    "rejected_attempts": 0,
                    "inference_call_id": state["call_id"],
                    "prompt_tokens": action.prompt_tokens,
                    "completion_tokens": action.completion_tokens,
                    "response_sha256": response_sha256,
                },
                evidence_refs=tuple(
                    sorted({action_evidence_sha256, response_sha256})
                ),
            )
        except BaseException:
            self._poisoned = True
            raise
        slot_key = (attempt.group_id, attempt.target_id, attempt.action_slot)
        self._candidate_slots[slot_key] = action.digest
        self._candidate_action_sha256s[slot_key] = action_evidence_sha256
        self._candidate_receipts[slot_key] = receipt
        del self._pending_candidates[slot_key]
        return receipt

    @_writer_transaction
    def mark_candidate_response_observed(
        self,
        attempt: CandidateAttempt,
        *,
        response_sha256: str,
    ) -> None:
        """Anchor exact returned bytes before any renderer or action parsing."""
        state = self._candidate_state(attempt)
        self._require_evidence(response_sha256)
        if not state["started"] or state.get("response_sha256") is not None:
            raise LedgerError("candidate response witness is not exactly once and in flight")
        self._append_event(
            "model_call_response_observed",
            {
                "attempt_kind": "candidate",
                "attempt_id": attempt.attempt_id,
                "call_id": state["call_id"],
                "response_sha256": response_sha256,
            },
            evidence_refs=(response_sha256,),
        )
        state["response_sha256"] = response_sha256

    @_writer_transaction
    def record_zero_call_candidate_failure(
        self,
        attempt: CandidateAttempt,
        *,
        reason: str,
        supervisor_evidence_sha256: str,
    ) -> bytes:
        state = self._candidate_state(attempt)
        self._require_evidence(supervisor_evidence_sha256)
        if state["started"]:
            raise LedgerError("zero-call status is impossible after model_call_started")
        if not reason:
            raise ValueError("zero-call reason must be nonempty")
        offset = len(self._records)
        receipt = self._append_receipt(
            "zero_call_infrastructure_failure",
            {
                "ledger_id": self._ledger_id,
                "ledger_offset": offset,
                "prior_chain_sha256": self.head_sha256,
                "group_id": attempt.group_id,
                "target_id": attempt.target_id,
                "action_slot": attempt.action_slot,
                "action_seed": attempt.action_seed,
                "attempt_ordinal": attempt.attempt_ordinal,
                "attempt_id": attempt.attempt_id,
                "attempt_model_calls": 0,
                "attempt_overrides": 0,
                "prior_candidate_completions": len(self._candidate_slots),
                "prior_execution_completions": len(self._executions),
                "repair_sequence": self._zero_call_failure_count,
                "successor_permitted": self._zero_call_failure_count == 0,
                "reason": reason,
            },
            evidence_refs=(supervisor_evidence_sha256,),
        )
        slot_key = (attempt.group_id, attempt.target_id, attempt.action_slot)
        self._candidate_zero_call_failures[slot_key] = attempt.attempt_ordinal + 1
        self._zero_call_failure_count += 1
        del self._pending_candidates[slot_key]
        return receipt

    @_writer_transaction
    def begin_execution(
        self,
        *,
        group_id: str,
        target_id: str,
        arm_id: str,
        action: BehaviorAction,
        continuation_replicate: int,
    ) -> ExecutionAttempt:
        key = self._require_scientific_ready(group_id, target_id)
        commitment = self._commitments[key]
        if not arm_id.startswith("arm-"):
            raise ValueError("arm_id must use the canonical arm-N form")
        try:
            slot = int(arm_id.removeprefix("arm-"))
        except ValueError as error:
            raise ValueError("arm_id must use the canonical arm-N form") from error
        if slot < 0 or slot >= commitment["branch_count"]:
            raise ValueError("arm is outside frozen K")
        expected_digest = (
            self._recorded_action_digests[key]
            if slot == 0
            else self._candidate_slots.get((*key, slot))
        )
        if expected_digest != action.digest:
            raise LedgerError("execution action is not the durable arm action")
        _exact_int(continuation_replicate, "continuation_replicate", minimum=1)
        if continuation_replicate > commitment["continuation_replicates"]:
            raise ValueError("continuation replicate exceeds frozen c")
        execution_key = (*key, arm_id, continuation_replicate)
        if execution_key in self._executions or execution_key in self._pending_executions:
            raise LedgerError("arm replicate already has an attempt or outcome")
        attempt_ordinal = self._execution_zero_call_failures.get(execution_key, 0)
        if attempt_ordinal > 1:
            raise LedgerError("execution exhausted its one bounded repair")
        attempt = ExecutionAttempt(
            self._ledger_id,
            group_id,
            target_id,
            arm_id,
            action.digest,
            continuation_replicate,
            attempt_ordinal,
            self._fresh_id("execution"),
        )
        self._append_event(
            "execution_attempt",
            {
                "group_id": group_id,
                "target_id": target_id,
                "arm_id": arm_id,
                "action_digest": action.digest,
                "continuation_replicate": continuation_replicate,
                "attempt_ordinal": attempt_ordinal,
                "attempt_id": attempt.attempt_id,
            },
            evidence_refs=(),
        )
        self._pending_executions[execution_key] = {
            "attempt": attempt,
            "action": action,
            "calls": [],
            "in_flight": {},
            "context_sha256": None,
            "dispatched": False,
            "overrides": {},
            "responses": {},
        }
        return attempt

    @_writer_transaction
    def record_zero_call_execution_failure(
        self,
        attempt: ExecutionAttempt,
        *,
        reason: str,
        supervisor_evidence_sha256: str,
    ) -> bytes:
        """Close a dispatched execution only when no scientific action was observed."""
        state = self._execution_state(attempt)
        self._require_evidence(supervisor_evidence_sha256)
        if state["context_sha256"] is None or not state["dispatched"]:
            raise LedgerError("zero-call execution failure requires durable dispatch")
        if state["calls"] or state["in_flight"]:
            raise LedgerError("zero-call execution failure follows scientific activity")
        discarded_override_ids = sorted(
            item["ticket"].override_id
            for item in state["overrides"].values()
            if not item["delivered"]
        )
        if len(discarded_override_ids) != len(state["overrides"]):
            raise LedgerError("zero-call execution failure follows delivered scientific activity")
        if not reason:
            raise ValueError("zero-call reason must be nonempty")
        offset = len(self._records)
        receipt = self._append_receipt(
            "zero_call_execution_failure",
            {
                "ledger_id": self._ledger_id,
                "ledger_offset": offset,
                "prior_chain_sha256": self.head_sha256,
                "group_id": attempt.group_id,
                "target_id": attempt.target_id,
                "arm_id": attempt.arm_id,
                "action_digest": attempt.action_digest,
                "continuation_replicate": attempt.continuation_replicate,
                "attempt_ordinal": attempt.attempt_ordinal,
                "attempt_id": attempt.attempt_id,
                "attempt_model_calls": 0,
                "attempt_overrides": len(discarded_override_ids),
                "discarded_override_ids": discarded_override_ids,
                "prior_candidate_completions": len(self._candidate_slots),
                "prior_execution_completions": len(self._executions),
                "repair_sequence": self._zero_call_failure_count,
                "successor_permitted": self._zero_call_failure_count == 0,
                "reason": reason,
            },
            evidence_refs=(supervisor_evidence_sha256,),
        )
        execution_key = (
            attempt.group_id,
            attempt.target_id,
            attempt.arm_id,
            attempt.continuation_replicate,
        )
        self._execution_zero_call_failures[execution_key] = attempt.attempt_ordinal + 1
        self._zero_call_failure_count += 1
        del self._pending_executions[execution_key]
        return receipt

    @_writer_transaction
    def bind_execution_context(
        self,
        attempt: ExecutionAttempt,
        *,
        context_sha256: str,
    ) -> None:
        """Anchor the exact supervisor dispatch context before worker execution."""
        state = self._execution_state(attempt)
        self._require_evidence(context_sha256)
        if state["context_sha256"] is not None:
            raise LedgerError("execution context is already bound")
        if state["calls"] or state["in_flight"]:
            raise LedgerError("execution context must precede downstream model calls")
        self._append_event(
            "execution_context_bound",
            {
                "attempt_id": attempt.attempt_id,
                "group_id": attempt.group_id,
                "target_id": attempt.target_id,
                "arm_id": attempt.arm_id,
                "continuation_replicate": attempt.continuation_replicate,
                "context_sha256": context_sha256,
            },
            evidence_refs=(context_sha256,),
        )
        state["context_sha256"] = context_sha256

    @_writer_transaction
    def mark_execution_dispatched(self, attempt: ExecutionAttempt) -> None:
        """Close the zero-information repair window before sandbox/action dispatch."""
        state = self._execution_state(attempt)
        if state["context_sha256"] is None:
            raise LedgerError("execution context must be bound before dispatch")
        if state["dispatched"]:
            raise LedgerError("execution was already dispatched")
        if state["calls"] or state["in_flight"]:
            raise LedgerError("execution dispatch must precede downstream calls")
        self._append_event(
            "execution_dispatched",
            {
                "attempt_id": attempt.attempt_id,
                "group_id": attempt.group_id,
                "target_id": attempt.target_id,
                "arm_id": attempt.arm_id,
                "continuation_replicate": attempt.continuation_replicate,
            },
            evidence_refs=(),
        )
        state["dispatched"] = True

    @_writer_transaction
    def mark_execution_model_call_started(
        self,
        attempt: ExecutionAttempt,
        *,
        address: PolicyEventAddress,
        scheduled_seed: ScheduledSeed,
        request_sha256: str,
    ) -> ModelCallAttempt:
        state = self._execution_state(attempt)
        self._require_evidence(request_sha256)
        if state["context_sha256"] is None:
            raise LedgerError("execution context must be bound before model dispatch")
        if not state["dispatched"]:
            raise LedgerError("execution must be dispatched before a model call")
        if _execution_address_used(state, address):
            raise LedgerError("one execution may not reuse a scientific event address")
        if scheduled_seed.address != address:
            raise ValueError("scheduled seed is bound to a different event address")
        call = ModelCallAttempt(
            attempt.attempt_id,
            self._fresh_id("execution-call"),
            address,
            scheduled_seed,
        )
        self._append_event(
            "model_call_started",
            {
                "attempt_kind": "execution",
                "attempt_id": attempt.attempt_id,
                "call_id": call.call_id,
                "group_id": attempt.group_id,
                "target_id": attempt.target_id,
                "arm_id": attempt.arm_id,
                "continuation_replicate": attempt.continuation_replicate,
                "address": _address_payload(address),
                "seed": scheduled_seed.seed,
                "cache_salt": scheduled_seed.cache_salt,
                "coupling_mode": scheduled_seed.coupling_mode.value,
            },
            evidence_refs=(request_sha256,),
        )
        state["in_flight"][address] = call
        return call

    @_writer_transaction
    def commit_execution_override(
        self,
        attempt: ExecutionAttempt,
        *,
        address: PolicyEventAddress,
        action_digest: str,
        disposition: Literal["reuse", "inject"],
        request_sha256: str,
        response_content_sha256: str,
        prompt_tokens: int,
        completion_tokens: int,
        counts_toward_logical_cost: bool,
    ) -> ReplayOverrideTicket:
        """Commit a zero-POST action before exposing it to the replay runtime."""
        state = self._execution_state(attempt)
        self._require_evidence(request_sha256)
        self._require_evidence(response_content_sha256)
        _require_sha256(action_digest, "action_digest")
        if disposition not in {"reuse", "inject"}:
            raise ValueError("override disposition must be reuse or inject")
        prompt_tokens = _exact_int(prompt_tokens, "prompt_tokens")
        completion_tokens = _exact_int(completion_tokens, "completion_tokens", minimum=1)
        if type(counts_toward_logical_cost) is not bool:
            raise ValueError("counts_toward_logical_cost must be bool")
        if disposition == "inject" and counts_toward_logical_cost:
            raise ValueError("injected target action is already counted exactly once")
        if state["context_sha256"] is None or not state["dispatched"]:
            raise LedgerError("execution must be bound and dispatched before an override")
        if _execution_address_used(state, address):
            raise LedgerError("one execution may not reuse a scientific event address")
        if disposition == "inject" and action_digest != attempt.action_digest:
            raise LedgerError("injected override is not the durable arm action")
        commitment = self._commitments[(attempt.group_id, attempt.target_id)]
        if disposition == "inject" and (
            _address_payload(address) != commitment["target_address"]
        ):
            raise LedgerError("injected override is not at the committed target address")
        if (
            disposition == "reuse"
            and self._branch_target_roster_sha256 is not None
            and action_digest != self._source_action_digest(attempt, address)
        ):
            raise LedgerError("reused override differs from the frozen source action")
        ticket = ReplayOverrideTicket(
            attempt.attempt_id,
            self._fresh_id("execution-override"),
            address,
            action_digest,
            disposition,
            response_content_sha256,
            prompt_tokens,
            completion_tokens,
            counts_toward_logical_cost,
        )
        self._append_event(
            "execution_override_committed",
            {
                "attempt_id": attempt.attempt_id,
                "override_id": ticket.override_id,
                "group_id": attempt.group_id,
                "target_id": attempt.target_id,
                "arm_id": attempt.arm_id,
                "continuation_replicate": attempt.continuation_replicate,
                "address": _address_payload(address),
                "action_digest": action_digest,
                "disposition": disposition,
                "request_sha256": request_sha256,
                "response_content_sha256": response_content_sha256,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "counts_toward_logical_cost": counts_toward_logical_cost,
            },
            evidence_refs=tuple(sorted((request_sha256, response_content_sha256))),
        )
        state["overrides"][address] = {"ticket": ticket, "delivered": False}
        return ticket

    @_writer_transaction
    def mark_execution_override_delivered(
        self,
        attempt: ExecutionAttempt,
        ticket: ReplayOverrideTicket,
        *,
        typed_response_sha256: str,
    ) -> None:
        """Acknowledge that the ordinary typed response path consumed an override."""
        state = self._execution_state(attempt)
        self._require_evidence(typed_response_sha256)
        if ticket.execution_attempt_id != attempt.attempt_id:
            raise LedgerError("override ticket belongs to another execution")
        override = state["overrides"].get(ticket.address)
        if override is None or override["ticket"] != ticket:
            raise LedgerError("override ticket is not durably committed")
        if override["delivered"]:
            raise LedgerError("override was already delivered")
        self._append_event(
            "execution_override_delivered",
            {
                "attempt_id": attempt.attempt_id,
                "override_id": ticket.override_id,
                "typed_response_sha256": typed_response_sha256,
            },
            evidence_refs=(typed_response_sha256,),
        )
        override["delivered"] = True

    @_writer_transaction
    def complete_execution_model_call(
        self,
        attempt: ExecutionAttempt,
        call: ModelCallAttempt,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        response_sha256: str,
    ) -> None:
        state = self._execution_state(attempt)
        self._require_evidence(response_sha256)
        if state["in_flight"].get(call.address) != call:
            raise LedgerError("execution completion does not match the in-flight call")
        if call.call_id not in state.setdefault("responses", {}):
            raise LedgerError("execution completion lacks its durable raw response witness")
        prompt_tokens = _exact_int(prompt_tokens, "prompt_tokens")
        completion_tokens = _exact_int(completion_tokens, "completion_tokens")
        self._append_event(
            "model_call_completed",
            {
                "attempt_kind": "execution",
                "attempt_id": attempt.attempt_id,
                "call_id": call.call_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            evidence_refs=(response_sha256,),
        )
        state["calls"].append(
            {
                "call_id": call.call_id,
                "address": call.address,
                "scheduled_seed": call.scheduled_seed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )
        del state["in_flight"][call.address]

    @_writer_transaction
    def mark_execution_response_observed(
        self,
        attempt: ExecutionAttempt,
        call: ModelCallAttempt,
        *,
        response_sha256: str,
    ) -> None:
        """Anchor provider bytes before typed response parsing."""
        state = self._execution_state(attempt)
        self._require_evidence(response_sha256)
        if state["in_flight"].get(call.address) != call:
            raise LedgerError("execution response does not match the in-flight call")
        responses = state.setdefault("responses", {})
        if call.call_id in responses:
            raise LedgerError("execution response was observed twice")
        self._append_event(
            "model_call_response_observed",
            {
                "attempt_kind": "execution",
                "attempt_id": attempt.attempt_id,
                "call_id": call.call_id,
                "response_sha256": response_sha256,
            },
            evidence_refs=(response_sha256,),
        )
        responses[call.call_id] = response_sha256

    @_writer_transaction
    def finish_execution(
        self,
        attempt: ExecutionAttempt,
        *,
        outcome_kind: OutcomeKind,
        scored_reward: float,
        scorer_evidence_sha256: str,
        latency_seconds: float,
        dollars: float,
        judge_calls: int,
        cpu_seconds: float,
        gpu_seconds: float,
        wall_seconds: float,
        storage_bytes: int,
    ) -> bytes:
        state = self._execution_state(attempt)
        self._require_evidence(scorer_evidence_sha256)
        if state["context_sha256"] is None:
            raise LedgerError("execution context must be bound before its outcome")
        if not state["dispatched"]:
            raise LedgerError("execution must be dispatched before its outcome")
        if state["in_flight"]:
            raise LedgerError("cannot finish with a model call in flight")
        if any(not override["delivered"] for override in state["overrides"].values()):
            raise LedgerError("cannot finish with an undelivered replay override")
        if not isinstance(outcome_kind, OutcomeKind):
            raise ValueError("outcome_kind must be OutcomeKind")
        calls = state["calls"]
        replayed = [
            override["ticket"] for override in state["overrides"].values()
        ]
        if self._branch_target_roster_sha256 is not None:
            injections = [ticket for ticket in replayed if ticket.disposition == "inject"]
            commitment = self._commitments[(attempt.group_id, attempt.target_id)]
            if (
                len(injections) != 1
                or _address_payload(injections[0].address) != commitment["target_address"]
                or injections[0].action_digest != attempt.action_digest
            ):
                raise LedgerError("execution requires exactly one committed target injection")
        logical_replay_tokens = sum(
            ticket.completion_tokens
            for ticket in replayed
            if ticket.counts_toward_logical_cost
        )
        if outcome_kind is OutcomeKind.SUCCESS and not calls and logical_replay_tokens == 0:
            raise ValueError("zero-call success must be terminal_without_downstream")
        if outcome_kind is OutcomeKind.TERMINAL_WITHOUT_DOWNSTREAM and (
            calls or logical_replay_tokens
        ):
            raise ValueError("terminal_without_downstream cannot contain model calls")
        commitment = self._commitments[(attempt.group_id, attempt.target_id)]
        failure_kinds = {
            OutcomeKind.MALFORMED_ACTION,
            OutcomeKind.RUNTIME_EXCEPTION,
            OutcomeKind.TIMEOUT,
            OutcomeKind.RESOURCE_LIMIT,
        }
        reward = (
            commitment["failure_reward"]
            if outcome_kind in failure_kinds
            else _finite_float(scored_reward, "scored_reward")
        )
        action: BehaviorAction = state["action"]
        if action.parse_status == "malformed" and outcome_kind is not OutcomeKind.MALFORMED_ACTION:
            raise ValueError("malformed action must retain a malformed outcome")
        downstream_tokens = sum(call["completion_tokens"] for call in calls)
        logical = LogicalDeploymentCost(
            output_tokens=(
                action.completion_tokens + downstream_tokens + logical_replay_tokens
            ),
            latency_seconds=_finite_nonnegative(latency_seconds, "latency_seconds"),
            dollars=_finite_nonnegative(dollars, "dollars"),
        )
        actual = ActualEvaluationCost(
            generated_tokens=downstream_tokens,
            judge_calls=_exact_int(judge_calls, "judge_calls"),
            cpu_seconds=_finite_nonnegative(cpu_seconds, "cpu_seconds"),
            gpu_seconds=_finite_nonnegative(gpu_seconds, "gpu_seconds"),
            wall_seconds=_finite_nonnegative(wall_seconds, "wall_seconds"),
            storage_bytes=_exact_int(storage_bytes, "storage_bytes"),
        )
        receipt = self._append_receipt(
            "scientific_arm_execution",
            {
                "group_id": attempt.group_id,
                "target_id": attempt.target_id,
                "arm_id": attempt.arm_id,
                "action_digest": attempt.action_digest,
                "continuation_replicate": attempt.continuation_replicate,
                "execution_id": attempt.attempt_id,
                "outcome_kind": outcome_kind.value,
                "reward": reward,
                "calls": [
                    {
                        "call_id": call["call_id"],
                        "address": _address_payload(call["address"]),
                        "seed": call["scheduled_seed"].seed,
                        "coupling_mode": call["scheduled_seed"].coupling_mode.value,
                        "prompt_tokens": call["prompt_tokens"],
                        "completion_tokens": call["completion_tokens"],
                        "disposition": "generated",
                    }
                    for call in calls
                ],
                "replayed_calls": [
                    {
                        "override_id": ticket.override_id,
                        "address": _address_payload(ticket.address),
                        "action_digest": ticket.action_digest,
                        "disposition": ticket.disposition,
                        "prompt_tokens": ticket.prompt_tokens,
                        "completion_tokens": ticket.completion_tokens,
                        "counts_toward_logical_cost": (
                            ticket.counts_toward_logical_cost
                        ),
                    }
                    for ticket in replayed
                ],
                "logical_cost": {
                    "output_tokens": logical.output_tokens,
                    "latency_seconds": logical.latency_seconds,
                    "dollars": logical.dollars,
                },
                "actual_non_token_cost": {
                    "judge_calls": actual.judge_calls,
                    "cpu_seconds": actual.cpu_seconds,
                    "gpu_seconds": actual.gpu_seconds,
                    "wall_seconds": actual.wall_seconds,
                    "storage_bytes": actual.storage_bytes,
                },
            },
            evidence_refs=(scorer_evidence_sha256,),
        )
        execution_key = (
            attempt.group_id,
            attempt.target_id,
            attempt.arm_id,
            attempt.continuation_replicate,
        )
        self._executions.add(execution_key)
        self._execution_receipts[execution_key] = receipt
        del self._pending_executions[execution_key]
        return receipt

    @_writer_transaction
    def record_branch_group_artifact_completed(
        self,
        *,
        group_id: str,
        target_id: str,
        artifact_sha256: str,
        training_batch_identity: str,
    ) -> bytes:
        """Anchor a fully materialized K-by-c artifact after every primitive receipt."""
        key = self._require_scientific_ready(group_id, target_id)
        self._require_evidence(artifact_sha256)
        _require_sha256(training_batch_identity, "training_batch_identity")
        if key in self._branch_artifacts:
            raise LedgerError("branch group artifact is already completed")
        commitment = self._commitments[key]
        expected_candidate_slots = {
            (*key, slot) for slot in range(1, commitment["branch_count"])
        }
        actual_candidate_slots = {
            slot for slot in self._candidate_slots if slot[:2] == key
        }
        expected_executions = {
            (*key, f"arm-{slot}", replicate)
            for slot in range(commitment["branch_count"])
            for replicate in range(1, commitment["continuation_replicates"] + 1)
        }
        actual_executions = {item for item in self._executions if item[:2] == key}
        if (
            actual_candidate_slots != expected_candidate_slots
            or actual_executions != expected_executions
        ):
            raise LedgerError("branch artifact completion lacks its exact K-by-c denominator")
        receipt = self._append_receipt(
            "branch_group_artifact_completed",
            {
                "group_id": group_id,
                "target_id": target_id,
                "artifact_sha256": artifact_sha256,
                "training_batch_identity": training_batch_identity,
                "branch_count": commitment["branch_count"],
                "continuation_replicates": commitment["continuation_replicates"],
            },
            evidence_refs=(artifact_sha256,),
        )
        self._branch_artifacts[key] = artifact_sha256
        return receipt

    @_writer_transaction
    def claim_training_batch(
        self,
        *,
        training_batch_identity: str,
        artifact_sha256s: Sequence[str],
        consumer_id: str,
    ) -> bytes:
        self._require_writable()
        _require_sha256(training_batch_identity, "training_batch_identity")
        if not consumer_id:
            raise ValueError("consumer_id must be nonempty")
        artifacts = tuple(artifact_sha256s)
        if not artifacts or tuple(sorted(set(artifacts))) != artifacts:
            raise ValueError("artifact SHA-256 values must be sorted, unique, and nonempty")
        for digest in artifacts:
            _require_sha256(digest, "artifact_sha256")
        if training_batch_identity in self._batch_claims:
            raise BatchAlreadyClaimed(training_batch_identity)
        receipt = self._append_receipt(
            "training_batch_consumption",
            {
                "ledger_id": self._ledger_id,
                "training_batch_identity": training_batch_identity,
                "artifact_sha256s": list(artifacts),
                "consumer_id": consumer_id,
                "claim_sequence": len(self._records),
                "single_use": True,
            },
            evidence_refs=artifacts,
        )
        self._batch_claims.add(training_batch_identity)
        return receipt

    @_writer_transaction
    def _record_verified_support_gate(self, support_report_sha256: str) -> bytes:
        """Anchor the controller-verified support pass before training authorization."""
        self._require_writable()
        _require_sha256(support_report_sha256, "support_report_sha256")
        self._require_evidence(support_report_sha256)
        if self._verified_support_report_sha256 is not None:
            if self._verified_support_report_sha256 != support_report_sha256:
                raise LedgerError("scientific campaign already has a different support pass")
            return next(
                canonical_json(receipt)
                for (kind, _), receipt in self._receipts.items()
                if kind == "stage_d_support_gate_pass"
            )
        if (
            not self._source_rollout_completed
            or set(self._branch_artifacts) != self._branch_target_keys
        ):
            raise LedgerError("support pass requires complete source and branch rosters")
        receipt = self._append_receipt(
            "stage_d_support_gate_pass",
            {
                "ledger_id": self._ledger_id,
                "support_rules_sha256": self.genesis_binding.support_rules_sha256,
                "support_report_sha256": support_report_sha256,
                "source_sha256s": list(self.completed_source_sha256s),
                "branch_artifact_sha256s": list(
                    self.completed_branch_artifact_sha256s
                ),
            },
            evidence_refs=(support_report_sha256,),
        )
        self._verified_support_report_sha256 = support_report_sha256
        return receipt

    @_writer_transaction
    def authorize_stage_d_training_batch(
        self,
        *,
        arm: Literal["stock", "branch-global", "local"],
        training_batch_identity: str,
        sealed_batch_sha256: str,
        objective_sha256: str,
        objective_authorization_sha256: str,
        collection_plan_sha256: str,
        collection_receipt_sha256: str,
        support_report_sha256: str,
        source_sha256s: Sequence[str],
        branch_artifact_sha256s: Sequence[str],
        consumer_id: str,
    ) -> StageDTrainingBatchAuthorization:
        """Issue one same-ledger, single-use authorization for an exact live arm batch."""
        self._require_writable()
        if arm not in {"stock", "branch-global", "local"}:
            raise ValueError("unsupported Stage D training arm")
        for digest, name in (
            (training_batch_identity, "training_batch_identity"),
            (sealed_batch_sha256, "sealed_batch_sha256"),
            (objective_sha256, "objective_sha256"),
            (objective_authorization_sha256, "objective_authorization_sha256"),
            (collection_plan_sha256, "collection_plan_sha256"),
            (collection_receipt_sha256, "collection_receipt_sha256"),
            (support_report_sha256, "support_report_sha256"),
        ):
            _require_sha256(digest, name)
        if not consumer_id:
            raise ValueError("consumer_id must be nonempty")
        sources = tuple(source_sha256s)
        artifacts = tuple(branch_artifact_sha256s)
        if not sources or tuple(sorted(set(sources))) != sources:
            raise ValueError("source SHA-256 roster must be sorted, unique, and nonempty")
        if tuple(sorted(set(artifacts))) != artifacts:
            raise ValueError("branch artifact SHA-256 roster must be sorted and unique")
        if (arm == "stock") != (not artifacts):
            raise ValueError("only the stock arm may have an empty branch-artifact roster")
        for digest in (*sources, *artifacts):
            _require_sha256(digest, "Stage D batch evidence SHA-256")
        completed_sources = {
            completion.source_sha256 for completion in self._source_rollout_completed.values()
        }
        if set(sources) != completed_sources:
            raise LedgerError("Stage D batch source roster differs from completed rollouts")
        completed_artifacts = set(self._branch_artifacts.values())
        if arm != "stock" and (
            set(self._branch_artifacts) != self._branch_target_keys
            or set(artifacts) != completed_artifacts
        ):
            raise LedgerError(
                "Stage D batch artifact roster differs from completed branch groups"
            )
        if self._verified_support_report_sha256 != support_report_sha256:
            raise LedgerError("Stage D training requires the verified support pass")
        evidence_refs = tuple(
            sorted(
                {
                    sealed_batch_sha256,
                    objective_authorization_sha256,
                    collection_plan_sha256,
                    collection_receipt_sha256,
                    support_report_sha256,
                    *artifacts,
                }
            )
        )
        for digest in evidence_refs:
            self._require_evidence(digest)
        requested = {
            "arm": arm,
            "training_batch_identity": training_batch_identity,
            "sealed_batch_sha256": sealed_batch_sha256,
            "objective_sha256": objective_sha256,
            "objective_authorization_sha256": objective_authorization_sha256,
            "collection_plan_sha256": collection_plan_sha256,
            "collection_receipt_sha256": collection_receipt_sha256,
            "support_report_sha256": support_report_sha256,
            "source_sha256s": list(sources),
            "branch_artifact_sha256s": list(artifacts),
            "consumer_id": consumer_id,
        }
        existing = self._stage_d_training_authorizations.get(arm)
        if existing is not None:
            if any(existing.get(name) != value for name, value in requested.items()):
                raise LedgerError("Stage D arm already has a different authorization")
            return StageDTrainingBatchAuthorization(
                self._ledger_id,
                arm,
                training_batch_identity,
                sealed_batch_sha256,
                canonical_json(existing),
            )
        if training_batch_identity in self._batch_claims:
            raise BatchAlreadyClaimed(training_batch_identity)
        claim_sequence = len(self._records)
        receipt = self._append_receipt(
            "stage_d_training_batch_authorization",
            {
                "ledger_id": self._ledger_id,
                "ledger_offset": claim_sequence,
                "prior_chain_sha256": self.head_sha256,
                "arm": arm,
                "training_batch_identity": training_batch_identity,
                "sealed_batch_sha256": sealed_batch_sha256,
                "objective_sha256": objective_sha256,
                "objective_authorization_sha256": objective_authorization_sha256,
                "collection_plan_sha256": collection_plan_sha256,
                "collection_receipt_sha256": collection_receipt_sha256,
                "support_report_sha256": support_report_sha256,
                "source_sha256s": list(sources),
                "branch_artifact_sha256s": list(artifacts),
                "consumer_id": consumer_id,
                "claim_sequence": claim_sequence,
                "single_use": True,
            },
            evidence_refs=evidence_refs,
        )
        self._batch_claims.add(training_batch_identity)
        self._stage_d_training_authorizations[arm] = json.loads(receipt)
        return StageDTrainingBatchAuthorization(
            self._ledger_id,
            arm,
            training_batch_identity,
            sealed_batch_sha256,
            receipt,
        )

    @_writer_transaction
    def seal(self) -> LedgerSeal:
        self._require_writable()
        if (
            self._pending_candidates
            or self._pending_executions
            or self._source_policy_pending
            or any(not state["materialized"] for state in self._recorded_reservations.values())
        ):
            raise LedgerPoisoned("cannot seal with dangling scientific attempts")
        covered = {
            (completion.rollout_id, decision_id)
            for completion in self._source_rollout_completed.values()
            for decision_id in json.loads(completion.receipt)["decision_ids"]
        }
        if covered != set(self._source_policy_completed):
            raise LedgerPoisoned("cannot seal with unbound source policy completions")
        if (
            self._source_rollout_completed
            and self._commitments
            and self._branch_target_roster_sha256 is None
        ):
            raise LedgerPoisoned("cannot seal completed sources without a target roster")
        receipt_count = len(self._receipts)
        genesis_sha256 = self._record_sha256s[0]
        seal_offset = len(self._records)
        self._append_event(
            "seal",
            {
                "genesis_sha256": genesis_sha256,
                "pre_seal_head_sha256": self.head_sha256,
                "record_count_including_seal": seal_offset + 1,
                "receipt_count": receipt_count,
            },
            evidence_refs=(),
        )
        seal = LedgerSeal(
            self._ledger_id,
            genesis_sha256,
            self.head_sha256,
            len(self._records),
            receipt_count,
        )
        self.close()
        return seal

    @_writer_transaction
    def seal_scientific_campaign(self) -> LedgerSeal:
        """Seal only after the frozen QA and artifact rosters are complete."""
        if self._branch_target_roster_sha256 is None:
            raise LedgerError("scientific campaign seal requires a frozen target roster")
        if (
            self._reconstruction_qa_barrier_sha256 is None
            or set(self._branch_artifacts) != self._branch_target_keys
        ):
            raise LedgerError("scientific campaign is not roster-complete")
        if set(self._stage_d_training_authorizations) != {
            "stock",
            "branch-global",
            "local",
        }:
            raise LedgerError("scientific campaign lacks its three training authorizations")
        if self._verified_support_report_sha256 is None:
            raise LedgerError("scientific campaign lacks its verified support pass")
        return self.seal()

    def _append_receipt(
        self,
        receipt_kind: str,
        payload: Mapping[str, Any],
        *,
        evidence_refs: Sequence[str],
    ) -> bytes:
        if not receipt_kind:
            raise ValueError("receipt_kind must be nonempty")
        receipt = canonical_json({"schema_version": 1, "receipt_kind": receipt_kind, **payload})
        if len(receipt) > _MAX_RECEIPT_BYTES:
            raise ValueError("receipt exceeds the one-MiB cap")
        digest = _sha256(receipt)
        key = (receipt_kind, digest)
        if key in self._receipts:
            raise LedgerError("receipt bytes are already anchored in this ledger")
        self._append_record(
            "receipt",
            {
                "receipt_kind": receipt_kind,
                "receipt_sha256": digest,
                "receipt": json.loads(receipt),
                "evidence_refs": list(self._validated_evidence_refs(evidence_refs)),
            },
        )
        self._receipts[key] = json.loads(receipt)
        return receipt

    def _append_event(
        self,
        record_kind: str,
        event: Mapping[str, Any],
        *,
        evidence_refs: Sequence[str],
    ) -> None:
        self._append_record(
            record_kind,
            {
                "event": dict(event),
                "evidence_refs": list(self._validated_evidence_refs(evidence_refs)),
            },
        )

    def _append_record(self, record_kind: str, body: Mapping[str, Any]) -> None:
        self._require_writable()
        record = {
            "schema_version": 1,
            "domain": _DOMAIN,
            "ledger_id": self._ledger_id,
            "offset": len(self._records),
            "prior_record_sha256": self.head_sha256,
            "record_kind": record_kind,
            "body": dict(body),
        }
        encoded = canonical_json(record)
        if len(encoded) > _MAX_RECORD_BYTES:
            raise ValueError("ledger record exceeds the two-MiB cap")
        try:
            _atomic_record_write(
                self._records_dir / _record_name(len(self._records)),
                encoded,
            )
        except BaseException:
            self._poisoned = True
            raise
        self._records.append(record)
        self._record_sha256s.append(_sha256(encoded))

    def _validated_evidence_refs(self, refs: Sequence[str]) -> tuple[str, ...]:
        values = tuple(refs)
        if tuple(sorted(set(values))) != values:
            raise ValueError("evidence refs must be sorted and unique")
        for digest in values:
            self._require_evidence(digest)
        return values

    def _require_evidence(self, digest: str) -> None:
        _require_sha256(digest, "evidence_sha256")
        path = self._evidence_dir / digest
        if not path.is_file() or _sha256(path.read_bytes()) != digest:
            raise LedgerError("evidence ref is absent or corrupted")
        self._evidence_refs.add(digest)

    def _fresh_id(self, domain: str) -> str:
        return f"{self._ledger_id}:{domain}:{secrets.token_hex(12)}"

    def _require_committed(self, group_id: str, target_id: str) -> tuple[str, str]:
        key = _group_key(group_id, target_id)
        if key not in self._commitments:
            raise LedgerError("group target is not durably committed")
        return key

    def _require_reconstruction_qa_ready(
        self,
        *,
        group_id: str,
        target_id: str,
        recorded_action: BehaviorAction,
        actual_cost: ActualEvaluationCost,
    ) -> None:
        self._require_writable()
        key = self._require_committed(group_id, target_id)
        commitment = self._commitments[key]
        rollout_key = (group_id, cast(str, commitment["rollout_id"]))
        if rollout_key not in self._source_rollout_completed:
            raise LedgerError("reconstruction QA requires a completed source rollout")
        if self._branch_target_roster_sha256 is None:
            raise LedgerError("reconstruction QA requires a frozen target roster")
        if key not in self._branch_target_keys:
            raise LedgerError("reconstruction QA target is absent from the frozen roster")
        if key not in self._correspondence:
            raise LedgerError("correspondence must be frozen before reconstruction QA")
        expected_action_digest = self._recorded_action_digests.get(key)
        if expected_action_digest is None:
            raise LedgerError("recorded action must materialize before reconstruction QA")
        if expected_action_digest != recorded_action.digest:
            raise ValueError("recorded action changed after commitment")
        if key in self._qa:
            raise LedgerError("reconstruction QA is already recorded")
        if self._reconstruction_qa_barrier_sha256 is not None:
            raise LedgerError("reconstruction QA barrier is already sealed")
        if (
            self._candidate_slots
            or self._pending_candidates
            or self._executions
            or self._pending_executions
            or self._branch_artifacts
            or self._batch_claims
            or self._stage_d_training_authorizations
            or self._verified_support_report_sha256 is not None
        ):
            raise LedgerError("every reconstruction QA must precede scientific activity")
        if type(actual_cost) is not ActualEvaluationCost:
            raise TypeError("reconstruction QA cost must be ActualEvaluationCost")
        if (
            type(actual_cost.generated_tokens) is not int
            or type(actual_cost.judge_calls) is not int
            or actual_cost.generated_tokens != 0
            or actual_cost.judge_calls != 0
        ):
            raise ValueError("reconstruction QA must make zero model calls")

    def _require_scientific_ready(self, group_id: str, target_id: str) -> tuple[str, str]:
        key = self._require_committed(group_id, target_id)
        self._require_target_rostered(key)
        if key not in self._correspondence or self._qa_passed.get(key) is not True:
            raise LedgerError("group target is not ready for scientific activity")
        if self._branch_target_roster_sha256 is not None and (
            self._reconstruction_qa_barrier_sha256 is None
        ):
            raise LedgerError("whole-roster reconstruction QA barrier is not sealed")
        return key

    def _require_target_rostered(self, key: tuple[str, str]) -> None:
        if self._source_rollout_completed and self._branch_target_roster_sha256 is None:
            raise LedgerError("completed source rollouts require a frozen branch target roster")
        if self._branch_target_roster_sha256 is not None and key not in self._branch_target_keys:
            raise LedgerError("scientific target is absent from the frozen branch target roster")

    def _validate_roster_target(
        self,
        value: object,
        *,
        excluded: bool = False,
    ) -> dict[str, Any]:
        expected = {
            "source_sha256",
            "group_id",
            "rollout_id",
            "decision_id",
            "target_id",
            "target_ordinal",
            "event_address",
        }
        if excluded:
            expected.add("reason")
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("branch target roster entry fields differ")
        target = cast(dict[str, Any], value)
        for field in ("group_id", "rollout_id", "decision_id", "target_id"):
            if not isinstance(target[field], str) or not target[field]:
                raise ValueError("branch target roster identifiers must be nonempty")
        _require_sha256(target["source_sha256"], "source_sha256")
        if excluded and (not isinstance(target["reason"], str) or not target["reason"]):
            raise ValueError("excluded branch target requires a reason")
        if type(target["target_ordinal"]) is not int or target["target_ordinal"] < 0:
            raise ValueError("branch target roster ordinal is invalid")
        source = self._source_rollout_completed.get(
            (target["group_id"], target["rollout_id"])
        )
        reservation = self._source_policy_reservations.get(
            (target["rollout_id"], target["decision_id"])
        )
        commitment = self._commitments.get((target["group_id"], target["target_id"]))
        if (
            source is None
            or source.source_sha256 != target["source_sha256"]
            or reservation is None
            or reservation.get("group_id") != target["group_id"]
            or reservation.get("node_kind") != "child"
            or reservation.get("branch_selected") is not True
            or reservation.get("target_id") != target["target_id"]
            or reservation.get("target_ordinal") != target["target_ordinal"]
            or reservation.get("target_address") != target["event_address"]
            or commitment is None
            or commitment.get("rollout_id") != target["rollout_id"]
            or commitment.get("target_ordinal") != target["target_ordinal"]
            or commitment.get("target_address") != target["event_address"]
        ):
            raise ValueError("branch target roster entry lacks exact source provenance")
        return target

    def _candidate_state(self, attempt: CandidateAttempt) -> dict[str, Any]:
        if attempt.ledger_id != self._ledger_id:
            raise LedgerError("candidate attempt belongs to another ledger")
        key = (attempt.group_id, attempt.target_id, attempt.action_slot)
        state = self._pending_candidates.get(key)
        if state is None or state["attempt"] != attempt:
            raise LedgerError("candidate attempt is not active")
        return state

    def _source_action_digest(
        self,
        attempt: ExecutionAttempt,
        address: PolicyEventAddress,
    ) -> str:
        commitment = self._commitments[(attempt.group_id, attempt.target_id)]
        rollout_id = commitment["rollout_id"]
        address_payload = _address_payload(address)
        matches = [
            self._source_policy_action_digests[(source_rollout, decision_id)]
            for (source_rollout, decision_id), reservation in (
                self._source_policy_reservations.items()
            )
            if source_rollout == rollout_id
            and reservation["target_address"] == address_payload
            and (source_rollout, decision_id) in self._source_policy_action_digests
        ]
        if len(matches) != 1:
            raise LedgerError("reuse address lacks one frozen source action")
        return matches[0]

    def _recorded_action_state(
        self,
        reservation: RecordedActionReservation,
    ) -> dict[str, Any]:
        if reservation.ledger_id != self._ledger_id:
            raise LedgerError("recorded action reservation belongs to another ledger")
        key = (reservation.group_id, reservation.target_id)
        state = self._recorded_reservations.get(key)
        if state is None or state["reservation"] != reservation:
            raise LedgerError("recorded action reservation is not active")
        return state

    def _execution_state(self, attempt: ExecutionAttempt) -> dict[str, Any]:
        if attempt.ledger_id != self._ledger_id:
            raise LedgerError("execution attempt belongs to another ledger")
        key = (
            attempt.group_id,
            attempt.target_id,
            attempt.arm_id,
            attempt.continuation_replicate,
        )
        state = self._pending_executions.get(key)
        if state is None or state["attempt"] != attempt:
            raise LedgerError("execution attempt is not active")
        return state

    def _candidate_slots_for(self, key: tuple[str, str]) -> bool:
        slots = self._candidate_slots.keys() | self._pending_candidates.keys()
        return any(slot[:2] == key for slot in slots)

    def _execution_keys_for(self, key: tuple[str, str]) -> bool:
        executions = self._executions | self._pending_executions.keys()
        return any(item[:2] == key for item in executions)

    def _require_writable(self) -> None:
        if self._closed:
            raise LedgerError("ledger writer is closed")
        if self._poisoned:
            raise LedgerPoisoned("ledger writer is poisoned")
        if self._finalization_active:
            raise LedgerError("ledger finalization is already active")

    def _release_lock(self) -> None:
        descriptor = getattr(self, "_lock_descriptor", None)
        if descriptor is not None:
            _release_os_lock(descriptor)
            self._lock_descriptor = None

    def _rebuild_state(self) -> None:
        self._commitments: dict[tuple[str, str], dict[str, Any]] = {}
        self._recorded_reservations: dict[tuple[str, str], dict[str, Any]] = {}
        self._recorded_action_digests: dict[tuple[str, str], str] = {}
        self._correspondence: set[tuple[str, str]] = set()
        self._qa: set[tuple[str, str]] = set()
        self._qa_passed: dict[tuple[str, str], bool] = {}
        self._qa_receipt_sha256s: dict[tuple[str, str], str] = {}
        self._candidate_slots: dict[tuple[str, str, int], str] = {}
        self._candidate_action_sha256s: dict[tuple[str, str, int], str] = {}
        self._candidate_receipts: dict[tuple[str, str, int], bytes] = {}
        self._candidate_zero_call_failures: dict[tuple[str, str, int], int] = {}
        self._zero_call_failure_count = 0
        self._pending_candidates: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._executions: set[tuple[str, str, str, int]] = set()
        self._execution_receipts: dict[tuple[str, str, str, int], bytes] = {}
        self._execution_zero_call_failures: dict[
            tuple[str, str, str, int], int
        ] = {}
        self._pending_executions: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        self._source_policy_pending: dict[tuple[str, str], SourcePolicyCallReservation] = {}
        self._source_policy_reservations: dict[tuple[str, str], dict[str, Any]] = {}
        self._source_policy_responses: dict[tuple[str, str], str] = {}
        self._source_policy_completed: dict[tuple[str, str], str] = {}
        self._source_policy_action_digests: dict[tuple[str, str], str] = {}
        self._source_rollout_completed: dict[tuple[str, str], SourceRolloutCompletion] = {}
        self._branch_target_roster_sha256 = None
        self._branch_target_keys = set()
        self._reconstruction_qa_barrier_sha256 = None
        self._branch_artifacts: dict[tuple[str, str], str] = {}
        self._batch_claims: set[str] = set()
        self._stage_d_training_authorizations: dict[
            Literal["stock", "branch-global", "local"], dict[str, Any]
        ] = {}
        self._verified_support_report_sha256: str | None = None
        for record in self._records:
            if record["record_kind"] == "action_reservation":
                event = _event(_body(record))
                key = (event["group_id"], event["target_id"])
                commitment = self._commitments[key]
                reservation = RecordedActionReservation(
                    self._ledger_id,
                    event["group_id"],
                    event["target_id"],
                    event["reservation_id"],
                    event["recorded_action_seed"],
                    event["exact_action_key_digest"],
                    event["request_sha256"],
                    canonical_json(commitment),
                )
                self._recorded_reservations[key] = {
                    "reservation": reservation,
                    "started": False,
                    "materialized": False,
                }
                continue
            if record["record_kind"] == "model_call_started":
                event = _event(_body(record))
                if event["attempt_kind"] == "recorded_action":
                    state = next(
                        state
                        for state in self._recorded_reservations.values()
                        if state["reservation"].reservation_id == event["attempt_id"]
                    )
                    state["started"] = True
                    state["call_id"] = event["call_id"]
                continue
            if record["record_kind"] == "recorded_action_materialized":
                event = _event(_body(record))
                key = (event["group_id"], event["target_id"])
                self._recorded_reservations[key]["materialized"] = True
                self._recorded_action_digests[key] = event["action_digest"]
                continue
            if record["record_kind"] == "candidate_attempt":
                event = _event(_body(record))
                attempt = CandidateAttempt(
                    self._ledger_id,
                    event["group_id"],
                    event["target_id"],
                    event["action_slot"],
                    event["action_seed"],
                    event["attempt_ordinal"],
                    event["attempt_id"],
                )
                self._pending_candidates[
                    (attempt.group_id, attempt.target_id, attempt.action_slot)
                ] = {"attempt": attempt, "started": False}
                continue
            if record["record_kind"] == "execution_attempt":
                event = _event(_body(record))
                attempt = ExecutionAttempt(
                    self._ledger_id,
                    event["group_id"],
                    event["target_id"],
                    event["arm_id"],
                    event["action_digest"],
                    event["continuation_replicate"],
                    event["attempt_ordinal"],
                    event["attempt_id"],
                )
                self._pending_executions[
                    (
                        attempt.group_id,
                        attempt.target_id,
                        attempt.arm_id,
                        attempt.continuation_replicate,
                    )
                ] = {
                    "attempt": attempt,
                    "action": None,
                    "calls": [],
                    "in_flight": {},
                    "context_sha256": None,
                    "dispatched": False,
                    "overrides": {},
                    "responses": {},
                }
                continue
            if record["record_kind"] in {
                "execution_context_bound",
                "execution_dispatched",
            }:
                event = _event(_body(record))
                state = next(
                    state
                    for state in self._pending_executions.values()
                    if state["attempt"].attempt_id == event["attempt_id"]
                )
                if record["record_kind"] == "execution_context_bound":
                    state["context_sha256"] = event["context_sha256"]
                else:
                    state["dispatched"] = True
                continue
            if record["record_kind"] == "execution_override_committed":
                event = _event(_body(record))
                state = next(
                    state
                    for state in self._pending_executions.values()
                    if state["attempt"].attempt_id == event["attempt_id"]
                )
                address = _scanned_policy_address(event["address"])
                ticket = ReplayOverrideTicket(
                    event["attempt_id"],
                    event["override_id"],
                    address,
                    event["action_digest"],
                    event["disposition"],
                    event["response_content_sha256"],
                    event["prompt_tokens"],
                    event["completion_tokens"],
                    event["counts_toward_logical_cost"],
                )
                state["overrides"][address] = {"ticket": ticket, "delivered": False}
                continue
            if record["record_kind"] == "execution_override_delivered":
                event = _event(_body(record))
                state = next(
                    state
                    for state in self._pending_executions.values()
                    if state["attempt"].attempt_id == event["attempt_id"]
                )
                matching = next(
                    item
                    for item in state["overrides"].values()
                    if item["ticket"].override_id == event["override_id"]
                )
                matching["delivered"] = True
                continue
            if record["record_kind"] != "receipt":
                continue
            receipt = _body(record)["receipt"]
            kind = receipt["receipt_kind"]
            if kind == "pre_action_group_commitment":
                key = _group_key(receipt["group_id"], receipt["target_id"])
                self._commitments[key] = receipt
            elif kind == "source_policy_call_reserved":
                key = (receipt["rollout_id"], receipt["decision_id"])
                self._source_policy_reservations[key] = receipt
                self._source_policy_pending[key] = SourcePolicyCallReservation(
                    self._ledger_id,
                    receipt["group_id"],
                    receipt["rollout_id"],
                    receipt["decision_id"],
                    canonical_json(receipt),
                    receipt["exact_action_key_digest"],
                    receipt["request_sequence"],
                )
            elif kind == "source_policy_call_completed":
                key = (receipt["rollout_id"], receipt["decision_id"])
                self._source_policy_pending.pop(key)
                self._source_policy_completed[key] = _sha256(canonical_json(receipt))
                self._source_policy_action_digests[key] = receipt["action_digest"]
            elif kind == "source_policy_response_observed":
                key = (receipt["rollout_id"], receipt["decision_id"])
                if key not in self._source_policy_pending or key in self._source_policy_responses:
                    raise LedgerError("source policy response witness is out of order")
                self._source_policy_responses[key] = receipt["raw_response_sha256"]
            elif kind == "source_rollout_completed":
                key = (receipt["group_id"], receipt["rollout_id"])
                self._source_rollout_completed[key] = SourceRolloutCompletion(
                    self._ledger_id,
                    receipt["group_id"],
                    receipt["rollout_id"],
                    receipt["source_sha256"],
                    canonical_json(receipt),
                )
            elif kind == "branch_target_roster":
                self._branch_target_roster_sha256 = receipt["roster_sha256"]
                self._branch_target_keys = {
                    (target["group_id"], target["target_id"])
                    for target in receipt["targets"]
                }
            elif kind == "seed_correspondence_map":
                key = _group_key(receipt["group_id"], receipt["target_id"])
                self._correspondence.add(key)
                self._recorded_action_digests[key] = receipt["recorded_action_digest"]
            elif kind == "reconstruction_qa":
                key = _group_key(receipt["group_id"], receipt["target_id"])
                self._qa.add(key)
                self._qa_passed[key] = receipt["passed"]
                self._qa_receipt_sha256s[key] = _sha256(canonical_json(receipt))
            elif kind == "reconstruction_qa_barrier":
                self._reconstruction_qa_barrier_sha256 = _sha256(
                    canonical_json(receipt)
                )
            elif kind == "candidate_action_inference":
                slot = (
                    receipt["group_id"],
                    receipt["target_id"],
                    receipt["action_slot"],
                )
                self._candidate_slots[slot] = receipt["action_digest"]
                self._candidate_action_sha256s[slot] = receipt[
                    "action_evidence_sha256"
                ]
                self._candidate_receipts[slot] = canonical_json(receipt)
                self._pending_candidates.pop(slot, None)
            elif kind == "zero_call_infrastructure_failure":
                slot = (receipt["group_id"], receipt["target_id"], receipt["action_slot"])
                self._candidate_zero_call_failures[slot] = receipt["attempt_ordinal"] + 1
                self._zero_call_failure_count += 1
                self._pending_candidates.pop(slot, None)
            elif kind == "scientific_arm_execution":
                execution_key = (
                    receipt["group_id"],
                    receipt["target_id"],
                    receipt["arm_id"],
                    receipt["continuation_replicate"],
                )
                self._executions.add(execution_key)
                self._execution_receipts[execution_key] = canonical_json(receipt)
                self._pending_executions.pop(execution_key, None)
            elif kind == "zero_call_execution_failure":
                execution_key = (
                    receipt["group_id"],
                    receipt["target_id"],
                    receipt["arm_id"],
                    receipt["continuation_replicate"],
                )
                self._execution_zero_call_failures[execution_key] = (
                    receipt["attempt_ordinal"] + 1
                )
                self._zero_call_failure_count += 1
                self._pending_executions.pop(execution_key, None)
            elif kind == "branch_group_artifact_completed":
                self._branch_artifacts[
                    receipt["group_id"], receipt["target_id"]
                ] = receipt["artifact_sha256"]
            elif kind == "stage_d_support_gate_pass":
                self._verified_support_report_sha256 = receipt[
                    "support_report_sha256"
                ]
            elif kind in {
                "training_batch_consumption",
                "stage_d_training_batch_authorization",
            }:
                self._batch_claims.add(receipt["training_batch_identity"])
                if kind == "stage_d_training_batch_authorization":
                    self._stage_d_training_authorizations[receipt["arm"]] = receipt


def inspect_ledger(
    root: Path,
    *,
    allow_source_inflight: bool = False,
    allow_repairable_zero_call: bool = False,
) -> _ScanResult:
    """Scan a ledger without repairing it and classify its recovery state."""
    try:
        return _scan_ledger(
            root,
            allow_source_inflight=allow_source_inflight,
            allow_repairable_zero_call=allow_repairable_zero_call,
        )
    except BaseException as error:
        return _ScanResult("poisoned", str(error), (), (), {}, frozenset(), None, None)


def _scan_ledger(
    root: Path,
    *,
    allow_source_inflight: bool = False,
    allow_repairable_zero_call: bool = False,
) -> _ScanResult:
    records_dir = root / "records"
    evidence_dir = root / "evidence"
    if not records_dir.is_dir() or not evidence_dir.is_dir():
        raise LedgerPoisoned("ledger directories are missing")
    unknown_records = [
        path.name for path in records_dir.iterdir() if not _valid_record_name(path.name)
    ]
    unknown_evidence = [
        path.name
        for path in evidence_dir.iterdir()
        if not path.is_file() or not _is_sha256(path.name)
    ]
    if unknown_records or unknown_evidence:
        raise LedgerPoisoned("ledger contains unknown or temporary files")
    paths = sorted(records_dir.iterdir())
    if not paths:
        raise LedgerPoisoned("ledger has no genesis")
    records: list[dict[str, Any]] = []
    digests: list[str] = []
    receipts: dict[tuple[str, str], dict[str, Any]] = {}
    evidence_refs: set[str] = set()
    prior = _GENESIS_PRIOR
    ledger_id: str | None = None
    for expected_offset, path in enumerate(paths):
        if path.name != _record_name(expected_offset):
            raise LedgerPoisoned("ledger offsets are not contiguous")
        encoded = path.read_bytes()
        if not encoded or len(encoded) > _MAX_RECORD_BYTES:
            raise LedgerPoisoned("ledger record is empty or oversized")
        value = _strict_canonical_object(encoded, "ledger record")
        if set(value) != _RECORD_KEYS:
            raise LedgerPoisoned("ledger record fields differ from schema")
        if value["schema_version"] != 1 or value["domain"] != _DOMAIN:
            raise LedgerPoisoned("ledger record envelope is invalid")
        if value["offset"] != expected_offset or value["prior_record_sha256"] != prior:
            raise LedgerPoisoned("ledger record order or prior hash is invalid")
        current_ledger = value["ledger_id"]
        if not isinstance(current_ledger, str) or not current_ledger:
            raise LedgerPoisoned("ledger_id is invalid")
        if ledger_id is None:
            ledger_id = current_ledger
        elif current_ledger != ledger_id:
            raise LedgerPoisoned("cross-ledger record splice detected")
        body = _body(value)
        refs = body.get("evidence_refs", [])
        if not isinstance(refs, list) or refs != sorted(set(refs)):
            raise LedgerPoisoned("evidence refs must be sorted and unique")
        for ref in refs:
            if not _is_sha256(ref):
                raise LedgerPoisoned("evidence ref is malformed")
            evidence_path = evidence_dir / ref
            if not evidence_path.is_file() or _sha256(evidence_path.read_bytes()) != ref:
                raise LedgerPoisoned("evidence ref is absent or corrupted")
            evidence_refs.add(ref)
        if value["record_kind"] == "receipt":
            if set(body) != {
                "receipt_kind",
                "receipt_sha256",
                "receipt",
                "evidence_refs",
            }:
                raise LedgerPoisoned("receipt record fields differ from schema")
            receipt = body["receipt"]
            receipt_bytes = canonical_json(receipt)
            if len(receipt_bytes) > _MAX_RECEIPT_BYTES:
                raise LedgerPoisoned("receipt exceeds the size cap")
            if (
                not isinstance(receipt, dict)
                or receipt.get("schema_version") != 1
                or receipt.get("receipt_kind") != body["receipt_kind"]
                or _sha256(receipt_bytes) != body["receipt_sha256"]
            ):
                raise LedgerPoisoned("receipt record is internally inconsistent")
            if receipt.get("receipt_kind") == "branch_target_roster":
                roster_digest = receipt.get("roster_sha256")
                if not _is_sha256(roster_digest):
                    raise LedgerPoisoned("branch target roster evidence hash is invalid")
                roster = _strict_canonical_object(
                    (evidence_dir / cast(str, roster_digest)).read_bytes(),
                    "Stage D branch target roster evidence",
                )
                expected_roster = {
                    "schema_version": 2,
                    "domain": "redco-stage-d-branch-target-roster-v2",
                    "planned_source_count": receipt.get("planned_source_count"),
                    "completed_source_count": receipt.get("completed_source_count"),
                    "eligible_source_count": receipt.get("eligible_source_count"),
                    "ineligible_source_count": receipt.get("ineligible_source_count"),
                    "minimum_eligible_sources": receipt.get("minimum_eligible_sources"),
                    "eligibility_passed": receipt.get("eligibility_passed"),
                    "source_sha256s": receipt.get("source_sha256s"),
                    "targets": receipt.get("targets"),
                    "excluded_targets": receipt.get("excluded_targets"),
                }
                if roster != expected_roster:
                    raise LedgerPoisoned("branch target roster receipt differs from its evidence")
            receipt_key = (body["receipt_kind"], body["receipt_sha256"])
            if receipt_key in receipts:
                raise LedgerPoisoned("duplicate receipt anchoring is forbidden")
            receipts[receipt_key] = receipt
            if receipt.get("ledger_id") is not None and receipt["ledger_id"] != ledger_id:
                raise LedgerPoisoned("position-bound receipt names another ledger")
            if (
                receipt.get("ledger_offset") is not None
                and receipt["ledger_offset"] != expected_offset
            ):
                raise LedgerPoisoned("position-bound receipt has the wrong offset")
            if (
                receipt.get("prior_chain_sha256") is not None
                and receipt["prior_chain_sha256"] != prior
            ):
                raise LedgerPoisoned("position-bound receipt has the wrong prior hash")
        records.append(value)
        digest = _sha256(encoded)
        digests.append(digest)
        prior = digest
    assert ledger_id is not None
    repairable_attempt = _validate_state_machine(
        records,
        allow_source_inflight=allow_source_inflight,
        allow_repairable_zero_call=allow_repairable_zero_call,
    )
    seal = _seal_from_records(records, digests, len(receipts))
    status: RecoveryStatus
    if seal is not None:
        status = "sealed-valid"
    elif repairable_attempt is not None:
        status = "active-repairable-zero-call"
    else:
        status = "active-clean"
    return _ScanResult(
        status,
        None,
        tuple(records),
        tuple(digests),
        receipts,
        frozenset(evidence_refs),
        seal,
        repairable_attempt,
    )


def _seal_from_records(
    records: Sequence[Mapping[str, Any]],
    digests: Sequence[str],
    receipt_count: int,
) -> LedgerSeal | None:
    if records[-1]["record_kind"] != "seal":
        return None
    event = _event(_body(records[-1]))
    if (
        event["genesis_sha256"] != digests[0]
        or event["pre_seal_head_sha256"] != digests[-2]
        or event["record_count_including_seal"] != len(records)
        or event["receipt_count"] != receipt_count
    ):
        raise LedgerPoisoned("terminal seal summary is invalid")
    return LedgerSeal(
        str(records[0]["ledger_id"]),
        digests[0],
        digests[-1],
        len(records),
        receipt_count,
    )


def _lookup_receipt(
    scan: _ScanResult,
    receipt: bytes,
    receipt_kind: str,
) -> Mapping[str, Any]:
    if type(receipt) is not bytes:
        raise ValueError("receipt must be immutable bytes")
    value = _strict_canonical_object(receipt, "receipt")
    if value.get("receipt_kind") != receipt_kind:
        raise ValueError("receipt kind differs from requested verifier scope")
    anchored = scan.receipts.get((receipt_kind, _sha256(receipt)))
    if anchored != value:
        raise ValueError("receipt is not anchored in this ledger")
    return anchored


def _finalization_manifest_path(spec: _FinalizationTransactionSpec) -> Path:
    return spec.root / f".{spec.transaction_id}.finalization.json"


def _canonical_root_path(root: Path) -> str:
    _validate_root_ancestors(root)
    return os.path.normcase(os.path.abspath(os.fspath(root)))


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x00000400)
    )


def _validate_root_ancestors(root: Path) -> None:
    """Reject symlink/reparse ancestors before any ledger path is used."""

    current = Path(os.path.abspath(os.fspath(root)))
    while True:
        if _is_link_or_reparse(current):
            raise LedgerPoisoned("ledger root or ancestor is a symlink/reparse point")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _root_binding(root: Path) -> str:
    return _sha256(_canonical_root_path(root).encode("utf-8"))


def _strict_keyset(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(value)
    if observed != expected:
        raise LedgerPoisoned(
            f"{name} fields differ: missing={sorted(expected - observed)} "
            f"unknown={sorted(observed - expected)}"
        )


def _hex_bytes(value: object, name: str, *, maximum: int) -> bytes:
    if not isinstance(value, str) or len(value) % 2 or len(value) > maximum * 2:
        raise LedgerPoisoned(f"{name} must be bounded lowercase hex")
    if any(character not in "0123456789abcdef" for character in value):
        raise LedgerPoisoned(f"{name} contains noncanonical hex")
    try:
        return bytes.fromhex(value)
    except ValueError as error:
        raise LedgerPoisoned(f"{name} is not valid hex") from error


def _transaction_baseline_paths(root: Path) -> list[dict[str, int | str]]:
    values: list[dict[str, int | str]] = []
    for directory in ("records", "evidence"):
        base = root / directory
        if not base.is_dir() or _is_link_or_reparse(base):
            raise LedgerPoisoned("finalization output directory is not a real directory")
        for path in sorted(base.iterdir(), key=lambda item: item.name):
            if not path.is_file() or _is_link_or_reparse(path):
                raise LedgerPoisoned("finalization baseline contains a non-file entry")
            if getattr(path.lstat(), "st_nlink", 1) != 1:
                raise LedgerPoisoned("finalization baseline contains an aliased file")
            if "/" in path.name or "\\" in path.name or path.name in {"", ".", ".."}:
                raise LedgerPoisoned("finalization baseline contains an invalid name")
            encoded = path.read_bytes()
            relative = f"{directory}/{path.name}"
            values.append(
                {
                    "relative_path": relative,
                    "sha256": _sha256(encoded),
                    "bytes": len(encoded),
                }
            )
    return sorted(values, key=lambda item: cast(str, item["relative_path"]))


def _transaction_file_map(root: Path, directory: str) -> dict[str, bytes]:
    base = root / directory
    values: dict[str, bytes] = {}
    for path in base.iterdir():
        if path.is_file() and not _is_link_or_reparse(path):
            values[path.name] = path.read_bytes()
    return values


def _transaction_patch(
    spec: _FinalizationTransactionSpec,
    base: dict[str, bytes],
    prepared: dict[str, bytes],
    directory: str,
) -> tuple[dict[str, object], ...]:
    for name, encoded in base.items():
        if prepared.get(name) != encoded:
            raise LedgerError(f"finalization changed an existing {directory} file")
    patches: list[dict[str, object]] = []
    for name, encoded in sorted(prepared.items()):
        if name in base:
            continue
        if name.startswith("."):
            raise LedgerPoisoned("finalization cannot add hidden output files")
        relative = f"{directory}/{name}"
        patches.append(
            {
                "relative_path": relative,
                "temporary_relative_path": f"{directory}/.{spec.transaction_id}.{name}.tmp",
                "sha256": _sha256(encoded),
                "bytes": len(encoded),
                "content_hex": encoded.hex(),
            }
        )
    return tuple(patches)


def _action_from_bytes(encoded: bytes) -> _ActionDigestProxy:
    try:
        envelope = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LedgerPoisoned("recorded action request is malformed") from error
    if (
        not isinstance(envelope, dict)
        or canonical_json(envelope) != encoded
        or set(envelope) != {"schema_version", "domain", "action", "digest"}
        or not isinstance(envelope.get("action"), dict)
    ):
        raise LedgerPoisoned("recorded action request envelope is invalid")
    action_payload = cast(dict[str, Any], envelope["action"])
    token_values = action_payload.get("action_token_ids")
    key_payload = action_payload.get("key")
    if not isinstance(token_values, list) or not isinstance(key_payload, dict):
        raise LedgerPoisoned("recorded action request lacks its exact key or tokens")
    prompt_values = key_payload.get("prompt_token_ids")
    if not isinstance(prompt_values, list):
        raise LedgerPoisoned("recorded action request lacks prompt tokens")
    schema_version = key_payload.get("schema_version")

    def render_prompt(_: Mapping[str, Any]) -> tuple[int, ...]:
        return tuple(prompt_values)

    try:
        if schema_version == 1:
            action = BehaviorAction.from_bytes(
                encoded,
                encode_action=lambda _request, _message: tuple(token_values),
                render_prompt=render_prompt,
            )
        elif schema_version == 2:
            action = BehaviorAction.from_bytes(
                encoded,
                validate_action=lambda _request, _message, tokens: None,
                render_prompt=render_prompt,
            )
        else:
            raise LedgerPoisoned("recorded action schema version is unsupported")
    except LedgerPoisoned:
        raise
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LedgerPoisoned("recorded action fields or hashes are invalid") from error
    if action.to_bytes() != encoded:
        raise LedgerPoisoned("recorded action canonical bytes changed")
    return _ActionDigestProxy(action.digest)


def _request_attempt(value: object) -> ExecutionAttempt:
    if not isinstance(value, dict):
        raise LedgerPoisoned("execution attempt request must be an object")
    _strict_keyset(
        value,
        {
            "ledger_id",
            "group_id",
            "target_id",
            "arm_id",
            "action_digest",
            "continuation_replicate",
            "attempt_ordinal",
            "attempt_id",
        },
        "execution attempt",
    )
    if any(not isinstance(value[name], str) or not value[name] for name in (
        "ledger_id", "group_id", "target_id", "arm_id", "action_digest", "attempt_id"
    )):
        raise LedgerPoisoned("execution attempt has an empty identity")
    if not _is_sha256(value["action_digest"]):
        raise LedgerPoisoned("execution attempt action digest is invalid")
    if type(value["continuation_replicate"]) is not int or value["continuation_replicate"] < 0:
        raise LedgerPoisoned("execution attempt continuation is invalid")
    if type(value["attempt_ordinal"]) is not int or value["attempt_ordinal"] < 0:
        raise LedgerPoisoned("execution attempt ordinal is invalid")
    return ExecutionAttempt(
        value["ledger_id"],
        value["group_id"],
        value["target_id"],
        value["arm_id"],
        value["action_digest"],
        value["continuation_replicate"],
        value["attempt_ordinal"],
        value["attempt_id"],
    )


def _request_cost(value: object) -> ActualEvaluationCost:
    if not isinstance(value, dict):
        raise LedgerPoisoned("actual cost request must be an object")
    _strict_keyset(
        value,
        {
            "generated_tokens",
            "judge_calls",
            "cpu_seconds",
            "gpu_seconds",
            "wall_seconds",
            "storage_bytes",
        },
        "actual cost",
    )
    for name in ("generated_tokens", "judge_calls", "storage_bytes"):
        if type(value[name]) is not int or value[name] < 0:
            raise LedgerPoisoned(f"actual cost {name} is invalid")
    for name in ("cpu_seconds", "gpu_seconds", "wall_seconds"):
        _finite_float(value[name], f"actual cost {name}")
    return ActualEvaluationCost(
        generated_tokens=value["generated_tokens"],
        judge_calls=value["judge_calls"],
        cpu_seconds=value["cpu_seconds"],
        gpu_seconds=value["gpu_seconds"],
        wall_seconds=value["wall_seconds"],
        storage_bytes=value["storage_bytes"],
    )


def _finalization_request_bytes(spec: _FinalizationTransactionSpec) -> bytes:
    action_hex = (
        None
        if spec.recorded_action_bytes is None
        else spec.recorded_action_bytes.hex()
    )
    attempt = None
    if spec.execution_attempt is not None:
        attempt = {
            "ledger_id": spec.execution_attempt.ledger_id,
            "group_id": spec.execution_attempt.group_id,
            "target_id": spec.execution_attempt.target_id,
            "arm_id": spec.execution_attempt.arm_id,
            "action_digest": spec.execution_attempt.action_digest,
            "continuation_replicate": spec.execution_attempt.continuation_replicate,
            "attempt_ordinal": spec.execution_attempt.attempt_ordinal,
            "attempt_id": spec.execution_attempt.attempt_id,
        }
    cost = None if spec.actual_cost is None else _actual_cost_payload(spec.actual_cost)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "domain": _FINALIZATION_REQUEST_DOMAIN,
        "transaction_id": spec.transaction_id,
        "root_path": _canonical_root_path(spec.root),
        "root_binding_sha256": _root_binding(spec.root),
        "master_seed": spec.master_seed,
        "master_seed_sha256": _sha256(spec.master_seed.encode("utf-8")),
        "ledger_id": spec.ledger_id,
        "genesis_sha256": spec.genesis_sha256,
        "base_head_sha256": spec.base_head_sha256,
        "base_record_count": spec.base_record_count,
        "operation": spec.operation,
        "evidence_hex": spec.evidence.hex(),
        "evidence_sha256": _sha256(spec.evidence),
        "group_id": spec.group_id,
        "target_id": spec.target_id,
        "recorded_action_hex": action_hex,
        "passed": spec.passed,
        "actual_cost": cost,
        "execution_attempt": attempt,
        "outcome_kind": None if spec.outcome_kind is None else spec.outcome_kind.value,
        "scored_reward": spec.scored_reward,
        "latency_seconds": spec.latency_seconds,
        "dollars": spec.dollars,
        "judge_calls": spec.judge_calls,
        "cpu_seconds": spec.cpu_seconds,
        "gpu_seconds": spec.gpu_seconds,
        "wall_seconds": spec.wall_seconds,
        "storage_bytes": spec.storage_bytes,
    }
    _strict_keyset(payload, _FINALIZATION_REQUEST_KEYS, "finalization request")
    encoded = cast(bytes, canonical_json(payload))
    if len(encoded) > _MAX_FINALIZATION_REQUEST_BYTES:
        raise LedgerPoisoned("finalization request exceeds its bound")
    return encoded


def _finalization_child_command(
    mode: Literal["--finalize-transaction-stdin", "--cleanup-finalization-stdin"],
    root: Path,
) -> tuple[str, ...]:
    """Return an isolated child command with a fixed repository import root."""

    import_code = (
        "import sys; sys.path.insert(0, "
        + json.dumps(_FINALIZATION_SOURCE_ROOT)
        + "); from redco.analysis.stage_d_receipt_ledger import "
        "_finalization_transaction_main; raise SystemExit("
        "_finalization_transaction_main(sys.argv[1:]))"
    )
    return (sys.executable, "-I", "-c", import_code, mode, str(root))


def _spec_from_request(
    encoded: bytes,
    root: Path,
    *,
    require_action: bool = True,
) -> tuple[_FinalizationTransactionSpec, str]:
    value = _strict_canonical_object(encoded, "finalization request")
    _strict_keyset(value, _FINALIZATION_REQUEST_KEYS, "finalization request")
    if value.get("schema_version") != 2 or value.get("domain") != _FINALIZATION_REQUEST_DOMAIN:
        raise LedgerPoisoned("finalization request schema is unsupported")
    if (
        value.get("root_path") != _canonical_root_path(root)
        or value.get("root_binding_sha256") != _root_binding(root)
    ):
        raise LedgerPoisoned("finalization request root binding changed")
    transaction_id = value.get("transaction_id")
    if (
        not isinstance(transaction_id, str)
        or len(transaction_id) != 32
        or any(character not in "0123456789abcdef" for character in transaction_id)
    ):
        raise LedgerPoisoned("finalization transaction ID is invalid")
    master_seed = value.get("master_seed")
    if not isinstance(master_seed, str) or not master_seed:
        raise LedgerPoisoned("finalization request master seed is invalid")
    if value.get("master_seed_sha256") != _sha256(master_seed.encode("utf-8")):
        raise LedgerPoisoned("finalization request master seed binding changed")
    for name in ("ledger_id", "group_id", "target_id"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise LedgerPoisoned(f"finalization request {name} is invalid")
    base_head = value.get("base_head_sha256")
    if not _is_sha256(base_head):
        raise LedgerPoisoned("finalization request base head is invalid")
    genesis_sha256 = value.get("genesis_sha256")
    if not _is_sha256(genesis_sha256):
        raise LedgerPoisoned("finalization request genesis is invalid")
    base_record_count = value.get("base_record_count")
    if type(base_record_count) is not int or base_record_count < 1:
        raise LedgerPoisoned("finalization request base record count is invalid")
    operation = value.get("operation")
    if operation not in {"qa", "execution"}:
        raise LedgerPoisoned("finalization request operation is invalid")
    evidence = _hex_bytes(
        value.get("evidence_hex"), "finalization evidence", maximum=_MAX_RECEIPT_BYTES
    )
    if not evidence or value.get("evidence_sha256") != _sha256(evidence):
        raise LedgerPoisoned("finalization evidence binding changed")
    action: BehaviorAction | _ActionDigestProxy | None = None
    action_hex = value.get("recorded_action_hex")
    action_bytes: bytes | None = None
    if action_hex is not None:
        action_bytes = _hex_bytes(action_hex, "recorded action", maximum=_MAX_RECORD_BYTES)
        if require_action:
            action = _action_from_bytes(action_bytes)
    passed = value.get("passed")
    cost_value = value.get("actual_cost")
    attempt_value = value.get("execution_attempt")
    outcome_value = value.get("outcome_kind")
    cost = None if cost_value is None else _request_cost(cost_value)
    attempt = None if attempt_value is None else _request_attempt(attempt_value)
    outcome = None
    if outcome_value is not None:
        try:
            outcome = OutcomeKind(outcome_value)
        except (TypeError, ValueError) as error:
            raise LedgerPoisoned("finalization outcome kind is invalid") from error
    if operation == "qa":
        if (
            (require_action and action is None)
            or type(passed) is not bool
            or cost is None
            or attempt is not None
            or outcome is not None
        ):
            raise LedgerPoisoned("QA finalization request fields are inconsistent")
    elif (
        action is not None
        or passed is not None
        or cost is not None
        or attempt is None
        or outcome is None
    ):
        raise LedgerPoisoned("execution finalization request fields are inconsistent")
    for name in (
        "scored_reward",
        "latency_seconds",
        "dollars",
        "cpu_seconds",
        "gpu_seconds",
        "wall_seconds",
    ):
        _finite_float(value.get(name), name)
    for name in ("judge_calls", "storage_bytes"):
        if type(value.get(name)) is not int or value[name] < 0:
            raise LedgerPoisoned(f"finalization request {name} is invalid")
    spec = _FinalizationTransactionSpec(
        operation=operation,
        root=root,
        master_seed=master_seed,
        transaction_id=transaction_id,
        ledger_id=value["ledger_id"],
        genesis_sha256=cast(str, genesis_sha256),
        base_head_sha256=cast(str, base_head),
        base_record_count=base_record_count,
        evidence=evidence,
        group_id=value["group_id"],
        target_id=value["target_id"],
        recorded_action=action,
        recorded_action_bytes=action_bytes,
        passed=passed,
        actual_cost=cost,
        execution_attempt=attempt,
        outcome_kind=outcome,
        scored_reward=value["scored_reward"],
        latency_seconds=value["latency_seconds"],
        dollars=value["dollars"],
        judge_calls=value["judge_calls"],
        cpu_seconds=value["cpu_seconds"],
        gpu_seconds=value["gpu_seconds"],
        wall_seconds=value["wall_seconds"],
        storage_bytes=value["storage_bytes"],
    )
    return spec, _sha256(encoded)


def _transaction_temp_path(spec: _FinalizationTransactionSpec, relative: str) -> str:
    directory, name = relative.split("/", 1)
    return f"{directory}/.{spec.transaction_id}.{name}.tmp"


def _transaction_patch_payloads(
    patches: Sequence[Mapping[str, Any]],
    *,
    transaction_id: str,
    directory: str,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    paths: set[str] = set()
    temporary_paths: set[str] = set()
    for patch in patches:
        if not isinstance(patch, dict):
            raise LedgerPoisoned("finalization patch is not an object")
        _strict_keyset(patch, _FINALIZATION_PATCH_KEYS, "finalization patch")
        relative = patch.get("relative_path")
        temporary = patch.get("temporary_relative_path")
        if (
            not isinstance(relative, str)
            or not isinstance(temporary, str)
            or relative in paths
            or temporary in temporary_paths
            or "\\" in relative
            or "\\" in temporary
        ):
            raise LedgerPoisoned("finalization patch path is invalid or duplicated")
        parts = relative.split("/")
        if len(parts) != 2 or parts[0] != directory or not parts[1] or ".." in parts:
            raise LedgerPoisoned("finalization patch escapes its output directory")
        expected_temporary = f"{directory}/.{transaction_id}.{parts[1]}.tmp"
        if temporary != expected_temporary:
            raise LedgerPoisoned("finalization temporary path is not transaction-specific")
        encoded = _hex_bytes(
            patch.get("content_hex"), "finalization patch content", maximum=_MAX_RECORD_BYTES
        )
        if (
            type(patch.get("bytes")) is not int
            or patch.get("bytes") != len(encoded)
            or patch.get("sha256") != _sha256(encoded)
        ):
            raise LedgerPoisoned("finalization patch content binding changed")
        if directory == "records" and not _valid_record_name(parts[1]):
            raise LedgerPoisoned("finalization record path is invalid")
        if directory == "evidence" and not _is_sha256(parts[1]):
            raise LedgerPoisoned("finalization evidence path is invalid")
        paths.add(relative)
        temporary_paths.add(temporary)
        values.append({**patch, "content": encoded})
    return values


def _baseline_payload(
    baseline_paths: Sequence[Mapping[str, object]],
) -> list[dict[str, int | str]]:
    expected_entry_keys = {"relative_path", "sha256", "bytes"}
    result: list[dict[str, int | str]] = []
    previous = ""
    for item in baseline_paths:
        if not isinstance(item, Mapping) or set(item) != expected_entry_keys:
            raise LedgerPoisoned("finalization baseline entry schema differs")
        relative = item.get("relative_path")
        sha256 = item.get("sha256")
        size = item.get("bytes")
        if (
            not isinstance(relative, str)
            or "\\" in relative
            or relative <= previous
            or relative.count("/") != 1
            or relative.split("/", 1)[0] not in {"records", "evidence"}
            or not relative.split("/", 1)[1]
            or relative.split("/", 1)[1] in {".", ".."}
            or not _is_sha256(sha256)
            or type(size) is not int
            or size < 0
        ):
            raise LedgerPoisoned("finalization baseline path or digest is invalid")
        name = relative.split("/", 1)[1]
        if relative.startswith("records/") and not _valid_record_name(name):
            raise LedgerPoisoned("finalization baseline record name is invalid")
        if relative.startswith("evidence/") and not _is_sha256(name):
            raise LedgerPoisoned("finalization baseline evidence name is invalid")
        result.append(
            {
                "relative_path": relative,
                "sha256": cast(str, sha256),
                "bytes": size,
            }
        )
        previous = relative
    return result


def _baseline_digest(baseline_paths: Sequence[Mapping[str, object]]) -> str:
    return _sha256(canonical_json([dict(item) for item in baseline_paths]))


def _transaction_manifest(
    spec: _FinalizationTransactionSpec,
    *,
    state: Literal["committing", "committed"],
    request: bytes,
    baseline_paths: Sequence[Mapping[str, object]],
    result: Mapping[str, Any],
) -> bytes:
    canonical_baseline = _baseline_payload(baseline_paths)
    payload = {
        "schema_version": _FINALIZATION_TRANSACTION_SCHEMA_VERSION,
        "domain": _FINALIZATION_TRANSACTION_DOMAIN,
        "transaction_id": spec.transaction_id,
        "root_path": _canonical_root_path(spec.root),
        "root_binding_sha256": _root_binding(spec.root),
        "ledger_id": spec.ledger_id,
        "genesis_sha256": spec.genesis_sha256,
        "base_head_sha256": spec.base_head_sha256,
        "operation": spec.operation,
        "state": state,
        "request_sha256": _sha256(request),
        "result_sha256": result["result_sha256"],
        "result": dict(result),
        "baseline_paths": canonical_baseline,
        "baseline_sha256": _baseline_digest(canonical_baseline),
        "request_hex": request.hex(),
    }
    _strict_keyset(payload, _FINALIZATION_MANIFEST_KEYS, "finalization manifest")
    return cast(bytes, canonical_json(payload))


def _finalization_manifest_name(transaction_id: str) -> str:
    return f".{transaction_id}.finalization.json"


def _transaction_stage_path(spec: _FinalizationTransactionSpec) -> Path:
    return spec.root.parent / f".{spec.transaction_id}.stage-d-finalization-copy"


def _transaction_manifest_commit_temp(spec: _FinalizationTransactionSpec) -> Path:
    return _finalization_manifest_path(spec).with_name(
        _finalization_manifest_path(spec).name + ".commit.tmp"
    )


def _validate_transaction_spec_against_root(
    spec: _FinalizationTransactionSpec,
    *,
    root: Path,
    require_base: bool = True,
) -> None:
    scan = inspect_ledger(root)
    if scan.status != "active-clean":
        raise LedgerPoisoned("finalization root is not active-clean")
    if not scan.records:
        raise LedgerPoisoned("finalization root has no records")
    genesis = _body(scan.records[0])
    base_matches = (
        scan.record_sha256s[-1] == spec.base_head_sha256
        and len(scan.records) == spec.base_record_count
    )
    if (
        scan.records[0].get("ledger_id") != spec.ledger_id
        or scan.record_sha256s[0] != spec.genesis_sha256
        or (require_base and not base_matches)
        or genesis.get("master_seed_sha256") != _sha256(spec.master_seed.encode("utf-8"))
    ):
        raise LedgerPoisoned("finalization request is not bound to the live ledger base")


def _safe_output_path(root: Path, relative: str) -> Path:
    if "\\" in relative or relative.count("/") != 1:
        raise LedgerPoisoned("finalization path is not canonical")
    directory, name = relative.split("/", 1)
    if directory not in {"records", "evidence"} or not name or name in {".", ".."}:
        raise LedgerPoisoned("finalization path is outside the output directories")
    directory_path = root / directory
    path = directory_path / name
    if (
        _is_link_or_reparse(root)
        or not root.is_dir()
        or _is_link_or_reparse(directory_path)
        or not directory_path.is_dir()
        or path.parent != directory_path
        or path.name != name
        or (path.exists() and _is_link_or_reparse(path))
    ):
        raise LedgerPoisoned("finalization path escapes the transaction root")
    return path


def _validate_manifest_state(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[_FinalizationTransactionSpec, dict[str, Any], list[dict[str, int | str]]]:
    _strict_keyset(manifest, _FINALIZATION_MANIFEST_KEYS, "finalization manifest")
    if (
        manifest.get("schema_version") != _FINALIZATION_TRANSACTION_SCHEMA_VERSION
        or manifest.get("domain") != _FINALIZATION_TRANSACTION_DOMAIN
        or manifest.get("root_path") != _canonical_root_path(root)
        or manifest.get("root_binding_sha256") != _root_binding(root)
    ):
        raise LedgerPoisoned("finalization manifest root binding changed")
    transaction_id = manifest.get("transaction_id")
    if (
        not isinstance(transaction_id, str)
        or len(transaction_id) != 32
        or any(character not in "0123456789abcdef" for character in transaction_id)
    ):
        raise LedgerPoisoned("finalization manifest transaction ID is invalid")
    state = manifest.get("state")
    if type(state) is not str or state not in {"committing", "committed"}:
        raise LedgerPoisoned("finalization manifest state is invalid")
    manifest_path = root / _finalization_manifest_name(transaction_id)
    if manifest_path.name != _finalization_manifest_name(transaction_id):
        raise LedgerPoisoned("finalization manifest path is invalid")
    request = _hex_bytes(
        manifest.get("request_hex"),
        "finalization request",
        maximum=_MAX_FINALIZATION_REQUEST_BYTES,
    )
    spec, request_sha256 = _spec_from_request(request, root, require_action=False)
    if (
        spec.transaction_id != transaction_id
        or request_sha256 != manifest.get("request_sha256")
        or spec.ledger_id != manifest.get("ledger_id")
        or spec.genesis_sha256 != manifest.get("genesis_sha256")
        or spec.base_head_sha256 != manifest.get("base_head_sha256")
        or spec.operation != manifest.get("operation")
    ):
        raise LedgerPoisoned("finalization manifest request binding changed")
    baseline = _baseline_payload(cast(Sequence[Mapping[str, object]], manifest["baseline_paths"]))
    if manifest.get("baseline_sha256") != _baseline_digest(baseline):
        raise LedgerPoisoned("finalization baseline digest changed")
    result = _validate_finalization_result(spec, manifest["result"], spec.base_record_count)
    if manifest.get("result_sha256") != result["result"]["result_sha256"]:
        raise LedgerPoisoned("finalization manifest result binding changed")
    return spec, result, baseline


def _read_transaction_manifest(
    root: Path,
    path: Path,
) -> tuple[
    dict[str, Any],
    _FinalizationTransactionSpec,
    dict[str, Any],
    list[dict[str, int | str]],
]:
    if _is_link_or_reparse(path) or not path.is_file():
        raise LedgerPoisoned("finalization manifest is not a regular file")
    value = _strict_canonical_object(path.read_bytes(), "finalization manifest")
    spec, result, baseline = _validate_manifest_state(root, value)
    if path.name != _finalization_manifest_name(spec.transaction_id):
        raise LedgerPoisoned("finalization manifest filename is not transaction-specific")
    return value, spec, result, baseline


def _current_output_map(root: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for directory in ("records", "evidence"):
        base = root / directory
        if not base.is_dir() or _is_link_or_reparse(base):
            raise LedgerPoisoned("finalization output directory changed")
        for path in base.iterdir():
            if _is_link_or_reparse(path) or not path.is_file():
                raise LedgerPoisoned("finalization output contains a non-file")
            if getattr(path.lstat(), "st_nlink", 1) != 1:
                raise LedgerPoisoned("finalization output contains an aliased file")
            result[f"{directory}/{path.name}"] = path.read_bytes()
    return result


def _validate_manifest_outputs(
    root: Path,
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    baseline: Sequence[Mapping[str, object]],
    *,
    committed: bool,
) -> None:
    current = _current_output_map(root)
    expected_baseline = {
        cast(str, item["relative_path"]): item for item in baseline
    }
    patches = [
        *cast(Sequence[Mapping[str, Any]], result["evidence_patches"]),
        *cast(Sequence[Mapping[str, Any]], result["record_patches"]),
    ]
    expected_patches = {cast(str, item["relative_path"]): item for item in patches}
    expected_temp = {cast(str, item["temporary_relative_path"]): item for item in patches}
    # A committing transaction may have durable patch temporaries while the
    # child is between its manifest and final replacement.  They are allowed
    # only when their exact transaction-specific names are declared below;
    # committed state must still contain no temporaries.
    if set(current) - set(expected_baseline) - set(expected_patches) - set(expected_temp):
        raise LedgerPoisoned("finalization output contains an unbound path")
    for relative, item in expected_baseline.items():
        encoded = current.get(relative)
        if encoded is None or len(encoded) != item["bytes"] or _sha256(encoded) != item["sha256"]:
            raise LedgerPoisoned("finalization baseline was changed")
    for relative, item in expected_patches.items():
        if relative in expected_baseline:
            raise LedgerPoisoned("finalization patch replaces a baseline path")
        path = _safe_output_path(root, relative)
        encoded = current.get(relative)
        if committed:
            if encoded != item["content"]:
                raise LedgerPoisoned("committed finalization output differs from its patch")
        elif encoded is not None and encoded != item["content"]:
            raise LedgerPoisoned("partial finalization output differs from its patch")
        if path.exists() and _is_link_or_reparse(path):
            raise LedgerPoisoned("finalization output is a symlink")
    for temporary in expected_temp:
        if temporary in expected_baseline:
            raise LedgerPoisoned("finalization temporary path was present in the baseline")
        path = _safe_output_path(root, temporary)
        if path.exists():
            if committed:
                raise LedgerPoisoned("committed finalization left a temporary output")
            if _is_link_or_reparse(path):
                raise LedgerPoisoned("finalization temporary output is a symlink")


def _cleanup_request_bytes(spec: _FinalizationTransactionSpec, action: str) -> bytes:
    if action not in {"resolve", "ack"}:
        raise ValueError("unknown finalization cleanup action")
    payload = {
        "schema_version": _FINALIZATION_TRANSACTION_SCHEMA_VERSION,
        "domain": _FINALIZATION_CLEANUP_DOMAIN,
        "transaction_id": spec.transaction_id,
        "root_path": _canonical_root_path(spec.root),
        "root_binding_sha256": _root_binding(spec.root),
        "request_sha256": _sha256(_finalization_request_bytes(spec)),
        "action": action,
        "request_hex": _finalization_request_bytes(spec).hex(),
    }
    _strict_keyset(payload, _FINALIZATION_CLEANUP_KEYS, "finalization cleanup request")
    encoded = cast(bytes, canonical_json(payload))
    if len(encoded) > _MAX_FINALIZATION_CLEANUP_BYTES:
        raise LedgerPoisoned("finalization cleanup request exceeds its bound")
    return encoded


def _remove_owned_transaction_file(path: Path) -> None:
    if _is_link_or_reparse(path) or not path.is_file():
        raise LedgerPoisoned("transaction-owned path is not a regular file")
    if getattr(path.lstat(), "st_nlink", 1) != 1:
        raise LedgerPoisoned("transaction-owned path has an alias")
    path.unlink()


def _rollback_manifest(
    root: Path,
    result: Mapping[str, Any],
    baseline: Sequence[Mapping[str, object]],
) -> None:
    baseline_paths = {cast(str, item["relative_path"]) for item in baseline}
    patches = [
        *cast(Sequence[Mapping[str, Any]], result["evidence_patches"]),
        *cast(Sequence[Mapping[str, Any]], result["record_patches"]),
    ]
    removals: list[Path] = []
    for patch in patches:
        relative = cast(str, patch["relative_path"])
        temporary = cast(str, patch["temporary_relative_path"])
        if relative in baseline_paths or temporary in baseline_paths:
            raise LedgerPoisoned("rollback path was present in the immutable baseline")
        final_path = _safe_output_path(root, relative)
        temporary_path = _safe_output_path(root, temporary)
        if final_path.exists():
            if (
                _is_link_or_reparse(final_path)
                or getattr(final_path.lstat(), "st_nlink", 1) != 1
                or final_path.read_bytes() != patch["content"]
            ):
                raise LedgerPoisoned("rollback encountered a changed final output")
            removals.append(final_path)
        if temporary_path.exists():
            if (
                _is_link_or_reparse(temporary_path)
                or getattr(temporary_path.lstat(), "st_nlink", 1) != 1
            ):
                raise LedgerPoisoned("rollback encountered a symlink temporary output")
            # A transaction-specific name is not ownership proof by itself:
            # an attacker could pre-create that name.  The child writes the
            # complete canonical patch before publishing the manifest, so the
            # exact bytes are the second, immutable ownership proof.  Reject
            # partial or substituted contents rather than deleting them.
            if temporary_path.read_bytes() != patch["content"]:
                raise LedgerPoisoned("rollback encountered an unbound temporary output")
            removals.append(temporary_path)
    # No filesystem mutation occurs until every baseline/path/ownership check
    # above has succeeded.
    for path in removals:
        _remove_owned_transaction_file(path)
    if os.name != "nt":
        _fsync_directory(root / "records")
        _fsync_directory(root / "evidence")


def _remove_manifest_and_stage(
    root: Path,
    manifest_path: Path,
    spec: _FinalizationTransactionSpec,
    manifest: Mapping[str, Any],
) -> None:
    if manifest_path.name != _finalization_manifest_name(spec.transaction_id):
        raise LedgerPoisoned("finalization manifest filename is not transaction-specific")
    commit_temp = manifest_path.with_name(manifest_path.name + ".commit.tmp")
    if commit_temp.exists():
        if _is_link_or_reparse(commit_temp) or not commit_temp.is_file():
            raise LedgerPoisoned("finalization manifest temporary is invalid")
        if getattr(commit_temp.lstat(), "st_nlink", 1) != 1:
            raise LedgerPoisoned("finalization manifest temporary has an alias")
        committed_manifest = dict(manifest)
        committed_manifest["state"] = "committed"
        if commit_temp.read_bytes() != canonical_json(committed_manifest):
            raise LedgerPoisoned("finalization manifest temporary is not transaction-owned")
    stage = _transaction_stage_path(spec)
    if stage.exists():
        if _is_link_or_reparse(stage) or not stage.is_dir():
            raise LedgerPoisoned("finalization stage is not a real directory")
        marker = stage / ".transaction-owner"
        if _is_link_or_reparse(marker) or not marker.is_file():
            raise LedgerPoisoned("finalization stage owner marker is invalid")
        if marker.read_bytes() != _sha256(_finalization_request_bytes(spec)).encode("ascii"):
            raise LedgerPoisoned("finalization stage owner binding changed")
    _remove_owned_transaction_file(manifest_path)
    if commit_temp.exists():
        commit_temp.unlink()
    if stage.exists():
        shutil.rmtree(stage)
    if os.name != "nt":
        _fsync_directory(root)
        _fsync_directory(root.parent)


def _resolve_stale_finalization_manifests(root: Path) -> None:
    for path in sorted(root.glob(".*.finalization.json"), key=lambda item: item.name):
        if _is_link_or_reparse(path) or not path.is_file():
            raise LedgerPoisoned("finalization manifest is not a regular file")
        manifest, spec, result, baseline = _read_transaction_manifest(root, path)
        state = manifest["state"]
        _validate_manifest_outputs(
            root,
            manifest,
            result,
            baseline,
            committed=state == "committed",
        )
        if state == "committed":
            _validate_transaction_spec_against_root(spec, root=root, require_base=False)
            _validate_committed_finalization(spec, result)
        else:
            _rollback_manifest(root, result, baseline)
        _remove_manifest_and_stage(root, path, spec, manifest)


async def _run_finalization_cleanup_owner(
    spec: _FinalizationTransactionSpec,
    *,
    action: Literal["resolve", "ack"],
) -> dict[str, Any]:
    request = _cleanup_request_bytes(spec, action)
    process = await asyncio.create_subprocess_exec(
        *_finalization_child_command("--cleanup-finalization-stdin", spec.root),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(request),
            timeout=_FINALIZATION_KILL_WAIT_SECONDS,
        )
    except BaseException:
        await _terminate_finalization_process(process)
        raise
    if process.returncode != 0:
        raise LedgerPoisoned(
            "finalization cleanup owner failed"
            + (f": {stderr.decode('utf-8', errors='replace').strip()}" if stderr else "")
        )
    if len(stdout) > _MAX_FINALIZATION_CLEANUP_RESULT_BYTES:
        raise LedgerPoisoned("finalization cleanup result exceeds its bound")
    result = _strict_canonical_object(stdout, "finalization cleanup result")
    _strict_keyset(result, _FINALIZATION_CLEANUP_RESULT_KEYS, "finalization cleanup result")
    if (
        result.get("schema_version") != _FINALIZATION_TRANSACTION_SCHEMA_VERSION
        or result.get("domain") != _FINALIZATION_CLEANUP_DOMAIN
        or result.get("transaction_id") != spec.transaction_id
        or result.get("request_sha256") != _sha256(_finalization_request_bytes(spec))
        or result.get("state") not in {"committed", "rolled_back"}
    ):
        raise LedgerPoisoned("finalization cleanup result binding changed")
    return result


def _result_payload(
    spec: _FinalizationTransactionSpec,
    request_sha256: str,
    receipt: bytes,
    record_patches: Sequence[Mapping[str, Any]],
    evidence_patches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    core = {
        "schema_version": 2,
        "domain": _FINALIZATION_RESULT_DOMAIN,
        "transaction_id": spec.transaction_id,
        "request_sha256": request_sha256,
        "receipt_hex": receipt.hex(),
        "record_patches": [dict(patch) for patch in record_patches],
        "evidence_patches": [dict(patch) for patch in evidence_patches],
    }
    return {**core, "result_sha256": _sha256(canonical_json(core))}


def _result_bytes(result: Mapping[str, Any]) -> bytes:
    return cast(bytes, canonical_json(result))


def _emit_finalization_output(encoded: bytes, *, maximum: int) -> None:
    if len(encoded) > maximum:
        raise LedgerPoisoned("finalization child output exceeds its bound")
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def _validate_finalization_result(
    spec: _FinalizationTransactionSpec,
    encoded: bytes | Mapping[str, Any],
    baseline_record_count: int | None = None,
) -> dict[str, Any]:
    if isinstance(encoded, bytes) and len(encoded) > _MAX_FINALIZATION_RESULT_BYTES:
        raise LedgerPoisoned("finalization result exceeds its bound")
    value = (
        _strict_canonical_object(encoded, "finalization result")
        if isinstance(encoded, bytes)
        else dict(encoded)
    )
    _strict_keyset(value, _FINALIZATION_RESULT_KEYS, "finalization result")
    if (
        value.get("schema_version") != 2
        or value.get("domain") != _FINALIZATION_RESULT_DOMAIN
        or value.get("transaction_id") != spec.transaction_id
        or value.get("request_sha256") != _sha256(_finalization_request_bytes(spec))
    ):
        raise LedgerPoisoned("finalization result binding changed")
    result_sha256 = value.get("result_sha256")
    core = dict(value)
    core.pop("result_sha256", None)
    if result_sha256 != _sha256(canonical_json(core)):
        raise LedgerPoisoned("finalization result digest changed")
    receipt = _hex_bytes(
        value.get("receipt_hex"), "finalization receipt", maximum=_MAX_RECEIPT_BYTES
    )
    receipt_value = _strict_canonical_object(receipt, "finalization receipt")
    expected_kind = "reconstruction_qa" if spec.operation == "qa" else "scientific_arm_execution"
    if receipt_value.get("receipt_kind") != expected_kind:
        raise LedgerPoisoned("finalization receipt kind is inconsistent")
    records = _transaction_patch_payloads(
        cast(Sequence[Mapping[str, Any]], value["record_patches"]),
        transaction_id=spec.transaction_id,
        directory="records",
    )
    evidence = _transaction_patch_payloads(
        cast(Sequence[Mapping[str, Any]], value["evidence_patches"]),
        transaction_id=spec.transaction_id,
        directory="evidence",
    )
    if not records:
        raise LedgerPoisoned("finalization result has no record patch")
    expected_offset = (
        spec.base_record_count if baseline_record_count is None else baseline_record_count
    )
    if type(expected_offset) is not int or expected_offset != spec.base_record_count:
        raise LedgerPoisoned("finalization baseline record count changed")
    prior = spec.base_head_sha256
    receipt_found = False
    for index, patch in enumerate(records):
        record = _strict_canonical_object(cast(bytes, patch["content"]), "finalization record")
        if (
            record.get("ledger_id") != spec.ledger_id
            or record.get("offset") != expected_offset + index
            or record.get("prior_record_sha256") != prior
        ):
            raise LedgerPoisoned("finalization record chain differs from the authenticated base")
        prior = _sha256(cast(bytes, patch["content"]))
        if record.get("record_kind") == "receipt":
            body = _body(record)
            if (
                body.get("receipt") == receipt_value
                and body.get("receipt_sha256") == _sha256(receipt)
            ):
                receipt_found = True
    if not receipt_found:
        raise LedgerPoisoned("finalization receipt is not in the new record patch")
    return {
        "receipt": receipt,
        "record_patches": records,
        "evidence_patches": evidence,
        "result": value,
    }


def _validate_committed_finalization(
    spec: _FinalizationTransactionSpec,
    result: Mapping[str, Any],
) -> None:
    """Bind a committed transaction to the reopened ledger, not its manifest."""

    scan = inspect_ledger(spec.root)
    if scan.status != "active-clean":
        raise LedgerPoisoned("committed finalization is not active-clean")
    records = cast(Sequence[Mapping[str, Any]], result["record_patches"])
    expected_record_hashes = tuple(
        _sha256(cast(bytes, patch["content"])) for patch in records
    )
    if (
        len(scan.records) != spec.base_record_count + len(records)
        or tuple(scan.record_sha256s[spec.base_record_count:]) != expected_record_hashes
    ):
        raise LedgerPoisoned("committed finalization record chain is not anchored")
    receipt = cast(bytes, result["receipt"])
    receipt_value = _strict_canonical_object(receipt, "finalization receipt")
    receipt_kind = cast(str, receipt_value["receipt_kind"])
    if scan.receipts.get((receipt_kind, _sha256(receipt))) != receipt_value:
        raise LedgerPoisoned("committed finalization receipt is not ledger-anchored")
    for patch in cast(Sequence[Mapping[str, Any]], result["evidence_patches"]):
        relative = cast(str, patch["relative_path"])
        digest = relative.split("/", 1)[1]
        evidence_path = _safe_output_path(spec.root, relative)
        if (
            digest not in scan.evidence_refs
            or not evidence_path.is_file()
            or _is_link_or_reparse(evidence_path)
            or evidence_path.read_bytes() != cast(bytes, patch["content"])
        ):
            raise LedgerPoisoned("committed finalization evidence is not anchored")


def _finalization_stage_glob(spec: _FinalizationTransactionSpec) -> str:
    return f".{spec.transaction_id}.stage-d-finalization-copy-*"


def _cleanup_finalization_stages(spec: _FinalizationTransactionSpec) -> None:
    removed = False
    for path in spec.root.parent.glob(_finalization_stage_glob(spec)):
        if path.is_dir() and not _is_link_or_reparse(path):
            shutil.rmtree(path)
            removed = True
    if removed and os.name != "nt":
        _fsync_directory(spec.root.parent)


def _run_finalization_transaction_worker(spec: _FinalizationTransactionSpec) -> None:
    _validate_transaction_spec_against_root(spec, root=spec.root)
    baseline_paths = _transaction_baseline_paths(spec.root)
    stage_directory = _transaction_stage_path(spec)
    if stage_directory.exists() or _is_link_or_reparse(stage_directory):
        raise LedgerPoisoned("finalization stage path already exists")
    manifest_path = _finalization_manifest_path(spec)
    if manifest_path.exists() or _is_link_or_reparse(manifest_path):
        raise LedgerPoisoned("finalization manifest path already exists")
    if _transaction_manifest_commit_temp(spec).exists():
        raise LedgerPoisoned("finalization manifest temporary path already exists")
    request = _finalization_request_bytes(spec)
    request_sha256 = _sha256(request)
    stage_directory.mkdir()
    if os.name != "nt":
        _fsync_directory(spec.root.parent)
    (stage_directory / ".transaction-owner").write_bytes(request_sha256.encode("ascii"))
    if os.name != "nt":
        _fsync_directory(stage_directory)
    stage_root = stage_directory / "ledger"
    try:
        stage_root.mkdir()
        shutil.copytree(spec.root / "records", stage_root / "records", symlinks=False)
        shutil.copytree(spec.root / "evidence", stage_root / "evidence", symlinks=False)
        staged = StageDReceiptLedger(stage_root, master_seed=spec.master_seed)
        try:
            if spec.operation == "qa":
                if (
                    spec.recorded_action is None
                    or spec.passed is None
                    or spec.actual_cost is None
                ):
                    raise LedgerError("QA finalization request is incomplete")
                report_sha256 = staged.put_evidence(spec.evidence)
                receipt = staged.record_reconstruction_qa(
                    group_id=spec.group_id,
                    target_id=spec.target_id,
                    recorded_action=spec.recorded_action,
                    passed=spec.passed,
                    report_sha256=report_sha256,
                    actual_cost=spec.actual_cost,
                )
            else:
                if spec.execution_attempt is None or spec.outcome_kind is None:
                    raise LedgerError("execution finalization request is incomplete")
                evidence_sha256 = staged.put_evidence(spec.evidence)
                receipt = staged.finish_execution(
                    spec.execution_attempt,
                    outcome_kind=spec.outcome_kind,
                    scored_reward=spec.scored_reward,
                    scorer_evidence_sha256=evidence_sha256,
                    latency_seconds=spec.latency_seconds,
                    dollars=spec.dollars,
                    judge_calls=spec.judge_calls,
                    cpu_seconds=spec.cpu_seconds,
                    gpu_seconds=spec.gpu_seconds,
                    wall_seconds=spec.wall_seconds,
                    storage_bytes=spec.storage_bytes,
                )
        finally:
            staged.close()
        base_records = _transaction_file_map(spec.root, "records")
        base_evidence = _transaction_file_map(spec.root, "evidence")
        prepared_records = _transaction_file_map(stage_root, "records")
        prepared_evidence = _transaction_file_map(stage_root, "evidence")
        record_patches = _transaction_patch(spec, base_records, prepared_records, "records")
        evidence_patches = _transaction_patch(spec, base_evidence, prepared_evidence, "evidence")
        if not record_patches:
            raise LedgerError("finalization transaction did not prepare a ledger record")
        result = _result_payload(
            spec,
            request_sha256,
            receipt,
            record_patches,
            evidence_patches,
        )
        committing = _transaction_manifest(
            spec,
            state="committing",
            request=request,
            baseline_paths=baseline_paths,
            result=result,
        )
        _exclusive_durable_write(manifest_path, committing)
        for patch in evidence_patches:
            relative = cast(str, patch["relative_path"])
            _atomic_record_write(
                _safe_output_path(spec.root, relative),
                prepared_evidence[Path(relative).name],
                temporary_path=spec.root / cast(str, patch["temporary_relative_path"]),
            )
        for patch in record_patches:
            relative = cast(str, patch["relative_path"])
            _atomic_record_write(
                _safe_output_path(spec.root, relative),
                prepared_records[Path(relative).name],
                temporary_path=spec.root / cast(str, patch["temporary_relative_path"]),
            )
        if inspect_ledger(spec.root).status != "active-clean":
            raise LedgerPoisoned("prepared finalization did not produce a valid ledger")
        committed = _transaction_manifest(
            spec,
            state="committed",
            request=request,
            baseline_paths=baseline_paths,
            result=result,
        )
        _replace_durable_bytes(manifest_path, committed)
        _emit_finalization_output(
            _result_bytes(result), maximum=_MAX_FINALIZATION_RESULT_BYTES
        )
    finally:
        shutil.rmtree(stage_directory, ignore_errors=True)


async def _terminate_finalization_process(
    process: asyncio.subprocess.Process,
) -> None:
    """Terminate and reap one isolated commit process within a fixed bound."""

    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(
            process.wait(),
            timeout=_FINALIZATION_TERM_GRACE_SECONDS,
        )
        return
    except TimeoutError:
        process.kill()
        await asyncio.wait_for(
            process.wait(),
            timeout=_FINALIZATION_KILL_WAIT_SECONDS,
        )


def _exclusive_durable_write(path: Path, encoded: bytes) -> None:
    if path.exists() or _is_link_or_reparse(path):
        raise FileExistsError(path)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short durable transaction write")
            view = view[written:]
        os.fsync(descriptor)
        if os.name != "nt":
            _fsync_directory(path.parent)
    except BaseException:
        try:
            os.close(descriptor)
        finally:
            if path.exists() and not _is_link_or_reparse(path):
                path.unlink()
        raise
    else:
        os.close(descriptor)


def _replace_durable_bytes(path: Path, encoded: bytes) -> None:
    if not path.is_file() or _is_link_or_reparse(path):
        raise LedgerPoisoned("transaction manifest is not replaceable")
    temporary = path.with_name(path.name + ".commit.tmp")
    _exclusive_durable_write(temporary, encoded)
    os.replace(temporary, path)
    if os.name != "nt":
        _fsync_directory(path.parent)


def _atomic_record_write(
    path: Path,
    encoded: bytes,
    *,
    temporary_path: Path | None = None,
) -> None:
    if path.exists():
        raise FileExistsError(path)
    if _is_link_or_reparse(path):
        raise LedgerPoisoned("transaction output is a symlink")
    if temporary_path is None:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
    else:
        if temporary_path.exists() or _is_link_or_reparse(temporary_path):
            raise LedgerPoisoned("transaction temporary path already exists")
        descriptor = os.open(
            temporary_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        temporary = temporary_path
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _durable_rename(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_blob_write(path: Path, data: bytes) -> None:
    _atomic_record_write(path, data)


def _acquire_writer_lock(path: Path) -> int:
    created = False
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        created = True
    except FileExistsError:
        descriptor = os.open(path, os.O_RDWR)
    try:
        if created or os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"0")
            os.fsync(descriptor)
            if os.name != "nt":
                _fsync_directory(path.parent)
        os.lseek(descriptor, 0, os.SEEK_SET)
        _acquire_os_lock(descriptor)
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _acquire_os_lock(descriptor: int) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            fcntl = importlib.import_module("fcntl")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        raise LedgerError("ledger already has an exclusive writer") from error


def _release_os_lock(descriptor: int) -> None:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            fcntl = importlib.import_module("fcntl")
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _durable_rename(source: Path, destination: Path) -> None:
    """Atomically publish a file and durably commit its directory entry."""
    if destination.exists():
        raise FileExistsError(destination)
    if os.name == "nt":
        import ctypes

        move_file = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        move_file.restype = ctypes.c_int
        movefile_write_through = 0x8
        if not move_file(str(source), str(destination), movefile_write_through):
            raise ctypes.WinError(ctypes.get_last_error())
        return
    os.rename(source, destination)
    _fsync_directory(destination.parent)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        raise LedgerError("directory fsync is unsupported; use durable Windows rename")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _record_name(offset: int) -> str:
    return f"{offset:020d}.json"


def _valid_record_name(name: str) -> bool:
    return len(name) == 25 and name[:20].isdigit() and name.endswith(".json")


def _strict_canonical_object(encoded: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LedgerPoisoned(f"{name} is not JSON") from error
    if not isinstance(value, dict) or canonical_json(value) != encoded:
        raise LedgerPoisoned(f"{name} is not canonical JSON")
    return value


def _cleanup_result(
    spec: _FinalizationTransactionSpec,
    request_sha256: str,
    state: str,
    result: Mapping[str, Any] | None,
) -> bytes:
    payload = {
        "schema_version": _FINALIZATION_TRANSACTION_SCHEMA_VERSION,
        "domain": _FINALIZATION_CLEANUP_DOMAIN,
        "transaction_id": spec.transaction_id,
        "request_sha256": request_sha256,
        "state": state,
        "result": None if result is None else dict(result),
    }
    _strict_keyset(payload, _FINALIZATION_CLEANUP_RESULT_KEYS, "finalization cleanup result")
    encoded = cast(bytes, canonical_json(payload))
    if len(encoded) > _MAX_FINALIZATION_CLEANUP_RESULT_BYTES:
        raise LedgerPoisoned("finalization cleanup result exceeds its bound")
    return encoded


def _cleanup_owned_stage_if_present(spec: _FinalizationTransactionSpec) -> None:
    stage = _transaction_stage_path(spec)
    if not stage.exists():
        return
    if _is_link_or_reparse(stage) or not stage.is_dir():
        raise LedgerPoisoned("finalization stage is not a real directory")
    marker = stage / ".transaction-owner"
    if not marker.is_file() or _is_link_or_reparse(marker):
        raise LedgerPoisoned("unbound finalization stage cannot be removed")
    if marker.read_bytes() != _sha256(_finalization_request_bytes(spec)).encode("ascii"):
        raise LedgerPoisoned("finalization stage owner binding changed")
    shutil.rmtree(stage)
    if os.name != "nt":
        _fsync_directory(stage.parent)


def _run_finalization_cleanup_worker(encoded: bytes, root: Path) -> None:
    if len(encoded) > _MAX_FINALIZATION_CLEANUP_BYTES:
        raise LedgerPoisoned("finalization cleanup request exceeds its bound")
    value = _strict_canonical_object(encoded, "finalization cleanup request")
    _strict_keyset(value, _FINALIZATION_CLEANUP_KEYS, "finalization cleanup request")
    if (
        value.get("schema_version") != _FINALIZATION_TRANSACTION_SCHEMA_VERSION
        or value.get("domain") != _FINALIZATION_CLEANUP_DOMAIN
        or value.get("root_path") != _canonical_root_path(root)
        or value.get("root_binding_sha256") != _root_binding(root)
    ):
        raise LedgerPoisoned("finalization cleanup root binding changed")
    request = _hex_bytes(
        value.get("request_hex"),
        "finalization cleanup request payload",
        maximum=_MAX_FINALIZATION_REQUEST_BYTES,
    )
    spec, request_sha256 = _spec_from_request(request, root, require_action=False)
    if (
        value.get("transaction_id") != spec.transaction_id
        or value.get("request_sha256") != request_sha256
    ):
        raise LedgerPoisoned("finalization cleanup request binding changed")
    action = value.get("action")
    if action not in {"resolve", "ack"}:
        raise LedgerPoisoned("finalization cleanup action is invalid")
    manifest_path = _finalization_manifest_path(spec)
    if not manifest_path.exists():
        _cleanup_owned_stage_if_present(spec)
        _emit_finalization_output(
            _cleanup_result(spec, request_sha256, "rolled_back", None),
            maximum=_MAX_FINALIZATION_CLEANUP_RESULT_BYTES,
        )
        return
    manifest, manifest_spec, result, baseline = _read_transaction_manifest(root, manifest_path)
    if manifest_spec != spec:
        raise LedgerPoisoned("finalization cleanup manifest spec changed")
    state = manifest["state"]
    if state == "committed":
        _validate_transaction_spec_against_root(spec, root=root, require_base=False)
        _validate_manifest_outputs(root, manifest, result, baseline, committed=True)
        _validate_committed_finalization(spec, result)
        if action == "resolve":
            cleanup_output = _cleanup_result(
                spec, request_sha256, "committed", result["result"]
            )
            _emit_finalization_output(
                cleanup_output,
                maximum=_MAX_FINALIZATION_CLEANUP_RESULT_BYTES,
            )
            return
        cleanup_output = _cleanup_result(
            spec, request_sha256, "committed", result["result"]
        )
        _remove_manifest_and_stage(root, manifest_path, spec, manifest)
        _emit_finalization_output(
            cleanup_output,
            maximum=_MAX_FINALIZATION_CLEANUP_RESULT_BYTES,
        )
        return
    if action == "ack":
        raise LedgerPoisoned("uncommitted finalization cannot be acknowledged")
    _validate_manifest_outputs(root, manifest, result, baseline, committed=False)
    cleanup_output = _cleanup_result(spec, request_sha256, "rolled_back", None)
    _rollback_manifest(root, result, baseline)
    _remove_manifest_and_stage(root, manifest_path, spec, manifest)
    _emit_finalization_output(
        cleanup_output,
        maximum=_MAX_FINALIZATION_CLEANUP_RESULT_BYTES,
    )


def _finalization_transaction_main(argv: Sequence[str]) -> int:
    if len(argv) != 2 or argv[0] not in {
        "--finalize-transaction-stdin",
        "--cleanup-finalization-stdin",
    }:
        return 2
    root = Path(argv[1])
    try:
        maximum = (
            _MAX_FINALIZATION_CLEANUP_BYTES
            if argv[0] == "--cleanup-finalization-stdin"
            else _MAX_FINALIZATION_REQUEST_BYTES
        )
        encoded = sys.stdin.buffer.read(maximum + 1)
        if not encoded:
            raise LedgerPoisoned("finalization child received no request")
        if len(encoded) > maximum:
            raise LedgerPoisoned("finalization child request exceeds its bound")
        if argv[0] == "--finalize-transaction-stdin":
            spec, _ = _spec_from_request(encoded, root)
            _run_finalization_transaction_worker(spec)
        else:
            _run_finalization_cleanup_worker(encoded, root)
        return 0
    except BaseException as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


def _body(record: Mapping[str, Any]) -> dict[str, Any]:
    body = record.get("body")
    if not isinstance(body, dict):
        raise LedgerPoisoned("ledger record body must be an object")
    return body


def _event(body: Mapping[str, Any]) -> dict[str, Any]:
    event = body.get("event")
    if not isinstance(event, dict):
        raise LedgerPoisoned("event record lacks an event object")
    return event


def _group_key(group_id: str, target_id: str) -> tuple[str, str]:
    if not group_id or not target_id:
        raise ValueError("group_id and target_id must be nonempty")
    return group_id, target_id


def _address_payload(address: PolicyEventAddress) -> dict[str, str | int]:
    return {**address.as_payload(), "turn": address.turn}


def _scientific_address_key(address: PolicyEventAddress) -> bytes:
    return canonical_json(address.as_payload())


def _execution_address_used(state: Mapping[str, Any], address: PolicyEventAddress) -> bool:
    key = _scientific_address_key(address)
    return any(
        _scientific_address_key(existing) == key
        for existing in (
            *state["in_flight"],
            *state["overrides"],
            *(call["address"] for call in state["calls"]),
        )
    )


def _scanned_address_key(value: object) -> bytes:
    return _scientific_address_key(_scanned_policy_address(value))


def _scanned_policy_address(value: object) -> PolicyEventAddress:
    if not isinstance(value, dict) or set(value) != {
        "depth",
        "lineage",
        "session_call_ordinal",
        "turn",
        "call_kind",
    }:
        raise LedgerPoisoned("ledger event contains an invalid policy address")
    try:
        address = PolicyEventAddress(
            cast(int, value["depth"]),
            cast(str, value["lineage"]),
            cast(int, value["session_call_ordinal"]),
            cast(int, value["turn"]),
            cast(str, value["call_kind"]),
        )
    except (TypeError, ValueError) as error:
        raise LedgerPoisoned("ledger event contains an invalid policy address") from error
    if _address_payload(address) != value:
        raise LedgerPoisoned("ledger event policy address changed types")
    return address


def _scanned_scheduled_cache_salt(event: Mapping[str, Any]) -> str:
    seed = event.get("seed")
    coupling = event.get("coupling_mode")
    if type(seed) is not int or seed < 0 or coupling not in {"paired", "exogenous"}:
        raise LedgerPoisoned("execution model call has an invalid scheduled seed")
    address = _scanned_policy_address(event.get("address"))
    return ScheduledSeed(seed, CouplingMode(coupling), address).cache_salt


def _actual_cost_payload(cost: ActualEvaluationCost) -> dict[str, int | float]:
    return {
        "generated_tokens": cost.generated_tokens,
        "judge_calls": cost.judge_calls,
        "cpu_seconds": cost.cpu_seconds,
        "gpu_seconds": cost.gpu_seconds,
        "wall_seconds": cost.wall_seconds,
        "storage_bytes": cost.storage_bytes,
    }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return str(value)


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not (-float("inf") < value < float("inf")):
        raise ValueError(f"{name} must be a finite float")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


if __name__ == "__main__":
    raise SystemExit(_finalization_transaction_main(sys.argv[1:]))
