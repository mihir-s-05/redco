"""Durable, fail-closed receipt ledger for Stage D scientific execution."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import secrets
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Literal, cast

from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.analysis.stage_d_scientific_branch_group import (
    OutcomeKind,
    behavior_law_digest,
)
from redco.analysis.stage_d_spawn_provenance import (
    EventSeedScheduler,
    PolicyEventAddress,
    ScheduledSeed,
)
from redco.contracts import ActualEvaluationCost, LogicalDeploymentCost, canonical_json

_DOMAIN = "redco-stage-d-receipt-ledger-v1"
_GENESIS_PRIOR = "0" * 64
_MAX_RECEIPT_BYTES = 1 << 20
_MAX_RECORD_BYTES = 2 << 20
_RECORD_KEYS = {
    "schema_version",
    "domain",
    "ledger_id",
    "offset",
    "prior_record_sha256",
    "record_kind",
    "body",
}

RecoveryStatus = Literal["active-clean", "sealed-valid", "poisoned"]
FaultHook = Callable[[str, str], None]


def _writer_transaction[**P, R](method: Callable[P, R]) -> Callable[P, R]:
    """Serialize a complete public writer transition, including its state checks."""

    @wraps(method)
    def locked(*args: P.args, **kwargs: P.kwargs) -> R:
        owner = cast(Any, args[0])
        with owner._state_lock:
            return method(*args, **kwargs)

    return locked


class LedgerError(RuntimeError):
    """The ledger cannot safely continue or verify."""


class LedgerPoisoned(LedgerError):
    """A torn, ambiguous, or scientifically dangling ledger was observed."""


class BatchAlreadyClaimed(LedgerError):
    """A training batch already has a durable single-use claim."""


@dataclass(frozen=True, slots=True)
class GenesisBinding:
    preregistration_sha256: str
    source_sha256: str
    runtime_sha256: str
    config_sha256: str
    master_seed_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "preregistration_sha256",
            "source_sha256",
            "runtime_sha256",
            "config_sha256",
            "master_seed_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    def to_payload(self) -> dict[str, str]:
        return {
            "preregistration_sha256": self.preregistration_sha256,
            "source_sha256": self.source_sha256,
            "runtime_sha256": self.runtime_sha256,
            "config_sha256": self.config_sha256,
            "master_seed_sha256": self.master_seed_sha256,
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


@dataclass(frozen=True, slots=True)
class CandidateAttempt:
    ledger_id: str
    group_id: str
    target_id: str
    action_slot: int
    action_seed: int
    attempt_id: str


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
class ExecutionAttempt:
    ledger_id: str
    group_id: str
    target_id: str
    arm_id: str
    action_digest: str
    continuation_replicate: int
    attempt_id: str


@dataclass(frozen=True, slots=True)
class ModelCallAttempt:
    execution_attempt_id: str
    call_id: str
    address: PolicyEventAddress
    scheduled_seed: ScheduledSeed


@dataclass(frozen=True, slots=True)
class _ScanResult:
    status: RecoveryStatus
    reason: str | None
    records: tuple[dict[str, Any], ...]
    record_sha256s: tuple[str, ...]
    receipts: Mapping[tuple[str, str], dict[str, Any]]
    evidence_refs: frozenset[str]
    seal: LedgerSeal | None


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

    def __init__(
        self,
        root: Path,
        *,
        master_seed: str,
        fault_hook: FaultHook | None = None,
    ) -> None:
        self._root = root
        self._records_dir = root / "records"
        self._evidence_dir = root / "evidence"
        self._lock_path = root / "writer.lock"
        self._fault_hook = fault_hook
        self._state_lock = threading.RLock()
        self._closed = False
        self._poisoned = False
        self._lock_descriptor: int | None = _acquire_writer_lock(self._lock_path)
        try:
            scan = inspect_ledger(root)
            if scan.status != "active-clean":
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
        fault_hook: FaultHook | None = None,
    ) -> StageDReceiptLedger:
        if not master_seed:
            raise ValueError("master_seed must be nonempty")
        if binding.master_seed_sha256 != _sha256(master_seed.encode("utf-8")):
            raise ValueError("binding does not match master_seed")
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
        _atomic_record_write(root / "records" / _record_name(0), canonical_json(genesis), None)
        return cls(root, master_seed=master_seed, fault_hook=fault_hook)

    @property
    def ledger_id(self) -> str:
        return self._ledger_id

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def head_sha256(self) -> str:
        return self._record_sha256s[-1]

    def __enter__(self) -> StageDReceiptLedger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @_writer_transaction
    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._release_lock()

    @_writer_transaction
    def __call__(self, receipt: bytes, *, receipt_kind: str) -> Mapping[str, Any]:
        self._require_writable()
        scan = inspect_ledger(self._root)
        if scan.status != "active-clean":
            self._poisoned = True
            raise LedgerPoisoned("live ledger is no longer active-clean")
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
                "pre_action_snapshot_sha256": commitment[
                    "pre_action_snapshot_sha256"
                ],
                "recorded_action_digest": recorded_action.digest,
                "matched_addresses": [
                    _address_payload(address)
                    for address in ordered_addresses
                ],
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
        key = self._require_committed(group_id, target_id)
        self._require_evidence(report_sha256)
        if key not in self._correspondence:
            raise LedgerError("correspondence must be frozen before reconstruction QA")
        if key in self._qa:
            raise LedgerError("reconstruction QA is already recorded")
        if type(passed) is not bool:
            raise ValueError("passed must be bool")
        if self._recorded_action_digests[key] != recorded_action.digest:
            raise ValueError("recorded action changed after commitment")
        commitment = self._commitments[key]
        receipt = self._append_receipt(
            "reconstruction_qa",
            {
                "group_id": group_id,
                "target_id": target_id,
                "pre_action_snapshot_sha256": commitment[
                    "pre_action_snapshot_sha256"
                ],
                "recorded_action_digest": recorded_action.digest,
                "passed": passed,
                "report_sha256": report_sha256,
                "actual_cost": _actual_cost_payload(actual_cost),
            },
            evidence_refs=(report_sha256,),
        )
        self._qa.add(key)
        self._qa_passed[key] = passed
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
            attempt_id,
        )
        self._append_event(
            "candidate_attempt",
            {
                "group_id": group_id,
                "target_id": target_id,
                "action_slot": action_slot,
                "action_seed": action_seed,
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
        if action.key.sampler.seed != attempt.action_seed:
            raise ValueError("candidate action seed differs from reserved seed")
        commitment = self._commitments[(attempt.group_id, attempt.target_id)]
        if behavior_law_digest(action.key) != commitment["behavior_law_sha256"]:
            raise ValueError("candidate action changed the frozen behavior law")
        self._append_event(
            "model_call_completed",
            {
                "attempt_kind": "candidate",
                "attempt_id": attempt.attempt_id,
                "call_id": state["call_id"],
                "action_digest": action.digest,
                "prompt_tokens": action.prompt_tokens,
                "completion_tokens": action.completion_tokens,
            },
            evidence_refs=(response_sha256,),
        )
        try:
            receipt = self._append_receipt(
                "candidate_action_inference",
                {
                    "group_id": attempt.group_id,
                    "target_id": attempt.target_id,
                    "action_slot": attempt.action_slot,
                    "action_seed": attempt.action_seed,
                    "action_digest": action.digest,
                    "behavior_law_sha256": behavior_law_digest(action.key),
                    "selection_policy": "direct_single_sample",
                    "sample_attempts": 1,
                    "rejected_attempts": 0,
                    "inference_call_id": state["call_id"],
                },
                evidence_refs=(response_sha256,),
            )
        except BaseException:
            self._poisoned = True
            raise
        slot_key = (attempt.group_id, attempt.target_id, attempt.action_slot)
        self._candidate_slots[slot_key] = action.digest
        del self._pending_candidates[slot_key]
        return receipt

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
                "attempt_id": attempt.attempt_id,
                "scientific_model_calls": 0,
                "reason": reason,
            },
            evidence_refs=(supervisor_evidence_sha256,),
        )
        slot_key = (attempt.group_id, attempt.target_id, attempt.action_slot)
        self._candidate_slots[slot_key] = "zero-call-failure"
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
        attempt = ExecutionAttempt(
            self._ledger_id,
            group_id,
            target_id,
            arm_id,
            action.digest,
            continuation_replicate,
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
                "attempt_id": attempt.attempt_id,
            },
            evidence_refs=(),
        )
        self._pending_executions[execution_key] = {
            "attempt": attempt,
            "action": action,
            "calls": [],
            "in_flight": None,
        }
        return attempt

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
        if state["in_flight"] is not None:
            raise LedgerError("one execution model call is already in flight")
        if any(call["address"] == address for call in state["calls"]):
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
                "coupling_mode": scheduled_seed.coupling_mode.value,
            },
            evidence_refs=(request_sha256,),
        )
        state["in_flight"] = call
        return call

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
        if state["in_flight"] != call:
            raise LedgerError("execution completion does not match the in-flight call")
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
        state["in_flight"] = None

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
        if state["in_flight"] is not None:
            raise LedgerError("cannot finish with a model call in flight")
        if not isinstance(outcome_kind, OutcomeKind):
            raise ValueError("outcome_kind must be OutcomeKind")
        calls = state["calls"]
        if outcome_kind is OutcomeKind.SUCCESS and not calls:
            raise ValueError("zero-call success must be terminal_without_downstream")
        if outcome_kind is OutcomeKind.TERMINAL_WITHOUT_DOWNSTREAM and calls:
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
        if (
            action.parse_status == "malformed"
            and outcome_kind is not OutcomeKind.MALFORMED_ACTION
        ):
            raise ValueError("malformed action must retain a malformed outcome")
        downstream_tokens = sum(call["completion_tokens"] for call in calls)
        logical = LogicalDeploymentCost(
            output_tokens=action.completion_tokens + downstream_tokens,
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
        del self._pending_executions[execution_key]
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
    def seal(self) -> LedgerSeal:
        self._require_writable()
        if (
            self._pending_candidates
            or self._pending_executions
            or any(
                not state["materialized"]
                for state in self._recorded_reservations.values()
            )
        ):
            raise LedgerPoisoned("cannot seal with dangling scientific attempts")
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

    def _append_receipt(
        self,
        receipt_kind: str,
        payload: Mapping[str, Any],
        *,
        evidence_refs: Sequence[str],
    ) -> bytes:
        if not receipt_kind:
            raise ValueError("receipt_kind must be nonempty")
        receipt = canonical_json(
            {"schema_version": 1, "receipt_kind": receipt_kind, **payload}
        )
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
                self._fault_hook,
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

    def _require_scientific_ready(self, group_id: str, target_id: str) -> tuple[str, str]:
        key = self._require_committed(group_id, target_id)
        if key not in self._correspondence or self._qa_passed.get(key) is not True:
            raise LedgerError("group target is not ready for scientific activity")
        return key

    def _candidate_state(self, attempt: CandidateAttempt) -> dict[str, Any]:
        if attempt.ledger_id != self._ledger_id:
            raise LedgerError("candidate attempt belongs to another ledger")
        key = (attempt.group_id, attempt.target_id, attempt.action_slot)
        state = self._pending_candidates.get(key)
        if state is None or state["attempt"] != attempt:
            raise LedgerError("candidate attempt is not active")
        return state

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
        self._candidate_slots: dict[tuple[str, str, int], str] = {}
        self._pending_candidates: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._executions: set[tuple[str, str, str, int]] = set()
        self._pending_executions: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        self._batch_claims: set[str] = set()
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
            if record["record_kind"] != "receipt":
                continue
            receipt = _body(record)["receipt"]
            kind = receipt["receipt_kind"]
            if kind == "pre_action_group_commitment":
                key = _group_key(receipt["group_id"], receipt["target_id"])
                self._commitments[key] = receipt
            elif kind == "seed_correspondence_map":
                key = _group_key(receipt["group_id"], receipt["target_id"])
                self._correspondence.add(key)
                self._recorded_action_digests[key] = receipt["recorded_action_digest"]
            elif kind == "reconstruction_qa":
                key = _group_key(receipt["group_id"], receipt["target_id"])
                self._qa.add(key)
                self._qa_passed[key] = receipt["passed"]
            elif kind == "candidate_action_inference":
                self._candidate_slots[
                    receipt["group_id"], receipt["target_id"], receipt["action_slot"]
                ] = receipt["action_digest"]
            elif kind == "zero_call_infrastructure_failure":
                self._candidate_slots[
                    receipt["group_id"], receipt["target_id"], receipt["action_slot"]
                ] = "zero-call-failure"
            elif kind == "scientific_arm_execution":
                self._executions.add(
                    (
                        receipt["group_id"],
                        receipt["target_id"],
                        receipt["arm_id"],
                        receipt["continuation_replicate"],
                    )
                )
            elif kind == "training_batch_consumption":
                self._batch_claims.add(receipt["training_batch_identity"])


def inspect_ledger(root: Path) -> _ScanResult:
    """Scan a ledger without repairing it and classify its recovery state."""
    try:
        return _scan_ledger(root)
    except BaseException as error:
        return _ScanResult("poisoned", str(error), (), (), {}, frozenset(), None)


def _scan_ledger(root: Path) -> _ScanResult:
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
    _validate_state_machine(records)
    seal = _seal_from_records(records, digests, len(receipts))
    status: RecoveryStatus = "sealed-valid" if seal is not None else "active-clean"
    return _ScanResult(
        status,
        None,
        tuple(records),
        tuple(digests),
        receipts,
        frozenset(evidence_refs),
        seal,
    )


def _validate_state_machine(records: Sequence[Mapping[str, Any]]) -> None:
    if records[0]["record_kind"] != "genesis" or records[0]["offset"] != 0:
        raise LedgerPoisoned("first record must be genesis")
    if any(record["record_kind"] == "genesis" for record in records[1:]):
        raise LedgerPoisoned("ledger contains multiple genesis records")
    if any(record["record_kind"] == "seal" for record in records[:-1]):
        raise LedgerPoisoned("seal must be the terminal record")
    commitments: dict[tuple[str, str], tuple[int, str]] = {}
    reservations: dict[tuple[str, str], dict[str, Any]] = {}
    recorded_action_materialized: set[str] = set()
    correspondence: set[tuple[str, str]] = set()
    candidate_attempts: dict[str, dict[str, Any]] = {}
    candidate_attempt_slots: set[tuple[str, str, int]] = set()
    execution_attempts: dict[str, dict[str, Any]] = {}
    execution_attempt_slots: set[tuple[str, str, str, int]] = set()
    starts: dict[str, dict[str, Any]] = {}
    completed: set[str] = set()
    finished_candidates: set[str] = set()
    finished_executions: set[str] = set()
    candidate_slots: set[tuple[str, str, int]] = set()
    execution_slots: set[tuple[str, str, str, int]] = set()
    batch_claims: set[str] = set()
    for record in records[1:]:
        kind = record["record_kind"]
        body = _body(record)
        if kind == "receipt":
            receipt = body["receipt"]
            receipt_kind = receipt["receipt_kind"]
            if receipt_kind == "pre_action_group_commitment":
                key = _group_key(receipt["group_id"], receipt["target_id"])
                if key in commitments:
                    raise LedgerPoisoned("duplicate pre-action commitment")
                if (
                    receipt["commitment_sequence"] != record["offset"]
                    or receipt["action_reservation_sequence"] != record["offset"] + 1
                ):
                    raise LedgerPoisoned("commitment/reservation ordering proof is invalid")
                commitments[key] = (record["offset"], body["receipt_sha256"])
            elif receipt_kind == "seed_correspondence_map":
                key = _group_key(receipt["group_id"], receipt["target_id"])
                if key not in reservations or key in correspondence:
                    raise LedgerPoisoned("correspondence is missing its unique reservation")
                if reservations[key]["reservation_id"] not in recorded_action_materialized:
                    raise LedgerPoisoned("correspondence predates the recorded action output")
                if any(
                    (attempt["group_id"], attempt["target_id"]) == key
                    for attempt in (*candidate_attempts.values(), *execution_attempts.values())
                ):
                    raise LedgerPoisoned("correspondence was frozen after scientific activity")
                correspondence.add(key)
            elif receipt_kind == "candidate_action_inference":
                slot = (receipt["group_id"], receipt["target_id"], receipt["action_slot"])
                matching = [
                    attempt_id
                    for attempt_id, attempt in candidate_attempts.items()
                    if (
                        attempt["group_id"],
                        attempt["target_id"],
                        attempt["action_slot"],
                    )
                    == slot
                ]
                if len(matching) != 1 or matching[0] in finished_candidates:
                    raise LedgerPoisoned("candidate receipt lacks one unique attempt")
                call_id = receipt["inference_call_id"]
                if (
                    call_id not in completed
                    or starts.get(call_id, {}).get("attempt_id") != matching[0]
                ):
                    raise LedgerPoisoned("candidate receipt lacks one completed model call")
                finished_candidates.add(matching[0])
                candidate_slots.add(slot)
            elif receipt_kind == "zero_call_infrastructure_failure":
                attempt_id = receipt["attempt_id"]
                attempt = candidate_attempts.get(attempt_id)
                if attempt is None or attempt_id in finished_candidates:
                    raise LedgerPoisoned("zero-call receipt lacks one candidate attempt")
                if any(start["attempt_id"] == attempt_id for start in starts.values()):
                    raise LedgerPoisoned("zero-call receipt follows model_call_started")
                finished_candidates.add(attempt_id)
                candidate_slots.add(
                    (receipt["group_id"], receipt["target_id"], receipt["action_slot"])
                )
            elif receipt_kind == "scientific_arm_execution":
                execution_key = (
                    receipt["group_id"],
                    receipt["target_id"],
                    receipt["arm_id"],
                    receipt["continuation_replicate"],
                )
                matching = [
                    attempt_id
                    for attempt_id, attempt in execution_attempts.items()
                    if (
                        attempt["group_id"],
                        attempt["target_id"],
                        attempt["arm_id"],
                        attempt["continuation_replicate"],
                    )
                    == execution_key
                ]
                if len(matching) != 1 or matching[0] in finished_executions:
                    raise LedgerPoisoned("execution receipt lacks one unique attempt")
                expected_calls = {
                    call_id
                    for call_id, start in starts.items()
                    if start["attempt_id"] == matching[0]
                }
                receipt_calls = {call["call_id"] for call in receipt["calls"]}
                if expected_calls != receipt_calls or not expected_calls <= completed:
                    raise LedgerPoisoned("execution receipt does not cover completed calls exactly")
                finished_executions.add(matching[0])
                execution_slots.add(execution_key)
            elif receipt_kind == "training_batch_consumption":
                identity = receipt["training_batch_identity"]
                if identity in batch_claims:
                    raise LedgerPoisoned("training batch was claimed twice")
                batch_claims.add(identity)
        elif kind == "action_reservation":
            event = _event(body)
            key = _group_key(event["group_id"], event["target_id"])
            commitment = commitments.get(key)
            if (
                commitment is None
                or commitment[0] + 1 != record["offset"]
                or event["commitment_receipt_sha256"] != commitment[1]
                or not _is_sha256(event.get("exact_action_key_digest"))
                or not _is_sha256(event.get("request_sha256"))
                or not isinstance(event.get("reservation_id"), str)
                or not event["reservation_id"]
            ):
                raise LedgerPoisoned("reservation is not immediately after its commitment")
            if key in reservations:
                raise LedgerPoisoned("duplicate recorded action reservation")
            reservations[key] = event
        elif kind == "recorded_action_materialized":
            event = _event(body)
            key = _group_key(event["group_id"], event["target_id"])
            reservation = reservations.get(key)
            call_id = event.get("call_id")
            if (
                reservation is None
                or event.get("reservation_id") != reservation["reservation_id"]
                or event.get("exact_action_key_digest")
                != reservation["exact_action_key_digest"]
                or not _is_sha256(event.get("action_digest"))
                or call_id not in completed
                or starts.get(call_id, {}).get("attempt_id") != event["reservation_id"]
                or event["reservation_id"] in recorded_action_materialized
            ):
                raise LedgerPoisoned("recorded action materialization is not reservation-bound")
            recorded_action_materialized.add(event["reservation_id"])
        elif kind == "candidate_attempt":
            event = _event(body)
            attempt_id = event["attempt_id"]
            slot = (event["group_id"], event["target_id"], event["action_slot"])
            if attempt_id in candidate_attempts or slot in candidate_attempt_slots:
                raise LedgerPoisoned("duplicate candidate attempt")
            candidate_attempts[attempt_id] = event
            candidate_attempt_slots.add(slot)
        elif kind == "execution_attempt":
            event = _event(body)
            attempt_id = event["attempt_id"]
            execution_key = (
                event["group_id"],
                event["target_id"],
                event["arm_id"],
                event["continuation_replicate"],
            )
            if attempt_id in execution_attempts or execution_key in execution_attempt_slots:
                raise LedgerPoisoned("duplicate execution attempt")
            execution_attempts[attempt_id] = event
            execution_attempt_slots.add(execution_key)
        elif kind == "model_call_started":
            event = _event(body)
            call_id = event["call_id"]
            if call_id in starts:
                raise LedgerPoisoned("duplicate model call ID")
            attempt_kind = event.get("attempt_kind")
            attempt_id = event.get("attempt_id")
            if (
                (
                    attempt_kind == "candidate"
                    and attempt_id not in candidate_attempts
                )
                or (
                    attempt_kind == "execution"
                    and attempt_id not in execution_attempts
                )
                or (
                    attempt_kind == "recorded_action"
                    and attempt_id
                    not in {event["reservation_id"] for event in reservations.values()}
                )
                or attempt_kind not in {"candidate", "execution", "recorded_action"}
            ):
                raise LedgerPoisoned("model call start lacks its typed scientific attempt")
            starts[call_id] = event
        elif kind == "model_call_completed":
            event = _event(body)
            call_id = event["call_id"]
            if call_id not in starts or call_id in completed:
                raise LedgerPoisoned("completion lacks one in-flight model call")
            if starts[call_id]["attempt_id"] != event["attempt_id"]:
                raise LedgerPoisoned("model call completion changed attempt")
            completed.add(call_id)
    if set(commitments) != set(reservations):
        raise LedgerPoisoned("commitment is missing its promised action reservation")
    if set(candidate_attempts) != finished_candidates:
        raise LedgerPoisoned("ledger has a dangling candidate attempt")
    if set(execution_attempts) != finished_executions:
        raise LedgerPoisoned("ledger has a dangling execution attempt")
    if set(starts) != completed:
        raise LedgerPoisoned("ledger has a dangling model_call_started record")
    started_recorded_actions = {
        start["attempt_id"]
        for start in starts.values()
        if start["attempt_kind"] == "recorded_action"
    }
    if started_recorded_actions != recorded_action_materialized:
        raise LedgerPoisoned("recorded action call lacks a materialized action output")


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


def _atomic_record_write(path: Path, encoded: bytes, fault_hook: FaultHook | None) -> None:
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if fault_hook is not None:
            fault_hook("after_file_fsync", path.name)
        _durable_rename(temporary, path)
        if fault_hook is not None:
            fault_hook("after_rename", path.name)
        if fault_hook is not None:
            fault_hook("after_directory_fsync", path.name)
    finally:
        if temporary.exists() and fault_hook is None:
            temporary.unlink()


def _atomic_blob_write(path: Path, data: bytes) -> None:
    _atomic_record_write(path, data, None)


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
