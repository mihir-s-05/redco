"""Durable reserve-before-spawn transactions for evaluation actuators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

from redco.analysis.stage_d_evaluation_actuation import EvaluationSupervisorIdentity
from redco.analysis.stage_d_evaluation_codec import EvaluationEvidenceStore, exclusive_lock
from redco.analysis.stage_d_evaluation_state import (
    EvaluationActuationAttempt,
    EvaluationLedgerSnapshot,
)
from redco.analysis.stage_d_objective_binding import ArmName
from redco.contracts import canonical_json

ActuationDisposition = Literal[
    "claimed",
    "lost-claim-and-drained",
    "spawn-failed",
    "not-observed-after-crash",
    "cleanup-failed",
]


class EvaluationAttemptLedger(Protocol):
    lock_path: Path
    evidence: EvaluationEvidenceStore

    def inspect(self) -> EvaluationLedgerSnapshot: ...

    def _append_unlocked(self, kind: str, event: dict[str, Any]) -> str: ...


def reserve_actuation_attempt(
    ledger: EvaluationAttemptLedger,
    *,
    actuation_attempt_id: str,
    arm: ArmName,
    role: Literal["client", "server"],
    epoch: int,
    launch_record_sha256: str,
    supervisor: EvaluationSupervisorIdentity,
) -> EvaluationActuationAttempt:
    supervisor_bytes = canonical_json(supervisor.to_payload())
    with exclusive_lock(ledger.lock_path):
        snapshot = ledger.inspect()
        if snapshot.sealed:
            raise RuntimeError("evaluation actuation is forbidden after seal")
        supervisor_sha256 = ledger.evidence.put(supervisor_bytes)
        ledger._append_unlocked(
            "actuation_attempt_reserved",
            {
                "actuation_attempt_id": actuation_attempt_id,
                "arm": arm,
                "role": role,
                "epoch": epoch,
                "launch_record_sha256": launch_record_sha256,
                "supervisor_identity_sha256": supervisor_sha256,
            },
        )
        return ledger.inspect().actuation_attempts[-1]


def finish_actuation_attempt(
    ledger: EvaluationAttemptLedger,
    *,
    actuation_attempt_id: str,
    disposition: ActuationDisposition,
    process_receipt_bytes: bytes | None = None,
    error_evidence_bytes: bytes | None = None,
    cleanup_evidence_bytes: bytes | None = None,
) -> EvaluationActuationAttempt:
    with exclusive_lock(ledger.lock_path):
        snapshot = ledger.inspect()
        matches = [
            item
            for item in snapshot.actuation_attempts
            if item.actuation_attempt_id == actuation_attempt_id
        ]
        if len(matches) != 1:
            raise RuntimeError("evaluation actuation attempt is not uniquely reserved")
        existing = matches[0]
        event = {
            "actuation_attempt_id": actuation_attempt_id,
            "disposition": disposition,
            "process_receipt_sha256": (
                None
                if process_receipt_bytes is None
                else ledger.evidence.put(process_receipt_bytes)
            ),
            "error_evidence_sha256": (
                None if error_evidence_bytes is None else ledger.evidence.put(error_evidence_bytes)
            ),
            "cleanup_evidence_sha256": (
                None
                if cleanup_evidence_bytes is None
                else ledger.evidence.put(cleanup_evidence_bytes)
            ),
        }
        if existing.disposition is not None:
            observed = {
                "actuation_attempt_id": existing.actuation_attempt_id,
                "disposition": existing.disposition,
                "process_receipt_sha256": existing.process_receipt_sha256,
                "error_evidence_sha256": existing.error_evidence_sha256,
                "cleanup_evidence_sha256": existing.cleanup_evidence_sha256,
            }
            if observed != event:
                raise FileExistsError("evaluation actuation disposition differs")
            return existing
        ledger._append_unlocked("actuation_attempt_disposition", event)
        return ledger.inspect().actuation_attempts[-1]


__all__ = [
    "ActuationDisposition",
    "EvaluationAttemptLedger",
    "finish_actuation_attempt",
    "reserve_actuation_attempt",
]
