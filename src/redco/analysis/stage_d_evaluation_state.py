"""Immutable state exposed by the append-only Stage-D evaluation ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from redco.analysis.stage_d_evaluation_contracts import EvaluationScheduleUnit
from redco.analysis.stage_d_objective_binding import ArmName


@dataclass(frozen=True, slots=True)
class EvaluationCallState:
    call_id: str
    task_attempt_id: str
    call_ordinal: int
    event_address_sha256: str
    seed: int
    cache_salt: str
    request_sha256: str
    transport_sha256: str
    dispatch_receipt_sha256: str | None = None
    response_envelope_sha256: str | None = None
    raw_response_sha256: str | None = None
    outcome_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationTaskState:
    unit: EvaluationScheduleUnit
    task_attempt_id: str
    client_epoch: int
    server_attestation_sha256: str
    calls: tuple[EvaluationCallState, ...] = ()
    terminal_result_sha256: str | None = None
    task_metrics_sha256: str | None = None

    @property
    def completed(self) -> bool:
        return self.terminal_result_sha256 is not None


@dataclass(frozen=True, slots=True)
class EvaluationProcessEpoch:
    arm: ArmName
    epoch: int
    launch_record_sha256: str
    resume_task_attempt_id: str | None
    process_receipt_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationServerEpoch:
    arm: ArmName
    epoch: int
    launch_record_sha256: str
    process_receipt_sha256: str | None = None
    server_attestation_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationActuationAttempt:
    actuation_attempt_id: str
    arm: ArmName
    role: Literal["client", "server"]
    epoch: int
    launch_record_sha256: str
    supervisor_identity_sha256: str
    disposition: (
        Literal[
            "claimed",
            "lost-claim-and-drained",
            "spawn-failed",
            "not-observed-after-crash",
            "cleanup-failed",
        ]
        | None
    ) = None
    process_receipt_sha256: str | None = None
    error_evidence_sha256: str | None = None
    cleanup_evidence_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationLedgerSnapshot:
    authorization_sha256: str
    execution_manifest_sha256: str
    evaluation_plan_sha256: str
    created_at_unix_ns: int
    server_claims: tuple[tuple[ArmName, str], ...]
    server_attestations: tuple[tuple[ArmName, str], ...]
    server_epochs: tuple[EvaluationServerEpoch, ...]
    client_epochs: tuple[EvaluationProcessEpoch, ...]
    actuation_attempts: tuple[EvaluationActuationAttempt, ...]
    tasks: tuple[EvaluationTaskState, ...]
    arm_completions: tuple[tuple[ArmName, str], ...]
    arm_metrics: tuple[tuple[ArmName, str], ...]
    sealed: bool
    terminal_status: Literal[
        "active",
        "ambiguous-dispatch",
        "orphaned-open-task",
        "orphaned-server",
        "sealed",
    ]
    head_sha256: str
    record_count: int

    @property
    def current_task(self) -> EvaluationTaskState | None:
        return self.tasks[-1] if self.tasks and not self.tasks[-1].completed else None

    def latest_epoch(self, arm: ArmName) -> EvaluationProcessEpoch | None:
        return next((item for item in reversed(self.client_epochs) if item.arm == arm), None)

    def latest_server_epoch(self, arm: ArmName) -> EvaluationServerEpoch | None:
        return next((item for item in reversed(self.server_epochs) if item.arm == arm), None)


__all__ = [
    "EvaluationActuationAttempt",
    "EvaluationCallState",
    "EvaluationLedgerSnapshot",
    "EvaluationProcessEpoch",
    "EvaluationServerEpoch",
    "EvaluationTaskState",
]
