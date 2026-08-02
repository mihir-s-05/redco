"""Immutable capabilities passed between Stage-D evaluation transactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from redco.analysis.stage_d_evaluation_codec import sha256
from redco.analysis.stage_d_evaluation_contracts import EvaluationScheduleUnit
from redco.analysis.stage_d_objective_binding import ArmName


@dataclass(frozen=True, slots=True)
class EvaluationTaskAttempt:
    task_attempt_id: str
    unit: EvaluationScheduleUnit
    client_epoch: int


@dataclass(frozen=True, slots=True)
class EvaluationClientSession:
    arm: ArmName
    epoch: int
    process_receipt_bytes: bytes

    @property
    def process_receipt_sha256(self) -> str:
        return cast(str, sha256(self.process_receipt_bytes))


@dataclass(frozen=True, slots=True)
class EvaluationServerLaunch:
    arm: ArmName
    epoch: int
    launch_record_sha256: str


@dataclass(frozen=True, slots=True)
class EvaluationClientLaunch:
    arm: ArmName
    epoch: int
    launch_record_sha256: str
    resume_task_attempt_id: str | None


@dataclass(frozen=True, slots=True)
class EvaluationCallAuthorization:
    call_id: str
    task_attempt_id: str
    call_ordinal: int
    request_sha256: str
    transport_sha256: str


@dataclass(frozen=True, slots=True)
class EvaluationDispatchAuthorization:
    call: EvaluationCallAuthorization
    dispatch_receipt_sha256: str


__all__ = [
    "EvaluationCallAuthorization",
    "EvaluationClientLaunch",
    "EvaluationClientSession",
    "EvaluationDispatchAuthorization",
    "EvaluationServerLaunch",
    "EvaluationTaskAttempt",
]
