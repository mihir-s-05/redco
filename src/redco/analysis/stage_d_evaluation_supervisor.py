"""Pure orchestration policy for the single-process Stage-D evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from redco.analysis.stage_d_evaluation_capabilities import (
    EvaluationClientLaunch,
    EvaluationServerLaunch,
)
from redco.analysis.stage_d_evaluation_contracts import StageDEvaluationExecutionManifest
from redco.analysis.stage_d_evaluation_state import EvaluationLedgerSnapshot
from redco.analysis.stage_d_objective_binding import ArmName

_ARMS: tuple[ArmName, ...] = ("stock", "branch-global", "local")
ProcessLiveness = Literal[
    "live",
    "dead",
    "contained-orphan",
    "unknown",
    "foreign-draining",
]


class EvaluationBlockCode(StrEnum):
    AMBIGUOUS_DISPATCH = "ambiguous-dispatch"
    FOREIGN_PROCESS = "foreign-process"
    PROCESS_LIVENESS_UNKNOWN = "process-liveness-unknown"
    SERVER_DIED_AFTER_DISPATCH = "server-died-after-dispatch"
    INCONSISTENT_COMPLETION = "inconsistent-completion"
    UNRESOLVED_ACTUATION = "unresolved-actuation"


@dataclass(frozen=True, slots=True)
class EvaluationSupervisorAction:
    kind: Literal[
        "reserve-server",
        "spawn-server",
        "attest-server",
        "reserve-client",
        "spawn-client",
        "wait-client",
        "complete-arm",
        "stop-server",
        "cleanup-process",
        "seal",
        "done",
        "blocked",
    ]
    arm: ArmName | None = None
    server_launch: EvaluationServerLaunch | None = None
    client_launch: EvaluationClientLaunch | None = None
    process_receipt_sha256: str | None = None
    block_code: EvaluationBlockCode | None = None

    def __post_init__(self) -> None:
        expected = {
            "reserve-server": (True, False, False, False),
            "spawn-server": (True, True, False, False),
            "attest-server": (True, True, False, True),
            "reserve-client": (True, False, False, False),
            "spawn-client": (True, False, True, False),
            "wait-client": (True, False, False, True),
            "complete-arm": (True, False, False, False),
            "stop-server": (True, False, False, True),
            "cleanup-process": (True, False, False, True),
            "seal": (False, False, False, False),
            "done": (False, False, False, False),
        }
        if self.kind == "blocked":
            if (
                self.block_code is None
                or self.server_launch is not None
                or self.client_launch is not None
                or self.process_receipt_sha256 is not None
            ):
                raise ValueError("blocked evaluation supervisor action is malformed")
            return
        shape = expected[self.kind]
        observed = (
            self.arm is not None,
            self.server_launch is not None,
            self.client_launch is not None,
            self.process_receipt_sha256 is not None,
        )
        if observed != shape or self.block_code is not None:
            raise ValueError("evaluation supervisor action fields differ from its kind")
        if self.server_launch is not None and self.server_launch.arm != self.arm:
            raise ValueError("evaluation server action arm differs from its launch")
        if self.client_launch is not None and self.client_launch.arm != self.arm:
            raise ValueError("evaluation client action arm differs from its launch")


def next_evaluation_supervisor_action(
    *,
    manifest: StageDEvaluationExecutionManifest,
    snapshot: EvaluationLedgerSnapshot,
    process_liveness: Mapping[str, ProcessLiveness],
) -> EvaluationSupervisorAction:
    """Choose one idempotent transition without performing I/O."""
    if snapshot.execution_manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("evaluation supervisor snapshot differs from its manifest")
    if snapshot.sealed:
        return EvaluationSupervisorAction("done")
    if snapshot.terminal_status == "ambiguous-dispatch":
        return EvaluationSupervisorAction(
            "blocked",
            block_code=EvaluationBlockCode.AMBIGUOUS_DISPATCH,
        )
    if any(item.disposition is None for item in snapshot.actuation_attempts):
        return EvaluationSupervisorAction(
            "blocked",
            block_code=EvaluationBlockCode.UNRESOLVED_ACTUATION,
        )

    completed = dict(snapshot.arm_completions)
    for arm in _ARMS:
        if arm not in completed:
            break
        cleanup = _completed_arm_cleanup(snapshot, arm, process_liveness)
        if cleanup is not None:
            return cleanup
    else:
        return EvaluationSupervisorAction("seal")

    arm = next(item for item in _ARMS if item not in completed)
    arm_units = tuple(unit for unit in manifest.schedule if unit.arm == arm)
    arm_tasks = tuple(task for task in snapshot.tasks if task.unit.arm == arm)
    if len(arm_tasks) == len(arm_units) and all(task.completed for task in arm_tasks):
        return EvaluationSupervisorAction("complete-arm", arm=arm)

    server = snapshot.latest_server_epoch(arm)
    if server is None:
        return EvaluationSupervisorAction("reserve-server", arm=arm)
    launch = EvaluationServerLaunch(arm, server.epoch, server.launch_record_sha256)
    if server.process_receipt_sha256 is None:
        return EvaluationSupervisorAction("spawn-server", arm=arm, server_launch=launch)
    server_status = _liveness(process_liveness, server.process_receipt_sha256)
    if server_status == "foreign-draining":
        return EvaluationSupervisorAction(
            "blocked",
            arm=arm,
            block_code=EvaluationBlockCode.FOREIGN_PROCESS,
        )
    if server_status == "unknown":
        return EvaluationSupervisorAction(
            "blocked",
            arm=arm,
            block_code=EvaluationBlockCode.PROCESS_LIVENESS_UNKNOWN,
        )
    if server_status == "contained-orphan":
        return EvaluationSupervisorAction(
            "cleanup-process",
            arm=arm,
            process_receipt_sha256=server.process_receipt_sha256,
        )
    if server_status == "dead":
        if _arm_has_dispatch(snapshot, arm):
            return EvaluationSupervisorAction(
                "blocked",
                arm=arm,
                block_code=EvaluationBlockCode.SERVER_DIED_AFTER_DISPATCH,
            )
        return EvaluationSupervisorAction("reserve-server", arm=arm)
    if server.server_attestation_sha256 is None:
        return EvaluationSupervisorAction(
            "attest-server",
            arm=arm,
            server_launch=launch,
            process_receipt_sha256=server.process_receipt_sha256,
        )

    client = snapshot.latest_epoch(arm)
    if client is None:
        return EvaluationSupervisorAction("reserve-client", arm=arm)
    if client.process_receipt_sha256 is None:
        return EvaluationSupervisorAction(
            "spawn-client",
            arm=arm,
            client_launch=EvaluationClientLaunch(
                arm,
                client.epoch,
                client.launch_record_sha256,
                client.resume_task_attempt_id,
            ),
        )
    client_status = _liveness(process_liveness, client.process_receipt_sha256)
    if client_status == "foreign-draining":
        return EvaluationSupervisorAction(
            "blocked",
            arm=arm,
            block_code=EvaluationBlockCode.FOREIGN_PROCESS,
        )
    if client_status == "unknown":
        return EvaluationSupervisorAction(
            "blocked",
            arm=arm,
            block_code=EvaluationBlockCode.PROCESS_LIVENESS_UNKNOWN,
        )
    if client_status == "contained-orphan":
        return EvaluationSupervisorAction(
            "cleanup-process",
            arm=arm,
            process_receipt_sha256=client.process_receipt_sha256,
        )
    if client_status == "dead":
        if snapshot.terminal_status in {"orphaned-open-task", "orphaned-server"}:
            current = snapshot.current_task
            if current is not None and any(
                call.dispatch_receipt_sha256 is not None and call.response_envelope_sha256 is None
                for call in current.calls
            ):
                return EvaluationSupervisorAction(
                    "blocked",
                    arm=arm,
                    block_code=EvaluationBlockCode.AMBIGUOUS_DISPATCH,
                )
        return EvaluationSupervisorAction("reserve-client", arm=arm)
    return EvaluationSupervisorAction(
        "wait-client",
        arm=arm,
        process_receipt_sha256=client.process_receipt_sha256,
    )


def _completed_arm_cleanup(
    snapshot: EvaluationLedgerSnapshot,
    arm: ArmName,
    process_liveness: Mapping[str, ProcessLiveness],
) -> EvaluationSupervisorAction | None:
    client = snapshot.latest_epoch(arm)
    if client is not None:
        if client.process_receipt_sha256 is None:
            return EvaluationSupervisorAction(
                "blocked",
                arm=arm,
                block_code=EvaluationBlockCode.INCONSISTENT_COMPLETION,
            )
        status = _liveness(process_liveness, client.process_receipt_sha256)
        if status == "live":
            return EvaluationSupervisorAction(
                "wait-client",
                arm=arm,
                process_receipt_sha256=client.process_receipt_sha256,
            )
        if status == "unknown":
            return EvaluationSupervisorAction(
                "blocked",
                arm=arm,
                block_code=EvaluationBlockCode.PROCESS_LIVENESS_UNKNOWN,
            )
        if status == "foreign-draining":
            return EvaluationSupervisorAction(
                "blocked",
                arm=arm,
                block_code=EvaluationBlockCode.FOREIGN_PROCESS,
            )
        if status == "contained-orphan":
            return EvaluationSupervisorAction(
                "cleanup-process",
                arm=arm,
                process_receipt_sha256=client.process_receipt_sha256,
            )
    server = snapshot.latest_server_epoch(arm)
    if server is None or server.process_receipt_sha256 is None:
        return None
    status = _liveness(process_liveness, server.process_receipt_sha256)
    if status == "live":
        return EvaluationSupervisorAction(
            "stop-server",
            arm=arm,
            process_receipt_sha256=server.process_receipt_sha256,
        )
    if status == "unknown":
        return EvaluationSupervisorAction(
            "blocked",
            arm=arm,
            block_code=EvaluationBlockCode.PROCESS_LIVENESS_UNKNOWN,
        )
    if status == "foreign-draining":
        return EvaluationSupervisorAction(
            "blocked",
            arm=arm,
            block_code=EvaluationBlockCode.FOREIGN_PROCESS,
        )
    if status == "contained-orphan":
        return EvaluationSupervisorAction(
            "cleanup-process",
            arm=arm,
            process_receipt_sha256=server.process_receipt_sha256,
        )
    return None


def _liveness(
    values: Mapping[str, ProcessLiveness],
    digest: str,
) -> ProcessLiveness:
    value = values.get(digest, "unknown")
    if value not in {
        "live",
        "dead",
        "contained-orphan",
        "unknown",
        "foreign-draining",
    }:
        raise ValueError("evaluation supervisor process liveness is invalid")
    return value


def _arm_has_dispatch(snapshot: EvaluationLedgerSnapshot, arm: ArmName) -> bool:
    return any(
        call.dispatch_receipt_sha256 is not None
        for task in snapshot.tasks
        if task.unit.arm == arm
        for call in task.calls
    )


__all__ = [
    "EvaluationBlockCode",
    "EvaluationSupervisorAction",
    "ProcessLiveness",
    "next_evaluation_supervisor_action",
]
