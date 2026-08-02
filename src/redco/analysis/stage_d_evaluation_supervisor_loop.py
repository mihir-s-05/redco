"""Single-owner bounded supervisor for the Stage-D held-out evaluator."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from redco.analysis.stage_d_evaluation_actuation import (
    ActuatedProcessReceipt,
    EvaluationSupervisorIdentity,
)
from redco.analysis.stage_d_evaluation_actuator import preflight_cgroup_v2
from redco.analysis.stage_d_evaluation_ledger import StageDEvaluationLedger
from redco.analysis.stage_d_evaluation_supervisor import (
    EvaluationBlockCode,
    EvaluationSupervisorAction,
    next_evaluation_supervisor_action,
)
from redco.analysis.stage_d_evaluation_supervisor_executor import (
    attest_evaluation_server,
    launch_bound_evaluation_process,
    observed_process_liveness,
    stop_actuated_process,
)
from redco.analysis.stage_d_evaluation_worker import cleanup_evaluation_containers
from redco.analysis.stage_d_file_lock import exclusive_file_lock
from redco.analysis.stage_d_handoff_coordinator import StageDHandoffCoordinator


@dataclass(frozen=True, slots=True)
class EvaluationSupervisorResult:
    disposition: Literal["completed", "blocked", "deadline-exceeded"]
    ledger_head_sha256: str
    ledger_record_count: int
    block_code: EvaluationBlockCode | None = None

    def __post_init__(self) -> None:
        if (self.disposition == "blocked") != (self.block_code is not None):
            raise ValueError("evaluation supervisor result fields differ")


def run_evaluation_supervisor(
    *,
    coordinator: StageDHandoffCoordinator,
    handoff_root: Path,
    evaluation_root: Path,
) -> EvaluationSupervisorResult:
    """Run the frozen evaluator to completion under one lifetime advisory lock."""
    if os.name == "nt":
        raise RuntimeError("evaluation supervisor requires Linux")
    ledger = coordinator.materialize_evaluation_ledger(evaluation_root)
    manifest = ledger.manifest
    limits = manifest.supervisor_limits
    supervisor = EvaluationSupervisorIdentity.current()
    lock_path = Path(limits.control_root) / f".{manifest.evaluation_ledger_id}.supervisor.lock"
    with exclusive_file_lock(lock_path):
        try:
            preflight_cgroup_v2(
                Path(limits.cgroup_root),
                executable=manifest.program("stock", "server").absolute_executable,
                timeout_seconds=limits.stop_timeout_seconds,
            )
            return _run_locked(
                coordinator=coordinator,
                ledger=ledger,
                handoff_root=handoff_root,
                evaluation_root=evaluation_root,
                supervisor=supervisor,
            )
        except BaseException:
            _cleanup_owned_processes(ledger, supervisor)
            raise


def _run_locked(
    *,
    coordinator: StageDHandoffCoordinator,
    ledger: StageDEvaluationLedger,
    handoff_root: Path,
    evaluation_root: Path,
    supervisor: EvaluationSupervisorIdentity,
) -> EvaluationSupervisorResult:
    manifest = ledger.manifest
    limits = manifest.supervisor_limits
    deadline_unix_ns = (
        ledger.inspect().created_at_unix_ns + limits.evaluation_timeout_seconds * 1_000_000_000
    )
    while True:
        snapshot = ledger.inspect()
        if time.time_ns() >= deadline_unix_ns:
            _cleanup_owned_processes(ledger, supervisor)
            current = ledger.inspect()
            return EvaluationSupervisorResult(
                "deadline-exceeded",
                current.head_sha256,
                current.record_count,
            )
        liveness = observed_process_liveness(ledger, supervisor=supervisor)
        action = next_evaluation_supervisor_action(
            manifest=manifest,
            snapshot=snapshot,
            process_liveness=liveness,
        )
        result = _execute_action(
            action,
            coordinator=coordinator,
            ledger=ledger,
            handoff_root=handoff_root,
            evaluation_root=evaluation_root,
            supervisor=supervisor,
        )
        if result is not None:
            return result


def _execute_action(
    action: EvaluationSupervisorAction,
    *,
    coordinator: StageDHandoffCoordinator,
    ledger: StageDEvaluationLedger,
    handoff_root: Path,
    evaluation_root: Path,
    supervisor: EvaluationSupervisorIdentity,
) -> EvaluationSupervisorResult | None:
    limits = ledger.manifest.supervisor_limits
    if action.kind == "reserve-server":
        assert action.arm is not None
        ledger.reserve_server_launch(action.arm)
    elif action.kind == "spawn-server":
        assert action.server_launch is not None
        launch_bound_evaluation_process(
            ledger=ledger,
            handoff_root=handoff_root,
            evaluation_root=evaluation_root,
            supervisor=supervisor,
            server_launch=action.server_launch,
        )
    elif action.kind == "attest-server":
        assert action.server_launch is not None and action.process_receipt_sha256 is not None
        receipt = _receipt(ledger, action.process_receipt_sha256)
        if receipt.supervisor != supervisor or not receipt.is_same_live_tree():
            raise RuntimeError("evaluation server changed before attestation")
        attest_evaluation_server(
            coordinator=coordinator,
            ledger=ledger,
            launch=action.server_launch,
            process_receipt=receipt,
            probe_timeout_seconds=limits.probe_timeout_seconds,
        )
    elif action.kind == "reserve-client":
        assert action.arm is not None
        cleanup_evaluation_containers(
            ledger.manifest,
            timeout_seconds=limits.stop_timeout_seconds,
        )
        ledger.reserve_client_launch(action.arm)
    elif action.kind == "spawn-client":
        assert action.client_launch is not None
        launch_bound_evaluation_process(
            ledger=ledger,
            handoff_root=handoff_root,
            evaluation_root=evaluation_root,
            supervisor=supervisor,
            client_launch=action.client_launch,
        )
    elif action.kind == "wait-client":
        time.sleep(limits.poll_interval_milliseconds / 1000)
    elif action.kind == "complete-arm":
        assert action.arm is not None
        ledger.complete_arm(action.arm)
    elif action.kind in {"stop-server", "cleanup-process"}:
        assert action.process_receipt_sha256 is not None
        stop_actuated_process(
            _receipt(ledger, action.process_receipt_sha256),
            cgroup_root=Path(limits.cgroup_root),
            timeout_seconds=limits.stop_timeout_seconds,
        )
    elif action.kind == "seal":
        cleanup_evaluation_containers(
            ledger.manifest,
            timeout_seconds=limits.stop_timeout_seconds,
        )
        ledger.seal()
    elif action.kind == "done":
        snapshot = ledger.inspect()
        return EvaluationSupervisorResult(
            "completed",
            snapshot.head_sha256,
            snapshot.record_count,
        )
    elif action.kind == "blocked":
        code = action.block_code
        assert code is not None
        if code == EvaluationBlockCode.FOREIGN_PROCESS:
            time.sleep(limits.poll_interval_milliseconds / 1000)
            return None
        _cleanup_owned_processes(ledger, supervisor)
        snapshot = ledger.inspect()
        return EvaluationSupervisorResult(
            "blocked",
            snapshot.head_sha256,
            snapshot.record_count,
            code,
        )
    else:
        raise AssertionError(f"unhandled evaluation supervisor action: {action.kind}")
    return None


def _cleanup_owned_processes(
    ledger: StageDEvaluationLedger,
    supervisor: EvaluationSupervisorIdentity,
) -> None:
    limits = ledger.manifest.supervisor_limits
    snapshot = ledger.inspect()
    digests = [
        item.process_receipt_sha256
        for item in snapshot.client_epochs
        if item.process_receipt_sha256 is not None
    ] + [
        item.process_receipt_sha256
        for item in snapshot.server_epochs
        if item.process_receipt_sha256 is not None
    ]
    for digest in reversed(digests):
        receipt = _receipt(ledger, digest)
        if receipt.supervisor == supervisor or not receipt.supervisor.is_same_live_process():
            stop_actuated_process(
                receipt,
                cgroup_root=Path(limits.cgroup_root),
                timeout_seconds=limits.stop_timeout_seconds,
            )
    cleanup_evaluation_containers(
        ledger.manifest,
        timeout_seconds=limits.stop_timeout_seconds,
    )


def _receipt(ledger: StageDEvaluationLedger, digest: str) -> ActuatedProcessReceipt:
    return ActuatedProcessReceipt.from_bytes(ledger.evidence.get(digest))


__all__ = [
    "EvaluationSupervisorResult",
    "run_evaluation_supervisor",
]
