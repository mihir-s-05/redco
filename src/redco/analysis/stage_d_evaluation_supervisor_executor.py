"""Linux executor for one action chosen by the pure Stage-D supervisor."""

from __future__ import annotations

import os
import secrets
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Literal

from redco.analysis.stage_d_evaluation_actuation import (
    ActuatedProcessReceipt,
    EvaluationSupervisorIdentity,
)
from redco.analysis.stage_d_evaluation_actuator import CgroupV2
from redco.analysis.stage_d_evaluation_capabilities import (
    EvaluationClientLaunch,
    EvaluationServerLaunch,
)
from redco.analysis.stage_d_evaluation_codec import atomic_publish
from redco.analysis.stage_d_evaluation_contracts import (
    evaluation_environment_sha256,
)
from redco.analysis.stage_d_evaluation_ledger import StageDEvaluationLedger
from redco.analysis.stage_d_evaluation_server import (
    capture_linux_server_process,
    probe_local_evaluation_server,
)
from redco.analysis.stage_d_evaluation_supervisor import ProcessLiveness
from redco.analysis.stage_d_handoff_coordinator import StageDHandoffCoordinator
from redco.analysis.stage_d_process_supervision import command_sha256
from redco.contracts import canonical_json


def launch_bound_evaluation_process(
    *,
    ledger: StageDEvaluationLedger,
    handoff_root: Path,
    evaluation_root: Path,
    supervisor: EvaluationSupervisorIdentity,
    server_launch: EvaluationServerLaunch | None = None,
    client_launch: EvaluationClientLaunch | None = None,
) -> ActuatedProcessReceipt:
    """Launch one actuator and synchronously observe its durable target claim."""
    if (server_launch is None) == (client_launch is None):
        raise ValueError("evaluation launch requires exactly one process capability")
    if os.name == "nt":
        raise RuntimeError("evaluation process launch requires Linux")
    if not supervisor.is_same_live_process() or supervisor.pid != os.getpid():
        raise ValueError("evaluation actuator must be launched by its bound supervisor")
    if server_launch is not None:
        arm = server_launch.arm
        role: Literal["client", "server"] = "server"
        epoch = server_launch.epoch
        capability = server_launch.launch_record_sha256
    else:
        assert client_launch is not None
        arm = client_launch.arm
        role = "client"
        epoch = client_launch.epoch
        capability = client_launch.launch_record_sha256
    program = ledger.manifest.program(arm, role)
    limits = ledger.manifest.supervisor_limits
    process_receipt_dir = Path(limits.control_root) / ledger.manifest.evaluation_ledger_id
    log_root = Path(limits.log_root) / ledger.manifest.evaluation_ledger_id
    for root in (handoff_root, evaluation_root, process_receipt_dir, log_root):
        if not root.is_absolute():
            raise ValueError("evaluation executor paths must be absolute")
    actuation_attempt_id = secrets.token_hex(16)
    ledger.reserve_actuation_attempt(
        actuation_attempt_id=actuation_attempt_id,
        arm=arm,
        role=role,
        epoch=epoch,
        launch_record_sha256=capability,
        supervisor=supervisor,
    )
    attempt_label = f"{role}-{arm}-{epoch}-{capability[:16]}-{actuation_attempt_id}"
    attempt_root = process_receipt_dir / actuation_attempt_id
    attempt_root.mkdir(parents=True, mode=0o700)
    log_root.mkdir(parents=True, exist_ok=True)
    supervisor_path = attempt_root / "supervisor.json"
    environment_path = attempt_root / "target-environment.json"
    receipt_path = attempt_root / "target-receipt.json"
    stop_path = attempt_root / "stop-request.json"
    atomic_publish(supervisor_path, canonical_json(supervisor.to_payload()))
    atomic_publish(environment_path, canonical_json(dict(program.environment)))
    wrapper_command = _evaluation_wrapper_command(
        program.absolute_executable,
        handoff_root=handoff_root,
        evaluation_root=evaluation_root,
        arm=arm,
        role=role,
        server_launch=server_launch,
        client_launch=client_launch,
    )
    actuator_command = [
        program.absolute_executable,
        "-m",
        "redco.analysis.stage_d_evaluation_actuator",
        "--supervisor-identity",
        str(supervisor_path),
        "--receipt",
        str(receipt_path),
        "--stop-request",
        str(stop_path),
        "--cgroup-root",
        limits.cgroup_root,
        "--target-environment",
        str(environment_path),
        "--target-cwd",
        program.working_directory,
        "--target-stdout",
        str(log_root / f"{attempt_label}.target.stdout.log"),
        "--target-stderr",
        str(log_root / f"{attempt_label}.target.stderr.log"),
        "--arm",
        arm,
        "--role",
        role,
        "--epoch",
        str(epoch),
        "--launch-capability-sha256",
        capability,
        "--actuation-attempt-id",
        actuation_attempt_id,
        "--program-command-sha256",
        command_sha256(program.argv),
        "--program-environment-sha256",
        evaluation_environment_sha256(program.environment),
        "--poll-interval-seconds",
        str(limits.poll_interval_milliseconds / 1000),
        "--stop-timeout-seconds",
        str(limits.stop_timeout_seconds),
        "--max-log-bytes",
        str(limits.max_log_bytes),
        "--",
        *wrapper_command,
    ]
    actuator_stdout = log_root / f"{attempt_label}.actuator.stdout.log"
    actuator_stderr = log_root / f"{attempt_label}.actuator.stderr.log"
    try:
        with actuator_stdout.open("xb") as stdout, actuator_stderr.open("xb") as stderr:
            process = subprocess.Popen(
                actuator_command,
                cwd=program.working_directory,
                env=dict(program.environment),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
                start_new_session=True,
            )
    except BaseException as error:
        ledger.finish_actuation_attempt(
            actuation_attempt_id=actuation_attempt_id,
            disposition="spawn-failed",
            error_evidence_bytes=_actuation_error_evidence("spawn", error),
        )
        raise
    deadline = time.monotonic() + limits.claim_timeout_seconds
    while time.monotonic() < deadline:
        receipt = _claimed_receipt(
            ledger,
            server_launch=server_launch,
            client_launch=client_launch,
        )
        if receipt is not None:
            if receipt.supervisor != supervisor:
                _stop_direct_child(process, timeout_seconds=limits.stop_timeout_seconds)
                ledger.finish_actuation_attempt(
                    actuation_attempt_id=actuation_attempt_id,
                    disposition="lost-claim-and-drained",
                    cleanup_evidence_bytes=_actuation_cleanup_evidence(
                        "foreign-claim",
                        process.returncode,
                    ),
                )
                return receipt
            if receipt.actuator_pid != process.pid:
                _stop_direct_child(process, timeout_seconds=limits.stop_timeout_seconds)
                ledger.finish_actuation_attempt(
                    actuation_attempt_id=actuation_attempt_id,
                    disposition="lost-claim-and-drained",
                    cleanup_evidence_bytes=_actuation_cleanup_evidence(
                        "lost-claim",
                        process.returncode,
                    ),
                )
                return receipt
            ledger.finish_actuation_attempt(
                actuation_attempt_id=actuation_attempt_id,
                disposition="claimed",
                process_receipt_bytes=receipt.to_bytes(),
            )
            return receipt
        if process.poll() is not None:
            early_exit_error = RuntimeError(
                f"evaluation {role} exited before its durable claim; see {actuator_stderr}"
            )
            ledger.finish_actuation_attempt(
                actuation_attempt_id=actuation_attempt_id,
                disposition="spawn-failed",
                error_evidence_bytes=_actuation_error_evidence(
                    "pre-claim-exit",
                    early_exit_error,
                ),
            )
            raise early_exit_error
        time.sleep(limits.poll_interval_milliseconds / 1000)
    _stop_direct_child(process, timeout_seconds=limits.stop_timeout_seconds)
    ledger.finish_actuation_attempt(
        actuation_attempt_id=actuation_attempt_id,
        disposition="spawn-failed",
        error_evidence_bytes=_actuation_error_evidence(
            "claim-timeout",
            TimeoutError(f"evaluation {role} claim timeout"),
        ),
    )
    raise TimeoutError(f"evaluation {role} did not claim its launch before timeout")


def _actuation_error_evidence(stage: str, error: BaseException) -> bytes:
    return canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-evaluation-actuation-error-v1",
            "stage": stage,
            "error_type": type(error).__qualname__,
            "error_message": str(error),
        }
    )


def _actuation_cleanup_evidence(reason: str, return_code: int | None) -> bytes:
    return canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-evaluation-actuation-cleanup-v1",
            "reason": reason,
            "actuator_return_code": return_code,
        }
    )


def _evaluation_wrapper_command(
    executable: str,
    *,
    handoff_root: Path,
    evaluation_root: Path,
    arm: str,
    role: Literal["client", "server"],
    server_launch: EvaluationServerLaunch | None,
    client_launch: EvaluationClientLaunch | None,
) -> list[str]:
    command = [
        executable,
        "-m",
        "redco.analysis.stage_d_evaluation_process",
        "--handoff-root",
        str(handoff_root),
        "--evaluation-root",
        str(evaluation_root),
        "--arm",
        arm,
        "--role",
        role,
    ]
    if server_launch is not None:
        command.extend(
            (
                "--server-epoch",
                str(server_launch.epoch),
                "--server-launch-record-sha256",
                server_launch.launch_record_sha256,
            )
        )
    else:
        assert client_launch is not None
        command.extend(
            (
                "--client-epoch",
                str(client_launch.epoch),
                "--client-launch-record-sha256",
                client_launch.launch_record_sha256,
            )
        )
        if client_launch.resume_task_attempt_id is not None:
            command.extend(("--resume-task-attempt-id", client_launch.resume_task_attempt_id))
    return command


def _stop_direct_child(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    if process.poll() is not None:
        process.wait()
        return
    process.terminate()
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=max(0.0, deadline - time.monotonic()))


def attest_evaluation_server(
    *,
    coordinator: StageDHandoffCoordinator,
    ledger: StageDEvaluationLedger,
    launch: EvaluationServerLaunch,
    process_receipt: ActuatedProcessReceipt,
    probe_timeout_seconds: float,
) -> str:
    program = ledger.manifest.program(launch.arm, "server")
    checkpoint = coordinator.materialize_evaluation_checkpoint(
        launch.arm,
        Path(program.checkpoint_root),
    )
    probe = probe_local_evaluation_server(program, timeout_seconds=probe_timeout_seconds)
    observation = capture_linux_server_process(
        launch=launch,
        receipt=process_receipt,
        program=program,
        checkpoint_manifest_bytes=checkpoint.to_bytes(),
    )
    return ledger.attest_server(
        launch=launch,
        process_receipt_bytes=process_receipt.to_bytes(),
        process_observation_bytes=observation.to_bytes(),
        probe_response_bytes=probe,
    )


def observed_process_liveness(
    ledger: StageDEvaluationLedger,
    *,
    supervisor: EvaluationSupervisorIdentity,
) -> dict[str, ProcessLiveness]:
    snapshot = ledger.inspect()
    digests = {
        item.process_receipt_sha256
        for item in snapshot.server_epochs
        if item.process_receipt_sha256 is not None
    } | {
        item.process_receipt_sha256
        for item in snapshot.client_epochs
        if item.process_receipt_sha256 is not None
    }
    result: dict[str, ProcessLiveness] = {}
    cgroup_root = Path(ledger.manifest.supervisor_limits.cgroup_root)
    for digest in digests:
        receipt = ActuatedProcessReceipt.from_bytes(ledger.evidence.get(digest))
        try:
            contained = _receipt_cgroup_populated(receipt, cgroup_root=cgroup_root)
        except (FileNotFoundError, OSError, ValueError):
            result[digest] = "unknown"
            continue
        target_live = receipt.is_same_live_process()
        actuator_live = receipt.actuator_is_same_live_process()
        prior_supervisor_live = receipt.supervisor.is_same_live_process()
        if receipt.supervisor != supervisor:
            if prior_supervisor_live and (target_live or actuator_live or contained):
                result[digest] = "foreign-draining"
            elif target_live or actuator_live or contained:
                result[digest] = "contained-orphan"
            else:
                result[digest] = "dead"
        elif receipt.is_same_live_tree():
            result[digest] = "live"
        elif contained:
            result[digest] = "contained-orphan"
        elif not target_live and not actuator_live:
            result[digest] = "dead"
        else:
            result[digest] = "unknown"
    return result


def stop_actuated_process(
    receipt: ActuatedProcessReceipt,
    *,
    cgroup_root: Path,
    timeout_seconds: float,
) -> None:
    """Request actuator cleanup, then use the exact cgroup as a bounded fallback."""
    if os.name == "nt":
        raise RuntimeError("evaluation process termination requires Linux")
    cgroup_path = Path(receipt.cgroup_path)
    if cgroup_path.parent != cgroup_root or not receipt.has_dedicated_topology():
        raise ValueError("evaluation stop receipt differs from its containment root")
    _receipt_cgroup_populated(receipt, cgroup_root=cgroup_root)
    stop_path = Path(receipt.stop_request_path)
    atomic_publish(
        stop_path,
        canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-evaluation-stop-request-v1",
                "process_receipt_sha256": receipt.receipt_sha256,
            }
        ),
    )
    deadline = time.monotonic() + timeout_seconds
    graceful_deadline = min(deadline, time.monotonic() + min(1.0, timeout_seconds / 2))
    while time.monotonic() < graceful_deadline:
        if (
            not receipt.is_same_live_process()
            and not receipt.actuator_is_same_live_process()
            and not _receipt_cgroup_populated(receipt, cgroup_root=cgroup_root)
        ):
            reap_child_process(receipt.actuator_pid)
            return
        time.sleep(0.05)
    if receipt.actuator_is_same_live_process():
        with suppress(ProcessLookupError):
            os.kill(receipt.actuator_pid, signal.SIGTERM)
    fallback = CgroupV2(cgroup_path)
    if fallback.path.exists() and fallback.populated():
        if receipt.is_same_live_process():
            if fallback.verify_membership(receipt.pid) != receipt.cgroup_lines:
                raise RuntimeError("evaluation fallback cgroup identity differs")
        elif fallback.path.parent != cgroup_root:
            raise RuntimeError("evaluation fallback cgroup root differs")
        fallback.kill()
        fallback.wait_empty(deadline)
    if receipt.is_same_live_process():
        if fallback.verify_membership(receipt.pid) != receipt.cgroup_lines:
            raise RuntimeError("evaluation fallback cgroup identity differs")
        raise RuntimeError("evaluation target survived its cgroup kill")
    while time.monotonic() < deadline:
        if not receipt.actuator_is_same_live_process():
            reap_child_process(receipt.actuator_pid)
            if fallback.path.exists():
                with suppress(FileNotFoundError):
                    fallback.remove()
            return
        time.sleep(0.05)
    raise TimeoutError("evaluation actuator did not terminate")


def _receipt_cgroup_populated(
    receipt: ActuatedProcessReceipt,
    *,
    cgroup_root: Path,
) -> bool:
    cgroup_path = Path(receipt.cgroup_path)
    if cgroup_path.parent != cgroup_root or not receipt.has_dedicated_topology():
        raise ValueError("evaluation receipt cgroup differs from its containment root")
    if not cgroup_path.exists():
        return False
    observed = cgroup_path.stat()
    if (observed.st_dev, observed.st_ino) != (
        receipt.cgroup_device_id,
        receipt.cgroup_inode,
    ):
        raise ValueError("evaluation receipt cgroup identity changed")
    return CgroupV2(cgroup_path).populated()


def reap_child_process(pid: int) -> bool:
    """Reap an exited direct child without mistaking a live non-child for dead."""
    if os.name == "nt":
        raise RuntimeError("evaluation process reaping requires Linux")
    try:
        observed, _status = os.waitpid(pid, getattr(os, "WNOHANG", 1))
    except ChildProcessError:
        return False
    return observed == pid


def _claimed_receipt(
    ledger: StageDEvaluationLedger,
    *,
    server_launch: EvaluationServerLaunch | None,
    client_launch: EvaluationClientLaunch | None,
) -> ActuatedProcessReceipt | None:
    snapshot = ledger.inspect()
    if server_launch is not None:
        server_epoch = snapshot.latest_server_epoch(server_launch.arm)
        if server_epoch is None or (
            server_epoch.epoch,
            server_epoch.launch_record_sha256,
        ) != (server_launch.epoch, server_launch.launch_record_sha256):
            raise RuntimeError("evaluation server launch changed while spawning")
        receipt_sha256 = server_epoch.process_receipt_sha256
    else:
        assert client_launch is not None
        client_epoch = snapshot.latest_epoch(client_launch.arm)
        if client_epoch is None or (
            client_epoch.epoch,
            client_epoch.launch_record_sha256,
            client_epoch.resume_task_attempt_id,
        ) != (
            client_launch.epoch,
            client_launch.launch_record_sha256,
            client_launch.resume_task_attempt_id,
        ):
            raise RuntimeError("evaluation client launch changed while spawning")
        receipt_sha256 = client_epoch.process_receipt_sha256
    if receipt_sha256 is None:
        return None
    return ActuatedProcessReceipt.from_bytes(ledger.evidence.get(receipt_sha256))


__all__ = [
    "attest_evaluation_server",
    "launch_bound_evaluation_process",
    "observed_process_liveness",
    "reap_child_process",
    "stop_actuated_process",
]
