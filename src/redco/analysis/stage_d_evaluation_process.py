"""Bound same-PID claim-and-exec transition for Stage-D evaluation processes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal, NoReturn

from redco.analysis.stage_d_checkpoint_evidence import StageDCheckpointManifest
from redco.analysis.stage_d_checkpoint_materialization import (
    verify_materialized_checkpoint,
)
from redco.analysis.stage_d_evaluation_actuation import ActuatedProcessReceipt
from redco.analysis.stage_d_evaluation_capabilities import (
    EvaluationClientLaunch,
    EvaluationServerLaunch,
)
from redco.analysis.stage_d_evaluation_contracts import (
    EvaluationProgramBinding,
    StageDEvaluationExecutionManifest,
    hash_file,
)
from redco.analysis.stage_d_evaluation_ledger import StageDEvaluationLedger


def await_actuator_release(start_gate_fd: int) -> None:
    """Require the actuator's one-byte release after its receipt is durable."""
    if type(start_gate_fd) is not int or start_gate_fd < 3:
        raise ValueError("evaluation start-gate descriptor is invalid")
    try:
        value = os.read(start_gate_fd, 2)
    finally:
        os.close(start_gate_fd)
    if value != b"1":
        raise RuntimeError("evaluation actuator did not release the target")


def claim_and_exec_evaluation_process(
    *,
    ledger: StageDEvaluationLedger,
    manifest: StageDEvaluationExecutionManifest,
    program: EvaluationProgramBinding,
    role: Literal["client", "server"],
    actuated_receipt_path: Path,
    checkpoint_manifest: StageDCheckpointManifest,
    server_launch: EvaluationServerLaunch | None = None,
    client_launch: EvaluationClientLaunch | None = None,
) -> NoReturn:
    if program.role != role:
        raise ValueError("evaluation process role differs from its program")
    if hash_file(Path(manifest.runtime_bundle_path)) != manifest.runtime_bundle_sha256:
        raise ValueError("evaluation runtime bundle differs from its manifest")
    if hash_file(Path(program.absolute_executable)) != program.executable_sha256:
        raise ValueError("evaluation executable differs from its manifest")
    for source_name, expected_sha256 in program.source_sha256s:
        source = Path(program.working_directory) / source_name
        if source.is_symlink() or hash_file(source) != expected_sha256:
            raise ValueError("evaluation source differs from its manifest")
    if str(Path.cwd().absolute()) != program.working_directory:
        raise ValueError("evaluation working directory differs from its manifest")
    if os.environ != dict(program.environment):
        raise ValueError("evaluation environment differs from its manifest")
    if not actuated_receipt_path.is_absolute() or actuated_receipt_path.is_symlink():
        raise ValueError("evaluation actuated receipt path is invalid")
    if role == "server":
        if server_launch is None or client_launch is not None or server_launch.arm != program.arm:
            raise ValueError("evaluation server lacks its launch reservation")
        launch_epoch = server_launch.epoch
        launch_capability_sha256 = server_launch.launch_record_sha256
    else:
        if client_launch is None or server_launch is not None or client_launch.arm != program.arm:
            raise ValueError("evaluation client lacks its launch reservation")
        launch_epoch = client_launch.epoch
        launch_capability_sha256 = client_launch.launch_record_sha256
    receipt_bytes = actuated_receipt_path.read_bytes()
    receipt = ActuatedProcessReceipt.from_bytes(receipt_bytes)
    receipt.verify_program(
        arm=program.arm,
        role=role,
        epoch=launch_epoch,
        launch_capability_sha256=launch_capability_sha256,
        argv=program.argv,
        environment=program.environment,
        cgroup_root=manifest.supervisor_limits.cgroup_root,
        control_root=manifest.supervisor_limits.control_root,
        evaluation_ledger_id=manifest.evaluation_ledger_id,
        require_current=True,
    )
    if role == "server":
        assert server_launch is not None
        ledger.claim_server(server_launch, receipt_bytes)
    else:
        assert client_launch is not None
        ledger.claim_client(client_launch, receipt_bytes)
    if (
        checkpoint_manifest.arm != program.arm
        or checkpoint_manifest.manifest_sha256 != program.checkpoint_manifest_sha256
    ):
        raise ValueError("evaluation checkpoint differs from its program")
    verify_materialized_checkpoint(checkpoint_manifest, Path(program.checkpoint_root))
    os.execve(program.absolute_executable, program.argv, dict(program.environment))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("stock", "branch-global", "local"), required=True)
    parser.add_argument("--role", choices=("client", "server"), required=True)
    parser.add_argument("--actuated-receipt", type=Path, required=True)
    parser.add_argument("--start-gate-fd", type=int, required=True)
    parser.add_argument("--server-epoch", type=int)
    parser.add_argument("--server-launch-record-sha256")
    parser.add_argument("--client-epoch", type=int)
    parser.add_argument("--client-launch-record-sha256")
    parser.add_argument("--resume-task-attempt-id")
    return parser.parse_args()


def main() -> None:
    from redco.analysis.stage_d_handoff_coordinator import StageDHandoffCoordinator

    arguments = _arguments()
    await_actuator_release(arguments.start_gate_fd)
    coordinator = StageDHandoffCoordinator(arguments.handoff_root)
    ledger = coordinator.materialize_evaluation_ledger(arguments.evaluation_root)
    manifest = ledger.manifest
    program = manifest.program(arguments.arm, arguments.role)
    checkpoint_manifest = coordinator.materialize_evaluation_checkpoint(
        program.arm,
        Path(program.checkpoint_root),
    )
    server_launch = None
    client_launch = None
    if arguments.role == "server":
        if arguments.server_epoch is None or arguments.server_launch_record_sha256 is None:
            raise ValueError("evaluation server wrapper lacks its launch reservation")
        if any(
            item is not None
            for item in (
                arguments.client_epoch,
                arguments.client_launch_record_sha256,
                arguments.resume_task_attempt_id,
            )
        ):
            raise ValueError("evaluation server wrapper received client launch fields")
        server_launch = EvaluationServerLaunch(
            arguments.arm,
            arguments.server_epoch,
            arguments.server_launch_record_sha256,
        )
    else:
        if (
            arguments.client_epoch is None
            or arguments.client_launch_record_sha256 is None
            or arguments.server_epoch is not None
            or arguments.server_launch_record_sha256 is not None
        ):
            raise ValueError("evaluation client wrapper lacks its launch reservation")
        client_launch = EvaluationClientLaunch(
            arguments.arm,
            arguments.client_epoch,
            arguments.client_launch_record_sha256,
            arguments.resume_task_attempt_id,
        )
    claim_and_exec_evaluation_process(
        ledger=ledger,
        manifest=manifest,
        program=program,
        role=arguments.role,
        actuated_receipt_path=arguments.actuated_receipt,
        checkpoint_manifest=checkpoint_manifest,
        server_launch=server_launch,
        client_launch=client_launch,
    )


__all__ = ["await_actuator_release", "claim_and_exec_evaluation_process", "main"]


if __name__ == "__main__":
    main()
