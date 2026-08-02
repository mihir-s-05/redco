from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from redco.analysis.stage_d_checkpoint_evidence import (
    CheckpointMember,
    StageDCheckpointManifest,
)
from redco.analysis.stage_d_evaluation_actuation import ActuatedProcessReceipt
from redco.analysis.stage_d_evaluation_barrier import (
    EvaluationCheckpointBinding,
    StageDEvaluationAuthorization,
    StageDEvaluationPlan,
    StageDEvaluationTask,
)
from redco.analysis.stage_d_evaluation_capabilities import (
    EvaluationServerLaunch,
)
from redco.analysis.stage_d_evaluation_contracts import (
    EvaluationProgramBinding,
    EvaluationRuntimeEntrypoint,
    EvaluationScheduleUnit,
    EvaluationSupervisorLimits,
    StageDEvaluationExecutionManifest,
    evaluation_environment_sha256,
    hash_file,
)
from redco.analysis.stage_d_evaluation_ledger import StageDEvaluationLedger
from redco.analysis.stage_d_process_supervision import command_sha256

_ARMS = ("stock", "branch-global", "local")


def _sha(character: str) -> str:
    return character * 64


def _runtime_archive(path: Path) -> None:
    source_root = Path(__file__).parents[1] / "src"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted((source_root / "redco").rglob("*.py")):
            archive.write(source, source.relative_to(source_root).as_posix())


def _frozen_ledger(
    tmp_path: Path,
    *,
    target: Path,
    output: Path,
    runtime: Path,
    environment: tuple[tuple[str, str], ...],
) -> tuple[
    StageDEvaluationLedger,
    StageDEvaluationExecutionManifest,
    dict[str, StageDCheckpointManifest],
]:
    plan = StageDEvaluationPlan(
        tasks=(StageDEvaluationTask("heldout-1", 9101),),
        reward_min=0.0,
        reward_max=1.0,
        success_reward_threshold=0.5,
    ).to_bytes()
    config = b"frozen=true\n"
    executable = str(Path(sys.executable).resolve())
    checkpoint_manifests = {}
    for arm in _ARMS:
        checkpoint_root = tmp_path / f"checkpoint-{arm}"
        checkpoint_root.mkdir()
        for name, value in (
            ("STABLE", b""),
            ("adapter_config.json", b"{}"),
            ("adapter_model.safetensors", f"adapter-{arm}".encode()),
        ):
            (checkpoint_root / name).write_bytes(value)
        checkpoint_manifests[arm] = StageDCheckpointManifest(
            arm=arm,  # type: ignore[arg-type]
            trainer_step=1,
            base_model_manifest_sha256=_sha("e"),
            post_model_sha256=_sha("a"),
            members=tuple(
                CheckpointMember(
                    path.name,
                    path.stat().st_size,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                for path in sorted(checkpoint_root.iterdir())
            ),
        )
        if os.name != "nt":
            for path in checkpoint_root.iterdir():
                path.chmod(0o444)
            checkpoint_root.chmod(0o555)
    checkpoints = tuple(
        EvaluationCheckpointBinding(
            arm,  # type: ignore[arg-type]
            checkpoint_manifests[arm].manifest_sha256,
            checkpoint_manifests[arm].post_model_sha256,
            _sha("b"),
        )
        for arm in _ARMS
    )
    programs = tuple(
        EvaluationProgramBinding(
            arm=arm,
            role=role,
            absolute_executable=executable,
            executable_sha256=hash_file(Path(executable)),
            argv=(
                executable,
                target.name,
                str(output),
                *([str((tmp_path / f"checkpoint-{arm}").resolve())] if role == "server" else []),
            ),
            working_directory=str(tmp_path.resolve()),
            checkpoint_root=str((tmp_path / f"checkpoint-{arm}").resolve()),
            environment=environment,
            source_sha256s=((target.name, hash_file(target)),),
            checkpoint_manifest_sha256=checkpoints[index].checkpoint_manifest_sha256,
            post_model_sha256=checkpoints[index].post_model_sha256,
            reload_evidence_sha256=checkpoints[index].reload_evidence_sha256,
            endpoint=f"http://127.0.0.1:{8700 + index}",
            gpu_assignment=(index,),
            cache_namespace=f"process-test-{arm}",
        )
        for index, arm in enumerate(_ARMS)
        for role in ("server", "client")
    )
    manifest = StageDEvaluationExecutionManifest(
        evaluation_ledger_id=_sha("6"),
        protocol_manifest_sha256=_sha("e"),
        trainer_ledger_head_sha256=_sha("f"),
        trainer_record_count=41,
        heldout_eval_config_sha256=hashlib.sha256(config).hexdigest(),
        evaluation_plan_sha256=hashlib.sha256(plan).hexdigest(),
        decision_rule_sha256=_sha("1"),
        runtime_entrypoints=(
            EvaluationRuntimeEntrypoint(
                "task_runner",
                "task_runtime.py",
                "task_runtime",
                "run_task",
                "redco-stage-d-worker-ipc-v1",
                _sha("2"),
            ),
            EvaluationRuntimeEntrypoint(
                "scorer", "scorer.py", "scorer", "score", "redco-stage-d-scorer-v1", _sha("3")
            ),
            EvaluationRuntimeEntrypoint(
                "request_serializer",
                "serializer.py",
                "serializer",
                "serialize",
                "redco-stage-d-request-serializer-v1",
                _sha("4"),
            ),
        ),
        runtime_worker_image="python@sha256:" + "5" * 64,
        runtime_bundle_path=str(runtime.resolve()),
        runtime_bundle_sha256=hash_file(runtime),
        container_runtime_executable="/usr/bin/docker",
        container_runtime_executable_sha256=_sha("7"),
        supervisor_limits=EvaluationSupervisorLimits(
            str((tmp_path / "evaluation-control").resolve()),
            str((tmp_path / "evaluation-logs").resolve()),
            "/sys/fs/cgroup",
            7200,
            30,
            30,
            5,
            50,
            1048576,
        ),
        max_server_launches_per_arm=2,
        max_client_launches_per_arm=2,
        server_replacement_policy="before-first-dispatch-only-v1",
        programs=programs,
        schedule=tuple(
            EvaluationScheduleUnit(index, arm, 0, "heldout-1", 9101)
            for index, arm in enumerate(_ARMS)
        ),
    )
    authorization = StageDEvaluationAuthorization(
        handoff_training_adoption_record_sha256=_sha("7"),
        campaign_manifest_sha256=_sha("5"),
        protocol_manifest_sha256=manifest.protocol_manifest_sha256,
        trainer_ledger_head_sha256=manifest.trainer_ledger_head_sha256,
        trainer_record_count=manifest.trainer_record_count,
        heldout_eval_config_sha256=manifest.heldout_eval_config_sha256,
        evaluation_plan_sha256=manifest.evaluation_plan_sha256,
        execution_manifest_sha256=manifest.manifest_sha256,
        checkpoints=checkpoints,
    )
    ledger = StageDEvaluationLedger.create(
        tmp_path / manifest.evaluation_ledger_id,
        authorization_bytes=authorization.to_bytes(),
        execution_manifest_bytes=manifest.to_bytes(),
        evaluation_plan_bytes=plan,
        runtime_bundle_bytes=runtime.read_bytes(),
    )
    return ledger, manifest, checkpoint_manifests


def _claim_exec_launcher(
    *,
    manifest_path: Path,
    ledger_root: Path,
    checkpoint_manifest_path: Path,
    role: str,
    server_launch: EvaluationServerLaunch,
) -> str:
    return "\n".join(
        (
            "from pathlib import Path",
            "from redco.analysis.stage_d_evaluation_contracts import "
            "StageDEvaluationExecutionManifest",
            "from redco.analysis.stage_d_evaluation_ledger import StageDEvaluationLedger",
            "from redco.analysis.stage_d_checkpoint_evidence import StageDCheckpointManifest",
            "from redco.analysis.stage_d_evaluation_capabilities import EvaluationServerLaunch",
            "from redco.analysis.stage_d_evaluation_process import "
            "await_actuator_release, claim_and_exec_evaluation_process",
            "import argparse",
            "parser = argparse.ArgumentParser()",
            "parser.add_argument('--actuated-receipt', type=Path, required=True)",
            "parser.add_argument('--start-gate-fd', type=int, required=True)",
            "args = parser.parse_args()",
            "await_actuator_release(args.start_gate_fd)",
            f"manifest = StageDEvaluationExecutionManifest.from_bytes("
            f"Path({str(manifest_path)!r}).read_bytes())",
            f"ledger = StageDEvaluationLedger(Path({str(ledger_root)!r}))",
            f"checkpoint_manifest = StageDCheckpointManifest.from_bytes("
            f"Path({str(checkpoint_manifest_path)!r}).read_bytes())",
            "server_launch = EvaluationServerLaunch("
            f"'stock', {server_launch.epoch}, {server_launch.launch_record_sha256!r})",
            "claim_and_exec_evaluation_process(",
            "    ledger=ledger,",
            "    manifest=manifest,",
            f"    program=manifest.program('stock', {role!r}),",
            f"    role={role!r},",
            "    actuated_receipt_path=args.actuated_receipt,",
            "    checkpoint_manifest=checkpoint_manifest,",
            "    server_launch=server_launch,",
            "    client_launch=None,",
            ")",
        )
    )


def _wait_for(path: Path, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {path}")


def _actuator_command(
    *,
    tmp_path: Path,
    ledger: StageDEvaluationLedger,
    manifest: StageDEvaluationExecutionManifest,
    launch: EvaluationServerLaunch,
    launcher: str,
    suffix: str,
) -> tuple[list[str], Path, Path]:
    from redco.analysis.stage_d_evaluation_actuation import EvaluationSupervisorIdentity
    from redco.contracts import canonical_json

    supervisor = EvaluationSupervisorIdentity.current()
    program = manifest.program("stock", "server")
    actuation_attempt_id = hashlib.sha256(suffix.encode()).hexdigest()[:32]
    ledger.reserve_actuation_attempt(
        actuation_attempt_id=actuation_attempt_id,
        arm="stock",
        role="server",
        epoch=launch.epoch,
        launch_record_sha256=launch.launch_record_sha256,
        supervisor=supervisor,
    )
    attempt_root = (
        Path(manifest.supervisor_limits.control_root)
        / manifest.evaluation_ledger_id
        / actuation_attempt_id
    )
    attempt_root.mkdir(parents=True)
    supervisor_path = attempt_root / "supervisor.json"
    environment_path = attempt_root / "environment.json"
    receipt_path = attempt_root / "target-receipt.json"
    stop_path = attempt_root / "stop-request.json"
    supervisor_path.write_bytes(canonical_json(supervisor.to_payload()))
    environment_path.write_bytes(canonical_json(dict(program.environment)))
    command = [
        sys.executable,
        "-m",
        "redco.analysis.stage_d_evaluation_actuator",
        "--supervisor-identity",
        str(supervisor_path),
        "--receipt",
        str(receipt_path),
        "--stop-request",
        str(stop_path),
        "--cgroup-root",
        manifest.supervisor_limits.cgroup_root,
        "--target-environment",
        str(environment_path),
        "--target-cwd",
        program.working_directory,
        "--target-stdout",
        str(tmp_path / f"target-{suffix}.stdout"),
        "--target-stderr",
        str(tmp_path / f"target-{suffix}.stderr"),
        "--arm",
        "stock",
        "--role",
        "server",
        "--epoch",
        str(launch.epoch),
        "--launch-capability-sha256",
        launch.launch_record_sha256,
        "--actuation-attempt-id",
        actuation_attempt_id,
        "--program-command-sha256",
        command_sha256(program.argv),
        "--program-environment-sha256",
        evaluation_environment_sha256(program.environment),
        "--poll-interval-seconds",
        "0.02",
        "--stop-timeout-seconds",
        "5",
        "--max-log-bytes",
        "1048576",
        "--",
        sys.executable,
        "-c",
        launcher,
    ]
    return command, receipt_path, stop_path


@pytest.mark.skipif(
    os.name == "nt" or getattr(os, "geteuid", lambda: -1)() != 0,
    reason="same-PID actuator test requires root cgroup-v2 delegation",
)
def test_claim_and_exec_preserves_process_identity(tmp_path: Path) -> None:
    target = tmp_path / "evaluation_target.py"
    target.write_text(
        "import json, os, pathlib, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid': os.getpid()}))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    output = tmp_path / "server-output.json"
    runtime = tmp_path / "runtime.zip"
    _runtime_archive(runtime)
    environment = tuple(
        sorted(
            {
                "LC_ALL": "C.UTF-8",
                "PATH": os.environ["PATH"],
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(runtime.resolve()),
            }.items()
        )
    )
    ledger, manifest, checkpoint_manifests = _frozen_ledger(
        tmp_path,
        target=target,
        output=output,
        runtime=runtime,
        environment=environment,
    )
    manifest_path = tmp_path / "execution-manifest.json"
    manifest_path.write_bytes(manifest.to_bytes())
    checkpoint_manifest_path = tmp_path / "stock-checkpoint-manifest.json"
    checkpoint_manifest_path.write_bytes(checkpoint_manifests["stock"].to_bytes())
    server_launch = ledger.reserve_server_launch("stock")
    launcher = _claim_exec_launcher(
        manifest_path=manifest_path,
        ledger_root=ledger.root,
        checkpoint_manifest_path=checkpoint_manifest_path,
        role="server",
        server_launch=server_launch,
    )
    command, receipt_path, stop_path = _actuator_command(
        tmp_path=tmp_path,
        ledger=ledger,
        manifest=manifest,
        launch=server_launch,
        launcher=launcher,
        suffix="single",
    )
    actuator = subprocess.Popen(
        command,
        cwd=tmp_path,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _wait_for(receipt_path)
        _wait_for(output)
        receipt = ActuatedProcessReceipt.from_bytes(receipt_path.read_bytes())
        assert receipt.pid == json.loads(output.read_text(encoding="utf-8"))["pid"]
        assert dict(ledger.inspect().server_claims)["stock"] == receipt.receipt_sha256
        ledger.finish_actuation_attempt(
            actuation_attempt_id=receipt.actuation_attempt_id,
            disposition="claimed",
            process_receipt_bytes=receipt.to_bytes(),
        )
        stop_path.write_bytes(b"stop")
        _stdout, stderr = actuator.communicate(timeout=10)
        assert actuator.returncode == 125, stderr
    finally:
        if actuator.poll() is None:
            actuator.kill()
            actuator.wait(timeout=5)


@pytest.mark.skipif(
    os.name == "nt" or getattr(os, "geteuid", lambda: -1)() != 0,
    reason="duplicate-actuator test requires root cgroup-v2 delegation",
)
def test_two_actuators_execute_one_claimed_target(tmp_path: Path) -> None:
    target = tmp_path / "evaluation_target.py"
    output = tmp_path / "executions.txt"
    target.write_text(
        "import os, pathlib, sys, time\n"
        "with pathlib.Path(sys.argv[1]).open('a', encoding='utf-8') as stream:\n"
        "    stream.write(f'{os.getpid()}\\n')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime.zip"
    _runtime_archive(runtime)
    environment = tuple(
        sorted(
            {
                "LC_ALL": "C.UTF-8",
                "PATH": os.environ["PATH"],
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(runtime.resolve()),
            }.items()
        )
    )
    ledger, manifest, checkpoint_manifests = _frozen_ledger(
        tmp_path,
        target=target,
        output=output,
        runtime=runtime,
        environment=environment,
    )
    manifest_path = tmp_path / "execution-manifest.json"
    manifest_path.write_bytes(manifest.to_bytes())
    checkpoint_manifest_path = tmp_path / "stock-checkpoint-manifest.json"
    checkpoint_manifest_path.write_bytes(checkpoint_manifests["stock"].to_bytes())
    launch = ledger.reserve_server_launch("stock")
    launcher = _claim_exec_launcher(
        manifest_path=manifest_path,
        ledger_root=ledger.root,
        checkpoint_manifest_path=checkpoint_manifest_path,
        role="server",
        server_launch=launch,
    )
    specs = [
        _actuator_command(
            tmp_path=tmp_path,
            ledger=ledger,
            manifest=manifest,
            launch=launch,
            launcher=launcher,
            suffix=str(index),
        )
        for index in range(2)
    ]
    processes = [
        subprocess.Popen(
            command,
            cwd=tmp_path,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        for command, _receipt, _stop in specs
    ]
    try:
        for _command, receipt_path, _stop in specs:
            _wait_for(receipt_path)
        _wait_for(output)
        winner_sha256 = dict(ledger.inspect().server_claims)["stock"]
        receipts = [
            ActuatedProcessReceipt.from_bytes(receipt_path.read_bytes())
            for _command, receipt_path, _stop in specs
        ]
        winner_index = next(
            index
            for index, receipt in enumerate(receipts)
            if receipt.receipt_sha256 == winner_sha256
        )
        specs[winner_index][2].write_bytes(b"stop")
        for process in processes:
            process.communicate(timeout=10)
        for index, receipt in enumerate(receipts):
            if index == winner_index:
                ledger.finish_actuation_attempt(
                    actuation_attempt_id=receipt.actuation_attempt_id,
                    disposition="claimed",
                    process_receipt_bytes=receipt.to_bytes(),
                )
            else:
                ledger.finish_actuation_attempt(
                    actuation_attempt_id=receipt.actuation_attempt_id,
                    disposition="lost-claim-and-drained",
                    cleanup_evidence_bytes=b"losing actuator drained",
                )
        assert sorted(process.returncode for process in processes) == [1, 125]
        assert len(output.read_text(encoding="utf-8").splitlines()) == 1
        assert any(
            b"already claimed" in (tmp_path / f"target-{index}.stderr").read_bytes()
            for index in range(2)
        )
        assert all(not Path(receipt.cgroup_path).exists() for receipt in receipts)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
