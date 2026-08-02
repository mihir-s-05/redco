from __future__ import annotations

import hashlib
import io
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_stage_d_evaluation_ledger import _RUNTIME, _frozen_inputs

import redco.analysis.stage_d_evaluation_worker as worker_module
from redco.analysis.stage_d_evaluation_contracts import (
    StageDEvaluationExecutionManifest,
)
from redco.analysis.stage_d_evaluation_worker import DockerEvaluationRuntime


def _runtime(tmp_path: Path, *, timeout: float = 1.0) -> DockerEvaluationRuntime:
    _, manifest_bytes, _ = _frozen_inputs()
    bundle = tmp_path / "runtime.zip"
    bundle.write_bytes(_RUNTIME)
    executable = tmp_path / "docker"
    executable.write_bytes(b"frozen docker executable")
    manifest = replace(
        StageDEvaluationExecutionManifest.from_bytes(manifest_bytes),
        runtime_bundle_path=str(bundle.resolve()),
        container_runtime_executable=str(executable.resolve()),
        container_runtime_executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
    )
    return DockerEvaluationRuntime(
        manifest=manifest,
        docker_executable=executable.resolve(),
        docker_executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        task_timeout_seconds=timeout,
    )


def test_container_command_has_network_and_privilege_denials(tmp_path: Path) -> None:
    command = _runtime(tmp_path).command(task_attempt_id="a" * 64)
    assert command[1:4] == ("run", "--rm", "--interactive")
    assert command[command.index("--name") + 1].startswith("redco-d-")
    assert f"redco.stage_d.task_attempt={'a' * 64}" in command
    assert "--network=none" in command
    assert "--read-only" in command
    assert (
        command[command.index("--cap-drop=ALL")],
        command[command.index("--security-opt=no-new-privileges")],
    ) == ("--cap-drop=ALL", "--security-opt=no-new-privileges")
    assert command[command.index("--user=65534:65534")] == "--user=65534:65534"
    assert command[command.index("--mount") + 1].endswith(",dst=/runtime/runtime.zip,readonly")
    assert "sys.path.insert(0,'/runtime/runtime.zip')" in command[-1]
    assert "_module.__file__ == '/runtime/runtime.zip/task_runtime.py'" in command[-1]


class _BlockingReader:
    def __init__(self) -> None:
        self.killed = threading.Event()

    def readline(self, _limit: int) -> bytes:
        self.killed.wait(1.0)
        return b""

    def read(self, _limit: int) -> bytes:
        return b""


class _HungProcess:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.stdin = io.BytesIO()
        self.stdout = _BlockingReader()
        self._return_code: int | None = None

    def kill(self) -> None:
        self._return_code = -9
        self.stdout.killed.set()

    def wait(self) -> int:
        self.stdout.killed.wait(1.0)
        assert self._return_code is not None
        return self._return_code

    def poll(self) -> int | None:
        return self._return_code


def test_worker_wall_deadline_kills_blocked_ipc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker_module.subprocess, "Popen", _HungProcess)
    monkeypatch.setattr(
        worker_module.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": 0})(),
    )
    with pytest.raises(TimeoutError, match="task deadline"):
        _runtime(tmp_path, timeout=0.01).run_task(
            task_id="heldout-1",
            seed=9101,
            task_attempt_id="a" * 64,
            model_port=object(),  # type: ignore[arg-type]
        )
