"""Network-denied Docker worker protocol for frozen held-out task logic."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_evaluation_contracts import (
    EvaluationRuntimeEntrypoint,
    StageDEvaluationExecutionManifest,
    hash_file,
)
from redco.analysis.stage_d_evaluation_model_port import (
    EvaluationCallSpec,
    EvaluationModelPort,
)
from redco.analysis.stage_d_openai_response import ParsedOpenAIResponse
from redco.contracts import EventAddress, canonical_json

_MAX_MESSAGE_BYTES = 4 * 1024 * 1024
_MAX_CONTAINER_LIST_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class RuntimeTaskOutput:
    terminal_output_bytes: bytes
    task_evidence_bytes: bytes


class DockerEvaluationRuntime:
    def __init__(
        self,
        *,
        manifest: StageDEvaluationExecutionManifest,
        docker_executable: Path,
        docker_executable_sha256: str,
        task_timeout_seconds: float,
    ) -> None:
        if (
            not docker_executable.is_absolute()
            or docker_executable.is_symlink()
            or hash_file(docker_executable) != docker_executable_sha256
            or str(docker_executable) != manifest.container_runtime_executable
            or docker_executable_sha256 != manifest.container_runtime_executable_sha256
        ):
            raise ValueError("evaluation container runtime executable differs")
        if task_timeout_seconds <= 0:
            raise ValueError("evaluation worker timeout must be positive")
        runtime = Path(manifest.runtime_bundle_path)
        if runtime.is_symlink() or hash_file(runtime) != manifest.runtime_bundle_sha256:
            raise ValueError("evaluation worker runtime bundle differs")
        self._manifest = manifest
        self._docker = docker_executable
        self._timeout = task_timeout_seconds

    def command(self, *, task_attempt_id: str) -> tuple[str, ...]:
        if len(task_attempt_id) != 64 or any(
            character not in "0123456789abcdef" for character in task_attempt_id
        ):
            raise ValueError("evaluation worker task attempt ID is invalid")
        entrypoint = _task_runner(self._manifest)
        code = _worker_bootstrap(entrypoint)
        runtime = str(Path(self._manifest.runtime_bundle_path).resolve())
        container_name = _container_name(self._manifest, task_attempt_id)
        return (
            str(self._docker),
            "run",
            "--rm",
            "--interactive",
            "--name",
            container_name,
            "--label",
            f"redco.stage_d.ledger={self._manifest.evaluation_ledger_id}",
            "--label",
            f"redco.stage_d.task_attempt={task_attempt_id}",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=64",
            "--memory=1g",
            "--cpus=1",
            "--user=65534:65534",
            "--mount",
            f"type=bind,src={runtime},dst=/runtime/runtime.zip,readonly",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            self._manifest.runtime_worker_image,
            "python",
            "-I",
            "-S",
            "-c",
            code,
        )

    def run_task(
        self,
        *,
        task_id: str,
        seed: int,
        task_attempt_id: str,
        model_port: EvaluationModelPort,
    ) -> RuntimeTaskOutput:
        container_name = _container_name(self._manifest, task_attempt_id)
        with tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                self.command(task_attempt_id=task_attempt_id),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                cwd=None if os.name == "nt" else "/",
            )
            if process.stdin is None or process.stdout is None:
                process.kill()
                raise RuntimeError("evaluation worker lacks its IPC pipes")
            timed_out = threading.Event()

            def kill_at_deadline() -> None:
                timed_out.set()
                process.kill()

            watchdog = threading.Timer(self._timeout, kill_at_deadline)
            watchdog.daemon = True
            watchdog.start()
            try:
                _write_message(
                    process.stdin,
                    {
                        "schema_version": 1,
                        "domain": "redco-stage-d-worker-start-v1",
                        "task_id": task_id,
                        "seed": seed,
                    },
                )
                output = _serve_worker(process, model_port)
                process.stdin.close()
                return_code = process.wait()
                stderr_file.seek(0)
                stderr = stderr_file.read(_MAX_MESSAGE_BYTES + 1)
                trailing = process.stdout.read(_MAX_MESSAGE_BYTES + 1)
                if timed_out.is_set():
                    raise TimeoutError("evaluation worker exceeded its task deadline")
                if return_code != 0 or trailing or len(stderr) > _MAX_MESSAGE_BYTES:
                    raise RuntimeError("evaluation worker did not terminate cleanly")
                return output
            except BaseException as error:
                if process.poll() is None:
                    process.kill()
                process.wait()
                if timed_out.is_set() and not isinstance(error, TimeoutError):
                    raise TimeoutError("evaluation worker exceeded its task deadline") from error
                raise
            finally:
                watchdog.cancel()
                self._remove_container(container_name)

    def _remove_container(self, container_name: str) -> None:
        result = subprocess.run(
            (str(self._docker), "rm", "--force", container_name),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=self._timeout,
        )
        if result.returncode != 0 and (
            result.returncode != 1 or b"No such container" not in result.stderr
        ):
            raise RuntimeError("evaluation worker container cleanup failed")


def _container_name(
    manifest: StageDEvaluationExecutionManifest,
    task_attempt_id: str,
) -> str:
    return f"redco-d-{manifest.evaluation_ledger_id[:12]}-{task_attempt_id[:32]}"


def cleanup_evaluation_containers(
    manifest: StageDEvaluationExecutionManifest,
    *,
    timeout_seconds: float,
) -> tuple[str, ...]:
    """Remove only containers carrying this evaluation ledger's exact label."""
    executable = Path(manifest.container_runtime_executable)
    if (
        timeout_seconds <= 0
        or executable.is_symlink()
        or hash_file(executable) != manifest.container_runtime_executable_sha256
    ):
        raise ValueError("evaluation container cleanup binding differs")
    selector = f"label=redco.stage_d.ledger={manifest.evaluation_ledger_id}"
    listed = subprocess.run(
        (str(executable), "ps", "--all", "--quiet", "--filter", selector),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    if listed.returncode != 0 or len(listed.stdout) > _MAX_CONTAINER_LIST_BYTES or listed.stderr:
        raise RuntimeError("evaluation container roster query failed")
    identifiers = tuple(line for line in listed.stdout.decode("ascii").splitlines() if line)
    if any(
        len(identifier) < 12
        or len(identifier) > 64
        or any(character not in "0123456789abcdef" for character in identifier)
        for identifier in identifiers
    ):
        raise RuntimeError("evaluation container roster is malformed")
    if identifiers:
        removed = subprocess.run(
            (str(executable), "rm", "--force", *identifiers),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
        if removed.returncode != 0:
            raise RuntimeError("evaluation container cleanup failed")
    verified = subprocess.run(
        (str(executable), "ps", "--all", "--quiet", "--filter", selector),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    if verified.returncode != 0 or verified.stdout or verified.stderr:
        raise RuntimeError("evaluation containers remain after cleanup")
    return identifiers


def _worker_bootstrap(entrypoint: EvaluationRuntimeEntrypoint) -> str:
    return (
        "import importlib,sys;"
        "sys.path.insert(0,'/runtime/runtime.zip');"
        f"_module=importlib.import_module({entrypoint.module!r});"
        f"assert _module.__file__ == {'/runtime/runtime.zip/' + entrypoint.member_path!r};"
        f"_entry=getattr(_module,{entrypoint.callable_name!r});"
        "_entry()"
    )


def _task_runner(
    manifest: StageDEvaluationExecutionManifest,
) -> EvaluationRuntimeEntrypoint:
    matches = [item for item in manifest.runtime_entrypoints if item.role == "task_runner"]
    if len(matches) != 1 or matches[0].api_schema != "redco-stage-d-worker-ipc-v1":
        raise ValueError("evaluation task-runner entrypoint schema differs")
    return matches[0]


def _serve_worker(
    process: subprocess.Popen[bytes],
    model_port: EvaluationModelPort,
) -> RuntimeTaskOutput:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("evaluation worker lacks its IPC pipes")
    while True:
        message = _read_message(process.stdout)
        domain = message.get("domain")
        if domain == "redco-stage-d-worker-call-v1":
            if set(message) != {"schema_version", "domain", "address", "payload"}:
                raise ValueError("evaluation worker call fields differ")
            address = message["address"]
            if not isinstance(address, dict) or set(address) != {
                "parent_node_id",
                "turn_index",
                "call_slot_index",
                "occurrence_index",
            }:
                raise ValueError("evaluation worker address fields differ")
            if not isinstance(message["payload"], dict):
                raise ValueError("evaluation worker call payload is invalid")
            parsed = model_port.call(
                EvaluationCallSpec(EventAddress(**address), message["payload"])
            )
            _write_model_response(process.stdin, parsed)
        elif domain == "redco-stage-d-worker-terminal-v1":
            if set(message) != {
                "schema_version",
                "domain",
                "terminal_output",
                "task_evidence",
            }:
                raise ValueError("evaluation worker terminal fields differ")
            return RuntimeTaskOutput(
                canonical_json(message["terminal_output"]),
                canonical_json(message["task_evidence"]),
            )
        else:
            raise ValueError("evaluation worker message domain is invalid")


def _write_model_response(pipe: Any, parsed: ParsedOpenAIResponse) -> None:
    _write_message(
        pipe,
        {
            "schema_version": 1,
            "domain": "redco-stage-d-worker-model-response-v1",
            "response": json.loads(parsed.to_bytes()),
        },
    )


def _write_message(pipe: Any, payload: dict[str, Any]) -> None:
    value = canonical_json(payload)
    if len(value) > _MAX_MESSAGE_BYTES:
        raise ValueError("evaluation worker IPC message is too large")
    pipe.write(value + b"\n")
    pipe.flush()


def _read_message(pipe: Any) -> dict[str, Any]:
    value = pipe.readline(_MAX_MESSAGE_BYTES + 2)
    if not value or len(value) > _MAX_MESSAGE_BYTES + 1 or not value.endswith(b"\n"):
        raise ValueError("evaluation worker IPC framing is invalid")
    raw = value[:-1]
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation worker IPC is not JSON") from error
    if not isinstance(payload, dict) or canonical_json(payload) != raw:
        raise ValueError("evaluation worker IPC is not canonical JSON")
    if payload.get("schema_version") != 1:
        raise ValueError("evaluation worker IPC schema differs")
    return payload


__all__ = [
    "DockerEvaluationRuntime",
    "RuntimeTaskOutput",
    "cleanup_evaluation_containers",
]
