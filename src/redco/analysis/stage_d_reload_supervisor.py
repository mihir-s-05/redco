"""Supervisor-owned, two-process checkpoint reload evidence for Stage D."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from redco.analysis.stage_d_checkpoint_evidence import (
    StageDCheckpointManifest,
    StageDReloadEvidence,
)
from redco.analysis.stage_d_objective_binding import ArmName
from redco.analysis.stage_d_process_supervision import linux_process_identity
from redco.contracts import canonical_json

_PROBE_DOMAIN = "redco-stage-d-reload-probe-v1"
_RESULT_DOMAIN = "redco-stage-d-reload-worker-result-v1"
_COMPLETION_DOMAIN = "redco-stage-d-reload-worker-completion-v1"
_ARMS = {"stock", "branch-global", "local"}
ReloadFaultHook = Callable[[str, Path], None]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class StageDReloadProbe:
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    tokenizer_manifest_sha256: str
    renderer_manifest_sha256: str
    runtime_manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.prompt_token_ids
            or len(self.prompt_token_ids) > 4096
            or any(type(item) is not int or item < 0 for item in self.prompt_token_ids)
        ):
            raise ValueError("reload probe token IDs are invalid")
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 64:
            raise ValueError("reload probe generation budget is invalid")
        for name in (
            "tokenizer_manifest_sha256",
            "renderer_manifest_sha256",
            "runtime_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    @property
    def probe_sha256(self) -> str:
        return _sha256(self.to_bytes())

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": _PROBE_DOMAIN,
                "prompt_token_ids": list(self.prompt_token_ids),
                "max_new_tokens": self.max_new_tokens,
                "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
                "renderer_manifest_sha256": self.renderer_manifest_sha256,
                "runtime_manifest_sha256": self.runtime_manifest_sha256,
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDReloadProbe:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("reload probe is not JSON") from error
        fields = {
            "schema_version",
            "domain",
            "prompt_token_ids",
            "max_new_tokens",
            "tokenizer_manifest_sha256",
            "renderer_manifest_sha256",
            "runtime_manifest_sha256",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or payload.get("schema_version") != 1
            or payload.get("domain") != _PROBE_DOMAIN
            or canonical_json(payload) != value
            or not isinstance(payload.get("prompt_token_ids"), list)
        ):
            raise ValueError("reload probe is noncanonical or has different fields")
        return cls(
            tuple(payload["prompt_token_ids"]),
            payload["max_new_tokens"],
            payload["tokenizer_manifest_sha256"],
            payload["renderer_manifest_sha256"],
            payload["runtime_manifest_sha256"],
        )


@dataclass(frozen=True, slots=True)
class ReloadWorkerResult:
    arm: ArmName
    launch_nonce: str
    pid: int
    boot_id: str
    process_start_ticks: str
    checkpoint_manifest_sha256: str
    post_model_sha256: str
    loaded_model_sha256: str
    reload_probe_sha256: str
    base_model_manifest_sha256: str
    tokenizer_manifest_sha256: str
    renderer_manifest_sha256: str
    runtime_manifest_sha256: str
    python_executable_sha256: str
    worker_source_sha256: str
    worker_command_sha256: str
    worker_environment_sha256: str
    working_directory_sha256: str
    output_sha256: str

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise ValueError("reload result arm is invalid")
        if len(self.launch_nonce) != 64:
            raise ValueError("reload launch nonce is invalid")
        _require_sha256(self.launch_nonce, "reload launch nonce")
        if type(self.pid) is not int or self.pid < 1:
            raise ValueError("reload result PID is invalid")
        if not self.boot_id or not self.process_start_ticks.isdigit():
            raise ValueError("reload result process identity is invalid")
        for name in (
            "checkpoint_manifest_sha256",
            "post_model_sha256",
            "loaded_model_sha256",
            "reload_probe_sha256",
            "base_model_manifest_sha256",
            "tokenizer_manifest_sha256",
            "renderer_manifest_sha256",
            "runtime_manifest_sha256",
            "python_executable_sha256",
            "worker_source_sha256",
            "worker_command_sha256",
            "worker_environment_sha256",
            "working_directory_sha256",
            "output_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    @property
    def identity(self) -> str:
        return _sha256(self.to_bytes())

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": _RESULT_DOMAIN,
                "arm": self.arm,
                "launch_nonce": self.launch_nonce,
                "pid": self.pid,
                "boot_id": self.boot_id,
                "process_start_ticks": self.process_start_ticks,
                "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
                "post_model_sha256": self.post_model_sha256,
                "loaded_model_sha256": self.loaded_model_sha256,
                "reload_probe_sha256": self.reload_probe_sha256,
                "base_model_manifest_sha256": self.base_model_manifest_sha256,
                "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
                "renderer_manifest_sha256": self.renderer_manifest_sha256,
                "runtime_manifest_sha256": self.runtime_manifest_sha256,
                "python_executable_sha256": self.python_executable_sha256,
                "worker_source_sha256": self.worker_source_sha256,
                "worker_command_sha256": self.worker_command_sha256,
                "worker_environment_sha256": self.worker_environment_sha256,
                "working_directory_sha256": self.working_directory_sha256,
                "output_sha256": self.output_sha256,
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> ReloadWorkerResult:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("reload worker result is not JSON") from error
        fields = {
            "schema_version",
            "domain",
            "arm",
            "launch_nonce",
            "pid",
            "boot_id",
            "process_start_ticks",
            "checkpoint_manifest_sha256",
            "post_model_sha256",
            "loaded_model_sha256",
            "reload_probe_sha256",
            "base_model_manifest_sha256",
            "tokenizer_manifest_sha256",
            "renderer_manifest_sha256",
            "runtime_manifest_sha256",
            "python_executable_sha256",
            "worker_source_sha256",
            "worker_command_sha256",
            "worker_environment_sha256",
            "working_directory_sha256",
            "output_sha256",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or payload.get("schema_version") != 1
            or payload.get("domain") != _RESULT_DOMAIN
            or canonical_json(payload) != value
        ):
            raise ValueError("reload worker result is noncanonical or has different fields")
        return cls(**{key: payload[key] for key in fields - {"schema_version", "domain"}})


@dataclass(frozen=True, slots=True)
class ReloadWorkerCompletion:
    result: ReloadWorkerResult
    generated_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(type(item) is not int or item < 0 for item in self.generated_token_ids):
            raise ValueError("reload completion token IDs are invalid")
        if self.result.output_sha256 != _sha256(self.output_bytes):
            raise ValueError("reload completion output digest differs")

    @property
    def output_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-reload-output-v1",
                "generated_token_ids": list(self.generated_token_ids),
            }
        )

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": _COMPLETION_DOMAIN,
                "result": json.loads(self.result.to_bytes()),
                "generated_token_ids": list(self.generated_token_ids),
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> ReloadWorkerCompletion:
        payload = _canonical_object(value, "reload worker completion")
        if (
            set(payload) != {"schema_version", "domain", "result", "generated_token_ids"}
            or payload.get("schema_version") != 1
            or payload.get("domain") != _COMPLETION_DOMAIN
            or not isinstance(payload.get("result"), dict)
            or not isinstance(payload.get("generated_token_ids"), list)
        ):
            raise ValueError("reload worker completion fields differ")
        result = ReloadWorkerResult.from_bytes(canonical_json(payload["result"]))
        return cls(result, tuple(payload["generated_token_ids"]))


def run_fresh_reload_pair(
    *,
    arm: ArmName,
    checkpoint_root: Path,
    checkpoint_manifest_path: Path,
    reload_probe_path: Path,
    base_model_root: Path,
    base_model_manifest_path: Path,
    tokenizer_manifest_path: Path,
    renderer_manifest_path: Path,
    runtime_manifest_path: Path,
    evidence_root: Path,
    timeout_seconds: int,
    backend: Literal["transformers-peft", "test-fixture"] = "transformers-peft",
    allow_test_backend: bool = False,
    fault_hook: ReloadFaultHook | None = None,
) -> tuple[
    StageDReloadEvidence,
    tuple[bytes, bytes],
    tuple[bytes, bytes],
]:
    """Spawn exactly two sequential workers and validate their real process identities."""
    if arm not in _ARMS:
        raise ValueError("reload arm is invalid")
    if backend == "test-fixture" and not allow_test_backend:
        raise ValueError("test reload backend is forbidden in deployment")
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ValueError("reload timeout must be positive")
    manifest_bytes = checkpoint_manifest_path.read_bytes()
    manifest = StageDCheckpointManifest.from_bytes(manifest_bytes)
    manifest.verify_directory(checkpoint_root, verify_semantic=backend != "test-fixture")
    probe_bytes = reload_probe_path.read_bytes()
    probe = StageDReloadProbe.from_bytes(probe_bytes)
    expected_files = {
        base_model_manifest_path: manifest.base_model_manifest_sha256,
        tokenizer_manifest_path: probe.tokenizer_manifest_sha256,
        renderer_manifest_path: probe.renderer_manifest_sha256,
        runtime_manifest_path: probe.runtime_manifest_sha256,
    }
    for path, expected in expected_files.items():
        if _file_sha256(path) != expected:
            raise ValueError(f"reload identity file differs: {path.name}")
    evidence_root.mkdir(parents=True, exist_ok=True)
    results: list[ReloadWorkerResult] = []
    outputs: list[bytes] = []
    for ordinal in range(2):
        result, output = _run_or_resume_worker(
            arm=arm,
            ordinal=ordinal + 1,
            backend=backend,
            checkpoint_root=checkpoint_root,
            checkpoint_manifest_path=checkpoint_manifest_path,
            reload_probe_path=reload_probe_path,
            base_model_root=base_model_root,
            base_model_manifest_path=base_model_manifest_path,
            tokenizer_manifest_path=tokenizer_manifest_path,
            renderer_manifest_path=renderer_manifest_path,
            runtime_manifest_path=runtime_manifest_path,
            evidence_root=evidence_root,
            timeout_seconds=timeout_seconds,
            manifest=manifest,
            manifest_bytes=manifest_bytes,
            probe=probe,
            probe_bytes=probe_bytes,
            fault_hook=fault_hook,
        )
        results.append(result)
        outputs.append(output)
    evidence = StageDReloadEvidence(
        arm=arm,
        checkpoint_manifest_sha256=_sha256(manifest_bytes),
        post_model_sha256=manifest.post_model_sha256,
        reload_probe_sha256=_sha256(probe_bytes),
        process_identities=(results[0].identity, results[1].identity),
        output_sha256s=(_sha256(outputs[0]), _sha256(outputs[1])),
    )
    evidence.verify_output_bytes((outputs[0], outputs[1]))
    result_bytes = (results[0].to_bytes(), results[1].to_bytes())
    evidence.verify_process_result_bytes(result_bytes)
    return evidence, (outputs[0], outputs[1]), result_bytes


def _run_or_resume_worker(
    *,
    arm: ArmName,
    ordinal: int,
    backend: str,
    checkpoint_root: Path,
    checkpoint_manifest_path: Path,
    reload_probe_path: Path,
    base_model_root: Path,
    base_model_manifest_path: Path,
    tokenizer_manifest_path: Path,
    renderer_manifest_path: Path,
    runtime_manifest_path: Path,
    evidence_root: Path,
    timeout_seconds: int,
    manifest: StageDCheckpointManifest,
    manifest_bytes: bytes,
    probe: StageDReloadProbe,
    probe_bytes: bytes,
    fault_hook: ReloadFaultHook | None,
) -> tuple[ReloadWorkerResult, bytes]:
    prefix = evidence_root / f"reload-{ordinal}"
    nonce_path = prefix.with_suffix(".nonce.json")
    intent_path = prefix.with_suffix(".intent.json")
    claim_path = prefix.with_suffix(".claim.json")
    gate_path = prefix.with_suffix(".gate.json")
    completion_path = prefix.with_suffix(".completion.json")
    python_executable = Path(sys.executable).absolute()
    worker_source = Path(__file__).resolve()
    working_directory = Path.cwd().resolve()
    worker_environment = _reload_worker_environment()
    static_intent = {
        "schema_version": 1,
        "domain": "redco-stage-d-reload-intent-v1",
        "arm": arm,
        "ordinal": ordinal,
        "backend": backend,
        "checkpoint_manifest_sha256": _sha256(manifest_bytes),
        "reload_probe_sha256": _sha256(probe_bytes),
        "base_model_manifest_sha256": manifest.base_model_manifest_sha256,
        "tokenizer_manifest_sha256": probe.tokenizer_manifest_sha256,
        "renderer_manifest_sha256": probe.renderer_manifest_sha256,
        "runtime_manifest_sha256": probe.runtime_manifest_sha256,
        "python_executable_sha256": _file_sha256(python_executable),
        "worker_source_sha256": _file_sha256(worker_source),
        "worker_environment_sha256": _environment_sha256(worker_environment),
        "working_directory_sha256": _sha256(str(working_directory).encode("utf-8")),
    }
    if nonce_path.exists():
        nonce_payload = _canonical_object(nonce_path.read_bytes(), "reload nonce")
    else:
        proposal = canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-reload-nonce-v1",
                "launch_nonce": secrets.token_hex(32),
            }
        )
        with suppress(FileExistsError):
            _exclusive_write(nonce_path, proposal)
        nonce_payload = _canonical_object(nonce_path.read_bytes(), "reload nonce")
    if (
        set(nonce_payload) != {"schema_version", "domain", "launch_nonce"}
        or nonce_payload.get("schema_version") != 1
        or nonce_payload.get("domain") != "redco-stage-d-reload-nonce-v1"
    ):
        raise RuntimeError("reload nonce receipt differs")
    nonce = _require_sha256(nonce_payload.get("launch_nonce"), "reload launch nonce")
    intent = (
        _canonical_object(intent_path.read_bytes(), "reload intent")
        if intent_path.exists()
        else None
    )
    command = [
        str(python_executable),
        "-m",
        "redco.analysis.stage_d_reload_supervisor",
        "--worker",
        "--arm",
        arm,
        "--launch-nonce",
        nonce,
        "--ordinal",
        str(ordinal),
        "--backend",
        backend,
        "--checkpoint-root",
        str(checkpoint_root),
        "--checkpoint-manifest",
        str(checkpoint_manifest_path),
        "--reload-probe",
        str(reload_probe_path),
        "--base-model-root",
        str(base_model_root),
        "--base-model-manifest",
        str(base_model_manifest_path),
        "--tokenizer-manifest",
        str(tokenizer_manifest_path),
        "--renderer-manifest",
        str(renderer_manifest_path),
        "--runtime-manifest",
        str(runtime_manifest_path),
        "--claim",
        str(claim_path),
        "--gate",
        str(gate_path),
        "--completion",
        str(completion_path),
        "--python-executable-sha256",
        static_intent["python_executable_sha256"],
        "--worker-source-sha256",
        static_intent["worker_source_sha256"],
        "--worker-command-sha256",
        "PENDING",
        "--worker-environment-sha256",
        static_intent["worker_environment_sha256"],
        "--working-directory-sha256",
        static_intent["working_directory_sha256"],
    ]
    command_sha256 = _command_sha256(tuple(command))
    command[command.index("PENDING")] = command_sha256
    expected_intent = {
        **static_intent,
        "launch_nonce": nonce,
        "worker_command_sha256": command_sha256,
    }
    if intent is not None:
        if intent != expected_intent:
            raise RuntimeError("reload intent differs from the frozen worker inputs")
    else:
        _write_idempotent(intent_path, canonical_json(expected_intent))
        if fault_hook is not None:
            fault_hook("after-reload-intent", intent_path)
    gate = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-reload-start-gate-v1",
            "arm": arm,
            "ordinal": ordinal,
            "launch_nonce": nonce,
        }
    )
    if completion_path.exists():
        if not gate_path.is_file() or not claim_path.is_file():
            raise RuntimeError("reload completion lacks its durable authorization chain")
        started = _canonical_object(claim_path.read_bytes(), "reload process claim")
        expected_process = (
            started.get("pid"),
            started.get("boot_id"),
            started.get("process_start_ticks"),
        )
        if (
            started.get("arm") != arm
            or started.get("ordinal") != ordinal
            or started.get("launch_nonce") != nonce
            or started.get("domain") != "redco-stage-d-reload-process-claim-v1"
            or gate_path.read_bytes() != gate
        ):
            raise RuntimeError("completed reload start receipt differs")
        return _validate_worker_completion(
            completion_path.read_bytes(),
            arm=arm,
            nonce=nonce,
            manifest=manifest,
            manifest_bytes=manifest_bytes,
            probe=probe,
            probe_bytes=probe_bytes,
            expected_process=expected_process,
            expected_provenance=expected_intent,
        )
    process: subprocess.Popen[bytes] | None = None
    if not claim_path.exists():
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=os.name != "nt",
            cwd=working_directory,
            env=worker_environment,
        )
        if fault_hook is not None:
            fault_hook("after-reload-worker-spawn", intent_path)
        try:
            _wait_for_claim(
                claim_path,
                arm=arm,
                ordinal=ordinal,
                nonce=nonce,
                spawned=process,
                timeout_seconds=10,
            )
        except BaseException:
            _kill_process_tree(process)
            raise
    started = _canonical_object(claim_path.read_bytes(), "reload process claim")
    expected_started = {
        "schema_version": 1,
        "domain": "redco-stage-d-reload-process-claim-v1",
        "arm": arm,
        "ordinal": ordinal,
        "launch_nonce": nonce,
        "pid": started.get("pid"),
        "boot_id": started.get("boot_id"),
        "process_start_ticks": started.get("process_start_ticks"),
    }
    if started != expected_started:
        raise RuntimeError("reload process start receipt differs")
    pid = started.get("pid")
    if type(pid) is not int or pid < 1:
        raise RuntimeError("reload process start PID is invalid")
    process_identity = (started.get("boot_id"), started.get("process_start_ticks"))
    owns_worker = process is not None and pid == process.pid
    if process is not None and not owns_worker:
        stdout, stderr = process.communicate(timeout=10)
        if process.returncode != 0 or stdout:
            raise RuntimeError(
                "competing reload worker did not exit cleanly: "
                f"{stderr.decode('utf-8', errors='replace')[-4000:]}"
            )
        process = None
    if process is None and not gate_path.exists() and not _same_live_process(pid, process_identity):
        raise RuntimeError("reload worker died before authorization; no probe was observed")
    _write_idempotent(gate_path, gate)
    if fault_hook is not None:
        fault_hook("after-reload-start-gate", gate_path)
    if process is None:
        if not _same_live_process(pid, process_identity):
            raise RuntimeError("reload worker died after authorization without complete evidence")
        _wait_for_file(completion_path, process=None, timeout_seconds=timeout_seconds)
    else:
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except BaseException:
            _kill_process_tree(process)
            raise
        if process.returncode != 0:
            raise RuntimeError(
                f"reload worker {ordinal} failed: "
                f"{stderr.decode('utf-8', errors='replace')[-4000:]}"
            )
        if stdout:
            raise RuntimeError("reload worker wrote unexpected stdout")
    if fault_hook is not None:
        fault_hook("after-reload-worker-completion", completion_path)
    return _validate_worker_completion(
        completion_path.read_bytes(),
        arm=arm,
        nonce=nonce,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        probe=probe,
        probe_bytes=probe_bytes,
        expected_process=(pid, *process_identity),
        expected_provenance=expected_intent,
    )


def _validate_worker_completion(
    completion_bytes: bytes,
    *,
    arm: ArmName,
    nonce: str,
    manifest: StageDCheckpointManifest,
    manifest_bytes: bytes,
    probe: StageDReloadProbe,
    probe_bytes: bytes,
    expected_process: tuple[object, object, object] | None,
    expected_provenance: dict[str, object],
) -> tuple[ReloadWorkerResult, bytes]:
    completion = ReloadWorkerCompletion.from_bytes(completion_bytes)
    result = completion.result
    output = completion.output_bytes
    if (
        result.arm != arm
        or result.launch_nonce != nonce
        or (
            expected_process is not None
            and (result.pid, result.boot_id, result.process_start_ticks) != expected_process
        )
        or result.checkpoint_manifest_sha256 != _sha256(manifest_bytes)
        or result.post_model_sha256 != manifest.post_model_sha256
        or result.loaded_model_sha256 != manifest.post_model_sha256
        or result.reload_probe_sha256 != _sha256(probe_bytes)
        or result.base_model_manifest_sha256 != manifest.base_model_manifest_sha256
        or result.tokenizer_manifest_sha256 != probe.tokenizer_manifest_sha256
        or result.renderer_manifest_sha256 != probe.renderer_manifest_sha256
        or result.runtime_manifest_sha256 != probe.runtime_manifest_sha256
        or result.python_executable_sha256 != expected_provenance["python_executable_sha256"]
        or result.worker_source_sha256 != expected_provenance["worker_source_sha256"]
        or result.worker_command_sha256 != expected_provenance["worker_command_sha256"]
        or result.worker_environment_sha256 != expected_provenance["worker_environment_sha256"]
        or result.working_directory_sha256 != expected_provenance["working_directory_sha256"]
        or result.output_sha256 != _sha256(output)
    ):
        raise RuntimeError("reload worker result differs from supervisor bindings")
    return result, output


def _wait_for_process_identity(pid: int) -> tuple[str, str]:
    deadline = time.monotonic() + 10.0
    while True:
        try:
            return linux_process_identity(pid)
        except (FileNotFoundError, ProcessLookupError):
            if time.monotonic() >= deadline:
                raise RuntimeError("reload worker exited before process identity capture") from None
            time.sleep(0.01)


def _same_live_process(pid: int, identity: tuple[object, object]) -> bool:
    try:
        return linux_process_identity(pid) == identity
    except (FileNotFoundError, ProcessLookupError):
        return False


def _command_sha256(argv: tuple[str, ...]) -> str:
    normalized = list(argv)
    try:
        index = normalized.index("--worker-command-sha256") + 1
    except (ValueError, IndexError) as error:
        raise ValueError("reload worker command lacks its digest slot") from error
    normalized[index] = "PENDING"
    return _sha256(
        canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-reload-worker-command-v1",
                "argv": normalized,
            }
        )
    )


def _reload_worker_environment() -> dict[str, str]:
    prefixes = ("CUDA_", "NVIDIA_", "HF_", "TRANSFORMERS_", "TORCH_")
    names = {
        "APPDATA",
        "HOME",
        "LD_LIBRARY_PATH",
        "LOCALAPPDATA",
        "PATH",
        "PYTHONPATH",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
        "XDG_CACHE_HOME",
    }
    result = {
        key: value for key, value in os.environ.items() if key in names or key.startswith(prefixes)
    }
    source_root = str(Path(__file__).resolve().parents[2])
    existing = result.get("PYTHONPATH")
    existing_parts = [] if not existing else existing.split(os.pathsep)
    result["PYTHONPATH"] = os.pathsep.join(
        (source_root, *(item for item in existing_parts if item != source_root))
    )
    result["PYTHONHASHSEED"] = "0"
    result["PYTHONNOUSERSITE"] = "1"
    return dict(sorted(result.items()))


def _environment_sha256(environment: dict[str, str]) -> str:
    return _sha256(
        canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-reload-worker-environment-v1",
                "environment": environment,
            }
        )
    )


def _wait_for_claim(
    claim_path: Path,
    *,
    arm: ArmName,
    ordinal: int,
    nonce: str,
    spawned: subprocess.Popen[bytes],
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not claim_path.is_file():
        if spawned.poll() is not None:
            stderr = b"" if spawned.stderr is None else spawned.stderr.read()
            raise RuntimeError(
                "reload worker exited before process claim: "
                f"{stderr.decode('utf-8', errors='replace')[-4000:]}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for reload process claim")
        time.sleep(0.01)
    claim = _canonical_object(claim_path.read_bytes(), "reload process claim")
    expected = {
        "schema_version": 1,
        "domain": "redco-stage-d-reload-process-claim-v1",
        "arm": arm,
        "ordinal": ordinal,
        "launch_nonce": nonce,
        "pid": claim.get("pid"),
        "boot_id": claim.get("boot_id"),
        "process_start_ticks": claim.get("process_start_ticks"),
    }
    if claim != expected:
        raise RuntimeError("reload process claim differs")
    pid = claim.get("pid")
    identity = (claim.get("boot_id"), claim.get("process_start_ticks"))
    if type(pid) is not int or not _same_live_process(pid, identity):
        raise RuntimeError("claimed reload worker died before authorization")


def _wait_for_file(
    path: Path,
    *,
    process: subprocess.Popen[bytes] | None,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if process is not None and process.poll() is not None:
            stderr = b"" if process.stderr is None else process.stderr.read()
            raise RuntimeError(
                "reload worker exited before durable evidence: "
                f"{stderr.decode('utf-8', errors='replace')[-4000:]}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for reload evidence: {path.name}")
        time.sleep(0.02)


def _canonical_object(value: bytes, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(payload, dict) or canonical_json(payload) != value:
        raise ValueError(f"{name} is not canonical JSON")
    return payload


def _write_idempotent(path: Path, value: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != value:
            raise FileExistsError(f"reload durable state differs: {path}")
        return
    _exclusive_write(path, value)


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        if process.poll() is None:
            process.kill()
    else:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _worker(arguments: argparse.Namespace) -> None:
    arm = cast(ArmName, arguments.arm)
    boot_id, start_ticks = linux_process_identity(os.getpid())
    process_identity = {
        "arm": arm,
        "ordinal": arguments.ordinal,
        "launch_nonce": arguments.launch_nonce,
        "pid": os.getpid(),
        "boot_id": boot_id,
        "process_start_ticks": start_ticks,
    }
    claim = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-reload-process-claim-v1",
            **process_identity,
        }
    )
    try:
        _exclusive_write(arguments.claim, claim)
    except FileExistsError:
        return
    while not arguments.gate.is_file():
        time.sleep(0.02)
    expected_gate = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-reload-start-gate-v1",
            "arm": arm,
            "ordinal": arguments.ordinal,
            "launch_nonce": arguments.launch_nonce,
        }
    )
    if arguments.gate.read_bytes() != expected_gate:
        raise ValueError("reload worker start gate differs from its launch")
    actual_command = (
        str(Path(sys.executable).absolute()),
        "-m",
        "redco.analysis.stage_d_reload_supervisor",
        *sys.argv[1:],
    )
    actual_provenance = {
        "python_executable_sha256": _file_sha256(Path(sys.executable).absolute()),
        "worker_source_sha256": _file_sha256(Path(__file__).resolve()),
        "worker_command_sha256": _command_sha256(actual_command),
        "worker_environment_sha256": _environment_sha256(_reload_worker_environment()),
        "working_directory_sha256": _sha256(str(Path.cwd().resolve()).encode("utf-8")),
    }
    expected_provenance = {name: getattr(arguments, name) for name in actual_provenance}
    if actual_provenance != expected_provenance:
        raise ValueError("reload worker execution provenance differs")
    manifest_bytes = arguments.checkpoint_manifest.read_bytes()
    manifest = StageDCheckpointManifest.from_bytes(manifest_bytes)
    if manifest.arm != arm:
        raise ValueError("reload worker arm differs from checkpoint")
    manifest.verify_directory(
        arguments.checkpoint_root,
        verify_semantic=arguments.backend != "test-fixture",
    )
    probe_bytes = arguments.reload_probe.read_bytes()
    probe = StageDReloadProbe.from_bytes(probe_bytes)
    files = {
        arguments.base_model_manifest: manifest.base_model_manifest_sha256,
        arguments.tokenizer_manifest: probe.tokenizer_manifest_sha256,
        arguments.renderer_manifest: probe.renderer_manifest_sha256,
        arguments.runtime_manifest: probe.runtime_manifest_sha256,
    }
    for path, expected in files.items():
        if _file_sha256(path) != expected:
            raise ValueError(f"reload worker identity file differs: {path.name}")
    if arguments.backend == "transformers-peft":
        _verify_base_model_snapshot(
            arguments.base_model_root,
            arguments.base_model_manifest,
        )
    if arguments.backend == "test-fixture":
        generated = tuple(
            (token + probe.max_new_tokens) % 32000
            for token in probe.prompt_token_ids[-probe.max_new_tokens :]
        )
        loaded_model_sha256 = manifest.post_model_sha256
    else:
        generated, loaded_model_sha256 = _transformers_peft_reload(
            arguments.base_model_root,
            arguments.checkpoint_root,
            probe,
            base_model_manifest_sha256=manifest.base_model_manifest_sha256,
        )
    if loaded_model_sha256 != manifest.post_model_sha256:
        raise ValueError("loaded PEFT adapter differs from the checkpoint manifest")
    output = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-reload-output-v1",
            "generated_token_ids": list(generated),
        }
    )
    result = ReloadWorkerResult(
        arm=arm,
        launch_nonce=arguments.launch_nonce,
        pid=os.getpid(),
        boot_id=boot_id,
        process_start_ticks=start_ticks,
        checkpoint_manifest_sha256=_sha256(manifest_bytes),
        post_model_sha256=manifest.post_model_sha256,
        loaded_model_sha256=loaded_model_sha256,
        reload_probe_sha256=_sha256(probe_bytes),
        base_model_manifest_sha256=manifest.base_model_manifest_sha256,
        tokenizer_manifest_sha256=probe.tokenizer_manifest_sha256,
        renderer_manifest_sha256=probe.renderer_manifest_sha256,
        runtime_manifest_sha256=probe.runtime_manifest_sha256,
        **actual_provenance,
        output_sha256=_sha256(output),
    )
    completion = ReloadWorkerCompletion(result, generated)
    _exclusive_write(arguments.completion, completion.to_bytes())


def _transformers_peft_reload(
    base_model_root: Path,
    checkpoint_root: Path,
    probe: StageDReloadProbe,
    *,
    base_model_manifest_sha256: str,
) -> tuple[tuple[int, ...], str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base_model_root,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, checkpoint_root, is_trainable=False)
    model.eval()
    from redco.analysis.stage_d_live_update import loaded_peft_adapter_state_sha256

    loaded_model_sha256 = loaded_peft_adapter_state_sha256(
        model,
        base_snapshot_manifest_sha256=base_model_manifest_sha256,
    )
    device = next(model.parameters()).device
    input_ids = torch.tensor([probe.prompt_token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=probe.max_new_tokens,
            use_cache=True,
        )
    return (
        tuple(int(item) for item in output[0, input_ids.shape[1] :].tolist()),
        loaded_model_sha256,
    )


def _verify_base_model_snapshot(root: Path, manifest_path: Path) -> None:
    try:
        payload = json.loads(manifest_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("base model manifest is not JSON") from error
    if (
        not isinstance(payload, dict)
        or canonical_json(payload) != manifest_path.read_bytes()
        or payload.get("domain") != "redco-stage-d-e2-base-snapshot-v1"
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("files"), list)
    ):
        raise ValueError("base model manifest is noncanonical or unsupported")
    expected_paths: set[str] = set()
    for item in payload["files"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "size", "sha256"}
            or not isinstance(item["path"], str)
            or Path(item["path"]).is_absolute()
            or "\\" in item["path"]
            or any(part in {"", ".", ".."} for part in Path(item["path"]).parts)
            or type(item["size"]) is not int
            or item["size"] < 0
        ):
            raise ValueError("base model manifest member is invalid")
        expected_paths.add(item["path"])
        path = root / item["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != item["size"]
            or _file_sha256(path) != _require_sha256(item["sha256"], "base member")
        ):
            raise ValueError(f"base model snapshot member differs: {item['path']}")
    if len(expected_paths) != len(payload["files"]):
        raise ValueError("base model manifest contains duplicate paths")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("base model snapshot contains a symbolic link")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    }
    if actual_paths != expected_paths:
        raise ValueError("base model snapshot roster differs from its manifest")


def _exclusive_write(
    path: Path,
    value: bytes,
    *,
    fault_hook: ReloadFaultHook | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != value:
            raise FileExistsError(f"reload durable state differs: {path}")
        return
    pending = path.with_name(f".{path.name}.pending.{os.getpid()}.{secrets.token_hex(8)}")
    descriptor = os.open(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        if fault_hook is not None:
            fault_hook("after-reload-pending-fsync", pending)
        try:
            os.link(pending, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != value:
                raise FileExistsError(f"reload durable state differs: {path}") from None
        if fault_hook is not None:
            fault_hook("after-reload-link", path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        pending.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=sorted(_ARMS), required=True)
    parser.add_argument("--launch-nonce", required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    parser.add_argument("--backend", choices=("transformers-peft", "test-fixture"), required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--reload-probe", type=Path, required=True)
    parser.add_argument("--base-model-root", type=Path, required=True)
    parser.add_argument("--base-model-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--renderer-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--claim", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--python-executable-sha256", required=True)
    parser.add_argument("--worker-source-sha256", required=True)
    parser.add_argument("--worker-command-sha256", required=True)
    parser.add_argument("--worker-environment-sha256", required=True)
    parser.add_argument("--working-directory-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.worker:
        raise ValueError("reload module CLI is worker-only")
    _worker(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ReloadWorkerResult",
    "StageDReloadProbe",
    "run_fresh_reload_pair",
]
