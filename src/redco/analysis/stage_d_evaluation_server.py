"""Linux serving-process and read-only checkpoint observation contract."""

from __future__ import annotations

import base64
import http.client
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from redco.analysis.stage_d_checkpoint_evidence import StageDCheckpointManifest
from redco.analysis.stage_d_checkpoint_materialization import (
    verify_materialized_checkpoint,
)
from redco.analysis.stage_d_evaluation_actuation import ActuatedProcessReceipt
from redco.analysis.stage_d_evaluation_capabilities import EvaluationServerLaunch
from redco.analysis.stage_d_evaluation_contracts import (
    EvaluationProgramBinding,
    evaluation_environment_sha256,
    hash_file,
)
from redco.analysis.stage_d_process_supervision import command_sha256, linux_process_identity
from redco.contracts import canonical_json
from redco.integrity import require_sha256_hex as _require_sha256

_MAX_PROC_BYTES = 4 * 1024 * 1024
_MAX_PROBE_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class EvaluationServerProcessObservation:
    arm: str
    server_epoch: int
    launch_record_sha256: str
    process_receipt_sha256: str
    program_binding_sha256: str
    pid: int
    boot_id: str
    process_start_ticks: str
    executable_path: str
    executable_sha256: str
    argv: tuple[str, ...]
    working_directory: str
    environment_manifest_sha256: str
    cgroup_lines: tuple[str, ...]
    checkpoint_root: str
    checkpoint_manifest_sha256: str
    endpoint: str
    cache_namespace: str

    def __post_init__(self) -> None:
        if self.arm not in {"stock", "branch-global", "local"}:
            raise ValueError("server observation arm is invalid")
        if type(self.server_epoch) is not int or self.server_epoch < 0:
            raise ValueError("server observation epoch is invalid")
        for name in (
            "launch_record_sha256",
            "process_receipt_sha256",
            "program_binding_sha256",
            "executable_sha256",
            "environment_manifest_sha256",
            "checkpoint_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.pid) is not int or self.pid < 1:
            raise ValueError("server observation PID is invalid")
        if not self.boot_id or not self.process_start_ticks.isdigit():
            raise ValueError("server observation process identity is invalid")
        for name in (
            "executable_path",
            "working_directory",
            "checkpoint_root",
            "endpoint",
            "cache_namespace",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or not value.isprintable():
                raise ValueError(f"server observation {name} is invalid")
        if not self.argv or any(not item or not item.isprintable() for item in self.argv):
            raise ValueError("server observation argv is invalid")
        if any(not item.isprintable() for item in self.cgroup_lines):
            raise ValueError("server observation cgroup is invalid")

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-evaluation-server-process-v1",
                "arm": self.arm,
                "server_epoch": self.server_epoch,
                "launch_record_sha256": self.launch_record_sha256,
                "process_receipt_sha256": self.process_receipt_sha256,
                "program_binding_sha256": self.program_binding_sha256,
                "pid": self.pid,
                "boot_id": self.boot_id,
                "process_start_ticks": self.process_start_ticks,
                "executable_path": self.executable_path,
                "executable_sha256": self.executable_sha256,
                "argv": list(self.argv),
                "working_directory": self.working_directory,
                "environment_manifest_sha256": self.environment_manifest_sha256,
                "cgroup_lines": list(self.cgroup_lines),
                "checkpoint_root": self.checkpoint_root,
                "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
                "endpoint": self.endpoint,
                "cache_namespace": self.cache_namespace,
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> EvaluationServerProcessObservation:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("server process observation is not JSON") from error
        fields = {
            "schema_version",
            "domain",
            "arm",
            "server_epoch",
            "launch_record_sha256",
            "process_receipt_sha256",
            "program_binding_sha256",
            "pid",
            "boot_id",
            "process_start_ticks",
            "executable_path",
            "executable_sha256",
            "argv",
            "working_directory",
            "environment_manifest_sha256",
            "cgroup_lines",
            "checkpoint_root",
            "checkpoint_manifest_sha256",
            "endpoint",
            "cache_namespace",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or payload.get("schema_version") != 1
            or payload.get("domain") != "redco-stage-d-evaluation-server-process-v1"
            or not isinstance(payload.get("argv"), list)
            or not isinstance(payload.get("cgroup_lines"), list)
            or canonical_json(payload) != value
        ):
            raise ValueError("server process observation fields differ")
        return cls(
            payload["arm"],
            payload["server_epoch"],
            payload["launch_record_sha256"],
            payload["process_receipt_sha256"],
            payload["program_binding_sha256"],
            payload["pid"],
            payload["boot_id"],
            payload["process_start_ticks"],
            payload["executable_path"],
            payload["executable_sha256"],
            tuple(payload["argv"]),
            payload["working_directory"],
            payload["environment_manifest_sha256"],
            tuple(payload["cgroup_lines"]),
            payload["checkpoint_root"],
            payload["checkpoint_manifest_sha256"],
            payload["endpoint"],
            payload["cache_namespace"],
        )

    def verify(
        self,
        *,
        launch: EvaluationServerLaunch,
        receipt: ActuatedProcessReceipt,
        program: EvaluationProgramBinding,
    ) -> None:
        if (
            (self.arm, self.server_epoch, self.launch_record_sha256)
            != (launch.arm, launch.epoch, launch.launch_record_sha256)
            or self.process_receipt_sha256 != receipt.receipt_sha256
            or self.program_binding_sha256 != program.binding_sha256
            or (self.pid, self.boot_id, self.process_start_ticks)
            != (receipt.pid, receipt.boot_id, receipt.process_start_ticks)
            or self.executable_sha256 != program.executable_sha256
            or self.argv != program.argv
            or command_sha256(self.argv) != receipt.command_sha256
            or self.working_directory != program.working_directory
            or self.checkpoint_root != program.checkpoint_root
            or self.environment_manifest_sha256
            != evaluation_environment_sha256(program.environment)
            or self.environment_manifest_sha256 != receipt.environment_manifest_sha256
            or self.checkpoint_manifest_sha256 != program.checkpoint_manifest_sha256
            or self.endpoint != program.endpoint
            or self.cache_namespace != program.cache_namespace
        ):
            raise ValueError("server process observation differs from frozen bindings")


def capture_linux_server_process(
    *,
    launch: EvaluationServerLaunch,
    receipt: ActuatedProcessReceipt,
    program: EvaluationProgramBinding,
    checkpoint_manifest_bytes: bytes,
) -> EvaluationServerProcessObservation:
    if os.name == "nt":
        raise RuntimeError("server process capture requires Linux /proc")
    if not receipt.is_same_live_process():
        raise ValueError("server process is not live at observation")
    checkpoint_root = Path(program.checkpoint_root)
    checkpoint_manifest = StageDCheckpointManifest.from_bytes(checkpoint_manifest_bytes)
    if (
        checkpoint_manifest.arm != program.arm
        or checkpoint_manifest.manifest_sha256 != program.checkpoint_manifest_sha256
    ):
        raise ValueError("server checkpoint differs from its program")
    verify_materialized_checkpoint(checkpoint_manifest, checkpoint_root)
    process_root = Path(f"/proc/{receipt.pid}")
    executable_path = str((process_root / "exe").resolve())
    argv = tuple(
        item.decode("utf-8")
        for item in _bounded_proc_read(process_root / "cmdline").split(b"\0")
        if item
    )
    environment_items = tuple(
        item.decode("utf-8")
        for item in _bounded_proc_read(process_root / "environ").split(b"\0")
        if item
    )
    if len({item.split("=", 1)[0] for item in environment_items}) != len(environment_items) or any(
        "=" not in item for item in environment_items
    ):
        raise ValueError("server process environment is malformed or duplicated")
    environment = tuple(
        sorted((name, value) for item in environment_items for name, value in (item.split("=", 1),))
    )
    cgroup_lines = tuple(_bounded_proc_read(process_root / "cgroup").decode("utf-8").splitlines())
    boot_id, start_ticks = linux_process_identity(receipt.pid)
    observation = EvaluationServerProcessObservation(
        arm=launch.arm,
        server_epoch=launch.epoch,
        launch_record_sha256=launch.launch_record_sha256,
        process_receipt_sha256=receipt.receipt_sha256,
        program_binding_sha256=program.binding_sha256,
        pid=receipt.pid,
        boot_id=boot_id,
        process_start_ticks=start_ticks,
        executable_path=executable_path,
        executable_sha256=hash_file(Path(executable_path)),
        argv=argv,
        working_directory=str((process_root / "cwd").resolve()),
        environment_manifest_sha256=evaluation_environment_sha256(environment),
        cgroup_lines=cgroup_lines,
        checkpoint_root=str(checkpoint_root.resolve()),
        checkpoint_manifest_sha256=checkpoint_manifest.manifest_sha256,
        endpoint=program.endpoint,
        cache_namespace=program.cache_namespace,
    )
    observation.verify(
        launch=launch,
        receipt=receipt,
        program=program,
    )
    return observation


def probe_local_evaluation_server(
    program: EvaluationProgramBinding,
    *,
    timeout_seconds: float,
) -> bytes:
    """Probe one bound loopback server without retrying or sampling tokens."""
    endpoint = urlsplit(program.endpoint)
    if endpoint.hostname != "127.0.0.1" or endpoint.port is None:
        raise ValueError("evaluation server probe endpoint is not loopback")
    responses: list[dict[str, object]] = []
    models_body: bytes | None = None
    for path in ("/health", "/v1/models"):
        connection = http.client.HTTPConnection(
            endpoint.hostname,
            endpoint.port,
            timeout=timeout_seconds,
        )
        try:
            connection.request("GET", path, headers={"host": f"127.0.0.1:{endpoint.port}"})
            response = connection.getresponse()
            body = response.read(_MAX_PROBE_BYTES + 1)
            status = response.status
        finally:
            connection.close()
        if status != 200:
            raise ValueError(f"evaluation server probe failed: {path}")
        if len(body) > _MAX_PROBE_BYTES:
            raise ValueError(f"evaluation server probe response is oversized: {path}")
        if path == "/v1/models":
            models_body = body
        responses.append(
            {
                "path": path,
                "status_code": status,
                "body_base64": base64.b64encode(body).decode("ascii"),
            }
        )
    if models_body is None:
        raise RuntimeError("evaluation server models probe was not executed")
    try:
        models = json.loads(models_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation server models probe is not JSON") from error
    data = models.get("data") if isinstance(models, dict) else None
    expected_model_id = _served_model_id(program)
    if not isinstance(data, list) or not any(
        isinstance(item, dict) and item.get("id") == expected_model_id for item in data
    ):
        raise ValueError("evaluation server probe exposes a different model identity")
    return canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-evaluation-server-probe-v1",
            "program_binding_sha256": program.binding_sha256,
            "expected_model_id": expected_model_id,
            "responses": responses,
        }
    )


def _served_model_id(program: EvaluationProgramBinding) -> str:
    indexes = [index for index, value in enumerate(program.argv) if value == "--served-model-name"]
    if len(indexes) > 1 or (indexes and indexes[0] + 1 >= len(program.argv)):
        raise ValueError("evaluation server served-model command is ambiguous")
    return program.argv[indexes[0] + 1] if indexes else program.checkpoint_root


def _bounded_proc_read(path: Path) -> bytes:
    value = path.read_bytes()
    if len(value) > _MAX_PROC_BYTES:
        raise ValueError(f"server process observation is too large: {path.name}")
    return value


__all__ = [
    "EvaluationServerProcessObservation",
    "capture_linux_server_process",
    "probe_local_evaluation_server",
]
