"""Orphan-safe trainer process identity and pre-exec receipt contract."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from redco.analysis.stage_d_objective_binding import ArmName
from redco.contracts import canonical_json

_DOMAIN = "redco-stage-d-trainer-process-start-v1"
_ARMS = {"stock", "branch-global", "local"}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def command_sha256(argv: tuple[str, ...]) -> str:
    if not argv or any(not item or not item.isprintable() for item in argv):
        raise ValueError("trainer command must contain printable nonempty arguments")
    return _sha256(
        canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-trainer-command-v1",
                "argv": list(argv),
            }
        )
    )


def linux_process_identity(pid: int) -> tuple[str, str]:
    if type(pid) is not int or pid < 1:
        raise ValueError("trainer process ID must be positive")
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    closing = stat.rfind(")")
    fields = stat[closing + 2 :].split()
    if closing < 1 or len(fields) <= 19 or not fields[19].isdigit() or not boot_id:
        raise ValueError("Linux trainer process identity is malformed")
    return boot_id, fields[19]


def linux_process_state(pid: int) -> str:
    """Return the single-letter Linux state without treating zombies as live work."""
    if type(pid) is not int or pid < 1:
        raise ValueError("trainer process ID must be positive")
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    closing = stat.rfind(")")
    fields = stat[closing + 2 :].split()
    if closing < 1 or not fields or len(fields[0]) != 1:
        raise ValueError("Linux trainer process state is malformed")
    return fields[0]


def linux_process_cgroup(pid: int) -> tuple[str, ...]:
    if type(pid) is not int or pid < 1:
        raise ValueError("trainer process ID must be positive")
    lines = tuple(Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines())
    if not lines or any(not line or not line.isprintable() for line in lines):
        raise ValueError("Linux trainer process cgroup is malformed")
    return lines


def linux_process_group_identity(pid: int) -> tuple[int, int]:
    if os.name == "nt":
        raise RuntimeError("process groups require Linux")
    getpgid = cast(Callable[[int], int], os.getpgid)  # type: ignore[attr-defined]
    getsid = cast(Callable[[int], int], os.getsid)  # type: ignore[attr-defined]
    return getpgid(pid), getsid(pid)


@dataclass(frozen=True, slots=True)
class TrainerProcessStartReceipt:
    arm: ArmName
    launch_id: str
    pid: int
    boot_id: str
    process_start_ticks: str
    command_sha256: str
    environment_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise ValueError("trainer process receipt arm is invalid")
        if not self.launch_id or not self.launch_id.isprintable():
            raise ValueError("trainer process receipt launch ID is invalid")
        if type(self.pid) is not int or self.pid < 1:
            raise ValueError("trainer process receipt PID is invalid")
        if not self.boot_id or not self.boot_id.isprintable():
            raise ValueError("trainer process receipt boot ID is invalid")
        if not self.process_start_ticks.isdigit():
            raise ValueError("trainer process receipt start time is invalid")
        _require_sha256(self.command_sha256, "trainer command sha256")
        _require_sha256(
            self.environment_manifest_sha256,
            "trainer environment manifest sha256",
        )

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_bytes())

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": _DOMAIN,
                "arm": self.arm,
                "launch_id": self.launch_id,
                "pid": self.pid,
                "boot_id": self.boot_id,
                "process_start_ticks": self.process_start_ticks,
                "command_sha256": self.command_sha256,
                "environment_manifest_sha256": self.environment_manifest_sha256,
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> TrainerProcessStartReceipt:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("trainer process receipt is not JSON") from error
        expected = {
            "schema_version",
            "domain",
            "arm",
            "launch_id",
            "pid",
            "boot_id",
            "process_start_ticks",
            "command_sha256",
            "environment_manifest_sha256",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload.get("schema_version") != 1
            or payload.get("domain") != _DOMAIN
            or canonical_json(payload) != value
        ):
            raise ValueError("trainer process receipt is noncanonical or has different fields")
        arm = payload["arm"]
        if arm not in _ARMS:
            raise ValueError("trainer process receipt arm is invalid")
        return cls(
            cast(ArmName, arm),
            payload["launch_id"],
            payload["pid"],
            payload["boot_id"],
            payload["process_start_ticks"],
            payload["command_sha256"],
            payload["environment_manifest_sha256"],
        )

    def is_same_live_process(self) -> bool:
        try:
            boot_id, start_ticks = linux_process_identity(self.pid)
            state = linux_process_state(self.pid)
        except (FileNotFoundError, ProcessLookupError):
            return False
        return state != "Z" and (boot_id, start_ticks) == (
            self.boot_id,
            self.process_start_ticks,
        )


def write_preexec_receipt(
    path: Path,
    *,
    arm: ArmName,
    launch_id: str,
    argv: tuple[str, ...],
    environment_manifest_sha256: str,
) -> TrainerProcessStartReceipt:
    boot_id, start_ticks = linux_process_identity(os.getpid())
    receipt = TrainerProcessStartReceipt(
        arm,
        launch_id,
        os.getpid(),
        boot_id,
        start_ticks,
        command_sha256(argv),
        _require_sha256(
            environment_manifest_sha256,
            "trainer environment manifest sha256",
        ),
    )
    if path.exists():
        raise FileExistsError("trainer process receipt already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    descriptor = os.open(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(receipt.to_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)
    _fsync_directory(path.parent)
    return receipt


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "TrainerProcessStartReceipt",
    "command_sha256",
    "linux_process_cgroup",
    "linux_process_group_identity",
    "linux_process_identity",
    "linux_process_state",
    "write_preexec_receipt",
]
