"""Identity-bound process actuation contracts for Stage-D held-out evaluation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from redco.analysis.stage_d_evaluation_codec import sha256
from redco.analysis.stage_d_evaluation_contracts import evaluation_environment_sha256
from redco.analysis.stage_d_objective_binding import ArmName
from redco.analysis.stage_d_process_supervision import (
    command_sha256,
    linux_process_cgroup,
    linux_process_group_identity,
    linux_process_identity,
    linux_process_state,
)
from redco.contracts import canonical_json

_DOMAIN = "redco-stage-d-actuated-process-receipt-v3"
_ARMS = {"stock", "branch-global", "local"}


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _attempt_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("actuation attempt ID must be 128-bit lowercase hex")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.isprintable():
        raise ValueError(f"{name} must be printable and nonempty")
    return value


def _is_absolute_path(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


@dataclass(frozen=True, slots=True)
class EvaluationSupervisorIdentity:
    pid: int
    boot_id: str
    process_start_ticks: str

    def __post_init__(self) -> None:
        _positive(self.pid, "evaluation supervisor PID")
        _text(self.boot_id, "evaluation supervisor boot ID")
        if not self.process_start_ticks.isdigit():
            raise ValueError("evaluation supervisor start ticks are invalid")

    @classmethod
    def current(cls) -> EvaluationSupervisorIdentity:
        boot_id, start_ticks = linux_process_identity(os.getpid())
        return cls(os.getpid(), boot_id, start_ticks)

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

    def to_payload(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "boot_id": self.boot_id,
            "process_start_ticks": self.process_start_ticks,
        }


@dataclass(frozen=True, slots=True)
class ActuatedProcessReceipt:
    arm: ArmName
    role: Literal["client", "server"]
    epoch: int
    launch_capability_sha256: str
    actuation_attempt_id: str
    pid: int
    boot_id: str
    process_start_ticks: str
    process_group_id: int
    process_session_id: int
    cgroup_path: str
    cgroup_device_id: int
    cgroup_inode: int
    cgroup_lines: tuple[str, ...]
    supervisor: EvaluationSupervisorIdentity
    actuator_pid: int
    actuator_boot_id: str
    actuator_start_ticks: str
    command_sha256: str
    environment_manifest_sha256: str
    stop_request_path: str

    def __post_init__(self) -> None:
        if self.arm not in _ARMS or self.role not in {"client", "server"}:
            raise ValueError("actuated process arm or role is invalid")
        if type(self.epoch) is not int or self.epoch < 0:
            raise ValueError("actuated process epoch is invalid")
        _digest(self.launch_capability_sha256, "actuated launch capability")
        _attempt_id(self.actuation_attempt_id)
        for integer_value, name in (
            (self.pid, "actuated target PID"),
            (self.process_group_id, "actuated target process group"),
            (self.process_session_id, "actuated target session"),
            (self.actuator_pid, "evaluation actuator PID"),
        ):
            _positive(integer_value, name)
        for text_value, name in (
            (self.boot_id, "actuated target boot ID"),
            (self.actuator_boot_id, "evaluation actuator boot ID"),
            (self.cgroup_path, "actuated cgroup path"),
            (self.stop_request_path, "actuated stop request path"),
        ):
            _text(text_value, name)
        _nonnegative(self.cgroup_device_id, "actuated cgroup device")
        _positive(self.cgroup_inode, "actuated cgroup inode")
        if not self.process_start_ticks.isdigit() or not self.actuator_start_ticks.isdigit():
            raise ValueError("actuated process start ticks are invalid")
        if not _is_absolute_path(self.cgroup_path) or not _is_absolute_path(self.stop_request_path):
            raise ValueError("actuated process control paths must be absolute")
        if not self.cgroup_lines or any(
            not line or not line.isprintable() for line in self.cgroup_lines
        ):
            raise ValueError("actuated cgroup observation is invalid")
        _digest(self.command_sha256, "actuated command")
        _digest(self.environment_manifest_sha256, "actuated environment")

    @property
    def receipt_sha256(self) -> str:
        return sha256(self.to_bytes())

    @property
    def launch_id(self) -> str:
        return (
            f"stage-d-evaluation-{self.role}-{self.arm}-"
            f"{self.epoch}-{self.launch_capability_sha256[:16]}"
        )

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 3,
                "domain": _DOMAIN,
                "arm": self.arm,
                "role": self.role,
                "epoch": self.epoch,
                "launch_capability_sha256": self.launch_capability_sha256,
                "actuation_attempt_id": self.actuation_attempt_id,
                "pid": self.pid,
                "boot_id": self.boot_id,
                "process_start_ticks": self.process_start_ticks,
                "process_group_id": self.process_group_id,
                "process_session_id": self.process_session_id,
                "cgroup_path": self.cgroup_path,
                "cgroup_device_id": self.cgroup_device_id,
                "cgroup_inode": self.cgroup_inode,
                "cgroup_lines": list(self.cgroup_lines),
                "supervisor": self.supervisor.to_payload(),
                "actuator_pid": self.actuator_pid,
                "actuator_boot_id": self.actuator_boot_id,
                "actuator_start_ticks": self.actuator_start_ticks,
                "command_sha256": self.command_sha256,
                "environment_manifest_sha256": self.environment_manifest_sha256,
                "stop_request_path": self.stop_request_path,
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> ActuatedProcessReceipt:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("actuated process receipt is not JSON") from error
        fields = {
            "schema_version",
            "domain",
            "arm",
            "role",
            "epoch",
            "launch_capability_sha256",
            "actuation_attempt_id",
            "pid",
            "boot_id",
            "process_start_ticks",
            "process_group_id",
            "process_session_id",
            "cgroup_path",
            "cgroup_device_id",
            "cgroup_inode",
            "cgroup_lines",
            "supervisor",
            "actuator_pid",
            "actuator_boot_id",
            "actuator_start_ticks",
            "command_sha256",
            "environment_manifest_sha256",
            "stop_request_path",
        }
        supervisor = payload.get("supervisor") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or payload.get("schema_version") != 3
            or payload.get("domain") != _DOMAIN
            or not isinstance(payload.get("cgroup_lines"), list)
            or not isinstance(supervisor, dict)
            or set(supervisor) != {"pid", "boot_id", "process_start_ticks"}
            or canonical_json(payload) != value
        ):
            raise ValueError("actuated process receipt fields differ")
        return cls(
            payload["arm"],
            payload["role"],
            payload["epoch"],
            payload["launch_capability_sha256"],
            payload["actuation_attempt_id"],
            payload["pid"],
            payload["boot_id"],
            payload["process_start_ticks"],
            payload["process_group_id"],
            payload["process_session_id"],
            payload["cgroup_path"],
            payload["cgroup_device_id"],
            payload["cgroup_inode"],
            tuple(payload["cgroup_lines"]),
            EvaluationSupervisorIdentity(
                supervisor["pid"],
                supervisor["boot_id"],
                supervisor["process_start_ticks"],
            ),
            payload["actuator_pid"],
            payload["actuator_boot_id"],
            payload["actuator_start_ticks"],
            payload["command_sha256"],
            payload["environment_manifest_sha256"],
            payload["stop_request_path"],
        )

    def is_same_live_process(self) -> bool:
        try:
            boot_id, start_ticks = linux_process_identity(self.pid)
            state = linux_process_state(self.pid)
            group_id, session_id = linux_process_group_identity(self.pid)
            cgroup = linux_process_cgroup(self.pid)
        except (FileNotFoundError, ProcessLookupError):
            return False
        return state != "Z" and (
            boot_id,
            start_ticks,
            group_id,
            session_id,
            cgroup,
        ) == (
            self.boot_id,
            self.process_start_ticks,
            self.process_group_id,
            self.process_session_id,
            self.cgroup_lines,
        )

    def actuator_is_same_live_process(self) -> bool:
        try:
            boot_id, start_ticks = linux_process_identity(self.actuator_pid)
            state = linux_process_state(self.actuator_pid)
            group_id, session_id = linux_process_group_identity(self.actuator_pid)
        except (FileNotFoundError, ProcessLookupError):
            return False
        return state != "Z" and (
            boot_id,
            start_ticks,
            group_id,
            session_id,
        ) == (
            self.actuator_boot_id,
            self.actuator_start_ticks,
            self.actuator_pid,
            self.actuator_pid,
        )

    def has_dedicated_topology(self) -> bool:
        return (
            self.process_group_id == self.pid
            and self.process_session_id == self.actuator_pid
            and self.actuator_pid != self.pid
        )

    def is_same_live_tree(self) -> bool:
        return (
            self.supervisor.is_same_live_process()
            and self.actuator_is_same_live_process()
            and self.is_same_live_process()
        )

    def verify_program(
        self,
        *,
        arm: ArmName,
        role: Literal["client", "server"],
        epoch: int,
        launch_capability_sha256: str,
        argv: tuple[str, ...],
        environment: tuple[tuple[str, str], ...],
        cgroup_root: str,
        control_root: str,
        evaluation_ledger_id: str,
        require_current: bool,
    ) -> None:
        cgroup_path = PurePosixPath(self.cgroup_path)
        expected_cgroup_name = f"redco-{launch_capability_sha256[:24]}-{self.actuator_pid}"
        expected_stop_path = (
            PurePosixPath(control_root)
            / evaluation_ledger_id
            / self.actuation_attempt_id
            / "stop-request.json"
        )
        if (
            (self.arm, self.role, self.epoch, self.launch_capability_sha256)
            != (arm, role, epoch, launch_capability_sha256)
            or self.command_sha256 != command_sha256(argv)
            or self.environment_manifest_sha256 != evaluation_environment_sha256(environment)
            or cgroup_path.parent != PurePosixPath(cgroup_root)
            or cgroup_path.name != expected_cgroup_name
            or PurePosixPath(self.stop_request_path) != expected_stop_path
            or not self.has_dedicated_topology()
            or (require_current and self.pid != os.getpid())
            or not self.is_same_live_tree()
        ):
            raise ValueError("actuated process receipt differs from its frozen launch")


__all__ = [
    "ActuatedProcessReceipt",
    "EvaluationSupervisorIdentity",
]
