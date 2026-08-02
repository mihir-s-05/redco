"""Linux watchdog actuator for one identity-bound evaluation process group."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, NoReturn, cast

from redco.analysis.stage_d_evaluation_actuation import (
    ActuatedProcessReceipt,
    EvaluationSupervisorIdentity,
)
from redco.analysis.stage_d_evaluation_codec import atomic_publish
from redco.analysis.stage_d_process_supervision import (
    linux_process_cgroup,
    linux_process_group_identity,
    linux_process_identity,
)

_PR_SET_PDEATHSIG = 1
_PR_SET_CHILD_SUBREAPER = 36
_MAX_CONTROL_BYTES = 1024 * 1024


def _mask_signals(how_name: str, signals: Iterable[signal.Signals]) -> None:
    if os.name == "nt":
        raise RuntimeError("signal masking requires Linux")
    mask = cast(
        Callable[[int, Iterable[signal.Signals]], set[signal.Signals]],
        signal.__dict__["pthread_sigmask"],
    )
    mask(cast(int, signal.__dict__[how_name]), signals)


def _prctl(option: int, value: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, value, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def bind_parent_death(expected: EvaluationSupervisorIdentity) -> None:
    """Arm PDEATHSIG and close the parent-died-before-prctl race."""
    if os.name == "nt":
        raise RuntimeError("parent-death binding requires Linux")
    _prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    if os.getppid() != expected.pid or not expected.is_same_live_process():
        raise RuntimeError("actuator parent changed before parent-death binding")


@dataclass(frozen=True, slots=True)
class CgroupV2:
    path: Path

    @classmethod
    def create(cls, delegated_root: Path, name: str) -> CgroupV2:
        if (
            not delegated_root.is_absolute()
            or delegated_root.is_symlink()
            or not (delegated_root / "cgroup.controllers").is_file()
            or "/" in name
            or not name
        ):
            raise ValueError("evaluation cgroup root or name is invalid")
        path = delegated_root / name
        path.mkdir(mode=0o700)
        if any(
            not (path / control).is_file()
            for control in ("cgroup.procs", "cgroup.events", "cgroup.kill")
        ):
            path.rmdir()
            raise RuntimeError("evaluation cgroup-v2 controls are unavailable")
        return cls(path)

    def add(self, pid: int) -> None:
        (self.path / "cgroup.procs").write_text(f"{pid}\n", encoding="ascii")

    def verify_membership(self, pid: int) -> tuple[str, ...]:
        members = {
            int(value)
            for value in (self.path / "cgroup.procs").read_text(encoding="ascii").splitlines()
        }
        lines = linux_process_cgroup(pid)
        unified = tuple(line for line in lines if line.startswith("0::"))
        if (
            pid not in members
            or len(unified) != 1
            or not unified[0].removeprefix("0::").rstrip("/").endswith(f"/{self.path.name}")
        ):
            raise RuntimeError("evaluation target did not enter its exact cgroup")
        return lines

    def populated(self) -> bool:
        fields = dict(
            line.split(maxsplit=1)
            for line in (self.path / "cgroup.events").read_text(encoding="ascii").splitlines()
        )
        if fields.get("populated") not in {"0", "1"}:
            raise ValueError("evaluation cgroup populated state is malformed")
        return fields["populated"] == "1"

    def kill(self) -> None:
        control = self.path / "cgroup.kill"
        if not control.is_file():
            raise RuntimeError("evaluation cgroup.kill is unavailable")
        control.write_text("1\n", encoding="ascii")

    def wait_empty(self, deadline: float) -> None:
        while time.monotonic() < deadline:
            if not self.populated():
                return
            time.sleep(0.05)
        raise TimeoutError("evaluation cgroup did not become empty")

    def remove(self) -> None:
        if self.populated():
            raise RuntimeError("evaluation cgroup is still populated")
        self.path.rmdir()


def _read_supervisor_identity(path: Path) -> EvaluationSupervisorIdentity:
    value = path.read_bytes()
    if len(value) > _MAX_CONTROL_BYTES:
        raise ValueError("evaluation supervisor identity is oversized")
    try:
        payload = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation supervisor identity is not JSON") from error
    if not isinstance(payload, dict) or set(payload) != {
        "pid",
        "boot_id",
        "process_start_ticks",
    }:
        raise ValueError("evaluation supervisor identity fields differ")
    return EvaluationSupervisorIdentity(
        payload["pid"], payload["boot_id"], payload["process_start_ticks"]
    )


def _read_environment(path: Path) -> dict[str, str]:
    value = path.read_bytes()
    if len(value) > _MAX_CONTROL_BYTES:
        raise ValueError("evaluation target environment is oversized")
    try:
        payload = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation target environment is not JSON") from error
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in payload.items()
    ):
        raise ValueError("evaluation target environment is invalid")
    return cast(dict[str, str], payload)


def _target_preexec(actuator: EvaluationSupervisorIdentity) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    _mask_signals("SIG_UNBLOCK", {signal.SIGTERM, signal.SIGINT})
    _prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    if os.getppid() != actuator.pid or not actuator.is_same_live_process():
        os._exit(126)


def _drain_bounded(
    source: BinaryIO,
    destination: BinaryIO,
    limit_bytes: int,
    errors: list[str],
) -> None:
    written = 0
    writable = True
    try:
        while chunk := source.read(64 * 1024):
            if writable and written < limit_bytes:
                retained = chunk[: limit_bytes - written]
                try:
                    destination.write(retained)
                    written += len(retained)
                except OSError as error:
                    errors.append(f"{type(error).__name__}: {error}")
                    writable = False
    finally:
        source.close()


def _terminate_target(
    target: subprocess.Popen[bytes],
    cgroup: CgroupV2 | None,
    *,
    deadline: float,
) -> None:
    target_live = target.poll() is None
    cgroup_populated = cgroup is not None and cgroup.populated()
    if not target_live and not cgroup_populated:
        return
    with suppress(ProcessLookupError):
        os.kill(-target.pid, signal.SIGTERM)
    grace_deadline = min(deadline, time.monotonic() + 1.0)
    while time.monotonic() < grace_deadline:
        target_live = target.poll() is None
        cgroup_populated = cgroup is not None and cgroup.populated()
        if not target_live and not cgroup_populated:
            return
        time.sleep(0.02)
    if cgroup is not None and cgroup.populated():
        cgroup.kill()
        cgroup.wait_empty(deadline)
    elif target.poll() is None:
        with suppress(ProcessLookupError):
            os.kill(-target.pid, getattr(signal, "SIGKILL", 9))
    if target.poll() is None:
        remaining = max(0.0, deadline - time.monotonic())
        target.wait(timeout=remaining)


def _reap_descendants(deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            pid, _status = os.waitpid(-1, getattr(os, "WNOHANG", 1))
        except ChildProcessError:
            return
        if pid == 0:
            time.sleep(0.01)
            continue
    raise TimeoutError("evaluation actuator descendants were not reaped")


def preflight_cgroup_v2(
    delegated_root: Path,
    *,
    executable: str,
    timeout_seconds: float,
) -> None:
    """Exercise the exact create, move, observe, kill, empty, remove contract."""
    if os.name == "nt" or timeout_seconds <= 0:
        raise RuntimeError("cgroup-v2 preflight requires Linux and a positive timeout")
    cgroup = CgroupV2.create(
        delegated_root,
        f"redco-preflight-{os.getpid()}-{time.time_ns()}",
    )
    process: subprocess.Popen[bytes] | None = None
    deadline = time.monotonic() + timeout_seconds
    try:
        process = subprocess.Popen(
            [executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            process_group=0,
        )
        cgroup.add(process.pid)
        cgroup.verify_membership(process.pid)
    finally:
        if process is not None:
            _terminate_target(process, cgroup, deadline=deadline)
        cgroup.remove()


def run_actuator(arguments: argparse.Namespace) -> int:
    if os.name == "nt":
        raise RuntimeError("evaluation actuator requires Linux")
    supervisor = _read_supervisor_identity(arguments.supervisor_identity)
    bind_parent_death(supervisor)
    _prctl(_PR_SET_CHILD_SUBREAPER, 1)
    actuator_boot, actuator_start = linux_process_identity(os.getpid())
    actuator_identity = EvaluationSupervisorIdentity(os.getpid(), actuator_boot, actuator_start)
    actuator_group, actuator_session = linux_process_group_identity(os.getpid())
    if (actuator_group, actuator_session) != (os.getpid(), os.getpid()):
        raise RuntimeError("evaluation actuator is not its own session leader")
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    blocked = {signal.SIGTERM, signal.SIGINT}
    _mask_signals("SIG_BLOCK", blocked)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    gate_read, gate_write = os.pipe()
    target: subprocess.Popen[bytes] | None = None
    cgroup: CgroupV2 | None = None
    stdout_file: BinaryIO | None = None
    stderr_file: BinaryIO | None = None
    stdout_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    drain_errors: list[str] = []
    try:
        environment = _read_environment(arguments.target_environment)
        cgroup = CgroupV2.create(
            arguments.cgroup_root,
            f"redco-{arguments.launch_capability_sha256[:24]}-{os.getpid()}",
        )
        stdout_file = arguments.target_stdout.open("xb")
        stderr_file = arguments.target_stderr.open("xb")
        target_command = [
            *arguments.target_command,
            "--actuated-receipt",
            str(arguments.receipt),
            "--start-gate-fd",
            str(gate_read),
        ]
        target = subprocess.Popen(
            target_command,
            cwd=arguments.target_cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            pass_fds=(gate_read,),
            process_group=0,
            preexec_fn=lambda: _target_preexec(actuator_identity),
        )
        assert target.stdout is not None and target.stderr is not None
        stdout_thread = threading.Thread(
            target=_drain_bounded,
            args=(target.stdout, stdout_file, arguments.max_log_bytes, drain_errors),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_drain_bounded,
            args=(target.stderr, stderr_file, arguments.max_log_bytes, drain_errors),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        cgroup.add(target.pid)
        target_boot, target_start = linux_process_identity(target.pid)
        target_group, target_session = linux_process_group_identity(target.pid)
        cgroup_lines = cgroup.verify_membership(target.pid)
        cgroup_stat = cgroup.path.stat()
        receipt = ActuatedProcessReceipt(
            arm=arguments.arm,
            role=arguments.role,
            epoch=arguments.epoch,
            launch_capability_sha256=arguments.launch_capability_sha256,
            actuation_attempt_id=arguments.actuation_attempt_id,
            pid=target.pid,
            boot_id=target_boot,
            process_start_ticks=target_start,
            process_group_id=target_group,
            process_session_id=target_session,
            cgroup_path=str(cgroup.path),
            cgroup_device_id=cgroup_stat.st_dev,
            cgroup_inode=cgroup_stat.st_ino,
            cgroup_lines=cgroup_lines,
            supervisor=supervisor,
            actuator_pid=os.getpid(),
            actuator_boot_id=actuator_boot,
            actuator_start_ticks=actuator_start,
            command_sha256=arguments.program_command_sha256,
            environment_manifest_sha256=arguments.program_environment_sha256,
            stop_request_path=str(arguments.stop_request),
        )
        if not receipt.has_dedicated_topology():
            raise RuntimeError("evaluation actuator created the wrong process topology")
        atomic_publish(arguments.receipt, receipt.to_bytes())
        _mask_signals("SIG_UNBLOCK", blocked)
        if stop_requested.is_set() or not supervisor.is_same_live_process():
            return 125
        os.write(gate_write, b"1")
        os.close(gate_write)
        gate_write = -1
        while target.poll() is None:
            if (
                stop_requested.is_set()
                or arguments.stop_request.exists()
                or not supervisor.is_same_live_process()
            ):
                break
            time.sleep(arguments.poll_interval_seconds)
        return_code = target.poll()
        if return_code is None:
            return 125
        return return_code
    finally:
        _mask_signals("SIG_UNBLOCK", blocked)
        for descriptor in (gate_read, gate_write):
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)
        cleanup_deadline = time.monotonic() + arguments.stop_timeout_seconds
        if target is not None:
            _terminate_target(target, cgroup, deadline=cleanup_deadline)
            _reap_descendants(cleanup_deadline)
        for thread in (stdout_thread, stderr_thread):
            if thread is not None:
                thread.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
                if thread.is_alive():
                    raise TimeoutError("evaluation log drain did not terminate")
        for stream in (stdout_file, stderr_file):
            if stream is not None:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
        if cgroup is not None:
            cgroup.remove()
        if drain_errors:
            raise RuntimeError(f"evaluation log drain failed: {drain_errors[0]}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor-identity", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--stop-request", type=Path, required=True)
    parser.add_argument("--cgroup-root", type=Path, required=True)
    parser.add_argument("--target-environment", type=Path, required=True)
    parser.add_argument("--target-cwd", type=Path, required=True)
    parser.add_argument("--target-stdout", type=Path, required=True)
    parser.add_argument("--target-stderr", type=Path, required=True)
    parser.add_argument("--arm", choices=("stock", "branch-global", "local"), required=True)
    parser.add_argument("--role", choices=("client", "server"), required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--launch-capability-sha256", required=True)
    parser.add_argument("--actuation-attempt-id", required=True)
    parser.add_argument("--program-command-sha256", required=True)
    parser.add_argument("--program-environment-sha256", required=True)
    parser.add_argument("--poll-interval-seconds", type=float, required=True)
    parser.add_argument("--stop-timeout-seconds", type=float, required=True)
    parser.add_argument("--max-log-bytes", type=int, required=True)
    parser.add_argument("target_command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if not arguments.target_command or arguments.target_command[0] != "--":
        parser.error("target command must follow --")
    arguments.target_command = arguments.target_command[1:]
    paths = (
        arguments.supervisor_identity,
        arguments.receipt,
        arguments.stop_request,
        arguments.cgroup_root,
        arguments.target_environment,
        arguments.target_cwd,
        arguments.target_stdout,
        arguments.target_stderr,
    )
    if any(not path.is_absolute() for path in paths):
        parser.error("evaluation actuator paths must be absolute")
    if (
        arguments.poll_interval_seconds <= 0
        or arguments.stop_timeout_seconds <= 0
        or arguments.max_log_bytes < 1
    ):
        parser.error("evaluation actuator limits must be positive")
    if (
        len(arguments.actuation_attempt_id) != 32
        or any(character not in "0123456789abcdef" for character in arguments.actuation_attempt_id)
        or arguments.receipt.parent.name != arguments.actuation_attempt_id
        or arguments.supervisor_identity.parent != arguments.receipt.parent
        or arguments.stop_request.parent != arguments.receipt.parent
        or arguments.target_environment.parent != arguments.receipt.parent
        or arguments.receipt.name != "target-receipt.json"
        or arguments.stop_request.name != "stop-request.json"
    ):
        parser.error("evaluation actuator control paths differ from its attempt")
    if arguments.stop_request.exists():
        parser.error("evaluation actuator stop request already exists")
    return arguments


def main() -> NoReturn:
    raise SystemExit(run_actuator(_arguments()))


__all__ = [
    "CgroupV2",
    "bind_parent_death",
    "main",
    "preflight_cgroup_v2",
    "run_actuator",
]


if __name__ == "__main__":
    main()
