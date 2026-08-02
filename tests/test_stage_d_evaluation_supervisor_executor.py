from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from redco.analysis.stage_d_evaluation_actuation import (
    ActuatedProcessReceipt,
    EvaluationSupervisorIdentity,
)
from redco.analysis.stage_d_evaluation_actuator import preflight_cgroup_v2
from redco.analysis.stage_d_evaluation_supervisor_executor import reap_child_process
from redco.contracts import canonical_json


def _wait_for(path: Path, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {path}")


@pytest.mark.skipif(os.name == "nt", reason="waitpid requires Linux")
def test_reap_child_process_is_nonblocking() -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.monotonic() + 5.0
    reaped = reap_child_process(process.pid)
    while time.monotonic() < deadline and not reaped:
        time.sleep(0.01)
        reaped = reap_child_process(process.pid)
    assert reaped
    with pytest.raises(ChildProcessError):
        os.waitpid(process.pid, os.WNOHANG)


@pytest.mark.skipif(
    os.name == "nt" or getattr(os, "geteuid", lambda: -1)() != 0,
    reason="real actuator containment requires root cgroup-v2 delegation",
)
def test_actuator_gates_exec_and_cleans_target_descendant_and_cgroup(tmp_path: Path) -> None:
    controllers = Path("/sys/fs/cgroup/cgroup.controllers")
    if not controllers.is_file() or not os.access(controllers.parent, os.W_OK):
        pytest.skip("writable cgroup-v2 delegation is unavailable")
    preflight_cgroup_v2(
        controllers.parent,
        executable=sys.executable,
        timeout_seconds=5,
    )
    target = tmp_path / "target.py"
    sentinel = tmp_path / "sentinel.json"
    target.write_text(
        "import argparse, json, os, pathlib, subprocess, sys, time\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--actuated-receipt', required=True)\n"
        "parser.add_argument('--start-gate-fd', required=True, type=int)\n"
        "args = parser.parse_args()\n"
        "gate = os.read(args.start_gate_fd, 2)\n"
        "os.close(args.start_gate_fd)\n"
        "assert gate == b'1'\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(sentinel)!r}).write_text(json.dumps("
        "{'pid': os.getpid(), 'child_pid': child.pid}))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    supervisor = EvaluationSupervisorIdentity.current()
    attempt_root = tmp_path / ("d" * 32)
    attempt_root.mkdir()
    supervisor_path = attempt_root / "supervisor.json"
    environment_path = attempt_root / "environment.json"
    receipt_path = attempt_root / "target-receipt.json"
    stop_path = attempt_root / "stop-request.json"
    supervisor_path.write_bytes(canonical_json(supervisor.to_payload()))
    environment_path.write_bytes(canonical_json(dict(os.environ)))
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
        "/sys/fs/cgroup",
        "--target-environment",
        str(environment_path),
        "--target-cwd",
        str(tmp_path),
        "--target-stdout",
        str(tmp_path / "target.stdout"),
        "--target-stderr",
        str(tmp_path / "target.stderr"),
        "--arm",
        "stock",
        "--role",
        "server",
        "--epoch",
        "0",
        "--launch-capability-sha256",
        "a" * 64,
        "--actuation-attempt-id",
        "d" * 32,
        "--program-command-sha256",
        "b" * 64,
        "--program-environment-sha256",
        "c" * 64,
        "--poll-interval-seconds",
        "0.02",
        "--stop-timeout-seconds",
        "5",
        "--max-log-bytes",
        "1048576",
        "--",
        sys.executable,
        str(target),
    ]
    actuator = subprocess.Popen(
        command,
        cwd=tmp_path,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _wait_for(receipt_path)
        _wait_for(sentinel)
        receipt = ActuatedProcessReceipt.from_bytes(receipt_path.read_bytes())
        observed = json.loads(sentinel.read_text(encoding="utf-8"))
        assert receipt.pid == observed["pid"]
        assert receipt.supervisor == supervisor
        assert receipt.is_same_live_tree()
        cgroup_path = Path(receipt.cgroup_path)
        assert cgroup_path.is_dir()
        stop_path.write_bytes(b"stop")
        stdout, stderr = actuator.communicate(timeout=10)
        assert actuator.returncode == 125, (stdout, stderr)
        assert not cgroup_path.exists()
        for pid in (observed["pid"], observed["child_pid"]):
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        if actuator.poll() is None:
            actuator.kill()
            actuator.wait(timeout=5)
