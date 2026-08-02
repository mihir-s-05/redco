from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from redco.analysis.stage_d_process_supervision import (
    TrainerProcessStartReceipt,
    linux_process_identity,
    linux_process_state,
)


@pytest.mark.skipif(os.name == "nt", reason="Linux /proc identity is deployment-specific")
def test_wrapper_receipt_precedes_exec_and_tracks_the_same_process(tmp_path: Path) -> None:
    receipt_path = tmp_path / "process.json"
    command_observation = tmp_path / "command-observation"
    wrapper = Path(__file__).parents[1] / "scripts" / "stage_d_trainer_exec_wrapper.py"
    process = subprocess.Popen(
        [
            sys.executable,
            str(wrapper),
            "--receipt",
            str(receipt_path),
            "--arm",
            "stock",
            "--launch-id",
            "stock-live",
            "--environment-manifest-sha256",
            "a" * 64,
            "--",
            "/bin/sh",
            "-c",
            f"test -s {receipt_path!s} && touch {command_observation!s}; exec sleep 30",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not command_observation.exists():
            time.sleep(0.01)
        assert command_observation.is_file()
        receipt = TrainerProcessStartReceipt.from_bytes(receipt_path.read_bytes())
        assert receipt.pid == process.pid
        assert receipt.is_same_live_process() is True
        stale = TrainerProcessStartReceipt(
            receipt.arm,
            receipt.launch_id,
            receipt.pid,
            receipt.boot_id,
            str(int(receipt.process_start_ticks) + 1),
            receipt.command_sha256,
            receipt.environment_manifest_sha256,
        )
        assert stale.is_same_live_process() is False
    finally:
        process.terminate()
        process.wait(timeout=10)
    assert receipt.is_same_live_process() is False


@pytest.mark.skipif(os.name == "nt", reason="Linux zombie state is deployment-specific")
def test_zombie_process_is_not_reported_as_live() -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.1)"])
    boot_id, start_ticks = linux_process_identity(process.pid)
    receipt = TrainerProcessStartReceipt(
        "stock",
        "stock-zombie",
        process.pid,
        boot_id,
        start_ticks,
        "a" * 64,
        "b" * 64,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and linux_process_state(process.pid) != "Z":
            time.sleep(0.01)
        assert linux_process_state(process.pid) == "Z"
        assert receipt.is_same_live_process() is False
    finally:
        process.wait(timeout=5)
