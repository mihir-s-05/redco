"""Run one fixed Stage D Prime capacity-monitor heartbeat."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = (
    Path(os.environ["APPDATA"]) / "uv" / "tools" / "prime" / "Scripts" / "python.exe"
).resolve()
if sys.platform != "win32" or sys.version_info[:3] != (3, 13, 2):
    raise SystemExit(20)
if Path(sys.executable).resolve() != EXPECTED_PYTHON:
    raise SystemExit(20)
os.environ["PATH"] = str(Path.home() / ".local" / "bin") + os.pathsep + os.environ["PATH"]
source = str(ROOT / "src")
if source not in sys.path:
    sys.path.insert(0, source)

from redco.analysis.stage_d_v13_prime_capacity_monitor_v1 import (  # noqa: E402
    AUTHORIZATION_FALSE,
    CONTRACT_DOMAIN,
    MONITOR_ID,
    run_capacity_monitor_heartbeat_v1,
)


def _terminal_value() -> dict[str, object]:
    return {
        "schema_version": 1,
        "domain": CONTRACT_DOMAIN,
        "monitor_id": MONITOR_ID,
        "state": "runner_failure_terminal",
        "disposition": "runner_failure_terminal",
        "continue_monitoring": False,
        "observation_ordinal": None,
        "next_not_before_epoch": None,
        "ledger_sha256": None,
        "authorization": AUTHORIZATION_FALSE,
    }


def _emit(value: dict[str, object]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def main() -> int:
    if sys.argv[1:]:
        _emit(_terminal_value())
        return 20
    try:
        result = run_capacity_monitor_heartbeat_v1()
    except (
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ):
        _emit(_terminal_value())
        return 20
    _emit(result.value())
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
