"""Run one Stage-C3 arm and enforce its frozen first-batch invariants."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from redco.analysis.stage_c3_invariants import (
    check_first_training_row,
    first_training_row,
)


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)  # type: ignore[attr-defined]
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)  # type: ignore[attr-defined]
        process.wait(timeout=10)


def supervise(
    launcher: Path,
    config: Path,
    output_dir: Path,
    result_path: Path,
    *,
    mode: str,
    invariant_timeout_seconds: int,
) -> int:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    process = subprocess.Popen(
        ["bash", str(launcher), str(config), str(output_dir)],
        start_new_session=True,
    )
    metrics = output_dir / "run_default" / "metrics.jsonl"
    deadline = time.monotonic() + invariant_timeout_seconds
    invariant: dict[str, object] | None = None
    try:
        while time.monotonic() < deadline:
            row = first_training_row(metrics)
            if row is not None:
                invariant = check_first_training_row(row, mode=mode)  # type: ignore[arg-type]
                break
            if process.poll() is not None:
                break
            time.sleep(2)
        if invariant is None:
            invariant = {
                "schema_version": 1,
                "mode": mode,
                "checks": {"first_training_row_before_timeout": False},
                "passed": False,
            }
        if not invariant["passed"]:
            _terminate_group(process)
            exit_code = 42
        else:
            exit_code = process.wait()
        result = {
            **invariant,
            "launcher_exit_code": process.returncode,
            "supervisor_exit_code": exit_code,
            "config": config.as_posix(),
            "output_dir": output_dir.as_posix(),
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return exit_code
    finally:
        _terminate_group(process)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "arm"), required=True)
    parser.add_argument("--invariant-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    raise SystemExit(
        supervise(
            args.launcher,
            args.config,
            args.output_dir,
            args.result,
            mode=args.mode,
            invariant_timeout_seconds=args.invariant_timeout_seconds,
        )
    )


if __name__ == "__main__":
    main()
