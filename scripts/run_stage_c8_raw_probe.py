"""Run the Stage-C8 grouped-MM probe in isolated subprocesses."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_probes(output_dir: Path, shared_cache: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_cache.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = (
        ("eager", "eager"),
        ("compiled-first", "compiled"),
        ("compiled-second", "compiled"),
    )
    for name, mode in cases:
        result_path = output_dir / f"{name}.json"
        environment = dict(os.environ)
        environment["CUDA_LAUNCH_BLOCKING"] = "1"
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(shared_cache)
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/stage_c8_grouped_mm_probe.py",
                "--mode",
                mode,
                "--output",
                str(result_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        (output_dir / f"{name}.stdout.log").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (output_dir / f"{name}.stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        result = None
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": name,
                "mode": mode,
                "exit_code": completed.returncode,
                "passed": completed.returncode == 0
                and result is not None
                and result.get("status") == "passed",
                "result": result,
            }
        )

    passed = {str(row["name"]): bool(row["passed"]) for row in rows}
    if all(passed.values()):
        interpretation = "old_failure_unresolved_or_transient"
    elif (
        passed["eager"]
        and passed["compiled-first"]
        and not passed["compiled-second"]
    ):
        interpretation = "shared_compiled_cache_failure_supported"
    else:
        interpretation = "private_grouped_kernel_failure_observed"
    summary = {
        "status": "completed",
        "cases": rows,
        "interpretation": interpretation,
        "fallback_trainer_may_proceed": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shared-cache", type=Path, required=True)
    args = parser.parse_args()
    summary = run_probes(args.output_dir, args.shared_cache)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
