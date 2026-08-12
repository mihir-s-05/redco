"""Build the expanded QASPER evaluation matrix from the archived source."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from redco.contracts import canonical_json
from redco.experiments.qasper_evidence import (
    PilotBudget,
    assert_matrix_continuity,
    build_pilot_tasks,
    load_pilot_tasks,
)
from redco.integrity import sha256_bytes

SOURCE_COMMIT = "53a7c67c9cb6df39e44454f364aaf3c9ca352966"
SOURCE_PATH = "datasets/stage-d/qasper-support-v13-launch-input-v1.jsonl"
OUTPUT_PATH = Path("data/qasper-evidence-matrix-v1.json")
PILOT_PATH = Path("data/qasper-evidence-pilot-v1.json")
MATRIX_BUDGET = PilotBudget(eval_tasks=24)


def _git(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def build_bytes() -> bytes:
    source = _git("show", f"{SOURCE_COMMIT}:{SOURCE_PATH}")
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    tasks = build_pilot_tasks(
        rows,
        train_tasks=MATRIX_BUDGET.train_tasks,
        eval_tasks=MATRIX_BUDGET.eval_tasks,
    )
    assert_matrix_continuity(load_pilot_tasks(PILOT_PATH), tasks)
    payload = {
        "budget": asdict(MATRIX_BUDGET),
        "source": {
            "commit": SOURCE_COMMIT,
            "path": SOURCE_PATH,
            "git_blob": _git("rev-parse", f"{SOURCE_COMMIT}:{SOURCE_PATH}")
            .decode()
            .strip(),
            "raw_sha256": sha256_bytes(source),
            "rows": len(rows),
        },
        "tasks": [json.loads(task.to_json()) for task in tasks],
    }
    envelope = {
        "payload": payload,
        "payload_sha256": sha256_bytes(canonical_json(payload)),
        "schema_version": 1,
    }
    return (json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    rendered = build_bytes()
    if arguments.check:
        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_bytes() != rendered:
            raise SystemExit("matrix dataset is missing or stale")
        return
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(rendered)


if __name__ == "__main__":
    main()
