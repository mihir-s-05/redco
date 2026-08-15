"""Build the small authenticated MuSiQue-Ans capability snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json
from redco.experiments.musique_capability import (
    MuSiQueTask,
    load_gate_config,
    select_official_tasks,
)


def _task_value(task: MuSiQueTask) -> dict[str, Any]:
    return {
        "answer": task.answer,
        "answer_aliases": list(task.answer_aliases),
        "candidates": [
            {"text": candidate.text, "title": candidate.title} for candidate in task.candidates
        ],
        "hop_count": task.hop_count,
        "question": task.question,
        "support_components": list(task.support_components),
        "support_path": list(task.support_path),
        "task_id": task.task_id,
    }


def _read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError(f"official row {line_number} is not an object")
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/musique-ans-capability-gate-v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/musique-ans-capability-v1.json"),
    )
    arguments = parser.parse_args()
    config = load_gate_config(arguments.manifest)
    manifest = json.loads(arguments.manifest.read_bytes())
    source = manifest["source"]
    source_bytes = arguments.source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != source["raw_dev_sha256"]:
        raise ValueError("official source hash does not match the reviewed manifest")
    tasks = select_official_tasks(_read_rows(arguments.source), config)
    payload = {
        "cohort": "short_document_linear_chain",
        "dataset": "MuSiQue-Ans",
        "schema_version": 1,
        "tasks": [_task_value(task) for task in tasks],
    }
    output_bytes = canonical_json(payload) + b"\n"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(output_bytes)
    print(
        json.dumps(
            {
                "source_sha256": source_sha256,
                "snapshot_sha256": hashlib.sha256(output_bytes).hexdigest(),
                "snapshot_bytes": len(output_bytes),
                "tasks_by_depth": {
                    str(depth): sum(task.hop_count == depth for task in tasks) for depth in (2, 4)
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
