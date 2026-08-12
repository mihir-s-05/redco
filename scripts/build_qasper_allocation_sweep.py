"""Build a 24/96 QASPER cohort from the authenticated full source."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json
from redco.experiments.qasper_evidence import (
    EvidenceTask,
    PilotBudget,
    _candidate_from_row,
    _digest,
    assert_evaluation_extension,
    load_pilot_tasks,
)
from redco.integrity import sha256_bytes

SOURCE_COMMIT = "53a7c67c9cb6df39e44454f364aaf3c9ca352966"
SOURCE_PATH = (
    "datasets/stage-d/source-auth-v13/"
    "qasper-train-06806e4608976fc2fac0a090ac425d5b2b29caf4-0000.parquet"
)
SOURCE_SHA256 = "9af08092ee26c4f700202c1f90d1592b662926f23f3a308a10ff0a53345e37fe"
PARENT_PATH = Path("data/qasper-evidence-matrix-v1.json")
PARENT_SHA256 = "11f7d6b4edadd0d8d423a22f0ff0e2ed53d2dfe224c01c41e0ce04678042a7e5"
OUTPUT_PATH = Path("data/qasper-allocation-sweep-v1.json")
SWEEP_BUDGET = PilotBudget(eval_tasks=96)


def _git(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def _render_paper(row: Mapping[str, Any]) -> str:
    parts = [f"### PAPER: {row['title']}", "<abstract>", row["abstract"], "</abstract>"]
    full_text = row["full_text"]
    for section, paragraphs in zip(
        full_text["section_name"], full_text["paragraphs"], strict=True
    ):
        parts.append(f"\n## {section}")
        parts.extend(paragraphs)
    return "\n".join(parts)


def _source_rows(rows: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for paper_row in rows:
        paper = _render_paper(paper_row)
        if len(paper) > 60_000:
            continue
        qas = paper_row["qas"]
        for index, question in enumerate(qas["question"]):
            evidence = list(
                dict.fromkeys(
                    span.strip()
                    for annotation in qas["answers"][index]["answer"]
                    if not annotation["unanswerable"]
                    for span in annotation["evidence"]
                    if len(span.strip()) >= 20 and span.strip() in paper
                )
            )
            if not evidence:
                continue
            yield {
                "example_id": f"qasper-{qas['question_id'][index]}",
                "paper_id": paper_row["id"],
                "question": question,
                "paper": paper,
                "reference_evidence": evidence,
            }


def _extension_tasks(source: bytes, excluded_papers: set[str]) -> list[EvidenceTask]:
    import pyarrow.parquet as parquet

    table = parquet.read_table(io.BytesIO(source))
    selected_by_paper: dict[str, EvidenceTask] = {}
    for row in _source_rows(table.to_pylist()):
        paper_id = row["paper_id"]
        if paper_id in excluded_papers or paper_id in selected_by_paper:
            continue
        candidate = _candidate_from_row(row)
        if candidate is not None:
            selected_by_paper[paper_id] = candidate
    return sorted(
        selected_by_paper.values(),
        key=lambda task: _digest("allocation-sweep-v1", task.task_id),
    )


def build_bytes() -> bytes:
    if sha256_bytes(PARENT_PATH.read_bytes()) != PARENT_SHA256:
        raise ValueError("parent matrix dataset has the wrong digest")
    parent = load_pilot_tasks(PARENT_PATH)
    source = _git("show", f"{SOURCE_COMMIT}:{SOURCE_PATH}")
    if sha256_bytes(source) != SOURCE_SHA256:
        raise ValueError("full QASPER source has the wrong digest")
    excluded = {task.source_paper_id for task in parent}
    extension = _extension_tasks(source, excluded)
    needed = SWEEP_BUDGET.eval_tasks - 24
    if len(extension) < needed:
        raise ValueError(f"full source yields only {len(extension)} extension tasks")
    tasks = list(parent)
    tasks.extend(
        EvidenceTask(
            task_id=task.task_id,
            source_example_id=task.source_example_id,
            source_paper_id=task.source_paper_id,
            split="eval",
            question=task.question,
            paragraphs=task.paragraphs,
            gold_paragraph_index=task.gold_paragraph_index,
            reference_evidence=task.reference_evidence,
        )
        for task in extension[:needed]
    )
    ordered = tuple(sorted(tasks, key=lambda task: (task.split, task.task_id)))
    assert_evaluation_extension(
        parent,
        ordered,
        parent_eval_tasks=24,
        expanded_eval_tasks=96,
    )
    payload = {
        "budget": asdict(SWEEP_BUDGET),
        "source": {
            "bytes": len(source),
            "commit": SOURCE_COMMIT,
            "git_blob": _git("rev-parse", f"{SOURCE_COMMIT}:{SOURCE_PATH}")
            .decode()
            .strip(),
            "parent_matrix_sha256": PARENT_SHA256,
            "path": SOURCE_PATH,
            "raw_sha256": SOURCE_SHA256,
            "rows": 888,
        },
        "tasks": [json.loads(task.to_json()) for task in ordered],
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
            raise SystemExit("allocation-sweep dataset is missing or stale")
        return
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(rendered)


if __name__ == "__main__":
    main()
