from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import load_dataset

QASPER_SOURCE_REVISION = "fdc9d8214fbab5dd782958601db4d678e6934a54"
QASPER_PARQUET_REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"
PARQUET_BASE = (
    "https://huggingface.co/datasets/allenai/qasper/resolve/"
    f"{QASPER_PARQUET_REVISION}/qasper"
)


def render_paper(row: dict[str, Any]) -> str:
    parts = [
        f"### PAPER: {row['title']}",
        "<abstract>",
        row["abstract"],
        "</abstract>",
    ]
    sections = row["full_text"]
    for name, paragraphs in zip(
        sections["section_name"], sections["paragraphs"], strict=True
    ):
        parts.append(f"\n## {name}")
        parts.extend(paragraphs)
    return "\n".join(parts)


def exact_reference(
    paper: str, answers: dict[str, Any]
) -> tuple[str, ...] | None:
    candidates: list[tuple[str, ...]] = []
    for annotation in answers["answer"]:
        if annotation["unanswerable"]:
            continue
        evidence = tuple(
            span.strip()
            for span in annotation["evidence"]
            if span.strip() and span.strip() in paper
        )
        if evidence:
            candidates.append(evidence)
    if not candidates:
        return None
    return min(candidates, key=lambda spans: (sum(map(len, spans)), len(spans)))


def select_examples(
    rows: Iterable[dict[str, Any]],
    *,
    split: str,
    limit: int,
    maximum_paper_characters: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        paper = render_paper(row)
        if len(paper) > maximum_paper_characters:
            continue
        qas = row["qas"]
        for index, question in enumerate(qas["question"]):
            reference = exact_reference(paper, qas["answers"][index])
            if reference is None:
                continue
            question_id = qas["question_id"][index]
            examples.append(
                {
                    "example_id": f"qasper-{question_id}",
                    "paper_id": row["id"],
                    "title": row["title"],
                    "question": question,
                    "split": split,
                    "paper": paper,
                    "reference_evidence": list(reference),
                }
            )
            if len(examples) == limit:
                return examples
    raise RuntimeError(
        f"only found {len(examples)} eligible {split} examples, needed {limit}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-examples", type=int, default=32)
    parser.add_argument("--validation-examples", type=int, default=16)
    parser.add_argument("--maximum-paper-characters", type=int, default=60_000)
    args = parser.parse_args()

    selected: list[dict[str, Any]] = []
    for source_split, output_split, limit in (
        ("train", "train", args.train_examples),
        ("validation", "validation", args.validation_examples),
    ):
        url = f"{PARQUET_BASE}/{source_split}/0000.parquet"
        rows = load_dataset(
            "parquet",
            data_files={source_split: url},
            split=source_split,
            streaming=True,
        )
        selected.extend(
            select_examples(
                rows,
                split=output_split,
                limit=limit,
                maximum_paper_characters=args.maximum_paper_characters,
            )
        )

    serialized = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in selected
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(serialized)
    manifest = {
        "schema_version": 1,
        "dataset": "allenai/qasper",
        "source_revision": QASPER_SOURCE_REVISION,
        "converted_parquet_revision": QASPER_PARQUET_REVISION,
        "license": "cc-by-4.0",
        "selection": {
            "order": "source order, then question order",
            "train_examples": args.train_examples,
            "validation_examples": args.validation_examples,
            "maximum_paper_characters": args.maximum_paper_characters,
            "requirements": [
                "answer is not marked unanswerable",
                "at least one annotation evidence string occurs verbatim in rendered paper",
                "choose the eligible annotation with the shortest total evidence length",
            ],
        },
        "output": {
            "path": args.output.as_posix(),
            "bytes": len(serialized),
            "sha256": hashlib.sha256(serialized).hexdigest(),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
