from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
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


def answer_type(annotation: dict[str, Any]) -> str:
    if annotation.get("yes_no") is not None:
        return "yes_no"
    if annotation.get("extractive_spans"):
        return "extractive"
    if annotation.get("free_form_answer"):
        return "abstractive"
    return "other"


def exact_reference(
    paper: str,
    answers: dict[str, Any],
    *,
    minimum_span_characters: int,
) -> tuple[tuple[str, ...], str] | None:
    candidates: list[tuple[tuple[str, ...], str]] = []
    for annotation in answers["answer"]:
        if annotation["unanswerable"]:
            continue
        evidence = tuple(
            dict.fromkeys(
                span.strip()
                for span in annotation["evidence"]
                if (
                    len(span.strip()) >= minimum_span_characters
                    and span.strip() in paper
                )
            )
        )
        if evidence:
            candidates.append((evidence, answer_type(annotation)))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (sum(map(len, item[0])), len(item[0]), item[1]),
    )


def select_paper_disjoint_examples(
    rows: Iterable[dict[str, Any]],
    *,
    split: str,
    limit: int,
    maximum_paper_characters: int,
    minimum_span_characters: int,
    forbidden_paper_ids: set[str],
    forbidden_reference_spans: set[str],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        if row["id"] in forbidden_paper_ids:
            continue
        paper = render_paper(row)
        if len(paper) > maximum_paper_characters:
            continue
        qas = row["qas"]
        chosen = None
        for index, question in enumerate(qas["question"]):
            reference = exact_reference(
                paper,
                qas["answers"][index],
                minimum_span_characters=minimum_span_characters,
            )
            if reference is None:
                continue
            evidence, kind = reference
            if set(evidence) & forbidden_reference_spans:
                continue
            question_id = qas["question_id"][index]
            chosen = {
                "example_id": f"qasper-{question_id}",
                "paper_id": row["id"],
                "title": row["title"],
                "question": question,
                "answer_type": kind,
                "split": split,
                "paper": paper,
                "reference_evidence": list(evidence),
            }
            break
        if chosen is not None:
            examples.append(chosen)
        if len(examples) == limit:
            return examples
    raise RuntimeError(
        f"only found {len(examples)} eligible {split} papers, needed {limit}"
    )


def distribution(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "minimum": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "p90": float(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]),
        "maximum": float(ordered[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-papers", type=int, default=32)
    parser.add_argument("--validation-papers", type=int, default=32)
    parser.add_argument("--maximum-paper-characters", type=int, default=60_000)
    parser.add_argument("--minimum-span-characters", type=int, default=20)
    args = parser.parse_args()

    selected: list[dict[str, Any]] = []
    forbidden_paper_ids: set[str] = set()
    forbidden_reference_spans: set[str] = set()
    for source_split, output_split, limit in (
        ("train", "train", args.train_papers),
        ("validation", "validation", args.validation_papers),
    ):
        url = f"{PARQUET_BASE}/{source_split}/0000.parquet"
        rows = load_dataset(
            "parquet",
            data_files={source_split: url},
            split=source_split,
            streaming=True,
        )
        examples = select_paper_disjoint_examples(
            rows,
            split=output_split,
            limit=limit,
            maximum_paper_characters=args.maximum_paper_characters,
            minimum_span_characters=args.minimum_span_characters,
            forbidden_paper_ids=forbidden_paper_ids,
            forbidden_reference_spans=forbidden_reference_spans,
        )
        selected.extend(examples)
        forbidden_paper_ids.update(row["paper_id"] for row in examples)
        forbidden_reference_spans.update(
            span for row in examples for span in row["reference_evidence"]
        )

    serialized = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in selected
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(serialized)
    split_counts = Counter(row["split"] for row in selected)
    answer_types = {
        split: dict(
            Counter(
                row["answer_type"]
                for row in selected
                if row["split"] == split
            )
        )
        for split in split_counts
    }
    manifest = {
        "schema_version": 2,
        "dataset": "allenai/qasper",
        "source_revision": QASPER_SOURCE_REVISION,
        "converted_parquet_revision": QASPER_PARQUET_REVISION,
        "license": "cc-by-4.0",
        "task": "deterministic single-paper exact-evidence extraction",
        "selection": {
            "order": "source paper order, then first eligible question",
            "one_question_per_paper": True,
            "paper_disjoint_splits": True,
            "reference_span_disjoint_splits": True,
            "train_papers": args.train_papers,
            "validation_papers": args.validation_papers,
            "maximum_paper_characters": args.maximum_paper_characters,
            "minimum_span_characters": args.minimum_span_characters,
            "exclude_unanswerable_or_empty_evidence": True,
        },
        "audit": {
            "split_counts": dict(split_counts),
            "unique_papers": len({row["paper_id"] for row in selected}),
            "unique_questions": len({row["example_id"] for row in selected}),
            "answer_types": answer_types,
            "paper_characters": distribution(
                [len(row["paper"]) for row in selected]
            ),
            "reference_characters": distribution(
                [
                    sum(map(len, row["reference_evidence"]))
                    for row in selected
                ]
            ),
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
