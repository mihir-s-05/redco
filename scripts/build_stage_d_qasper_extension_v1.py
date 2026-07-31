"""Build a fresh QASPER extension disjoint from the immutable Stage D v4 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

QASPER_SOURCE_REVISION = "fdc9d8214fbab5dd782958601db4d678e6934a54"
QASPER_PARQUET_REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"
PARQUET_BASE = (
    "https://huggingface.co/datasets/allenai/qasper/resolve/"
    f"{QASPER_PARQUET_REVISION}/qasper"
)
EXTENSION_SPLITS = {
    "successor_support": 64,
    "successor_science_train": 16,
    "successor_science_eval": 32,
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _answer_type(annotation: dict[str, Any]) -> str:
    if annotation.get("yes_no") is not None:
        return "yes_no"
    if annotation.get("extractive_spans"):
        return "extractive"
    if annotation.get("free_form_answer"):
        return "abstractive"
    return "other"


def _exact_reference(
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
                if len(span.strip()) >= minimum_span_characters
                and span.strip() in paper
            )
        )
        if evidence:
            candidates.append((evidence, _answer_type(annotation)))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (sum(map(len, item[0])), len(item[0]), item[1]),
    )


def select_extension_examples(
    rows: Iterable[dict[str, Any]],
    *,
    output_split: str,
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
            reference = _exact_reference(
                paper,
                qas["answers"][index],
                minimum_span_characters=minimum_span_characters,
            )
            if reference is None:
                continue
            evidence, kind = reference
            if set(evidence) & forbidden_reference_spans:
                continue
            chosen = {
                "example_id": f"qasper-{qas['question_id'][index]}",
                "paper_id": row["id"],
                "title": row["title"],
                "question": question,
                "answer_type": kind,
                "split": output_split,
                "paper": paper,
                "reference_evidence": list(evidence),
            }
            break
        if chosen is not None:
            examples.append(chosen)
            forbidden_paper_ids.add(chosen["paper_id"])
            forbidden_reference_spans.update(chosen["reference_evidence"])
        if len(examples) == limit:
            return examples
    raise RuntimeError(
        f"only found {len(examples)} fresh {output_split} papers, needed {limit}"
    )


def materialize_extension(
    *,
    old_rows: list[dict[str, Any]],
    train_rows: Iterable[dict[str, Any]],
    validation_rows: Iterable[dict[str, Any]],
    maximum_paper_characters: int,
    minimum_span_characters: int,
) -> list[dict[str, Any]]:
    forbidden_paper_ids = {str(row["paper_id"]) for row in old_rows}
    forbidden_reference_spans = {
        str(span)
        for row in old_rows
        for span in row["reference_evidence"]
    }
    support = select_extension_examples(
        train_rows,
        output_split="successor_support",
        limit=64,
        maximum_paper_characters=maximum_paper_characters,
        minimum_span_characters=minimum_span_characters,
        forbidden_paper_ids=forbidden_paper_ids,
        forbidden_reference_spans=forbidden_reference_spans,
    )
    science_train = select_extension_examples(
        train_rows,
        output_split="successor_science_train",
        limit=16,
        maximum_paper_characters=maximum_paper_characters,
        minimum_span_characters=minimum_span_characters,
        forbidden_paper_ids=forbidden_paper_ids,
        forbidden_reference_spans=forbidden_reference_spans,
    )
    science_eval = select_extension_examples(
        validation_rows,
        output_split="successor_science_eval",
        limit=32,
        maximum_paper_characters=maximum_paper_characters,
        minimum_span_characters=minimum_span_characters,
        forbidden_paper_ids=forbidden_paper_ids,
        forbidden_reference_spans=forbidden_reference_spans,
    )
    return [*support, *science_train, *science_eval]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-snapshot", type=Path, required=True)
    parser.add_argument("--old-snapshot-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--maximum-paper-characters", type=int, default=60_000)
    parser.add_argument("--minimum-span-characters", type=int, default=20)
    args = parser.parse_args()

    old_bytes = args.old_snapshot.read_bytes()
    if _sha256(old_bytes) != args.old_snapshot_sha256:
        raise ValueError("immutable old snapshot hash mismatch")
    old_rows = [
        json.loads(line)
        for line in old_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(old_rows) != 120:
        raise ValueError("immutable v4 snapshot must contain 120 papers")

    from datasets import load_dataset

    train_rows = load_dataset(
        "parquet",
        data_files={
            "train": f"{PARQUET_BASE}/train/0000.parquet",
        },
        split="train",
        streaming=True,
    )
    validation_rows = load_dataset(
        "parquet",
        data_files={
            "validation": f"{PARQUET_BASE}/validation/0000.parquet",
        },
        split="validation",
        streaming=True,
    )
    rows = materialize_extension(
        old_rows=old_rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
        maximum_paper_characters=args.maximum_paper_characters,
        minimum_span_characters=args.minimum_span_characters,
    )
    payload = b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    old_ids = {str(row["paper_id"]) for row in old_rows}
    old_refs = {
        str(span)
        for row in old_rows
        for span in row["reference_evidence"]
    }
    new_ids = {str(row["paper_id"]) for row in rows}
    new_refs = {
        str(span) for row in rows for span in row["reference_evidence"]
    }
    split_counts = Counter(str(row["split"]) for row in rows)
    checks = {
        "immutable_old_snapshot_retained": _sha256(old_bytes)
        == args.old_snapshot_sha256,
        "extension_has_112_unique_papers": len(rows) == len(new_ids) == 112,
        "split_sizes_exact": split_counts == Counter(EXTENSION_SPLITS),
        "old_and_extension_papers_disjoint": not old_ids & new_ids,
        "old_and_extension_references_disjoint": not old_refs & new_refs,
        "extension_references_unique": len(new_refs)
        == sum(len(row["reference_evidence"]) for row in rows),
        "every_reference_exact": all(
            span in row["paper"]
            for row in rows
            for span in row["reference_evidence"]
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"QASPER extension audit failed: {checks}")
    manifest = {
        "schema_version": 1,
        "dataset": "allenai/qasper",
        "source_revision": QASPER_SOURCE_REVISION,
        "converted_parquet_revision": QASPER_PARQUET_REVISION,
        "license": "cc-by-4.0",
        "old_snapshot": {
            "path": args.old_snapshot.as_posix(),
            "sha256": args.old_snapshot_sha256,
            "papers": 120,
            "disposition": "immutable development/retired quarantine",
        },
        "selection": {
            "source_order": True,
            "seed_forbidden_ids_and_spans_from_all_old_120": True,
            "one_question_per_paper": True,
            "maximum_paper_characters": args.maximum_paper_characters,
            "minimum_span_characters": args.minimum_span_characters,
            "fresh_reference_positions_or_midpoint_coverage_inspected": False,
        },
        "partitions": {
            split: {
                "papers": expected,
                "paper_ids": [
                    row["paper_id"] for row in rows if row["split"] == split
                ],
                "answer_types": dict(
                    sorted(
                        Counter(
                            row["answer_type"]
                            for row in rows
                            if row["split"] == split
                        ).items()
                    )
                ),
            }
            for split, expected in EXTENSION_SPLITS.items()
        },
        "checks": checks,
        "output": {
            "path": args.output.as_posix(),
            "bytes": len(payload),
            "sha256": _sha256(payload),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )


if __name__ == "__main__":
    main()
