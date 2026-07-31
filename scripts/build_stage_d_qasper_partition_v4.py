from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PARTITION_SIZES = {
    "fewshot_support": 8,
    "power_audit": 64,
    "science_train": 16,
    "science_eval": 32,
}


def partition(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if len(train) != 88 or len(validation) != 32:
        raise ValueError("expected exactly 88 train and 32 validation papers")
    assignments = (
        ("fewshot_support", train[:8]),
        ("power_audit", train[8:72]),
        ("science_train", train[72:88]),
        ("science_eval", validation),
    )
    result: list[dict[str, Any]] = []
    for split, source_rows in assignments:
        for row in source_rows:
            copied = dict(row)
            copied["source_split"] = row["split"]
            copied["split"] = split
            result.append(copied)
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.read_bytes()
    if _sha256(source) != args.input_sha256:
        raise ValueError("input hash mismatch")
    source_rows = [
        json.loads(line)
        for line in source.decode("utf-8").splitlines()
        if line.strip()
    ]
    rows = partition(source_rows)
    payload = b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    partitions = {
        split: {
            "examples": sum(row["split"] == split for row in rows),
            "paper_ids": sorted(
                row["paper_id"] for row in rows if row["split"] == split
            ),
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
        for split in PARTITION_SIZES
    }
    sets = {
        name: set(record["paper_ids"])
        for name, record in partitions.items()
    }
    checks = {
        "partition_sizes_exact": all(
            partitions[name]["examples"] == expected
            for name, expected in PARTITION_SIZES.items()
        ),
        "all_120_papers_unique": (
            len({row["paper_id"] for row in rows}) == 120
        ),
        "paper_ids_pairwise_disjoint": all(
            not sets[left] & sets[right]
            for index, left in enumerate(sets)
            for right in list(sets)[index + 1 :]
        ),
        "support_and_power_cover_all_answer_types": all(
            set(partitions[name]["answer_types"])
            == {"abstractive", "extractive", "yes_no"}
            for name in ("fewshot_support", "power_audit")
        ),
    }
    manifest = {
        "schema_version": 1,
        "source": {
            "path": args.input.as_posix(),
            "sha256": args.input_sha256,
        },
        "partition_rule": (
            "Preserve source order: train[0:8] fewshot support, "
            "train[8:72] unique-paper power audit, train[72:88] science "
            "train, and all 32 validation papers science eval."
        ),
        "replication": {
            "fewshot_support": {
                "papers": 8,
                "seeds_per_paper": 8,
                "rollouts": 64,
            },
            "power_audit": {
                "papers": 64,
                "seeds_per_paper": 1,
                "rollouts": 64,
                "inferential_unit": "paper",
            },
            "science_train": {"papers": 16},
            "science_eval": {"papers": 32},
        },
        "partitions": partitions,
        "checks": checks,
        "output": {
            "path": args.output.as_posix(),
            "bytes": len(payload),
            "sha256": _sha256(payload),
        },
        "passes": all(checks.values()),
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not manifest["passes"]:
        raise SystemExit("v4 QASPER partition audit failed")


if __name__ == "__main__":
    main()
