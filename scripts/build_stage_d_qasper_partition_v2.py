from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PARTITION_SIZES = {
    "fewshot_support": 8,
    "power_audit": 8,
    "science_train": 16,
    "science_eval": 32,
}


def _load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def partition(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if len(train) != 32 or len(validation) != 32:
        raise ValueError("expected exactly 32 train and 32 validation papers")

    assigned: list[dict[str, Any]] = []
    offsets = (
        ("fewshot_support", train[:8]),
        ("power_audit", train[8:16]),
        ("science_train", train[16:]),
        ("science_eval", validation),
    )
    for partition_name, source_rows in offsets:
        for row in source_rows:
            copied = dict(row)
            copied["source_split"] = row["split"]
            copied["split"] = partition_name
            assigned.append(copied)
    return assigned


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
    source_sha256 = _sha256(source)
    if source_sha256 != args.input_sha256:
        raise ValueError(
            f"input hash mismatch: {source_sha256} != {args.input_sha256}"
        )
    rows = partition(_load(args.input))
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)

    by_partition = {
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
    paper_sets = {
        split: set(record["paper_ids"])
        for split, record in by_partition.items()
    }
    pairwise_disjoint = all(
        not paper_sets[left] & paper_sets[right]
        for index, left in enumerate(PARTITION_SIZES)
        for right in list(PARTITION_SIZES)[index + 1 :]
    )
    checks = {
        "partition_sizes_exact": all(
            by_partition[name]["examples"] == expected
            for name, expected in PARTITION_SIZES.items()
        ),
        "paper_ids_pairwise_disjoint": pairwise_disjoint,
        "all_papers_unique": len({row["paper_id"] for row in rows}) == 64,
        "support_blocks_cover_all_answer_types": all(
            set(by_partition[name]["answer_types"])
            == {"abstractive", "extractive", "yes_no"}
            for name in ("fewshot_support", "power_audit")
        ),
    }
    manifest = {
        "schema_version": 1,
        "source": {
            "path": args.input.as_posix(),
            "sha256": source_sha256,
        },
        "partition_rule": (
            "Preserve frozen source order: train[0:8] fewshot_support, "
            "train[8:16] power_audit, train[16:32] science_train, and all "
            "validation rows science_eval."
        ),
        "replication": {
            "fewshot_support": {
                "papers": 8,
                "episode_seeds_per_paper": 8,
                "rollouts": 64,
            },
            "power_audit": {
                "papers": 8,
                "episode_seeds_per_paper": 8,
                "rollouts": 64,
            },
            "science_train": {
                "papers": 16,
                "role": "scientific optimizer data only",
            },
            "science_eval": {
                "papers": 32,
                "role": "held-out scientific evaluation only",
            },
        },
        "partitions": by_partition,
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
        raise SystemExit("Stage D QASPER partition audit failed")


if __name__ == "__main__":
    main()
