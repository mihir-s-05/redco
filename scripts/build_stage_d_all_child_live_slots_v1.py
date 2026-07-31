"""Freeze and materialize the exact Stage D all-child live slot order."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "environments" / "redco_evidence_selection_v2"))

from redco_evidence_selection_v2.seeding import (  # noqa: E402
    derive_episode_seed,
)

from redco.contracts import canonical_json  # noqa: E402
from redco.integrations.signed_subprocess import (  # noqa: E402
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)

FIXTURE_MASTER_SEED = "redco-stage-d0-all-child-fixture-v1"
FIXTURE_REPLAY_MASTER_SEED = "redco-stage-d0-all-child-fixture-replay-v1"
SUPPORT_MASTER_SEED = "redco-stage-d0-all-child-support-v1"
SUPPORT_REPLAY_MASTER_SEED = "redco-stage-d0-all-child-support-replay-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _slot(
    row: dict[str, Any],
    *,
    kind: str,
    index: int,
    master_seed: str,
    replay_master_seed: str,
) -> dict[str, Any]:
    example_id = str(row["example_id"])
    return {
        "kind": kind,
        "slot_index": index,
        "slot_id": f"{example_id}::replicate-0",
        "example_id": example_id,
        "paper_id": str(row["paper_id"]),
        "split": str(row["split"]),
        "replicate": 0,
        "episode_seed": derive_episode_seed(master_seed, example_id, 0),
        "episode_master_seed": master_seed,
        "replay_master_seed": replay_master_seed,
        "row_sha256": hashlib.sha256(canonical_json(row) + b"\n").hexdigest(),
    }


def build(
    *,
    fixture_dataset: Path,
    fixture_manifest: Path,
    support_dataset: Path,
    support_manifest: Path,
) -> dict[str, Any]:
    fixture_rows = _rows(fixture_dataset)
    support_rows = [row for row in _rows(support_dataset) if row["split"] == "successor_support"]
    fixture_audit = json.loads(fixture_manifest.read_text(encoding="utf-8"))
    support_audit = json.loads(support_manifest.read_text(encoding="utf-8"))
    fixture_ids = [row["fixture_id"] for row in fixture_audit["rows"][:2]]
    selected_fixtures = [row for row in fixture_rows if row["example_id"] in fixture_ids]
    expected_support_ids = support_audit["partitions"]["successor_support"]["paper_ids"]
    if [row["paper_id"] for row in support_rows] != expected_support_ids:
        raise ValueError("support dataset order differs from frozen manifest")
    if [row["example_id"] for row in selected_fixtures] != fixture_ids:
        raise ValueError("fixture order differs from frozen manifest")
    fixture_orders = [
        row["shard_order"] for row in fixture_audit["rows"] if row["fixture_id"] in fixture_ids
    ]
    if fixture_orders != [[0, 1], [1, 0]]:
        raise ValueError("two fixture sentinels must cover both shard orders")
    slots = [
        *[
            _slot(
                row,
                kind="fixture",
                index=index,
                master_seed=FIXTURE_MASTER_SEED,
                replay_master_seed=FIXTURE_REPLAY_MASTER_SEED,
            )
            for index, row in enumerate(selected_fixtures)
        ],
        *[
            _slot(
                row,
                kind="support",
                index=index,
                master_seed=SUPPORT_MASTER_SEED,
                replay_master_seed=SUPPORT_REPLAY_MASTER_SEED,
            )
            for index, row in enumerate(support_rows)
        ],
    ]
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-all-child-live-slots-v1",
            "datasets": {
                "fixture": {
                    "path": fixture_dataset.as_posix(),
                    "sha256": _sha256(fixture_dataset),
                    "manifest": fixture_manifest.as_posix(),
                    "manifest_sha256": _sha256(fixture_manifest),
                },
                "support": {
                    "path": support_dataset.as_posix(),
                    "sha256": _sha256(support_dataset),
                    "manifest": support_manifest.as_posix(),
                    "manifest_sha256": _sha256(support_manifest),
                },
            },
            "seed_derivation": "derive_episode_seed(master, example_id, 0)",
            "fixture_slots": 2,
            "support_slots": 64,
            "fixture_shard_orders": fixture_orders,
            "slots": slots,
        }
    )


def materialize(
    *,
    manifest_path: Path,
    fixture_dataset: Path,
    support_dataset: Path,
    output_dir: Path,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_signed_payload(manifest)
    datasets = {
        "fixture": fixture_dataset,
        "support": support_dataset,
    }
    rows_by_kind = {
        kind: {str(row["example_id"]): row for row in _rows(path)}
        for kind, path in datasets.items()
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    handles: dict[str, list[str]] = {"fixture": [], "support": []}
    for slot in manifest["slots"]:
        kind = str(slot["kind"])
        row = rows_by_kind[kind][str(slot["example_id"])]
        payload = canonical_json(row) + b"\n"
        if hashlib.sha256(payload).hexdigest() != slot["row_sha256"]:
            raise ValueError("slot row hash mismatch")
        kind_dir = output_dir / kind
        kind_dir.mkdir(exist_ok=True)
        path = kind_dir / f"{int(slot['slot_index']):03d}.jsonl"
        path.write_bytes(payload)
        handles[kind].append(
            "\t".join(
                [
                    f"{int(slot['slot_index']):03d}",
                    str(slot["example_id"]),
                    str(slot["paper_id"]),
                    str(slot["episode_seed"]),
                    str(slot["episode_master_seed"]),
                    str(slot["replay_master_seed"]),
                    path.as_posix(),
                    slot["row_sha256"],
                ]
            )
        )
    for kind, lines in handles.items():
        (output_dir / f"{kind}.tsv").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--fixture-dataset", type=Path, required=True)
    freeze.add_argument("--fixture-manifest", type=Path, required=True)
    freeze.add_argument("--support-dataset", type=Path, required=True)
    freeze.add_argument("--support-manifest", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    emit = subparsers.add_parser("materialize")
    emit.add_argument("--manifest", type=Path, required=True)
    emit.add_argument("--fixture-dataset", type=Path, required=True)
    emit.add_argument("--support-dataset", type=Path, required=True)
    emit.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        atomic_write_json(
            args.output,
            build(
                fixture_dataset=args.fixture_dataset,
                fixture_manifest=args.fixture_manifest,
                support_dataset=args.support_dataset,
                support_manifest=args.support_manifest,
            ),
        )
    else:
        materialize(
            manifest_path=args.manifest,
            fixture_dataset=args.fixture_dataset,
            support_dataset=args.support_dataset,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
