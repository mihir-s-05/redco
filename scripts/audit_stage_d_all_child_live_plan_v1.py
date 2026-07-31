"""Audit Stage D live slot plans, summaries, and canonical support progress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_all_child_support import aggregate_paper_support
from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def _slots(path: Path, kind: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load(path)
    verify_signed_payload(manifest)
    return manifest, [row for row in manifest["slots"] if row["kind"] == kind]


def audit_dry_run(slots_path: Path, kind: str, dry_run_path: Path) -> dict[str, Any]:
    manifest, slots = _slots(slots_path, kind)
    observed = _load(dry_run_path)["episode_seed_plan"]
    expected = [
        {
            "task_position": index,
            "example_id": slot["example_id"],
            "replicate": 0,
            "seed": slot["episode_seed"],
        }
        for index, slot in enumerate(slots)
    ]
    projected = [
        {
            "task_position": int(row["task_position"]),
            "example_id": str(row["example_id"]),
            "replicate": int(row["replicate"]),
            "seed": int(row["seed"]),
        }
        for row in observed
    ]
    if projected != expected:
        raise ValueError(f"{kind} dry-run plan differs from frozen slots")
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-all-child-live-dry-run-audit",
            "kind": kind,
            "slot_manifest_signature": manifest["signed_payload_sha256"],
            "slots": len(slots),
            "exact": True,
        }
    )


def audit_summary(slots_path: Path, kind: str, index: int, summary_path: Path) -> dict[str, Any]:
    manifest, slots = _slots(slots_path, kind)
    slot = slots[index]
    summary = _load(summary_path)
    records = summary.get("records") or []
    if len(records) != 1:
        raise ValueError("one-slot live run must contain exactly one record")
    record = records[0]
    exact = (
        summary.get("master_seed") == slot["episode_master_seed"]
        and record.get("slot_id") == slot["slot_id"]
        and record.get("example_id") == slot["example_id"]
        and int(record.get("replicate", -1)) == 0
        and int(record.get("seed", -1)) == int(slot["episode_seed"])
    )
    if not exact:
        raise ValueError("live run summary differs from frozen slot")
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-all-child-live-summary-audit",
            "kind": kind,
            "slot_index": index,
            "paper_id": slot["paper_id"],
            "slot_manifest_signature": manifest["signed_payload_sha256"],
            "exact": True,
        }
    )


def audit_progress(slots_path: Path, records_dir: Path) -> dict[str, Any]:
    manifest, slots = _slots(slots_path, "support")
    paths = sorted(records_dir.glob("*.json")) if records_dir.exists() else []
    records = [_load(path) for path in paths]
    for record in records:
        verify_signed_payload(record)
    if len(records) > 64:
        raise ValueError("too many support records")
    expected_ids = [slot["paper_id"] for slot in slots[: len(records)]]
    actual_ids = [record.get("paper_id") for record in records]
    if actual_ids != expected_ids:
        raise ValueError("support record prefix differs from frozen slot order")
    successes = sum(record.get("paper_joint_pass") is True for record in records)
    failures = len(records) - successes
    untouched = 64 - len(records)
    terminal_failure = successes + untouched < 58
    complete = len(records) == 64
    passes = complete and successes >= 58
    if terminal_failure:
        decision = "terminal_fail"
    elif passes:
        decision = "pass"
    elif complete:
        decision = "terminal_fail"
    else:
        decision = "continue"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-d-all-child-live-support-progress",
        "slot_manifest_signature": manifest["signed_payload_sha256"],
        "observed_prefix": len(records),
        "successes": successes,
        "failures": failures,
        "untouched_remaining": untouched,
        "required_successes": 58,
        "decision": decision,
        "early_success_forbidden": True,
        "truncated_rate_or_wilson_forbidden": terminal_failure and not complete,
    }
    if complete:
        aggregate = aggregate_paper_support(records)
        payload["aggregate"] = aggregate
        payload["aggregate_signature"] = aggregate["signed_payload_sha256"]
    return sign_payload(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry = subparsers.add_parser("dry-run")
    dry.add_argument("--slots", type=Path, required=True)
    dry.add_argument("--kind", choices=("fixture", "support"), required=True)
    dry.add_argument("--input", type=Path, required=True)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--slots", type=Path, required=True)
    summary.add_argument("--kind", choices=("fixture", "support"), required=True)
    summary.add_argument("--index", type=int, required=True)
    summary.add_argument("--input", type=Path, required=True)
    progress = subparsers.add_parser("progress")
    progress.add_argument("--slots", type=Path, required=True)
    progress.add_argument("--records-dir", type=Path, required=True)
    for subparser in (dry, summary, progress):
        subparser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "dry-run":
        report = audit_dry_run(args.slots, args.kind, args.input)
    elif args.command == "summary":
        report = audit_summary(args.slots, args.kind, args.index, args.input)
    else:
        report = audit_progress(args.slots, args.records_dir)
    atomic_write_json(args.output, report)


if __name__ == "__main__":
    main()
