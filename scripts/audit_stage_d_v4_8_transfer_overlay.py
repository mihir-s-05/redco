"""Audit a byte-preserving v4.8 transfer overlay against frozen v4.7 sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected = protocol["source_sha256"]
    results = {}
    member_names = []
    duplicate_names = []
    with tarfile.open(args.overlay, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        for member in members:
            name = member.name.removeprefix("./").replace("\\", "/")
            if name in member_names:
                duplicate_names.append(name)
                continue
            member_names.append(name)
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"cannot read overlay member: {name}")
            actual = hashlib.sha256(handle.read()).hexdigest()
            results[name] = {
                "expected": expected.get(name),
                "actual": actual,
                "passes": expected.get(name) == actual,
            }

    checks = {
        "exact_member_set": (
            set(member_names) == set(expected)
            and len(member_names) == len(expected)
        ),
        "no_duplicates": not duplicate_names,
        "all_member_hashes_exact": (
            bool(results)
            and all(item["passes"] for item in results.values())
        ),
        "local_worktree_still_exact": all(
            Path(name).is_file() and _sha256(Path(name)) == digest
            for name, digest in expected.items()
        ),
    }
    report = sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-v4-8-byte-preserving-transfer-overlay",
            "protocol": args.protocol.as_posix(),
            "protocol_sha256": _sha256(args.protocol),
            "overlay": args.overlay.as_posix(),
            "overlay_sha256": _sha256(args.overlay),
            "expected_member_count": len(expected),
            "actual_member_count": len(member_names),
            "duplicate_names": duplicate_names,
            "results": results,
            "checks": checks,
            "passes": all(checks.values()),
        }
    )
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
