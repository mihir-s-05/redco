"""Record the WSL serialization regressions for the Stage D v4.9 repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_signed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed_payload(payload)
    return payload


def report(
    migration_equivalence: Path,
    archive_expected: Path,
    archive_actual: Path,
) -> dict[str, Any]:
    migration = _load_signed(migration_equivalence)
    expected = _load_signed(archive_expected)
    actual = _load_signed(archive_actual)
    archive_equal = archive_expected.read_bytes() == archive_actual.read_bytes()
    checks = {
        "executed_on_linux": platform.system() == "Linux",
        "migration_strict_signed_objects_equal": migration["passes"],
        "migration_files_differ_only_in_serialization": (
            migration["expected_file_sha256"]
            != migration["actual_file_sha256"]
            and migration["expected_signed_payload_sha256"]
            == migration["actual_signed_payload_sha256"]
        ),
        "archive_manifest_byte_identical": archive_equal,
        "archive_manifest_signed_objects_equal": expected == actual,
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-v4-9-linux-serialization-regression",
            "platform": platform.platform(),
            "migration_equivalence": migration_equivalence.as_posix(),
            "migration_equivalence_sha256": _sha256(migration_equivalence),
            "migration_equivalence_signature": migration[
                "signed_payload_sha256"
            ],
            "archive_expected": archive_expected.as_posix(),
            "archive_expected_sha256": _sha256(archive_expected),
            "archive_expected_signature": expected["signed_payload_sha256"],
            "archive_actual": archive_actual.as_posix(),
            "archive_actual_sha256": _sha256(archive_actual),
            "archive_actual_signature": actual["signed_payload_sha256"],
            "checks": checks,
            "passes": all(checks.values()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migration-equivalence", type=Path, required=True)
    parser.add_argument("--archive-expected", type=Path, required=True)
    parser.add_argument("--archive-actual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = report(
        args.migration_equivalence,
        args.archive_expected,
        args.archive_actual,
    )
    atomic_write_json(args.output, payload)
    if not payload["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
