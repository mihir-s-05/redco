"""Audit the single-field Stage D v4.10 eager inference config change."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import tomllib
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(parent: Path, eager: Path) -> dict[str, Any]:
    parent_data = tomllib.loads(parent.read_text(encoding="utf-8"))
    eager_data = tomllib.loads(eager.read_text(encoding="utf-8"))
    parent_without = {**parent_data, "model": {**parent_data["model"]}}
    eager_without = {**eager_data, "model": {**eager_data["model"]}}
    parent_value = parent_without["model"].pop("enforce_eager")
    eager_value = eager_without["model"].pop("enforce_eager")
    diff = list(
        difflib.unified_diff(
            parent.read_text(encoding="utf-8").splitlines(),
            eager.read_text(encoding="utf-8").splitlines(),
            fromfile=parent.as_posix(),
            tofile=eager.as_posix(),
            lineterm="",
        )
    )
    checks = {
        "parent_is_not_eager": parent_value is False,
        "successor_is_eager": eager_value is True,
        "all_other_toml_fields_equal": parent_without == eager_without,
        "exactly_one_diff_hunk": sum(line.startswith("@@") for line in diff) == 1,
        "exactly_one_removed_value": sum(line == "-enforce_eager = false" for line in diff)
        == 1,
        "exactly_one_added_value": sum(line == "+enforce_eager = true" for line in diff)
        == 1,
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-v4-10-eager-config",
            "parent": parent.as_posix(),
            "parent_sha256": _sha256(parent),
            "eager": eager.as_posix(),
            "eager_sha256": _sha256(eager),
            "unified_diff": diff,
            "checks": checks,
            "passes": all(checks.values()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--eager", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.parent, args.eager)
    atomic_write_json(args.output, payload)
    if not payload["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
