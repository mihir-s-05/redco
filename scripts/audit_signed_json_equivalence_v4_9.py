"""Verify exact semantic equality of two strictly parsed signed JSON objects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON numeric constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_strict_signed_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("signed JSON payload must be an object")
    verify_signed_payload(payload)
    return payload


def _typed_canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def audit(expected_path: Path, actual_path: Path) -> dict[str, Any]:
    expected = load_strict_signed_json(expected_path)
    actual = load_strict_signed_json(actual_path)
    equal = _typed_canonical(expected) == _typed_canonical(actual)
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "strict-signed-json-equivalence-v4-9",
            "expected_path": expected_path.as_posix(),
            "expected_file_sha256": _sha256(expected_path),
            "expected_signed_payload_sha256": expected[
                "signed_payload_sha256"
            ],
            "actual_path": actual_path.as_posix(),
            "actual_file_sha256": _sha256(actual_path),
            "actual_signed_payload_sha256": actual[
                "signed_payload_sha256"
            ],
            "strict_full_object_equal": equal,
            "passes": equal,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.expected, args.actual)
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
