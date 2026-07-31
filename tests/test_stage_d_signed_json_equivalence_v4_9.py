from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from redco.integrations.signed_subprocess import sign_payload

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_signed_json_equivalence_v4_9 import (  # noqa: E402
    audit,
    load_strict_signed_json,
)


def _write(path: Path, payload: dict, newline: str = "\n") -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_bytes(text.replace("\n", newline).encode())


def test_crlf_and_lf_are_equivalent(tmp_path: Path) -> None:
    payload = sign_payload({"schema_version": 1, "value": [1, True, None]})
    expected = tmp_path / "expected.json"
    actual = tmp_path / "actual.json"
    _write(expected, payload, "\r\n")
    _write(actual, payload, "\n")
    report = audit(expected, actual)
    assert report["passes"] is True
    assert report["expected_file_sha256"] != report["actual_file_sha256"]


def test_invalid_signature_fails(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    _write(path, {"value": 2, "signed_payload_sha256": "0" * 64})
    with pytest.raises(ValueError, match="signed payload SHA-256 mismatch"):
        load_strict_signed_json(path)


def test_freshly_resigned_semantic_change_fails(tmp_path: Path) -> None:
    expected = tmp_path / "expected.json"
    actual = tmp_path / "actual.json"
    _write(expected, sign_payload({"value": 1}))
    _write(actual, sign_payload({"value": 2}))
    assert audit(expected, actual)["passes"] is False


@pytest.mark.parametrize(
    "actual_payload",
    [sign_payload({"value": 1, "extra": 2}), sign_payload({})],
)
def test_missing_or_extra_keys_fail(
    tmp_path: Path, actual_payload: dict
) -> None:
    expected = tmp_path / "expected.json"
    actual = tmp_path / "actual.json"
    _write(expected, sign_payload({"value": 1}))
    _write(actual, actual_payload)
    assert audit(expected, actual)["passes"] is False


def test_duplicate_keys_and_nonstandard_numbers_fail(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"value":1,"value":1,"signed_payload_sha256":"' + "0" * 64 + '"}'
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_strict_signed_json(duplicate)

    nonstandard = tmp_path / "nan.json"
    nonstandard.write_text(
        '{"value":NaN,"signed_payload_sha256":"' + "0" * 64 + '"}'
    )
    with pytest.raises(ValueError, match="nonstandard JSON numeric"):
        load_strict_signed_json(nonstandard)
