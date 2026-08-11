from __future__ import annotations

import hashlib

import pytest

from redco.integrity import is_sha256_hex, require_sha256_hex, sha256_bytes


class _Digest(str):
    def __eq__(self, _other: object) -> bool:
        return True

    def __ne__(self, _other: object) -> bool:
        return False


@pytest.mark.parametrize("value", [b"", b"redco", "µ".encode()])
def test_sha256_bytes_matches_hashlib(value: bytes) -> None:
    assert sha256_bytes(value) == hashlib.sha256(value).hexdigest()


def test_digest_validation_accepts_exact_lowercase_strings() -> None:
    digest = "0123456789abcdef" * 4

    assert is_sha256_hex(digest)
    assert require_sha256_hex(digest, "digest") is digest


@pytest.mark.parametrize(
    "value",
    [
        None,
        b"0" * 64,
        "0" * 63,
        "0" * 65,
        "A" * 64,
        "g" * 64,
        _Digest("0" * 64),
    ],
)
def test_digest_validation_rejects_noncanonical_values(value: object) -> None:
    assert not is_sha256_hex(value)
    with pytest.raises(ValueError, match=r"^digest must be a lowercase SHA-256$"):
        require_sha256_hex(value, "digest")
