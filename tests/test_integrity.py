from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping

import pytest

from redco.integrity import is_sha256_hex, require_sha256_hex, sha256_bytes
from scripts.verify_repository import (
    AMBIENT_PYTEST_CONTROLS,
    EXPECTED_REQUIRED_NODE_IDS_SHA256,
    EXPECTED_REQUIRED_NODES,
    OUTCOME_NAMES,
    _prepare_pytest_environment,
    node_ids_sha256,
    profile_exit_status,
    profile_outcomes_are_green,
    required_membership_matches,
)


class _Digest(str):
    def __eq__(self, _other: object) -> bool:
        return True

    def __ne__(self, _other: object) -> bool:
        return False


def _matrix_exit_status(
    outcomes: Mapping[str, int],
    *,
    collect_only: bool = False,
    membership_ok: bool = True,
    selected: int = 1,
    executed: int = 1,
) -> int:
    return profile_exit_status(
        0,
        collect_only=collect_only,
        membership_ok=membership_ok,
        selected=selected,
        executed=executed,
        outcomes=outcomes,
    )


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


def test_repository_matrix_owns_environment_membership_and_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert EXPECTED_REQUIRED_NODES > 0
    assert is_sha256_hex(EXPECTED_REQUIRED_NODE_IDS_SHA256)
    assert EXPECTED_REQUIRED_NODE_IDS_SHA256 != "0" * 64

    green = {name: int(name == "passed") for name in OUTCOME_NAMES}
    assert profile_outcomes_are_green(1, 1, green)
    assert not profile_outcomes_are_green(2, 1, green)
    assert _matrix_exit_status(green) == 0
    assert _matrix_exit_status(
        {name: 0 for name in OUTCOME_NAMES},
        collect_only=True,
        membership_ok=False,
        executed=0,
    ) == 1
    for outcome in OUTCOME_NAMES[1:]:
        red = {**green, "passed": 0, outcome: 1}
        assert not profile_outcomes_are_green(1, 1, red)
        assert _matrix_exit_status(red) == 1
        assert _matrix_exit_status(red, collect_only=True) == 0

    for name, value in zip(
        AMBIENT_PYTEST_CONTROLS,
        ("--ignore=tests/test_integrity.py", "tests.test_integrity"),
        strict=True,
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "0")
    assert _prepare_pytest_environment() == AMBIENT_PYTEST_CONTROLS
    assert all(name not in os.environ for name in AMBIENT_PYTEST_CONTROLS)
    assert os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"

    expected = ("tests/a.py::test_a", "tests/b.py::test_b")
    expected_sha256 = node_ids_sha256(expected)
    assert required_membership_matches(
        expected,
        expected_count=2,
        expected_sha256=expected_sha256,
    )
    assert not required_membership_matches(
        (expected[0], "tests/b.py::test_c"),
        expected_count=2,
        expected_sha256=expected_sha256,
    )
    assert not required_membership_matches(
        (),
        expected_count=2,
        expected_sha256=expected_sha256,
    )
    duplicate = (expected[0], expected[0])
    assert not required_membership_matches(
        duplicate,
        expected_count=2,
        expected_sha256=node_ids_sha256(duplicate),
    )
