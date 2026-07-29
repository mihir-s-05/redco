"""Durable signed-artifact boundaries for native-library worker processes."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn


def canonical_sha256(payload: dict[str, Any]) -> str:
    """Hash a JSON payload using the project's canonical representation."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sign_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *payload* with its canonical SHA-256 signature."""
    if "signed_payload_sha256" in payload:
        raise ValueError("refusing to sign a payload that already has a signature")
    return {
        **payload,
        "signed_payload_sha256": canonical_sha256(payload),
    }


def verify_signed_payload(payload: dict[str, Any]) -> str:
    """Verify and return a signed payload's canonical SHA-256."""
    signature = payload.get("signed_payload_sha256")
    if not isinstance(signature, str) or len(signature) != 64:
        raise ValueError("missing or malformed signed_payload_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "signed_payload_sha256"}
    expected = canonical_sha256(unsigned)
    if signature != expected:
        raise ValueError(
            f"signed payload SHA-256 mismatch: recorded={signature}, recomputed={expected}"
        )
    return signature


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write and fsync JSON in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_descriptor)
        except OSError:
            pass
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def run_and_hard_exit(
    entrypoint: Callable[[], object],
    *,
    exit_process: Callable[[int], NoReturn] = os._exit,
) -> NoReturn:
    """Run a native worker and exit without invoking extension finalizers."""
    status = 0
    try:
        entrypoint()
    except BaseException:
        traceback.print_exc()
        status = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    exit_process(status)
