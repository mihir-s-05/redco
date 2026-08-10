"""Atomic storage and hash-chain codec for Stage-D evaluation state."""

from __future__ import annotations

import importlib
import json
import os
import secrets
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json
from redco.integrity import sha256_bytes

_RECORD_DOMAIN = "redco-stage-d-evaluation-ledger-record-v1"
FaultHook = Callable[[str, Path], None]


def sha256(value: bytes) -> str:
    """Compatibility name for the evaluation codec's public digest helper."""
    return sha256_bytes(value)


def canonical_object(value: bytes, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(payload, dict) or canonical_json(payload) != value:
        raise ValueError(f"{name} is not canonical JSON")
    return payload


def atomic_publish(
    path: Path,
    value: bytes,
    *,
    fault_hook: FaultHook | None = None,
) -> None:
    """Publish exact bytes create-only; a killed writer cannot expose a partial final."""
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != value:
            raise FileExistsError(f"durable evaluation state differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending.{os.getpid()}.{secrets.token_hex(8)}")
    descriptor = os.open(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        if fault_hook is not None:
            fault_hook("after-evaluation-ledger-pending-fsync", pending)
        try:
            os.link(pending, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != value:
                raise FileExistsError(f"durable evaluation state differs: {path}") from None
        if fault_hook is not None:
            fault_hook("after-evaluation-ledger-link", path)
        fsync_directory(path.parent)
    finally:
        pending.unlink(missing_ok=True)


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class EvaluationEvidenceStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, value: bytes, *, fault_hook: FaultHook | None = None) -> str:
        if type(value) is not bytes:
            raise TypeError("evaluation evidence must be immutable bytes")
        digest = sha256(value)
        atomic_publish(self.root / digest, value, fault_hook=fault_hook)
        return digest

    def get(self, digest: str) -> bytes:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("evaluation evidence digest is invalid")
        path = self.root / digest
        if path.is_symlink() or not path.is_file():
            raise ValueError("evaluation evidence is absent or symbolic")
        value = path.read_bytes()
        if sha256(value) != digest:
            raise ValueError("evaluation evidence differs from its digest")
        return value


def encode_record(
    *,
    offset: int,
    prior_record_sha256: str | None,
    record_kind: str,
    event: Mapping[str, Any],
) -> bytes:
    if type(offset) is not int or offset < 0:
        raise ValueError("evaluation record offset is invalid")
    if (offset == 0) != (prior_record_sha256 is None):
        raise ValueError("evaluation record prior hash is inconsistent")
    if prior_record_sha256 is not None and (
        len(prior_record_sha256) != 64
        or any(character not in "0123456789abcdef" for character in prior_record_sha256)
    ):
        raise ValueError("evaluation record prior hash is invalid")
    if not record_kind or not record_kind.isprintable():
        raise ValueError("evaluation record kind is invalid")
    return canonical_json(
        {
            "schema_version": 1,
            "domain": _RECORD_DOMAIN,
            "offset": offset,
            "prior_record_sha256": prior_record_sha256,
            "record_kind": record_kind,
            "event": dict(event),
        }
    )


def decode_record(value: bytes) -> dict[str, Any]:
    payload = canonical_object(value, "evaluation ledger record")
    if (
        set(payload)
        != {
            "schema_version",
            "domain",
            "offset",
            "prior_record_sha256",
            "record_kind",
            "event",
        }
        or payload.get("schema_version") != 1
        or payload.get("domain") != _RECORD_DOMAIN
        or not isinstance(payload.get("event"), dict)
    ):
        raise ValueError("evaluation ledger record fields differ")
    if (
        encode_record(
            offset=payload["offset"],
            prior_record_sha256=payload["prior_record_sha256"],
            record_kind=payload["record_kind"],
            event=payload["event"],
        )
        != value
    ):
        raise ValueError("evaluation ledger record is noncanonical")
    return payload


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            msvcrt = importlib.import_module("msvcrt")

            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl = importlib.import_module("fcntl")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            msvcrt = importlib.import_module("msvcrt")

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl = importlib.import_module("fcntl")
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


__all__ = [
    "EvaluationEvidenceStore",
    "atomic_publish",
    "canonical_object",
    "decode_record",
    "encode_record",
    "exclusive_lock",
    "sha256",
]
