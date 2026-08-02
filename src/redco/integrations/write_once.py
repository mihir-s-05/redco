"""One durable, non-overwriting byte writer shared by evidence stores."""

from __future__ import annotations

import os
import secrets
from pathlib import Path


def write_once(
    path: Path,
    value: bytes,
    *,
    allow_existing_same: bool = True,
    error_type: type[Exception] = FileExistsError,
) -> None:
    if type(value) is not bytes or not value:
        raise ValueError("write-once value must be nonempty bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise error_type(f"write-once collision at {path}")
    if path.exists():
        if allow_existing_same and path.read_bytes() == value:
            return
        raise error_type(f"write-once collision at {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if (
                path.is_symlink()
                or not allow_existing_same
                or path.read_bytes() != value
            ):
                raise error_type(f"write-once collision at {path}") from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
