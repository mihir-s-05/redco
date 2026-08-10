"""Dependency-free integrity primitives shared by Redco protocol owners."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypeGuard


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of immutable bytes."""
    return hashlib.sha256(value).hexdigest()


def is_sha256_hex(value: object) -> TypeGuard[str]:
    """Return whether *value* is exactly a lowercase hexadecimal SHA-256 digest."""
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def require_sha256_hex(value: object, name: str) -> str:
    """Return a valid digest or raise the protocol-standard validation error."""
    if not is_sha256_hex(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def resolve_contained_file(root: Path, relative: str | Path) -> Path | None:
    """Resolve one regular non-symlink file only when it stays below *root*."""

    resolved_root = root.resolve()
    candidate = resolved_root / relative
    if not candidate.is_file() or candidate.is_symlink():
        return None
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


__all__ = [
    "is_sha256_hex",
    "require_sha256_hex",
    "resolve_contained_file",
    "sha256_bytes",
]
