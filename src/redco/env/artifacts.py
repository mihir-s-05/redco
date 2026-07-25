"""Immutable, content-addressed, SSA-versioned artifacts."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    digest: str
    media_type: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class VersionedArtifact:
    logical_name: str
    version: int
    ref: ArtifactRef

    @property
    def artifact_id(self) -> str:
        return f"{self.logical_name}^{self.version}"


class ArtifactStore:
    """Disk-backed CAS using atomic, idempotent writes."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        digest = hashlib.sha256(data).hexdigest()
        path = self._path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise RuntimeError("content-address collision")
        else:
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_bytes(data)
            os.replace(temporary, path)
        return ArtifactRef(digest=digest, media_type=media_type, size_bytes=len(data))

    def put_json(self, value: Any) -> ArtifactRef:
        return self.put_bytes(canonical_json(value), media_type="application/json")

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        data = self._path(ref.digest).read_bytes()
        if len(data) != ref.size_bytes or hashlib.sha256(data).hexdigest() != ref.digest:
            raise RuntimeError(f"artifact verification failed: {ref.digest}")
        return data

    def _path(self, digest: str) -> Path:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("digest must be lowercase sha256 hex")
        return self.root / digest[:2] / digest[2:]


class ArtifactLedger:
    """Assign monotonically increasing SSA versions to logical artifact names."""

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store
        self._latest: dict[str, VersionedArtifact] = {}

    def write_json(self, logical_name: str, value: Any) -> VersionedArtifact:
        if not logical_name:
            raise ValueError("logical_name must be non-empty")
        previous = self._latest.get(logical_name)
        version = 0 if previous is None else previous.version + 1
        artifact = VersionedArtifact(logical_name, version, self.store.put_json(value))
        self._latest[logical_name] = artifact
        return artifact

    def latest(self, logical_name: str) -> VersionedArtifact:
        try:
            return self._latest[logical_name]
        except KeyError as error:
            raise KeyError(f"unknown logical artifact: {logical_name}") from error
