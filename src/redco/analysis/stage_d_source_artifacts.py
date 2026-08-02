"""Crash-safe persistence for prepared and completed Stage-D source rollouts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_receipt_ledger import inspect_ledger
from redco.analysis.stage_d_source_contracts import SourceRollout
from redco.contracts import canonical_json
from redco.integrations.write_once import write_once


class SourceArtifactError(RuntimeError):
    """A source artifact cannot be persisted or recovered without ambiguity."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class StageDSourceArtifactStore:
    """Persist each source around its ledger receipt without an unrecoverable gap."""

    def __init__(self, root: Path) -> None:
        if root.is_symlink():
            raise SourceArtifactError("source artifact root cannot be a symbolic link")
        self._root = root
        self._pending = root / "pending"
        self._sources = root / "sources"
        root.mkdir(parents=True, exist_ok=True)
        if self._pending.is_symlink() or self._sources.is_symlink():
            raise SourceArtifactError("source artifact directories cannot be symbolic links")
        self._pending.mkdir(exist_ok=True)
        self._sources.mkdir(exist_ok=True)
        _fsync_directory(root)

    def prepare(self, value: bytes) -> str:
        """Durably stage a source payload before its completion receipt is appended."""
        prepared = _parse_prepared(value)
        digest = str(prepared["source_sha256"])
        write_once(
            self._pending / f"{digest}.json",
            value,
            error_type=SourceArtifactError,
        )
        return digest

    def commit(self, source: SourceRollout) -> Path:
        """Install a receipt-bearing source and retire its matching prepared payload."""
        if type(source) is not SourceRollout or source.evidence_class != "live":
            raise ValueError("artifact store accepts only live verified source rollouts")
        pending_path = self._pending / f"{source.source_sha256}.json"
        if not pending_path.is_file():
            raise SourceArtifactError("source completion lacks its prepared artifact")
        prepared = _parse_prepared(pending_path.read_bytes())
        if (
            prepared["source_sha256"] != source.source_sha256
            or prepared["source"] != source.to_payload()
        ):
            raise SourceArtifactError("completed source differs from its prepared artifact")
        destination = self._sources / f"{source.source_sha256}.json"
        write_once(destination, source.to_bytes(), error_type=SourceArtifactError)
        pending_path.unlink()
        _fsync_directory(self._pending)
        return destination

    def recover_completed(self, ledger_root: Path) -> tuple[Path, ...]:
        """Promote receipt-completed pending sources after a process interruption."""
        scan = inspect_ledger(ledger_root)
        if scan.status != "active-clean":
            raise SourceArtifactError(
                f"source recovery requires an active-clean ledger, got {scan.status}"
            )
        completions: dict[tuple[str, str, str], dict[str, Any]] = {}
        for (kind, _), receipt in scan.receipts.items():
            if kind != "source_rollout_completed":
                continue
            key = (
                str(receipt.get("group_id")),
                str(receipt.get("rollout_id")),
                str(receipt.get("source_sha256")),
            )
            if key in completions:
                raise SourceArtifactError("duplicate source completion receipt")
            completions[key] = receipt

        recovered: list[Path] = []
        for pending_path in sorted(self._pending.glob("*.json")):
            encoded = pending_path.read_bytes()
            prepared = _parse_prepared(encoded)
            payload = prepared["source"]
            key = (
                str(payload["group_id"]),
                str(payload["rollout_id"]),
                prepared["source_sha256"],
            )
            completion_receipt = completions.get(key)
            if completion_receipt is None:
                raise SourceArtifactError(
                    "prepared source has no durable completion receipt"
                )
            source_bytes = canonical_json(
                {
                    "schema_version": 1,
                    "domain": "redco-stage-d-source-rollout-v1",
                    "source": payload,
                    "source_sha256": prepared["source_sha256"],
                    "producer_receipt": completion_receipt,
                }
            )
            destination = self._sources / f"{prepared['source_sha256']}.json"
            write_once(destination, source_bytes, error_type=SourceArtifactError)
            pending_path.unlink()
            _fsync_directory(self._pending)
            recovered.append(destination)
        return tuple(recovered)

    def source_paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self._sources.glob("*.json")))

    def assert_pristine(self) -> None:
        """Require a brand-new artifact root before a one-shot collection starts."""
        expected = {self._pending, self._sources}
        actual = set(self._root.iterdir())
        if actual != expected or any(not path.is_dir() for path in expected):
            raise SourceArtifactError(
                "source artifact root contains unexpected startup entries"
            )
        if any(self._pending.iterdir()):
            raise SourceArtifactError(
                "source artifact root contains stale pending payloads"
            )
        if any(self._sources.iterdir()):
            raise SourceArtifactError(
                "source artifact root contains stale completed sources"
            )

    def assert_no_pending(self) -> None:
        if any(self._pending.iterdir()):
            raise SourceArtifactError("source artifact store still has pending payloads")


def _parse_prepared(value: bytes) -> dict[str, Any]:
    if type(value) is not bytes:
        raise ValueError("prepared source artifact must be immutable bytes")
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("prepared source artifact must be JSON") from error
    if not isinstance(parsed, dict) or canonical_json(parsed) != value:
        raise ValueError("prepared source artifact must be canonical JSON")
    if set(parsed) != {"schema_version", "domain", "source", "source_sha256"}:
        raise ValueError("prepared source artifact fields differ")
    if (
        parsed["schema_version"] != 1
        or parsed["domain"] != "redco-stage-d-prepared-source-rollout-v1"
        or not isinstance(parsed["source"], dict)
    ):
        raise ValueError("prepared source artifact envelope is invalid")
    expected = _sha256(
        canonical_json(
            {
                "domain": "redco-stage-d-source-rollout-v1",
                "source": parsed["source"],
            }
        )
    )
    if not isinstance(parsed["source_sha256"], str) or parsed["source_sha256"] != expected:
        raise ValueError("prepared source artifact digest mismatch")
    return parsed


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
