"""Crash-idempotent archive and ledger repair for one frozen zero-call failure."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_receipt_ledger import (
    StageDReceiptLedger,
    inspect_ledger,
)
from redco.contracts import canonical_json

_DOMAIN = "redco-stage-d-zero-call-repair-archive-v2"
_REASON = "explicit supervisor-confirmed zero-call scientific worker failure"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def recover_or_open_scientific_ledger(
    *,
    ledger_root: Path,
    master_seed: str,
    recover_requested: bool,
    supervisor_evidence_path: Path | None,
    repair_archive: Path | None,
    episode_output: Path,
) -> StageDReceiptLedger:
    """Perform, resume, or verify one exact recovery transaction before live imports."""
    scan = inspect_ledger(ledger_root, allow_repairable_zero_call=True)
    if not recover_requested:
        if scan.status == "active-repairable-zero-call":
            raise RuntimeError(
                "repairable zero-call state requires explicit supervisor evidence and archive"
            )
        return StageDReceiptLedger(ledger_root, master_seed=master_seed)
    if supervisor_evidence_path is None or repair_archive is None:
        raise RuntimeError("zero-call recovery requires supervisor evidence and archive")
    evidence = supervisor_evidence_path.read_bytes()
    if not evidence:
        raise ValueError("zero-call supervisor evidence must be nonempty")
    evidence_sha256 = _sha256(evidence)
    if scan.status == "active-repairable-zero-call":
        if scan.repairable_attempt is None:
            raise RuntimeError("repairable ledger lacks its exact attempt")
        attempt = dict(scan.repairable_attempt)
        _install_or_verify_archive(
            archive=repair_archive,
            episode_output=episode_output,
            repairable_attempt=attempt,
            supervisor_evidence_sha256=evidence_sha256,
        )
        return StageDReceiptLedger.recover_zero_call_failure(
            ledger_root,
            master_seed=master_seed,
            reason=_REASON,
            supervisor_evidence=evidence,
        )
    if scan.status != "active-clean":
        raise RuntimeError("zero-call recovery requested for a non-repairable ledger")
    archive_payload = _verify_archive(
        repair_archive,
        supervisor_evidence_sha256=evidence_sha256,
    )
    attempt = archive_payload["repairable_attempt"]
    matching = [
        receipt
        for (kind, _), receipt in scan.receipts.items()
        if kind in {"zero_call_infrastructure_failure", "zero_call_execution_failure"}
        and receipt.get("attempt_id") == attempt.get("attempt_id")
        and receipt.get("reason") == _REASON
    ]
    if len(matching) != 1 or evidence_sha256 not in scan.evidence_refs:
        raise RuntimeError("completed zero-call repair lacks one exact ledger receipt")
    return StageDReceiptLedger(ledger_root, master_seed=master_seed)


def _install_or_verify_archive(
    *,
    archive: Path,
    episode_output: Path,
    repairable_attempt: dict[str, Any],
    supervisor_evidence_sha256: str,
) -> None:
    archive_resolved = archive.resolve()
    episode_resolved = episode_output.resolve()
    if (
        archive_resolved == episode_resolved
        or archive_resolved in episode_resolved.parents
        or episode_resolved in archive_resolved.parents
    ):
        raise ValueError("repair archive and episode output must be path-disjoint")
    if archive.exists():
        payload = _verify_archive(
            archive,
            supervisor_evidence_sha256=supervisor_evidence_sha256,
        )
        if payload["repairable_attempt"] != repairable_attempt:
            raise ValueError("repair archive belongs to a different ledger attempt")
        if episode_output.exists():
            raise ValueError("archived episode output still exists at its live path")
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    pending = archive.with_name(f".{archive.name}.pending")
    manifest_path = pending / "repair.json"
    if not pending.exists():
        pending.mkdir()
        payload = {
            "schema_version": 1,
            "domain": _DOMAIN,
            "repairable_attempt": repairable_attempt,
            "supervisor_evidence_sha256": supervisor_evidence_sha256,
            "episode_output_present": episode_output.exists(),
            "episode_output_manifest_sha256": (
                _tree_sha256(episode_output) if episode_output.exists() else None
            ),
        }
        _exclusive_write(manifest_path, canonical_json(payload))
        _fsync_directory(pending)
    payload = _read_archive_manifest(pending)
    if (
        payload["repairable_attempt"] != repairable_attempt
        or payload["supervisor_evidence_sha256"] != supervisor_evidence_sha256
    ):
        raise ValueError("pending repair archive belongs to a different attempt")
    archived_output = pending / "episode-output"
    if payload["episode_output_present"]:
        if not archived_output.exists():
            if not episode_output.exists():
                raise RuntimeError("repair crashed before preserving its declared episode output")
            os.replace(episode_output, archived_output)
            _fsync_directory(episode_output.parent)
            _fsync_directory(pending)
        elif episode_output.exists():
            raise RuntimeError("repair has duplicate live and archived episode output")
    elif episode_output.exists() or archived_output.exists():
        raise RuntimeError("repair archive episode-output state differs from its manifest")
    os.replace(pending, archive)
    _fsync_directory(archive.parent)
    _verify_archive(
        archive,
        supervisor_evidence_sha256=supervisor_evidence_sha256,
    )


def _verify_archive(
    archive: Path,
    *,
    supervisor_evidence_sha256: str,
) -> dict[str, Any]:
    if not archive.is_dir():
        raise FileNotFoundError("zero-call repair archive is absent")
    payload = _read_archive_manifest(archive)
    if payload["supervisor_evidence_sha256"] != supervisor_evidence_sha256:
        raise ValueError("repair archive supervisor evidence differs")
    output = archive / "episode-output"
    if output.exists() != payload["episode_output_present"]:
        raise ValueError("repair archive episode-output roster differs")
    observed_output_sha256 = _tree_sha256(output) if output.exists() else None
    if observed_output_sha256 != payload["episode_output_manifest_sha256"]:
        raise ValueError("repair archive episode output differs from its manifest")
    expected = {"repair.json"} | ({"episode-output"} if output.exists() else set())
    if {path.name for path in archive.iterdir()} != expected:
        raise ValueError("repair archive contains unexpected members")
    return payload


def _read_archive_manifest(root: Path) -> dict[str, Any]:
    value = (root / "repair.json").read_bytes()
    try:
        payload = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("repair archive manifest is not JSON") from error
    expected = {
        "schema_version",
        "domain",
        "repairable_attempt",
        "supervisor_evidence_sha256",
        "episode_output_present",
        "episode_output_manifest_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload.get("schema_version") != 1
        or payload.get("domain") != _DOMAIN
        or canonical_json(payload) != value
        or not isinstance(payload.get("repairable_attempt"), dict)
        or type(payload.get("episode_output_present")) is not bool
        or (
            payload.get("episode_output_manifest_sha256") is not None
            and (
                not isinstance(payload["episode_output_manifest_sha256"], str)
                or len(payload["episode_output_manifest_sha256"]) != 64
            )
        )
        or payload["episode_output_present"]
        != (payload["episode_output_manifest_sha256"] is not None)
    ):
        raise ValueError("repair archive manifest is noncanonical or has different fields")
    return payload


def _exclusive_write(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _tree_sha256(path: Path) -> str:
    entries: list[dict[str, object]] = []
    if path.is_symlink():
        raise ValueError("repair archive forbids symbolic episode output")
    if path.is_file():
        value = path.read_bytes()
        entries.append({"path": ".", "size": len(value), "sha256": _sha256(value)})
    elif path.is_dir():
        for item in sorted(path.rglob("*"), key=lambda candidate: candidate.as_posix()):
            if item.is_symlink() or not (item.is_file() or item.is_dir()):
                raise ValueError("repair archive contains a non-regular output member")
            relative = item.relative_to(path).as_posix()
            if item.is_dir():
                entries.append({"path": relative + "/", "size": 0, "sha256": None})
            else:
                value = item.read_bytes()
                entries.append(
                    {"path": relative, "size": len(value), "sha256": _sha256(value)}
                )
    else:
        raise ValueError("repair episode output is neither a regular file nor directory")
    return _sha256(canonical_json({"entries": entries}))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["recover_or_open_scientific_ledger"]
