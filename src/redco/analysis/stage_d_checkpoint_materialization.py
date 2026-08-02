"""Crash-safe read-only checkpoint materialization from adopted trainer CAS bytes."""

from __future__ import annotations

import os
import secrets
import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from redco.analysis.stage_d_checkpoint_evidence import StageDCheckpointManifest
from redco.analysis.stage_d_objective_binding import ArmName
from redco.analysis.stage_d_training_completion import StageDTrainingCompletion


def materialize_adopted_checkpoint(
    *,
    training_entries: Mapping[str, bytes],
    arm: ArmName,
    destination: Path,
) -> StageDCheckpointManifest:
    completion_bytes = training_entries.get("completion.json")
    if completion_bytes is None:
        raise ValueError("training adoption lacks its completion")
    completion = StageDTrainingCompletion.from_bytes(completion_bytes)
    arm_completion = next(item for item in completion.arms if item.arm == arm)
    manifest_bytes = training_entries.get(f"evidence/{arm_completion.checkpoint_manifest_sha256}")
    if manifest_bytes is None:
        raise ValueError("training adoption lacks its checkpoint manifest")
    manifest = StageDCheckpointManifest.from_bytes(manifest_bytes)
    if (
        manifest.arm != arm
        or manifest.manifest_sha256 != arm_completion.checkpoint_manifest_sha256
        or tuple(member.sha256 for member in manifest.members)
        != arm_completion.checkpoint_member_sha256s
    ):
        raise ValueError("adopted checkpoint manifest differs from training completion")
    member_bytes = {}
    for member in manifest.members:
        value = training_entries.get(f"evidence/{member.sha256}")
        if value is None:
            raise ValueError("training adoption lacks checkpoint member evidence")
        member_bytes[member.path] = value
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise FileExistsError("checkpoint materialization destination is invalid")
    if destination.exists():
        verify_materialized_checkpoint(manifest, destination)
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.pending"
    )
    pending.mkdir(mode=0o700)
    try:
        for member in manifest.members:
            path = pending / Path(*PurePosixPath(member.path).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            _exclusive_file(path, member_bytes[member.path])
        manifest.verify_directory(pending)
        _make_read_only(pending)
        os.replace(pending, destination)
        _fsync_directory(destination.parent)
    finally:
        if pending.exists():
            for path in pending.rglob("*"):
                if path.is_file():
                    path.chmod(0o600)
            pending.chmod(0o700)
            shutil.rmtree(pending)
    verify_materialized_checkpoint(manifest, destination)
    return manifest


def verify_materialized_checkpoint(
    manifest: StageDCheckpointManifest,
    root: Path,
) -> None:
    manifest.verify_directory(root, verify_semantic=False)
    _require_read_only(root)


def _exclusive_file(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _make_read_only(root: Path) -> None:
    if os.name == "nt":
        return
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    root.chmod(0o555)
    _fsync_directory(root)


def _require_read_only(root: Path) -> None:
    if os.name == "nt":
        return
    if root.stat().st_mode & 0o222 or any(
        path.stat().st_mode & 0o222 for path in root.rglob("*") if path.is_file()
    ):
        raise ValueError("materialized checkpoint is not read-only")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["materialize_adopted_checkpoint", "verify_materialized_checkpoint"]
