"""Fail-closed verification for the retained Stage-D evaluation runtime archive."""

from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from pathlib import PurePosixPath

from redco.analysis.stage_d_evaluation_contracts import (
    StageDEvaluationExecutionManifest,
)

_MAX_MEMBERS = 4096
_MAX_EXPANDED_BYTES = 256 * 1024 * 1024
_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 250


def verify_evaluation_runtime_bundle(
    value: bytes,
    *,
    manifest: StageDEvaluationExecutionManifest,
) -> tuple[tuple[str, str], ...]:
    if hashlib.sha256(value).hexdigest() != manifest.runtime_bundle_sha256:
        raise ValueError("evaluation runtime bundle differs from its manifest")
    try:
        archive = zipfile.ZipFile(io.BytesIO(value), "r")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("evaluation runtime bundle is not a valid ZIP archive") from error
    with archive:
        members = archive.infolist()
        if not members or len(members) > _MAX_MEMBERS:
            raise ValueError("evaluation runtime bundle member count is invalid")
        names: list[str] = []
        expanded = 0
        digests: dict[str, str] = {}
        for member in members:
            name = _safe_member_name(member.filename)
            if name in names:
                raise ValueError("evaluation runtime bundle has duplicate members")
            names.append(name)
            if member.flag_bits & 0x1:
                raise ValueError("evaluation runtime bundle has encrypted members")
            mode = member.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError("evaluation runtime bundle has special members")
            if member.is_dir():
                continue
            if member.file_size > _MAX_MEMBER_BYTES:
                raise ValueError("evaluation runtime bundle member is too large")
            expanded += member.file_size
            if expanded > _MAX_EXPANDED_BYTES:
                raise ValueError("evaluation runtime bundle expands beyond its limit")
            if member.compress_size == 0 and member.file_size != 0:
                raise ValueError("evaluation runtime bundle has an invalid compression size")
            if (
                member.compress_size > 0
                and member.file_size / member.compress_size > _MAX_COMPRESSION_RATIO
            ):
                raise ValueError("evaluation runtime bundle compression ratio is excessive")
            try:
                member_bytes = archive.read(member)
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                raise ValueError("evaluation runtime bundle member cannot be read") from error
            if len(member_bytes) != member.file_size:
                raise ValueError("evaluation runtime bundle member size differs")
            digests[name] = hashlib.sha256(member_bytes).hexdigest()
        required_named = {
            source_name: expected_sha256
            for program in manifest.programs
            for source_name, expected_sha256 in program.source_sha256s
        }
        for name, expected_sha256 in required_named.items():
            if digests.get(name) != expected_sha256:
                raise ValueError("evaluation runtime program source differs")
        for entrypoint in manifest.runtime_entrypoints:
            if digests.get(entrypoint.member_path) != entrypoint.source_sha256:
                raise ValueError("evaluation runtime entrypoint source differs")
        return tuple(sorted(digests.items()))


def _safe_member_name(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("evaluation runtime bundle member path is unsafe")
    return value


__all__ = ["verify_evaluation_runtime_bundle"]
