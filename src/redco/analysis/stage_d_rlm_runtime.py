"""Fail-closed binding of the frozen patched-RLM install bundle to live configs."""

from __future__ import annotations

import hashlib
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from redco.analysis.stage_d_dependency_stack import StageDDependencyStackManifest
from redco.analysis.stage_d_protocol_manifest import StageDProtocolManifest


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_regular_absolute(path: Path, name: str) -> bytes:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"Stage-D {name} must be an absolute regular file")
    return path.read_bytes()


@dataclass(frozen=True, slots=True)
class StageDRLMInstallBundle:
    archive_path: Path
    uv_path: Path
    cache_archive_path: Path
    launcher_path: Path

    def verify(self, manifest: StageDDependencyStackManifest) -> None:
        archive = _require_regular_absolute(self.archive_path, "RLM archive")
        uv = _require_regular_absolute(self.uv_path, "uv executable")
        cache = _require_regular_absolute(self.cache_archive_path, "uv cache archive")
        launcher = _require_regular_absolute(self.launcher_path, "RLM launcher")
        if (
            _sha256(archive) != manifest.rlm_archive_sha256
            or _sha256(uv) != manifest.rlm_uv_binary_sha256
            or _sha256(cache) != manifest.rlm_uv_cache_archive_sha256
            or _sha256(launcher) != manifest.rlm_executable_sha256
        ):
            raise ValueError("Stage-D frozen RLM install asset differs from dependency stack")
        if _archive_uv_lock_sha256(self.archive_path) != manifest.uv_lock_sha256:
            raise ValueError("Stage-D RLM archive uv.lock differs from dependency stack")
        if not _archive_has_provenance_modules(self.archive_path):
            raise ValueError("Stage-D RLM archive lacks patched spawn provenance")


def load_stage_d_rlm_runtime(
    *,
    protocol: StageDProtocolManifest,
    dependency_stack_path: Path,
    archive_path: Path,
    uv_path: Path,
    cache_archive_path: Path,
    launcher_path: Path,
) -> tuple[StageDDependencyStackManifest, StageDRLMInstallBundle]:
    dependency_bytes = _require_regular_absolute(
        dependency_stack_path,
        "dependency stack",
    )
    if _sha256(dependency_bytes) != protocol.dependency_stack_sha256:
        raise ValueError("Stage-D dependency stack differs from protocol manifest")
    manifest = StageDDependencyStackManifest.from_bytes(dependency_bytes)
    bundle = StageDRLMInstallBundle(
        archive_path,
        uv_path,
        cache_archive_path,
        launcher_path,
    )
    bundle.verify(manifest)
    return manifest, bundle


class RLMHarnessConfig(Protocol):
    id: str
    checkout_archive_path: str | None
    checkout_archive_sha256: str | None
    checkout_uv_path: str | None
    checkout_uv_sha256: str | None
    checkout_cache_archive_path: str | None
    checkout_cache_archive_sha256: str | None
    checkout_uv_lock_sha256: str | None
    checkout_launcher_path: str | None
    checkout_launcher_sha256: str | None


class EnvConfigWithHarnesses(Protocol):
    def agent_harnesses(self) -> dict[str, RLMHarnessConfig]: ...


def verify_stage_d_rlm_harness(
    harness: RLMHarnessConfig,
    *,
    manifest: StageDDependencyStackManifest,
    bundle: StageDRLMInstallBundle,
) -> None:
    """Require the exact frozen install bundle on every resolved RLM harness."""
    bundle.verify(manifest)
    expected = {
        "id": "rlm",
        "checkout_archive_path": str(bundle.archive_path),
        "checkout_archive_sha256": manifest.rlm_archive_sha256,
        "checkout_uv_path": str(bundle.uv_path),
        "checkout_uv_sha256": manifest.rlm_uv_binary_sha256,
        "checkout_cache_archive_path": str(bundle.cache_archive_path),
        "checkout_cache_archive_sha256": manifest.rlm_uv_cache_archive_sha256,
        "checkout_uv_lock_sha256": manifest.uv_lock_sha256,
        "checkout_launcher_path": str(bundle.launcher_path),
        "checkout_launcher_sha256": manifest.rlm_executable_sha256,
    }
    observed = {name: getattr(harness, name, None) for name in expected}
    if observed != expected:
        raise ValueError("Stage-D resolved RLM harness is not bound to the frozen install bundle")


def verify_stage_d_env_rlm_harnesses(
    env_config: EnvConfigWithHarnesses,
    *,
    manifest: StageDDependencyStackManifest,
    bundle: StageDRLMInstallBundle,
) -> None:
    harnesses = env_config.agent_harnesses()
    if not harnesses:
        raise ValueError("Stage-D environment has no resolved agent harnesses")
    for harness in harnesses.values():
        verify_stage_d_rlm_harness(harness, manifest=manifest, bundle=bundle)


def _archive_uv_lock_sha256(path: Path) -> str:
    return _sha256(_read_regular_archive_member(path, "uv.lock"))


def _archive_has_provenance_modules(path: Path) -> bool:
    required = {
        "src/rlm/provenance.py",
        "src/rlm/session.py",
        "src/rlm/cli.py",
    }
    with tarfile.open(path, mode="r") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
    if not required.issubset(names):
        return False
    provenance = _read_regular_archive_member(path, "src/rlm/provenance.py")
    return b"redco.stage-d.spawn-lineage.v2" in provenance


def _read_regular_archive_member(path: Path, name: str) -> bytes:
    expected = PurePosixPath(name)
    if expected.is_absolute() or ".." in expected.parts:
        raise ValueError("Stage-D RLM archive member name is unsafe")
    with tarfile.open(path, mode="r") as archive:
        matches = [member for member in archive.getmembers() if member.name == name]
        if len(matches) != 1 or not matches[0].isfile():
            raise ValueError(f"Stage-D RLM archive lacks one regular {name}")
        stream = archive.extractfile(matches[0])
        if stream is None:
            raise ValueError(f"Stage-D RLM archive member is unreadable: {name}")
        return stream.read()


__all__ = [
    "StageDRLMInstallBundle",
    "load_stage_d_rlm_runtime",
    "verify_stage_d_env_rlm_harnesses",
    "verify_stage_d_rlm_harness",
]
