from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_stage_d_dependency_stack import _manifest

from redco.analysis.stage_d_dependency_stack import write_canonical_tree_tar
from redco.analysis.stage_d_rlm_runtime import (
    StageDRLMInstallBundle,
    verify_stage_d_rlm_harness,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture(tmp_path):
    tree = tmp_path / "rlm"
    (tree / "src" / "rlm").mkdir(parents=True)
    (tree / "src" / "rlm" / "provenance.py").write_bytes(
        b'DOMAIN = "redco.stage-d.spawn-lineage.v2"\n'
    )
    (tree / "src" / "rlm" / "session.py").write_bytes(b"# patched session\n")
    (tree / "src" / "rlm" / "cli.py").write_bytes(b"# patched cli\n")
    (tree / "uv.lock").write_bytes(b"version = 1\n")
    archive = tmp_path / "rlm.tar"
    write_canonical_tree_tar(tree, archive)
    uv = tmp_path / "uv"
    uv.write_bytes(b"pinned uv")
    cache = tmp_path / "cache.tar.gz"
    cache.write_bytes(b"pinned cache")
    launcher = tmp_path / "rlm-wrapper"
    launcher.write_bytes(b"pinned launcher")
    manifest = replace(
        _manifest(),
        rlm_archive_sha256=_sha256(archive.read_bytes()),
        rlm_uv_binary_sha256=_sha256(uv.read_bytes()),
        rlm_uv_cache_archive_sha256=_sha256(cache.read_bytes()),
        rlm_executable_sha256=_sha256(launcher.read_bytes()),
        uv_lock_sha256=_sha256((tree / "uv.lock").read_bytes()),
    )
    bundle = StageDRLMInstallBundle(archive, uv, cache, launcher)
    harness = SimpleNamespace(
        id="rlm",
        checkout_archive_path=str(archive),
        checkout_archive_sha256=manifest.rlm_archive_sha256,
        checkout_uv_path=str(uv),
        checkout_uv_sha256=manifest.rlm_uv_binary_sha256,
        checkout_cache_archive_path=str(cache),
        checkout_cache_archive_sha256=manifest.rlm_uv_cache_archive_sha256,
        checkout_uv_lock_sha256=manifest.uv_lock_sha256,
        checkout_launcher_path=str(launcher),
        checkout_launcher_sha256=manifest.rlm_executable_sha256,
    )
    return manifest, bundle, harness


def test_exact_frozen_rlm_bundle_is_bound_to_resolved_harness(tmp_path) -> None:
    manifest, bundle, harness = _fixture(tmp_path)
    verify_stage_d_rlm_harness(harness, manifest=manifest, bundle=bundle)


def test_vanilla_or_partially_bound_rlm_harness_is_rejected(tmp_path) -> None:
    manifest, bundle, harness = _fixture(tmp_path)
    harness.checkout_archive_path = None
    with pytest.raises(ValueError, match="not bound to the frozen install bundle"):
        verify_stage_d_rlm_harness(harness, manifest=manifest, bundle=bundle)


def test_archive_without_spawn_provenance_is_rejected(tmp_path) -> None:
    manifest, bundle, harness = _fixture(tmp_path)
    tree = tmp_path / "unpatched"
    (tree / "src" / "rlm").mkdir(parents=True)
    (tree / "src" / "rlm" / "provenance.py").write_bytes(b"# vanilla\n")
    (tree / "src" / "rlm" / "session.py").write_bytes(b"# vanilla\n")
    (tree / "src" / "rlm" / "cli.py").write_bytes(b"# vanilla\n")
    (tree / "uv.lock").write_bytes(b"version = 1\n")
    archive = tmp_path / "unpatched.tar"
    write_canonical_tree_tar(tree, archive)
    changed_bundle = replace(bundle, archive_path=archive)
    changed_manifest = replace(manifest, rlm_archive_sha256=_sha256(archive.read_bytes()))
    harness.checkout_archive_path = str(archive)
    harness.checkout_archive_sha256 = changed_manifest.rlm_archive_sha256
    with pytest.raises(ValueError, match="lacks patched spawn provenance"):
        verify_stage_d_rlm_harness(
            harness,
            manifest=changed_manifest,
            bundle=changed_bundle,
        )
