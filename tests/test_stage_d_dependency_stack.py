from __future__ import annotations

import os
from pathlib import Path

import pytest

from redco.analysis.stage_d_dependency_stack import (
    ComponentBinding,
    ImportedModuleBinding,
    PatchBinding,
    StageDDependencyStackManifest,
    canonical_tree_manifest_bytes,
    write_canonical_tree_tar,
    write_canonical_tree_tar_gzip,
)


def _patches(*names: str) -> tuple[PatchBinding, ...]:
    return tuple(PatchBinding(name, f"{index:x}" * 64) for index, name in enumerate(names, 1))


def _manifest() -> StageDDependencyStackManifest:
    components = (
        ComponentBinding(
            "prime-rl",
            "1" * 40,
            _patches(
                "prime-rl-redco-stage-c9-practical-efficiency.patch",
                "prime-rl-stage-d-live-update-gate-v1.patch",
                "prime-rl-stage-d-objective-gate-v1.patch",
                "prime-rl-strict-tool-env-guard.patch",
            ),
            "a" * 64,
        ),
        ComponentBinding(
            "renderers",
            "2" * 40,
            _patches(
                "renderers-stage-d-prepared-observer-v1.patch",
                "renderers-stage-d-replay-directives-v1.patch",
            ),
            "b" * 64,
        ),
        ComponentBinding(
            "verifiers",
            "3" * 40,
            _patches(
                "verifiers-stage-d-provenance-baseline-v1.patch",
                "verifiers-stage-d-prepared-observer-v1.patch",
                "verifiers-stage-d-replay-directives-v1.patch",
                "verifiers-stage-d-sampling-director-v1.patch",
                "verifiers-stage-d-isolated-docker-v1.patch",
                "verifiers-stage-d-pre-generation-preflight-v1.patch",
                "verifiers-stage-d-patched-rlm-archive-v1.patch",
                "verifiers-stage-d-frozen-rlm-install-v1.patch",
                "verifiers-stage-d-observer-failfast-v1.patch",
            ),
            "c" * 64,
        ),
        ComponentBinding(
            "rlm",
            "4" * 40,
            _patches(
                "rlm-event-replay-provenance.patch",
                "rlm-mcp-client-symbol-compat.patch",
                "rlm-root-initial-required-tool-choice.patch",
                "rlm-spawn-provenance-v2.patch",
            ),
            "d" * 64,
        ),
    )
    return StageDDependencyStackManifest(
        redco_commit="5" * 40,
        redco_tree_sha256="e" * 64,
        components=components,
        rlm_archive_sha256="1" * 64,
        rlm_uv_binary_sha256="2" * 64,
        rlm_uv_cache_archive_sha256="3" * 64,
        rlm_executable_sha256="4" * 64,
        uv_lock_sha256="5" * 64,
        container_image="stage-d@sha256:" + "6" * 64,
        runtime_manifest_sha256="7" * 64,
        imported_modules=(
            ImportedModuleBinding("redco", "/workspace/redco/src/redco/__init__.py", "8" * 64),
        ),
        program_sha256s=(
            ("scientific_launcher", "9" * 64),
            ("trainer_entrypoint", "a" * 64),
            ("evaluator", "b" * 64),
            ("reporter", "c" * 64),
        ),
    )


def test_dependency_stack_round_trips_canonically() -> None:
    manifest = _manifest()
    assert StageDDependencyStackManifest.from_bytes(manifest.to_bytes()) == manifest


def test_dependency_stack_rejects_patch_reordering() -> None:
    first = _manifest().components[0]
    try:
        ComponentBinding(
            first.name,
            first.base_commit,
            tuple(reversed(first.patches)),
            first.post_tree_sha256,
        )
    except ValueError as error:
        assert "patch order" in str(error)
    else:
        raise AssertionError("reordered dependency patches were accepted")


def test_dependency_stack_still_parses_exact_frozen_legacy_bindings() -> None:
    root = Path(__file__).parents[1] / "configs/stage-d"
    paths = sorted(root.glob("stage-d1-dependency-stack-v*.json"))
    assert paths
    for path in paths:
        manifest = StageDDependencyStackManifest.from_bytes(path.read_bytes())
        expected_last_patch = (
            "verifiers-stage-d-observer-failfast-v1.patch"
            if path.name.startswith("stage-d1-dependency-stack-v11")
            else "verifiers-stage-d-frozen-rlm-install-v1.patch"
        )
        assert tuple(patch.name for patch in manifest.components[2].patches)[-1] == (
            expected_last_patch
        )


def test_dependency_stack_rejects_new_binding_with_legacy_verifier_sequence() -> None:
    manifest = _manifest()
    verifier = manifest.components[2]
    with pytest.raises(ValueError, match="patch order"):
        ComponentBinding(
            verifier.name,
            verifier.base_commit,
            verifier.patches[:-1],
            verifier.post_tree_sha256,
        )


def test_canonical_dependency_archive_is_byte_identical_across_builds(
    tmp_path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "src").mkdir(parents=True)
        (root / "src" / "module.py").write_bytes(b"print('stable')\n")
        (root / "uv.lock").write_bytes(b"version = 1\n")
    (first / "src" / "module.py").chmod(0o755)
    (second / "src" / "module.py").chmod(0o755)
    assert canonical_tree_manifest_bytes(first) == canonical_tree_manifest_bytes(second)
    first_tar = tmp_path / "first.tar"
    second_tar = tmp_path / "second.tar"
    assert write_canonical_tree_tar(first, first_tar) == write_canonical_tree_tar(
        second, second_tar
    )
    assert first_tar.read_bytes() == second_tar.read_bytes()
    first_gzip = tmp_path / "first.tar.gz"
    second_gzip = tmp_path / "second.tar.gz"
    assert write_canonical_tree_tar_gzip(first, first_gzip) == (
        write_canonical_tree_tar_gzip(second, second_gzip)
    )
    assert first_gzip.read_bytes() == second_gzip.read_bytes()


def test_canonical_cache_archive_allows_only_internal_relative_symlinks(
    tmp_path,
) -> None:
    if os.name == "nt":
        pytest.skip("POSIX symlink semantics are validated in pinned Linux")
    root = tmp_path / "cache"
    target = root / "archive-v0" / "payload"
    target.mkdir(parents=True)
    (target / "METADATA").write_bytes(b"frozen")
    link = root / "wheels-v6" / "package"
    link.parent.mkdir()
    link.symlink_to("../archive-v0/payload", target_is_directory=True)
    write_canonical_tree_tar(root, tmp_path / "cache.tar", allow_relative_symlinks=True)
    link.unlink()
    link.symlink_to(tmp_path, target_is_directory=True)
    try:
        write_canonical_tree_tar(root, tmp_path / "bad.tar", allow_relative_symlinks=True)
    except ValueError as error:
        assert "absolute symbolic link" in str(error)
    else:
        raise AssertionError("absolute cache symlink was accepted")
