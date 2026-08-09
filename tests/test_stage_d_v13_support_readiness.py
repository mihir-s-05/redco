from __future__ import annotations

import inspect
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from redco.analysis import stage_d_v13_support_readiness as readiness
from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes

ROOT = Path(__file__).parents[1].resolve()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "--all")
    _git(
        root,
        "-c",
        "user.name=Redco Test",
        "-c",
        "user.email=test@redco.local",
        "commit",
        "-m",
        message,
    )
    return _git(root, "rev-parse", "HEAD")


def _synthetic_readiness_repo(tmp_path: Path) -> tuple[Path, str]:
    target = tmp_path / "repo"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--shared",
            "--no-checkout",
            str(ROOT),
            str(target),
        ],
        check=True,
    )
    _git(target, "config", "core.autocrlf", "true")
    _git(target, "checkout", "--quiet", readiness.READINESS_PARENT_COMMIT)
    for relative in readiness.READINESS_PATHS:
        source = ROOT / relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return target, _commit(target, "readiness repair")


def test_readiness_build_is_canonical_non_authorizing_and_deterministic() -> None:
    first = readiness.build_readiness_artifacts(ROOT)
    second = readiness.build_readiness_artifacts(ROOT)
    assert first == second
    assert set(first) == {
        readiness.DEPENDENCY_MANIFEST_RELATIVE,
        readiness.READINESS_MANIFEST_RELATIVE,
        readiness.READINESS_AUDIT_RELATIVE,
    }
    manifest = json.loads(first[readiness.READINESS_MANIFEST_RELATIVE])
    assert canonical_json_bytes(manifest) == first[readiness.READINESS_MANIFEST_RELATIVE]
    assert manifest["parent"] == {
        "commit": readiness.READINESS_PARENT_COMMIT,
        "tree": readiness.READINESS_PARENT_TREE,
    }
    assert manifest["future_authorization"]["present"] is False
    assert manifest["artifact_root"] == {
        "platform": "windows",
        "fixed_path": readiness.FIXED_LOCAL_ARTIFACT_ROOT,
        "caller_override_allowed": False,
        "required_entries": manifest["artifact_root"]["required_entries"],
        "max_transfer_bytes": readiness.MAX_TRANSFER_BYTES,
        "max_archive_extracted_bytes": readiness.MAX_ARCHIVE_EXTRACTED_BYTES,
        "max_archive_members": readiness.MAX_ARCHIVE_MEMBERS,
        "minimum_post_transfer_free_bytes": readiness.MIN_POST_TRANSFER_FREE_BYTES,
    }
    assert not any(manifest["authorization"].values())
    assert manifest["stop_rules"]["support_success_authorizes_science"] is False


def test_successor_dependency_manifest_binds_reviewed_live_stack() -> None:
    payload = json.loads(readiness.build_dependency_manifest(ROOT))
    components = {
        component["name"]: component
        for component in payload["live_owner_stack"]["components"]
    }
    assert components["renderers"]["post_tree_sha256"] == (
        "bd43d515c12dcaa1e1c0279941a1397d4ffba31a1557d6d7342a1322b195fcc4"
    )
    assert components["verifiers"]["post_tree_sha256"] == (
        "9dcf9e98dea73c2487d2165cd6cae35dc61fb66e00d377d85d5466886b3ea4e0"
    )
    assert components["renderers"]["patches"][-1]["name"] == (
        "renderers-stage-d-watchdog-owner-v1.patch"
    )
    assert components["verifiers"]["patches"][-1]["name"] == (
        "verifiers-stage-d-watchdog-owner-v1.patch"
    )
    assert payload["uv_lock"]["sha256"] == readiness.CURRENT_UV_LOCK_SHA256
    assert payload["runtime"]["python_version"] == "3.12.3"
    assert payload["abi"]["split_engine_sampling"] is True
    assert payload["abi"]["sampling_contract_sha256"] == (
        "819222244a81565a67331826be3dd362e14e1481043d60fccb569551a4471f6d"
    )
    assert set(payload["owner_bindings"]) == set(readiness.SUPPORT_OWNER_PATHS)


def test_historical_v1_and_v12_are_immutable_and_stale_for_this_chain() -> None:
    assert sha256_bytes((ROOT / readiness.HISTORICAL_V1_AUTH_RELATIVE).read_bytes()) == (
        readiness.HISTORICAL_V1_AUTH_SHA256
    )
    assert sha256_bytes(
        (ROOT / readiness.HISTORICAL_V12_DEPENDENCY_RELATIVE).read_bytes()
    ) == readiness.HISTORICAL_V12_DEPENDENCY_SHA256
    manifest = json.loads(
        readiness.build_readiness_artifacts(ROOT)[readiness.READINESS_MANIFEST_RELATIVE]
    )
    assert manifest["historical_v1"]["ancestry_compatible"] is False
    with pytest.raises(ValueError, match=r"direct child|parent|b0a416"):
        from redco.analysis import stage_d_v13_support_launch as historical

        historical._authenticate_parent(ROOT, require_post_commit=True)


def test_fixed_artifact_root_validates_exact_files_archives_and_budget(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    plain = root / "model.bin"
    plain.write_bytes(b"model")
    archive = root / "cache.tar"
    member = tmp_path / "member.txt"
    member.write_bytes(b"cache")
    with tarfile.open(archive, "w") as handle:
        handle.add(member, arcname="cache/member.txt")
    entries: list[dict[str, object]] = [
        {
            "name": "model",
            "relative_path": "model.bin",
            "sha256": sha256_bytes(plain.read_bytes()),
            "expected_bytes": plain.stat().st_size,
            "kind": "regular_file",
        },
        {
            "name": "cache",
            "relative_path": "cache.tar",
            "sha256": sha256_bytes(archive.read_bytes()),
            "expected_bytes": archive.stat().st_size,
            "kind": "tar_archive",
        },
    ]
    result = readiness._validate_artifact_root(
        root, entries, disk_free_bytes=readiness.MIN_POST_TRANSFER_FREE_BYTES + 1_000_000
    )
    assert result["files"] == 2
    plain.write_bytes(b"changed")
    with pytest.raises(readiness.ReadinessBlocked, match="hash differs"):
        readiness._validate_artifact_root(
            root, entries, disk_free_bytes=readiness.MIN_POST_TRANSFER_FREE_BYTES + 1_000_000
        )


def test_artifact_root_rejects_missing_links_hardlinks_and_unsafe_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(readiness.ReadinessBlocked, match="absent or linked"):
        readiness._validate_artifact_root(tmp_path / "missing", [])
    root = tmp_path / "root"
    root.mkdir()
    source = root / "source"
    alias = root / "alias"
    source.write_bytes(b"same")
    os.link(source, alias)
    entry = {
        "name": "alias",
        "relative_path": "alias",
        "sha256": sha256_bytes(alias.read_bytes()),
        "expected_bytes": 4,
        "kind": "regular_file",
    }
    with pytest.raises(readiness.ReadinessBlocked, match="hard-link"):
        readiness._validate_artifact_root(root, [entry], disk_free_bytes=10**10)
    monkeypatch.setattr(readiness, "_is_link_or_reparse", lambda path: path == root)
    with pytest.raises(readiness.ReadinessBlocked, match="absent or linked"):
        readiness._validate_artifact_root(root, [])


def _tree_snapshot(root: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        sorted(
            (
                path.relative_to(root).as_posix(),
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in root.rglob("*")
        )
    )


def test_archive_rejects_every_special_member_before_writes(tmp_path: Path) -> None:
    special_types = (
        tarfile.FIFOTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        b"s",
        b"Z",
    )
    for index, member_type in enumerate(special_types):
        archive = tmp_path / f"special-{index}.tar"
        with tarfile.open(archive, "w") as handle:
            member = tarfile.TarInfo(f"special-{index}")
            member.type = member_type
            member.size = 0
            handle.addfile(member)
        before = _tree_snapshot(tmp_path)
        with pytest.raises(readiness.ReadinessBlocked, match="member is unsafe"):
            readiness._validate_archive(archive)
        assert _tree_snapshot(tmp_path) == before


def test_archive_rejects_expanded_size_and_member_count_before_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "bounded.tar"
    with tarfile.open(archive, "w") as handle:
        for name, value in (("one", b"ab"), ("two", b"cd")):
            member = tarfile.TarInfo(name)
            member.size = len(value)
            handle.addfile(member, io.BytesIO(value))
    before = _tree_snapshot(tmp_path)
    monkeypatch.setattr(readiness, "MAX_ARCHIVE_EXTRACTED_BYTES", 3)
    with pytest.raises(readiness.ReadinessBlocked, match="extracted size"):
        readiness._validate_archive(archive)
    assert _tree_snapshot(tmp_path) == before
    monkeypatch.setattr(readiness, "MAX_ARCHIVE_EXTRACTED_BYTES", 4)
    monkeypatch.setattr(readiness, "MAX_ARCHIVE_MEMBERS", 1)
    with pytest.raises(readiness.ReadinessBlocked, match="too many members"):
        readiness._validate_archive(archive)
    assert _tree_snapshot(tmp_path) == before


def test_archive_rejects_duplicate_normalized_paths_before_writes(tmp_path: Path) -> None:
    cases = (
        (("duplicate", tarfile.REGTYPE), ("duplicate", tarfile.REGTYPE)),
        (("kind-collision", tarfile.REGTYPE), ("kind-collision/", tarfile.DIRTYPE)),
        (("alias/path", tarfile.REGTYPE), ("alias//path", tarfile.REGTYPE)),
        (("dot/path", tarfile.REGTYPE), ("dot/./path", tarfile.REGTYPE)),
        (("parent", tarfile.REGTYPE), ("parent/child", tarfile.REGTYPE)),
        (("descendant/child", tarfile.REGTYPE), ("descendant", tarfile.REGTYPE)),
    )
    for index, members in enumerate(cases):
        archive = tmp_path / f"duplicate-{index}.tar"
        with tarfile.open(archive, "w") as handle:
            for name, member_type in members:
                member = tarfile.TarInfo(name)
                member.type = member_type
                if member_type == tarfile.REGTYPE:
                    member.size = 1
                    handle.addfile(member, io.BytesIO(b"x"))
                else:
                    handle.addfile(member)
        before = _tree_snapshot(tmp_path)
        with pytest.raises(
            readiness.ReadinessBlocked,
            match=r"duplicate normalized|collides",
        ):
            readiness._validate_archive(archive)
        assert _tree_snapshot(tmp_path) == before


def test_public_readiness_authority_has_no_caller_overrides() -> None:
    assert not inspect.signature(readiness.validate_local_artifacts).parameters
    assert not inspect.signature(readiness.validate_future_prime_observation).parameters
    assert not inspect.signature(readiness.validate_future_support_authorization).parameters
    with pytest.raises(readiness.ReadinessBlocked):
        readiness.validate_local_artifacts()
    with pytest.raises((FileNotFoundError, ValueError)):
        readiness.validate_future_support_authorization()


def test_readiness_commit_and_future_authorization_are_exact_direct_children(
    tmp_path: Path,
) -> None:
    repo, repair = _synthetic_readiness_repo(tmp_path)
    readiness.authenticate_readiness_commit(repo, repair)
    authorization = readiness._future_authorization_payload(repo, repair)
    path = repo / readiness.FUTURE_AUTH_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(authorization))
    authorization_commit = _commit(repo, "support-only authorization")
    assert _git(repo, "rev-parse", f"{authorization_commit}^") == repair
    assert readiness._validate_future_authorization(repo) == authorization


def test_future_authorization_rejects_dirty_or_staged_caller_state(
    tmp_path: Path,
) -> None:
    for dirty_mode in ("untracked", "staged"):
        repo, repair = _synthetic_readiness_repo(tmp_path / dirty_mode)
        path = repo / readiness.FUTURE_AUTH_RELATIVE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            canonical_json_bytes(readiness._future_authorization_payload(repo, repair))
        )
        _commit(repo, "authorization")
        extra = repo / "caller-authority.txt"
        extra.write_text("forged", encoding="utf-8")
        if dirty_mode == "staged":
            _git(repo, "add", "caller-authority.txt")
        with pytest.raises(ValueError, match="clean superproject"):
            readiness._validate_future_authorization(repo)


def test_readiness_commit_rejects_wrong_parent_extra_path_and_merge(tmp_path: Path) -> None:
    repo, repair = _synthetic_readiness_repo(tmp_path)
    extra = repo / "extra.txt"
    extra.write_text("extra", encoding="utf-8")
    bad = _commit(repo, "extra successor")
    with pytest.raises(ValueError, match="direct child"):
        readiness.authenticate_readiness_commit(repo, bad)
    _git(repo, "checkout", "--quiet", "-b", "side", readiness.READINESS_PARENT_COMMIT)
    side = repo / "side.txt"
    side.write_text("side", encoding="utf-8")
    _commit(repo, "side")
    _git(repo, "checkout", "--quiet", "--detach", repair)
    _git(repo, "merge", "--no-ff", "-m", "merge", "side")
    merge = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="single-parent"):
        readiness.authenticate_readiness_commit(repo, merge)


def test_future_preflight_contract_is_strictly_non_provisioning() -> None:
    manifest = json.loads(
        readiness.build_readiness_artifacts(ROOT)[readiness.READINESS_MANIFEST_RELATIVE]
    )
    preflight = manifest["future_preflight"]
    assert preflight == {
        "prime_cli_version": "0.6.20",
        "fixed_observation_path": readiness.FIXED_PRIME_OBSERVATION_RELATIVE,
        "ttl_seconds": 900,
        "raw_stdout_stderr_hashes_required": True,
        "wallet_minimum_usd": 30,
        "pods_required": 0,
        "disks_required": 0,
        "qualifying_resources_required": 1,
        "resource": "non-spot 2x48GB L40/L40S/RTX6000Ada",
        "maximum_hourly_rate_usd": 2,
    }
    assert manifest["authorization"]["prime_authorized"] is False
    assert manifest["authorization"]["science_authorized"] is False
