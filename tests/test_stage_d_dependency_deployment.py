from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from redco.analysis.stage_d_dependency_stack import canonical_tree_manifest_bytes
from scripts.verify_stage_d_dependency_deployment import _gitlinks, _tree_sha256
from scripts.verify_stage_d_dependency_deployment import _require_clean as require_clean


def test_tree_hash_excludes_nested_gitlink_roots(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "owned.txt").write_text("owned", encoding="utf-8")
    child = root / "deps/child"
    child.mkdir(parents=True)
    (child / "nested.txt").write_text("first", encoding="utf-8")
    first = _tree_sha256(root, ("deps/child",))
    (child / "nested.txt").write_text("second", encoding="utf-8")
    assert _tree_sha256(root, ("deps/child",)) == first
    assert _tree_sha256(root) != first


def test_tree_exclusion_does_not_match_similar_prefix(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "deps/child").mkdir(parents=True)
    (root / "deps/childish").mkdir()
    (root / "deps/child/ignored").write_text("ignored", encoding="utf-8")
    included = root / "deps/childish/included"
    included.write_text("first", encoding="utf-8")
    first = _tree_sha256(root, ("deps/child",))
    included.write_text("second", encoding="utf-8")
    assert _tree_sha256(root, ("deps/child",)) != first


def test_tree_hash_includes_safe_relative_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    first_target = root / "first"
    second_target = root / "second"
    first_target.mkdir(parents=True)
    second_target.mkdir()
    link = root / "link"
    os.symlink("first", link, target_is_directory=True)
    first = _tree_sha256(root)
    link.unlink()
    os.symlink("second", link, target_is_directory=True)
    assert _tree_sha256(root) != first


@pytest.mark.parametrize(
    "value",
    ("", "/absolute", "../escape", "a/../b", "./a", "a\\b"),
)
def test_tree_exclusion_rejects_unsafe_roots(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match="exclusion root is unsafe"):
        canonical_tree_manifest_bytes(tmp_path, excluded_roots=(value,))


def test_gitlinks_reads_exact_index_map(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "update-index",
            "--add",
            "--cacheinfo",
            "160000,0123456789012345678901234567890123456789,deps/child",
        ],
        cwd=root,
        check=True,
    )
    assert _gitlinks(root) == {
        "deps/child": "0123456789012345678901234567890123456789"
    }


def test_pristine_dependency_rejects_untracked_and_ignored_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text("ignored\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "init",
        ],
        cwd=root,
        check=True,
    )
    require_clean(root, "fixture")
    (root / "ignored").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not pristine"):
        require_clean(root, "fixture")


def test_canonical_tree_hash_is_stable(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a").write_bytes(b"a")
    first = _tree_sha256(root)
    payload = {
        "path": "a",
        "type": "file",
        "mode": "0644",
        "size": 1,
        "sha256": hashlib.sha256(b"a").hexdigest(),
    }
    expected = hashlib.sha256(
        json.dumps(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-canonical-tree-v1",
                "entries": [payload],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert first == expected
