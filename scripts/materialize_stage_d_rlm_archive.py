#!/usr/bin/env python3
"""Build the exact normalized patched-RLM source archive for Stage D."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

from redco.analysis.stage_d_dependency_stack import (
    canonical_tree_manifest_bytes,
    write_canonical_tree_tar,
)
from redco.contracts import canonical_json

_BASE_COMMIT = "56218f33796ecbe465445bc43948886354fde196"
_PATCHES = (
    "rlm-event-replay-provenance.patch",
    "rlm-mcp-client-symbol-compat.patch",
    "rlm-root-initial-required-tool-choice.patch",
    "rlm-spawn-provenance-v2.patch",
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _run(*command: str, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True, capture_output=True)


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    descriptor = os.open(pending, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    patches = tuple(args.patch_root / name for name in _PATCHES)
    if any(not patch.is_file() for patch in patches):
        raise FileNotFoundError("the frozen RLM patch stack is incomplete")
    with tempfile.TemporaryDirectory(prefix="redco-stage-d-rlm-") as temporary:
        checkout = Path(temporary) / "rlm"
        _run(
            "git",
            "clone",
            "--quiet",
            "--no-hardlinks",
            "--no-checkout",
            str(args.source_repo),
            str(checkout),
        )
        _run("git", "checkout", "--quiet", "--detach", _BASE_COMMIT, cwd=checkout)
        for patch in patches:
            _run("git", "apply", "--check", str(patch), cwd=checkout)
            _run("git", "apply", str(patch), cwd=checkout)
        tree_manifest = canonical_tree_manifest_bytes(checkout)
        archive_sha256 = write_canonical_tree_tar(checkout, args.output)
    report = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-patched-rlm-archive-build-v1",
            "base_commit": _BASE_COMMIT,
            "patches": [
                {"name": patch.name, "sha256": _sha256(patch.read_bytes())}
                for patch in patches
            ],
            "post_tree_manifest_sha256": _sha256(tree_manifest),
            "uv_lock_sha256": _sha256(
                _read_archive_member(args.output, "uv.lock")
            ),
            "archive_sha256": archive_sha256,
        }
    )
    _atomic_write(args.report, report)


def _read_archive_member(path: Path, name: str) -> bytes:
    import tarfile

    with tarfile.open(path, mode="r") as archive:
        member = archive.getmember(name)
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f"archive member is not a regular file: {name}")
        return stream.read()


if __name__ == "__main__":
    main()
