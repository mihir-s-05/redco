#!/usr/bin/env python3
"""Normalize an already-proven offline uv cache and pin its uv executable."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from redco.analysis.stage_d_dependency_stack import (
    canonical_tree_manifest_bytes,
    write_canonical_tree_tar_gzip,
)
from redco.contracts import canonical_json


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    descriptor = os.open(pending, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


def main() -> None:
    args = _arguments()
    uv_bytes = args.uv.read_bytes()
    if not uv_bytes or not os.access(args.uv, os.X_OK):
        raise ValueError("pinned uv executable is absent or not executable")
    tree_manifest = canonical_tree_manifest_bytes(
        args.cache,
        allow_relative_symlinks=True,
    )
    archive_sha256 = write_canonical_tree_tar_gzip(
        args.cache,
        args.output,
        allow_relative_symlinks=True,
    )
    _atomic_write(
        args.report,
        canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-rlm-offline-cache-build-v1",
                "uv_sha256": _sha256(uv_bytes),
                "cache_tree_manifest_sha256": _sha256(tree_manifest),
                "cache_archive_sha256": archive_sha256,
                "cache_archive_size": args.output.stat().st_size,
            }
        ),
    )


if __name__ == "__main__":
    main()
