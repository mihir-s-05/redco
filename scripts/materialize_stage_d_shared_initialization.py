#!/usr/bin/env python3
"""Materialize Stage D's shared-initialization trust root from retained bytes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from redco.analysis.stage_d_shared_initialization import (
    StageDSharedInitializationManifest,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialization-id", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--base-model-manifest", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    manifest = StageDSharedInitializationManifest.from_retained_adapter(
        initialization_id=args.initialization_id,
        checkpoint_id=args.checkpoint_id,
        base_model_manifest_path=args.base_model_manifest,
        adapter_manifest_path=args.adapter_manifest,
        adapter_path=args.adapter,
    )
    value = manifest.to_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    print(manifest.manifest_sha256)


if __name__ == "__main__":
    main()
