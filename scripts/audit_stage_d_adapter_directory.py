"""Create a signed tensor-level manifest for an extracted adapter directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_stage_d_adapter_archive import _tensor_manifest

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit(directory: Path) -> dict[str, Any]:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    members = {
        path.relative_to(directory).as_posix(): {
            "byte_length": len(data := path.read_bytes()),
            "sha256": _sha256(data),
        }
        for path in files
    }
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if set(members) != required:
        raise ValueError(
            f"adapter directory member set differs: {sorted(members)}"
        )
    config = json.loads(
        (directory / "adapter_config.json").read_text(encoding="utf-8")
    )
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d0-extracted-adapter-directory-manifest",
            "directory": directory.resolve().as_posix(),
            "members": members,
            "adapter_config": config,
            "safetensors": _tensor_manifest(
                (directory / "adapter_model.safetensors").read_bytes()
            ),
            "passes": True,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(args.output, audit(args.directory))


if __name__ == "__main__":
    main()
