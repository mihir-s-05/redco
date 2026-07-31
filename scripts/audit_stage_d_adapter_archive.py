"""Create a signed, tensor-level manifest for a Stage D adapter archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import struct
import tarfile
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_manifest(data: bytes) -> dict[str, Any]:
    if len(data) < 8:
        raise ValueError("safetensors payload is shorter than its header length")
    header_length = struct.unpack("<Q", data[:8])[0]
    header_end = 8 + header_length
    if header_end > len(data):
        raise ValueError("safetensors header extends beyond the payload")
    header_bytes = data[8:header_end]
    header = json.loads(header_bytes)
    tensor_data = data[header_end:]
    tensors: dict[str, Any] = {}
    spans: list[tuple[int, int, str]] = []
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        start, end = (int(value) for value in metadata["data_offsets"])
        if not 0 <= start <= end <= len(tensor_data):
            raise ValueError(f"invalid tensor byte span for {name}")
        spans.append((start, end, str(name)))
        tensors[str(name)] = {
            "dtype": str(metadata["dtype"]),
            "shape": [int(value) for value in metadata["shape"]],
            "data_offsets": [start, end],
            "byte_length": end - start,
            "data_sha256": _sha256(tensor_data[start:end]),
        }
    ordered = sorted(spans)
    if any(
        left[1] > right[0]
        for left, right in itertools.pairwise(ordered)
    ):
        raise ValueError("safetensors tensor byte spans overlap")
    return {
        "header_length": header_length,
        "header_sha256": _sha256(header_bytes),
        "tensor_data_sha256": _sha256(tensor_data),
        "tensor_count": len(tensors),
        "tensors": tensors,
    }


def audit(archive: Path) -> dict[str, Any]:
    archive_bytes = archive.read_bytes()
    members: dict[str, Any] = {}
    payloads: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as bundle:
        for member in bundle.getmembers():
            normalized = member.name.removeprefix("./")
            if normalized in {"", "."} and member.isdir():
                continue
            if not normalized or normalized in payloads:
                raise ValueError("archive has an empty or duplicate member")
            if not member.isfile():
                raise ValueError(f"archive member is not a regular file: {normalized}")
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not read archive member: {normalized}")
            data = extracted.read()
            if len(data) != member.size:
                raise ValueError(f"archive member size mismatch: {normalized}")
            payloads[normalized] = data
            members[normalized] = {
                "byte_length": len(data),
                "sha256": _sha256(data),
            }
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if set(payloads) != required:
        raise ValueError(
            f"adapter archive member set differs: {sorted(payloads)}"
        )
    config = json.loads(payloads["adapter_config.json"])
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d0-step8-adapter-archive-manifest",
            "archive": archive.as_posix(),
            "archive_byte_length": len(archive_bytes),
            "archive_sha256": _sha256(archive_bytes),
            "members": members,
            "adapter_config": config,
            "safetensors": _tensor_manifest(
                payloads["adapter_model.safetensors"]
            ),
            "passes": True,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(args.output, audit(args.archive))


if __name__ == "__main__":
    main()
