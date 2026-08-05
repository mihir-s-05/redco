"""Fail-closed Phase-A artifact publication using shared draft machinery."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_v13_draft import sha256_bytes
from redco.analysis.stage_d_v13_draft_publication import (
    atomic_write,
    canonical_json_payload,
    validate_output_paths,
)


def validate_phase_a_paths(
    root: Path,
    output_paths: tuple[str, ...],
    immutable_paths: Mapping[str, str],
) -> None:
    validate_output_paths(root, immutable_paths, output_paths=output_paths)


def write_phase_a_outputs(
    root: Path,
    payloads: Mapping[str, bytes],
    *,
    immutable_paths: Mapping[str, str],
) -> dict[str, str]:
    output_paths = tuple(payloads)
    validate_phase_a_paths(root, output_paths, immutable_paths)
    for relative, payload in payloads.items():
        parsed: Any
        if relative.endswith(".json"):
            import json

            parsed = json.loads(payload)
            if not isinstance(parsed, dict) or payload != canonical_json_payload(parsed):
                raise ValueError(f"Phase-A JSON is not canonical: {relative}")
        atomic_write(root, relative, payload, output_paths=output_paths)
    return {relative: sha256_bytes(payload) for relative, payload in payloads.items()}


__all__ = ["validate_phase_a_paths", "write_phase_a_outputs"]
