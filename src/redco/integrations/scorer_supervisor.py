"""Subprocess supervisor for durable, signed vLLM scorer outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    canonical_sha256,
    sign_payload,
    verify_signed_payload,
)


def _load_signed_output(
    path: Path,
    *,
    expected_cases_sha256: str,
    expected_model: str,
    expected_analysis: str | None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scorer output must be a JSON object")
    verify_signed_payload(payload)
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("scorer output has no source object")
    if source.get("cases_sha256") != expected_cases_sha256:
        raise ValueError("scorer output cases SHA-256 does not match the frozen cases")
    if source.get("model") != expected_model:
        raise ValueError("scorer output model does not match the requested model")
    if expected_analysis is not None and payload.get("analysis") != expected_analysis:
        raise ValueError("scorer output analysis kind does not match")
    return payload


def supervise(
    command: list[str],
    *,
    output: Path,
    verified: Path,
    expected_cases_sha256: str,
    expected_model: str,
    expected_analysis: str | None,
) -> dict[str, Any]:
    """Run one scorer and commit a parent-verified sentinel."""
    if not command:
        raise ValueError("a scorer command is required after --")
    if output.exists() or verified.exists():
        raise FileExistsError("refusing to overwrite scorer output or verification sentinel")

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"scorer subprocess exited with status {completed.returncode}")
    if not output.is_file():
        raise FileNotFoundError(f"scorer did not create {output}")

    payload = _load_signed_output(
        output,
        expected_cases_sha256=expected_cases_sha256,
        expected_model=expected_model,
        expected_analysis=expected_analysis,
    )
    output_bytes = output.read_bytes()
    sentinel = sign_payload(
        {
            "schema_version": 1,
            "status": "verified",
            "child_returncode": completed.returncode,
            "command_sha256": hashlib.sha256(
                json.dumps(command, separators=(",", ":")).encode()
            ).hexdigest(),
            "output_path": output.as_posix(),
            "output_file_sha256": hashlib.sha256(output_bytes).hexdigest(),
            "output_signed_payload_sha256": payload["signed_payload_sha256"],
            "expected_cases_sha256": expected_cases_sha256,
            "expected_model": expected_model,
            "expected_analysis": expected_analysis,
        }
    )
    atomic_write_json(verified, sentinel)
    return sentinel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--expected-cases-sha256", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-analysis")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    sentinel = supervise(
        command,
        output=args.output,
        verified=args.verified,
        expected_cases_sha256=args.expected_cases_sha256,
        expected_model=args.expected_model,
        expected_analysis=args.expected_analysis,
    )
    print(
        json.dumps(
            {
                "status": sentinel["status"],
                "verification_sha256": canonical_sha256(
                    {
                        key: value
                        for key, value in sentinel.items()
                        if key != "signed_payload_sha256"
                    }
                ),
            },
            sort_keys=True,
        )
    )
