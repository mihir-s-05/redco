"""Run two isolated exact canonical loads of one retained Stage D adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from audit_stage_d_adapter_directory import audit as audit_directory

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_manifest(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "members": report["members"],
        "adapter_config": report["adapter_config"],
        "safetensors": report["safetensors"],
        "passes": report["passes"],
    }


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, mode="r:gz") as bundle:
        bundle.extractall(destination, filter="data")


def run(
    *,
    archive: Path,
    frozen_archive_manifest: Path,
    base_model: Path,
    action_cases: Path,
    root_cases: Path,
    scorer: Path,
    uv_binary: Path,
    prime_project: Path,
    output_dir: Path,
    stable_adapter_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen_manifest = json.loads(
        frozen_archive_manifest.read_text(encoding="utf-8")
    )
    verify_signed_payload(frozen_manifest)
    if _sha256(archive) != frozen_manifest["archive_sha256"]:
        raise ValueError("retained adapter archive hash differs")

    workspace = Path(
        tempfile.mkdtemp(prefix="redco-stage-d-v4-6-retention-")
    )
    entries = []
    action_paths = []
    root_paths = []
    try:
        for replicate in (1, 2):
            extraction = workspace / f"extraction-{replicate}"
            extraction.mkdir()
            _safe_extract(archive, extraction)
            directory_manifest = audit_directory(extraction)
            manifest_path = output_dir / f"extraction-{replicate}-manifest.json"
            atomic_write_json(manifest_path, directory_manifest)
            if (
                _normalized_manifest(directory_manifest)
                != {
                    "members": frozen_manifest["members"],
                    "adapter_config": frozen_manifest["adapter_config"],
                    "safetensors": frozen_manifest["safetensors"],
                    "passes": frozen_manifest["passes"],
                }
            ):
                raise ValueError("extracted adapter manifest differs")

            if stable_adapter_path.is_symlink():
                stable_adapter_path.unlink()
            elif stable_adapter_path.exists():
                raise ValueError("stable adapter path exists and is not a symlink")
            stable_adapter_path.symlink_to(extraction, target_is_directory=True)
            if stable_adapter_path.resolve() != extraction.resolve():
                raise ValueError("stable adapter symlink target differs")

            action_output = output_dir / f"canonical-action-{replicate}.json"
            root_output = output_dir / f"canonical-root-{replicate}.json"
            command = [
                str(uv_binary),
                "run",
                "--frozen",
                "--project",
                str(prime_project),
                "--extra",
                "flash-attn",
                "python",
                str(scorer),
                "--model",
                str(base_model),
                "--adapter",
                str(stable_adapter_path),
                "--adapter-name",
                "retained",
                "--action-cases",
                str(action_cases),
                "--root-cases",
                str(root_cases),
                "--action-output",
                str(action_output),
                "--root-output",
                str(root_output),
                "--device",
                "cuda:0",
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"canonical scorer replicate {replicate} failed: "
                    f"{completed.returncode}"
                )
            action = json.loads(action_output.read_text(encoding="utf-8"))
            root = json.loads(root_output.read_text(encoding="utf-8"))
            verify_signed_payload(action)
            verify_signed_payload(root)
            action_paths.append(action_output)
            root_paths.append(root_output)
            entries.append(
                {
                    "replicate": replicate,
                    "physical_extraction": extraction.resolve().as_posix(),
                    "stable_logical_adapter_path": (
                        stable_adapter_path.as_posix()
                    ),
                    "observed_symlink_target": (
                        stable_adapter_path.resolve().as_posix()
                    ),
                    "directory_manifest": manifest_path.as_posix(),
                    "directory_manifest_sha256": _sha256(manifest_path),
                    "adapter_name": "retained",
                    "process_returncode": completed.returncode,
                    "action_payload_sha256": _sha256(action_output),
                    "root_payload_sha256": _sha256(root_output),
                }
            )
            stable_adapter_path.unlink()

        action_exact = action_paths[0].read_bytes() == action_paths[1].read_bytes()
        root_exact = root_paths[0].read_bytes() == root_paths[1].read_bytes()
        report = sign_payload(
            {
                "schema_version": 1,
                "analysis": "stage-d0-v4-6-isolated-canonical-retention",
                "archive": archive.as_posix(),
                "archive_sha256": _sha256(archive),
                "frozen_archive_manifest": (
                    frozen_archive_manifest.as_posix()
                ),
                "frozen_archive_manifest_sha256": _sha256(
                    frozen_archive_manifest
                ),
                "base_model": base_model.as_posix(),
                "action_cases_sha256": _sha256(action_cases),
                "root_cases_sha256": _sha256(root_cases),
                "scorer_sha256": _sha256(scorer),
                "invocations": entries,
                "action_payloads_byte_identical": action_exact,
                "root_payloads_byte_identical": root_exact,
                "passes": action_exact and root_exact,
            }
        )
        return report
    finally:
        if stable_adapter_path.is_symlink():
            stable_adapter_path.unlink()
        shutil.rmtree(workspace, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--frozen-archive-manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--action-cases", type=Path, required=True)
    parser.add_argument("--root-cases", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--uv-binary", type=Path, required=True)
    parser.add_argument("--prime-project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stable-adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        archive=args.archive,
        frozen_archive_manifest=args.frozen_archive_manifest,
        base_model=args.base_model,
        action_cases=args.action_cases,
        root_cases=args.root_cases,
        scorer=args.scorer,
        uv_binary=args.uv_binary,
        prime_project=args.prime_project,
        output_dir=args.output_dir,
        stable_adapter_path=args.stable_adapter_path,
    )
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
