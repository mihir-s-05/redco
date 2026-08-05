"""Authenticated immutable-input and archive parsing for the v12 audit.

This module owns only read-only authentication and durable archive inspection.
It never writes to the checkout or to the extracted archive copy.
"""

from __future__ import annotations

import json
import os
import tarfile
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_v12_audit_common import (
    ARCHIVE_SHA256,
    EVIDENCE_MANIFEST_SHA256,
    FROZEN_ARCHIVE_RELATIVE,
    FROZEN_ARCHIVE_ROOT_RELATIVE,
    FROZEN_MANIFEST_RELATIVE,
    FROZEN_REPO_FILE_SHA256,
    FROZEN_REPORT_RELATIVE,
    FROZEN_RUNTIME_CODE_COMMIT,
    FROZEN_TERMINAL_REPORT_SCHEMA_VERSION,
    TERMINAL_REPORT_SHA256,
    _bounded,
    _mapping,
    sha256_file,
)


def _safe_extract(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        destination_resolved = destination.resolve()
        for member in members:
            target = (destination / member.name).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError("terminal archive contains a path traversal member")
        handle.extractall(destination, filter="data")
    roots = {member.name.split("/", 1)[0] for member in members if member.name}
    data_roots = {root for root in roots if not root.endswith("-evidence-sha256.txt")}
    if data_roots != {"stage-d1-support-v12"}:
        raise ValueError("terminal archive does not have the pinned v12 data root")
    return destination / "stage-d1-support-v12"


def _manifest_entries(manifest: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError("evidence manifest contains a malformed line")
        digest, relative = parts
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative in seen
        ):
            raise ValueError("evidence manifest contains a duplicate or invalid entry")
        seen.add(relative)
        entries.append((digest, relative))
    return entries


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON evidence at {path.name}") from error


def _receipt_records(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "ledger" / "records").glob("*.json"))
    if len(paths) != 14:
        raise ValueError("terminal ledger does not contain the pinned fourteen records")
    return [_mapping(_load_json(path), f"ledger record {path.name}") for path in paths]


def _require_path(path: Path, expected: Path, name: str) -> Path:
    resolved = path.resolve()
    if resolved != expected.resolve():
        raise ValueError(f"{name} is not the pinned immutable v12 path")
    if not resolved.is_file():
        raise ValueError(f"{name} is missing")
    return resolved


def _source_hashes(repo_root: Path) -> dict[str, str]:
    """Hash every pinned source/config file; missing files are fatal."""
    result: dict[str, str] = {}
    for relative, expected in sorted(FROZEN_REPO_FILE_SHA256.items()):
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"required immutable repository file is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"immutable repository hash differs for {relative}: {actual}")
        result[relative] = actual
    return result


def _authenticate_repo_root(repo_root: Path) -> dict[str, str]:
    expected_root = Path(__file__).resolve().parents[3]
    resolved = repo_root.resolve()
    if resolved != expected_root:
        raise ValueError("repo_root is not the authenticated Redco checkout")
    hashes = _source_hashes(resolved)
    protocol = _mapping(
        _load_json(resolved / "configs/stage-d/stage-d1-support-protocol-v12.json"), "protocol"
    )
    prereg = _mapping(
        _load_json(resolved / "configs/stage-d/stage-d1-support-preregistration-v12.json"),
        "preregistration",
    )
    genesis = _mapping(
        _load_json(resolved / "configs/stage-d/stage-d1-support-genesis-v12.json"), "genesis"
    )
    source = _mapping(
        _load_json(resolved / "configs/stage-d/stage-d1-support-source-v12.json"), "source"
    )
    if (
        protocol.get("schema_version") != 1
        or protocol.get("preregistration_sha256")
        != hashes["configs/stage-d/stage-d1-support-preregistration-v12.json"]
        or protocol.get("dependency_stack_sha256")
        != hashes["configs/stage-d/stage-d1-dependency-stack-v12.json"]
        or protocol.get("genesis_config_sha256")
        != hashes["configs/stage-d/stage-d1-support-genesis-v12.json"]
        or protocol.get("source_sha256")
        != hashes["configs/stage-d/stage-d1-support-source-v12.json"]
    ):
        raise ValueError("v12 protocol claims do not bind to the authenticated files")
    if (
        prereg.get("schema_version") != 1
        or _mapping(prereg.get("cpu_hardening"), "preregistration cpu_hardening").get(
            "bound_code_commit"
        )
        != FROZEN_RUNTIME_CODE_COMMIT
        or _mapping(
            prereg.get("deployment_authentication"), "preregistration deployment_authentication"
        ).get("code_commit")
        != FROZEN_RUNTIME_CODE_COMMIT
    ):
        raise ValueError("v12 preregistration claims are not authenticated")
    if (
        genesis.get("schema_version") != 1
        or genesis.get("domain") != "redco-stage-d1-support-genesis-v12"
        or genesis.get("preregistration_sha256")
        != hashes["configs/stage-d/stage-d1-support-preregistration-v12.json"]
        or genesis.get("source_sha256")
        != hashes["configs/stage-d/stage-d1-support-source-v12.json"]
        or genesis.get("dependency_stack_sha256")
        != hashes["configs/stage-d/stage-d1-dependency-stack-v12.json"]
    ):
        raise ValueError("v12 genesis claims are not authenticated")
    if (
        source.get("schema_version") != 1
        or source.get("redco_commit") != FROZEN_RUNTIME_CODE_COMMIT
        or source.get("collection_plan_sha256")
        != hashes["configs/stage-d/stage-d1-support-collection-plan-v11.json"]
        or source.get("support_only") is not True
    ):
        raise ValueError("v12 source claims are not authenticated")
    amendment_claims = {
        "deployment_amendment_v12_2_sha256": (
            "configs/stage-d/stage-d1-support-deployment-amendment-v12-2.json"
        ),
        "deployment_amendment_v12_3_sha256": (
            "configs/stage-d/stage-d1-support-deployment-amendment-v12-3.json"
        ),
        "deployment_amendment_v12_4_sha256": (
            "configs/stage-d/stage-d1-support-deployment-amendment-v12-4.json"
        ),
    }
    for relative in amendment_claims.values():
        amendment = _mapping(_load_json(resolved / relative), relative)
        controls = _mapping(amendment.get("controlling_artifacts"), f"{relative} controls")
        protocol_claim = controls.get("protocol_sha256", controls.get("successor_protocol_sha256"))
        if (
            amendment.get("schema_version") != 1
            or amendment.get("status")
            not in {
                "cpu_only_pending_independent_review_and_second_provisioning",
                "approved_for_one_in_place_zero_call_repair_and_final_relaunch",
                "approved_for_exact_prelaunch_ref_correction",
            }
            or protocol_claim != hashes["configs/stage-d/stage-d1-support-protocol-v12.json"]
            or controls.get("preregistration_sha256")
            != hashes["configs/stage-d/stage-d1-support-preregistration-v12.json"]
        ):
            raise ValueError(f"v12 deployment amendment claims are not authenticated: {relative}")
    eval_payload = tomllib.loads(
        (resolved / "configs/stage-d/stage-d1-support-source-eval-v12.toml").read_text(
            encoding="utf-8"
        )
    )
    eval_env = _mapping(eval_payload.get("env"), "v12 source-eval env")
    if (
        eval_env.get("source_sha256") != hashes["configs/stage-d/stage-d1-support-source-v12.json"]
        or eval_env.get("config_sha256")
        != hashes["configs/stage-d/stage-d1-support-genesis-v12.json"]
        or eval_env.get("preregistration_sha256")
        != hashes["configs/stage-d/stage-d1-support-preregistration-v12.json"]
        or eval_env.get("support_rules_sha256")
        != hashes["configs/stage-d/stage-d1-support-rules-v1.json"]
    ):
        raise ValueError("v12 source-eval claims are not authenticated")
    return hashes


def _validate_terminal_report_schema(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != FROZEN_TERMINAL_REPORT_SCHEMA_VERSION:
        raise ValueError("terminal report schema version is not the frozen v12 version")
    if report.get("status") != "terminal-incomplete-live-transport-trace-invariant":
        raise ValueError("terminal report status is not the frozen v12 terminal status")
    frozen = _mapping(report.get("frozen_inputs"), "terminal report frozen_inputs")
    if (
        frozen.get("runtime_code_commit") != FROZEN_RUNTIME_CODE_COMMIT
        or frozen.get("protocol_sha256")
        != FROZEN_REPO_FILE_SHA256["configs/stage-d/stage-d1-support-protocol-v12.json"]
        or frozen.get("preregistration_sha256")
        != FROZEN_REPO_FILE_SHA256["configs/stage-d/stage-d1-support-preregistration-v12.json"]
        or frozen.get("deployment_amendment_v12_2_sha256")
        != FROZEN_REPO_FILE_SHA256[
            "configs/stage-d/stage-d1-support-deployment-amendment-v12-2.json"
        ]
        or frozen.get("deployment_amendment_v12_3_sha256")
        != FROZEN_REPO_FILE_SHA256[
            "configs/stage-d/stage-d1-support-deployment-amendment-v12-3.json"
        ]
        or frozen.get("deployment_amendment_v12_4_sha256")
        != FROZEN_REPO_FILE_SHA256[
            "configs/stage-d/stage-d1-support-deployment-amendment-v12-4.json"
        ]
    ):
        raise ValueError("terminal report frozen input claims are not authenticated")
    evidence = _mapping(report.get("evidence"), "terminal report evidence")
    if (
        evidence.get("evidence_file_count") != 38
        or evidence.get("terminal_archive_sha256") != ARCHIVE_SHA256
        or evidence.get("evidence_manifest_sha256") != EVIDENCE_MANIFEST_SHA256
        or evidence.get("terminal_archive_path") != str(FROZEN_ARCHIVE_RELATIVE).replace("\\", "/")
        or evidence.get("evidence_manifest_path")
        != str(FROZEN_MANIFEST_RELATIVE).replace("\\", "/")
    ):
        raise ValueError("terminal report evidence claims are not authenticated")
    diagnosis = _mapping(report.get("diagnosis"), "terminal report diagnosis")
    live = _mapping(report.get("live_observation"), "terminal report live_observation")
    if (
        diagnosis.get("failure") != "captured transport message differs from the Verifiers trace"
        or diagnosis.get("failure_phase") != "source finalization after live responses"
        or diagnosis.get("ledger_status") != "poisoned"
        or diagnosis.get("ledger_reason") != "ledger records an aborted source rollout finalization"
        or live.get("model_calls") != 4
        or live.get("root_calls") != 2
        or live.get("child_calls") != 2
        or live.get("ledger_records") != 14
        or live.get("ledger_evidence_files") != 13
        or live.get("source_rollouts_committed") != 0
        or live.get("support_measurements") != 0
    ):
        raise ValueError("terminal report durable outcome claims are not authenticated")


def _authenticate_terminal_report(path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = _require_path(path, repo_root / FROZEN_REPORT_RELATIVE, "terminal report")
    if sha256_file(resolved) != TERMINAL_REPORT_SHA256:
        raise ValueError("terminal report hash differs from its frozen value")
    report = _mapping(_load_json(resolved), "terminal report")
    _validate_terminal_report_schema(report)
    return report


def _authenticate_inputs(
    archive: Path,
    evidence_manifest: Path,
    repo_root: Path,
    terminal_report: Path,
) -> tuple[Path, Path, dict[str, str], dict[str, Any]]:
    root = repo_root.resolve()
    _authenticate_repo_root(root)
    archive_path = _require_path(archive, root / FROZEN_ARCHIVE_RELATIVE, "terminal archive")
    manifest_path = _require_path(
        evidence_manifest,
        root / FROZEN_MANIFEST_RELATIVE,
        "evidence manifest",
    )
    if sha256_file(archive_path) != ARCHIVE_SHA256:
        raise ValueError("terminal archive hash differs from its frozen value")
    if sha256_file(manifest_path) != EVIDENCE_MANIFEST_SHA256:
        raise ValueError("evidence manifest hash differs from its frozen value")
    terminal = _authenticate_terminal_report(terminal_report, root)
    return archive_path, manifest_path, _source_hashes(root), terminal


def _immutable_input_paths(repo_root: Path) -> tuple[Path, ...]:
    root = repo_root.resolve()
    paths: set[Path] = {
        root / FROZEN_ARCHIVE_RELATIVE,
        root / FROZEN_MANIFEST_RELATIVE,
        root / FROZEN_REPORT_RELATIVE,
        root / FROZEN_ARCHIVE_ROOT_RELATIVE,
    }
    paths.update(root / relative for relative in FROZEN_REPO_FILE_SHA256)
    archive_root = root / FROZEN_ARCHIVE_ROOT_RELATIVE
    if archive_root.is_dir():
        paths.update(archive_root.rglob("*"))
    return tuple(sorted(paths))


def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError):
        return False


def _reject_output_alias(resolved: Path, immutable_paths: Sequence[Path]) -> None:
    immutable_files = {path.resolve() for path in immutable_paths if path.is_file()}
    immutable_directories = {path.resolve() for path in immutable_paths if path.is_dir()}
    if any(
        resolved == directory or directory in resolved.parents
        for directory in immutable_directories
    ):
        raise ValueError("audit output aliases or descends from an immutable v12 directory")
    if resolved in immutable_files:
        raise ValueError("audit output aliases an authenticated immutable v12 file")
    if resolved.exists() and any(_same_file(resolved, path) for path in immutable_files):
        raise ValueError("audit output is a hard-link alias of an immutable v12 file")


def _validate_output_path(output: Path, repo_root: Path) -> Path:
    resolved = output.resolve()
    _reject_output_alias(resolved, _immutable_input_paths(repo_root))
    return resolved


def _manifest_audit(root: Path, manifest: Path) -> dict[str, Any]:
    entries = _manifest_entries(manifest)
    if len(entries) != 38:
        raise ValueError("evidence manifest does not contain the pinned 38 entries")
    results: list[dict[str, Any]] = []
    listed_paths: set[str] = set()
    for expected, relative in entries:
        if not relative.startswith("stage-d1-support-v12/"):
            raise ValueError("evidence manifest contains a path outside the pinned archive root")
        path = root.parent / relative
        present = path.is_file()
        actual = sha256_file(path) if present else None
        results.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "present": present,
                "matches": present and actual == expected,
            }
        )
        listed_paths.add(relative)
    archive_files = {
        str(path.relative_to(root.parent)).replace("\\", "/")
        for path in root.parent.rglob("*")
        if path.is_file()
    }
    allowed_unlisted = {
        "stage-d1-support-v12/ledger/writer.lock",
        "stage-d1-support-v12-evidence-sha256.txt",
    }
    if archive_files - listed_paths - allowed_unlisted:
        raise ValueError("terminal archive contains an unmanifested durable file")
    if listed_paths - archive_files:
        raise ValueError("evidence manifest names a missing durable file")
    return {
        "listed_count": len(results),
        "matched_count": sum(item["matches"] for item in results),
        "all_match": all(item["matches"] for item in results),
        "entries": results,
    }


def _error_evidence(root: Path, abort_receipt: Mapping[str, Any]) -> dict[str, Any]:
    digest = abort_receipt.get("error_sha256")
    if not isinstance(digest, str):
        raise ValueError("finalization abort lacks error_sha256")
    path = root / "ledger" / "evidence" / digest
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError("finalization abort error evidence hash is not durable")
    error = _mapping(_load_json(path), "finalization abort error")
    if (
        error.get("schema_version") != 1
        or error.get("domain") != "redco-stage-d-source-finalization-abort-v1"
    ):
        raise ValueError("finalization abort error evidence schema is not pinned")
    message = error.get("error_message")
    if not isinstance(message, str) or not message:
        raise ValueError("finalization abort error evidence lacks a message")
    return {
        "error_sha256": digest,
        "error_type": error.get("error_type"),
        "error_message": message,
        "error_message_bounded": _bounded(message),
        "status": "pass",
    }
