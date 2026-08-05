"""Candidate-null Foundation F tree allowlist and canonical manifest."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes
from redco.analysis.stage_d_v13_source_phase_a import (
    FOUNDATION_NULL_CANDIDATE,
    SOURCE_ARTIFACT_RELATIVE,
    SOURCE_BYTES,
    SOURCE_LOGICAL_URL,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    SOURCE_ROW_COUNT,
    SOURCE_SCHEMA_SHA256,
    SOURCE_SHA256,
    foundation_envelope,
    validate_foundation_envelope,
)
from redco.analysis.stage_d_v13_source_phase_a_decoder import git_blob_sha1, hardened_git

FOUNDATION_PARENT_COMMIT = "c41fd18446cecf1c7c98e5aa3a962d1568072c1b"
PRIME_RL_RELATIVE = "external/prime-rl"
PRIME_RL_GITLINK_MODE = "160000"
PRIME_RL_GITLINK_OBJECT = "3b22dd951cad1036d1fe8dd0a0bfc40807a9b360"
FOUNDATION_MANIFEST_RELATIVE = (
    "reports/stage-d1-support-v13-foundation-tree-manifest-f1.json"
)
SOURCE_PROVENANCE_RELATIVE = (
    "datasets/stage-d/source-auth-v13/qasper-train-06806e4608976fc2fac0a090ac425d5b2b29caf4-"
    "provenance-v1.json"
)
APPROVAL_ANCHOR_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-source-phase-a-approval-anchor-v1.json"
)
BINDINGS_RELATIVE = "src/redco/analysis/stage_d_v13_source_phase_a_bindings.py"

# This is the exact proposed F staging boundary.  The manifest intentionally
# omits itself from file bindings to avoid self-hash circularity.
FOUNDATION_ALLOWLIST: tuple[str, ...] = (
    "configs/stage-d/v13-draft/stage-d1-support-source-authentication-phase-a-v1.json",
    "configs/stage-d/v13-draft/stage-d1-support-source-phase-a-approval-anchor-v1.json",
    SOURCE_PROVENANCE_RELATIVE,
    SOURCE_ARTIFACT_RELATIVE,
    FOUNDATION_MANIFEST_RELATIVE,
    "reports/stage-d1-support-v13-source-phase-a-artifact-manifest-v1.json",
    "reports/stage-d1-support-v13-source-phase-a-audit-v1.json",
    "reports/stage-d1-support-v13-source-phase-a-cpu-manifest-v1.json",
    "reports/stage-d1-support-v13-source-phase-a-status-v2.json",
    "scripts/build_stage_d_qasper_extension_v1.py",
    "scripts/build_stage_d_v13_foundation_manifest.py",
    "scripts/build_stage_d_v13_source_phase_a.py",
    "src/redco/analysis/stage_d_collection.py",
    "src/redco/analysis/stage_d_v13_draft.py",
    "src/redco/analysis/stage_d_v13_draft_inputs.py",
    "src/redco/analysis/stage_d_v13_draft_publication.py",
    "src/redco/analysis/stage_d_v13_foundation.py",
    "src/redco/analysis/stage_d_v13_source_phase_a.py",
    "src/redco/analysis/stage_d_v13_source_phase_a_bindings.py",
    "src/redco/analysis/stage_d_v13_source_phase_a_decoder.py",
    "src/redco/analysis/stage_d_v13_source_phase_a_publication.py",
    "src/redco/analysis/stage_d_v13_source_phase_a_selector.py",
    "src/redco/analysis/stage_d_v13_source_phase_a_trust.py",
    "src/redco/analysis/stage_d_v13_source_phase_a_witness.py",
    "tests/test_stage_d_qasper_partition_v4.py",
    "tests/test_stage_d_support_successor.py",
    "tests/test_stage_d_v13_draft.py",
    "tests/test_stage_d_v13_source_phase_a.py",
    "pyproject.toml",
    "uv.lock",
)

FOUNDATION_EXCLUDED_CLASSES = (
    "candidate_and_post_179_evidence",
    "phase_b_authorization_c",
    "launch_hardware_wallet_material",
    "cache_temp_state",
    "external_prime_rl_submodule",
)


def build_source_provenance() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        foundation_envelope(
            {
                "schema_version": 1,
                "domain": "redco-stage-d1-support-v13-qasper-source-provenance-v1",
                "source": {
                    "repository": SOURCE_REPOSITORY,
                    "logical_url": SOURCE_LOGICAL_URL,
                    "revision": SOURCE_REVISION,
                    "path": "qasper/train/0000.parquet",
                    "bytes": SOURCE_BYTES,
                    "sha256": SOURCE_SHA256,
                    "schema_sha256": SOURCE_SCHEMA_SHA256,
                    "row_count": SOURCE_ROW_COUNT,
                    "license": "cc-by-4.0",
                    "attribution": "allenai/qasper",
                },
                "policy": {
                    "full_file_authenticated": True,
                    "metadata_only_for_row_count_and_schema": True,
                    "phase_a_cutoff": 179,
                    "post_cutoff_rows_read": False,
                },
            },
        ),
    )


def build_integrity_anchor(root: Path) -> dict[str, Any]:
    """Materialize the F-only registry witness; it grants no authority."""

    from redco.analysis.stage_d_v13_source_phase_a_trust import _policy_projection

    registry = root / BINDINGS_RELATIVE
    if not registry.is_file():
        raise FileNotFoundError(BINDINGS_RELATIVE)
    policy = _policy_projection()
    return cast(
        dict[str, Any],
        foundation_envelope(
            {
                "schema_version": 1,
                "domain": "redco-stage-d1-support-v13-source-phase-a-approval-v1",
                "registry": {
                    "path": BINDINGS_RELATIVE,
                    "sha256": sha256_bytes(registry.read_bytes()),
                },
                "policy": policy,
                "trust_root": {
                    "kind": "future_reviewed_git_commit",
                    "status": "uncommitted_integrity_witness_only",
                    "externally_authorized": False,
                    "review_required_before_freeze": True,
                    "approval_record": (
                        "A future reviewed Git commit or canonical approval record must bind "
                        "these exact bytes before any Phase-B or launch authorization."
                    ),
                },
            },
        ),
    )


def _git_head(root: Path) -> str:
    result = hardened_git(root, "rev-parse", "HEAD", text=True)
    if result.returncode != 0 or not isinstance(result.stdout, str):
        raise ValueError("Foundation F HEAD authentication failed")
    return result.stdout.strip()


def _validate_prime_rl_gitlink(root: Path) -> None:
    result = hardened_git(root, "ls-files", "--stage", "--", PRIME_RL_RELATIVE, text=True)
    if result.returncode != 0 or not isinstance(result.stdout, str):
        raise ValueError("external/prime-rl gitlink index authentication failed")
    lines = result.stdout.splitlines()
    if len(lines) != 1 or lines[0].split() != [
        PRIME_RL_GITLINK_MODE,
        PRIME_RL_GITLINK_OBJECT,
        "0",
        PRIME_RL_RELATIVE,
    ]:
        raise ValueError("external/prime-rl gitlink mode/object differs from the fixed witness")


def _porcelain_records(raw: bytes) -> list[tuple[str, list[str]]]:
    """Parse porcelain-v1 -z records without path quoting or prefix guesses."""

    parts = raw.split(b"\0")
    records: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(parts) and parts[index]:
        record = parts[index]
        if len(record) < 4 or record[2:3] != b" ":
            raise ValueError("malformed Git porcelain status record")
        status = record[:2].decode("ascii", errors="strict")
        paths = [os.fsdecode(record[3:])]
        index += 1
        if "R" in status or "C" in status:
            if index >= len(parts) or not parts[index]:
                raise ValueError("Git rename/copy status record has no source path")
            paths.append(os.fsdecode(parts[index]))
            index += 1
        records.append((status, paths))
    return records


def _status_paths(root: Path) -> set[str]:
    result = hardened_git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if result.returncode != 0 or not isinstance(result.stdout, bytes):
        raise ValueError("Foundation F status authentication failed")
    paths: set[str] = set()
    prime_witness_seen = False
    for status, record_paths in _porcelain_records(result.stdout):
        normalized = [path.replace("\\", "/") for path in record_paths]
        if normalized == [PRIME_RL_RELATIVE] and status == " M":
            if prime_witness_seen:
                raise ValueError("external/prime-rl dirty witness is duplicated")
            _validate_prime_rl_gitlink(root)
            prime_witness_seen = True
            continue
        paths.update(normalized)
    unexpected = sorted(paths.difference(FOUNDATION_ALLOWLIST))
    if unexpected:
        raise ValueError(
            "Foundation F tree has dirty or untracked paths outside the exact allowlist: "
            + ", ".join(unexpected)
        )
    if not prime_witness_seen:
        raise ValueError(
            "Foundation F requires exactly one authenticated external/prime-rl dirty witness"
        )
    return paths


def _file_entry(root: Path, relative: str) -> dict[str, str | int]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Foundation F allowlist input is missing: {relative}")
    data = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "git_blob_sha1": git_blob_sha1(data),
    }


def _validate_provenance(root: Path) -> dict[str, Any]:
    path = root / SOURCE_PROVENANCE_RELATIVE
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ValueError("QASPER source provenance is not canonical")
    validate_foundation_envelope(value)
    expected = build_source_provenance()
    if value != expected:
        raise ValueError("QASPER source provenance differs from the pinned source")
    return value


def build_foundation_manifest(root: Path) -> dict[str, Any]:
    """Build the deterministic F tree witness without binding its own bytes."""

    head = _git_head(root)
    if head != FOUNDATION_PARENT_COMMIT:
        raise ValueError(f"Foundation F parent must be {FOUNDATION_PARENT_COMMIT}, got {head}")
    allowlist = tuple(sorted(FOUNDATION_ALLOWLIST))
    # The shared checkout contains unrelated historical artifacts.  They are
    # deliberately neither authenticated nor staged by this exact allowlist;
    # only an allowlisted path can enter ``files`` below.
    _status_paths(root)
    provenance = _validate_provenance(root)
    entries = [
        _file_entry(root, relative)
        for relative in allowlist
        if relative != FOUNDATION_MANIFEST_RELATIVE
    ]
    manifest = cast(
        dict[str, Any],
        foundation_envelope(
        {
            "schema_version": 1,
            "domain": "redco-stage-d1-support-v13-foundation-tree-manifest-f1",
            "foundation_parent_commit": FOUNDATION_PARENT_COMMIT,
            "foundation_status": "candidate_null_uncommitted_foundation_only",
            "allowlist": list(allowlist),
            "files": entries,
            "manifest_self_hash": "omitted_to_avoid_self_hash_circularity",
            "excluded_classes": list(FOUNDATION_EXCLUDED_CLASSES),
            "source": provenance["source"],
            "assertions": {
                "candidate": FOUNDATION_NULL_CANDIDATE,
                "no_post_179_source_rows": True,
                "phase_b_authorization_artifact_present": False,
                "launch_authorized": False,
                "provider_calls_authorized": False,
                "phase_b_authorized": False,
                "foundation_only": True,
                "non_authorizing": True,
                "trust_anchor_external_authorization": False,
                "parent_head_verified": head,
                "external_prime_rl_in_allowlist": False,
            },
            "qasper": {
                **provenance["source"],
                "schema_fingerprint": SOURCE_SCHEMA_SHA256,
                "license": "cc-by-4.0",
                "attribution": "allenai/qasper",
            },
            "reproducibility": {
                "canonical_json": True,
                "trailing_newline": False,
                "absolute_paths": False,
                "timings": False,
                "source_rows_after_cutoff_read": False,
            },
        }
        ),
    )
    validate_foundation_envelope(manifest)
    return manifest


def validate_foundation_manifest(root: Path, value: dict[str, Any]) -> None:
    expected = build_foundation_manifest(root)
    if value != expected:
        raise ValueError("Foundation F manifest differs from independently rebuilt witness")


__all__ = [
    "FOUNDATION_ALLOWLIST",
    "FOUNDATION_EXCLUDED_CLASSES",
    "FOUNDATION_MANIFEST_RELATIVE",
    "FOUNDATION_PARENT_COMMIT",
    "PRIME_RL_GITLINK_MODE",
    "PRIME_RL_GITLINK_OBJECT",
    "PRIME_RL_RELATIVE",
    "SOURCE_PROVENANCE_RELATIVE",
    "build_foundation_manifest",
    "build_integrity_anchor",
    "build_source_provenance",
    "validate_foundation_manifest",
]
