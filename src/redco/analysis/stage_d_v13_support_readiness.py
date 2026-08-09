"""Non-authorizing successor readiness contract for Stage D v13 support.

This module deliberately does not provide an execute surface.  It freezes the
CPU-reviewed dependency, owner, asset, and future-authorization contracts for
one direct child of the Phase-2 checkpoint.  Historical launch-v1 remains
unchanged and cannot authorize this ancestry.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tarfile
from pathlib import Path
from typing import Any, cast

from redco.analysis.stage_d_dependency_stack import live_owner_dependency_payload
from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes
from redco.analysis.stage_d_v13_foundation import (
    PRIME_RL_GITLINK_MODE,
    PRIME_RL_GITLINK_OBJECT,
    PRIME_RL_RELATIVE,
)
from redco.analysis.stage_d_v13_launch_observations import validate_prime_observation
from redco.analysis.stage_d_v13_source_phase_a_decoder import hardened_git

ROOT = Path(__file__).parents[3].resolve()
READINESS_PARENT_COMMIT = "c77d57ce88380b838f92971648969dc3c6c4e759"
READINESS_PARENT_TREE = "d0a190eea635ef369c8e970933a6845ee4a20458"
READINESS_DOMAIN = "redco-stage-d1-support-v13-readiness-repair-v1"
DEPENDENCY_DOMAIN = "redco-stage-d-dependency-stack-v13-readiness-v1"
FUTURE_AUTH_DOMAIN = "redco-stage-d1-support-v13-launch-authorization-v2"
AUTHORIZATION_TEXT = (
    "I accept that the Stage D scientific campaign is exploratory, with paired n=32 "
    "providing approximately 72.14% power, and authorize up to $12 for the support "
    "phase under the frozen protocol."
)
AUTHORIZATION_TEXT_SHA256 = (
    "62f0433268ca4c3eba9e6387bf0d9e74ad8d831a7138bba353bfd9dcb00a0129"
)
ORCHESTRATOR_THREAD = "019f9ab9-ec45-7ac3-82b1-09757b92a7c3"
CURRENT_UV_LOCK_SHA256 = (
    "60e9fe7396d45d8e8edd13d2de708fa4895452410b43e1ad860f720047634d31"
)
PHASE2_AUDIT_RELATIVE = "reports/stage-d2-qa-localization-audit-v2.json"
PHASE2_AUDIT_SHA256 = (
    "d8baa34cbd55c74213913825e186580ab9bef69ad34174f38cebfd071460d5a9"
)
FROZEN_PROTOCOL_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-v13-frozen-support-protocol-v1.json"
)
FROZEN_PROTOCOL_SHA256 = (
    "65734f3dc5caeb1866e25b535d5b91d17ffbc434b69fbe0baf5efe63d339145b"
)
FROZEN_COHORT_RELATIVE = "datasets/stage-d/qasper-support-v13-launch-input-v1.jsonl"
FROZEN_COHORT_SHA256 = (
    "02571660e6893b4fe804c504c90d7a4027a50609dc2520c3e865b95e8af9b8f8"
)
FROZEN_PLAN_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-v13-launch-collection-plan-v1.json"
)
FROZEN_PLAN_SHA256 = (
    "a63e2e43d009e576ce79d22f56c978e93b359a3fa244cdf1fca89020699c1f7c"
)
HISTORICAL_V1_AUTH_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-v13-launch-authorization-v1.json"
)
HISTORICAL_V1_AUTH_SHA256 = (
    "30020b15b5929af1bf668de1bd6b3eb15fe068ec86b24d2dc9a05a8b3b72a7be"
)
HISTORICAL_V12_DEPENDENCY_RELATIVE = "configs/stage-d/stage-d1-dependency-stack-v12.json"
HISTORICAL_V12_DEPENDENCY_SHA256 = (
    "cda524c6ecea9821b1e36290da64df465aa46fad9ec174881c24d3dc895b2831"
)

DEPENDENCY_MANIFEST_RELATIVE = (
    "configs/stage-d/stage-d1-support-readiness-dependency-manifest-v1.json"
)
READINESS_MANIFEST_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-v13-readiness-repair-v1.json"
)
READINESS_AUDIT_RELATIVE = "reports/stage-d1-support-v13-readiness-audit-v1.json"
FUTURE_AUTH_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-v13-launch-authorization-v2.json"
)
FIXED_LOCAL_ARTIFACT_ROOT = "D:/redco-artifacts/stage-d-v13-support-v2"
FIXED_PRIME_OBSERVATION_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-readiness/prime-observation-v2.json"
)
MAX_TRANSFER_BYTES = 16 * 1024**3
MAX_ARCHIVE_EXTRACTED_BYTES = MAX_TRANSFER_BYTES
MAX_ARCHIVE_MEMBERS = 100_000
MIN_POST_TRANSFER_FREE_BYTES = 4 * 1024**3
PRIME_OBSERVATION_TTL_SECONDS = 900

READINESS_PATHS = frozenset(
    {
        DEPENDENCY_MANIFEST_RELATIVE,
        READINESS_MANIFEST_RELATIVE,
        READINESS_AUDIT_RELATIVE,
        "scripts/build_stage_d_v13_support_readiness.py",
        "src/redco/analysis/stage_d_v13_support_readiness.py",
        "tests/test_stage_d_v13_support_readiness.py",
    }
)

SUPPORT_OWNER_PATHS = (
    "environments/redco_evidence_selection_v2/redco_evidence_selection_v2/scientific_campaign_driver.py",
    "environments/redco_evidence_selection_v2/redco_evidence_selection_v2/scientific_env.py",
    "environments/redco_evidence_selection_v2/redco_evidence_selection_v2/source_env.py",
    "scripts/run_stage_d_scientific_campaign.py",
    "scripts/run_stage_d_source_collection.py",
    "scripts/run_stage_d_v13_local_orchestrator.py",
    "scripts/run_stage_d_v13_remote_bootstrap.py",
    "scripts/run_stage_d_v13_support.py",
    "src/redco/analysis/stage_d_action_closure.py",
    "src/redco/analysis/stage_d_dependency_stack.py",
    "src/redco/analysis/stage_d_receipt_ledger.py",
    "src/redco/analysis/stage_d_replay_controller.py",
    "src/redco/analysis/stage_d_scientific_campaign.py",
    "src/redco/analysis/stage_d_source_producer.py",
    "src/redco/analysis/stage_d_v13_launch_lifecycle.py",
    "src/redco/analysis/stage_d_v13_launch_observations.py",
    "src/redco/analysis/stage_d_v13_support_launch_runtime.py",
    "src/redco/analysis/stage_d_v13_support_readiness.py",
)


class ReadinessBlocked(RuntimeError):
    """The successor bundle is valid but cannot authorize provisioning."""


def _git(root: Path, *args: str, text: bool = True) -> str | bytes:
    # The shared checkout is owned by Windows Git.  Pin its clean-filter view so
    # WSL validation does not misclassify committed CRLF files as dirty.
    result = hardened_git(root, "-c", "core.autocrlf=true", *args, text=text)
    if result.returncode != 0:
        raise ValueError(f"Git authentication failed: {' '.join(args)}")
    if text:
        if not isinstance(result.stdout, str):
            raise ValueError("Git text authentication returned non-text output")
        return result.stdout.strip()
    if not isinstance(result.stdout, bytes):
        raise ValueError("Git object authentication returned non-bytes output")
    return result.stdout


def _bound_file(root: Path, relative: str, expected: str) -> bytes:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"bound readiness input is missing: {relative}")
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected:
        raise ValueError(f"bound readiness input changed: {relative}")
    return raw


def _validate_gitlink(root: Path) -> None:
    line = cast(str, _git(root, "ls-files", "--stage", "--", PRIME_RL_RELATIVE))
    if line.split() != [
        PRIME_RL_GITLINK_MODE,
        PRIME_RL_GITLINK_OBJECT,
        "0",
        PRIME_RL_RELATIVE,
    ]:
        raise ValueError("external/prime-rl gitlink differs from the fixed witness")


def _status_paths(root: Path) -> set[str]:
    raw = cast(
        bytes,
        _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", text=False),
    )
    records = raw.split(b"\0")
    paths: set[str] = set()
    index = 0
    while index < len(records) and records[index]:
        record = records[index]
        if len(record) < 4 or record[2:3] != b" ":
            raise ValueError("malformed readiness Git status")
        status = record[:2].decode("ascii")
        path = os.fsdecode(record[3:]).replace("\\", "/")
        index += 1
        if "R" in status or "C" in status:
            if index >= len(records) or not records[index]:
                raise ValueError("readiness rename/copy status lacks a source")
            paths.add(os.fsdecode(records[index]).replace("\\", "/"))
            index += 1
        if path == PRIME_RL_RELATIVE and status == " M":
            _validate_gitlink(root)
            continue
        paths.add(path)
    return paths


def _authenticate_precommit(root: Path) -> None:
    if _git(root, "rev-parse", "HEAD") != READINESS_PARENT_COMMIT:
        raise ValueError("readiness build requires exact committed parent c77d57c")
    if _git(root, "rev-parse", "HEAD^{tree}") != READINESS_PARENT_TREE:
        raise ValueError("readiness parent tree changed")
    unexpected = _status_paths(root).difference(READINESS_PATHS)
    if unexpected:
        raise ValueError(
            "readiness worktree contains paths outside the exact allowlist: "
            + ", ".join(sorted(unexpected))
        )


def _commit_diff(root: Path, parent: str, child: str) -> dict[str, str]:
    lines = cast(
        str,
        _git(root, "diff", "--name-status", "--no-renames", parent, child),
    ).splitlines()
    result: dict[str, str] = {}
    for line in lines:
        status, separator, path = line.partition("\t")
        if not separator or status not in {"A", "M", "D"} or path in result:
            raise ValueError("readiness commit diff is malformed")
        result[path] = status
    return result


def authenticate_readiness_commit(root: Path, commit: str) -> None:
    parents = cast(str, _git(root, "rev-list", "--parents", "-n", "1", commit)).split()
    if len(parents) != 2 or parents[1] != READINESS_PARENT_COMMIT:
        raise ValueError("readiness repair must be a single-parent direct child of c77d57c")
    expected = {relative: "A" for relative in READINESS_PATHS}
    if _commit_diff(root, READINESS_PARENT_COMMIT, commit) != expected:
        raise ValueError("readiness repair commit differs from the exact six-path allowlist")


def _phase2_payload(root: Path) -> dict[str, Any]:
    raw = _bound_file(root, PHASE2_AUDIT_RELATIVE, PHASE2_AUDIT_SHA256)
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError("Phase-2 audit is not canonical")
    evidence = value.get("verification_evidence")
    if not isinstance(evidence, dict) or not isinstance(evidence.get("payload"), dict):
        raise ValueError("Phase-2 audit lacks authenticated CPU evidence")
    return cast(dict[str, Any], evidence["payload"])


def build_dependency_manifest(root: Path) -> bytes:
    root = root.resolve()
    _bound_file(root, HISTORICAL_V12_DEPENDENCY_RELATIVE, HISTORICAL_V12_DEPENDENCY_SHA256)
    if sha256_bytes((root / "uv.lock").read_bytes()) != CURRENT_UV_LOCK_SHA256:
        raise ValueError("current uv.lock differs from the readiness binding")
    phase2 = _phase2_payload(root)
    runtime = phase2.get("runtime")
    abi = phase2.get("abi_probe")
    if not isinstance(runtime, dict) or runtime != {
        "offline": True,
        "project_sync": False,
        "python_executable_sha256": (
            "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
        ),
        "python_implementation": "CPython",
        "python_version": "3.12.3",
        "uv_executable_sha256": (
            "da15297d6879b2cfbe5ea3cb03725c1613d51ba72892cc996468d871f0a532fb"
        ),
        "uv_lock_sha256": CURRENT_UV_LOCK_SHA256,
        "uv_version": "uv 0.11.32 (x86_64-unknown-linux-gnu)",
    }:
        raise ValueError("Phase-2 runtime binding differs from the reviewed CPU evidence")
    if (
        not isinstance(abi, dict)
        or abi.get("split_engine_sampling") is not True
        or abi.get("episode_trace_model_call_v2_validated") is not True
        or abi.get("controller_watchdog_interface") is not True
        or abi.get("sampling_contract_sha256")
        != "819222244a81565a67331826be3dd362e14e1481043d60fccb569551a4471f6d"
        or abi.get("sampling_field_count") != 12
        or not isinstance(abi.get("module_bindings"), list)
        or not abi["module_bindings"]
    ):
        raise ValueError("Phase-2 ABI binding differs from the reviewed CPU evidence")
    owner_bindings = {
        relative: sha256_bytes((root / relative).read_bytes()) for relative in SUPPORT_OWNER_PATHS
    }
    dependency = live_owner_dependency_payload(root)
    return cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": 1,
                "domain": DEPENDENCY_DOMAIN,
                "historical_v12": {
                    "path": HISTORICAL_V12_DEPENDENCY_RELATIVE,
                    "sha256": HISTORICAL_V12_DEPENDENCY_SHA256,
                    "immutable": True,
                    "authorizes_successor": False,
                },
                "live_owner_stack": dependency,
                "runtime": runtime,
                "abi": abi,
                "uv_lock": {"path": "uv.lock", "sha256": CURRENT_UV_LOCK_SHA256},
                "phase2_evidence": {
                    "path": PHASE2_AUDIT_RELATIVE,
                    "sha256": PHASE2_AUDIT_SHA256,
                    "parent_commit": READINESS_PARENT_COMMIT,
                },
                "owner_bindings": dict(sorted(owner_bindings.items())),
            }
        ),
    )


def _artifact_entries(root: Path) -> list[dict[str, object]]:
    base = json.loads((root / "configs/stage-d/stage-d1-base-model-manifest.json").read_bytes())
    adapter = json.loads(
        (root / "reports/stage-d0-scaffold-step8-adapter-manifest-v1.json").read_bytes()
    )
    entries: list[dict[str, object]] = []
    for name, binding in sorted(cast(dict[str, dict[str, object]], base["files"]).items()):
        entries.append(
            {
                "name": f"base-model:{name}",
                "relative_path": f"base-model/{name}",
                "sha256": binding["sha256"],
                "expected_bytes": binding["bytes"],
                "kind": "regular_file",
            }
        )
    entries.extend(
        (
            {
                "name": "adapter:model",
                "relative_path": "adapter/adapter_model.safetensors",
                "sha256": base["adapter_model_sha256"],
                "expected_bytes": adapter["members"]["adapter_model.safetensors"][
                    "byte_length"
                ],
                "kind": "regular_file",
            },
            {
                "name": "adapter:archive",
                "relative_path": "archives/selected-adapter.tar.gz",
                "sha256": adapter["archive_sha256"],
                "expected_bytes": adapter["archive_byte_length"],
                "kind": "tar_archive",
            },
            {
                "name": "offline:rlm-archive",
                "relative_path": "runtime/rlm-patched-v1.tar",
                "sha256": "0db97a244b88b5186a27fe1af14fa33a89b69e39a68a064cf7d10bb6f3084669",
                "expected_bytes": None,
                "kind": "tar_archive",
            },
            {
                "name": "offline:uv-cache",
                "relative_path": "runtime/rlm-cache-v2.tar.gz",
                "sha256": "fb1ff3fe82a5109db0662092b344e0069ada5365f137d01b3d7010b84d7e37be",
                "expected_bytes": None,
                "kind": "tar_archive",
            },
            {
                "name": "offline:uv",
                "relative_path": "runtime/uv",
                "sha256": "da15297d6879b2cfbe5ea3cb03725c1613d51ba72892cc996468d871f0a532fb",
                "expected_bytes": None,
                "kind": "executable",
            },
            {
                "name": "offline:launcher",
                "relative_path": "runtime/rlm-wrapper",
                "sha256": "7f6d55f352d521a4d34c675e2bc5cb9581fde2fb635c826a9486470aa0de6cd4",
                "expected_bytes": None,
                "kind": "executable",
            },
        )
    )
    return sorted(entries, key=lambda item: cast(str, item["name"]))


def build_readiness_artifacts(root: Path) -> dict[str, bytes]:
    root = root.resolve()
    _authenticate_precommit(root)
    _bound_file(root, FROZEN_PROTOCOL_RELATIVE, FROZEN_PROTOCOL_SHA256)
    _bound_file(root, FROZEN_COHORT_RELATIVE, FROZEN_COHORT_SHA256)
    _bound_file(root, FROZEN_PLAN_RELATIVE, FROZEN_PLAN_SHA256)
    _bound_file(root, HISTORICAL_V1_AUTH_RELATIVE, HISTORICAL_V1_AUTH_SHA256)
    dependency = build_dependency_manifest(root)
    readiness = cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": 1,
                "domain": READINESS_DOMAIN,
                "state": "non_authorizing_readiness_repair",
                "parent": {
                    "commit": READINESS_PARENT_COMMIT,
                    "tree": READINESS_PARENT_TREE,
                },
                "historical_v1": {
                    "path": HISTORICAL_V1_AUTH_RELATIVE,
                    "sha256": HISTORICAL_V1_AUTH_SHA256,
                    "ancestry_compatible": False,
                    "unchanged": True,
                },
                "dependency_manifest": {
                    "path": DEPENDENCY_MANIFEST_RELATIVE,
                    "sha256": sha256_bytes(dependency),
                },
                "artifact_root": {
                    "platform": "windows",
                    "fixed_path": FIXED_LOCAL_ARTIFACT_ROOT,
                    "caller_override_allowed": False,
                    "required_entries": _artifact_entries(root),
                    "max_transfer_bytes": MAX_TRANSFER_BYTES,
                    "max_archive_extracted_bytes": MAX_ARCHIVE_EXTRACTED_BYTES,
                    "max_archive_members": MAX_ARCHIVE_MEMBERS,
                    "minimum_post_transfer_free_bytes": MIN_POST_TRANSFER_FREE_BYTES,
                },
                "frozen_support": {
                    "authorization_text_sha256": AUTHORIZATION_TEXT_SHA256,
                    "authorization_text_bytes": len(AUTHORIZATION_TEXT.encode("utf-8")),
                    "protocol": {
                        "path": FROZEN_PROTOCOL_RELATIVE,
                        "sha256": FROZEN_PROTOCOL_SHA256,
                    },
                    "cohort": {
                        "path": FROZEN_COHORT_RELATIVE,
                        "sha256": FROZEN_COHORT_SHA256,
                        "rows": 64,
                    },
                    "collection_plan": {
                        "path": FROZEN_PLAN_RELATIVE,
                        "sha256": FROZEN_PLAN_SHA256,
                    },
                    "support_success_floor": 58,
                    "branch_count_k": 4,
                    "minimum_score_range": 0.05,
                    "support_cap_usd": 12,
                    "attempt_limit": 1,
                    "retry_after_outcome_activity": False,
                },
                "future_authorization": {
                    "path": FUTURE_AUTH_RELATIVE,
                    "domain": FUTURE_AUTH_DOMAIN,
                    "must_be_direct_child": True,
                    "sole_diff_path": FUTURE_AUTH_RELATIVE,
                    "present": False,
                    "one_use_handoff_domain": (
                        "redco-stage-d1-support-v13-execute-handoff-v2"
                    ),
                },
                "future_preflight": {
                    "prime_cli_version": "0.6.20",
                    "fixed_observation_path": FIXED_PRIME_OBSERVATION_RELATIVE,
                    "ttl_seconds": PRIME_OBSERVATION_TTL_SECONDS,
                    "raw_stdout_stderr_hashes_required": True,
                    "wallet_minimum_usd": 30,
                    "pods_required": 0,
                    "disks_required": 0,
                    "qualifying_resources_required": 1,
                    "resource": "non-spot 2x48GB L40/L40S/RTX6000Ada",
                    "maximum_hourly_rate_usd": 2,
                },
                "stop_rules": {
                    "missing_or_ambiguous_asset": "stop_before_provisioning",
                    "missing_or_ambiguous_billing": "terminal_support_stop",
                    "missing_or_ambiguous_evidence": "terminal_support_stop",
                    "teardown_not_proven": "terminal_support_stop",
                    "support_success_authorizes_science": False,
                },
                "authorization": {
                    "support_launch_authorized": False,
                    "support_spend_authorized": False,
                    "provider_calls_authorized": False,
                    "model_calls_authorized": False,
                    "prime_authorized": False,
                    "science_authorized": False,
                    "training_authorized": False,
                    "heldout_evaluation_authorized": False,
                    "scientific_transition_authorized": False,
                },
            }
        ),
    )
    file_bindings = {
        "scripts/build_stage_d_v13_support_readiness.py": sha256_bytes(
            (root / "scripts/build_stage_d_v13_support_readiness.py").read_bytes()
        ),
        "src/redco/analysis/stage_d_v13_support_readiness.py": sha256_bytes(
            (root / "src/redco/analysis/stage_d_v13_support_readiness.py").read_bytes()
        ),
        "tests/test_stage_d_v13_support_readiness.py": sha256_bytes(
            (root / "tests/test_stage_d_v13_support_readiness.py").read_bytes()
        ),
        DEPENDENCY_MANIFEST_RELATIVE: sha256_bytes(dependency),
        READINESS_MANIFEST_RELATIVE: sha256_bytes(readiness),
    }
    audit = cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": 1,
                "domain": "redco-stage-d1-support-v13-readiness-audit-v1",
                "state": "non_authorizing_cpu_readiness_evidence",
                "parent": {
                    "commit": READINESS_PARENT_COMMIT,
                    "tree": READINESS_PARENT_TREE,
                },
                "allowlist": sorted(READINESS_PATHS),
                "file_bindings": dict(sorted(file_bindings.items())),
                "self_hash": "excluded_to_avoid_circular_binding",
                "artifact_root_observed": False,
                "prime_observation_performed": False,
                "support_launch_authorized": False,
                "science_authorized": False,
            }
        ),
    )
    return {
        DEPENDENCY_MANIFEST_RELATIVE: dependency,
        READINESS_MANIFEST_RELATIVE: readiness,
        READINESS_AUDIT_RELATIVE: audit,
    }


def verify_readiness_bundle(root: Path, output_root: Path) -> dict[str, str]:
    expected = build_readiness_artifacts(root)
    hashes: dict[str, str] = {}
    for relative, value in expected.items():
        path = output_root / relative
        if path.is_symlink() or not path.is_file() or path.read_bytes() != value:
            raise ValueError(f"readiness artifact differs from reconstructed bytes: {relative}")
        hashes[relative] = sha256_bytes(value)
    return hashes


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & reparse
    )


def _normalized_archive_member_path(name: str, archive_name: str) -> str:
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise ReadinessBlocked(f"archive member path is unsafe: {archive_name}")
    raw_parts = name.split("/")
    if ".." in raw_parts:
        raise ReadinessBlocked(f"archive member path is unsafe: {archive_name}")
    parts = tuple(part for part in raw_parts if part not in {"", "."})
    if not parts or parts[0].endswith(":"):
        raise ReadinessBlocked(f"archive member path is unsafe: {archive_name}")
    return "/".join(parts)


def _validate_archive(path: Path) -> None:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            members = archive.getmembers()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise ReadinessBlocked(f"archive has too many members: {path.name}")
            extracted_bytes = 0
            normalized_types: dict[str, bytes] = {}
            for member in members:
                normalized = _normalized_archive_member_path(member.name, path.name)
                if member.type not in {tarfile.REGTYPE, tarfile.DIRTYPE}:
                    raise ReadinessBlocked(f"archive member is unsafe: {path.name}")
                if normalized in normalized_types:
                    raise ReadinessBlocked(
                        f"archive has a duplicate normalized member path: {path.name}"
                    )
                parent_parts = normalized.split("/")[:-1]
                for index in range(1, len(parent_parts) + 1):
                    parent = "/".join(parent_parts[:index])
                    if normalized_types.get(parent) == tarfile.REGTYPE:
                        raise ReadinessBlocked(
                            f"archive member path collides with a file: {path.name}"
                        )
                if member.type == tarfile.REGTYPE and any(
                    existing.startswith(f"{normalized}/") for existing in normalized_types
                ):
                    raise ReadinessBlocked(
                        f"archive member path collides with descendants: {path.name}"
                    )
                normalized_types[normalized] = member.type
                if member.type == tarfile.REGTYPE:
                    if type(member.size) is not int or member.size < 0:
                        raise ReadinessBlocked(f"archive member size is invalid: {path.name}")
                    extracted_bytes += member.size
                    if extracted_bytes > MAX_ARCHIVE_EXTRACTED_BYTES:
                        raise ReadinessBlocked(
                            f"archive extracted size exceeds the fixed budget: {path.name}"
                        )
    except tarfile.TarError as error:
        raise ReadinessBlocked(f"archive is invalid: {path.name}") from error


def _validate_artifact_root(
    root: Path,
    entries: list[dict[str, object]],
    *,
    disk_free_bytes: int | None = None,
) -> dict[str, int]:
    if not root.exists() or not root.is_dir() or _is_link_or_reparse(root):
        raise ReadinessBlocked("fixed local artifact root is absent or linked")
    current = Path(root.anchor) if root.is_absolute() else Path()
    for part in root.parts[1:] if root.is_absolute() else root.parts:
        current /= part
        if _is_link_or_reparse(current):
            raise ReadinessBlocked("artifact root has a symlink/reparse ancestor")
    total = 0
    seen_inodes: set[tuple[int, int]] = set()
    for entry in entries:
        relative = Path(cast(str, entry["relative_path"]))
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ReadinessBlocked("artifact entry escapes the fixed root")
        path = root / relative
        cursor = root
        for part in relative.parts:
            cursor /= part
            if _is_link_or_reparse(cursor):
                raise ReadinessBlocked("artifact entry has a linked/reparse ancestor")
        if not path.is_file() or path.is_symlink():
            raise ReadinessBlocked(f"required artifact is absent: {relative.as_posix()}")
        info = path.stat()
        if info.st_nlink != 1:
            raise ReadinessBlocked("required artifact is a hard-link alias")
        identity = (info.st_dev, info.st_ino)
        if identity in seen_inodes:
            raise ReadinessBlocked("required artifact aliases another entry")
        seen_inodes.add(identity)
        raw = path.read_bytes()
        if sha256_bytes(raw) != entry["sha256"]:
            raise ReadinessBlocked(f"required artifact hash differs: {relative.as_posix()}")
        expected_bytes = entry["expected_bytes"]
        if expected_bytes is not None and info.st_size != expected_bytes:
            raise ReadinessBlocked(f"required artifact size differs: {relative.as_posix()}")
        if entry["kind"] == "tar_archive":
            _validate_archive(path)
        total += info.st_size
    free = shutil.disk_usage(root).free if disk_free_bytes is None else disk_free_bytes
    if total > MAX_TRANSFER_BYTES:
        raise ReadinessBlocked("authenticated transfer set exceeds the fixed size budget")
    if free < total + MIN_POST_TRANSFER_FREE_BYTES:
        raise ReadinessBlocked("artifact volume lacks the fixed post-transfer reserve")
    return {"files": len(entries), "transfer_bytes": total, "free_bytes": free}


def validate_local_artifacts() -> dict[str, int]:
    """Validate only the fixed production artifact root; callers cannot override it."""

    if os.name != "nt":
        raise ReadinessBlocked("local artifact readiness must run on the Windows owner")
    return _validate_artifact_root(Path(FIXED_LOCAL_ARTIFACT_ROOT), _artifact_entries(ROOT))


def validate_future_prime_observation() -> dict[str, Any]:
    """Validate only the future fixed observation path after artifact readiness."""

    validate_local_artifacts()
    observation = ROOT / FIXED_PRIME_OBSERVATION_RELATIVE
    value = validate_prime_observation(ROOT, observation)
    inventory = cast(dict[str, Any], value["inventory"])
    resource = cast(dict[str, Any], value["resource"])
    if inventory != {"pods": [], "disks": []}:
        raise ReadinessBlocked("Prime observation must prove zero pods and disks")
    gpu = str(resource.get("gpu_type", "")).replace(" ", "").upper()
    if (
        gpu not in {"L40", "L40S", "RTX6000ADA"}
        or resource.get("gpu_count") != 2
        or resource.get("gpu_memory_gb") != 48
        or resource.get("is_spot") is not False
        or type(resource.get("price_per_hour")) not in {int, float}
        or cast(float, resource["price_per_hour"]) > 2
    ):
        raise ReadinessBlocked("Prime observation lacks the one exact qualifying resource")
    return cast(dict[str, Any], value)


def _git_blob(root: Path, commit: str, relative: str) -> bytes:
    return cast(bytes, _git(root, "show", f"{commit}:{relative}", text=False))


def _future_authorization_payload(root: Path, readiness_commit: str) -> dict[str, Any]:
    readiness_manifest = _git_blob(root, readiness_commit, READINESS_MANIFEST_RELATIVE)
    dependency_manifest = _git_blob(root, readiness_commit, DEPENDENCY_MANIFEST_RELATIVE)
    return {
        "schema_version": 2,
        "domain": FUTURE_AUTH_DOMAIN,
        "state": "support_only_authorization",
        "readiness_commit": readiness_commit,
        "readiness_tree": _git(root, "rev-parse", f"{readiness_commit}^{{tree}}"),
        "readiness_manifest": {
            "path": READINESS_MANIFEST_RELATIVE,
            "sha256": sha256_bytes(readiness_manifest),
            "git_blob": _git(
                root, "rev-parse", f"{readiness_commit}:{READINESS_MANIFEST_RELATIVE}"
            ),
        },
        "dependency_manifest": {
            "path": DEPENDENCY_MANIFEST_RELATIVE,
            "sha256": sha256_bytes(dependency_manifest),
            "git_blob": _git(
                root, "rev-parse", f"{readiness_commit}:{DEPENDENCY_MANIFEST_RELATIVE}"
            ),
        },
        "artifact_root": FIXED_LOCAL_ARTIFACT_ROOT,
        "authorization": {
            "text": AUTHORIZATION_TEXT,
            "text_bytes": len(AUTHORIZATION_TEXT.encode("utf-8")),
            "text_sha256": AUTHORIZATION_TEXT_SHA256,
            "orchestrator_thread_id": ORCHESTRATOR_THREAD,
        },
        "frozen_support": {
            "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "cohort_sha256": FROZEN_COHORT_SHA256,
            "collection_plan_sha256": FROZEN_PLAN_SHA256,
            "support_cap_usd": 12,
            "support_papers": 64,
            "support_attempt_limit": 1,
        },
        "one_use_handoff_domain": "redco-stage-d1-support-v13-execute-handoff-v2",
        "scope": {
            "support_launch_authorized": True,
            "support_spend_authorized": True,
            "provider_calls_authorized": True,
            "model_calls_authorized": True,
            "candidate_selection_authorized": False,
            "science_authorized": False,
            "training_authorized": False,
            "heldout_evaluation_authorized": False,
            "scientific_transition_authorized": False,
            "retry_after_outcome_activity": False,
        },
    }


def _validate_future_authorization(root: Path) -> dict[str, Any]:
    if _status_paths(root):
        raise ValueError("future support authorization requires an exact clean superproject")
    head = cast(str, _git(root, "rev-parse", "HEAD"))
    parents = cast(str, _git(root, "rev-list", "--parents", "-n", "1", head)).split()
    if len(parents) != 2:
        raise ValueError("future support authorization rejects merge commits")
    readiness_commit = parents[1]
    authenticate_readiness_commit(root, readiness_commit)
    if _commit_diff(root, readiness_commit, head) != {FUTURE_AUTH_RELATIVE: "A"}:
        raise ValueError("future support authorization must be the sole direct-child addition")
    raw = _git_blob(root, head, FUTURE_AUTH_RELATIVE)
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError("future support authorization is not canonical")
    expected = _future_authorization_payload(root, readiness_commit)
    if value != expected:
        raise ValueError("future support authorization differs from the frozen v2 schema")
    return cast(dict[str, Any], value)


def validate_future_support_authorization() -> dict[str, Any]:
    """Production no-argument future validator; currently fails because v2 is absent."""

    return _validate_future_authorization(ROOT)


__all__ = [
    "AUTHORIZATION_TEXT_SHA256",
    "DEPENDENCY_MANIFEST_RELATIVE",
    "FIXED_LOCAL_ARTIFACT_ROOT",
    "FUTURE_AUTH_RELATIVE",
    "READINESS_AUDIT_RELATIVE",
    "READINESS_MANIFEST_RELATIVE",
    "READINESS_PARENT_COMMIT",
    "READINESS_PATHS",
    "ReadinessBlocked",
    "authenticate_readiness_commit",
    "build_dependency_manifest",
    "build_readiness_artifacts",
    "validate_future_prime_observation",
    "validate_future_support_authorization",
    "validate_local_artifacts",
    "verify_readiness_bundle",
]
