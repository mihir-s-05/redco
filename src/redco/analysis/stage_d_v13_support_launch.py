"""Authenticated, support-only Stage-D v13 launch bundle.

This module owns the preflight boundary for the one authorized support attempt.
It deliberately consumes only committed derived support inputs; it never opens
the authenticated QASPER Parquet object.  Live execution is exposed through a
small private runtime protocol so the command-line verifier cannot be used to
override scientific settings or silently execute a second attempt.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import time
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from redco.analysis.stage_d_collection import StageDCollectionPlan, derive_scientific_group_id
from redco.analysis.stage_d_dependency_stack import live_owner_dependency_payload
from redco.analysis.stage_d_protocol_manifest import (
    StageDPolicyIdentity,
    StageDProtocolManifest,
)
from redco.analysis.stage_d_returning_root_correspondence import contract_artifact_bytes
from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes
from redco.analysis.stage_d_v13_draft_publication import validate_output_paths
from redco.analysis.stage_d_v13_launch_lifecycle import SigningIdentity
from redco.analysis.stage_d_v13_launch_observations import (
    validate_pod_runtime_observation,
    validate_prime_observation,
)
from redco.analysis.stage_d_v13_support_contract import (
    AUTHENTICATED_PREDECESSOR_HASHES,
    CANDIDATE_EXAMPLE_ID,
    CANDIDATE_PAPER_ID,
    CANDIDATE_QUESTION_INDEX,
    CANDIDATE_RELATIVE,
    CANDIDATE_ROW_SHA256,
    CANDIDATE_SELECTION_ADDRESS_SHA256,
    CANDIDATE_SOURCE_ORDINAL,
    COMPOSITION_RELATIVE,
    FROZEN_SUPPORT_RULES_RELATIVE,
    FROZEN_SUPPORT_RULES_SHA256,
    MASTER_SEED,
    PROTOCOL_RELATIVE,
    RETAINED_SUPPORT_RELATIVE,
    RETAINED_SUPPORT_SHA256,
    SCIENTIFIC_NAMESPACE,
    SELECTION_CLAIM_RELATIVE,
    SELECTION_CLAIM_SHA256,
    SELECTION_MANIFEST_RELATIVE,
    SELECTION_MANIFEST_SHA256,
    SELECTION_RECEIPT_RELATIVE,
    SELECTION_RECEIPT_SHA256,
    SOURCE_LOGICAL_URL,
    SOURCE_PATH,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    SOURCE_ROW_COUNT,
    SOURCE_SCHEMA_SHA256,
    SOURCE_SEMANTIC_COMMIT,
    SOURCE_SHA256,
    SUPPORTED_DATASETS,
    SUPPORTED_PYARROW,
    SUPPORTED_PYTHON,
    V12_ARCHIVE_RELATIVE,
    V12_ARCHIVE_SHA256,
    V12_EVIDENCE_MANIFEST_RELATIVE,
    V12_EVIDENCE_MANIFEST_SHA256,
    V12_FINALIZATION_AUDIT_RELATIVE,
    V12_FINALIZATION_AUDIT_SHA256,
    V12_PREREG_RELATIVE,
    V12_PREREG_SHA256,
    V12_PROTOCOL_RELATIVE,
    V12_PROTOCOL_SHA256,
    V12_SOURCE_EVAL_RELATIVE,
    V12_SOURCE_EVAL_SHA256,
    V12_TERMINAL_REPORT_RELATIVE,
    V12_TERMINAL_REPORT_SHA256,
    sampling_contract_binding,
)
from redco.analysis.stage_d_v13_support_launch_runtime import execute_support_once

LAUNCH_DOMAIN = "redco-stage-d1-support-v13-launch-authorization-v1"
LAUNCH_SCHEMA_VERSION = 1
PARENT_COMMIT = "b0a41608ede24e4e3216db2160971c255ef82bba"
PARENT_TREE = "bba9bc4fb2433cf7dd9c0216d4d6d6e33d426421"
ORCHESTRATOR_THREAD = "019f9ab9-ec45-7ac3-82b1-09757b92a7c3"
AUTHORIZATION_TEXT = (
    "I accept that the Stage D scientific campaign is exploratory, with paired n=32 "
    "providing approximately 72.14% power, and authorize up to $12 for the support "
    "phase under the frozen protocol."
)
AUTHORIZATION_TEXT_SHA256 = "62f0433268ca4c3eba9e6387bf0d9e74ad8d831a7138bba353bfd9dcb00a0129"
SIGNING_PUBLIC_KEY_TYPE = "ssh-rsa"
SIGNING_PUBLIC_KEY_BASE64 = (
    "AAAAB3NzaC1yc2EAAAADAQABAAACAQC37IhPgU3QIeja+98pLT0bf5I3W4/bknSBeMxaQXH81la/"
    "VBbSst7RgZsm3gzMvWSZE7xrnYWX43lC7hykpUFEQZbUqzrc/QLgbRTaG3xgKEpA1abqTGgipYY/"
    "vb8AlEvngzNf2jBmKSDyIB+278tEFpnhB135rICdilZXO4pVJLkVtOf41yzfXZbnUTwubGCjbp7f+"
    "zY5OUQkoWhFbxqO0JKxIUmeW0t0lbNW3vfx++jJIBj6HB2b5fRgVVDK0qCFf4nW0XwMoYxuYXkU40"
    "yaB4WlBiASNJzRC9imWPUeaXgjW0Cq4uqK1+QVMhl9tPnQnHAwsRRtt2s9gxyXeR/Xq+zvt1Z4AHZ7"
    "M5JwQZDx4tFCV7IYsISHIINDvI9gXazJ1n0GnpzYpQg8GgZsi4GGpm+NJeaeNULE5W674mgJDeDXVr"
    "+e4Rds7gHwcSqeqei0dnN5uEa3EWJaymdO+WZCGDqv7tPOrAq87lno2WC9eJB1O8vbyxciKVQDau4m"
    "T8JYmdaOFTsWbdmTF2thq1ASH/Zq1hZWw0zg2+Cw20JGpA7DFRH/Du2AXZCUJxKMHjlSgF69ytsFAs"
    "THol5/GlzbWdWkkHKxeTwvTyWWjYjaAZJHdp8CXmcgpKLpLztUDZKheI1UviKIrSdrS1x1N4LeXwt"
    "XYk7X/uvtxNtqOwg5UQ=="
)
SIGNING_FINGERPRINT_SHA256 = "SHA256:LNuExn82n/p//myB4cc0pYv7yA1rzlbeI8Qyi6mXM3U"
SIGNING_PRINCIPAL = "mihir"
SIGNING_NAMESPACE = "redco-stage-d1-support-v13-execute-handoff-v2-signing"
ALLOWED_SIGNERS_SHA256 = "ff2a200af8bbdf8aea8d724c804dc9ba638534a33f490473ddcc668b449a9dd4"

LAUNCH_AUTH_RELATIVE = "configs/stage-d/v13-draft/stage-d1-support-v13-launch-authorization-v1.json"
LAUNCH_DATASET_RELATIVE = "datasets/stage-d/qasper-support-v13-launch-input-v1.jsonl"
LAUNCH_PLAN_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-v13-launch-collection-plan-v1.json"
)
LAUNCH_SOURCE_EVAL_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-v13-launch-source-eval-v1.toml"
)
LAUNCH_PROTOCOL_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-v13-launch-protocol-manifest-v1.json"
)
LAUNCH_BRANCH_RUNTIME_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-v13-launch-branch-runtime-v1.toml"
)
LAUNCH_AUDIT_RELATIVE = "reports/stage-d1-support-v13-launch-audit-v1.json"
RETURNING_ROOT_CONTRACT_RELATIVE = (
    "reports/stage-d1-support-v13-returning-root-correspondence-contract-v1.json"
)
LAUNCH_ATTEMPT_RELATIVE = "runs/stage-d/stage-d1-support-v13-launch/attempt-v1.json"
LAUNCH_TERMINAL_RELATIVE = "reports/stage-d1-support-v13-launch-terminal-v1.json"
LAUNCH_SUPPORT_REPORT_RELATIVE = "reports/stage-d1-support-v13-support-report-v1.json"
LAUNCH_PREFLIGHT_SNAPSHOT_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-launch/preflight-snapshot-v1.json"
)
LAUNCH_PRIME_OBSERVATION_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-launch/prime-observation-v1.json"
)
LAUNCH_POD_OBSERVATION_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-launch/pod-observation-v1.json"
)

# The TOML serializer omits ``reasoning_effort`` because its authenticated
# default is JSON null.  Every other request-owned field is explicit here;
# the persisted projection is completed by the production sampler contract.
LAUNCH_SAMPLING: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 1.0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "seed": 1,
    "max_tokens": 768,
    "n": 1,
    "tool_choice": "auto",
    "parallel_tool_calls": False,
}
LAUNCH_PERSISTED_SAMPLING: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 1.0,
    "reasoning_effort": None,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "seed": 1,
    "max_tokens": 768,
    "n": 1,
    "tool_choice": "auto",
    "parallel_tool_calls": False,
}
LAUNCH_PROVISIONING_LEDGER_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-launch/provisioning-ledger-v1.json"
)
LAUNCH_HANDOFF_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-launch/execute-handoff-v2.json"
)
LAUNCH_HANDOFF_SIGNATURE_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-launch/execute-handoff-v2.sig"
)
LAUNCH_PROVISION_CLAIM_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-launch/provision-claim-v2.json"
)
LAUNCH_KNOWN_HOSTS_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-launch/known_hosts-v2"
)
LAUNCH_RUNTIME_MANIFEST_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-launch/runtime/execution-manifest-v1.json"
)
LAUNCH_LIFECYCLE_RELATIVE = "reports/stage-d1-support-v13-launch-lifecycle-v1.json"

# These are the only pre-existing production owners changed by the launch
# bundle.  Their exact source bytes are included in the authorization/audit
# bindings and in the post-commit diff allowlist.
LAUNCH_OWNER_PATHS = frozenset(
    {
        "src/redco/analysis/stage_d_live_observer.py",
        "environments/redco_evidence_selection_v2/redco_evidence_selection_v2/source_env.py",
        "environments/redco_evidence_selection_v2/redco_evidence_selection_v2/scientific_campaign_driver.py",
    }
)

# Execution-owner bytes are bound in the launch authorization independently of
# the frozen scientific roots.  These are read-only source bindings; the
# launch commit allowlist remains the smaller set declared below.
LAUNCH_CODE_PATHS = frozenset(
    {
        "scripts/run_stage_d_source_collection.py",
        "scripts/run_stage_d_scientific_campaign.py",
        "scripts/run_stage_d_v13_support.py",
        "scripts/run_stage_d_v13_launch_observation.py",
        "scripts/run_stage_d_v13_local_orchestrator.py",
        "scripts/run_stage_d_v13_remote_bootstrap.py",
        "src/redco/analysis/stage_d_v13_launch_lifecycle.py",
        "src/redco/analysis/stage_d_v13_launch_observations.py",
        "src/redco/analysis/stage_d_source_contracts.py",
        "src/redco/analysis/stage_d_source_producer.py",
        "src/redco/analysis/stage_d_receipt_ledger.py",
        "src/redco/analysis/stage_d_returning_root_contract.py",
        "src/redco/analysis/stage_d_returning_root_correspondence.py",
        "src/redco/analysis/stage_d_v13_support_launch.py",
        "src/redco/analysis/stage_d_v13_support_launch_runtime.py",
    }
)

EXPECTED_COLLECTION_RUNNER_SHA256 = (
    "5f77273341bbf16c34aa2bb23bb88b60418006f14ae6361a8363148b706c48e8"
)
EXPECTED_PRODUCER_SHA256 = "45ececee92acd46857c37e0d7ec08c036604789f308e3fc1e4629029e7a7090c"
EXPECTED_SAMPLING_SHA256 = "819222244a81565a67331826be3dd362e14e1481043d60fccb569551a4471f6d"
EXPECTED_SUPPORT_RULES_SHA256 = FROZEN_SUPPORT_RULES_SHA256
EXPECTED_DEPENDENCY_STACK_SHA256 = (
    "cda524c6ecea9821b1e36290da64df465aa46fad9ec174881c24d3dc895b2831"
)
EXPECTED_RENDERER_TREE_SHA256 = "bd43d515c12dcaa1e1c0279941a1397d4ffba31a1557d6d7342a1322b195fcc4"
EXPECTED_VERIFIER_TREE_SHA256 = "9dcf9e98dea73c2487d2165cd6cae35dc61fb66e00d377d85d5466886b3ea4e0"

ACTION_CONFIG_RELATIVE = "configs/stage-d/stage-d1-action-closure-corpus-v2.json"
ACTION_AUDIT_RELATIVE = "reports/stage-d1-action-closure-corpus-audit-v2.json"
COLLECTION_RUNNER_RELATIVE = "scripts/run_stage_d_source_collection.py"
PRODUCER_RELATIVE = "src/redco/analysis/stage_d_source_producer.py"
DEPENDENCY_RELATIVE = "src/redco/analysis/stage_d_dependency_stack.py"
DEPENDENCY_MANIFEST_RELATIVE = "configs/stage-d/stage-d1-dependency-stack-v12.json"
PROTOCOL_AUDIT_ROOT_RELATIVE = "reports/stage-d1-support-v13-protocol-audit-v1.json"
DEPENDENCY_SOURCE_SHA256 = "7feba4914177d9475ecc936447cd5b7aa0a6e9df891fcf7592fa84ccb9c4c95e"

FROZEN_ROOT_HASHES: dict[str, str] = {
    PROTOCOL_RELATIVE: ("65734f3dc5caeb1866e25b535d5b91d17ffbc434b69fbe0baf5efe63d339145b"),
    PROTOCOL_AUDIT_ROOT_RELATIVE: (
        "6df3b0e98aa0ca27b72c7abd443cfbf003f7f99e0ce1880ec2e8b1fd3801d2f3"
    ),
    CANDIDATE_RELATIVE: ("3df14acf9bf5f71736511aa9115f5e49ceab14a191bc1b634e3f82f21ca3f4a1"),
    COMPOSITION_RELATIVE: ("3cb26d9aec634e96fb342f87ea807711ed943a64073a9e37c5b7a546294638bc"),
    ACTION_CONFIG_RELATIVE: "50152ebbaea6cecce63c167c13d56050c4feb50782f838d69ea34840b29670c0",
    ACTION_AUDIT_RELATIVE: "60631a5153c2434682642f5aecaf5f55e61f368c8b96d721982eea7c9c158646",
    COLLECTION_RUNNER_RELATIVE: EXPECTED_COLLECTION_RUNNER_SHA256,
    PRODUCER_RELATIVE: EXPECTED_PRODUCER_SHA256,
    DEPENDENCY_MANIFEST_RELATIVE: EXPECTED_DEPENDENCY_STACK_SHA256,
    FROZEN_SUPPORT_RULES_RELATIVE: EXPECTED_SUPPORT_RULES_SHA256,
    SELECTION_CLAIM_RELATIVE: SELECTION_CLAIM_SHA256,
    SELECTION_RECEIPT_RELATIVE: SELECTION_RECEIPT_SHA256,
    SELECTION_MANIFEST_RELATIVE: SELECTION_MANIFEST_SHA256,
    V12_ARCHIVE_RELATIVE: V12_ARCHIVE_SHA256,
    V12_EVIDENCE_MANIFEST_RELATIVE: V12_EVIDENCE_MANIFEST_SHA256,
    V12_TERMINAL_REPORT_RELATIVE: V12_TERMINAL_REPORT_SHA256,
    V12_FINALIZATION_AUDIT_RELATIVE: V12_FINALIZATION_AUDIT_SHA256,
    V12_PREREG_RELATIVE: V12_PREREG_SHA256,
    V12_PROTOCOL_RELATIVE: V12_PROTOCOL_SHA256,
    V12_SOURCE_EVAL_RELATIVE: V12_SOURCE_EVAL_SHA256,
}
FROZEN_ROOT_HASHES.update(AUTHENTICATED_PREDECESSOR_HASHES)

POLICY_FILES: dict[str, str] = {
    "configs/stage-d/stage-d1-base-model-manifest.json": (
        "45ed89b1c482b5189c6ca86db128c1d22782f24ac6c578fb19fbaa4a15d90dbc"
    ),
    "reports/stage-d0-scaffold-step8-adapter-manifest-v1.json": (
        "1b3cc7f5e374919f90f41b537f398ec59c12b1c31e25482b5bcf6f3b7bc1ddbe"
    ),
    "configs/stage-d/stage-d1-tokenizer-manifest.json": (
        "65676a0c390086672375cf9acbe2698a461e2f403e1c5e7556647d114c204a38"
    ),
    "configs/stage-d/stage-d1-renderer-manifest.json": (
        "a9e6b8823af5e8e89c65e1382af0d07cfad1bd32ce6fe17bb72b994b8a84366c"
    ),
    "configs/stage-d/stage-d1-sampler-conformance-manifest.json": (
        "e5ba2716be1d977cbf4047958b948f170d6d88c75c768a7c8fe1b8c1c8a9d196"
    ),
}

RUNTIME = {"python": SUPPORTED_PYTHON, "pyarrow": SUPPORTED_PYARROW, "datasets": SUPPORTED_DATASETS}
EXPECTED_UV_LOCK_SHA256 = (
    "d98a2958c7d73cb4d300e40d3b80cfc49a7f6d11f0e76a8a181b932f58e68f4e"
)
OFFLINE_RLM_BINDINGS: dict[str, str] = {
    "checkout_archive_path": "/workspace/redco/.runtime/stage-d/rlm-patched-v1.tar",
    "checkout_archive_sha256": "0db97a244b88b5186a27fe1af14fa33a89b69e39a68a064cf7d10bb6f3084669",
    "checkout_uv_path": "/workspace/redco/.runtime/stage-d/uv",
    "checkout_uv_sha256": "da15297d6879b2cfbe5ea3cb03725c1613d51ba72892cc996468d871f0a532fb",
    "checkout_cache_archive_path": "/workspace/redco/.runtime/stage-d/rlm-cache-v2.tar.gz",
    "checkout_cache_archive_sha256": (
        "fb1ff3fe82a5109db0662092b344e0069ada5365f137d01b3d7010b84d7e37be"
    ),
    "checkout_uv_lock_sha256": "d98a2958c7d73cb4d300e40d3b80cfc49a7f6d11f0e76a8a181b932f58e68f4e",
    "checkout_launcher_path": "/workspace/redco/.runtime/stage-d/rlm-wrapper",
    "checkout_launcher_sha256": "7f6d55f352d521a4d34c675e2bc5cb9581fde2fb635c826a9486470aa0de6cd4",
}
ASSET_BINDING_SCHEMA_VERSION = 1
ASSET_ARTIFACT_ROOT_LOCATOR = "operator-supplied-artifact-root"


def asset_binding_contract(root: Path) -> dict[str, dict[str, str]]:
    """Return the signed local-locator/remote-destination asset contract.

    ``artifact-store:`` locators are resolved only against the explicit local
    artifact root supplied by the operator.  ``repository:`` locators are
    resolved only against the authenticated checkout.  No Linux destination
    is ever reused as a Windows/local source path.
    """

    root = root.resolve()
    result: dict[str, dict[str, str]] = {}

    def add(
        name: str,
        locator: str,
        destination: str,
        digest: str,
    ) -> None:
        if name in result:
            raise ValueError(f"duplicate launch asset binding: {name}")
        result[name] = {
            "local_locator": locator,
            "remote_destination": destination,
            "sha256": digest,
        }

    for relative, digest in sorted(POLICY_FILES.items()):
        add(
            f"repository:{relative}",
            f"repository:{relative}",
            f"/workspace/redco/{relative}",
            digest,
        )
    add(
        "repository:uv.lock",
        "repository:uv.lock",
        "/workspace/redco/uv.lock",
        EXPECTED_UV_LOCK_SHA256,
    )

    # Generated launch artifacts are already authenticated by the bundle
    # manifest and are present in the checked-out commit.  Including their
    # hashes here would make authorization circular (protocol/source config
    # hashes include the authorization hash).  Bind only the immutable
    # repository inputs transferred independently of that generated set.
    for relative in sorted(
        {DEPENDENCY_MANIFEST_RELATIVE, FROZEN_SUPPORT_RULES_RELATIVE}
    ):
        root_digest = FROZEN_ROOT_HASHES.get(relative)
        if root_digest is None:
            root_digest = sha256_bytes((root / relative).read_bytes())
        if not isinstance(root_digest, str):
            raise ValueError("launch asset hash is not a string")
        add(
            f"repository:{relative}",
            f"repository:{relative}",
            f"/workspace/redco/{relative}",
            root_digest,
        )

    dependency = json.loads(
        _read_bound(root, DEPENDENCY_MANIFEST_RELATIVE, EXPECTED_DEPENDENCY_STACK_SHA256)
    )
    components = dependency.get("components", [])
    if not isinstance(components, list):
        raise ValueError("dependency stack asset list is invalid")
    for component in components:
        if not isinstance(component, dict) or not isinstance(component.get("patches"), list):
            raise ValueError("dependency stack patch list is invalid")
        for patch in component["patches"]:
            if not isinstance(patch, dict) or not isinstance(patch.get("name"), str):
                raise ValueError("dependency stack patch binding is invalid")
            relative = f"patches/{patch['name']}"
            patch_digest = patch.get("sha256")
            if not isinstance(patch_digest, str):
                raise ValueError("dependency stack patch hash is invalid")
            add(
                f"repository:{relative}",
                f"repository:{relative}",
                f"/workspace/redco/{relative}",
                patch_digest,
            )

    base = json.loads(
        _read_bound(
            root,
            "configs/stage-d/stage-d1-base-model-manifest.json",
            POLICY_FILES["configs/stage-d/stage-d1-base-model-manifest.json"],
        )
    )
    base_destination = str(base["base_model"])
    base_prefix = "artifact-store:base-model/"
    for name, item in sorted(base["files"].items()):
        if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
            raise ValueError("base model asset binding is invalid")
        add(
            f"base-model:{name}",
            f"{base_prefix}{name}",
            f"{base_destination.rstrip('/')}/{name}",
            item["sha256"],
        )
    adapter = json.loads(
        _read_bound(
            root,
            "reports/stage-d0-scaffold-step8-adapter-manifest-v1.json",
            POLICY_FILES["reports/stage-d0-scaffold-step8-adapter-manifest-v1.json"],
        )
    )
    adapter_root = str(base["adapter"])
    add(
        "adapter:model",
        "artifact-store:adapter/adapter_model.safetensors",
        f"{adapter_root.rstrip('/')}/adapter_model.safetensors",
        str(base["adapter_model_sha256"]),
    )
    archive = adapter.get("archive")
    archive_sha = adapter.get("archive_sha256")
    if not isinstance(archive, str) or not isinstance(archive_sha, str):
        raise ValueError("adapter archive binding is invalid")
    add(
        "adapter:archive",
        "artifact-store:archives/selected-adapter.tar.gz",
        "/workspace/redco/.runtime/stage-d1/adapter.tar.gz",
        archive_sha,
    )
    for name, value in sorted(OFFLINE_RLM_BINDINGS.items()):
        if not name.endswith("_path"):
            continue
        offline_digest = OFFLINE_RLM_BINDINGS.get(name.replace("_path", "_sha256"))
        if offline_digest is None:
            raise ValueError(f"offline asset hash is missing: {name}")
        if not isinstance(offline_digest, str):
            raise ValueError(f"offline asset hash is invalid: {name}")
        locator_name = name.removeprefix("checkout_").removesuffix("_path")
        add(
            f"offline:{name}",
            f"artifact-store:runtime/{locator_name}",
            value,
            offline_digest,
        )
    for relative, digest in (
        (
            "scripts/merge_stage_c_warmstart.py",
            "06046f345d0ac29e0919d43fce50c2a6ae20a29d03be215785323413e28d0416",
        ),
        (
            "scripts/stage_c_lora.py",
            "0647f6ed77e11757fe25279e5f29e560ddd3bd8a4c1fb8db26b944553d52d846",
        ),
    ):
        add(
            f"repository:{relative}",
            f"repository:{relative}",
            f"/workspace/redco/{relative}",
            digest,
        )
    return dict(sorted(result.items()))


def resolve_local_asset_locator(
    repository: Path,
    artifact_root: Path,
    locator: str,
) -> Path:
    """Resolve a signed locator without interpreting remote Linux paths."""

    if locator.startswith("repository:"):
        base, relative = repository.absolute(), locator.removeprefix("repository:")
    elif locator.startswith("artifact-store:"):
        base, relative = artifact_root.absolute(), locator.removeprefix("artifact-store:")
    else:
        raise ValueError("launch asset locator has an unsupported namespace")
    if _is_link_or_reparse(base):
        raise ValueError("launch asset root is linked or reparse-backed")
    relative_path = Path(relative)
    if relative_path.is_absolute() or any(
        part in {"", ".", ".."} for part in relative_path.parts
    ):
        raise ValueError("launch asset locator escapes its approved root")
    current = base
    for part in relative_path.parts:
        current /= part
        if _is_link_or_reparse(current):
            raise ValueError("launch asset path contains a linked or reparse ancestor")
    candidate = current.resolve(strict=False)
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise ValueError("launch asset locator escapes its approved root") from error
    return candidate


def _is_link_or_reparse(path: Path) -> bool:
    """Reject symlink/junction/reparse points before resolving an asset path."""

    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(info, "st_file_attributes", 0) & reparse)
LAUNCH_BUNDLE_PATHS = frozenset(
    {
        LAUNCH_AUTH_RELATIVE,
        LAUNCH_DATASET_RELATIVE,
        LAUNCH_PLAN_RELATIVE,
        LAUNCH_SOURCE_EVAL_RELATIVE,
        LAUNCH_PROTOCOL_RELATIVE,
        LAUNCH_BRANCH_RUNTIME_RELATIVE,
        LAUNCH_AUDIT_RELATIVE,
        RETURNING_ROOT_CONTRACT_RELATIVE,
        "scripts/build_stage_d_v13_support_launch.py",
        "scripts/run_stage_d_v13_support.py",
        "scripts/run_stage_d_source_collection.py",
        "scripts/run_stage_d_scientific_campaign.py",
        "src/redco/analysis/stage_d_source_contracts.py",
        "src/redco/analysis/stage_d_source_producer.py",
        "src/redco/analysis/stage_d_receipt_ledger.py",
        "src/redco/analysis/stage_d_returning_root_contract.py",
        "src/redco/analysis/stage_d_returning_root_correspondence.py",
        "src/redco/analysis/stage_d_v13_support_launch.py",
        "src/redco/analysis/stage_d_v13_support_launch_runtime.py",
        "src/redco/analysis/stage_d_v13_launch_observations.py",
        "src/redco/analysis/stage_d_v13_launch_lifecycle.py",
        "scripts/run_stage_d_v13_launch_observation.py",
        "scripts/run_stage_d_v13_local_orchestrator.py",
        "scripts/run_stage_d_v13_remote_bootstrap.py",
        "tests/test_stage_d_v13_support_launch.py",
        "tests/test_stage_d_source_finalization_integration.py",
        "tests/test_stage_d_source_producer.py",
        "tests/test_stage_d_returning_root_correspondence.py",
        *LAUNCH_OWNER_PATHS,
    }
)
OUTPUT_PATHS = (
    LAUNCH_AUTH_RELATIVE,
    LAUNCH_DATASET_RELATIVE,
    LAUNCH_PLAN_RELATIVE,
    LAUNCH_SOURCE_EVAL_RELATIVE,
    LAUNCH_PROTOCOL_RELATIVE,
    LAUNCH_BRANCH_RUNTIME_RELATIVE,
    LAUNCH_AUDIT_RELATIVE,
    RETURNING_ROOT_CONTRACT_RELATIVE,
)
ATTEMPT_PATHS = (
    LAUNCH_ATTEMPT_RELATIVE,
    LAUNCH_PROVISIONING_LEDGER_RELATIVE,
    LAUNCH_HANDOFF_RELATIVE,
    LAUNCH_HANDOFF_SIGNATURE_RELATIVE,
    LAUNCH_PROVISION_CLAIM_RELATIVE,
    LAUNCH_KNOWN_HOSTS_RELATIVE,
    LAUNCH_TERMINAL_RELATIVE,
    LAUNCH_SUPPORT_REPORT_RELATIVE,
    LAUNCH_LIFECYCLE_RELATIVE,
)


def _git_stdout(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    )
    return bytes(result.stdout)


def _sha256_json(value: object) -> str:
    return cast(str, sha256_bytes(canonical_json_bytes(value)))


def _read_bound(root: Path, relative: str, expected: str) -> bytes:
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"launch input is missing: {relative}")
    value = path.read_bytes()
    if sha256_bytes(value) != expected:
        raise ValueError(f"launch input hash differs: {relative}")
    return value


def _git_value(root: Path, *args: str) -> str:
    return _git_stdout(root, *args).decode("utf-8").strip()


def _status_paths(root: Path) -> tuple[tuple[str, str], ...]:
    raw = _git_stdout(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    records = raw.split(b"\0")
    parsed: list[tuple[str, str]] = []
    for record in records:
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise ValueError("git status record is malformed")
        path = record[3:].decode("utf-8", "surrogateescape")
        parsed.append((record[:2].decode("ascii"), path))
    return tuple(parsed)


def _authenticate_worktree(root: Path, *, require_post_commit: bool) -> None:
    for status, path in _status_paths(root):
        if status in {" M", " m"} and path == "external/prime-rl":
            continue
        if not require_post_commit and status == "??" and path in LAUNCH_BUNDLE_PATHS:
            continue
        if (
            not require_post_commit
            and status == " M"
            and path in LAUNCH_BUNDLE_PATHS
            and _git_path_exists(root, PARENT_COMMIT, path)
        ):
            continue
        raise ValueError(f"launch worktree is not an authenticated clean view: {status} {path}")


def _git_path_exists(root: Path, commit: str, relative: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{commit}:{relative}"],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _authenticate_committed_diff(root: Path, head: str) -> None:
    lines = _git_value(root, "diff", "--name-status", "--no-renames", PARENT_COMMIT, head)
    entries: list[tuple[str, str]] = []
    for line in lines.splitlines():
        status, separator, path = line.partition("\t")
        if not separator or not status or not path:
            raise ValueError("launch commit diff is malformed")
        entries.append((status, path))
    expected_status = {
        path: ("M" if _git_path_exists(root, PARENT_COMMIT, path) else "A")
        for path in LAUNCH_BUNDLE_PATHS
    }
    if {path for _status, path in entries} != set(expected_status):
        raise ValueError("launch commit diff is not the exact bundle allowlist")
    if {path: status for status, path in entries} != expected_status:
        raise ValueError("launch bundle commit has an unexpected path status")


def _authenticate_parent(root: Path, *, require_post_commit: bool = False) -> str:
    head = _git_value(root, "rev-parse", "HEAD")
    if _git_value(root, "rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise ValueError("launch baseline tree differs from the reviewed baseline")
    if require_post_commit:
        if head == PARENT_COMMIT:
            raise ValueError("launch execution requires the committed bundle child")
        parents = _git_value(root, "rev-list", "--parents", "-n", "1", head).split()
        if len(parents) != 2 or parents[0] != head:
            raise ValueError("launch execution rejects merge commits")
        if _git_value(root, "rev-parse", "HEAD^") != PARENT_COMMIT:
            raise ValueError("launch execution requires a direct child of the reviewed baseline")
        _authenticate_committed_diff(root, head)
        _authenticate_worktree(root, require_post_commit=True)
        return "committed_direct_child"
    if head != PARENT_COMMIT or _git_value(root, "rev-parse", "HEAD^{tree}") != PARENT_TREE:
        raise ValueError("pre-commit bundle verification requires the reviewed baseline")
    _authenticate_worktree(root, require_post_commit=False)
    return "precommit_non_authorizing"


def _authenticate_roots(root: Path, *, require_post_commit: bool = False) -> dict[str, str]:
    _authenticate_parent(root, require_post_commit=require_post_commit)
    actual: dict[str, str] = {}
    for relative, expected in FROZEN_ROOT_HASHES.items():
        actual[relative] = sha256_bytes(_read_bound(root, relative, expected))
    producer = root / PRODUCER_RELATIVE
    if sha256_bytes(producer.read_bytes()) != EXPECTED_PRODUCER_SHA256:
        raise ValueError("producer source hash differs from the reviewed launch binding")
    dependency = live_owner_dependency_payload(root)
    components = {str(item["name"]): item for item in dependency["components"]}
    if (
        components["renderers"]["post_tree_sha256"] != EXPECTED_RENDERER_TREE_SHA256
        or components["verifiers"]["post_tree_sha256"] != EXPECTED_VERIFIER_TREE_SHA256
    ):
        raise ValueError("dependency post-tree binding differs from the reviewed launch binding")
    sampling = sampling_contract_binding(root)
    if sampling["sha256"] != EXPECTED_SAMPLING_SHA256:
        raise ValueError("sampling contract binding differs from the reviewed launch binding")
    for relative, expected in POLICY_FILES.items():
        actual[relative] = sha256_bytes(_read_bound(root, relative, expected))
    for relative in sorted(LAUNCH_OWNER_PATHS):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"launch owner source is missing: {relative}")
        actual[relative] = sha256_bytes(path.read_bytes())
    for relative in sorted(LAUNCH_CODE_PATHS):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"launch code source is missing: {relative}")
        actual[relative] = sha256_bytes(path.read_bytes())
    actual[DEPENDENCY_RELATIVE] = sha256_bytes(
        _read_bound(root, DEPENDENCY_RELATIVE, DEPENDENCY_SOURCE_SHA256)
    )
    actual["dependency_stack_contract_sha256"] = EXPECTED_DEPENDENCY_STACK_SHA256
    actual["renderer_post_tree_sha256"] = EXPECTED_RENDERER_TREE_SHA256
    actual["verifier_post_tree_sha256"] = EXPECTED_VERIFIER_TREE_SHA256
    actual["sampling_contract_sha256"] = sampling["sha256"]
    return actual


def _candidate_row(root: Path) -> dict[str, Any]:
    payload = json.loads(
        _read_bound(root, CANDIDATE_RELATIVE, FROZEN_ROOT_HASHES[CANDIDATE_RELATIVE])
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("candidate"), dict):
        raise ValueError("candidate artifact does not contain its row projection")
    source = payload.get("source")
    candidate = cast(dict[str, Any], payload["candidate"])
    rollout = payload.get("fresh_support_rollout")
    if (
        not isinstance(source, dict)
        or not isinstance(rollout, dict)
        or source.get("ordinal") != CANDIDATE_SOURCE_ORDINAL
        or source.get("paper_id") != CANDIDATE_PAPER_ID
        or source.get("example_id") != CANDIDATE_EXAMPLE_ID
        or source.get("question_index") != CANDIDATE_QUESTION_INDEX
        or source.get("row_sha256") != CANDIDATE_ROW_SHA256
        or set(candidate)
        != {
            "answer_type",
            "example_id",
            "paper",
            "paper_id",
            "question",
            "reference_evidence",
            "split",
            "title",
        }
        or candidate.get("paper_id") != CANDIDATE_PAPER_ID
        or candidate.get("example_id") != CANDIDATE_EXAMPLE_ID
        or candidate.get("split") != "successor_support"
        or source.get("selection_receipt_sha256") != SELECTION_RECEIPT_SHA256
        or source.get("selection_evidence_manifest_sha256") != SELECTION_MANIFEST_SHA256
        or source.get("selection_claim_sha256") != SELECTION_CLAIM_SHA256
        or rollout.get("selection_address_sha256") != CANDIDATE_SELECTION_ADDRESS_SHA256
        or not isinstance(rollout.get("address_sha256"), str)
        or rollout.get("address_sha256") == CANDIDATE_SELECTION_ADDRESS_SHA256
        or type(rollout.get("seed")) is not int
    ):
        raise ValueError("candidate artifact is outside the launch contract")
    return candidate


def _retained_support(root: Path) -> tuple[list[dict[str, Any]], list[bytes]]:
    raw = _read_bound(root, RETAINED_SUPPORT_RELATIVE, RETAINED_SUPPORT_SHA256)
    rows: list[dict[str, Any]] = []
    lines: list[bytes] = []
    for line in raw.splitlines(keepends=True):
        stripped = line.rstrip(b"\r\n")
        if not stripped:
            continue
        value = json.loads(stripped)
        if not isinstance(value, dict):
            raise ValueError("retained support JSONL contains a non-object row")
        if value.get("split") == "successor_support":
            rows.append(cast(dict[str, Any], value))
            lines.append(line if line.endswith((b"\n", b"\r")) else line + b"\n")
    if len(rows) != 63 or len({row.get("example_id") for row in rows}) != 63:
        raise ValueError("retained support input does not contain exactly 63 unique rows")
    return rows, lines


def _build_collection_inputs(root: Path) -> tuple[bytes, bytes, StageDCollectionPlan, str]:
    retained, retained_lines = _retained_support(root)
    candidate = _candidate_row(root)
    rows = [*retained, candidate]
    if len(rows) != 64 or any(row.get("split") != "successor_support" for row in rows):
        raise ValueError("launch support input is not exactly 64 support rows")
    if len({row.get("paper_id") for row in rows}) != 64:
        raise ValueError("support input paper identities are not unique")
    task_data = [
        {
            "scientific_group_id": derive_scientific_group_id(
                namespace=SCIENTIFIC_NAMESPACE,
                example_id=str(row["example_id"]),
            ),
            "example_id": str(row["example_id"]),
            "rollout_slot": 0,
        }
        for row in rows
    ]
    plan = StageDCollectionPlan.build(task_data, master_seed=MASTER_SEED)
    dataset = b"".join(retained_lines) + canonical_json_bytes(candidate) + b"\n"
    return dataset, plan.to_bytes(), plan, sha256_bytes(dataset)


def _toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise TypeError(f"unsupported TOML scalar: {type(value).__name__}")


def _toml_bytes(value: Mapping[str, Any]) -> bytes:
    lines: list[str] = []

    def emit_table(table: Mapping[str, Any], prefix: str = "") -> None:
        scalars = [(key, item) for key, item in table.items() if not isinstance(item, dict)]
        children = [(key, item) for key, item in table.items() if isinstance(item, dict)]
        for key, item in scalars:
            lines.append(f"{key} = {_toml_scalar(item)}")
        for key, child in children:
            if lines and lines[-1] != "":
                lines.append("")
            name = f"{prefix}.{key}" if prefix else key
            lines.append(f"[{name}]")
            emit_table(cast(Mapping[str, Any], child), name)

    emit_table(value)
    return "\n".join(lines).encode("utf-8")


def _source_eval(
    *,
    dataset_sha256: str,
    plan_sha256: str,
    authorization_sha256: str,
    protocol_anchor_sha256: str,
) -> bytes:
    genesis_config_sha256 = _sha256_json(
        {
            "domain": "redco-stage-d1-support-v13-launch-genesis-v1",
            "authorization_sha256": authorization_sha256,
        }
    )
    value: dict[str, Any] = {
        "model": "/workspace/models/stage-d1-merged",
        "num_tasks": 64,
        "num_rollouts": 1,
        "shuffle": False,
        "max_concurrent": 1,
        "rich": False,
        "server": False,
        "push": False,
        "output_dir": "runs/stage-d/stage-d1-support-v13/source-eval",
        "client": {
            "type": "train",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key_var": "VLLM_API_KEY",
            "pool_size": 1,
            "renderer_model_name": "Qwen/Qwen3-4B-Instruct-2507",
            "renderer": {"name": "auto"},
        },
        "sampling": {
            **LAUNCH_SAMPLING,
            "extra_body": {"cache_salt": "placeholder-only-before-episode-addressing"},
        },
        "env": {
            "id": "redco-evidence-selection-v2",
            "ledger_path": "runs/stage-d/stage-d1-support-v13/ledger",
            "artifact_path": "runs/stage-d/stage-d1-support-v13/source-artifacts",
            "master_seed": MASTER_SEED,
            "preregistration_sha256": PROTOCOL_ROOT_SHA256,
            "source_sha256": dataset_sha256,
            "runtime_sha256": _sha256_json(RUNTIME),
            "config_sha256": genesis_config_sha256,
            "protocol_manifest_sha256": protocol_anchor_sha256,
            "support_rules_sha256": EXPECTED_SUPPORT_RULES_SHA256,
            "checkpoint_id": "/workspace/models/stage-d1-merged",
            "base_model_manifest_path": "configs/stage-d/stage-d1-base-model-manifest.json",
            "base_model_manifest_sha256": POLICY_FILES[
                "configs/stage-d/stage-d1-base-model-manifest.json"
            ],
            "adapter_manifest_path": "reports/stage-d0-scaffold-step8-adapter-manifest-v1.json",
            "adapter_manifest_sha256": POLICY_FILES[
                "reports/stage-d0-scaffold-step8-adapter-manifest-v1.json"
            ],
            "tokenizer_manifest_path": "configs/stage-d/stage-d1-tokenizer-manifest.json",
            "tokenizer_manifest_sha256": POLICY_FILES[
                "configs/stage-d/stage-d1-tokenizer-manifest.json"
            ],
            "renderer_manifest_path": "configs/stage-d/stage-d1-renderer-manifest.json",
            "renderer_manifest_sha256": POLICY_FILES[
                "configs/stage-d/stage-d1-renderer-manifest.json"
            ],
            "sampler_conformance_manifest_path": (
                "configs/stage-d/stage-d1-sampler-conformance-manifest.json"
            ),
            "sampler_conformance_manifest_sha256": POLICY_FILES[
                "configs/stage-d/stage-d1-sampler-conformance-manifest.json"
            ],
            "resolved_agent_sampling_law_sha256": (
                "5f16a53881bf375cdadcd7cf85d44e3671bd100b18cfb5bfb9b2a34503658f4e"
            ),
            "resolved_train_client_sha256": (
                "94788de2b522b2187ad96e2cbd775cda9d81e487168f2844ac347cde5c6497f7"
            ),
            "branch_count": 4,
            "continuation_replicates": 1,
            "failure_reward": 0.0,
            "root_policy_turn_count": 2,
            "maximum_observed_root_policy_turn_count": 4,
            "maximum_captured_session_call_count": 16,
            "max_concurrent": 1,
            "retries": {"max_retries": 0},
            "taskset": {
                "id": "redco-evidence-selection-v2",
                "dataset_path": LAUNCH_DATASET_RELATIVE,
                "dataset_sha256": dataset_sha256,
                "split": "successor_support",
                "prompt_profile": "fewshot_scaffold_v2",
                "policy_checkpoint_id": "/workspace/models/stage-d1-merged",
                "scaffold_prompt_path": "configs/stage-d/stage-d0-scaffold-fewshot-v4.txt",
                "scaffold_prompt_sha256": (
                    "b27653e90f52a20f26ac79e3d0569275e9ba0ed2b07abbe06f060dd2486aee73"
                ),
                "scientific_group_namespace": SCIENTIFIC_NAMESPACE,
                "rollouts_per_task": 1,
            },
            "agent": {
                "model": "/workspace/models/stage-d1-merged",
                "max_turns": 8,
                "max_total_tokens": 8192,
                "retries": {"max_retries": 0},
                "timeout": {"setup": 180.0, "rollout": 900.0, "finalize": 120.0, "scoring": 120.0},
                "harness": {
                    "id": "rlm",
                    "version": "56218f33796ecbe465445bc43948886354fde196",
                    "max_depth": 1,
                    **OFFLINE_RLM_BINDINGS,
                    "runtime": {"type": "subprocess"},
                },
            },
        },
    }
    return _toml_bytes(value)


def _branch_runtime(
    source_eval: bytes,
) -> bytes:
    """Build the separate one-target runtime used by each branch owner."""

    try:
        value = tomllib.loads(source_eval.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("cannot derive branch runtime from source-eval config") from error
    value["num_tasks"] = 1
    value["num_rollouts"] = 1
    value["max_concurrent"] = 1
    value["output_dir"] = "runs/stage-d/stage-d1-support-v13/branch-runtime"
    return _toml_bytes(value)


PROTOCOL_ROOT_SHA256 = FROZEN_ROOT_HASHES[PROTOCOL_RELATIVE]


def launch_signing_identity() -> SigningIdentity:
    return SigningIdentity.from_payload(
        {
            "public_key_type": SIGNING_PUBLIC_KEY_TYPE,
            "public_key_base64": SIGNING_PUBLIC_KEY_BASE64,
            "fingerprint_sha256": SIGNING_FINGERPRINT_SHA256,
            "principal": SIGNING_PRINCIPAL,
            "namespace": SIGNING_NAMESPACE,
            "allowed_signers_sha256": ALLOWED_SIGNERS_SHA256,
        }
    )


def _authorization(
    *,
    dataset_sha256: str,
    plan_sha256: str,
    protocol_anchor_sha256: str,
    root_hashes: Mapping[str, str],
    asset_bindings: Mapping[str, Mapping[str, str]],
) -> bytes:
    payload: dict[str, Any] = {
        "schema_version": LAUNCH_SCHEMA_VERSION,
        "domain": LAUNCH_DOMAIN,
        "state": "support_launch_authorized_one_attempt",
        "parent": {"commit": PARENT_COMMIT, "tree": PARENT_TREE},
        "orchestrator_thread_id": ORCHESTRATOR_THREAD,
        "authorization": {
            "text_utf8": AUTHORIZATION_TEXT,
            "text_sha256": AUTHORIZATION_TEXT_SHA256,
            "text_byte_length": len(AUTHORIZATION_TEXT.encode("utf-8")),
        },
        "signing": launch_signing_identity().to_payload(),
        "execution_gate": {
            "surface": "post_commit_direct_child_only",
            "requires_parent_commit": PARENT_COMMIT,
            "precommit_build_non_authorizing": True,
            "precommit_execute_allowed": False,
        },
        "scope": {
            "support_launch_authorized": True,
            "launch_authorized": True,
            "provider_calls_authorized": True,
            "model_calls_authorized": True,
            "support_spend_authorized": True,
            "support_attempt_limit": 1,
            "support_papers": 64,
            "support_success_floor": 58,
            "branch_count_k": 4,
            "minimum_score_range": 0.05,
            "no_early_pass": True,
            "retry_after_provider_dispatch": False,
            "science_authorized": False,
            "training_authorized": False,
            "heldout_evaluation_authorized": False,
            "scientific_transition_authorized": False,
            "prime_gpu_scientific_launch_authorized": False,
        },
        "input_bindings": {
            "dataset": {
                "path": LAUNCH_DATASET_RELATIVE,
                "sha256": dataset_sha256,
                "rows": 64,
                "support_rows": 64,
                "science_rows": 0,
            },
            "collection_plan": {"path": LAUNCH_PLAN_RELATIVE, "sha256": plan_sha256},
            "protocol_anchor_sha256": protocol_anchor_sha256,
            "source_contract": {
                "repository": SOURCE_REPOSITORY,
                "logical_url": SOURCE_LOGICAL_URL,
                "revision": SOURCE_REVISION,
                "path": SOURCE_PATH,
                "sha256": SOURCE_SHA256,
                "schema_sha256": SOURCE_SCHEMA_SHA256,
                "row_count": SOURCE_ROW_COUNT,
                "semantic_commit": SOURCE_SEMANTIC_COMMIT,
                "accessed_during_bundle_build": False,
            },
            "assets": {
                "schema_version": ASSET_BINDING_SCHEMA_VERSION,
                "local_root_locator": ASSET_ARTIFACT_ROOT_LOCATOR,
                "entries": {
                    name: dict(value)
                    for name, value in sorted(asset_bindings.items())
                },
            },
        },
        "frozen_bindings": dict(sorted(root_hashes.items())),
        "obligations": {
            "wallet_min_before_support_usd": 30,
            "support_cap_usd": 12,
            "science_reserve_cap_usd": 16,
            "teardown_contingency_min_usd": 2,
            "resource": "one non-spot 2x48GB L40/L40S/RTX 6000 Ada",
            "hourly_cap_usd": 2,
            "wallet_and_hardware_check": "read_only_at_launch_fail_closed",
            "optional_redeployment": (
                "only before any provider POST, response, trace, source, score, or science evidence"
            ),
            "deadlines_seconds": {
                "setup": 180,
                "provider": 900,
                "episode": 900,
                "concurrent_child": 900,
                "scoring": 120,
                "finalizer": 120,
                "campaign": 21600,
                "pod": 21600,
                "termination": 2,
            },
            "attempt_record_before_execute": True,
            "stop_after_canonical_support_report": True,
        },
    }
    if payload["authorization"]["text_sha256"] != AUTHORIZATION_TEXT_SHA256:
        raise AssertionError("authorization text hash constant is inconsistent")
    return cast(bytes, canonical_json_bytes(payload))


def _policy_identity(root: Path) -> StageDPolicyIdentity:
    for relative, expected in POLICY_FILES.items():
        _read_bound(root, relative, expected)
    return StageDPolicyIdentity(
        checkpoint_id="/workspace/models/stage-d1-merged",
        base_model_manifest_sha256=POLICY_FILES[
            "configs/stage-d/stage-d1-base-model-manifest.json"
        ],
        adapter_manifest_sha256=POLICY_FILES[
            "reports/stage-d0-scaffold-step8-adapter-manifest-v1.json"
        ],
        tokenizer_manifest_sha256=POLICY_FILES["configs/stage-d/stage-d1-tokenizer-manifest.json"],
        renderer_manifest_sha256=POLICY_FILES["configs/stage-d/stage-d1-renderer-manifest.json"],
        sampler_conformance_manifest_sha256=POLICY_FILES[
            "configs/stage-d/stage-d1-sampler-conformance-manifest.json"
        ],
        resolved_agent_sampling_law_sha256="5f16a53881bf375cdadcd7cf85d44e3671bd100b18cfb5bfb9b2a34503658f4e",
        resolved_train_client_sha256="94788de2b522b2187ad96e2cbd775cda9d81e487168f2844ac347cde5c6497f7",
    )


def _protocol_manifest(
    root: Path,
    *,
    dataset_sha256: str,
    source_eval_sha256: str,
    branch_runtime_sha256: str,
    plan: StageDCollectionPlan,
    authorization_sha256: str,
) -> bytes:
    arm_names = ("stock", "branch-global", "local")
    arm_hashes = tuple(
        (name, _sha256_json({"arm": name, "protocol_sha256": PROTOCOL_ROOT_SHA256}))
        for name in arm_names
    )
    manifest = StageDProtocolManifest(
        preregistration_sha256=PROTOCOL_ROOT_SHA256,
        dependency_stack_sha256=EXPECTED_DEPENDENCY_STACK_SHA256,
        genesis_config_sha256=_sha256_json(
            {
                "domain": "redco-stage-d1-support-v13-launch-genesis-v1",
                "authorization_sha256": authorization_sha256,
            }
        ),
        master_seed_sha256=sha256_bytes(MASTER_SEED.encode("utf-8")),
        source_sha256=dataset_sha256,
        runtime_sha256=_sha256_json(RUNTIME),
        source_eval_config_sha256=source_eval_sha256,
        # The support attempt uses the authenticated source-evaluation config
        # for its branch/replay owner.  Science remains separately
        # unauthorized and is never entered by this bundle.
        scientific_eval_config_sha256=branch_runtime_sha256,
        heldout_eval_config_sha256=_sha256_json({"heldout_papers": 32, "authorized": False}),
        collection_plan_sha256=plan.plan_sha256,
        evaluation_plan_sha256=_sha256_json(
            {
                "domain": "redco-stage-d1-support-v13-evaluation-plan-v1",
                "k": 4,
                "source_order": True,
            }
        ),
        decision_rule_sha256="792decd5e6887efd494d2dba40d8ac00ff0fd243f72ff171a371d4cb7eb87306",
        support_rules_sha256=EXPECTED_SUPPORT_RULES_SHA256,
        reload_probe_sha256=_sha256_json(
            {"domain": "redco-stage-d1-support-v13-reload-probe-v1", "exact": True}
        ),
        shared_initialization_sha256=_sha256_json(
            {
                "domain": "redco-stage-d1-support-v13-shared-initialization-v1",
                "byte_identical": True,
            }
        ),
        objective_authorization_sha256=authorization_sha256,
        objective_binding_sha256s=arm_hashes,
        trainer_config_sha256s=arm_hashes,
        policy_identity=_policy_identity(root),
        arm_order=arm_names,
        branch_global_scope="within-source-group-all-target-branches-v1",
        trainer_step=1,
        seq_len=4096,
    )
    return cast(bytes, manifest.to_bytes())


def _validate_source_eval_contract(
    value: bytes,
    *,
    expected_num_tasks: int = 64,
    expected_output_dir: str = "runs/stage-d/stage-d1-support-v13/source-eval",
    branch_runtime: bool = False,
) -> None:
    try:
        parsed = tomllib.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("launch source-eval config is not valid TOML") from error
    expected_top_level = {
        "model",
        "num_tasks",
        "num_rollouts",
        "shuffle",
        "max_concurrent",
        "rich",
        "server",
        "push",
        "output_dir",
        "client",
        "sampling",
        "env",
    }
    if set(parsed) != expected_top_level:
        raise ValueError("launch source-eval fields differ")
    if (
        parsed.get("model") != "/workspace/models/stage-d1-merged"
        or parsed.get("num_tasks") != expected_num_tasks
        or parsed.get("num_rollouts") != 1
        or parsed.get("shuffle") is not False
        or parsed.get("max_concurrent") != 1
        or parsed.get("rich") is not False
        or parsed.get("server") is not False
        or parsed.get("push") is not False
        or parsed.get("output_dir") != expected_output_dir
    ):
        raise ValueError("launch source-eval runtime values differ")
    env = parsed.get("env")
    agent = env.get("agent") if isinstance(env, dict) else None
    harness = agent.get("harness") if isinstance(agent, dict) else None
    if not isinstance(harness, dict):
        raise ValueError("launch source-eval config lacks the RLM harness contract")
    for key, expected in OFFLINE_RLM_BINDINGS.items():
        if harness.get(key) != expected:
            raise ValueError(f"launch source-eval offline binding differs: {key}")
    if harness.get("runtime") != {"type": "subprocess"}:
        raise ValueError("launch source-eval runtime contract differs")
    sampling = parsed.get("sampling")
    expected_sampling = {
        **LAUNCH_SAMPLING,
        "extra_body": {"cache_salt": "placeholder-only-before-episode-addressing"},
    }
    if sampling != expected_sampling:
        raise ValueError("launch sampling configuration differs from the pinned request map")


def _validate_branch_runtime(value: bytes) -> None:
    try:
        parsed = tomllib.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("launch branch runtime is not valid TOML") from error
    if (
        parsed["num_tasks"] != 1
        or parsed["num_rollouts"] != 1
        or parsed["max_concurrent"] != 1
        or parsed["server"] is not False
        or parsed["push"] is not False
    ):
        raise ValueError("launch branch runtime must be exactly one-by-one")
    env = parsed.get("env")
    taskset = env.get("taskset") if isinstance(env, dict) else None
    if not isinstance(env, dict) or env.get("branch_count") != 4:
        raise ValueError("launch branch runtime must preserve K=4 in the environment contract")
    if not isinstance(taskset, dict) or taskset.get("rollouts_per_task") != 1:
        raise ValueError("launch branch runtime must preserve one rollout per target")
    _validate_source_eval_contract(
        value,
        expected_num_tasks=1,
        expected_output_dir="runs/stage-d/stage-d1-support-v13/branch-runtime",
        branch_runtime=True,
    )


def _validate_preflight_snapshot(
    value: bytes,
    *,
    expected_bundle_mode: str,
    expected_bundle_commit: str,
    expected_bundle_tree: str,
) -> dict[str, Any]:
    try:
        snapshot = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("launch preflight snapshot is not JSON") from error
    if not isinstance(snapshot, dict) or canonical_json_bytes(snapshot) != value:
        raise ValueError("launch preflight snapshot must be canonical JSON")
    expected_keys = {
        "schema_version",
        "domain",
        "state",
        "captured_at_epoch",
        "expires_at_epoch",
        "bundle",
        "resource",
        "runtime",
        "dependency",
        "assets",
        "outputs",
        "vllm",
        "offline",
    }
    if set(snapshot) != expected_keys:
        raise ValueError("launch preflight snapshot fields differ")
    if snapshot["schema_version"] != 1 or snapshot["domain"] != (
        "redco-stage-d1-support-v13-launch-preflight-v1"
    ) or snapshot["state"] != "verified":
        raise ValueError("launch preflight snapshot state is not verified")
    captured = snapshot["captured_at_epoch"]
    expires = snapshot["expires_at_epoch"]
    if type(captured) is not int or type(expires) is not int or expires <= captured:
        raise ValueError("launch preflight snapshot timestamps are invalid")
    if expires < int(time.time()):
        raise ValueError("launch preflight snapshot is stale")
    bundle = snapshot["bundle"]
    if not isinstance(bundle, dict) or set(bundle) != {"mode", "commit", "tree"}:
        raise ValueError("launch preflight bundle binding is incomplete")
    if (
        bundle["mode"] != expected_bundle_mode
        or bundle["commit"] != expected_bundle_commit
        or bundle["tree"] != expected_bundle_tree
    ):
        raise ValueError("launch preflight bundle mode differs from authentication mode")
    resource = snapshot["resource"]
    if not isinstance(resource, dict) or set(resource) != {
        "provider",
        "location",
        "resource_type",
        "gpu_count",
        "gpu_memory_gb",
        "spot",
        "hourly_rate_usd",
        "persistent_storage",
        "active_duplicate_pods",
        "ephemeral_storage_gb",
        "wallet_usd",
        "support_cap_usd",
        "science_reserve_usd",
        "teardown_reserve_usd",
    }:
        raise ValueError("launch preflight resource witness is incomplete")
    if (
        resource["provider"] != "prime"
        or not isinstance(resource["location"], str)
        or not resource["location"]
        or resource["resource_type"] != "2x48GB L40/L40S/RTX 6000 Ada"
        or type(resource["gpu_count"]) is not int
        or resource["gpu_count"] != 2
        or type(resource["gpu_memory_gb"]) is not int
        or resource["gpu_memory_gb"] != 48
        or resource["spot"] is not False
        or type(resource["hourly_rate_usd"]) not in {int, float}
        or resource["hourly_rate_usd"] > 2
        or resource["persistent_storage"] is not False
        or type(resource["active_duplicate_pods"]) is not int
        or resource["active_duplicate_pods"] != 0
        or type(resource["ephemeral_storage_gb"]) not in {int, float}
        or resource["ephemeral_storage_gb"] <= 0
        or type(resource["wallet_usd"]) not in {int, float}
        or resource["wallet_usd"] < 30
        or resource["support_cap_usd"] != 12
        or resource["science_reserve_usd"] != 16
        or resource["teardown_reserve_usd"] != 2
        or resource["wallet_usd"] < 12 + 16 + 2
    ):
        raise ValueError("launch preflight resource witness is outside the frozen gate")
    if snapshot["runtime"] != {
        "python": SUPPORTED_PYTHON,
        "pyarrow": SUPPORTED_PYARROW,
        "datasets": SUPPORTED_DATASETS,
        "uv_native_only": True,
    }:
        raise ValueError("launch preflight runtime differs from the frozen CPU stack")
    dependency = snapshot["dependency"]
    if not isinstance(dependency, dict) or dependency != {
        "stack_sha256": EXPECTED_DEPENDENCY_STACK_SHA256,
        "renderer_post_tree_sha256": EXPECTED_RENDERER_TREE_SHA256,
        "verifier_post_tree_sha256": EXPECTED_VERIFIER_TREE_SHA256,
    }:
        raise ValueError("launch preflight dependency witness differs")
    assets = snapshot["assets"]
    if not isinstance(assets, dict) or assets != dict(sorted(POLICY_FILES.items())):
        raise ValueError("launch preflight model/renderer/tokenizer assets differ")
    outputs = snapshot["outputs"]
    if not isinstance(outputs, dict) or outputs != {
        "ledger_pristine": True,
        "output_root_pristine": True,
    }:
        raise ValueError("launch preflight output roots are not pristine")
    vllm = snapshot["vllm"]
    if not isinstance(vllm, dict) or vllm != {
        "health": "ready",
        "model_list": ["/workspace/models/stage-d1-merged"],
        "completion_requests": 0,
    }:
        raise ValueError("launch preflight vLLM witness is incomplete or used generation")
    offline = snapshot["offline"]
    if not isinstance(offline, dict) or offline != {
        "bindings": dict(sorted(OFFLINE_RLM_BINDINGS.items())),
        "lock_sha256": EXPECTED_UV_LOCK_SHA256,
    }:
        raise ValueError("launch preflight offline RLM witness differs")
    return cast(dict[str, Any], snapshot)


def build_preflight_snapshot(
    root: Path,
    *,
    location: str,
    captured_at_epoch: int,
    expires_at_epoch: int,
    require_post_commit: bool = False,
) -> bytes:
    """Build the canonical, read-only witness supplied by launch-time checks.

    This helper only serializes evidence supplied by the deployment system; it
    never queries a provider or claims that hardware/wallet checks occurred.
    """

    root = root.resolve()
    mode = _authenticate_parent(root, require_post_commit=require_post_commit)
    bundle_commit = _git_value(root, "rev-parse", "HEAD")
    bundle_tree = _git_value(root, "rev-parse", "HEAD^{tree}")
    return cast(bytes, canonical_json_bytes(
        {
            "schema_version": 1,
            "domain": "redco-stage-d1-support-v13-launch-preflight-v1",
            "state": "verified",
            "captured_at_epoch": captured_at_epoch,
            "expires_at_epoch": expires_at_epoch,
            "bundle": {"mode": mode, "commit": bundle_commit, "tree": bundle_tree},
            "resource": {
                "provider": "prime",
                "location": location,
                "resource_type": "2x48GB L40/L40S/RTX 6000 Ada",
                "gpu_count": 2,
                "gpu_memory_gb": 48,
                "spot": False,
                "hourly_rate_usd": 2,
                "persistent_storage": False,
                "active_duplicate_pods": 0,
                "ephemeral_storage_gb": 1,
                "wallet_usd": 30,
                "support_cap_usd": 12,
                "science_reserve_usd": 16,
                "teardown_reserve_usd": 2,
            },
            "runtime": {
                **RUNTIME,
                "uv_native_only": True,
            },
            "dependency": {
                "stack_sha256": EXPECTED_DEPENDENCY_STACK_SHA256,
                "renderer_post_tree_sha256": EXPECTED_RENDERER_TREE_SHA256,
                "verifier_post_tree_sha256": EXPECTED_VERIFIER_TREE_SHA256,
            },
            "assets": dict(sorted(POLICY_FILES.items())),
            "outputs": {"ledger_pristine": True, "output_root_pristine": True},
            "vllm": {
                "health": "ready",
                "model_list": ["/workspace/models/stage-d1-merged"],
                "completion_requests": 0,
            },
            "offline": {
                "bindings": dict(sorted(OFFLINE_RLM_BINDINGS.items())),
                "lock_sha256": EXPECTED_UV_LOCK_SHA256,
            },
        }
    ))


def preflight_validate(
    root: Path,
    snapshot_path: Path,
    *,
    require_post_commit: bool = False,
    runtime_observation_path: Path | None = None,
    synthetic: bool = False,
) -> dict[str, str]:
    """Validate an owned launch observation before any attempt claim.

    ``synthetic=True`` is test-only and is never accepted by the execute-once
    path.  Production validation consumes raw Prime and pod observations and
    never trusts a caller-supplied resource assertion.
    """

    root = root.resolve()
    mode = _authenticate_parent(root, require_post_commit=require_post_commit)
    hashes = verify_launch_bundle(root, require_post_commit=require_post_commit)
    snapshot = snapshot_path.resolve()
    if snapshot.is_symlink() or not snapshot.is_file():
        raise ValueError("launch preflight observation is missing")
    protected = tuple(_bundle_immutable_paths(root)) + tuple(
        str(root / relative) for relative in (*OUTPUT_PATHS, *ATTEMPT_PATHS)
    )
    for immutable in protected:
        source = Path(immutable)
        if source.is_file() and os.path.samefile(snapshot, source):
            raise ValueError("launch preflight observation aliases an immutable input")
    if synthetic:
        _validate_preflight_snapshot(
            snapshot.read_bytes(),
            expected_bundle_mode=mode,
            expected_bundle_commit=_git_value(root, "rev-parse", "HEAD"),
            expected_bundle_tree=_git_value(root, "rev-parse", "HEAD^{tree}"),
        )
    else:
        try:
            snapshot.relative_to(root)
        except ValueError as error:
            raise ValueError("Prime observation escapes the repository") from error
        validate_prime_observation(root, snapshot)
        if runtime_observation_path is None:
            raise ValueError("production preflight requires a pod runtime observation")
        runtime_observation = runtime_observation_path.resolve()
        try:
            runtime_observation.relative_to(root)
        except ValueError as error:
            raise ValueError("pod runtime observation escapes the repository") from error
        if runtime_observation.is_symlink() or not runtime_observation.is_file():
            raise ValueError("pod runtime observation is missing")
        for immutable in protected:
            source = Path(immutable)
            if source.is_file() and os.path.samefile(runtime_observation, source):
                raise ValueError("pod runtime observation aliases an immutable input")
        validate_pod_runtime_observation(
            runtime_observation.read_bytes(),
            expected_asset_hashes={
                **POLICY_FILES,
                "uv.lock": EXPECTED_UV_LOCK_SHA256,
            },
            expected_runtime={
                "python": SUPPORTED_PYTHON,
                "datasets": SUPPORTED_DATASETS,
                "pyarrow": SUPPORTED_PYARROW,
            },
        )
    return hashes


def build_launch_artifacts(
    root: Path,
    *,
    require_post_commit: bool = False,
) -> dict[str, bytes]:
    """Reconstruct the complete launch set from authenticated local inputs."""
    root = root.resolve()
    root_hashes = _authenticate_roots(root, require_post_commit=require_post_commit)
    dataset, plan_bytes, plan, dataset_sha256 = _build_collection_inputs(root)
    protocol_anchor = _sha256_json(
        {
            "domain": "redco-stage-d1-support-v13-protocol-anchor-v1",
            "parent_commit": PARENT_COMMIT,
            "parent_tree": PARENT_TREE,
            "authorization_text_sha256": AUTHORIZATION_TEXT_SHA256,
            "collection_plan_sha256": plan.plan_sha256,
            "dataset_sha256": dataset_sha256,
        }
    )
    authorization = _authorization(
        dataset_sha256=dataset_sha256,
        plan_sha256=plan.plan_sha256,
        protocol_anchor_sha256=protocol_anchor,
        root_hashes=root_hashes,
        asset_bindings=asset_binding_contract(root),
    )
    authorization_sha256 = sha256_bytes(authorization)
    source_eval = _source_eval(
        dataset_sha256=dataset_sha256,
        plan_sha256=plan.plan_sha256,
        authorization_sha256=authorization_sha256,
        protocol_anchor_sha256=protocol_anchor,
    )
    branch_runtime = _branch_runtime(
        source_eval,
    )
    branch_runtime_sha256 = sha256_bytes(branch_runtime)
    returning_root_contract = contract_artifact_bytes()
    protocol = _protocol_manifest(
        root,
        dataset_sha256=dataset_sha256,
        source_eval_sha256=sha256_bytes(source_eval),
        branch_runtime_sha256=branch_runtime_sha256,
        plan=plan,
        authorization_sha256=authorization_sha256,
    )
    audit_payload = {
        "schema_version": 1,
        "domain": "redco-stage-d1-support-v13-launch-audit-v1",
        "state": "cpu_bundle_verified_not_launched",
        "parent": {"commit": PARENT_COMMIT, "tree": PARENT_TREE},
        "authorization_sha256": authorization_sha256,
        "protocol_anchor_sha256": protocol_anchor,
        "protocol_manifest_sha256": sha256_bytes(protocol),
        "artifact_bindings": {
            LAUNCH_AUTH_RELATIVE: {"sha256": authorization_sha256, "bytes": len(authorization)},
            LAUNCH_DATASET_RELATIVE: {"sha256": dataset_sha256, "bytes": len(dataset), "rows": 64},
            LAUNCH_PLAN_RELATIVE: {"sha256": sha256_bytes(plan_bytes), "bytes": len(plan_bytes)},
            LAUNCH_SOURCE_EVAL_RELATIVE: {
                "sha256": sha256_bytes(source_eval),
                "bytes": len(source_eval),
            },
            LAUNCH_BRANCH_RUNTIME_RELATIVE: {
                "sha256": branch_runtime_sha256,
                "bytes": len(branch_runtime),
            },
            LAUNCH_PROTOCOL_RELATIVE: {"sha256": sha256_bytes(protocol), "bytes": len(protocol)},
            RETURNING_ROOT_CONTRACT_RELATIVE: {
                "sha256": sha256_bytes(returning_root_contract),
                "bytes": len(returning_root_contract),
            },
        },
        "frozen_bindings": dict(sorted(root_hashes.items())),
        "support_contract": {
            "papers": 64,
            "success_floor": 58,
            "k": 4,
            "minimum_score_range": 0.05,
            "no_early_pass": True,
            "no_retry": True,
            "science_authorized": False,
            "training_authorized": False,
            "heldout_evaluation_authorized": False,
            "scientific_transition_authorized": False,
        },
        "live_activity_performed": False,
        "source_access_performed": False,
        "provider_calls_performed": False,
        "wallet_or_hardware_checked": False,
    }
    audit = canonical_json_bytes(audit_payload)
    _validate_source_eval_contract(source_eval)
    return {
        LAUNCH_AUTH_RELATIVE: authorization,
        LAUNCH_DATASET_RELATIVE: dataset,
        LAUNCH_PLAN_RELATIVE: plan_bytes,
        LAUNCH_SOURCE_EVAL_RELATIVE: source_eval,
        LAUNCH_BRANCH_RUNTIME_RELATIVE: branch_runtime,
        LAUNCH_PROTOCOL_RELATIVE: protocol,
        LAUNCH_AUDIT_RELATIVE: audit,
        RETURNING_ROOT_CONTRACT_RELATIVE: returning_root_contract,
    }


def _bundle_immutable_paths(root: Path) -> dict[str, str]:
    return {
        str((root / relative).resolve()): expected
        for relative, expected in FROZEN_ROOT_HASHES.items()
    } | {str((root / relative).resolve()): expected for relative, expected in POLICY_FILES.items()}


def verify_launch_bundle(
    root: Path,
    artifact_root: Path | None = None,
    *,
    require_post_commit: bool = False,
) -> dict[str, str]:
    """Verify reconstructed bytes, complete 64-row input, and all launch gates."""
    root = root.resolve()
    artifact_root = (artifact_root or root).resolve()
    expected = build_launch_artifacts(root, require_post_commit=require_post_commit)
    validate_output_paths(
        artifact_root,
        _bundle_immutable_paths(root),
        output_paths=OUTPUT_PATHS,
    )
    actual: dict[str, bytes] = {}
    for relative, expected_bytes in expected.items():
        path = artifact_root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"launch bundle artifact is missing: {relative}")
        actual[relative] = path.read_bytes()
        if actual[relative] != expected_bytes:
            raise ValueError(f"launch bundle artifact differs from reconstructed bytes: {relative}")
    auth = json.loads(actual[LAUNCH_AUTH_RELATIVE])
    if (
        auth.get("authorization", {}).get("text_sha256") != AUTHORIZATION_TEXT_SHA256
        or auth.get("signing") != launch_signing_identity().to_payload()
        or auth.get("orchestrator_thread_id") != ORCHESTRATOR_THREAD
        or auth.get("parent") != {"commit": PARENT_COMMIT, "tree": PARENT_TREE}
        or auth.get("execution_gate")
        != {
            "precommit_build_non_authorizing": True,
            "precommit_execute_allowed": False,
            "requires_parent_commit": PARENT_COMMIT,
            "surface": "post_commit_direct_child_only",
        }
        or auth.get("scope", {}).get("support_launch_authorized") is not True
        or auth.get("scope", {}).get("science_authorized") is not False
        or auth.get("scope", {}).get("training_authorized") is not False
        or auth.get("scope", {}).get("heldout_evaluation_authorized") is not False
        or auth.get("scope", {}).get("scientific_transition_authorized") is not False
    ):
        raise ValueError("launch authorization is outside the support-only scope")
    dataset_rows = actual[LAUNCH_DATASET_RELATIVE].splitlines()
    if len(dataset_rows) != 64 or any(
        json.loads(row).get("split") != "successor_support" for row in dataset_rows
    ):
        raise ValueError("launch dataset is not exactly 64 support rows")
    plan = StageDCollectionPlan.from_bytes(actual[LAUNCH_PLAN_RELATIVE])
    if len(plan.slots) != 64:
        raise ValueError("launch collection plan does not contain exactly 64 slots")
    expected_branch_runtime = _branch_runtime(
        actual[LAUNCH_SOURCE_EVAL_RELATIVE],
    )
    if actual[LAUNCH_BRANCH_RUNTIME_RELATIVE] != expected_branch_runtime:
        raise ValueError("launch branch runtime differs from reconstructed bytes")
    _validate_branch_runtime(actual[LAUNCH_BRANCH_RUNTIME_RELATIVE])
    protocol = StageDProtocolManifest.from_bytes(actual[LAUNCH_PROTOCOL_RELATIVE])
    if protocol.collection_plan_sha256 != plan.plan_sha256:
        raise ValueError("launch protocol manifest does not bind its collection plan")
    _validate_source_eval_contract(actual[LAUNCH_SOURCE_EVAL_RELATIVE])
    return {relative: sha256_bytes(value) for relative, value in actual.items()}


def summarize_bundle(
    root: Path,
    artifact_root: Path | None = None,
    *,
    require_post_commit: bool = False,
) -> dict[str, Any]:
    hashes = verify_launch_bundle(
        root,
        artifact_root,
        require_post_commit=require_post_commit,
    )
    return {
        "domain": LAUNCH_DOMAIN,
        "state": "verified_cpu_only_not_launched",
        "artifact_hashes": dict(sorted(hashes.items())),
        "support_rows": 64,
        "science_rows": 0,
        "provider_calls_performed": False,
        "source_access_performed": False,
        "wallet_or_hardware_checked": False,
        "science_authorized": False,
        "scientific_transition_authorized": False,
        "activation_mode": (
            "committed_direct_child" if require_post_commit else "precommit_non_authorizing"
        ),
    }


__all__ = [
    "ALLOWED_SIGNERS_SHA256",
    "AUTHORIZATION_TEXT",
    "AUTHORIZATION_TEXT_SHA256",
    "LAUNCH_AUDIT_RELATIVE",
    "LAUNCH_AUTH_RELATIVE",
    "LAUNCH_BRANCH_RUNTIME_RELATIVE",
    "LAUNCH_BUNDLE_PATHS",
    "LAUNCH_CODE_PATHS",
    "LAUNCH_DATASET_RELATIVE",
    "LAUNCH_DOMAIN",
    "LAUNCH_HANDOFF_RELATIVE",
    "LAUNCH_HANDOFF_SIGNATURE_RELATIVE",
    "LAUNCH_KNOWN_HOSTS_RELATIVE",
    "LAUNCH_LIFECYCLE_RELATIVE",
    "LAUNCH_OWNER_PATHS",
    "LAUNCH_PLAN_RELATIVE",
    "LAUNCH_POD_OBSERVATION_RELATIVE",
    "LAUNCH_PREFLIGHT_SNAPSHOT_RELATIVE",
    "LAUNCH_PRIME_OBSERVATION_RELATIVE",
    "LAUNCH_PROTOCOL_RELATIVE",
    "LAUNCH_PROVISIONING_LEDGER_RELATIVE",
    "LAUNCH_PROVISION_CLAIM_RELATIVE",
    "LAUNCH_RUNTIME_MANIFEST_RELATIVE",
    "LAUNCH_SOURCE_EVAL_RELATIVE",
    "OFFLINE_RLM_BINDINGS",
    "PARENT_COMMIT",
    "PARENT_TREE",
    "RETURNING_ROOT_CONTRACT_RELATIVE",
    "SIGNING_FINGERPRINT_SHA256",
    "SIGNING_NAMESPACE",
    "SIGNING_PRINCIPAL",
    "build_launch_artifacts",
    "build_preflight_snapshot",
    "execute_support_once",
    "launch_signing_identity",
    "preflight_validate",
    "summarize_bundle",
    "verify_launch_bundle",
]
