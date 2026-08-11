"""Non-authorizing successor readiness and sole authorization-child contract.

This module deliberately owns no Prime, wallet, pod, SSH, model, or source
operation.  It authenticates the completed v2 no-capacity evidence, freezes a
new monotonic v3 evidence root, and exposes the only future authorization
builder.  The existing v2 lifecycle remains the runtime owner after that
future child is independently reviewed and committed.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, cast

from redco.analysis import stage_d_v13_prime_inventory_v5 as v5
from redco.analysis import stage_d_v13_prime_test_one_shot_contract_v2 as v2
from redco.analysis.stage_d_v13_prime_test_one_shot_evidence_v2 import (
    verify_terminal_evidence,
)
from redco.analysis.stage_d_v13_prime_test_one_shot_runtime_binding_v2 import (
    V3_READINESS_AUTHORITY,
    V3_RUNTIME_AUTHORITY,
)

ROOT = Path(__file__).resolve().parents[3]

PARENT_COMMIT = "b2b7190dc07510af657c32c1037f8f0a6df3beb7"
PARENT_TREE = "7f6e506a6545627aa8a1903a4f6b1c7a422fb08b"
V2_PARENT_COMMIT = "f4c27b5ff3ed950887d777d80a7680b66e27a5f0"
V2_PARENT_TREE = "3d645cbd91e46547939d942fac06762148b68623"

READINESS_DOMAIN = "redco-stage-d1-prime-test-one-shot-readiness-v3"
AUDIT_DOMAIN = "redco-stage-d1-prime-test-one-shot-readiness-audit-v3"
AUTHORIZATION_DOMAIN = "redco-stage-d1-prime-test-one-shot-authorization-v3"
AUTHORIZATION_PATH = "configs/stage-d/stage-d1-prime-test-one-shot-authorization-v3.json"
CONTRACT_PATH = "configs/stage-d/stage-d1-prime-test-one-shot-readiness-v3.json"
AUDIT_PATH = "reports/stage-d1-prime-test-one-shot-readiness-audit-v3.json"
MODULE_PATH = "src/redco/analysis/stage_d_v13_prime_test_one_shot_successor_v3.py"
BUILDER_PATH = "scripts/build_stage_d_v13_prime_test_one_shot_readiness_v3.py"
TEST_PATH = "tests/test_stage_d_v13_prime_test_one_shot_successor_v3.py"
RUNTIME_MODULE_PATH = "src/redco/analysis/stage_d_v13_prime_test_one_shot_runtime_v3.py"
RUNTIME_BINDING_MODULE_PATH = (
    "src/redco/analysis/stage_d_v13_prime_test_one_shot_runtime_binding_v2.py"
)
HANDOFF_MODULE_PATH = "src/redco/analysis/stage_d_v13_prime_test_one_shot_handoff_v2.py"
GIT_OWNER_MODULE_PATH = "src/redco/analysis/stage_d_v13_prime_test_one_shot_git_owner_v2.py"
RUNTIME_BINDING_TEST_PATH = "tests/test_stage_d_v13_prime_test_one_shot_runtime_binding_v3.py"
EVIDENCE_MATRIX_TEST_PATH = "tests/test_stage_d_v13_prime_test_one_shot_evidence_matrix_v3.py"
RUNNER_PATH = "scripts/run_stage_d_v13_prime_test_one_shot_v3.py"
MODIFIED_OWNER_PATHS = frozenset(
    {
        "src/redco/analysis/stage_d_v13_prime_test_one_shot_contract_v2.py",
        "src/redco/analysis/stage_d_v13_prime_test_one_shot_prime_v2.py",
        "src/redco/analysis/stage_d_v13_prime_test_one_shot_lifecycle_v2.py",
        "src/redco/analysis/stage_d_v13_prime_test_one_shot_evidence_v2.py",
        "src/redco/analysis/stage_d_v13_prime_test_one_shot_remote_v2.py",
        "tests/test_stage_d_v13_prime_test_one_shot_contract_v2.py",
        "tests/test_stage_d_v13_prime_test_one_shot_evidence_v2.py",
        "tests/test_stage_d_v13_prime_test_one_shot_lifecycle_v2.py",
    }
)
NEW_PATHS = frozenset(
    {
        MODULE_PATH,
        BUILDER_PATH,
        TEST_PATH,
        CONTRACT_PATH,
        AUDIT_PATH,
        RUNTIME_MODULE_PATH,
        RUNTIME_BINDING_MODULE_PATH,
        HANDOFF_MODULE_PATH,
        GIT_OWNER_MODULE_PATH,
        RUNTIME_BINDING_TEST_PATH,
        EVIDENCE_MATRIX_TEST_PATH,
        RUNNER_PATH,
    }
)
READINESS_PATHS = MODIFIED_OWNER_PATHS | NEW_PATHS
SOURCE_PATHS = READINESS_PATHS - {CONTRACT_PATH, AUDIT_PATH}

V2_EVIDENCE_ROOT = "runs/stage-d/stage-d1-prime-test-one-shot-v2"
EVIDENCE_ROOT = "runs/stage-d/stage-d1-prime-test-one-shot-v3"
V2_AUTHORIZATION_PATH = v2.AUTHORIZATION_PATH
V2_TERMINAL_DOMAIN = v2.TERMINAL_DOMAIN
CLAIM_DOMAIN_V3 = "redco-stage-d1-prime-test-one-shot-claim-v3"
TERMINAL_DOMAIN_V3 = "redco-stage-d1-prime-test-one-shot-terminal-v3"
TERMINAL_NAMESPACE_V3 = "redco-stage-d1-prime-test-one-shot-terminal-v3"
TERMINAL_PURPOSE_V3 = v2.TERMINAL_PURPOSE
V2_TERMINAL_FILE_NAMES = frozenset(
    {
        "assessment-envelope.json",
        "assessment.json",
        "claim.json",
        "command-records.json",
        "terminal-envelope.json",
        "terminal.json",
        "transcript.json",
    }
)

SUCCESSOR_AUTHORIZATION_TEXT = (
    "I authorize one fresh immediate Prime attempt after the terminal v2 no-capacity "
    "observation. Use uv only; never pip. No monitoring, no retry after the actual "
    "Prime observation, and model calls, training, science, source, and Parquet "
    "remain unauthorized."
)
SUCCESSOR_AUTHORIZATION_SHA256 = hashlib.sha256(
    SUCCESSOR_AUTHORIZATION_TEXT.encode("utf-8")
).hexdigest()
ORCHESTRATOR_THREAD = v2.ORCHESTRATOR_THREAD
THREAT_MODEL = {
    "supported": (
        "honest-but-fallible operator and untrusted provider/network/artifact bytes"
    ),
    "out_of_scope": (
        "arbitrary hostile in-process Python code execution or monkeypatching of "
        "authenticated module globals; equivalent to process compromise"
    ),
}

READINESS_AUTHORITY = V3_READINESS_AUTHORITY
RUNTIME_AUTHORITY = V3_RUNTIME_AUTHORITY

V2_TERMINAL_TOP_LEVEL_KEYS = frozenset(
    {
        "assessment_sha256",
        "attempt_consumed",
        "authority",
        "authorization",
        "cleanup_failures",
        "cleanup_proven",
        "command_count",
        "create_dispatched",
        "disposition",
        "domain",
        "elapsed_seconds",
        "evidence_dag",
        "monitoring",
        "primary_failure",
        "prime_cli_call_count",
        "publication_failures",
        "purpose",
        "recovery_failures",
        "retry",
        "schema_version",
        "state",
        "tests_passed",
        "wallet_api_call_count",
    }
)


def canonical_json(value: object) -> bytes:
    return v2.canonical_json(value)


def sha256_bytes(raw: bytes) -> str:
    return v2.sha256_bytes(raw)


def _git_output(root: Path, *arguments: str) -> str:
    return v2.git_output(root, *arguments)


def _git_status(root: Path) -> set[tuple[str, str]]:
    return v2._status(root)


def current_head(root: Path) -> str:
    return _git_output(root.resolve(), "rev-parse", "HEAD")


def _regular_file(path: Path, label: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is absent") from error
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or getattr(info, "st_file_attributes", 0) & reparse
    ):
        raise ValueError(f"{label} is not an unaliased regular file")
    return path.read_bytes()


def _binding(root: Path, relative: str) -> dict[str, object]:
    raw = _regular_file(root / relative, relative)
    return {"bytes": len(raw), "sha256": sha256_bytes(raw)}


def _safe_absent(root: Path, relative: str) -> None:
    target = root / relative
    if target.exists() or target.is_symlink():
        raise ValueError(f"{relative} must remain absent")
    current = target.parent
    root = root.resolve()
    while current != root:
        if current.exists() or current.is_symlink():
            info = current.lstat()
            reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if current.is_symlink() or getattr(info, "st_file_attributes", 0) & reparse:
                raise ValueError(f"{relative} has an aliased ancestor")
        current = current.parent


def _v2_signing_identity() -> v2.SigningIdentity:
    raw = v5._load_terminal_signing_identity()
    return v2.SigningIdentity(
        raw.principal,
        raw.key_type,
        raw.public_key_base64,
        raw.fingerprint_sha256,
        raw.allowed_signers_sha256,
    )


def _v2_history(root: Path) -> dict[str, object]:
    history_root = root / V2_EVIDENCE_ROOT
    if not history_root.is_dir() or history_root.is_symlink():
        raise ValueError("v2 terminal evidence root is absent or aliased")
    children = {item.name for item in history_root.iterdir()}
    if children != set(V2_TERMINAL_FILE_NAMES):
        raise ValueError("v2 terminal evidence topology differs")
    files: dict[str, dict[str, object]] = {}
    parsed: dict[str, object] = {}
    for name in sorted(V2_TERMINAL_FILE_NAMES):
        raw = _regular_file(history_root / name, f"v2/{name}")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"v2/{name} is not JSON") from error
        if canonical_json(value) != raw:
            raise ValueError(f"v2/{name} is not canonical JSON")
        files[name] = {"bytes": len(raw), "sha256": sha256_bytes(raw)}
        parsed[name] = value

    try:
        signed_terminal = verify_terminal_evidence(history_root, _v2_signing_identity())
    except Exception as error:
        raise ValueError("v2 terminal signed evidence is not authentic") from error
    if (
        signed_terminal.get("state") != "no_qualifying_capacity"
        or signed_terminal.get("disposition") != "no_qualifying_capacity"
    ):
        raise ValueError("v2 signed terminal is not the no-capacity predecessor")

    terminal_value = parsed["terminal.json"]
    if not isinstance(terminal_value, dict):
        raise ValueError("v2 terminal is not an object")
    terminal = cast(dict[str, Any], terminal_value)
    if set(terminal) != set(V2_TERMINAL_TOP_LEVEL_KEYS):
        raise ValueError("v2 terminal schema differs")
    expected = {
        "schema_version": 2,
        "domain": V2_TERMINAL_DOMAIN,
        "purpose": v2.TERMINAL_PURPOSE,
        "state": "no_qualifying_capacity",
        "disposition": "no_qualifying_capacity",
        "attempt_consumed": True,
        "monitoring": False,
        "retry": False,
        "create_dispatched": False,
        "cleanup_proven": True,
        "command_count": 0,
        "prime_cli_call_count": 0,
        "wallet_api_call_count": 0,
        "tests_passed": False,
        "primary_failure": None,
        "cleanup_failures": [],
        "recovery_failures": [],
        "publication_failures": [],
    }
    for key, expected_value in expected.items():
        if terminal[key] != expected_value:
            raise ValueError(f"v2 terminal {key} differs")
    if terminal["authority"] != v2.READINESS_AUTHORITY:
        raise ValueError("v2 terminal authority is not all false")
    dag = terminal["evidence_dag"]
    if not isinstance(dag, dict) or set(dag) != {
        "assessment",
        "assessment-envelope",
        "claim",
        "command-records",
        "transcript",
    }:
        raise ValueError("v2 terminal evidence DAG differs")
    for key, entry in cast(dict[str, Any], dag).items():
        if not isinstance(entry, dict) or set(entry) != {"bytes", "path", "sha256"}:
            raise ValueError("v2 terminal evidence DAG entry differs")
        name = f"{key}.json"
        if entry["path"] != name or entry["bytes"] != files[name]["bytes"]:
            raise ValueError("v2 terminal evidence DAG path differs")
        if entry["sha256"] != files[name]["sha256"]:
            raise ValueError("v2 terminal evidence DAG hash differs")
    authorization = terminal["authorization"]
    if not isinstance(authorization, dict):
        raise ValueError("v2 terminal authorization is malformed")
    auth = cast(dict[str, Any], authorization)
    auth_raw = _regular_file(root / V2_AUTHORIZATION_PATH, "v2 authorization")
    if auth["authorization_path"] != V2_AUTHORIZATION_PATH:
        raise ValueError("v2 authorization path differs")
    if auth["authorization_sha256"] != sha256_bytes(auth_raw):
        raise ValueError("v2 authorization hash differs")
    if auth["commit"] != PARENT_COMMIT or auth["tree"] != PARENT_TREE:
        raise ValueError("v2 authorization lineage differs")
    if auth["parent"] != V2_PARENT_COMMIT or auth["authorization_blob"] != _git_output(
        root, "rev-parse", f"HEAD:{V2_AUTHORIZATION_PATH}"
    ):
        raise ValueError("v2 authorization binding differs")
    return {
        "root": V2_EVIDENCE_ROOT,
        "files": files,
        "terminal": {
            "bytes": files["terminal.json"]["bytes"],
            "sha256": files["terminal.json"]["sha256"],
            "state": terminal["state"],
            "disposition": terminal["disposition"],
            "attempt_consumed": terminal["attempt_consumed"],
            "create_dispatched": terminal["create_dispatched"],
            "cleanup_proven": terminal["cleanup_proven"],
            "authority": terminal["authority"],
        },
        "authorization": {
            "path": V2_AUTHORIZATION_PATH,
            "sha256": auth["authorization_sha256"],
            "commit": auth["commit"],
            "parent": auth["parent"],
            "tree": auth["tree"],
            "blob": auth["authorization_blob"],
        },
    }


def _expected_status(*, artifacts_present: bool) -> set[tuple[str, str]]:
    status = {(" M", v2.EXTERNAL_GITLINK)}
    paths = READINESS_PATHS if artifacts_present else SOURCE_PATHS
    return status | {
        (" M", path) if path in MODIFIED_OWNER_PATHS else ("??", path)
        for path in paths
    }


def _readiness_diff() -> set[str]:
    return {f"A\t{path}" for path in NEW_PATHS} | {
        f"M\t{path}" for path in MODIFIED_OWNER_PATHS
    }


def authenticate_successor(
    root: Path, *, committed: bool, head_override: str | None = None
) -> dict[str, str]:
    root = root.resolve()
    _safe_absent(root, EVIDENCE_ROOT)
    _v2_history(root)
    v2._external_binding(root)
    if _git_output(root, "rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise ValueError("successor parent commit differs")
    if _git_output(root, "rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise ValueError("successor parent tree differs")
    head = _git_output(root, "rev-parse", "HEAD") if head_override is None else head_override
    artifact_presence = {
        relative: (root / relative).is_file() and not (root / relative).is_symlink()
        for relative in (CONTRACT_PATH, AUDIT_PATH)
    }
    if len(set(artifact_presence.values())) != 1:
        raise ValueError("successor artifacts are partially present")
    if committed:
        if head == PARENT_COMMIT:
            raise ValueError("successor readiness is not committed")
        lineage = _git_output(root, "rev-list", "--parents", "-n", "1", head).split()
        if lineage != [head, PARENT_COMMIT]:
            raise ValueError("successor must be a direct child of the current parent")
        changes = set(
            _git_output(
                root, "diff", "--name-status", "--no-renames", PARENT_COMMIT, head
            ).splitlines()
        )
        if changes != _readiness_diff():
            raise ValueError("successor direct-child diff is not exact")
        if _git_status(root) != {(" M", v2.EXTERNAL_GITLINK)}:
            raise ValueError("successor committed checkout is not exact")
    else:
        if head != PARENT_COMMIT:
            raise ValueError("pre-commit successor build requires the current parent")
        if _git_status(root) != _expected_status(artifacts_present=all(artifact_presence.values())):
            raise ValueError("successor pre-commit checkout is not exact")
    return {"commit": head, "tree": _git_output(root, "show", "-s", "--format=%T", head)}


def _v2_readiness_bindings(root: Path) -> dict[str, dict[str, object]]:
    return {relative: _binding(root, relative) for relative in sorted(v2.READINESS_PATHS)}


def _source_bindings(root: Path) -> dict[str, dict[str, object]]:
    return {relative: _binding(root, relative) for relative in sorted(SOURCE_PATHS)}


def _contract_value(root: Path, history: dict[str, object]) -> dict[str, object]:
    source_bindings = _source_bindings(root)
    return {
        "schema_version": 3,
        "domain": READINESS_DOMAIN,
        "state": "non_authorizing_successor_readiness",
        "parent": {"commit": PARENT_COMMIT, "tree": PARENT_TREE},
        "threat_model": THREAT_MODEL,
        "readiness_paths": sorted(READINESS_PATHS),
        "source_bindings": source_bindings,
        "future_authorization": {
            "path": AUTHORIZATION_PATH,
            "domain": AUTHORIZATION_DOMAIN,
            "sole_child_diff": [f"A\t{AUTHORIZATION_PATH}"],
            "requires_readiness_direct_child": True,
            "present": False,
        },
        "runtime_binding": {
            "authorization_path": AUTHORIZATION_PATH,
            "authorization_authenticator": "authenticate_authorization_v3",
            "evidence_root": EVIDENCE_ROOT,
            "claim_domain": CLAIM_DOMAIN_V3,
            "claim_schema_version": 2,
            "claim_authority": RUNTIME_AUTHORITY,
            "assessment": {
                "domain": v2.ASSESSMENT_DOMAIN,
                "namespace": v2.ASSESSMENT_NAMESPACE,
                "schema_version": 2,
                "ttl_seconds": v2.ASSESSMENT_TTL_SECONDS,
                "authority": RUNTIME_AUTHORITY,
            },
            "create": {
                "schema_version": 2,
                "authority": RUNTIME_AUTHORITY,
            },
            "signed_envelope_domain": v2.SIGNED_ENVELOPE_DOMAIN,
            "handoff_namespace": v2.HANDOFF_NAMESPACE,
            "handoff_schema_version": 2,
            "handoff_authority": READINESS_AUTHORITY,
            "terminal": {
                "schema_version": 2,
                "domain": TERMINAL_DOMAIN_V3,
                "namespace": TERMINAL_NAMESPACE_V3,
                "purpose": TERMINAL_PURPOSE_V3,
                "authority": READINESS_AUTHORITY,
            },
            "result_authority": READINESS_AUTHORITY,
            "result_schema_version": 2,
            "schema_versions": {
                key: 2
                for key in (
                    "claim", "assessment", "assessment-envelope", "handoff",
                    "handoff-envelope", "terminal", "terminal-envelope", "result",
                )
            },
            "artifact_filenames": dict(v2.ARTIFACT_FILENAMES),
        },
        "evidence": {
            "predecessor_root": V2_EVIDENCE_ROOT,
            "successor_root": EVIDENCE_ROOT,
            "successor_root_must_be_absent": True,
            "predecessor_terminal": history,
        },
        "user_scope": {
            "successor_text_sha256": SUCCESSOR_AUTHORIZATION_SHA256,
            "successor_text_bytes": len(SUCCESSOR_AUTHORIZATION_TEXT.encode("utf-8")),
            "orchestrator_thread": ORCHESTRATOR_THREAD,
            "no_monitoring": True,
            "no_retry_after_observation": True,
            "model_training_science_source_parquet": False,
        },
        "reused_v2": {
            "readiness_config": {
                "path": v2.CONTRACT_PATH,
                **_binding(root, v2.CONTRACT_PATH),
            },
            "readiness_audit": {
                "path": v2.AUDIT_PATH,
                **_binding(root, v2.AUDIT_PATH),
            },
            "path_bindings": _v2_readiness_bindings(root),
            "contract_module": {
                "path": v2.CONTRACT_MODULE,
                **_binding(root, v2.CONTRACT_MODULE),
            },
        },
        "prime_test_scope": {
            "purpose": v2.TERMINAL_PURPOSE,
            "availability_attempt_limit": 1,
            "provisioning_dispatch_limit": 1,
            "retries": 0,
            "create_endpoint": v2.PODS_CREATE_ENDPOINT,
            "cli_create_forbidden": True,
            "same_pod_full_test_plan": True,
            "test_nodes": list(v2.TEST_NODES),
            "source_dataset_parquet_reads": False,
            "model_calls": False,
        },
        "limits": {
            "assessment_ttl_seconds": v2.ASSESSMENT_TTL_SECONDS,
            "maximum_pod_seconds": v2.MAXIMUM_POD_SECONDS,
            "support_cap_usd": v2.SUPPORT_CAP_USD,
            "reserve_usd": v2.RESERVE_USD,
            "wallet_minimum_usd": v2.WALLET_MINIMUM_USD,
            "maximum_rate_usd": v2.MAXIMUM_RATE_USD,
            "cleanup_timeout_seconds": v2.CLEANUP_TIMEOUT_SECONDS,
            "maximum_prime_cli_calls": v2.MAX_PRIME_CLI_CALLS,
            "maximum_wallet_api_calls": v2.MAX_WALLET_API_CALLS,
        },
        "authority": READINESS_AUTHORITY,
    }


def build_readiness_artifacts(
    root: Path, *, committed: bool, head_override: str | None = None
) -> dict[str, bytes]:
    root = root.resolve()
    authenticate_successor(root, committed=committed, head_override=head_override)
    history = _v2_history(root)
    contract = canonical_json(_contract_value(root, history))
    audit = canonical_json(
        {
            "schema_version": 3,
            "domain": AUDIT_DOMAIN,
            "state": "cpu_only_not_authorizing",
            "parent": {"commit": PARENT_COMMIT, "tree": PARENT_TREE},
            "contract": {
                "path": CONTRACT_PATH,
                "bytes": len(contract),
                "sha256": sha256_bytes(contract),
            },
            "successor": {
                "evidence_root": EVIDENCE_ROOT,
                "future_authorization_path": AUTHORIZATION_PATH,
                "future_authorization_present": False,
                "live_capture_executed": False,
                "prime_calls": 0,
                "network_calls": 0,
                "wallet_calls": 0,
                "provisioning_calls": 0,
                "remote_calls": 0,
                "model_calls": 0,
                "source_reads": 0,
            },
            "predecessor": history,
            "file_bindings": _source_bindings(root),
            "threat_model": THREAT_MODEL,
            "authority": READINESS_AUTHORITY,
        }
    )
    return {CONTRACT_PATH: contract, AUDIT_PATH: audit}


def verify_readiness_artifacts(
    root: Path, *, committed: bool, head_override: str | None = None
) -> dict[str, str]:
    expected = build_readiness_artifacts(
        root, committed=committed, head_override=head_override
    )
    result: dict[str, str] = {}
    for relative, raw in expected.items():
        actual = _regular_file(root / relative, relative)
        if actual != raw:
            raise ValueError(f"successor artifact differs: {relative}")
        result[relative] = sha256_bytes(raw)
    _safe_absent(root, AUTHORIZATION_PATH)
    return result


def authorization_value(
    root: Path, *, readiness_commit: str
) -> dict[str, object]:
    root = root.resolve()
    authenticate_successor(root, committed=True, head_override=readiness_commit)
    history = _v2_history(root)
    contract = _regular_file(root / CONTRACT_PATH, CONTRACT_PATH)
    audit = _regular_file(root / AUDIT_PATH, AUDIT_PATH)
    expected = build_readiness_artifacts(
        root, committed=True, head_override=readiness_commit
    )
    if contract != expected[CONTRACT_PATH] or audit != expected[AUDIT_PATH]:
        raise ValueError("successor readiness artifacts differ")
    return {
        "schema_version": 3,
        "domain": AUTHORIZATION_DOMAIN,
        "state": "one_shot_test_only_authorized",
        "lineage": {
            "parent": PARENT_COMMIT,
            "readiness_commit": readiness_commit,
            "readiness_tree": _git_output(root, "show", "-s", "--format=%T", readiness_commit),
            "readiness_contract_path": CONTRACT_PATH,
            "readiness_contract_sha256": sha256_bytes(contract),
            "readiness_audit_path": AUDIT_PATH,
            "readiness_audit_sha256": sha256_bytes(audit),
            "authorization_path": AUTHORIZATION_PATH,
        },
        "predecessor": history,
        "evidence": {
            "root": EVIDENCE_ROOT,
            "v2_terminal_state": "no_qualifying_capacity",
            "v2_terminal_sha256": cast(dict[str, Any], history["terminal"])["sha256"],
            "new_attempt_ordinal": 3,
        },
        "user": {
            "text": SUCCESSOR_AUTHORIZATION_TEXT,
            "bytes": len(SUCCESSOR_AUTHORIZATION_TEXT.encode("utf-8")),
            "sha256": SUCCESSOR_AUTHORIZATION_SHA256,
            "thread_id": ORCHESTRATOR_THREAD,
        },
        "scope": {
            "availability_attempt_limit": 1,
            "provisioning_dispatch_limit": 1,
            "monitoring": False,
            "retry": False,
            "same_pod_full_test_plan": True,
            "support_cap_usd": v2.SUPPORT_CAP_USD,
            "reserve_usd": v2.RESERVE_USD,
            "test_only": True,
            "authority": RUNTIME_AUTHORITY,
        },
    }


def authenticate_authorization_v3(root: Path) -> dict[str, str]:
    root = root.resolve()
    head = current_head(root)
    lineage = _git_output(root, "rev-list", "--parents", "-n", "1", head).split()
    if len(lineage) != 2 or lineage[0] != head:
        raise ValueError("v3 authorization is not a direct child")
    readiness_commit = lineage[1]
    if readiness_commit == PARENT_COMMIT:
        raise ValueError("v3 authorization has no committed readiness predecessor")
    authenticate_successor(
        root, committed=True, head_override=readiness_commit
    )
    changes = set(
        _git_output(
            root,
            "diff",
            "--name-status",
            "--no-renames",
            readiness_commit,
            head,
        ).splitlines()
    )
    if changes != {f"A\t{AUTHORIZATION_PATH}"}:
        raise ValueError("v3 authorization child diff is not exact")
    raw = _regular_file(root / AUTHORIZATION_PATH, AUTHORIZATION_PATH)
    if canonical_json(json.loads(raw)) != raw:
        raise ValueError("v3 authorization is not canonical JSON")
    expected = canonical_json(
        authorization_value(root, readiness_commit=readiness_commit)
    )
    if raw != expected:
        raise ValueError("v3 authorization document differs")
    committed_raw = _git_output(root, "show", f"{head}:{AUTHORIZATION_PATH}").encode("utf-8")
    if committed_raw != raw:
        raise ValueError("v3 authorization working tree differs from committed bytes")
    blob = _git_output(root, "rev-parse", f"HEAD:{AUTHORIZATION_PATH}")
    if len(blob) != 40 or any(character not in "0123456789abcdef" for character in blob):
        raise ValueError("v3 authorization blob is malformed")
    return {
        "commit": head,
        "tree": _git_output(root, "show", "-s", "--format=%T", head),
        "parent": readiness_commit,
        "authorization_path": AUTHORIZATION_PATH,
        "authorization_sha256": sha256_bytes(raw),
        "authorization_blob": blob,
    }


def publish_exclusive(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


__all__ = [
    "AUDIT_DOMAIN",
    "AUDIT_PATH",
    "AUTHORIZATION_DOMAIN",
    "AUTHORIZATION_PATH",
    "CONTRACT_PATH",
    "EVIDENCE_ROOT",
    "GIT_OWNER_MODULE_PATH",
    "MODIFIED_OWNER_PATHS",
    "NEW_PATHS",
    "PARENT_COMMIT",
    "PARENT_TREE",
    "READINESS_AUTHORITY",
    "READINESS_DOMAIN",
    "READINESS_PATHS",
    "ROOT",
    "RUNNER_PATH",
    "RUNTIME_BINDING_MODULE_PATH",
    "RUNTIME_MODULE_PATH",
    "SUCCESSOR_AUTHORIZATION_SHA256",
    "SUCCESSOR_AUTHORIZATION_TEXT",
    "THREAT_MODEL",
    "V2_EVIDENCE_ROOT",
    "V2_TERMINAL_FILE_NAMES",
    "authenticate_authorization_v3",
    "authenticate_successor",
    "authorization_value",
    "build_readiness_artifacts",
    "canonical_json",
    "current_head",
    "publish_exclusive",
    "sha256_bytes",
    "verify_readiness_artifacts",
]
