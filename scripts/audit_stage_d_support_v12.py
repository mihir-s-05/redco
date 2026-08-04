#!/usr/bin/env python3
# ruff: noqa: E501
"""Audit the zero-observation Stage-D v12 support successor and run its preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_dependency_stack import canonical_tree_manifest_bytes
from redco.contracts import canonical_json

RUNTIME_COMMIT = "7b54f25912a9842c000291d20314dc831eca776b"
RUNTIME_PROGRAM_PATHS = {"scientific_launcher": "scripts/run_stage_d_scientific_campaign.py", "trainer_entrypoint": "src/redco/analysis/stage_d_prime_trainer.py", "reporter": "src/redco/analysis/stage_d_campaign_store.py"}
PRESERVED_PREREGISTRATION_FIELDS = ["branch_protocol", "cohort", "interpretation_and_disposition", "policy", "repair_and_attempt_policy", "replacement", "resource_guardrails", "support_rule", "target_selection"]
SOURCE_CHANGE_FIELDS = set(
    ["action_contract", "closure_readiness_report", "closure_readiness_report_sha256", "dependency_auth_amendment_sha256", "domain", "historical_replay_report", "historical_replay_report_sha256", "measurement_semantics_changes", "redco_commit", "redco_cpu_amendment_sha256"]
)
PROTOCOL_CHANGE_FIELDS = set(
    ["dependency_stack_sha256", "genesis_config_sha256", "preregistration_sha256", "scientific_eval_config_sha256", "source_eval_config_sha256", "source_sha256"]
)
TOML_CHANGE_PATHS = {("output_dir",), ("env", "artifact_path"), ("env", "config_sha256"), ("env", "ledger_path"), ("env", "preregistration_sha256"), ("env", "source_sha256")}
EXPECTED_PREREGISTRATION_CHANGE_FIELDS = set(
    ["cpu_hardening", "deployment_authentication", "domain", "parent_terminal", "preflight", "purpose", "user_authorization"]
)
EXPECTED_ACTION_CONTRACT = "closure-based-exact-engine-actions-v3-max-token-canonical"
EXPECTED_PREFLIGHT_MODULE_SHA256 = "000a5e1ea4c266c4727c68ada0ff25169976d59321196bcac13b1691c08fe6f2"
EXPECTED_SOURCE_RUNNERS = {"branch_runner": "scripts/run_stage_d_scientific_campaign.py", "source_runner": "scripts/run_stage_d_source_collection.py"}
EXPECTED_TOML_PATHS = {
    "source": {"output_dir": "/workspace/redco/runs/stage-d/stage-d1-support-v12/source-eval", "ledger_path": "/workspace/redco/runs/stage-d/stage-d1-support-v12/ledger", "artifact_path": "/workspace/redco/runs/stage-d/stage-d1-support-v12/source-artifacts"},
    "replay": {"output_dir": "/workspace/redco/runs/stage-d/stage-d1-support-v12/replay-template", "ledger_path": "/workspace/redco/runs/stage-d/stage-d1-support-v12/ledger", "artifact_path": "/workspace/redco/runs/stage-d/stage-d1-support-v12/source-artifacts"},
}
EXPECTED_MEASUREMENT_SEMANTICS = {
    "source": ["max-token actions are canonical after exact termination proof even if truncated typed content does not round-trip", "strict restart reload preserves those completed max-token actions"],
    "preregistration": ["proven max-token responses remain canonical actions even when truncated typed content cannot semantically round-trip", "completed max-token actions survive strict serialization and restart reload"],
    "dependency_auth": ["Proven max-token responses remain canonical sampled actions even when truncated typed content cannot semantically round-trip.", "Completed max-token actions survive strict serialization and restart reload."],
    "readiness": ["A response proven to have exhausted its frozen max-token budget is retained as a canonical sampled action even when its typed message cannot semantically round-trip truncated token bytes.", "The canonical max-token action survives strict serialization and restart reload; every non-max-token action retains strict semantic validation."],
}
EXPECTED_ACTION_CLOSURE = ["eos", "max_tokens", "tool_calls", "textual_refusal", "empty_content", "multi_turn_child", "concurrent_gather"]
EXPECTED_DEPENDENCY_AUTH_KEYS = set(
    ["classification", "dependency_stack_sha256", "domain", "independent_child_tree_sha256s", "measurement_semantics_changes", "parent", "preobservation_evidence", "preserved", "prime_binding", "redco_binding", "same_pod_continuation_authorized", "sampling_changes", "schema_version", "scientific_decision_rule_changes", "verifier"]
)
EXPECTED_READINESS_KEYS = set(
    ["artifacts", "domain", "measurement_semantics", "preflight", "preobservation_lineage", "redco_commit", "schema_version", "status"]
)
EXPECTED_PURPOSE = (
    "Measure the natural eligible-and-informative branch-target density of the "
    "unchanged Stage D scaffold policy after restart-safe canonical max-token ingestion."
)
EXPECTED_USER_AUTHORIZATION = {"date": "2026-08-03", "statement": "Continue the bounded Stage D support work after CPU hardening and independent review; no scientific training is authorized by this preregistration."}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_canonical(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise ValueError(f"{path} is not canonical JSON")
    return value


def _require_hash(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    if _sha256(raw) != expected:
        raise ValueError(f"{path} differs from frozen SHA-256")
    return raw


def _git_bytes(repository: Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout


def _read_commit_canonical(repository: Path, path: str) -> dict[str, Any]:
    raw = _git_bytes(repository, RUNTIME_COMMIT, path)
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise ValueError(f"{path} is not canonical JSON at the frozen commit")
    return value


def _clean_tree_sha256(repository: Path, commit: str) -> str:
    with tempfile.TemporaryDirectory(prefix="redco-stage-d-v12-tree-") as temporary:
        root = Path(temporary)
        archive = root / "tree.tar"
        extracted = root / "tree"
        extracted.mkdir()
        subprocess.run(
            ["git", "archive", "--format=tar", "-o", str(archive), commit],
            cwd=repository,
            check=True,
        )
        with tarfile.open(archive) as bundle:
            bundle.extractall(extracted, filter="data")
        return _sha256(
            canonical_tree_manifest_bytes(extracted, allow_relative_symlinks=True)
        )


def _dependency_tree_sha256(root: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="redco-stage-d-v12-dependency-") as temporary:
        clean = Path(temporary) / "tree"
        shutil.copytree(
            root,
            clean,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
        )
        return _sha256(canonical_tree_manifest_bytes(clean, allow_relative_symlinks=True))


def _without_paths(value: Any, paths: set[tuple[str, ...]]) -> Any:
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        matching = {path[1:] for path in paths if path and path[0] == key}
        if () in matching:
            continue
        result[key] = _without_paths(item, matching)
    return result


def _fields(specification: str) -> set[str]:
    return set(specification.split())


def audit(
    repository: Path,
    *,
    tokenizer: Path,
    renderers: Path,
    verifiers: Path,
) -> dict[str, Any]:
    config = repository / "configs/stage-d"
    reports = repository / "reports"
    paths = {
        "dependency_auth_amendment": config
        / "stage-d1-support-dependency-auth-amendment-v12.json",
        "dependency_stack": config / "stage-d1-dependency-stack-v12.json",
        "genesis": config / "stage-d1-support-genesis-v12.json",
        "preregistration": config / "stage-d1-support-preregistration-v12.json",
        "protocol": config / "stage-d1-support-protocol-v12.json",
        "readiness": reports / "stage-d1-support-v12-cpu-readiness.json",
        "replay_config": config / "stage-d1-support-replay-eval-v12.toml",
        "source": config / "stage-d1-support-source-v12.json",
        "source_config": config / "stage-d1-support-source-eval-v12.toml",
    }
    values = {
        name: _read_canonical(path)
        for name, path in paths.items()
        if path.suffix == ".json"
    }
    hashes = {name: _sha256(path.read_bytes()) for name, path in paths.items()}
    parent_prereg = _read_commit_canonical(
        repository, "configs/stage-d/stage-d1-support-preregistration-v11-2.json"
    )
    prereg = values["preregistration"]
    changed_preregistration = {
        key
        for key in set(prereg) | set(parent_prereg)
        if key not in prereg
        or key not in parent_prereg
        or canonical_json(prereg[key]) != canonical_json(parent_prereg[key])
    }
    if changed_preregistration != EXPECTED_PREREGISTRATION_CHANGE_FIELDS:
        raise ValueError("v12 preregistration changes fields outside the allowlist")
    if (
        prereg["domain"] != "redco-stage-d1-support-preregistration-v12"
        or prereg["purpose"] != EXPECTED_PURPOSE
        or prereg["user_authorization"] != EXPECTED_USER_AUTHORIZATION
    ):
        raise ValueError("v12 purpose or user authorization differs")
    for field in PRESERVED_PREREGISTRATION_FIELDS:
        if canonical_json(prereg[field]) != canonical_json(parent_prereg[field]):
            raise ValueError(f"v12 changes frozen scientific field {field}")
    if prereg["cohort"]["collection_plan_sha256"] != parent_prereg["cohort"][
        "collection_plan_sha256"
    ]:
        raise ValueError("v12 changes the ordered cohort")
    parent_source = _read_commit_canonical(
        repository, "configs/stage-d/stage-d1-support-source-v11-2.json"
    )
    source = values["source"]
    shared_source_fields = set(parent_source) - SOURCE_CHANGE_FIELDS
    if any(
        canonical_json(source.get(field)) != canonical_json(parent_source[field])
        for field in shared_source_fields
    ):
        raise ValueError("v12 source changes a non-whitelisted field")
    if set(source) - set(parent_source) != {
        "historical_replay_report",
        "historical_replay_report_sha256",
        "measurement_semantics_changes",
        "redco_cpu_amendment_sha256",
    }:
        raise ValueError("v12 source additions differ from the reviewed allowlist")
    if (
        source["closure_readiness_report_sha256"] != hashes["readiness"]
        or source["dependency_auth_amendment_sha256"]
        != hashes["dependency_auth_amendment"]
        or source["historical_replay_report_sha256"]
        != values["readiness"]["artifacts"]["historical_replay_report_sha256"]
        or source["redco_cpu_amendment_sha256"]
        != values["readiness"]["artifacts"]["redco_cpu_amendment_sha256"]
    ):
        raise ValueError("v12 source binds a different readiness or dependency audit")
    if (
        source["redco_commit"] != RUNTIME_COMMIT
        or source["action_contract"] != EXPECTED_ACTION_CONTRACT
        or source["domain"] != "redco-stage-d1-support-source-v12"
        or source["closure_readiness_report"]
        != "reports/stage-d1-support-v12-cpu-readiness.json"
        or source["historical_replay_report"]
        != "reports/stage-d1-historical-semantic-replay-v1.json"
        or source["measurement_semantics_changes"]
        != EXPECTED_MEASUREMENT_SEMANTICS["source"]
        or any(source[field] != path for field, path in EXPECTED_SOURCE_RUNNERS.items())
    ):
        raise ValueError("v12 source identity or measurement semantics differ")
    for path_field, hash_field in (
        ("branch_runner", "branch_runner_sha256"),
        ("source_runner", "source_runner_sha256"),
    ):
        if _sha256(
            _git_bytes(repository, RUNTIME_COMMIT, source[path_field])
        ) != source[hash_field]:
            raise ValueError(f"v12 source runner differs: {path_field}")
    parent_protocol = _read_commit_canonical(
        repository, "configs/stage-d/stage-d1-support-protocol-v11-2.json"
    )
    protocol = values["protocol"]
    changed_protocol = {
        key
        for key in set(protocol) | set(parent_protocol)
        if key not in protocol
        or key not in parent_protocol
        or canonical_json(protocol[key]) != canonical_json(parent_protocol[key])
    }
    if changed_protocol != PROTOCOL_CHANGE_FIELDS:
        raise ValueError("v12 protocol changes scientific fields")
    parent_stack = _read_commit_canonical(
        repository, "configs/stage-d/stage-d1-dependency-stack-v11-2.json"
    )
    stack = values["dependency_stack"]
    if stack["redco_commit"] != RUNTIME_COMMIT:
        raise ValueError("v12 dependency stack binds a different Redco commit")
    if set(stack) != set(parent_stack) or any(
        set(item) != _fields("absolute_path name sha256")
        for item in stack["imported_modules"]
    ):
        raise ValueError("v12 dependency stack schema differs")
    if canonical_json(stack["components"]) != canonical_json(parent_stack["components"]):
        raise ValueError("v12 changes the pinned external dependency stack")
    component_trees = {
        item["name"]: item["post_tree_sha256"] for item in stack["components"]
    }
    if (
        _dependency_tree_sha256(renderers) != component_trees["renderers"]
        or _dependency_tree_sha256(verifiers) != component_trees["verifiers"]
    ):
        raise ValueError("v12 reconstructed dependency tree differs")
    for field in set(parent_stack) - {
        "imported_modules",
        "redco_commit",
        "redco_tree_sha256",
    }:
        if canonical_json(stack[field]) != canonical_json(parent_stack[field]):
            raise ValueError(f"v12 changes inherited dependency field {field}")
    if [
        (item["name"], item["absolute_path"]) for item in stack["imported_modules"]
    ] != [
        (item["name"], item["absolute_path"])
        for item in parent_stack["imported_modules"]
    ]:
        raise ValueError("v12 imported-module roster differs")
    if stack["redco_tree_sha256"] != _clean_tree_sha256(repository, RUNTIME_COMMIT):
        raise ValueError("v12 clean Redco archive tree differs")
    parent_imports = {item["name"]: item for item in parent_stack["imported_modules"]}
    for item in stack["imported_modules"]:
        name = item["name"]
        absolute = item["absolute_path"]
        if absolute.startswith("/workspace/redco/") and not absolute.startswith(
            "/workspace/redco/external/"
        ):
            relative = absolute.removeprefix("/workspace/redco/")
            actual = _sha256(_git_bytes(repository, RUNTIME_COMMIT, relative))
        elif name == "renderers":
            actual = _sha256((renderers / "renderers/__init__.py").read_bytes())
        elif name == "verifiers.v1":
            actual = _sha256((verifiers / "verifiers/v1/__init__.py").read_bytes())
        else:
            if item != parent_imports[name]:
                raise ValueError(f"v12 changes externally authenticated import {name}")
            continue
        if actual != item["sha256"]:
            raise ValueError(f"v12 imported module differs: {name}")
    if stack["program_sha256s"] != parent_stack["program_sha256s"]:
        raise ValueError("v12 changes a frozen program")
    for name, path in RUNTIME_PROGRAM_PATHS.items():
        if _sha256(_git_bytes(repository, RUNTIME_COMMIT, path)) != stack[
            "program_sha256s"
        ][name]:
            raise ValueError(f"v12 runtime program differs: {name}")
    for suffix in ("source", "replay"):
        parent_toml = tomllib.loads(
            _git_bytes(
                repository,
                RUNTIME_COMMIT,
                f"configs/stage-d/stage-d1-support-{suffix}-eval-v11-2.toml",
            ).decode()
        )
        current_toml = tomllib.loads(
            (config / f"stage-d1-support-{suffix}-eval-v12.toml").read_text()
        )
        if canonical_json(
            _without_paths(parent_toml, TOML_CHANGE_PATHS)
        ) != canonical_json(_without_paths(current_toml, TOML_CHANGE_PATHS)):
            raise ValueError(f"v12 {suffix} config changes a scientific field")
        if (
            current_toml["env"]["preregistration_sha256"]
            != hashes["preregistration"]
            or current_toml["env"]["source_sha256"] != hashes["source"]
            or current_toml["env"]["config_sha256"] != hashes["genesis"]
            or current_toml["output_dir"]
            != EXPECTED_TOML_PATHS[suffix]["output_dir"]
            or current_toml["env"]["ledger_path"]
            != EXPECTED_TOML_PATHS[suffix]["ledger_path"]
            or current_toml["env"]["artifact_path"]
            != EXPECTED_TOML_PATHS[suffix]["artifact_path"]
        ):
            raise ValueError(f"v12 {suffix} execution bindings differ")
    genesis = values["genesis"]
    parent_genesis = _read_commit_canonical(
        repository, "configs/stage-d/stage-d1-support-genesis-v11-2.json"
    )
    changed_genesis = {
        key
        for key in genesis
        if canonical_json(genesis[key]) != canonical_json(parent_genesis[key])
    }
    if changed_genesis != {
        "dependency_stack_sha256",
        "domain",
        "preregistration_sha256",
        "source_sha256",
    } or set(genesis) != set(parent_genesis):
        raise ValueError("v12 genesis changes a frozen runtime or scientific field")
    if genesis["dependency_stack_sha256"] != hashes["dependency_stack"] or genesis[
        "preregistration_sha256"
    ] != hashes["preregistration"] or genesis["source_sha256"] != hashes["source"]:
        raise ValueError("v12 genesis hash chain differs")
    if (
        protocol["dependency_stack_sha256"] != hashes["dependency_stack"]
        or protocol["genesis_config_sha256"] != hashes["genesis"]
        or protocol["preregistration_sha256"] != hashes["preregistration"]
        or protocol["scientific_eval_config_sha256"] != hashes["replay_config"]
        or protocol["source_eval_config_sha256"] != hashes["source_config"]
        or protocol["source_sha256"] != hashes["source"]
    ):
        raise ValueError("v12 protocol hash chain differs")
    readiness = values["readiness"]
    dependency_auth = values["dependency_auth_amendment"]
    parent_dependency_auth = _read_commit_canonical(
        repository,
        "configs/stage-d/stage-d1-support-dependency-auth-amendment-v11-2.json",
    )
    if (
        dependency_auth["dependency_stack_sha256"] != hashes["dependency_stack"]
        or dependency_auth["parent"]["readiness_v12_sha256"]
        != hashes["readiness"]
        or prereg["cpu_hardening"]["readiness_report_sha256"]
        != hashes["readiness"]
        or prereg["deployment_authentication"]["amendment_sha256"]
        != hashes["dependency_auth_amendment"]
        or prereg["deployment_authentication"]["dependency_stack_sha256"]
        != hashes["dependency_stack"]
        or prereg["deployment_authentication"]["code_commit"] != RUNTIME_COMMIT
        or prereg["deployment_authentication"]["verifier_sha256"]
        != dependency_auth["verifier"]["sha256"]
        or readiness["redco_commit"] != RUNTIME_COMMIT
        or dependency_auth["redco_binding"]["commit"] != RUNTIME_COMMIT
        or dependency_auth["redco_binding"]["clean_archive_tree_sha256"]
        != stack["redco_tree_sha256"]
    ):
        raise ValueError("v12 readiness and deployment cross-bindings differ")
    if (
        set(dependency_auth) != EXPECTED_DEPENDENCY_AUTH_KEYS
        or dependency_auth["classification"]
        != "outcome-independent-pre-model-action-contract-successor"
        or dependency_auth["domain"]
        != "redco-stage-d1-support-dependency-auth-amendment-v12"
        or dependency_auth["schema_version"] != 1
        or canonical_json(dependency_auth["prime_binding"])
        != canonical_json(parent_dependency_auth["prime_binding"])
        or canonical_json(dependency_auth["independent_child_tree_sha256s"])
        != canonical_json(parent_dependency_auth["independent_child_tree_sha256s"])
        or canonical_json(dependency_auth["preserved"])
        != canonical_json(parent_dependency_auth["preserved"])
        or dependency_auth["preobservation_evidence"][
            "inherited_infrastructure_pod_id"
        ]
        != parent_dependency_auth["preobservation_evidence"]["pod_id"]
        or set(dependency_auth["preobservation_evidence"])
        != _fields(
            "inherited_infrastructure_pod_id model_calls provider_posts response_bytes "
            "scientific_outputs source_rollouts support_gate_evaluations "
            "support_measurements"
        )
        or set(dependency_auth["redco_binding"])
        != _fields("clean_archive_tree_sha256 commit module_sha256s")
        or set(dependency_auth["verifier"]) != _fields("commit path sha256")
        or set(dependency_auth["parent"])
        != _fields(
            "dependency_auth_amendment_v11_2_sha256 dependency_auth_audit_v11_2_sha256 "
            "historical_replay_report_sha256 preregistration_v11_2_sha256 "
            "protocol_v11_2_sha256 readiness_v12_sha256"
        )
        or set(dependency_auth["redco_binding"]["module_sha256s"])
        != _fields(
            "src/redco/analysis/stage_d_exact_action.py "
            "src/redco/analysis/stage_d_live_observer.py "
            "src/redco/analysis/stage_d_receipt_ledger.py "
            "src/redco/analysis/stage_d_source_producer.py"
        )
        or dependency_auth["verifier"]["path"]
        != "scripts/verify_stage_d_dependency_deployment.py"
    ):
        raise ValueError("v12 dependency-auth inheritance or schema differs")
    dependency_parent_paths = {
        "dependency_auth_amendment_v11_2_sha256": "configs/stage-d/stage-d1-support-dependency-auth-amendment-v11-2.json",
        "dependency_auth_audit_v11_2_sha256": "reports/stage-d1-support-dependency-auth-audit-v11-2.json",
        "historical_replay_report_sha256": "reports/stage-d1-historical-semantic-replay-v1.json",
        "preregistration_v11_2_sha256": "configs/stage-d/stage-d1-support-preregistration-v11-2.json",
        "protocol_v11_2_sha256": "configs/stage-d/stage-d1-support-protocol-v11-2.json",
    }
    for field, path in dependency_parent_paths.items():
        if _sha256(_git_bytes(repository, RUNTIME_COMMIT, path)) != dependency_auth[
            "parent"
        ][field]:
            raise ValueError(f"v12 predecessor evidence differs: {path}")
    _require_hash(paths["readiness"], dependency_auth["parent"]["readiness_v12_sha256"])
    for path, digest in dependency_auth["redco_binding"]["module_sha256s"].items():
        if _sha256(_git_bytes(repository, RUNTIME_COMMIT, path)) != digest:
            raise ValueError(f"v12 Redco deployment binding differs: {path}")
    verifier_binding = dependency_auth["verifier"]
    if (
        verifier_binding["commit"] != RUNTIME_COMMIT
        or _sha256(_git_bytes(repository, RUNTIME_COMMIT, verifier_binding["path"]))
        != verifier_binding["sha256"]
    ):
        raise ValueError("v12 deployment verifier binding differs")
    artifact_paths = {
        "action_closure_corpus_sha256": "configs/stage-d/stage-d1-action-closure-corpus-v1.json",
        "action_closure_corpus_audit_sha256": "reports/stage-d1-action-closure-corpus-audit-v1.json",
        "historical_audit_script_sha256": "scripts/audit_stage_d_historical_semantics.py",
        "historical_replay_report_sha256": "reports/stage-d1-historical-semantic-replay-v1.json",
        "redco_cpu_amendment_sha256": "configs/stage-d/stage-d1-historical-replay-redco-amendment-v1.json",
    }
    for field, path in artifact_paths.items():
        if _sha256(_git_bytes(repository, RUNTIME_COMMIT, path)) != readiness["artifacts"][
            field
        ]:
            raise ValueError(f"v12 retained CPU artifact differs: {path}")
    _require_hash(
        repository / "scripts/stage_d_v12_preflight_plugin.py",
        readiness["artifacts"]["preflight_plugin_sha256"],
    )
    _require_hash(
        repository / "scripts/stage_d_v12_preflight.py",
        EXPECTED_PREFLIGHT_MODULE_SHA256,
    )
    if (
        readiness["artifacts"]["preflight_module_sha256"]
        != EXPECTED_PREFLIGHT_MODULE_SHA256
    ):
        raise ValueError("v12 preflight module binding differs")
    module_path = repository / "scripts/stage_d_v12_preflight.py"
    module_spec = importlib.util.spec_from_file_location("stage_d_v12_preflight", module_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError("v12 preflight module cannot be loaded")
    preflight_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(preflight_module)
    lineage = readiness["preobservation_lineage"]
    if (
        set(readiness) != EXPECTED_READINESS_KEYS
        or readiness["domain"] != "redco-stage-d1-support-v12-cpu-readiness"
        or readiness["schema_version"] != 1
        or set(lineage)
        != _fields(
            "dependency_auth_audit_v11_2_sha256 dependency_auth_v11_2_sha256 "
            "infrastructure_pod_activity_recorded model_responses provider_posts "
            "scientific_arm_outcomes source_rollouts support_gate_evaluations "
            "support_measurements"
        )
        or lineage["dependency_auth_v11_2_sha256"]
        != dependency_auth["parent"]["dependency_auth_amendment_v11_2_sha256"]
        or lineage["dependency_auth_audit_v11_2_sha256"]
        != dependency_auth["parent"]["dependency_auth_audit_v11_2_sha256"]
        or lineage["infrastructure_pod_activity_recorded"] is not True
        or set(readiness["artifacts"])
        != _fields(
            "action_closure_corpus_audit_sha256 action_closure_corpus_sha256 "
            "historical_audit_script_sha256 historical_replay_report_sha256 "
            "preflight_module_sha256 preflight_plugin_sha256 redco_cpu_amendment_sha256"
        )
        or set(readiness["measurement_semantics"])
        != _fields(
            "decision_rule_changes sampling_changes source_action_contract "
            "source_action_contract_changes"
        )
    ):
        raise ValueError("v12 readiness lineage or schema differs")
    cohort_artifacts = {
        source["collection_plan"]: source["collection_plan_sha256"],
        source["dataset"]: source["dataset_sha256"],
        source["dataset_manifest"]: source["dataset_manifest_sha256"],
        source["address_audit"]: source["address_audit_sha256"],
        source["scaffold"]: source["scaffold_sha256"],
        "configs/stage-d/stage-d1-base-model-manifest.json": prereg["policy"][
            "base_model_manifest_sha256"
        ],
        "reports/stage-d0-scaffold-step8-adapter-manifest-v1.json": prereg["policy"][
            "adapter_manifest_sha256"
        ],
        "configs/stage-d/stage-d1-tokenizer-manifest.json": prereg["policy"][
            "tokenizer_manifest_sha256"
        ],
        "configs/stage-d/stage-d1-renderer-manifest.json": prereg["policy"][
            "renderer_manifest_sha256"
        ],
        "configs/stage-d/stage-d1-sampler-conformance-manifest.json": prereg[
            "policy"
        ]["sampler_conformance_manifest_sha256"],
    }
    for path, digest in cohort_artifacts.items():
        if _sha256(_git_bytes(repository, RUNTIME_COMMIT, path)) != digest:
            raise ValueError(f"v12 cohort or policy artifact differs: {path}")
    parent_evidence = prereg["parent_terminal"]
    if (
        set(parent_evidence)
        != _fields(
            "dependency_auth_infrastructure_activity_recorded "
            "dependency_auth_v11_2_sha256 last_observed_terminal_report_sha256 "
            "model_responses predecessor_preregistration_sha256 "
            "predecessor_protocol_sha256 provider_posts scientific_arm_outcomes "
            "source_rollouts_committed support_gate_evaluations support_measurements"
        )
        or parent_evidence["dependency_auth_infrastructure_activity_recorded"]
        is not True
        or parent_evidence["dependency_auth_v11_2_sha256"]
        != dependency_auth["parent"]["dependency_auth_amendment_v11_2_sha256"]
        or parent_evidence["predecessor_preregistration_sha256"]
        != dependency_auth["parent"]["preregistration_v11_2_sha256"]
        or parent_evidence["predecessor_protocol_sha256"]
        != dependency_auth["parent"]["protocol_v11_2_sha256"]
        or _sha256(
            _git_bytes(repository, RUNTIME_COMMIT, "reports/stage-d1-support-v10-terminal.json")
        )
        != parent_evidence["last_observed_terminal_report_sha256"]
    ):
        raise ValueError("v12 parent-terminal evidence differs")
    if any(
        parent_evidence[field] != 0
        for field in (
            "model_responses",
            "provider_posts",
            "scientific_arm_outcomes",
            "source_rollouts_committed",
            "support_gate_evaluations",
            "support_measurements",
        )
    ):
        raise ValueError("v12 is not a zero-observation successor")
    if prereg["deployment_authentication"]["decision_rule_changes"] or prereg[
        "deployment_authentication"
    ]["sampling_changes"]:
        raise ValueError("v12 changes decision or sampling rules")
    cpu_hardening = prereg["cpu_hardening"]
    if (
        set(cpu_hardening)
        != _fields(
            "action_closure bound_code_commit completed_action_replay_count "
            "historical_failure_corpus_replayed historical_semantic_replay_count "
            "historical_semantic_replay_report_sha256 "
            "historical_support_density_identifiable readiness_report_sha256 "
            "restart_safe_max_token_action watchdog"
        )
        or cpu_hardening["action_closure"] != EXPECTED_ACTION_CLOSURE
        or cpu_hardening["bound_code_commit"] != RUNTIME_COMMIT
        or cpu_hardening["completed_action_replay_count"] != 12
        or cpu_hardening["historical_semantic_replay_count"] != 14
        or cpu_hardening["historical_failure_corpus_replayed"] is not True
        or cpu_hardening["historical_support_density_identifiable"] is not False
        or cpu_hardening["restart_safe_max_token_action"] is not True
        or cpu_hardening["historical_semantic_replay_report_sha256"]
        != readiness["artifacts"]["historical_replay_report_sha256"]
    ):
        raise ValueError("v12 CPU hardening evidence differs")
    prereg_runner = prereg["preflight"]["runner"]
    if (
        set(prereg["preflight"])
        != _fields(
            "apply_pinned_dependency_patches_before_import exact_collection "
            "fail_on_collection_mismatch fail_on_deselection fail_on_skip_or_xfail "
            "mandatory_failfast_assertions mandatory_prime_tests require_zero_skips "
            "runner single_fresh_process"
        )
        or set(prereg_runner)
        != _fields(
            "apply_pinned_dependency_patches_before_import argv_prefix "
            "single_fresh_process"
        )
        or set(prereg["deployment_authentication"])
        != _fields(
            "amendment_sha256 code_commit decision_rule_changes dependency_stack_sha256 "
            "measurement_semantics_changes sampling_changes verifier_sha256"
        )
        or prereg["preflight"]["mandatory_failfast_assertions"]
        != parent_prereg["preflight"]["mandatory_failfast_assertions"]
        or prereg["preflight"]["apply_pinned_dependency_patches_before_import"] is not True
        or prereg["preflight"]["fail_on_collection_mismatch"] is not True
        or prereg["preflight"]["fail_on_deselection"] is not True
        or prereg["preflight"]["fail_on_skip_or_xfail"] is not True
        or prereg["preflight"]["require_zero_skips"] is not True
        or prereg_runner["argv_prefix"] != preflight_module.EXPECTED_PREFLIGHT_RUNNER
        or prereg_runner["single_fresh_process"] is not True
        or prereg_runner["apply_pinned_dependency_patches_before_import"] is not True
        or prereg["preflight"]["single_fresh_process"] is not True
    ):
        raise ValueError("v12 preflight runner differs")
    if (
        prereg["deployment_authentication"]["measurement_semantics_changes"]
        != EXPECTED_MEASUREMENT_SEMANTICS["preregistration"]
        or dependency_auth["measurement_semantics_changes"]
        != EXPECTED_MEASUREMENT_SEMANTICS["dependency_auth"]
        or dependency_auth["sampling_changes"]
        or dependency_auth["scientific_decision_rule_changes"]
        or readiness["measurement_semantics"]["source_action_contract"]
        != EXPECTED_ACTION_CONTRACT
        or readiness["measurement_semantics"]["source_action_contract_changes"]
        != EXPECTED_MEASUREMENT_SEMANTICS["readiness"]
        or readiness["measurement_semantics"]["decision_rule_changes"]
        or readiness["measurement_semantics"]["sampling_changes"]
    ):
        raise ValueError("v12 disclosed measurement semantics differ")
    dependency_zero_fields = (
        "model_calls",
        "provider_posts",
        "response_bytes",
        "scientific_outputs",
        "source_rollouts",
        "support_gate_evaluations",
        "support_measurements",
    )
    readiness_zero_fields = (
        "model_responses",
        "provider_posts",
        "scientific_arm_outcomes",
        "source_rollouts",
        "support_gate_evaluations",
        "support_measurements",
    )
    if (
        dependency_auth["same_pod_continuation_authorized"] is not False
        or any(
            dependency_auth["preobservation_evidence"][field] != 0
            for field in dependency_zero_fields
        )
        or any(
            readiness["preobservation_lineage"][field] != 0
            for field in readiness_zero_fields
        )
        or readiness["status"] != "pass_cpu_only_live_not_authorized"
        or prereg["interpretation_and_disposition"]["scientific_training_authorized"]
        is not False
    ):
        raise ValueError("v12 exceeds the CPU-only zero-observation authorization")
    preflight = preflight_module.run_preflight(
        repository,
        prereg,
        tokenizer=tokenizer,
        renderers=renderers,
        verifiers=verifiers,
    )
    active_imports = preflight.pop("active_imports")
    if readiness["preflight"] != {
        "errors": preflight["collection_errors"],
        "failed": preflight["failed"],
        "nodeid_manifest_count": preflight["collected"],
        "nodeid_manifest_sha256": preflight["nodeid_manifest_sha256"],
        "passed": preflight["passed"],
        "runner": preflight_module.EXPECTED_PREFLIGHT_RUNNER,
        "selector_count": preflight["selector_count"],
        "selector_manifest_sha256": preflight["selector_manifest_sha256"],
        "skipped": preflight["skipped"],
        "xfail": preflight["xfail"],
    }:
        raise ValueError("v12 readiness report differs from fresh preflight")
    return {
        "checks": {
            "canonical_json_chain": True,
            "clean_runtime_commit_bound": True,
            "exact_cohort_and_scientific_fields_preserved": True,
            "external_dependency_stack_unchanged": True,
            "active_import_bindings_match_manifest": True,
            "max_token_measurement_change_disclosed": True,
            "preobservation_zero_counters_bound": True,
            "protocol_and_genesis_hash_chain_valid": True,
            "zero_skip_exact_preflight": True,
        },
        "audit_script_sha256": _sha256(Path(__file__).read_bytes()),
        "active_imports": active_imports,
        "domain": "redco-stage-d1-support-successor-preregistration-audit-v12",
        "hashes": hashes,
        "preflight": preflight,
        "preflight_module_sha256": EXPECTED_PREFLIGHT_MODULE_SHA256,
        "preflight_plugin_sha256": readiness["artifacts"]["preflight_plugin_sha256"],
        "redco_clean_archive_tree_sha256": stack["redco_tree_sha256"],
        "redco_commit": RUNTIME_COMMIT,
        "schema_version": 1,
        "status": "pass_cpu_only_live_requires_hardware_amendment",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--renderers-root", type=Path, required=True)
    parser.add_argument("--verifiers-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(
        args.repository.resolve(),
        tokenizer=args.tokenizer_path.resolve(),
        renderers=args.renderers_root.resolve(),
        verifiers=args.verifiers_root.resolve(),
    )
    encoded = canonical_json(report)
    if args.output is not None:
        args.output.write_bytes(encoded)
    print(encoded.decode())


if __name__ == "__main__":
    main()
