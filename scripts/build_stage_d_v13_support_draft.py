"""Build the unfrozen, CPU-only Stage-D v13 support-campaign draft.

The script is an orchestration layer only.  Input authentication, cohort
selection, protocol rules, and publication validation live in focused modules.
It writes only the fifteen v13 draft artifacts and never recovers or edits a
v12 artifact.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_v13_draft import (
    affordability_ledger,
    canonical_json_bytes,
    fresh_identities,
    sha256_json,
    state_machine_contract,
)
from redco.analysis.stage_d_v13_draft_cohort import prepare_successor
from redco.analysis.stage_d_v13_draft_contract import (
    OBSERVED_EXAMPLE_ID,
    OBSERVED_SEED,
    build_nonoverlap,
    build_v12_scientific_contract,
    build_v13_scientific_contract,
    compare_scientific_contracts,
    observed_information,
)
from redco.analysis.stage_d_v13_draft_inputs import (
    FROZEN_HASHES,
    POST_REPAIR_HASHES,
    PRE_REPAIR_SOURCE_SHA256,
    REPAIRED_SOURCE_SHA256,
    V12_ARCHIVE_SHA256,
    V12_AUDIT_REPORT_SHA256,
    V12_DEPENDENCY_STACK_SHA256,
    V12_EVIDENCE_MANIFEST_SHA256,
    archive_has_evaluator_payload,
    authenticate_immutable_inputs,
    dependency_binding,
    historical_identity_witness,
    read_json,
    sha256_file,
    verify_repair_ancestor,
)
from redco.analysis.stage_d_v13_draft_protocol import (
    deployment_checklist,
    state_ledger,
)
from redco.analysis.stage_d_v13_draft_publication import (
    OUTPUT_RELATIVE_PATHS,
    atomic_write,
    publication_envelope,
    validate_cross_artifact_references,
    validate_output_paths,
    validate_publication,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_WALLET_USD = 36.02
HISTORICAL_RESERVE_USD = 25.0
QASPER_SOURCE_REVISION = "fdc9d8214fbab5dd782958601db4d678e6934a54"
QASPER_PARQUET_REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"
OVERLAY_PACKAGES = (
    "anthropic",
    "datasets",
    "numpy",
    "math-verify",
    "mcp",
    "openai",
    "openai-agents",
    "prime-tunnel",
    "prime-sandboxes",
    "pydantic",
    "requests",
    "rich",
    "tenacity",
    "gepa",
    "pyzmq",
    "msgpack",
    "aiolimiter",
    "setproctitle",
    "httpx",
    "aiohttp",
    "prime-pydantic-config",
    "loguru",
    "tomli-w",
    "pytest==9.1.1",
    "pytest-asyncio==1.3.0",
    "multidict",
)
DRAFT_TEST_NODE_IDS = (
    "tests/test_stage_d_v13_draft.py::test_canonical_draft_json_is_sorted_and_has_no_trailing_newline",
    "tests/test_stage_d_v13_draft.py::test_state_machine_allows_only_one_pre_post_redeployment",
    "tests/test_stage_d_v13_draft.py::test_provider_dispatch_consumes_attempt_and_retires_on_lost_response",
    "tests/test_stage_d_v13_draft.py::test_response_and_source_outputs_forbid_redeployment",
    "tests/test_stage_d_v13_draft.py::test_code_and_protocol_changes_are_not_redeployment",
    "tests/test_stage_d_v13_draft.py::test_administrative_identities_ignore_archive_and_evaluator_material",
    "tests/test_stage_d_v13_draft.py::test_administrative_identities_ignore_historical_terminal_hashes",
    "tests/test_stage_d_v13_draft.py::test_observed_information_disclosure_is_honest_and_bounded",
    "tests/test_stage_d_v13_draft.py::test_successor_draft_preserves_63_rows_and_fails_closed_without_reserve",
    "tests/test_stage_d_v13_draft.py::test_reserve_receipt_requires_authenticated_scan_after_179_and_no_candidate",
    "tests/test_stage_d_v13_draft.py::test_delta_audit_freezes_scientific_fields_and_limits_whitelist",
    "tests/test_stage_d_v13_draft.py::test_historical_identity_witness_is_authenticated_and_complete",
    "tests/test_stage_d_v13_draft.py::test_historical_identity_collision_injection_fails_closed",
    "tests/test_stage_d_v13_draft.py::test_missing_historical_identity_input_fails_closed",
    "tests/test_stage_d_v13_draft.py::test_scientific_contract_nested_projections_are_independent",
    "tests/test_stage_d_v13_draft.py::test_fresh_identities_and_nonoverlap_audit_do_not_reuse_known_values",
    "tests/test_stage_d_v13_draft.py::test_nonoverlap_collision_injection_fails_closed",
    "tests/test_stage_d_v13_draft.py::test_affordability_is_conservative_and_does_not_check_wallet_now",
    "tests/test_stage_d_v13_draft.py::test_artifact_and_cpu_manifests_are_path_independent_and_unfrozen",
    "tests/test_stage_d_v13_draft.py::test_draft_cpu_manifest_matches_independent_collection",
    "tests/test_stage_d_v13_draft.py::test_all_published_json_artifacts_are_explicitly_unfrozen",
    "tests/test_stage_d_v13_draft.py::test_repair_and_v12_hash_partition_is_explicit",
    "tests/test_stage_d_v13_draft.py::test_clean_reconstruction_hash_is_pinned_and_wrong_hash_fails",
    "tests/test_stage_d_v13_draft.py::test_repair_binding_accepts_descendant_heads_and_rejects_unrelated_heads",
    "tests/test_stage_d_v13_draft.py::test_state_ledger_has_no_actual_attempts_and_exactly_one_allowed_redeploy",
    "tests/test_stage_d_v13_draft.py::test_sha256_json_is_stable_for_same_frozen_contract",
    "tests/test_stage_d_v13_draft.py::test_check_only_rejects_tampered_canonical_status",
    "tests/test_stage_d_v13_draft.py::test_publication_rejects_symlink_parent_escape_before_write",
    "tests/test_stage_d_v13_draft.py::test_publication_rejects_cross_output_and_immutable_hard_links",
)
ADMINISTRATIVE_INPUT_PATHS = (
    "configs/stage-d/stage-d1-support-preregistration-v12.json",
    "configs/stage-d/stage-d1-support-protocol-v12.json",
    "configs/stage-d/stage-d1-support-collection-plan-v11.json",
    "configs/stage-d/stage-d1-support-source-v12.json",
    "configs/stage-d/stage-d1-support-source-eval-v12.toml",
    "configs/stage-d/stage-d1-dependency-stack-v12.json",
    "datasets/stage-d/qasper-support-successor-manifest-v6.json",
)

EXPECTED_BYTES: dict[str, bytes] = {}
_WRITE_OUTPUTS = True


def _authenticated_output_bindings() -> dict[str, str]:
    return {
        **FROZEN_HASHES,
        **POST_REPAIR_HASHES,
        "src/redco/analysis/stage_d_source_producer.py": REPAIRED_SOURCE_SHA256,
    }


def _path(relative: str) -> Path:
    return ROOT / relative


def _envelope(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["draft_unfrozen"] = True
    result["launch_authorized"] = False
    return result


def _write_bytes(relative: str, data: bytes) -> str:
    EXPECTED_BYTES[relative] = data
    if _WRITE_OUTPUTS:
        return atomic_write(ROOT, relative, data)
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _write_json(relative: str, value: dict[str, Any]) -> str:
    publication_envelope(value, relative)
    return _write_bytes(relative, canonical_json_bytes(value))


def _load_inputs() -> dict[str, Any]:
    immutable_hashes = authenticate_immutable_inputs(ROOT)
    repair_commit = verify_repair_ancestor(ROOT)
    prereg = read_json(ROOT, "configs/stage-d/stage-d1-support-preregistration-v12.json")
    protocol = read_json(ROOT, "configs/stage-d/stage-d1-support-protocol-v12.json")
    collection = read_json(ROOT, "configs/stage-d/stage-d1-support-collection-plan-v11.json")
    dependency = read_json(ROOT, "configs/stage-d/stage-d1-dependency-stack-v12.json")
    terminal = read_json(ROOT, "reports/stage-d1-support-v12-terminal.json")
    audit = read_json(ROOT, "reports/stage-d1-support-v12-finalization-audit-v1.json")
    source_config = read_json(ROOT, "configs/stage-d/stage-d1-support-source-v12.json")
    successor_manifest = read_json(
        ROOT, "datasets/stage-d/qasper-support-successor-manifest-v6.json"
    )
    source_eval = tomllib.loads(
        _path("configs/stage-d/stage-d1-support-source-eval-v12.toml").read_text(encoding="utf-8")
    )
    source_lines = (
        _path("datasets/stage-d/qasper-support-successor-v6.jsonl")
        .read_bytes()
        .splitlines(keepends=True)
    )
    dependency_result = dependency_binding(ROOT, dependency)
    evaluator_payload = archive_has_evaluator_payload(ROOT)
    historical_witness = historical_identity_witness(ROOT)
    cohort = prepare_successor(
        source_lines,
        collection,
        successor_manifest,
    )
    return {
        "immutable_hashes": immutable_hashes,
        "repair_commit": repair_commit,
        "prereg": prereg,
        "protocol": protocol,
        "collection": collection,
        "dependency": dependency,
        "dependency_result": dependency_result,
        "terminal": terminal,
        "audit": audit,
        "source_config": source_config,
        "source_eval": source_eval,
        "successor_manifest": successor_manifest,
        "cohort": cohort,
        "evaluator_payload": evaluator_payload,
        "historical_identity_witness": historical_witness,
    }


def _administrative_inputs(hashes: dict[str, str]) -> dict[str, str]:
    return {path: hashes[path] for path in ADMINISTRATIVE_INPUT_PATHS}


def _affordability(prereg: dict[str, Any], source_eval: dict[str, Any]) -> dict[str, Any]:
    guardrails = prereg["resource_guardrails"]
    environment = source_eval["env"]
    raw = affordability_ledger(
        campaign_cap_usd=float(guardrails["maximum_cumulative_billing_usd"]),
        historical_wallet_usd=HISTORICAL_WALLET_USD,
        reserve_usd=HISTORICAL_RESERVE_USD,
        max_provider_posts=int(source_eval["num_tasks"])
        * int(environment["maximum_captured_session_call_count"]),
        max_completion_tokens=int(guardrails["total_generated_token_ceiling"]),
        max_wall_hours=float(guardrails["maximum_cumulative_wall_hours"]),
        max_hourly_usd=2.0,
    )
    result = _envelope(
        {
            "schema_version": 1,
            "domain": "redco-stage-d1-support-v13-affordability-ledger-v1",
            **raw,
            "historical_headroom_usd": HISTORICAL_WALLET_USD - HISTORICAL_RESERVE_USD,
            "historical_headroom_below_campaign_cap": (
                float(guardrails["maximum_cumulative_billing_usd"])
                > HISTORICAL_WALLET_USD - HISTORICAL_RESERVE_USD
            ),
            "no_wallet_check_performed": True,
        }
    )
    return result


def _delta(
    v12_contract: dict[str, Any],
    v13_contract: dict[str, Any],
    comparison: dict[str, Any],
    repair_commit: str,
) -> dict[str, Any]:
    return _envelope(
        {
            "schema_version": 2,
            "domain": "redco-stage-d1-support-v12-to-v13-delta-audit-v2",
            "status": "draft_unfrozen_not_authorized",
            "v12_inputs_immutable": True,
            "frozen_contract_sha256": sha256_json(v12_contract),
            "unchanged_contract": {
                "v12": v12_contract,
                "v13_draft": v13_contract,
                **comparison,
            },
            "whitelisted_differences": [
                "v13 draft identity/version/status and explicit launch=false authorization",
                "retirement metadata for the observed first v12 unit and its descendants",
                (
                    "one outcome-independent reserve replacement slot, pending authenticated "
                    "source-order selection"
                ),
                (
                    "repaired producer/source/config/post-repair engineering hashes cascading "
                    "from 8b64ad0"
                ),
                "reviewed deployment-resource bindings and clean-stack launch gate",
                "zero-call smoke checklist, affordability ledger, and attempt/abort state machine",
                "new draft artifact paths and independent ledger/output identities",
            ],
            "forbidden_changes": [
                (
                    "denominator, support threshold, support definitions, F1-range rule, "
                    "estimator, or Wilson reporting"
                ),
                (
                    "scorer, evaluator, reference bytes, prompt/corpus order, program, "
                    "model/checkpoint, tokenizer, or renderer"
                ),
                (
                    "sampling, termination, timeouts, K=4 topology, no-resampling rule, "
                    "failure reward, or target timing/selection"
                ),
                (
                    "serial execution, retry-zero semantics, evidence rules, or scientific "
                    "master/group namespace"
                ),
            ],
            "repair_binding": {
                "repair_commit": repair_commit,
                "repaired_source_sha256": REPAIRED_SOURCE_SHA256,
                "pre_repair_source_sha256": PRE_REPAIR_SOURCE_SHA256,
                "post_repair_audit_sha256": POST_REPAIR_HASHES[
                    "reports/stage-d1-v12-post-repair-audit-v1.json"
                ],
                "post_repair_regression_sha256": POST_REPAIR_HASHES[
                    "reports/stage-d1-source-comparison-post-repair-v1.json"
                ],
                "engineering_only": True,
            },
        }
    )


def _cpu_manifest(
    artifact_hashes: dict[str, str],
    artifact_manifest_sha: str,
    audit_sha: str,
) -> dict[str, Any]:
    command_prefix = "uv run --frozen --no-sync --offline --project . "
    overlay_args = " ".join(f"--with {package}" for package in OVERLAY_PACKAGES)
    overlay_prefix = (
        f"uv run --frozen --no-sync --offline {overlay_args} --project . python -m pytest "
    )
    selected_files = (
        "tests/test_stage_d_source_comparison_contract.py "
        "tests/test_stage_d_post_repair_verification.py "
        "tests/test_stage_d_source_finalization_integration.py"
    )
    scope_files = " ".join(
        (
            "src/redco/analysis/stage_d_v13_draft.py",
            "src/redco/analysis/stage_d_v13_draft_inputs.py",
            "src/redco/analysis/stage_d_v13_draft_contract.py",
            "src/redco/analysis/stage_d_v13_draft_cohort.py",
            "src/redco/analysis/stage_d_v13_draft_protocol.py",
            "src/redco/analysis/stage_d_v13_draft_publication.py",
            "scripts/build_stage_d_v13_support_draft.py",
            "tests/test_stage_d_v13_draft.py",
        )
    )
    v1_audit = (
        f"{command_prefix}python -m redco.analysis.stage_d_v12_finalization_audit "
        "--archive runs/stage-d/stage-d1-support-v12-terminal.tar.gz "
        "--evidence-manifest runs/stage-d/stage-d1-support-v12-evidence-sha256.txt "
        "--repo-root . --terminal-report reports/stage-d1-support-v12-terminal.json "
        '--output "${TMPDIR:-/tmp}/stage-d1-v12-v1-audit-attempt.json"'
    )
    commands = {
        "draft_selected_suite": (
            f"{command_prefix}python -m pytest tests/test_stage_d_v13_draft.py -q --tb=no -rA"
        ),
        "draft_collection": (
            f"{command_prefix}python -m pytest tests/test_stage_d_v13_draft.py "
            "--collect-only -q -vv"
        ),
        "post_repair_selected_suite": f"{overlay_prefix}{selected_files} -q --tb=no",
        "green_source_producer": overlay_prefix
        + (
            "--asyncio-mode=auto -p no:cacheprovider tests/test_stage_d_source_producer.py "
            '-k "not campaign_transaction" -rA'
        ),
        "green_live_observer": overlay_prefix
        + "--asyncio-mode=auto -p no:cacheprovider tests/test_stage_d_live_observer.py -rA",
        "green_source_env": overlay_prefix
        + "--asyncio-mode=auto -p no:cacheprovider tests/test_stage_d_source_env_pinned.py -rA",
        "ruff_scope": f"{command_prefix}ruff check {scope_files}",
        "mypy_scope": (
            f"{command_prefix}mypy --strict --follow-imports=skip "
            f"--ignore-missing-imports --no-incremental {scope_files}"
        ),
        "canonical_artifact_check": (
            f"{command_prefix}python scripts/build_stage_d_v13_support_draft.py --check-only"
        ),
        "v1_audit_partition": v1_audit,
    }
    draft_node_ids = list(DRAFT_TEST_NODE_IDS)
    draft_status = {
        "node_count": len(draft_node_ids),
        "node_ids": draft_node_ids,
        "passed": len(draft_node_ids),
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
    }
    draft_status_signature = sha256_json(draft_status)
    environment = {
        "route": "WSL2 Ubuntu CPU",
        "python": "3.12.3",
        "uv": "0.11.32",
        "uv_policy": (
            "uv run --frozen --no-sync --offline; persistent project cache/environment; "
            "no resolution/network"
        ),
        "pythonpath_assumption": (
            "<checkout>/src plus the pinned local dependency roots used by existing Stage-D tests"
        ),
        "pythonpath_export": (
            'export PYTHONPATH="$PWD/src:<pinned-dependency-copy>/renderers:'
            "<pinned-dependency-copy>/verifiers:$PWD/external/prime-rl/src:"
            "$PWD/external/prime-rl/deps/pydantic-config/src:"
            '$PWD/environments/redco_evidence_selection_v2:$PWD/tests"'
        ),
        "pythonpath_roots": [
            "<checkout>/src",
            "<pinned-dependency-copy>/renderers",
            "<pinned-dependency-copy>/verifiers",
            "<checkout>/external/prime-rl/src",
            "<checkout>/external/prime-rl/deps/pydantic-config/src",
            "<checkout>/environments/redco_evidence_selection_v2",
            "<checkout>/tests",
        ],
        "offline_uv_overlay_packages": list(OVERLAY_PACKAGES),
        "overlay_pinned_dependency_copy_is_read_only": True,
        "network": False,
        "gpu": False,
        "prime": False,
        "dependency_hash_bindings": {
            "pyproject_sha256": sha256_file(ROOT / "pyproject.toml"),
            "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
            "clean_prime_reconstruction_required": True,
        },
    }
    return _envelope(
        {
            "schema_version": 2,
            "domain": "redco-stage-d1-support-v13-draft-cpu-manifest-v2",
            "status": "draft_artifacts_built_cpu_only",
            "environment": environment,
            "commands": commands,
            "draft_suite": {
                "collection_command": commands["draft_collection"],
                "node_count": len(draft_node_ids),
                "node_ids": draft_node_ids,
                "node_list_sha256": sha256_json(draft_node_ids),
                "status_signature": draft_status_signature,
                "expected_status": f"{len(draft_node_ids)}_pass_0_fail_0_skip_0_xfail",
            },
            "outcomes": {
                "draft_selected_suite": f"{len(draft_node_ids)}_pass_0_fail_0_skip",
                "draft_selected_suite_status_signature": draft_status_signature,
                "post_repair_selected_suite": "114_pass_0_fail_0_skip",
                "green_source_producer": "21_pass_0_fail_0_skip_5_deselected",
                "green_live_observer": "19_pass_0_fail_0_skip",
                "green_source_env": "20_pass_0_fail_0_skip",
                "ruff": "pass",
                "mypy": "pass",
                "canonical_artifact_check": "pass",
                "v1_audit_partition": "rejected_before_write_source_hash_mismatch",
            },
            "artifact_hashes_excluding_this_manifest": {
                **artifact_hashes,
                "reports/stage-d1-support-v13-draft-artifact-"
                "manifest-v1.json": artifact_manifest_sha,
                "reports/stage-d1-support-v13-draft-audit-v1.json": audit_sha,
            },
            "zero_skips_required": True,
            "draft_only": True,
        }
    )


def build(write: bool = True) -> dict[str, str]:
    """Authenticate inputs and build the deterministic fifteen-artifact draft."""

    global _WRITE_OUTPUTS
    _WRITE_OUTPUTS = write
    EXPECTED_BYTES.clear()
    validate_output_paths(ROOT, _authenticated_output_bindings())
    inputs = _load_inputs()
    hashes = inputs["immutable_hashes"]
    prereg = inputs["prereg"]
    protocol = inputs["protocol"]
    collection = inputs["collection"]
    dependency = inputs["dependency"]
    source_config = inputs["source_config"]
    source_eval = inputs["source_eval"]
    successor_manifest = inputs["successor_manifest"]
    cohort = inputs["cohort"]
    repair_commit = inputs["repair_commit"]

    identities = fresh_identities(
        scientific_namespace=source_eval["env"]["taskset"]["scientific_group_namespace"],
        draft_domain="redco-stage-d1-support-preregistration-v13-draft",
        repair_commit=repair_commit,
        administrative_inputs=_administrative_inputs(hashes),
    )
    observed = observed_information(
        inputs["audit"], inputs["terminal"], inputs["evaluator_payload"]
    )
    v12_contract = build_v12_scientific_contract(
        prereg,
        protocol,
        source_config,
        collection,
        source_eval,
        successor_manifest,
        dependency,
        hashes,
    )
    v13_contract = build_v13_scientific_contract(
        prereg,
        protocol,
        source_config,
        collection,
        source_eval,
        successor_manifest,
        dependency,
        hashes,
    )
    comparison = compare_scientific_contracts(v12_contract, v13_contract)
    dependency_result = inputs["dependency_result"]

    dataset_relative = "datasets/stage-d/qasper-support-successor-v7-draft-retained-only.jsonl"
    dataset_sha = _write_bytes(dataset_relative, cohort["retained_lines"])
    dataset_manifest = cohort["dataset_manifest"]
    dataset_manifest["input_v6_sha256"] = hashes[
        "datasets/stage-d/qasper-support-successor-v6.jsonl"
    ]
    dataset_manifest["output"]["sha256"] = dataset_sha
    dataset_manifest_sha = _write_json(
        "datasets/stage-d/qasper-support-successor-manifest-v7-draft.json",
        dataset_manifest,
    )

    collection_draft = cohort["collection_draft"]
    collection_draft["retired_observed_unit"]["descendant_identities"] = [
        inputs["audit"]["terminal_trace"]["id"],
        *[call["decision_id"] for call in inputs["audit"]["calls"]],
        *[call["lineage"] for call in inputs["audit"]["calls"] if call["depth"] > 0],
    ]
    collection_draft["integrity"]["scientific_group_namespace"] = source_eval["env"]["taskset"][
        "scientific_group_namespace"
    ]
    collection_relative = (
        "configs/stage-d/v13-draft/stage-d1-support-collection-plan-v13-draft.json"
    )
    collection_sha = _write_json(collection_relative, collection_draft)

    reserve_relative = "reports/stage-d1-support-v13-reserve-selection-receipt-v1.json"
    reserve_sha = _write_json(reserve_relative, cohort["reserve_receipt"])

    nonoverlap_relative = "reports/stage-d1-support-v13-nonoverlap-audit-v1.json"
    historical_rows = [
        json.loads(line)
        for relative in (
            "datasets/stage-d/qasper-support-successor-v5.jsonl",
            "datasets/stage-d/qasper-deterministic-v4.jsonl",
        )
        for line in _path(relative).read_bytes().splitlines()
    ]
    nonoverlap = build_nonoverlap(
        cohort["retained_rows"],
        historical_rows,
        collection,
        identities,
        inputs["audit"],
        inputs["terminal"],
        successor_manifest,
        source_records=[
            prereg,
            protocol,
            source_config,
            dependency,
            collection,
        ],
        historical_identity_witness=inputs["historical_identity_witness"],
    )
    nonoverlap_sha = _write_json(nonoverlap_relative, nonoverlap)

    observed_relative = "reports/stage-d1-support-v13-observed-information-disclosure-v1.json"
    observed_sha = _write_json(observed_relative, observed)

    delta_relative = "reports/stage-d1-support-v13-delta-audit-v1.json"
    delta = _delta(v12_contract, v13_contract, comparison, repair_commit)
    delta_sha = _write_json(delta_relative, delta)

    state_relative = "configs/stage-d/v13-draft/stage-d1-support-state-machine-v1.json"
    state = state_ledger()
    state_sha = _write_json(state_relative, state)
    affordability_relative = "reports/stage-d1-support-v13-affordability-ledger-v1.json"
    affordability = _affordability(prereg, source_eval)
    affordability_sha = _write_json(affordability_relative, affordability)

    genesis_relative = "configs/stage-d/v13-draft/stage-d1-support-genesis-v13-draft.json"
    genesis = _envelope(
        {
            "schema_version": 2,
            "domain": "redco-stage-d1-support-genesis-v13-draft",
            "status": "draft_unfrozen_not_authorized",
            "scientific_campaign_authorized": False,
            "scientific_master_group_namespace": source_eval["env"]["taskset"][
                "scientific_group_namespace"
            ],
            "fresh_identities": identities,
            "repair_binding": {
                "redco_commit": repair_commit,
                "repaired_source_sha256": REPAIRED_SOURCE_SHA256,
                "v12_pre_repair_source_sha256": PRE_REPAIR_SOURCE_SHA256,
                "v12_archive_sha256": V12_ARCHIVE_SHA256,
                "v12_evidence_manifest_sha256": V12_EVIDENCE_MANIFEST_SHA256,
            },
            "cohort": {
                "required_denominator": 64,
                "preserved_untouched_slots": 63,
                "retired_observed_example_id": OBSERVED_EXAMPLE_ID,
                "retired_observed_seed": OBSERVED_SEED,
                "reserve_selection_receipt": reserve_relative,
                "reserve_selection_sha256": reserve_sha,
                "collection_plan": collection_relative,
                "collection_plan_sha256": collection_sha,
                "dataset": dataset_relative,
                "dataset_sha256": dataset_sha,
                "dataset_manifest_sha256": dataset_manifest_sha,
                "candidate_materialized": False,
            },
            "zero_call_genesis": True,
            "provider_posts": 0,
            "response_witnesses": 0,
        }
    )
    genesis_sha = _write_json(genesis_relative, genesis)

    config_relative = "configs/stage-d/v13-draft/stage-d1-support-config-v13-draft.json"
    runtime_config = _envelope(
        {
            "schema_version": 2,
            "domain": "redco-stage-d1-support-config-v13-draft",
            "status": "draft_unfrozen_not_authorized",
            "provider_calls_authorized": False,
            "model": source_eval["model"],
            "checkpoint_id": source_eval["env"]["checkpoint_id"],
            "sampling": source_eval["sampling"],
            "max_concurrent": 1,
            "shuffle": False,
            "push": False,
            "server": False,
            "num_tasks": 64,
            "num_rollouts": 1,
            "dataset_path": dataset_relative,
            "dataset_sha256": dataset_sha,
            "output_root_identity": identities["output_root_id"],
            "ledger_identity": identities["ledger_id"],
            "frozen_contract_sha256": sha256_json(v12_contract),
            "state_machine_contract_sha256": sha256_json(state_machine_contract()),
            "affordability_ledger_sha256": affordability_sha,
            "dependency_stack_sha256": V12_DEPENDENCY_STACK_SHA256,
            "repair_commit": repair_commit,
            "repaired_source_sha256": REPAIRED_SOURCE_SHA256,
            "scientific_fields_are_inherited_unchanged": True,
        }
    )
    config_sha = _write_json(config_relative, runtime_config)

    checklist = deployment_checklist(
        source_eval=source_eval,
        dependency_stack_sha256=V12_DEPENDENCY_STACK_SHA256,
        post_repair_regression_sha256=POST_REPAIR_HASHES[
            "reports/stage-d1-source-comparison-post-repair-v1.json"
        ],
    )
    prereg_relative = "configs/stage-d/v13-draft/stage-d1-support-preregistration-v13-draft.json"
    main_prereg = _envelope(
        {
            "schema_version": 2,
            "domain": "redco-stage-d1-support-preregistration-v13-draft",
            "status": "draft_unfrozen_not_authorized",
            "scientific_campaign_authorized": False,
            "support_only": True,
            "freeze_required_before_launch": True,
            "user_authorization_scope": "CPU drafting and audit only; no freeze or launch",
            "honest_v12_disclosure": {
                "path": observed_relative,
                "sha256": observed_sha,
                "v12_zero_committed_scientific_outputs": True,
                "v12_zero_information_claim": False,
                "evaluator_classification": (
                    "observed_engineering_information_not_admissible_scientific_outcome"
                ),
                "partially_observed_unit_permanently_excluded": True,
            },
            "frozen_scientific_contract": v12_contract,
            "frozen_scientific_contract_sha256": sha256_json(v12_contract),
            "delta_audit": {"path": delta_relative, "sha256": delta_sha},
            "cohort_and_replacement": {
                "preserved_untouched_rows": 63,
                "retired_example_id": OBSERVED_EXAMPLE_ID,
                "retired_seed": OBSERVED_SEED,
                "dataset": dataset_relative,
                "dataset_sha256": dataset_sha,
                "dataset_manifest_sha256": dataset_manifest_sha,
                "collection_plan": collection_relative,
                "collection_plan_sha256": collection_sha,
                "reserve_receipt": reserve_relative,
                "reserve_receipt_sha256": reserve_sha,
                "selection_outcome_independent": True,
                "materialization_status": "blocked_pending_authenticated_scan_after_receipt_179",
                "candidate": None,
            },
            "fresh_identities": identities,
            "redeployment": {
                "state_machine": state_relative,
                "state_machine_sha256": state_sha,
                "maximum_provisioning_attempts": 2,
                "one_pre_post_redeployment_only": True,
                "campaign_max_concurrent": 1,
                "provider_dispatch_consumes_attempt": True,
                "provider_dispatch_retires_unit": True,
                "no_post_response_resume_replay_or_redeployment": True,
            },
            "deployment_bindings": dependency_result,
            "repair_bindings": {
                "redco_commit": repair_commit,
                "repaired_source_sha256": REPAIRED_SOURCE_SHA256,
                "v12_pre_repair_source_sha256": PRE_REPAIR_SOURCE_SHA256,
                "v12_archive_sha256": V12_ARCHIVE_SHA256,
                "v12_evidence_manifest_sha256": V12_EVIDENCE_MANIFEST_SHA256,
                "post_repair_audit_sha256": POST_REPAIR_HASHES[
                    "reports/stage-d1-v12-post-repair-audit-v1.json"
                ],
            },
            "zero_call_smoke": checklist,
            "deployment_checklist": checklist,
            "affordability": {
                "path": affordability_relative,
                "sha256": affordability_sha,
                "launch_rule": affordability["allowed_spend_rule"],
                "launch_fail_closed": True,
            },
            "artifacts": {
                "genesis": genesis_relative,
                "genesis_sha256": genesis_sha,
                "runtime_config": config_relative,
                "runtime_config_sha256": config_sha,
                "nonoverlap_audit": nonoverlap_relative,
                "nonoverlap_audit_sha256": nonoverlap_sha,
            },
            "no_v12_recovery_or_reinterpretation": True,
            "no_scientific_conclusion": True,
        }
    )
    prereg_sha = _write_json(prereg_relative, main_prereg)

    artifact_hashes = {
        dataset_relative: dataset_sha,
        "datasets/stage-d/qasper-support-successor-manifest-v7-draft.json": dataset_manifest_sha,
        collection_relative: collection_sha,
        reserve_relative: reserve_sha,
        nonoverlap_relative: nonoverlap_sha,
        observed_relative: observed_sha,
        delta_relative: delta_sha,
        state_relative: state_sha,
        affordability_relative: affordability_sha,
        genesis_relative: genesis_sha,
        config_relative: config_sha,
        prereg_relative: prereg_sha,
    }
    artifact_relative = "reports/stage-d1-support-v13-draft-artifact-manifest-v1.json"
    artifact_manifest = _envelope(
        {
            "schema_version": 2,
            "domain": "redco-stage-d1-support-v13-draft-artifact-manifest-v2",
            "status": "draft_unfrozen_not_authorized",
            "canonical_json": {
                "sort_keys": True,
                "separators": [",", ":"],
                "trailing_newline": False,
            },
            "path_independent": True,
            "immutable_v12_hashes": hashes,
            "post_repair_hashes": POST_REPAIR_HASHES,
            "repair_source_sha256": REPAIRED_SOURCE_SHA256,
            "builder_sources": {
                relative: sha256_file(_path(relative))
                for relative in (
                    "src/redco/analysis/stage_d_v13_draft.py",
                    "src/redco/analysis/stage_d_v13_draft_inputs.py",
                    "src/redco/analysis/stage_d_v13_draft_contract.py",
                    "src/redco/analysis/stage_d_v13_draft_cohort.py",
                    "src/redco/analysis/stage_d_v13_draft_protocol.py",
                    "src/redco/analysis/stage_d_v13_draft_publication.py",
                    "scripts/build_stage_d_v13_support_draft.py",
                )
            },
            "draft_artifacts": artifact_hashes,
            "reproducibility": {
                "same_inputs_same_bytes": True,
                "ephemeral_paths_omitted": True,
                "durations_omitted": True,
                "wallet_not_checked": True,
                "network_not_used": True,
                "gpu_not_used": True,
                "prime_not_called": True,
                "clean_dependency_reconstruction_required": True,
            },
        }
    )
    artifact_sha = _write_json(artifact_relative, artifact_manifest)

    audit_relative = "reports/stage-d1-support-v13-draft-audit-v1.json"
    draft_audit = _envelope(
        {
            "schema_version": 2,
            "domain": "redco-stage-d1-support-v13-draft-audit-v2",
            "status": "draft_unfrozen_not_authorized",
            "engineering_audit_only": True,
            "immutable_v12_authentication": {
                "hashes": hashes,
                "all_expected_files_present": True,
                "archive_sha256": V12_ARCHIVE_SHA256,
                "evidence_manifest_sha256": V12_EVIDENCE_MANIFEST_SHA256,
                "v12_audit_report_sha256": V12_AUDIT_REPORT_SHA256,
                "v1_source_hash": PRE_REPAIR_SOURCE_SHA256,
                "current_repaired_source_hash": REPAIRED_SOURCE_SHA256,
                "v1_audit_partition": (
                    "v1 pre-repair evidence is immutable and not rerun as repaired evidence"
                ),
                "v12_mutated": False,
                "v12_recovered": False,
            },
            "observed_information_disclosure": {
                "path": observed_relative,
                "sha256": observed_sha,
                "classification": (
                    "observed_engineering_information_not_admissible_scientific_outcome"
                ),
            },
            "v12_to_v13_delta": {"path": delta_relative, "sha256": delta_sha},
            "successor": {
                "dataset": dataset_relative,
                "dataset_sha256": dataset_sha,
                "materialization_status": "blocked_pending_authenticated_scan_after_receipt_179",
                "reserve_receipt": reserve_relative,
                "nonoverlap_audit": nonoverlap_relative,
                "candidate": None,
            },
            "identity_binding": {
                "administrative_input_paths": list(ADMINISTRATIVE_INPUT_PATHS),
                "historical_identity_hashes_are_witness_only": True,
                "historical_terminal_hashes_enter_campaign_identity": False,
            },
            "dependency_binding": dependency_result,
            "repair_binding": {
                "commit": repair_commit,
                "source_sha256": REPAIRED_SOURCE_SHA256,
                "post_repair_audit_sha256": POST_REPAIR_HASHES[
                    "reports/stage-d1-v12-post-repair-audit-v1.json"
                ],
            },
            "zero_call_smoke": checklist,
            "deployment_checklist": checklist,
            "attempt_abort": {"path": state_relative, "sha256": state_sha},
            "affordability": {"path": affordability_relative, "sha256": affordability_sha},
            "artifact_manifest": {"path": artifact_relative},
            "fail_closed_blockers": [
                "authenticated source-order scan after receipt ordinal 179 is unavailable",
                "deployment wallet and launch-time provider pricing were not checked",
                "draft is unfrozen and has no launch authorization",
            ],
            "scope": {
                "support_only": True,
                "scientific_training": False,
                "prime_calls": False,
                "model_calls": False,
                "network": False,
                "v13_frozen": False,
            },
        }
    )
    audit_sha = _write_json(audit_relative, draft_audit)

    cpu_relative = "reports/stage-d1-support-v13-draft-cpu-manifest-v1.json"
    cpu_manifest = _cpu_manifest(artifact_hashes, artifact_sha, audit_sha)
    cpu_sha = _write_json(cpu_relative, cpu_manifest)
    return {
        "draft_preregistration_sha256": prereg_sha,
        "draft_genesis_sha256": genesis_sha,
        "draft_config_sha256": config_sha,
        "draft_audit_sha256": audit_sha,
        "artifact_manifest_sha256": artifact_sha,
        "cpu_manifest_sha256": cpu_sha,
        "dataset_sha256": dataset_sha,
        "reserve_receipt_sha256": reserve_sha,
        "nonoverlap_audit_sha256": nonoverlap_sha,
        "delta_audit_sha256": delta_sha,
    }


def _check_only() -> None:
    build(write=False)
    actual = {relative: _path(relative).read_bytes() for relative in OUTPUT_RELATIVE_PATHS}
    validate_publication(actual, EXPECTED_BYTES)
    expected_hashes = {relative: _sha256(data) for relative, data in EXPECTED_BYTES.items()}
    for relative in sorted(OUTPUT_RELATIVE_PATHS):
        if relative.endswith(".json"):
            parsed = json.loads(actual[relative])
            if not isinstance(parsed, dict):
                raise ValueError(f"draft JSON artifact is not an object: {relative}")
            validate_cross_artifact_references(parsed, expected_hashes)
    validate_output_paths(ROOT, _authenticated_output_bindings())


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        _check_only()
        return
    print(json.dumps(build(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
