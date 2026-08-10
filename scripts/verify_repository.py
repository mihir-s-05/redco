"""One fail-closed owner for Redco's offline repository verification matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ENV_ROOT = ROOT / "environments" / "redco_evidence_selection_v2"
for _entry in reversed((ROOT, ROOT / "src", ROOT / "scripts", EVIDENCE_ENV_ROOT)):
    _entry_text = str(_entry)
    if _entry_text not in sys.path:
        sys.path.insert(0, _entry_text)

REQUIRED = "required-green"
REQUIRED_DEPENDENCY = "required-dependency"
PROTECTED = "protected-paused"
OUTCOME_NAMES = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
EXPECTED_REQUIRED_NODES = 1288
EXPECTED_REQUIRED_NODE_IDS_SHA256 = (
    "808f3863c123d2f277c96d6f9b9e7644007d7b068a86f55d4c2d1e9c0d15d22a"
)
ALL_PASS_PROFILES = frozenset((REQUIRED, "optional-stack"))
AMBIENT_PYTEST_CONTROLS = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")
PROFILE_CLASSES: dict[str, frozenset[str]] = {
    REQUIRED: frozenset((REQUIRED, REQUIRED_DEPENDENCY)),
    "platform-linux": frozenset(("platform-linux",)),
    "platform-windows": frozenset(("platform-windows",)),
    "optional-stack": frozenset(("optional-stack",)),
    "retained-provenance": frozenset(("retained-provenance",)),
    "inherited-known-failure": frozenset(("inherited-known-failure",)),
}
REFUSED_PROFILES = {
    "authenticated-windows": "requires separately authorized retained credentials/material",
    "external-checkout": "would inspect the user-owned external/prime-rl checkout",
    "source-auth": (
        "requires authenticated source material and the exact Python 3.12.3, "
        "PyArrow 25.0.0, datasets 5.0.0 runtime"
    ),
}

OPTIONAL_STACK_VERSIONS = {"tomli-w": "1.2.0"}
OPTIONAL_STACK_MODULES = (
    "httpx", "msgspec", "multidict", "peft", "pydantic", "safetensors.torch", "tomli_w",
    "torch", "transformers",
    "prime_rl.configs.trainer", "prime_rl.transport", "prime_rl.trainer.batch",
    "prime_rl.trainer.rl.data", "prime_rl.trainer.rl.loss",
    "prime_rl.trainer.rl.redco_loss", "prime_rl.trainer.rl.train", "prime_rl.trainer.utils",
    "redco_evidence_selection_v2.live_candidate",
    "redco_evidence_selection_v2.run_feasibility",
    "redco_evidence_selection_v2.scientific_env",
    "redco_evidence_selection_v2.source_env", "redco_evidence_selection_v2.taskset",
    "renderers.base", "renderers.client",
    "verifiers.v1.agent", "verifiers.v1.cli.eval.runner", "verifiers.v1.cli.output",
    "verifiers.v1.clients", "verifiers.v1.clients.config", "verifiers.v1.clients.train",
    "verifiers.v1.configs.eval", "verifiers.v1.dialects", "verifiers.v1.dialects.chat",
    "verifiers.v1.harness", "verifiers.v1.interception.server", "verifiers.v1.judges.rubric",
    "verifiers.v1.loaders", "verifiers.v1.rollout", "verifiers.v1.runtimes.base",
    "verifiers.v1.runtimes.docker", "verifiers.v1.serve.server", "verifiers.v1.session",
    "verifiers.v1.task",
)

CAPACITY_TEST = "tests/test_stage_d_v13_prime_capacity_monitor_v1.py"
CAPACITY_CONTRACT = "configs/stage-d/stage-d1-support-prime-capacity-monitor-v1.json"
CAPACITY_AUDIT = "reports/stage-d1-support-prime-capacity-monitor-audit-v1.json"
TOY_EXECUTOR_TEST = "tests/test_stage_d_toy_executor.py"

WHOLE_PATH_CLASSES: dict[str, str] = {
    CAPACITY_TEST: PROTECTED,
    "tests/test_stage_c9_efficiency.py": "optional-stack",
    "tests/test_stage_c9_partial_recovery.py": "optional-stack",
    "tests/test_stage_d_collection_runner.py": "optional-stack",
    "tests/test_stage_d_historical_semantics.py": "retained-provenance",
    "tests/test_stage_d_live_candidate_pinned.py": "optional-stack",
    "tests/test_stage_d_live_update_torch.py": "optional-stack",
    "tests/test_stage_d_objective_binding_prime.py": "optional-stack",
    "tests/test_stage_d_qwen_action_regression.py": "optional-stack",
    "tests/test_stage_d_scientific_env_pinned.py": "optional-stack",
    "tests/test_stage_d_source_env_pinned.py": "optional-stack",
    "tests/test_stage_d_source_finalization_integration.py": "optional-stack",
    TOY_EXECUTOR_TEST: "platform-windows",
    "tests/test_stage_d_patch_sequence.py": "external-checkout",
    "tests/test_stage_d_v13_prime_inventory_v3.py": "authenticated-windows",
    "tests/test_stage_d_v13_prime_inventory_v5.py": "authenticated-windows",
}

FUNCTION_CLASS_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    REQUIRED: {
        "tests/test_stage_d_qwen_action_regression.py": (
            "test_qwen_fixture_has_reviewed_bytes_and_token_counts",
        ),
        "tests/test_stage_d_v13_prime_inventory_v5.py": (
            "test_price_disk_memory_and_label_laws_are_exact",
            "test_unknown_key_is_row_local_and_duplicate_cloud_is_ambiguous",
            "test_only_literal_false_spot_qualifies",
            "test_community_price_only_literal_none_permits_on_demand_fallback",
        ),
        TOY_EXECUTOR_TEST: (
            "test_expired_callback_deadline_never_dispatches",
            "test_candidate_gateway_must_match_ledger_genesis",
            "test_candidate_gateway_binding_is_rechecked_at_dispatch",
            "test_cas_and_workspace_validation_fail_closed",
            "test_workspace_contract_rejects_unsafe_paths_and_links",
        ),
    },
    REQUIRED_DEPENDENCY: {
        "tests/test_stage_d_returning_root_correspondence.py": (
            "test_contract_schemas_are_resolvable_and_reject_mutations",
            "test_terminal_disposition_requires_source_ledger_owner",
        ),
        "tests/test_stage_d_source_producer.py": (
            "test_real_producer_trace_matches_the_persisted_raw_schema",
        ),
    },
    "platform-linux": {
        "tests/test_stage_d_dependency_stack.py": (
            "test_canonical_cache_archive_allows_only_internal_relative_symlinks",
        ),
        "tests/test_stage_d_evaluation_ledger.py": (
            "test_generic_trainer_receipt_cannot_claim_an_evaluation_launch",
        ),
        "tests/test_stage_d_evaluation_process.py": (
            "test_claim_and_exec_preserves_process_identity",
            "test_two_actuators_execute_one_claimed_target",
        ),
        "tests/test_stage_d_evaluation_supervisor_executor.py": (
            "test_reap_child_process_is_nonblocking",
            "test_actuator_gates_exec_and_cleans_target_descendant_and_cgroup",
        ),
        "tests/test_stage_d_process_supervision.py": (
            "test_wrapper_receipt_precedes_exec_and_tracks_the_same_process",
            "test_zombie_process_is_not_reported_as_live",
        ),
        "tests/test_stage_d_reload_supervisor.py": (
            "test_supervisor_owns_two_real_reload_processes",
            "test_two_supervisors_race_to_the_same_single_worker_pair",
            "test_reload_timeout_kills_the_entire_worker_process_group",
            "test_reload_cleanup_kills_descendants_after_session_leader_exits",
        ),
    },
    "optional-stack": {
        "tests/test_stage_d_deterministic_evidence.py": (
            "test_actual_frozen_qasper_taskset_ids_checkpoint_and_seed_domain",
            "test_prompt_profiles_separate_science_from_trace_fixture",
            "test_served_snapshot_and_renderer_identity_are_separate",
            "test_feasibility_forwards_complete_frozen_rlm_bundle",
            "test_run_grouped_installs_distinct_episode_cache_salts",
        ),
        "tests/test_stage_d_fixture_v4_7.py": ("test_v2_fixture_loads_through_real_taskset",),
        "tests/test_stage_d_judge_audit.py": (
            "test_judge_prompt_contains_reference_and_prediction",
        ),
        "tests/test_stage_d_live_observer.py": (
            "test_actual_interception_train_renderer_path_observes_bytes_once",
            "test_actual_two_turn_child_finalizes_as_excluded_without_replay",
        ),
        "tests/test_stage_d_reload_supervisor.py": (
            "test_real_transformers_peft_reload_hashes_the_loaded_adapter",
        ),
        "tests/test_stage_d_three_arm_bridge.py": (
            "test_actual_prime_packer_losses_and_gradients_match_independent_objectives",
            "test_actual_prime_rollout_payload_is_byte_stable_and_arm_specific",
            "test_actual_prime_tensor_conversion_passes_single_use_runtime_gate",
        ),
        "tests/test_stage_d_training_bridge.py": (
            "test_actual_prime_loss_drives_one_durable_optimizer_step",
            "test_actual_prime_msgpack_packer_and_clean_loss_match_manual_formula",
        ),
        "tests/test_stage_d_v13_support_launch.py": (
            "test_launch_tomls_pass_pinned_eval_config_and_reject_unsupported_tables",
        ),
    },
    "retained-provenance": {
        "tests/test_rlm_episode_replay.py": (
            "test_recovered_trace_is_parented_but_not_silently_migrated",
        ),
        "tests/test_stage_c6_repository_audits.py": (
            "test_v3_tree_fails_only_intermediate_repair_hash_checks",
        ),
        "tests/test_stage_c6_v2_preregistration.py": (
            "test_frozen_stage_c6_v2_protocol_passes_machine_audit",
        ),
        "tests/test_stage_c6_v2_repair.py": (
            "test_stage_c6_v2_bounded_repair_passes_machine_audit",
        ),
        "tests/test_stage_c6_v3_preregistration.py": (
            "test_stage_c6_v3_protocol_passes_machine_audit",
        ),
        "tests/test_stage_c6_v3_repair.py": (
            "test_stage_c6_v3_outcome_independent_repair_passes_audit",
        ),
        "tests/test_stage_c6_v3_repair2.py": (
            "test_stage_c6_v3_in_place_parser_repair_passes_audit",
        ),
        "tests/test_stage_d_action_closure.py": (
            "test_retained_raw_fixture_manifest_authenticates_exactly_fourteen",
        ),
        "tests/test_stage_d_post_repair_verification.py": (
            "test_postrepair_audit_partitions_immutable_v1_from_repaired_tree",
            "test_postrepair_report_is_canonical_and_has_no_frozen_v1_overwrite",
            "test_postrepair_manifest_is_canonical_and_records_polarity_transition",
        ),
        "tests/test_stage_d_v12_finalization_audit.py": (
            "test_terminal_v12_audit_is_total_and_immutable",
            "test_generated_audit_report_is_canonical_and_reproducible",
        ),
        "tests/test_stage_d_v13_prime_inventory_v2.py": (
            "test_artifact_build_is_deterministic_and_non_authorizing",
        ),
        "tests/test_stage_d_v13_prime_inventory_v4.py": (
            "test_terminal_v3_reassessment_is_exact_sanitized_and_non_authorizing",
            "test_offline_assessment_is_fixed_no_overwrite_and_tamper_evident",
            "test_raw_alias_linked_ancestor_and_output_alias_fail_before_writes",
            "test_v1_through_v3_are_immutable_and_v4_build_is_deterministic",
        ),
        "tests/test_stage_d_v13_selection_evidence.py": (
            "test_selection_evidence_manifest_binds_candidate_and_no_raw_transcript",
        ),
        "tests/test_stage_d_v13_draft.py": (
            "test_scientific_contract_nested_projections_are_independent",
            "test_check_only_rejects_tampered_canonical_status",
        ),
        "tests/test_stage_d_v13_prime_inventory_v5.py": ("test_no_shared_v5_live_artifact_exists",),
        "tests/test_stage_d_v13_source_phase_a.py": (
            "test_immutable_v1_audit_rejects_repaired_tree",
            "test_phase_a_cpu_manifest_matches_collection",
            "test_selector_universe_contains_all_authenticated_classes",
        ),
        "tests/test_stage_d_v13_support_launch.py": (
            "test_launch_bundle_rebuild_is_byte_identical",
            "test_launch_bundle_is_exactly_support_only",
            "test_launch_bundle_verifies_and_binds_plan_and_manifest",
            "test_check_only_does_not_write_or_repair",
            "test_check_only_rejects_tamper_without_mutation",
            "test_coordinated_authorization_mutation_fails_closed",
            "test_output_and_immutable_input_aliases_fail_closed",
            "test_source_free_build_never_reads_authenticated_parquet",
            "test_preflight_snapshot_is_canonical_and_real",
            "test_precommit_execution_is_non_authorizing_and_claim_free",
            "test_real_launch_owners_run_source_free_in_subprocess",
        ),
        "tests/test_stage_d_v13_support_readiness.py": (
            "test_future_preflight_contract_is_strictly_non_provisioning",
            "test_historical_v1_and_v12_are_immutable_and_stale_for_this_chain",
            "test_readiness_build_is_canonical_non_authorizing_and_deterministic",
        ),
    },
    "authenticated-windows": {
        "tests/test_stage_d_v13_prime_inventory_v4.py": (
            "test_installed_semantic_owners_bind_optional_spot_and_direct_memory",
        ),
    },
    "inherited-known-failure": {
        "tests/test_stage_d_v13_source_phase_a.py": (
            "test_pinned_source_artifact_metadata",
            "test_historical_receipts_174_through_179",
            "test_six_retired_rows_and_observed_unit",
            "test_forbidden_witness_is_complete_and_hashed",
            "test_phase_a_candidate_and_launch_flags_remain_unresolved",
            "test_scientific_binding_reuses_v1_law",
            "test_phase_a_artifacts_are_canonical_and_unfrozen",
            "test_two_fresh_output_roots_are_byte_identical",
            "test_phase_a_check_only_rejects_tampering",
            "test_forbidden_witness_rebuild_rejects_each_mutation",
            "test_forbidden_witness_rejects_recomputed_self_hash",
            "test_phase_a_publication_rejects_authenticated_input_hardlink",
            "test_phase_a_publication_rejects_cross_output_hardlink",
            "test_phase_a_publication_rejects_symlink_parent_before_write",
            "test_phase_a_status_capture_is_independent_and_exact",
            "test_phase_a_approval_anchor_authenticates_registry_policy",
            "test_phase_a_resume_invocation_count_is_zero",
        ),
    },
    "source-auth": {
        "tests/test_stage_d_v13_source_phase_a.py": (
            "test_legacy_datasets_decoder_batch_policy_is_rejected",
            "test_real_pinned_decoder_emits_one_bounded_batch",
            "test_dormant_resume_decoder_starts_at_ordinal_180",
            "test_dormant_resume_decoder_rejects_wrong_binding",
            "test_dormant_resume_decoder_requires_reviewed_checkpoint",
        ),
        "tests/test_stage_d_v13_support_protocol.py": (
            "test_candidate_materializer_stops_at_authenticated_ordinal_180",
            "test_support_protocol_dual_build_is_byte_identical",
            "test_candidate_composition_has_sixty_four_support_units_without_row_duplication",
            "test_upstream_input_failure_precedes_candidate_output",
            "test_runtime_mismatch_fails_before_source_access",
        ),
    },
    "external-checkout": {
        "tests/test_stage_d_support_successor.py": (
            "test_successor_protocol_freezes_only_authorized_changes",
            "test_later_successor_protocols_freeze_only_authorized_changes",
        ),
        "tests/test_stage_d_live_update.py": (
            "test_trainer_source_has_only_the_two_narrow_default_off_hooks",
        ),
    },
}

FUNCTION_CLASSES: dict[str, str] = {}
for _category, _paths in FUNCTION_CLASS_GROUPS.items():
    for _path, _names in _paths.items():
        for _name in _names:
            _node_id = f"{_path}::{_name}"
            _previous = FUNCTION_CLASSES.setdefault(_node_id, _category)
            if _previous != _category:
                raise RuntimeError(f"conflicting classifications for {_node_id}")

CLASSIFIED_TEST_PATHS = frozenset(WHOLE_PATH_CLASSES) | frozenset(
    node_id.split("::", 1)[0] for node_id in FUNCTION_CLASSES
)

FROZEN_RUFF_EXCEPTIONS = {
    "src/redco/analysis/stage_c5_v3_preregistration.py": (
        "E501",
        "8d134ec45c9a4c2381d72a4577a204f8a73bcb0a58522921d96d594e7bc2b58c",
    ),
    "src/redco/analysis/stage_c6_preregistration.py": (
        "E501",
        "9c21b2dea312f4de4640b60f3b41e36d8eb0b48c1c3c2e41e53fa8a49364a4d5",
    ),
    "tests/test_stage_c5_runtime_regression.py": (
        "I001",
        "afc863cf8b3e853e65e6e612a56377661cc6bae511e778f493061f9efb10d87d",
    ),
}

MYPY_REQUIRED_SCOPE = (
    "src/redco/integrity.py",
    "src/redco/analysis/frozen_rollout.py",
    "src/redco/analysis/gate_gb.py",
    "src/redco/analysis/stage_c4_selection_bundle_verification.py",
    "src/redco/analysis/stage_d_dependency_stack.py",
    "src/redco/analysis/stage_d_evaluation_barrier.py",
    "src/redco/analysis/stage_d_evaluation_capabilities.py",
    "src/redco/analysis/stage_d_evaluation_codec.py",
    "src/redco/analysis/stage_d_evaluation_contracts.py",
    "src/redco/analysis/stage_d_evaluation_driver.py",
    "src/redco/analysis/stage_d_evaluation_ledger.py",
    "src/redco/analysis/stage_d_evaluation_reducer.py",
    "src/redco/analysis/stage_d_evaluation_server.py",
    "src/redco/analysis/stage_d_evaluation_state.py",
    "src/redco/analysis/stage_d_v13_support_contract.py",
    "src/redco/analysis/stage_d_v13_support_protocol.py",
    "src/redco/analysis/stage_d_v13_support_publication.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_bytes())
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path.relative_to(ROOT).as_posix()} must contain a JSON object")
    return cast(dict[str, object], value)


def _object_at(owner: dict[str, object], key: str, source: str) -> dict[str, object]:
    value = owner.get(key)
    if not isinstance(value, dict) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{source}.{key} must be an object with string keys")
    return cast(dict[str, object], value)


def _string_at(owner: dict[str, object], key: str, source: str) -> str:
    value = owner.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{source}.{key} must be a string")
    return value


def _verify_capacity_binding() -> str:
    test_path = ROOT / CAPACITY_TEST
    contract_path = ROOT / CAPACITY_CONTRACT
    audit_path = ROOT / CAPACITY_AUDIT
    contract = _json_object(contract_path)
    audit = _json_object(audit_path)
    source_hashes = _object_at(contract, "source_hashes", CAPACITY_CONTRACT)
    file_bindings = _object_at(audit, "file_bindings", CAPACITY_AUDIT)
    audit_contract = _object_at(audit, "contract", CAPACITY_AUDIT)
    test_digest = _sha256(test_path)
    bound_digests = (
        _string_at(source_hashes, CAPACITY_TEST, f"{CAPACITY_CONTRACT}.source_hashes"),
        _string_at(file_bindings, CAPACITY_TEST, f"{CAPACITY_AUDIT}.file_bindings"),
    )
    if any(digest != test_digest for digest in bound_digests):
        raise ValueError(f"protected capacity test digest mismatch: actual {test_digest}")
    if _string_at(audit_contract, "path", f"{CAPACITY_AUDIT}.contract") != CAPACITY_CONTRACT:
        raise ValueError("capacity audit names an unexpected contract path")
    if _string_at(audit_contract, "sha256", f"{CAPACITY_AUDIT}.contract") != _sha256(contract_path):
        raise ValueError("capacity audit contract digest mismatch")
    return test_digest


def _relative(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


def _profile_selects(profile: str, category: str, path: str) -> bool:
    return category in PROFILE_CLASSES[profile] or (
        profile == "platform-windows" and path == TOY_EXECUTOR_TEST and category == REQUIRED
    )


def profile_outcomes_are_green(
    selected: int,
    executed: int,
    outcomes: Mapping[str, int],
) -> bool:
    return selected == executed == outcomes.get("passed", 0) and all(
        outcomes.get(name, 0) == 0 for name in OUTCOME_NAMES if name != "passed"
    )


def profile_exit_status(
    exitstatus: int,
    *,
    collect_only: bool,
    membership_ok: bool,
    selected: int,
    executed: int,
    outcomes: Mapping[str, int],
) -> int:
    if exitstatus != 0:
        return exitstatus
    if not membership_ok:
        return 1
    if collect_only or profile_outcomes_are_green(selected, executed, outcomes):
        return 0
    return 1


def _prepare_pytest_environment() -> tuple[str, ...]:
    inherited = tuple(name for name in AMBIENT_PYTEST_CONTROLS if name in os.environ)
    for name in inherited:
        del os.environ[name]
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return inherited


def node_ids_sha256(node_ids: Iterable[str]) -> str:
    return hashlib.sha256(
        b"\0".join(node_id.encode("utf-8") for node_id in sorted(node_ids))
    ).hexdigest()


def required_membership_matches(
    node_ids: Iterable[str],
    *,
    expected_count: int,
    expected_sha256: str,
) -> bool:
    collected = tuple(node_ids)
    return (
        len(collected) == len(set(collected)) == expected_count
        and node_ids_sha256(collected) == expected_sha256
    )


def _module_origin(name: str) -> str:
    module = importlib.import_module(name)
    origin = getattr(module, "__file__", None)
    return str(Path(origin).resolve()) if isinstance(origin, str) else "<namespace>"


def _run_pytest(profile: str, collect_only: bool) -> int:
    inherited_controls = _prepare_pytest_environment()
    if inherited_controls:
        print(
            "cleared ambient pytest controls: "
            + ", ".join(inherited_controls)
            + "; the repository runner owns pytest selection and plugins",
            file=sys.stderr,
        )
    if profile in REFUSED_PROFILES:
        print(f"refusing profile {profile}: {REFUSED_PROFILES[profile]}", file=sys.stderr)
        return 2
    selected_classes = PROFILE_CLASSES[profile]
    missing_classified_paths = sorted(
        path for path in CLASSIFIED_TEST_PATHS if not (ROOT / path).is_file()
    )
    if missing_classified_paths:
        print(
            "stale classified test paths: " + ", ".join(missing_classified_paths),
            file=sys.stderr,
        )
        return 2
    capacity_digest = _verify_capacity_binding()
    if profile == REQUIRED:
        try:
            jsonschema_version = importlib.metadata.version("jsonschema")
        except importlib.metadata.PackageNotFoundError:
            jsonschema_version = None
        if jsonschema_version != "4.26.0":
            print(
                "required dependency unavailable: expected jsonschema==4.26.0, got "
                f"{jsonschema_version or 'not installed'}; three portable security/schema nodes "
                "remain required and were not deselected",
                file=sys.stderr,
            )
            return 2
    if profile == "optional-stack":
        unavailable: list[str] = []
        observed: list[str] = []
        for distribution, expected in OPTIONAL_STACK_VERSIONS.items():
            try:
                actual = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                actual = "not installed"
            observed.append(f"{distribution}=={actual}")
            if actual != expected:
                unavailable.append(f"{distribution} expected {expected}, got {actual}")
        package_distributions = importlib.metadata.packages_distributions()
        failed_roots: set[str] = set()
        for module_name in OPTIONAL_STACK_MODULES:
            module_root = module_name.partition(".")[0]
            if module_root in failed_roots:
                continue
            try:
                origin = _module_origin(module_name)
            except Exception as error:
                unavailable.append(
                    f"module {module_name}: {type(error).__name__}: {error}"
                )
                failed_roots.add(module_root)
                continue
            distributions = package_distributions.get(module_root, ())
            versions = ",".join(
                f"{distribution}=={importlib.metadata.version(distribution)}"
                for distribution in distributions
            )
            observed.append(f"{module_name}={versions or 'unversioned'}@{origin}")
        if unavailable:
            print(
                "optional-stack runtime unavailable; no optional nodes collected: "
                + "; ".join(unavailable)
                + "; observed: "
                + "; ".join(observed),
                file=sys.stderr,
            )
            return 2
        print("optional-stack runtime: " + "; ".join(observed))

    import pytest
    from _pytest.config import Config
    from _pytest.main import Session
    from _pytest.nodes import Item
    from _pytest.reports import TestReport
    from _pytest.terminal import TerminalReporter

    selected_paths = {
        node_id.split("::", 1)[0]
        for node_id, category in FUNCTION_CLASSES.items()
        if _profile_selects(profile, category, node_id.split("::", 1)[0])
    }

    class MatrixPlugin:
        def __init__(self) -> None:
            self.preignored: Counter[str] = Counter()
            self.selected: Counter[str] = Counter()
            self.deselected: Counter[str] = Counter()
            self.executed: set[str] = set()
            self.selected_node_count = 0
            self.selected_node_sha256 = node_ids_sha256(())
            self.selected_node_ids_unique = True
            self.membership_ok = profile != REQUIRED

        def pytest_ignore_collect(self, collection_path: Path, config: Config) -> bool | None:
            del config
            relative = _relative(collection_path)
            if relative == CAPACITY_TEST:
                self.preignored[PROTECTED] += 1
                return True
            if (
                relative is None
                or not relative.startswith("tests/")
                or not relative.endswith(".py")
            ):
                return None
            whole_category = WHOLE_PATH_CLASSES.get(relative, REQUIRED)
            if profile == REQUIRED:
                if whole_category in selected_classes or relative in selected_paths:
                    return None
            elif _profile_selects(profile, whole_category, relative) or relative in selected_paths:
                return None
            self.preignored[whole_category] += 1
            return True

        @pytest.hookimpl(trylast=True)
        def pytest_collection_modifyitems(self, config: Config, items: list[Item]) -> None:
            kept: list[Item] = []
            removed: list[Item] = []
            collected_ids = {item.nodeid.split("[", 1)[0] for item in items}
            collected_paths = {item.nodeid.split("::", 1)[0] for item in items}
            stale = sorted(
                node_id
                for node_id in FUNCTION_CLASSES
                if node_id.split("::", 1)[0] in collected_paths and node_id not in collected_ids
            )
            if stale:
                raise pytest.UsageError("stale exact node classifications: " + ", ".join(stale))
            for item in items:
                path = item.nodeid.split("::", 1)[0]
                function_id = item.nodeid.split("[", 1)[0]
                category = FUNCTION_CLASSES.get(function_id, WHOLE_PATH_CLASSES.get(path, REQUIRED))
                if _profile_selects(profile, category, path):
                    kept.append(item)
                else:
                    self.deselected[category] += 1
                    removed.append(item)
            items[:] = kept
            if removed:
                config.hook.pytest_deselected(items=removed)

        @pytest.hookimpl(trylast=True)
        def pytest_collection_finish(self, session: Session) -> None:
            self.selected.clear()
            for item in session.items:
                path = item.nodeid.split("::", 1)[0]
                function_id = item.nodeid.split("[", 1)[0]
                category = FUNCTION_CLASSES.get(
                    function_id,
                    WHOLE_PATH_CLASSES.get(path, REQUIRED),
                )
                if not _profile_selects(profile, category, path):
                    raise pytest.UsageError(
                        f"final collection retained out-of-profile node: {item.nodeid}"
                    )
                self.selected[category] += 1
            node_ids = tuple(item.nodeid for item in session.items)
            self.selected_node_count = len(node_ids)
            self.selected_node_sha256 = node_ids_sha256(node_ids)
            self.selected_node_ids_unique = len(node_ids) == len(set(node_ids))
            if profile == REQUIRED:
                self.membership_ok = required_membership_matches(
                    node_ids,
                    expected_count=EXPECTED_REQUIRED_NODES,
                    expected_sha256=EXPECTED_REQUIRED_NODE_IDS_SHA256,
                )

        def pytest_runtest_logreport(self, report: TestReport) -> None:
            if report.when == "call" or (report.when == "setup" and not report.passed):
                self.executed.add(report.nodeid)

        def pytest_sessionfinish(self, session: Session, exitstatus: int) -> None:
            if profile not in ALL_PASS_PROFILES:
                return
            terminalreporter = cast(
                TerminalReporter,
                session.config.pluginmanager.get_plugin("terminalreporter"),
            )
            outcomes = {name: len(terminalreporter.stats.get(name, ())) for name in OUTCOME_NAMES}
            session.exitstatus = profile_exit_status(
                exitstatus,
                collect_only=session.config.option.collectonly,
                membership_ok=self.membership_ok,
                selected=sum(self.selected.values()),
                executed=len(self.executed),
                outcomes=outcomes,
            )

        def pytest_terminal_summary(
            self,
            terminalreporter: TerminalReporter,
            exitstatus: int,
            config: Config,
        ) -> None:
            del exitstatus
            stats = terminalreporter.stats
            outcomes = {name: len(stats.get(name, ())) for name in OUTCOME_NAMES}
            terminalreporter.write_sep("=", "Redco matrix classification")
            terminalreporter.write_line(f"profile: {profile}")
            terminalreporter.write_line(
                "selected nodes: "
                + str(sum(self.selected.values()))
                + f" {dict(sorted(self.selected.items()))}"
            )
            if profile == REQUIRED:
                verdict = "matched" if self.membership_ok else "MISMATCH"
                terminalreporter.write_line(
                    "required membership: "
                    f"expected count={EXPECTED_REQUIRED_NODES} "
                    f"sha256={EXPECTED_REQUIRED_NODE_IDS_SHA256}; "
                    f"observed count={self.selected_node_count} "
                    f"sha256={self.selected_node_sha256} "
                    f"unique={self.selected_node_ids_unique} ({verdict})"
                )
                if config.option.collectonly and self.membership_ok:
                    terminalreporter.write_line(
                        "mode: collect-only; trusted membership authenticated; "
                        "execution not asserted"
                    )
            terminalreporter.write_line(f"executed nodes: {len(self.executed)}")
            terminalreporter.write_line(
                "deselected nodes: "
                + str(sum(self.deselected.values()))
                + f" {dict(sorted(self.deselected.items()))}"
            )
            terminalreporter.write_line(
                "unavailable/preignored files: "
                + str(sum(self.preignored.values()))
                + f" {dict(sorted(self.preignored.items()))}"
            )
            terminalreporter.write_line(
                "outcomes: " + ", ".join(f"{name}={count}" for name, count in outcomes.items())
            )
            terminalreporter.write_line(
                f"protected files: 1 (capacity digest verified: {capacity_digest})"
            )

    args = [
        "tests",
        "-o",
        "addopts=",
        "--strict-markers",
        "--strict-config",
        "-p",
        "no:cacheprovider",
        "-ra",
    ]
    if collect_only:
        args.append("--collect-only")
    return int(pytest.main(args, plugins=[MatrixPlugin()]))


def _run_ruff() -> int:
    ignores: list[str] = []
    for relative, (rule, expected) in FROZEN_RUFF_EXCEPTIONS.items():
        actual = _sha256(ROOT / relative)
        if actual != expected:
            print(
                f"refusing frozen Ruff exception for {relative}: expected {expected}, got {actual}",
                file=sys.stderr,
            )
            return 2
        ignores.append(f"{relative}:{rule}")
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "src",
        "scripts",
        "tests",
        "environments",
        "--per-file-ignores",
        ",".join(ignores),
    ]
    print("Ruff scope: repository-wide; three exact-byte frozen violations authenticated")
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _run_mypy() -> int:
    print(
        f"required mypy scope: {len(MYPY_REQUIRED_SCOPE)} affected production modules; "
        "broad repository mypy: inherited-red and not claimed by this command"
    )
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--strict",
        "--no-incremental",
        *MYPY_REQUIRED_SCOPE,
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _run_compile() -> int:
    paths = sorted(
        path
        for owner in ("src", "scripts", "tests", "environments")
        for path in (ROOT / owner).rglob("*.py")
    )
    for path in paths:
        compile(path.read_bytes(), str(path), "exec")
    print(f"cache-free compile: {len(paths)} Python files")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pytest_parser = commands.add_parser("pytest", help="run one classified pytest row")
    pytest_parser.add_argument(
        "--profile",
        choices=sorted((*PROFILE_CLASSES, *REFUSED_PROFILES)),
        default=REQUIRED,
    )
    pytest_parser.add_argument(
        "--collect-only",
        action="store_true",
        help="authenticate classified membership without executing tests",
    )
    commands.add_parser("ruff", help="run repository Ruff with authenticated frozen exceptions")
    commands.add_parser("mypy", help="strict-check the affected production boundary")
    commands.add_parser("compile", help="compile repository Python bytes without caches")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "pytest":
        return _run_pytest(cast(str, args.profile), cast(bool, args.collect_only))
    if args.command == "ruff":
        return _run_ruff()
    if args.command == "mypy":
        return _run_mypy()
    if args.command == "compile":
        return _run_compile()
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
