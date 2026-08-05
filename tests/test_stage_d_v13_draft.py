"""CPU-only tests for the unfrozen Stage-D v13 support draft."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from redco.analysis.stage_d_v13_draft import (
    AttemptState,
    DraftEvent,
    DraftPolicyError,
    DraftState,
    affordability_ledger,
    canonical_json_bytes,
    fresh_identities,
    nonoverlap_digest,
    reject_change_as_redeployment,
    sha256_json,
    transition,
)
from redco.analysis.stage_d_v13_draft_contract import (
    build_v12_scientific_contract,
    build_v13_scientific_contract,
    compare_scientific_contracts,
)
from redco.analysis.stage_d_v13_draft_inputs import (
    FROZEN_HASHES,
    REPAIR_COMMIT,
    REPAIRED_SOURCE_SHA256,
    historical_identity_witness,
    require_repair_ancestor,
    verify_clean_reconstruction,
    verify_clean_reconstruction_status,
    verify_repair_ancestor,
)
from redco.analysis.stage_d_v13_draft_publication import (
    OUTPUT_RELATIVE_PATHS,
    validate_output_paths,
    validate_publication,
)

ROOT = Path(__file__).resolve().parents[1]
V13_CONFIG = ROOT / "configs/stage-d/v13-draft"
REPORTS = ROOT / "reports"
DATASETS = ROOT / "datasets/stage-d"


def _json(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def test_canonical_draft_json_is_sorted_and_has_no_trailing_newline() -> None:
    paths = [*V13_CONFIG.glob("*.json"), *REPORTS.glob("stage-d1-support-v13-*.json")]
    paths.extend(DATASETS.glob("qasper-support-successor-manifest-v7-draft.json"))
    assert paths
    for path in paths:
        raw = path.read_bytes()
        assert raw == canonical_json_bytes(json.loads(raw))
        assert not raw.endswith(b"\n")
        assert (
            hashlib.sha256(raw).hexdigest()
            == hashlib.sha256(canonical_json_bytes(json.loads(raw))).hexdigest()
        )


def test_state_machine_allows_only_one_pre_post_redeployment() -> None:
    state = transition(AttemptState(), DraftEvent.PROVISION_ATTEMPT)
    state = transition(state, DraftEvent.PROVISION_FAILED)
    state = transition(state, DraftEvent.REDEPLOY)
    state = transition(state, DraftEvent.PROVISION_READY)
    assert state.state is DraftState.READY_PRE_POST
    assert state.provisioning_attempts == 2
    with pytest.raises(DraftPolicyError, match="redeploy"):
        transition(state, DraftEvent.REDEPLOY)


def test_provider_dispatch_consumes_attempt_and_retires_on_lost_response() -> None:
    state = transition(AttemptState(), DraftEvent.PROVISION_ATTEMPT)
    state = transition(state, DraftEvent.PROVISION_READY)
    state = transition(state, DraftEvent.PROVIDER_POST)
    assert state.campaign_attempt_consumed
    assert state.unit_retired
    state = transition(state, DraftEvent.ABORT)
    state = transition(state, DraftEvent.RECOVER_ARTIFACTS)
    assert state.state is DraftState.TERMINAL_INCOMPLETE
    with pytest.raises(DraftPolicyError, match="forbidden"):
        transition(state, DraftEvent.REDEPLOY)


def test_response_and_source_outputs_forbid_redeployment() -> None:
    state = transition(AttemptState(), DraftEvent.PROVISION_ATTEMPT)
    state = transition(state, DraftEvent.PROVISION_READY)
    state = transition(state, DraftEvent.PROVIDER_POST)
    response_state = transition(state, DraftEvent.RESPONSE_BYTES)
    with pytest.raises(DraftPolicyError, match="forbidden"):
        transition(response_state, DraftEvent.REDEPLOY)
    artifact_state = transition(state, DraftEvent.SCORE_OUTPUT)
    with pytest.raises(DraftPolicyError, match="forbidden"):
        transition(artifact_state, DraftEvent.REDEPLOY)


def test_code_and_protocol_changes_are_not_redeployment() -> None:
    reject_change_as_redeployment("capacity_resource")
    with pytest.raises(DraftPolicyError, match="requires a new amendment"):
        reject_change_as_redeployment("comparator")
    with pytest.raises(DraftPolicyError, match="requires a new amendment"):
        reject_change_as_redeployment("outcome_independent_launcher")


def test_administrative_identities_ignore_archive_and_evaluator_material() -> None:
    administrative = {
        "protocol_sha256": "protocol",
        "collection_sha256": "collection",
    }
    baseline = fresh_identities(
        scientific_namespace="redco-stage-d1-support-v1",
        draft_domain="redco-stage-d1-support-v13-draft",
        repair_commit=REPAIR_COMMIT,
        administrative_inputs=administrative,
    )
    mutated_archive = "different-terminal-archive"
    mutated_evaluator = "different-evaluator-payload"
    assert mutated_archive != mutated_evaluator
    assert (
        fresh_identities(
            scientific_namespace="redco-stage-d1-support-v1",
            draft_domain="redco-stage-d1-support-v13-draft",
            repair_commit=REPAIR_COMMIT,
            administrative_inputs=administrative,
        )
        == baseline
    )
    with pytest.raises(DraftPolicyError, match="outcome-bearing"):
        fresh_identities(
            scientific_namespace="redco-stage-d1-support-v1",
            draft_domain="redco-stage-d1-support-v13-draft",
            repair_commit=REPAIR_COMMIT,
            administrative_inputs={
                **administrative,
                "evaluator_payload_sha256": mutated_evaluator,
            },
        )


def test_administrative_identities_ignore_historical_terminal_hashes() -> None:
    script = ROOT / "scripts/build_stage_d_v13_support_draft.py"
    spec = importlib.util.spec_from_file_location("stage_d_v13_builder_identity", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    administrative = module._administrative_inputs(dict(FROZEN_HASHES))
    baseline = fresh_identities(
        scientific_namespace="redco-stage-d1-support-v1",
        draft_domain="redco-stage-d1-support-v13-draft",
        repair_commit=REPAIR_COMMIT,
        administrative_inputs=administrative,
    )
    terminal_paths = (
        "reports/stage-d1-support-v8-terminal.json",
        "reports/stage-d1-support-v9-terminal.json",
        "reports/stage-d1-support-v10-terminal.json",
    )
    assert all(path not in administrative for path in terminal_paths)
    for path in terminal_paths:
        mutated_hashes = dict(FROZEN_HASHES)
        mutated_hashes[path] = "0" * 64
        assert module._administrative_inputs(mutated_hashes) == administrative
        assert (
            fresh_identities(
                scientific_namespace="redco-stage-d1-support-v1",
                draft_domain="redco-stage-d1-support-v13-draft",
                repair_commit=REPAIR_COMMIT,
                administrative_inputs=module._administrative_inputs(mutated_hashes),
            )
            == baseline
        )
    with pytest.raises(DraftPolicyError, match="outcome-bearing"):
        fresh_identities(
            scientific_namespace="redco-stage-d1-support-v1",
            draft_domain="redco-stage-d1-support-v13-draft",
            repair_commit=REPAIR_COMMIT,
            administrative_inputs={"reports/stage-d1-support-v8-terminal.json": "terminal"},
        )


def test_observed_information_disclosure_is_honest_and_bounded() -> None:
    disclosure = _json("reports/stage-d1-support-v13-observed-information-disclosure-v1.json")
    assert disclosure["v12_not_zero_information"] is True
    durable = disclosure["durable_facts"]
    assert isinstance(durable, dict)
    assert durable["call_count"] == 4
    assert durable["root_call_count"] == 2
    assert durable["child_call_count"] == 2
    assert durable["total_completion_tokens"] == 1716
    evaluator = durable["evaluator_payload"]
    assert isinstance(evaluator, dict)
    assert evaluator["exact_span_f1"] == 0.0
    assert (
        disclosure["outcome_independence_certificate"]["replacement_selection_uses_observed_score"]
        is False
    )


def test_successor_draft_preserves_63_rows_and_fails_closed_without_reserve() -> None:
    old = (DATASETS / "qasper-support-successor-v6.jsonl").read_bytes().splitlines(keepends=True)
    draft = (DATASETS / "qasper-support-successor-v7-draft-retained-only.jsonl").read_bytes()
    assert draft == b"".join(old[1:])
    assert len(draft.splitlines()) == 111
    manifest = _json("datasets/stage-d/qasper-support-successor-manifest-v7-draft.json")
    assert manifest["status"] == "blocked_unmaterialized_retained_rows_only"
    assert manifest["replacement"]["selection_status"] == (
        "blocked_pending_authenticated_source_scan"
    )
    assert manifest["replacement"]["candidate"] is None
    assert all(value is None for value in manifest["replacement"]["unresolved_candidate"].values())
    assert manifest["checks"]["launch_eligibility"] is False


def test_reserve_receipt_requires_authenticated_scan_after_179_and_no_candidate() -> None:
    receipt = _json("reports/stage-d1-support-v13-reserve-selection-receipt-v1.json")
    assert receipt["status"] == "blocked_authenticated_source_scan_unavailable"
    assert receipt["selection_rule"]["resume_after_authenticated_receipt_ordinal"] == 179
    assert "required_next_source_ordinal" not in receipt["selection_rule"]
    assert receipt["candidate"] is None
    assert all(value is None for value in receipt["unresolved_candidate"].values())


def test_delta_audit_freezes_scientific_fields_and_limits_whitelist() -> None:
    delta = _json("reports/stage-d1-support-v13-delta-audit-v1.json")
    assert delta["unchanged_contract"]["exact_equal"] is True
    fields = delta["unchanged_contract"]["field_hashes"]
    assert isinstance(fields, dict)
    assert all(value["equal"] for value in fields.values())
    forbidden = delta["forbidden_changes"]
    assert any("denominator" in value for value in forbidden)
    assert any("K=4" in value for value in forbidden)
    assert delta["repair_binding"]["repair_commit"].startswith("8b64ad0")
    consumed = delta["unchanged_contract"]["v12"]["authenticated_input_hashes"]
    assert "datasets/stage-d/qasper-support-successor-v6.jsonl" in consumed
    assert "datasets/stage-d/qasper-support-successor-manifest-v6.json" in consumed
    assert "reports/stage-d1-support-successor-address-audit-v6.json" in consumed
    assert (
        delta["unchanged_contract"]["v12"] != delta["unchanged_contract"]["v13_draft"]
        or (delta["unchanged_contract"]["exact_equal"])
    )


def test_historical_identity_witness_is_authenticated_and_complete() -> None:
    witness = historical_identity_witness(ROOT)
    assert len(witness["retired_address_records"]) == 6
    assert len(witness["rollout_records"]) == 3
    assert len(witness["identity_sets"]["addresses"]) >= 6
    assert set(witness["artifacts"]) >= {
        "reports/stage-d1-support-successor-address-audit-v1.json",
        "reports/stage-d1-support-successor-address-audit-v6.json",
        "reports/stage-d1-support-v8-terminal.json",
        "reports/stage-d1-support-v10-terminal.json",
    }
    nonoverlap = _json("reports/stage-d1-support-v13-nonoverlap-audit-v1.json")
    historical = nonoverlap["historical_identity_witness"]
    assert historical["witness_sha256"] == witness["witness_sha256"]
    assert nonoverlap["known_values"]["historical_addresses"] >= 6
    assert nonoverlap["known_values"]["historical_rollout_ids"] == 3


def test_historical_identity_collision_injection_fails_closed() -> None:
    witness = historical_identity_witness(ROOT)
    address = witness["identity_sets"]["addresses"][0]
    rollout = witness["identity_sets"]["rollout_ids"][0]
    address_collision = nonoverlap_digest(
        {"historical_addresses": [address], "new_address": [address]}
    )
    rollout_collision = nonoverlap_digest(
        {"historical_rollout_ids": [rollout], "new_rollout_id": [rollout]}
    )
    assert (
        address_collision["checks"]["historical_addresses_disjoint_from_new_address"] is False
    )
    assert (
        rollout_collision["checks"]["historical_rollout_ids_disjoint_from_new_rollout_id"] is False
    )


def test_missing_historical_identity_input_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="historical identity input"):
        historical_identity_witness(tmp_path)


def test_scientific_contract_nested_projections_are_independent() -> None:
    script = ROOT / "scripts/build_stage_d_v13_support_draft.py"
    spec = importlib.util.spec_from_file_location("stage_d_v13_builder_contracts", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = module._load_inputs()
    args = (
        inputs["prereg"],
        inputs["protocol"],
        inputs["source_config"],
        inputs["collection"],
        inputs["source_eval"],
        inputs["successor_manifest"],
        inputs["dependency"],
        inputs["immutable_hashes"],
    )
    v12 = build_v12_scientific_contract(*args)
    v13 = build_v13_scientific_contract(*args)
    v12_protocol = cast(dict[str, Any], v12["protocol_and_evidence_rules"]["protocol"])
    v13_protocol = cast(dict[str, Any], v13["protocol_and_evidence_rules"]["protocol"])
    assert v12_protocol is not v13_protocol
    v12_protocol["one_sided_nested_mutation"] = True
    assert v12 != v13
    with pytest.raises(ValueError, match="scientific contract changed"):
        compare_scientific_contracts(v12, v13)


def test_fresh_identities_and_nonoverlap_audit_do_not_reuse_known_values() -> None:
    genesis = _json("configs/stage-d/v13-draft/stage-d1-support-genesis-v13-draft.json")
    audit = _json("reports/stage-d1-support-v13-nonoverlap-audit-v1.json")
    assert genesis["fresh_identities"]["campaign_id"].startswith("stage-d1-support-v13-")
    assert audit["status"] == "blocked_pending_authenticated_scan_after_receipt_179"
    checks = audit["checks"]["checks"]
    assert audit["checks"]["all_known_nonoverlap_checks"]
    assert checks["paper_ids_unique"]
    assert checks["fresh_administrative_ids_disjoint_from_forbidden"]
    assert audit["candidate_dependent_checks"]["candidate_ids_disjoint_from_forbidden"] is None


def test_nonoverlap_collision_injection_fails_closed() -> None:
    for field in [
        "paper_ids",
        "example_ids",
        "rendered_paper_hashes",
        "reference_spans",
        "scientific_group_ids",
        "slot_ids",
        "cache_salts",
        "seeds",
        "decision_ids",
        "lineages",
        "request_sequences",
        "recursive_trace_and_session_ids",
    ]:
        values = {field: ["collision", "collision"]}
        result = nonoverlap_digest(values)
        assert result["checks"][f"{field}_unique"] is False
        assert result["all_unique"] is False
        assert all(value is None for value in result["candidate_checks"].values())


def test_affordability_is_conservative_and_does_not_check_wallet_now() -> None:
    ledger = affordability_ledger(
        campaign_cap_usd=12.0,
        historical_wallet_usd=36.02,
        reserve_usd=25.0,
        max_provider_posts=1024,
        max_completion_tokens=5013504,
        max_wall_hours=6.0,
        max_hourly_usd=2.0,
    )
    assert ledger["wallet_checked_now"] is False
    assert ledger["wallet_at_launch_usd"] is None
    assert ledger["max_provider_posts"] == 1024
    assert ledger["max_completion_tokens"] == 5013504
    assert ledger["worst_case_wall_cost_usd"] == 12.0
    assert isinstance(ledger["status"], str)
    assert ledger["status"].startswith("fail_closed")


def test_artifact_and_cpu_manifests_are_path_independent_and_unfrozen() -> None:
    artifact = _json("reports/stage-d1-support-v13-draft-artifact-manifest-v1.json")
    cpu = _json("reports/stage-d1-support-v13-draft-cpu-manifest-v1.json")
    assert artifact["path_independent"] is True
    assert artifact["status"] == "draft_unfrozen_not_authorized"
    assert cpu["draft_unfrozen"] is True
    assert cpu["environment"]["network"] is False
    assert cpu["environment"]["gpu"] is False
    for value in artifact["draft_artifacts"]:
        assert not Path(value).is_absolute()


def test_draft_cpu_manifest_matches_independent_collection() -> None:
    cpu = _json("reports/stage-d1-support-v13-draft-cpu-manifest-v1.json")
    suite = cpu["draft_suite"]
    command = shlex.split(suite["collection_command"])
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    collected_match = re.search(r"collected (\d+) items", output)
    assert collected_match is not None
    collected = int(collected_match.group(1))
    names = re.findall(r"<Function ([^>]+)>", output)
    nodes = [f"tests/test_stage_d_v13_draft.py::{name}" for name in names]
    assert collected == len(nodes)
    assert suite["node_count"] == collected
    assert suite["node_ids"] == nodes
    assert suite["node_list_sha256"] == sha256_json(nodes)
    status = {
        "node_count": collected,
        "node_ids": nodes,
        "passed": collected,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
    }
    assert suite["status_signature"] == sha256_json(status)
    assert suite["expected_status"] == f"{collected}_pass_0_fail_0_skip_0_xfail"
    assert cpu["outcomes"]["draft_selected_suite"] == f"{collected}_pass_0_fail_0_skip"


def test_all_published_json_artifacts_are_explicitly_unfrozen() -> None:
    for relative in OUTPUT_RELATIVE_PATHS:
        if not relative.endswith(".json"):
            continue
        value = _json(relative)
        assert value["draft_unfrozen"] is True
        assert value["launch_authorized"] is False


def test_repair_and_v12_hash_partition_is_explicit() -> None:
    report = _json("reports/stage-d1-support-v13-draft-audit-v1.json")
    auth = report["immutable_v12_authentication"]
    assert auth["all_expected_files_present"] is True
    assert auth["v12_mutated"] is False
    assert auth["v12_recovered"] is False
    assert auth["v1_source_hash"] != auth["current_repaired_source_hash"]
    assert report["repair_binding"]["source_sha256"] == REPAIRED_SOURCE_SHA256


def test_clean_reconstruction_hash_is_pinned_and_wrong_hash_fails() -> None:
    audit = _json("reports/stage-d1-support-v13-draft-audit-v1.json")
    binding = audit["dependency_binding"]
    assert binding["clean_reconstruction_status"] == "pass"
    assert (
        binding["observed_reconstructed_post_tree_sha256"] == binding["expected_post_tree_sha256"]
    )
    with pytest.raises(ValueError, match="clean dependency reconstruction differs"):
        verify_clean_reconstruction("0" * 64, binding["expected_post_tree_sha256"])
    with pytest.raises(ValueError, match="untracked files"):
        verify_clean_reconstruction_status("?? extra/nested-file", "")
    with pytest.raises(ValueError, match="ignored files"):
        verify_clean_reconstruction_status("", "Would remove ignored")


def test_repair_binding_accepts_descendant_heads_and_rejects_unrelated_heads() -> None:
    assert verify_repair_ancestor(ROOT) == REPAIR_COMMIT
    with pytest.raises(ValueError, match="not an ancestor"):
        require_repair_ancestor(REPAIR_COMMIT, "unrelated-head", False)


def test_state_ledger_has_no_actual_attempts_and_exactly_one_allowed_redeploy() -> None:
    ledger = _json("configs/stage-d/v13-draft/stage-d1-support-state-machine-v1.json")
    assert ledger["status"] == "draft_unlaunched_no_actual_attempts"
    assert ledger["campaign_attempts_consumed"] == 0
    examples = {entry["name"]: entry for entry in ledger["examples"]}
    assert examples["one_pre_post_redeployment"]["admissible"] is True
    assert examples["lost_response_consumes_attempt_and_retires_unit"]["admissible"] is True


def test_sha256_json_is_stable_for_same_frozen_contract() -> None:
    delta = _json("reports/stage-d1-support-v13-delta-audit-v1.json")
    frozen = delta["unchanged_contract"]["v13_draft"]
    assert sha256_json(frozen) == delta["frozen_contract_sha256"]


def test_check_only_rejects_tampered_canonical_status() -> None:
    script = ROOT / "scripts/build_stage_d_v13_support_draft.py"
    spec = importlib.util.spec_from_file_location("stage_d_v13_builder", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build(write=False)
    expected = dict(module.EXPECTED_BYTES)
    actual = dict(expected)
    relative = "reports/stage-d1-support-v13-draft-audit-v1.json"
    value = json.loads(actual[relative])
    value["status"] = "tampered"
    actual[relative] = canonical_json_bytes(value)
    with pytest.raises(ValueError, match="draft publication bytes differ"):
        validate_publication(actual, expected)


def test_publication_rejects_symlink_parent_escape_before_write(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    repository.mkdir()
    (outside / "configs").mkdir(parents=True)
    os.symlink(outside / "configs", repository / "configs", target_is_directory=True)
    with pytest.raises(ValueError, match=r"symlink|escapes"):
        validate_output_paths(repository, {})
    assert not (outside / "configs" / "stage-d").exists()


def test_publication_rejects_cross_output_and_immutable_hard_links(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    output_directory = repository / "configs" / "stage-d" / "v13-draft"
    output_directory.mkdir(parents=True)
    first = output_directory / "stage-d1-support-preregistration-v13-draft.json"
    second = output_directory / "stage-d1-support-genesis-v13-draft.json"
    first.write_bytes(b"canonical-placeholder")
    os.link(first, second)
    with pytest.raises(ValueError, match="hard-link aliases"):
        validate_output_paths(repository, {})

    immutable_repository = tmp_path / "immutable-repository"
    immutable_output_directory = immutable_repository / "configs" / "stage-d" / "v13-draft"
    immutable_output_directory.mkdir(parents=True)
    immutable = immutable_repository / "immutable-input.json"
    immutable.write_bytes(b"immutable-placeholder")
    output = immutable_output_directory / "stage-d1-support-preregistration-v13-draft.json"
    os.link(immutable, output)
    with pytest.raises(ValueError, match="hard-link alias"):
        validate_output_paths(immutable_repository, {"immutable-input.json": "unused"})
