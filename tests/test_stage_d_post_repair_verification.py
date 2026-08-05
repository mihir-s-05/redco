from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from stage_d_source_comparison_oracle import (
    MESSAGE_COMPARISON_ERROR,
    MESSAGE_COMPARISON_OWNER,
    MUTATION_CASES,
    RECORD_EXACTNESS_ONLY_SCOPE,
    production_boundary_observation,
    production_derived_owner_observation,
    production_record_owner_observation,
)

import redco.analysis.stage_d_v12_postrepair_audit as postrepair_audit_module
from redco.analysis.stage_d_receipt_ledger import inspect_ledger
from redco.analysis.stage_d_source_producer import (
    StageDSourceRolloutProducer,
    verify_source_trace_semantics,
)
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress
from redco.analysis.stage_d_v12_audit_common import (
    ARCHIVE_SHA256,
    EVIDENCE_MANIFEST_SHA256,
    FROZEN_REPO_FILE_SHA256,
    TERMINAL_REPORT_SHA256,
)
from redco.analysis.stage_d_v12_audit_common import sha256_file as immutable_sha256_file
from redco.analysis.stage_d_v12_finalization_audit import audit_archive
from redco.analysis.stage_d_v12_postrepair_audit import (
    APPROVED_REPAIRED_SOURCE_SHA256,
    POST_REPAIR_DOMAIN,
    SOURCE_RELATIVE,
    audit_postrepair,
)
from redco.contracts import canonical_json

ROOT = Path(__file__).parents[1]
ARCHIVE = ROOT / "runs" / "stage-d" / "stage-d1-support-v12-terminal.tar.gz"
MANIFEST = ROOT / "runs" / "stage-d" / "stage-d1-support-v12-evidence-sha256.txt"
TERMINAL_REPORT = ROOT / "reports" / "stage-d1-support-v12-terminal.json"
POST_REPAIR_REPORT = ROOT / "reports" / "stage-d1-v12-post-repair-audit-v1.json"
POST_REPAIR_MANIFEST = ROOT / "reports" / "stage-d1-source-comparison-post-repair-v1.json"

MESSAGE_CASE_IDS = tuple(
    case_id
    for case_id in sorted(MUTATION_CASES)
    if not case_id.startswith("record-")
    and case_id not in {"finish-reason-disagreement", "token-cap-disagreement"}
)
RECORD_CASE_IDS = tuple(
    case_id
    for case_id in sorted(MUTATION_CASES)
    if case_id.startswith("record-")
    or case_id in {"finish-reason-disagreement", "token-cap-disagreement"}
)


def test_postrepair_audit_partitions_immutable_v1_from_repaired_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = audit_postrepair(ROOT)
    assert report["domain"] == POST_REPAIR_DOMAIN
    assert report["status"] == "engineering_post_repair_verification_only"
    partition = report["v1_partition"]
    assert partition["archive_sha256"] == ARCHIVE_SHA256
    assert partition["evidence_manifest_sha256"] == EVIDENCE_MANIFEST_SHA256
    assert partition["terminal_report_sha256"] == TERMINAL_REPORT_SHA256
    assert partition["archive_manifest"] == {
        "all_match": True,
        "listed_count": 38,
        "matched_count": 38,
    }
    assert partition["pre_repair_source_sha256"] == FROZEN_REPO_FILE_SHA256[SOURCE_RELATIVE]
    assert partition["v1_inputs_untouched"] is True
    assert partition["v1_evidence_is_not_reinterpreted"] is True
    repaired = report["repaired_tree"]
    assert repaired["approved_repaired_source_sha256"] == APPROVED_REPAIRED_SOURCE_SHA256
    assert repaired["repaired_source_sha256"] == APPROVED_REPAIRED_SOURCE_SHA256
    assert repaired["approved_source_hash_authenticated"] is True
    assert repaired["record_binding"] == {
        "message_cases_bound": True,
        "record_cases_bound": False,
        "record_cases_scope": RECORD_EXACTNESS_ONLY_SCOPE,
        "record_binding_hook": (
            "tests/stage_d_source_comparison_oracle.py:"
            "record_exactness_binding_observation"
        ),
    }
    assert report["repaired_tree"]["source_hash_changed"] is True
    with pytest.raises(ValueError, match="immutable repository hash differs"):
        audit_archive(
            ARCHIVE,
            MANIFEST,
            repo_root=ROOT,
            terminal_report=TERMINAL_REPORT,
        )
    with monkeypatch.context() as patch:
        patch.setattr(
            postrepair_audit_module,
            "APPROVED_REPAIRED_SOURCE_SHA256",
            "0" * 64,
        )
        with pytest.raises(ValueError, match="not the approved post-repair value"):
            postrepair_audit_module.audit_postrepair(ROOT)

    real_sha256_file = immutable_sha256_file
    wrong_hash = "f" * 64

    def observed_hash(path: Path) -> str:
        if path.resolve() == ROOT / SOURCE_RELATIVE:
            return wrong_hash
        return real_sha256_file(path)

    with monkeypatch.context() as patch:
        patch.setattr(postrepair_audit_module, "sha256_file", observed_hash)
        with pytest.raises(ValueError, match="not the approved post-repair value"):
            postrepair_audit_module.audit_postrepair(ROOT)


def test_postrepair_report_is_canonical_and_has_no_frozen_v1_overwrite() -> None:
    raw = POST_REPAIR_REPORT.read_bytes()
    assert not raw.endswith(b"\n")
    assert raw == canonical_json(json.loads(raw))
    report = json.loads(raw)
    assert report["domain"] == POST_REPAIR_DOMAIN
    assert report["v1_partition"]["archive_sha256"] == ARCHIVE_SHA256
    assert report["v1_partition"]["evidence_manifest_sha256"] == EVIDENCE_MANIFEST_SHA256
    assert report["v1_partition"]["terminal_report_sha256"] == TERMINAL_REPORT_SHA256
    assert (ROOT / "reports/stage-d1-support-v12-finalization-audit-v1.json").read_bytes()
    assert all(
        hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
        for relative, expected in FROZEN_REPO_FILE_SHA256.items()
        if relative != SOURCE_RELATIVE
    )


def test_postrepair_manifest_is_canonical_and_records_polarity_transition() -> None:
    raw = POST_REPAIR_MANIFEST.read_bytes()
    assert not raw.endswith(b"\n")
    assert raw == canonical_json(json.loads(raw))
    manifest = json.loads(raw)
    assert manifest["domain"] == "redco-stage-d1-source-comparison-post-repair-manifest-v1"
    assert manifest["outcome"]["failed"] == 0
    assert manifest["outcome"]["skipped"] == 0
    assert manifest["polarity_transition"] == {
        "pre_repair_designated_red": True,
        "post_repair_designated_red": False,
        "repair_is_the_only_production_difference": True,
    }
    assert manifest["v1_partition"]["v1_report_unchanged"] is True
    assert manifest["v1_partition"]["v1_manifest_unchanged"] is True
    assert manifest["v1_partition"]["v1_archive_unchanged"] is True
    assert manifest["approved_repaired_source_sha256"] == APPROVED_REPAIRED_SOURCE_SHA256
    assert manifest["record_binding"] == {
        "message_cases_bound": True,
        "record_cases_bound": False,
        "record_cases_scope": RECORD_EXACTNESS_ONLY_SCOPE,
        "record_binding_hook": (
            "tests/stage_d_source_comparison_oracle.py:"
            "record_exactness_binding_observation"
        ),
    }


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "case_id", MESSAGE_CASE_IDS
)
def test_repaired_message_matrix_reaches_message_owner(case_id: str) -> None:
    transport, trace, expected = MUTATION_CASES[case_id]
    observation = production_boundary_observation(transport, trace)
    assert observation["transport_bytes"] == canonical_json(transport)
    assert observation["trace_bytes"] == canonical_json(trace)
    assert observation["transport_sha256"] == hashlib.sha256(
        canonical_json(transport)
    ).hexdigest()
    assert observation["trace_sha256"] == hashlib.sha256(canonical_json(trace)).hexdigest()
    if expected:
        assert observation["accepted"] is True, case_id
        assert observation["failure_owner"] is None, case_id
        assert observation["error_origin"] is None, case_id
    else:
        assert observation["accepted"] is False, case_id
        assert observation["failure_owner"] == MESSAGE_COMPARISON_OWNER, case_id
        assert observation["error_origin"] == MESSAGE_COMPARISON_ERROR, case_id


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "case_id", RECORD_CASE_IDS
)
def test_repaired_record_matrix_reports_unbound_exactness_scope(case_id: str) -> None:
    observation = production_record_owner_observation(case_id)
    assert observation["bound"] is False, case_id
    assert observation["status"] == "not_bound_post_repair", case_id
    assert observation["production_owner"] is None, case_id
    assert observation["owner"] == (
        "tests/stage_d_source_comparison_oracle.py:record_exactness_binding_observation"
    ), case_id
    assert observation["scope"] == RECORD_EXACTNESS_ONLY_SCOPE, case_id
    assert observation["error_origin"] == (
        "record mutation is frozen exactness-only and not production-bound"
    ), case_id


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("case_id", "expected_error"),
    (
        (
            "derived-sampled-node-bijection",
            "captured policy-call count differs from the Verifiers trace",
        ),
        (
            "derived-reward-components",
            "successful source trace requires explicit reward components",
        ),
        ("derived-finite-reward", "reward exact_span_f1 must be finite"),
    ),
)
def test_derived_mutations_reach_named_derive_owner(
    case_id: str, expected_error: str
) -> None:
    observation = production_derived_owner_observation(case_id)
    assert observation == {
        "bound": True,
        "case_id": case_id,
        "owner": "redco.analysis.stage_d_source_producer:derive_source_trace",
        "status": "rejected",
        "error_origin": expected_error,
    }


def test_source_semantic_mutation_reaches_semantic_owner(tmp_path: Path) -> None:
    from test_stage_d_source_producer import _episode, _produce

    source, ledger = _produce(tmp_path / "source-semantic-owner")
    try:
        mutated = type(source)._construct(
            source.group_id,
            source.rollout_id,
            source.reward + 1.0,
            source.stock_sequences,
            source.stock_sequence_decision_ids,
            source.decisions,
            source.child_target_roster,
            source.branch_eligible,
            source.ineligibility_reason,
            source.trace_sha256,
            source.reward_evidence_sha256,
            source.stock_sequences_evidence_sha256,
            source.base_model_manifest_sha256,
            source.evidence_class,
            source.producer_receipt,
        )
        with pytest.raises(ValueError, match="source rollout fields differ from semantic trace"):
            verify_source_trace_semantics(
                mutated,
                raw_episode=_episode(),
            )
    finally:
        ledger.close()


def test_ledger_mutation_reaches_poisoning_owner(tmp_path: Path) -> None:
    from test_stage_d_live_observer import _ledger
    from test_stage_d_source_producer import _action

    ledger = _ledger(tmp_path / "ledger-owner")
    producer = StageDSourceRolloutProducer(
        ledger=ledger,
        group_id="group-1",
        rollout_id="ledger-owner",
        child_target_roster=("target-0",),
        allow_test_fixture_roster=True,
        base_model_manifest_sha256="b" * 64,
    )
    action = _action(71)
    producer.intercept_policy_call(
        event_address=PolicyEventAddress(0, "root", 0, 0),
        action_key=action.key,
        node_kind="root",
        target_id=None,
        branch_selected=False,
        forward_once=lambda _key: action,
    )
    assert producer.abort_finalization(RuntimeError("owner probe")) is not None
    assert inspect_ledger(tmp_path / "ledger-owner").status == "poisoned"
    assert inspect_ledger(tmp_path / "ledger-owner").reason == (
        "ledger records an aborted source rollout finalization"
    )
    ledger.close()
