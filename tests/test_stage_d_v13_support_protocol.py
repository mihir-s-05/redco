from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from redco.analysis import stage_d_v13_support_contract as contract
from redco.analysis import stage_d_v13_support_protocol as protocol
from redco.analysis.stage_d_v13_draft_publication import atomic_publish_set
from redco.analysis.stage_d_v13_support_protocol import (
    CANDIDATE_EXAMPLE_ID,
    CANDIDATE_ROW_SHA256,
    CANDIDATE_SOURCE_ORDINAL,
    CandidateReadInstrumentation,
    build_protocol_artifacts,
    check_protocol_artifacts,
    materialize_candidate,
    rebuild_protocol_artifacts_from_existing,
)


def test_candidate_materializer_stops_at_authenticated_ordinal_180(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    observer = CandidateReadInstrumentation()
    output = tmp_path / "candidate.json"
    payload = materialize_candidate(root, output, instrumentation=observer)
    assert observer.requested_ordinals == list(range(181))
    assert observer.arrow_batch_ranges == [(ordinal, ordinal) for ordinal in range(181)]
    assert observer.arrow_batch_cardinalities == [1] * 181
    assert observer.materialized_ordinals == [180]
    assert observer.canonicalized_ordinals == [180]
    assert observer.evaluated_ordinals == [180]
    assert observer.to_payload()["post_180_requested"] is False
    assert observer.to_payload()["post_180_materialized"] is False
    assert observer.to_payload()["arrow_logical_rows_emitted"] == 181
    assert payload["source"]["ordinal"] == CANDIDATE_SOURCE_ORDINAL
    assert payload["source"]["example_id"] == CANDIDATE_EXAMPLE_ID
    assert payload["source"]["row_sha256"] == CANDIDATE_ROW_SHA256
    assert payload["authority"]["source_selection_repeated"] is False
    assert payload["authority"]["provider_calls_authorized"] is False
    assert output.read_bytes()[-1:] != b"\n"


def test_support_protocol_dual_build_is_byte_identical(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    first = build_protocol_artifacts(root, tmp_path / "first")
    second = build_protocol_artifacts(root, tmp_path / "second")
    assert set(first) == set(second)
    assert all(first[path] == second[path] for path in first)
    protocol = json.loads(
        first["configs/stage-d/v13-draft/stage-d1-support-v13-frozen-support-protocol-v1.json"]
    )
    assert protocol["support_rule"]["denominator"] == 64
    assert protocol["support_rule"]["required_joint_successes"] == 58
    assert protocol["scientific_protocol"]["arms"] == ["stock", "branch-global", "local"]
    assert protocol["authorization"]["launch_authorized"] is False
    assert protocol["authorization"]["provider_calls_authorized"] is False
    assert protocol["attempt_policy"]["maximum_live_support_attempts_global"] == 1
    assert protocol["attempt_policy"]["outcome_bearing_cohorts"] == 1
    assert protocol["attempt_policy"]["second_outcome_bearing_attempt"] == (
        "forbidden_unconditionally"
    )
    assert protocol["source"]["required_runtime"] == {
        "python": "3.12.3",
        "pyarrow": "25.0.0",
        "datasets": "5.0.0",
    }
    assert protocol["scientific_protocol"]["checkpoint_retention_preflight"][
        "save_reload_reproduce_outputs"
    ] is True
    assert protocol["support_pass_transition"] == (
        "user_checkpoint_required_before_any_support_spend_or_science_transition"
    )
    assert protocol["authorization"]["readiness_blocker"] == (
        "exploratory_science_not_user_accepted"
    )


def test_candidate_composition_has_sixty_four_support_units_without_row_duplication(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    artifacts = build_protocol_artifacts(root, tmp_path)
    composition = json.loads(
        artifacts[
            "datasets/stage-d/qasper-support-successor-v8-candidate-composition-manifest-v1.json"
        ]
    )
    assert composition["support_cohort"] == {
        "required_papers": 64,
        "retained_support_rows": 63,
        "authenticated_replacement_rows": 1,
        "science_train_rows": 16,
        "science_eval_rows": 32,
    }
    assert composition["retained_base"]["byte_identity_preserved"] is True
    assert composition["nonoverlap"]["example_id"] is True
    assert composition["nonoverlap"]["historical_address_identities"] is True
    assert composition["authenticated_address_audit"] == {
        "preserved_count": 63,
        "retired_count": 1,
        "reserve_count": 1,
        "checks": {
            "preserved_63_addresses_exact": True,
            "prior_plan_has_64_unique_slots": True,
            "reserve_address_fresh": True,
            "retired_address_absent": True,
            "successor_plan_has_64_unique_slots": True,
        },
    }
    assert composition["authorization"]["science_authorized"] is False


def test_support_check_only_authenticates_published_bytes_without_source_materialization(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    artifacts = build_protocol_artifacts(root, tmp_path / "build")
    for relative, data in artifacts.items():
        path = tmp_path / "published" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    hashes = check_protocol_artifacts(root, tmp_path / "published")
    assert hashes == {
        relative: hashlib.sha256(data).hexdigest() for relative, data in artifacts.items()
    }


def test_support_check_only_uses_independent_reviewed_bytes_and_rejects_coordinated_tamper(
    tmp_path: Path,
) -> None:
    """The read-only checker cannot be blessed by rewriting its audit hash."""

    root = Path(__file__).parents[1]
    output_root = tmp_path / "published"
    for relative in protocol.REVIEWED_PROTOCOL_ARTIFACT_SHA256:
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / relative).read_bytes())

    def snapshot() -> dict[str, tuple[bytes, int, int]]:
        return {
            path.relative_to(output_root).as_posix(): (
                path.read_bytes(),
                path.stat().st_mtime_ns,
                path.stat().st_ino,
            )
            for path in output_root.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    assert protocol.check_protocol_artifacts(root, output_root) == (
        protocol.REVIEWED_PROTOCOL_ARTIFACT_SHA256
    )
    assert snapshot() == before

    candidate = output_root / protocol.CANDIDATE_RELATIVE
    candidate_payload = json.loads(candidate.read_bytes())
    candidate_payload["unknown_null"] = None
    candidate.write_bytes(protocol.canonical_json_bytes(candidate_payload))
    audit = output_root / protocol.PROTOCOL_AUDIT_RELATIVE
    audit_payload = json.loads(audit.read_bytes())
    audit_payload["candidate_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    audit.write_bytes(protocol.canonical_json_bytes(audit_payload))
    tampered = snapshot()
    with pytest.raises(ValueError, match="reviewed byte set"):
        protocol.check_protocol_artifacts(root, output_root)
    assert snapshot() == tampered

    audit.write_bytes((root / protocol.PROTOCOL_AUDIT_RELATIVE).read_bytes())
    audit.unlink()
    missing = snapshot()
    with pytest.raises(ValueError, match="published support artifact is missing"):
        protocol.check_protocol_artifacts(root, output_root)
    assert snapshot() == missing


def test_source_free_protocol_rebuild_is_deterministic(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    first = rebuild_protocol_artifacts_from_existing(root)
    second = rebuild_protocol_artifacts_from_existing(root)
    assert first == second
    assert {relative: hashlib.sha256(raw).hexdigest() for relative, raw in first.items()} == (
        protocol.REVIEWED_PROTOCOL_ARTIFACT_SHA256
    )


def test_support_check_only_rejects_tamper_or_missing_without_writes(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    output_root = tmp_path / "published"
    artifacts = build_protocol_artifacts(root, tmp_path / "build")
    atomic_publish_set(
        output_root,
        artifacts,
        manifest_path=protocol.PROTOCOL_AUDIT_RELATIVE,
    )
    before = {
        relative: (output_root / relative).read_bytes()
        for relative in artifacts
    }

    candidate_path = output_root / protocol.CANDIDATE_RELATIVE
    candidate_path.write_bytes(candidate_path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        check_protocol_artifacts(root, output_root)
    assert candidate_path.read_bytes() == before[protocol.CANDIDATE_RELATIVE] + b" "

    candidate_path.write_bytes(before[protocol.CANDIDATE_RELATIVE])
    audit_path = output_root / protocol.PROTOCOL_AUDIT_RELATIVE
    audit_bytes = audit_path.read_bytes()
    audit_path.unlink()
    with pytest.raises(ValueError, match="published support artifact is missing"):
        check_protocol_artifacts(root, output_root)
    assert not audit_path.exists()
    assert candidate_path.read_bytes() == before[protocol.CANDIDATE_RELATIVE]
    assert audit_bytes == before[protocol.PROTOCOL_AUDIT_RELATIVE]


def test_atomic_support_publication_check_only_is_read_only_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "publication"
    root.mkdir()
    payloads = {
        "candidate.json": b'{"candidate":null}',
        "audit.json": b'{"status":"candidate-null"}',
    }
    for relative, data in payloads.items():
        path = root / relative
        path.write_bytes(data)
    before = {
        relative: (
            (root / relative).read_bytes(),
            (root / relative).stat().st_mtime_ns,
            (root / relative).stat().st_ino,
        )
        for relative in payloads
    }
    assert atomic_publish_set(root, payloads, manifest_path="audit.json", check_only=True)
    after = {
        relative: (
            (root / relative).read_bytes(),
            (root / relative).stat().st_mtime_ns,
            (root / relative).stat().st_ino,
        )
        for relative in payloads
    }
    assert after == before

    (root / "candidate.json").write_bytes(b'{"candidate":"tampered"}')
    tampered_before = (root / "candidate.json").read_bytes()
    with pytest.raises(ValueError, match="publication bytes differ"):
        atomic_publish_set(root, payloads, manifest_path="audit.json", check_only=True)
    assert (root / "candidate.json").read_bytes() == tampered_before

    (root / "audit.json").unlink()
    missing_before = (root / "candidate.json").read_bytes()
    with pytest.raises(ValueError, match="published output is missing"):
        atomic_publish_set(root, payloads, manifest_path="audit.json", check_only=True)
    assert not (root / "audit.json").exists()
    assert (root / "candidate.json").read_bytes() == missing_before

    (root / "candidate.json").write_bytes(payloads["candidate.json"])
    (root / "audit.json").write_bytes(payloads["audit.json"])
    replace_count = 0
    real_replace = os.replace

    def fail_second_replace(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("injected publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="injected publication failure"):
        atomic_publish_set(root, payloads, manifest_path="audit.json")
    assert {relative: (root / relative).read_bytes() for relative in payloads} == payloads

    fresh = tmp_path / "fresh-publication-root"
    assert atomic_publish_set(fresh, payloads, manifest_path="audit.json")
    assert {relative: (fresh / relative).read_bytes() for relative in payloads} == payloads


def test_atomic_publication_restores_with_staged_replacements_when_rollback_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "rollback"
    payloads = {"a.json": b'{"a":1}', "manifest.json": b'{"manifest":1}'}
    root.mkdir()
    for relative, data in payloads.items():
        (root / relative).write_bytes(data.replace(b"1", b"0"))
    before = {relative: (root / relative).read_bytes() for relative in payloads}
    calls = 0
    real_replace = os.replace

    def fail_publish_and_first_rollback(
        source: os.PathLike[str], destination: os.PathLike[str]
    ) -> None:
        nonlocal calls
        calls += 1
        if calls in {2, 3}:
            raise OSError("injected replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_publish_and_first_rollback)
    with pytest.raises(OSError, match="injected replacement failure"):
        atomic_publish_set(root, payloads, manifest_path="manifest.json")
    assert {relative: (root / relative).read_bytes() for relative in payloads} == before


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "relative",
    [
        protocol.SELECTION_RECEIPT_RELATIVE,
        protocol.SELECTION_MANIFEST_RELATIVE,
        protocol.SELECTION_CLAIM_RELATIVE,
        protocol.SELECTION_ORIGINAL_CLAIM_RELATIVE,
        protocol.V12_ARCHIVE_RELATIVE,
        protocol.V12_EVIDENCE_MANIFEST_RELATIVE,
        protocol.V12_TERMINAL_REPORT_RELATIVE,
        protocol.V12_FINALIZATION_AUDIT_RELATIVE,
        protocol.V12_PREREG_RELATIVE,
        protocol.V12_PROTOCOL_RELATIVE,
        protocol.V12_SOURCE_EVAL_RELATIVE,
        protocol.FROZEN_SUPPORT_RULES_RELATIVE,
        protocol.RETAINED_SUPPORT_RELATIVE,
        protocol.COLLECTION_PLAN_RELATIVE,
        protocol.V6_MANIFEST_RELATIVE,
        protocol.ADDRESS_AUDIT_RELATIVE,
        protocol.SOURCE_ARTIFACT_RELATIVE,
    ],
)
def test_upstream_input_failure_precedes_candidate_output(
    relative: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).parents[1]
    original_reader = contract.read_authenticated

    def reject_mutated_input(
        input_root: Path, input_relative: str, expected_sha256: str
    ) -> bytes:
        if input_relative == relative:
            raise ValueError(f"authenticated upstream evidence changed: {relative}")
        return original_reader(input_root, input_relative, expected_sha256)

    if relative == protocol.SOURCE_ARTIFACT_RELATIVE:
        def reject_source_contract(*_args: object, **_kwargs: object) -> object:
            raise ValueError(f"authenticated upstream evidence changed: {relative}")

        monkeypatch.setattr(protocol, "source_contract", reject_source_contract)
    else:
        monkeypatch.setattr(contract, "read_authenticated", reject_mutated_input)
    output = tmp_path / "candidate.json"
    with pytest.raises(ValueError, match="authenticated upstream evidence changed"):
        protocol.materialize_candidate(root, output)
    assert not output.exists()


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "relative",
    [
        protocol.SELECTION_RECEIPT_RELATIVE,
        protocol.SELECTION_MANIFEST_RELATIVE,
        protocol.SELECTION_CLAIM_RELATIVE,
        protocol.SELECTION_ORIGINAL_CLAIM_RELATIVE,
        protocol.V12_ARCHIVE_RELATIVE,
        protocol.V12_EVIDENCE_MANIFEST_RELATIVE,
        protocol.V12_TERMINAL_REPORT_RELATIVE,
        protocol.V12_FINALIZATION_AUDIT_RELATIVE,
        protocol.V12_PREREG_RELATIVE,
        protocol.V12_PROTOCOL_RELATIVE,
        protocol.V12_SOURCE_EVAL_RELATIVE,
        protocol.FROZEN_SUPPORT_RULES_RELATIVE,
        protocol.RETAINED_SUPPORT_RELATIVE,
        protocol.COLLECTION_PLAN_RELATIVE,
        protocol.V6_MANIFEST_RELATIVE,
        protocol.ADDRESS_AUDIT_RELATIVE,
        protocol.SOURCE_ARTIFACT_RELATIVE,
    ],
)
def test_support_outputs_reject_hard_link_to_every_authenticated_input(
    relative: str, tmp_path: Path
) -> None:
    from redco.analysis.stage_d_v13_draft_publication import validate_output_paths

    root = tmp_path / "repository"
    root.mkdir()
    source = root / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"immutable-input")
    output = root / protocol.CANDIDATE_RELATIVE
    output.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, output)
    with pytest.raises(ValueError, match="hard-link alias"):
        validate_output_paths(
            root,
            {str(source.resolve()): hashlib.sha256(b"immutable-input").hexdigest()},
            output_paths=(protocol.CANDIDATE_RELATIVE,),
        )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "runtime_field",
    ["python", "pyarrow", "datasets"],
)
def test_runtime_mismatch_fails_before_source_access(
    runtime_field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).parents[1]
    expected = {
        "python": "3.12.3",
        "pyarrow": "25.0.0",
        "datasets": "5.0.0",
        "supported": True,
    }
    expected[runtime_field] = "mismatch"
    expected["supported"] = False
    monkeypatch.setattr(contract, "runtime_payload", lambda: expected)
    source_path = (root / protocol.SOURCE_ARTIFACT_RELATIVE).resolve()
    original_read_bytes = Path.read_bytes

    def reject_source_read(path: Path) -> bytes:
        if path.resolve() == source_path:
            raise AssertionError("runtime mismatch opened the authenticated source")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_source_read)
    output = tmp_path / "candidate.json"
    with pytest.raises(RuntimeError, match=r"Python 3\.12\.3"):
        materialize_candidate(root, output)
    assert not output.exists()
