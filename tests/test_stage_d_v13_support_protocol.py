from __future__ import annotations

import hashlib
import json
import os
from collections.abc import ItemsView, Iterable, Mapping
from pathlib import Path

import pytest

from redco.analysis import stage_d_dependency_stack as dependency_stack
from redco.analysis import stage_d_v13_support_contract as contract
from redco.analysis import stage_d_v13_support_protocol as protocol
from redco.analysis import stage_d_v13_support_publication as publication
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

ROOT = Path(__file__).parents[1]
REQUIRED_RUNTIME = {"python": "3.12.3", "pyarrow": "25.0.0", "datasets": "5.0.0"}
AUTHENTICATED_INPUTS = (
    contract.SELECTION_RECEIPT_RELATIVE,
    contract.SELECTION_MANIFEST_RELATIVE,
    contract.SELECTION_CLAIM_RELATIVE,
    contract.SELECTION_ORIGINAL_CLAIM_RELATIVE,
    *contract.UPSTREAM_EVIDENCE_SHA256,
    protocol.SOURCE_ARTIFACT_RELATIVE,
)


def _source_free_required_paths(root: Path) -> tuple[str, ...]:
    stack = dependency_stack.live_owner_patch_payload(root)
    return (
        contract.SELECTION_RECEIPT_RELATIVE,
        contract.SELECTION_MANIFEST_RELATIVE,
        contract.SELECTION_CLAIM_RELATIVE,
        *(
            relative
            for relative in contract.UPSTREAM_EVIDENCE_SHA256
            if relative not in contract.SOURCE_FREE_OPTIONAL_EVIDENCE
        ),
        contract.ACTION_CLOSURE_RELATIVE,
        contract.ACTION_CLOSURE_AUDIT_RELATIVE,
        contract.LAUNCH_AUTHORIZATION_RELATIVE,
        contract.SAMPLING_CONTRACT_SOURCE_RELATIVE,
        *(
            f"patches/{patch['name']}"
            for component in stack["components"]
            for patch in component["patches"]
        ),
    )


def _write_files(root: Path, files: Mapping[str, bytes]) -> None:
    for relative, raw in files.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)


def _read_files(root: Path, relatives: Iterable[str]) -> dict[str, bytes]:
    return {relative: (root / relative).read_bytes() for relative in relatives}


def _copy_files(source: Path, target: Path, relatives: Iterable[str]) -> None:
    _write_files(target, _read_files(source, relatives))


def _source_free_repository(parent: Path) -> tuple[Path, tuple[str, ...]]:
    repository = parent / "repository"
    required = _source_free_required_paths(ROOT)
    _copy_files(ROOT, repository, (*protocol.REVIEWED_PROTOCOL_ARTIFACT_SHA256, *required))
    return repository, required


def _file_snapshot(root: Path) -> dict[str, tuple[bytes, int, int]]:
    return {
        path.relative_to(root).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
            path.stat().st_ino,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def _assert_rebuild_rejected_without_writes(repository: Path) -> None:
    before = _file_snapshot(repository)
    with pytest.raises(ValueError):
        rebuild_protocol_artifacts_from_existing(repository)
    assert _file_snapshot(repository) == before


def _failing_replace(*failed_calls: int) -> object:
    call = 0
    real_replace = os.replace

    def replace(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        nonlocal call
        call += 1
        if call in failed_calls:
            raise OSError("injected replacement failure")
        real_replace(source, destination)

    return replace


def test_candidate_materializer_stops_at_authenticated_ordinal_180(tmp_path: Path) -> None:
    observer = CandidateReadInstrumentation()
    output = tmp_path / "candidate.json"
    payload = materialize_candidate(ROOT, output, instrumentation=observer)
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
    first = build_protocol_artifacts(ROOT, tmp_path / "first")
    second = build_protocol_artifacts(ROOT, tmp_path / "second")
    assert first == second
    published = tmp_path / "published"
    assert atomic_publish_set(
        published, first, manifest_path=protocol.PROTOCOL_AUDIT_RELATIVE
    )
    expected_hashes = {
        relative: hashlib.sha256(raw).hexdigest() for relative, raw in first.items()
    }
    assert check_protocol_artifacts(ROOT, published) == expected_hashes
    payload = json.loads(first[protocol.PROTOCOL_RELATIVE])
    support = payload["support_rule"]
    attempt = payload["attempt_policy"]
    authorization = payload["authorization"]
    assert (support["denominator"], support["required_joint_successes"]) == (64, 58)
    assert payload["scientific_protocol"]["arms"] == ["stock", "branch-global", "local"]
    assert authorization["launch_authorized"] is authorization["provider_calls_authorized"] is False
    assert (
        attempt["maximum_live_support_attempts_global"],
        attempt["outcome_bearing_cohorts"],
        attempt["second_outcome_bearing_attempt"],
    ) == (1, 1, "forbidden_unconditionally")
    assert payload["source"]["required_runtime"] == REQUIRED_RUNTIME
    preflight = payload["scientific_protocol"]["checkpoint_retention_preflight"]
    assert preflight["save_reload_reproduce_outputs"] is True
    assert payload["support_pass_transition"] == (
        "user_checkpoint_required_before_any_support_spend_or_science_transition"
    )
    assert authorization["readiness_blocker"] == "exploratory_science_not_user_accepted"


def test_candidate_composition_has_sixty_four_support_units_without_row_duplication(
    tmp_path: Path,
) -> None:
    artifacts = build_protocol_artifacts(ROOT, tmp_path)
    composition = json.loads(artifacts[protocol.COMPOSITION_RELATIVE])
    assert composition["support_cohort"] == {
        "required_papers": 64,
        "retained_support_rows": 63,
        "authenticated_replacement_rows": 1,
        "science_train_rows": 16,
        "science_eval_rows": 32,
    }
    assert composition["retained_base"]["byte_identity_preserved"] is True
    nonoverlap = composition["nonoverlap"]
    assert nonoverlap["example_id"] is True
    assert nonoverlap["historical_address_identities"] is True
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


def test_support_check_only_uses_independent_reviewed_bytes_and_rejects_coordinated_tamper(
    tmp_path: Path,
) -> None:
    """The read-only checker cannot be blessed by rewriting its audit hash."""

    output_root = tmp_path / "published"
    _copy_files(ROOT, output_root, protocol.REVIEWED_PROTOCOL_ARTIFACT_SHA256)
    before = _file_snapshot(output_root)
    assert protocol.check_protocol_artifacts(ROOT, output_root) == (
        protocol.REVIEWED_PROTOCOL_ARTIFACT_SHA256
    )
    assert _file_snapshot(output_root) == before

    reviewed = _read_files(output_root, protocol.REVIEWED_PROTOCOL_ARTIFACT_SHA256)
    changed = dict(reviewed)
    changed_candidate = json.loads(changed[protocol.CANDIDATE_RELATIVE])
    changed_candidate["unknown_null"] = None
    changed[protocol.CANDIDATE_RELATIVE] = protocol.canonical_json_bytes(changed_candidate)

    class ChangingArtifacts(dict[str, bytes]):
        def __init__(self) -> None:
            super().__init__(reviewed)
            self.item_reads = 0

        def items(self) -> ItemsView[str, bytes]:
            self.item_reads += 1
            return (reviewed if self.item_reads == 1 else changed).items()

    stateful = ChangingArtifacts()
    assert publication.authenticate_protocol_artifact_bytes(ROOT, stateful) == (
        protocol.REVIEWED_PROTOCOL_ARTIFACT_SHA256
    )
    assert stateful.item_reads == 1

    unexpected = output_root / "unexpected-dangling-link"
    unexpected.symlink_to(output_root / "absent-target")
    with pytest.raises(ValueError, match="unexpected file"):
        protocol.check_protocol_artifacts(ROOT, output_root)
    unexpected.unlink()
    assert _file_snapshot(output_root) == before
    output_alias = tmp_path / "published-alias"
    output_alias.symlink_to(output_root, target_is_directory=True)
    with pytest.raises(ValueError, match="output root must be a non-symlink directory"):
        protocol.check_protocol_artifacts(ROOT, output_alias)
    output_alias.unlink()
    assert _file_snapshot(output_root) == before
    candidate = output_root / protocol.CANDIDATE_RELATIVE
    candidate_payload = json.loads(candidate.read_bytes())
    candidate_payload["unknown_null"] = None
    candidate.write_bytes(protocol.canonical_json_bytes(candidate_payload))
    audit = output_root / protocol.PROTOCOL_AUDIT_RELATIVE
    audit_payload = json.loads(audit.read_bytes())
    audit_payload["candidate_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    audit.write_bytes(protocol.canonical_json_bytes(audit_payload))
    tampered = _file_snapshot(output_root)
    with pytest.raises(ValueError, match="reviewed byte set"):
        protocol.check_protocol_artifacts(ROOT, output_root)
    assert _file_snapshot(output_root) == tampered
    audit.unlink()
    missing = _file_snapshot(output_root)
    with pytest.raises(ValueError, match="published support artifact is missing"):
        protocol.check_protocol_artifacts(ROOT, output_root)
    assert _file_snapshot(output_root) == missing


def test_source_free_verifier_passes_in_minimal_git_free_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _ = _source_free_repository(tmp_path)
    output = tmp_path / "published"
    _copy_files(ROOT, output, protocol.REVIEWED_PROTOCOL_ARTIFACT_SHA256)
    absent = {".git", "external", contract.SOURCE_ARTIFACT_RELATIVE}
    absent.update(contract.SOURCE_FREE_OPTIONAL_EVIDENCE)
    assert all(not (repository / relative).exists() for relative in absent)
    repository_before = _file_snapshot(repository)
    output_before = _file_snapshot(output)
    real_read_bytes = Path.read_bytes
    repository_root = repository.resolve()
    output_root = output.resolve()

    def reject_forbidden_read(path: Path) -> bytes:
        resolved = path.resolve()
        if resolved.is_relative_to(output_root):
            return real_read_bytes(path)
        try:
            relative = resolved.relative_to(repository_root)
        except ValueError as error:
            raise AssertionError(f"source-free verifier opened {resolved}") from error
        if relative == Path(contract.SOURCE_ARTIFACT_RELATIVE) or (
            relative.parts and relative.parts[0] == "external"
        ):
            raise AssertionError(f"source-free verifier opened {relative.as_posix()}")
        return real_read_bytes(path)

    def reject_subprocess(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("source-free verifier invoked a dependency subprocess")

    monkeypatch.setattr(Path, "read_bytes", reject_forbidden_read)
    monkeypatch.setattr(dependency_stack.subprocess, "run", reject_subprocess)
    artifacts = _read_files(repository, protocol.REVIEWED_PROTOCOL_ARTIFACT_SHA256)
    reviewed = protocol.REVIEWED_PROTOCOL_ARTIFACT_SHA256
    assert publication.authenticate_protocol_artifact_bytes(repository, artifacts) == reviewed
    assert check_protocol_artifacts(repository, output) == reviewed
    rebuilt = rebuild_protocol_artifacts_from_existing(repository)
    assert rebuilt == artifacts
    assert rebuild_protocol_artifacts_from_existing(repository) == rebuilt
    assert _file_snapshot(repository) == repository_before
    assert _file_snapshot(output) == output_before


@pytest.mark.parametrize(
    "escaped_parent",
    ("reports", "src/redco/analysis", "patches"),
)  # type: ignore[untyped-decorator]
def test_source_free_verifier_rejects_parent_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    escaped_parent: str,
) -> None:
    repository, _ = _source_free_repository(tmp_path)
    artifacts = _read_files(repository, protocol.REVIEWED_PROTOCOL_ARTIFACT_SHA256)
    output = tmp_path / "published"
    _write_files(output, artifacts)
    source = repository / escaped_parent
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped = source.rename(outside / source.name)
    source.symlink_to(escaped, target_is_directory=True)
    repository_before = _file_snapshot(repository)
    output_before = _file_snapshot(output)
    real_read_bytes = Path.read_bytes
    escaped_root = escaped.resolve()

    def reject_escaped_read(path: Path) -> bytes:
        if path.resolve().is_relative_to(escaped_root):
            raise AssertionError(f"source-free verifier opened escaped path {path}")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_escaped_read)

    entrypoints = (
        lambda: publication.authenticate_protocol_artifact_bytes(repository, artifacts),
        lambda: check_protocol_artifacts(repository, output),
        lambda: rebuild_protocol_artifacts_from_existing(repository),
    )
    for entrypoint in entrypoints:
        with pytest.raises(ValueError, match="missing"):
            entrypoint()
        assert _file_snapshot(repository) == repository_before
        assert _file_snapshot(output) == output_before


def test_source_free_verifier_rejects_every_required_repository_binding(
    tmp_path: Path,
) -> None:
    repository, required = _source_free_repository(tmp_path)
    for relative in required:
        path = repository / relative
        raw = path.read_bytes()
        path.unlink()
        _assert_rebuild_rejected_without_writes(repository)
        path.write_bytes(raw + b" ")
        _assert_rebuild_rejected_without_writes(repository)
        path.write_bytes(raw)


def test_source_free_optional_evidence_is_absent_or_exact(
    tmp_path: Path,
) -> None:
    repository, _ = _source_free_repository(tmp_path)
    expected = rebuild_protocol_artifacts_from_existing(repository)
    original_claim = repository / contract.SELECTION_ORIGINAL_CLAIM_RELATIVE
    original_claim.parent.mkdir(parents=True, exist_ok=True)
    original_claim.write_bytes((repository / contract.SELECTION_CLAIM_RELATIVE).read_bytes())
    assert rebuild_protocol_artifacts_from_existing(repository) == expected
    original_claim.unlink()
    for relative in contract.SOURCE_FREE_OPTIONAL_EVIDENCE:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"wrong retained-local evidence")
        _assert_rebuild_rejected_without_writes(repository)
        path.unlink()
        path.mkdir()
        _assert_rebuild_rejected_without_writes(repository)
        path.rmdir()
        path.symlink_to(repository / contract.SELECTION_CLAIM_RELATIVE)
        _assert_rebuild_rejected_without_writes(repository)
        path.unlink()


def test_rebuild_and_check_share_semantic_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _ = _source_free_repository(tmp_path)
    artifacts = _read_files(repository, protocol.REVIEWED_PROTOCOL_ARTIFACT_SHA256)
    candidate = json.loads(artifacts[protocol.CANDIDATE_RELATIVE])
    candidate["authority"]["provider_calls_authorized"] = True
    artifacts[protocol.CANDIDATE_RELATIVE] = protocol.canonical_json_bytes(candidate)
    composition = json.loads(artifacts[protocol.COMPOSITION_RELATIVE])
    composition["candidate"]["sha256"] = hashlib.sha256(
        artifacts[protocol.CANDIDATE_RELATIVE]
    ).hexdigest()
    artifacts[protocol.COMPOSITION_RELATIVE] = protocol.canonical_json_bytes(composition)
    audit = json.loads(artifacts[protocol.PROTOCOL_AUDIT_RELATIVE])
    audit["candidate_sha256"] = composition["candidate"]["sha256"]
    audit["composition_sha256"] = protocol.sha256_json(composition)
    artifacts[protocol.PROTOCOL_AUDIT_RELATIVE] = protocol.canonical_json_bytes(audit)
    replacement_hashes = {
        relative: hashlib.sha256(raw).hexdigest() for relative, raw in artifacts.items()
    }
    monkeypatch.setattr(contract, "REVIEWED_PROTOCOL_ARTIFACT_SHA256", replacement_hashes)
    output = tmp_path / "published"
    _write_files(repository, artifacts)
    _write_files(output, artifacts)
    for verify in (
        lambda: publication.authenticate_protocol_artifact_bytes(repository, artifacts),
        lambda: check_protocol_artifacts(repository, output),
        lambda: rebuild_protocol_artifacts_from_existing(repository),
    ):
        with pytest.raises(ValueError, match="authenticated materialization"):
            verify()


def test_atomic_support_publication_check_only_is_read_only_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "publication"
    payloads = {
        "candidate.json": b'{"candidate":null}',
        "audit.json": b'{"status":"candidate-null"}',
    }
    _write_files(root, payloads)
    before = _file_snapshot(root)
    assert atomic_publish_set(root, payloads, manifest_path="audit.json", check_only=True)
    assert _file_snapshot(root) == before
    (root / "candidate.json").write_bytes(b'{"candidate":"tampered"}')
    tampered_before = (root / "candidate.json").read_bytes()
    with pytest.raises(ValueError, match="publication bytes differ"):
        atomic_publish_set(root, payloads, manifest_path="audit.json", check_only=True)
    assert (root / "candidate.json").read_bytes() == tampered_before
    (root / "audit.json").unlink()
    missing_before = _file_snapshot(root)
    with pytest.raises(ValueError, match="published output is missing"):
        atomic_publish_set(root, payloads, manifest_path="audit.json", check_only=True)
    assert _file_snapshot(root) == missing_before
    _write_files(root, payloads)
    monkeypatch.setattr(os, "replace", _failing_replace(2))
    with pytest.raises(OSError, match="injected replacement failure"):
        atomic_publish_set(root, payloads, manifest_path="audit.json")
    assert _read_files(root, payloads) == payloads
    fresh = tmp_path / "fresh-publication-root"
    assert atomic_publish_set(fresh, payloads, manifest_path="audit.json")
    assert _read_files(fresh, payloads) == payloads


def test_atomic_publication_restores_with_staged_replacements_when_rollback_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "rollback"
    payloads = {"a.json": b'{"a":1}', "manifest.json": b'{"manifest":1}'}
    _write_files(root, {relative: raw.replace(b"1", b"0") for relative, raw in payloads.items()})
    before = _read_files(root, payloads)
    monkeypatch.setattr(os, "replace", _failing_replace(2, 3))
    with pytest.raises(OSError, match="injected replacement failure"):
        atomic_publish_set(root, payloads, manifest_path="manifest.json")
    assert _read_files(root, payloads) == before


@pytest.mark.parametrize("relative", AUTHENTICATED_INPUTS)  # type: ignore[untyped-decorator]
def test_upstream_input_failure_precedes_candidate_output(
    relative: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_reader = contract.read_authenticated

    def reject_mutated_input(input_root: Path, input_relative: str, expected_sha256: str) -> bytes:
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
        protocol.materialize_candidate(ROOT, output)
    assert not output.exists()


@pytest.mark.parametrize("relative", AUTHENTICATED_INPUTS)  # type: ignore[untyped-decorator]
def test_support_outputs_reject_hard_link_to_every_authenticated_input(
    relative: str, tmp_path: Path
) -> None:
    from redco.analysis.stage_d_v13_draft_publication import validate_output_paths

    root = tmp_path / "repository"
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
    "runtime_field", tuple(REQUIRED_RUNTIME)
)
def test_runtime_mismatch_fails_before_source_access(
    runtime_field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {**REQUIRED_RUNTIME, "supported": True}
    expected[runtime_field] = "mismatch"
    expected["supported"] = False
    monkeypatch.setattr(contract, "runtime_payload", lambda: expected)
    source_path = (ROOT / protocol.SOURCE_ARTIFACT_RELATIVE).resolve()
    original_read_bytes = Path.read_bytes

    def reject_source_read(path: Path) -> bytes:
        if path.resolve() == source_path:
            raise AssertionError("runtime mismatch opened the authenticated source")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_source_read)
    output = tmp_path / "candidate.json"
    with pytest.raises(RuntimeError, match=r"Python 3\.12\.3"):
        materialize_candidate(ROOT, output)
    assert not output.exists()
