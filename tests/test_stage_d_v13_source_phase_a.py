"""CPU-only tests for the authenticated, cutoff-walled v13 Phase A source path."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import types
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, cast

import pytest

from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes, sha256_json
from redco.analysis.stage_d_v13_source_phase_a import (
    PHASE_A_CUTOFF,
    PHASE_A_OUTPUTS,
    SOURCE_ARTIFACT_RELATIVE,
    SOURCE_BYTES,
    SOURCE_SHA256,
    _authenticated_behavior_bindings,
    authenticate_source_artifact,
    authenticated_historical_inputs,
    build_forbidden_witness,
    build_phase_a_result,
    collision_disposition,
    iter_cutoff_rows,
    phase_a_immutable_paths,
    phase_a_payloads,
    reconstruct_retired_units,
    source_row_sha256,
    validate_forbidden_witness,
    write_phase_a_outputs,
)
from redco.analysis.stage_d_v13_source_phase_a_decoder import (
    PHASE_B_BINDING_RELATIVE,
    DecoderInstrumentation,
    PhaseAWallError,
    PhaseBResumeAuthorizationError,
    PhaseBResumeUnavailable,
    bounded_source_rows,
    legacy_datasets_decoder_probe,
    resume_decoder_invocation_count,
    resume_source_rows,
    source_schema_sha256,
)
from redco.analysis.stage_d_v13_source_phase_a_selector import (
    CollisionClassification,
    TerminalIdentityCollision,
    classify_candidate_collisions,
    normalize_digest_values,
    render_paper,
    select_first_eligible,
)
from redco.analysis.stage_d_v13_source_phase_a_trust import (
    APPROVAL_ANCHOR_RELATIVE,
    BINDINGS_RELATIVE,
    authenticate_external_anchor,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_NODE_IDS = (
    "test_pinned_source_artifact_metadata",
    "test_historical_receipts_174_through_179",
    "test_six_retired_rows_and_observed_unit",
    "test_forbidden_witness_is_complete_and_hashed",
    "test_phase_a_wall_stops_before_ordinal_180",
    "test_phase_a_candidate_and_launch_flags_remain_unresolved",
    "test_collision_disposition_is_predeclared",
    "test_scientific_binding_reuses_v1_law",
    "test_phase_a_artifacts_are_canonical_and_unfrozen",
    "test_phase_a_cpu_manifest_matches_collection",
    "test_two_fresh_output_roots_are_byte_identical",
    "test_phase_a_check_only_rejects_tampering",
    "test_legacy_datasets_decoder_batch_policy_is_rejected",
    "test_real_pinned_decoder_emits_one_bounded_batch",
    "test_decoder_instrumentation_rejects_oversized_object",
    "test_terminal_selector_collisions_fail_closed_without_later_candidate[example]",
    "test_terminal_selector_collisions_fail_closed_without_later_candidate[rendered]",
    "test_terminal_selector_collisions_fail_closed_without_later_candidate[row]",
    "test_terminal_selector_collisions_fail_closed_without_later_candidate[address]",
    "test_terminal_collision_dominates_continuable_collision[example-paper]",
    "test_terminal_collision_dominates_continuable_collision[rendered-paper]",
    "test_terminal_collision_dominates_continuable_collision[row-paper]",
    "test_terminal_collision_dominates_continuable_collision[address-paper]",
    "test_terminal_collision_dominates_continuable_collision[example-reference]",
    "test_terminal_collision_dominates_continuable_collision[rendered-reference]",
    "test_terminal_collision_dominates_continuable_collision[row-reference]",
    "test_terminal_collision_dominates_continuable_collision[address-reference]",
    "test_multiple_terminal_collisions_have_complete_set_and_primary_reason",
    "test_paper_and_reference_collisions_are_the_only_continuable_set",
    "test_paper_collision_continues_to_next_candidate",
    "test_selector_continues_after_raw_reference_collision",
    "test_selector_universe_contains_all_authenticated_classes",
    "test_forbidden_witness_rebuild_rejects_each_mutation[retired_units]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[retired_papers]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[examples]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[rows]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[rendered]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[references]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[source_addresses]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[historical_addresses]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[old_snapshot_papers]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[predecessor_examples]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[exclusion_hash]",
    "test_forbidden_witness_rejects_recomputed_self_hash",
    "test_phase_a_publication_rejects_authenticated_input_hardlink",
    "test_phase_a_publication_rejects_cross_output_hardlink",
    "test_phase_a_publication_rejects_symlink_parent_before_write",
    "test_phase_a_status_capture_is_independent_and_exact",
    "test_phase_a_missing_source_fails_closed",
    "test_immutable_v1_audit_rejects_repaired_tree",
    "test_phase_a_approval_anchor_authenticates_registry_policy",
    "test_phase_a_approval_anchor_mutations_fail_before_publication[selector]",
    "test_phase_a_approval_anchor_mutations_fail_before_publication[selector_and_registry]",
    "test_phase_a_approval_anchor_mutations_fail_before_publication[registry]",
    "test_phase_a_approval_anchor_mutations_fail_before_publication[derivation]",
    "test_phase_a_approval_anchor_mutations_fail_before_publication[status]",
    "test_phase_a_approval_anchor_mutations_fail_before_publication[resume]",
    "test_phase_a_approval_anchor_mutations_fail_before_publication[anchor]",
    "test_phase_a_resume_invocation_count_is_zero",
    "test_dormant_resume_decoder_starts_at_ordinal_180",
    "test_dormant_resume_decoder_rejects_wrong_binding[source]",
    "test_dormant_resume_decoder_rejects_wrong_binding[schema]",
    "test_dormant_resume_decoder_rejects_wrong_binding[version]",
    "test_dormant_resume_decoder_rejects_wrong_binding[config]",
    "test_dormant_resume_decoder_requires_reviewed_checkpoint",
    "test_foundation_resume_entrypoint_rejects_all_caller_authority",
    "test_future_phase_b_authorization_is_unusable_without_committed_c[missing]",
    "test_future_phase_b_authorization_is_unusable_without_committed_c[wrong_ancestry]",
    "test_future_phase_b_authorization_is_unusable_without_committed_c[working_tree]",
    "test_future_phase_b_authorization_is_unusable_without_committed_c[wrong_source]",
)


_PHASE_A_RESULT: dict[str, Any] | None = None


def phase_a_result() -> dict[str, Any]:
    global _PHASE_A_RESULT
    if _PHASE_A_RESULT is None:
        _PHASE_A_RESULT = build_phase_a_result(ROOT)
    return _PHASE_A_RESULT


def test_pinned_source_artifact_metadata() -> None:
    source = phase_a_result()["source"]
    assert source["local_artifact"] == SOURCE_ARTIFACT_RELATIVE
    assert source["bytes"] == SOURCE_BYTES
    assert source["sha256"] == SOURCE_SHA256
    assert source["row_count"] == 888
    assert source["row_groups"] == 1
    assert source["decoder"] == {
        "python": "3.12.3",
        "datasets": "5.0.0",
        "pyarrow": "25.0.0",
        "loader": (
            "pyarrow.parquet.ParquetFile.iter_batches(batch_size=180, "
            "row_groups=[0], use_threads=False)"
        ),
        "batch_size": 180,
        "use_threads": False,
        "logical_readahead": False,
        "metadata_only_for_authentication": True,
    }


def test_historical_receipts_174_through_179() -> None:
    receipts = phase_a_result()["historical_receipts"]
    assert len(receipts) == 6
    assert [receipt["source_ordinal"] for receipt in receipts] == list(range(174, 180))
    assert all(receipt["status"] == "authenticated" for receipt in receipts)
    assert all(receipt["address_audit_path"].endswith(".json") for receipt in receipts)


def test_six_retired_rows_and_observed_unit() -> None:
    units = phase_a_result()["retired_units"]
    assert len(units) == 7
    assert {unit["paper_id"] for unit in units} >= {
        "1911.03894",
        "2001.09899",
        "1710.01492",
        "1912.01673",
        "1909.12231",
        "1706.08032",
        "1811.01399",
    }
    assert all(unit["status"].startswith("authenticated") for unit in units)
    assert next(unit for unit in units if unit["paper_id"] == "1706.08032")["source_ordinal"] == 94
    assert (
        next(unit for unit in units if unit["paper_id"] == "1811.01399")["source_ordinal"] is None
    )


def test_forbidden_witness_is_complete_and_hashed() -> None:
    witness = phase_a_result()["forbidden_witness"]
    validate_forbidden_witness(
        ROOT,
        witness,
        expected_units=reconstruct_retired_units(ROOT),
        address_inputs=authenticated_historical_inputs(ROOT),
    )
    assert witness["old_deterministic_snapshot"]["rows"] == 120
    assert witness["complete_v6_predecessor"]["rows"] == 112
    assert witness["forbidden_sets"]["reference_span_sha256"]
    assert len(witness["forbidden_sets"]["historical_address_sha256"]) == 74
    assert witness["authenticated_historical_identity_witness"]["address_count"] == 74
    assert witness["exclusion_hashes"]

    missing_retired = copy.deepcopy(witness)
    missing_retired["retired_units"] = [
        unit for unit in missing_retired["retired_units"] if unit["paper_id"] != "1706.08032"
    ]
    with pytest.raises(ValueError, match="forbidden witness differs"):
        validate_forbidden_witness(
            ROOT,
            missing_retired,
            expected_units=reconstruct_retired_units(ROOT),
            address_inputs=authenticated_historical_inputs(ROOT),
        )

    missing_reference = copy.deepcopy(witness)
    retired = next(
        unit for unit in missing_reference["retired_units"] if unit["paper_id"] == "1706.08032"
    )
    retired["reference_span_sha256"] = []
    with pytest.raises(ValueError, match="forbidden witness differs"):
        validate_forbidden_witness(
            ROOT,
            missing_reference,
            expected_units=reconstruct_retired_units(ROOT),
            address_inputs=authenticated_historical_inputs(ROOT),
        )


def test_phase_a_wall_stops_before_ordinal_180() -> None:
    yielded: list[int] = []
    canonicalized: list[int] = []

    def rows() -> Any:
        for ordinal in range(PHASE_A_CUTOFF + 2):
            yielded.append(ordinal)
            yield {"id": f"paper-{ordinal}"}

    for ordinal, row in iter_cutoff_rows(rows()):
        canonicalized.append(ordinal)
        assert row["id"] == f"paper-{ordinal}"
    assert yielded == list(range(PHASE_A_CUTOFF + 2))
    assert canonicalized == list(range(PHASE_A_CUTOFF + 1))
    assert 180 not in canonicalized
    with pytest.raises(ValueError, match="non-negative"):
        list(iter_cutoff_rows(({"id": "x"} for _ in range(181)), cutoff=-1))


def test_phase_a_candidate_and_launch_flags_remain_unresolved() -> None:
    phase_a = phase_a_result()
    assert all(value is None for value in phase_a["candidate"].values())
    assert phase_a["phase_a_wall"]["phase_b_started"] is False
    assert phase_a["authorization_provenance"]["freeze"] is False
    assert phase_a["authorization_provenance"]["launch_authorized"] is False
    assert phase_a["authorization_provenance"]["provider_calls"] is False
    assert phase_a["phase_a_state_machine"]["phase_b_transition_authorized"] is False


def test_collision_disposition_is_predeclared() -> None:
    assert collision_disposition("paper_or_reference", True) == "continue_source_order_scan"
    assert collision_disposition("paper_or_reference", False) == "accept"
    assert collision_disposition("address_or_identity", True) == "terminal_fail_closed"
    assert collision_disposition("address_or_identity", False) == "accept"


def test_scientific_binding_reuses_v1_law() -> None:
    binding = phase_a_result()["scientific_binding"]
    assert binding["scientific_group_namespace"] == "redco-stage-d1-support-v1"
    assert binding["master_seed"] == "redco-stage-d1-support-v1-20260802-78b65e4cc16ac31f"
    assert binding["administrative_seed_used"] is False
    assert binding["v13_administrative_identity_used"] is False
    assert binding["candidate_seed"] is None
    assert binding["candidate_address"] is None


def test_phase_a_artifacts_are_canonical_and_unfrozen() -> None:
    payloads = phase_a_payloads(ROOT, test_node_ids=TEST_NODE_IDS)
    assert set(payloads) == set(PHASE_A_OUTPUTS)
    for relative, payload in payloads.items():
        parsed = json.loads(payload)
        assert parsed["draft_unfrozen"] is True
        assert parsed["launch_authorized"] is False
        assert payload == canonical_json_bytes(parsed)
        assert not payload.endswith(b"\n"), relative


def test_phase_a_cpu_manifest_matches_collection() -> None:
    from scripts.build_stage_d_v13_source_phase_a import PHASE_A_TEST_NODE_IDS

    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_stage_d_v13_source_phase_a.py",
        "--collect-only",
        "-q",
        "-vv",
        "--capture=no",
    ]
    environment = os.environ.copy()
    pythonpath = [str(ROOT), str(ROOT / "src")]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    collected = re.findall(r"<Function (test_[^>]+)>", result.stdout)
    assert tuple(collected) == PHASE_A_TEST_NODE_IDS
    manifest = json.loads(
        (ROOT / "reports/stage-d1-support-v13-source-phase-a-cpu-manifest-v1.json").read_bytes()
    )
    suite = manifest["suite"]
    assert suite["interpreter_binding"] == "sys.executable"
    assert suite["runtime"] == {
        "python": ".".join(map(str, sys.version_info[:3])),
        "datasets": importlib.metadata.version("datasets"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "pytest": importlib.metadata.version("pytest"),
    }
    assert suite["node_ids"] == list(PHASE_A_TEST_NODE_IDS)
    assert suite["node_count"] == len(collected)
    assert suite["verification"]["collection_reproduced"] is True
    assert suite["verification"]["status_signature"]
    assert suite["verification"]["status_capture_path"] == PHASE_A_OUTPUTS[4]
    assert suite["verification"]["status_capture_sha256"] == sha256_bytes(
        (ROOT / PHASE_A_OUTPUTS[4]).read_bytes()
    )
    assert suite["verification"]["independent_status_capture"] is True
    assert suite["expected"] == {
        "passed": len(collected),
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
    }


def test_two_fresh_output_roots_are_byte_identical(tmp_path: Path) -> None:
    payloads = phase_a_payloads(ROOT, test_node_ids=TEST_NODE_IDS)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    immutable_paths = phase_a_immutable_paths(ROOT)
    write_phase_a_outputs(first, payloads, immutable_paths=immutable_paths)
    write_phase_a_outputs(second, payloads, immutable_paths=immutable_paths)
    for relative in PHASE_A_OUTPUTS:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_phase_a_check_only_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = phase_a_payloads(ROOT, test_node_ids=TEST_NODE_IDS)
    tmp_path.mkdir(exist_ok=True)
    write_phase_a_outputs(tmp_path, payloads, immutable_paths=phase_a_immutable_paths(ROOT))
    audit_path = tmp_path / PHASE_A_OUTPUTS[1]
    audit = json.loads(audit_path.read_bytes())
    audit["status"] = "tampered"
    audit_path.write_bytes(canonical_json_bytes(audit))
    from scripts.build_stage_d_v13_source_phase_a import check_only

    with pytest.raises(ValueError, match="existing bytes differ"):
        check_only(tmp_path)

    # The Foundation F check-only boundary is separately read-only.  Exercise
    # the real root once, then use a tiny disposable stand-in to cover missing
    # and tampered generated inputs without touching authenticated files.
    foundation_script = importlib.import_module(
        "scripts.build_stage_d_v13_foundation_manifest"
    )
    foundation = importlib.import_module("redco.analysis.stage_d_v13_foundation")
    root_paths = [
        ROOT / foundation.SOURCE_PROVENANCE_RELATIVE,
        ROOT / foundation.APPROVAL_ANCHOR_RELATIVE,
        ROOT / foundation.FOUNDATION_MANIFEST_RELATIVE,
    ]

    def snapshot(paths: list[Path]) -> dict[str, tuple[bool, bytes | None, int | None, int | None]]:
        result: dict[str, tuple[bool, bytes | None, int | None, int | None]] = {}
        for path in paths:
            if not path.exists():
                result[str(path)] = (False, None, None, None)
                continue
            stat = path.stat()
            result[str(path)] = (
                True,
                path.read_bytes(),
                stat.st_mtime_ns,
                getattr(stat, "st_ino", None),
            )
        return result

    before = snapshot(root_paths)
    # The shared checkout deliberately contains unrelated historical
    # untracked artifacts.  Exercise the full manifest/check-only path under
    # an exact-tree staging view, then test the real status gate below.
    monkeypatch.setattr(
        foundation,
        "_git_head",
        lambda _root: "c41fd18446cecf1c7c98e5aa3a962d1568072c1b",
    )
    monkeypatch.setattr(foundation, "_status_paths", lambda _root: set())
    foundation_script.check_only(root=ROOT)
    assert snapshot(root_paths) == before
    monkeypatch.undo()

    standin = tmp_path / "foundation-standin"
    standin.mkdir()
    expected_provenance = {"kind": "provenance"}
    expected_anchor = {"kind": "anchor"}
    expected_manifest = {"kind": "manifest"}
    monkeypatch.setattr(
        foundation_script, "build_source_provenance", lambda: expected_provenance
    )
    monkeypatch.setattr(
        foundation_script, "build_integrity_anchor", lambda _root: expected_anchor
    )
    monkeypatch.setattr(
        foundation_script,
        "validate_foundation_manifest",
        lambda _root, _value: None,
    )
    standin_paths = [
        standin / foundation.SOURCE_PROVENANCE_RELATIVE,
        standin / foundation.APPROVAL_ANCHOR_RELATIVE,
        standin / foundation.FOUNDATION_MANIFEST_RELATIVE,
    ]
    for path, value in zip(
        standin_paths,
        (expected_provenance, expected_anchor, expected_manifest),
        strict=True,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(value))
    standin_before = snapshot(standin_paths)
    foundation_script.check_only(root=standin)
    assert snapshot(standin_paths) == standin_before

    missing = tmp_path / "missing-foundation"
    missing_before = snapshot(
        [
            missing / foundation.SOURCE_PROVENANCE_RELATIVE,
            missing / foundation.APPROVAL_ANCHOR_RELATIVE,
        ]
    )
    with pytest.raises(FileNotFoundError):
        foundation_script.check_only(root=missing)
    assert snapshot(
        [
            missing / foundation.SOURCE_PROVENANCE_RELATIVE,
            missing / foundation.APPROVAL_ANCHOR_RELATIVE,
        ]
    ) == missing_before

    tampered = standin / foundation.APPROVAL_ANCHOR_RELATIVE
    tampered.write_bytes(b"tampered")
    tampered_before = snapshot([tampered])
    with pytest.raises(ValueError, match="approval anchor bytes differ"):
        foundation_script.check_only(root=standin)
    assert snapshot([tampered]) == tampered_before

    # Exact-tree authentication rejects a same-prefix/near-miss parent and an
    # unrelated dirty path; the pre-existing external submodule remains the
    # sole explicitly excluded boundary.
    for near_miss in ("c41fd18", "c41fd18446cecf1c7c98e5aa3a962d1568072c1c"):
        monkeypatch.setattr(foundation, "_git_head", lambda _root, value=near_miss: value)
        with pytest.raises(ValueError, match="parent must be"):
            foundation.build_foundation_manifest(ROOT)
    monkeypatch.undo()
    real_before = snapshot(root_paths)
    with pytest.raises(ValueError, match="outside the exact allowlist"):
        foundation_script.check_only(root=ROOT)
    assert snapshot(root_paths) == real_before
    status_result = type(
        "StatusResult",
        (),
        {"returncode": 0, "stdout": b"?? unallowlisted-foundation-test.txt\0"},
    )()
    monkeypatch.setattr(foundation, "hardened_git", lambda *_args, **_kwargs: status_result)
    with pytest.raises(ValueError, match="outside the exact allowlist"):
        foundation._status_paths(ROOT)

    # The only accepted submodule record is the exact pre-existing dirty
    # worktree witness, and its index entry must remain the fixed gitlink.
    def fake_git(*args: Any, **kwargs: Any) -> Any:
        if "ls-files" in args:
            return type(
                "Result",
                (),
                {
                    "returncode": 0,
                    "stdout": (
                        "160000 3b22dd951cad1036d1fe8dd0a0bfc40807a9b360 0 "
                        "external/prime-rl\n"
                    ),
                },
            )()
        return type("Result", (), {"returncode": 0, "stdout": b" M external/prime-rl\0"})()

    monkeypatch.setattr(foundation, "hardened_git", fake_git)
    assert foundation._status_paths(ROOT) == set()
    monkeypatch.setattr(
        foundation,
        "hardened_git",
        lambda *args, **kwargs: type(
            "Result",
            (),
            {"returncode": 0, "stdout": b" M scripts/build_stage_d_v13_source_phase_a.py\0"},
        )(),
    )
    with pytest.raises(ValueError, match="exactly one"):
        foundation._status_paths(ROOT)
    def duplicate_fake_git(*args: Any, **kwargs: Any) -> Any:
        if "ls-files" in args:
            return type(
                "Result",
                (),
                {
                    "returncode": 0,
                    "stdout": (
                        "160000 3b22dd951cad1036d1fe8dd0a0bfc40807a9b360 0 "
                        "external/prime-rl\n"
                    ),
                },
            )()
        return type(
            "Result",
            (),
            {"returncode": 0, "stdout": b" M external/prime-rl\0 M external/prime-rl\0"},
        )()

    monkeypatch.setattr(foundation, "hardened_git", duplicate_fake_git)
    with pytest.raises(ValueError, match="duplicated"):
        foundation._status_paths(ROOT)
    for status_bytes in (
        b" m external/prime-rl\0",
        b"M  external/prime-rl\0",
        b"D  external/prime-rl\0",
        b"R  external/prime-rl\0external/prime-rl-old\0",
        b" m external/prime-rl/child with spaces\0",
        b"?? external/prime-rl/child\0",
    ):
        monkeypatch.setattr(
            foundation,
            "hardened_git",
            lambda *args, status_bytes=status_bytes, **kwargs: type(
                "Result", (), {"returncode": 0, "stdout": status_bytes}
            )(),
        )
        with pytest.raises(ValueError):
            foundation._status_paths(ROOT)
    monkeypatch.setattr(
        foundation,
        "hardened_git",
        lambda *args, **kwargs: type(
            "Result", (), {"returncode": 0, "stdout": b" M external/prime-rl\0"}
        )(),
    )
    # A clean/staged index entry cannot bless the dirty witness.
    monkeypatch.setattr(
        foundation,
        "_validate_prime_rl_gitlink",
        lambda _root: (_ for _ in ()).throw(ValueError("wrong gitlink")),
    )
    with pytest.raises(ValueError, match="wrong gitlink"):
        foundation._status_paths(ROOT)
    for index_bytes in (
        b"",
        b"160000 000000000000000000000000000000000000 0 external/prime-rl\n",
        b"100644 3b22dd951cad1036d1fe8dd0a0bfc40807a9b360 0 external/prime-rl\n",
        (
            b"160000 3b22dd951cad1036d1fe8dd0a0bfc40807a9b360 0 "
            b"external/prime-rl\nexternal/prime-rl\n"
        ),
    ):
        def wrong_index_git(*args: Any, index_bytes: bytes = index_bytes, **kwargs: Any) -> Any:
            if "ls-files" in args:
                return type(
                    "Result", (), {"returncode": 0, "stdout": index_bytes.decode()}
                )()
            return type("Result", (), {"returncode": 0, "stdout": b" M external/prime-rl\0"})()

        monkeypatch.setattr(foundation, "hardened_git", wrong_index_git)
        with pytest.raises(ValueError, match="gitlink"):
            foundation._status_paths(ROOT)


def test_legacy_datasets_decoder_batch_policy_is_rejected() -> None:
    probe = legacy_datasets_decoder_probe(ROOT / SOURCE_ARTIFACT_RELATIVE)
    assert probe["configured_batch_size"] is None
    assert probe["effective_first_table_batch_size"] == 888
    assert probe["would_cross_phase_a_wall"] is True
    assert probe["rows_iterated"] == 0
    assert probe["rows_deserialized"] == 0


def test_real_pinned_decoder_emits_one_bounded_batch() -> None:
    instrumentation = DecoderInstrumentation()
    rows = list(
        bounded_source_rows(
            ROOT / SOURCE_ARTIFACT_RELATIVE,
            instrumentation=instrumentation,
        )
    )
    assert len(rows) == 180
    assert rows[0][0] == 0 and rows[-1][0] == PHASE_A_CUTOFF
    assert instrumentation.decoded_objects == [
        {
            "kind": "pyarrow_record_batch",
            "rows": 180,
            "start_ordinal": 0,
            "end_ordinal": 179,
        }
    ]
    assert instrumentation.canonicalized_ordinals == list(range(180))
    payload = instrumentation.to_payload()
    assert payload["maximum_decoded_cardinality"] == 180
    assert payload["maximum_decoded_ordinal"] == PHASE_A_CUTOFF
    assert payload["all_decoded_objects_within_cutoff"] is True
    assert payload["post_cutoff_logical_row_materialized"] is False
    assert payload["physical_io_may_span_row_group"] is True


def test_decoder_instrumentation_rejects_oversized_object() -> None:
    with pytest.raises(PhaseAWallError, match="beyond ordinal 179"):
        DecoderInstrumentation().record_batch(start=0, rows=888)


def _single_question_candidate() -> tuple[dict[str, Any], dict[str, Any]]:
    row: dict[str, Any] | None = None
    for ordinal, candidate in bounded_source_rows(ROOT / SOURCE_ARTIFACT_RELATIVE):
        if ordinal == 179:
            row = copy.deepcopy(candidate)
            break
    assert row is not None
    selected_result = select_first_eligible(row)
    assert selected_result is not None
    _selected, question_index = selected_result
    qas = row["qas"]
    for field in ("question", "answers", "question_id"):
        qas[field] = [qas[field][question_index]]
    selected_result = select_first_eligible(row)
    assert selected_result is not None
    return row, selected_result[0]


def _two_question_candidate() -> dict[str, Any]:
    first_span = "first evidence span is at least twenty chars"
    return {
        "id": "terminal-paper",
        "title": "Terminal paper",
        "abstract": first_span,
        "full_text": {"section_name": [], "paragraphs": []},
        "qas": {
            "question": ["first", "unreachable"],
            "question_id": ["one", "two"],
            "answers": [
                {"answer": [{"unanswerable": False, "evidence": [first_span]}]},
                {"answer": [{"unanswerable": True}]},
            ],
        },
    }


def _later_candidate() -> dict[str, Any]:
    span = "later evidence span is at least twenty chars"
    return {
        "id": "later-paper",
        "title": "Later paper",
        "abstract": span,
        "full_text": {"section_name": [], "paragraphs": []},
        "qas": {
            "question": ["later"],
            "question_id": ["later"],
            "answers": [
                {"answer": [{"unanswerable": False, "evidence": [span]}]},
            ],
        },
    }


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("collision_class", "kwargs_factory"),
    (
        ("example_id_collision", lambda row: {"forbidden_example_ids": {"qasper-one"}}),
        (
            "rendered_paper_collision",
            lambda row: {
                "forbidden_rendered_paper_sha256": {
                    sha256_bytes(render_paper(row).encode("utf-8"))
                }
            },
        ),
        ("source_row_collision", lambda row: {"forbidden_row_sha256": {source_row_sha256(row)}}),
        (
            "source_address_collision",
            lambda row: {
                "forbidden_address_sha256": {"a" * 64},
                "candidate_address_sha256": "a" * 64,
            },
        ),
    ),
    ids=("example", "rendered", "row", "address"),
)
def test_terminal_selector_collisions_fail_closed_without_later_candidate(
    collision_class: str,
    kwargs_factory: Any,
) -> None:
    first = _two_question_candidate()
    later = _later_candidate()
    evaluated: list[str] = []
    with pytest.raises(TerminalIdentityCollision) as caught:
        for candidate in (first, later):
            evaluated.append(str(candidate["id"]))
            selected = select_first_eligible(candidate, **kwargs_factory(first))
            if selected is not None:
                break
    assert caught.value.collision_class == collision_class
    assert str(caught.value) == f"terminal identity collision: {collision_class}"
    assert evaluated == ["terminal-paper"]


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("terminal_class", "continuable_class"),
    (
        ("example_id_collision", "paper_id_collision"),
        ("rendered_paper_collision", "paper_id_collision"),
        ("source_row_collision", "paper_id_collision"),
        ("source_address_collision", "paper_id_collision"),
        ("example_id_collision", "reference_span_collision"),
        ("rendered_paper_collision", "reference_span_collision"),
        ("source_row_collision", "reference_span_collision"),
        ("source_address_collision", "reference_span_collision"),
    ),
    ids=(
        "example-paper",
        "rendered-paper",
        "row-paper",
        "address-paper",
        "example-reference",
        "rendered-reference",
        "row-reference",
        "address-reference",
    ),
)
def test_terminal_collision_dominates_continuable_collision(
    terminal_class: str, continuable_class: str
) -> None:
    first = _two_question_candidate()
    later = _later_candidate()
    first_span = first["qas"]["answers"][0]["answer"][0]["evidence"][0]
    kwargs: dict[str, Any] = {
        "forbidden_paper_ids": {"terminal-paper"}
        if continuable_class == "paper_id_collision"
        else set(),
        "forbidden_reference_spans": {first_span}
        if continuable_class == "reference_span_collision"
        else set(),
    }
    if terminal_class == "example_id_collision":
        kwargs["forbidden_example_ids"] = {"qasper-one"}
    elif terminal_class == "rendered_paper_collision":
        kwargs["forbidden_rendered_paper_sha256"] = {
            sha256_bytes(render_paper(first).encode("utf-8"))
        }
    elif terminal_class == "source_row_collision":
        kwargs["forbidden_row_sha256"] = {source_row_sha256(first)}
    else:
        kwargs["forbidden_address_sha256"] = {"a" * 64}
        kwargs["candidate_address_sha256"] = "a" * 64
    evaluated: list[str] = []
    with pytest.raises(TerminalIdentityCollision) as caught:
        for candidate in (first, later):
            evaluated.append(str(candidate["id"]))
            if select_first_eligible(candidate, **kwargs) is not None:
                break
    assert caught.value.collision_class == terminal_class
    assert terminal_class in caught.value.collision_set
    assert continuable_class in caught.value.collision_set
    assert evaluated == ["terminal-paper"]


def test_multiple_terminal_collisions_have_complete_set_and_primary_reason() -> None:
    row = _two_question_candidate()
    classification = classify_candidate_collisions(
        row=row,
        example_id="qasper-one",
        paper=render_paper(row),
        evidence=(row["qas"]["answers"][0]["answer"][0]["evidence"][0],),
        forbidden_paper_ids={"terminal-paper"},
        forbidden_example_ids={"qasper-one"},
        forbidden_rendered_paper_sha256={sha256_bytes(render_paper(row).encode("utf-8"))},
        forbidden_reference_spans={
            row["qas"]["answers"][0]["answer"][0]["evidence"][0]
        },
        forbidden_row_sha256={source_row_sha256(row)},
        candidate_address_sha256="a" * 64,
        forbidden_address_sha256={"a" * 64},
    )
    assert classification == CollisionClassification(
        collision_set=(
            "paper_id_collision",
            "reference_span_collision",
            "example_id_collision",
            "rendered_paper_collision",
            "source_row_collision",
            "source_address_collision",
        ),
        primary_terminal="example_id_collision",
    )
    with pytest.raises(TerminalIdentityCollision) as caught:
        select_first_eligible(
            row,
            forbidden_paper_ids={"terminal-paper"},
            forbidden_example_ids={"qasper-one"},
            forbidden_rendered_paper_sha256={sha256_bytes(render_paper(row).encode("utf-8"))},
            forbidden_reference_spans={
                row["qas"]["answers"][0]["answer"][0]["evidence"][0]
            },
            forbidden_row_sha256={source_row_sha256(row)},
            candidate_address_sha256="a" * 64,
            forbidden_address_sha256={"a" * 64},
        )
    assert caught.value.collision_set == classification.collision_set


def test_paper_and_reference_collisions_are_the_only_continuable_set() -> None:
    row = _two_question_candidate()
    first_span = row["qas"]["answers"][0]["answer"][0]["evidence"][0]
    classification = classify_candidate_collisions(
        row=row,
        example_id="qasper-one",
        paper=render_paper(row),
        evidence=(first_span,),
        forbidden_paper_ids={"terminal-paper"},
        forbidden_example_ids=set(),
        forbidden_rendered_paper_sha256=set(),
        forbidden_reference_spans={first_span},
        forbidden_row_sha256=set(),
        candidate_address_sha256=None,
        forbidden_address_sha256=set(),
    )
    assert classification.continuable is True
    assert classification.primary_terminal is None
    assert classification.collision_set == (
        "paper_id_collision",
        "reference_span_collision",
    )
    assert select_first_eligible(row, forbidden_paper_ids={"terminal-paper"},
                                 forbidden_reference_spans={first_span}) is None


def test_paper_collision_continues_to_next_candidate() -> None:
    first = _two_question_candidate()
    for field in ("question", "question_id", "answers"):
        first["qas"][field] = first["qas"][field][:1]
    later = _later_candidate()
    assert select_first_eligible(first, forbidden_paper_ids={"terminal-paper"}) is None
    selected = select_first_eligible(later, forbidden_paper_ids={"terminal-paper"})
    assert selected is not None
    assert selected[0]["paper_id"] == "later-paper"


def test_selector_continues_after_raw_reference_collision() -> None:
    first_span = "first evidence span is at least twenty chars"
    second_span = "second evidence span is also twenty chars"
    row = {
        "id": "fresh-paper",
        "title": "Fresh paper",
        "abstract": f"{first_span} {second_span}",
        "full_text": {"section_name": [], "paragraphs": []},
        "qas": {
            "question": ["first", "second"],
            "question_id": ["one", "two"],
            "answers": [
                {"answer": [{"unanswerable": False, "evidence": [first_span]}]},
                {"answer": [{"unanswerable": False, "evidence": [second_span]}]},
            ],
        },
    }
    selected = select_first_eligible(
        row,
        forbidden_reference_spans={first_span},
    )
    assert selected is not None
    assert selected[1] == 1
    assert selected[0]["reference_evidence"] == [second_span]
    assert (
        select_first_eligible(
            row,
            forbidden_reference_spans={sha256_bytes(first_span.encode("utf-8"))},
        )
        is not None
    )
    with pytest.raises(ValueError, match="collection"):
        normalize_digest_values("a" * 64, field="rendered_paper_sha256")


def test_selector_universe_contains_all_authenticated_classes() -> None:
    from redco.analysis.stage_d_v13_source_phase_a import _build_universe

    units = reconstruct_retired_units(ROOT)
    witness, raw_references = build_forbidden_witness(
        ROOT,
        units,
        authenticated_historical_inputs(ROOT),
    )
    universe = _build_universe(witness, raw_references)
    sets = witness["forbidden_sets"]
    assert universe["paper_ids"] == (
        set(sets["retired_paper_ids"])
        | set(sets["old_snapshot_paper_ids"])
        | set(sets["predecessor_paper_ids"])
        | set(sets["historical_paper_ids"])
    )
    assert set(sets["old_snapshot_rendered_paper_sha256"]).issubset(
        set(universe["rendered_paper_sha256"])
    )
    assert set(sets["predecessor_rendered_paper_sha256"]).issubset(
        set(universe["rendered_paper_sha256"])
    )
    assert set(sets["historical_address_sha256"]).issubset(set(universe["addresses"]))
    assert raw_references.issubset(universe["reference_spans"])


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mutation",
    (
        "retired_units",
        "retired_papers",
        "examples",
        "rows",
        "rendered",
        "references",
        "source_addresses",
        "historical_addresses",
        "old_snapshot_papers",
        "predecessor_examples",
        "exclusion_hash",
    ),
)
def test_forbidden_witness_rebuild_rejects_each_mutation(mutation: str) -> None:
    witness = copy.deepcopy(phase_a_result()["forbidden_witness"])
    if mutation == "retired_units":
        witness["retired_units"].pop()
    elif mutation == "retired_papers":
        witness["forbidden_sets"]["retired_paper_ids"].pop()
    elif mutation == "examples":
        witness["forbidden_sets"]["retired_example_ids"][0] = "substituted-example"
    elif mutation == "rows":
        witness["forbidden_sets"]["retired_row_sha256"].pop()
    elif mutation == "rendered":
        witness["forbidden_sets"]["rendered_paper_sha256"][0] = "0" * 64
    elif mutation == "references":
        witness["forbidden_sets"]["reference_span_sha256"].pop()
    elif mutation == "source_addresses":
        witness["forbidden_sets"]["source_address_sha256"][0] = "1" * 64
    elif mutation == "historical_addresses":
        witness["forbidden_sets"]["historical_address_sha256"][0] = "2" * 64
    elif mutation == "old_snapshot_papers":
        witness["forbidden_sets"]["old_snapshot_paper_ids"].pop()
    elif mutation == "predecessor_examples":
        witness["forbidden_sets"]["predecessor_example_ids"][0] = "substituted-predecessor"
    else:
        witness["exclusion_hashes"]["retired_paper_ids"] = "3" * 64
    with pytest.raises(ValueError, match=r"forbidden witness|cardinality|exclusion"):
        validate_forbidden_witness(ROOT, witness)


def test_forbidden_witness_rejects_recomputed_self_hash() -> None:
    witness = copy.deepcopy(phase_a_result()["forbidden_witness"])
    witness["forbidden_sets"]["retired_paper_ids"][0] = "substituted-paper"
    witness_without_hash = {key: value for key, value in witness.items() if key != "witness_sha256"}
    witness["witness_sha256"] = sha256_json(witness_without_hash)
    with pytest.raises(ValueError, match="forbidden witness differs"):
        validate_forbidden_witness(ROOT, witness)


def test_phase_a_publication_rejects_authenticated_input_hardlink(tmp_path: Path) -> None:
    payloads = phase_a_payloads(ROOT, test_node_ids=TEST_NODE_IDS)
    destination = tmp_path / PHASE_A_OUTPUTS[0]
    destination.parent.mkdir(parents=True)
    copied_input = tmp_path / "receipt-v1-copy.json"
    shutil.copyfile(
        ROOT / "datasets/stage-d/qasper-support-successor-manifest-v1.json",
        copied_input,
    )
    os.link(copied_input, destination)
    immutable = phase_a_immutable_paths(ROOT)
    immutable[str(copied_input)] = "temporary-authenticated-receipt"
    with pytest.raises(ValueError, match="hard-link alias"):
        write_phase_a_outputs(tmp_path, payloads, immutable_paths=immutable)


def test_phase_a_publication_rejects_cross_output_hardlink(tmp_path: Path) -> None:
    payloads = phase_a_payloads(ROOT, test_node_ids=TEST_NODE_IDS)
    first = tmp_path / PHASE_A_OUTPUTS[0]
    second = tmp_path / PHASE_A_OUTPUTS[1]
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(payloads[PHASE_A_OUTPUTS[0]])
    os.link(first, second)
    with pytest.raises(ValueError, match="hard-link aliases"):
        write_phase_a_outputs(
            tmp_path,
            payloads,
            immutable_paths=phase_a_immutable_paths(ROOT),
        )


def test_phase_a_publication_rejects_symlink_parent_before_write(tmp_path: Path) -> None:
    payloads = phase_a_payloads(ROOT, test_node_ids=TEST_NODE_IDS)
    outside = tmp_path / "outside"
    outside.mkdir()
    parent = tmp_path / "configs" / "stage-d"
    parent.mkdir(parents=True)
    os.symlink(outside, parent / "v13-draft", target_is_directory=True)
    with pytest.raises(ValueError, match=r"symlink|escapes"):
        write_phase_a_outputs(
            tmp_path,
            payloads,
            immutable_paths=phase_a_immutable_paths(ROOT),
        )
    assert not list(outside.iterdir())


def test_phase_a_status_capture_is_independent_and_exact() -> None:
    from redco.analysis.stage_d_v13_source_phase_a_bindings import PHASE_A_STATUS_SIGNATURE

    payloads = phase_a_payloads(ROOT, test_node_ids=TEST_NODE_IDS)
    status = json.loads(payloads[PHASE_A_OUTPUTS[4]])
    assert status["node_ids"] == list(TEST_NODE_IDS)
    expected = {"passed": len(TEST_NODE_IDS), "failed": 0, "skipped": 0, "xfailed": 0}
    assert status["status"] == expected
    assert status["status_signature"] == sha256_json(
        {"node_ids": list(TEST_NODE_IDS), "status": expected}
    )
    assert status["status_signature"] == PHASE_A_STATUS_SIGNATURE


def test_phase_a_missing_source_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Parquet artifact"):
        authenticate_source_artifact(tmp_path)


def test_immutable_v1_audit_rejects_repaired_tree() -> None:
    from redco.analysis.stage_d_v12_finalization_audit import audit_archive

    with pytest.raises(ValueError, match="immutable repository hash differs"):
        audit_archive(
            ROOT / "runs/stage-d/stage-d1-support-v12-terminal.tar.gz",
            ROOT / "runs/stage-d/stage-d1-support-v12-evidence-sha256.txt",
            repo_root=ROOT,
            terminal_report=ROOT / "reports/stage-d1-support-v12-terminal.json",
        )


def _approval_overlay(tmp_path: Path) -> Path:
    from redco.analysis.stage_d_v13_source_phase_a_bindings import BEHAVIOR_BINDING_FILES
    from redco.analysis.stage_d_v13_source_phase_a_trust import TRUST_MODULE_RELATIVE

    overlay = tmp_path / "approval-overlay"
    for relative in (
        *BEHAVIOR_BINDING_FILES,
        APPROVAL_ANCHOR_RELATIVE,
        BINDINGS_RELATIVE,
        TRUST_MODULE_RELATIVE,
    ):
        destination = overlay / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return overlay


def test_phase_a_approval_anchor_authenticates_registry_policy() -> None:
    approval = authenticate_external_anchor(ROOT)
    assert approval["anchor_path"] == APPROVAL_ANCHOR_RELATIVE
    assert approval["registry_path"] == BINDINGS_RELATIVE
    immutable = phase_a_immutable_paths(ROOT)
    assert str((ROOT / APPROVAL_ANCHOR_RELATIVE).resolve()) in immutable
    assert str((ROOT / BINDINGS_RELATIVE).resolve()) in immutable


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mutation",
    ("selector", "selector_and_registry", "registry", "derivation", "status", "resume", "anchor"),
)
def test_phase_a_approval_anchor_mutations_fail_before_publication(
    tmp_path: Path, mutation: str
) -> None:
    overlay = _approval_overlay(tmp_path)
    selector = overlay / "src/redco/analysis/stage_d_v13_source_phase_a_selector.py"
    registry = overlay / BINDINGS_RELATIVE
    anchor = overlay / APPROVAL_ANCHOR_RELATIVE
    if mutation in {"selector", "selector_and_registry"}:
        mutated_selector = selector.read_bytes().replace(
            b"class TerminalIdentityCollision", b"class TamperedTerminalIdentityCollision"
        )
        selector.write_bytes(mutated_selector)
        if mutation == "selector_and_registry":
            from redco.analysis.stage_d_v13_source_phase_a_bindings import APPROVED_BEHAVIOR_HASHES

            old_digest = APPROVED_BEHAVIOR_HASHES[
                "src/redco/analysis/stage_d_v13_source_phase_a_selector.py"
            ].encode("ascii")
            new_digest = sha256_bytes(mutated_selector).encode("ascii")
            registry.write_bytes(registry.read_bytes().replace(old_digest, new_digest))
    elif mutation == "registry":
        registry.write_bytes(registry.read_bytes() + b"\n# registry tamper\n")
    elif mutation == "derivation":
        registry.write_bytes(
            registry.read_bytes().replace(b"redco-stage-d1-support-v1", b"tampered")
        )
    elif mutation == "status":
        from redco.analysis.stage_d_v13_source_phase_a_bindings import PHASE_A_STATUS_SIGNATURE

        original_registry = registry.read_bytes()
        mutated_registry = original_registry.replace(
            PHASE_A_STATUS_SIGNATURE.encode("ascii"), b"0" * 64, 1
        )
        assert mutated_registry != original_registry
        status_mutations: list[tuple[Path, bytes]] = [(registry, mutated_registry)]
        original_anchor = anchor.read_bytes()
        for field, value in (
            ("draft_unfrozen", False),
            ("launch_authorized", True),
            ("provider_calls_authorized", True),
            ("phase_b_authorized", True),
            ("foundation_only", False),
            ("non_authorizing", False),
            ("seed", "tampered"),
            ("address", "tampered"),
        ):
            mutated_anchor = json.loads(original_anchor)
            mutated_anchor[field] = value
            status_mutations.append((anchor, canonical_json_bytes(mutated_anchor)))
        mutated_anchor = json.loads(original_anchor)
        mutated_anchor["candidate"]["source_ordinal"] = 180
        status_mutations.append((anchor, canonical_json_bytes(mutated_anchor)))
        mutated_anchor = json.loads(original_anchor)
        mutated_anchor["policy"]["status_envelope"]["status_signature"] = "0" * 64
        status_mutations.append((anchor, canonical_json_bytes(mutated_anchor)))
        for path, mutated in status_mutations:
            original = path.read_bytes()
            path.write_bytes(mutated)
            with pytest.raises((FileNotFoundError, ValueError)):
                _authenticated_behavior_bindings(overlay)
            assert not (tmp_path / "published").exists()
            path.write_bytes(original)
        return
    elif mutation == "resume":
        registry.write_bytes(
            registry.read_bytes().replace(b"stage-d-v13-phase-b-resume-v2", b"tampered-resume-v2")
        )
    else:
        anchor.write_bytes(
            anchor.read_bytes().replace(b'"schema_version":1', b'"schema_version":2')
        )
    with pytest.raises((FileNotFoundError, ValueError)):
        _authenticated_behavior_bindings(overlay)
    assert not (tmp_path / "published").exists()


def _synthetic_resume_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    path = tmp_path / "resume-fixture.parquet"
    table = pa.table({"ordinal": list(range(360)), "token": ["x"] * 360})
    parquet.write_table(table, path, row_group_size=360)
    from redco.analysis.stage_d_v13_source_phase_a_bindings import PHASE_B_RESUME_CONTRACT

    authorization = {
        "state": "reviewed_preselection_checkpoint",
        "phase_b_authorized": True,
        "preselection_checkpoint_sha256": "c" * 64,
        "source_sha256": sha256_bytes(path.read_bytes()),
        "schema_sha256": source_schema_sha256(path),
        "decoder_contract_sha256": sha256_json(PHASE_B_RESUME_CONTRACT),
        "start_ordinal": 180,
        "batch_size": 180,
        "row_groups": [0],
        "use_threads": False,
        "source_order": "physical_ordinal",
    }
    return path, authorization


@dataclass(slots=True)
class _SyntheticResumeInstrumentation:
    decoded_objects: list[dict[str, int | str]] = dataclass_field(default_factory=list)
    materialized_ranges: list[list[int]] = dataclass_field(default_factory=list)
    canonicalized_ordinals: list[int] = dataclass_field(default_factory=list)
    evaluated_ordinals: list[int] = dataclass_field(default_factory=list)
    emitted_ordinals: list[int] = dataclass_field(default_factory=list)

    def record_decoded(self, *, start: int, rows: int) -> None:
        end = start + rows - 1
        previous_end = (
            int(self.decoded_objects[-1]["end_ordinal"]) if self.decoded_objects else None
        )
        if rows <= 0 or (previous_end is not None and start != previous_end + 1):
            raise ValueError("synthetic resume decoder logical batch order differs")
        self.decoded_objects.append(
            {
                "kind": "pyarrow_record_batch",
                "rows": rows,
                "start_ordinal": start,
                "end_ordinal": end,
            }
        )

    def record_materialized(self, *, start: int, rows: int) -> None:
        end = start + rows - 1
        if start < 180:
            raise ValueError("synthetic resume materialization began before ordinal 180")
        self.materialized_ranges.append([start, end])

    def record_canonicalized(self, ordinal: int) -> None:
        if ordinal < 180:
            raise ValueError("synthetic resume canonicalization began before ordinal 180")
        self.canonicalized_ordinals.append(ordinal)

    def record_evaluated(self, ordinal: int) -> None:
        if ordinal < 180:
            raise ValueError("synthetic resume evaluation began before ordinal 180")
        self.evaluated_ordinals.append(ordinal)

    def record_emitted(self, ordinal: int) -> None:
        if ordinal < 180 or (
            self.emitted_ordinals and ordinal != self.emitted_ordinals[-1] + 1
        ):
            raise ValueError("synthetic resume emitted ordinals are not contiguous")
        self.emitted_ordinals.append(ordinal)


def _synthetic_resume_source_rows(
    path: Path,
    *,
    authorization: Mapping[str, Any],
    instrumentation: _SyntheticResumeInstrumentation | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Test-local synthetic ordering fixture; never used by production code."""

    required = {
        "state",
        "phase_b_authorized",
        "preselection_checkpoint_sha256",
        "source_sha256",
        "schema_sha256",
        "decoder_contract_sha256",
        "start_ordinal",
        "batch_size",
        "row_groups",
        "use_threads",
        "source_order",
    }
    if not required.issubset(authorization):
        raise PhaseBResumeAuthorizationError("synthetic resume binding is incomplete")
    from redco.analysis.stage_d_v13_source_phase_a_bindings import PHASE_B_RESUME_CONTRACT

    if (
        authorization["state"] != "reviewed_preselection_checkpoint"
        or authorization["phase_b_authorized"] is not True
        or authorization["start_ordinal"] != 180
        or authorization["batch_size"] != 180
        or authorization["row_groups"] != [0]
        or authorization["use_threads"] is not False
        or authorization["source_order"] != "physical_ordinal"
        or authorization["decoder_contract_sha256"] != sha256_json(PHASE_B_RESUME_CONTRACT)
    ):
        raise PhaseBResumeAuthorizationError("synthetic resume binding differs")
    if not path.is_file():
        raise FileNotFoundError(path)
    if authorization["source_sha256"] != sha256_bytes(path.read_bytes()):
        raise PhaseBResumeAuthorizationError("synthetic resume source hash differs")
    if authorization["schema_sha256"] != source_schema_sha256(path):
        raise PhaseBResumeAuthorizationError("synthetic resume source schema differs")
    import pyarrow.parquet as parquet

    parquet_file = parquet.ParquetFile(path)
    metadata = parquet_file.metadata
    if metadata is None or metadata.num_rows <= 180:
        raise PhaseBResumeAuthorizationError("synthetic resume source has no post-cutoff rows")
    observer = instrumentation or _SyntheticResumeInstrumentation()

    def rows() -> Iterator[tuple[int, dict[str, Any]]]:
        next_ordinal = 0
        for batch in parquet_file.iter_batches(
            batch_size=180, row_groups=[0], use_threads=False
        ):
            row_count = int(batch.num_rows)
            observer.record_decoded(start=next_ordinal, rows=row_count)
            batch_start = next_ordinal
            next_ordinal += row_count
            if next_ordinal <= 180:
                continue
            if batch_start < 180:
                raise PhaseBResumeAuthorizationError(
                    "synthetic resume batch straddles ordinal 180"
                )
            decoded_rows = batch.to_pylist()
            if len(decoded_rows) != row_count:
                raise ValueError("synthetic resume batch cardinality changed")
            observer.record_materialized(start=batch_start, rows=row_count)
            for offset, row in enumerate(decoded_rows):
                ordinal = batch_start + offset
                if not isinstance(row, dict):
                    raise ValueError(f"synthetic resume row {ordinal} is not an object")
                observer.record_canonicalized(ordinal)
                observer.record_evaluated(ordinal)
                observer.record_emitted(ordinal)
                yield ordinal, row

    return rows()


def test_phase_a_resume_invocation_count_is_zero() -> None:
    assert phase_a_result()["phase_a_wall"]["phase_b_resume_decoder_invocations"] == 0
    assert resume_decoder_invocation_count() == 0


def test_dormant_resume_decoder_starts_at_ordinal_180(tmp_path: Path) -> None:
    path, authorization = _synthetic_resume_fixture(tmp_path)
    instrumentation = _SyntheticResumeInstrumentation()
    rows = list(
        _synthetic_resume_source_rows(
            path, authorization=authorization, instrumentation=instrumentation
        )
    )
    assert [ordinal for ordinal, _row in rows] == list(range(180, 360))
    assert instrumentation.decoded_objects == [
        {"kind": "pyarrow_record_batch", "rows": 180, "start_ordinal": 0, "end_ordinal": 179},
        {"kind": "pyarrow_record_batch", "rows": 180, "start_ordinal": 180, "end_ordinal": 359},
    ]
    assert instrumentation.materialized_ranges == [[180, 359]]
    assert instrumentation.emitted_ordinals == list(range(180, 360))
    assert instrumentation.canonicalized_ordinals == list(range(180, 360))
    assert instrumentation.evaluated_ordinals == list(range(180, 360))


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mutation", ("source", "schema", "version", "config")
)
def test_dormant_resume_decoder_rejects_wrong_binding(tmp_path: Path, mutation: str) -> None:
    path, authorization = _synthetic_resume_fixture(tmp_path)
    if mutation == "source":
        authorization["source_sha256"] = "a" * 64
    elif mutation == "schema":
        authorization["schema_sha256"] = "b" * 64
    elif mutation == "version":
        authorization["decoder_contract_sha256"] = "d" * 64
    else:
        authorization["batch_size"] = 181
    with pytest.raises(PhaseBResumeAuthorizationError):
        _synthetic_resume_source_rows(path, authorization=authorization)


def test_dormant_resume_decoder_requires_reviewed_checkpoint(tmp_path: Path) -> None:
    path, authorization = _synthetic_resume_fixture(tmp_path)
    authorization["state"] = "phase_a_prefix_only"
    with pytest.raises(PhaseBResumeAuthorizationError):
        _synthetic_resume_source_rows(path, authorization=authorization)


def test_foundation_resume_entrypoint_rejects_all_caller_authority() -> None:
    resume_entrypoint = cast(Any, resume_source_rows)
    with pytest.raises(PhaseBResumeUnavailable, match="repeatable raw resume iterator is retired"):
        resume_source_rows()
    with pytest.raises(TypeError):
        resume_entrypoint(Path("alternate.parquet"))
    with pytest.raises(TypeError):
        resume_entrypoint(authorization=True)


def test_gate_g_removes_production_synthetic_resume_seam() -> None:
    decoder = importlib.import_module("redco.analysis.stage_d_v13_source_phase_a_decoder")
    assert not hasattr(decoder, "_synthetic_resume_source_rows")
    assert not hasattr(decoder, "_validate_synthetic_resume_authorization")
    assert not hasattr(decoder, "_resume_rows")
    assert not hasattr(decoder, "ResumeDecoderInstrumentation")
    assert "_synthetic_resume_source_rows" not in decoder.__all__


def _git_test(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_repair_r_legacy_c3_owners_are_retired_and_gate_rejects_legacy_path(
    tmp_path: Path,
) -> None:
    decoder = importlib.import_module("redco.analysis.stage_d_v13_source_phase_a_decoder")
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")

    for retired in (
        "_validate_c3_artifact",
        "_validate_head_authorization_c3",
        "_authenticate_c3_and_source",
        "PHASE_B_AUTHORIZATION_RELATIVE",
        "PHASE_C3_AUTHORIZATION_RELATIVE",
    ):
        assert not hasattr(decoder, retired)

    with pytest.raises(PhaseBResumeUnavailable, match="C3-v1"):
        decoder.validate_future_phase_b_authorization_artifact()
    with pytest.raises(PhaseBResumeUnavailable, match="repeatable raw resume iterator"):
        decoder.resume_source_rows()

    repo, _gate = _synthetic_gate_chain(tmp_path, legacy_c=True)
    with pytest.raises(selection.SelectionGateError, match="legacy"):
        selection._validate_c3_v2(repo)



def _synthetic_gate_chain(
    tmp_path: Path,
    *,
    terminal: str = "c3",
    mutation: str | None = None,
    extra_g: bool = False,
    legacy_c: bool = False,
) -> tuple[Path, str | None]:
    """Create only synthetic F -> B -> R -> G -> C3-v2 history."""

    from redco.analysis.stage_d_v13_source_phase_a_bindings import (
        PHASE_A_AUDIT_RELATIVE,
        PHASE_A_STATUS_SIGNATURE,
        PHASE_B_BINDING_DOMAIN,
        PHASE_B_RESUME_CONTRACT_V2_SHA256,
        PHASE_B_SOURCE_SELECTION_CONTRACT_V4,
        PHASE_B_SOURCE_SELECTION_CONTRACT_V4_SHA256,
        PHASE_C3_V2_AUTHORIZATION_DOMAIN,
        PHASE_C3_V2_AUTHORIZATION_RELATIVE,
        SELECTION_CLAIM_RELATIVE,
        SELECTION_GATE_APPROVAL_TEXT_SHA256,
        SELECTION_GATE_APPROVAL_THREAD_ID,
        SELECTION_RECEIPT_RELATIVE,
    )
    from redco.analysis.stage_d_v13_source_phase_a_decoder import (
        PHASE_A_CONFIG_RELATIVE,
        REPAIR_DIFF_ALLOWLIST,
        _resume_contract_v3_hash,
        git_blob_sha1,
    )
    from redco.analysis.stage_d_v13_source_selection import (
        LEGACY_C_AUTHORIZATION_RELATIVE,
        compute_scan_id,
    )

    repo = tmp_path / "gate-g-repo"
    repo.mkdir(parents=True)
    _git_test(repo, "init", "--quiet")
    _git_test(repo, "config", "user.email", "gate-g@example.invalid")
    _git_test(repo, "config", "user.name", "Gate G Test")
    (repo / "parent.txt").write_text("pre-F", encoding="utf-8")
    _git_test(repo, "add", "parent.txt")
    _git_test(repo, "commit", "--quiet", "-m", "pre-F")

    source_bytes = b"synthetic source bytes; no production rows"
    source_contract = {
        "path": "qasper/train/0000.parquet",
        "revision": "synthetic-revision",
        "sha256": sha256_bytes(source_bytes),
        "schema_sha256": "s" * 64,
        "row_count": 888,
    }
    manifest_relative = Path(
        "reports/stage-d1-support-v13-foundation-tree-manifest-f1.json"
    )
    manifest_path = repo / manifest_relative
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(
        canonical_json_bytes(
            {"qasper": copy.deepcopy(source_contract), "source": copy.deepcopy(source_contract)}
        )
    )
    config_path = repo / PHASE_A_CONFIG_RELATIVE
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(b"synthetic phase-a config")
    audit_path = repo / PHASE_A_AUDIT_RELATIVE
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_bytes(
        canonical_json_bytes(
            {"source_authentication": {"forbidden_witness": {"witness_sha256": "a" * 64}}}
        )
    )
    for relative in REPAIR_DIFF_ALLOWLIST:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("foundation", encoding="utf-8")
    _git_test(repo, "add", ".")
    _git_test(repo, "commit", "--quiet", "-m", "F")
    foundation_commit = _git_test(repo, "rev-parse", "HEAD")
    foundation_tree = _git_test(repo, "rev-parse", "HEAD^{tree}")

    b_path = repo / PHASE_B_BINDING_RELATIVE
    b_path.parent.mkdir(parents=True, exist_ok=True)
    b_payload: dict[str, Any] = {
        "schema_version": 1,
        "domain": PHASE_B_BINDING_DOMAIN,
        "state": "B",
        "draft_unfrozen": True,
        "foundation_only": True,
        "non_authorizing": True,
        "candidate": {
            "source_ordinal": None,
            "paper_id": None,
            "example_id": None,
            "row": None,
            "seed": None,
            "address": None,
        },
        "seed": None,
        "address": None,
        "phase_b_authorized": False,
        "source_selection_authorized": False,
        "launch_authorized": False,
        "provider_calls_authorized": False,
        "model_calls_authorized": False,
        "prime_gpu_scientific_launch_authorized": False,
        "status_signature": PHASE_A_STATUS_SIGNATURE,
        "foundation_commit": foundation_commit,
        "foundation_tree_sha1": foundation_tree,
        "foundation_manifest": {
            "path": str(manifest_relative).replace("\\", "/"),
            "sha256": sha256_bytes(manifest_path.read_bytes()),
            "git_blob_sha1": git_blob_sha1(manifest_path.read_bytes()),
        },
        "bindings": {
            "source_artifact": {
                "path": SOURCE_ARTIFACT_RELATIVE,
                "sha256": source_contract["sha256"],
                "schema_sha256": source_contract["schema_sha256"],
                "row_count": 888,
            },
            "phase_a_config": {
                "path": PHASE_A_CONFIG_RELATIVE,
                "sha256": sha256_bytes(config_path.read_bytes()),
            },
            "decoder_contract_sha256": PHASE_B_RESUME_CONTRACT_V2_SHA256,
        },
    }
    b_path.write_bytes(canonical_json_bytes(b_payload))
    _git_test(repo, "add", PHASE_B_BINDING_RELATIVE)
    _git_test(repo, "commit", "--quiet", "-m", "B")
    b_raw = b_path.read_bytes()
    if terminal == "b":
        return repo, None

    for relative in REPAIR_DIFF_ALLOWLIST:
        path = repo / relative
        path.write_bytes(path.read_bytes() + b"\nrepair")
    _git_test(repo, "add", *REPAIR_DIFF_ALLOWLIST)
    _git_test(repo, "commit", "--quiet", "-m", "R")
    if terminal == "r":
        return repo, None

    gate_path = repo / "src/redco/analysis/stage_d_v13_source_selection.py"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_bytes(b"synthetic Gate G module; no source access")
    for relative in REPAIR_DIFF_ALLOWLIST:
        path = repo / relative
        path.write_bytes(path.read_bytes() + b"\ngate")
    if extra_g:
        (repo / "unapproved-g.txt").write_text("unapproved", encoding="utf-8")
    _git_test(repo, "add", "src/redco/analysis/stage_d_v13_source_selection.py")
    _git_test(repo, "add", *REPAIR_DIFF_ALLOWLIST)
    if extra_g:
        _git_test(repo, "add", "unapproved-g.txt")
    _git_test(repo, "commit", "--quiet", "-m", "G")
    gate_commit = _git_test(repo, "rev-parse", "HEAD")
    if legacy_c:
        legacy_path = repo / LEGACY_C_AUTHORIZATION_RELATIVE
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_bytes(b"{}")
        _git_test(repo, "add", LEGACY_C_AUTHORIZATION_RELATIVE)
        _git_test(repo, "commit", "--quiet", "-m", "legacy-C")
        gate_commit = _git_test(repo, "rev-parse", "HEAD")
    if terminal == "g":
        return repo, gate_commit

    c3_path = repo / PHASE_C3_V2_AUTHORIZATION_RELATIVE
    c3_path.parent.mkdir(parents=True, exist_ok=True)
    audit_raw = audit_path.read_bytes()
    binding_commit = _git_test(repo, "rev-parse", "HEAD~2")
    repair_commit = _git_test(repo, "rev-parse", "HEAD~1")
    c3_payload: dict[str, Any] = {
        "schema_version": 2,
        "domain": PHASE_C3_V2_AUTHORIZATION_DOMAIN,
        "state": "C3-v2",
        "draft_unfrozen": False,
        "candidate": None,
        "seed": None,
        "address": None,
        "phase_b_authorized": False,
        "phase_b_source_selection_authorized": True,
        "source_selection_authorized": False,
        "launch_authorized": False,
        "provider_calls_authorized": False,
        "model_calls_authorized": False,
        "prime_gpu_scientific_launch_authorized": False,
        "science_authorized": False,
        "status_signature": PHASE_A_STATUS_SIGNATURE,
        "foundation_commit": foundation_commit,
        "foundation_tree_sha1": foundation_tree,
        "binding_commit": binding_commit,
        "repair_commit": repair_commit,
        "gate_commit": gate_commit,
        "binding_artifact": {
            "path": PHASE_B_BINDING_RELATIVE,
            "sha256": sha256_bytes(b_raw),
            "git_blob_sha1": git_blob_sha1(b_raw),
        },
        "source": copy.deepcopy(source_contract),
        "contracts": {
            "v2_sha256": PHASE_B_RESUME_CONTRACT_V2_SHA256,
            "v3_sha256": _resume_contract_v3_hash(),
            "v4_sha256": PHASE_B_SOURCE_SELECTION_CONTRACT_V4_SHA256,
            "preselection_checkpoint_sha256": sha256_bytes(b_raw),
        },
        "runtime_versions": {
            "python": "3.12.3",
            "datasets": "5.0.0",
            "pyarrow": "25.0.0",
        },
        "approval": {
            "thread_id": SELECTION_GATE_APPROVAL_THREAD_ID,
            "text_sha256": SELECTION_GATE_APPROVAL_TEXT_SHA256,
        },
        "scan": {
            "scan_id": compute_scan_id(gate_commit),
            "attempt_limit": 1,
            "retry": False,
            "start_ordinal": 180,
            "final_possible_ordinal": 887,
            "stop_rule": cast(str, PHASE_B_SOURCE_SELECTION_CONTRACT_V4["stop_rule"]),
        },
        "paths": {"claim": SELECTION_CLAIM_RELATIVE, "receipt": SELECTION_RECEIPT_RELATIVE},
        "forbidden_universe": {
            "artifact_path": PHASE_A_AUDIT_RELATIVE,
            "artifact_sha256": sha256_bytes(audit_raw),
            "witness_sha256": "a" * 64,
        },
    }
    if mutation == "scan":
        cast(dict[str, Any], c3_payload["scan"])["scan_id"] = "tampered"
    elif mutation == "v4":
        cast(dict[str, Any], c3_payload["contracts"])["v4_sha256"] = "0" * 64
    elif mutation == "approval":
        cast(dict[str, Any], c3_payload["approval"])["text_sha256"] = "0" * 64
    elif mutation == "source":
        cast(dict[str, Any], c3_payload["source"])["sha256"] = "0" * 64
    elif mutation == "candidate":
        c3_payload["candidate"] = {"source_ordinal": 180}
    elif mutation == "forbidden":
        cast(dict[str, Any], c3_payload["forbidden_universe"])["witness_sha256"] = "0" * 64
    elif mutation == "raw_refs":
        cast(dict[str, Any], c3_payload["forbidden_universe"])["raw_reference_spans"] = []
    elif mutation == "unknown":
        c3_payload["unknown"] = None
    c3_path.write_bytes(canonical_json_bytes(c3_payload))
    _git_test(repo, "add", PHASE_C3_V2_AUTHORIZATION_RELATIVE)
    _git_test(repo, "commit", "--quiet", "-m", "C3-v2")
    return repo, gate_commit


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mutation",
    ("scan", "v4", "approval", "source", "candidate", "forbidden", "raw_refs", "unknown"),
)
def test_gate_g_c3_v2_schema_mutations_fail_closed(tmp_path: Path, mutation: str) -> None:
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")
    repo, _gate = _synthetic_gate_chain(tmp_path, mutation=mutation)
    with pytest.raises(selection.SelectionGateError):
        selection._validate_c3_v2(repo)


def test_gate_g_validates_exact_future_chain_without_source_access(tmp_path: Path) -> None:
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")
    repo, gate = _synthetic_gate_chain(tmp_path)
    artifact, gate_commit, inputs = selection._validate_c3_v2(repo)
    assert artifact["state"] == "C3-v2"
    assert gate_commit == gate
    assert inputs["source_contract"]["row_count"] == 888
    assert inputs["audit_raw"]


def test_gate_g_rejects_legacy_c_and_extra_gate_paths(tmp_path: Path) -> None:
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")
    legacy_repo, _gate = _synthetic_gate_chain(
        tmp_path / "legacy", terminal="g", legacy_c=True
    )
    with pytest.raises(selection.SelectionGateError, match="legacy"):
        selection._validate_c3_v2(legacy_repo)
    extra_repo, _gate = _synthetic_gate_chain(tmp_path / "extra", extra_g=True)
    with pytest.raises(selection.SelectionGateError, match="R to G"):
        selection._validate_c3_v2(extra_repo)


def test_gate_g_absent_c3_fails_before_runtime_source_or_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")
    decoder = importlib.import_module("redco.analysis.stage_d_v13_source_phase_a_decoder")
    repo, _gate = _synthetic_gate_chain(tmp_path, terminal="g")
    monkeypatch.setattr(decoder, "PROJECT_ROOT", repo)
    monkeypatch.setattr(
        decoder,
        "_require_runtime_versions_only",
        lambda: pytest.fail("runtime was inspected before C3-v2"),
    )
    monkeypatch.setattr(
        decoder,
        "_validate_production_source_metadata",
        lambda *_args: pytest.fail("source was inspected before C3-v2"),
    )
    with pytest.raises(selection.SelectionGateError, match="C3-v2 is absent"):
        selection.activate_selection_gate()
    assert not (repo / selection.SELECTION_CLAIM_RELATIVE).exists()


def _synthetic_selection_row(
    paper_id: str, question_id: str, *, two_questions: bool = False
) -> dict[str, Any]:
    span = f"{paper_id} first synthetic evidence span is long enough"
    answers: list[dict[str, Any]] = [
        {"answer": [{"unanswerable": False, "evidence": [span]}]}
    ]
    questions = [question_id]
    question_ids = [question_id]
    if two_questions:
        answers.append({"answer": [{"unanswerable": True}]})
        questions.append("unreachable")
        question_ids.append("unreachable")
    return {
        "id": paper_id,
        "title": f"Synthetic {paper_id}",
        "abstract": span,
        "full_text": {"section_name": [], "paragraphs": []},
        "qas": {"question": questions, "question_id": question_ids, "answers": answers},
    }


class _SyntheticRows:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.requested: list[int] = []

    def iter_rows(
        self, *, start_ordinal: int, end_ordinal: int
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        assert (start_ordinal, end_ordinal) == (180, 887)
        for offset, row in enumerate(self.rows):
            ordinal = 180 + offset
            self.requested.append(ordinal)
            yield ordinal, row


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "collision_kind", ("example", "rendered", "row", "address")
)
def test_gate_g_terminal_collision_stops_before_later_candidate(
    collision_kind: str,
) -> None:
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")
    first = _synthetic_selection_row("first-paper", "first", two_questions=True)
    later = _synthetic_selection_row("later-paper", "later")
    selected = select_first_eligible(first)
    assert selected is not None
    chosen, _index = selected
    universe = selection.SelectionUniverse(
        paper_ids=frozenset(),
        example_ids=(
            frozenset({chosen["example_id"]})
            if collision_kind == "example"
            else frozenset()
        ),
        rendered_paper_sha256=(
            frozenset({sha256_bytes(str(chosen["paper"]).encode("utf-8"))})
            if collision_kind == "rendered"
            else frozenset()
        ),
        reference_spans=frozenset(),
        row_sha256=(
            frozenset({source_row_sha256(first)})
            if collision_kind == "row"
            else frozenset()
        ),
        addresses=frozenset(),
    )
    if collision_kind == "address":
        universe = selection.SelectionUniverse(
            paper_ids=frozenset(),
            example_ids=frozenset(),
            rendered_paper_sha256=frozenset(),
            reference_spans=frozenset(),
            row_sha256=frozenset(),
            addresses=frozenset({selection._candidate_address_sha256(first, chosen)}),
        )
    fake = _SyntheticRows([first, later])
    outcome = selection._scan_once(fake, universe)
    assert outcome["disposition"] == "terminal_collision"
    assert outcome["collision_class"] == {
        "example": "example_id_collision",
        "rendered": "rendered_paper_collision",
        "row": "source_row_collision",
        "address": "source_address_collision",
    }[collision_kind]
    assert fake.requested == [180]
    assert outcome["candidate"] is None


def test_gate_g_continuable_paper_and_reference_collisions_continue() -> None:
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")
    for kind in ("paper", "reference"):
        first = _synthetic_selection_row("first-paper", "first")
        later = _synthetic_selection_row("later-paper", "later")
        selected = select_first_eligible(first)
        assert selected is not None
        span = selected[0]["reference_evidence"][0]
        universe = selection.SelectionUniverse(
            paper_ids=frozenset({"first-paper"}) if kind == "paper" else frozenset(),
            example_ids=frozenset(),
            rendered_paper_sha256=frozenset(),
            reference_spans=frozenset({span}) if kind == "reference" else frozenset(),
            row_sha256=frozenset(),
            addresses=frozenset(),
        )
        fake = _SyntheticRows([first, later])
        outcome = selection._scan_once(fake, universe)
        assert outcome["disposition"] == "eligible_candidate"
        assert outcome["stop_ordinal"] == 181
        assert fake.requested == [180, 181]


def test_gate_g_exhaustion_stops_at_final_ordinal() -> None:
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")
    empty_rows = [
        {
            "id": f"empty-{ordinal}",
            "title": "Synthetic empty",
            "abstract": "no exact evidence",
            "full_text": {"section_name": [], "paragraphs": []},
            "qas": {"question": [], "question_id": [], "answers": []},
        }
        for ordinal in range(708)
    ]
    fake = _SyntheticRows(empty_rows)
    outcome = selection._scan_once(
        fake,
        selection.SelectionUniverse(
            paper_ids=frozenset(),
            example_ids=frozenset(),
            rendered_paper_sha256=frozenset(),
            reference_spans=frozenset(),
            row_sha256=frozenset(),
            addresses=frozenset(),
        ),
    )
    assert outcome["disposition"] == "exhausted"
    assert outcome["stop_ordinal"] == 887
    assert fake.requested[-1] == 887
    assert len(fake.requested) == 708


def test_gate_g_reconstructs_authenticated_reference_projection_without_source_access() -> None:
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")
    decoder = importlib.import_module("redco.analysis.stage_d_v13_source_phase_a_decoder")
    bindings = importlib.import_module("redco.analysis.stage_d_v13_source_phase_a_bindings")
    head = decoder._git_text(ROOT, "rev-parse", "--verify", "HEAD^{commit}")
    audit_raw = decoder._git_blob_at_commit(ROOT, head, bindings.PHASE_A_AUDIT_RELATIVE)
    universe = selection._authenticated_forbidden_universe(ROOT, head, audit_raw)
    assert len(universe.reference_spans) == 414
    assert len(universe.paper_ids) == 238
    assert bindings.RECOVERED_REFERENCE_PAPER_ID in universe.paper_ids
    assert bindings.RECOVERED_REFERENCE_SHA256 in {
        sha256_bytes(value.encode("utf-8")) for value in universe.reference_spans
    }


def _committed_recovery_extension_raw() -> bytes:
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")
    decoder = importlib.import_module("redco.analysis.stage_d_v13_source_phase_a_decoder")
    return cast(
        bytes,
        decoder._git_blob_at_commit(
            ROOT,
            decoder._git_text(ROOT, "rev-parse", "--verify", "HEAD^{commit}"),
            selection.SUCCESSOR_EXTENSION_RELATIVE,
        ),
    )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mutation", ("missing", "duplicate", "altered", "extra_reference")
)
def test_gate_g_recovery_record_mutations_fail_closed(mutation: str) -> None:
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")
    records = [json.loads(line) for line in _committed_recovery_extension_raw().splitlines()]
    target_index = next(
        index
        for index, record in enumerate(records)
        if record.get("example_id") == selection.RECOVERED_REFERENCE_EXAMPLE_ID
    )
    if mutation == "missing":
        del records[target_index]
    elif mutation == "duplicate":
        records.append(copy.deepcopy(records[target_index]))
    elif mutation == "altered":
        records[target_index]["paper"] += " altered"
    else:
        records[target_index]["reference_evidence"].append("extra")
    mutated = b"\n".join(canonical_json_bytes(record) for record in records)
    with pytest.raises(selection.SelectionGateError):
        selection._locate_recovery_record(mutated)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "field_name", ("extension_sha256", "canonical_row_sha256", "rendered_paper_sha256")
)
def test_gate_g_recovery_projection_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch, field_name: str
) -> None:
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")
    decoder = importlib.import_module("redco.analysis.stage_d_v13_source_phase_a_decoder")
    bindings = importlib.import_module("redco.analysis.stage_d_v13_source_phase_a_bindings")
    head = decoder._git_text(ROOT, "rev-parse", "--verify", "HEAD^{commit}")
    audit = json.loads(
        decoder._git_blob_at_commit(ROOT, head, bindings.PHASE_A_AUDIT_RELATIVE)
    )
    projection = cast(
        dict[str, Any],
        cast(dict[str, Any], selection.PHASE_B_SOURCE_SELECTION_CONTRACT_V4["forbidden_universe"])[
            "recovery_projection"
        ],
    )
    monkeypatch.setitem(projection, field_name, "0" * 64)
    with pytest.raises(selection.SelectionGateError):
        selection._authenticated_recovery_reference(ROOT, head, audit)


def test_gate_g_recovery_wrong_address_manifest_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")
    decoder = importlib.import_module("redco.analysis.stage_d_v13_source_phase_a_decoder")
    bindings = importlib.import_module("redco.analysis.stage_d_v13_source_phase_a_bindings")
    head = decoder._git_text(ROOT, "rev-parse", "--verify", "HEAD^{commit}")
    audit = json.loads(
        decoder._git_blob_at_commit(ROOT, head, bindings.PHASE_A_AUDIT_RELATIVE)
    )
    monkeypatch.setattr(selection, "SUCCESSOR_ADDRESS_AUDIT_V1_SHA256", "0" * 64)
    with pytest.raises(selection.SelectionGateError):
        selection._authenticated_recovery_reference(ROOT, head, audit)


def test_gate_g_recovery_omission_fails_before_any_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")
    decoder = importlib.import_module("redco.analysis.stage_d_v13_source_phase_a_decoder")
    repo, _gate = _synthetic_gate_chain(tmp_path)
    monkeypatch.setattr(decoder, "PROJECT_ROOT", repo)
    events: list[str] = []

    def omitted(*_args: Any) -> Any:
        events.append("universe")
        raise selection.SelectionGateError("simulated 413/414 omission")

    monkeypatch.setattr(selection, "_authenticated_forbidden_universe", omitted)
    monkeypatch.setattr(
        selection,
        "validate_output_paths",
        lambda *_args, **_kwargs: pytest.fail("output paths validated before universe"),
    )
    monkeypatch.setattr(
        selection,
        "_create_exclusive_claim",
        lambda *_args, **_kwargs: pytest.fail("claim created before universe"),
    )
    monkeypatch.setattr(
        decoder,
        "_require_runtime_versions_only",
        lambda: pytest.fail("runtime inspected before universe"),
    )
    monkeypatch.setattr(
        decoder,
        "_validate_production_source_metadata",
        lambda *_args: pytest.fail("source inspected before universe"),
    )
    with pytest.raises(selection.SelectionGateError, match="413/414"):
        selection.activate_selection_gate()
    assert events == ["universe"]
    assert not (repo / selection.SELECTION_CLAIM_RELATIVE).exists()
    assert not (repo / selection.SELECTION_RECEIPT_RELATIVE).exists()


def test_gate_g_immutable_reference_inputs_reject_claim_and_receipt_aliases(
    tmp_path: Path,
) -> None:
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")
    bindings = importlib.import_module("redco.analysis.stage_d_v13_source_phase_a_bindings")
    forbidden = cast(
        Mapping[str, Any], bindings.PHASE_B_SOURCE_SELECTION_CONTRACT_V4["forbidden_universe"]
    )
    raw_sources = cast(Mapping[str, str], forbidden["raw_reference_source_hashes"])
    immutable = {
        *raw_sources,
        bindings.PHASE_A_AUDIT_RELATIVE,
        bindings.SUCCESSOR_EXTENSION_RELATIVE,
        bindings.SUCCESSOR_EXTENSION_MANIFEST_RELATIVE,
        bindings.SUCCESSOR_MANIFEST_RELATIVE,
        bindings.SUCCESSOR_ADDRESS_AUDIT_V1_RELATIVE,
    }
    for output in (selection.SELECTION_CLAIM_RELATIVE, selection.SELECTION_RECEIPT_RELATIVE):
        for relative in sorted(immutable):
            source = tmp_path / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"immutable input")
            destination = tmp_path / output
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            with pytest.raises(ValueError, match="hard-link alias"):
                selection.validate_output_paths(
                    tmp_path,
                    {relative: "0" * 64},
                    output_paths=(
                        selection.SELECTION_CLAIM_RELATIVE,
                        selection.SELECTION_RECEIPT_RELATIVE,
                    ),
                )
            destination.unlink()


def test_gate_g_actuator_claims_before_runtime_source_and_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = importlib.import_module("redco.analysis.stage_d_v13_source_selection")
    decoder = importlib.import_module("redco.analysis.stage_d_v13_source_phase_a_decoder")
    repo, _gate = _synthetic_gate_chain(tmp_path)
    monkeypatch.setattr(decoder, "PROJECT_ROOT", repo)
    events: list[str] = []
    original_claim = selection._create_exclusive_claim
    original_validate_paths = selection.validate_output_paths

    def claim(root: Path, payload: dict[str, Any]) -> bytes:
        events.append("claim")
        return cast(bytes, original_claim(root, payload))

    monkeypatch.setattr(selection, "_create_exclusive_claim", claim)

    def validate_paths(*args: Any, **kwargs: Any) -> None:
        events.append("paths")
        original_validate_paths(*args, **kwargs)

    monkeypatch.setattr(selection, "validate_output_paths", validate_paths)

    def runtime() -> tuple[Any, str]:
        events.append("runtime")
        return types.SimpleNamespace(__version__="25.0.0"), "5.0.0"

    def source(*_args: Any) -> object:
        events.append("source")
        return object()

    def universe(*_args: Any) -> Any:
        events.append("universe")
        return selection.SelectionUniverse(
            paper_ids=frozenset(),
            example_ids=frozenset(),
            rendered_paper_sha256=frozenset(),
            reference_spans=frozenset(),
            row_sha256=frozenset(),
            addresses=frozenset(),
        )

    def scan(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("scan")
        return {
            "disposition": "exhausted",
            "stop_ordinal": 887,
            "candidate": None,
            "transcript_sha256": "a" * 64,
        }

    monkeypatch.setattr(decoder, "_require_runtime_versions_only", runtime)
    monkeypatch.setattr(decoder, "_validate_production_source_metadata", source)
    monkeypatch.setattr(selection, "_authenticated_forbidden_universe", universe)
    monkeypatch.setattr(selection, "_scan_once", scan)
    receipt = selection.activate_selection_gate()
    assert events == ["universe", "paths", "claim", "runtime", "source", "scan"]
    assert receipt["attempt"] == 1
    assert receipt["retry"] is False
    assert receipt["candidate"] is None
    assert (repo / selection.SELECTION_CLAIM_RELATIVE).is_file()
    assert (repo / selection.SELECTION_RECEIPT_RELATIVE).is_file()
    with pytest.raises(selection.SelectionGateAlreadyClaimed):
        selection.activate_selection_gate()
    assert events == [
        "universe",
        "paths",
        "claim",
        "runtime",
        "source",
        "scan",
        "universe",
        "paths",
        "claim",
    ]


def test_gate_g_v2_v3_history_is_frozen_and_v4_is_distinct() -> None:
    from redco.analysis.stage_d_v13_source_phase_a_bindings import (
        PHASE_B_RESUME_CONTRACT_V2_SHA256,
        PHASE_B_SOURCE_SELECTION_CONTRACT_V4,
        PHASE_B_SOURCE_SELECTION_CONTRACT_V4_SHA256,
    )
    from redco.analysis.stage_d_v13_source_phase_a_decoder import (
        _resume_contract_hash,
        _resume_contract_v3_hash,
    )

    assert _resume_contract_hash() == PHASE_B_RESUME_CONTRACT_V2_SHA256
    assert PHASE_B_RESUME_CONTRACT_V2_SHA256 == (
        "cade25b90061b817423307b5e63fb6c76756ac3f5b365671572a6d16eb2e8e08"
    )
    assert _resume_contract_v3_hash() == (
        "6dbe84a853f0e995e6a6821b0340b44e0c197b7a7783af709c668d0ae96f6151"
    )
    assert _resume_contract_v3_hash() != PHASE_B_RESUME_CONTRACT_V2_SHA256
    assert sha256_json(PHASE_B_SOURCE_SELECTION_CONTRACT_V4) == (
        PHASE_B_SOURCE_SELECTION_CONTRACT_V4_SHA256
    )
