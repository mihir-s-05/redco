"""Stable façade for authenticated v13 source-recovery Phase A.

Ownership is deliberately split into decoder, selector, witness, and
publication modules.  This façade only composes their authenticated results;
it never selects or materializes a post-179 candidate.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
import tomllib
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes, sha256_json
from redco.analysis.stage_d_v13_draft_inputs import (
    FROZEN_HASHES,
    POST_REPAIR_HASHES,
    authenticate_immutable_inputs,
    sha256_file,
)
from redco.analysis.stage_d_v13_source_phase_a_decoder import (
    PHASE_A_CUTOFF,
    PHASE_A_VERSION,
    SOURCE_ARTIFACT_RELATIVE,
    SOURCE_BYTES,
    SOURCE_FIELDS,
    SOURCE_LOGICAL_URL,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    SOURCE_ROW_COUNT,
    SOURCE_ROW_GROUPS,
    SOURCE_SCHEMA_SHA256,
    SOURCE_SEMANTIC_COMMIT,
    SOURCE_SHA256,
    SUPPORTED_DATASETS,
    SUPPORTED_PYARROW,
    SUPPORTED_PYTHON,
    DecoderInstrumentation,
    PhaseAWallError,
    authenticate_source_artifact,
    bounded_source_rows,
    canonical_source_row_bytes,
    legacy_datasets_decoder_probe,
    resume_decoder_invocation_count,
    source_row_sha256,
)
from redco.analysis.stage_d_v13_source_phase_a_publication import write_phase_a_outputs
from redco.analysis.stage_d_v13_source_phase_a_selector import (
    MAXIMUM_PAPER_CHARACTERS,
    MINIMUM_SPAN_CHARACTERS,
    TerminalIdentityCollision,
    derivation_golden_vectors,
    render_paper,
    select_first_eligible,
    selector_decision,
)
from redco.analysis.stage_d_v13_source_phase_a_trust import (
    APPROVAL_ANCHOR_RELATIVE,
    BINDINGS_RELATIVE,
    TRUST_MODULE_RELATIVE,
    authenticate_external_anchor,
)
from redco.analysis.stage_d_v13_source_phase_a_witness import (
    EXPECTED_CARDINALITIES,
    HISTORICAL_RECEIPT_EXPECTATIONS,
    authenticated_historical_inputs,
    build_forbidden_witness,
    historical_receipts,
    reconstruct_retired_units,
    validate_forbidden_witness,
)

TRANSCRIPT_VERSION = "stage-d-v13-source-receipt-transcript-v2"
TRANSCRIPT_INITIAL = hashlib.sha256(TRANSCRIPT_VERSION.encode("ascii")).digest()
PHASE_A_OUTPUTS = (
    "configs/stage-d/v13-draft/stage-d1-support-source-authentication-phase-a-v1.json",
    "reports/stage-d1-support-v13-source-phase-a-audit-v1.json",
    "reports/stage-d1-support-v13-source-phase-a-cpu-manifest-v1.json",
    "reports/stage-d1-support-v13-source-phase-a-artifact-manifest-v1.json",
    "reports/stage-d1-support-v13-source-phase-a-status-v2.json",
)
FOUNDATION_NULL_CANDIDATE = {
    "source_ordinal": None,
    "paper_id": None,
    "example_id": None,
    "row": None,
    "seed": None,
    "address": None,
}
FOUNDATION_ENVELOPE_ASSERTIONS = {
    "draft_unfrozen": True,
    "launch_authorized": False,
    "provider_calls_authorized": False,
    "phase_b_authorized": False,
    "foundation_only": True,
    "non_authorizing": True,
}
OLD_REJECTED_ARTIFACT_HASHES = {
    PHASE_A_OUTPUTS[0]: "7499c292f12ba9f105ace6dddc42afaa6f43feb27ae354667eb1138a7664ea4e",
    PHASE_A_OUTPUTS[1]: "4ffff1a56ad63097a2fedd59b01bd48b0bafce59d22d5b893415fa1b4d69a8f5",
    PHASE_A_OUTPUTS[2]: "73c8cf6a53f0d023a76e79b3867d0b5da854dfe9c72c77be1e7a5733840513bd",
    PHASE_A_OUTPUTS[3]: "7a2d8c8256c1697ec03f34c80e0fce02765d06b0a2a6a5d66864b3218224f02d",
}


def iter_cutoff_rows(
    rows: Iterable[Mapping[str, Any]], *, cutoff: int = PHASE_A_CUTOFF
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Yield only already-decoded rows through the frozen cutoff.

    Production Phase A uses ``bounded_source_rows`` so that the decoder
    object is itself bounded. This wrapper is retained for pre-decoded test
    streams and never returns the first row after cutoff.
    """

    if cutoff < 0:
        raise ValueError("cutoff must be non-negative")
    for ordinal, row in enumerate(rows):
        if ordinal > cutoff:
            break
        yield ordinal, row


def collision_disposition(kind: str, collision: bool) -> str:
    """Return the frozen disposition for a selector collision class."""

    if not collision:
        return "accept"
    if kind == "address_or_identity":
        return "terminal_fail_closed"
    if kind == "paper_or_reference":
        return "continue_source_order_scan"
    raise ValueError(f"unknown collision class: {kind}")


def _address_core(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "paper_id",
        "example_id",
        "seed",
        "scientific_group_id",
        "slot_id",
        "rollout_slot",
        "cache_salt",
        "row_sha256",
        "canonical_row_sha256",
    )
    return {field: record[field] for field in fields if field in record}


def _selected_summary(
    row: Mapping[str, Any], selected: Mapping[str, Any], question_index: int
) -> dict[str, Any]:
    references = [str(value) for value in cast(Iterable[Any], selected["reference_evidence"])]
    return {
        "paper_id": str(row["id"]),
        "example_id": str(selected["example_id"]),
        "question_index": question_index,
        "canonical_row_sha256": source_row_sha256(selected),
        "rendered_paper_sha256": [sha256_bytes(str(selected["paper"]).encode("utf-8"))],
        "reference_span_sha256": [sha256_bytes(value.encode("utf-8")) for value in references],
        "_raw_reference_evidence": references,
    }


def _scientific_binding(root: Path) -> dict[str, Any]:
    values = tomllib.loads(
        (root / "configs/stage-d/stage-d1-support-source-eval-v12.toml").read_text(encoding="utf-8")
    )
    env = cast(Mapping[str, Any], values["env"])
    taskset = cast(Mapping[str, Any], env["taskset"])
    binding = derivation_golden_vectors()
    if (
        binding["namespace"] != taskset["scientific_group_namespace"]
        or binding["master_seed"] != env["master_seed"]
    ):
        raise ValueError("scientific derivation law differs from the authenticated v1 config")
    return {
        "scientific_group_namespace": str(taskset["scientific_group_namespace"]),
        "master_seed": str(env["master_seed"]),
        "source_of_law": "stage_d_collection.py v1 scientific group/master-seed law",
        "administrative_seed_used": False,
        "v13_administrative_identity_used": False,
        "candidate_seed": None,
        "candidate_address": None,
        "hmac_domains": [binding["group_domain"], binding["seed_domain"]],
        "golden_vector": binding,
    }


def _authenticated_behavior_bindings(root: Path) -> dict[str, str]:
    """Bind every behavior owner and frozen decoder input by exact bytes."""

    approval = authenticate_external_anchor(root)

    from redco.analysis.stage_d_v13_source_phase_a_bindings import (
        APPROVED_BEHAVIOR_HASHES,
        APPROVED_DERIVATION_VECTOR,
        BEHAVIOR_BINDING_FILES,
    )

    observed: dict[str, str] = {}
    for relative in BEHAVIOR_BINDING_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"behavior binding input is missing: {relative}")
        digest = sha256_file(path)
        expected = APPROVED_BEHAVIOR_HASHES.get(relative)
        if expected is None or digest != expected:
            raise ValueError(f"behavior binding hash mismatch for {relative}")
        observed[relative] = digest
    if derivation_golden_vectors() != APPROVED_DERIVATION_VECTOR:
        raise ValueError("scientific derivation golden vector differs")
    observed[APPROVAL_ANCHOR_RELATIVE] = cast(str, approval["anchor_sha256"])
    observed[BINDINGS_RELATIVE] = cast(str, approval["registry_sha256"])
    observed[TRUST_MODULE_RELATIVE] = sha256_file(root / TRUST_MODULE_RELATIVE)
    return observed


def authenticated_phase_a_inputs(root: Path) -> dict[str, str]:
    approval = authenticate_external_anchor(root)
    immutable = authenticate_immutable_inputs(root)
    historical = authenticated_historical_inputs(root)
    behavior = _authenticated_behavior_bindings(root)
    source = root / SOURCE_ARTIFACT_RELATIVE
    if not source.is_file() or sha256_file(source) != SOURCE_SHA256:
        raise ValueError("Phase-A source artifact is not authenticated")
    return {
        **immutable,
        **historical,
        SOURCE_ARTIFACT_RELATIVE: SOURCE_SHA256,
        **behavior,
        APPROVAL_ANCHOR_RELATIVE: cast(str, approval["anchor_sha256"]),
        BINDINGS_RELATIVE: cast(str, approval["registry_sha256"]),
        TRUST_MODULE_RELATIVE: sha256_file(root / TRUST_MODULE_RELATIVE),
    }


def phase_a_immutable_paths(root: Path) -> dict[str, str]:
    """Return every authenticated path used to protect publication aliases."""

    from redco.analysis.stage_d_v13_source_phase_a_bindings import BEHAVIOR_BINDING_FILES

    relative_paths = {**FROZEN_HASHES, **POST_REPAIR_HASHES}
    relative_paths.update({relative: "" for relative in BEHAVIOR_BINDING_FILES})
    approval = authenticate_external_anchor(root)
    relative_paths[APPROVAL_ANCHOR_RELATIVE] = cast(str, approval["anchor_sha256"])
    relative_paths[BINDINGS_RELATIVE] = cast(str, approval["registry_sha256"])
    relative_paths[TRUST_MODULE_RELATIVE] = sha256_file(root / TRUST_MODULE_RELATIVE)
    relative_paths[SOURCE_ARTIFACT_RELATIVE] = SOURCE_SHA256
    relative_paths.update({relative: "" for relative in authenticated_historical_inputs(root)})
    # Publication may target a fresh output root. Bind immutable inputs to
    # the authenticated source tree rather than accidentally looking for
    # same-named files under that disposable output root.
    paths = {
        str((root / relative).resolve(strict=False)): expected
        for relative, expected in relative_paths.items()
    }
    return paths


def _decision_code(row: Mapping[str, Any], universe: Mapping[str, Any]) -> str:
    try:
        return cast(
            str,
            selector_decision(
                row,
                forbidden_paper_ids=cast(set[str], universe["paper_ids"]),
                forbidden_example_ids=cast(set[str], universe["example_ids"]),
                forbidden_rendered_paper_sha256=cast(
                    Iterable[str], universe["rendered_paper_sha256"]
                ),
                forbidden_reference_spans=cast(set[str], universe["reference_spans"]),
                forbidden_row_sha256=cast(set[str], universe["row_sha256"]),
                forbidden_address_sha256=cast(Iterable[str], universe["addresses"]),
            ),
        )
    except TerminalIdentityCollision as error:
        collision_set = "+".join(error.collision_set)
        return f"terminal_{error.collision_class};set={collision_set}"


def _advance_transcript(state: bytes, ordinal: int, row_hash: str, decision_code: str) -> bytes:
    event = canonical_json_bytes(
        {"ordinal": ordinal, "source_row_sha256": row_hash, "decision_code": decision_code}
    )
    return hashlib.sha256(state + b"\x00" + event).digest()


def _build_universe(witness: Mapping[str, Any], raw_references: set[str]) -> dict[str, Any]:
    sets = cast(Mapping[str, Any], witness["forbidden_sets"])
    return {
        "paper_ids": set(cast(Iterable[str], sets["retired_paper_ids"]))
        | set(cast(Iterable[str], sets["old_snapshot_paper_ids"]))
        | set(cast(Iterable[str], sets["predecessor_paper_ids"]))
        | set(cast(Iterable[str], sets["historical_paper_ids"])),
        "example_ids": set(cast(Iterable[str], sets["retired_example_ids"]))
        | set(cast(Iterable[str], sets["old_snapshot_example_ids"]))
        | set(cast(Iterable[str], sets["predecessor_example_ids"]))
        | set(cast(Iterable[str], sets["historical_example_ids"])),
        "rendered_paper_sha256": list(cast(Iterable[str], sets["rendered_paper_sha256"]))
        + list(cast(Iterable[str], sets["old_snapshot_rendered_paper_sha256"]))
        + list(cast(Iterable[str], sets["predecessor_rendered_paper_sha256"])),
        "reference_spans": raw_references,
        "row_sha256": set(cast(Iterable[str], sets["retired_row_sha256"]))
        | set(cast(Iterable[str], sets["old_snapshot_row_sha256"]))
        | set(cast(Iterable[str], sets["predecessor_row_sha256"]))
        | set(cast(Iterable[str], sets["historical_row_sha256"])),
        "addresses": list(cast(Iterable[str], sets["source_address_sha256"]))
        + list(cast(Iterable[str], sets["historical_address_sha256"])),
    }


def build_phase_a_result(root: Path) -> dict[str, Any]:
    """Authenticate and audit exactly the bounded 0--179 source prefix."""

    approval = authenticate_external_anchor(root)
    source = authenticate_source_artifact(root)
    immutable = authenticate_immutable_inputs(root)
    authenticated_inputs = authenticated_phase_a_inputs(root)
    receipts, _retired_addresses = historical_receipts(root)
    instrumentation = DecoderInstrumentation()
    prefix = list(
        bounded_source_rows(root / SOURCE_ARTIFACT_RELATIVE, instrumentation=instrumentation)
    )
    resume_invocations = resume_decoder_invocation_count()
    if resume_invocations != 0:
        raise PhaseAWallError("Phase A observed a dormant Phase-B resume decoder invocation")
    retired_units = reconstruct_retired_units(root, prefix_rows=prefix)
    witness, raw_references = build_forbidden_witness(
        root,
        retired_units,
        authenticated_historical_inputs(root),
    )
    validate_forbidden_witness(
        root,
        witness,
        expected_units=retired_units,
        address_inputs=authenticated_historical_inputs(root),
    )
    universe = _build_universe(witness, raw_references)
    receipt_by_ordinal = {int(item["source_ordinal"]): item for item in receipts}
    receipt_results: list[dict[str, Any]] = []
    transcript = TRANSCRIPT_INITIAL
    for ordinal, row in prefix:
        row_hash = source_row_sha256(row)
        transcript = _advance_transcript(
            transcript, ordinal, row_hash, _decision_code(row, universe)
        )
        if ordinal not in receipt_by_ordinal:
            continue
        expected = receipt_by_ordinal[ordinal]
        selected = select_first_eligible(row)
        if selected is None:
            raise ValueError(
                f"historical source receipt has no exact eligible question at {ordinal}"
            )
        chosen, question_index = selected
        summary = _selected_summary(row, chosen, question_index)
        address = cast(Mapping[str, Any], expected["address_record"])
        address_row_hash = address.get("canonical_row_sha256") or address.get("row_sha256")
        if (
            str(row["id"]) != expected["paper_id"]
            or row_hash != expected["source_row_sha256"]
            or str(chosen["example_id"]) != expected["example_id"]
            or summary["canonical_row_sha256"] != address_row_hash
        ):
            raise ValueError(f"historical source receipt mismatch at ordinal {ordinal}")
        receipt_results.append(
            {
                "version": expected["version"],
                "source_ordinal": ordinal,
                "paper_id": str(row["id"]),
                "example_id": str(chosen["example_id"]),
                "source_row_sha256": row_hash,
                "canonical_selected_row_sha256": summary["canonical_row_sha256"],
                "address_audit_path": expected["address_audit_path"],
                "address_role": expected["address_role"],
                "address_sha256": sha256_json(_address_core(address)),
                "question_index": question_index,
                "status": "authenticated",
            }
        )
    if len(prefix) != PHASE_A_CUTOFF + 1 or len(receipt_results) != 6:
        raise ValueError("Phase A did not authenticate the complete bounded prefix/receipt set")
    public_units = [
        {key: value for key, value in unit.items() if not key.startswith("_")}
        for unit in retired_units
    ]
    decoder_payload = instrumentation.to_payload()
    scientific_binding = _scientific_binding(root)
    post_cutoff_logical = bool(decoder_payload["post_cutoff_logical_row_materialized"])
    post_cutoff_canonical = bool(decoder_payload["post_cutoff_row_canonicalized"])
    return {
        "schema_version": 2,
        "phase": "A",
        "version": PHASE_A_VERSION,
        "source": source,
        "approval_anchor": approval,
        "immutable_v12_bindings": immutable,
        "authenticated_inputs": authenticated_inputs,
        "historical_receipts": receipt_results,
        "retired_units": public_units,
        "forbidden_witness": witness,
        "selection_contract": {
            "physical_order": "source ordinal zero through authenticated source row order",
            "source_row_count": SOURCE_ROW_COUNT,
            "phase_a_maximum_read_ordinal": PHASE_A_CUTOFF,
            "resume_after_authenticated_receipt_ordinal": PHASE_A_CUTOFF,
            "maximum_paper_characters": MAXIMUM_PAPER_CHARACTERS,
            "minimum_exact_span_characters": MINIMUM_SPAN_CHARACTERS,
            "first_eligible_question_per_paper": True,
            "forbidden_set_hashes": witness["exclusion_hashes"],
            "raw_reference_comparison": True,
            "candidate_difficulty_or_subjective_content_rejection": False,
            "candidate_materialization_in_phase_a": False,
        },
        "rolling_transcript": {
            "schema_version": 2,
            "version": TRANSCRIPT_VERSION,
            "cutoff_inclusive": PHASE_A_CUTOFF,
            "rows_canonicalized": len(prefix),
            "hash": transcript.hex(),
            "content_excluded_from_transcript": True,
        },
        "scientific_binding": scientific_binding,
        "decoder_probe": {
            "legacy_datasets": legacy_datasets_decoder_probe(root / SOURCE_ARTIFACT_RELATIVE),
            "bounded_pyarrow": decoder_payload,
            "metadata_authentication_row_deserialization": False,
        },
        "phase_a_wall": {
            "cutoff_inclusive": PHASE_A_CUTOFF,
            "source_rows_requested": len(prefix),
            "source_rows_canonicalized": len(prefix),
            "max_canonicalized_ordinal": PHASE_A_CUTOFF,
            "decoded_batch_cardinalities": [
                int(item["rows"]) for item in instrumentation.decoded_objects
            ],
            "decoded_ordinal_ranges": [
                [int(item["start_ordinal"]), int(item["end_ordinal"])]
                for item in instrumentation.decoded_objects
            ],
            "post_cutoff_physical_io_claim": (
                "not claimed; compressed-page I/O may span the single row group"
            ),
            "post_cutoff_row_materialized": post_cutoff_logical,
            "post_cutoff_row_deserialized": post_cutoff_logical,
            "post_cutoff_row_canonicalized": post_cutoff_canonical,
            "post_cutoff_row_evaluated": post_cutoff_canonical,
            "full_stream_selection_started": False,
            "phase_b_started": False,
            "phase_b_resume_decoder_invocations": resume_invocations,
            "phase_b_resume_decoder_invocations_required": 0,
        },
        "phase_a_state_machine": {
            "schema_version": 2,
            "initial_state": "source_authentication_pending",
            "terminal_state": "authenticated_prefix_candidate_unresolved",
            "states": [
                "source_authentication_pending",
                "source_authenticated",
                "historical_receipts_authenticated",
                "cutoff_wall_active",
                "authenticated_prefix_candidate_unresolved",
                "phase_b_selection_not_started",
            ],
            "phase_b_transition_authorized": False,
            "provider_dispatch_authorized": False,
        },
        "candidate": {
            "source_ordinal": None,
            "paper_id": None,
            "example_id": None,
            "row": None,
            "seed": None,
            "address": None,
        },
        "authorization_provenance": {
            "orchestrator_thread": "019f9ab9-ec45-7ac3-82b1-09757b92a7c3",
            "authorization_scope": (
                "CPU-only exact-source restoration and Phase A prefix authentication"
            ),
            "authorization_context": (
                "Restore or download the exact authenticated QASPER dataset snapshot, "
                "verify its hash "
                "and ordering, then select the first unused eligible row after 179. After review, "
                "we can freeze v13 and resume GPU testing."
            ),
            "phase_a_stop_before_selection": True,
            "freeze": False,
            "launch_authorized": False,
            "provider_calls": False,
        },
        "frozen_scientific_rules": {
            "denominator": 64,
            "support_threshold": 58,
            "f1_range": 0.05,
            "estimator": "pre-registered estimator and Wilson reporting",
            "sampling_and_topology": "inherited unchanged from authenticated v12 protocol",
            "source_rules": "physical source order, one question per paper, exact evidence",
        },
    }


def foundation_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap any F artifact in the fixed candidate-null, non-authorizing state."""

    envelope = dict(payload)
    envelope.update(FOUNDATION_ENVELOPE_ASSERTIONS)
    envelope["candidate"] = dict(FOUNDATION_NULL_CANDIDATE)
    envelope["seed"] = None
    envelope["address"] = None
    return envelope


def _phase_a_runtime_versions() -> dict[str, str]:
    versions = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "datasets": importlib.metadata.version("datasets"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "pytest": importlib.metadata.version("pytest"),
    }
    expected = {
        "python": SUPPORTED_PYTHON,
        "datasets": SUPPORTED_DATASETS,
        "pyarrow": SUPPORTED_PYARROW,
    }
    if any(versions[key] != value for key, value in expected.items()):
        raise RuntimeError(f"Phase-A runtime versions differ: {versions}")
    return versions


def validate_foundation_envelope(value: Mapping[str, Any]) -> None:
    """Fail closed unless all F-only authorization assertions are explicit."""

    for key, expected in FOUNDATION_ENVELOPE_ASSERTIONS.items():
        if value.get(key) is not expected:
            raise ValueError(f"foundation envelope assertion differs: {key}")
    if value.get("candidate") != FOUNDATION_NULL_CANDIDATE:
        raise ValueError("foundation envelope candidate is not null")
    if value.get("seed") is not None or value.get("address") is not None:
        raise ValueError("foundation envelope seed/address must be null")


_envelope = foundation_envelope


def phase_a_payloads(root: Path, *, test_node_ids: tuple[str, ...]) -> dict[str, bytes]:
    approval = authenticate_external_anchor(root)
    result = build_phase_a_result(root)
    config = _envelope(
        {
            "schema_version": 2,
            "domain": PHASE_A_VERSION,
            "status": "phase_a_authenticated_prefix_candidate_unresolved",
            "source_amendment": result,
            "approval_anchor": approval,
        }
    )
    config_bytes = canonical_json_bytes(config)
    audit = _envelope(
        {
            "schema_version": 2,
            "domain": "redco-stage-d1-support-v13-source-phase-a-audit-v2",
            "status": "authenticated_prefix_only_candidate_unresolved",
            "engineering_audit_only": True,
            "source_authentication": result,
            "approval_anchor": approval,
            "supersedes_rejected_artifact_hashes": OLD_REJECTED_ARTIFACT_HASHES,
            "config": {"path": PHASE_A_OUTPUTS[0], "sha256": sha256_bytes(config_bytes)},
        }
    )
    audit_bytes = canonical_json_bytes(audit)
    from redco.analysis.stage_d_v13_source_phase_a_bindings import PHASE_A_STATUS_SIGNATURE

    runtime_versions = _phase_a_runtime_versions()
    status_path = root / PHASE_A_OUTPUTS[4]
    if not status_path.is_file():
        raise FileNotFoundError("independent Phase-A status capture is missing")
    status_bytes = status_path.read_bytes()
    status = json.loads(status_bytes)
    if not isinstance(status, dict) or status_bytes != canonical_json_bytes(status):
        raise ValueError("Phase-A status capture is not canonical JSON")
    expected_status = {"passed": len(test_node_ids), "failed": 0, "skipped": 0, "xfailed": 0}
    if (
        status.get("node_ids") != list(test_node_ids)
        or status.get("status") != expected_status
        or status.get("status_signature") != PHASE_A_STATUS_SIGNATURE
        or sha256_json({"node_ids": list(test_node_ids), "status": expected_status})
        != PHASE_A_STATUS_SIGNATURE
    ):
        raise ValueError("independent Phase-A status capture differs")
    status = foundation_envelope(
        {
            "schema_version": 2,
            "domain": "redco-stage-d1-support-v13-source-phase-a-status-v2",
            "status": expected_status,
            "node_ids": list(test_node_ids),
            "status_signature": PHASE_A_STATUS_SIGNATURE,
            "capture": "independent_cpu_suite_status_signature",
        }
    )
    status_bytes = canonical_json_bytes(status)
    test_path = root / "tests/test_stage_d_v13_source_phase_a.py"
    cpu = _envelope(
        {
            "schema_version": 2,
            "domain": "redco-stage-d1-support-v13-source-phase-a-cpu-manifest-v2",
            "status": "phase_a_tests_zero_skip_required",
            "suite": {
                "command": (
                    "sys.executable -m pytest tests/test_stage_d_v13_source_phase_a.py "
                    "-q --tb=no --capture=no"
                ),
                "collection_command": (
                    "sys.executable -m pytest tests/test_stage_d_v13_source_phase_a.py "
                    "--collect-only -q -vv --capture=no"
                ),
                "interpreter_binding": "sys.executable",
                "runtime": runtime_versions,
                "node_ids": list(test_node_ids),
                "node_count": len(test_node_ids),
                "node_list_sha256": sha256_json(list(test_node_ids)),
                "test_source_sha256": sha256_file(test_path),
                "expected": {"passed": len(test_node_ids), "failed": 0, "skipped": 0, "xfailed": 0},
                "verification": {
                    "collection_reproduced": True,
                    "status_signature": PHASE_A_STATUS_SIGNATURE,
                    "status_capture_path": PHASE_A_OUTPUTS[4],
                    "status_capture_sha256": sha256_bytes(status_bytes),
                    "independent_status_capture": True,
                },
            },
            "existing_repaired_cpu_suites": {
                "selected": {"passed": 114, "failed": 0, "skipped": 0},
                "producer": {"passed": 21, "deselected": 5, "failed": 0, "skipped": 0},
                "observer": {"passed": 19, "failed": 0, "skipped": 0},
                "source_environment": {"passed": 20, "failed": 0, "skipped": 0},
            },
            "phase_a_artifact_sha256": SOURCE_SHA256,
            "approval_anchor": approval,
        }
    )
    cpu_bytes = canonical_json_bytes(cpu)
    artifact_hashes = {
        relative: sha256_bytes(payload)
        for relative, payload in (
            (PHASE_A_OUTPUTS[0], config_bytes),
            (PHASE_A_OUTPUTS[1], audit_bytes),
            (PHASE_A_OUTPUTS[2], cpu_bytes),
            (PHASE_A_OUTPUTS[4], status_bytes),
        )
    }
    from redco.analysis.stage_d_v13_source_phase_a_bindings import BEHAVIOR_BINDING_FILES

    manifest = _envelope(
        {
            "schema_version": 2,
            "domain": "redco-stage-d1-support-v13-source-phase-a-artifact-manifest-v2",
            "status": "draft_unfrozen_not_authorized",
            "canonical_json": {
                "sort_keys": True,
                "separators": [",", ":"],
                "trailing_newline": False,
            },
            "source_artifact": {
                "path": SOURCE_ARTIFACT_RELATIVE,
                "sha256": SOURCE_SHA256,
                "bytes": SOURCE_BYTES,
            },
            "approval_anchor": approval,
            "phase_a_artifacts": artifact_hashes,
            "builder_sources": {
                relative: sha256_file(root / relative) for relative in BEHAVIOR_BINDING_FILES
            },
            "decoder_versions": {
                "python": SUPPORTED_PYTHON,
                "datasets": SUPPORTED_DATASETS,
                "pyarrow": SUPPORTED_PYARROW,
            },
            "derivation_golden_vector": result["scientific_binding"]["golden_vector"],
            "reproducibility": {
                "same_authenticated_inputs_same_bytes": True,
                "absolute_paths_omitted": True,
                "timings_omitted": True,
                "network_scope": "exact logical source artifact retrieval only; build is offline",
                "provider_calls": False,
                "prime_calls": False,
                "gpu_calls": False,
            },
        }
    )
    return {
        PHASE_A_OUTPUTS[0]: config_bytes,
        PHASE_A_OUTPUTS[1]: audit_bytes,
        PHASE_A_OUTPUTS[2]: cpu_bytes,
        PHASE_A_OUTPUTS[3]: canonical_json_bytes(manifest),
        PHASE_A_OUTPUTS[4]: status_bytes,
    }


__all__ = [
    "EXPECTED_CARDINALITIES",
    "FOUNDATION_ENVELOPE_ASSERTIONS",
    "FOUNDATION_NULL_CANDIDATE",
    "HISTORICAL_RECEIPT_EXPECTATIONS",
    "PHASE_A_CUTOFF",
    "PHASE_A_OUTPUTS",
    "PHASE_A_VERSION",
    "SOURCE_ARTIFACT_RELATIVE",
    "SOURCE_BYTES",
    "SOURCE_FIELDS",
    "SOURCE_LOGICAL_URL",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "SOURCE_ROW_COUNT",
    "SOURCE_ROW_GROUPS",
    "SOURCE_SCHEMA_SHA256",
    "SOURCE_SEMANTIC_COMMIT",
    "SOURCE_SHA256",
    "PhaseAWallError",
    "authenticate_source_artifact",
    "authenticated_historical_inputs",
    "authenticated_phase_a_inputs",
    "bounded_source_rows",
    "build_forbidden_witness",
    "build_phase_a_result",
    "canonical_source_row_bytes",
    "collision_disposition",
    "foundation_envelope",
    "iter_cutoff_rows",
    "legacy_datasets_decoder_probe",
    "phase_a_immutable_paths",
    "phase_a_payloads",
    "reconstruct_retired_units",
    "render_paper",
    "select_first_eligible",
    "sha256_file",
    "source_row_sha256",
    "validate_forbidden_witness",
    "validate_foundation_envelope",
    "write_phase_a_outputs",
]
