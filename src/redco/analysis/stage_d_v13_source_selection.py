"""One-attempt, first-disposition Stage-D source-selection actuator.

Gate G owns the production activation surface after Repair R.  It authenticates
the exact future C3-v2 chain, claims the attempt before runtime or source
access, and then consumes one source-order cursor until the first disposition.
No caller-provided path, commit, universe, or authority is accepted.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from redco.analysis import stage_d_v13_source_phase_a_decoder as decoder
from redco.analysis.stage_d_collection import (
    SourceCollectionSlot,
    derive_scientific_group_id,
    derive_source_episode_seed_and_salt,
)
from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes, sha256_json
from redco.analysis.stage_d_v13_draft_inputs import (
    FROZEN_HASHES,
    HISTORICAL_ADDRESS_HASHES,
    HISTORICAL_ROLLOUT_HASHES,
    historical_identity_witness,
)
from redco.analysis.stage_d_v13_source_phase_a_bindings import (
    APPROVED_DERIVATION_VECTOR,
    PHASE_A_AUDIT_RELATIVE,
    PHASE_A_AUDIT_SHA256,
    PHASE_A_STATUS_SIGNATURE,
    PHASE_A_WITNESS_SHA256,
    PHASE_B_BINDING_RELATIVE,
    PHASE_B_RESUME_CONTRACT_V2_SHA256,
    PHASE_B_SOURCE_SELECTION_CONTRACT_V4,
    PHASE_B_SOURCE_SELECTION_CONTRACT_V4_SHA256,
    PHASE_C3_V2_AUTHORIZATION_DOMAIN,
    PHASE_C3_V2_AUTHORIZATION_RELATIVE,
    RECOVERED_REFERENCE_CANONICAL_ROW_SHA256,
    RECOVERED_REFERENCE_CARDINALITY,
    RECOVERED_REFERENCE_EXAMPLE_ID,
    RECOVERED_REFERENCE_EXPECTED_DIGEST_COUNT,
    RECOVERED_REFERENCE_PAPER_ID,
    RECOVERED_REFERENCE_QUESTION_INDEX,
    RECOVERED_REFERENCE_RENDERED_PAPER_SHA256,
    RECOVERED_REFERENCE_SHA256,
    RECOVERED_REFERENCE_SOURCE_ORDINAL,
    REPAIR_R_COMMIT,
    SELECTION_CLAIM_RELATIVE,
    SELECTION_GATE_APPROVAL_TEXT_SHA256,
    SELECTION_GATE_APPROVAL_THREAD_ID,
    SELECTION_RECEIPT_RELATIVE,
    SUCCESSOR_ADDRESS_AUDIT_V1_GIT_BLOB_SHA1,
    SUCCESSOR_ADDRESS_AUDIT_V1_RELATIVE,
    SUCCESSOR_ADDRESS_AUDIT_V1_SHA256,
    SUCCESSOR_EXTENSION_GIT_BLOB_SHA1,
    SUCCESSOR_EXTENSION_INTRODUCED_COMMIT,
    SUCCESSOR_EXTENSION_MANIFEST_GIT_BLOB_SHA1,
    SUCCESSOR_EXTENSION_MANIFEST_RELATIVE,
    SUCCESSOR_EXTENSION_MANIFEST_SHA256,
    SUCCESSOR_EXTENSION_RELATIVE,
    SUCCESSOR_EXTENSION_SHA256,
    SUCCESSOR_MANIFEST_GIT_BLOB_SHA1,
    SUCCESSOR_MANIFEST_RELATIVE,
    SUCCESSOR_MANIFEST_SHA256,
)
from redco.analysis.stage_d_v13_source_phase_a_bindings import (
    PHASE_C3_AUTHORIZATION_RELATIVE as LEGACY_C3_AUTHORIZATION_RELATIVE,
)
from redco.analysis.stage_d_v13_source_phase_a_publication import (
    atomic_write,
    validate_output_paths,
)
from redco.analysis.stage_d_v13_source_phase_a_selector import (
    TerminalIdentityCollision,
    classify_candidate_collisions,
    select_first_eligible,
)
from redco.analysis.stage_d_v13_source_phase_a_witness import EXPECTED_CARDINALITIES

LEGACY_C_AUTHORIZATION_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-v13-phase-b-authorization-c-v1.json"
)

G_SOURCE_SELECTION_RELATIVE = "src/redco/analysis/stage_d_v13_source_selection.py"
G_DIFF_ALLOWLIST = (
    *decoder.REPAIR_DIFF_ALLOWLIST,
    G_SOURCE_SELECTION_RELATIVE,
)
_TRANSCRIPT_VERSION = "stage-d-v13-source-selection-transcript-v4"
_FULL_COMMIT = decoder._FULL_COMMIT_SHA


def _advance_transcript(
    state: bytes, ordinal: int, row_hash: str, decision_code: str
) -> bytes:
    event = canonical_json_bytes(
        {
            "ordinal": ordinal,
            "source_row_sha256": row_hash,
            "decision_code": decision_code,
        }
    )
    return hashlib.sha256(state + b"\x00" + event).digest()


class SelectionGateError(decoder.PhaseBResumeAuthorizationError):
    """A fail-closed Gate-G validation or execution error."""


class SelectionGateAlreadyClaimed(SelectionGateError):
    """The durable claim proves that the sole attempt was already consumed."""


@dataclass(frozen=True, slots=True)
class SelectionUniverse:
    paper_ids: frozenset[str]
    example_ids: frozenset[str]
    rendered_paper_sha256: frozenset[str]
    reference_spans: frozenset[str]
    row_sha256: frozenset[str]
    addresses: frozenset[str]


@dataclass(slots=True)
class SelectionScanInstrumentation:
    """Bounded logical observations for synthetic cursor tests."""

    requested_start_ordinal: int = decoder.PHASE_B_RESUME_START_ORDINAL
    batches: list[dict[str, int]] | None = None
    materialized_ordinals: list[int] | None = None
    evaluated_ordinals: list[int] | None = None

    def __post_init__(self) -> None:
        self.batches = [] if self.batches is None else self.batches
        self.materialized_ordinals = (
            [] if self.materialized_ordinals is None else self.materialized_ordinals
        )
        self.evaluated_ordinals = (
            [] if self.evaluated_ordinals is None else self.evaluated_ordinals
        )


def compute_scan_id(gate_commit: str) -> str:
    """Derive the outcome-independent scan identity from the frozen inputs."""

    if _FULL_COMMIT.fullmatch(gate_commit) is None:
        raise SelectionGateError("Gate G commit is not a full Git SHA")
    payload = {
        "domain": PHASE_C3_V2_AUTHORIZATION_DOMAIN,
        "gate_commit": gate_commit,
        "v4_contract_sha256": PHASE_B_SOURCE_SELECTION_CONTRACT_V4_SHA256,
        "approval_text_sha256": SELECTION_GATE_APPROVAL_TEXT_SHA256,
    }
    return f"stage-d-source-selection-{sha256_json(payload)}"


def _parse_canonical(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SelectionGateError(f"{label} is not JSON") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise SelectionGateError(f"{label} is not canonical JSON")
    return value


def _parse_authenticated_json(raw: bytes, label: str) -> dict[str, Any]:
    """Parse an exact-hash historical JSON input without rewriting its bytes."""

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SelectionGateError(f"{label} is not JSON") from error
    if not isinstance(value, dict):
        raise SelectionGateError(f"{label} is not an object")
    return value


def _require_exact_mapping(
    value: object, expected_keys: set[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise SelectionGateError(f"{label} schema is not exact")
    return cast(dict[str, Any], value)


def _expected_source(base_inputs: Mapping[str, Any]) -> dict[str, Any]:
    source = base_inputs.get("source_contract")
    if not isinstance(source, dict):
        raise SelectionGateError("Foundation source contract is unavailable")
    required = {"path", "revision", "sha256", "schema_sha256", "row_count"}
    if set(source) != required:
        raise SelectionGateError("Foundation source contract schema differs")
    return dict(source)


def _validate_c3_v2(repo_root: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Authenticate future F->B->R->G->C3-v2 without source access."""

    production = repo_root.resolve() == decoder._DEFAULT_PROJECT_ROOT
    head = decoder._git_text(repo_root, "rev-parse", "--verify", "HEAD^{commit}")
    if _FULL_COMMIT.fullmatch(head) is None:
        raise SelectionGateError("current HEAD is not a full Git commit SHA")
    forbidden_paths = (
        LEGACY_C_AUTHORIZATION_RELATIVE,
        LEGACY_C3_AUTHORIZATION_RELATIVE,
    )
    for relative in forbidden_paths:
        if (repo_root / relative).is_file() or decoder._git_path_exists_at_commit(
            repo_root, head, relative
        ):
            raise SelectionGateError("legacy C/C3-v1 authorization path is forbidden")
    c3_path = repo_root / PHASE_C3_V2_AUTHORIZATION_RELATIVE
    if not c3_path.is_file() or not decoder._git_path_exists_at_commit(
        repo_root, head, PHASE_C3_V2_AUTHORIZATION_RELATIVE
    ):
        raise SelectionGateError("Authorization C3-v2 is absent from exact HEAD")
    c3_parents = decoder._commit_parents(repo_root, head)
    if len(c3_parents) != 1:
        raise SelectionGateError("C3-v2 must have exactly one direct parent G")
    gate_commit = c3_parents[0]
    gate_parents = decoder._commit_parents(repo_root, gate_commit)
    if len(gate_parents) != 1:
        raise SelectionGateError("Gate G must have exactly one direct parent R")
    repair_commit = gate_parents[0]
    repair_parents = decoder._commit_parents(repo_root, repair_commit)
    if len(repair_parents) != 1:
        raise SelectionGateError("Repair R must have exactly one direct parent B")
    binding_commit = repair_parents[0]
    binding_parents = decoder._commit_parents(repo_root, binding_commit)
    if len(binding_parents) != 1:
        raise SelectionGateError("Binding B must have exactly one direct parent F")
    foundation_commit = binding_parents[0]
    foundation_parents = decoder._commit_parents(repo_root, foundation_commit)
    if len(foundation_parents) != 1:
        raise SelectionGateError("Foundation F must have exactly one direct parent")
    if production and (
        repair_commit != REPAIR_R_COMMIT
        or binding_commit != decoder.BINDING_B_COMMIT
        or foundation_commit != decoder.FOUNDATION_F_COMMIT
        or foundation_parents[0] != decoder.FOUNDATION_F_PARENT_COMMIT
    ):
        raise SelectionGateError("Gate G ancestry is not the approved F-B-R chain")

    expected_f_to_b = [f"A\t{PHASE_B_BINDING_RELATIVE}"]
    expected_b_to_r = sorted(f"M\t{path}" for path in decoder.REPAIR_DIFF_ALLOWLIST)
    expected_r_to_g = sorted(
        [f"M\t{path}" for path in decoder.REPAIR_DIFF_ALLOWLIST]
        + [f"A\t{G_SOURCE_SELECTION_RELATIVE}"]
    )
    expected_g_to_c3 = [f"A\t{PHASE_C3_V2_AUTHORIZATION_RELATIVE}"]
    if sorted(decoder._diff_paths(repo_root, foundation_commit, binding_commit)) != (
        expected_f_to_b
    ):
        raise SelectionGateError("F to B differs outside the Binding B artifact")
    if sorted(decoder._diff_paths(repo_root, binding_commit, repair_commit)) != (
        expected_b_to_r
    ):
        raise SelectionGateError("B to R differs outside the Repair R allowlist")
    if sorted(decoder._diff_paths(repo_root, repair_commit, gate_commit)) != (
        expected_r_to_g
    ):
        raise SelectionGateError("R to G differs outside the four-file Gate G allowlist")
    if sorted(decoder._diff_paths(repo_root, gate_commit, head)) != expected_g_to_c3:
        raise SelectionGateError("G to C3-v2 differs outside the one authorization artifact")

    b_raw = decoder._git_blob_at_commit(repo_root, binding_commit, PHASE_B_BINDING_RELATIVE)
    b, base_inputs = decoder._validate_b_checkpoint(
        repo_root,
        foundation_commit,
        binding_commit,
        b_raw,
        production=production,
    )
    c3_raw = decoder._git_blob_at_commit(repo_root, head, PHASE_C3_V2_AUTHORIZATION_RELATIVE)
    if c3_path.read_bytes() != c3_raw:
        raise SelectionGateError("C3-v2 worktree bytes differ from Git")
    parsed = _parse_canonical(c3_raw, "Authorization C3-v2")
    expected_keys = {
        "schema_version",
        "domain",
        "state",
        "draft_unfrozen",
        "candidate",
        "seed",
        "address",
        "phase_b_authorized",
        "phase_b_source_selection_authorized",
        "source_selection_authorized",
        "launch_authorized",
        "provider_calls_authorized",
        "model_calls_authorized",
        "prime_gpu_scientific_launch_authorized",
        "science_authorized",
        "status_signature",
        "foundation_commit",
        "foundation_tree_sha1",
        "binding_commit",
        "repair_commit",
        "gate_commit",
        "binding_artifact",
        "source",
        "contracts",
        "runtime_versions",
        "approval",
        "scan",
        "paths",
        "forbidden_universe",
    }
    if set(parsed) != expected_keys:
        raise SelectionGateError("Authorization C3-v2 has unexpected or missing fields")
    if (
        parsed["schema_version"] != 2
        or parsed["domain"] != PHASE_C3_V2_AUTHORIZATION_DOMAIN
        or parsed["state"] != "C3-v2"
        or parsed["draft_unfrozen"] is not False
        or parsed["phase_b_authorized"] is not False
        or parsed["phase_b_source_selection_authorized"] is not True
        or parsed["source_selection_authorized"] is not False
        or parsed["launch_authorized"] is not False
        or parsed["provider_calls_authorized"] is not False
        or parsed["model_calls_authorized"] is not False
        or parsed["prime_gpu_scientific_launch_authorized"] is not False
        or parsed["science_authorized"] is not False
        or parsed["status_signature"] != PHASE_A_STATUS_SIGNATURE
        or parsed["foundation_commit"] != foundation_commit
        or parsed["binding_commit"] != binding_commit
        or parsed["repair_commit"] != repair_commit
        or parsed["gate_commit"] != gate_commit
        or parsed["foundation_tree_sha1"]
        != decoder._git_tree_at_commit(repo_root, foundation_commit)
        or parsed["candidate"] is not None
        or parsed["seed"] is not None
        or parsed["address"] is not None
    ):
        raise SelectionGateError("Authorization C3-v2 state or identity differs")
    if production and gate_parents[0] != REPAIR_R_COMMIT:
        raise SelectionGateError("Gate G is not the direct child of approved Repair R")

    binding_artifact = _require_exact_mapping(
        parsed["binding_artifact"], {"path", "sha256", "git_blob_sha1"}, "C3-v2 B binding"
    )
    expected_b_sha = decoder.BINDING_B_SHA256 if production else sha256_bytes(b_raw)
    expected_b_blob = (
        decoder.BINDING_B_GIT_BLOB_SHA1
        if production
        else decoder.git_blob_sha1(b_raw)
    )
    if (
        binding_artifact["path"] != PHASE_B_BINDING_RELATIVE
        or binding_artifact["sha256"] != expected_b_sha
        or binding_artifact["git_blob_sha1"] != expected_b_blob
    ):
        raise SelectionGateError("C3-v2 B binding differs")

    source = _require_exact_mapping(
        parsed["source"],
        {"path", "revision", "sha256", "schema_sha256", "row_count"},
        "C3-v2 source",
    )
    expected_source = _expected_source(base_inputs)
    if source != expected_source:
        raise SelectionGateError("C3-v2 source binding differs from authenticated F")

    contracts = _require_exact_mapping(
        parsed["contracts"],
        {"v2_sha256", "v3_sha256", "v4_sha256", "preselection_checkpoint_sha256"},
        "C3-v2 contracts",
    )
    expected_preselection = (
        decoder.B_PRESELECTION_CHECKPOINT_SHA256
        if production
        else sha256_bytes(b_raw)
    )
    if contracts != {
        "v2_sha256": PHASE_B_RESUME_CONTRACT_V2_SHA256,
        "v3_sha256": decoder._resume_contract_v3_hash(),
        "v4_sha256": PHASE_B_SOURCE_SELECTION_CONTRACT_V4_SHA256,
        "preselection_checkpoint_sha256": expected_preselection,
    }:
        raise SelectionGateError("C3-v2 contract digests differ")

    runtime = _require_exact_mapping(
        parsed["runtime_versions"], {"python", "datasets", "pyarrow"}, "C3-v2 runtime"
    )
    if runtime != {
        "python": decoder.SUPPORTED_PYTHON,
        "datasets": decoder.SUPPORTED_DATASETS,
        "pyarrow": decoder.SUPPORTED_PYARROW,
    }:
        raise SelectionGateError("C3-v2 runtime contract differs")
    approval = _require_exact_mapping(
        parsed["approval"], {"thread_id", "text_sha256"}, "C3-v2 approval"
    )
    if approval != {
        "thread_id": SELECTION_GATE_APPROVAL_THREAD_ID,
        "text_sha256": SELECTION_GATE_APPROVAL_TEXT_SHA256,
    }:
        raise SelectionGateError("C3-v2 external approval binding differs")
    scan = _require_exact_mapping(
        parsed["scan"],
        {
            "scan_id",
            "attempt_limit",
            "retry",
            "start_ordinal",
            "final_possible_ordinal",
            "stop_rule",
        },
        "C3-v2 scan",
    )
    if scan != {
        "scan_id": compute_scan_id(gate_commit),
        "attempt_limit": 1,
        "retry": False,
        "start_ordinal": 180,
        "final_possible_ordinal": 887,
        "stop_rule": PHASE_B_SOURCE_SELECTION_CONTRACT_V4["stop_rule"],
    }:
        raise SelectionGateError("C3-v2 scan bounds or identity differ")
    paths = _require_exact_mapping(parsed["paths"], {"claim", "receipt"}, "C3-v2 paths")
    if paths != {
        "claim": SELECTION_CLAIM_RELATIVE,
        "receipt": SELECTION_RECEIPT_RELATIVE,
    }:
        raise SelectionGateError("C3-v2 claim/receipt paths differ")
    forbidden = _require_exact_mapping(
        parsed["forbidden_universe"],
        {"artifact_path", "artifact_sha256", "witness_sha256"},
        "C3-v2 forbidden universe",
    )
    audit_raw = decoder._git_blob_at_commit(repo_root, gate_commit, PHASE_A_AUDIT_RELATIVE)
    audit_sha = sha256_bytes(audit_raw)
    if forbidden != {
        "artifact_path": PHASE_A_AUDIT_RELATIVE,
        "artifact_sha256": audit_sha,
        "witness_sha256": _witness_hash_from_audit(audit_raw),
    }:
        raise SelectionGateError("C3-v2 forbidden-universe binding differs")
    if production and (
        audit_sha != PHASE_A_AUDIT_SHA256
        or forbidden["witness_sha256"] != PHASE_A_WITNESS_SHA256
    ):
        raise SelectionGateError("C3-v2 forbidden-universe artifact is not approved")
    return parsed, gate_commit, {"source_contract": expected_source, "audit_raw": audit_raw, "b": b}


def _witness_hash_from_audit(audit_raw: bytes) -> str:
    audit = _parse_canonical(audit_raw, "Phase-A audit")
    source_auth = audit.get("source_authentication")
    if not isinstance(source_auth, dict):
        raise SelectionGateError("Phase-A audit source authentication is missing")
    witness = source_auth.get("forbidden_witness")
    if not isinstance(witness, dict) or not isinstance(witness.get("witness_sha256"), str):
        raise SelectionGateError("Phase-A audit forbidden witness is missing")
    return cast(str, witness["witness_sha256"])


def _authenticated_git_input(
    repo_root: Path,
    gate_commit: str,
    relative: str,
    *,
    expected_sha256: str,
    expected_git_blob_sha1: str,
) -> bytes:
    raw = decoder._git_blob_at_commit(repo_root, gate_commit, relative)
    if sha256_bytes(raw) != expected_sha256:
        raise SelectionGateError(f"authenticated input hash differs: {relative}")
    if decoder.git_blob_sha1(raw) != expected_git_blob_sha1:
        raise SelectionGateError(f"authenticated input Git blob differs: {relative}")
    path = repo_root / relative
    if not path.is_file() or path.read_bytes() != raw:
        raise SelectionGateError(f"authenticated input worktree differs: {relative}")
    return raw


def _locate_recovery_record(extension_raw: bytes) -> str:
    identity_matches: list[dict[str, Any]] = []
    for line in extension_raw.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise SelectionGateError("successor extension is not valid JSONL") from error
        if not isinstance(record, dict):
            raise SelectionGateError("successor extension record is not an object")
        if (
            record.get("paper_id") == RECOVERED_REFERENCE_PAPER_ID
            and record.get("example_id") == RECOVERED_REFERENCE_EXAMPLE_ID
        ):
            identity_matches.append(cast(dict[str, Any], record))
    if len(identity_matches) != 1:
        raise SelectionGateError(
            "recovery projection does not identify exactly one extension record"
        )
    record = identity_matches[0]
    if record.get("split") != "successor_support" or not isinstance(
        record.get("question"), str
    ):
        raise SelectionGateError("recovery extension record schema differs")
    evidence = record.get("reference_evidence")
    if not isinstance(evidence, list) or len(evidence) != RECOVERED_REFERENCE_CARDINALITY:
        raise SelectionGateError("recovery extension reference cardinality differs")
    reference = str(evidence[0])
    if sha256_bytes(reference.encode("utf-8")) != RECOVERED_REFERENCE_SHA256:
        raise SelectionGateError("recovery extension reference digest differs")
    if not isinstance(record.get("paper"), str) or sha256_bytes(
        record["paper"].encode("utf-8")
    ) != RECOVERED_REFERENCE_RENDERED_PAPER_SHA256:
        raise SelectionGateError("recovery extension rendered-paper digest differs")
    return reference


def _authenticated_recovery_reference(
    repo_root: Path, gate_commit: str, audit: Mapping[str, Any]
) -> str:
    forbidden = cast(
        Mapping[str, Any], PHASE_B_SOURCE_SELECTION_CONTRACT_V4["forbidden_universe"]
    )
    projection = _require_exact_mapping(
        forbidden["recovery_projection"],
        {
            "extension_path",
            "extension_sha256",
            "extension_git_blob_sha1",
            "extension_manifest_path",
            "extension_manifest_sha256",
            "extension_manifest_git_blob_sha1",
            "successor_manifest_path",
            "successor_manifest_sha256",
            "successor_manifest_git_blob_sha1",
            "address_audit_path",
            "address_audit_sha256",
            "address_audit_git_blob_sha1",
            "introduced_commit",
            "source_ordinal",
            "paper_id",
            "example_id",
            "question_index",
            "canonical_row_sha256",
            "rendered_paper_sha256",
            "reference_cardinality",
            "reference_sha256",
            "expected_reference_digest_count",
        },
        "recovery projection",
    )
    expected_projection = {
        "extension_path": SUCCESSOR_EXTENSION_RELATIVE,
        "extension_sha256": SUCCESSOR_EXTENSION_SHA256,
        "extension_git_blob_sha1": SUCCESSOR_EXTENSION_GIT_BLOB_SHA1,
        "extension_manifest_path": SUCCESSOR_EXTENSION_MANIFEST_RELATIVE,
        "extension_manifest_sha256": SUCCESSOR_EXTENSION_MANIFEST_SHA256,
        "extension_manifest_git_blob_sha1": SUCCESSOR_EXTENSION_MANIFEST_GIT_BLOB_SHA1,
        "successor_manifest_path": SUCCESSOR_MANIFEST_RELATIVE,
        "successor_manifest_sha256": SUCCESSOR_MANIFEST_SHA256,
        "successor_manifest_git_blob_sha1": SUCCESSOR_MANIFEST_GIT_BLOB_SHA1,
        "address_audit_path": SUCCESSOR_ADDRESS_AUDIT_V1_RELATIVE,
        "address_audit_sha256": SUCCESSOR_ADDRESS_AUDIT_V1_SHA256,
        "address_audit_git_blob_sha1": SUCCESSOR_ADDRESS_AUDIT_V1_GIT_BLOB_SHA1,
        "introduced_commit": SUCCESSOR_EXTENSION_INTRODUCED_COMMIT,
        "source_ordinal": RECOVERED_REFERENCE_SOURCE_ORDINAL,
        "paper_id": RECOVERED_REFERENCE_PAPER_ID,
        "example_id": RECOVERED_REFERENCE_EXAMPLE_ID,
        "question_index": RECOVERED_REFERENCE_QUESTION_INDEX,
        "canonical_row_sha256": RECOVERED_REFERENCE_CANONICAL_ROW_SHA256,
        "rendered_paper_sha256": RECOVERED_REFERENCE_RENDERED_PAPER_SHA256,
        "reference_cardinality": RECOVERED_REFERENCE_CARDINALITY,
        "reference_sha256": RECOVERED_REFERENCE_SHA256,
        "expected_reference_digest_count": RECOVERED_REFERENCE_EXPECTED_DIGEST_COUNT,
    }
    if projection != expected_projection:
        raise SelectionGateError("recovery projection differs from approved bindings")
    decoder._git_ancestor(repo_root, SUCCESSOR_EXTENSION_INTRODUCED_COMMIT, gate_commit)

    extension_raw = _authenticated_git_input(
        repo_root,
        gate_commit,
        SUCCESSOR_EXTENSION_RELATIVE,
        expected_sha256=SUCCESSOR_EXTENSION_SHA256,
        expected_git_blob_sha1=SUCCESSOR_EXTENSION_GIT_BLOB_SHA1,
    )
    extension_manifest_raw = _authenticated_git_input(
        repo_root,
        gate_commit,
        SUCCESSOR_EXTENSION_MANIFEST_RELATIVE,
        expected_sha256=SUCCESSOR_EXTENSION_MANIFEST_SHA256,
        expected_git_blob_sha1=SUCCESSOR_EXTENSION_MANIFEST_GIT_BLOB_SHA1,
    )
    extension_manifest = _parse_authenticated_json(
        extension_manifest_raw, "successor extension manifest"
    )
    output = _require_exact_mapping(
        extension_manifest.get("output"),
        {"path", "sha256", "bytes"},
        "successor extension output",
    )
    if output != {
        "path": SUCCESSOR_EXTENSION_RELATIVE,
        "sha256": SUCCESSOR_EXTENSION_SHA256,
        "bytes": len(extension_raw),
    }:
        raise SelectionGateError("successor extension manifest output differs")
    if (
        extension_manifest.get("dataset") != "allenai/qasper"
        or extension_manifest.get("source_revision")
        != "fdc9d8214fbab5dd782958601db4d678e6934a54"
        or extension_manifest.get("converted_parquet_revision")
        != "06806e4608976fc2fac0a090ac425d5b2b29caf4"
        or extension_manifest.get("selection")
        != {
            "fresh_reference_positions_or_midpoint_coverage_inspected": False,
            "maximum_paper_characters": 60000,
            "minimum_span_characters": 20,
            "one_question_per_paper": True,
            "seed_forbidden_ids_and_spans_from_all_old_120": True,
            "source_order": True,
        }
    ):
        raise SelectionGateError("successor extension manifest provenance differs")

    successor_manifest_raw = _authenticated_git_input(
        repo_root,
        gate_commit,
        SUCCESSOR_MANIFEST_RELATIVE,
        expected_sha256=SUCCESSOR_MANIFEST_SHA256,
        expected_git_blob_sha1=SUCCESSOR_MANIFEST_GIT_BLOB_SHA1,
    )
    successor_manifest = _parse_authenticated_json(
        successor_manifest_raw, "successor support manifest"
    )
    if successor_manifest.get("dataset") != "allenai/qasper" or successor_manifest.get(
        "source_revision"
    ) != "fdc9d8214fbab5dd782958601db4d678e6934a54":
        raise SelectionGateError("successor support manifest provenance differs")
    prior_extension = _require_exact_mapping(
        successor_manifest.get("prior_extension"),
        {"path", "sha256", "papers"},
        "successor prior extension",
    )
    if prior_extension != {
        "path": SUCCESSOR_EXTENSION_RELATIVE,
        "sha256": SUCCESSOR_EXTENSION_SHA256,
        "papers": 112,
    }:
        raise SelectionGateError("successor support manifest extension binding differs")
    successor = successor_manifest.get("successor")
    if not isinstance(successor, dict):
        raise SelectionGateError("successor support manifest successor section is missing")
    retired = _require_exact_mapping(
        successor.get("retired"),
        {"example_id", "paper_id", "row_sha256"},
        "successor retired row",
    )
    if retired != {
        "example_id": RECOVERED_REFERENCE_EXAMPLE_ID,
        "paper_id": RECOVERED_REFERENCE_PAPER_ID,
        "row_sha256": RECOVERED_REFERENCE_CANONICAL_ROW_SHA256,
    }:
        raise SelectionGateError("successor retired-row binding differs")
    checks = successor.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise SelectionGateError("successor support manifest checks are not authenticated")

    address_audit_raw = _authenticated_git_input(
        repo_root,
        gate_commit,
        SUCCESSOR_ADDRESS_AUDIT_V1_RELATIVE,
        expected_sha256=SUCCESSOR_ADDRESS_AUDIT_V1_SHA256,
        expected_git_blob_sha1=SUCCESSOR_ADDRESS_AUDIT_V1_GIT_BLOB_SHA1,
    )
    address_audit = _parse_authenticated_json(address_audit_raw, "successor address audit v1")
    address_checks = address_audit.get("checks")
    if not isinstance(address_checks, dict) or not address_checks or not all(
        address_checks.values()
    ):
        raise SelectionGateError("successor address audit checks are not authenticated")
    address_retired = _require_exact_mapping(
        address_audit.get("retired"),
        {
            "cache_salt",
            "example_id",
            "paper_id",
            "rollout_slot",
            "row_sha256",
            "scientific_group_id",
            "seed",
            "slot_id",
        },
        "successor address retired row",
    )
    if address_retired.get("example_id") != RECOVERED_REFERENCE_EXAMPLE_ID or (
        address_retired.get("paper_id") != RECOVERED_REFERENCE_PAPER_ID
        or address_retired.get("row_sha256") != RECOVERED_REFERENCE_CANONICAL_ROW_SHA256
    ):
        raise SelectionGateError("successor address retired-row identity differs")

    witness = cast(Mapping[str, Any], audit["source_authentication"])["forbidden_witness"]
    retired_units = cast(Mapping[str, Any], witness)["retired_units"]
    if not isinstance(retired_units, list):
        raise SelectionGateError("Phase-A retired-unit audit is missing")
    matching_units = [
        cast(dict[str, Any], unit)
        for unit in retired_units
        if isinstance(unit, dict)
        and unit.get("source_ordinal") == RECOVERED_REFERENCE_SOURCE_ORDINAL
    ]
    if len(matching_units) != 1:
        raise SelectionGateError("Phase-A ordinal-89 recovery tuple is not unique")
    expected_unit = {
        "canonical_row_sha256": RECOVERED_REFERENCE_CANONICAL_ROW_SHA256,
        "example_id": RECOVERED_REFERENCE_EXAMPLE_ID,
        "paper_id": RECOVERED_REFERENCE_PAPER_ID,
        "question_index": RECOVERED_REFERENCE_QUESTION_INDEX,
        "reference_span_sha256": [RECOVERED_REFERENCE_SHA256],
        "rendered_paper_sha256": [RECOVERED_REFERENCE_RENDERED_PAPER_SHA256],
        "source_ordinal": RECOVERED_REFERENCE_SOURCE_ORDINAL,
    }
    unit = matching_units[0]
    if any(unit.get(key) != value for key, value in expected_unit.items()):
        raise SelectionGateError("Phase-A ordinal-89 recovery tuple differs")
    return _locate_recovery_record(extension_raw)


def _authenticated_forbidden_universe(
    repo_root: Path, gate_commit: str, audit_raw: bytes
) -> SelectionUniverse:
    """Rebuild the selector universe from authenticated committed artifacts only."""

    audit = _parse_canonical(audit_raw, "Phase-A audit")
    source_auth = audit.get("source_authentication")
    if not isinstance(source_auth, dict):
        raise SelectionGateError("Phase-A audit source authentication is missing")
    witness = source_auth.get("forbidden_witness")
    if not isinstance(witness, dict):
        raise SelectionGateError("Phase-A forbidden witness is missing")
    if witness.get("witness_sha256") != PHASE_A_WITNESS_SHA256:
        raise SelectionGateError("Phase-A forbidden witness digest differs")
    without_hash = {key: value for key, value in witness.items() if key != "witness_sha256"}
    if sha256_json(without_hash) != witness["witness_sha256"]:
        raise SelectionGateError("Phase-A forbidden witness self-hash differs")
    sets = witness.get("forbidden_sets")
    if not isinstance(sets, dict):
        raise SelectionGateError("Phase-A forbidden sets are missing")
    for name, expected_count in EXPECTED_CARDINALITIES.items():
        values = sets.get(name)
        if not isinstance(values, list) or len(values) != expected_count:
            raise SelectionGateError(f"Phase-A forbidden set cardinality differs: {name}")

    raw_sources = cast(
        Mapping[str, str],
        cast(Mapping[str, Any], PHASE_B_SOURCE_SELECTION_CONTRACT_V4["forbidden_universe"])[
            "raw_reference_source_hashes"
        ],
    )
    raw_references: set[str] = set()
    for relative, expected_hash in raw_sources.items():
        raw = decoder._git_blob_at_commit(repo_root, gate_commit, relative)
        if sha256_bytes(raw) != expected_hash:
            raise SelectionGateError(f"authenticated reference source differs: {relative}")
        for line in raw.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SelectionGateError(f"reference source is not JSONL: {relative}") from error
            if not isinstance(row, dict):
                raise SelectionGateError(f"reference source row is not an object: {relative}")
            evidence = row.get("reference_evidence")
            if not isinstance(evidence, list):
                raise SelectionGateError(f"reference source evidence is missing: {relative}")
            raw_references.update(str(span) for span in evidence)

    for relative, expected_hash in {
        **HISTORICAL_ADDRESS_HASHES,
        **HISTORICAL_ROLLOUT_HASHES,
    }.items():
        raw = decoder._git_blob_at_commit(repo_root, gate_commit, relative)
        if sha256_bytes(raw) != expected_hash:
            raise SelectionGateError(f"historical identity source differs: {relative}")
        path = repo_root / relative
        if not path.is_file() or path.read_bytes() != raw:
            raise SelectionGateError(f"historical identity worktree differs: {relative}")
    historical = historical_identity_witness(repo_root)
    expected_historical = cast(
        Mapping[str, Any], witness["authenticated_historical_identity_witness"]
    )
    historical_identity_sets = cast(Mapping[str, Iterable[str]], historical["identity_sets"])
    sanitized_historical: dict[str, list[str]] = {}
    for name, values in historical_identity_sets.items():
        if name == "reference_spans":
            sanitized_historical["reference_span_sha256"] = sorted(
                sha256_bytes(str(value).encode("utf-8")) for value in values
            )
        else:
            sanitized_historical[name] = sorted(str(value) for value in values)
    historical_projection = {
        "artifact_hashes": historical["artifacts"],
        "witness_sha256": historical["witness_sha256"],
        "identity_sets": sanitized_historical,
        "address_count": len(sanitized_historical.get("addresses", [])),
    }
    if expected_historical != historical_projection:
        raise SelectionGateError("historical identity witness differs from committed inputs")
    raw_references.update(str(value) for value in historical_identity_sets["reference_spans"])
    raw_references.add(_authenticated_recovery_reference(repo_root, gate_commit, audit))

    def union(*names: str) -> frozenset[str]:
        return frozenset(
            str(value) for name in names for value in cast(Iterable[Any], sets[name])
        )

    expected_reference_digests = union(
        "reference_span_sha256",
        "old_snapshot_reference_span_sha256",
        "predecessor_reference_span_sha256",
        "historical_reference_span_sha256",
    )
    actual_reference_digests = frozenset(
        sha256_bytes(value.encode("utf-8")) for value in raw_references
    )
    if len(expected_reference_digests) != RECOVERED_REFERENCE_EXPECTED_DIGEST_COUNT:
        raise SelectionGateError("authenticated reference digest cardinality differs")
    if actual_reference_digests != expected_reference_digests:
        raise SelectionGateError(
            "raw reference spans cannot be reconstructed from authenticated committed artifacts"
        )
    return SelectionUniverse(
        paper_ids=union(
            "retired_paper_ids",
            "old_snapshot_paper_ids",
            "predecessor_paper_ids",
            "historical_paper_ids",
        ),
        example_ids=union(
            "retired_example_ids",
            "old_snapshot_example_ids",
            "predecessor_example_ids",
            "historical_example_ids",
        ),
        rendered_paper_sha256=union(
            "rendered_paper_sha256",
            "old_snapshot_rendered_paper_sha256",
            "predecessor_rendered_paper_sha256",
        ),
        reference_spans=frozenset(raw_references),
        row_sha256=union(
            "retired_row_sha256",
            "old_snapshot_row_sha256",
            "predecessor_row_sha256",
            "historical_row_sha256",
        ),
        addresses=union("source_address_sha256", "historical_address_sha256"),
    )


def _candidate_address_sha256(row: Mapping[str, Any], chosen: Mapping[str, Any]) -> str:
    vector = APPROVED_DERIVATION_VECTOR
    example_id = str(chosen["example_id"])
    namespace = cast(str, vector["namespace"])
    master_seed = cast(str, vector["master_seed"])
    group_id = derive_scientific_group_id(namespace=namespace, example_id=example_id)
    seed, cache_salt = derive_source_episode_seed_and_salt(
        master_seed=master_seed,
        scientific_group_id=group_id,
        rollout_slot=0,
    )
    slot = SourceCollectionSlot.build(
        {"scientific_group_id": group_id, "example_id": example_id, "rollout_slot": 0},
        master_seed=master_seed,
    )
    return cast(
        str,
        sha256_json(
            {
                "paper_id": str(row["id"]),
                "example_id": example_id,
                "seed": seed,
                "scientific_group_id": group_id,
                "slot_id": slot.slot_id,
                "rollout_slot": 0,
                "cache_salt": cache_salt,
                "canonical_row_sha256": decoder.source_row_sha256(row),
            }
        ),
    )


def _iter_rows_from_180(
    parquet_file: Any,
    *,
    instrumentation: SelectionScanInstrumentation | None = None,
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    observer = instrumentation or SelectionScanInstrumentation()
    iter_rows = getattr(parquet_file, "iter_rows", None)
    if callable(iter_rows):
        for ordinal, row in iter_rows(start_ordinal=180, end_ordinal=887):
            if ordinal < 180 or ordinal > 887:
                raise SelectionGateError("source cursor crossed the Gate-G bounds")
            if observer.evaluated_ordinals is not None:
                observer.evaluated_ordinals.append(int(ordinal))
            yield int(ordinal), cast(Mapping[str, Any], row)
        return
    iter_batches = getattr(parquet_file, "iter_batches", None)
    if not callable(iter_batches):
        raise SelectionGateError("authenticated source has no bounded physical-order cursor")
    next_ordinal = 0
    for batch in iter_batches(batch_size=180, row_groups=[0], use_threads=False):
        rows = int(batch.num_rows)
        if observer.batches is not None:
            observer.batches.append(
                {
                    "start_ordinal": next_ordinal,
                    "rows": rows,
                    "end_ordinal": next_ordinal + rows - 1,
                }
            )
        batch_start = next_ordinal
        next_ordinal += rows
        if next_ordinal <= 180:
            continue
        if batch_start < 180:
            raise SelectionGateError("source batch straddles the Gate-G start ordinal")
        decoded_rows = batch.to_pylist()
        if len(decoded_rows) != rows:
            raise SelectionGateError("source batch cardinality changed")
        for offset, row in enumerate(decoded_rows):
            ordinal = batch_start + offset
            if ordinal > 887:
                raise SelectionGateError("source cursor exceeded the final possible ordinal")
            if not isinstance(row, dict):
                raise SelectionGateError("source cursor yielded a non-object row")
            if observer.materialized_ordinals is not None:
                observer.materialized_ordinals.append(ordinal)
            yield ordinal, cast(Mapping[str, Any], row)


def _scan_once(
    parquet_file: Any,
    universe: SelectionUniverse,
    *,
    instrumentation: SelectionScanInstrumentation | None = None,
) -> dict[str, Any]:
    transcript = hashlib.sha256(_TRANSCRIPT_VERSION.encode("ascii")).digest()
    expected_ordinal = 180
    for ordinal, row in _iter_rows_from_180(parquet_file, instrumentation=instrumentation):
        if ordinal != expected_ordinal:
            raise SelectionGateError("source cursor has a gap, duplicate, or order drift")
        row_hash = decoder.source_row_sha256(row)
        decision = "reject_no_exact_evidence_or_authenticated_collision"
        try:
            selected = select_first_eligible(
                row,
                forbidden_paper_ids=set(universe.paper_ids),
                forbidden_example_ids=set(universe.example_ids),
                forbidden_rendered_paper_sha256=universe.rendered_paper_sha256,
                forbidden_reference_spans=set(universe.reference_spans),
                forbidden_row_sha256=set(universe.row_sha256),
                forbidden_address_sha256=universe.addresses,
            )
            if selected is not None:
                chosen, question_index = selected
                address_sha = _candidate_address_sha256(row, chosen)
                classification = classify_candidate_collisions(
                    row=row,
                    example_id=str(chosen["example_id"]),
                    paper=str(chosen["paper"]),
                    evidence=cast(Iterable[str], chosen["reference_evidence"]),
                    forbidden_paper_ids=set(universe.paper_ids),
                    forbidden_example_ids=set(universe.example_ids),
                    forbidden_rendered_paper_sha256=set(universe.rendered_paper_sha256),
                    forbidden_reference_spans=set(universe.reference_spans),
                    forbidden_row_sha256=set(universe.row_sha256),
                    candidate_address_sha256=address_sha,
                    forbidden_address_sha256=set(universe.addresses),
                )
                if classification.primary_terminal is not None:
                    raise TerminalIdentityCollision(classification)
                decision = "eligible_candidate"
                transcript = _advance_transcript(transcript, ordinal, row_hash, decision)
                return {
                    "disposition": "eligible_candidate",
                    "stop_ordinal": ordinal,
                    "candidate": {
                        "source_ordinal": ordinal,
                        "paper_id": str(chosen["paper_id"]),
                        "example_id": str(chosen["example_id"]),
                        "question_index": question_index,
                        "source_row_sha256": row_hash,
                        "address_sha256": address_sha,
                    },
                    "transcript_sha256": transcript.hex(),
                }
        except TerminalIdentityCollision as error:
            decision = f"terminal_{error.collision_class}"
            transcript = _advance_transcript(transcript, ordinal, row_hash, decision)
            return {
                "disposition": "terminal_collision",
                "stop_ordinal": ordinal,
                "collision_class": error.collision_class,
                "collision_set": list(error.collision_set),
                "candidate": None,
                "transcript_sha256": transcript.hex(),
            }
        transcript = _advance_transcript(transcript, ordinal, row_hash, decision)
        expected_ordinal += 1
    if expected_ordinal != 888:
        raise SelectionGateError("source cursor ended before ordinal 887")
    return {
        "disposition": "exhausted",
        "stop_ordinal": 887,
        "candidate": None,
        "transcript_sha256": transcript.hex(),
    }


def _immutable_paths() -> dict[str, str]:
    paths = dict(FROZEN_HASHES)
    forbidden = cast(
        Mapping[str, Any], PHASE_B_SOURCE_SELECTION_CONTRACT_V4["forbidden_universe"]
    )
    paths.update(
        {
            str(relative): str(expected)
            for relative, expected in cast(
                Mapping[str, Any], forbidden["raw_reference_source_hashes"]
            ).items()
        }
    )
    paths.update(
        {
            PHASE_A_AUDIT_RELATIVE: PHASE_A_AUDIT_SHA256,
            decoder.SOURCE_ARTIFACT_RELATIVE: decoder.SOURCE_SHA256,
            PHASE_B_BINDING_RELATIVE: decoder.BINDING_B_SHA256,
            SUCCESSOR_EXTENSION_RELATIVE: SUCCESSOR_EXTENSION_SHA256,
            SUCCESSOR_EXTENSION_MANIFEST_RELATIVE: SUCCESSOR_EXTENSION_MANIFEST_SHA256,
            SUCCESSOR_MANIFEST_RELATIVE: SUCCESSOR_MANIFEST_SHA256,
            SUCCESSOR_ADDRESS_AUDIT_V1_RELATIVE: SUCCESSOR_ADDRESS_AUDIT_V1_SHA256,
            LEGACY_C3_AUTHORIZATION_RELATIVE: "",
            PHASE_C3_V2_AUTHORIZATION_RELATIVE: "",
        }
    )
    return paths


def _claim_payload(scan_id: str, gate_commit: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "domain": "redco-stage-d1-support-v13-source-selection-claim-v1",
        "state": "claimed",
        "scan_id": scan_id,
        "gate_commit": gate_commit,
        "attempt_limit": 1,
        "retry": False,
        "candidate": None,
        "seed": None,
        "address": None,
        "phase_b_source_selection_authorized": True,
        "launch_authorized": False,
        "provider_calls_authorized": False,
        "model_calls_authorized": False,
        "prime_gpu_scientific_launch_authorized": False,
        "science_authorized": False,
        "claim_path": SELECTION_CLAIM_RELATIVE,
        "receipt_path": SELECTION_RECEIPT_RELATIVE,
    }


def _create_exclusive_claim(root: Path, payload: dict[str, Any]) -> bytes:
    claim_path = root / SELECTION_CLAIM_RELATIVE
    receipt_path = root / SELECTION_RECEIPT_RELATIVE
    if claim_path.exists() or receipt_path.exists():
        raise SelectionGateAlreadyClaimed("Gate-G attempt is already claimed")
    data = cast(bytes, canonical_json_bytes(payload))
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(claim_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise SelectionGateAlreadyClaimed(
            "Gate-G claim was created by another attempt"
        ) from error
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    return data


def _receipt(
    scan_id: str,
    gate_commit: str,
    claim_bytes: bytes,
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": 2,
        "domain": "redco-stage-d1-support-v13-source-selection-receipt-v2",
        "state": "terminal",
        "scan_id": scan_id,
        "gate_commit": gate_commit,
        "attempt": 1,
        "attempt_limit": 1,
        "retry": False,
        "claim_path": SELECTION_CLAIM_RELATIVE,
        "claim_sha256": sha256_bytes(claim_bytes),
        "candidate": outcome.get("candidate"),
        "disposition": outcome.get("disposition"),
        "stop_ordinal": outcome.get("stop_ordinal"),
        "transcript_sha256": outcome.get("transcript_sha256"),
        "phase_b_source_selection_authorized": True,
        "launch_authorized": False,
        "provider_calls_authorized": False,
        "model_calls_authorized": False,
        "prime_gpu_scientific_launch_authorized": False,
        "science_authorized": False,
        "receipt_path": SELECTION_RECEIPT_RELATIVE,
    }
    for key in ("collision_class", "collision_set", "error_type", "error_sha256"):
        if key in outcome:
            value[key] = outcome[key]
    return value


def _publish_receipt(root: Path, value: dict[str, Any]) -> None:
    atomic_write(
        root,
        SELECTION_RECEIPT_RELATIVE,
        canonical_json_bytes(value),
        output_paths=(SELECTION_CLAIM_RELATIVE, SELECTION_RECEIPT_RELATIVE),
    )


def activate_selection_gate() -> dict[str, Any]:
    """Run the sole no-argument Gate-G source-selection attempt."""

    root = decoder.PROJECT_ROOT
    artifact, gate_commit, base_inputs = _validate_c3_v2(root)
    del artifact
    # The complete historical universe is authenticated before any output
    # validation, claim creation, runtime inspection, or source access.
    universe = _authenticated_forbidden_universe(
        root, gate_commit, cast(bytes, base_inputs["audit_raw"])
    )
    immutable = _immutable_paths()
    validate_output_paths(
        root,
        immutable,
        output_paths=(SELECTION_CLAIM_RELATIVE, SELECTION_RECEIPT_RELATIVE),
    )
    scan_id = compute_scan_id(gate_commit)
    claim_bytes = _create_exclusive_claim(root, _claim_payload(scan_id, gate_commit))
    try:
        pyarrow, _datasets = decoder._require_runtime_versions_only()
        parquet_file = decoder._validate_production_source_metadata(
            root, cast(Mapping[str, Any], base_inputs["source_contract"]), pyarrow
        )
        outcome = _scan_once(parquet_file, universe)
    except Exception as error:
        outcome = {
            "disposition": "failure",
            "candidate": None,
            "stop_ordinal": None,
            "transcript_sha256": None,
            "error_type": type(error).__name__,
            "error_sha256": sha256_bytes(str(error).encode("utf-8")),
        }
    receipt = _receipt(scan_id, gate_commit, claim_bytes, outcome)
    _publish_receipt(root, receipt)
    return receipt


__all__ = [
    "G_DIFF_ALLOWLIST",
    "G_SOURCE_SELECTION_RELATIVE",
    "LEGACY_C3_AUTHORIZATION_RELATIVE",
    "LEGACY_C_AUTHORIZATION_RELATIVE",
    "SELECTION_CLAIM_RELATIVE",
    "SELECTION_GATE_APPROVAL_TEXT_SHA256",
    "SELECTION_GATE_APPROVAL_THREAD_ID",
    "SELECTION_RECEIPT_RELATIVE",
    "SelectionGateAlreadyClaimed",
    "SelectionGateError",
    "SelectionScanInstrumentation",
    "SelectionUniverse",
    "_authenticated_forbidden_universe",
    "_iter_rows_from_180",
    "_scan_once",
    "_validate_c3_v2",
    "activate_selection_gate",
    "compute_scan_id",
]
