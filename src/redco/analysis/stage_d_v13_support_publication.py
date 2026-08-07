"""Read-only support artifact authentication owned outside the materializer."""

from __future__ import annotations

import json
from pathlib import Path

from redco.analysis.stage_d_dependency_stack import live_owner_dependency_payload
from redco.analysis.stage_d_v13_draft import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
)
from redco.analysis.stage_d_v13_support_contract import (
    CANDIDATE_EXAMPLE_ID,
    CANDIDATE_PAPER_ID,
    CANDIDATE_QUESTION_INDEX,
    CANDIDATE_RELATIVE,
    CANDIDATE_ROW_SHA256,
    CANDIDATE_SOURCE_ORDINAL,
    COMPOSITION_RELATIVE,
    PROTOCOL_AUDIT_RELATIVE,
    PROTOCOL_RELATIVE,
    REVIEWED_PROTOCOL_ARTIFACT_SHA256,
    SELECTION_CLAIM_SHA256,
    SELECTION_MANIFEST_SHA256,
    SELECTION_RECEIPT_SHA256,
    SOURCE_ARTIFACT_RELATIVE,
    SOURCE_BYTES,
    SOURCE_LOGICAL_URL,
    SOURCE_PATH,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    SOURCE_ROW_COUNT,
    SOURCE_SCHEMA_SHA256,
    SOURCE_SEMANTIC_COMMIT,
    SOURCE_SHA256,
    SUPPORTED_DATASETS,
    SUPPORTED_PYARROW,
    SUPPORTED_PYTHON,
    authenticate_upstream_evidence,
    require_supported_runtime,
    sampling_contract_binding,
)


def check_protocol_artifacts(root: Path, output_root: Path) -> dict[str, str]:
    """Authenticate the published set without opening or decoding QASPER."""

    runtime = require_supported_runtime()
    upstream = authenticate_upstream_evidence(root)
    paths = {
        CANDIDATE_RELATIVE,
        COMPOSITION_RELATIVE,
        PROTOCOL_RELATIVE,
        PROTOCOL_AUDIT_RELATIVE,
    }
    for entry in (output_root.rglob("*") if output_root.exists() else ()):
        if entry.is_file() and entry.relative_to(output_root).as_posix() not in paths:
            raise ValueError(
                "published support output contains an unexpected file: "
                f"{entry.relative_to(output_root).as_posix()}"
            )
    actual: dict[str, bytes] = {}
    for relative in paths:
        path = output_root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"published support artifact is missing: {relative}")
        actual[relative] = path.read_bytes()
    hashes = {relative: sha256_bytes(raw) for relative, raw in actual.items()}
    if hashes != REVIEWED_PROTOCOL_ARTIFACT_SHA256:
        raise ValueError("published support artifact differs from the reviewed byte set")
    try:
        candidate = json.loads(actual[CANDIDATE_RELATIVE])
        composition = json.loads(actual[COMPOSITION_RELATIVE])
        protocol = json.loads(actual[PROTOCOL_RELATIVE])
        audit = json.loads(actual[PROTOCOL_AUDIT_RELATIVE])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("published support artifact is not JSON") from error
    if any(
        not isinstance(value, dict)
        or canonical_json_bytes(value) != actual[relative]
        for relative, value in (
            (CANDIDATE_RELATIVE, candidate),
            (COMPOSITION_RELATIVE, composition),
            (PROTOCOL_RELATIVE, protocol),
            (PROTOCOL_AUDIT_RELATIVE, audit),
        )
    ):
        raise ValueError("published support artifact is not canonical")
    candidate_source = candidate.get("source")
    instrumentation = candidate.get("instrumentation")
    candidate_authority = candidate.get("authority")
    composition_candidate = composition.get("candidate")
    composition_cohort = composition.get("support_cohort")
    composition_authorization = composition.get("authorization")
    protocol_source = protocol.get("source")
    protocol_authorization = protocol.get("authorization")
    protocol_attempt_policy = protocol.get("attempt_policy")
    protocol_selection = protocol.get("selection_evidence")
    audit_selection = audit.get("selection_evidence")
    dependency_stack = live_owner_dependency_payload(root)
    sampling_contract = sampling_contract_binding(root)
    expected_protocol_source = {
        "repository": SOURCE_REPOSITORY,
        "logical_url": SOURCE_LOGICAL_URL,
        "revision": SOURCE_REVISION,
        "semantic_source_commit": SOURCE_SEMANTIC_COMMIT,
        "path": SOURCE_PATH,
        "local_artifact": SOURCE_ARTIFACT_RELATIVE,
        "sha256": SOURCE_SHA256,
        "schema_sha256": SOURCE_SCHEMA_SHA256,
        "bytes": SOURCE_BYTES,
        "rows": SOURCE_ROW_COUNT,
        "logical_read_wall": (
            "Arrow emits logical ordinals 0..180 in bounded one-row batches; only ordinal "
            "180 is Python-converted, canonicalized, and evaluated; ordinal 181 is never "
            "requested or emitted"
        ),
        "decoder": {
            "batch_size": 1,
            "use_threads": False,
            "row_groups": [0],
            "logical_readahead": False,
            "physical_compressed_page_io_may_span_row_group": True,
        },
        "required_runtime": {
            "python": SUPPORTED_PYTHON,
            "pyarrow": SUPPORTED_PYARROW,
            "datasets": SUPPORTED_DATASETS,
        },
    }
    if (
        candidate.get("domain") != "redco-stage-d1-support-v13-candidate-ordinal-180-v1"
        or not isinstance(candidate_source, dict)
        or candidate_source.get("ordinal") != CANDIDATE_SOURCE_ORDINAL
        or candidate_source.get("paper_id") != CANDIDATE_PAPER_ID
        or candidate_source.get("example_id") != CANDIDATE_EXAMPLE_ID
        or candidate_source.get("question_index") != CANDIDATE_QUESTION_INDEX
        or candidate_source.get("row_sha256") != CANDIDATE_ROW_SHA256
        or candidate_source.get("selection_claim_sha256") != SELECTION_CLAIM_SHA256
        or candidate_source.get("selection_receipt_sha256") != SELECTION_RECEIPT_SHA256
        or candidate_source.get("selection_evidence_manifest_sha256")
        != SELECTION_MANIFEST_SHA256
        or candidate_source.get("upstream_evidence_hashes") != upstream["upstream_hashes"]
        or candidate_source.get("frozen_decision_rule_sha256") != upstream["decision_rule_sha256"]
        or not isinstance(instrumentation, dict)
        or instrumentation.get("post_180_requested") is not False
        or instrumentation.get("post_180_materialized") is not False
        or instrumentation.get("post_180_canonicalized") is not False
        or instrumentation.get("post_180_evaluated") is not False
        or instrumentation.get("materialized_ordinals") != [CANDIDATE_SOURCE_ORDINAL]
        or instrumentation.get("canonicalized_ordinals") != [CANDIDATE_SOURCE_ORDINAL]
        or instrumentation.get("evaluated_ordinals") != [CANDIDATE_SOURCE_ORDINAL]
        or candidate_source.get("runtime") != runtime
        or candidate_authority
        != {
            "candidate_materialized": True,
            "source_selection_repeated": False,
            "provider_calls_authorized": False,
            "model_calls_authorized": False,
            "science_authorized": False,
            "launch_authorized": False,
        }
        or protocol_source != expected_protocol_source
        or protocol.get("dependency_stack") != dependency_stack
        or audit.get("dependency_stack") != dependency_stack
        or candidate.get("sampling_contract") != sampling_contract
        or composition.get("sampling_contract") != sampling_contract
        or protocol.get("sampling_contract") != sampling_contract
        or audit.get("sampling_contract") != sampling_contract
    ):
        raise ValueError("published candidate does not bind the authenticated materialization")
    if (
        not isinstance(composition_candidate, dict)
        or composition_candidate.get("sha256") != hashes[CANDIDATE_RELATIVE]
        or composition_candidate.get("ordinal") != CANDIDATE_SOURCE_ORDINAL
        or composition_candidate.get("paper_id") != CANDIDATE_PAPER_ID
        or composition_candidate.get("example_id") != CANDIDATE_EXAMPLE_ID
        or composition_candidate.get("row_sha256") != CANDIDATE_ROW_SHA256
        or composition_cohort
        != {
            "required_papers": 64,
            "retained_support_rows": 63,
            "authenticated_replacement_rows": 1,
            "science_train_rows": 16,
            "science_eval_rows": 32,
        }
        or composition_authorization
        != {
            "candidate_fixed": True,
            "provider_calls_authorized": False,
            "science_authorized": False,
            "launch_authorized": False,
            "support_spend_authorized": False,
            "exploratory_science_user_accepted": False,
            "readiness_blocker": "exploratory_science_not_user_accepted",
        }
        or audit.get("candidate_sha256") != hashes[CANDIDATE_RELATIVE]
        or audit.get("composition_sha256") != sha256_json(composition)
        or audit.get("protocol_sha256") != sha256_json(protocol)
        or not isinstance(protocol_selection, dict)
        or protocol_selection.get("manifest_sha256") != SELECTION_MANIFEST_SHA256
        or protocol_selection.get("receipt_sha256") != SELECTION_RECEIPT_SHA256
        or protocol_selection.get("claim_sha256") != SELECTION_CLAIM_SHA256
        or protocol_selection.get("upstream_v12_hashes") != upstream["upstream_hashes"]
        or not isinstance(audit_selection, dict)
        or audit_selection.get("upstream_v12_hashes") != upstream["upstream_hashes"]
        or not isinstance(protocol_attempt_policy, dict)
        or protocol_attempt_policy.get("maximum_live_support_attempts_global") != 1
        or protocol_attempt_policy.get("outcome_bearing_cohorts") != 1
        or protocol_attempt_policy.get("second_outcome_bearing_attempt")
        != "forbidden_unconditionally"
        or not isinstance(protocol_authorization, dict)
        or protocol_authorization.get("provider_calls_authorized") is not False
        or protocol_authorization.get("model_calls_authorized") is not False
        or protocol_authorization.get("science_authorized") is not False
        or protocol_authorization.get("launch_authorized") is not False
        or protocol_authorization.get("format_only_sft_iteration_allowed") is not False
        or protocol_authorization.get("exploratory_science_user_accepted") is not False
        or protocol_authorization.get("support_spend_authorized") is not False
        or protocol.get("support_pass_transition")
        != "user_checkpoint_required_before_any_support_spend_or_science_transition"
        or protocol_authorization.get("readiness_blocker")
        != "exploratory_science_not_user_accepted"
        or audit.get("domain") != "redco-stage-d1-support-v13-protocol-audit-v1"
        or audit.get("ready_for_live_support") is not False
        or audit.get("live_activity_performed") is not False
        or audit.get("runtime") != runtime
    ):
        raise ValueError("published support artifact cross-references differ")
    return hashes

__all__ = ["check_protocol_artifacts"]
