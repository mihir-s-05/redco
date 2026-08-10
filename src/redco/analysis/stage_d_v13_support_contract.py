"""Authenticated source/protocol inputs shared by v13 materialization and publication.

This module authenticates committed inputs and constructs the immutable contract
used by both the materializer and the read-only publication owner.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_source_producer import (
    SAMPLING_CONTRACT_SHA256,
    SAMPLING_CONTRACT_VERSION,
)
from redco.analysis.stage_d_v13_draft import sha256_bytes
from redco.analysis.stage_d_v13_source_phase_a_decoder import (
    SOURCE_ARTIFACT_RELATIVE,
    SOURCE_BYTES,
    SOURCE_FIELDS,
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
)
from redco.contracts import canonical_json
from redco.integrity import resolve_contained_file

CANDIDATE_SOURCE_ORDINAL = 180
CANDIDATE_PAPER_ID = "2001.09332"
CANDIDATE_EXAMPLE_ID = "qasper-9447ec36e397853c04dcb8f67492ca9f944dbd4b"
CANDIDATE_QUESTION_INDEX = 0
CANDIDATE_ROW_SHA256 = "fd2891a4f162f2c9f949ec08dcc24f60402fc0ba49ed5b17c3c0c5f08808d5a2"
CANDIDATE_SELECTION_ADDRESS_SHA256 = (
    "91ac240ddb2da96eb21cd969b648ab00c4d54811052d0e2c4798cab06ad9bc78"
)
SELECTION_RECEIPT_SHA256 = "a0ef52769f2ea2f81a23b5c9d74ef124024ff4ad2254ccd2aad7e7d810402102"
SELECTION_RECEIPT_RELATIVE = "reports/stage-d1-support-v13-source-selection-receipt-v2.json"
SELECTION_MANIFEST_RELATIVE = (
    "reports/stage-d1-support-v13-source-selection-evidence-manifest-v1.json"
)
SELECTION_MANIFEST_SHA256 = "4fde8cb1ef189b525a2585bb6f44e8888e47e4e17c66f1a4c4ca68934201cd02"
SELECTION_CLAIM_RELATIVE = "reports/stage-d1-support-v13-source-selection-claim-v1.json"
SELECTION_CLAIM_SHA256 = "240b30bc283c993bf007834f7ce7524a97a177da1678cddaeff04e60d7c8edac"
SELECTION_ORIGINAL_CLAIM_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-source-selection-claim-v1.json"
)
V12_ARCHIVE_RELATIVE = "runs/stage-d/stage-d1-support-v12-terminal.tar.gz"
V12_ARCHIVE_SHA256 = "c2bb6713234653fc08e10778aa17815ad5a26f769406806037f50c390820b894"
V12_EVIDENCE_MANIFEST_RELATIVE = "runs/stage-d/stage-d1-support-v12-evidence-sha256.txt"
V12_EVIDENCE_MANIFEST_SHA256 = "90c694a45ece9887ea658adc36a8b30f6b7d78f8b111c017f54b5e5d51003671"
V12_TERMINAL_REPORT_RELATIVE = "reports/stage-d1-support-v12-terminal.json"
V12_TERMINAL_REPORT_SHA256 = "3b79bbf541ee4210744b37f2df49531ca4e6b601f500e0ecaa43e3fa9e8ca9ec"
V12_FINALIZATION_AUDIT_RELATIVE = "reports/stage-d1-support-v12-finalization-audit-v1.json"
V12_FINALIZATION_AUDIT_SHA256 = "97f743f9dfee0c5f2988073dc00efc7a83765698ead298d89c9f9ae26714a588"
FROZEN_SUPPORT_RULES_RELATIVE = "configs/stage-d/stage-d1-support-rules-v1.json"
FROZEN_SUPPORT_RULES_SHA256 = "088e3990c270881435214c725c0ca984462f9a824a1e0708d1bcbebd4264d235"
RETAINED_SUPPORT_RELATIVE = "datasets/stage-d/qasper-support-successor-v7-draft-retained-only.jsonl"
RETAINED_SUPPORT_SHA256 = "c6f99a40578c44c20b3b703316440d54c29821911b20fc09426e0cb44e921d07"
COLLECTION_PLAN_RELATIVE = "configs/stage-d/stage-d1-support-collection-plan-v11.json"
COLLECTION_PLAN_SHA256 = "9870c18fd43ee3a08bb212d5f9b506104e39b3d2dc2ccd7b689240799d2696cb"
V6_MANIFEST_RELATIVE = "datasets/stage-d/qasper-support-successor-manifest-v6.json"
V6_MANIFEST_SHA256 = "5b1667fa9f17c7e733276b17534de6453598e60b8e52a733a2028c7dab671697"
ADDRESS_AUDIT_RELATIVE = "reports/stage-d1-support-successor-address-audit-v6.json"
ADDRESS_AUDIT_SHA256 = "8f0e081fe2b3ed8b15254ac5068569d9e0a493fc1e8b832d7956af43ed1f50dd"
AUTHENTICATED_PREDECESSOR_HASHES = {
    RETAINED_SUPPORT_RELATIVE: RETAINED_SUPPORT_SHA256,
    COLLECTION_PLAN_RELATIVE: COLLECTION_PLAN_SHA256,
    V6_MANIFEST_RELATIVE: V6_MANIFEST_SHA256,
    ADDRESS_AUDIT_RELATIVE: ADDRESS_AUDIT_SHA256,
}
V12_PREREG_RELATIVE = "configs/stage-d/stage-d1-support-preregistration-v12.json"
V12_PREREG_SHA256 = "8cf086e4b198c45306a0cb7d3289b72e65fa2ee9ae34940a42e141542003b429"
V12_PROTOCOL_RELATIVE = "configs/stage-d/stage-d1-support-protocol-v12.json"
V12_PROTOCOL_SHA256 = "2be6b64916ef3620dc15fade89916b616de1ea8f54db0109c7f0ff5c3be8e9fd"
V12_SOURCE_EVAL_RELATIVE = "configs/stage-d/stage-d1-support-source-eval-v12.toml"
V12_SOURCE_EVAL_SHA256 = "704eca5bb15a7ee52572653110639857dcffe422a2d5cfe1a66b498959e88351"
UPSTREAM_EVIDENCE_SHA256 = {
    V12_ARCHIVE_RELATIVE: V12_ARCHIVE_SHA256,
    V12_EVIDENCE_MANIFEST_RELATIVE: V12_EVIDENCE_MANIFEST_SHA256,
    V12_TERMINAL_REPORT_RELATIVE: V12_TERMINAL_REPORT_SHA256,
    V12_FINALIZATION_AUDIT_RELATIVE: V12_FINALIZATION_AUDIT_SHA256,
    V12_PREREG_RELATIVE: V12_PREREG_SHA256,
    V12_PROTOCOL_RELATIVE: V12_PROTOCOL_SHA256,
    V12_SOURCE_EVAL_RELATIVE: V12_SOURCE_EVAL_SHA256,
    FROZEN_SUPPORT_RULES_RELATIVE: FROZEN_SUPPORT_RULES_SHA256,
    **AUTHENTICATED_PREDECESSOR_HASHES,
}
SOURCE_FREE_OPTIONAL_EVIDENCE = frozenset(
    {
        SELECTION_ORIGINAL_CLAIM_RELATIVE,
        V12_ARCHIVE_RELATIVE,
        V12_EVIDENCE_MANIFEST_RELATIVE,
    }
)
SUPPORT_RULES_SHA256 = "088e3990c270881435214c725c0ca984462f9a824a1e0708d1bcbebd4264d235"
MASTER_SEED = "redco-stage-d1-support-v1-20260802-78b65e4cc16ac31f"
SCIENTIFIC_NAMESPACE = "redco-stage-d1-support-v1"

CANDIDATE_RELATIVE = "datasets/stage-d/qasper-support-successor-candidate-ordinal-180-v1.json"
COMPOSITION_RELATIVE = (
    "datasets/stage-d/qasper-support-successor-v8-candidate-composition-manifest-v1.json"
)
PROTOCOL_RELATIVE = "configs/stage-d/v13-draft/stage-d1-support-v13-frozen-support-protocol-v1.json"
PROTOCOL_AUDIT_RELATIVE = "reports/stage-d1-support-v13-protocol-audit-v1.json"
SAMPLING_CONTRACT_SOURCE_RELATIVE = "src/redco/analysis/stage_d_source_producer.py"
ACTION_CLOSURE_RELATIVE = "configs/stage-d/stage-d1-action-closure-corpus-v2.json"
ACTION_CLOSURE_SHA256 = "50152ebbaea6cecce63c167c13d56050c4feb50782f838d69ea34840b29670c0"
ACTION_CLOSURE_AUDIT_RELATIVE = "reports/stage-d1-action-closure-corpus-audit-v2.json"
ACTION_CLOSURE_AUDIT_SHA256 = "60631a5153c2434682642f5aecaf5f55e61f368c8b96d721982eea7c9c158646"
LAUNCH_AUTHORIZATION_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-v13-launch-authorization-v1.json"
)
LAUNCH_AUTHORIZATION_SHA256 = "30020b15b5929af1bf668de1bd6b3eb15fe068ec86b24d2dc9a05a8b3b72a7be"
CANDIDATE_AUTHORITY = {
    "candidate_materialized": True,
    "source_selection_repeated": False,
    "provider_calls_authorized": False,
    "model_calls_authorized": False,
    "science_authorized": False,
    "launch_authorized": False,
}
SUPPORT_COHORT = {
    "required_papers": 64,
    "retained_support_rows": 63,
    "authenticated_replacement_rows": 1,
    "science_train_rows": 16,
    "science_eval_rows": 32,
}
COMPOSITION_AUTHORIZATION = {
    "candidate_fixed": True,
    "provider_calls_authorized": False,
    "science_authorized": False,
    "launch_authorized": False,
    "support_spend_authorized": False,
    "exploratory_science_user_accepted": False,
    "readiness_blocker": "exploratory_science_not_user_accepted",
}
PROTOCOL_AUTHORIZATION = {
    "provider_calls_authorized": False,
    "model_calls_authorized": False,
    "science_authorized": False,
    "launch_authorized": False,
    "format_only_sft_iteration_allowed": False,
    "exploratory_science_user_accepted": False,
    "support_spend_authorized": False,
    "readiness_blocker": "exploratory_science_not_user_accepted",
}


def sampling_contract_binding(root: Path) -> dict[str, str]:
    source = resolve_contained_file(root, SAMPLING_CONTRACT_SOURCE_RELATIVE)
    if source is None:
        raise ValueError("sampling contract source is missing")
    return {
        "version": SAMPLING_CONTRACT_VERSION,
        "sha256": SAMPLING_CONTRACT_SHA256,
        "producer_source_path": SAMPLING_CONTRACT_SOURCE_RELATIVE,
        "producer_source_sha256": sha256_bytes(source.read_bytes()),
    }


def _decoder_contract() -> dict[str, object]:
    return {
        "batch_size": 1, "use_threads": False,
        "row_groups": [0], "logical_readahead": False,
        "physical_compressed_page_io_may_span_row_group": True,
    }


def protocol_source_binding() -> dict[str, object]:
    """Return the immutable source projection embedded in the reviewed protocol."""

    return {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "logical_url": SOURCE_LOGICAL_URL,
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
        "decoder": _decoder_contract(),
        "required_runtime": {
            "python": SUPPORTED_PYTHON,
            "pyarrow": SUPPORTED_PYARROW,
            "datasets": SUPPORTED_DATASETS,
        },
    }


# Independent reviewed bytes for the current candidate-null protocol set.
# These are deliberately code-owned constants, rather than values read from
# the audit artifact being checked, so a coordinated cross-hash rewrite cannot
# bless a structural mutation.
REVIEWED_PROTOCOL_ARTIFACT_SHA256 = {
    CANDIDATE_RELATIVE: "3df14acf9bf5f71736511aa9115f5e49ceab14a191bc1b634e3f82f21ca3f4a1",
    COMPOSITION_RELATIVE: "3cb26d9aec634e96fb342f87ea807711ed943a64073a9e37c5b7a546294638bc",
    PROTOCOL_RELATIVE: "65734f3dc5caeb1866e25b535d5b91d17ffbc434b69fbe0baf5efe63d339145b",
    PROTOCOL_AUDIT_RELATIVE: "6df3b0e98aa0ca27b72c7abd443cfbf003f7f99e0ce1880ec2e8b1fd3801d2f3",
}
LAUNCH_PREDECESSOR_BINDINGS = {
    ACTION_CLOSURE_RELATIVE: ACTION_CLOSURE_SHA256,
    ACTION_CLOSURE_AUDIT_RELATIVE: ACTION_CLOSURE_AUDIT_SHA256,
    **REVIEWED_PROTOCOL_ARTIFACT_SHA256,
}


@dataclass(slots=True)
class CandidateReadInstrumentation:
    arrow_batch_cardinalities: list[int] = field(default_factory=list)
    arrow_batch_ranges: list[tuple[int, int]] = field(default_factory=list)
    requested_ordinals: list[int] = field(default_factory=list)
    materialized_ordinals: list[int] = field(default_factory=list)
    canonicalized_ordinals: list[int] = field(default_factory=list)
    evaluated_ordinals: list[int] = field(default_factory=list)

    def record_arrow_batch(self, first_ordinal: int, cardinality: int) -> None:
        last_ordinal = first_ordinal + cardinality - 1
        expected_first = sum(self.arrow_batch_cardinalities)
        if (
            cardinality < 1
            or first_ordinal != expected_first
            or last_ordinal > CANDIDATE_SOURCE_ORDINAL
        ):
            raise RuntimeError("candidate decoder emitted an Arrow batch beyond ordinal 180")
        self.arrow_batch_cardinalities.append(cardinality)
        self.arrow_batch_ranges.append((first_ordinal, last_ordinal))

    def record_request(self, ordinal: int) -> None:
        if ordinal != len(self.requested_ordinals) or ordinal > CANDIDATE_SOURCE_ORDINAL:
            raise RuntimeError("candidate materializer requested ordinal 181 or later")
        self.requested_ordinals.append(ordinal)

    def record_materialized(self, ordinal: int) -> None:
        if ordinal > CANDIDATE_SOURCE_ORDINAL:
            raise RuntimeError("candidate materializer materialized ordinal 181 or later")
        self.materialized_ordinals.append(ordinal)

    def record_canonicalized(self, ordinal: int) -> None:
        if ordinal > CANDIDATE_SOURCE_ORDINAL:
            raise RuntimeError("candidate materializer canonicalized ordinal 181 or later")
        self.canonicalized_ordinals.append(ordinal)

    def record_evaluated(self, ordinal: int) -> None:
        if ordinal > CANDIDATE_SOURCE_ORDINAL:
            raise RuntimeError("candidate materializer evaluated ordinal 181 or later")
        self.evaluated_ordinals.append(ordinal)

    def to_payload(self) -> dict[str, Any]:
        return {
            "arrow_batch_cardinalities": list(self.arrow_batch_cardinalities),
            "arrow_batch_ranges": [list(value) for value in self.arrow_batch_ranges],
            "arrow_logical_rows_emitted": sum(self.arrow_batch_cardinalities),
            "requested_ordinals": list(self.requested_ordinals),
            "materialized_ordinals": list(self.materialized_ordinals),
            "canonicalized_ordinals": list(self.canonicalized_ordinals),
            "evaluated_ordinals": list(self.evaluated_ordinals),
            "python_converted_ordinals": list(self.materialized_ordinals),
            "requested_last": self.requested_ordinals[-1] if self.requested_ordinals else None,
            "post_180_requested": any(item > 180 for item in self.requested_ordinals),
            "post_180_materialized": any(item > 180 for item in self.materialized_ordinals),
            "post_180_canonicalized": any(item > 180 for item in self.canonicalized_ordinals),
            "post_180_evaluated": any(item > 180 for item in self.evaluated_ordinals),
        }


def load_parquet(path: Path) -> Any:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyArrow is required for the candidate materializer") from error
    return parquet.ParquetFile(path, memory_map=True)


def runtime_payload() -> dict[str, str | bool]:
    try:
        import pyarrow
    except ImportError:
        return {
            "python": ".".join(map(str, sys.version_info[:3])),
            "pyarrow": "missing",
            "datasets": "missing",
            "supported": False,
        }
    try:
        import datasets

        datasets_version = str(getattr(datasets, "__version__", ""))
    except ImportError:
        datasets_version = "missing"
    python_version = ".".join(map(str, sys.version_info[:3]))
    return {
        "python": python_version,
        "pyarrow": str(pyarrow.__version__),
        "datasets": datasets_version,
        "supported": (
            python_version == SUPPORTED_PYTHON
            and str(pyarrow.__version__) == SUPPORTED_PYARROW
            and datasets_version == SUPPORTED_DATASETS
        ),
    }


def require_supported_runtime() -> dict[str, str | bool]:
    runtime = runtime_payload()
    if runtime.get("supported") is not True:
        raise RuntimeError(
            "authenticated candidate materialization requires Python 3.12.3, "
            "PyArrow 25.0.0, and datasets 5.0.0"
        )
    return runtime


def source_contract(root: Path, parquet_path: Path) -> tuple[dict[str, Any], Any]:
    del root
    runtime = require_supported_runtime()
    raw = parquet_path.read_bytes()
    if len(raw) != SOURCE_BYTES or sha256_bytes(raw) != SOURCE_SHA256:
        raise ValueError("authenticated QASPER source bytes differ")
    parquet_file = load_parquet(parquet_path)
    metadata = parquet_file.metadata
    if metadata is None or metadata.num_rows != SOURCE_ROW_COUNT or metadata.num_row_groups != 1:
        raise ValueError("authenticated QASPER Parquet metadata differs")
    schema_sha = sha256_bytes(str(parquet_file.schema_arrow).encode("utf-8"))
    if schema_sha != SOURCE_SCHEMA_SHA256:
        raise ValueError("authenticated QASPER schema differs")
    return {
        "repository": SOURCE_REPOSITORY,
        "logical_url": SOURCE_LOGICAL_URL,
        "revision": SOURCE_REVISION,
        "semantic_source_commit": SOURCE_SEMANTIC_COMMIT,
        "path": SOURCE_PATH,
        "local_artifact": SOURCE_ARTIFACT_RELATIVE,
        "bytes": len(raw),
        "sha256": SOURCE_SHA256,
        "schema_sha256": SOURCE_SCHEMA_SHA256,
        "row_count": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "fields": list(SOURCE_FIELDS),
        "decoder": _decoder_contract(),
        "runtime": runtime,
    }, parquet_file


def read_authenticated(root: Path, relative: str, expected_sha256: str) -> bytes:
    path = resolve_contained_file(root, relative)
    if path is None:
        raise ValueError(f"authenticated upstream evidence is missing: {relative}")
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"authenticated upstream evidence changed: {relative}")
    return raw


def authenticate_upstream_evidence(root: Path) -> dict[str, Any]:
    """Authenticate committed selection/v12 evidence before candidate access."""

    receipt_raw = read_authenticated(root, SELECTION_RECEIPT_RELATIVE, SELECTION_RECEIPT_SHA256)
    manifest_raw = read_authenticated(root, SELECTION_MANIFEST_RELATIVE, SELECTION_MANIFEST_SHA256)
    mirror_raw = read_authenticated(root, SELECTION_CLAIM_RELATIVE, SELECTION_CLAIM_SHA256)
    original_raw = read_authenticated(
        root,
        SELECTION_ORIGINAL_CLAIM_RELATIVE,
        SELECTION_CLAIM_SHA256,
    )
    if mirror_raw != original_raw:
        raise ValueError("selection claim mirror differs from the immutable original claim")
    try:
        receipt = json.loads(receipt_raw)
        manifest = json.loads(manifest_raw)
        claim = json.loads(mirror_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("selection evidence is not valid JSON") from error
    receipt_fields = {
        "attempt",
        "attempt_limit",
        "candidate",
        "claim_path",
        "claim_sha256",
        "disposition",
        "domain",
        "gate_commit",
        "launch_authorized",
        "model_calls_authorized",
        "phase_b_source_selection_authorized",
        "prime_gpu_scientific_launch_authorized",
        "provider_calls_authorized",
        "receipt_path",
        "retry",
        "scan_id",
        "schema_version",
        "science_authorized",
        "state",
        "stop_ordinal",
        "transcript_sha256",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != receipt_fields
        or canonical_json(receipt) != receipt_raw
        or receipt.get("schema_version") != 2
        or receipt.get("domain") != "redco-stage-d1-support-v13-source-selection-receipt-v2"
        or receipt.get("state") != "terminal"
        or receipt.get("disposition") != "eligible_candidate"
        or receipt.get("attempt") != 1
        or receipt.get("attempt_limit") != 1
        or receipt.get("retry") is not False
        or receipt.get("stop_ordinal") != CANDIDATE_SOURCE_ORDINAL
        or receipt.get("claim_path") != SELECTION_ORIGINAL_CLAIM_RELATIVE
        or receipt.get("claim_sha256") != SELECTION_CLAIM_SHA256
        or receipt.get("receipt_path") != SELECTION_RECEIPT_RELATIVE
        or receipt.get("phase_b_source_selection_authorized") is not True
        or any(
            receipt.get(name) is not False
            for name in (
                "launch_authorized",
                "model_calls_authorized",
                "prime_gpu_scientific_launch_authorized",
                "provider_calls_authorized",
                "science_authorized",
            )
        )
    ):
        raise ValueError("selection receipt is outside the frozen terminal contract")
    candidate = receipt.get("candidate")
    expected_candidate = {
        "address_sha256": CANDIDATE_SELECTION_ADDRESS_SHA256,
        "example_id": CANDIDATE_EXAMPLE_ID,
        "paper_id": CANDIDATE_PAPER_ID,
        "question_index": CANDIDATE_QUESTION_INDEX,
        "source_ordinal": CANDIDATE_SOURCE_ORDINAL,
        "source_row_sha256": CANDIDATE_ROW_SHA256,
    }
    if candidate != expected_candidate:
        raise ValueError("selection receipt candidate differs from the frozen candidate")
    if (
        not isinstance(claim, dict)
        or canonical_json(claim) != mirror_raw
        or claim.get("scan_id") != receipt.get("scan_id")
        or claim.get("gate_commit") != receipt.get("gate_commit")
        or claim.get("candidate") is not None
        or claim.get("address") is not None
        or claim.get("seed") is not None
    ):
        raise ValueError("selection claim is not the matching pre-action claim")
    if not isinstance(manifest, dict) or canonical_json(manifest) != manifest_raw:
        raise ValueError("selection evidence manifest is not canonical")
    receipt_binding = manifest.get("receipt")
    claim_binding = manifest.get("claim")
    if (
        not isinstance(receipt_binding, dict)
        or not isinstance(claim_binding, dict)
        or receipt_binding.get("path") != SELECTION_RECEIPT_RELATIVE
        or receipt_binding.get("sha256") != SELECTION_RECEIPT_SHA256
        or claim_binding.get("sha256") != SELECTION_CLAIM_SHA256
        or manifest.get("attempt")
        != {"consumed": 1, "no_row_after_ordinal": CANDIDATE_SOURCE_ORDINAL, "retry": False}
        or manifest.get("candidate") != expected_candidate
    ):
        raise ValueError("selection evidence manifest does not bind the receipt")

    authenticated_predecessors: dict[str, bytes] = {}
    upstream_hashes: dict[str, str] = {}
    for relative, expected in UPSTREAM_EVIDENCE_SHA256.items():
        raw = read_authenticated(root, relative, expected)
        upstream_hashes[relative] = sha256_bytes(raw)
        if relative in AUTHENTICATED_PREDECESSOR_HASHES or relative in {
            V12_PROTOCOL_RELATIVE,
            FROZEN_SUPPORT_RULES_RELATIVE,
        }:
            authenticated_predecessors[relative] = raw
    protocol_raw = authenticated_predecessors[V12_PROTOCOL_RELATIVE]
    rules_raw = authenticated_predecessors[FROZEN_SUPPORT_RULES_RELATIVE]
    try:
        protocol = json.loads(protocol_raw)
        rules = json.loads(rules_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("frozen support protocol/rules are not JSON") from error
    if (
        not isinstance(protocol, dict)
        or protocol.get("decision_rule_sha256")
        != "792decd5e6887efd494d2dba40d8ac00ff0fd243f72ff171a371d4cb7eb87306"
        or not isinstance(rules, dict)
        or rules.get("domain") != "redco-stage-d-support-rules-v1"
        or rules.get("schema_version") != 1
        or rules.get("required_papers") != 64
        or rules.get("required_successes") != 58
        or rules.get("minimum_reward_range") != 0.05
    ):
        raise ValueError("frozen support-rule or decision binding differs")
    return {
        "selection_receipt": receipt,
        "selection_manifest_sha256": SELECTION_MANIFEST_SHA256,
        "selection_claim_sha256": SELECTION_CLAIM_SHA256,
        "upstream_hashes": upstream_hashes,
        "authenticated_predecessors": authenticated_predecessors,
        "decision_rule_sha256": protocol["decision_rule_sha256"],
        "support_rules_sha256": FROZEN_SUPPORT_RULES_SHA256,
    }


__all__ = [
    "ACTION_CLOSURE_AUDIT_RELATIVE",
    "ACTION_CLOSURE_AUDIT_SHA256",
    "ACTION_CLOSURE_RELATIVE",
    "ACTION_CLOSURE_SHA256",
    "ADDRESS_AUDIT_RELATIVE",
    "ADDRESS_AUDIT_SHA256",
    "AUTHENTICATED_PREDECESSOR_HASHES",
    "CANDIDATE_AUTHORITY",
    "CANDIDATE_EXAMPLE_ID",
    "CANDIDATE_PAPER_ID",
    "CANDIDATE_QUESTION_INDEX",
    "CANDIDATE_RELATIVE",
    "CANDIDATE_ROW_SHA256",
    "CANDIDATE_SELECTION_ADDRESS_SHA256",
    "CANDIDATE_SOURCE_ORDINAL",
    "COLLECTION_PLAN_RELATIVE",
    "COLLECTION_PLAN_SHA256",
    "COMPOSITION_AUTHORIZATION",
    "COMPOSITION_RELATIVE",
    "FROZEN_SUPPORT_RULES_RELATIVE",
    "FROZEN_SUPPORT_RULES_SHA256",
    "LAUNCH_AUTHORIZATION_RELATIVE",
    "LAUNCH_AUTHORIZATION_SHA256",
    "LAUNCH_PREDECESSOR_BINDINGS",
    "MASTER_SEED",
    "PROTOCOL_AUDIT_RELATIVE",
    "PROTOCOL_AUTHORIZATION",
    "PROTOCOL_RELATIVE",
    "RETAINED_SUPPORT_RELATIVE",
    "RETAINED_SUPPORT_SHA256",
    "REVIEWED_PROTOCOL_ARTIFACT_SHA256",
    "SAMPLING_CONTRACT_SOURCE_RELATIVE",
    "SCIENTIFIC_NAMESPACE",
    "SELECTION_CLAIM_RELATIVE",
    "SELECTION_CLAIM_SHA256",
    "SELECTION_MANIFEST_RELATIVE",
    "SELECTION_MANIFEST_SHA256",
    "SELECTION_ORIGINAL_CLAIM_RELATIVE",
    "SELECTION_RECEIPT_RELATIVE",
    "SELECTION_RECEIPT_SHA256",
    "SOURCE_ARTIFACT_RELATIVE",
    "SOURCE_BYTES",
    "SOURCE_FIELDS",
    "SOURCE_FREE_OPTIONAL_EVIDENCE",
    "SOURCE_LOGICAL_URL",
    "SOURCE_PATH",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "SOURCE_ROW_COUNT",
    "SOURCE_SCHEMA_SHA256",
    "SOURCE_SEMANTIC_COMMIT",
    "SOURCE_SHA256",
    "SUPPORTED_DATASETS",
    "SUPPORTED_PYARROW",
    "SUPPORTED_PYTHON",
    "SUPPORT_COHORT",
    "SUPPORT_RULES_SHA256",
    "UPSTREAM_EVIDENCE_SHA256",
    "V6_MANIFEST_RELATIVE",
    "V6_MANIFEST_SHA256",
    "V12_ARCHIVE_RELATIVE",
    "V12_ARCHIVE_SHA256",
    "V12_EVIDENCE_MANIFEST_RELATIVE",
    "V12_EVIDENCE_MANIFEST_SHA256",
    "V12_FINALIZATION_AUDIT_RELATIVE",
    "V12_FINALIZATION_AUDIT_SHA256",
    "V12_PREREG_RELATIVE",
    "V12_PREREG_SHA256",
    "V12_PROTOCOL_RELATIVE",
    "V12_PROTOCOL_SHA256",
    "V12_SOURCE_EVAL_RELATIVE",
    "V12_SOURCE_EVAL_SHA256",
    "V12_TERMINAL_REPORT_RELATIVE",
    "V12_TERMINAL_REPORT_SHA256",
    "CandidateReadInstrumentation",
    "authenticate_upstream_evidence",
    "load_parquet",
    "protocol_source_binding",
    "read_authenticated",
    "require_supported_runtime",
    "runtime_payload",
    "sampling_contract_binding",
    "source_contract",
]
