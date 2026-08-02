"""Transitive evidence-closure verification for Stage-D evaluation ledgers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from redco.analysis.stage_d_evaluation_codec import (
    EvaluationEvidenceStore,
    canonical_object,
    decode_record,
)

EVIDENCE_FIELDS = {
    "process_receipt_sha256",
    "prior_process_receipt_sha256",
    "dead_process_evidence_sha256",
    "server_attestation_sha256",
    "event_address_sha256",
    "request_sha256",
    "transport_sha256",
    "dispatch_receipt_sha256",
    "response_envelope_sha256",
    "raw_response_sha256",
    "outcome_sha256",
    "terminal_result_sha256",
    "task_metrics_sha256",
    "arm_metrics_sha256",
    "supervisor_identity_sha256",
    "error_evidence_sha256",
    "cleanup_evidence_sha256",
}
_NESTED_FIELDS = {
    "redco-stage-d-evaluation-dead-client-v1": ("prior_process_receipt_sha256",),
    "redco-stage-d-evaluation-server-attestation-v1": (
        "process_receipt_sha256",
        "process_observation_sha256",
        "probe_response_sha256",
    ),
    "redco-stage-d-evaluation-transport-v1": ("body_sha256",),
    "redco-stage-d-evaluation-dispatch-v1": ("request_sha256", "transport_sha256"),
    "redco-stage-d-evaluation-response-envelope-v1": (
        "dispatch_receipt_sha256",
        "raw_response_sha256",
    ),
    "redco-stage-d-evaluation-call-outcome-v1": (
        "response_envelope_sha256",
        "parsed_response_sha256",
    ),
    "redco-stage-d-evaluation-task-metrics-v1": (
        "terminal_result_sha256",
        "scorer_evidence_sha256",
    ),
}


def reachable_evidence(
    records_root: Path,
    evidence: EvaluationEvidenceStore,
) -> tuple[str, ...]:
    roots: list[str] = []
    for path in sorted(records_root.glob("*.json")):
        record = decode_record(path.read_bytes())
        roots.extend(
            digest
            for name, digest in record["event"].items()
            if name in EVIDENCE_FIELDS and digest is not None
        )
    return tuple(sorted(verify_evidence_closure(evidence, roots)))


def verify_evidence_closure(
    evidence: EvaluationEvidenceStore,
    roots: Iterable[str],
) -> set[str]:
    pending = list(roots)
    visited: set[str] = set()
    while pending:
        digest = pending.pop()
        if digest in visited:
            continue
        value = evidence.get(digest)
        visited.add(digest)
        try:
            payload = canonical_object(value, "nested evaluation evidence")
        except ValueError:
            continue
        domain = payload.get("domain")
        if not isinstance(domain, str):
            continue
        for field in _NESTED_FIELDS.get(domain, ()):
            nested = payload.get(field)
            if (
                not isinstance(nested, str)
                or len(nested) != 64
                or any(character not in "0123456789abcdef" for character in nested)
            ):
                raise ValueError("nested evaluation evidence reference is invalid")
            pending.append(nested)
        if domain == "redco-stage-d-heldout-metrics-v1":
            examples = payload.get("examples")
            if not isinstance(examples, list):
                raise ValueError("held-out metrics evidence examples are invalid")
            for example in examples:
                if not isinstance(example, dict):
                    raise ValueError("held-out metrics evidence example is invalid")
                nested = example.get("raw_output_sha256")
                if not isinstance(nested, str):
                    raise ValueError("held-out metrics raw evidence reference is invalid")
                pending.append(nested)
    return visited


__all__ = ["EVIDENCE_FIELDS", "reachable_evidence", "verify_evidence_closure"]
