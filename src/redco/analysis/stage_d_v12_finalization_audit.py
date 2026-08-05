"""Read-only, fail-closed engineering audit of the terminal Stage-D v12 trace.

The public ``audit_archive`` and ``main`` entry points remain here as a small
facade.  Immutable-input authentication, durable action/call checks, and
disposable production-semantic reconstruction live in focused sibling modules.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_receipt_ledger import inspect_ledger
from redco.analysis.stage_d_v12_audit_common import (
    _POST_CALL_INVARIANT_NAMES,
    _SEMANTIC_RECONSTRUCTION_NAMES,
    _STATUS,
    ARCHIVE_SHA256,
    AUDIT_DOMAIN,
    AUDIT_SCHEMA_VERSION,
    EVIDENCE_MANIFEST_SHA256,
    FROZEN_ARCHIVE_ROOT_RELATIVE,
    FROZEN_REPO_FILE_SHA256,
    FROZEN_RUNTIME_CODE_COMMIT,
    FROZEN_TRACE_ID,
    KNOWN_FAILURE_DECISION_ID,
    KNOWN_FAILURE_LINEAGE,
    KNOWN_FAILURE_NODE,
    TERMINAL_REPORT_SHA256,
    _mapping,
    _require_sha256,
    _status,
    sha256_file,
)
from redco.analysis.stage_d_v12_audit_inputs import (
    _authenticate_inputs,
    _error_evidence,
    _load_json,
    _manifest_audit,
    _receipt_records,
    _reject_output_alias,
    _safe_extract,
    _source_hashes,
    _validate_output_path,
    _validate_terminal_report_schema,
)
from redco.analysis.stage_d_v12_audit_semantics import _semantic_reconstruction
from redco.analysis.stage_d_v12_audit_trace import (
    _action_by_address,
    _address_key,
    _call_audit,
    _post_call_invariants,
)
from redco.contracts import canonical_json

__all__ = [
    "ARCHIVE_SHA256",
    "EVIDENCE_MANIFEST_SHA256",
    "FROZEN_ARCHIVE_ROOT_RELATIVE",
    "FROZEN_REPO_FILE_SHA256",
    "_reject_output_alias",
    "_require_sha256",
    "_source_hashes",
    "_validate_output_path",
    "_validate_terminal_report_schema",
    "audit_archive",
    "main",
]


def _checklist_complete(checks: list[dict[str, Any]], expected: tuple[str, ...]) -> bool:
    names = [item.get("name") for item in checks]
    return names == list(expected) and all(
        item.get("status")
        in {
            "pass",
            "fail",
            "not_observable_from_persisted_schema",
            "reconstructed_on_disposable_copy",
        }
        for item in checks
    )


def audit_archive(
    archive: Path,
    evidence_manifest: Path,
    *,
    repo_root: Path,
    terminal_report: Path,
) -> dict[str, Any]:
    """Audit authenticated terminal bytes using disposable extraction only."""
    archive_path, manifest_path, source_hashes, terminal_report_payload = _authenticate_inputs(
        archive,
        evidence_manifest,
        repo_root,
        terminal_report,
    )
    with tempfile.TemporaryDirectory(prefix="redco-v12-audit-") as temporary:
        extracted_parent = Path(temporary)
        root = _safe_extract(archive_path, extracted_parent)
        manifest_result = _manifest_audit(root, manifest_path)
        trace_payload = _mapping(_load_json(root / "source-eval" / "traces.jsonl"), "trace payload")
        traces = trace_payload.get("traces")
        if not isinstance(traces, list) or len(traces) != 1:
            raise ValueError("terminal archive must contain exactly one source trace")
        trace = _mapping(traces[0], "terminal trace")
        calls_raw = trace.get("calls")
        nodes_raw = trace.get("nodes")
        if not isinstance(calls_raw, list) or not isinstance(nodes_raw, list):
            raise ValueError("terminal trace calls/nodes are not lists")
        calls = [_mapping(call, "trace call") for call in calls_raw]
        nodes = [_mapping(node, "trace node") for node in nodes_raw]
        if trace.get("id") != FROZEN_TRACE_ID:
            raise ValueError("terminal trace identity differs from the frozen v12 trace")
        action_map, action_hash_checks = _action_by_address(root)
        call_addresses = {
            _address_key(_mapping(call.get("rlm"), "trace call rlm")) for call in calls
        }
        if call_addresses != set(action_map):
            raise ValueError("durable action/address mapping is not a call bijection")
        call_results = [
            _call_audit(
                trace,
                call,
                call_index=index,
                nodes=nodes,
                action_entry=action_map[_address_key(_mapping(call.get("rlm"), "trace call rlm"))],
            )
            for index, call in enumerate(calls)
        ]

        records = _receipt_records(root)
        ledger_scan = inspect_ledger(root / "ledger")
        durable_evidence_count = sum(
            path.is_file() for path in (root / "ledger" / "evidence").glob("*")
        )
        abort_receipts = [
            _mapping(record.get("body", {}).get("receipt"), "abort receipt")
            for record in records
            if record.get("record_kind") == "receipt"
            and record.get("body", {}).get("receipt", {}).get("receipt_kind")
            == "source_rollout_finalization_aborted"
        ]
        completion_receipts = [
            record
            for record in records
            if record.get("record_kind") == "receipt"
            and record.get("body", {}).get("receipt", {}).get("receipt_kind")
            == "source_rollout_completed"
        ]
        if len(abort_receipts) != 1 or completion_receipts:
            raise ValueError("terminal source finalization receipts are inconsistent")
        abort_error = _error_evidence(root, abort_receipts[0])
        source_counts = (
            sum(path.is_file() for path in (root / "source-artifacts" / "sources").glob("*")),
            sum(path.is_file() for path in (root / "source-artifacts" / "pending").glob("*")),
        )
        trace_manifest_entry = next(
            (
                entry
                for entry in manifest_result["entries"]
                if entry["path"] == "stage-d1-support-v12/source-eval/traces.jsonl"
            ),
            None,
        )
        trace_artifact = root / "source-eval" / "traces.jsonl"
        trace_artifact_hash_status: _STATUS = (
            "pass"
            if isinstance(trace_manifest_entry, dict)
            and trace_artifact.is_file()
            and trace_manifest_entry.get("matches") is True
            and trace_manifest_entry.get("actual_sha256") == sha256_file(trace_artifact)
            else "fail"
        )
        semantic_reconstruction = _semantic_reconstruction(
            trace,
            calls,
            action_map,
            records,
        )
        post_call = _post_call_invariants(
            trace,
            calls,
            nodes,
            action_map,
            str(ledger_scan.status),
            ledger_scan.reason,
            abort_receipts,
            source_counts,
            trace_artifact_hash_status,
            "pass" if abort_error.get("status") == "pass" else "fail",
        )
        post_call_invariants_executed = _checklist_complete(post_call, _POST_CALL_INVARIANT_NAMES)
        post_call_status: _STATUS = (
            "pass"
            if post_call_invariants_executed and all(item["status"] == "pass" for item in post_call)
            else "fail"
        )
        semantic_reconstruction_executed = (
            _checklist_complete(semantic_reconstruction["checks"], _SEMANTIC_RECONSTRUCTION_NAMES)
            and semantic_reconstruction["status"] == "pass"
            and all(item.get("result") == "pass" for item in semantic_reconstruction["checks"])
        )
        downstream_status: _STATUS = (
            "pass"
            if semantic_reconstruction_executed
            and semantic_reconstruction["status"] == "pass"
            and post_call_status == "pass"
            else "fail"
        )
        completed_decision_ids = {
            _mapping(record.get("body", {}).get("receipt"), "completed receipt").get("decision_id")
            for record in records
            if record.get("record_kind") == "receipt"
            and record.get("body", {}).get("receipt", {}).get("receipt_kind")
            == "source_policy_call_completed"
        }
        abort_decision_ids = set(abort_receipts[0].get("decision_ids", []))
        abort_receipt_status: _STATUS = (
            "pass"
            if abort_receipts[0].get("phase") == "source_finalization"
            and abort_decision_ids == completed_decision_ids
            and abort_error["status"] == "pass"
            else "fail"
        )
        production_status = (
            "aborted"
            if abort_receipt_status == "pass" and not completion_receipts
            else "inconsistent"
        )
        ledger_chain_status: _STATUS = (
            "pass"
            if str(ledger_scan.status) == "poisoned"
            and ledger_scan.reason == "ledger records an aborted source rollout finalization"
            else "fail"
        )
        terminal_evidence = _mapping(terminal_report_payload.get("evidence"), "terminal evidence")
        live_observation = _mapping(
            terminal_report_payload.get("live_observation"), "terminal live_observation"
        )
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "domain": AUDIT_DOMAIN,
            "status": "engineering_audit_only",
            "scientific_interpretation": (
                "This output is engineering audit evidence, not a recovered v12 source, "
                "support observation, target roster, branch outcome, or scientific conclusion. "
                "Archived-schema non-observability is not a scientific or integration failure."
            ),
            "inputs_untouched": True,
            "status_definitions": {
                "pass": "The frozen durable evidence directly satisfies the named invariant.",
                "fail": "The frozen durable evidence directly contradicts the named invariant.",
                "not_observable_from_persisted_schema": (
                    "A nullable field was omitted by the persisted Verifiers schema; this is "
                    "not evidence of a v12 scientific or integration failure."
                ),
                "directly_verified_from_archive": (
                    "Checked from immutable archive bytes or durable receipts."
                ),
                "reconstructed_on_disposable_copy": (
                    "Reconstructed only in temporary extraction; no frozen input "
                    "or source artifact changed."
                ),
            },
            "hashes": {
                "archive_sha256": sha256_file(archive_path),
                "evidence_manifest_sha256": sha256_file(manifest_path),
                "terminal_report_sha256": TERMINAL_REPORT_SHA256,
                "runtime_code_commit": FROZEN_RUNTIME_CODE_COMMIT,
                "protocol_sha256": source_hashes[
                    "configs/stage-d/stage-d1-support-protocol-v12.json"
                ],
                "preregistration_sha256": source_hashes[
                    "configs/stage-d/stage-d1-support-preregistration-v12.json"
                ],
                "runtime_and_source_sha256": source_hashes,
            },
            "archive_manifest": manifest_result,
            "terminal_trace": {
                "id": trace.get("id"),
                "stop_condition": trace.get("stop_condition"),
                "is_completed": trace.get("is_completed"),
                "ok": trace.get("ok"),
                "call_count": len(calls),
                "node_count": len(nodes),
                "root_call_count": sum(
                    _mapping(call.get("rlm"), "trace call rlm").get("depth") == 0 for call in calls
                ),
                "child_call_count": sum(
                    _mapping(call.get("rlm"), "trace call rlm").get("depth") == 1 for call in calls
                ),
                "evidence_file_count": terminal_evidence.get("evidence_file_count"),
                "model_call_count": live_observation.get("model_calls"),
            },
            "known_failure": {
                "decision_id": KNOWN_FAILURE_DECISION_ID,
                "lineage": KNOWN_FAILURE_LINEAGE,
                "node": KNOWN_FAILURE_NODE,
                "field": "/content",
                "transport_presence": "present-null",
                "trace_presence": "absent",
                "engineering_only": True,
            },
            "calls": call_results,
            "action_evidence_hashes": action_hash_checks,
            "ledger": {
                "status": str(ledger_scan.status),
                "reason": ledger_scan.reason,
                "record_count": len(records),
                "evidence_count": durable_evidence_count,
                "chain_and_poison_invariant": _status(
                    "ledger_chain_and_poison",
                    ledger_chain_status,
                    "directly_verified_from_archive",
                ),
            },
            "downstream_audit": {
                "status": downstream_status,
                "call_count_audited": len(call_results),
                "post_call_invariants_executed": post_call_invariants_executed,
                "semantic_reconstruction_executed": semantic_reconstruction_executed,
            },
            "semantic_reconstruction": semantic_reconstruction,
            "post_call_invariants": post_call,
            "source_finalization": {
                "production_status": production_status,
                "abort_receipt_count": len(abort_receipts),
                "completion_receipt_count": len(completion_receipts),
                "abort_receipt": {
                    "status": abort_receipt_status,
                    "phase": abort_receipts[0].get("phase"),
                    "error_sha256": abort_error["error_sha256"],
                    "decision_count": len(abort_receipts[0].get("decision_ids", [])),
                },
                "error_evidence": abort_error,
                "committed_source_artifacts": source_counts[0],
                "pending_source_artifacts": source_counts[1],
                "failure_message": abort_error["error_message"],
            },
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--terminal-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = _validate_output_path(args.output, args.repo_root)
    result = audit_archive(
        args.archive,
        args.evidence_manifest,
        repo_root=args.repo_root,
        terminal_report=args.terminal_report,
    )
    encoded = canonical_json(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    print(encoded.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
