"""Read-only post-repair verification partitioned from the frozen v1 audit."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_v12_audit_common import (
    ARCHIVE_SHA256,
    EVIDENCE_MANIFEST_SHA256,
    FROZEN_ARCHIVE_RELATIVE,
    FROZEN_MANIFEST_RELATIVE,
    FROZEN_REPO_FILE_SHA256,
    FROZEN_REPORT_RELATIVE,
    TERMINAL_REPORT_SHA256,
    sha256_file,
)
from redco.analysis.stage_d_v12_audit_inputs import (
    _authenticate_terminal_report,
    _manifest_audit,
    _safe_extract,
    _validate_output_path,
)
from redco.analysis.stage_d_v12_finalization_audit import audit_archive
from redco.contracts import canonical_json

POST_REPAIR_SCHEMA_VERSION = 1
POST_REPAIR_DOMAIN = "redco-stage-d1-v12-post-repair-engineering-audit-v1"
SOURCE_RELATIVE = "src/redco/analysis/stage_d_source_producer.py"
APPROVED_REPAIRED_SOURCE_SHA256 = (
    "2e13f156b9dd078ce02bb06eeeb9a69f122b8fe25c48c05abce3290d702ee522"
)


def audit_postrepair(repo_root: Path) -> dict[str, Any]:
    """Authenticate immutable v1 evidence and separately verify the repaired tree."""
    root = repo_root.resolve()
    archive = root / FROZEN_ARCHIVE_RELATIVE
    manifest = root / FROZEN_MANIFEST_RELATIVE
    terminal_report = root / FROZEN_REPORT_RELATIVE
    for path, expected, name in (
        (archive, ARCHIVE_SHA256, "v1 archive"),
        (manifest, EVIDENCE_MANIFEST_SHA256, "v1 evidence manifest"),
        (terminal_report, TERMINAL_REPORT_SHA256, "v1 terminal report"),
    ):
        if not path.is_file():
            raise ValueError(f"{name} is missing")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{name} hash differs from its immutable v1 value")
    terminal = _authenticate_terminal_report(terminal_report, root)

    current_hashes: dict[str, str] = {}
    for relative, expected in sorted(FROZEN_REPO_FILE_SHA256.items()):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required v1 input is missing: {relative}")
        actual = sha256_file(path)
        current_hashes[relative] = actual
        if relative != SOURCE_RELATIVE and actual != expected:
            raise ValueError(f"unchanged v1 input hash differs: {relative}")
    repaired_source_sha256 = current_hashes[SOURCE_RELATIVE]
    pre_repair_source_sha256 = FROZEN_REPO_FILE_SHA256[SOURCE_RELATIVE]
    if repaired_source_sha256 != APPROVED_REPAIRED_SOURCE_SHA256:
        raise ValueError(
            "repaired source hash is not the approved post-repair value: "
            f"expected {APPROVED_REPAIRED_SOURCE_SHA256}, got {repaired_source_sha256}"
        )
    if pre_repair_source_sha256 == APPROVED_REPAIRED_SOURCE_SHA256:
        raise ValueError("approved post-repair source hash equals the frozen v1 hash")

    with tempfile.TemporaryDirectory(prefix="redco-v12-postrepair-") as temporary:
        extracted = _safe_extract(archive, Path(temporary))
        archive_manifest = _manifest_audit(extracted, manifest)
    if not archive_manifest["all_match"] or archive_manifest["listed_count"] != 38:
        raise ValueError("immutable v1 archive/evidence manifest verification failed")

    try:
        audit_archive(
            archive,
            manifest,
            repo_root=root,
            terminal_report=terminal_report,
        )
    except ValueError as error:
        rejection = str(error)
        expected_prefix = f"immutable repository hash differs for {SOURCE_RELATIVE}:"
        if not rejection.startswith(expected_prefix):
            raise ValueError("frozen v1 audit did not fail at the repaired source hash") from error
    else:
        raise ValueError("frozen v1 audit unexpectedly accepted the repaired source tree")

    return {
        "schema_version": POST_REPAIR_SCHEMA_VERSION,
        "domain": POST_REPAIR_DOMAIN,
        "status": "engineering_post_repair_verification_only",
        "v1_partition": {
            "archive_sha256": ARCHIVE_SHA256,
            "evidence_manifest_sha256": EVIDENCE_MANIFEST_SHA256,
            "terminal_report_sha256": TERMINAL_REPORT_SHA256,
            "terminal_report_schema_version": terminal["schema_version"],
            "archive_manifest": {
                "all_match": archive_manifest["all_match"],
                "listed_count": archive_manifest["listed_count"],
                "matched_count": archive_manifest["matched_count"],
            },
            "source_hash_mismatch_rejection": rejection,
            "pre_repair_source_sha256": pre_repair_source_sha256,
            "v1_inputs_untouched": True,
            "v1_evidence_is_not_reinterpreted": True,
        },
        "repaired_tree": {
            "source_relative": SOURCE_RELATIVE,
            "approved_repaired_source_sha256": APPROVED_REPAIRED_SOURCE_SHA256,
            "repaired_source_sha256": repaired_source_sha256,
            "approved_source_hash_authenticated": True,
            "source_hash_changed": True,
            "repair": {
                "comparison_owner": "redco.analysis.stage_d_source_producer:_verify_trace_call",
                "pointer": "/content",
                "transport_presence": "present-null",
                "trace_presence": "absent",
                "message_role": "assistant",
                "tool_calls": "absent on both sides",
                "directional": True,
                "exact_equality_checked_first": True,
                "raw_forms_preserved": True,
            },
            "record_binding": {
                "message_cases_bound": True,
                "record_cases_bound": False,
                "record_cases_scope": (
                    "frozen record exactness vectors only; no record-field mutation is "
                    "production-bound by this post-repair fixture"
                ),
                "record_binding_hook": (
                    "tests/stage_d_source_comparison_oracle.py:"
                    "record_exactness_binding_observation"
                ),
            },
        },
        "scientific_scope": {
            "v12_recovered": False,
            "v13_touched": False,
            "scientific_settings_changed": False,
            "engineering_only": True,
        },
        "runtime_source_hashes": current_hashes,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = _validate_output_path(args.output, args.repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(audit_postrepair(args.repo_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
