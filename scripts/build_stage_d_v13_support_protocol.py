#!/usr/bin/env python3
"""Build candidate-null-independent, CPU-only v13 support artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from redco.analysis.stage_d_v13_draft_publication import (
    atomic_publish_set,
    validate_output_paths,
)
from redco.analysis.stage_d_v13_source_phase_a_decoder import (
    SOURCE_ARTIFACT_RELATIVE,
    SOURCE_SHA256,
)
from redco.analysis.stage_d_v13_support_protocol import (
    ADDRESS_AUDIT_RELATIVE,
    ADDRESS_AUDIT_SHA256,
    CANDIDATE_RELATIVE,
    COLLECTION_PLAN_RELATIVE,
    COLLECTION_PLAN_SHA256,
    COMPOSITION_RELATIVE,
    FROZEN_SUPPORT_RULES_RELATIVE,
    FROZEN_SUPPORT_RULES_SHA256,
    PROTOCOL_AUDIT_RELATIVE,
    PROTOCOL_RELATIVE,
    RETAINED_SUPPORT_RELATIVE,
    RETAINED_SUPPORT_SHA256,
    SELECTION_CLAIM_RELATIVE,
    SELECTION_CLAIM_SHA256,
    SELECTION_MANIFEST_RELATIVE,
    SELECTION_MANIFEST_SHA256,
    SELECTION_ORIGINAL_CLAIM_RELATIVE,
    SELECTION_RECEIPT_RELATIVE,
    SELECTION_RECEIPT_SHA256,
    V12_ARCHIVE_RELATIVE,
    V12_ARCHIVE_SHA256,
    V12_EVIDENCE_MANIFEST_RELATIVE,
    V12_EVIDENCE_MANIFEST_SHA256,
    V12_FINALIZATION_AUDIT_RELATIVE,
    V12_FINALIZATION_AUDIT_SHA256,
    V12_PREREG_RELATIVE,
    V12_PREREG_SHA256,
    V12_PROTOCOL_RELATIVE,
    V12_PROTOCOL_SHA256,
    V12_SOURCE_EVAL_RELATIVE,
    V12_SOURCE_EVAL_SHA256,
    V12_TERMINAL_REPORT_RELATIVE,
    V12_TERMINAL_REPORT_SHA256,
    build_protocol_artifacts,
    check_protocol_artifacts,
    rebuild_protocol_artifacts_from_existing,
)
from redco.contracts import canonical_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--rebuild-existing", action="store_true")
    args = parser.parse_args()
    if args.check_only and args.rebuild_existing:
        parser.error("--check-only and --rebuild-existing are mutually exclusive")
    root = args.repository.resolve()
    output_root = (args.output_root or root).resolve()
    immutable = {
        relative: expected
        for relative, expected in {
            RETAINED_SUPPORT_RELATIVE: RETAINED_SUPPORT_SHA256,
            COLLECTION_PLAN_RELATIVE: COLLECTION_PLAN_SHA256,
            ADDRESS_AUDIT_RELATIVE: ADDRESS_AUDIT_SHA256,
            SELECTION_RECEIPT_RELATIVE: SELECTION_RECEIPT_SHA256,
            SELECTION_MANIFEST_RELATIVE: SELECTION_MANIFEST_SHA256,
            SELECTION_CLAIM_RELATIVE: SELECTION_CLAIM_SHA256,
            SELECTION_ORIGINAL_CLAIM_RELATIVE: SELECTION_CLAIM_SHA256,
            V12_ARCHIVE_RELATIVE: V12_ARCHIVE_SHA256,
            V12_EVIDENCE_MANIFEST_RELATIVE: V12_EVIDENCE_MANIFEST_SHA256,
            V12_FINALIZATION_AUDIT_RELATIVE: V12_FINALIZATION_AUDIT_SHA256,
            V12_TERMINAL_REPORT_RELATIVE: V12_TERMINAL_REPORT_SHA256,
            V12_PREREG_RELATIVE: V12_PREREG_SHA256,
            V12_PROTOCOL_RELATIVE: V12_PROTOCOL_SHA256,
            V12_SOURCE_EVAL_RELATIVE: V12_SOURCE_EVAL_SHA256,
            FROZEN_SUPPORT_RULES_RELATIVE: FROZEN_SUPPORT_RULES_SHA256,
            SOURCE_ARTIFACT_RELATIVE: SOURCE_SHA256,
        }.items()
    }
    validate_output_paths(output_root, {
        str((root / relative).resolve()): expected
        for relative, expected in immutable.items()
    }, output_paths=tuple(
        {
            CANDIDATE_RELATIVE,
            COMPOSITION_RELATIVE,
            PROTOCOL_RELATIVE,
            PROTOCOL_AUDIT_RELATIVE,
        }
    ))
    if args.check_only:
        hashes = check_protocol_artifacts(root, output_root)
        artifacts = {relative: (output_root / relative).read_bytes() for relative in hashes}
    else:
        artifacts = (
            rebuild_protocol_artifacts_from_existing(root)
            if args.rebuild_existing
            else build_protocol_artifacts(root, output_root)
        )
        hashes = atomic_publish_set(
            output_root,
            artifacts,
            immutable_paths={
                str((root / relative).resolve()): expected
                for relative, expected in immutable.items()
            },
            manifest_path=PROTOCOL_AUDIT_RELATIVE,
        )
    result = {
        "check_only": args.check_only,
        "artifact_count": len(artifacts),
        "artifacts": [
            {"path": path, "bytes": len(data), "sha256": hashes[path]}
            for path, data in sorted(artifacts.items())
        ],
    }
    print(canonical_json(result).decode())


if __name__ == "__main__":
    main()
