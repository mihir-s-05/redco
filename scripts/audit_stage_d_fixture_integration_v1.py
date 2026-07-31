"""Gate live fixtures only on outcome-independent all-child integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)

SEMANTIC_FIELDS = {
    "recorded_trace_output_valid",
    "regenerated_original_output_valid",
    "all_alternative_outputs_valid",
}


def audit(path: Path) -> dict[str, Any]:
    target_audit = json.loads(path.read_text(encoding="utf-8"))
    verify_signed_payload(target_audit)
    structural_checks = []
    for report in target_audit["target_reports"]:
        verify_signed_payload(report)
        fields = report.get("exact_field_checks") or {}
        structural_checks.append(
            bool(fields)
            and all(value is True for key, value in fields.items() if key not in SEMANTIC_FIELDS)
        )
    checks = {
        "two_to_four_canonical_targets": 2 <= int(target_audit["candidate_count"]) <= 4,
        "complete_structural_replay_chain": (
            len(structural_checks) == int(target_audit["candidate_count"])
            and all(structural_checks)
        ),
        "exact_weights_sum_to_one": (
            target_audit["exact_decision_unit_weight_contract"] is True
            and target_audit["outer_decision_unit_weight_sum"]
            == {
                "numerator": int(target_audit["candidate_count"]),
                "denominator": int(target_audit["candidate_count"]),
            }
        ),
        "trace_precommit_replay_scorer_chain_present": all(
            len(str(target_audit.get(field, ""))) == 64
            for field in (
                "source_trace_sha256",
                "precommit_signed_payload_sha256",
                "candidate_set_sha256",
                "replay_signed_payload_sha256",
                "scorer_signed_payload_sha256",
            )
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-fixture-integration-v1",
            "target_audit_signature": target_audit["signed_payload_sha256"],
            "checks": checks,
            "passes": all(checks.values()),
            "diagnostics_not_gate_inputs": {
                "parseability": True,
                "verbatimness": True,
                "f1": True,
                "informativeness": True,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.target_audit)
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(24)


if __name__ == "__main__":
    main()
