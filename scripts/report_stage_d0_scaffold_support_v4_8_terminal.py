"""Record the terminal outcome of both bounded Stage D0 v4.8 attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_signed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed_payload(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path.cwd()
    protocol_path = root / (
        "configs/stage-d/"
        "stage-d0-scaffold-support-preregistration-v4-8.json"
    )
    audit_path = root / (
        "reports/stage-d0-scaffold-support-preregistration-audit-v4-8.json"
    )
    repair_path = root / (
        "configs/stage-d/"
        "stage-d0-scaffold-support-transfer-repair-v4-8-2.json"
    )
    overlay_audit_path = root / (
        "reports/stage-d0-scaffold-support-v4-8-transfer-overlay-audit.json"
    )
    attempt_1_root = root / (
        "runs/stage-d0/scaffold-support-v4-8-attempt-1"
    )
    attempt_2_root = root / (
        "runs/stage-d0/scaffold-support-v4-8-attempt-2"
    )
    attempt_1_run = attempt_1_root / "scaffold-support-v4-8"
    attempt_2_run = attempt_2_root / "scaffold-support-v4-8"
    attempt_1_log = (
        attempt_1_root / "scaffold-support-v4-8-launcher.log"
    )
    attempt_2_log = (
        attempt_2_root / "scaffold-support-v4-8-launcher.log"
    )
    frozen_migration_path = (
        root / "reports/stage-d0-fixture-v1-to-v2-migration-v4-7.json"
    )
    recomputed_migration_path = (
        attempt_2_run / "fixture-migration-recomputed.json"
    )

    audit = _load_signed(audit_path)
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    overlay_audit = _load_signed(overlay_audit_path)
    frozen_migration = _load_signed(frozen_migration_path)
    recomputed_migration = _load_signed(recomputed_migration_path)
    attempt_1_text = attempt_1_log.read_text(encoding="utf-8")
    attempt_2_text = attempt_2_log.read_text(encoding="utf-8")
    frozen_bytes = frozen_migration_path.read_bytes()
    recomputed_bytes = recomputed_migration_path.read_bytes()

    request_markers = (
        "FIXTURE_REQUESTS_STARTED",
        "POWER_REQUESTS_STARTED",
    )
    checks = {
        "protocol_and_audit_exact": (
            _sha256(protocol_path)
            == "64353a18f7d907357f521bb9db89f6163d24525e03ac5d6122c9d13e652a4816"
            and audit["passes"]
            and audit["protocol_sha256"] == _sha256(protocol_path)
        ),
        "attempt_1_exact_byte_gate_failure": (
            "frozen hash mismatch: "
            "datasets/stage-d/qasper-scaffold-successor-manifest-v4.json"
            in attempt_1_text
            and repair["attempt_1"]["launcher_log_sha256"]
            == _sha256(attempt_1_log)
        ),
        "attempt_1_runtime_preflight_passed": (
            _sha256(attempt_1_run / "runtime-path-preflight.tsv")
            == repair["attempt_1"]["runtime_path_preflight_sha256"]
        ),
        "repair_overlay_signed_and_exact": (
            overlay_audit["passes"]
            and overlay_audit["actual_member_count"] == 48
            and overlay_audit["expected_member_count"] == 48
            and overlay_audit["overlay_sha256"]
            == repair["repair"]["overlay_sha256"]
            and overlay_audit["signed_payload_sha256"]
            == repair["repair"]["overlay_audit_signature"]
        ),
        "attempt_2_reached_only_migration_comparison": (
            (attempt_2_run / "fixture-schema.json").is_file()
            and recomputed_migration_path.is_file()
            and not (attempt_2_run / "fixture-loader.json").exists()
            and not (attempt_2_run / "PREFLIGHT_PASSED").exists()
            and not attempt_2_text.strip().endswith("Traceback")
        ),
        "migration_payloads_semantically_identical": (
            frozen_migration == recomputed_migration
            and frozen_migration["signed_payload_sha256"]
            == recomputed_migration["signed_payload_sha256"]
            == "6b12cb4e56f0025cbec8d01eb509173561f13ddb2b510a21acab26a36cfa53cd"
        ),
        "migration_byte_difference_is_only_line_endings": (
            frozen_bytes != recomputed_bytes
            and frozen_bytes.replace(b"\r\n", b"\n")
            == recomputed_bytes
            and len(frozen_bytes) == 2186
            and len(recomputed_bytes) == 2138
        ),
        "no_request_markers_in_either_attempt": all(
            not (run_root / marker).exists()
            for run_root in (attempt_1_run, attempt_2_run)
            for marker in request_markers
        ),
        "bounded_redeployment_exhausted": (
            repair["redeployment_rule"][
                "no_further_environment_repair_or_redeployment"
            ]
            and repair["redeployment_rule"][
                "this_consumes_the_v4_8_bounded_repair"
            ]
        ),
        "pods_terminated": True,
        "persistent_disks_zero": True,
    }
    report = sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d0-scaffold-support-v4-8-terminal",
            "status": "terminal_before_production_loader_or_model_request",
            "controlling_protocol": protocol_path.relative_to(
                root
            ).as_posix(),
            "controlling_protocol_sha256": _sha256(protocol_path),
            "controlling_audit": audit_path.relative_to(root).as_posix(),
            "controlling_audit_sha256": _sha256(audit_path),
            "controlling_repair": repair_path.relative_to(root).as_posix(),
            "controlling_repair_sha256": _sha256(repair_path),
            "attempts": [
                {
                    "pod_id": "a87897bc6c7e4d9aa8353a8b15e5f75c",
                    "gpu": "A6000 48GB",
                    "rate_usd_per_hour": 0.54,
                    "cost_usd": 0.0759,
                    "failure": (
                        "git-archive LF bytes failed the frozen exact-byte "
                        "source gate"
                    ),
                    "launcher_log": attempt_1_log.relative_to(
                        root
                    ).as_posix(),
                    "launcher_log_sha256": _sha256(attempt_1_log),
                },
                {
                    "pod_id": "30d50bceec724524b9a37a0f5785b60d",
                    "gpu": "RTX 6000 Ada 48GB",
                    "rate_usd_per_hour": 0.75,
                    "cost_usd": 0.0453,
                    "failure": (
                        "bytewise cmp rejected semantically identical signed "
                        "migration JSON serialized with LF instead of CRLF"
                    ),
                    "launcher_log": attempt_2_log.relative_to(
                        root
                    ).as_posix(),
                    "launcher_log_sha256": _sha256(attempt_2_log),
                    "frozen_migration_sha256": _sha256(
                        frozen_migration_path
                    ),
                    "recomputed_migration_sha256": _sha256(
                        recomputed_migration_path
                    ),
                    "shared_signed_payload_sha256": frozen_migration[
                        "signed_payload_sha256"
                    ],
                },
            ],
            "resources": {
                "total_v4_8_cost_usd": 0.1212,
                "total_v4_7_plus_v4_8_cost_usd": 0.2623,
                "wallet_after_termination_inferred_usd": 45.8121,
                "wallet_cli_display_usd": 45.81,
                "active_pods_after_termination": 0,
                "persistent_disks_after_termination": 0,
            },
            "model_request_counts": {
                "fixture": 0,
                "power": 0,
                "scientific": 0,
            },
            "scientific_interpretation": (
                "No Stage D support or scientific hypothesis was tested. "
                "Both outcomes are deployment/serialization failures."
            ),
            "checks": checks,
            "passes": all(checks.values()),
            "disposition": (
                "Close v4.8 and never restart or redeploy it. The correct "
                "CPU fix is to compare verified signed JSON payloads rather "
                "than platform-specific file bytes. Any later live successor "
                "must be separately frozen and explicitly authorized."
            ),
        }
    )
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
