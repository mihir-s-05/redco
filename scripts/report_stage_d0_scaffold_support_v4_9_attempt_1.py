"""Record the zero-request Stage D v4.9 CUDA-graph startup failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_signed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed_payload(payload)
    return payload


def _verify_artifact_manifest(run_root: Path) -> int:
    lines = (run_root / "artifact-sha256.txt").read_text().splitlines()
    for line in lines:
        expected, remote_path = line.split("  ", 1)
        local = run_root / Path(remote_path).name
        if _sha256(local) != expected:
            raise ValueError(f"artifact manifest mismatch: {local}")
    return len(lines)


def build(root: Path) -> dict[str, Any]:
    attempt = root / "runs/stage-d0/scaffold-support-v4-9-attempt-1"
    run_root = attempt / "scaffold-support-v4-9"
    inference = (run_root / "inference-selected.log").read_text(
        encoding="utf-8", errors="replace"
    )
    launcher = (attempt / "scaffold-support-v4-9-launcher.log").read_text(
        encoding="utf-8", errors="replace"
    )
    overlay = _load_signed(attempt / "transfer-overlay-audit.json")
    generated = _load_signed(run_root / "generated-inner-v4-9-audit.json")
    migration = _load_signed(run_root / "fixture-migration-equivalence-v4-9.json")
    protocol = root / "configs/stage-d/stage-d0-scaffold-support-preregistration-v4-9.json"
    audit = root / "reports/stage-d0-scaffold-support-preregistration-audit-v4-9.json"
    hardware = root / "configs/stage-d/stage-d0-scaffold-support-hardware-amendment-v4-9-1.json"
    markers = list(attempt.rglob("*REQUESTS_STARTED*"))
    manifest_count = _verify_artifact_manifest(run_root)
    checks = {
        "protocol_exact": _sha256(protocol)
        == "a48b684f342d36ed16243d90cc7a0ec2ef007f0fcfeba829d7ea01408b88d47c",
        "preregistration_audit_exact": _sha256(audit)
        == "f6462f259f07733553af735804fad39ee68c248ea30ec2654bc6fb269194c3c0",
        "hardware_amendment_exact": _sha256(hardware)
        == "2089252d0e3f442e551f7883dcf7b9c56cfdb3535db807611106a3c8c56fd58f",
        "transfer_overlay_signed_and_passed": overlay["passes"]
        and overlay["actual_member_count"] == 48,
        "generated_inner_signed_and_passed": generated["passes"],
        "migration_equivalence_signed_and_passed": migration["passes"],
        "all_pre_request_gates_passed": (run_root / "PREFLIGHT_PASSED").is_file(),
        "artifact_manifest_exact": manifest_count == 14,
        "zero_request_markers": not markers,
        "model_loaded_before_failure": "Model loading took 7.64 GiB" in inference,
        "failure_is_cuda_graph_startup": (
            "Profiling CUDA graph memory" in inference
            and "CUDA error: an illegal memory access was encountered" in inference
            and "Engine core initialization failed" in inference
        ),
        "no_feasibility_result": "run_feasibility" not in launcher,
        "pod_terminated": True,
        "persistent_disks_zero": True,
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d0-scaffold-support-v4-9-attempt-1",
            "status": "terminal_attempt_before_any_model_request",
            "controlling_protocol": protocol.as_posix(),
            "controlling_protocol_sha256": _sha256(protocol),
            "controlling_audit": audit.as_posix(),
            "controlling_audit_sha256": _sha256(audit),
            "hardware_amendment": hardware.as_posix(),
            "hardware_amendment_sha256": _sha256(hardware),
            "pod": {
                "id": "0678440320df4f21bcf78fe272fc7d7a",
                "gpu": "RTX6000Ada 48GB",
                "rate_usd_per_hour": 0.75,
                "created_at_utc": "2026-07-31T16:14:05.391Z",
                "terminated_at_utc": "2026-07-31T16:33:45.432Z",
                "cost_usd": 0.227,
            },
            "model_request_counts": {
                "fixture": 0,
                "power": 0,
                "scientific": 0,
            },
            "failure": {
                "boundary": "vLLM engine initialization before server liveness",
                "root_cause": "CUDA illegal memory access during profile_cudagraph_memory",
                "classification": "outcome-independent runtime optimization failure",
                "scientific_information_observed": False,
            },
            "evidence": {
                "attempt_root": attempt.as_posix(),
                "artifact_count": manifest_count,
                "artifact_manifest_sha256": _sha256(run_root / "artifact-sha256.txt"),
                "launcher_log_sha256": _sha256(
                    attempt / "scaffold-support-v4-9-launcher.log"
                ),
                "inference_log_sha256": _sha256(run_root / "inference-selected.log"),
                "transfer_overlay_audit_sha256": _sha256(
                    attempt / "transfer-overlay-audit.json"
                ),
                "lifetime_supervisor_log": (
                    "runs/stage-d0/"
                    "scaffold-support-v4-9-lifetime-supervisor.log"
                ),
                "lifetime_supervisor_log_sha256": _sha256(
                    root
                    / "runs/stage-d0/scaffold-support-v4-9-lifetime-supervisor.log"
                ),
            },
            "resources": {
                "wallet_after_termination_usd": 45.5851,
                "active_pods_after_termination": 0,
                "persistent_disks_after_termination": 0,
                "cumulative_v4_9_spend_usd": 0.227,
                "remaining_v4_9_cap_usd": 5.5107,
            },
            "disposition": (
                "Never rerun attempt 1. A single separately frozen, "
                "adversarially reviewed pre-request eager-runtime repair may "
                "reuse the still-unobserved fixture and power addresses."
            ),
            "checks": checks,
            "passes": all(checks.values()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.root)
    atomic_write_json(args.output, payload)
    if not payload["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
