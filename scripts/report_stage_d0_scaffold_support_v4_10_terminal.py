"""Record the terminal Stage D0 scaffold-support v4.10 fixture result."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/stage-d0/scaffold-support-v4-10"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_signed(path: Path) -> dict[str, Any]:
    payload = _load(path)
    verify_signed_payload(payload)
    return payload


def _manifest_exact() -> tuple[int, list[str]]:
    manifest = RUN / "artifact-sha256.txt"
    failures = []
    rows = manifest.read_text(encoding="utf-8").splitlines()
    marker = "runs/stage-d0/scaffold-support-v4-10/"
    for row in rows:
        expected, remote_path = row.split("  ", 1)
        relative = remote_path.split(marker, 1)[1]
        local = RUN / relative
        if not local.is_file() or _sha256(local) != expected:
            failures.append(relative)
    return len(rows), failures


def _utc_seconds(start: str, end: str) -> float:
    return (
        datetime.fromisoformat(end).replace(tzinfo=None)
        - datetime.fromisoformat(start).replace(tzinfo=None)
    ).total_seconds()


def build_report() -> dict[str, Any]:
    protocol_path = ROOT / (
        "configs/stage-d/stage-d0-scaffold-support-preregistration-v4-10.json"
    )
    audit_path = ROOT / (
        "reports/stage-d0-scaffold-support-preregistration-audit-v4-10.json"
    )
    hardware_path = ROOT / (
        "configs/stage-d/stage-d0-scaffold-support-hardware-amendment-v4-10-1.json"
    )
    protocol = _load(protocol_path)
    audit = _load_signed(audit_path)
    hardware = _load(hardware_path)
    transfer = _load_signed(
        ROOT / "runs/stage-d0/scaffold-support-v4-10-transfer-overlay-audit.json"
    )
    eager_tail = _load_signed(RUN / "generated-eager-tail-v4-10-audit.json")
    inner = _load_signed(RUN / "generated-inner-v4-10-audit.json")
    eligibility = _load_signed(RUN / "selected-fixture-eligibility.json")
    replay = _load_signed(RUN / "selected-fixture-replay.json")
    scores = _load_signed(RUN / "selected-fixture-scores.json")
    summary = _load(RUN / "selected-fixture/run-summary.json")
    inference = (RUN / "inference-selected.log").read_text(
        encoding="utf-8", errors="replace"
    )
    fixture_control = (RUN / "selected-fixture-control.log").read_text(
        encoding="utf-8", errors="replace"
    )
    artifact_count, manifest_failures = _manifest_exact()

    created_at = "2026-07-31T16:59:40.199000"
    terminated_at = "2026-07-31T17:10:34.426000"
    cost_usd = 0.1171
    rate_usd_per_hour = 0.75
    checks = {
        "protocol_exact": (
            _sha256(protocol_path)
            == "3709246858a5e3d1815289f40e990c780cdefcec29cc7695536c54c80aa74b76"
        ),
        "preregistration_audit_exact_and_passing": (
            _sha256(audit_path)
            == "4210e444ef7bb6e0b07a67d6f27c13f7b06683dd73727ae778171e6d2ebb7a29"
            and audit["signed_payload_sha256"]
            == "87ad7743eb6d445fe53eab4f565e6e42e1fb70f5fca55371bd37ab56f85fca52"
            and audit["passes"]
        ),
        "hardware_amendment_exact": (
            _sha256(hardware_path)
            == "e032187ea6705ef311c4c36327bae54674fcc883ed6d40e47fafdac8e83b52b3"
            and hardware["selected_resource"]["resource_id"] == "03c728"
            and hardware["repair_boundary"]["this_is_the_only_v4_10_deployment"]
        ),
        "transfer_overlay_signed_and_exact": (
            transfer["passes"]
            and transfer["actual_member_count"] == 48
            and transfer["signed_payload_sha256"]
            == "acfc1e8de68b15dbcc398061c0cf7730bbc3ec5137d14e224a626022716447f3"
        ),
        "generated_runners_signed_and_passing": (
            eager_tail["passes"]
            and inner["passes"]
            and eager_tail["signed_payload_sha256"]
            == protocol["runtime"]["generated_eager_tail_audit_signature"]
            and inner["signed_payload_sha256"]
            == protocol["runtime"]["generated_inner_audit_signature"]
        ),
        "artifact_manifest_exact": artifact_count == 25 and not manifest_failures,
        "all_pre_request_gates_passed": (
            (RUN / "PREFLIGHT_PASSED").is_file()
            and (RUN / "EAGER_RUNTIME_PREFLIGHT_PASSED").is_file()
        ),
        "live_eager_runtime_proven": (
            "enforce_eager=True" in inference
            and "<CompilationMode.NONE: 0>" in inference
            and "<CUDAGraphMode.NONE: 0>" in inference
            and "Profiling CUDA graph memory" not in inference
            and "Capturing CUDA graphs" not in inference
        ),
        "fixture_request_observed_power_never_started": (
            (RUN / "FIXTURE_REQUESTS_STARTED").is_file()
            and not (RUN / "POWER_REQUESTS_STARTED").exists()
        ),
        "fixture_rollout_completed_successfully": (
            len(summary["records"]) == 1
            and summary["records"][0]["ok"]
            and "reward=1.000" in fixture_control
        ),
        "branch_replay_and_scoring_plumbing_passed": (
            replay["paired_branches"] == 3
            and replay["cached_action_mismatches"] == 0
            and replay["reward_mismatches"] == 0
            and scores["all_alternatives_distinct_from_original"]
            and scores["all_downstream_prompts_changed"]
        ),
        "fixture_gate_failed_as_frozen": (
            eligibility["eligible"] is False
            and eligibility["informative"] is False
            and eligibility["joint_eligible_and_informative"] is False
            and eligibility["root_calls"] == 3
            and eligibility["child_calls"] == 2
            and eligibility["exact_field_checks"]["exactly_two_root_calls"] is False
            and eligibility["reward_informativeness"]["range"] == 0.0
        ),
        "resources_terminated": True,
        "persistent_disks_zero": True,
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d0-scaffold-support-v4-10-terminal",
            "status": "terminal_fixture_gate_failure",
            "decision": "fail",
            "passes": all(checks.values()),
            "checks": checks,
            "controlling_protocol": protocol_path.relative_to(ROOT).as_posix(),
            "controlling_protocol_sha256": _sha256(protocol_path),
            "controlling_audit": audit_path.relative_to(ROOT).as_posix(),
            "controlling_audit_sha256": _sha256(audit_path),
            "hardware_amendment": hardware_path.relative_to(ROOT).as_posix(),
            "hardware_amendment_sha256": _sha256(hardware_path),
            "result": {
                "fixture_rollouts": 1,
                "fixture_reward": 1.0,
                "recorded_policy_calls": 5,
                "root_calls": eligibility["root_calls"],
                "child_calls": eligibility["child_calls"],
                "branch_alternatives": replay["alternatives_per_target"],
                "fixture_eligible": eligibility["eligible"],
                "fixture_informative": eligibility["informative"],
                "joint_gate": eligibility["joint_eligible_and_informative"],
                "failed_topology_check": "exactly_two_root_calls",
                "observed_root_calls": eligibility["root_calls"],
                "expected_root_calls": 2,
                "regenerated_and_alternative_f1": [
                    eligibility["reward_informativeness"]["regenerated_original_f1"],
                    *eligibility["reward_informativeness"]["alternative_f1"],
                ],
                "f1_range": eligibility["reward_informativeness"]["range"],
                "power_requests": 0,
                "scientific_training_requests": 0,
            },
            "interpretation": {
                "what_succeeded": (
                    "The eager runtime repair, real tool-call scaffold, one live fixture "
                    "rollout, target selection, K=4 branch generation, exact replay, and "
                    "deterministic scoring all executed successfully."
                ),
                "what_failed": (
                    "The single frozen fixture produced a richer root-child-root-child-root "
                    "topology (3 root and 2 child calls) instead of the required two-root "
                    "shape, and all four branch outcomes scored F1=1.0, yielding zero "
                    "informativeness."
                ),
                "claim_scope": (
                    "This is a negative Stage D scaffold-support gate, not an algorithm "
                    "result. No power block and no stock, branch-global, or local-credit "
                    "training arm ran."
                ),
            },
            "ledgers": {
                "fixture_wall_seconds": summary["total_wall_seconds"],
                "recorded_rollout_generated_tokens": replay["baseline_generated_tokens"],
                "alternative_action_generated_tokens": replay[
                    "alternative_action_generated_tokens"
                ],
                "branch_downstream_generated_tokens": replay[
                    "downstream_generated_tokens"
                ],
                "branch_model_request_wall_seconds": replay[
                    "model_request_wall_seconds"
                ],
                "provider_billed_gpu_hours": cost_usd / rate_usd_per_hour,
                "pod_wall_lifetime_seconds": _utc_seconds(created_at, terminated_at),
                "exact_billing_usd": cost_usd,
            },
            "artifacts": {
                "run_root": RUN.relative_to(ROOT).as_posix(),
                "artifact_manifest_count": artifact_count,
                "artifact_manifest_failures": manifest_failures,
                "launcher_log_sha256": _sha256(
                    ROOT / "runs/stage-d0/scaffold-support-v4-10-launcher.log"
                ),
                "lifetime_supervisor_log_sha256": _sha256(
                    ROOT
                    / "runs/stage-d0/scaffold-support-v4-10-lifetime-supervisor.log"
                ),
                "transfer_overlay_audit_sha256": _sha256(
                    ROOT
                    / "runs/stage-d0/scaffold-support-v4-10-transfer-overlay-audit.json"
                ),
            },
            "pod": {
                "id": "cfb7963d6edc49bca2b6c364137de974",
                "gpu": "RTX6000Ada 48GB",
                "rate_usd_per_hour": rate_usd_per_hour,
                "created_at_utc": created_at,
                "terminated_at_utc": terminated_at,
                "cost_usd": cost_usd,
            },
            "resources": {
                "wallet_after_termination_usd": 45.468,
                "active_pods_after_termination": 0,
                "persistent_disks_after_termination": 0,
                "total_support_spend_usd": round(0.2623 + 0.227 + cost_usd, 4),
                "original_support_ceiling_remaining_usd": round(
                    6.0 - (0.2623 + 0.227 + cost_usd), 4
                ),
            },
            "disposition": (
                "Terminal v4.10 result. Never retry this fixture address or power block. "
                "Any successor requires explicit user authorization and a separately "
                "frozen design; v4.10 itself has no repair deployment remaining."
            ),
        }
    )


def main() -> None:
    output = ROOT / "reports/stage-d0-scaffold-support-v4-10-terminal-2026-07-31.json"
    report = build_report()
    atomic_write_json(output, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
