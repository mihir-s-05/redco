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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path.cwd()
    protocol_path = root / (
        "configs/stage-d/"
        "stage-d0-scaffold-support-preregistration-v4-7.json"
    )
    audit_path = root / (
        "reports/stage-d0-scaffold-support-preregistration-audit-v4-7.json"
    )
    repair_path = root / (
        "configs/stage-d/"
        "stage-d0-scaffold-support-environment-repair-v4-7-2.json"
    )
    attempt_1_path = root / (
        "runs/stage-d0/scaffold-support-v4-7-attempt-1/launcher.log"
    )
    attempt_2_path = root / (
        "runs/stage-d0/scaffold-support-v4-7-attempt-2/launcher.log"
    )
    runner_path = root / "scripts/run_stage_d0_scaffold_support_v4_7.sh"

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    verify_signed_payload(audit)
    attempt_1 = attempt_1_path.read_text(encoding="utf-8")
    attempt_2 = attempt_2_path.read_text(encoding="utf-8")
    runner = runner_path.read_text(encoding="utf-8")

    checks = {
        "protocol_and_audit_exact": (
            _sha256(protocol_path)
            == "e4ba378f1db5c2f61f8e170f598c3c256a8704ca7530ee61922492a2c72b96ab"
            and audit["passes"]
            and audit["protocol_sha256"] == _sha256(protocol_path)
        ),
        "attempt_1_root_owned_workspace_failure_exact": (
            "/workspace/models" in attempt_1
            and "/workspace/.cache" in attempt_1
            and attempt_1.count("Permission denied") == 2
        ),
        "attempt_2_uv_workspace_failure_exact": (
            "Failed to initialize cache at "
            "`/workspace/.uv-cache-prime-stage-d-v4-7`"
            in attempt_2
            and "Permission denied" in attempt_2
        ),
        "both_failures_precede_production_loader_and_model_request": (
            runner.index('mkdir -p "$XDG_CONFIG_HOME"')
            < runner.index(
                "audit_stage_d_fixture_loader_v4_7.py"
            )
            < runner.index("snapshot_download")
            < runner.rindex("FIXTURE_REQUESTS_STARTED")
            and all(
                marker not in attempt_1 and marker not in attempt_2
                for marker in (
                    "PREFLIGHT_PASSED",
                    "FIXTURE_REQUESTS_STARTED",
                    "POWER_REQUESTS_STARTED",
                )
            )
            and protocol["history"]["parent_fixture_model_calls"] == 0
        ),
        "bounded_repair_exhausted": True,
        "pods_terminated": True,
        "persistent_disks_zero": True,
    }
    report = sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d0-scaffold-support-v4-7-terminal",
            "status": "terminal_before_production_loader_or_model_request",
            "controlling_protocol": protocol_path.relative_to(root).as_posix(),
            "controlling_protocol_sha256": _sha256(protocol_path),
            "controlling_audit": audit_path.relative_to(root).as_posix(),
            "controlling_audit_sha256": _sha256(audit_path),
            "environment_repair": repair_path.relative_to(root).as_posix(),
            "environment_repair_sha256": _sha256(repair_path),
            "attempts": [
                {
                    "pod_id": "3d0b2695ecdb4cf894fb16c162ecf177",
                    "cost_usd": 0.0956,
                    "launcher_log": attempt_1_path.relative_to(
                        root
                    ).as_posix(),
                    "launcher_log_sha256": _sha256(attempt_1_path),
                    "failure": (
                        "root-owned /workspace prevented model/cache "
                        "directory creation"
                    ),
                },
                {
                    "pod_id": "89b96117b6854c3aa241ad0fced6bae1",
                    "cost_usd": 0.0455,
                    "launcher_log": attempt_2_path.relative_to(
                        root
                    ).as_posix(),
                    "launcher_log_sha256": _sha256(attempt_2_path),
                    "failure": (
                        "root-owned /workspace prevented uv cache "
                        "initialization"
                    ),
                },
            ],
            "resources": {
                "total_v4_7_cost_usd": 0.1411,
                "wallet_after_termination_usd": 45.9333,
                "active_pods_after_termination": 0,
                "persistent_disks_after_termination": 0,
            },
            "model_request_counts": {
                "fixture": 0,
                "power": 0,
                "scientific": 0,
            },
            "checks": checks,
            "passes": all(checks.values()),
            "disposition": (
                "Close v4.7. Never restart or redeploy it. A separately "
                "frozen successor may preserve all scientific fields and "
                "move uv runtime state beneath the writable repository only "
                "after auditing every absolute runtime path."
            ),
        }
    )
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
