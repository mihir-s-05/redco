"""Audit the Stage D0 v4.5 pre-model flash-attn activation repair."""

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


def audit(
    root: Path,
    protocol_path: Path,
    amendment_v4_1_path: Path,
    amendment_v4_2_path: Path,
    amendment_v4_5_path: Path,
    audit_v4_2_path: Path,
    patch_audit_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    amendment_v4_1 = json.loads(
        amendment_v4_1_path.read_text(encoding="utf-8")
    )
    amendment_v4_2 = json.loads(
        amendment_v4_2_path.read_text(encoding="utf-8")
    )
    amendment_v4_5 = json.loads(
        amendment_v4_5_path.read_text(encoding="utf-8")
    )
    audit_v4_2 = json.loads(audit_v4_2_path.read_text(encoding="utf-8"))
    patch_audit = json.loads(patch_audit_path.read_text(encoding="utf-8"))
    for report in (audit_v4_2, patch_audit):
        verify_signed_payload(report)

    effective_sources = dict(protocol["source_sha256"])
    for amendment in (
        amendment_v4_1,
        amendment_v4_2,
        amendment_v4_5,
    ):
        effective_sources.update(amendment["source_sha256_replacements"])
        effective_sources.update(amendment["source_sha256_additions"])

    source_results = {}
    for relative, expected in effective_sources.items():
        clean_prime_relative = relative.removeprefix("external/prime-rl/")
        clean_prime_result = patch_audit["source_results"].get(
            clean_prime_relative
        )
        if (
            relative.startswith("external/prime-rl/")
            and clean_prime_result is not None
        ):
            actual = clean_prime_result["actual"]
            provenance = "signed_clean_prime_patch_audit"
        else:
            path = root / relative
            actual = _sha256(path) if path.is_file() else None
            provenance = "current_worktree"
        source_results[relative] = {
            "expected": expected,
            "actual": actual,
            "passes": actual == expected,
            "provenance": provenance,
        }

    runner = (
        root / "scripts/run_stage_d0_scaffold_support_v4.sh"
    ).read_text(encoding="utf-8")
    renderer_command = (
        '"$uv_bin" run --frozen --project external/prime-rl \\\n'
        "  --extra flash-attn \\\n"
        "  python scripts/audit_stage_d_sft_renderer_v4.py"
    )
    dry_run_command = (
        '"$uv_bin" run --frozen --project external/prime-rl \\\n'
        "  --extra flash-attn \\\n"
        '  sft @ "$sft_train_config" --dry-run'
    )
    live_sft_command = (
        '"$uv_bin" run --frozen --project external/prime-rl \\\n'
        "    --extra flash-attn \\\n"
        '    sft @ "$sft_train_config"'
    )
    preflight_logs = amendment_v4_5["preobservation_evidence"][
        "recovered_logs"
    ]
    log_results = {}
    for relative, expected in preflight_logs.items():
        path = root / relative
        actual = _sha256(path) if path.is_file() else None
        log_results[relative] = {
            "expected": expected,
            "actual": actual,
            "passes": actual == expected,
        }

    checks = {
        "signed_ancestor_audits_pass": (
            audit_v4_2["passes"] and patch_audit["passes"]
        ),
        "amendment_chain_exact": (
            amendment_v4_5["base_protocol_sha256"]
            == _sha256(protocol_path)
            and amendment_v4_5["amendment_v4_1_sha256"]
            == _sha256(amendment_v4_1_path)
            and amendment_v4_5["amendment_v4_2_sha256"]
            == _sha256(amendment_v4_2_path)
            and amendment_v4_5["audit_v4_2_sha256"]
            == _sha256(audit_v4_2_path)
        ),
        "failure_is_pre_model_and_outcome_independent": (
            amendment_v4_5["status"]
            == "frozen_after_outcome_independent_pre_model_failure"
            and amendment_v4_5["preobservation_evidence"]["model_calls"] == 0
            and amendment_v4_5["preobservation_evidence"][
                "optimizer_steps"
            ]
            == 0
            and amendment_v4_5["preobservation_evidence"][
                "candidate_score_payloads"
            ]
            == 0
            and amendment_v4_5["preobservation_evidence"][
                "scientific_arm_outcomes"
            ]
            == 0
            and amendment_v4_5["preobservation_evidence"][
                "gpu_memory_mib"
            ]
            <= 2
        ),
        "repair_is_dependency_activation_only": (
            amendment_v4_5["scientific_changes"] == []
            and amendment_v4_5["repair"]["dependency_extra"]
            == "flash-attn"
            and amendment_v4_5["repair"]["commands_changed"]
            == [
                "renderer audit preflight",
                "SFT launcher dry-run preflight",
            ]
            and amendment_v4_5["repair"]["actual_sft_command_changed"]
            is False
        ),
        "runner_merges_v4_5": (
            "stage-d0-scaffold-support-amendment-v4-5.json" in runner
            and "for amendment_name in sys.argv[2:]:" in runner
        ),
        "same_extra_on_all_sft_stack_commands": (
            renderer_command in runner
            and dry_run_command in runner
            and live_sft_command in runner
            and runner.count("--extra flash-attn") == 3
        ),
        "prior_execution_repairs_persist": (
            'timeout --signal=TERM --kill-after=120 21600 bash "$0" "$@"'
            in runner
            and "timeout --signal=TERM --kill-after=120 3600" in runner
            and 'rm -rf "$sft_dir" "$sft_reloaded" "$sft_merged"'
            in runner
        ),
        "recovered_logs_exact": (
            bool(log_results)
            and all(row["passes"] for row in log_results.values())
        ),
        "all_effective_source_hashes_exact": (
            bool(source_results)
            and all(row["passes"] for row in source_results.values())
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": (
                "stage-d0-scaffold-support-preregistration-audit-v4-5"
            ),
            "protocol": protocol_path.resolve()
            .relative_to(root.resolve())
            .as_posix(),
            "amendment_v4_5": amendment_v4_5_path.resolve()
            .relative_to(root.resolve())
            .as_posix(),
            "amendment_v4_5_sha256": _sha256(amendment_v4_5_path),
            "effective_source_count": len(effective_sources),
            "source_results": source_results,
            "preflight_log_results": log_results,
            "checks": checks,
            "passes": all(checks.values()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "configs/stage-d/"
            "stage-d0-scaffold-support-preregistration-v4.json"
        ),
    )
    parser.add_argument(
        "--amendment-v4-1",
        type=Path,
        default=Path(
            "configs/stage-d/"
            "stage-d0-scaffold-support-amendment-v4-1.json"
        ),
    )
    parser.add_argument(
        "--amendment-v4-2",
        type=Path,
        default=Path(
            "configs/stage-d/"
            "stage-d0-scaffold-support-amendment-v4-2.json"
        ),
    )
    parser.add_argument(
        "--amendment-v4-5",
        type=Path,
        default=Path(
            "configs/stage-d/"
            "stage-d0-scaffold-support-amendment-v4-5.json"
        ),
    )
    parser.add_argument(
        "--audit-v4-2",
        type=Path,
        default=Path(
            "reports/"
            "stage-d0-scaffold-support-preregistration-audit-v4-2.json"
        ),
    )
    parser.add_argument(
        "--patch-audit",
        type=Path,
        default=Path(
            "reports/stage-d-prime-sft-runtime-patch-v2-audit.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.root.resolve(),
        args.protocol.resolve(),
        args.amendment_v4_1.resolve(),
        args.amendment_v4_2.resolve(),
        args.amendment_v4_5.resolve(),
        args.audit_v4_2.resolve(),
        args.patch_audit.resolve(),
    )
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
