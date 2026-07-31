"""Audit the outcome-independent Stage D0 scaffold-support v4.1 amendment."""

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
    amendment_path: Path,
    base_audit_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    base_audit = json.loads(base_audit_path.read_text(encoding="utf-8"))
    verify_signed_payload(base_audit)
    protocol_relative = protocol_path.resolve().relative_to(
        root.resolve()
    ).as_posix()

    replacements = amendment["source_sha256_replacements"]
    additions = amendment["source_sha256_additions"]
    effective_sources = dict(protocol["source_sha256"])
    effective_sources.update(replacements)
    effective_sources.update(additions)
    source_results = {}
    for relative, expected in effective_sources.items():
        path = root / relative
        actual = _sha256(path) if path.is_file() else None
        source_results[relative] = {
            "expected": expected,
            "actual": actual,
            "passes": actual == expected,
        }

    runner = (
        root / "scripts/run_stage_d0_scaffold_support_v4.sh"
    ).read_text(encoding="utf-8")
    replay = (
        root / "src/redco/analysis/empirical_branch_replay.py"
    ).read_text(encoding="utf-8")
    group = (
        root / "src/redco/analysis/stage_d_branch_group.py"
    ).read_text(encoding="utf-8")

    checks = {
        "base_protocol_exact": (
            amendment["base_protocol"] == protocol_relative
            and amendment["base_protocol_sha256"] == _sha256(protocol_path)
            and amendment["base_commit"] == "0f00d29"
        ),
        "base_audit_exact_and_passing": (
            base_audit["passes"]
            and base_audit["protocol_sha256"] == _sha256(protocol_path)
        ),
        "preobservation_state_explicit": (
            amendment["status"] == "frozen_before_any_v4_model_call"
            and amendment["preobservation_evidence"]["prime_pods_created"] == 0
            and amendment["preobservation_evidence"]["model_calls"] == 0
            and amendment["preobservation_evidence"]["optimizer_steps"] == 0
            and amendment["preobservation_evidence"][
                "candidate_score_payloads"
            ]
            == 0
            and amendment["preobservation_evidence"][
                "scientific_arm_outcomes"
            ]
            == 0
        ),
        "replacement_keys_were_frozen": (
            set(replacements) <= set(protocol["source_sha256"])
        ),
        "addition_keys_are_new": (
            not (set(additions) & set(protocol["source_sha256"]))
        ),
        "all_effective_source_hashes_exact": (
            bool(source_results)
            and all(row["passes"] for row in source_results.values())
        ),
        "scientific_design_unchanged": (
            amendment["scientific_changes"] == []
            and amendment["unchanged_fields"]
            == [
                "dataset partitions and corpus bytes",
                "initialization and SFT optimizer",
                "sampling and seed rules",
                "candidate order and selection rule",
                "eligibility and informativeness definitions",
                "58-of-64 joint support threshold",
                "future scientific arms, outcomes, and budget envelope",
            ]
        ),
        "hard_timeout_escalation_present": (
            'timeout --signal=TERM --kill-after=120 21600 bash "$0" "$@"'
            in runner
            and "timeout --signal=TERM --kill-after=120 3600" in runner
        ),
        "all_sft_work_cleaned_on_exit": (
            'rm -rf "$sft_dir" "$sft_reloaded" "$sft_merged"' in runner
        ),
        "portable_prime_patch_applied_before_hash_check": (
            "prime-rl-stage-d-sft-local-json-v1.patch" in runner
            and runner.index("git -C external/prime-rl apply --check")
            < runner.index("source_sha256 = dict(protocol")
        ),
        "only_declared_ineligibility_becomes_negative": (
            "class DeterministicReplayIneligibility(ValueError):" in replay
            and "except DeterministicReplayIneligibility as error:" in group
            and "except (ValueError, RuntimeError)" not in group
        ),
        "seed_semantics_narrowed": (
            amendment["seed_semantics"]["planned_slots"]
            == "all 64 planned episode seeds are exact"
            and amendment["seed_semantics"]["present_traces"]
            == "every present observed root seed must align exactly"
            and amendment["seed_semantics"]["missing_traces"]
            == (
                "missing traces remain denominator negatives and do not "
                "count as observed-seed successes"
            )
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": (
                "stage-d0-scaffold-support-preregistration-audit-v4-1"
            ),
            "protocol": protocol_relative,
            "protocol_sha256": _sha256(protocol_path),
            "amendment": amendment_path.as_posix(),
            "amendment_sha256": _sha256(amendment_path),
            "base_audit": base_audit_path.as_posix(),
            "effective_source_count": len(effective_sources),
            "source_results": source_results,
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
        "--amendment",
        type=Path,
        default=Path(
            "configs/stage-d/"
            "stage-d0-scaffold-support-amendment-v4-1.json"
        ),
    )
    parser.add_argument(
        "--base-audit",
        type=Path,
        default=Path(
            "reports/stage-d0-scaffold-support-preregistration-audit-v4.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.root,
        args.protocol,
        args.amendment,
        args.base_audit,
    )
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
