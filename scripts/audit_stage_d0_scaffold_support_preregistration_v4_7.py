"""Audit the frozen Stage D0 scaffold-support v4.7 continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
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


def audit(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    parent = _load_signed(root / protocol["history"]["parent_terminal_report"])
    migration = _load_signed(
        root / protocol["fixture_migration"]["signed_audit"]
    )
    manifest = _load_signed(
        root / protocol["fixed_initialization"]["adapter_manifest"]
    )
    bootstrap = protocol["verifiers_transfer_bootstrap"]
    transfer_repair = json.loads(
        (
            root / bootstrap["inherited_transfer_repair"]
        ).read_text(encoding="utf-8")
    )

    source_results = {}
    for relative, expected in protocol["source_sha256"].items():
        path = root / relative
        actual = _sha256(path) if path.is_file() else None
        source_results[relative] = {
            "expected": expected,
            "actual": actual,
            "passes": actual == expected,
        }

    dataset = root / protocol["frozen_sequence"][
        "step_4_previously_unobserved_power_audit"
    ]["dataset"]
    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split_counts = Counter(str(row["split"]) for row in rows)

    runner_path = root / "scripts/run_stage_d0_scaffold_support_v4_7.sh"
    runner = runner_path.read_text(encoding="utf-8")
    inherited_tail = (
        root / "scripts/run_stage_d0_scaffold_support_v4_6.sh"
    ).read_text(encoding="utf-8")
    fixture = protocol["frozen_sequence"][
        "step_3_previously_unobserved_fixture"
    ]
    power = protocol["frozen_sequence"][
        "step_4_previously_unobserved_power_audit"
    ]
    budget = protocol["budget"]
    envelope = (
        budget["v4_7_support_ceiling_usd"]
        + budget["scientific_training_reserved_ceiling_usd"]
        + budget["science_evaluation_reserved_usd"]
        + budget["untouchable_reserve_usd"]
    )
    first_model_download = runner.index("snapshot_download")
    loader_command = runner.index(
        "audit_stage_d_fixture_loader_v4_7.py"
    )
    production_dry_run = runner.index(
        "python -m redco_evidence_selection_v2.run_feasibility"
    )
    preflight_marker = runner.index('touch "$run_root/PREFLIGHT_PASSED"')
    verifiers_commit_check = runner.index(
        'git -C "$verifiers_source" rev-parse HEAD'
    )
    verifiers_prepare = runner.index(
        "bash scripts/run_rlm_structural_trace_audit.sh"
    )

    checks = {
        "frozen_before_any_v4_7_request": (
            protocol["status"]
            == "frozen_before_any_v4_7_model_request"
        ),
        "parent_terminal_signed_exact_and_zero_request": (
            _sha256(root / protocol["history"]["parent_terminal_report"])
            == protocol["history"]["parent_terminal_report_sha256"]
            and parent["signed_payload_sha256"]
            == protocol["history"]["parent_terminal_report_signature"]
            and parent["terminal_failure"]["fixture_model_calls"] == 0
            and parent["terminal_failure"]["power_audit_model_calls"] == 0
            and parent["terminal_failure"]["scientific_arm_outcomes"] == 0
        ),
        "parent_evidence_bundle_exact": (
            _sha256(root / protocol["history"]["parent_evidence_bundle"])
            == protocol["history"]["parent_evidence_bundle_sha256"]
            == parent["evidence"]["archive_sha256"]
        ),
        "inherited_retention_and_health_exact": (
            parent["validated_results"]["retention"][
                "signed_payload_sha256"
            ]
            == protocol["inherited_validated_evidence"][
                "canonical_retention"
            ]["signed_payload_sha256"]
            and parent["validated_results"]["merged_vllm_health"][
                "signed_payload_sha256"
            ]
            == protocol["inherited_validated_evidence"][
                "merged_vllm_health"
            ]["signed_payload_sha256"]
        ),
        "signed_migration_exact_and_complete": (
            _sha256(root / protocol["fixture_migration"]["signed_audit"])
            == protocol["fixture_migration"]["signed_audit_sha256"]
            and migration["signed_payload_sha256"]
            == protocol["fixture_migration"]["signed_payload_sha256"]
            and migration["passes"]
            and all(migration["checks"].values())
            and len(migration["rows"]) == 3
            and len(
                {row["example_id"] for row in migration["rows"]}
            )
            == 3
            and len(
                {row["prompt_sha256"] for row in migration["rows"]}
            )
            == 3
            and len({row["episode_seed"] for row in migration["rows"]})
            == 3
        ),
        "fixture_only_adds_extractively_typed_metadata": (
            migration["parent_fixture_sha256"]
            == protocol["fixture_migration"]["parent_sha256"]
            and migration["successor_fixture_sha256"]
            == protocol["fixture_migration"]["successor_sha256"]
            and all(
                row["answer_type"] == "extractive"
                for row in migration["rows"]
            )
        ),
        "all_source_hashes_exact": (
            bool(source_results)
            and all(row["passes"] for row in source_results.values())
        ),
        "clean_verifiers_bootstrap_frozen_and_exact": (
            _sha256(root / bootstrap["inherited_transfer_repair"])
            == bootstrap["inherited_transfer_repair_sha256"]
            and transfer_repair["repair"]["clean_verifiers_commit"]
            == bootstrap["clean_commit"]
            and transfer_repair["repair"]["clean_source_sha256"]
            == bootstrap["clean_source_sha256"]
            and transfer_repair["repair"]["no_prepatched_transfer"]
            and bootstrap[
                "known_initial_bootstrap_does_not_consume_bounded_repair"
            ]
        ),
        "runner_verifies_clean_verifiers_before_patch": (
            bootstrap["clean_commit"] in runner
            and 'git -C "$verifiers_source" status --porcelain' in runner
            and all(
                digest in runner
                for digest in bootstrap["clean_source_sha256"].values()
            )
            and verifiers_commit_check < verifiers_prepare
        ),
        "retained_adapter_and_tensors_exact": (
            _sha256(
                root
                / protocol["fixed_initialization"]["adapter_archive"]
            )
            == protocol["fixed_initialization"]["adapter_archive_sha256"]
            == manifest["archive_sha256"]
            and manifest["members"]["adapter_model.safetensors"]["sha256"]
            == protocol["fixed_initialization"]["adapter_model_sha256"]
            and manifest["safetensors"]["tensor_count"]
            == protocol["fixed_initialization"]["adapter_tensor_count"]
            and manifest["safetensors"]["tensor_data_sha256"]
            == protocol["fixed_initialization"][
                "adapter_tensor_data_sha256"
            ]
        ),
        "production_loader_preflight_is_exact_and_precedes_model": (
            "audit_stage_d_fixture_loader_v4_7.py" in runner
            and "--dry-run" in runner
            and "audit_stage_d_fixture_dry_run_v4_7.py" in runner
            and "python -m redco_evidence_selection_v2.run_feasibility"
            in runner
            and loader_command < production_dry_run < preflight_marker
            and preflight_marker < first_model_download
        ),
        "same_patched_runtime_for_preflight_and_live": (
            runner.count('cd "$verifiers_worktree"') >= 2
            and runner.count(
                'UV_PROJECT_ENVIRONMENT="$verifiers_environment"'
            )
            >= 2
            and "bash scripts/run_rlm_structural_trace_audit.sh"
            in runner
            and protocol["prime_stack"][
                "inference_server_post_patch_sha256"
            ]
            in runner
        ),
        "canonical_and_vllm_score_case_reruns_absent": (
            "run_stage_d_retention_canonical_v4_6.py" not in runner
            and "score_stage_c_policies_vllm.py" not in runner
            and "score_stage_c3_root_routes_vllm.py" not in runner
        ),
        "archive_deployment_and_strict_server_preserved": (
            "audit_stage_d_adapter_archive.py" in runner
            and "audit_stage_d_adapter_directory.py" in runner
            and "merge_stage_c_warmstart.py" in runner
            and 'start_inference "$selected_config" "selected"'
            in inherited_tail
            and "REDCO_STRICT_TOOL_CALLING_ENV=1" in inherited_tail
        ),
        "request_boundaries_injected_before_both_live_blocks": (
            runner.count("REQUESTS_STARTED") == 6
            and "if ($0 ~ /^run_eval \\\\$/)" in runner
            and "source \"$instrumented_tail\"" in runner
            and fixture["request_boundary"].startswith(
                "write FIXTURE_REQUESTS_STARTED"
            )
            and power["request_boundary"].startswith(
                "write POWER_REQUESTS_STARTED"
            )
        ),
        "repair_rules_bind_first_request_not_output": (
            "before_first_fixture_request"
            in protocol["repair_policy"]
            and "after_fixture_request_started"
            in protocol["repair_policy"]
            and "after_power_request_started"
            in protocol["repair_policy"]
            and "even if no output was persisted"
            in protocol["repair_policy"][
                "after_fixture_request_started"
            ]
        ),
        "fixture_and_power_addresses_unchanged": (
            fixture["master_seed"]
            == "redco-stage-d0-selected-fixture-v4"
            and fixture["replay_master_seed"]
            == "redco-stage-d0-selected-fixture-replay-v4"
            and power["master_seed"] == "redco-stage-d0-power-audit-v4"
            and power["replay_master_seed"]
            == "redco-stage-d0-power-replay-v4"
        ),
        "power_design_unchanged": (
            power["papers"] == 64
            and power["replicates_per_paper"] == 1
            and power["K_including_regenerated_original"] == 4
            and power["minimum_f1_range"] == 0.05
            and power["joint_pass_rule"]
            == (
                "at least 58 of 64 unique papers are both eligible "
                "and informative"
            )
        ),
        "power_partition_is_64_unique_papers": (
            split_counts["power_audit"] == 64
            and len(
                {
                    row["paper_id"]
                    for row in rows
                    if row["split"] == "power_audit"
                }
            )
            == 64
        ),
        "hard_six_hour_timeout_executable": (
            "timeout --signal=TERM --kill-after=120 21600" in runner
        ),
        "budget_envelope_exact_and_covered": (
            math.isclose(
                envelope,
                budget["full_envelope_usd"],
                abs_tol=1e-9,
            )
            and math.isclose(
                budget["wallet_verified_usd"] - envelope,
                budget["headroom_usd"],
                abs_tol=1e-9,
            )
            and budget["headroom_usd"] > 0
        ),
        "hardware_and_storage_bounds_exact": (
            protocol["hardware"]["maximum_rate_usd_per_hour"] == 1.0
            and all(
                "non-spot 1x48GB" in item
                for item in protocol["hardware"]["allowed"]
            )
            and "persistent storage" in protocol["hardware"]["forbidden"]
        ),
        "no_scientific_call_authorized": protocol["future_science"][
            "no_scientific_model_call_authorized_by_v4_7"
        ],
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": (
                "stage-d0-scaffold-support-preregistration-audit-v4-7"
            ),
            "protocol": protocol_path.as_posix(),
            "protocol_sha256": _sha256(protocol_path),
            "source_results": source_results,
            "split_counts": dict(split_counts),
            "recomputed_budget_envelope_usd": envelope,
            "checks": checks,
            "passes": all(checks.values()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "configs/stage-d/"
            "stage-d0-scaffold-support-preregistration-v4-7.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, args.protocol)
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
