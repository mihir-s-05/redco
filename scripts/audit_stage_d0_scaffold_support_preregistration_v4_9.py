"""Audit the serialization-only Stage D0 scaffold-support v4.9 successor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_signed(path: Path) -> dict[str, Any]:
    payload = _load(path)
    verify_signed_payload(payload)
    return payload


def audit(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = _load(root / protocol_path)
    history = protocol["history"]
    parent = _load(root / history["parent_protocol"])
    parent_audit = _load_signed(root / history["parent_audit"])
    terminal = _load_signed(root / history["parent_terminal_report"])
    inner_audit = _load_signed(root / protocol["runtime"]["generated_inner_audit"])
    linux_audit = _load_signed(root / protocol["runtime"]["linux_serialization_audit"])
    overlay_audit = _load_signed(root / protocol["transfer"]["overlay_audit"])
    wrapper = (root / protocol["runtime"]["wrapper"]).read_text(encoding="utf-8")
    generator = (root / protocol["runtime"]["generator"]).read_text(encoding="utf-8")
    verifier = (root / protocol["runtime"]["strict_json_verifier"]).read_text(
        encoding="utf-8"
    )

    source_results = {
        relative: {
            "expected": expected,
            "actual": _sha256(root / relative),
            "passes": _sha256(root / relative) == expected,
        }
        for relative, expected in protocol["source_sha256"].items()
    }
    budget = protocol["budget"]
    envelope = (
        budget["remaining_support_ceiling_usd"]
        + budget["scientific_training_reserved_ceiling_usd"]
        + budget["science_evaluation_reserved_usd"]
        + budget["untouchable_reserve_usd"]
    )
    runtime = protocol["runtime"]
    transfer = protocol["transfer"]
    checks = {
        "distinct_successor_frozen_before_request": (
            protocol["experiment"] == "stage-d0-scaffold-support-v4-9"
            and protocol["status"] == "frozen_before_any_v4_9_model_request"
        ),
        "v4_8_terminal_signed_exact_and_zero_requests": (
            _sha256(root / history["parent_terminal_report"])
            == history["parent_terminal_report_sha256"]
            and terminal["signed_payload_sha256"]
            == history["parent_terminal_report_signature"]
            and terminal["passes"]
            and all(value == 0 for value in terminal["model_request_counts"].values())
        ),
        "parent_protocol_and_audit_signed_exact": (
            _sha256(root / history["parent_protocol"])
            == history["parent_protocol_sha256"]
            and _sha256(root / history["parent_audit"])
            == history["parent_audit_sha256"]
            and parent_audit["passes"]
            and parent_audit["protocol_sha256"]
            == history["parent_protocol_sha256"]
        ),
        "all_scientific_fields_inherited_except_named_validation_line": (
            {
                key: value
                for key, value in protocol["scientific_inheritance"].items()
                if key != "policy"
            }
            == {
                key: value
                for key, value in parent["scientific_inheritance"].items()
                if key != "policy"
            }
            and protocol["scientific_inheritance"]["policy"]
            == (
                "All scientific and request-producing behavior is inherited "
                "byte-identically from v4.7; exactly one pre-request "
                "migration-validation line changes from byte comparison to "
                "strict verified signed-object equality."
            )
        ),
        "source_hashes_exact": (
            bool(source_results)
            and all(item["passes"] for item in source_results.values())
        ),
        "generated_runner_is_one_hunk_only": (
            inner_audit["passes"]
            and _sha256(root / runtime["generated_inner_audit"])
            == runtime["generated_inner_audit_sha256"]
            and inner_audit["signed_payload_sha256"]
            == runtime["generated_inner_audit_signature"]
            and inner_audit["generated_sha256"]
            == runtime["generated_inner_sha256"]
            and all(inner_audit["checks"].values())
            and len(
                [
                    line
                    for line in inner_audit["unified_diff"]
                    if line.startswith("@@")
                ]
            )
            == 1
        ),
        "only_signed_json_comparison_changes": (
            generator.count("audit_signed_json_equivalence_v4_9.py") == 3
            and 'cmp -s "$migration_report"' in generator
            and verifier.count("verify_signed_payload") == 2
            and "strict_full_object_equal" in verifier
            and "duplicate JSON key" in verifier
            and "nonstandard JSON numeric" in verifier
        ),
        "linux_serialization_regressions_pass": (
            linux_audit["passes"]
            and _sha256(root / runtime["linux_serialization_audit"])
            == runtime["linux_serialization_audit_sha256"]
            and linux_audit["signed_payload_sha256"]
            == runtime["linux_serialization_audit_signature"]
            and all(linux_audit["checks"].values())
        ),
        "wrapper_generates_audits_and_parses_inner_before_run": (
            wrapper.index("generate_stage_d_v4_9_inner_runner.py")
            < wrapper.index('bash -n "$generated_inner"')
            < wrapper.index("preflight_stage_d_runtime_paths_v4_8.sh")
            < wrapper.index('bash "$generated_inner"')
            and "generated inner hash mismatch" in wrapper
            and "generated inner audit signature mismatch" in wrapper
            and "preflight_stage_d_runtime_paths_v4_8.sh" in wrapper
            and 'run_root="runs/stage-d0/scaffold-support-v4-9"' in wrapper
        ),
        "github_https_plus_exact_overlay_frozen": (
            transfer["repository_https"] == "https://github.com/mihir-s-05/redco.git"
            and transfer["clone_branch"] == "main"
            and transfer["commit_pinning"] == "required in hardware amendment"
            and transfer["overlay_sha256"] == overlay_audit["overlay_sha256"]
            and _sha256(root / transfer["overlay_audit"])
            == transfer["overlay_audit_sha256"]
            and overlay_audit["signed_payload_sha256"]
            == transfer["overlay_audit_signature"]
            and transfer["overlay_member_count"] == 48
            and overlay_audit["expected_member_count"] == 48
            and overlay_audit["actual_member_count"] == 48
            and overlay_audit["passes"]
            and all(overlay_audit["checks"].values())
        ),
        "transfer_order_and_empty_failure_gate_frozen": (
            transfer["ordered_steps"]
            == [
                "https_clone_and_checkout_exact_commit",
                "initialize_clean_pinned_submodules",
                "apply_exact_48_member_byte_overlay",
                "require_empty_overlay_hash_failure_set",
                "run_outer_source_hash_gate",
                "generate_and_audit_inner_runner",
                "run_runtime_path_preflight",
                "invoke_inner_runner",
            ]
            and transfer["overlay_hash_gate"] == "empty failure set required"
        ),
        "budget_arithmetic_exact": (
            math.isclose(
                budget["wallet_after_v4_8_usd"] - envelope,
                budget["headroom_usd"],
                abs_tol=1e-9,
            )
            and math.isclose(envelope, budget["remaining_full_envelope_usd"], abs_tol=1e-9)
            and budget["headroom_usd"] > 0
            and budget["cumulative_v4_9_cap_usd"]
            == budget["remaining_support_ceiling_usd"]
            and terminal["resources"]["wallet_after_termination_inferred_usd"]
            == budget["wallet_after_v4_8_usd"]
        ),
        "timeout_and_depleted_cap_executable": (
            "REDCO_V4_9_AUTHORIZED_TIMEOUT_SECONDS" in wrapper
            and 'test "$timeout_seconds" -le 17100' in wrapper
            and budget["maximum_wrapper_timeout_seconds"] == 17100
            and budget["maximum_total_billed_pod_lifetime_seconds_at_max_rate"]
            == 19800
            and budget["minimum_non_wrapper_lifetime_reserve_seconds"] == 2700
            and budget["maximum_wrapper_timeout_seconds"]
            + budget["minimum_non_wrapper_lifetime_reserve_seconds"]
            <= budget["maximum_total_billed_pod_lifetime_seconds_at_max_rate"]
            and budget["hardware_amendment_rule"]
            == (
                "rate_usd_per_hour * authorized_total_billed_pod_lifetime_seconds / 3600 "
                "+ cumulative_v4_9_spend_usd must be no greater than 5.7377"
            )
            and "local supervisor" in budget["hardware_amendment_lifetime_contract"]
            and "120-second kill grace" in budget["hardware_amendment_lifetime_contract"]
        ),
        "bounded_repair_and_request_boundaries_preserved": (
            protocol["repair_policy"]["maximum_pre_request_redeployments"] == 1
            and protocol["repair_policy"]["after_first_fixture_request"].startswith(
                "terminal"
            )
            and protocol["repair_policy"]["scientific_changes"] == "forbidden"
            and protocol["authorization"]["scientific_training_model_calls"] == 0
        ),
        "uv_only": (
            protocol["runtime"]["dependency_policy"] == "uv only; pip is forbidden"
            and "pip " not in wrapper
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d0-scaffold-support-preregistration-audit-v4-9",
            "protocol": protocol_path.as_posix(),
            "protocol_sha256": _sha256(root / protocol_path),
            "source_results": source_results,
            "recomputed_remaining_envelope_usd": envelope,
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
            "configs/stage-d/stage-d0-scaffold-support-preregistration-v4-9.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.root, args.protocol)
    atomic_write_json(args.output, payload)
    if not payload["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
