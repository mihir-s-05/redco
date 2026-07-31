"""Audit the frozen Stage D all-child live support protocol."""

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


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def audit(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = _load(root / protocol_path)
    slots = _load(root / protocol["slots"]["manifest"])
    verify_signed_payload(slots)
    generated = _load(root / protocol["runtime"]["generated_runner_audit"])
    verify_signed_payload(generated)
    design_audit = _load(root / "reports/stage-d0-all-child-successor-design-audit-v1.json")
    verify_signed_payload(design_audit)
    source_results = {
        relative: {
            "expected": expected,
            "actual": _sha256(root / relative),
            "passes": _sha256(root / relative) == expected,
        }
        for relative, expected in protocol["source_sha256"].items()
    }
    wrapper = (root / protocol["runtime"]["wrapper"]).read_text(encoding="utf-8")
    generator = (root / protocol["runtime"]["runner_generator"]).read_text(encoding="utf-8")
    plan_audit = (root / "scripts/audit_stage_d_all_child_live_plan_v1.py").read_text(
        encoding="utf-8"
    )
    fixture_audit = (root / "scripts/audit_stage_d_fixture_integration_v1.py").read_text(
        encoding="utf-8"
    )
    budget = protocol["budget"]
    checks = {
        "frozen_before_live_completion": (
            protocol["status"] == "frozen_before_any_all_child_live_model_completion"
            and protocol["authorization"]["scientific_training_model_calls"] == 0
        ),
        "cpu_design_audit_passed": design_audit["passes"] is True,
        "source_hashes_exact": all(row["passes"] for row in source_results.values()),
        "slots_signed_and_exact": (
            _sha256(root / protocol["slots"]["manifest"]) == protocol["slots"]["sha256"]
            and slots["signed_payload_sha256"] == protocol["slots"]["signature"]
            and slots["fixture_slots"] == 2
            and slots["support_slots"] == 64
            and len({row["paper_id"] for row in slots["slots"]}) == 66
        ),
        "full_seed_strings_frozen": all(
            protocol["slots"][field] == expected
            for field, expected in (
                ("fixture_master_seed", "redco-stage-d0-all-child-fixture-v1"),
                (
                    "fixture_replay_master_seed",
                    "redco-stage-d0-all-child-fixture-replay-v1",
                ),
                ("support_master_seed", "redco-stage-d0-all-child-support-v1"),
                (
                    "support_replay_master_seed",
                    "redco-stage-d0-all-child-support-replay-v1",
                ),
            )
        ),
        "generated_runner_exact": (
            generated["generated_sha256"] == protocol["runtime"]["generated_runner_sha256"]
            and generated["signed_payload_sha256"]
            == protocol["runtime"]["generated_runner_audit_signature"]
            and generated["passes"] is True
        ),
        "fixture_gate_excludes_semantics": (
            set(protocol["fixture_integration"]["diagnostic_only_never_veto"])
            == {"parseability", "verbatimness", "F1", "reward range", "informativeness"}
            and "SEMANTIC_FIELDS" in fixture_audit
            and "diagnostics_not_gate_inputs" in fixture_audit
        ),
        "paper_rule_and_early_stop_exact": (
            protocol["support_gate"]["papers"] == 64
            and protocol["support_gate"]["required_successes"] == 58
            and "seventh paper failure" in protocol["support_gate"]["early_failure"]
            and protocol["support_gate"]["early_success"] == "forbidden"
            and "successes + untouched < 58" in plan_audit
        ),
        "partial_observation_rule_exact": (
            protocol["address_and_interruption_policy"]["observation_boundary"]
            == (
                "an address becomes irrevocably started when ADDRESS_STARTED is "
                "persisted immediately before its first model request"
            )
            and "terminal failure for the entire support campaign"
            in protocol["address_and_interruption_policy"]["failure_after_any_address_started"]
            and "continuation from a partial prefix is forbidden"
            in protocol["address_and_interruption_policy"]["continuation_after_repair"]
            and protocol["address_and_interruption_policy"]["automatic_retry"] == "forbidden"
            and '"$kind" "$slot" "$example_id" "$paper_id" >"$work/ADDRESS_STARTED"' in generator
        ),
        "sequential_atomic_runner": (
            "process_slot()" in generator
            and 'mv "$work" "$run_root/completed/$kind-$slot"' in generator
            and 'done <"$materialized/support.tsv"' in generator
            and "SUPPORT_EARLY_FAILED" in generator
        ),
        "pre_request_and_runtime_gates_present": all(
            token in generator + wrapper
            for token in (
                "dry-run",
                "audit_stage_d_runtime_context_v1.py",
                "precommit",
                "enforce_eager=True",
                "profile_cudagraph_memory",
                "preflight_stage_d_runtime_paths_v4_8.sh",
            )
        ),
        "budget_and_reserve_exact": (
            budget["maximum_new_charge_usd"] == 4.5
            and budget["maximum_rate_usd_per_hour"] == 0.75
            and budget["absolute_lifetime_hours"] == 6.0
            and budget["wallet_after_max_charge_usd"]
            == budget["wallet_at_preregistration_usd"] - budget["maximum_new_charge_usd"]
            and budget["wallet_after_max_charge_usd"]
            >= budget["required_wallet_after_max_charge_usd"]
            == budget["science_reserve_usd"] + budget["untouchable_reserve_usd"]
        ),
        "timeout_inside_absolute_lifetime": (
            protocol["runtime"]["wrapper_timeout_seconds"] == 18000
            and protocol["runtime"]["absolute_pod_lifetime_seconds"] == 21600
            and protocol["runtime"]["setup_recovery_reserve_seconds"] == 3600
            and protocol["runtime"]["wrapper_timeout_seconds"]
            + protocol["runtime"]["setup_recovery_reserve_seconds"]
            <= protocol["runtime"]["absolute_pod_lifetime_seconds"]
            and 'test "$timeout_seconds" -le 18000' in wrapper
        ),
        "https_uv_no_persistent_storage": (
            protocol["transfer"]["repository_https"] == "https://github.com/mihir-s-05/redco.git"
            and protocol["runtime"]["dependency_policy"] == "uv only; pip is forbidden"
            and "pip " not in wrapper
            and "persistent Prime storage" in protocol["transfer"]["forbidden"]
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d0-all-child-support-preregistration-audit-v1",
            "protocol": protocol_path.as_posix(),
            "protocol_sha256": _sha256(root / protocol_path),
            "source_results": source_results,
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
        default=Path("configs/stage-d/stage-d0-all-child-support-preregistration-v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, args.protocol)
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
