"""Audit the frozen Stage D0 scaffold-support v4.6 continuation."""

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


def _wilson_lower(successes: int, total: int, z: float = 1.95996398454) -> float:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = proportion + z * z / (2 * total)
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total
        + z * z / (4 * total * total)
    )
    return (center - radius) / denominator


def _load_signed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed_payload(payload)
    return payload


def audit(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    parent = _load_signed(
        root / protocol["history"]["closed_parent_report"]
    )
    manifest = _load_signed(
        root / protocol["fixed_initialization"]["adapter_manifest"]
    )
    observed = _load_signed(
        root / protocol["observed_address_lock"]["report"]
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

    adapter = root / protocol["fixed_initialization"]["adapter_archive"]
    dataset = root / protocol["frozen_sequence"][
        "step_3_previously_unobserved_power_audit"
    ]["dataset"]
    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split_counts = Counter(str(row["split"]) for row in rows)

    runner = (
        root / "scripts/run_stage_d0_scaffold_support_v4_6.sh"
    ).read_text(encoding="utf-8")
    retention = (
        root / "scripts/run_stage_d_retention_canonical_v4_6.py"
    ).read_text(encoding="utf-8")
    scorer = (
        root / "scripts/score_stage_d_retained_adapter_canonical_v4_6.py"
    ).read_text(encoding="utf-8")
    health = (
        root / "scripts/audit_stage_d_vllm_health_v4_6.py"
    ).read_text(encoding="utf-8")
    adapter_context = scorer[
        scorer.index("with adapter_hooks(model, args.adapter):") :
        scorer.index("action = _action_payload(")
    ]
    power = protocol["frozen_sequence"][
        "step_3_previously_unobserved_power_audit"
    ]
    budget = protocol["budget"]
    recomputed_envelope = (
        budget["v4_6_compute_ceiling_usd"]
        + budget["scientific_training_reserved_ceiling_usd"]
        + budget["science_evaluation_reserved_usd"]
        + budget["untouchable_reserve_usd"]
    )

    checks = {
        "frozen_before_v4_6_output": (
            protocol["status"] == "frozen_before_any_v4_6_model_call"
        ),
        "parent_terminal_signed_and_exact": (
            _sha256(root / protocol["history"]["closed_parent_report"])
            == protocol["history"]["closed_parent_report_sha256"]
            and parent["status"]
            == "terminal_before_selected_fixture_and_power_audit"
        ),
        "parent_has_no_fixture_power_or_science": (
            parent["frozen_cascade"]["selected_fixture_model_calls"] == 0
            and parent["frozen_cascade"]["power_audit_model_calls"] == 0
            and parent["frozen_cascade"]["scientific_arm_outcomes"] == 0
        ),
        "observed_address_report_signed_and_exact": (
            _sha256(root / protocol["observed_address_lock"]["report"])
            == protocol["observed_address_lock"]["report_sha256"]
            and len(observed["fewshot_support"]) == 64
            and observed["sft_optimizer_steps"] == list(range(1, 9))
            and observed["selected_fixture_observed"] is False
            and observed["power_audit_addresses_observed"] == []
            and observed["scientific_arm_addresses_observed"] == []
        ),
        "all_source_hashes_exact": (
            bool(source_results)
            and all(row["passes"] for row in source_results.values())
        ),
        "retained_archive_exact": (
            adapter.is_file()
            and _sha256(adapter)
            == protocol["fixed_initialization"]["adapter_archive_sha256"]
            == manifest["archive_sha256"]
        ),
        "retained_tensor_identity_exact": (
            manifest["members"]["adapter_model.safetensors"]["sha256"]
            == protocol["fixed_initialization"]["adapter_model_sha256"]
            and manifest["safetensors"]["tensor_count"]
            == protocol["fixed_initialization"]["adapter_tensor_count"]
            and manifest["safetensors"]["tensor_data_sha256"]
            == protocol["fixed_initialization"][
                "adapter_tensor_data_sha256"
            ]
        ),
        "no_support_or_sft_rerun": (
            "redco-stage-d0-fewshot-support-v4" not in runner
            and "sft @" not in runner
            and "stage-d0-scaffold-sft-v4.toml" not in runner
            and "selected-adapter.tar.gz" in runner
        ),
        "prime_inference_source_pinned_before_and_after_patch": (
            protocol["source_sha256"][
                "external/prime-rl/src/prime_rl/inference/server.py"
            ]
            == protocol["prime_stack"]["inference_server_base_sha256"]
            == "d36e25ba2484e7b85dac96a591c18f7599203a437228cbedc675ecd1cf67ddeb"
            and protocol["source_sha256"][
                "patches/prime-rl-strict-tool-env-guard.patch"
            ]
            == protocol["prime_stack"]["strict_tool_patch_sha256"]
            and protocol["prime_stack"][
                "inference_server_post_patch_sha256"
            ]
            in runner
            and runner.index("bash scripts/run_rlm_structural_trace_audit.sh")
            < runner.index(
                protocol["prime_stack"][
                    "inference_server_post_patch_sha256"
                ]
            )
            < runner.index("run_stage_d_retention_canonical_v4_6.py")
        ),
        "two_fresh_isolated_canonical_processes": (
            "for replicate in (1, 2):" in retention
            and "subprocess.run(command, check=False)" in retention
            and "physical_extraction" in retention
            and "stable_logical_adapter_path" in retention
            and "action_paths[0].read_bytes() == action_paths[1].read_bytes()"
            in retention
            and "root_paths[0].read_bytes() == root_paths[1].read_bytes()"
            in retention
        ),
        "adapter_covers_action_and_root_scores": (
            "_action_model(" in adapter_context
            and "_root_payload(" in adapter_context
        ),
        "vllm_is_health_not_equality_gate": (
            "canonical_runtime_greedy_tokens_agree" in health
            and "all_runtime_values_finite" in health
            and "EXPECTED_ROOT_CASES_SIGNATURE" in health
            and "canonical_runtime_root_token_ids_exact" in health
            and "--canonical-root" in runner
            and "--expected-runtime-model" in runner
            and "probability_delta" not in health
            and "logprob_delta" not in health
        ),
        "unobserved_fixture_precedes_power": (
            runner.index('"$run_root/selected-fixture"')
            < runner.index('"$run_root/power-audit" 64 1')
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
            and power["master_seed"] == "redco-stage-d0-power-audit-v4"
            and power["replay_master_seed"]
            == "redco-stage-d0-power-replay-v4"
        ),
        "power_partition_still_64_unique_papers": (
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
        "wilson_lower_exceeds_0_808": _wilson_lower(58, 64) > 0.808,
        "hard_six_hour_timeout_executable": (
            "timeout --signal=TERM --kill-after=120 21600" in runner
        ),
        "budget_envelope_exact_and_covered": (
            math.isclose(
                recomputed_envelope,
                budget["full_envelope_usd"],
                abs_tol=1e-9,
            )
            and math.isclose(
                budget["wallet_verified_usd"] - recomputed_envelope,
                budget["headroom_usd"],
                abs_tol=1e-9,
            )
            and budget["headroom_usd"] > 0
        ),
        "one_nonspot_48gb_gpu_rate_bound": (
            protocol["hardware"]["maximum_rate_usd_per_hour"] == 1.0
            and all(
                "non-spot 1x48GB" in item
                for item in protocol["hardware"]["allowed"]
            )
            and "persistent storage" in protocol["hardware"]["forbidden"]
        ),
        "no_scientific_call_authorized": protocol["future_science"][
            "no_scientific_model_call_authorized_by_v4_6"
        ],
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": (
                "stage-d0-scaffold-support-preregistration-audit-v4-6"
            ),
            "protocol": protocol_path.as_posix(),
            "protocol_sha256": _sha256(protocol_path),
            "source_results": source_results,
            "split_counts": dict(split_counts),
            "wilson_lower_58_of_64": _wilson_lower(58, 64),
            "recomputed_budget_envelope_usd": recomputed_envelope,
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
            "stage-d0-scaffold-support-preregistration-v4-6.json"
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
