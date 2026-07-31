"""Audit the bounded eager-runtime Stage D0 v4.10 successor."""

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
    terminal = _load_signed(root / history["parent_terminal_report"])
    config_audit = _load_signed(root / protocol["runtime"]["eager_config_audit"])
    eager_tail_audit = _load_signed(
        root / protocol["runtime"]["generated_eager_tail_audit"]
    )
    inner_audit = _load_signed(root / protocol["runtime"]["generated_inner_audit"])
    wrapper = (root / protocol["runtime"]["wrapper"]).read_text(encoding="utf-8")
    path_preflight = (root / protocol["runtime"]["path_preflight"]).read_text(
        encoding="utf-8"
    )
    path_regression = (
        root / protocol["runtime"]["path_contract_regression"]
    ).read_text(encoding="utf-8")
    inference_config_source = (
        root
        / "external/prime-rl/packages/prime-rl-configs/src/prime_rl/configs/inference.py"
    ).read_text(encoding="utf-8")
    server_source = (
        root / "external/prime-rl/src/prime_rl/inference/vllm/server.py"
    ).read_text(encoding="utf-8")
    worker_source = (
        root / "external/prime-rl/src/prime_rl/inference/vllm/worker/filesystem.py"
    ).read_text(encoding="utf-8")
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
    checks = {
        "distinct_successor_frozen_before_request": (
            protocol["experiment"] == "stage-d0-scaffold-support-v4-10"
            and protocol["status"] == "frozen_before_any_v4_10_model_request"
        ),
        "v4_9_failure_signed_outcome_independent_and_zero_requests": (
            _sha256(root / history["parent_terminal_report"])
            == history["parent_terminal_report_sha256"]
            and terminal["signed_payload_sha256"]
            == history["parent_terminal_report_signature"]
            and terminal["passes"]
            and terminal["failure"]["scientific_information_observed"] is False
            and terminal["failure"]["root_cause"]
            == "CUDA illegal memory access during profile_cudagraph_memory"
            and all(value == 0 for value in terminal["model_request_counts"].values())
        ),
        "source_hashes_exact": (
            bool(source_results)
            and all(item["passes"] for item in source_results.values())
        ),
        "eager_config_is_exactly_one_field": (
            config_audit["passes"]
            and all(config_audit["checks"].values())
            and config_audit["eager_sha256"] == runtime["eager_config_sha256"]
            and config_audit["signed_payload_sha256"]
            == runtime["eager_config_audit_signature"]
        ),
        "eager_tail_is_pre_request_and_request_tail_unchanged": (
            eager_tail_audit["passes"]
            and all(eager_tail_audit["checks"].values())
            and eager_tail_audit["generated_sha256"]
            == runtime["generated_eager_tail_sha256"]
            and eager_tail_audit["signed_payload_sha256"]
            == runtime["generated_eager_tail_audit_signature"]
            and eager_tail_audit["checks"]["request_producing_tail_byte_identical"]
        ),
        "inner_has_only_three_named_hunks": (
            inner_audit["passes"]
            and all(inner_audit["checks"].values())
            and inner_audit["generated_sha256"]
            == runtime["generated_inner_sha256"]
            and inner_audit["signed_payload_sha256"]
            == runtime["generated_inner_audit_signature"]
            and len(
                [
                    line
                    for line in inner_audit["unified_diff"]
                    if line.startswith("@@")
                ]
            )
            == 3
        ),
        "pinned_prime_maps_eager_to_vllm": (
            '"model.enforce_eager": "enforce_eager"' in inference_config_source
            and "namespace = config.to_vllm()" in server_source
            and "When enforce_eager=True" in worker_source
            and "model isn't wrapped by torch.compile" in worker_source
        ),
        "wrapper_proves_live_eager_before_any_request": (
            wrapper.index("generate_stage_d_v4_10_eager_tail.py")
            < wrapper.index('bash -n "$generated_eager_tail"')
            < wrapper.index("generate_stage_d_v4_10_inner_runner.py")
            < wrapper.index('bash -n "$generated_inner"')
            < wrapper.index("preflight_stage_d_runtime_paths_v4_8.sh")
            < wrapper.index('bash "$generated_inner"')
            and 'raise SystemExit(f"v4.10 generated {label} hash mismatch")'
            in wrapper
            and 'raise SystemExit(f"v4.10 generated {label} audit signature mismatch")'
            in wrapper
        ),
        "runtime_path_contract_exact_across_wrapper_protocol_and_preflight": (
            'runtime_root="$repo_root/.runtime/stage-d-v4-8"' in wrapper
            and 'export REDCO_RUNTIME_ROOT="$runtime_root"' in wrapper
            and protocol["runtime"]["uv_environment"]
            == "/workspace/redco/.runtime/stage-d-v4-8/prime-env"
            and protocol["runtime"]["uv_cache"]
            == "/workspace/redco/.runtime/stage-d-v4-8/uv-cache"
            and 'runtime_root="${REDCO_RUNTIME_ROOT:-$repo_root/.runtime/stage-d-v4-8}"'
            in path_preflight
            and 'test "$runtime_root" = "/workspace/redco/.runtime/stage-d-v4-8"'
            in path_preflight
            and '"/workspace/redco/.runtime/stage-d-v4-8/prime-env"'
            in path_preflight
            and '"/workspace/redco/.runtime/stage-d-v4-8/uv-cache"'
            in path_preflight
            and "wsl @arguments" in path_regression
            and '"bash", "-x", $preflight' in path_regression
            and 'if ($trace.Contains("+ mkdir -p"))' in path_regression
        ),
        "scientific_symmetry_and_claim_scope_frozen": (
            protocol["scientific_inheritance"]["shared_runtime_policy"]
            == (
                "The exact eager config must be used by the fixture, power audit, "
                "every future scientific arm and baseline, and every evaluation."
            )
            and protocol["scientific_inheritance"]["eager_equivalence_claim"]
            == "No numerical or distributional equivalence to CUDA-graph mode is claimed."
        ),
        "budget_arithmetic_exact": (
            math.isclose(
                budget["wallet_after_v4_9_usd"] - envelope,
                budget["headroom_usd"],
                abs_tol=1e-9,
            )
            and math.isclose(envelope, budget["remaining_full_envelope_usd"], abs_tol=1e-9)
            and budget["cumulative_support_spend_usd"] == terminal["pod"]["cost_usd"]
            and budget["remaining_support_ceiling_usd"]
            == terminal["resources"]["remaining_v4_9_cap_usd"]
            and budget["headroom_usd"] > 0
        ),
        "timeout_and_lifetime_cap_bounded": (
            "REDCO_V4_10_AUTHORIZED_TIMEOUT_SECONDS" in wrapper
            and 'test "$timeout_seconds" -le 17100' in wrapper
            and budget["maximum_wrapper_timeout_seconds"] == 17100
            and budget["maximum_total_billed_pod_lifetime_seconds_at_max_rate"]
            == 19800
            and budget["minimum_non_wrapper_lifetime_reserve_seconds"] == 2700
            and budget["maximum_wrapper_timeout_seconds"]
            + budget["minimum_non_wrapper_lifetime_reserve_seconds"]
            <= budget["maximum_total_billed_pod_lifetime_seconds_at_max_rate"]
        ),
        "one_final_deployment_and_no_automatic_repair": (
            protocol["repair_policy"]["maximum_v4_10_deployments"] == 1
            and protocol["repair_policy"]["automatic_repair"] == "forbidden"
            and protocol["repair_policy"]["successor_beyond_v4_10"]
            == "requires explicit user authorization and a new preregistration"
        ),
        "runtime_ledgers_required": (
            protocol["artifact_policy"]["runtime_ledgers"]
            == ["wall_time", "generated_tokens", "gpu_hours", "exact_billing"]
        ),
        "https_transfer_and_uv_only": (
            protocol["transfer"]["repository_https"]
            == "https://github.com/mihir-s-05/redco.git"
            and runtime["dependency_policy"] == "uv only; pip is forbidden"
            and "pip " not in wrapper
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d0-scaffold-support-preregistration-audit-v4-10",
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
            "configs/stage-d/stage-d0-scaffold-support-preregistration-v4-10.json"
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
