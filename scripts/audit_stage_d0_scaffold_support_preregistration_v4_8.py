"""Audit the environment-only Stage D0 scaffold-support v4.8 successor."""

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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_signed(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    verify_signed_payload(payload)
    return payload


def audit(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = _load_json(root / protocol_path)
    parent_path = root / protocol["history"]["parent_protocol"]
    parent = _load_json(parent_path)
    parent_audit = _load_signed(
        root / protocol["history"]["parent_audit"]
    )
    terminal = _load_signed(
        root / protocol["history"]["parent_terminal_report"]
    )
    wrapper_path = root / protocol["runtime"]["wrapper"]
    preflight_path = root / protocol["runtime"]["path_preflight"]
    parent_runner_path = root / protocol["runtime"]["inherited_runner"]
    wrapper = wrapper_path.read_text(encoding="utf-8")
    preflight = preflight_path.read_text(encoding="utf-8")
    parent_runner = parent_runner_path.read_text(encoding="utf-8")

    source_results = {}
    for relative, expected in protocol["source_sha256"].items():
        actual = _sha256(root / relative)
        source_results[relative] = {
            "expected": expected,
            "actual": actual,
            "passes": actual == expected,
        }

    runtime = protocol["runtime"]
    expected_paths = runtime["exact_writable_paths"]
    budget = protocol["budget"]
    recomputed_envelope = (
        budget["remaining_support_ceiling_usd"]
        + budget["scientific_training_reserved_ceiling_usd"]
        + budget["science_evaluation_reserved_usd"]
        + budget["untouchable_reserve_usd"]
    )
    fixture = parent["frozen_sequence"][
        "step_3_previously_unobserved_fixture"
    ]
    power = parent["frozen_sequence"][
        "step_4_previously_unobserved_power_audit"
    ]

    checks = {
        "distinct_successor_frozen_before_request": (
            protocol["experiment"]
            == "stage-d0-scaffold-support-v4-8"
            and protocol["status"]
            == "frozen_before_any_v4_8_model_request"
        ),
        "parent_protocol_and_audit_exact": (
            _sha256(parent_path)
            == protocol["history"]["parent_protocol_sha256"]
            and _sha256(
                root / protocol["history"]["parent_audit"]
            )
            == protocol["history"]["parent_audit_sha256"]
            and parent_audit["passes"]
            and parent_audit["protocol_sha256"]
            == protocol["history"]["parent_protocol_sha256"]
        ),
        "terminal_report_signed_exact_and_zero_requests": (
            _sha256(
                root / protocol["history"]["parent_terminal_report"]
            )
            == protocol["history"]["parent_terminal_report_sha256"]
            and terminal["signed_payload_sha256"]
            == protocol["history"]["parent_terminal_report_signature"]
            and terminal["passes"]
            and all(
                count == 0
                for count in terminal["model_request_counts"].values()
            )
        ),
        "all_scientific_fields_inherited": (
            protocol["scientific_inheritance"]["policy"]
            == "byte-identical through the inherited v4.7 runner"
            and fixture["dataset"]
            == protocol["scientific_inheritance"]["fixture_dataset"]
            and fixture["master_seed"]
            == protocol["scientific_inheritance"]["fixture_master_seed"]
            and fixture["replay_master_seed"]
            == protocol["scientific_inheritance"][
                "fixture_replay_master_seed"
            ]
            and power["dataset"]
            == protocol["scientific_inheritance"]["power_dataset"]
            and power["master_seed"]
            == protocol["scientific_inheritance"]["power_master_seed"]
            and power["replay_master_seed"]
            == protocol["scientific_inheritance"][
                "power_replay_master_seed"
            ]
            and power["papers"] == 64
            and power["K_including_regenerated_original"] == 4
            and power["minimum_f1_range"] == 0.05
            and power["joint_pass_rule"]
            == "at least 58 of 64 unique papers are both eligible and informative"
        ),
        "source_hashes_exact": (
            bool(source_results)
            and all(item["passes"] for item in source_results.values())
        ),
        "parent_runner_unchanged_and_request_boundaries_retained": (
            _sha256(parent_runner_path)
            == runtime["inherited_runner_sha256"]
            and parent_runner.count("REQUESTS_STARTED") == 6
            and "source \"$instrumented_tail\"" in parent_runner
        ),
        "wrapper_uses_only_writable_runtime_state": (
            'runtime_root="$repo_root/.runtime/stage-d-v4-8"'
            in wrapper
            and 'REDCO_UV_ENVIRONMENT="$runtime_root/prime-env"'
            in wrapper
            and 'REDCO_UV_CACHE_DIR="$runtime_root/uv-cache"'
            in wrapper
            and 'REDCO_RUN_ROOT="$run_root"' in wrapper
            and 'XDG_CONFIG_HOME="$repo_root/.runtime-config"'
            in parent_runner
            and "/workspace/.venv-prime" not in wrapper
            and "/workspace/.uv-cache-prime" not in wrapper
        ),
        "outer_sources_verified_before_bootstrap": (
            wrapper.index('python3 - "$protocol"')
            < wrapper.index("sudo -n install -d")
            and 'protocol["source_sha256"].items()' in wrapper
            and "v4.8 bootstrap hash mismatch" in wrapper
        ),
        "root_owned_paths_repaired_narrowly_before_probe": (
            wrapper.index("sudo -n install -d")
            < wrapper.index(
                "preflight_stage_d_runtime_paths_v4_8.sh"
            )
            < wrapper.index(
                "run_stage_d0_scaffold_support_v4_7.sh"
            )
            and "/workspace/models" in wrapper
            and "/workspace/.cache/huggingface" in wrapper
            and "/workspace/evidence_context.txt" in wrapper
            and "sudo -n" in wrapper
        ),
        "all_absolute_runtime_paths_probed_exactly": (
            len(expected_paths) == 11
            and len(set(expected_paths)) == len(expected_paths)
            and all(f'"{path}"' in preflight for path in expected_paths)
            and 'mktemp -d "$path/' in preflight
            and 'rmdir "$probe"' in preflight
            and "test ! -e \"$probe\"" in preflight
            and protocol["runtime"]["exact_writable_file"]
            == "/workspace/evidence_context.txt"
            and 'context_path="/workspace/evidence_context.txt"'
            in preflight
            and "context-write-read-truncate-probe" in preflight
            and ': >"$context_path"' in preflight
            and 'test ! -s "$context_path"' in preflight
        ),
        "disk_floor_is_executable_and_exact": (
            runtime["minimum_free_kib"] == 47_185_920
            and "df -Pk /workspace/redco" in preflight
            and "df -Pk /tmp" in preflight
            and preflight.count(
                'test "$workspace_free_kib" -ge "$minimum_free_kib"'
            )
            == 1
            and preflight.count(
                'test "$tmp_free_kib" -ge "$minimum_free_kib"'
            )
            == 1
        ),
        "uv_only_and_no_pip": (
            protocol["runtime"]["dependency_policy"]
            == "uv only; pip is forbidden"
            and "pip " not in wrapper
            and "pip " not in preflight
        ),
        "budget_arithmetic_and_added_balance_exact": (
            math.isclose(
                budget["wallet_verified_after_top_up_usd"],
                terminal["resources"]["wallet_after_termination_usd"],
                abs_tol=1e-9,
            )
            and math.isclose(
                budget["prior_support_spend_usd"],
                terminal["resources"]["total_v4_7_cost_usd"],
                abs_tol=1e-9,
            )
            and math.isclose(
                recomputed_envelope,
                budget["remaining_full_envelope_usd"],
                abs_tol=1e-9,
            )
            and math.isclose(
                budget["wallet_verified_after_top_up_usd"]
                - recomputed_envelope,
                budget["headroom_usd"],
                abs_tol=1e-9,
            )
            and budget["headroom_usd"] > 0
            and budget["cumulative_v4_8_cap_usd"]
            == budget["remaining_support_ceiling_usd"]
        ),
        "cumulative_budget_timeout_is_executable": (
            "REDCO_V4_8_AUTHORIZED_TIMEOUT_SECONDS" in wrapper
            and 'test "$timeout_seconds" -le 21600' in wrapper
            and "timeout --signal=TERM --kill-after=120" in wrapper
            and protocol["budget"]["hardware_amendment_rule"]
            == (
                "rate_usd_per_hour * authorized_timeout_seconds / 3600 "
                "+ cumulative_v4_8_spend_usd must be no greater than "
                "cumulative_v4_8_cap_usd"
            )
            and protocol["budget"]["redeployment_budget_rule"]
            == "the repair inherits the depleted cumulative cap; it never resets"
        ),
        "preflight_evidence_is_added_to_final_hash_manifest": (
            wrapper.index(
                '"$REDCO_RUNTIME_PREFLIGHT_REPORT"'
            )
            < wrapper.rindex("artifact-sha256.txt")
            and "find \"$repo_root/$run_root\"" in wrapper
        ),
        "bounded_environment_repair_not_scientific_retry": (
            protocol["repair_policy"][
                "maximum_outcome_independent_redeployments"
            ]
            == 1
            and "before the first fixture request"
            in protocol["repair_policy"]["eligible_boundary"]
            and protocol["repair_policy"][
                "after_fixture_request_started"
            ].startswith("terminal")
            and protocol["repair_policy"][
                "after_power_request_started"
            ].startswith("terminal")
            and protocol["repair_policy"]["frozen_science_changes"]
            == "forbidden"
        ),
        "no_scientific_training_authorized": (
            protocol["authorization"][
                "scientific_training_model_calls"
            ]
            == 0
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": (
                "stage-d0-scaffold-support-preregistration-audit-v4-8"
            ),
            "protocol": protocol_path.as_posix(),
            "protocol_sha256": _sha256(root / protocol_path),
            "source_results": source_results,
            "recomputed_remaining_envelope_usd": recomputed_envelope,
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
            "stage-d0-scaffold-support-preregistration-v4-8.json"
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
