"""Aggregate frozen CPU and live replay evidence into the Stage-B Gate GB decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from redco.contracts import canonical_json


@dataclass(frozen=True, slots=True)
class PairedEquivalenceInterval:
    pairs: int
    confidence: float
    margin: float
    mean_difference: float
    half_width: float
    lower: float
    upper: float
    passed: bool


@dataclass(frozen=True, slots=True)
class GpuResourceMetrics:
    samples: int
    elapsed_seconds: float
    mean_utilization_fraction: float
    utilization_weighted_gpu_seconds: float
    peak_memory_mib: float
    mean_power_watts: float
    energy_watt_hours: float
    passed_minimum_samples: bool


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], payload)


def _verify_signed_report(payload: dict[str, Any]) -> bool:
    claimed = payload.get("report_sha256")
    if not isinstance(claimed, str):
        return False
    unsigned = dict(payload)
    unsigned.pop("report_sha256")
    return hashlib.sha256(canonical_json(unsigned)).hexdigest() == claimed


def _paired_equivalence_interval(
    differences: list[float],
    *,
    confidence: float,
    margin: float,
) -> PairedEquivalenceInterval:
    if not differences:
        raise ValueError("paired differences must be non-empty")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if not 0 < margin < 1:
        raise ValueError("margin must be in (0, 1)")
    if any(not -1 <= value <= 1 for value in differences):
        raise ValueError("paired reward differences must be in [-1, 1]")
    alpha = 1.0 - confidence
    half_width = math.sqrt(2.0 * math.log(2.0 / alpha) / len(differences))
    mean = math.fsum(differences) / len(differences)
    lower = mean - half_width
    upper = mean + half_width
    return PairedEquivalenceInterval(
        pairs=len(differences),
        confidence=confidence,
        margin=margin,
        mean_difference=mean,
        half_width=half_width,
        lower=lower,
        upper=upper,
        passed=lower > -margin and upper < margin,
    )


def _gpu_resource_metrics(
    path: Path,
    *,
    minimum_samples: int,
) -> GpuResourceMetrics:
    rows: list[tuple[float, float, float, float]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#"):
                continue
            if len(row) != 4:
                raise ValueError("GPU sample rows need epoch, utilization, memory, power")
            rows.append(
                (
                    float(row[0].strip()),
                    float(row[1].strip()),
                    float(row[2].strip()),
                    float(row[3].strip()),
                )
            )
    if len(rows) < 2:
        raise ValueError("at least two GPU samples are required")
    rows.sort(key=lambda item: item[0])
    elapsed = rows[-1][0] - rows[0][0]
    if elapsed <= 0:
        raise ValueError("GPU sample timestamps must span positive time")
    weighted_seconds = 0.0
    energy_watt_seconds = 0.0
    for current, following in pairwise(rows):
        interval = following[0] - current[0]
        if interval <= 0:
            raise ValueError("GPU sample timestamps must be strictly increasing")
        weighted_seconds += interval * current[1] / 100.0
        energy_watt_seconds += interval * current[3]
    return GpuResourceMetrics(
        samples=len(rows),
        elapsed_seconds=elapsed,
        mean_utilization_fraction=weighted_seconds / elapsed,
        utilization_weighted_gpu_seconds=weighted_seconds,
        peak_memory_mib=max(row[2] for row in rows),
        mean_power_watts=energy_watt_seconds / elapsed,
        energy_watt_hours=energy_watt_seconds / 3600.0,
        passed_minimum_samples=len(rows) >= minimum_samples,
    )


def evaluate_gate_gb(
    *,
    static_path: Path,
    dynamic_path: Path,
    stochastic_path: Path,
    strict_trace_path: Path,
    broad_path: Path,
    gpu_samples_path: Path,
    expected_static_sha256: str,
    expected_dynamic_sha256: str,
    expected_stochastic_sha256: str,
    expected_strict_trace_sha256: str,
    minimum_live_pairs: int,
    live_confidence: float,
    overall_reward_margin: float,
    per_target_reward_margin: float,
    minimum_distinct_candidate_fraction: float,
    maximum_sliced_policy_event_fraction: float,
    minimum_gpu_samples: int,
) -> dict[str, Any]:
    paths = {
        "static_cpu": static_path,
        "dynamic_cpu": dynamic_path,
        "stochastic_cpu": stochastic_path,
        "strict_trace": strict_trace_path,
        "broad_live": broad_path,
        "gpu_samples": gpu_samples_path,
    }
    file_hashes = {name: _file_sha256(path) for name, path in paths.items()}
    expected_hashes_exact = {
        "static_cpu": file_hashes["static_cpu"] == expected_static_sha256,
        "dynamic_cpu": file_hashes["dynamic_cpu"] == expected_dynamic_sha256,
        "stochastic_cpu": (
            file_hashes["stochastic_cpu"] == expected_stochastic_sha256
        ),
        "strict_trace": (
            file_hashes["strict_trace"] == expected_strict_trace_sha256
        ),
    }
    static = _load_object(static_path)
    dynamic = _load_object(dynamic_path)
    stochastic = _load_object(stochastic_path)
    strict = _load_object(strict_trace_path)
    broad = _load_object(broad_path)

    static_passed = (
        _verify_signed_report(static)
        and static.get("interventions") == 10_000
        and static.get("deterministic_failures") == 0
        and static.get("all_six_dependency_kinds_exercised") is True
        and static.get("hand_audit_passed") is True
        and static.get("passed_cpu_gate") is True
    )
    dynamic_passed = (
        _verify_signed_report(dynamic)
        and cast(int, dynamic.get("paired_branches", 0)) >= 3_000
        and cast(int, dynamic.get("topology_divergences", 0)) >= 2_000
        and dynamic.get("deterministic_failures") == 0
        and dynamic.get("passed_rlm_shaped_cpu_proxy") is True
    )
    stochastic_passed = (
        _verify_signed_report(stochastic)
        and cast(int, stochastic.get("paired_branches", 0)) >= 25_000
        and stochastic.get("deterministic_state_mismatches") == 0
        and stochastic.get("passed_stochastic_gate") is True
    )
    provenance = cast(dict[str, Any], strict.get("provenance", {}))
    strict_trace_passed = (
        strict.get("passed") is True
        and strict.get("ready_for_empirical_branch_pair_replay") is True
        and provenance.get("structural_model_call_coverage") == 1.0
        and provenance.get("exact_prompt_provenance_coverage") == 1.0
        and provenance.get("exact_recursive_parent_coverage") == 1.0
        and provenance.get("exact_cross_component_links") == 4
        and provenance.get("cross_component_fallbacks") == 0
        and provenance.get("unresolved_cross_component_links") == 0
    )

    pairs = cast(list[dict[str, Any]], broad.get("pairs", []))
    differences = [
        float(pair["full_suffix"]["reward"]) - float(pair["sliced"]["reward"])
        for pair in pairs
    ]
    target_differences: dict[str, list[float]] = {}
    for pair, difference in zip(pairs, differences, strict=True):
        target = str(pair["target_node_id"])
        target_differences.setdefault(target, []).append(difference)
    overall_interval = _paired_equivalence_interval(
        differences,
        confidence=live_confidence,
        margin=overall_reward_margin,
    )
    target_intervals = {
        target: _paired_equivalence_interval(
            values,
            confidence=live_confidence,
            margin=per_target_reward_margin,
        )
        for target, values in sorted(target_differences.items())
    }
    distinct_fractions = cast(
        dict[str, float],
        broad.get("distinct_candidate_fraction_by_target", {}),
    )
    target_metrics = cast(
        list[dict[str, Any]],
        broad.get("target_metrics", []),
    )
    target_meter_fields = {
        "paired_branches",
        "alternative_action_generated_tokens",
        "downstream_generated_tokens",
        "generation_prompt_tokens",
        "model_request_wall_seconds",
        "full_suffix_policy_events_visited",
        "sliced_policy_events_visited",
        "sliced_policy_event_fraction",
        "full_arm_cost",
        "sliced_arm_cost",
    }
    live_target_meters_complete = (
        len(target_metrics) == 4
        and all(metric.get("target_agent_depth") == 1 for metric in target_metrics)
        and all(
            target_meter_fields <= metric.keys() for metric in target_metrics
        )
    )
    broad_passed = (
        _verify_signed_report(broad)
        and broad.get("paired_branches") == len(pairs)
        and len(pairs) >= minimum_live_pairs
        and broad.get("target_count") == 4
        and broad.get("deterministic_terminal_mismatches") == 0
        and broad.get("reward_mismatches") == 0
        and broad.get("cached_action_mismatches") == 0
        and broad.get("lossless_hybrid_preflight_exact") is True
        and broad.get("same_prompt_same_seed_reproducibility", {}).get("exact")
        is True
        and broad.get("distinct_candidate_gate_passed") is True
        and len(distinct_fractions) == 4
        and all(
            fraction >= minimum_distinct_candidate_fraction
            for fraction in distinct_fractions.values()
        )
        and float(broad.get("sliced_policy_event_fraction", 1.0))
        <= maximum_sliced_policy_event_fraction
        and overall_interval.passed
        and len(target_intervals) == 4
        and all(interval.passed for interval in target_intervals.values())
        and live_target_meters_complete
    )
    gpu = _gpu_resource_metrics(
        gpu_samples_path,
        minimum_samples=minimum_gpu_samples,
    )
    baseline_tokens = int(broad.get("baseline_generated_tokens", 0))
    candidate_tokens = int(
        broad.get("alternative_action_generated_tokens", 0)
    )
    downstream_tokens = int(broad.get("downstream_generated_tokens", 0))
    if baseline_tokens <= 0 or not pairs:
        raise ValueError("broad report needs positive baseline tokens and pairs")
    average_candidate_tokens = candidate_tokens / len(pairs)
    average_downstream_tokens = downstream_tokens / len(pairs)
    stage_c_alternatives = 3
    projected_stage_c_policy_token_raf = 1.0 + stage_c_alternatives * (
        average_candidate_tokens + average_downstream_tokens
    ) / baseline_tokens
    dynamic_full_work = cast(
        dict[str, Any],
        dynamic.get("full_work_by_role", {}),
    )
    dynamic_sliced_work = cast(
        dict[str, Any],
        dynamic.get("sliced_work_by_role", {}),
    )
    expected_roles = {"environment", "judge", "root_policy", "subcall_policy"}
    raf_meters_complete = (
        live_target_meters_complete
        and expected_roles <= dynamic_full_work.keys()
        and expected_roles <= dynamic_sliced_work.keys()
        and gpu.passed_minimum_samples
        and cast(int, broad.get("alternative_action_generated_tokens", 0)) > 0
        and cast(int, broad.get("downstream_generated_tokens", 0)) > 0
        and cast(float, broad.get("model_request_wall_seconds", 0.0)) > 0
        and cast(dict[str, Any], broad.get("full_arm_cost", {})).get(
            "storage_bytes", 0
        )
        > 0
        and cast(dict[str, Any], broad.get("sliced_arm_cost", {})).get(
            "storage_bytes", 0
        )
        > 0
    )
    all_checks = {
        "expected_input_file_hashes_exact": all(expected_hashes_exact.values()),
        "static_deterministic_cpu_campaign": static_passed,
        "dynamic_topology_cpu_campaign": dynamic_passed,
        "stochastic_reward_cpu_equivalence": stochastic_passed,
        "strict_live_trace_provenance": strict_trace_passed,
        "broad_frozen_model_live_replay": broad_passed,
        "gpu_resource_metering": gpu.passed_minimum_samples,
        "raf_meters_by_role_and_live_target": raf_meters_complete,
    }
    passed = all(all_checks.values())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate": "GB",
        "gate_gb_cleared": passed,
        "decision_scope": (
            "Stage-B replay correctness for the fixed-topology four-child live "
            "RLM protocol, supplemented by randomized static and dynamic-topology "
            "CPU campaigns and seeded stochastic-reward equivalence."
        ),
        "mandatory_checks": all_checks,
        "expected_input_file_hashes_exact": expected_hashes_exact,
        "input_file_sha256": file_hashes,
        "evidence": {
            "static_cpu_interventions": static.get("interventions"),
            "dynamic_cpu_paired_branches": dynamic.get("paired_branches"),
            "dynamic_cpu_topology_divergences": dynamic.get(
                "topology_divergences"
            ),
            "stochastic_cpu_paired_branches": stochastic.get("paired_branches"),
            "strict_trace_exact_links": provenance.get(
                "exact_cross_component_links"
            ),
            "live_paired_branches": len(pairs),
            "live_distinct_candidate_fraction_by_target": distinct_fractions,
            "live_sliced_policy_event_fraction": broad.get(
                "sliced_policy_event_fraction"
            ),
            "live_reward_overall_equivalence": asdict(overall_interval),
            "live_reward_per_target_equivalence": {
                target: asdict(interval)
                for target, interval in target_intervals.items()
            },
            "raf_meters": {
                "live_policy_depth_one_by_target": target_metrics,
                "dynamic_full_work_by_role": dynamic_full_work,
                "dynamic_sliced_work_by_role": dynamic_sliced_work,
                "live_alternative_action_generated_tokens": broad.get(
                    "alternative_action_generated_tokens"
                ),
                "live_regenerated_downstream_policy_tokens": broad.get(
                    "downstream_generated_tokens"
                ),
                "live_generation_prompt_tokens": broad.get(
                    "generation_prompt_tokens"
                ),
                "live_model_request_wall_seconds": broad.get(
                    "model_request_wall_seconds"
                ),
                "live_full_arm_cost": broad.get("full_arm_cost"),
                "live_sliced_arm_cost": broad.get("sliced_arm_cost"),
                "live_judge_calls": 0,
                "live_environment_events": 0,
                "protocol_note": (
                    "The frozen live protocol has policy calls only; environment "
                    "and judge roles are exercised and metered in the dynamic "
                    "CPU campaign."
                ),
            },
            "stage_c_n4_compute_projection": {
                "branch_group_size": 4,
                "alternative_branches": stage_c_alternatives,
                "continuations_per_branch": 1,
                "baseline_generated_tokens": baseline_tokens,
                "average_alternative_action_generated_tokens": (
                    average_candidate_tokens
                ),
                "average_regenerated_downstream_policy_tokens": (
                    average_downstream_tokens
                ),
                "projected_policy_token_raf": (
                    projected_stage_c_policy_token_raf
                ),
                "measured_sliced_policy_event_fraction": broad.get(
                    "sliced_policy_event_fraction"
                ),
                "interpretation": (
                    "Projection normalizes the 1,024-alternative stress campaign "
                    "back to the Stage-C n=4 branch group. It is trace-specific "
                    "and not a production cost forecast."
                ),
            },
        },
        "resource_use": asdict(gpu),
        "limitations": [
            "The live campaign branches only four depth-one child states from "
            "one recorded fixed-topology trace.",
            "Dynamic topology divergence is exercised by the CPU RLM-shaped "
            "campaign, not by the frozen-model live trace.",
            "The live reward is deterministic exact-match; the separately "
            "seeded CPU campaign supplies stochastic-reward equivalence.",
            "Gate GB is a replay-correctness gate and does not establish "
            "training effectiveness or estimator quality.",
        ],
    }
    payload["report_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--stochastic", type=Path, required=True)
    parser.add_argument("--strict-trace", type=Path, required=True)
    parser.add_argument("--broad", type=Path, required=True)
    parser.add_argument("--gpu-samples", type=Path, required=True)
    parser.add_argument("--expected-static-sha256", required=True)
    parser.add_argument("--expected-dynamic-sha256", required=True)
    parser.add_argument("--expected-stochastic-sha256", required=True)
    parser.add_argument("--expected-strict-trace-sha256", required=True)
    parser.add_argument("--minimum-live-pairs", type=int, required=True)
    parser.add_argument("--live-confidence", type=float, required=True)
    parser.add_argument("--overall-reward-margin", type=float, required=True)
    parser.add_argument("--per-target-reward-margin", type=float, required=True)
    parser.add_argument(
        "--minimum-distinct-candidate-fraction",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--maximum-sliced-policy-event-fraction",
        type=float,
        required=True,
    )
    parser.add_argument("--minimum-gpu-samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = evaluate_gate_gb(
        static_path=args.static,
        dynamic_path=args.dynamic,
        stochastic_path=args.stochastic,
        strict_trace_path=args.strict_trace,
        broad_path=args.broad,
        gpu_samples_path=args.gpu_samples,
        expected_static_sha256=args.expected_static_sha256,
        expected_dynamic_sha256=args.expected_dynamic_sha256,
        expected_stochastic_sha256=args.expected_stochastic_sha256,
        expected_strict_trace_sha256=args.expected_strict_trace_sha256,
        minimum_live_pairs=args.minimum_live_pairs,
        live_confidence=args.live_confidence,
        overall_reward_margin=args.overall_reward_margin,
        per_target_reward_margin=args.per_target_reward_margin,
        minimum_distinct_candidate_fraction=(
            args.minimum_distinct_candidate_fraction
        ),
        maximum_sliced_policy_event_fraction=(
            args.maximum_sliced_policy_event_fraction
        ),
        minimum_gpu_samples=args.minimum_gpu_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(payload) + b"\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["gate_gb_cleared"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
