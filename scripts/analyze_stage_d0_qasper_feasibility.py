from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_trace_contract import audit_rlm_trace

EXPECTED_CHECKPOINT = (
    "Qwen/Qwen3-4B-Instruct-2507@"
    "cdbee75f17c01a7cc42f958dc650907174af0554"
)
EXPECTED_NATURAL_TRACES = 32
EXPECTED_FIXTURE_TRACES = 1
MEANINGFUL_F1_RANGE = 0.05
MINIMUM_INFORMATIVE_GROUPS = 5


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def load_episodes(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(
            value.get("traces"), list
        ):
            raise TypeError(f"{path}:{line_number} is not an episode object")
        rows.append(value)
    return rows


def collect_run(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes = load_episodes(run_dir / "traces.jsonl")
    summary = load_json(run_dir / "run-summary.json")
    return episodes, summary


def _score(trace: dict[str, Any]) -> dict[str, float]:
    info = trace.get("info") or {}
    evidence = info.get("evidence_selection") or {}
    score = evidence.get("score")
    if not isinstance(score, dict):
        raise ValueError(f"trace {trace.get('id')} has no deterministic score")
    return {
        key: float(score[key])
        for key in (
            "f1",
            "precision",
            "recall",
            "parseable",
            "all_predicted_spans_verbatim",
        )
    }


def _call_counts(trace: dict[str, Any]) -> tuple[int, int, int]:
    calls = trace.get("calls") or []
    root = child = 0
    for call in calls:
        depth = (call.get("rlm") or {}).get("depth")
        if depth == 0:
            root += 1
        elif isinstance(depth, int) and depth > 0:
            child += 1
    return len(calls), root, child


def _exact_call_contract(trace: dict[str, Any]) -> bool:
    nodes = trace.get("nodes") or []
    calls = trace.get("calls") or []
    if not calls:
        return False
    for call in calls:
        node_index = call.get("node")
        if (
            not isinstance(node_index, int)
            or isinstance(node_index, bool)
            or not 0 <= node_index < len(nodes)
        ):
            return False
        node = nodes[node_index]
        token_ids = node.get("token_ids") or []
        mask = node.get("mask") or []
        logprobs = node.get("logprobs") or []
        sampled_count = sum(item is True for item in mask)
        seed = (call.get("sampling") or {}).get("seed")
        if (
            node.get("sampled") is not True
            or not token_ids
            or len(token_ids) != len(mask)
            or sampled_count < 1
            or sampled_count != len(logprobs)
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed < 1
            or not isinstance(call.get("usage"), dict)
        ):
            return False
    return (trace.get("info") or {}).get("checkpoint_id") == EXPECTED_CHECKPOINT


def _seed_contract(
    traces: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        trace_id: record["seed"]
        for record in summary.get("records") or []
        for trace_id in record.get("trace_ids") or []
    }
    observed: dict[str, int | None] = {}
    for trace in traces:
        root_calls = [
            call
            for call in trace.get("calls") or []
            if (call.get("rlm") or {}).get("depth") == 0
        ]
        observed[str(trace.get("id"))] = (
            (root_calls[0].get("sampling") or {}).get("seed")
            if root_calls
            else None
        )
    return {
        "expected_by_trace": expected,
        "observed_root_seed_by_trace": observed,
        "unique_expected_seeds": len(set(expected.values())),
        "passes": (
            len(expected) == len(traces)
            and len(set(expected.values())) == len(traces)
            and observed == expected
        ),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def _natural_metrics(
    episodes: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    traces = [
        trace
        for episode in episodes
        for trace in episode.get("traces") or []
    ]
    rows = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        for trace in episode.get("traces") or []:
            score = _score(trace)
            total, root, child = _call_counts(trace)
            task = (trace.get("task") or {}).get("data") or {}
            row = {
                "episode_ok": episode.get("ok") is True,
                "trace_ok": trace.get("ok") is True,
                "trace_id": trace.get("id"),
                "example_id": task.get("example_id"),
                "score": score,
                "total_calls": total,
                "root_calls": root,
                "child_calls": child,
                "eligible_child_target": child >= 1,
                "exact_call_contract": _exact_call_contract(trace),
            }
            rows.append(row)
            groups[str(task.get("example_id"))].append(row)

    informative = {}
    format_failure_variance = {}
    for example_id, group in sorted(groups.items()):
        rewards = [row["score"]["f1"] for row in group]
        all_valid = all(
            row["score"]["parseable"] == 1.0
            and row["score"]["all_predicted_spans_verbatim"] == 1.0
            for row in group
        )
        reward_range = max(rewards) - min(rewards) if rewards else 0.0
        distinct_advantages = len(set(rewards)) >= 2
        group_result = {
            "rollouts": len(group),
            "all_parseable_and_verbatim": all_valid,
            "reward_min": min(rewards) if rewards else None,
            "reward_max": max(rewards) if rewards else None,
            "reward_range": reward_range,
            "distinct_rewards": len(set(rewards)),
            "informative": (
                len(group) == 4
                and all_valid
                and reward_range >= MEANINGFUL_F1_RANGE
                and distinct_advantages
            ),
        }
        informative[example_id] = group_result
        if not all_valid and reward_range > 0:
            format_failure_variance[example_id] = group_result

    parseable_count = sum(
        row["score"]["parseable"] == 1.0 for row in rows
    )
    verbatim_count = sum(
        row["score"]["all_predicted_spans_verbatim"] == 1.0 for row in rows
    )
    eligible_count = sum(row["eligible_child_target"] for row in rows)
    exact_count = sum(row["exact_call_contract"] for row in rows)
    informative_count = sum(
        result["informative"] for result in informative.values()
    )
    call_totals = [row["total_calls"] for row in rows]
    root_totals = [row["root_calls"] for row in rows]
    child_totals = [row["child_calls"] for row in rows]
    seed_contract = _seed_contract(traces, summary)
    mandatory = {
        "trace_count_exact": len(rows) == EXPECTED_NATURAL_TRACES,
        "all_episodes_and_traces_ok": (
            len(rows) == EXPECTED_NATURAL_TRACES
            and all(row["episode_ok"] and row["trace_ok"] for row in rows)
        ),
        "parseable_at_least_29_of_32": parseable_count >= 29,
        "all_verbatim_at_least_29_of_32": verbatim_count >= 29,
        "eligible_child_at_least_26_of_32": eligible_count >= 26,
        "all_exact_call_and_checkpoint_contracts": (
            exact_count == EXPECTED_NATURAL_TRACES
        ),
        "median_total_calls_at_least_4": (
            bool(call_totals) and statistics.median(call_totals) >= 4
        ),
        "median_root_calls_at_least_2": (
            bool(root_totals) and statistics.median(root_totals) >= 2
        ),
        "median_child_calls_at_least_1": (
            bool(child_totals) and statistics.median(child_totals) >= 1
        ),
        "informative_groups_at_least_5_of_8": informative_count
        >= MINIMUM_INFORMATIVE_GROUPS,
        "episode_addressed_seed_contract": seed_contract["passes"],
    }
    return {
        "trace_count": len(rows),
        "parseable_count": parseable_count,
        "all_verbatim_count": verbatim_count,
        "eligible_child_count": eligible_count,
        "exact_call_contract_count": exact_count,
        "median_total_calls": statistics.median(call_totals)
        if call_totals
        else 0,
        "median_root_calls": statistics.median(root_totals)
        if root_totals
        else 0,
        "median_child_calls": statistics.median(child_totals)
        if child_totals
        else 0,
        "informative_group_count": informative_count,
        "informative_groups": informative,
        "format_failure_variance_groups": format_failure_variance,
        "seed_contract": seed_contract,
        "mandatory_checks": mandatory,
        "passes": all(mandatory.values()),
        "rows": rows,
    }


def _fixture_metrics(
    episodes: list[dict[str, Any]],
    summary: dict[str, Any],
    replay: dict[str, Any],
    scorer_plumbing: dict[str, Any],
) -> dict[str, Any]:
    traces = [
        trace
        for episode in episodes
        for trace in episode.get("traces") or []
    ]
    contract = audit_rlm_trace(traces[0]).to_dict() if len(traces) == 1 else {}
    seed_contract = _seed_contract(traces, summary)
    pairs = replay.get("pairs") or []
    scoring_pairs = scorer_plumbing.get("pairs") or []
    mandatory = {
        "trace_count_exact": len(traces) == EXPECTED_FIXTURE_TRACES,
        "episode_and_trace_ok": (
            len(episodes) == 1
            and episodes[0].get("ok") is True
            and len(traces) == 1
            and traces[0].get("ok") is True
        ),
        "full_trace_contract": contract.get("stage_d_science_ready") is True,
        "episode_addressed_seed_contract": seed_contract["passes"],
        "at_least_two_child_targets": contract.get("child_calls", 0) >= 2,
        "paired_cache_path_plumbing": (
            len(pairs) >= 2
            and all(
                pair.get("terminal_artifacts_exact") is True
                and pair.get("cached_actions_exact") is True
                for pair in pairs
            )
        ),
        "distinct_alternative_actions": scorer_plumbing.get(
            "all_alternatives_distinct_from_original"
        )
        is True,
        "downstream_prompt_changed": scorer_plumbing.get(
            "all_downstream_prompts_changed"
        )
        is True,
        "offline_deterministic_scorer_recorded": (
            len(scoring_pairs) == len(pairs)
            and len(scoring_pairs) >= 2
            and all(
                isinstance(pair.get("precision"), (int, float))
                and isinstance(pair.get("recall"), (int, float))
                and isinstance(pair.get("f1"), (int, float))
                and isinstance(pair.get("parsed_spans"), list)
                for pair in scoring_pairs
            )
        ),
    }
    return {
        "trace_contract": contract,
        "seed_contract": seed_contract,
        "paired_branches": len(pairs),
        "mandatory_checks": mandatory,
        "passes": all(mandatory.values()),
        "scope": (
            "forced trace plus paired cache/path traversal and deterministic "
            "scorer plumbing; not real-task full-vs-sliced reward equivalence"
        ),
    }


def _budget_projection(
    natural_summary: dict[str, Any],
    resource: dict[str, Any],
) -> dict[str, Any]:
    episode_seconds = [
        float(record["wall_seconds"])
        for record in natural_summary.get("records") or []
    ]
    p50 = statistics.median(episode_seconds) if episode_seconds else 0.0
    p95 = _percentile(episode_seconds, 0.95)
    planned_rollouts = 3 * 2 * 8 * 4
    setup_seconds = float(resource["setup_and_download_seconds"])
    training_and_checkpoint_seconds = 900.0
    future_rate = float(resource["future_maximum_rate_usd_per_hour"])
    projected_seconds_pre_contingency = (
        setup_seconds
        + planned_rollouts * p95
        + training_and_checkpoint_seconds
    )
    projected_seconds = 1.30 * projected_seconds_pre_contingency
    projected_cost = projected_seconds / 3600 * future_rate
    original_d0_cap = float(resource["original_d0_cap_usd"])
    prior_d0_cost = float(resource["prior_d0_cost_usd"])
    smoke_cost = float(resource["smoke_cost_usd"])
    remaining_d0 = original_d0_cap - prior_d0_cost - smoke_cost
    wallet_after_smoke = float(resource["wallet_after_smoke_usd"])
    reserve = float(resource["recovery_reserve_usd"])
    mandatory = {
        "projection_fits_remaining_d0_allocation": projected_cost <= remaining_d0,
        "projection_preserves_wallet_reserve": projected_cost
        <= wallet_after_smoke - reserve,
    }
    return {
        "observed_episode_service_seconds_p50": p50,
        "observed_episode_service_seconds_p95": p95,
        "planned_minimum_stock_gate": {
            "seeds": 3,
            "optimizer_updates_per_seed": 2,
            "task_groups_per_update": 8,
            "rollouts_per_group": 4,
            "total_rollouts": planned_rollouts,
            "interpretation": (
                "minimum budget-feasibility design only; the stock learning "
                "protocol and its power rule require separate preregistration"
            ),
        },
        "setup_and_download_seconds": setup_seconds,
        "fixed_training_and_checkpoint_seconds": training_and_checkpoint_seconds,
        "contingency_multiplier": 1.30,
        "projected_wall_seconds": projected_seconds,
        "future_maximum_rate_usd_per_hour": future_rate,
        "projected_cost_usd": projected_cost,
        "remaining_d0_allocation_usd": remaining_d0,
        "wallet_after_smoke_usd": wallet_after_smoke,
        "recovery_reserve_usd": reserve,
        "mandatory_checks": mandatory,
        "passes": all(mandatory.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural-run", type=Path, required=True)
    parser.add_argument("--fixture-run", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--scorer-plumbing", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    natural_episodes, natural_summary = collect_run(args.natural_run)
    fixture_episodes, fixture_summary = collect_run(args.fixture_run)
    natural = _natural_metrics(natural_episodes, natural_summary)
    fixture = _fixture_metrics(
        fixture_episodes,
        fixture_summary,
        load_json(args.replay_report),
        load_json(args.scorer_plumbing),
    )
    budget = _budget_projection(natural_summary, load_json(args.resource))
    stock_authorized = (
        natural["passes"] and fixture["passes"] and budget["passes"]
    )
    result = {
        "schema_version": 1,
        "scope": "deterministic single-paper QASPER exact-evidence D0 feasibility",
        "natural_qasper": natural,
        "forced_trace_fixture": fixture,
        "budget_projection": budget,
        "decision": (
            "pass_stock_gate_authorized"
            if stock_authorized
            else "fail_stop_before_stock_gate"
        ),
        "stock_learning_gate_authorized": stock_authorized,
        "stage_d1_authorized": False,
        "stage_d1_blockers": [
            "The smoke is engineering evidence, not a powered learning result.",
            (
                "A real task restore/inject executor must independently establish "
                "full-vs-sliced terminal-state and F1 equivalence."
            ),
            (
                "The stock learning gate requires a separate frozen protocol and "
                "power rule."
            ),
        ],
        "forbidden_claims": [
            "GA-full",
            "alphaXiv incumbent reproduction",
            "real-task sliced/full reward equivalence",
            "ReDCO learning benefit",
            "Stage D1 clearance",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not stock_authorized:
        raise SystemExit("Stage D0 QASPER feasibility gate failed")


if __name__ == "__main__":
    main()
