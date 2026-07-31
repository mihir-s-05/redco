"""Audit the frozen root-only Stage D scaffold support block."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import sign_payload
from redco.integrations.verifiers_provenance import import_trace
from redco.integrations.verifiers_trace import (
    extract_policy_calls,
    load_trace_records,
)

from .stage_d_target_eligibility import (
    _policy_node_ids,
    _target_behavior_logprobs,
)


def _derive_episode_seed(
    master_seed: str, example_id: str, replicate: int
) -> int:
    payload = json.dumps(
        {
            "master_seed": master_seed,
            "example_id": example_id,
            "replicate": replicate,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**31 - 1
    ) + 1


def _percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _row(trace: dict[str, Any]) -> dict[str, Any]:
    calls = tuple(
        sorted(
            extract_policy_calls(trace), key=lambda call: call.call_index
        )
    )
    roots = [
        call
        for call in calls
        if call.agent_depth == 0 and call.turn_index is not None
    ]
    children = [call for call in calls if call.agent_depth == 1]
    selected = children[0] if children else None
    provenance = import_trace(trace, source_bytes=0)
    node_ids = _policy_node_ids(provenance)
    downstream_root = False
    behavior_logprobs = False
    if selected is not None and selected.call_index in node_ids:
        selected_node = node_ids[selected.call_index]
        descendants = provenance.graph.descendants(selected_node)
        downstream_root = any(
            call.call_index > selected.call_index
            and call.agent_depth == 0
            and node_ids.get(call.call_index) in descendants
            for call in calls
        )
        behavior_logprobs = bool(
            _target_behavior_logprobs(trace, selected)
        )
    exact_calls = bool(calls) and all(
        call.prompt_token_ids
        and call.action_token_ids
        and call.checkpoint_id != "unknown"
        and call.decoding_config_hash
        and call.event_seed is not None
        for call in calls
    )
    score = (
        ((trace.get("info") or {}).get("evidence_selection") or {})
        .get("score")
        or {}
    )
    parseable = score.get("parseable") == 1.0
    verbatim = score.get("all_predicted_spans_verbatim") == 1.0
    task = (trace.get("task") or {}).get("data") or {}
    precursor_eligible = (
        trace.get("ok") is True
        and len(roots) >= 2
        and 1 <= len(children) <= 2
        and exact_calls
        and behavior_logprobs
        and downstream_root
        and parseable
        and verbatim
    )
    return {
        "trace_id": trace.get("id"),
        "example_id": task.get("example_id"),
        "paper_id": task.get("paper_id"),
        "root_calls": len(roots),
        "child_calls": len(children),
        "exact_call_contract": exact_calls,
        "behavior_logprobs_present": behavior_logprobs,
        "selected_child_reaches_later_root": downstream_root,
        "parseable": parseable,
        "verbatim": verbatim,
        "precursor_eligible": precursor_eligible,
    }


def evaluate(path: Path, summary_path: Path) -> dict[str, Any]:
    traces = load_trace_records(path)
    traces_by_id = {str(trace.get("id")): trace for trace in traces}
    if len(traces_by_id) != len(traces):
        raise ValueError("trace IDs must be unique")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    master_seed = str(summary["master_seed"])
    records = summary.get("records") or []
    rows = []
    referenced_trace_ids: set[str] = set()
    for record in records:
        example_id = str(record["example_id"])
        replicate = int(record["replicate"])
        expected_seed = _derive_episode_seed(
            master_seed, example_id, replicate
        )
        trace_ids = [str(value) for value in record.get("trace_ids") or []]
        trace = (
            traces_by_id.get(trace_ids[0])
            if len(trace_ids) == 1
            else None
        )
        if trace is None:
            row = {
                "trace_id": f"missing::{record['slot_id']}",
                "example_id": example_id,
                "paper_id": None,
                "root_calls": 0,
                "child_calls": 0,
                "exact_call_contract": False,
                "behavior_logprobs_present": False,
                "selected_child_reaches_later_root": False,
                "parseable": False,
                "verbatim": False,
                "precursor_eligible": False,
                "reason": "missing_or_nonunique_trace",
            }
            observed_seed = None
        else:
            referenced_trace_ids.add(trace_ids[0])
            try:
                row = _row(trace)
            except (IndexError, KeyError, TypeError, ValueError) as error:
                row = {
                    "trace_id": trace.get("id"),
                    "example_id": example_id,
                    "paper_id": (
                        ((trace.get("task") or {}).get("data") or {}).get(
                            "paper_id"
                        )
                    ),
                    "root_calls": 0,
                    "child_calls": 0,
                    "exact_call_contract": False,
                    "behavior_logprobs_present": False,
                    "selected_child_reaches_later_root": False,
                    "parseable": False,
                    "verbatim": False,
                    "precursor_eligible": False,
                    "reason": f"{type(error).__name__}: {error}",
                }
            root_calls = [
                call
                for call in extract_policy_calls(trace)
                if call.agent_depth == 0
            ]
            observed_seed = (
                root_calls[0].event_seed if root_calls else None
            )
        row.update(
            {
                "slot_id": record["slot_id"],
                "replicate": replicate,
                "expected_seed": expected_seed,
                "recorded_plan_seed": record["seed"],
                "observed_root_seed": observed_seed,
                "plan_seed_contract": expected_seed == record["seed"],
                "observed_seed_contract": (
                    observed_seed is None
                    or record["seed"] == observed_seed
                ),
            }
        )
        row["precursor_eligible"] = (
            row["precursor_eligible"]
            and row["plan_seed_contract"]
            and row["observed_seed_contract"]
        )
        rows.append(row)
    child_counts = [row["child_calls"] for row in rows]
    example_counts = Counter(str(row["example_id"]) for row in rows)
    successes = sum(row["precursor_eligible"] for row in rows)
    checks = {
        "exactly_64_rollouts": len(rows) == 64,
        "all_traces_accounted_for_once": (
            referenced_trace_ids == set(traces_by_id)
        ),
        "eight_examples_with_eight_seeds_each": (
            len(example_counts) == 8
            and set(example_counts.values()) == {8}
        ),
        "at_least_58_precursor_eligible": successes >= 58,
        "all_64_episode_seeds_unique_and_exact": (
            len({row["expected_seed"] for row in rows}) == 64
            and all(row["plan_seed_contract"] for row in rows)
            and all(row["observed_seed_contract"] for row in rows)
        ),
        "median_child_calls_in_cost_band": (
            bool(child_counts)
            and 1 <= statistics.median(child_counts) <= 2
        ),
        "p95_child_calls_at_most_2": (
            bool(child_counts) and _percentile(child_counts, 0.95) <= 2
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-root-only-scaffold-support",
            "scope": (
                "Binary few-shot-to-SFT cascade trigger only. This does not "
                "measure branch informativeness and cannot authorize science."
            ),
            "rollouts": len(rows),
            "unique_examples": len(example_counts),
            "precursor_eligible": successes,
            "required_precursor_eligible": 58,
            "median_child_calls": (
                statistics.median(child_counts) if child_counts else 0
            ),
            "p95_child_calls": (
                _percentile(child_counts, 0.95) if child_counts else 0
            ),
            "checks": checks,
            "passes": all(checks.values()),
            "cascade_disposition": (
                "use_shared_fewshot_initialization"
                if all(checks.values())
                else "run_exactly_the_prefrozen_step8_sft_fallback"
            ),
            "rows": rows,
        }
    )
