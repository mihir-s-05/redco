"""Prepare and aggregate exact-prefix Stage-C checkpoint scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} contains a non-object JSON line")
    return rows


def _sampled_action_node(trace: dict[str, Any]) -> dict[str, Any]:
    nodes = trace.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("trace is missing nodes")
    sampled = [
        node
        for node in nodes
        if isinstance(node, dict) and node.get("sampled") is True
    ]
    if len(sampled) != 1:
        raise ValueError("expected exactly one sampled node")
    return sampled[0]


def _prefix_and_action_token(trace: dict[str, Any]) -> tuple[list[int], int]:
    nodes = trace["nodes"]
    prefix: list[int] = []
    sampled_tokens: list[int] = []
    for node in nodes:
        token_ids = node.get("token_ids")
        mask = node.get("mask")
        if not isinstance(token_ids, list) or not isinstance(mask, list):
            raise ValueError("training trace is missing token ids or masks")
        if len(token_ids) != len(mask):
            raise ValueError("training trace token ids and masks differ in length")
        for token_id, trainable in zip(token_ids, mask, strict=True):
            if not isinstance(token_id, int) or not isinstance(trainable, bool):
                raise ValueError("training trace token metadata has invalid types")
            if trainable:
                sampled_tokens.append(token_id)
            else:
                if sampled_tokens:
                    raise ValueError("non-sampled token follows the sampled action")
                prefix.append(token_id)
    if len(sampled_tokens) != 1:
        raise ValueError("Stage-C action must be exactly one sampled token")
    return prefix, sampled_tokens[0]


def prepare_policy_cases(trace_paths: list[Path]) -> dict[str, Any]:
    """Extract exact rendered prefixes and digit token IDs from training traces."""
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    token_ids_by_action: dict[str, set[int]] = defaultdict(set)
    for path in trace_paths:
        for trace in _read_jsonl(path):
            agent = trace.get("agent")
            task = trace.get("task")
            if (
                not isinstance(agent, dict)
                or agent.get("name") not in {
                    "original",
                    "alternative_1",
                    "alternative_2",
                    "alternative_3",
                }
                or not isinstance(task, dict)
                or not isinstance(task.get("data"), dict)
            ):
                continue
            data = task["data"]
            probe = data.get("probe_name")
            route = data.get("context_route")
            actions = data.get("actions")
            if (
                not isinstance(probe, str)
                or not isinstance(route, str)
                or not isinstance(actions, list)
                or not actions
                or not all(isinstance(action, str) for action in actions)
            ):
                continue
            sampled_node = _sampled_action_node(trace)
            message = sampled_node.get("message")
            reply = message.get("content") if isinstance(message, dict) else None
            if not isinstance(reply, str) or reply.strip() not in actions:
                continue
            prefix, action_token_id = _prefix_and_action_token(trace)
            action = reply.strip()
            token_ids_by_action[action].add(action_token_id)
            candidates.setdefault(
                (probe, route),
                {
                    "case_id": f"{probe}:{route}",
                    "probe_name": probe,
                    "context_route": route,
                    "actions": actions,
                    "prefix_token_ids": prefix,
                    "prompt": data.get("prompt"),
                },
            )
    if not candidates:
        raise ValueError("no valid Stage-C policy cases were found")
    inconsistent = {
        action: sorted(token_ids)
        for action, token_ids in token_ids_by_action.items()
        if len(token_ids) != 1
    }
    if inconsistent:
        raise ValueError(f"an action used multiple token ids: {inconsistent}")
    action_token_ids = {
        action: next(iter(token_ids))
        for action, token_ids in sorted(token_ids_by_action.items())
    }
    missing = sorted(
        {
            action
            for case in candidates.values()
            for action in case["actions"]
            if action not in action_token_ids
        }
    )
    if missing:
        raise ValueError(f"no sampled token id was observed for actions: {missing}")
    cases = []
    for case in sorted(candidates.values(), key=lambda item: item["case_id"]):
        case["action_token_ids"] = {
            action: action_token_ids[action] for action in case["actions"]
        }
        cases.append(case)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c-policy-audit-cases",
        "source_traces": [str(path.as_posix()) for path in trace_paths],
        "case_count": len(cases),
        "action_token_ids": action_token_ids,
        "cases": cases,
    }
    signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return payload


def aggregate_policy_scores(raw: dict[str, Any]) -> dict[str, Any]:
    """Reduce GPU-produced per-case probabilities into preregistrable metrics."""
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("raw policy audit is missing models")
    summaries: dict[str, Any] = {}
    base_by_case: dict[str, dict[str, Any]] | None = None
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("name"), str):
            raise ValueError("raw policy audit has an invalid model entry")
        rows = model.get("cases")
        if not isinstance(rows, list) or not rows:
            raise ValueError("raw policy audit model has no cases")
        by_case = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
                raise ValueError("raw policy audit has an invalid case")
            for field in (
                "full_vocab_action_probabilities_t1",
                "full_vocab_action_probabilities_t2",
                "conditional_action_probabilities_t1",
                "conditional_action_probabilities_t2",
            ):
                values = row.get(field)
                if (
                    not isinstance(values, dict)
                    or not values
                    or not all(
                        isinstance(value, int | float)
                        and math.isfinite(float(value))
                        and 0 <= float(value) <= 1
                        for value in values.values()
                    )
                ):
                    raise ValueError(f"invalid {field} in {row['case_id']}")
            by_case[row["case_id"]] = row
        if model["name"] == "base":
            base_by_case = by_case
        needle_rows = [
            row for row in rows if row.get("probe_name") == "planted_needle"
        ]
        needle_t2 = [
            float(row["full_vocab_action_probabilities_t2"]["5"])
            for row in needle_rows
        ]
        summaries[model["name"]] = {
            "cases": len(rows),
            "greedy_allowed_rate": fmean(
                float(row.get("greedy_allowed_action") is not None) for row in rows
            ),
            "needle_cases": len(needle_rows),
            "needle_action_5_full_vocab_mass_t2": {
                "minimum": min(needle_t2),
                "mean": fmean(needle_t2),
                "maximum": max(needle_t2),
            },
            "needle_action_5_greedy_rate": fmean(
                float(row.get("greedy_allowed_action") == "5")
                for row in needle_rows
            ),
            "mean_full_vocab_kl_from_base_t1": (
                None
                if model["name"] == "base"
                else fmean(float(row["full_vocab_kl_from_base_t1"]) for row in rows)
            ),
            "mean_allowed_action_kl_from_base_t1": (
                None
                if model["name"] == "base"
                else fmean(
                    float(row["allowed_action_kl_from_base_t1"]) for row in rows
                )
            ),
        }
        if len(by_case) != len(rows):
            raise ValueError(f"model {model['name']} contains duplicate cases")
    if base_by_case is None:
        raise ValueError("raw policy audit must include a model named base")
    expected_cases = set(base_by_case)
    if any(
        {row["case_id"] for row in model["cases"]} != expected_cases
        for model in models
    ):
        raise ValueError("policy audit models do not contain identical cases")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c-policy-audit",
        "source": raw.get("source"),
        "summaries": summaries,
        "interpretation": {
            "needle_mass": (
                "Full-vocabulary probability at the live branch temperature (2.0); "
                "invalid tokens remain in the failure reward class."
            ),
            "movement": (
                "KL and greedy agreement quantify whether an adapter moved relative "
                "to initialization; they are diagnostics, not gate outcomes."
            ),
        },
    }
    signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--traces", type=Path, action="append", default=[])
    prepare.add_argument("--trace-root", type=Path, action="append", default=[])
    prepare.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--raw", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        trace_paths = list(args.traces)
        for trace_root in args.trace_root:
            trace_paths.extend(sorted(trace_root.glob("**/train/all/traces.jsonl")))
        if not trace_paths:
            parser.error("prepare requires --traces or --trace-root")
        report = prepare_policy_cases(trace_paths)
    else:
        report = aggregate_policy_scores(json.loads(args.raw.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
