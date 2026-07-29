"""Verify live constrained-root traces against the exact four-way policy."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from redco.analysis.stage_c3_power import ROUTES
from redco.analysis.stage_c5_constrained import (
    ROUTE_CHOICES,
    constrained_route_distribution,
)

CHOICE_LOGPROB_TOLERANCE_NATS = 0.02


def _route(reply: str) -> str | None:
    stripped = reply.strip()
    for route in ROUTES:
        if stripped == f"<route>{route}</route>":
            return route
    return None


def _choice_index(root_scores: dict[str, Any]) -> int:
    details = root_scores["temperature_2"]["token_details"]
    rows = [details[route] for route in ROUTES]
    divergent = [
        index
        for index in range(min(map(len, rows)))
        if len({row[index]["token_id"] for row in rows}) > 1
    ]
    if len(divergent) != 1:
        raise ValueError("root scores do not have one route-choice token")
    return divergent[0]


def verify_trace_file(
    traces_path: Path,
    root_scores: dict[str, Any],
    *,
    expected_context_traces: int,
) -> dict[str, Any]:
    """Verify request routing, output validity, and constrained logprobs."""
    traces = [
        json.loads(line)
        for line in traces_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    contexts = [
        trace
        for trace in traces
        if trace.get("agent", {}).get("name") == "context"
    ]
    probabilities = constrained_route_distribution(root_scores)
    choice_index = _choice_index(root_scores)
    expected_token_ids = {
        route: root_scores["temperature_2"]["token_details"][route][choice_index][
            "token_id"
        ]
        for route in ROUTES
    }
    observations = []
    for trace in contexts:
        choices = (
            trace.get("agent", {})
            .get("sampling", {})
            .get("extra_body", {})
            .get("structured_outputs", {})
            .get("choice")
        )
        sampled_nodes = [
            node for node in trace.get("nodes", []) if node.get("sampled") is True
        ]
        if len(sampled_nodes) != 1:
            raise ValueError("context trace must have one sampled node")
        node = sampled_nodes[0]
        reply = str(node.get("message", {}).get("content", ""))
        route = _route(reply)
        content_token_ids = [
            token_id
            for token_id, is_content in zip(
                node.get("token_ids", []),
                node.get("is_content", []),
                strict=True,
            )
            if is_content
        ]
        logprobs = [float(value) for value in node.get("logprobs", [])]
        if route is None or choice_index >= len(content_token_ids) or choice_index >= len(logprobs):
            observed_logprob = math.nan
            token_id = None
            error = math.inf
        else:
            observed_logprob = logprobs[choice_index]
            token_id = content_token_ids[choice_index]
            error = abs(observed_logprob - math.log(probabilities[route]))
        observations.append(
            {
                "route": route,
                "reply": reply,
                "choices_exact": choices == list(ROUTE_CHOICES),
                "choice_token_id": token_id,
                "expected_choice_token_id": (
                    expected_token_ids.get(route) if route is not None else None
                ),
                "observed_choice_logprob": observed_logprob,
                "expected_choice_logprob": (
                    math.log(probabilities[route]) if route is not None else None
                ),
                "absolute_logprob_error_nats": error,
            }
        )
    checks = {
        "exact_context_trace_count": len(contexts) == expected_context_traces,
        "every_request_uses_exact_four_choice_constraint": bool(observations)
        and all(row["choices_exact"] for row in observations),
        "every_completion_is_one_canonical_route": bool(observations)
        and all(row["route"] in ROUTES for row in observations),
        "every_choice_token_matches_exact_scorer": bool(observations)
        and all(
            row["choice_token_id"] == row["expected_choice_token_id"]
            for row in observations
        ),
        "every_choice_logprob_is_finite": bool(observations)
        and all(math.isfinite(row["observed_choice_logprob"]) for row in observations),
        "every_choice_logprob_matches_exact_categorical": bool(observations)
        and all(
            row["absolute_logprob_error_nats"] <= CHOICE_LOGPROB_TOLERANCE_NATS
            for row in observations
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "expected_context_traces": expected_context_traces,
        "constraint_logprob_tolerance_nats": CHOICE_LOGPROB_TOLERANCE_NATS,
        "constrained_route_probabilities_t2": probabilities,
        "observations": observations,
    }
