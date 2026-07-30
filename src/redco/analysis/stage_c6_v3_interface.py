"""Verify the exact constrained root-likelihood path used by Stage C6 v3."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from redco.analysis.stage_c3_power import ROUTES
from redco.analysis.stage_c5_constrained import (
    ROUTE_CHOICES,
    constrained_route_distribution,
)

TRANSPORT_TOLERANCE_NATS = 1e-5


def _choice_metadata(
    root_scores: dict[str, Any],
) -> tuple[dict[int, str], dict[str, float]]:
    details = root_scores["temperature_2"]["token_details"]
    rows = {route: details[route] for route in ROUTES}
    divergent = [
        index
        for index in range(min(len(row) for row in rows.values()))
        if len({row[index]["token_id"] for row in rows.values()}) > 1
    ]
    if len(divergent) != 1:
        raise ValueError("root choices must have one divergent token")
    choice_index = divergent[0]
    token_to_route = {
        int(rows[route][choice_index]["token_id"]): route for route in ROUTES
    }
    if len(token_to_route) != len(ROUTES):
        raise ValueError("root choices must use four distinct token ids")
    return token_to_route, constrained_route_distribution(root_scores)


def _quantized(rows: list[tuple[int, float]]) -> Counter[tuple[int, int]]:
    scale = round(1.0 / TRANSPORT_TOLERANCE_NATS)
    return Counter((token_id, round(logprob * scale)) for token_id, logprob in rows)


def verify_interface(
    *,
    traces: list[dict[str, Any]],
    batch: Any,
    token_exports: list[dict[str, Any]],
    root_scores: dict[str, Any],
    expected_context_traces: int,
) -> dict[str, Any]:
    """Verify sampling, transport, and trainer consumption of route likelihoods."""
    token_to_route, reference = _choice_metadata(root_scores)
    contexts = [
        trace
        for trace in traces
        if trace.get("agent", {}).get("name") == "context"
    ]
    trace_rows: list[tuple[int, float]] = []
    observations: list[dict[str, Any]] = []
    for trace in contexts:
        choices = (
            trace.get("agent", {})
            .get("sampling", {})
            .get("extra_body", {})
            .get("structured_outputs", {})
            .get("choice")
        )
        sampled = [
            node for node in trace.get("nodes", []) if node.get("sampled") is True
        ]
        if len(sampled) != 1:
            raise ValueError("each context trace must have one sampled node")
        node = sampled[0]
        content_token_ids = [
            int(token_id)
            for token_id, is_content in zip(
                node.get("token_ids", []),
                node.get("is_content", []),
                strict=True,
            )
            if is_content
        ]
        content = [
            (token_id, float(logprob))
            for token_id, logprob in zip(
                content_token_ids,
                node.get("logprobs", []),
                strict=True,
            )
            if token_id in token_to_route
        ]
        if len(content) != 1:
            raise ValueError("each context trace must have one route-choice token")
        token_id, logprob = content[0]
        route = token_to_route[token_id]
        trace_rows.append(content[0])
        observations.append(
            {
                "route": route,
                "token_id": token_id,
                "behavior_logprob": logprob,
                "static_reference_logprob": math.log(reference[route]),
                "static_reference_error_nats": abs(
                    logprob - math.log(reference[route])
                ),
                "choices_exact": choices == list(ROUTE_CHOICES),
            }
        )

    packed_rows: list[tuple[int, float]] = []
    for example in batch.examples:
        packed_rows.extend(
            (int(token_id), float(logprob))
            for token_id, trainable, logprob in zip(
                example.token_ids,
                example.mask,
                example.logprobs,
                strict=True,
            )
            if trainable and int(token_id) in token_to_route
        )

    export_rows: list[dict[str, float | int]] = []
    for record in token_exports:
        for token_id, trainable, inference, trainer, log_ratio in zip(
            record["token_ids"],
            record["loss_mask"],
            record["inference_logprobs"],
            record["trainer_logprobs"],
            record["log_importance_ratio"],
            strict=True,
        ):
            if not trainable or int(token_id) not in token_to_route:
                continue
            export_rows.append(
                {
                    "token_id": int(token_id),
                    "inference_logprob": float(inference),
                    "trainer_logprob": float(trainer),
                    "log_importance_ratio": float(log_ratio),
                }
            )

    exported_inference = [
        (int(row["token_id"]), float(row["inference_logprob"]))
        for row in export_rows
    ]
    importance_consistent = all(
        abs(
            float(row["log_importance_ratio"])
            - (
                float(row["trainer_logprob"])
                - float(row["inference_logprob"])
            )
        )
        <= TRANSPORT_TOLERANCE_NATS
        for row in export_rows
    )
    checks = {
        "exact_context_trace_count": (
            len(contexts) == expected_context_traces
        ),
        "every_request_uses_exact_four_choice_constraint": bool(observations)
        and all(row["choices_exact"] for row in observations),
        "trace_behavior_logprobs_are_finite": bool(trace_rows)
        and all(math.isfinite(logprob) for _, logprob in trace_rows),
        "packed_route_count_matches_traces": (
            len(packed_rows) == len(trace_rows)
        ),
        "packed_behavior_logprobs_match_traces": (
            _quantized(packed_rows) == _quantized(trace_rows)
        ),
        "trainer_export_route_count_matches_traces": (
            len(export_rows) == len(trace_rows)
        ),
        "trainer_export_behavior_logprobs_match_packed": (
            _quantized(exported_inference) == _quantized(packed_rows)
        ),
        "trainer_constrained_logprobs_are_finite_and_bounded": bool(
            export_rows
        )
        and all(
            math.isfinite(float(row["trainer_logprob"]))
            and float(row["trainer_logprob"]) <= TRANSPORT_TOLERANCE_NATS
            for row in export_rows
        ),
        "importance_ratios_use_constrained_trainer_and_behavior_logprobs": (
            bool(export_rows) and importance_consistent
        ),
    }
    return {
        "schema_version": 1,
        "analysis": "stage-c6-v3-exact-constrained-interface",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "expected_context_traces": expected_context_traces,
        "transport_tolerance_nats": TRANSPORT_TOLERANCE_NATS,
        "static_reference_is_descriptive_not_decision_bearing": True,
        "maximum_static_reference_error_nats": max(
            (
                float(row["static_reference_error_nats"])
                for row in observations
            ),
            default=None,
        ),
        "trace_observations": observations,
        "trainer_route_exports": export_rows,
    }
