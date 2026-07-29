"""Exact finite-choice route semantics for the bounded Stage-C5 successor."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from redco.analysis.stage_c3_power import ROUTES
from redco.analysis.stage_c4_warmstart import evaluate_candidate
from redco.integrations.signed_subprocess import sign_payload

ROUTE_CHOICES = tuple(f"<route>{route}</route>" for route in ROUTES)
SEMANTICS = "vllm-choice-trie-single-divergence-v1"


def constrained_route_distribution(root_scores: dict[str, Any]) -> dict[str, float]:
    """Recover the four-way guided-choice distribution at its divergent token."""
    temperature = root_scores.get("temperature_2")
    if not isinstance(temperature, dict):
        raise ValueError("root scores are missing temperature_2")
    details = temperature.get("token_details")
    if not isinstance(details, dict) or set(details) != set(ROUTES):
        raise ValueError("root token details must cover every route")

    rows = {route: details[route] for route in ROUTES}
    if any(not isinstance(row, list) or not row for row in rows.values()):
        raise ValueError("every route must have non-empty token details")
    lengths = {len(row) for row in rows.values()}
    if len(lengths) != 1:
        raise ValueError("canonical route tokenizations must have equal lengths")

    divergent = [
        index
        for index in range(next(iter(lengths)))
        if len({rows[route][index]["token_id"] for route in ROUTES}) > 1
    ]
    if len(divergent) != 1:
        raise ValueError("route choices must have exactly one divergent token position")
    choice_index = divergent[0]
    if len({rows[route][choice_index]["token_id"] for route in ROUTES}) != len(ROUTES):
        raise ValueError("route labels must use four distinct token ids")

    weights: dict[str, float] = {}
    for route in ROUTES:
        value = rows[route][choice_index].get("temperature_2_logprob")
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError("route choice logprob must be finite")
        weights[route] = math.exp(float(value))
    normalizer = math.fsum(weights.values())
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("route choice normalizer must be positive and finite")
    return {route: weights[route] / normalizer for route in ROUTES}


def constrained_root_scores(root_scores: dict[str, Any]) -> dict[str, Any]:
    """Return an unsigned analysis copy carrying constrained route masses."""
    constrained = copy.deepcopy(root_scores)
    probabilities = constrained_route_distribution(root_scores)
    temperature = constrained["temperature_2"]
    temperature["route_sequence_probabilities"] = probabilities
    temperature["route_sequence_logprobabilities"] = {
        route: math.log(probability)
        for route, probability in probabilities.items()
    }
    constrained.pop("signed_payload_sha256", None)
    constrained["analysis"] = "stage-c5-constrained-root-route-scores"
    constrained["constraint_semantics"] = SEMANTICS
    return constrained


def evaluate_constrained_candidate(
    *,
    step: int,
    action_scores: dict[str, Any],
    root_scores: dict[str, Any],
    dataset_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one prior candidate under exact constrained route semantics."""
    probabilities = constrained_route_distribution(root_scores)
    candidate = evaluate_candidate(
        step=step,
        action_scores=action_scores,
        root_scores=constrained_root_scores(root_scores),
        dataset_manifest=dataset_manifest,
    )
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-c5-constrained-candidate-rescore",
            "status": candidate["status"],
            "step": step,
            "constraint_semantics": SEMANTICS,
            "route_choices": list(ROUTE_CHOICES),
            "constrained_route_probabilities_t2": probabilities,
            "candidate": candidate,
            "sources": {
                "action_scores_signed_payload_sha256": action_scores[
                    "signed_payload_sha256"
                ],
                "root_scores_signed_payload_sha256": root_scores[
                    "signed_payload_sha256"
                ],
                "dataset_manifest_signed_payload_sha256": dataset_manifest[
                    "signed_payload_sha256"
                ],
            },
        }
    )


def select_constrained_candidates(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the earliest constrained candidate under the unchanged envelope."""
    if not reports:
        raise ValueError("no candidate reports supplied")
    ordered = sorted(reports, key=lambda report: int(report["step"]))
    if len({int(report["step"]) for report in ordered}) != len(ordered):
        raise ValueError("candidate steps must be unique")
    if any(report.get("constraint_semantics") != SEMANTICS for report in ordered):
        raise ValueError("candidate constraint semantics do not match")
    passing = [report for report in ordered if report["status"] == "passed"]
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-c5-constrained-warmstart-selection",
            "status": "passed" if passing else "failed",
            "selection_rule": (
                "Earliest optimizer checkpoint satisfying every unchanged v4 "
                "support, power, concentration, and factorization threshold "
                "under the frozen constrained root policy."
            ),
            "constraint_semantics": SEMANTICS,
            "route_choices": list(ROUTE_CHOICES),
            "selected_step": int(passing[0]["step"]) if passing else None,
            "evaluated_steps": [int(report["step"]) for report in ordered],
            "candidate_statuses": {
                str(report["step"]): report["status"] for report in ordered
            },
            "candidate_signed_payloads": {
                str(report["step"]): report["signed_payload_sha256"]
                for report in ordered
            },
        }
    )


def rescore_v4_bundle(bundle_root: Path) -> dict[str, Any]:
    """Rescore every frozen v4 candidate without changing source artifacts."""
    selection_root = bundle_root / "runs/stage-c4/warmstart-selection-v4"
    dataset_manifest = json.loads(
        (selection_root / "dataset-audit.json").read_text(encoding="utf-8")
    )
    candidates = []
    for step in range(2, 33, 2):
        candidate_root = selection_root / "candidates" / f"step_{step}"
        action_scores = json.loads(
            (candidate_root / "action-scores.json").read_text(encoding="utf-8")
        )
        root_scores = json.loads(
            (candidate_root / "root-scores.json").read_text(encoding="utf-8")
        )
        candidates.append(
            evaluate_constrained_candidate(
                step=step,
                action_scores=action_scores,
                root_scores=root_scores,
                dataset_manifest=dataset_manifest,
            )
        )
    passing = [candidate["step"] for candidate in candidates if candidate["status"] == "passed"]
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-c5-v4-constrained-rescore",
            "status": "passed" if passing else "failed",
            "interpretation": (
                "Exploratory initialization calibration only. This does not alter "
                "the terminal v4 decision and is not a scientific-arm result."
            ),
            "constraint_semantics": SEMANTICS,
            "route_choices": list(ROUTE_CHOICES),
            "evaluated_steps": [candidate["step"] for candidate in candidates],
            "passing_steps": passing,
            "earliest_passing_step": passing[0] if passing else None,
            "candidates": candidates,
        }
    )
