"""Analyze deployment scorer drift for a byte-identical Stage-C6 model."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from redco.analysis.stage_c5_constrained import (
    constrained_route_distribution,
    evaluate_constrained_candidate,
)
from redco.integrations.signed_subprocess import sign_payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        str(name): str(value["sha256"])
        for name, value in manifest["files"].items()
    }


def _model_rows(payload: dict[str, Any], temperature: str) -> dict[str, Any]:
    models = payload["models"]
    if len(models) != 1:
        raise ValueError("initialization score payload must contain one model")
    return {
        str(row["case_id"]): row
        for row in models[0]["temperatures"][temperature]
    }


def _action_drift(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    maximum_logprob = 0.0
    maximum_probability = 0.0
    for temperature in ("1.0", "2.0"):
        left_rows = _model_rows(left, temperature)
        right_rows = _model_rows(right, temperature)
        if set(left_rows) != set(right_rows):
            raise ValueError("action score cases differ")
        for case_id in left_rows:
            for action in left_rows[case_id]["action_logprobabilities"]:
                maximum_logprob = max(
                    maximum_logprob,
                    abs(
                        float(left_rows[case_id]["action_logprobabilities"][action])
                        - float(right_rows[case_id]["action_logprobabilities"][action])
                    ),
                )
                maximum_probability = max(
                    maximum_probability,
                    abs(
                        float(left_rows[case_id]["action_probabilities"][action])
                        - float(right_rows[case_id]["action_probabilities"][action])
                    ),
                )
    return {
        "maximum_absolute_selected_logprob_drift": maximum_logprob,
        "maximum_absolute_selected_probability_drift": maximum_probability,
    }


def _root_drift(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_details = left["temperature_2"]["token_details"]
    right_details = right["temperature_2"]["token_details"]
    if set(left_details) != set(right_details):
        raise ValueError("root routes differ")
    maximum_token_logprob = 0.0
    for route in left_details:
        if len(left_details[route]) != len(right_details[route]):
            raise ValueError("root tokenization lengths differ")
        for first, second in zip(
            left_details[route],
            right_details[route],
            strict=True,
        ):
            if int(first["token_id"]) != int(second["token_id"]):
                raise ValueError("root token ids differ")
            maximum_token_logprob = max(
                maximum_token_logprob,
                abs(
                    float(first["temperature_2_logprob"])
                    - float(second["temperature_2_logprob"])
                ),
            )
    left_routes = constrained_route_distribution(left)
    right_routes = constrained_route_distribution(right)
    return {
        "maximum_absolute_token_logprob_drift": maximum_token_logprob,
        "maximum_absolute_constrained_route_probability_drift": max(
            abs(left_routes[route] - right_routes[route])
            for route in left_routes
        ),
    }


def _constant_reward_gradient(
    probabilities: dict[str, float],
    reward: float,
) -> tuple[float, ...]:
    actions = tuple(sorted(probabilities))
    return tuple(
        math.fsum(
            probabilities[action]
            * reward
            * (float(action == logit_action) - probabilities[logit_action])
            for action in actions
        )
        for logit_action in actions
    )


def analyze(
    *,
    stage_c5_action: Path,
    stage_c5_root: Path,
    stage_c5_merge_manifest: Path,
    stage_c6_action: Path,
    stage_c6_root: Path,
    stage_c6_merge_manifest: Path,
    dataset_manifest: Path,
) -> dict[str, Any]:
    action_c5 = _load(stage_c5_action)
    root_c5 = _load(stage_c5_root)
    action_c6 = _load(stage_c6_action)
    root_c6 = _load(stage_c6_root)
    merge_c5 = _load(stage_c5_merge_manifest)
    merge_c6 = _load(stage_c6_merge_manifest)
    dataset = _load(dataset_manifest)
    hashes_c5 = _file_hashes(merge_c5)
    hashes_c6 = _file_hashes(merge_c6)
    if set(hashes_c5) != set(hashes_c6):
        raise ValueError("merged model file sets differ")

    candidate_c5 = evaluate_constrained_candidate(
        step=18,
        action_scores=action_c5,
        root_scores=root_c5,
        dataset_manifest=dataset,
    )
    candidate_c6 = evaluate_constrained_candidate(
        step=18,
        action_scores=action_c6,
        root_scores=root_c6,
        dataset_manifest=dataset,
    )
    nuisance_gradients: dict[str, Any] = {}
    for label, candidate in (("stage_c5", candidate_c5), ("stage_c6", candidate_c6)):
        by_route = candidate["candidate"]["model_factorization"][
            "normalized_digit_distribution_by_route"
        ]
        gradients = {
            route: _constant_reward_gradient(
                probabilities,
                float(route == "delta"),
            )
            for route, probabilities in by_route.items()
        }
        nuisance_gradients[label] = {
            "by_route": gradients,
            "maximum_absolute_exact_gradient": max(
                abs(value)
                for gradient in gradients.values()
                for value in gradient
            ),
        }

    tv_c5 = float(
        candidate_c5["candidate"]["model_factorization"][
            "route_digit_joint_total_variation"
        ]
    )
    tv_c6 = float(
        candidate_c6["candidate"]["model_factorization"][
            "route_digit_joint_total_variation"
        ]
    )
    identical_files = {
        name: hashes_c5[name] == hashes_c6[name] for name in hashes_c5
    }
    payload = {
        "schema_version": 1,
        "analysis": "stage-c6-byte-identical-scorer-stability",
        "status": "passed",
        "model_identity": {
            "all_file_hashes_identical": all(identical_files.values()),
            "file_checks": identical_files,
            "stage_c5_adapter_sha256": merge_c5["adapter_model_sha256"],
            "stage_c6_adapter_sha256": merge_c6["adapter_model_sha256"],
        },
        "score_drift": {
            "action": _action_drift(action_c5, action_c6),
            "root": _root_drift(root_c5, root_c6),
            "joint_tv": {
                "stage_c5": tv_c5,
                "stage_c6": tv_c6,
                "absolute_change": abs(tv_c6 - tv_c5),
                "frozen_limit": 0.05,
                "gate_decision_flipped": (
                    (tv_c5 <= 0.05) != (tv_c6 <= 0.05)
                ),
            },
        },
        "scientific_sensitivity": {
            "confusion_irrelevant_exact_target_gradient": nuisance_gradients,
            "interpretation": (
                "For confusion_irrelevant, reward is constant across target "
                "digits after conditioning on the route. Its exact target-logit "
                "gradient is therefore zero under both measured policies. The "
                "scorer drift flips the auxiliary factorization gate but does "
                "not create a causal target gradient."
            ),
        },
        "disposition": (
            "Keep Stage C6 v1 terminal. Replace single-shot vLLM support "
            "measurement with a separately frozen deterministic scorer; do not "
            "relax the 0.05 threshold based on this result."
        ),
        "sources": {
            "stage_c5_action_sha256": _sha256(stage_c5_action),
            "stage_c5_root_sha256": _sha256(stage_c5_root),
            "stage_c5_merge_manifest_sha256": _sha256(stage_c5_merge_manifest),
            "stage_c6_action_sha256": _sha256(stage_c6_action),
            "stage_c6_root_sha256": _sha256(stage_c6_root),
            "stage_c6_merge_manifest_sha256": _sha256(stage_c6_merge_manifest),
            "dataset_manifest_sha256": _sha256(dataset_manifest),
        },
    }
    return sign_payload(payload)
