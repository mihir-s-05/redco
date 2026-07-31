"""Audit the merged-vLLM operational health check for Stage D v4.6."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)

EXPECTED_ROOT_CASES_SIGNATURE = (
    "0274d78b630201b5363fcc1a6348eef3aa1a52c33153e0a25627d4cfb2dfcff9"
)
EXPECTED_ROUTES = {"alpha", "beta", "gamma", "delta"}


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _root_view(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("source")
    scores = payload.get("temperature_2")
    if not isinstance(source, dict) or not isinstance(scores, dict):
        raise ValueError("root payload is missing source or temperature_2")
    if source.get("cases_sha256") != EXPECTED_ROOT_CASES_SIGNATURE:
        raise ValueError("root payload uses the wrong frozen cases")
    logprobs = scores.get("route_sequence_logprobabilities")
    probabilities = scores.get("route_sequence_probabilities")
    token_details = scores.get("token_details")
    mass = scores.get("valid_route_sequence_mass")
    if not all(
        isinstance(value, dict)
        for value in (logprobs, probabilities, token_details)
    ):
        raise ValueError("root payload score mappings are missing")
    route_sets = (
        set(logprobs),
        set(probabilities),
        set(token_details),
    )
    if any(routes != EXPECTED_ROUTES for routes in route_sets):
        raise ValueError("root payload route keys differ")
    if not isinstance(mass, int | float) or not math.isfinite(mass) or mass <= 0:
        raise ValueError("root payload route mass is not positive and finite")
    if not all(
        isinstance(value, int | float) and math.isfinite(value)
        for mapping in (logprobs, probabilities)
        for value in mapping.values()
    ):
        raise ValueError("root payload scores are not finite")
    if not all(float(value) > 0 for value in probabilities.values()):
        raise ValueError("root payload route probabilities are not positive")
    if not math.isclose(
        float(mass),
        math.fsum(float(value) for value in probabilities.values()),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("root payload valid-route mass is inconsistent")
    tokens: dict[str, list[int]] = {}
    for route, rows in token_details.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError("root payload token details are empty")
        route_tokens = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "token_id",
                "temperature_2_logprob",
            }:
                raise ValueError("root token detail schema differs")
            token_id = row["token_id"]
            logprob = row["temperature_2_logprob"]
            if not isinstance(token_id, int) or not isinstance(
                logprob, int | float
            ) or not math.isfinite(logprob):
                raise ValueError("root token detail value differs")
            route_tokens.append(token_id)
        tokens[route] = route_tokens
    model = source.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("root payload model source is missing")
    return {
        "model": model,
        "cases_sha256": source["cases_sha256"],
        "routes": sorted(EXPECTED_ROUTES),
        "token_ids": tokens,
    }


def audit(
    canonical_action_path: Path,
    runtime_action_path: Path,
    canonical_root_path: Path,
    runtime_root_path: Path,
    expected_runtime_model: str,
) -> dict[str, Any]:
    canonical = json.loads(
        canonical_action_path.read_text(encoding="utf-8")
    )
    runtime = json.loads(runtime_action_path.read_text(encoding="utf-8"))
    canonical_root = json.loads(
        canonical_root_path.read_text(encoding="utf-8")
    )
    root = json.loads(runtime_root_path.read_text(encoding="utf-8"))
    for payload in (canonical, runtime, canonical_root, root):
        verify_signed_payload(payload)
    canonical_root_view = _root_view(canonical_root)
    runtime_root_view = _root_view(root)
    canonical_models = {
        str(model["name"]): model for model in canonical["models"]
    }
    runtime_models = {
        str(model["name"]): model for model in runtime["models"]
    }
    canonical_rows = canonical_models["retained"]["temperatures"]["1.0"]
    runtime_rows = runtime_models["selected"]["temperatures"]["1.0"]
    canonical_by_case = {row["case_id"]: row for row in canonical_rows}
    runtime_by_case = {row["case_id"]: row for row in runtime_rows}
    greedy_agreement = (
        set(canonical_by_case) == set(runtime_by_case)
        and all(
            int(canonical_by_case[case]["greedy_token_id"])
            == int(runtime_by_case[case]["greedy_token_id"])
            for case in canonical_by_case
        )
    )
    checks = {
        "all_payloads_signed": True,
        "all_runtime_values_finite": _finite(runtime) and _finite(root),
        "canonical_runtime_case_sets_exact": (
            set(canonical_by_case) == set(runtime_by_case)
        ),
        "canonical_runtime_greedy_tokens_agree": greedy_agreement,
        "canonical_runtime_root_cases_exact": (
            canonical_root_view["cases_sha256"]
            == runtime_root_view["cases_sha256"]
            == EXPECTED_ROOT_CASES_SIGNATURE
        ),
        "canonical_runtime_root_routes_exact": (
            canonical_root_view["routes"] == runtime_root_view["routes"]
        ),
        "canonical_runtime_root_token_ids_exact": (
            canonical_root_view["token_ids"]
            == runtime_root_view["token_ids"]
        ),
        "runtime_root_model_source_exact": (
            runtime_root_view["model"] == expected_runtime_model
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d0-v4-6-merged-vllm-health",
            "checks": checks,
            "canonical_greedy": {
                case: int(row["greedy_token_id"])
                for case, row in canonical_by_case.items()
            },
            "runtime_greedy": {
                case: int(row["greedy_token_id"])
                for case, row in runtime_by_case.items()
            },
            "canonical_root": canonical_root_view,
            "runtime_root": runtime_root_view,
            "expected_runtime_model": expected_runtime_model,
            "passes": all(checks.values()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-action", type=Path, required=True)
    parser.add_argument("--runtime-action", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--expected-runtime-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.canonical_action,
        args.runtime_action,
        args.canonical_root,
        args.runtime_root,
        args.expected_runtime_model,
    )
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
