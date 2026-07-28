"""Compare exact-prefix Hugging Face and vLLM Stage-C policy scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _models(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = payload.get("models")
    if not isinstance(values, list) or not values:
        raise ValueError("score payload has no models")
    result = {}
    for model in values:
        if not isinstance(model, dict) or not isinstance(model.get("name"), str):
            raise ValueError("score payload has an invalid model")
        result[model["name"]] = model
    return result


def _hf_rows(model: dict[str, Any], temperature: str) -> dict[str, dict[str, Any]]:
    cases = model.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Hugging Face model has no cases")
    field = f"full_vocab_action_probabilities_t{temperature[0]}"
    rows = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ValueError("Hugging Face case is invalid")
        probabilities = case.get(field)
        if not isinstance(probabilities, dict):
            raise ValueError(f"Hugging Face case has no {field}")
        rows[case["case_id"]] = {
            "greedy_token_id": case.get("greedy_token_id"),
            "probabilities": probabilities,
        }
    return rows


def _vllm_rows(model: dict[str, Any], temperature: str) -> dict[str, dict[str, Any]]:
    temperatures = model.get("temperatures")
    if not isinstance(temperatures, dict):
        raise ValueError("vLLM model has no temperatures")
    cases = temperatures.get(temperature)
    if not isinstance(cases, list):
        raise ValueError(f"vLLM model has no temperature {temperature}")
    rows = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ValueError("vLLM case is invalid")
        probabilities = case.get("action_probabilities")
        if not isinstance(probabilities, dict):
            raise ValueError("vLLM case has no action probabilities")
        rows[case["case_id"]] = {
            "greedy_token_id": case.get("greedy_token_id"),
            "probabilities": probabilities,
        }
    return rows


def compare_backends(
    hf_payload: dict[str, Any],
    vllm_payload: dict[str, Any],
    *,
    model_names: tuple[str, ...],
    maximum_logprob_difference: float,
) -> dict[str, Any]:
    """Compare absolute and adapter-relative action log probabilities."""
    if maximum_logprob_difference <= 0:
        raise ValueError("maximum_logprob_difference must be positive")
    hf_models = _models(hf_payload)
    vllm_models = _models(vllm_payload)
    if any(name not in hf_models or name not in vllm_models for name in model_names):
        raise ValueError("requested model is missing from a score payload")
    if "base" not in model_names:
        raise ValueError("model_names must include base")

    comparisons = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    greedy_mismatches = 0
    for temperature in ("1.0", "2.0"):
        hf_base = _hf_rows(hf_models["base"], temperature)
        vllm_base = _vllm_rows(vllm_models["base"], temperature)
        for name in model_names:
            hf_rows = _hf_rows(hf_models[name], temperature)
            vllm_rows = _vllm_rows(vllm_models[name], temperature)
            if hf_rows.keys() != vllm_rows.keys() or hf_rows.keys() != hf_base.keys():
                raise ValueError("backend case IDs differ")
            for case_id in hf_rows:
                hf_probs = hf_rows[case_id]["probabilities"]
                vllm_probs = vllm_rows[case_id]["probabilities"]
                if hf_probs.keys() != vllm_probs.keys():
                    raise ValueError("backend action IDs differ")
                greedy_equal = (
                    hf_rows[case_id]["greedy_token_id"]
                    == vllm_rows[case_id]["greedy_token_id"]
                )
                greedy_mismatches += int(not greedy_equal)
                for action in hf_probs:
                    hf_logprob = math.log(float(hf_probs[action]))
                    vllm_logprob = math.log(float(vllm_probs[action]))
                    absolute = abs(hf_logprob - vllm_logprob)
                    hf_relative = hf_logprob - math.log(
                        float(hf_base[case_id]["probabilities"][action])
                    )
                    vllm_relative = vllm_logprob - math.log(
                        float(vllm_base[case_id]["probabilities"][action])
                    )
                    relative = abs(hf_relative - vllm_relative)
                    maximum_absolute = max(maximum_absolute, absolute)
                    maximum_relative = max(maximum_relative, relative)
                    comparisons.append(
                        {
                            "model": name,
                            "temperature": temperature,
                            "case_id": case_id,
                            "action": action,
                            "absolute_logprob_difference": absolute,
                            "adapter_relative_logprob_difference": relative,
                        }
                    )
    status = (
        "pass"
        if maximum_absolute <= maximum_logprob_difference
        and maximum_relative <= maximum_logprob_difference
        and greedy_mismatches == 0
        else "fail"
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c-backend-parity",
        "status": status,
        "thresholds": {
            "maximum_logprob_difference": maximum_logprob_difference,
            "greedy_token_mismatches": 0,
        },
        "observed": {
            "maximum_absolute_logprob_difference": maximum_absolute,
            "maximum_adapter_relative_logprob_difference": maximum_relative,
            "greedy_token_mismatches": greedy_mismatches,
            "comparison_count": len(comparisons),
        },
        "largest_differences": sorted(
            comparisons,
            key=lambda item: max(
                item["absolute_logprob_difference"],
                item["adapter_relative_logprob_difference"],
            ),
            reverse=True,
        )[:20],
    }
    signed = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--maximum-logprob-difference", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_backends(
        json.loads(args.hf.read_text()),
        json.loads(args.vllm.read_text()),
        model_names=tuple(args.model),
        maximum_logprob_difference=args.maximum_logprob_difference,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
