"""Score exact temperature-2 probabilities of forced root route sequences."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams

from redco.analysis.vllm_temperature import retemper_selected_logprobs
from redco.integrations.signed_subprocess import (
    atomic_write_json,
    run_and_hard_exit,
    sign_payload,
)


def _logprob(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"unsupported vLLM logprob value: {type(value)}")


def _shutdown_llm(llm: LLM) -> None:
    """Stop vLLM's background engine before Python extension finalization."""
    engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if callable(shutdown):
        shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case_payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = case_payload["cases"]
    prompts = [
        {
            "prompt_token_ids": [
                *[int(value) for value in case["prefix_token_ids"]],
                *[int(value) for value in case["completion_token_ids"]],
            ]
        }
        for case in cases
    ]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=512,
        max_logprobs=-1,
        trust_remote_code=False,
    )
    params = SamplingParams(
        temperature=1.0,
        max_tokens=1,
        logprobs=0,
        prompt_logprobs=-1,
        detokenize=False,
        seed=9601,
    )
    try:
        outputs = llm.generate(prompts, params, use_tqdm=False)
        route_logprobs: dict[str, float] = {}
        token_details: dict[str, list[dict[str, Any]]] = {}
        for case, output in zip(cases, outputs, strict=True):
            prompt_logprobs = output.prompt_logprobs
            if prompt_logprobs is None:
                raise ValueError("vLLM omitted prompt logprobs")
            prefix_length = len(case["prefix_token_ids"])
            completion = [int(value) for value in case["completion_token_ids"]]
            details = []
            total = 0.0
            for offset, token_id in enumerate(completion):
                values = prompt_logprobs[prefix_length + offset]
                if values is None:
                    raise ValueError("vLLM omitted a completion prompt logprob")
                raw = {int(key): _logprob(value) for key, value in values.items()}
                selected = retemper_selected_logprobs(
                    raw,
                    [token_id],
                    temperature=2.0,
                )[token_id]
                total += selected
                details.append(
                    {
                        "token_id": token_id,
                        "temperature_2_logprob": selected,
                    }
                )
            route = str(case["route"])
            route_logprobs[route] = total
            token_details[route] = details
        probabilities = {route: math.exp(value) for route, value in route_logprobs.items()}
        payload: dict[str, Any] = {
            "schema_version": 1,
            "analysis": "stage-c3-root-route-sequence-scores",
            "temperature_2": {
                "route_sequence_logprobabilities": route_logprobs,
                "route_sequence_probabilities": probabilities,
                "valid_route_sequence_mass": math.fsum(probabilities.values()),
                "token_details": token_details,
            },
            "source": {
                "model": args.model,
                "cases_sha256": case_payload["signed_payload_sha256"],
            },
        }
        atomic_write_json(args.output, sign_payload(payload))
    finally:
        _shutdown_llm(llm)


if __name__ == "__main__":
    run_and_hard_exit(main)
