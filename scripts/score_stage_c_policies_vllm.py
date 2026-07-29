"""Score exact Stage-C prefixes with vLLM and optional LoRA adapters.

Run this in the pinned Prime-RL GPU environment. The script deliberately
accepts already-rendered token IDs so tokenizer and chat-template differences
cannot enter the backend-parity diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from redco.analysis.vllm_temperature import retemper_selected_logprobs


def _logprob(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, (float, int)):
        return float(value)
    raise TypeError(f"unsupported vLLM logprob value: {type(value)}")


def _shutdown_llm(llm: LLM) -> None:
    """Stop vLLM's background engine before Python extension finalization."""
    engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if callable(shutdown):
        shutdown()


def _score(
    llm: LLM,
    cases: list[dict[str, Any]],
    *,
    lora_request: LoRARequest | None,
) -> dict[str, list[dict[str, Any]]]:
    prompts = [
        {"prompt_token_ids": [int(token) for token in case["prefix_token_ids"]]} for case in cases
    ]
    params = SamplingParams(
        temperature=1.0,
        max_tokens=1,
        logprobs=-1,
        seed=7202803,
        detokenize=False,
    )
    outputs = llm.generate(
        prompts,
        params,
        lora_request=lora_request,
        use_tqdm=False,
    )
    rows: dict[str, list[dict[str, Any]]] = {"1.0": [], "2.0": []}
    for case, output in zip(cases, outputs, strict=True):
        completion = output.outputs[0]
        if completion.logprobs is None or len(completion.logprobs) != 1:
            raise ValueError("vLLM did not return one generated-token logprob map")
        logprobs = completion.logprobs[0]
        action_ids = {
            action: int(token_id) for action, token_id in case["action_token_ids"].items()
        }
        missing = [token_id for token_id in action_ids.values() if token_id not in logprobs]
        if missing:
            raise ValueError(f"vLLM omitted action token logprobs: {missing}")
        raw_logprobs = {int(token_id): _logprob(value) for token_id, value in logprobs.items()}
        greedy_token = max(raw_logprobs, key=raw_logprobs.__getitem__)
        greedy_allowed = next(
            (action for action, token_id in action_ids.items() if token_id == greedy_token),
            None,
        )
        for temperature in (1.0, 2.0):
            selected = retemper_selected_logprobs(
                raw_logprobs,
                list(action_ids.values()),
                temperature=temperature,
            )
            action_logprobs = {
                action: selected[token_id] for action, token_id in action_ids.items()
            }
            rows[str(temperature)].append(
                {
                    "case_id": case["case_id"],
                    "probe_name": case["probe_name"],
                    "context_route": case["context_route"],
                    "temperature": temperature,
                    "greedy_token_id": int(greedy_token),
                    "greedy_allowed_action": greedy_allowed,
                    "action_logprobabilities": action_logprobs,
                    "action_probabilities": {
                        action: math.exp(logprob) for action, logprob in action_logprobs.items()
                    },
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default="base")
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    case_payload = json.loads(args.cases.read_text())
    cases = case_payload["cases"]
    adapters: list[tuple[str, Path]] = []
    for value in args.adapter:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name == args.model_name:
            raise ValueError("--adapter must be a unique NAME=PATH")
        adapters.append((name, Path(path)))

    llm = LLM(
        model=args.model,
        enable_lora=bool(adapters),
        max_lora_rank=32,
        max_logprobs=-1,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=512,
        trust_remote_code=False,
    )
    try:
        models = []
        for index, (name, path) in enumerate(
            [(args.model_name, None), *adapters],
            start=0,
        ):
            request = (
                None
                if path is None
                else LoRARequest(
                    lora_name=name,
                    lora_int_id=index,
                    lora_path=str(path),
                )
            )
            models.append(
                {
                    "name": name,
                    "temperatures": _score(
                        llm,
                        cases,
                        lora_request=request,
                    ),
                }
            )
        payload = {
            "schema_version": 1,
            "backend": "vllm",
            "temperature_semantics": (
                "vLLM raw full-vocabulary logprobs retempered exactly as "
                "softmax(logits / temperature)"
            ),
            "source": {
                "model": args.model,
                "cases_sha256": case_payload["signed_payload_sha256"],
                "adapters": [
                    {"name": name, "path": str(path.as_posix())} for name, path in adapters
                ],
            },
            "models": models,
        }
        signed = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    finally:
        _shutdown_llm(llm)


if __name__ == "__main__":
    main()
