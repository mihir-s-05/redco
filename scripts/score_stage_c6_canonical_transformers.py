"""Canonically score Stage-C6 prefixes with deterministic eager Transformers.

This scorer deliberately avoids vLLM, sampling, CUDA graphs, Triton sampling
kernels, batching, and runtime LoRA. It loads one already-merged float32 model,
uses eager attention and deterministic CUDA algorithms, performs one batch-one
forward pass per prefix, and applies a Python ``math.fsum`` full-vocabulary
normalizer to float32 logits copied to CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from stage_c_lora import adapter_hooks
from transformers import AutoModelForCausalLM

from redco.analysis.canonical_logits import selected_logprobs
from redco.integrations.signed_subprocess import atomic_write_json, sign_payload

SETTINGS = {
    "backend": "transformers-eager-cuda",
    "model_dtype": "float32",
    "logit_export_dtype": "float32",
    "normalizer": "python-math-fsum-float64",
    "batch_size": 1,
    "attention": "eager",
    "tf32": False,
    "cublas_workspace_config": ":4096:8",
    "torch_deterministic_algorithms": True,
}


def _configure() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _next_logits(model: Any, token_ids: list[int], device: str) -> list[float]:
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
    return logits.detach().to(device="cpu", dtype=torch.float32).tolist()


def _sequence_logits(
    model: Any,
    prefix: list[int],
    completion: list[int],
    device: str,
) -> list[list[float]]:
    tokens = [*prefix, *completion]
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0]
    return [
        logits[len(prefix) + offset - 1]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .tolist()
        for offset in range(len(completion))
    ]


def _action_model(
    model: Any,
    case_payload: dict[str, Any],
    *,
    model_name: str,
    device: str,
) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {"1.0": [], "2.0": []}
    for case in case_payload["cases"]:
        logits = _next_logits(
            model,
            [int(value) for value in case["prefix_token_ids"]],
            device,
        )
        action_ids = {
            action: int(token_id)
            for action, token_id in case["action_token_ids"].items()
        }
        greedy_token = max(range(len(logits)), key=logits.__getitem__)
        greedy_allowed = next(
            (
                action
                for action, token_id in action_ids.items()
                if token_id == greedy_token
            ),
            None,
        )
        for temperature in (1.0, 2.0):
            selected = selected_logprobs(
                logits,
                list(action_ids.values()),
                temperature=temperature,
            )
            action_logprobs = {
                action: selected[token_id]
                for action, token_id in action_ids.items()
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
                        action: math.exp(logprob)
                        for action, logprob in action_logprobs.items()
                    },
                }
            )
    return {"name": model_name, "temperatures": rows}


def _action_payload(
    case_payload: dict[str, Any],
    *,
    model_path: str,
    models: list[dict[str, Any]],
    adapters: list[tuple[str, Path]],
) -> dict[str, Any]:
    return sign_payload(
        {
            "schema_version": 1,
            "backend": SETTINGS["backend"],
            "canonical_settings": SETTINGS,
            "temperature_semantics": (
                "Full-vocabulary float32 logits normalized with deterministic "
                "Python math.fsum after division by temperature."
            ),
            "source": {
                "model": model_path,
                "cases_sha256": case_payload["signed_payload_sha256"],
                "adapters": [
                    {"name": name, "path": path.as_posix()}
                    for name, path in adapters
                ],
            },
            "models": models,
        }
    )


def _root_payload(
    model: Any,
    case_payload: dict[str, Any],
    *,
    model_path: str,
    device: str,
) -> dict[str, Any]:
    route_logprobs: dict[str, float] = {}
    token_details: dict[str, list[dict[str, Any]]] = {}
    for case in case_payload["cases"]:
        completion = [int(value) for value in case["completion_token_ids"]]
        logits_by_position = _sequence_logits(
            model,
            [int(value) for value in case["prefix_token_ids"]],
            completion,
            device,
        )
        details = []
        total = 0.0
        for token_id, logits in zip(
            completion,
            logits_by_position,
            strict=True,
        ):
            value = selected_logprobs(
                logits,
                [token_id],
                temperature=2.0,
            )[token_id]
            total += value
            details.append(
                {
                    "token_id": token_id,
                    "temperature_2_logprob": value,
                }
            )
        route = str(case["route"])
        route_logprobs[route] = total
        token_details[route] = details
    probabilities = {
        route: math.exp(value) for route, value in route_logprobs.items()
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-c3-root-route-sequence-scores",
            "backend": SETTINGS["backend"],
            "canonical_settings": SETTINGS,
            "temperature_2": {
                "route_sequence_logprobabilities": route_logprobs,
                "route_sequence_probabilities": probabilities,
                "valid_route_sequence_mass": math.fsum(probabilities.values()),
                "token_details": token_details,
            },
            "source": {
                "model": model_path,
                "cases_sha256": case_payload["signed_payload_sha256"],
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--action-cases", type=Path, required=True)
    parser.add_argument("--root-cases", type=Path, required=True)
    parser.add_argument("--action-output", type=Path, required=True)
    parser.add_argument("--root-output", type=Path)
    parser.add_argument("--model-name", default="selected_initialization")
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.device != "cuda:0":
        raise ValueError("the frozen canonical scorer requires cuda:0")
    _configure()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    action_cases = json.loads(args.action_cases.read_text(encoding="utf-8"))
    root_cases = json.loads(args.root_cases.read_text(encoding="utf-8"))
    adapters: list[tuple[str, Path]] = []
    for value in args.adapter:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name == args.model_name:
            raise ValueError("--adapter must be a unique NAME=PATH")
        if name in {existing for existing, _ in adapters}:
            raise ValueError(f"duplicate adapter name: {name}")
        adapters.append((name, Path(path)))
    action_models = [
        _action_model(
            model,
            action_cases,
            model_name=args.model_name,
            device=args.device,
        )
    ]
    for name, adapter in adapters:
        with adapter_hooks(model, adapter):
            action_models.append(
                _action_model(
                    model,
                    action_cases,
                    model_name=name,
                    device=args.device,
                )
            )
    action = _action_payload(
        action_cases,
        model_path=args.model,
        models=action_models,
        adapters=adapters,
    )
    root = (
        None
        if args.root_output is None
        else _root_payload(
            model,
            root_cases,
            model_path=args.model,
            device=args.device,
        )
    )
    atomic_write_json(args.action_output, action)
    if args.root_output is not None and root is not None:
        atomic_write_json(args.root_output, root)
    print(
        json.dumps(
            {
                "action_signature": action["signed_payload_sha256"],
                "root_signature": (
                    None if root is None else root["signed_payload_sha256"]
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
