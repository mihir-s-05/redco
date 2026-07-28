"""Score exact Stage-C action probabilities for a base model and LoRA adapters.

Run this only in the GPU environment with the Prime-RL uv environment active.
The output is intentionally raw; aggregate it locally with
``redco.analysis.stage_c_policy_audit``.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def _distribution(
    model: torch.nn.Module,
    prefix: list[int],
    *,
    temperature: float,
) -> torch.Tensor:
    input_ids = torch.tensor([prefix], device=model.device, dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0, -1].float()
    return torch.softmax(logits / temperature, dim=-1).cpu()


def _kl(candidate: torch.Tensor, base: torch.Tensor) -> float:
    tiny = torch.finfo(candidate.dtype).tiny
    return float(
        torch.sum(
            candidate * (torch.log(candidate.clamp_min(tiny)) - torch.log(base.clamp_min(tiny)))
        )
    )


def _score_model(
    model: torch.nn.Module,
    cases: list[dict[str, Any]],
    *,
    base_distributions: dict[str, torch.Tensor] | None,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    output: list[dict[str, Any]] = []
    distributions: dict[str, torch.Tensor] = {}
    for case in cases:
        case_id = case["case_id"]
        t1 = _distribution(model, case["prefix_token_ids"], temperature=1.0)
        t2 = _distribution(model, case["prefix_token_ids"], temperature=2.0)
        distributions[case_id] = t1
        action_ids = {
            action: int(token_id)
            for action, token_id in case["action_token_ids"].items()
        }
        full_t1 = {action: float(t1[token_id]) for action, token_id in action_ids.items()}
        full_t2 = {action: float(t2[token_id]) for action, token_id in action_ids.items()}
        sum_t1 = sum(full_t1.values())
        sum_t2 = sum(full_t2.values())
        conditional_t1 = {action: value / sum_t1 for action, value in full_t1.items()}
        conditional_t2 = {action: value / sum_t2 for action, value in full_t2.items()}
        greedy_token = int(torch.argmax(t1))
        greedy_allowed = next(
            (action for action, token_id in action_ids.items() if token_id == greedy_token),
            None,
        )
        row: dict[str, Any] = {
            "case_id": case_id,
            "probe_name": case["probe_name"],
            "context_route": case["context_route"],
            "greedy_token_id": greedy_token,
            "greedy_allowed_action": greedy_allowed,
            "full_vocab_action_probabilities_t1": full_t1,
            "full_vocab_action_probabilities_t2": full_t2,
            "conditional_action_probabilities_t1": conditional_t1,
            "conditional_action_probabilities_t2": conditional_t2,
        }
        if base_distributions is not None:
            base = base_distributions[case_id]
            row["full_vocab_kl_from_base_t1"] = _kl(t1, base)
            base_allowed = torch.tensor(
                [float(base[token_id]) for token_id in action_ids.values()]
            )
            base_allowed /= base_allowed.sum()
            candidate_allowed = torch.tensor(list(conditional_t1.values()))
            row["allowed_action_kl_from_base_t1"] = _kl(
                candidate_allowed,
                base_allowed,
            )
        output.append(row)
    return output, distributions


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case_payload = json.loads(args.cases.read_text())
    cases = case_payload["cases"]
    adapters = []
    for value in args.adapter:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name == "base":
            raise ValueError("--adapter must be NAME=PATH and NAME cannot be base")
        adapters.append((name, path))

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    base_rows, base_distributions = _score_model(
        base_model,
        cases,
        base_distributions=None,
    )
    models: list[dict[str, Any]] = [{"name": args.model_name, "cases": base_rows}]
    if adapters:
        first_name, first_path = adapters[0]
        model = PeftModel.from_pretrained(
            base_model,
            first_path,
            adapter_name=first_name,
        )
        for name, path in adapters[1:]:
            model.load_adapter(path, adapter_name=name)
        for name, _ in adapters:
            model.set_adapter(name)
            context = nullcontext()
            with context:
                rows, _ = _score_model(
                    model,
                    cases,
                    base_distributions=base_distributions,
                )
            models.append({"name": name, "cases": rows})
    payload = {
        "schema_version": 1,
        "source": {
            "model": args.model,
            "cases_sha256": case_payload["signed_payload_sha256"],
            "adapters": [{"name": name, "path": path} for name, path in adapters],
        },
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
