"""Canonically score both action and root behavior under one retained adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from score_stage_c6_canonical_transformers import (
    _action_model,
    _action_payload,
    _configure,
    _root_payload,
)
from stage_c_lora import adapter_hooks
from transformers import AutoModelForCausalLM

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-name", default="retained")
    parser.add_argument("--action-cases", type=Path, required=True)
    parser.add_argument("--root-cases", type=Path, required=True)
    parser.add_argument("--action-output", type=Path, required=True)
    parser.add_argument("--root-output", type=Path, required=True)
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

    with adapter_hooks(model, args.adapter):
        action_model = _action_model(
            model,
            action_cases,
            model_name=args.adapter_name,
            device=args.device,
        )
        root = _root_payload(
            model,
            root_cases,
            model_path=args.model,
            device=args.device,
        )

    action = _action_payload(
        action_cases,
        model_path=args.model,
        models=[action_model],
        adapters=[(args.adapter_name, args.adapter)],
    )
    root.pop("signed_payload_sha256")
    root["analysis"] = "stage-d0-v4-6-retained-adapter-root-scores"
    root["source"]["adapter"] = {
        "name": args.adapter_name,
        "path": args.adapter.as_posix(),
    }
    root = sign_payload(root)

    atomic_write_json(args.action_output, action)
    atomic_write_json(args.root_output, root)
    print(
        json.dumps(
            {
                "action_signature": action["signed_payload_sha256"],
                "root_signature": root["signed_payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
