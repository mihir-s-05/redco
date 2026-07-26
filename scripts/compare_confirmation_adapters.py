"""Measure adapter deltas for paired stock/ReDCO confirmation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def compare(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, int | float]:
    if reference.keys() != candidate.keys():
        raise ValueError("adapter tensor keys differ")
    differing_elements = 0
    total_elements = 0
    max_abs = 0.0
    squared_l2 = 0.0
    differing_tensors = 0
    for name, reference_tensor in reference.items():
        delta = (candidate[name] - reference_tensor).float()
        count = int(torch.count_nonzero(delta).item())
        differing_elements += count
        total_elements += delta.numel()
        if count:
            differing_tensors += 1
        if delta.numel():
            max_abs = max(max_abs, float(delta.abs().max().item()))
        squared_l2 += float(torch.sum(delta * delta).item())
    return {
        "different_elements": differing_elements,
        "total_elements": total_elements,
        "fraction_different": differing_elements / total_elements,
        "differing_tensors": differing_tensors,
        "max_abs": max_abs,
        "l2": squared_l2**0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("seeds", nargs="+", type=int)
    args = parser.parse_args()
    adapter_relative = Path(
        "run_default/broadcasts/step_1/adapter_model.safetensors"
    )
    comparisons: list[dict[str, Any]] = []
    for seed in args.seeds:
        name = f"pair-s{seed}"
        stock = load_file(
            args.root / name / "stock" / adapter_relative,
            device="cpu",
        )
        redco = load_file(
            args.root / name / "redco" / adapter_relative,
            device="cpu",
        )
        comparisons.append(
            {
                "pair": name,
                "seed": seed,
                "first": "stock",
                "second": "redco",
                **compare(stock, redco),
            }
        )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
