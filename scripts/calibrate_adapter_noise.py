"""Measure all pairwise tensor deltas among stock LoRA adapters."""

from __future__ import annotations

import argparse
import itertools
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
        local_max = float(delta.abs().max().item()) if delta.numel() else 0.0
        max_abs = max(max_abs, local_max)
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
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    adapter_relative = Path(
        "run_default/broadcasts/step_1/adapter_model.safetensors"
    )
    tensors = {
        name: load_file(args.root / name / adapter_relative, device="cpu")
        for name in args.names
    }
    comparisons: list[dict[str, Any]] = []
    for first, second in itertools.combinations(args.names, 2):
        comparisons.append(
            {
                "first": first,
                "second": second,
                **compare(tensors[first], tensors[second]),
            }
        )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "run_names": args.names,
                "comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
