"""Quantify tensor differences between frozen-rollout LoRA adapter files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def compare(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = load_file(reference_path, device="cpu")
    candidate = load_file(candidate_path, device="cpu")
    if reference.keys() != candidate.keys():
        raise ValueError("adapter tensor keys differ")

    differing_elements = 0
    total_elements = 0
    max_abs = 0.0
    squared_l2 = 0.0
    rows: list[dict[str, int | float | str]] = []
    for name, reference_tensor in reference.items():
        delta = (candidate[name] - reference_tensor).float()
        count = int(torch.count_nonzero(delta).item())
        total_elements += delta.numel()
        differing_elements += count
        local_max = float(delta.abs().max().item()) if delta.numel() else 0.0
        max_abs = max(max_abs, local_max)
        squared_l2 += float(torch.sum(delta * delta).item())
        if count:
            rows.append(
                {
                    "tensor": name,
                    "different_elements": count,
                    "max_abs": local_max,
                }
            )
    return {
        "different_elements": differing_elements,
        "total_elements": total_elements,
        "fraction_different": differing_elements / total_elements,
        "max_abs": max_abs,
        "l2": squared_l2**0.5,
        "differing_tensors": len(rows),
        "largest_tensors": sorted(
            rows,
            key=lambda row: float(row["max_abs"]),
            reverse=True,
        )[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()
    result = {
        candidate.as_posix(): compare(args.reference, candidate)
        for candidate in args.candidates
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
