"""Synchronized CUDA probe for the private grouped-LoRA matrix multiply."""

from __future__ import annotations

import argparse
import json
import os
import platform
from collections.abc import Callable
from pathlib import Path

import torch

BF16_ATOL = 0.03
BF16_RTOL = 0.03


def _grouped(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    a_out = torch._grouped_mm(
        x,
        lora_a.transpose(-2, -1),
        offs=offsets,
    )
    torch.cuda.synchronize()
    output = torch._grouped_mm(
        a_out,
        lora_b.transpose(-2, -1),
        offs=offsets,
    )
    torch.cuda.synchronize()
    return output


def _fallback(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    del offsets
    return (
        x @ lora_a[0].transpose(-2, -1)
    ) @ lora_b[0].transpose(-2, -1)


def _errors(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(1e-2)
    return {
        "maximum_absolute": float(difference.max().item()),
        "maximum_relative": float(relative.max().item()),
    }


def _run_path(
    operation: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
    seeds: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    x, lora_a, lora_b = (
        tensor.detach().clone().requires_grad_(True) for tensor in seeds
    )
    output = operation(x, lora_a, lora_b, offsets)
    loss = output.float().square().mean()
    gradients = torch.autograd.grad(loss, (x, lora_a, lora_b))
    torch.cuda.synchronize()
    return output.detach(), tuple(gradient.detach() for gradient in gradients)


def run_probe(mode: str) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(7308101)
    device = torch.device("cuda")
    shape = {"tokens": 476, "in_features": 2560, "out_features": 4096, "rank": 32}
    seeds = (
        torch.randn(
            shape["tokens"],
            shape["in_features"],
            device=device,
            dtype=torch.bfloat16,
        ),
        torch.randn(
            1,
            shape["rank"],
            shape["in_features"],
            device=device,
            dtype=torch.bfloat16,
        ),
        torch.randn(
            1,
            shape["out_features"],
            shape["rank"],
            device=device,
            dtype=torch.bfloat16,
        ),
    )
    offsets = torch.tensor(
        [shape["tokens"]],
        device=device,
        dtype=torch.int32,
    )
    grouped_operation = _grouped
    if mode == "compiled":
        grouped_operation = torch.compile(_grouped, fullgraph=False)

    grouped_output, grouped_gradients = _run_path(
        grouped_operation,
        seeds,
        offsets,
    )
    fallback_output, fallback_gradients = _run_path(
        _fallback,
        seeds,
        offsets,
    )
    torch.testing.assert_close(
        grouped_output,
        fallback_output,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )
    for grouped_gradient, fallback_gradient in zip(
        grouped_gradients,
        fallback_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            grouped_gradient,
            fallback_gradient,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )

    return {
        "status": "passed",
        "mode": mode,
        "shape": shape,
        "input_layout": [
            {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
            }
            for tensor in seeds
        ],
        "offsets": {
            "values": offsets.tolist(),
            "dtype": str(offsets.dtype),
        },
        "tolerance": {"absolute": BF16_ATOL, "relative": BF16_RTOL},
        "forward_error": _errors(grouped_output, fallback_output),
        "gradient_errors": [
            _errors(grouped, fallback)
            for grouped, fallback in zip(
                grouped_gradients,
                fallback_gradients,
                strict=True,
            )
        ],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "inductor_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eager", "compiled"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
