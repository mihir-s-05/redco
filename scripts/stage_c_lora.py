"""Minimal standard-LoRA loading helpers for Stage-C scoring and merging."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def _adapter_state(adapter_dir: Path) -> tuple[dict[str, torch.Tensor], float]:
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank = int(config["r"])
    scale = float(config["lora_alpha"]) / rank
    state = load_file(adapter_dir / "adapter_model.safetensors", device="cpu")
    return state, scale


def _module_name(lora_a_key: str) -> str:
    prefix = "base_model.model."
    suffix = ".lora_A.weight"
    if not lora_a_key.startswith(prefix) or not lora_a_key.endswith(suffix):
        raise ValueError(f"unsupported LoRA tensor name: {lora_a_key}")
    return lora_a_key[len(prefix) : -len(suffix)]


def _pairs(
    adapter_dir: Path,
) -> Iterator[tuple[str, torch.Tensor, torch.Tensor, float]]:
    state, scale = _adapter_state(adapter_dir)
    a_keys = sorted(key for key in state if key.endswith(".lora_A.weight"))
    if not a_keys:
        raise ValueError("adapter has no LoRA-A tensors")
    for a_key in a_keys:
        b_key = a_key.removesuffix(".lora_A.weight") + ".lora_B.weight"
        if b_key not in state:
            raise ValueError(f"adapter is missing {b_key}")
        yield _module_name(a_key), state[a_key], state[b_key], scale


@contextmanager
def adapter_hooks(model: torch.nn.Module, adapter_dir: Path):
    """Apply one adapter without mutating the base weights."""
    modules = dict(model.named_modules())
    handles = []
    tensors: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        for name, a_cpu, b_cpu, scale in _pairs(adapter_dir):
            module = modules.get(name)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"adapter target is not a linear module: {name}")
            device = module.weight.device
            dtype = module.weight.dtype
            a = a_cpu.to(device=device, dtype=dtype)
            b = b_cpu.to(device=device, dtype=dtype)
            tensors.append((a, b))

            def hook(
                _module: torch.nn.Module,
                inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                a: torch.Tensor = a,
                b: torch.Tensor = b,
                scale: float = scale,
            ) -> torch.Tensor:
                return output + F.linear(F.linear(inputs[0], a), b) * scale

            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
        tensors.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def merge_adapter(model: torch.nn.Module, adapter_dir: Path) -> None:
    """Merge one standard LoRA adapter into its base linear weights in place."""
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, a_cpu, b_cpu, scale in _pairs(adapter_dir):
            weight_name = f"{name}.weight"
            if weight_name not in parameters:
                raise ValueError(f"adapter target weight is missing: {weight_name}")
            weight = parameters[weight_name]
            a = a_cpu.to(device=weight.device, dtype=weight.dtype)
            b = b_cpu.to(device=weight.device, dtype=weight.dtype)
            weight.addmm_(b, a, beta=1.0, alpha=scale)
