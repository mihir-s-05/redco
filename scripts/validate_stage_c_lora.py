"""Validate the dependency-free Stage-C LoRA hook and merge algebra."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from stage_c_lora import _module_name, adapter_hooks, merge_adapter


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(3, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs)


def main() -> None:
    assert _module_name("model.layer.lora_A.weight") == "model.layer"
    assert (
        _module_name("base_model.model.model.layer.lora_A.weight")
        == "model.layer"
    )
    base_weight = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    a = torch.tensor([[0.5, -1.0, 2.0]])
    b = torch.tensor([[1.5], [-0.25]])
    scale = 2.0
    inputs = torch.tensor([[2.0, -1.0, 0.5]])

    with tempfile.TemporaryDirectory() as temporary:
        adapter = Path(temporary)
        (adapter / "adapter_config.json").write_text(
            json.dumps({"r": 1, "lora_alpha": scale})
        )
        save_file(
            {
                "base_model.model.layer.lora_A.weight": a,
                "base_model.model.layer.lora_B.weight": b,
            },
            adapter / "adapter_model.safetensors",
        )
        hooked = ToyModel()
        hooked.layer.weight.data.copy_(base_weight)
        expected = torch.nn.functional.linear(inputs, base_weight) + (
            torch.nn.functional.linear(torch.nn.functional.linear(inputs, a), b)
            * scale
        )
        with adapter_hooks(hooked, adapter):
            hooked_output = hooked(inputs)

        merged = ToyModel()
        merged.layer.weight.data.copy_(base_weight)
        merge_adapter(merged, adapter)
        merged_output = merged(inputs)

    torch.testing.assert_close(hooked_output, expected, rtol=0, atol=0)
    torch.testing.assert_close(merged_output, expected, rtol=0, atol=0)
    print(
        json.dumps(
            {
                "status": "pass",
                "hook_matches_analytic": True,
                "merge_matches_analytic": True,
                "maximum_absolute_difference": 0.0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
