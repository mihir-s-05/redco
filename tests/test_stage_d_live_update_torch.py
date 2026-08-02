from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from redco.analysis.stage_d_live_update import (  # noqa: E402
    LiveAuthorizationToken,
    LiveUpdateBinding,
    LiveUpdateTrainerGate,
    TrainerPoststep,
    _model_sha256,
    adapter_file_state_sha256,
)


class _TinyLoRAModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.lora_A = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        self.layer.lora_B = torch.nn.Parameter(torch.tensor([0.5, 3.0]))


class _RunManager:
    def __init__(self, model: _TinyLoRAModel) -> None:
        self.model = model

    def get_state_dict_for_run(self, index: int) -> dict[str, torch.Tensor]:
        assert index == 0
        return {
            "layer.lora_A": self.model.layer.lora_A,
            "layer.lora_B": self.model.layer.lora_B,
        }


class _FakeDTensor:
    def __init__(self, full: torch.Tensor) -> None:
        self._full = full
        self.shape = (1,)

    def detach(self) -> _FakeDTensor:
        return self

    def full_tensor(self) -> torch.Tensor:
        return self._full

    def to_local(self) -> torch.Tensor:
        raise AssertionError("distributed semantic hashing must not hash a local shard")


def _binding() -> LiveUpdateBinding:
    return LiveUpdateBinding(
        producer_seal_sha256="1" * 64,
        training_batch_identity="2" * 64,
        bridge_payload_sha256="3" * 64,
        prime_payload_sha256="4" * 64,
        prime_runtime_sha256="5" * 64,
        trainer_config_sha256="6" * 64,
        base_snapshot_manifest_sha256="7" * 64,
        authorization_timeout_seconds=30,
    )


def test_real_torch_gate_hashes_one_adamw_step_and_saved_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyLoRAModel()
    manager = _RunManager(model)
    monkeypatch.setitem(
        sys.modules,
        "prime_rl.trainer.runs",
        types.SimpleNamespace(get_multi_run_manager=lambda: manager),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    gate = LiveUpdateTrainerGate(model, optimizer, _binding(), receipts)
    prestate = gate.prestate
    authorization = LiveAuthorizationToken(
        prestate.binding_sha256,
        prestate.nonce,
        "ledger",
        "8" * 64,
        "trainer",
        prestate.pre_model_sha256,
        prestate.pre_optimizer_sha256,
    )
    (receipts / "authorization.json").write_bytes(authorization.to_bytes())
    gate.publish_and_wait()

    loss = model.layer.lora_A.square().sum() + model.layer.lora_B.square().sum()
    loss.backward()
    gradient_l2 = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    gate.record_optimizer_step(optimizer_step=1, gradient_l2=float(gradient_l2))

    poststep = TrainerPoststep.verify_bytes((receipts / "poststep.json").read_bytes())
    assert poststep.post_model_sha256 != prestate.pre_model_sha256
    adapter = tmp_path / "adapter_model.safetensors"
    safetensors_torch.save_file(
        {
            "base_model.model.layer.lora_A": model.layer.lora_A.detach(),
            "base_model.model.layer.lora_B": model.layer.lora_B.detach(),
        },
        adapter,
    )
    assert (
        adapter_file_state_sha256(
            adapter,
            base_snapshot_manifest_sha256="7" * 64,
        )
        == poststep.post_model_sha256
    )


def test_distributed_tensor_hash_uses_reconstructed_full_tensor() -> None:
    full = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = _model_sha256((("base_model.model.layer.lora_A", full),), "7" * 64)
    observed = _model_sha256(
        (("base_model.model.layer.lora_A", _FakeDTensor(full)),),
        "7" * 64,
    )
    assert observed == expected


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_model_state_hash_rejects_nonfinite_tensors(nonfinite: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _model_sha256(
            (("base_model.model.layer.lora_A", torch.tensor([nonfinite])),),
            "7" * 64,
        )
