from __future__ import annotations

from pathlib import Path

import pytest

from redco.analysis.stage_d_checkpoint_evidence import adopt_prime_adapter_checkpoint


def _source(root: Path) -> Path:
    step = root / "weights" / "step_1"
    adapter = step / "lora_adapters"
    adapter.mkdir(parents=True)
    (step / "STABLE").write_bytes(b"")
    (adapter / "adapter_config.json").write_bytes(b"{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    return step


def test_prime_checkpoint_adoption_is_exact_atomic_and_restartable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path / "source")
    destination = tmp_path / "retained" / "stock"
    pending = destination.with_name(".stock.pending")
    pending.mkdir(parents=True)
    (pending / "STABLE").write_bytes(b"")
    post_model = "9" * 64
    monkeypatch.setattr(
        "redco.analysis.stage_d_checkpoint_evidence.adapter_file_state_sha256",
        lambda *_args, **_kwargs: post_model,
    )
    manifest = adopt_prime_adapter_checkpoint(
        source_step_root=source,
        destination=destination,
        arm="stock",
        trainer_step=1,
        base_model_manifest_sha256="e" * 64,
        observed_post_model_sha256=post_model,
    )
    assert not pending.exists()
    assert {item.name for item in destination.iterdir()} == {
        "STABLE",
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    assert adopt_prime_adapter_checkpoint(
        source_step_root=source,
        destination=destination,
        arm="stock",
        trainer_step=1,
        base_model_manifest_sha256="e" * 64,
        observed_post_model_sha256=post_model,
    ) == manifest


def test_prime_checkpoint_adoption_rejects_unexpected_source_bytes(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source")
    (source / "optimizer.pt").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="unexpected source roster"):
        adopt_prime_adapter_checkpoint(
            source_step_root=source,
            destination=tmp_path / "retained" / "stock",
            arm="stock",
            trainer_step=1,
            base_model_manifest_sha256="e" * 64,
            observed_post_model_sha256="9" * 64,
        )
