from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from redco.analysis.stage_d_bridge_golden import (
    build_synthetic_golden,
    encode_synthetic_action,
    render_synthetic_prompt,
    shared_token_credit,
)
from redco.analysis.stage_d_e2_live import base_snapshot_manifest
from redco.analysis.stage_d_live_update import (
    LiveAuthorizationToken,
    LiveUpdateBinding,
    TrainerPoststep,
    TrainerPrestate,
    authorize_live_update,
    complete_live_update,
    start_live_update_gate,
)
from redco.analysis.stage_d_scientific_branch_group import BranchGroupArtifact
from redco.analysis.stage_d_training_bridge import (
    ArtifactVerificationContext,
    TrainingBridgeBinding,
    compile_training_batch,
)
from redco.analysis.stage_d_update_ledger import SingleUseUpdateLedger

_MASTER_SEED = "stage-d-e2-synthetic-golden"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _binding() -> LiveUpdateBinding:
    return LiveUpdateBinding(
        producer_seal_sha256="1" * 64,
        training_batch_identity="2" * 64,
        bridge_payload_sha256="3" * 64,
        prime_payload_sha256="4" * 64,
        prime_runtime_sha256="5" * 64,
        trainer_config_sha256="6" * 64,
        base_snapshot_manifest_sha256="7" * 64,
        authorization_timeout_seconds=60,
    )


def _prestate(binding: LiveUpdateBinding) -> TrainerPrestate:
    return TrainerPrestate(
        _sha256(binding.to_bytes()),
        "8" * 64,
        "9" * 64,
        "a" * 64,
        ("model.layer.lora_A.0", "model.layer.lora_B.0"),
        128,
    )


def test_synthetic_golden_is_deterministic_distinct_and_shared_nonzero() -> None:
    first = build_synthetic_golden()
    second = build_synthetic_golden()
    assert first.artifact_bytes == second.artifact_bytes
    artifacts = tuple(
        BranchGroupArtifact.verify_bytes(
            value,
            verifier=first.store,
            encode_action=encode_synthetic_action,
            render_prompt=render_synthetic_prompt,
            master_seed=_MASTER_SEED,
        )
        for value in first.artifact_bytes
    )
    nonflat = artifacts[0]
    assert len({arm.action.action_token_ids for arm in nonflat.arms}) == 4
    assert sum(arm.advantage for arm in nonflat.arms) == pytest.approx(0.0)
    credit = shared_token_credit(artifacts)
    assert any(abs(value) > 0 for value in credit.values())
    assert credit[20] != credit[21]


def test_synthetic_golden_compiles_through_the_training_bridge() -> None:
    golden = build_synthetic_golden()
    batch = compile_training_batch(
        golden.artifact_bytes,
        verification_context=ArtifactVerificationContext(
            golden.store,
            encode_synthetic_action,
            render_synthetic_prompt,
            _MASTER_SEED,
        ),
        binding=TrainingBridgeBinding(
            "1" * 64,
            "2" * 64,
            "3" * 64,
            "4" * 64,
            golden.policy_sha256,
        ),
        trainer_step=1,
        seq_len=8,
    )
    assert len(batch.records) == 8
    assert len({record.token_ids for record in batch.records[:4]}) == 4
    assert any(any(value != 0.0 for value in record.advantages) for record in batch.records)


def test_local_controller_authorizes_and_completes_once(tmp_path: Path) -> None:
    binding = _binding()
    prestate = _prestate(binding)
    ledger_root = tmp_path / "ledger"
    authorization_bytes = authorize_live_update(
        binding_bytes=binding.to_bytes(),
        prestate_bytes=prestate.to_bytes(),
        ledger_root=ledger_root,
        consumer_id="prime-trainer-one-step",
    )
    assert SingleUseUpdateLedger.inspect_status(ledger_root) == "authorized-incomplete"
    token = LiveAuthorizationToken.verify_bytes(authorization_bytes)
    poststep = TrainerPoststep(
        prestate.binding_sha256,
        _sha256(prestate.to_bytes()),
        token.authorization_sha256,
        "b" * 64,
        "c" * 64,
        1,
        0.25,
    )
    completion = complete_live_update(
        binding_bytes=binding.to_bytes(),
        prestate_bytes=prestate.to_bytes(),
        authorization_bytes=authorization_bytes,
        poststep_bytes=poststep.to_bytes(),
        ledger_root=ledger_root,
    )
    assert completion.post_model_sha256 == "b" * 64
    assert SingleUseUpdateLedger.inspect_status(ledger_root) == "complete"


def test_local_controller_rejects_stale_or_forged_receipts(tmp_path: Path) -> None:
    binding = _binding()
    prestate = _prestate(binding)
    ledger_root = tmp_path / "ledger"
    authorization_bytes = authorize_live_update(
        binding_bytes=binding.to_bytes(),
        prestate_bytes=prestate.to_bytes(),
        ledger_root=ledger_root,
        consumer_id="prime-trainer-one-step",
    )
    token_payload = json.loads(authorization_bytes)
    token_payload["nonce"] = "d" * 64
    forged_token = json.dumps(token_payload, sort_keys=True, separators=(",", ":")).encode()
    poststep = TrainerPoststep(
        prestate.binding_sha256,
        _sha256(prestate.to_bytes()),
        token_payload["authorization_sha256"],
        "b" * 64,
        "c" * 64,
        1,
        0.25,
    )
    with pytest.raises(ValueError, match="authorization token differs"):
        complete_live_update(
            binding_bytes=binding.to_bytes(),
            prestate_bytes=prestate.to_bytes(),
            authorization_bytes=forged_token,
            poststep_bytes=poststep.to_bytes(),
            ledger_root=ledger_root,
        )
    assert SingleUseUpdateLedger.inspect_status(ledger_root) == "authorized-incomplete"


def test_gate_is_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDCO_LIVE_UPDATE_BINDING", raising=False)
    monkeypatch.delenv("REDCO_LIVE_UPDATE_RECEIPTS", raising=False)
    assert start_live_update_gate(None, None, world_size=8, rank=7, max_steps=100) is None


def test_base_snapshot_manifest_hashes_files_and_excludes_local_cache(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "model.safetensors").write_bytes(b"frozen-model")
    (model_root / "config.json").write_bytes(b"{}")
    (model_root / ".cache").mkdir()
    (model_root / ".cache" / "ignored").write_bytes(b"not-model-state")
    payload = json.loads(
        base_snapshot_manifest(model_root, revision="a" * 40)
    )
    assert [item["path"] for item in payload["files"]] == [
        "config.json",
        "model.safetensors",
    ]
    assert payload["files"][1]["sha256"] == _sha256(b"frozen-model")


def test_frozen_e2_configs_have_one_step_and_adapter_only_contract() -> None:
    trainer = tomllib.loads(
        Path("configs/stage-d/stage-d-e2-trainer-v1.toml").read_text()
    )
    control = tomllib.loads(
        Path("configs/stage-d/stage-d-e2-control-v1.toml").read_text()
    )
    assert trainer["max_steps"] == control["max_steps"] == 1
    assert trainer["max_concurrent_runs"] == 1
    assert trainer["model"]["optim_cpu_offload"] is False
    assert trainer["loss"]["kwargs"]["kl_tau"] == 0.0
    assert trainer["ckpt"]["weights_only"] is True
    assert trainer["ckpt"]["weights"]["adapter_only"] is True


def test_trainer_source_has_only_the_two_narrow_default_off_hooks() -> None:
    source = Path("external/prime-rl/src/prime_rl/trainer/rl/train.py").read_text()
    assert source.count("start_live_update_gate") == 2
    assert source.count("live_update_gate.record_optimizer_step") == 1
    assert 'os.environ.get("REDCO_LIVE_UPDATE_BINDING")' in source
    assert source.index("grad_norm = clip_grad_norm_") < source.index(
        "live_update_gate = start_live_update_gate"
    )
    assert source.index("live_update_gate = start_live_update_gate") < source.index(
        "optimizer.step()"
    )
