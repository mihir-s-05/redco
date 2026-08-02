from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import redco.analysis.stage_d_prime_trainer as trainer_entrypoint
import redco.analysis.stage_d_three_arm_prime as objective_module
from redco.analysis.stage_d_prime_trainer import main
from redco.analysis.stage_d_three_arm_prime import StageDPrimeRuntimeGate


def test_stage_d_trainer_entrypoint_refuses_every_missing_gate_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "REDCO_STAGE_D_OBJECTIVE_ARM",
        "REDCO_STAGE_D_OBJECTIVE_BINDING",
        "REDCO_STAGE_D_OBJECTIVE_AUTHORIZATION",
        "REDCO_STAGE_D_OBJECTIVE_AUTHORIZATION_SHA256",
        "REDCO_STAGE_D_SEALED_ARM_BATCH",
        "REDCO_STAGE_D_BATCH_AUTHORIZATION_RECEIPT",
        "REDCO_STAGE_D_LEDGER_ROOT",
        "REDCO_STAGE_D_LEDGER_SEAL",
        "REDCO_STAGE_D_LEDGER_SEAL_SHA256",
        "REDCO_STAGE_D_CAMPAIGN_MANIFEST",
        "REDCO_STAGE_D_CAMPAIGN_MANIFEST_SHA256",
        "REDCO_STAGE_D_TRAINER_RUN_LEDGER",
        "REDCO_STAGE_D_TRAINER_LAUNCH_ID",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="mandatory Stage D trainer gate"):
        main()


def test_runtime_gate_requires_exactly_one_consumed_batch_before_exit() -> None:
    gate = StageDPrimeRuntimeGate(
        binding=object(),  # type: ignore[arg-type]
        batch=object(),  # type: ignore[arg-type]
        objective_authorization_sha256="a" * 64,
        batch_authorization_sha256="b" * 64,
        ledger_seal_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="complete sealed update"):
        gate.verify_finished()
    gate._batch_verified = True
    with pytest.raises(ValueError, match="complete sealed update"):
        gate.verify_finished()
    gate._optimizer_completed = True
    gate.verify_finished()


@pytest.mark.parametrize("consume_batch", [False, True])
def test_trainer_entrypoint_checks_consumption_after_mocked_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consume_batch: bool,
) -> None:
    files = {
        "REDCO_STAGE_D_OBJECTIVE_ARM": "local",
        "REDCO_STAGE_D_OBJECTIVE_BINDING": "binding",
        "REDCO_STAGE_D_OBJECTIVE_AUTHORIZATION": "authorization",
        "REDCO_STAGE_D_OBJECTIVE_AUTHORIZATION_SHA256": "d" * 64,
        "REDCO_STAGE_D_SEALED_ARM_BATCH": "batch",
        "REDCO_STAGE_D_BATCH_AUTHORIZATION_RECEIPT": "batch-receipt",
        "REDCO_STAGE_D_LEDGER_ROOT": "ledger-root",
        "REDCO_STAGE_D_LEDGER_SEAL": "ledger-seal",
        "REDCO_STAGE_D_CAMPAIGN_MANIFEST": "campaign-manifest",
        "REDCO_STAGE_D_TRAINER_RUN_LEDGER": "trainer-run-ledger",
    }
    for name, filename in files.items():
        path = tmp_path / filename
        if name in {"REDCO_STAGE_D_LEDGER_ROOT", "REDCO_STAGE_D_TRAINER_RUN_LEDGER"}:
            path.mkdir()
        else:
            path.write_bytes(filename.encode())
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("REDCO_STAGE_D_OBJECTIVE_ARM", "local")
    seal_bytes = (tmp_path / "ledger-seal").read_bytes()
    monkeypatch.setenv(
        "REDCO_STAGE_D_LEDGER_SEAL_SHA256",
        hashlib.sha256(seal_bytes).hexdigest(),
    )
    manifest_bytes = (tmp_path / "campaign-manifest").read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    monkeypatch.setenv("REDCO_STAGE_D_CAMPAIGN_MANIFEST_SHA256", manifest_sha256)
    monkeypatch.setenv("REDCO_STAGE_D_TRAINER_LAUNCH_ID", "local-launch")

    gate = StageDPrimeRuntimeGate(
        binding=object(),  # type: ignore[arg-type]
        batch=object(),  # type: ignore[arg-type]
        objective_authorization_sha256="a" * 64,
        batch_authorization_sha256="b" * 64,
        ledger_seal_sha256="c" * 64,
    )
    train_module = SimpleNamespace(
        set_proc_title=lambda _title: None,
        train=lambda _config, *, redco_runtime_gate: setattr(
            redco_runtime_gate, "_optimizer_completed", consume_batch
        ),
    )
    modules = {
        "prime_rl.configs.trainer": SimpleNamespace(TrainerConfig=object),
        "prime_rl.utils.config": SimpleNamespace(cli=lambda _config: object()),
        "prime_rl.trainer.rl.train": train_module,
    }
    original_import = trainer_entrypoint.importlib.import_module
    monkeypatch.setattr(
        trainer_entrypoint.importlib,
        "import_module",
        lambda name: modules[name] if name in modules else original_import(name),
    )
    monkeypatch.setattr(
        objective_module,
        "capture_prime_objective_cli",
        lambda: SimpleNamespace(trainer_toml_bytes=b"trainer-toml"),
    )
    gate._batch_verified = consume_batch
    monkeypatch.setattr(
        objective_module,
        "verify_captured_prime_objective",
        lambda **_kwargs: (gate, object(), object()),
    )
    monkeypatch.setattr(
        trainer_entrypoint,
        "LedgerSeal",
        SimpleNamespace(from_bytes=lambda _value: object()),
    )
    monkeypatch.setattr(
        trainer_entrypoint,
        "SealedReceiptVerifier",
        lambda _root, _seal: object(),
    )
    run_snapshot = SimpleNamespace(
        campaign_manifest_sha256=manifest_sha256,
        trainer_config_sha256s=(
            ("local", hashlib.sha256(b"trainer-toml").hexdigest()),
        ),
        state=lambda _arm: SimpleNamespace(active_launch_id="local-launch"),
    )
    monkeypatch.setattr(
        trainer_entrypoint,
        "StageDTrainerRunLedger",
        lambda _root: SimpleNamespace(inspect=lambda: run_snapshot),
    )

    if consume_batch:
        main()
    else:
        with pytest.raises(ValueError, match="complete sealed update"):
            main()
