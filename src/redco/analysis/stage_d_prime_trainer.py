"""Mandatory-gate Prime trainer entrypoint for Stage-D scientific runs."""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
from typing import cast

from redco.analysis.stage_d_objective_binding import ArmName
from redco.analysis.stage_d_receipt_ledger import LedgerSeal, SealedReceiptVerifier
from redco.analysis.stage_d_trainer_supervisor import StageDTrainerRunLedger

_ENV = {
    "arm": "REDCO_STAGE_D_OBJECTIVE_ARM",
    "binding": "REDCO_STAGE_D_OBJECTIVE_BINDING",
    "authorization": "REDCO_STAGE_D_OBJECTIVE_AUTHORIZATION",
    "authorization_sha256": "REDCO_STAGE_D_OBJECTIVE_AUTHORIZATION_SHA256",
    "batch": "REDCO_STAGE_D_SEALED_ARM_BATCH",
    "batch_authorization": "REDCO_STAGE_D_BATCH_AUTHORIZATION_RECEIPT",
    "ledger_root": "REDCO_STAGE_D_LEDGER_ROOT",
    "ledger_seal": "REDCO_STAGE_D_LEDGER_SEAL",
    "ledger_seal_sha256": "REDCO_STAGE_D_LEDGER_SEAL_SHA256",
    "campaign_manifest": "REDCO_STAGE_D_CAMPAIGN_MANIFEST",
    "campaign_manifest_sha256": "REDCO_STAGE_D_CAMPAIGN_MANIFEST_SHA256",
    "trainer_run_ledger": "REDCO_STAGE_D_TRAINER_RUN_LEDGER",
    "launch_id": "REDCO_STAGE_D_TRAINER_LAUNCH_ID",
}


def main() -> None:
    """Refuse to enter Prime unless every frozen Stage-D gate input exists."""
    values = {name: os.environ.get(variable) for name, variable in _ENV.items()}
    if any(not value for value in values.values()):
        raise ValueError("mandatory Stage D trainer gate environment is incomplete")
    from redco.analysis.stage_d_three_arm_prime import (
        capture_prime_objective_cli,
        verify_captured_prime_objective,
    )

    capture = capture_prime_objective_cli()
    trainer_config_module = importlib.import_module("prime_rl.configs.trainer")
    config_utils = importlib.import_module("prime_rl.utils.config")
    train_module = importlib.import_module("prime_rl.trainer.rl.train")
    expected_binding = Path(str(values["binding"])).read_bytes()
    authorization = Path(str(values["authorization"])).read_bytes()
    sealed_batch = Path(str(values["batch"])).read_bytes()
    batch_authorization = Path(str(values["batch_authorization"])).read_bytes()
    ledger_seal_bytes = Path(str(values["ledger_seal"])).read_bytes()
    ledger_seal_sha256 = hashlib.sha256(ledger_seal_bytes).hexdigest()
    if ledger_seal_sha256 != values["ledger_seal_sha256"]:
        raise ValueError("Stage D ledger seal differs from the frozen digest")
    receipt_verifier = SealedReceiptVerifier(
        Path(str(values["ledger_root"])),
        LedgerSeal.from_bytes(ledger_seal_bytes),
    )
    campaign_manifest = Path(str(values["campaign_manifest"])).read_bytes()
    campaign_manifest_sha256 = hashlib.sha256(campaign_manifest).hexdigest()
    if campaign_manifest_sha256 != values["campaign_manifest_sha256"]:
        raise ValueError("Stage D campaign manifest differs from the frozen digest")
    trainer_run_ledger = StageDTrainerRunLedger(Path(str(values["trainer_run_ledger"])))
    run_snapshot = trainer_run_ledger.inspect()
    raw_arm = str(values["arm"])
    if raw_arm not in {"stock", "branch-global", "local"}:
        raise ValueError("Stage D trainer arm is invalid")
    arm = cast(ArmName, raw_arm)
    launch_id = str(values["launch_id"])
    if (
        run_snapshot.campaign_manifest_sha256 != campaign_manifest_sha256
        or run_snapshot.state(arm).active_launch_id != launch_id
        or dict(run_snapshot.trainer_config_sha256s).get(arm)
        != hashlib.sha256(capture.trainer_toml_bytes).hexdigest()
    ):
        raise ValueError("Stage D trainer launch differs from its supervisor authorization")
    config = config_utils.cli(trainer_config_module.TrainerConfig)
    gate, _, _ = verify_captured_prime_objective(
        config=config,
        capture=capture,
        arm=arm,
        train_module=train_module,
        expected_binding_bytes=expected_binding,
        authorization_bytes=authorization,
        expected_authorization_sha256=str(values["authorization_sha256"]),
        sealed_batch_bytes=sealed_batch,
        batch_authorization_receipt=batch_authorization,
        receipt_verifier=receipt_verifier,
        ledger_seal_sha256=ledger_seal_sha256,
        trainer_run_ledger=trainer_run_ledger,
        launch_id=launch_id,
    )
    train_module.set_proc_title("Stage D Trainer")
    train_module.train(config, redco_runtime_gate=gate)
    gate.verify_finished()


if __name__ == "__main__":
    main()
