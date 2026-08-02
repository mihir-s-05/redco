from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from redco.analysis.stage_d_three_arm_prime import StageDPrimeRuntimeGate
from redco.analysis.stage_d_trainer_supervisor import StageDTrainerRunLedger


def _sha(character: str) -> str:
    return character * 64


def _create(root: Path) -> StageDTrainerRunLedger:
    return StageDTrainerRunLedger.create(
        root,
        campaign_manifest_sha256=_sha("a"),
        initialization_sha256=_sha("b"),
        batch_identities={
            "stock": _sha("1"),
            "branch-global": _sha("2"),
            "local": _sha("3"),
        },
        trainer_config_sha256s={
            "stock": _sha("4"),
            "branch-global": _sha("5"),
            "local": _sha("6"),
        },
    )


def test_one_preupdate_repair_then_optimizer_is_permanently_single_use(tmp_path: Path) -> None:
    ledger = _create(tmp_path / "runs")
    ledger.claim_launch(arm="stock", launch_id="stock-0")
    ledger.mark_batch_verified(
        arm="stock", launch_id="stock-0", batch_identity=_sha("1")
    )
    ledger.record_preupdate_failure(
        arm="stock",
        launch_id="stock-0",
        reason="driver import failed",
        evidence_sha256=_sha("7"),
    )
    ledger.claim_launch(arm="stock", launch_id="stock-1")
    ledger.mark_batch_verified(
        arm="stock", launch_id="stock-1", batch_identity=_sha("1")
    )
    ledger.mark_optimizer_started(arm="stock", launch_id="stock-1", trainer_step=1)
    with pytest.raises(RuntimeError, match="out of order"):
        ledger.record_preupdate_failure(
            arm="stock",
            launch_id="stock-1",
            reason="too late",
            evidence_sha256=_sha("8"),
        )
    with pytest.raises(RuntimeError, match="active launch"):
        ledger.claim_launch(arm="stock", launch_id="stock-2")
    ledger.mark_optimizer_completed(arm="stock", launch_id="stock-1", trainer_step=1)
    ledger.commit_checkpoint(
        arm="stock",
        launch_id="stock-1",
        checkpoint_sha256=_sha("9"),
        metrics_sha256=_sha("c"),
        reload_evidence_sha256=_sha("d"),
    )
    with pytest.raises(RuntimeError, match="arm order"):
        ledger.claim_launch(arm="local", launch_id="local-0")
    ledger.claim_launch(arm="branch-global", launch_id="global-0")
    snapshot = ledger.inspect()
    assert snapshot.state("stock").checkpoint_sha256 == _sha("9")
    assert snapshot.state("stock").launch_attempts == 2
    assert snapshot.state("stock").preupdate_failures == 1


def test_concurrent_duplicate_launch_has_one_winner(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _create(root)

    def claim(launch_id: str) -> str:
        try:
            StageDTrainerRunLedger(root).claim_launch(arm="stock", launch_id=launch_id)
            return "won"
        except RuntimeError:
            return "lost"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ("one", "two")))
    assert sorted(results) == ["lost", "won"]
    assert StageDTrainerRunLedger(root).inspect().state("stock").launch_attempts == 1


def test_record_tampering_is_detected_before_mutation(tmp_path: Path) -> None:
    ledger = _create(tmp_path / "runs")
    path = ledger.records / "00000000.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="canonical JSON"):
        ledger.claim_launch(arm="stock", launch_id="stock-0")


def test_runtime_gate_durably_brackets_the_optimizer_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _create(tmp_path / "runs")
    ledger.claim_launch(arm="stock", launch_id="stock-live")
    distributed = SimpleNamespace(
        get_rank=lambda: 0,
        get_world_size=lambda: 1,
        all_gather_object=lambda output, value: output.__setitem__(0, value),
    )
    import redco.analysis.stage_d_three_arm_prime as prime_module

    original_import = prime_module.importlib.import_module
    monkeypatch.setattr(
        prime_module.importlib,
        "import_module",
        lambda name: distributed if name == "torch.distributed" else original_import(name),
    )
    gate = StageDPrimeRuntimeGate(
        binding=object(),  # type: ignore[arg-type]
        batch=SimpleNamespace(arm="stock", batch_identity=_sha("1"), trainer_step=1),
        objective_authorization_sha256=_sha("a"),
        batch_authorization_sha256=_sha("b"),
        ledger_seal_sha256=_sha("c"),
        trainer_run_ledger=ledger,
        launch_id="stock-live",
    )
    gate._record_supervisor("mark_batch_verified", batch_identity=_sha("1"))
    gate._batch_verified = True
    gate.before_optimizer_step(trainer_step=1)
    gate.after_optimizer_step(trainer_step=1)
    gate.verify_finished()
    state = ledger.inspect().state("stock")
    assert state.batch_verified is True
    assert state.optimizer_started is True
    assert state.optimizer_completed is True
