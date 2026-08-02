from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from redco.analysis.stage_d_checkpoint_evidence import (
    CheckpointMember,
    StageDCheckpointManifest,
    StageDReloadEvidence,
    StageDTrainerMetricsEvidence,
)
from redco.analysis.stage_d_process_supervision import TrainerProcessStartReceipt
from redco.analysis.stage_d_reload_supervisor import ReloadWorkerResult
from redco.analysis.stage_d_three_arm_prime import StageDPrimeRuntimeGate
from redco.analysis.stage_d_trainer_supervisor import StageDTrainerRunLedger
from redco.analysis.stage_d_training_completion import StageDTrainingCompletion


def _sha(character: str) -> str:
    return character * 64


def _run_hard_crash(code: str, root: Path, stage: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), environment.get("PYTHONPATH", "")))
    )
    return subprocess.run(
        [sys.executable, "-c", code, str(root), stage],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _create(root: Path) -> StageDTrainerRunLedger:
    return StageDTrainerRunLedger.create(
        root,
        campaign_manifest_sha256=_sha("a"),
        protocol_manifest_sha256=_sha("b"),
        shared_initialization_manifest_sha256=_sha("c"),
        expected_pre_model_sha256=_sha("d"),
        expected_base_model_manifest_sha256=_sha("e"),
        reload_probe_sha256=_sha("f"),
        trainer_step=1,
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
        process_command_sha256s={
            "stock": _sha("a"),
            "branch-global": _sha("b"),
            "local": _sha("c"),
        },
        process_environment_sha256s={
            "stock": _sha("7"),
            "branch-global": _sha("8"),
            "local": _sha("9"),
        },
    )


_CRASH_CREATE = r"""
import os
import sys
from pathlib import Path
from redco.analysis.stage_d_trainer_supervisor import StageDTrainerRunLedger

def sha(character):
    return character * 64

def crash(observed, _path):
    if observed == sys.argv[2]:
        os._exit(91)

StageDTrainerRunLedger.create(
    Path(sys.argv[1]),
    campaign_manifest_sha256=sha("a"),
    protocol_manifest_sha256=sha("b"),
    shared_initialization_manifest_sha256=sha("c"),
    expected_pre_model_sha256=sha("d"),
    expected_base_model_manifest_sha256=sha("e"),
    reload_probe_sha256=sha("f"),
    trainer_step=1,
    batch_identities={"stock": sha("1"), "branch-global": sha("2"), "local": sha("3")},
    trainer_config_sha256s={"stock": sha("4"), "branch-global": sha("5"), "local": sha("6")},
    process_command_sha256s={"stock": sha("a"), "branch-global": sha("b"), "local": sha("c")},
    process_environment_sha256s={"stock": sha("7"), "branch-global": sha("8"), "local": sha("9")},
    fault_hook=crash,
)
"""


def _mark_process_started(
    ledger: StageDTrainerRunLedger,
    *,
    arm: str,
    launch_id: str,
) -> None:
    receipt = TrainerProcessStartReceipt(
        arm=arm,  # type: ignore[arg-type]
        launch_id=launch_id,
        pid=123,
        boot_id="fixture-boot",
        process_start_ticks="456",
        command_sha256={"stock": _sha("a"), "branch-global": _sha("b"), "local": _sha("c")}[arm],
        environment_manifest_sha256={
            "stock": _sha("7"),
            "branch-global": _sha("8"),
            "local": _sha("9"),
        }[arm],
    )
    ledger.mark_process_started(
        arm=arm,  # type: ignore[arg-type]
        launch_id=launch_id,
        process_receipt_bytes=receipt.to_bytes(),
    )


def _checkpoint_evidence(
    arm: str,
    root: Path,
    post_model_sha256: str,
    *,
    launch_id: str = "stock-1",
) -> tuple[
    Path,
    str,
    bytes,
    bytes,
    bytes,
    tuple[bytes, bytes],
    tuple[bytes, bytes],
]:
    checkpoint_root = root / f"checkpoint-{arm}"
    checkpoint_root.mkdir()
    (checkpoint_root / "STABLE").write_bytes(b"")
    (checkpoint_root / "adapter_config.json").write_bytes(b"{}")
    (checkpoint_root / "adapter_model.safetensors").write_bytes(b"adapter")
    members = tuple(
        CheckpointMember(
            path.relative_to(checkpoint_root).as_posix(),
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(
            checkpoint_root.iterdir(),
            key=lambda candidate: candidate.relative_to(checkpoint_root).as_posix(),
        )
    )
    manifest = StageDCheckpointManifest(
        arm=arm,  # type: ignore[arg-type]
        trainer_step=1,
        base_model_manifest_sha256=_sha("e"),
        post_model_sha256=post_model_sha256,
        members=members,
    )
    output = b'{"result":"stable"}\n'
    output_sha256 = hashlib.sha256(output).hexdigest()
    result_values = tuple(
        ReloadWorkerResult(
            arm=arm,  # type: ignore[arg-type]
            launch_nonce=_sha(str(ordinal)),
            pid=100 + ordinal,
            boot_id=f"fixture-boot-{ordinal}",
            process_start_ticks=str(200 + ordinal),
            checkpoint_manifest_sha256=manifest.manifest_sha256,
            post_model_sha256=post_model_sha256,
            loaded_model_sha256=post_model_sha256,
            reload_probe_sha256=_sha("f"),
            base_model_manifest_sha256=_sha("e"),
            tokenizer_manifest_sha256=_sha("1"),
            renderer_manifest_sha256=_sha("2"),
            runtime_manifest_sha256=_sha("3"),
            python_executable_sha256=_sha("4"),
            worker_source_sha256=_sha("5"),
            worker_command_sha256=_sha("6"),
            worker_environment_sha256=_sha("7"),
            working_directory_sha256=_sha("8"),
            output_sha256=output_sha256,
        ).to_bytes()
        for ordinal in (7, 8)
    )
    process_identities = tuple(
        ReloadWorkerResult.from_bytes(value).identity for value in result_values
    )
    reload = StageDReloadEvidence(
        arm=arm,  # type: ignore[arg-type]
        checkpoint_manifest_sha256=manifest.manifest_sha256,
        post_model_sha256=post_model_sha256,
        reload_probe_sha256=_sha("f"),
        process_identities=process_identities,  # type: ignore[arg-type]
        output_sha256s=(output_sha256, output_sha256),
    )
    metrics = StageDTrainerMetricsEvidence(
        arm=arm,  # type: ignore[arg-type]
        launch_id=launch_id,
        batch_identity={"stock": _sha("1"), "branch-global": _sha("2"), "local": _sha("3")}[arm],
        trainer_step=1,
        pre_model_sha256=_sha("d"),
        post_model_sha256=post_model_sha256,
        model_changed=post_model_sha256 != _sha("d"),
        optimizer_updates=1,
        loss=0.25,
        grad_norm=1.5,
    )
    return (
        checkpoint_root,
        post_model_sha256,
        manifest.to_bytes(),
        metrics.to_bytes(),
        reload.to_bytes(),
        (output, output),
        result_values,
    )


def test_one_preupdate_repair_then_optimizer_is_permanently_single_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _create(tmp_path / "runs")
    ledger.claim_launch(arm="stock", launch_id="stock-0")
    _mark_process_started(ledger, arm="stock", launch_id="stock-0")
    ledger.mark_initialization_verified(
        arm="stock", launch_id="stock-0", observed_pre_model_sha256=_sha("d")
    )
    ledger.mark_batch_verified(arm="stock", launch_id="stock-0", batch_identity=_sha("1"))
    ledger.record_preupdate_failure(
        arm="stock",
        launch_id="stock-0",
        reason="driver import failed",
        evidence_bytes=b"driver import failed before optimizer start",
    )
    ledger.claim_launch(arm="stock", launch_id="stock-1")
    _mark_process_started(ledger, arm="stock", launch_id="stock-1")
    ledger.mark_initialization_verified(
        arm="stock", launch_id="stock-1", observed_pre_model_sha256=_sha("d")
    )
    ledger.mark_batch_verified(arm="stock", launch_id="stock-1", batch_identity=_sha("1"))
    ledger.mark_optimizer_started(arm="stock", launch_id="stock-1", trainer_step=1)
    with pytest.raises(RuntimeError, match="out of order"):
        ledger.record_preupdate_failure(
            arm="stock",
            launch_id="stock-1",
            reason="too late",
            evidence_bytes=b"too late",
        )
    with pytest.raises(RuntimeError, match="active launch"):
        ledger.claim_launch(arm="stock", launch_id="stock-2")
    post_model = _sha("9")
    (
        checkpoint_root,
        post_model,
        checkpoint,
        metrics,
        reload,
        outputs,
        reload_results,
    ) = _checkpoint_evidence("stock", tmp_path, post_model)
    monkeypatch.setattr(
        "redco.analysis.stage_d_checkpoint_evidence.adapter_file_state_sha256",
        lambda *_args, **_kwargs: post_model,
    )
    ledger.mark_optimizer_completed(
        arm="stock",
        launch_id="stock-1",
        trainer_step=1,
        post_model_sha256=post_model,
    )
    ledger.commit_checkpoint(
        arm="stock",
        launch_id="stock-1",
        checkpoint_root=checkpoint_root,
        checkpoint_manifest_bytes=checkpoint,
        metrics_bytes=metrics,
        reload_evidence_bytes=reload,
        reload_output_bytes=outputs,
        reload_process_result_bytes=reload_results,
        trainer_step=1,
    )
    with pytest.raises(RuntimeError, match="arm order"):
        ledger.claim_launch(arm="local", launch_id="local-0")
    ledger.claim_launch(arm="branch-global", launch_id="global-0")
    with pytest.raises(RuntimeError, match="globally exhausted"):
        ledger.record_preupdate_failure(
            arm="branch-global",
            launch_id="global-0",
            reason="second campaign repair",
            evidence_bytes=b"forbidden second repair",
        )
    snapshot = ledger.inspect()
    assert snapshot.state("stock").checkpoint_sha256 == hashlib.sha256(checkpoint).hexdigest()
    assert snapshot.state("stock").metrics_sha256 == hashlib.sha256(metrics).hexdigest()
    assert snapshot.state("stock").reload_evidence_sha256 == hashlib.sha256(reload).hexdigest()
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


def test_hard_kill_during_genesis_temp_write_is_restartable(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    result = _run_hard_crash(_CRASH_CREATE, root, "after-record-temp-fsync")
    assert result.returncode == 91
    assert not tuple((root / "records").glob("*.json"))
    assert tuple((root / "records").glob(".*.pending"))
    ledger = _create(root)
    assert ledger.inspect().record_count == 1
    assert not tuple(ledger.records.glob(".*.pending"))


def test_hard_kill_during_launch_claim_is_restartable(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _create(root)
    code = r"""
import os
import sys
from pathlib import Path
from redco.analysis.stage_d_trainer_supervisor import StageDTrainerRunLedger

def crash(observed, _path):
    if observed == sys.argv[2]:
        os._exit(92)

StageDTrainerRunLedger(Path(sys.argv[1]), fault_hook=crash).claim_launch(
    arm="stock", launch_id="stock-live"
)
"""
    result = _run_hard_crash(code, root, "after-record-temp-fsync")
    assert result.returncode == 92
    StageDTrainerRunLedger(root).claim_launch(arm="stock", launch_id="stock-live")
    snapshot = StageDTrainerRunLedger(root).inspect()
    assert snapshot.state("stock").launch_attempts == 1
    assert snapshot.state("stock").active_launch_id == "stock-live"


def test_hard_kill_during_evidence_write_is_restartable(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    ledger = _create(root)
    ledger.claim_launch(arm="stock", launch_id="stock-live")
    code = r"""
import os
import sys
from pathlib import Path
from redco.analysis.stage_d_process_supervision import TrainerProcessStartReceipt
from redco.analysis.stage_d_trainer_supervisor import StageDTrainerRunLedger

def sha(character):
    return character * 64

def crash(observed, _path):
    if observed == sys.argv[2]:
        os._exit(93)

receipt = TrainerProcessStartReceipt(
    arm="stock",
    launch_id="stock-live",
    pid=123,
    boot_id="fixture-boot",
    process_start_ticks="456",
    command_sha256=sha("a"),
    environment_manifest_sha256=sha("7"),
)
StageDTrainerRunLedger(Path(sys.argv[1]), fault_hook=crash).mark_process_started(
    arm="stock", launch_id="stock-live", process_receipt_bytes=receipt.to_bytes()
)
"""
    result = _run_hard_crash(code, root, "after-evidence-temp-fsync")
    assert result.returncode == 93
    _mark_process_started(ledger, arm="stock", launch_id="stock-live")
    assert ledger.inspect().state("stock").process_started is True
    assert not tuple(ledger.evidence.glob(".*.pending"))


def test_record_tampering_is_detected_before_mutation(tmp_path: Path) -> None:
    ledger = _create(tmp_path / "runs")
    path = ledger.records / "00000000.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="canonical JSON"):
        ledger.claim_launch(arm="stock", launch_id="stock-0")


def test_missing_retained_checkpoint_member_is_detected_on_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _create(tmp_path / "runs")
    ledger.claim_launch(arm="stock", launch_id="stock-live")
    _mark_process_started(ledger, arm="stock", launch_id="stock-live")
    ledger.mark_initialization_verified(
        arm="stock", launch_id="stock-live", observed_pre_model_sha256=_sha("d")
    )
    ledger.mark_batch_verified(arm="stock", launch_id="stock-live", batch_identity=_sha("1"))
    ledger.mark_optimizer_started(arm="stock", launch_id="stock-live", trainer_step=1)
    post_model = _sha("9")
    (
        checkpoint_root,
        _,
        checkpoint,
        metrics,
        reload,
        outputs,
        reload_results,
    ) = _checkpoint_evidence("stock", tmp_path, post_model, launch_id="stock-live")
    monkeypatch.setattr(
        "redco.analysis.stage_d_checkpoint_evidence.adapter_file_state_sha256",
        lambda *_args, **_kwargs: post_model,
    )
    ledger.mark_optimizer_completed(
        arm="stock",
        launch_id="stock-live",
        trainer_step=1,
        post_model_sha256=post_model,
    )
    ledger.commit_checkpoint(
        arm="stock",
        launch_id="stock-live",
        checkpoint_root=checkpoint_root,
        checkpoint_manifest_bytes=checkpoint,
        metrics_bytes=metrics,
        reload_evidence_bytes=reload,
        reload_output_bytes=outputs,
        reload_process_result_bytes=reload_results,
        trainer_step=1,
    )
    manifest = StageDCheckpointManifest.from_bytes(checkpoint)
    adapter = next(
        member for member in manifest.members if member.path == "adapter_model.safetensors"
    )
    (ledger.evidence / adapter.sha256).unlink()
    with pytest.raises(ValueError, match="missing evidence"):
        ledger.inspect()


def test_terminal_training_completion_reopens_full_three_arm_evidence_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _create(tmp_path / "runs")
    post_by_arm = {"stock": _sha("7"), "branch-global": _sha("8"), "local": _sha("9")}
    monkeypatch.setattr(
        "redco.analysis.stage_d_checkpoint_evidence.adapter_file_state_sha256",
        lambda path, **_kwargs: post_by_arm[path.parent.name.removeprefix("checkpoint-")],
    )
    for arm in ("stock", "branch-global", "local"):
        launch_id = f"{arm}-complete"
        ledger.claim_launch(arm=arm, launch_id=launch_id)
        _mark_process_started(ledger, arm=arm, launch_id=launch_id)
        ledger.mark_initialization_verified(
            arm=arm,
            launch_id=launch_id,
            observed_pre_model_sha256=_sha("d"),
        )
        ledger.mark_batch_verified(
            arm=arm,
            launch_id=launch_id,
            batch_identity={"stock": _sha("1"), "branch-global": _sha("2"), "local": _sha("3")}[
                arm
            ],
        )
        ledger.mark_optimizer_started(arm=arm, launch_id=launch_id, trainer_step=1)
        post_model = post_by_arm[arm]
        ledger.mark_optimizer_completed(
            arm=arm,
            launch_id=launch_id,
            trainer_step=1,
            post_model_sha256=post_model,
        )
        checkpoint = _checkpoint_evidence(
            arm,
            tmp_path,
            post_model,
            launch_id=launch_id,
        )
        ledger.commit_checkpoint(
            arm=arm,
            launch_id=launch_id,
            checkpoint_root=checkpoint[0],
            checkpoint_manifest_bytes=checkpoint[2],
            metrics_bytes=checkpoint[3],
            reload_evidence_bytes=checkpoint[4],
            reload_output_bytes=checkpoint[5],
            reload_process_result_bytes=checkpoint[6],
            trainer_step=1,
        )
    completion = StageDTrainingCompletion.build(ledger)
    assert StageDTrainingCompletion.from_bytes(completion.to_bytes()) == completion
    completion.verify_ledger(ledger)
    assert tuple(item.arm for item in completion.arms) == (
        "stock",
        "branch-global",
        "local",
    )
    missing = ledger.evidence / completion.arms[0].reload_process_result_sha256s[0]
    missing.unlink()
    with pytest.raises(ValueError):
        completion.verify_ledger(ledger)


def test_runtime_gate_durably_brackets_the_optimizer_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _create(tmp_path / "runs")
    ledger.claim_launch(arm="stock", launch_id="stock-live")
    _mark_process_started(ledger, arm="stock", launch_id="stock-live")
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
    import redco.analysis.stage_d_live_update as live_update

    monkeypatch.setattr(
        live_update,
        "exported_adapter_state_sha256",
        lambda **kwargs: _sha("d") if kwargs else _sha("0"),
    )
    gate = StageDPrimeRuntimeGate(
        binding=object(),  # type: ignore[arg-type]
        batch=SimpleNamespace(arm="stock", batch_identity=_sha("1"), trainer_step=1),
        objective_authorization_sha256=_sha("a"),
        batch_authorization_sha256=_sha("b"),
        ledger_seal_sha256=_sha("c"),
        expected_pre_model_sha256=_sha("d"),
        base_model_manifest_sha256=_sha("e"),
        trainer_run_ledger=ledger,
        launch_id="stock-live",
    )
    ledger.mark_initialization_verified(
        arm="stock",
        launch_id="stock-live",
        observed_pre_model_sha256=_sha("d"),
    )
    gate._initialization_verified = True
    gate._record_supervisor("mark_batch_verified", batch_identity=_sha("1"))
    gate._batch_verified = True
    gate.before_optimizer_step(trainer_step=1)
    gate.after_optimizer_step(trainer_step=1)
    gate.verify_finished()
    state = ledger.inspect().state("stock")
    assert state.batch_verified is True
    assert state.optimizer_started is True
    assert state.optimizer_completed is True
    assert state.model_changed is False


def test_actual_initialization_export_is_verified_before_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _create(tmp_path / "runs")
    ledger.claim_launch(arm="stock", launch_id="stock-live")
    _mark_process_started(ledger, arm="stock", launch_id="stock-live")
    distributed = SimpleNamespace(
        get_rank=lambda: 0,
        get_world_size=lambda: 1,
        all_gather_object=lambda output, value: output.__setitem__(0, value),
    )
    import redco.analysis.stage_d_live_update as live_update
    import redco.analysis.stage_d_three_arm_prime as prime_module

    original_import = prime_module.importlib.import_module
    monkeypatch.setattr(
        prime_module.importlib,
        "import_module",
        lambda name: distributed if name == "torch.distributed" else original_import(name),
    )
    monkeypatch.setattr(
        live_update,
        "exported_adapter_state_sha256",
        lambda **kwargs: _sha("d") if kwargs else _sha("0"),
    )
    gate = StageDPrimeRuntimeGate(
        binding=object(),  # type: ignore[arg-type]
        batch=SimpleNamespace(arm="stock", batch_identity=_sha("1"), trainer_step=1),
        objective_authorization_sha256=_sha("a"),
        batch_authorization_sha256=_sha("b"),
        ledger_seal_sha256=_sha("c"),
        expected_pre_model_sha256=_sha("d"),
        base_model_manifest_sha256=_sha("e"),
        trainer_run_ledger=ledger,
        launch_id="stock-live",
    )
    gate.verify_initialization()
    assert ledger.inspect().state("stock").initialization_verified is True


def test_wrong_step_and_arm_order_fail_without_appending(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="arm order"):
        StageDTrainerRunLedger.create(
            tmp_path / "wrong-order",
            campaign_manifest_sha256=_sha("a"),
            protocol_manifest_sha256=_sha("b"),
            shared_initialization_manifest_sha256=_sha("c"),
            expected_pre_model_sha256=_sha("d"),
            expected_base_model_manifest_sha256=_sha("e"),
            reload_probe_sha256=_sha("f"),
            trainer_step=1,
            batch_identities={"stock": _sha("1"), "branch-global": _sha("2"), "local": _sha("3")},
            trainer_config_sha256s={
                "stock": _sha("4"),
                "branch-global": _sha("5"),
                "local": _sha("6"),
            },
            process_command_sha256s={
                "stock": _sha("a"),
                "branch-global": _sha("b"),
                "local": _sha("c"),
            },
            process_environment_sha256s={
                "stock": _sha("7"),
                "branch-global": _sha("8"),
                "local": _sha("9"),
            },
            arm_order=("local", "branch-global", "stock"),
        )
    ledger = _create(tmp_path / "runs")
    ledger.claim_launch(arm="stock", launch_id="stock-live")
    _mark_process_started(ledger, arm="stock", launch_id="stock-live")
    ledger.mark_initialization_verified(
        arm="stock", launch_id="stock-live", observed_pre_model_sha256=_sha("d")
    )
    ledger.mark_batch_verified(arm="stock", launch_id="stock-live", batch_identity=_sha("1"))
    before = ledger.inspect().record_count
    with pytest.raises(RuntimeError, match="frozen trainer step"):
        ledger.mark_optimizer_started(arm="stock", launch_id="stock-live", trainer_step=2)
    assert ledger.inspect().record_count == before
