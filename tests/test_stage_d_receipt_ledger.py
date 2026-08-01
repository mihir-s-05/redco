from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import redco.analysis.stage_d_receipt_ledger as receipt_ledger_module
from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.analysis.stage_d_receipt_ledger import (
    BatchAlreadyClaimed,
    GenesisBinding,
    LedgerError,
    SealedReceiptVerifier,
    StageDReceiptLedger,
    inspect_ledger,
)
from redco.analysis.stage_d_scientific_branch_group import (
    BranchGroupArtifact,
    BranchGroupSpec,
    BranchSeedOracle,
    CandidateSubmission,
    OutcomeKind,
    PreActionTargetCommitment,
    SeedCorrespondenceMap,
    run_scientific_branch_group,
)
from redco.analysis.stage_d_spawn_provenance import (
    EventSeedScheduler,
    PolicyEventAddress,
)
from redco.contracts import ActualEvaluationCost, canonical_json

MASTER_SEED = "durable-master"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _binding() -> GenesisBinding:
    return GenesisBinding(
        preregistration_sha256="1" * 64,
        source_sha256="2" * 64,
        runtime_sha256="3" * 64,
        config_sha256="4" * 64,
        master_seed_sha256=_sha256(MASTER_SEED.encode()),
    )


def _conformance() -> bytes:
    payload = {
        "schema_version": 1,
        "analysis": "served-stack-categorical-logprob-conformance-v1",
        "passes": True,
        "logprob_semantics": "served_chosen_token_post_transform",
        "categorical_case_count": 3,
        "served_stack_sha256": "a" * 64,
        "tool_call_termination_includes_all_generated_tokens": True,
        "eos_is_included_in_action_tokens_and_logprobs": True,
    }
    payload["signed_payload_sha256"] = _sha256(canonical_json(payload))
    return canonical_json(payload)


def _request(seed: int) -> dict[str, object]:
    return {
        "model": "model@commit",
        "messages": [{"role": "user", "content": "q"}],
        "tools": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": None,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "logit_bias": {},
        "seed": seed,
        "max_tokens": 2,
        "stop": None,
        "n": 1,
        "best_of": None,
        "use_beam_search": False,
        "logprobs": True,
        "top_logprobs": 0,
        "ignore_eos": False,
        "min_tokens": 0,
        "extra_body": {"cache_salt": f"seed-{seed}"},
    }


def _key(seed: int) -> ExactActionKey:
    return ExactActionKey.build(
        checkpoint_id="model@commit",
        base_model_manifest=b"base",
        adapter_manifest=b"adapter",
        tokenizer_manifest=b"tokenizer",
        renderer_manifest=b"renderer",
        sampler_conformance_manifest=_conformance(),
        action_selection_policy="direct_single_sample",
        transport_retry_policy="fail_before_action_no_resample",
        request=_request(seed),
        prompt_token_ids=(10, 11),
        render_prompt=lambda _: (10, 11),
    )


def _materialize(key: ExactActionKey) -> BehaviorAction:
    return BehaviorAction.build(
        key=key,
        action_token_ids=(20, 2),
        behavior_logprobs=(-0.2, -0.1),
        raw_transport_message={"role": "assistant", "content": "duplicate"},
        finish_reason="stop",
        prompt_tokens=2,
        completion_tokens=2,
        termination_kind="eos",
        eos_token_id=2,
        encode_action=lambda _request, _message: (20, 2),
    )


def _action(seed: int) -> BehaviorAction:
    return _materialize(_key(seed))


def _create(root: Path, *, fault_hook: Any = None) -> StageDReceiptLedger:
    return StageDReceiptLedger.create(
        root,
        binding=_binding(),
        master_seed=MASTER_SEED,
        fault_hook=fault_hook,
    )


def _commit(
    writer: StageDReceiptLedger,
    *,
    branch_count: int = 2,
) -> tuple[BehaviorAction, PolicyEventAddress, bytes, bytes]:
    recorded_key = _key(17)
    target = PolicyEventAddress(1, "root/child", 0, 0)
    snapshot = writer.put_evidence(b"snapshot")
    reservation = writer.commit_pre_action_and_reserve(
        group_id="group-1",
        rollout_id="rollout-1",
        target_roster=("target-0",),
        target_ordinal=0,
        target_id="target-0",
        target_address=target,
        pre_action_snapshot_sha256=snapshot,
        recorded_action_key=recorded_key,
        branch_count=branch_count,
        continuation_replicates=1,
        failure_reward=-1.0,
    )
    request = writer.put_evidence(recorded_key.request)
    writer.mark_recorded_action_model_call_started(
        reservation,
        request_sha256=request,
    )
    recorded = _materialize(recorded_key)
    response = writer.put_evidence(recorded.to_bytes())
    writer.complete_recorded_action(
        reservation,
        action=recorded,
        response_sha256=response,
    )
    correspondence_evidence = writer.put_evidence(b"correspondence")
    matched = PolicyEventAddress(0, "root", 2, 2)
    correspondence_receipt = writer.freeze_correspondence(
        group_id="group-1",
        target_id="target-0",
        recorded_action=recorded,
        matched_addresses=(matched,),
        evidence_sha256=correspondence_evidence,
    )
    return recorded, matched, reservation.commitment_receipt, correspondence_receipt


def _complete_scientific_artifact(
    writer: StageDReceiptLedger,
) -> tuple[BranchGroupArtifact, bytes]:
    recorded, matched, commitment_receipt, correspondence_receipt = _commit(writer)
    commitment = PreActionTargetCommitment.from_receipt(
        commitment_receipt,
        verifier=writer,
    )
    correspondence = SeedCorrespondenceMap.from_receipt(
        correspondence_receipt,
        verifier=writer,
        commitment=commitment,
        recorded_action=recorded,
    )
    spec = BranchGroupSpec(commitment, recorded, correspondence, MASTER_SEED)

    def qa(_: BranchGroupSpec) -> bytes:
        evidence = writer.put_evidence(b"qa-report")
        return writer.record_reconstruction_qa(
            group_id="group-1",
            target_id="target-0",
            recorded_action=recorded,
            passed=True,
            report_sha256=evidence,
            actual_cost=ActualEvaluationCost(cpu_seconds=0.1, wall_seconds=0.1),
        )

    def sample(
        *,
        action_slot: int,
        action_seed: int,
        reference_key: ExactActionKey,
    ) -> CandidateSubmission:
        assert reference_key == recorded.key
        attempt = writer.begin_candidate_attempt(
            group_id="group-1",
            target_id="target-0",
            action_slot=action_slot,
        )
        request = writer.put_evidence(f"request-{action_slot}".encode())
        writer.mark_candidate_model_call_started(attempt, request_sha256=request)
        action = _action(action_seed)
        response = writer.put_evidence(action.to_bytes())
        receipt = writer.complete_candidate_call(
            attempt,
            action=action,
            response_sha256=response,
        )
        return CandidateSubmission(action, receipt)

    def execute(
        *,
        arm_id: str,
        action: BehaviorAction,
        continuation_replicate: int,
        seed_oracle: BranchSeedOracle,
    ) -> bytes:
        attempt = writer.begin_execution(
            group_id="group-1",
            target_id="target-0",
            arm_id=arm_id,
            action=action,
            continuation_replicate=continuation_replicate,
        )
        context = writer.put_evidence(f"execution-context-{arm_id}".encode())
        writer.bind_execution_context(attempt, context_sha256=context)
        request = writer.put_evidence(f"execution-request-{arm_id}".encode())
        call = writer.mark_execution_model_call_started(
            attempt,
            address=matched,
            scheduled_seed=seed_oracle.seed_for(matched),
            request_sha256=request,
        )
        response = writer.put_evidence(f"execution-response-{arm_id}".encode())
        writer.complete_execution_model_call(
            attempt,
            call,
            prompt_tokens=3,
            completion_tokens=2,
            response_sha256=response,
        )
        score = writer.put_evidence(f"score-{arm_id}".encode())
        return writer.finish_execution(
            attempt,
            outcome_kind=OutcomeKind.SUCCESS,
            scored_reward=1.0 if arm_id == "arm-0" else 0.0,
            scorer_evidence_sha256=score,
            latency_seconds=0.1,
            dollars=0.0,
            judge_calls=0,
            cpu_seconds=0.1,
            gpu_seconds=0.0,
            wall_seconds=0.1,
            storage_bytes=0,
        )

    artifact = run_scientific_branch_group(
        spec,
        verifier=writer,
        sample_candidate=sample,
        run_reconstruction_qa=qa,
        execute_arm=execute,
    )
    return artifact, artifact.to_bytes()


def test_golden_chain_seals_and_reloads_full_c1_artifact(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    artifact, encoded = _complete_scientific_artifact(writer)
    artifact_digest = writer.put_evidence(encoded)
    writer.claim_training_batch(
        training_batch_identity=artifact.training_batch_identity,
        artifact_sha256s=(artifact_digest,),
        consumer_id="prime-bridge-1",
    )
    seal = writer.seal()

    scan = inspect_ledger(root)
    assert scan.status == "sealed-valid"
    assert scan.seal == seal
    verifier = SealedReceiptVerifier(root, seal)
    loaded = BranchGroupArtifact.verify_bytes(
        encoded,
        verifier=verifier,
        encode_action=lambda _request, _message: (20, 2),
        render_prompt=lambda _: (10, 11),
        master_seed=MASTER_SEED,
    )
    assert loaded == artifact


def test_wrong_out_of_band_seal_and_append_after_seal_fail(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    seal = writer.seal()
    wrong = type(seal)(
        seal.ledger_id,
        seal.genesis_sha256,
        "f" * 64,
        seal.record_count,
        seal.receipt_count,
    )
    with pytest.raises(LedgerError, match="out-of-band"):
        SealedReceiptVerifier(root, wrong)
    with pytest.raises(LedgerError, match="active-clean"):
        StageDReceiptLedger(root, master_seed=MASTER_SEED)


def test_only_one_exclusive_writer_can_exist(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    first = _create(root)
    with pytest.raises(LedgerError, match="exclusive writer"):
        StageDReceiptLedger(root, master_seed=MASTER_SEED)
    first.close()
    reopened = StageDReceiptLedger(root, master_seed=MASTER_SEED)
    reopened.close()


def test_recorded_action_output_can_only_materialize_after_durable_commit(
    tmp_path: Path,
) -> None:
    parameters = inspect.signature(
        StageDReceiptLedger.commit_pre_action_and_reserve
    ).parameters
    assert "recorded_action" not in parameters
    assert "recorded_action_key" in parameters

    root = tmp_path / "ledger"
    writer = _create(root)
    key = _key(17)
    snapshot = writer.put_evidence(b"snapshot")
    reservation = writer.commit_pre_action_and_reserve(
        group_id="group-1",
        rollout_id="rollout-1",
        target_roster=("target-0",),
        target_ordinal=0,
        target_id="target-0",
        target_address=PolicyEventAddress(1, "root/child", 0, 0),
        pre_action_snapshot_sha256=snapshot,
        recorded_action_key=key,
        branch_count=2,
        continuation_replicates=1,
        failure_reward=-1.0,
    )
    commitment = PreActionTargetCommitment.from_receipt(
        reservation.commitment_receipt,
        verifier=writer,
    )
    assert commitment.commitment_sequence < commitment.action_reservation_sequence
    assert writer.record_count == commitment.action_reservation_sequence + 1

    writer.close()
    writer = StageDReceiptLedger(root, master_seed=MASTER_SEED)
    reservation = writer.resume_recorded_action_reservation(
        group_id="group-1",
        target_id="target-0",
    )
    request = writer.put_evidence(key.request)
    writer.mark_recorded_action_model_call_started(
        reservation,
        request_sha256=request,
    )
    records_before_output = writer.record_count
    action = _materialize(key)
    assert records_before_output >= 4
    response = writer.put_evidence(action.to_bytes())
    writer.complete_recorded_action(
        reservation,
        action=action,
        response_sha256=response,
    )
    writer.close()
    assert inspect_ledger(root).status == "active-clean"


def test_os_lock_is_released_by_hard_process_exit(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    writer.close()
    script = (
        "import os,sys; from pathlib import Path; "
        "from redco.analysis.stage_d_receipt_ledger import StageDReceiptLedger; "
        f"StageDReceiptLedger(Path(sys.argv[1]), master_seed={MASTER_SEED!r}); "
        "os._exit(0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    reopened = StageDReceiptLedger(root, master_seed=MASTER_SEED)
    reopened.close()


def test_os_lock_rejects_a_second_process(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    script = (
        "import sys; from pathlib import Path; "
        "from redco.analysis.stage_d_receipt_ledger import LedgerError,StageDReceiptLedger; "
        "\ntry:\n"
        f" StageDReceiptLedger(Path(sys.argv[1]), master_seed={MASTER_SEED!r})\n"
        "except LedgerError:\n sys.exit(7)\n"
        "sys.exit(0)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 7, completed.stderr
    writer.close()


def test_concurrent_commit_transactions_serialize_without_poisoning(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    snapshot = writer.put_evidence(b"snapshot")
    action_key = _key(17)

    def commit(index: int) -> object:
        return writer.commit_pre_action_and_reserve(
            group_id=f"group-{index}",
            rollout_id=f"rollout-{index}",
            target_roster=(f"target-{index}",),
            target_ordinal=0,
            target_id=f"target-{index}",
            target_address=PolicyEventAddress(1, f"root/child-{index}", 0, 0),
            pre_action_snapshot_sha256=snapshot,
            recorded_action_key=action_key,
            branch_count=2,
            continuation_replicates=1,
            failure_reward=-1.0,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(commit, (1, 2)))
    assert len(receipts) == 2
    writer.close()
    assert inspect_ledger(root).status == "active-clean"


def test_seal_and_write_race_is_serial_and_never_poisons(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)

    def seal() -> str:
        writer.seal()
        return "sealed"

    def write() -> str:
        try:
            writer.put_evidence(b"racing evidence")
        except LedgerError:
            return "closed"
        return "written"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = [future.result() for future in (pool.submit(seal), pool.submit(write))]
    assert "sealed" in outcomes
    assert set(outcomes) <= {"sealed", "closed", "written"}
    assert inspect_ledger(root).status == "sealed-valid"


def test_durability_publish_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported(_source: Path, _destination: Path) -> None:
        raise OSError("durable publish unsupported")

    monkeypatch.setattr(receipt_ledger_module, "_durable_rename", unsupported)
    with pytest.raises(OSError, match="unsupported"):
        _create(tmp_path / "ledger")


@pytest.mark.parametrize("operation", ["mutate", "delete", "reorder", "truncate"])
def test_chain_corruption_is_poisoned(tmp_path: Path, operation: str) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    _commit(writer)
    writer.close()
    records = sorted((root / "records").iterdir())
    if operation == "mutate":
        value = json.loads(records[1].read_bytes())
        value["body"]["receipt"]["branch_count"] = 9
        records[1].write_bytes(canonical_json(value))
    elif operation == "delete":
        records[1].unlink()
    elif operation == "reorder":
        first = records[1].read_bytes()
        second = records[2].read_bytes()
        records[1].write_bytes(second)
        records[2].write_bytes(first)
    else:
        records[-1].write_bytes(records[-1].read_bytes()[:20])
    assert inspect_ledger(root).status == "poisoned"


def test_cross_ledger_record_splice_is_poisoned(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = _create(first_root)
    second = _create(second_root)
    _commit(first)
    _commit(second)
    first.close()
    second.close()
    shutil.copyfile(
        second_root / "records" / "00000000000000000001.json",
        first_root / "records" / "00000000000000000001.json",
    )
    assert inspect_ledger(first_root).status == "poisoned"


def test_unknown_temp_or_gap_is_poisoned(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    writer.close()
    (root / "records" / ".unknown.tmp").write_text("partial", encoding="utf-8")
    assert inspect_ledger(root).status == "poisoned"


def test_commitment_crash_before_reservation_is_unresumable(tmp_path: Path) -> None:
    def fail(stage: str, name: str) -> None:
        if stage == "after_directory_fsync" and name == "00000000000000000001.json":
            raise RuntimeError("crash after commitment")

    root = tmp_path / "ledger"
    writer = _create(root, fault_hook=fail)
    recorded_key = _key(17)
    snapshot = writer.put_evidence(b"snapshot")
    with pytest.raises(RuntimeError, match="crash"):
        writer.commit_pre_action_and_reserve(
            group_id="group-1",
            rollout_id="rollout-1",
            target_roster=("target-0",),
            target_ordinal=0,
            target_id="target-0",
            target_address=PolicyEventAddress(1, "root/child", 0, 0),
            pre_action_snapshot_sha256=snapshot,
            recorded_action_key=recorded_key,
            branch_count=2,
            continuation_replicates=1,
            failure_reward=-1.0,
        )
    writer.close()
    assert inspect_ledger(root).status == "poisoned"


def test_crash_before_rename_leaves_fail_closed_temp(tmp_path: Path) -> None:
    def fail(stage: str, name: str) -> None:
        if stage == "after_file_fsync" and name == "00000000000000000001.json":
            raise RuntimeError("crash before rename")

    root = tmp_path / "ledger"
    writer = _create(root, fault_hook=fail)
    recorded_key = _key(17)
    snapshot = writer.put_evidence(b"snapshot")
    with pytest.raises(RuntimeError, match="crash"):
        writer.commit_pre_action_and_reserve(
            group_id="group-1",
            rollout_id="rollout-1",
            target_roster=("target-0",),
            target_ordinal=0,
            target_id="target-0",
            target_address=PolicyEventAddress(1, "root/child", 0, 0),
            pre_action_snapshot_sha256=snapshot,
            recorded_action_key=recorded_key,
            branch_count=2,
            continuation_replicates=1,
            failure_reward=-1.0,
        )
    writer.close()
    assert inspect_ledger(root).status == "poisoned"


def test_dangling_model_call_start_poisons_recovery_and_blocks_zero_call(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    recorded, _, _, _ = _commit(writer)
    qa = writer.put_evidence(b"qa")
    writer.record_reconstruction_qa(
        group_id="group-1",
        target_id="target-0",
        recorded_action=recorded,
        passed=True,
        report_sha256=qa,
        actual_cost=ActualEvaluationCost(),
    )
    attempt = writer.begin_candidate_attempt(
        group_id="group-1",
        target_id="target-0",
        action_slot=1,
    )
    request = writer.put_evidence(b"request")
    writer.mark_candidate_model_call_started(attempt, request_sha256=request)
    supervisor = writer.put_evidence(b"supervisor")
    with pytest.raises(LedgerError, match="impossible"):
        writer.record_zero_call_candidate_failure(
            attempt,
            reason="transport disappeared",
            supervisor_evidence_sha256=supervisor,
        )
    writer.close()
    assert inspect_ledger(root).status == "poisoned"


def test_zero_call_receipt_is_minted_only_before_start(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    recorded, _, commitment, _ = _commit(writer)
    qa = writer.put_evidence(b"qa")
    writer.record_reconstruction_qa(
        group_id="group-1",
        target_id="target-0",
        recorded_action=recorded,
        passed=True,
        report_sha256=qa,
        actual_cost=ActualEvaluationCost(),
    )
    attempt = writer.begin_candidate_attempt(
        group_id="group-1",
        target_id="target-0",
        action_slot=1,
    )
    supervisor = writer.put_evidence(b"supervisor")
    receipt = writer.record_zero_call_candidate_failure(
        attempt,
        reason="capacity vanished",
        supervisor_evidence_sha256=supervisor,
    )
    value = writer(receipt, receipt_kind="zero_call_infrastructure_failure")
    assert value["scientific_model_calls"] == 0
    assert value["ledger_offset"] > json.loads(commitment)["ledger_offset"]
    writer.close()
    reopened = StageDReceiptLedger(root, master_seed=MASTER_SEED)
    with pytest.raises(LedgerError, match="already"):
        reopened.begin_candidate_attempt(
            group_id="group-1",
            target_id="target-0",
            action_slot=1,
        )
    reopened.seal()


def test_duplicate_candidate_slot_and_execution_replicate_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    artifact, _ = _complete_scientific_artifact(writer)
    with pytest.raises(LedgerError, match="already"):
        writer.begin_candidate_attempt(
            group_id="group-1",
            target_id="target-0",
            action_slot=1,
        )
    with pytest.raises(LedgerError, match="already"):
        writer.begin_execution(
            group_id="group-1",
            target_id="target-0",
            arm_id="arm-0",
            action=artifact.recorded_action,
            continuation_replicate=1,
        )
    writer.seal()


def test_execution_cannot_dispatch_or_finish_without_a_frozen_context(
    tmp_path: Path,
) -> None:
    writer = _create(tmp_path / "ledger")
    recorded, matched, _, _ = _commit(writer)
    qa_evidence = writer.put_evidence(b"qa")
    writer.record_reconstruction_qa(
        group_id="group-1",
        target_id="target-0",
        recorded_action=recorded,
        passed=True,
        report_sha256=qa_evidence,
        actual_cost=ActualEvaluationCost(cpu_seconds=0.1, wall_seconds=0.1),
    )
    attempt = writer.begin_execution(
        group_id="group-1",
        target_id="target-0",
        arm_id="arm-0",
        action=recorded,
        continuation_replicate=1,
    )
    request = writer.put_evidence(b"request")
    scheduled = EventSeedScheduler(
        MASTER_SEED,
        "rollout-1",
        "target-0",
        1,
    ).paired_continuation_seed(
        matched,
        committed_address=matched,
    )
    with pytest.raises(LedgerError, match="context must be bound"):
        writer.mark_execution_model_call_started(
            attempt,
            address=matched,
            scheduled_seed=scheduled,
            request_sha256=request,
        )
    score = writer.put_evidence(b"score")
    with pytest.raises(LedgerError, match="context must be bound"):
        writer.finish_execution(
            attempt,
            outcome_kind=OutcomeKind.RUNTIME_EXCEPTION,
            scored_reward=0.0,
            scorer_evidence_sha256=score,
            latency_seconds=0.0,
            dollars=0.0,
            judge_calls=0,
            cpu_seconds=0.0,
            gpu_seconds=0.0,
            wall_seconds=0.0,
            storage_bytes=0,
        )
    writer.close()


def test_concurrent_single_use_batch_claim_has_one_winner(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    evidence = writer.put_evidence(b"artifact")
    identity = "b" * 64

    def claim(consumer: str) -> str:
        try:
            writer.claim_training_batch(
                training_batch_identity=identity,
                artifact_sha256s=(evidence,),
                consumer_id=consumer,
            )
        except BatchAlreadyClaimed:
            return "lost"
        return "won"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(claim, ("one", "two")))
    assert sorted(outcomes) == ["lost", "won"]
    writer.seal()


def test_batch_claim_remains_consumed_after_reopen(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    evidence = writer.put_evidence(b"artifact")
    writer.claim_training_batch(
        training_batch_identity="c" * 64,
        artifact_sha256s=(evidence,),
        consumer_id="one",
    )
    writer.close()
    reopened = StageDReceiptLedger(root, master_seed=MASTER_SEED)
    with pytest.raises(BatchAlreadyClaimed):
        reopened.claim_training_batch(
            training_batch_identity="c" * 64,
            artifact_sha256s=(evidence,),
            consumer_id="two",
        )
    reopened.seal()
