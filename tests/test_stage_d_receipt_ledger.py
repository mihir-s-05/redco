from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import pytest

import redco.analysis.stage_d_receipt_ledger as receipt_ledger_module
from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.analysis.stage_d_ledger_validation import validate_state_machine
from redco.analysis.stage_d_receipt_ledger import (
    BatchAlreadyClaimed,
    GenesisBinding,
    LedgerError,
    LedgerPoisoned,
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
from redco.analysis.stage_d_three_arm_bridge import DecisionProvenance
from redco.analysis.stage_d_zero_call_recovery import (
    _install_or_verify_archive,
    recover_or_open_scientific_ledger,
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
        protocol_manifest_sha256="5" * 64,
        master_seed_sha256=_sha256(MASTER_SEED.encode()),
        support_rules_sha256="6" * 64,
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


def _bind_source_rollout(
    writer: StageDReceiptLedger,
    completion: bytes,
) -> bytes:
    trace = writer.put_evidence(b"raw-trace")
    reward = writer.put_evidence(b"reward-evidence")
    stock = writer.put_evidence(b"stock-sequences")
    result = writer.record_source_rollout_completed(
        group_id="group-1",
        rollout_id="rollout-1",
        source_sha256="5" * 64,
        trace_sha256=trace,
        reward_evidence_sha256=reward,
        stock_sequences_evidence_sha256=stock,
        base_model_manifest_sha256="6" * 64,
        decision_ids=("root-turn-0",),
        decision_completion_receipt_sha256s=(_sha256(completion),),
    )
    return result.receipt


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
        writer.mark_candidate_response_observed(attempt, response_sha256=response)
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
        writer.mark_execution_dispatched(attempt)
        request = writer.put_evidence(f"execution-request-{arm_id}".encode())
        call = writer.mark_execution_model_call_started(
            attempt,
            address=matched,
            scheduled_seed=seed_oracle.seed_for(matched),
            request_sha256=request,
        )
        response = writer.put_evidence(f"execution-response-{arm_id}".encode())
        writer.mark_execution_response_observed(
            attempt,
            call,
            response_sha256=response,
        )
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


def test_source_policy_receipts_roundtrip_reopen_seal_and_verify(tmp_path: Path) -> None:
    root = tmp_path / "source-policy"
    writer = _create(root)
    key = _key(17)
    request = writer.put_evidence(key.request)
    reservation = writer.reserve_source_policy_call(
        group_id="group-1",
        rollout_id="rollout-1",
        decision_id="root-turn-0",
        node_kind="root",
        target_id=None,
        target_ordinal=None,
        target_address=PolicyEventAddress(0, "root", 0, 0),
        recorded_action_key=key,
        request_sha256=request,
        branch_selected=False,
        raw_response_required=True,
    )
    action = _materialize(key)
    raw_response = writer.put_evidence(b"exact-provider-response")
    writer.mark_source_policy_response_observed(
        reservation,
        response_sha256=raw_response,
    )
    response = writer.put_evidence(action.to_bytes())
    completion = writer.complete_source_policy_call(
        reservation,
        action=action,
        response_sha256=response,
    )
    _bind_source_rollout(writer, completion)
    writer.close()

    records = [
        json.loads(path.read_bytes())
        for path in sorted((root / "records").glob("*.json"))
    ]
    validate_state_machine(records)
    tampered = json.loads(json.dumps(records))
    witness_record = next(
        record
        for record in tampered
        if record["record_kind"] == "receipt"
        and record["body"]["receipt"]["receipt_kind"]
        == "source_policy_response_observed"
    )
    witness_record["body"]["receipt"]["raw_response_sha256"] = "f" * 64
    witness_record["body"]["evidence_refs"] = ["f" * 64]
    with pytest.raises(LedgerPoisoned, match="source policy completion is invalid"):
        validate_state_machine(tampered)

    reopened = StageDReceiptLedger(root, master_seed=MASTER_SEED)
    seal = reopened.seal()
    verifier = SealedReceiptVerifier(root, seal)
    provenance = DecisionProvenance.from_receipts(
        reservation.receipt,
        completion,
        verifier=verifier,
    )
    assert provenance.ledger_id == seal.ledger_id
    assert provenance.request_sequence < provenance.completion_sequence
    assert provenance.exact_action_key_digest == key.digest
    assert provenance.action_digest == action.digest


def test_source_policy_raw_response_is_required_exactly_once_and_survives_reopen(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source-raw-response"
    writer = _create(root)
    key = _key(17)
    request = writer.put_evidence(key.request)
    reservation = writer.reserve_source_policy_call(
        group_id="group-1",
        rollout_id="rollout-1",
        decision_id="root-turn-0",
        node_kind="root",
        target_id=None,
        target_ordinal=None,
        target_address=PolicyEventAddress(0, "root", 0, 0),
        recorded_action_key=key,
        request_sha256=request,
        branch_selected=False,
        raw_response_required=True,
    )
    action = _materialize(key)
    action_evidence = writer.put_evidence(action.to_bytes())
    with pytest.raises(LedgerError, match="lacks its durable raw response witness"):
        writer.complete_source_policy_call(
            reservation,
            action=action,
            response_sha256=action_evidence,
        )
    raw_response = writer.put_evidence(b"exact-provider-response")
    witness = writer.mark_source_policy_response_observed(
        reservation,
        response_sha256=raw_response,
    )
    assert json.loads(witness)["receipt_kind"] == "source_policy_response_observed"
    with pytest.raises(LedgerError, match="observed twice"):
        writer.mark_source_policy_response_observed(
            reservation,
            response_sha256=raw_response,
        )
    writer.close()

    with pytest.raises(LedgerError, match="requires active-clean"):
        StageDReceiptLedger(root, master_seed=MASTER_SEED)
    witness_receipts = [
        json.loads(path.read_bytes())["body"]["receipt"]
        for path in sorted((root / "records").glob("*.json"))
        if json.loads(path.read_bytes())["record_kind"] == "receipt"
    ]
    assert witness_receipts[-1]["raw_response_sha256"] == raw_response


def test_selected_source_policy_call_requires_same_ledger_commitment(
    tmp_path: Path,
) -> None:
    writer = _create(tmp_path / "selected")
    key = _key(17)
    request = writer.put_evidence(key.request)
    with pytest.raises(LedgerError, match="lacks pre-action commitment"):
        writer.reserve_source_policy_call(
            group_id="group-1",
            rollout_id="rollout-1",
            decision_id="child-0",
            node_kind="child",
            target_id="target-0",
            target_ordinal=0,
            target_address=PolicyEventAddress(1, "root/child", 0, 0),
            recorded_action_key=key,
            request_sha256=request,
            branch_selected=True,
        )
    writer.close()


def test_source_policy_call_rejects_dangling_duplicate_and_mismatched_evidence(
    tmp_path: Path,
) -> None:
    writer = _create(tmp_path / "source-negative")
    key = _key(17)
    wrong_request = writer.put_evidence(b"wrong-request")
    with pytest.raises(ValueError, match="request evidence"):
        writer.reserve_source_policy_call(
            group_id="group-1",
            rollout_id="rollout-1",
            decision_id="root-turn-0",
            node_kind="root",
            target_id=None,
            target_ordinal=None,
            target_address=PolicyEventAddress(0, "root", 0, 0),
            recorded_action_key=key,
            request_sha256=wrong_request,
            branch_selected=False,
        )

    request = writer.put_evidence(key.request)
    reservation = writer.reserve_source_policy_call(
        group_id="group-1",
        rollout_id="rollout-1",
        decision_id="root-turn-0",
        node_kind="root",
        target_id=None,
        target_ordinal=None,
        target_address=PolicyEventAddress(0, "root", 0, 0),
        recorded_action_key=key,
        request_sha256=request,
        branch_selected=False,
    )
    with pytest.raises(LedgerError, match="already reserved"):
        writer.reserve_source_policy_call(
            group_id="group-1",
            rollout_id="rollout-1",
            decision_id="root-turn-0",
            node_kind="root",
            target_id=None,
            target_ordinal=None,
            target_address=PolicyEventAddress(0, "root", 0, 0),
            recorded_action_key=key,
            request_sha256=request,
            branch_selected=False,
        )
    wrong_action = _action(18)
    wrong_response = writer.put_evidence(wrong_action.to_bytes())
    with pytest.raises(LedgerError, match="differs from its reservation"):
        writer.complete_source_policy_call(
            reservation,
            action=wrong_action,
            response_sha256=wrong_response,
        )
    with pytest.raises(LedgerPoisoned, match="dangling"):
        writer.seal()
    writer.close()


def test_concurrent_source_policy_calls_complete_out_of_order_but_recovery_is_strict(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source-concurrent"
    writer = _create(root)
    reservations = []
    actions = []
    for ordinal, seed in enumerate((17, 18)):
        key = _key(seed)
        request = writer.put_evidence(key.request)
        reservations.append(
            writer.reserve_source_policy_call(
                group_id="group-1",
                rollout_id="rollout-1",
                decision_id=f"child-{ordinal}",
                node_kind="child",
                target_id=f"target-{ordinal}",
                target_ordinal=ordinal,
                target_address=PolicyEventAddress(
                    1, f"root/child-{ordinal}", 0, 0
                ),
                recorded_action_key=key,
                request_sha256=request,
                branch_selected=False,
            )
        )
        actions.append(_materialize(key))

    assert inspect_ledger(root).status == "poisoned"
    completions: dict[int, bytes] = {}
    for ordinal in (1, 0):
        response = writer.put_evidence(actions[ordinal].to_bytes())
        completions[ordinal] = writer.complete_source_policy_call(
            reservations[ordinal],
            action=actions[ordinal],
            response_sha256=response,
        )

    trace = writer.put_evidence(b"raw-trace-concurrent")
    reward = writer.put_evidence(b"reward-evidence-concurrent")
    stock = writer.put_evidence(b"stock-sequences-concurrent")
    writer.record_source_rollout_completed(
        group_id="group-1",
        rollout_id="rollout-1",
        source_sha256="5" * 64,
        trace_sha256=trace,
        reward_evidence_sha256=reward,
        stock_sequences_evidence_sha256=stock,
        base_model_manifest_sha256="6" * 64,
        decision_ids=("child-0", "child-1"),
        decision_completion_receipt_sha256s=(
            _sha256(completions[0]),
            _sha256(completions[1]),
        ),
    )
    assert writer.seal().record_count > 1


def test_aborted_source_policy_call_is_durable_and_terminal(tmp_path: Path) -> None:
    root = tmp_path / "source-aborted"
    writer = _create(root)
    key = _key(17)
    request = writer.put_evidence(key.request)
    reservation = writer.reserve_source_policy_call(
        group_id="group-1",
        rollout_id="rollout-1",
        decision_id="root-turn-0",
        node_kind="root",
        target_id=None,
        target_ordinal=None,
        target_address=PolicyEventAddress(0, "root", 0, 0),
        recorded_action_key=key,
        request_sha256=request,
        branch_selected=False,
    )
    error = writer.put_evidence(b"transport outcome unknown")
    receipt = writer.abort_source_policy_call(
        reservation,
        phase="post_unknown",
        error_sha256=error,
    )
    assert json.loads(receipt)["receipt_kind"] == "source_policy_call_aborted"
    assert inspect_ledger(root).reason == "ledger records an aborted source policy call"
    with pytest.raises(LedgerPoisoned):
        writer.seal()
    writer.close()
    with pytest.raises(LedgerError, match="requires active-clean"):
        StageDReceiptLedger(root, master_seed=MASTER_SEED)


def test_source_policy_receipts_reject_cross_ledger_and_tampering(tmp_path: Path) -> None:
    first = _create(tmp_path / "first")
    second = _create(tmp_path / "second")
    key = _key(17)
    request = first.put_evidence(key.request)
    reservation = first.reserve_source_policy_call(
        group_id="group-1",
        rollout_id="rollout-1",
        decision_id="root-turn-0",
        node_kind="root",
        target_id=None,
        target_ordinal=None,
        target_address=PolicyEventAddress(0, "root", 0, 0),
        recorded_action_key=key,
        request_sha256=request,
        branch_selected=False,
    )
    action = _materialize(key)
    other_response = second.put_evidence(action.to_bytes())
    with pytest.raises(LedgerError, match="not pending"):
        second.complete_source_policy_call(
            reservation,
            action=action,
            response_sha256=other_response,
        )
    response = first.put_evidence(action.to_bytes())
    completion = first.complete_source_policy_call(
        reservation,
        action=action,
        response_sha256=response,
    )
    _bind_source_rollout(first, completion)
    seal = first.seal()
    verifier = SealedReceiptVerifier(tmp_path / "first", seal)
    tampered = json.loads(completion)
    tampered["action_digest"] = "f" * 64
    tampered_bytes = canonical_json(tampered)
    with pytest.raises(ValueError, match="not anchored"):
        DecisionProvenance.from_receipts(
            reservation.receipt,
            tampered_bytes,
            verifier=verifier,
        )
    second.close()


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
    assert value["attempt_model_calls"] == 0
    assert value["attempt_overrides"] == 0
    assert value["repair_sequence"] == 0
    assert value["attempt_ordinal"] == 0
    assert value["successor_permitted"] is True
    assert value["ledger_offset"] > json.loads(commitment)["ledger_offset"]
    writer.close()
    reopened = StageDReceiptLedger(root, master_seed=MASTER_SEED)
    repair = reopened.begin_candidate_attempt(
        group_id="group-1",
        target_id="target-0",
        action_slot=1,
    )
    assert repair.attempt_ordinal == 1
    supervisor = reopened.put_evidence(b"second supervisor")
    second_receipt = reopened.record_zero_call_candidate_failure(
        repair,
        reason="successor capacity vanished",
        supervisor_evidence_sha256=supervisor,
    )
    second = reopened(second_receipt, receipt_kind="zero_call_infrastructure_failure")
    assert second["attempt_ordinal"] == 1
    assert second["successor_permitted"] is False
    with pytest.raises(LedgerError, match="exhausted"):
        reopened.begin_candidate_attempt(
            group_id="group-1",
            target_id="target-0",
            action_slot=1,
        )
    reopened.seal()


def test_zero_call_execution_failure_resumes_once_without_scientific_activity(
    tmp_path: Path,
) -> None:
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
    attempt = writer.begin_execution(
        group_id="group-1",
        target_id="target-0",
        arm_id="arm-0",
        action=recorded,
        continuation_replicate=1,
    )
    context = writer.put_evidence(b"dispatch context")
    writer.bind_execution_context(attempt, context_sha256=context)
    writer.mark_execution_dispatched(attempt)
    supervisor = writer.put_evidence(b"docker unavailable before policy activity")
    receipt = writer.record_zero_call_execution_failure(
        attempt,
        reason="docker unavailable",
        supervisor_evidence_sha256=supervisor,
    )
    value = writer(receipt, receipt_kind="zero_call_execution_failure")
    assert value["attempt_ordinal"] == 0
    assert value["attempt_model_calls"] == 0
    assert value["attempt_overrides"] == 0
    assert value["repair_sequence"] == 0
    assert value["successor_permitted"] is True
    writer.close()

    reopened = StageDReceiptLedger(root, master_seed=MASTER_SEED)
    repair = reopened.begin_execution(
        group_id="group-1",
        target_id="target-0",
        arm_id="arm-0",
        action=recorded,
        continuation_replicate=1,
    )
    assert repair.attempt_ordinal == 1
    context = reopened.put_evidence(b"successor dispatch context")
    reopened.bind_execution_context(repair, context_sha256=context)
    reopened.mark_execution_dispatched(repair)
    score = reopened.put_evidence(b"terminal score")
    reopened.finish_execution(
        repair,
        outcome_kind=OutcomeKind.TERMINAL_WITHOUT_DOWNSTREAM,
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
    reopened.seal()


def test_reopen_recovers_hard_exit_before_candidate_model_call(tmp_path: Path) -> None:
    root = tmp_path / "candidate-hard-exit"
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
    writer.begin_candidate_attempt(
        group_id="group-1",
        target_id="target-0",
        action_slot=1,
    )
    writer.close()

    scan = inspect_ledger(root, allow_repairable_zero_call=True)
    assert scan.status == "active-repairable-zero-call"
    assert scan.repairable_attempt is not None
    recovered = StageDReceiptLedger.recover_zero_call_failure(
        root,
        master_seed=MASTER_SEED,
        reason="worker hard-exited before POST",
        supervisor_evidence=b"verified dead worker",
    )
    successor = recovered.begin_candidate_attempt(
        group_id="group-1",
        target_id="target-0",
        action_slot=1,
    )
    assert successor.attempt_ordinal == 1
    recovered.close()


@pytest.mark.parametrize(
    ("finish", "expected_status"),
    [(False, "poisoned"), (True, "active-clean")],
)
def test_candidate_post_kill_boundary_is_evidenced_or_atomically_complete(
    tmp_path: Path,
    finish: bool,
    expected_status: str,
) -> None:
    root = tmp_path / f"candidate-post-kill-{finish}"
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
    writer.close()
    script = """
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from test_stage_d_receipt_ledger import MASTER_SEED, _action
from redco.analysis.stage_d_receipt_ledger import StageDReceiptLedger
w = StageDReceiptLedger(Path(sys.argv[1]), master_seed=MASTER_SEED)
a = w.begin_candidate_attempt(group_id='group-1', target_id='target-0', action_slot=1)
r = w.put_evidence(b'candidate-request')
w.mark_candidate_model_call_started(a, request_sha256=r)
action = _action(a.action_seed)
response = w.put_evidence(b'exact-provider-response')
w.mark_candidate_response_observed(a, response_sha256=response)
if sys.argv[3] == 'finish':
    w.complete_candidate_call(a, action=action, response_sha256=response)
os._exit(0)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(root),
            str(Path(__file__).parent),
            "finish" if finish else "kill",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    scan = inspect_ledger(root)
    assert scan.status == expected_status
    response_records = [
        record
        for path in sorted((root / "records").glob("*.json"))
        if (record := json.loads(path.read_bytes()))["record_kind"]
        == "model_call_response_observed"
    ]
    assert len(response_records) == 1
    response_digest = _sha256(b"exact-provider-response")
    assert response_records[0]["body"]["evidence_refs"] == [response_digest]
    assert (root / "evidence" / response_digest).read_bytes() == b"exact-provider-response"
    if finish:
        reopened = StageDReceiptLedger(root, master_seed=MASTER_SEED)
        recovered = reopened.completed_candidate_evidence(
            group_id="group-1",
            target_id="target-0",
            action_slot=1,
        )
        assert recovered is not None
        reopened.close()


def test_zero_call_archive_and_ledger_recovery_are_crash_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    recorded, _, _, _ = _commit(writer)
    writer.record_reconstruction_qa(
        group_id="group-1",
        target_id="target-0",
        recorded_action=recorded,
        passed=True,
        report_sha256=writer.put_evidence(b"qa"),
        actual_cost=ActualEvaluationCost(),
    )
    writer.begin_candidate_attempt(
        group_id="group-1",
        target_id="target-0",
        action_slot=1,
    )
    writer.close()
    evidence = tmp_path / "supervisor.bin"
    evidence.write_bytes(b"verified dead worker")
    episode_output = tmp_path / "episode-output"
    episode_output.mkdir()
    (episode_output / "partial.json").write_bytes(b"partial")
    archive = tmp_path / "repair-archive"
    scan = inspect_ledger(root, allow_repairable_zero_call=True)
    assert scan.repairable_attempt is not None

    # Simulate a hard exit after the archive transaction but before ledger repair.
    _install_or_verify_archive(
        archive=archive,
        episode_output=episode_output,
        repairable_attempt=dict(scan.repairable_attempt),
        supervisor_evidence_sha256=_sha256(evidence.read_bytes()),
    )
    assert not episode_output.exists()
    recovered = recover_or_open_scientific_ledger(
        ledger_root=root,
        master_seed=MASTER_SEED,
        recover_requested=True,
        supervisor_evidence_path=evidence,
        repair_archive=archive,
        episode_output=episode_output,
    )
    recovered.close()

    # Simulate another hard exit after ledger repair; the same command only verifies.
    reopened = recover_or_open_scientific_ledger(
        ledger_root=root,
        master_seed=MASTER_SEED,
        recover_requested=True,
        supervisor_evidence_path=evidence,
        repair_archive=archive,
        episode_output=episode_output,
    )
    reopened.close()
    final = inspect_ledger(root)
    failures = [
        receipt
        for (kind, _), receipt in final.receipts.items()
        if kind == "zero_call_infrastructure_failure"
    ]
    assert len(failures) == 1
    assert (archive / "episode-output" / "partial.json").read_bytes() == b"partial"


def test_reopen_recovers_hard_exit_after_zero_activity_execution_dispatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "execution-hard-exit"
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
    attempt = writer.begin_execution(
        group_id="group-1",
        target_id="target-0",
        arm_id="arm-0",
        action=recorded,
        continuation_replicate=1,
    )
    context = writer.put_evidence(b"dispatch context")
    writer.bind_execution_context(attempt, context_sha256=context)
    writer.mark_execution_dispatched(attempt)
    writer.close()

    scan = inspect_ledger(root, allow_repairable_zero_call=True)
    assert scan.status == "active-repairable-zero-call"
    assert scan.repairable_attempt is not None
    recovered = StageDReceiptLedger.recover_zero_call_failure(
        root,
        master_seed=MASTER_SEED,
        reason="worker hard-exited before scientific activity",
        supervisor_evidence=b"verified dead worker",
    )
    successor = recovered.begin_execution(
        group_id="group-1",
        target_id="target-0",
        arm_id="arm-0",
        action=recorded,
        continuation_replicate=1,
    )
    assert successor.attempt_ordinal == 1
    recovered.close()


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


def test_execution_ledger_binds_cache_salt_to_structural_seed(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    _complete_scientific_artifact(writer)
    starts = []
    for path in sorted((root / "records").iterdir()):
        record = json.loads(path.read_bytes())
        if record["record_kind"] != "model_call_started":
            continue
        event = record["body"]["event"]
        if event["attempt_kind"] == "execution":
            starts.append(event)
    assert starts
    for event in starts:
        address = PolicyEventAddress(
            event["address"]["depth"],
            event["address"]["lineage"],
            event["address"]["session_call_ordinal"],
            event["address"]["turn"],
            event["address"]["call_kind"],
        )
        scheduled = EventSeedScheduler(
            MASTER_SEED,
            "rollout-1",
            "target-0",
            1,
        ).paired_continuation_seed(address, committed_address=address)
        assert event["seed"] == scheduled.seed
        assert event["cache_salt"] == scheduled.cache_salt
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
    context = writer.put_evidence(b"execution context")
    writer.bind_execution_context(attempt, context_sha256=context)
    with pytest.raises(LedgerError, match="must be dispatched"):
        writer.mark_execution_model_call_started(
            attempt,
            address=matched,
            scheduled_seed=scheduled,
            request_sha256=request,
        )
    with pytest.raises(LedgerError, match="must be dispatched"):
        writer.finish_execution(
            attempt,
            outcome_kind=OutcomeKind.TERMINAL_WITHOUT_DOWNSTREAM,
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
    writer.mark_execution_dispatched(attempt)
    writer.finish_execution(
        attempt,
        outcome_kind=OutcomeKind.TERMINAL_WITHOUT_DOWNSTREAM,
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
    writer.seal()


def test_replay_override_is_committed_before_delivery_and_must_be_acknowledged(
    tmp_path: Path,
) -> None:
    writer = _create(tmp_path / "ledger")
    recorded, _, _, _ = _commit(writer)
    target = PolicyEventAddress(1, "root/child", 0, 0)
    qa = writer.put_evidence(b"qa")
    writer.record_reconstruction_qa(
        group_id="group-1",
        target_id="target-0",
        recorded_action=recorded,
        passed=True,
        report_sha256=qa,
        actual_cost=ActualEvaluationCost(cpu_seconds=0.1, wall_seconds=0.1),
    )
    attempt = writer.begin_execution(
        group_id="group-1",
        target_id="target-0",
        arm_id="arm-0",
        action=recorded,
        continuation_replicate=1,
    )
    context = writer.put_evidence(b"context")
    writer.bind_execution_context(attempt, context_sha256=context)
    writer.mark_execution_dispatched(attempt)
    request = writer.put_evidence(b"prepared request")
    response = writer.put_evidence(b"raw engine response")
    ticket = writer.commit_execution_override(
        attempt,
        address=target,
        action_digest=recorded.digest,
        disposition="inject",
        request_sha256=request,
        response_content_sha256=response,
        prompt_tokens=recorded.prompt_tokens,
        completion_tokens=recorded.completion_tokens,
        counts_toward_logical_cost=False,
    )
    score = writer.put_evidence(b"score")
    with pytest.raises(LedgerError, match="undelivered"):
        writer.finish_execution(
            attempt,
            outcome_kind=OutcomeKind.TERMINAL_WITHOUT_DOWNSTREAM,
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
    typed = writer.put_evidence(b"typed response")
    writer.mark_execution_override_delivered(
        attempt,
        ticket,
        typed_response_sha256=typed,
    )
    with pytest.raises(LedgerError, match="already delivered"):
        writer.mark_execution_override_delivered(
            attempt,
            ticket,
            typed_response_sha256=typed,
        )
    writer.finish_execution(
        attempt,
        outcome_kind=OutcomeKind.TERMINAL_WITHOUT_DOWNSTREAM,
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
    writer.seal()
    assert inspect_ledger(tmp_path / "ledger").status == "sealed-valid"


def test_crash_after_undelivered_override_is_one_time_zero_post_repairable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    recorded, _, _, _ = _commit(writer)
    target = PolicyEventAddress(1, "root/child", 0, 0)
    qa = writer.put_evidence(b"qa")
    writer.record_reconstruction_qa(
        group_id="group-1",
        target_id="target-0",
        recorded_action=recorded,
        passed=True,
        report_sha256=qa,
        actual_cost=ActualEvaluationCost(cpu_seconds=0.1, wall_seconds=0.1),
    )
    attempt = writer.begin_execution(
        group_id="group-1",
        target_id="target-0",
        arm_id="arm-0",
        action=recorded,
        continuation_replicate=1,
    )
    context = writer.put_evidence(b"context")
    writer.bind_execution_context(attempt, context_sha256=context)
    writer.mark_execution_dispatched(attempt)
    ticket = writer.commit_execution_override(
        attempt,
        address=target,
        action_digest=recorded.digest,
        disposition="inject",
        request_sha256=writer.put_evidence(b"request"),
        response_content_sha256=writer.put_evidence(b"response"),
        prompt_tokens=recorded.prompt_tokens,
        completion_tokens=recorded.completion_tokens,
        counts_toward_logical_cost=False,
    )
    writer.close()

    scan = inspect_ledger(root, allow_repairable_zero_call=True)
    assert scan.status == "active-repairable-zero-call"
    recovered = StageDReceiptLedger.recover_zero_call_failure(
        root,
        master_seed=MASTER_SEED,
        reason="worker exited before typed override delivery",
        supervisor_evidence=b"subprocess exit evidence",
    )
    failure = next(
        receipt
        for (kind, _), receipt in inspect_ledger(root).receipts.items()
        if kind == "zero_call_execution_failure"
    )
    assert failure["attempt_overrides"] == 1
    assert failure["discarded_override_ids"] == [ticket.override_id]
    successor = recovered.begin_execution(
        group_id="group-1",
        target_id="target-0",
        arm_id="arm-0",
        action=recorded,
        continuation_replicate=1,
    )
    assert successor.attempt_ordinal == 1


def test_parallel_generated_calls_complete_out_of_order_and_seal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    recorded, first_address, _, _ = _commit(writer)
    qa = writer.put_evidence(b"qa")
    writer.record_reconstruction_qa(
        group_id="group-1",
        target_id="target-0",
        recorded_action=recorded,
        passed=True,
        report_sha256=qa,
        actual_cost=ActualEvaluationCost(cpu_seconds=0.1, wall_seconds=0.1),
    )
    attempt = writer.begin_execution(
        group_id="group-1",
        target_id="target-0",
        arm_id="arm-0",
        action=recorded,
        continuation_replicate=1,
    )
    writer.bind_execution_context(
        attempt,
        context_sha256=writer.put_evidence(b"context"),
    )
    writer.mark_execution_dispatched(attempt)
    second_address = PolicyEventAddress(1, "root/parallel-child", 0, 0)
    scheduler = EventSeedScheduler(MASTER_SEED, "rollout-1", "target-0", 1)
    first = writer.mark_execution_model_call_started(
        attempt,
        address=first_address,
        scheduled_seed=scheduler.exogenous_continuation_seed(
            first_address,
            action_arm="arm-0",
        ),
        request_sha256=writer.put_evidence(b"request:first"),
    )
    second = writer.mark_execution_model_call_started(
        attempt,
        address=second_address,
        scheduled_seed=scheduler.exogenous_continuation_seed(
            second_address,
            action_arm="arm-0",
        ),
        request_sha256=writer.put_evidence(b"request:second"),
    )
    with pytest.raises(LedgerError, match="reuse a scientific event address"):
        writer.mark_execution_model_call_started(
            attempt,
            address=first_address,
            scheduled_seed=scheduler.exogenous_continuation_seed(
                first_address,
                action_arm="arm-0",
            ),
            request_sha256=writer.put_evidence(b"request:duplicate"),
        )
    second_response = writer.put_evidence(b"response:second")
    writer.mark_execution_response_observed(
        attempt,
        second,
        response_sha256=second_response,
    )
    writer.complete_execution_model_call(
        attempt,
        second,
        prompt_tokens=3,
        completion_tokens=2,
        response_sha256=second_response,
    )
    first_response = writer.put_evidence(b"response:first")
    writer.mark_execution_response_observed(
        attempt,
        first,
        response_sha256=first_response,
    )
    writer.complete_execution_model_call(
        attempt,
        first,
        prompt_tokens=4,
        completion_tokens=1,
        response_sha256=first_response,
    )
    writer.finish_execution(
        attempt,
        outcome_kind=OutcomeKind.SUCCESS,
        scored_reward=0.5,
        scorer_evidence_sha256=writer.put_evidence(b"score"),
        latency_seconds=0.1,
        dollars=0.0,
        judge_calls=0,
        cpu_seconds=0.1,
        gpu_seconds=0.0,
        wall_seconds=0.1,
        storage_bytes=0,
    )
    writer.seal()
    assert inspect_ledger(root).status == "sealed-valid"


def test_crash_with_parallel_generated_calls_is_terminal(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    recorded, first_address, _, _ = _commit(writer)
    writer.record_reconstruction_qa(
        group_id="group-1",
        target_id="target-0",
        recorded_action=recorded,
        passed=True,
        report_sha256=writer.put_evidence(b"qa"),
        actual_cost=ActualEvaluationCost(cpu_seconds=0.1, wall_seconds=0.1),
    )
    attempt = writer.begin_execution(
        group_id="group-1",
        target_id="target-0",
        arm_id="arm-0",
        action=recorded,
        continuation_replicate=1,
    )
    writer.bind_execution_context(
        attempt,
        context_sha256=writer.put_evidence(b"context"),
    )
    writer.mark_execution_dispatched(attempt)
    scheduler = EventSeedScheduler(MASTER_SEED, "rollout-1", "target-0", 1)
    for address in (
        first_address,
        PolicyEventAddress(1, "root/parallel-child", 0, 0),
    ):
        writer.mark_execution_model_call_started(
            attempt,
            address=address,
            scheduled_seed=scheduler.exogenous_continuation_seed(
                address,
                action_arm="arm-0",
            ),
            request_sha256=writer.put_evidence(address.lineage.encode()),
        )
    writer.close()

    scan = inspect_ledger(root)
    assert scan.status == "poisoned"
    assert "dangling" in str(scan.reason)


def test_replayed_continuation_counts_logically_but_not_as_generated_cost(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    recorded, _, _, _ = _commit(writer)
    target = PolicyEventAddress(1, "root/child", 0, 0)
    qa = writer.put_evidence(b"qa")
    writer.record_reconstruction_qa(
        group_id="group-1",
        target_id="target-0",
        recorded_action=recorded,
        passed=True,
        report_sha256=qa,
        actual_cost=ActualEvaluationCost(cpu_seconds=0.1, wall_seconds=0.1),
    )
    attempt = writer.begin_execution(
        group_id="group-1",
        target_id="target-0",
        arm_id="arm-0",
        action=recorded,
        continuation_replicate=1,
    )
    writer.bind_execution_context(
        attempt,
        context_sha256=writer.put_evidence(b"context"),
    )
    writer.mark_execution_dispatched(attempt)

    def replay(
        address: PolicyEventAddress,
        disposition: Literal["reuse", "inject"],
        *,
        logical: bool,
    ) -> None:
        ticket = writer.commit_execution_override(
            attempt,
            address=address,
            action_digest=recorded.digest,
            disposition=disposition,
            request_sha256=writer.put_evidence(f"request:{address.lineage}".encode()),
            response_content_sha256=writer.put_evidence(
                f"response:{address.lineage}".encode()
            ),
            prompt_tokens=recorded.prompt_tokens,
            completion_tokens=recorded.completion_tokens,
            counts_toward_logical_cost=logical,
        )
        writer.mark_execution_override_delivered(
            attempt,
            ticket,
            typed_response_sha256=writer.put_evidence(
                f"typed:{address.lineage}".encode()
            ),
        )

    replay(target, "inject", logical=False)
    replay(PolicyEventAddress(1, "root/sibling", 0, 0), "reuse", logical=True)
    receipt = writer.finish_execution(
        attempt,
        outcome_kind=OutcomeKind.SUCCESS,
        scored_reward=0.5,
        scorer_evidence_sha256=writer.put_evidence(b"score"),
        latency_seconds=0.1,
        dollars=0.0,
        judge_calls=0,
        cpu_seconds=0.1,
        gpu_seconds=0.0,
        wall_seconds=0.1,
        storage_bytes=0,
    )
    payload = writer(receipt, receipt_kind="scientific_arm_execution")
    assert payload["calls"] == []
    assert len(payload["replayed_calls"]) == 2
    assert payload["logical_cost"]["output_tokens"] == (
        recorded.completion_tokens * 2
    )
    writer.seal()
    assert inspect_ledger(root).status == "sealed-valid"


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


def test_committed_child_pre_post_abort_is_durable_and_terminal(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    writer = _create(root)
    action_key = _key(17)
    snapshot = writer.put_evidence(b"snapshot")
    reservation = writer.commit_pre_action_and_reserve(
        group_id="group-1",
        rollout_id="rollout-1",
        target_roster=("target-0",),
        target_ordinal=0,
        target_id="target-0",
        target_address=PolicyEventAddress(1, "root/child", 0, 0),
        pre_action_snapshot_sha256=snapshot,
        recorded_action_key=action_key,
        branch_count=2,
        continuation_replicates=1,
        failure_reward=-1.0,
    )
    error = writer.put_evidence(b"child reservation failed before POST")

    receipt = writer.abort_source_child_before_post(
        reservation,
        rollout_id="rollout-1",
        error_sha256=error,
    )

    assert b'"receipt_kind":"source_child_pre_post_aborted"' in receipt
    assert inspect_ledger(root).status == "poisoned"
    writer.close()
    assert inspect_ledger(root).status == "poisoned"
