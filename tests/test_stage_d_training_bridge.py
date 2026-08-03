from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from test_stage_d_scientific_branch_group import (
    Fixture,
    _fixture,
    _run,
    _unexpected_prompt_render,
    _validate_prepared_action,
)

from redco.analysis.stage_d_prime_bridge import audit_prime_cpu_batch, verify_prime_payload
from redco.analysis.stage_d_prime_update_audit import (
    prepare_prime_cpu_update,
)
from redco.analysis.stage_d_training_bridge import (
    ArtifactVerificationContext,
    SealedTrainingBatch,
    TrainingBridgeBinding,
    compile_training_batch,
    policy_identity_sha256,
)
from redco.analysis.stage_d_update_ledger import (
    SingleUseUpdateLedger,
    UpdateAlreadyAuthorized,
    UpdateLedgerBinding,
    UpdateLedgerError,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _artifacts(
    *,
    second_temperature: float = 0.7,
    prepared: bool = False,
) -> tuple[tuple[bytes, bytes], Fixture]:
    first_artifact, _, _, first = _run(
        _fixture(target_ordinal=0, prepared=prepared)
    )
    second_artifact, _, _, second = _run(
        _fixture(
            target_ordinal=1,
            prompt_content="different target prompt",
            temperature=second_temperature,
            prepared=prepared,
        ),
        rewards=(0.5, 0.5, 0.5, 0.5),
    )
    for kind, digests in second.store.allowed.items():
        first.store.allowed.setdefault(kind, set()).update(digests)
    return (first_artifact.to_bytes(), second_artifact.to_bytes()), first


def test_prepared_actions_reverify_through_branch_artifact_and_training_batch() -> None:
    values, fixture = _artifacts(prepared=True)
    batch = compile_training_batch(
        values,
        verification_context=ArtifactVerificationContext(
            verifier=fixture.store,
            encode_action=None,
            render_prompt=_unexpected_prompt_render,
            master_seed="master",
            validate_action=_validate_prepared_action,
        ),
        binding=TrainingBridgeBinding(
            producer_seal_sha256="1" * 64,
            bridge_source_sha256="2" * 64,
            prime_runtime_sha256="3" * 64,
            trainer_config_sha256="4" * 64,
            expected_policy_sha256=policy_identity_sha256(
                fixture.spec.recorded_action.key
            ),
        ),
        trainer_step=1,
        seq_len=8,
    )

    assert len(batch.records) == 8
    assert all(record.mask == (False, False, True, True) for record in batch.records)


def _batch(*, seq_len: int = 8) -> SealedTrainingBatch:
    values, fixture = _artifacts()
    policy_sha256 = policy_identity_sha256(fixture.spec.recorded_action.key)
    return compile_training_batch(
        values,
        verification_context=ArtifactVerificationContext(
            verifier=fixture.store,
            encode_action=lambda _request, _message: (20, 2),
            render_prompt=lambda _request: (10, 11),
            master_seed="master",
        ),
        binding=TrainingBridgeBinding(
            producer_seal_sha256="1" * 64,
            bridge_source_sha256="2" * 64,
            prime_runtime_sha256="3" * 64,
            trainer_config_sha256="4" * 64,
            expected_policy_sha256=policy_sha256,
        ),
        trainer_step=1,
        seq_len=seq_len,
    )


def test_bridge_reverifies_complete_roster_and_preserves_exact_clean_loss_fields() -> None:
    batch = _batch()

    assert len(batch.records) == 8
    assert sum(record.rl_normalizer for record in batch.records) == 2
    assert all(record.record_weight == Fraction(1, 8) for record in batch.records)
    assert all(record.rl_normalizer == Fraction(1, 4) for record in batch.records)
    assert all(record.mask == (False, False, True, True) for record in batch.records)
    assert all(record.behavior_logprobs[:2] == (0.0, 0.0) for record in batch.records)
    assert all(record.rl_weights[:2] == (0.0, 0.0) for record in batch.records)
    assert len({record.behavior_law_sha256 for record in batch.records}) == 2
    flat = [record for record in batch.records if record.target_id == "target-1"]
    assert len(flat) == 4
    assert all(record.advantages == (0.0, 0.0, 0.0, 0.0) for record in flat)

    restored = SealedTrainingBatch.verify_bytes(batch.to_bytes())
    assert restored == batch


def test_bridge_rejects_incomplete_roster_and_prime_truncation() -> None:
    values, fixture = _artifacts()
    binding = TrainingBridgeBinding(
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        policy_identity_sha256(fixture.spec.recorded_action.key),
    )
    context = ArtifactVerificationContext(
        fixture.store,
        lambda _request, _message: (20, 2),
        lambda _request: (10, 11),
        "master",
    )
    with pytest.raises(ValueError, match="incomplete target roster"):
        compile_training_batch(
            values[:1],
            verification_context=context,
            binding=binding,
            trainer_step=1,
            seq_len=8,
        )
    with pytest.raises(ValueError, match="truncate"):
        compile_training_batch(
            values,
            verification_context=context,
            binding=binding,
            trainer_step=1,
            seq_len=3,
        )


def test_bridge_rejects_mixed_policy_families() -> None:
    values, fixture = _artifacts(
        second_temperature=0.8,
    )
    with pytest.raises(ValueError, match="mix behavior policies"):
        compile_training_batch(
            values,
            verification_context=ArtifactVerificationContext(
                fixture.store,
                lambda _request, _message: (20, 2),
                lambda _request: (10, 11),
                "master",
            ),
            binding=TrainingBridgeBinding(
                "1" * 64,
                "2" * 64,
                "3" * 64,
                "4" * 64,
                policy_identity_sha256(fixture.spec.recorded_action.key),
            ),
            trainer_step=1,
            seq_len=8,
        )


def test_bridge_reload_rejects_nested_tampering() -> None:
    batch = _batch()
    envelope = json.loads(batch.to_bytes())
    envelope["payload"]["records"][0]["advantages"][-1] = 999.0
    tampered = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(ValueError, match="digest mismatch"):
        SealedTrainingBatch.verify_bytes(tampered)


def test_trainer_record_rejects_malformed_clean_loss_streams() -> None:
    record = _batch().records[0]
    with pytest.raises(ValueError, match="suffix action"):
        replace(record, mask=(False, True, False, True))
    with pytest.raises(ValueError, match="cannot be positive"):
        replace(record, behavior_logprobs=(0.0, 0.0, 0.1, -0.1))
    with pytest.raises(ValueError, match="positive temperature"):
        replace(record, temperatures=(0.7, 0.7, 0.8, 0.8))
    with pytest.raises(ValueError, match="scalar advantage"):
        replace(record, advantages=(0.0, 0.0, 1.0, 2.0))


def test_actual_prime_msgpack_packer_and_clean_loss_match_manual_formula() -> None:
    pytest.importorskip("prime_rl")
    batch = _batch()

    audit = audit_prime_cpu_batch(batch)
    verify_prime_payload(audit, batch)
    assert audit.sample_count == 8
    assert audit.rl_normalizer_sum == pytest.approx(2.0)
    assert audit.prime_normalized_loss == pytest.approx(audit.manual_normalized_loss)
    original = audit.packed_sequences[0]
    advantages = list(original.advantages)
    advantages[original.mask.index(True)] += 1.0
    forged_sequence = replace(original, advantages=tuple(advantages))
    forged = replace(
        audit,
        packed_sequences=(forged_sequence, *audit.packed_sequences[1:]),
    )
    with pytest.raises(ValueError, match="differs from the rederived"):
        verify_prime_payload(forged, batch)


def test_actual_prime_loss_drives_one_durable_optimizer_step(tmp_path: Path) -> None:
    pytest.importorskip("prime_rl")
    batch = _batch()
    prime = audit_prime_cpu_batch(batch)
    prepared = prepare_prime_cpu_update(batch, prime)
    binding = replace(
        _update_binding(batch, prime.prime_payload_sha256),
        expected_input_policy_sha256=prepared.pre_model_sha256,
    )
    prepared.verify_binding(binding)
    with pytest.raises(ValueError, match="binding differs"):
        prepared.verify_binding(replace(binding, prime_payload_sha256="f" * 64))
    ledger = SingleUseUpdateLedger.create(tmp_path / "prime-update", binding=binding)
    result = prepared.run_with_ledger(
        ledger,
        consumer_id="actual-prime-clean-loss",
    )
    assert result.completion.post_model_sha256 == result.audit.post_model_sha256
    assert ledger.status == "complete"
    assert len(list((ledger.root / "records").glob("*.json"))) == 4
    ledger.close()


def _update_binding(batch: SealedTrainingBatch, prime_payload_sha256: str) -> UpdateLedgerBinding:
    return UpdateLedgerBinding(
        producer_seal_sha256=batch.binding.producer_seal_sha256,
        training_batch_identity=batch.training_batch_identity,
        bridge_payload_sha256=batch.payload_sha256,
        prime_payload_sha256=prime_payload_sha256,
        prime_runtime_sha256=batch.binding.prime_runtime_sha256,
        trainer_config_sha256=batch.binding.trainer_config_sha256,
        expected_input_policy_sha256=batch.policy_sha256,
    )


def test_update_ledger_records_one_success_and_rejects_second_attempt(tmp_path: Path) -> None:
    batch = _batch()
    ledger = SingleUseUpdateLedger.create(
        tmp_path / "update-ledger",
        binding=_update_binding(batch, "5" * 64),
    )
    calls = 0

    def update() -> tuple[str, str, str]:
        nonlocal calls
        calls += 1
        return "6" * 64, "7" * 64, "8" * 64

    completion = ledger.run_once(
        consumer_id="cpu-prime-audit",
        pre_model_sha256=batch.policy_sha256,
        pre_optimizer_sha256="9" * 64,
        update=update,
    )
    assert calls == 1
    assert completion.post_model_sha256 == "6" * 64
    with pytest.raises(UpdateAlreadyAuthorized):
        ledger.run_once(
            consumer_id="second-attempt",
            pre_model_sha256=batch.policy_sha256,
            pre_optimizer_sha256="9" * 64,
            update=update,
        )
    assert calls == 1
    ledger.close()

    reopened = SingleUseUpdateLedger(tmp_path / "update-ledger")
    assert reopened.status == "complete"
    with pytest.raises(UpdateAlreadyAuthorized):
        reopened.authorize(
            consumer_id="reopened-attempt",
            pre_model_sha256=batch.policy_sha256,
            pre_optimizer_sha256="9" * 64,
        )
    reopened.close()


def test_update_ledger_rejects_forged_authorization_fields(tmp_path: Path) -> None:
    batch = _batch()
    ledger = SingleUseUpdateLedger.create(
        tmp_path / "forged-authorization",
        binding=_update_binding(batch, "5" * 64),
    )
    authorization = ledger.authorize(
        consumer_id="real-consumer",
        pre_model_sha256=batch.policy_sha256,
        pre_optimizer_sha256="9" * 64,
    )
    forged = replace(authorization, pre_model_sha256="a" * 64)
    with pytest.raises(ValueError, match="fields differ"):
        ledger.complete(
            forged,
            post_model_sha256=batch.policy_sha256,
            post_optimizer_sha256="b" * 64,
            step_evidence_sha256="c" * 64,
        )
    assert ledger.status == "authorized-incomplete"
    ledger.close()


def test_crash_after_authorization_is_durable_and_never_retried(tmp_path: Path) -> None:
    batch = _batch()
    root = tmp_path / "crashed-ledger"
    ledger = SingleUseUpdateLedger.create(root, binding=_update_binding(batch, "5" * 64))
    calls = 0

    def crash() -> tuple[str, str, str]:
        nonlocal calls
        calls += 1
        raise RuntimeError("optimizer process died")

    with pytest.raises(RuntimeError, match="optimizer process died"):
        ledger.run_once(
            consumer_id="crashing-attempt",
            pre_model_sha256=batch.policy_sha256,
            pre_optimizer_sha256="9" * 64,
            update=crash,
        )
    assert calls == 1
    assert ledger.status == "authorized-incomplete"
    ledger.close()

    reopened = SingleUseUpdateLedger(root)
    assert reopened.status == "authorized-incomplete"
    with pytest.raises(UpdateAlreadyAuthorized):
        reopened.authorize(
            consumer_id="forbidden-retry",
            pre_model_sha256=batch.policy_sha256,
            pre_optimizer_sha256="9" * 64,
        )
    reopened.close()


def test_hard_process_death_remains_readable_and_consumed(tmp_path: Path) -> None:
    root = tmp_path / "hard-crash-ledger"
    script = """
import os
import sys
from pathlib import Path
from redco.analysis.stage_d_update_ledger import SingleUseUpdateLedger, UpdateLedgerBinding

root = Path(sys.argv[1])
binding = UpdateLedgerBinding(
    producer_seal_sha256="1" * 64,
    training_batch_identity="2" * 64,
    bridge_payload_sha256="3" * 64,
    prime_payload_sha256="4" * 64,
    prime_runtime_sha256="5" * 64,
    trainer_config_sha256="6" * 64,
    expected_input_policy_sha256="7" * 64,
)
ledger = SingleUseUpdateLedger.create(root, binding=binding)
ledger.authorize(
    consumer_id="hard-crash",
    pre_model_sha256="7" * 64,
    pre_optimizer_sha256="8" * 64,
)
os._exit(17)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        check=False,
    )
    assert completed.returncode == 17
    assert (root / "writer.lock").is_file()
    assert SingleUseUpdateLedger.inspect_status(root) == "authorized-incomplete"
    with pytest.raises(UpdateLedgerError, match="active writer"):
        SingleUseUpdateLedger(root)


def test_update_ledger_corruption_fails_closed(tmp_path: Path) -> None:
    batch = _batch()
    root = tmp_path / "ledger"
    ledger = SingleUseUpdateLedger.create(root, binding=_update_binding(batch, "5" * 64))
    ledger.close()
    genesis = root / "records" / "00000000.json"
    genesis.write_bytes(genesis.read_bytes().replace(b'"sequence":0', b'"sequence":1'))

    with pytest.raises(UpdateLedgerError, match="envelope"):
        SingleUseUpdateLedger(root)
