"""Narrow optional-dependency adapter for the pinned Prime-RL CPU path."""

from __future__ import annotations

import hashlib
import importlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

from redco.analysis.stage_d_training_bridge import SealedTrainingBatch, TrainerRecord
from redco.contracts import canonical_json


@dataclass(frozen=True, slots=True)
class PrimePackedSequence:
    token_ids: tuple[int, ...]
    mask: tuple[bool, ...]
    logprobs: tuple[float, ...]
    temperatures: tuple[float, ...]
    advantages: tuple[float, ...]
    rl_weights: tuple[float, ...]
    env_names: tuple[str, ...]
    rl_normalizer: float

    def __post_init__(self) -> None:
        length = len(self.token_ids)
        if length == 0 or any(
            len(stream) != length
            for stream in (
                self.mask,
                self.logprobs,
                self.temperatures,
                self.advantages,
                self.rl_weights,
                self.env_names,
            )
        ):
            raise ValueError("Prime packed sequence streams must be aligned")
        if self.rl_normalizer <= 0:
            raise ValueError("Prime packed sequence normalizer must be positive")

    def to_payload(self) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "mask": list(self.mask),
            "logprobs": list(self.logprobs),
            "temperatures": list(self.temperatures),
            "advantages": list(self.advantages),
            "rl_weights": list(self.rl_weights),
            "env_names": list(self.env_names),
            "rl_normalizer": self.rl_normalizer,
        }


@dataclass(frozen=True, slots=True)
class PrimeBatchAudit:
    training_batch_identity: str
    bridge_payload_sha256: str
    prime_payload: bytes
    prime_payload_sha256: str
    packed_sequences: tuple[PrimePackedSequence, ...]
    sample_count: int
    packed_record_count: int
    rl_normalizer_sum: float
    prime_normalized_loss: float
    manual_normalized_loss: float

    def __post_init__(self) -> None:
        if not self.prime_payload:
            raise ValueError("Prime payload must be nonempty")
        for value in (
            self.training_batch_identity,
            self.bridge_payload_sha256,
            self.prime_payload_sha256,
        ):
            if len(value) != 64:
                raise ValueError("Prime audit identity fields must be SHA-256 values")
        if _sha256(self.prime_payload) != self.prime_payload_sha256:
            raise ValueError("Prime payload digest mismatch")
        if (
            self.sample_count < 1
            or self.packed_record_count < 1
            or self.packed_record_count != len(self.packed_sequences)
        ):
            raise ValueError("Prime audit counts must be positive")
        if (
            not math.isfinite(self.rl_normalizer_sum)
            or self.rl_normalizer_sum <= 0
            or not math.isclose(
                self.rl_normalizer_sum,
                sum(sequence.rl_normalizer for sequence in self.packed_sequences),
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Prime RL normalizer sum must be positive")
        if not all(
            math.isfinite(value)
            for value in (self.prime_normalized_loss, self.manual_normalized_loss)
        ):
            raise ValueError("Prime normalized losses must be finite")
        if not math.isclose(
            self.prime_normalized_loss,
            self.manual_normalized_loss,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError("Prime clean loss differs from the independent formula")


def audit_prime_cpu_batch(batch: SealedTrainingBatch) -> PrimeBatchAudit:
    """Round-trip through actual pinned Prime types, packer, and clean loss."""
    batch = SealedTrainingBatch.verify_bytes(batch.to_bytes())
    msgspec = importlib.import_module("msgspec")
    torch = importlib.import_module("torch")
    transport = importlib.import_module("prime_rl.transport")
    trainer_batch = importlib.import_module("prime_rl.trainer.batch")
    trainer_utils = importlib.import_module("prime_rl.trainer.utils")
    loss_module = importlib.import_module("prime_rl.trainer.rl.loss")
    redco_loss = importlib.import_module("prime_rl.trainer.rl.redco_loss")

    training_sample = transport.TrainingSample
    training_batch = transport.TrainingBatch
    examples = [
        training_sample(
            token_ids=list(record.token_ids),
            mask=list(record.mask),
            logprobs=list(record.behavior_logprobs),
            temperatures=list(record.temperatures),
            env_name=record.env_name,
            rl_weights=list(record.rl_weights),
            advantages=list(record.advantages),
            rl_normalizer=float(record.rl_normalizer),
        )
        for record in batch.records
    ]
    original = training_batch(examples=examples, step=batch.trainer_step)
    payload = cast(bytes, msgspec.msgpack.encode(original))
    decoded = msgspec.msgpack.decode(payload, type=training_batch)
    _verify_decoded(decoded, batch)

    prepared = trainer_batch.prepare_batch(
        rollouts=decoded.examples,
        seq_len=batch.seq_len,
        num_train_workers=1,
        idxs=[0] * len(decoded.examples),
        num_loras=1,
        bin_cost=trainer_utils.build_bin_cost(None),
    )
    packed = [micro_batch for worker in prepared for micro_batch in worker]
    packed_normalizers = [
        value for micro_batch in packed for value in (micro_batch.rl_normalizers or [])
    ]
    normalizer_sum = float(sum(packed_normalizers))
    expected_normalizer = float(sum(record.rl_normalizer for record in batch.records))
    if not math.isclose(normalizer_sum, expected_normalizer, abs_tol=1e-12):
        raise ValueError("Prime packer changed the decision-unit normalizer")

    packed_sequences = _packed_sequences(packed)
    expected_sequences = [_record_sequence(record) for record in batch.records]
    if Counter(canonical_json(value.to_payload()) for value in packed_sequences) != Counter(
        canonical_json(value.to_payload()) for value in expected_sequences
    ):
        raise ValueError("Prime packer changed a scientific token stream")

    prime_loss = 0.0
    manual_loss = 0.0
    for sequence in packed_sequences:
        trainer_logprobs = torch.tensor(sequence.logprobs, dtype=torch.float64)
        inputs = loss_module.LossInputs(
            trainer_logprobs=trainer_logprobs,
            inference_logprobs=trainer_logprobs.clone(),
            ref_logprobs=None,
            advantages=torch.tensor(sequence.advantages, dtype=torch.float64),
            loss_mask=torch.tensor(sequence.mask, dtype=torch.bool),
            loss_weights=torch.tensor(sequence.rl_weights, dtype=torch.float64),
        )
        output = redco_loss.clean_decision_loss(inputs, kl_tau=0.0)
        prime_loss += float(output.loss.item())
        manual_loss += -sum(
            advantage * logprob * weight
            for selected, advantage, logprob, weight in zip(
                sequence.mask,
                sequence.advantages,
                sequence.logprobs,
                sequence.rl_weights,
                strict=True,
            )
            if selected
        )
    prime_normalized = prime_loss / normalizer_sum
    manual_normalized = manual_loss / expected_normalizer
    return PrimeBatchAudit(
        batch.training_batch_identity,
        batch.payload_sha256,
        payload,
        _sha256(payload),
        tuple(packed_sequences),
        len(examples),
        len(packed_sequences),
        normalizer_sum,
        prime_normalized,
        manual_normalized,
    )


def verify_prime_payload(audit: PrimeBatchAudit, batch: SealedTrainingBatch) -> None:
    """Rederive every Prime field and require one exact prepared-batch audit."""
    batch = SealedTrainingBatch.verify_bytes(batch.to_bytes())
    if audit != audit_prime_cpu_batch(batch):
        raise ValueError("Prime audit differs from the rederived packed batch")


def _verify_decoded(decoded: Any, batch: SealedTrainingBatch) -> None:
    if decoded.step != batch.trainer_step or len(decoded.examples) != len(batch.records):
        raise ValueError("Prime batch envelope differs from the sealed bridge batch")
    for sample, record in zip(decoded.examples, batch.records, strict=True):
        expected = {
            "token_ids": list(record.token_ids),
            "mask": list(record.mask),
            "logprobs": list(record.behavior_logprobs),
            "temperatures": list(record.temperatures),
            "env_name": record.env_name,
            "rl_weights": list(record.rl_weights),
            "advantages": list(record.advantages),
            "rl_normalizer": float(record.rl_normalizer),
        }
        actual = {name: getattr(sample, name) for name in expected}
        if actual != expected:
            raise ValueError("Prime msgpack round-trip changed a scientific field")


def _record_sequence(record: TrainerRecord) -> PrimePackedSequence:
    return PrimePackedSequence(
        record.token_ids,
        record.mask,
        record.behavior_logprobs,
        record.temperatures,
        record.advantages,
        record.rl_weights,
        (record.env_name,) * len(record.token_ids),
        float(record.rl_normalizer),
    )


def _packed_sequences(packed: list[Any]) -> list[PrimePackedSequence]:
    sequences: list[PrimePackedSequence] = []
    for micro_batch in packed:
        if micro_batch.rl_weights is None or micro_batch.rl_normalizers is None:
            raise ValueError("Prime packer dropped clean-loss fields")
        if len(micro_batch.rl_normalizers) != len(micro_batch.sequence_lengths):
            raise ValueError("Prime packed normalizers lost sequence alignment")
        offset = 0
        for length, normalizer in zip(
            micro_batch.sequence_lengths,
            micro_batch.rl_normalizers,
            strict=True,
        ):
            end = offset + length
            trimmed_end = end
            while trimmed_end > offset and micro_batch.env_names[trimmed_end - 1] == "":
                trimmed_end -= 1
            selected = slice(offset, trimmed_end)
            sequences.append(
                PrimePackedSequence(
                    tuple(micro_batch.input_ids[selected]),
                    tuple(micro_batch.loss_mask[selected]),
                    tuple(micro_batch.inference_logprobs[selected]),
                    tuple(micro_batch.temperatures[selected]),
                    tuple(micro_batch.advantages[selected]),
                    tuple(micro_batch.rl_weights[selected]),
                    tuple(micro_batch.env_names[selected]),
                    float(normalizer),
                )
            )
            offset = end
        if offset != len(micro_batch.input_ids):
            raise ValueError("Prime packed sequence boundaries do not cover the microbatch")
    return sequences


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
