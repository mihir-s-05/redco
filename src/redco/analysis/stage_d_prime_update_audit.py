"""One tiny differentiable Prime-RL optimizer-step audit for the E1 bridge."""

from __future__ import annotations

import hashlib
import importlib
import math
from dataclasses import dataclass
from typing import Any

from redco.analysis.stage_d_prime_bridge import PrimeBatchAudit, verify_prime_payload
from redco.analysis.stage_d_training_bridge import SealedTrainingBatch
from redco.analysis.stage_d_update_ledger import (
    SingleUseUpdateLedger,
    UpdateCompletion,
    UpdateLedgerBinding,
)
from redco.contracts import canonical_json


@dataclass(frozen=True, slots=True)
class PrimeUpdateAudit:
    pre_model_sha256: str
    pre_optimizer_sha256: str
    post_model_sha256: str
    post_optimizer_sha256: str
    step_evidence_sha256: str
    normalized_loss: float
    gradient_l2: float

    def __post_init__(self) -> None:
        for value in (
            self.pre_model_sha256,
            self.pre_optimizer_sha256,
            self.post_model_sha256,
            self.post_optimizer_sha256,
            self.step_evidence_sha256,
        ):
            if len(value) != 64:
                raise ValueError("Prime update audit hashes must be SHA-256 values")
        if self.pre_model_sha256 == self.post_model_sha256:
            raise ValueError("Prime update audit did not change the model")
        if self.pre_optimizer_sha256 == self.post_optimizer_sha256:
            raise ValueError("Prime update audit did not change the optimizer")
        if (
            not math.isfinite(self.normalized_loss)
            or not math.isfinite(self.gradient_l2)
            or self.gradient_l2 <= 0
        ):
            raise ValueError("Prime update audit must have finite loss and nonzero gradient")


@dataclass(frozen=True, slots=True)
class PrimeLedgerUpdateResult:
    completion: UpdateCompletion
    audit: PrimeUpdateAudit


class PreparedPrimeCpuUpdate:
    """Prepared state whose sole optimizer step can run only after authorization."""

    __slots__ = (
        "_batch",
        "_loss_module",
        "_model",
        "_optimizer",
        "_prime",
        "_redco_loss",
        "_torch",
        "_used",
        "pre_model_sha256",
        "pre_optimizer_sha256",
    )

    def __init__(self, batch: SealedTrainingBatch, prime: PrimeBatchAudit) -> None:
        self._batch = SealedTrainingBatch.verify_bytes(batch.to_bytes())
        verify_prime_payload(prime, self._batch)
        self._prime = prime
        self._torch = importlib.import_module("torch")
        self._loss_module = importlib.import_module("prime_rl.trainer.rl.loss")
        self._redco_loss = importlib.import_module("prime_rl.trainer.rl.redco_loss")
        self._model = self._torch.nn.Parameter(
            self._torch.zeros(len(self._prime.packed_sequences), dtype=self._torch.float64)
        )
        self._optimizer = self._torch.optim.AdamW(
            [self._model],
            lr=0.05,
            weight_decay=0.0,
        )
        self.pre_model_sha256 = _state_sha256({"decision_offsets": self._model.detach()})
        self.pre_optimizer_sha256 = _state_sha256(self._optimizer.state_dict())
        self._used = False

    def verify_binding(self, binding: UpdateLedgerBinding) -> None:
        """Require the durable genesis to bind this exact prepared update."""
        expected = {
            "producer_seal_sha256": self._batch.binding.producer_seal_sha256,
            "training_batch_identity": self._batch.training_batch_identity,
            "bridge_payload_sha256": self._batch.payload_sha256,
            "prime_payload_sha256": self._prime.prime_payload_sha256,
            "prime_runtime_sha256": self._batch.binding.prime_runtime_sha256,
            "trainer_config_sha256": self._batch.binding.trainer_config_sha256,
            "expected_input_policy_sha256": self.pre_model_sha256,
        }
        if binding.to_payload() != expected:
            raise ValueError("update ledger binding differs from the prepared Prime update")

    def run_with_ledger(
        self,
        ledger: SingleUseUpdateLedger,
        *,
        consumer_id: str,
    ) -> PrimeLedgerUpdateResult:
        """Bind, durably authorize, execute once, complete, and seal."""
        self.verify_binding(ledger.binding)
        observed: list[PrimeUpdateAudit] = []

        def execute() -> tuple[str, str, str]:
            if ledger.status != "authorized-incomplete":
                raise RuntimeError("Prime update ran before durable authorization")
            audit = self._execute()
            observed.append(audit)
            return (
                audit.post_model_sha256,
                audit.post_optimizer_sha256,
                audit.step_evidence_sha256,
            )

        completion = ledger.run_once(
            consumer_id=consumer_id,
            pre_model_sha256=self.pre_model_sha256,
            pre_optimizer_sha256=self.pre_optimizer_sha256,
            update=execute,
        )
        if len(observed) != 1:
            raise RuntimeError("Prime update completion lacks one local audit")
        return PrimeLedgerUpdateResult(completion, observed[0])

    def _execute(self) -> PrimeUpdateAudit:
        """Run exactly one differentiable Prime loss and optimizer step."""
        if self._used:
            raise RuntimeError("prepared Prime update has already been attempted")
        self._used = True
        normalizer = self._prime.rl_normalizer_sum
        total_loss = self._model.sum() * 0.0
        for index, sequence in enumerate(self._prime.packed_sequences):
            behavior = self._torch.tensor(
                sequence.logprobs,
                dtype=self._torch.float64,
            )
            mask = self._torch.tensor(sequence.mask, dtype=self._torch.bool)
            trainer_logprobs = behavior + mask.to(dtype=self._torch.float64) * self._model[index]
            inputs = self._loss_module.LossInputs(
                trainer_logprobs=trainer_logprobs,
                inference_logprobs=behavior,
                ref_logprobs=None,
                advantages=self._torch.tensor(
                    sequence.advantages,
                    dtype=self._torch.float64,
                ),
                loss_mask=mask,
                loss_weights=self._torch.tensor(
                    sequence.rl_weights,
                    dtype=self._torch.float64,
                ),
            )
            total_loss = (
                total_loss
                + self._redco_loss.clean_decision_loss(
                    inputs,
                    kl_tau=0.0,
                ).loss
            )
        normalized_loss = total_loss / normalizer
        normalized_loss.backward()
        if self._model.grad is None:
            raise ValueError("Prime update audit produced no gradient")
        gradient_l2 = float(self._torch.linalg.vector_norm(self._model.grad).item())
        if not math.isfinite(gradient_l2) or gradient_l2 <= 0:
            raise ValueError("Prime update audit produced an invalid gradient")
        self._optimizer.step()
        post_model = _state_sha256({"decision_offsets": self._model.detach()})
        post_optimizer = _state_sha256(self._optimizer.state_dict())
        evidence = _state_sha256(
            {
                "domain": "redco-stage-d-prime-update-audit-v1",
                "training_batch_identity": self._batch.training_batch_identity,
                "bridge_payload_sha256": self._batch.payload_sha256,
                "prime_payload_sha256": self._prime.prime_payload_sha256,
                "normalized_loss": float(normalized_loss.detach().item()),
                "gradient_l2": gradient_l2,
                "pre_model_sha256": self.pre_model_sha256,
                "pre_optimizer_sha256": self.pre_optimizer_sha256,
                "post_model_sha256": post_model,
                "post_optimizer_sha256": post_optimizer,
            }
        )
        return PrimeUpdateAudit(
            self.pre_model_sha256,
            self.pre_optimizer_sha256,
            post_model,
            post_optimizer,
            evidence,
            float(normalized_loss.detach().item()),
            gradient_l2,
        )


def prepare_prime_cpu_update(
    batch: SealedTrainingBatch,
    prime: PrimeBatchAudit,
) -> PreparedPrimeCpuUpdate:
    """Prepare deterministic pre-state hashes without taking an optimizer step."""
    return PreparedPrimeCpuUpdate(batch, prime)


def _state_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(_portable(value))).hexdigest()


def _portable(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        tensor = value.detach().cpu()
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "values": tensor.tolist(),
        }
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported state value: {type(value).__name__}")
