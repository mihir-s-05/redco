"""Pinned-Prime packer, loss, and analytic-gradient audit for Stage-D arm batches."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import struct
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from redco.analysis.stage_d_arm_contracts import ArmTrainerRecord, SealedArmBatch
from redco.analysis.stage_d_objective_binding import (
    ObjectiveAuthorization,
    ObjectiveBinding,
)
from redco.analysis.stage_d_receipt_ledger import SealedReceiptVerifier
from redco.analysis.stage_d_trainer_supervisor import StageDTrainerRunLedger
from redco.contracts import canonical_json

PackedSequence = tuple[
    tuple[int, ...],
    tuple[bool, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...] | None,
    float | None,
]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def materialize_prime_objective_binding(
    *,
    arm: str,
    evidence_class: str,
    effective_argv: tuple[str, ...],
    trainer_toml_path: Path,
    trainer_toml_bytes: bytes,
) -> tuple[ObjectiveBinding, Any, Any]:
    """Parse and resolve the exact objective that pinned Prime will execute."""
    if effective_argv != ("@", str(trainer_toml_path)):
        raise ValueError("trainer objective requires one exact TOML and no overrides")
    if type(trainer_toml_bytes) is not bytes or not trainer_toml_bytes:
        raise ValueError("trainer TOML must be nonempty immutable bytes")
    try:
        raw_config = tomllib.loads(trainer_toml_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("trainer TOML is invalid") from error
    trainer_config_module = importlib.import_module("prime_rl.configs.trainer")
    loss_module = importlib.import_module("prime_rl.trainer.rl.loss")
    train_module = importlib.import_module("prime_rl.trainer.rl.train")
    config = trainer_config_module.TrainerConfig.model_validate(raw_config)
    return _materialize_resolved_prime_objective(
        arm=arm,
        evidence_class=evidence_class,
        effective_argv=effective_argv,
        trainer_toml_bytes=trainer_toml_bytes,
        config=config,
        trainer_config_module=trainer_config_module,
        loss_module=loss_module,
        train_module=train_module,
    )


def _materialize_resolved_prime_objective(
    *,
    arm: str,
    evidence_class: str,
    effective_argv: tuple[str, ...],
    trainer_toml_bytes: bytes,
    config: Any,
    trainer_config_module: Any,
    loss_module: Any,
    train_module: Any,
) -> tuple[ObjectiveBinding, Any, Any]:
    if train_module.setup_rl_loss_fn is not loss_module.setup_rl_loss_fn:
        raise ValueError("trainer imported a different setup_rl_loss_fn symbol")
    if (
        train_module.apply_exact_categorical_normalization
        is not loss_module.apply_exact_categorical_normalization
    ):
        raise ValueError("trainer imported a different exact-categorical symbol")
    setup_loss = loss_module.setup_rl_loss_fn(config.loss)
    loss_payload = config.loss.model_dump(mode="json", exclude_unset=False)
    if arm == "stock":
        import_path = "prime_rl.trainer.rl.loss.default_loss_fn"
        resolved_loss = _resolve_plain_python_function(import_path)
        if resolved_loss is not loss_module.default_loss_fn:
            raise ValueError("Prime default loss symbol changed during resolution")
    elif arm in {"branch-global", "local"}:
        if not isinstance(config.loss, trainer_config_module.CustomLossConfig):
            raise ValueError("branch objective must parse as a custom Prime loss")
        import_path = config.loss.import_path
        resolved_loss = _resolve_plain_python_function(import_path)
    else:
        raise ValueError("unsupported objective arm")
    if not callable(setup_loss):
        raise ValueError("Prime setup_rl_loss_fn did not return a callable")
    if arm != "stock":
        closure_values = tuple(cell.cell_contents for cell in (setup_loss.__closure__ or ()))
        if resolved_loss not in closure_values or config.loss.kwargs not in closure_values:
            raise ValueError("Prime custom loss closure differs from parsed configuration")
    resolved_module = importlib.import_module(resolved_loss.__module__)
    exact_categorical = (
        None
        if config.exact_categorical is None
        else config.exact_categorical.model_dump(mode="json", exclude_unset=False)
    )
    payload = {
        "schema_version": 1,
        "domain": "redco-stage-d-objective-binding-v1",
        "arm": arm,
        "evidence_class": evidence_class,
        "effective_argv": list(effective_argv),
        "trainer_toml_sha256": _sha256(trainer_toml_bytes),
        "materialized_trainer_config_sha256": _sha256(
            canonical_json(config.model_dump(mode="json", exclude_unset=False))
        ),
        "loss_config": loss_payload,
        "exact_categorical": exact_categorical,
        "fused_lm_head_token_chunk_size": config.model.fused_lm_head_token_chunk_size,
        "loss_callable": {
            "kind": "function",
            "import_path": import_path,
            "module": resolved_loss.__module__,
            "qualname": resolved_loss.__qualname__,
        },
        "module_sha256s": {
            "prime_rl.configs.trainer": _module_sha256(trainer_config_module),
            "prime_rl.trainer.rl.loss": _module_sha256(loss_module),
            "prime_rl.trainer.rl.train": _module_sha256(train_module),
            "resolved_loss_callable_module": _module_sha256(resolved_module),
        },
    }
    return ObjectiveBinding.from_bytes(canonical_json(payload)), config, setup_loss


@dataclass(frozen=True, slots=True)
class PrimeObjectiveCapture:
    """Config bytes captured before Prime's CLI parser is invoked."""

    effective_argv: tuple[str, str]
    trainer_toml_path: Path
    trainer_toml_bytes: bytes


@dataclass(slots=True)
class StageDPrimeRuntimeGate:
    """One live arm's objective and exact consumed-batch authorization."""

    binding: ObjectiveBinding
    batch: SealedArmBatch
    objective_authorization_sha256: str
    batch_authorization_sha256: str
    ledger_seal_sha256: str
    expected_pre_model_sha256: str | None = None
    base_model_manifest_sha256: str | None = None
    trainer_run_ledger: StageDTrainerRunLedger | None = None
    launch_id: str | None = None
    _initialization_verified: bool = False
    _batch_verified: bool = False
    _optimizer_started: bool = False
    _optimizer_completed: bool = False
    _post_model_sha256: str | None = None

    def verify_distributed(self) -> None:
        torch_dist = importlib.import_module("torch.distributed")
        if not torch_dist.is_initialized():
            raise ValueError("distributed Stage D gate requires initialized ranks")
        identity = (
            self.binding.objective_sha256,
            self.batch.batch_identity,
            self.objective_authorization_sha256,
            self.batch_authorization_sha256,
            self.ledger_seal_sha256,
            self.expected_pre_model_sha256,
            self.base_model_manifest_sha256,
        )
        world_size = int(torch_dist.get_world_size())
        gathered: list[tuple[str, str, str, str, str, str | None, str | None] | None] = [
            None
        ] * world_size
        torch_dist.all_gather_object(gathered, identity)
        if gathered != [identity] * world_size:
            raise ValueError("trainer ranks disagree on Stage D objective or batch evidence")

    def verify_initialization(self) -> None:
        """Prove every arm loaded the exact frozen LoRA pre-model state."""
        if self._initialization_verified:
            raise ValueError("Stage D initialization was verified twice")
        if self.expected_pre_model_sha256 is None or self.base_model_manifest_sha256 is None:
            raise ValueError("Stage D initialization identity is absent")
        from redco.analysis.stage_d_live_update import exported_adapter_state_sha256

        observed = exported_adapter_state_sha256(
            base_snapshot_manifest_sha256=self.base_model_manifest_sha256
        )
        torch_dist = importlib.import_module("torch.distributed")
        gathered: list[str | None] = [None] * int(torch_dist.get_world_size())
        torch_dist.all_gather_object(gathered, observed)
        if gathered != [self.expected_pre_model_sha256] * len(gathered):
            raise ValueError("Stage D loaded state differs from the frozen shared initialization")
        self._record_supervisor(
            "mark_initialization_verified",
            observed_pre_model_sha256=observed,
        )
        self._initialization_verified = True

    def verify_consumed_micro_batches(
        self,
        micro_batches: list[Any],
        *,
        trainer_step: int,
        process_group: Any,
    ) -> None:
        if not self._initialization_verified:
            raise ValueError("Stage D batch arrived before initialization verification")
        if self._batch_verified:
            raise ValueError("Stage D runtime gate observed a second training batch")
        if trainer_step != self.batch.trainer_step:
            raise ValueError("Prime consumed the sealed batch at a different trainer step")
        local_sequences: list[PackedSequence] | None = None
        local_error: str | None = None
        try:
            local_sequences = _unpack_tensor_micro_batches(micro_batches)
        except Exception as error:
            local_error = f"{type(error).__qualname__}: {error}"
        sequences = _gather_dp_sequences(
            local_sequences,
            local_error=local_error,
            process_group=process_group,
        )
        validate_prime_packed_sequences(self.batch, sequences)
        self._record_supervisor(
            "mark_batch_verified",
            batch_identity=self.batch.batch_identity,
        )
        self._batch_verified = True

    def before_optimizer_step(self, *, trainer_step: int) -> None:
        if not self._batch_verified or self._optimizer_started:
            raise ValueError("Stage D optimizer start is out of order")
        if trainer_step != self.batch.trainer_step:
            raise ValueError("Stage D optimizer started at a different trainer step")
        self._record_supervisor("mark_optimizer_started", trainer_step=trainer_step)
        self._optimizer_started = True

    def after_optimizer_step(self, *, trainer_step: int) -> None:
        if not self._optimizer_started or self._optimizer_completed:
            raise ValueError("Stage D optimizer completion is out of order")
        if trainer_step != self.batch.trainer_step:
            raise ValueError("Stage D optimizer completed at a different trainer step")
        if self.base_model_manifest_sha256 is None:
            raise ValueError("Stage D post-update state lacks its base-model identity")
        from redco.analysis.stage_d_live_update import exported_adapter_state_sha256

        observed = exported_adapter_state_sha256(
            base_snapshot_manifest_sha256=self.base_model_manifest_sha256
        )
        torch_dist = importlib.import_module("torch.distributed")
        gathered: list[str | None] = [None] * int(torch_dist.get_world_size())
        torch_dist.all_gather_object(gathered, observed)
        if gathered != [observed] * len(gathered):
            raise ValueError("Stage D ranks disagree on the post-update model state")
        self._record_supervisor(
            "mark_optimizer_completed",
            trainer_step=trainer_step,
            post_model_sha256=observed,
        )
        self._post_model_sha256 = observed
        self._optimizer_completed = True

    def verify_finished(self) -> None:
        """Reject a nominally successful trainer exit that consumed no sealed batch."""
        if (
            not self._batch_verified
            or not self._optimizer_completed
            or self._post_model_sha256 is None
        ):
            raise ValueError("Stage D trainer exited without one complete sealed update")

    def _record_supervisor(self, method: str, **payload: object) -> None:
        if self.trainer_run_ledger is None:
            if self.launch_id is not None:
                raise ValueError("Stage D launch ID exists without a trainer-run ledger")
            return
        if not self.launch_id:
            raise ValueError("Stage D trainer-run ledger requires a launch ID")
        torch_dist = importlib.import_module("torch.distributed")
        local_error: str | None = None
        if int(torch_dist.get_rank()) == 0:
            try:
                callback = getattr(self.trainer_run_ledger, method)
                callback(
                    arm=self.batch.arm,
                    launch_id=self.launch_id,
                    **payload,
                )
            except Exception as error:
                local_error = f"{type(error).__qualname__}: {error}"
        gathered: list[str | None] = [None] * int(torch_dist.get_world_size())
        torch_dist.all_gather_object(gathered, local_error)
        failures = tuple(error for error in gathered if error is not None)
        if failures:
            raise ValueError(f"Stage D trainer supervisor rejected transition: {failures}")


def capture_prime_objective_cli(
    effective_argv: tuple[str, ...] | None = None,
) -> PrimeObjectiveCapture:
    argv = tuple(sys.argv[1:]) if effective_argv is None else effective_argv
    if len(argv) != 2 or argv[0] != "@" or not argv[1]:
        raise ValueError("trainer objective requires one exact TOML and no overrides")
    path = Path(argv[1])
    value = path.read_bytes()
    if not value:
        raise ValueError("trainer TOML must be nonempty")
    return PrimeObjectiveCapture((argv[0], argv[1]), path, value)


def verify_captured_prime_objective(
    *,
    config: Any,
    capture: PrimeObjectiveCapture,
    arm: str,
    train_module: Any,
    expected_binding_bytes: bytes,
    authorization_bytes: bytes,
    expected_authorization_sha256: str,
    sealed_batch_bytes: bytes,
    batch_authorization_receipt: bytes,
    receipt_verifier: SealedReceiptVerifier,
    ledger_seal_sha256: str,
    trainer_run_ledger: StageDTrainerRunLedger | None = None,
    launch_id: str | None = None,
    expected_pre_model_sha256: str | None = None,
    base_model_manifest_sha256: str | None = None,
) -> tuple[StageDPrimeRuntimeGate, Any, Any]:
    """Authorize the parsed in-process config against batch and preregistration."""
    if capture.trainer_toml_path.read_bytes() != capture.trainer_toml_bytes:
        raise ValueError("trainer TOML changed while Prime parsed it")
    trainer_config_module = importlib.import_module("prime_rl.configs.trainer")
    loss_module = importlib.import_module("prime_rl.trainer.rl.loss")
    result = _materialize_resolved_prime_objective(
        arm=arm,
        evidence_class="live",
        effective_argv=capture.effective_argv,
        trainer_toml_bytes=capture.trainer_toml_bytes,
        config=config,
        trainer_config_module=trainer_config_module,
        loss_module=loss_module,
        train_module=train_module,
    )
    binding = result[0]
    expected = ObjectiveBinding.from_bytes(expected_binding_bytes)
    authorization = ObjectiveAuthorization.from_bytes(authorization_bytes)
    batch = SealedArmBatch.verify_bytes(sealed_batch_bytes)
    if _sha256(authorization_bytes) != expected_authorization_sha256:
        raise ValueError("objective authorization differs from preregistered digest")
    authorization.authorize_one(binding)
    if (
        binding != expected
        or batch.evidence_class != "live"
        or batch.arm != binding.arm
        or batch.objective_binding != binding
    ):
        raise ValueError("in-process objective differs from frozen live authorization")
    batch_authorization_sha256 = _verify_stage_d_batch_authorization(
        batch_authorization_receipt,
        verifier=receipt_verifier,
        batch=batch,
        sealed_batch_bytes=sealed_batch_bytes,
        objective_authorization_sha256=expected_authorization_sha256,
    )
    if len(ledger_seal_sha256) != 64:
        raise ValueError("Stage D ledger seal digest must be SHA-256")
    return (
        StageDPrimeRuntimeGate(
            binding,
            batch,
            expected_authorization_sha256,
            batch_authorization_sha256,
            ledger_seal_sha256,
            expected_pre_model_sha256,
            base_model_manifest_sha256,
            trainer_run_ledger,
            launch_id,
        ),
        result[1],
        result[2],
    )


def _verify_stage_d_batch_authorization(
    receipt_bytes: bytes,
    *,
    verifier: SealedReceiptVerifier,
    batch: SealedArmBatch,
    sealed_batch_bytes: bytes,
    objective_authorization_sha256: str,
) -> str:
    if type(verifier) is not SealedReceiptVerifier:
        raise ValueError("Stage D batch authorization requires an out-of-band sealed ledger")
    if type(receipt_bytes) is not bytes or not receipt_bytes:
        raise ValueError("Stage D batch authorization receipt is absent")
    try:
        parsed = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Stage D batch authorization must be canonical JSON") from error
    if not isinstance(parsed, dict) or canonical_json(parsed) != receipt_bytes:
        raise ValueError("Stage D batch authorization must be canonical JSON")
    anchored = dict(
        verifier(
            receipt_bytes,
            receipt_kind="stage_d_training_batch_authorization",
        )
    )
    if anchored != parsed:
        raise ValueError("sealed ledger returned different batch authorization bytes")
    expected_fields = {
        "schema_version",
        "receipt_kind",
        "ledger_id",
        "ledger_offset",
        "prior_chain_sha256",
        "arm",
        "training_batch_identity",
        "sealed_batch_sha256",
        "objective_sha256",
        "objective_authorization_sha256",
        "collection_plan_sha256",
        "collection_receipt_sha256",
        "support_report_sha256",
        "source_sha256s",
        "branch_artifact_sha256s",
        "consumer_id",
        "claim_sequence",
        "single_use",
    }
    if set(parsed) != expected_fields or parsed != {
        "schema_version": 1,
        "receipt_kind": "stage_d_training_batch_authorization",
        "ledger_id": parsed["ledger_id"],
        "ledger_offset": parsed["ledger_offset"],
        "prior_chain_sha256": parsed["prior_chain_sha256"],
        "arm": batch.arm,
        "training_batch_identity": batch.batch_identity,
        "sealed_batch_sha256": _sha256(sealed_batch_bytes),
        "objective_sha256": batch.objective_binding.objective_sha256,
        "objective_authorization_sha256": objective_authorization_sha256,
        "collection_plan_sha256": parsed["collection_plan_sha256"],
        "collection_receipt_sha256": parsed["collection_receipt_sha256"],
        "support_report_sha256": parsed["support_report_sha256"],
        "source_sha256s": list(batch.source_sha256s),
        "branch_artifact_sha256s": list(batch.branch_artifact_sha256s),
        "consumer_id": f"stage-d-prime:{batch.arm}:step:{batch.trainer_step}",
        "claim_sequence": parsed["claim_sequence"],
        "single_use": True,
    }:
        raise ValueError("sealed batch differs from its single-use authorization")
    if any(
        not isinstance(parsed[field], str) or len(parsed[field]) != 64
        for field in (
            "collection_plan_sha256",
            "collection_receipt_sha256",
            "support_report_sha256",
        )
    ):
        raise ValueError("sealed batch lacks collection-roster authorization")
    if (
        type(parsed["ledger_offset"]) is not int
        or parsed["ledger_offset"] < 1
        or parsed["claim_sequence"] != parsed["ledger_offset"]
        or not isinstance(parsed["ledger_id"], str)
        or not parsed["ledger_id"]
        or not isinstance(parsed["prior_chain_sha256"], str)
        or len(parsed["prior_chain_sha256"]) != 64
    ):
        raise ValueError("Stage D batch authorization chronology is invalid")
    return _sha256(receipt_bytes)


def capture_and_materialize_prime_objective(
    *,
    arm: str,
    evidence_class: str,
    effective_argv: tuple[str, ...] | None = None,
) -> tuple[ObjectiveBinding, Any, Any]:
    """Open the sole CLI config once and reject any parse-time file drift."""
    capture = capture_prime_objective_cli(effective_argv)
    result = materialize_prime_objective_binding(
        arm=arm,
        evidence_class=evidence_class,
        effective_argv=capture.effective_argv,
        trainer_toml_path=capture.trainer_toml_path,
        trainer_toml_bytes=capture.trainer_toml_bytes,
    )
    if capture.trainer_toml_path.read_bytes() != capture.trainer_toml_bytes:
        raise ValueError("trainer TOML changed during objective materialization")
    return result


def verify_distributed_objective_binding(binding: ObjectiveBinding) -> None:
    """Require every initialized trainer rank to report one objective identity."""
    torch_dist = importlib.import_module("torch.distributed")
    if not torch_dist.is_initialized():
        raise ValueError("distributed objective verification requires initialized ranks")
    world_size = int(torch_dist.get_world_size())
    gathered: list[str | None] = [None] * world_size
    torch_dist.all_gather_object(gathered, binding.objective_sha256)
    if gathered != [binding.objective_sha256] * world_size:
        raise ValueError("trainer ranks disagree on the executable objective")


def _unpack_tensor_micro_batches(micro_batches: list[Any]) -> list[PackedSequence]:
    sequences: list[PackedSequence] = []
    for micro_batch in micro_batches:
        lengths = list(micro_batch["sequence_lengths"])
        if any(type(length) is not int or length < 1 for length in lengths):
            raise ValueError("Prime consumed invalid sequence lengths")
        normalizers = micro_batch["rl_normalizers"]
        if normalizers is not None and len(normalizers) != len(lengths):
            raise ValueError("Prime consumed normalizers with wrong sequence alignment")
        token_ids = _flat_cpu_values(micro_batch["input_ids"], int)
        mask = _flat_cpu_values(micro_batch["loss_mask"], bool)
        logprobs = _flat_cpu_values(micro_batch["inference_logprobs"], float)
        temperatures = _flat_cpu_values(micro_batch["temperatures"], float)
        advantages = _flat_cpu_values(micro_batch["advantages"], float)
        weights = (
            None
            if micro_batch["rl_weights"] is None
            else _flat_cpu_values(micro_batch["rl_weights"], float)
        )
        env_names = list(micro_batch["env_names"])
        total = len(token_ids)
        aligned_lengths = {
            len(mask),
            len(logprobs),
            len(temperatures),
            len(advantages),
            len(env_names),
        }
        if weights is not None:
            aligned_lengths.add(len(weights))
        if aligned_lengths != {total} or sum(lengths) != total:
            raise ValueError("Prime consumed misaligned or trailing tensor streams")
        offset = 0
        for index, length in enumerate(lengths):
            stop = offset + int(length)
            trimmed = stop
            while trimmed > offset and env_names[trimmed - 1] == "":
                trimmed -= 1
            selected = slice(offset, trimmed)
            sequence: PackedSequence = (
                tuple(token_ids[selected]),
                tuple(mask[selected]),
                tuple(logprobs[selected]),
                tuple(temperatures[selected]),
                tuple(advantages[selected]),
                None if weights is None else tuple(weights[selected]),
                None if normalizers is None else float(normalizers[index]),
            )
            if not any(sequence[1]):
                if any(value != 0.0 for value in sequence[4]):
                    raise ValueError("Prime dummy padding has nonzero advantages")
                if sequence[5] is not None and any(value != 0.0 for value in sequence[5]):
                    raise ValueError("Prime dummy padding has nonzero RL weights")
                if sequence[6] not in (None, 0.0):
                    raise ValueError("Prime dummy padding has a nonzero normalizer")
            else:
                sequences.append(sequence)
            offset = stop
        if offset != total:
            raise ValueError("Prime sequence lengths did not consume every token")
    return sequences


def _gather_dp_sequences(
    local_sequences: list[PackedSequence] | None,
    *,
    local_error: str | None,
    process_group: Any,
) -> list[PackedSequence]:
    """Gather the exact data-parallel shards before validating the global batch."""
    if process_group is None:
        raise ValueError("Stage D runtime gate requires the exact data-parallel group")
    torch_dist = importlib.import_module("torch.distributed")
    if not torch_dist.is_initialized():
        raise ValueError("distributed Stage D batch gate requires initialized ranks")
    world_size = int(torch_dist.get_world_size(group=process_group))
    if world_size < 1:
        raise ValueError("data-parallel group has invalid world size")
    local_payload = {"error": local_error, "sequences": local_sequences}
    gathered: list[dict[str, Any] | None] = [None] * world_size
    torch_dist.all_gather_object(gathered, local_payload, group=process_group)
    if any(shard is None for shard in gathered):
        raise ValueError("data-parallel sequence gather was incomplete")
    errors = [
        shard["error"] for shard in gathered if shard is not None and shard.get("error") is not None
    ]
    if errors:
        raise ValueError(f"data-parallel shard validation failed: {errors}")
    return [
        sequence
        for shard in gathered
        if shard is not None
        for sequence in (shard.get("sequences") or [])
    ]


def _flat_cpu_values(value: Any, cast_value: Any) -> list[Any]:
    return [cast_value(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _resolve_plain_python_function(import_path: str) -> Any:
    if not import_path or "." not in import_path:
        raise ValueError("loss import path is invalid")
    module_name, attribute = import_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    if "__getattr__" in vars(module):
        raise ValueError("dynamic module attributes are forbidden for scientific loss")
    value = vars(module).get(attribute)
    if (
        not inspect.isfunction(value)
        or hasattr(value, "__wrapped__")
        or value.__name__ != attribute
        or "<locals>" in value.__qualname__
    ):
        raise ValueError("scientific loss must be one undecorated module-level function")
    _module_path(module)
    return value


def _module_path(module: Any) -> Path:
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str):
        spec = importlib.util.find_spec(module.__name__)
        origin = None if spec is None else spec.origin
    if not isinstance(origin, str) or not origin.endswith(".py"):
        raise ValueError("scientific objective modules require concrete .py origins")
    path = Path(origin)
    if not path.is_file():
        raise ValueError("scientific objective module source is absent")
    return path


def _module_sha256(module: Any) -> str:
    return _sha256(_module_path(module).read_bytes())


@dataclass(frozen=True, slots=True)
class ThreeArmPrimeAudit:
    arm: str
    prime_payload_sha256: str
    sample_count: int
    packed_sequence_count: int
    normalization_mode: str
    rl_scale: float
    prime_loss: float
    manual_loss: float
    max_gradient_error: float

    def __post_init__(self) -> None:
        if self.arm not in {"stock", "branch-global", "local"}:
            raise ValueError("unsupported audited arm")
        if self.normalization_mode not in {"token", "decision"}:
            raise ValueError("unsupported Prime normalization mode")
        if self.sample_count < 1 or self.packed_sequence_count < 1:
            raise ValueError("Prime audit requires nonempty samples")
        if self.rl_scale <= 0.0:
            raise ValueError("Prime audit scale must be positive")
        if any(
            not math.isfinite(value)
            for value in (
                self.prime_loss,
                self.manual_loss,
                self.max_gradient_error,
            )
        ):
            raise ValueError("Prime audit metrics must be finite")
        if not math.isclose(self.prime_loss, self.manual_loss, abs_tol=1e-9):
            raise ValueError("Prime loss differs from the independent objective")
        if self.max_gradient_error > 1e-12:
            raise ValueError("Prime gradient differs from the analytic objective")


def materialize_prime_rollout_bytes(batch: SealedArmBatch) -> bytes:
    """Encode one sealed arm as Prime's exact ``train_rollouts.bin`` payload."""
    batch = SealedArmBatch.verify_bytes(batch.to_bytes())
    msgspec = importlib.import_module("msgspec")
    transport = importlib.import_module("prime_rl.transport")
    trainer_batch = importlib.import_module("prime_rl.trainer.batch")
    trainer_utils = importlib.import_module("prime_rl.trainer.utils")
    examples = [
        transport.TrainingSample(
            token_ids=list(record.token_ids),
            mask=list(record.mask),
            logprobs=list(record.behavior_logprobs),
            temperatures=list(record.temperatures),
            env_name=f"redco-stage-d-{batch.arm}",
            rl_weights=(None if record.rl_weights is None else list(record.rl_weights)),
            advantages=list(record.advantages),
            rl_normalizer=(
                None if record.rl_normalizer is None else float(record.rl_normalizer)
            ),
        )
        for record in batch.records
    ]
    original = transport.TrainingBatch(examples=examples, step=batch.trainer_step)
    payload = cast(bytes, msgspec.msgpack.encode(original))
    decoded = msgspec.msgpack.decode(payload, type=transport.TrainingBatch)
    if decoded.step != batch.trainer_step or len(decoded.examples) != len(batch.records):
        raise ValueError("Prime transport changed the sealed arm batch")
    prepared = trainer_batch.prepare_batch(
        rollouts=decoded.examples,
        seq_len=batch.seq_len,
        num_train_workers=1,
        idxs=[0] * len(decoded.examples),
        num_loras=1,
        bin_cost=trainer_utils.build_bin_cost(None),
    )
    packed = [micro_batch for worker in prepared for micro_batch in worker]
    sequences = _unpack_sequences(packed)
    if len(sequences) != len(batch.records):
        raise ValueError("Prime packer changed the scientific sample count")
    validate_prime_packed_sequences(batch, sequences)
    reencoded = cast(bytes, msgspec.msgpack.encode(decoded))
    if reencoded != payload:
        raise ValueError("Prime transport payload is not byte-stable after decoding")
    return payload


def audit_three_arm_prime_batch(
    batch: SealedArmBatch,
    *,
    resolved_objective: tuple[ObjectiveBinding, Any, Any] | None = None,
) -> ThreeArmPrimeAudit:
    """Round-trip one sealed arm through actual pinned Prime types and losses."""
    batch = SealedArmBatch.verify_bytes(batch.to_bytes())
    msgspec = importlib.import_module("msgspec")
    torch = importlib.import_module("torch")
    transport = importlib.import_module("prime_rl.transport")
    trainer_batch = importlib.import_module("prime_rl.trainer.batch")
    trainer_utils = importlib.import_module("prime_rl.trainer.utils")
    loss_module = importlib.import_module("prime_rl.trainer.rl.loss")
    redco_loss = importlib.import_module("prime_rl.trainer.rl.redco_loss")
    trainer_config = importlib.import_module("prime_rl.configs.trainer")

    if batch.evidence_class == "live" and resolved_objective is None:
        raise ValueError("live Prime audit requires the resolved executable objective")
    resolved_loss_fn = None
    if resolved_objective is not None:
        binding, _config, resolved_loss_fn = resolved_objective
        if binding != batch.objective_binding:
            raise ValueError("resolved executable objective differs from sealed batch")

    examples = [
        transport.TrainingSample(
            token_ids=list(record.token_ids),
            mask=list(record.mask),
            logprobs=list(record.behavior_logprobs),
            temperatures=list(record.temperatures),
            env_name=f"redco-stage-d-{batch.arm}",
            rl_weights=(None if record.rl_weights is None else list(record.rl_weights)),
            advantages=list(record.advantages),
            rl_normalizer=(None if record.rl_normalizer is None else float(record.rl_normalizer)),
        )
        for record in batch.records
    ]
    original = transport.TrainingBatch(examples=examples, step=batch.trainer_step)
    payload = cast(bytes, msgspec.msgpack.encode(original))
    decoded = msgspec.msgpack.decode(payload, type=transport.TrainingBatch)
    if decoded.step != batch.trainer_step or len(decoded.examples) != len(batch.records):
        raise ValueError("Prime transport changed the sealed arm batch")

    prepared = trainer_batch.prepare_batch(
        rollouts=decoded.examples,
        seq_len=batch.seq_len,
        num_train_workers=1,
        idxs=[0] * len(decoded.examples),
        num_loras=1,
        bin_cost=trainer_utils.build_bin_cost(None),
    )
    packed = [micro_batch for worker in prepared for micro_batch in worker]
    sequences = _unpack_sequences(packed)
    if len(sequences) != len(batch.records):
        raise ValueError("Prime packer changed the scientific sample count")

    decision_mode = all(record.rl_normalizer is not None for record in batch.records)
    token_mode = all(record.rl_normalizer is None for record in batch.records)
    if decision_mode == token_mode:
        raise ValueError("one arm batch must use exactly one normalization mode")
    rl_scale = validate_prime_packed_sequences(batch, sequences)

    prime_total = torch.zeros((), dtype=torch.float64)
    manual_total = _manual_sealed_loss(batch)
    gradient_errors: list[float] = []
    default_config = trainer_config.DefaultLossConfig(kl_tau=0.0)
    for sequence in sequences:
        behavior = torch.tensor(sequence[2], dtype=torch.float64)
        trainer = behavior.clone().detach().requires_grad_(True)
        mask = torch.tensor(sequence[1], dtype=torch.bool)
        advantages = torch.tensor(sequence[4], dtype=torch.float64)
        weights_value = sequence[5]
        weights = (
            None if weights_value is None else torch.tensor(weights_value, dtype=torch.float64)
        )
        inputs = loss_module.LossInputs(
            trainer_logprobs=trainer,
            inference_logprobs=behavior,
            ref_logprobs=None,
            advantages=advantages,
            loss_mask=mask,
            loss_weights=weights,
        )
        if resolved_loss_fn is not None:
            output = resolved_loss_fn(inputs)
        else:
            output = (
                loss_module.default_loss_fn(inputs, default_config)
                if batch.arm == "stock"
                else redco_loss.clean_decision_loss(inputs, kl_tau=0.0)
            )
        scaled = output.loss / rl_scale
        scaled.backward()
        prime_total = prime_total + output.loss.detach()
        expected_gradient = _sealed_gradient(sequence, rl_scale)
        gradient_errors.extend(
            abs(float(actual) - expected)
            for actual, expected in zip(
                trainer.grad.tolist(),
                expected_gradient,
                strict=True,
            )
        )
    return ThreeArmPrimeAudit(
        batch.arm,
        _sha256(payload),
        len(examples),
        len(sequences),
        "decision" if decision_mode else "token",
        rl_scale,
        float(prime_total.item()) / rl_scale,
        manual_total / rl_scale,
        max(gradient_errors, default=0.0),
    )


def validate_prime_packed_sequences(
    batch: SealedArmBatch,
    sequences: tuple[PackedSequence, ...] | list[PackedSequence],
) -> float:
    """Bind unpacked Prime streams and its denominator to sealed record evidence."""
    sealed_sequences = tuple(_record_sequence(record) for record in batch.records)
    actual_tensor_values = tuple(_float32_sequence(sequence) for sequence in sequences)
    expected_tensor_values = tuple(_float32_sequence(sequence) for sequence in sealed_sequences)
    if Counter(actual_tensor_values) != Counter(expected_tensor_values):
        raise ValueError("Prime packer changed sealed token-stream evidence")
    decision_mode = all(record.rl_normalizer is not None for record in batch.records)
    sealed_rl_scale = (
        math.fsum(float(cast(Fraction, record.rl_normalizer)) for record in batch.records)
        if decision_mode
        else float(sum(sum(record.mask) for record in batch.records))
    )
    packed_rl_scale = (
        math.fsum(cast(float, sequence[6]) for sequence in sequences)
        if decision_mode
        else float(
            sum(
                sum(
                    selected and (weights is None or weight != 0.0)
                    for selected, weight in zip(
                        sequence[1],
                        sequence[5] or (1.0,) * len(sequence[1]),
                        strict=True,
                    )
                )
                for sequence in sequences
                for weights in (sequence[5],)
            )
        )
    )
    sealed_scale = max(sealed_rl_scale, 1.0)
    if not math.isclose(max(packed_rl_scale, 1.0), sealed_scale, abs_tol=1e-12):
        raise ValueError("Prime packed denominator differs from sealed records")
    return sealed_scale


def _record_sequence(record: ArmTrainerRecord) -> PackedSequence:
    return (
        tuple(record.token_ids),
        tuple(record.mask),
        tuple(record.behavior_logprobs),
        tuple(record.temperatures),
        tuple(record.advantages),
        None if record.rl_weights is None else tuple(record.rl_weights),
        None if record.rl_normalizer is None else float(record.rl_normalizer),
    )


def _float32_sequence(sequence: PackedSequence) -> PackedSequence:
    """Canonicalize exactly as Prime's TensorMicroBatch float tensors do."""
    return (
        sequence[0],
        sequence[1],
        tuple(_float32(value) for value in sequence[2]),
        tuple(_float32(value) for value in sequence[3]),
        tuple(_float32(value) for value in sequence[4]),
        (None if sequence[5] is None else tuple(_float32(value) for value in sequence[5])),
        sequence[6],
    )


def _float32(value: float) -> float:
    return float(struct.unpack("!f", struct.pack("!f", value))[0])


def _manual_sealed_loss(batch: SealedArmBatch) -> float:
    total = 0.0
    for record in batch.records:
        if batch.arm == "stock":
            total += -math.fsum(
                advantage
                for selected, advantage in zip(
                    record.mask,
                    record.advantages,
                    strict=True,
                )
                if selected
            )
        else:
            assert record.rl_weights is not None
            total += -math.fsum(
                advantage * logprob * weight
                for selected, advantage, logprob, weight in zip(
                    record.mask,
                    record.advantages,
                    record.behavior_logprobs,
                    record.rl_weights,
                    strict=True,
                )
                if selected
            )
    return total


def _sealed_gradient(sequence: PackedSequence, rl_scale: float) -> list[float]:
    weights = sequence[5] or tuple(1.0 for _ in sequence[1])
    return [
        (-advantage * weight / rl_scale) if selected else 0.0
        for selected, advantage, weight in zip(
            sequence[1],
            sequence[4],
            weights,
            strict=True,
        )
    ]


def _unpack_sequences(
    packed: list[Any],
) -> list[PackedSequence]:
    sequences = []
    for micro_batch in packed:
        offset = 0
        normalizers = micro_batch.rl_normalizers
        if normalizers is not None and len(normalizers) != len(micro_batch.sequence_lengths):
            raise ValueError("Prime packed normalizers lost sequence alignment")
        for index, length in enumerate(micro_batch.sequence_lengths):
            stop = offset + length
            trimmed = stop
            while trimmed > offset and micro_batch.env_names[trimmed - 1] == "":
                trimmed -= 1
            selected = slice(offset, trimmed)
            sequences.append(
                (
                    tuple(int(value) for value in micro_batch.input_ids[selected]),
                    tuple(bool(value) for value in micro_batch.loss_mask[selected]),
                    tuple(float(value) for value in micro_batch.inference_logprobs[selected]),
                    tuple(float(value) for value in micro_batch.temperatures[selected]),
                    tuple(float(value) for value in micro_batch.advantages[selected]),
                    (
                        None
                        if micro_batch.rl_weights is None
                        else tuple(float(value) for value in micro_batch.rl_weights[selected])
                    ),
                    None if normalizers is None else float(normalizers[index]),
                )
            )
            offset = stop
    return sequences
