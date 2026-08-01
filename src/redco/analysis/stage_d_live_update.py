"""Durable one-step authorization seam for the bounded Stage-D 4B smoke."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_update_ledger import (
    SingleUseUpdateLedger,
    UpdateAuthorization,
    UpdateCompletion,
    UpdateLedgerBinding,
)
from redco.contracts import canonical_json

SCHEMA_VERSION = 1
_BINDING_DOMAIN = "redco-stage-d-live-update-binding-v1"
_PRESTATE_DOMAIN = "redco-stage-d-live-update-prestate-v1"
_AUTH_DOMAIN = "redco-stage-d-live-update-authorization-v1"
_POSTSTEP_DOMAIN = "redco-stage-d-live-update-poststep-v1"
_ENV_BINDING = "REDCO_LIVE_UPDATE_BINDING"
_ENV_RECEIPTS = "REDCO_LIVE_UPDATE_RECEIPTS"


@dataclass(frozen=True, slots=True)
class LiveUpdateBinding:
    producer_seal_sha256: str
    training_batch_identity: str
    bridge_payload_sha256: str
    prime_payload_sha256: str
    prime_runtime_sha256: str
    trainer_config_sha256: str
    base_snapshot_manifest_sha256: str
    authorization_timeout_seconds: int

    def __post_init__(self) -> None:
        for name in (
            "producer_seal_sha256",
            "training_batch_identity",
            "bridge_payload_sha256",
            "prime_payload_sha256",
            "prime_runtime_sha256",
            "trainer_config_sha256",
            "base_snapshot_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.authorization_timeout_seconds) is not int or not (
            30 <= self.authorization_timeout_seconds <= 1800
        ):
            raise ValueError("authorization timeout must be an integer from 30 to 1800 seconds")

    def to_payload(self) -> dict[str, str | int]:
        return {
            "producer_seal_sha256": self.producer_seal_sha256,
            "training_batch_identity": self.training_batch_identity,
            "bridge_payload_sha256": self.bridge_payload_sha256,
            "prime_payload_sha256": self.prime_payload_sha256,
            "prime_runtime_sha256": self.prime_runtime_sha256,
            "trainer_config_sha256": self.trainer_config_sha256,
            "base_snapshot_manifest_sha256": self.base_snapshot_manifest_sha256,
            "authorization_timeout_seconds": self.authorization_timeout_seconds,
        }

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": _BINDING_DOMAIN,
                "binding": self.to_payload(),
            }
        )

    @classmethod
    def verify_bytes(cls, value: bytes) -> LiveUpdateBinding:
        payload = _canonical_object(value, "live update binding")
        _strict_keys(payload, {"schema_version", "domain", "binding"}, "binding envelope")
        if payload["schema_version"] != SCHEMA_VERSION or payload["domain"] != _BINDING_DOMAIN:
            raise ValueError("unsupported live update binding")
        body = payload["binding"]
        if not isinstance(body, dict):
            raise ValueError("live update binding body must be an object")
        _strict_keys(body, set(cls.__dataclass_fields__), "binding")
        return cls(**body)


@dataclass(frozen=True, slots=True)
class TrainerPrestate:
    binding_sha256: str
    nonce: str
    pre_model_sha256: str
    pre_optimizer_sha256: str
    trainable_names: tuple[str, ...]
    trainable_parameter_count: int

    def __post_init__(self) -> None:
        for name in ("binding_sha256", "pre_model_sha256", "pre_optimizer_sha256"):
            _require_sha256(getattr(self, name), name)
        if len(self.nonce) != 64 or any(
            character not in "0123456789abcdef" for character in self.nonce
        ):
            raise ValueError("prestate nonce must contain 32 random bytes")
        if (
            not self.trainable_names
            or tuple(sorted(set(self.trainable_names))) != self.trainable_names
        ):
            raise ValueError("trainable names must be sorted, unique, and nonempty")
        if type(self.trainable_parameter_count) is not int or self.trainable_parameter_count < 1:
            raise ValueError("trainable parameter count must be positive")

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": _PRESTATE_DOMAIN,
                "binding_sha256": self.binding_sha256,
                "nonce": self.nonce,
                "pre_model_sha256": self.pre_model_sha256,
                "pre_optimizer_sha256": self.pre_optimizer_sha256,
                "trainable_names": list(self.trainable_names),
                "trainable_parameter_count": self.trainable_parameter_count,
            }
        )

    @classmethod
    def verify_bytes(cls, value: bytes) -> TrainerPrestate:
        payload = _canonical_object(value, "trainer prestate")
        expected = {
            "schema_version",
            "domain",
            "binding_sha256",
            "nonce",
            "pre_model_sha256",
            "pre_optimizer_sha256",
            "trainable_names",
            "trainable_parameter_count",
        }
        _strict_keys(payload, expected, "trainer prestate")
        if payload["schema_version"] != SCHEMA_VERSION or payload["domain"] != _PRESTATE_DOMAIN:
            raise ValueError("unsupported trainer prestate")
        names = payload["trainable_names"]
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError("trainer prestate names are invalid")
        return cls(
            payload["binding_sha256"],
            payload["nonce"],
            payload["pre_model_sha256"],
            payload["pre_optimizer_sha256"],
            tuple(names),
            payload["trainable_parameter_count"],
        )


@dataclass(frozen=True, slots=True)
class LiveAuthorizationToken:
    binding_sha256: str
    nonce: str
    ledger_id: str
    authorization_sha256: str
    consumer_id: str
    pre_model_sha256: str
    pre_optimizer_sha256: str

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": _AUTH_DOMAIN,
                **self._payload(),
            }
        )

    def _payload(self) -> dict[str, str]:
        return {
            "binding_sha256": self.binding_sha256,
            "nonce": self.nonce,
            "ledger_id": self.ledger_id,
            "authorization_sha256": self.authorization_sha256,
            "consumer_id": self.consumer_id,
            "pre_model_sha256": self.pre_model_sha256,
            "pre_optimizer_sha256": self.pre_optimizer_sha256,
        }

    @classmethod
    def verify_bytes(cls, value: bytes) -> LiveAuthorizationToken:
        payload = _canonical_object(value, "live authorization token")
        fields = {
            "binding_sha256",
            "nonce",
            "ledger_id",
            "authorization_sha256",
            "consumer_id",
            "pre_model_sha256",
            "pre_optimizer_sha256",
        }
        _strict_keys(payload, {"schema_version", "domain", *fields}, "authorization token")
        if payload["schema_version"] != SCHEMA_VERSION or payload["domain"] != _AUTH_DOMAIN:
            raise ValueError("unsupported live authorization token")
        values = {name: payload[name] for name in fields}
        if any(not isinstance(value, str) or not value for value in values.values()):
            raise ValueError("live authorization fields must be nonempty strings")
        for name in (
            "binding_sha256",
            "authorization_sha256",
            "pre_model_sha256",
            "pre_optimizer_sha256",
        ):
            _require_sha256(values[name], name)
        if len(values["nonce"]) != 64:
            raise ValueError("authorization nonce is invalid")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class TrainerPoststep:
    binding_sha256: str
    prestate_sha256: str
    authorization_sha256: str
    post_model_sha256: str
    post_optimizer_sha256: str
    optimizer_step: int
    gradient_l2: float

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": _POSTSTEP_DOMAIN,
                "binding_sha256": self.binding_sha256,
                "prestate_sha256": self.prestate_sha256,
                "authorization_sha256": self.authorization_sha256,
                "post_model_sha256": self.post_model_sha256,
                "post_optimizer_sha256": self.post_optimizer_sha256,
                "optimizer_step": self.optimizer_step,
                "gradient_l2": self.gradient_l2,
            }
        )

    @classmethod
    def verify_bytes(cls, value: bytes) -> TrainerPoststep:
        payload = _canonical_object(value, "trainer poststep")
        fields = {
            "binding_sha256",
            "prestate_sha256",
            "authorization_sha256",
            "post_model_sha256",
            "post_optimizer_sha256",
            "optimizer_step",
            "gradient_l2",
        }
        _strict_keys(payload, {"schema_version", "domain", *fields}, "trainer poststep")
        if payload["schema_version"] != SCHEMA_VERSION or payload["domain"] != _POSTSTEP_DOMAIN:
            raise ValueError("unsupported trainer poststep")
        result = cls(**{name: payload[name] for name in fields})
        for name in (
            "binding_sha256",
            "prestate_sha256",
            "authorization_sha256",
            "post_model_sha256",
            "post_optimizer_sha256",
        ):
            _require_sha256(getattr(result, name), name)
        if result.optimizer_step != 1:
            raise ValueError("trainer poststep must prove optimizer step one")
        if (
            type(result.gradient_l2) is not float
            or not math.isfinite(result.gradient_l2)
            or result.gradient_l2 <= 0
        ):
            raise ValueError("trainer poststep gradient must be finite and nonzero")
        return result


class LiveUpdateTrainerGate:
    """Trainer-owned pre/post state recorder blocked on an external authorization."""

    def __init__(self, model: Any, optimizer: Any, binding: LiveUpdateBinding, root: Path) -> None:
        if not root.is_dir() or any(root.iterdir()):
            raise ValueError("live update receipt directory must exist and be empty")
        self._model = model
        self._optimizer = optimizer
        self._binding = binding
        self._root = root
        self._binding_sha256 = _sha256(binding.to_bytes())
        trainable = _trainable_parameters(model)
        adapter = _exported_adapter_parameters()
        if sum(int(value.numel()) for _, value in adapter) != sum(
            int(value.numel()) for _, value in trainable
        ):
            raise ValueError("exported adapter and trainable LoRA parameter counts differ")
        self._prestate = TrainerPrestate(
            self._binding_sha256,
            secrets.token_hex(32),
            _model_sha256(adapter, binding.base_snapshot_manifest_sha256),
            _optimizer_sha256(optimizer, trainable),
            tuple(name for name, _ in trainable),
            sum(int(parameter.numel()) for _, parameter in trainable),
        )
        self._authorization: LiveAuthorizationToken | None = None
        self._poststep_written = False

    @property
    def prestate(self) -> TrainerPrestate:
        return self._prestate

    def publish_and_wait(self) -> None:
        prestate_bytes = self._prestate.to_bytes()
        _atomic_write(self._root / "prestate.json", prestate_bytes)
        deadline = time.monotonic() + self._binding.authorization_timeout_seconds
        authorization_path = self._root / "authorization.json"
        while not authorization_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("live update authorization timed out before optimizer execution")
            time.sleep(0.1)
        token = LiveAuthorizationToken.verify_bytes(authorization_path.read_bytes())
        expected = (
            self._binding_sha256,
            self._prestate.nonce,
            self._prestate.pre_model_sha256,
            self._prestate.pre_optimizer_sha256,
        )
        observed = (
            token.binding_sha256,
            token.nonce,
            token.pre_model_sha256,
            token.pre_optimizer_sha256,
        )
        if observed != expected:
            raise ValueError("live authorization token differs from the trainer prestate")
        self._authorization = token

    def record_optimizer_step(self, *, optimizer_step: int, gradient_l2: float) -> None:
        if self._authorization is None:
            raise RuntimeError("optimizer step occurred before live authorization")
        if self._poststep_written:
            raise RuntimeError("live update gate observed a second optimizer step")
        if optimizer_step != 1:
            raise ValueError("bounded live update permits only optimizer step one")
        trainable = _trainable_parameters(self._model)
        adapter = _exported_adapter_parameters()
        post_model = _model_sha256(adapter, self._binding.base_snapshot_manifest_sha256)
        post_optimizer = _optimizer_sha256(self._optimizer, trainable)
        if post_model == self._prestate.pre_model_sha256:
            raise ValueError("optimizer step did not change the LoRA state")
        if post_optimizer == self._prestate.pre_optimizer_sha256:
            raise ValueError("optimizer step did not change the optimizer state")
        poststep = TrainerPoststep(
            self._binding_sha256,
            _sha256(self._prestate.to_bytes()),
            self._authorization.authorization_sha256,
            post_model,
            post_optimizer,
            optimizer_step,
            float(gradient_l2),
        )
        _atomic_write(self._root / "poststep.json", poststep.to_bytes())
        self._poststep_written = True


def start_live_update_gate(
    model: Any,
    optimizer: Any,
    *,
    world_size: int,
    rank: int,
    max_steps: int | None,
) -> LiveUpdateTrainerGate | None:
    """Start the default-off gate and block until a local controller authorizes."""
    binding_path = os.environ.get(_ENV_BINDING)
    receipt_root = os.environ.get(_ENV_RECEIPTS)
    if binding_path is None and receipt_root is None:
        return None
    if not binding_path or not receipt_root:
        raise ValueError("both live update gate environment paths are required")
    if world_size != 1 or rank != 0 or max_steps != 1:
        raise ValueError(
            "bounded live update gate requires rank 0 of exactly one process and one step"
        )
    binding = LiveUpdateBinding.verify_bytes(Path(binding_path).read_bytes())
    gate = LiveUpdateTrainerGate(model, optimizer, binding, Path(receipt_root))
    gate.publish_and_wait()
    return gate


def authorize_live_update(
    *,
    binding_bytes: bytes,
    prestate_bytes: bytes,
    ledger_root: Path,
    consumer_id: str,
) -> bytes:
    """Create and durably authorize the local ledger from a remote prestate."""
    binding = LiveUpdateBinding.verify_bytes(binding_bytes)
    prestate = TrainerPrestate.verify_bytes(prestate_bytes)
    if prestate.binding_sha256 != _sha256(binding_bytes):
        raise ValueError("trainer prestate differs from the frozen live binding")
    ledger_binding = UpdateLedgerBinding(
        binding.producer_seal_sha256,
        binding.training_batch_identity,
        binding.bridge_payload_sha256,
        binding.prime_payload_sha256,
        binding.prime_runtime_sha256,
        binding.trainer_config_sha256,
        prestate.pre_model_sha256,
    )
    with SingleUseUpdateLedger.create(ledger_root, binding=ledger_binding) as ledger:
        authorization = ledger.authorize(
            consumer_id=consumer_id,
            pre_model_sha256=prestate.pre_model_sha256,
            pre_optimizer_sha256=prestate.pre_optimizer_sha256,
        )
    return LiveAuthorizationToken(
        prestate.binding_sha256,
        prestate.nonce,
        authorization.ledger_id,
        authorization.authorization_sha256,
        authorization.consumer_id,
        authorization.pre_model_sha256,
        authorization.pre_optimizer_sha256,
    ).to_bytes()


def complete_live_update(
    *,
    binding_bytes: bytes,
    prestate_bytes: bytes,
    authorization_bytes: bytes,
    poststep_bytes: bytes,
    ledger_root: Path,
) -> UpdateCompletion:
    """Verify the trainer post-state and immediately seal the local ledger."""
    LiveUpdateBinding.verify_bytes(binding_bytes)
    prestate = TrainerPrestate.verify_bytes(prestate_bytes)
    token = LiveAuthorizationToken.verify_bytes(authorization_bytes)
    poststep = TrainerPoststep.verify_bytes(poststep_bytes)
    if prestate.binding_sha256 != _sha256(binding_bytes):
        raise ValueError("prestate differs from the live update binding")
    if (
        token.binding_sha256 != prestate.binding_sha256
        or token.nonce != prestate.nonce
        or token.pre_model_sha256 != prestate.pre_model_sha256
        or token.pre_optimizer_sha256 != prestate.pre_optimizer_sha256
    ):
        raise ValueError("authorization token differs from the trainer prestate")
    if (
        poststep.binding_sha256 != prestate.binding_sha256
        or poststep.prestate_sha256 != _sha256(prestate_bytes)
        or poststep.authorization_sha256 != token.authorization_sha256
        or poststep.post_model_sha256 == prestate.pre_model_sha256
        or poststep.post_optimizer_sha256 == prestate.pre_optimizer_sha256
    ):
        raise ValueError("trainer poststep differs from its authorized prestate")
    authorization = UpdateAuthorization(
        token.ledger_id,
        token.authorization_sha256,
        token.consumer_id,
        token.pre_model_sha256,
        token.pre_optimizer_sha256,
    )
    with SingleUseUpdateLedger(ledger_root) as ledger:
        return ledger.complete(
            authorization,
            post_model_sha256=poststep.post_model_sha256,
            post_optimizer_sha256=poststep.post_optimizer_sha256,
            step_evidence_sha256=_sha256(poststep_bytes),
        )


def _trainable_parameters(model: Any) -> tuple[tuple[str, Any], ...]:
    named = tuple(
        sorted(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    )
    if not named:
        raise ValueError("live update model has no trainable parameters")
    unexpected = [name for name, _ in named if "lora_A" not in name and "lora_B" not in name]
    if unexpected:
        raise ValueError(f"live update found non-LoRA trainables: {unexpected[:3]}")
    if len({id(parameter) for _, parameter in named}) != len(named):
        raise ValueError("live update trainable parameter names are not one-to-one")
    return named


def _model_sha256(named: tuple[tuple[str, Any], ...], base_manifest_sha256: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(
        canonical_json(
            {
                "domain": "redco-stage-d-live-lora-state-v1",
                "base_snapshot_manifest_sha256": base_manifest_sha256,
            }
        )
    )
    for name, tensor in named:
        _update_tensor_hash(hasher, name, tensor)
    return hasher.hexdigest()


def adapter_file_state_sha256(
    adapter_path: Path,
    *,
    base_snapshot_manifest_sha256: str,
) -> str:
    """Hash a retained safetensors adapter using the trainer's model-state domain."""
    _require_sha256(base_snapshot_manifest_sha256, "base_snapshot_manifest_sha256")
    safetensors = importlib.import_module("safetensors")
    values = []
    with safetensors.safe_open(adapter_path, framework="pt", device="cpu") as handle:
        for name in sorted(handle.keys()):
            values.append((name, handle.get_tensor(name)))
    if not values:
        raise ValueError("retained adapter contains no tensors")
    return _model_sha256(tuple(values), base_snapshot_manifest_sha256)


def _exported_adapter_parameters() -> tuple[tuple[str, Any], ...]:
    runs = importlib.import_module("prime_rl.trainer.runs")
    state = runs.get_multi_run_manager().get_state_dict_for_run(0)
    values = tuple(
        sorted((f"base_model.model.{name}", tensor) for name, tensor in state.items())
    )
    if not values:
        raise ValueError("Prime exported adapter state is empty")
    return values


def _optimizer_sha256(optimizer: Any, named: tuple[tuple[str, Any], ...]) -> str:
    hasher = hashlib.sha256()
    hasher.update(canonical_json({"domain": "redco-stage-d-live-optimizer-state-v1"}))
    names_by_id = {id(parameter): name for name, parameter in named}
    groups = []
    for group in optimizer.param_groups:
        parameters = group.get("params")
        if not isinstance(parameters, list):
            parameters = list(parameters)
        names = []
        for parameter in parameters:
            name = names_by_id.get(id(parameter))
            if name is None:
                raise ValueError("optimizer contains a parameter outside the named LoRA state")
            names.append(name)
        config = {key: _portable(value) for key, value in group.items() if key != "params"}
        groups.append({"parameters": sorted(names), "config": config})
    hasher.update(canonical_json(groups))
    state = optimizer.state
    entries = []
    for parameter, values in state.items():
        name = names_by_id.get(id(parameter))
        if name is None:
            raise ValueError("optimizer state contains an unnamed parameter")
        entries.append((name, values))
    for name, values in sorted(entries):
        hasher.update(canonical_json({"parameter": name}))
        if not isinstance(values, dict):
            raise ValueError("optimizer parameter state must be a mapping")
        for key in sorted(values, key=str):
            value = values[key]
            label = f"{name}:{key}"
            if hasattr(value, "detach"):
                _update_tensor_hash(hasher, label, value)
            else:
                hasher.update(canonical_json({"name": label, "value": _portable(value)}))
    return hasher.hexdigest()


def _update_tensor_hash(hasher: Any, name: str, tensor: Any) -> None:
    value = tensor.detach()
    if hasattr(value, "to_local"):
        value = value.to_local()
    value = value.cpu().contiguous()
    hasher.update(
        canonical_json(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(tensor.shape),
                "local_shape": list(value.shape),
            }
        )
    )
    torch = importlib.import_module("torch")
    hasher.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def _portable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _portable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    return str(value)


def _canonical_object(value: bytes, label: str) -> dict[str, Any]:
    if type(value) is not bytes:
        raise ValueError(f"{label} must be immutable bytes")
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or canonical_json(parsed) != value:
        raise ValueError(f"{label} must be canonical JSON")
    return parsed


def _strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields differ from the frozen schema")


def _atomic_write(path: Path, value: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace live update receipt: {path.name}")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value
