"""Separate durable authorization ledger for one trainer update attempt."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json

SCHEMA_VERSION = 1
_DOMAIN = "redco-stage-d-update-ledger-v1"


class UpdateLedgerError(RuntimeError):
    pass


class UpdateAlreadyAuthorized(UpdateLedgerError):
    pass


@dataclass(frozen=True, slots=True)
class UpdateLedgerBinding:
    producer_seal_sha256: str
    training_batch_identity: str
    bridge_payload_sha256: str
    prime_payload_sha256: str
    prime_runtime_sha256: str
    trainer_config_sha256: str
    expected_input_policy_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "producer_seal_sha256",
            "training_batch_identity",
            "bridge_payload_sha256",
            "prime_payload_sha256",
            "prime_runtime_sha256",
            "trainer_config_sha256",
            "expected_input_policy_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    def to_payload(self) -> dict[str, str]:
        return {
            "producer_seal_sha256": self.producer_seal_sha256,
            "training_batch_identity": self.training_batch_identity,
            "bridge_payload_sha256": self.bridge_payload_sha256,
            "prime_payload_sha256": self.prime_payload_sha256,
            "prime_runtime_sha256": self.prime_runtime_sha256,
            "trainer_config_sha256": self.trainer_config_sha256,
            "expected_input_policy_sha256": self.expected_input_policy_sha256,
        }


@dataclass(frozen=True, slots=True)
class UpdateAuthorization:
    ledger_id: str
    authorization_sha256: str
    consumer_id: str
    pre_model_sha256: str
    pre_optimizer_sha256: str


@dataclass(frozen=True, slots=True)
class UpdateCompletion:
    ledger_id: str
    completion_sha256: str
    post_model_sha256: str
    post_optimizer_sha256: str
    step_evidence_sha256: str


class SingleUseUpdateLedger:
    """Four-record ledger: genesis, authorization, completion, and seal."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock_path = root / "writer.lock"
        self._lock_descriptor: int | None = None
        self._acquire_lock()
        try:
            self._records = _load_records(root)
            self._validate()
        except BaseException:
            self.close()
            raise

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        binding: UpdateLedgerBinding,
    ) -> SingleUseUpdateLedger:
        root.mkdir(parents=True, exist_ok=False)
        (root / "records").mkdir()
        ledger_id = secrets.token_hex(16)
        genesis = {
            "schema_version": SCHEMA_VERSION,
            "domain": _DOMAIN,
            "ledger_id": ledger_id,
            "sequence": 0,
            "prior_sha256": "0" * 64,
            "record_kind": "genesis",
            "body": binding.to_payload(),
        }
        _atomic_write(root / "records" / "00000000.json", canonical_json(genesis))
        return cls(root)

    @classmethod
    def inspect_status(cls, root: Path) -> str:
        """Validate and report durable state without acquiring the writer lock."""
        reader = object.__new__(cls)
        reader.root = root
        reader._lock_path = root / "writer.lock"
        reader._lock_descriptor = None
        reader._records = _load_records(root)
        reader._validate()
        return reader.status

    @property
    def binding(self) -> UpdateLedgerBinding:
        body = self._records[0]["body"]
        assert isinstance(body, dict)
        return UpdateLedgerBinding(**body)

    @property
    def ledger_id(self) -> str:
        return str(self._records[0]["ledger_id"])

    @property
    def status(self) -> str:
        return (
            "ready",
            "authorized-incomplete",
            "completed-unsealed",
            "complete",
        )[len(self._records) - 1]

    def authorize(
        self,
        *,
        consumer_id: str,
        pre_model_sha256: str,
        pre_optimizer_sha256: str,
    ) -> UpdateAuthorization:
        self._require_open()
        if len(self._records) != 1:
            raise UpdateAlreadyAuthorized(self.binding.training_batch_identity)
        if not consumer_id:
            raise ValueError("consumer_id must be nonempty")
        if pre_model_sha256 != self.binding.expected_input_policy_sha256:
            raise ValueError("pre-update model differs from the frozen input policy")
        _require_sha256(pre_optimizer_sha256, "pre_optimizer_sha256")
        record = self._append(
            "update_authorized",
            {
                "consumer_id": consumer_id,
                "pre_model_sha256": pre_model_sha256,
                "pre_optimizer_sha256": pre_optimizer_sha256,
                "single_use_attempt": True,
            },
        )
        return UpdateAuthorization(
            self.ledger_id,
            _sha256(canonical_json(record)),
            consumer_id,
            pre_model_sha256,
            pre_optimizer_sha256,
        )

    def complete(
        self,
        authorization: UpdateAuthorization,
        *,
        post_model_sha256: str,
        post_optimizer_sha256: str,
        step_evidence_sha256: str,
    ) -> UpdateCompletion:
        self._require_open()
        if len(self._records) != 2:
            raise UpdateLedgerError("one incomplete authorization is required")
        if authorization.ledger_id != self.ledger_id:
            raise ValueError("authorization belongs to a different update ledger")
        if authorization.authorization_sha256 != _sha256(canonical_json(self._records[1])):
            raise ValueError("authorization receipt differs from the durable record")
        durable_body = _body(self._records[1], "authorization")
        durable_authorization = UpdateAuthorization(
            self.ledger_id,
            authorization.authorization_sha256,
            str(durable_body["consumer_id"]),
            str(durable_body["pre_model_sha256"]),
            str(durable_body["pre_optimizer_sha256"]),
        )
        if authorization != durable_authorization:
            raise ValueError("authorization fields differ from the durable record")
        for value, name in (
            (post_model_sha256, "post_model_sha256"),
            (post_optimizer_sha256, "post_optimizer_sha256"),
            (step_evidence_sha256, "step_evidence_sha256"),
        ):
            _require_sha256(value, name)
        if post_model_sha256 == authorization.pre_model_sha256:
            raise ValueError("successful update must change the model state hash")
        record = self._append(
            "update_completed",
            {
                "authorization_sha256": authorization.authorization_sha256,
                "post_model_sha256": post_model_sha256,
                "post_optimizer_sha256": post_optimizer_sha256,
                "step_evidence_sha256": step_evidence_sha256,
                "successful_optimizer_steps": 1,
            },
        )
        completion = UpdateCompletion(
            self.ledger_id,
            _sha256(canonical_json(record)),
            post_model_sha256,
            post_optimizer_sha256,
            step_evidence_sha256,
        )
        self._append(
            "ledger_sealed",
            {
                "completion_sha256": completion.completion_sha256,
                "record_count": 4,
            },
        )
        return completion

    def run_once(
        self,
        *,
        consumer_id: str,
        pre_model_sha256: str,
        pre_optimizer_sha256: str,
        update: Callable[[], tuple[str, str, str]],
    ) -> UpdateCompletion:
        """Authorize once, invoke one callback, then record successful completion."""
        authorization = self.authorize(
            consumer_id=consumer_id,
            pre_model_sha256=pre_model_sha256,
            pre_optimizer_sha256=pre_optimizer_sha256,
        )
        post_model, post_optimizer, evidence = update()
        return self.complete(
            authorization,
            post_model_sha256=post_model,
            post_optimizer_sha256=post_optimizer,
            step_evidence_sha256=evidence,
        )

    def close(self) -> None:
        if self._lock_descriptor is not None:
            os.close(self._lock_descriptor)
            self._lock_descriptor = None
            self._lock_path.unlink(missing_ok=True)

    def __enter__(self) -> SingleUseUpdateLedger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _append(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        prior = self._records[-1]
        record = {
            "schema_version": SCHEMA_VERSION,
            "domain": _DOMAIN,
            "ledger_id": self.ledger_id,
            "sequence": len(self._records),
            "prior_sha256": _sha256(canonical_json(prior)),
            "record_kind": kind,
            "body": body,
        }
        path = self.root / "records" / f"{len(self._records):08d}.json"
        _atomic_write(path, canonical_json(record))
        self._records.append(record)
        return record

    def _validate(self) -> None:
        if not 1 <= len(self._records) <= 4:
            raise UpdateLedgerError("update ledger has an invalid record count")
        for sequence, record in enumerate(self._records):
            if set(record) != {
                "schema_version",
                "domain",
                "ledger_id",
                "sequence",
                "prior_sha256",
                "record_kind",
                "body",
            }:
                raise UpdateLedgerError("update ledger record envelope fields are invalid")
            if (
                record.get("schema_version") != SCHEMA_VERSION
                or record.get("domain") != _DOMAIN
                or record.get("sequence") != sequence
                or record.get("ledger_id") != self._records[0].get("ledger_id")
            ):
                raise UpdateLedgerError("update ledger record envelope is invalid")
            expected_prior = (
                "0" * 64 if sequence == 0 else _sha256(canonical_json(self._records[sequence - 1]))
            )
            if record.get("prior_sha256") != expected_prior:
                raise UpdateLedgerError("update ledger hash chain is invalid")
        if self._records[0].get("record_kind") != "genesis":
            raise UpdateLedgerError("update ledger genesis is missing")
        body = self._records[0].get("body")
        if not isinstance(body, dict):
            raise UpdateLedgerError("update ledger binding is invalid")
        try:
            _strict_keys(
                body,
                {
                    "producer_seal_sha256",
                    "training_batch_identity",
                    "bridge_payload_sha256",
                    "prime_payload_sha256",
                    "prime_runtime_sha256",
                    "trainer_config_sha256",
                    "expected_input_policy_sha256",
                },
                "genesis",
            )
            binding = UpdateLedgerBinding(**body)
            if len(self._records) >= 2:
                self._validate_authorization(binding)
            if len(self._records) >= 3:
                self._validate_completion()
            if len(self._records) == 4:
                self._validate_seal()
        except (TypeError, ValueError) as error:
            raise UpdateLedgerError("update ledger record body is invalid") from error

    def _validate_authorization(self, binding: UpdateLedgerBinding) -> None:
        record = self._records[1]
        if record.get("record_kind") != "update_authorized":
            raise UpdateLedgerError("update authorization record is invalid")
        body = _body(record, "authorization")
        _strict_keys(
            body,
            {
                "consumer_id",
                "pre_model_sha256",
                "pre_optimizer_sha256",
                "single_use_attempt",
            },
            "authorization",
        )
        if not isinstance(body["consumer_id"], str) or not body["consumer_id"]:
            raise ValueError("authorization consumer must be nonempty")
        if body["pre_model_sha256"] != binding.expected_input_policy_sha256:
            raise ValueError("authorization input policy differs from genesis")
        _require_sha256(body["pre_optimizer_sha256"], "pre_optimizer_sha256")
        if body["single_use_attempt"] is not True:
            raise ValueError("authorization must declare one attempt")

    def _validate_completion(self) -> None:
        record = self._records[2]
        if record.get("record_kind") != "update_completed":
            raise UpdateLedgerError("update completion record is invalid")
        body = _body(record, "completion")
        _strict_keys(
            body,
            {
                "authorization_sha256",
                "post_model_sha256",
                "post_optimizer_sha256",
                "step_evidence_sha256",
                "successful_optimizer_steps",
            },
            "completion",
        )
        if body["authorization_sha256"] != _sha256(canonical_json(self._records[1])):
            raise ValueError("completion does not bind its authorization")
        for name in (
            "post_model_sha256",
            "post_optimizer_sha256",
            "step_evidence_sha256",
        ):
            _require_sha256(body[name], name)
        authorization = _body(self._records[1], "authorization")
        if body["post_model_sha256"] == authorization["pre_model_sha256"]:
            raise ValueError("completion does not record a changed model")
        if body["successful_optimizer_steps"] != 1:
            raise ValueError("completion must record exactly one optimizer step")

    def _validate_seal(self) -> None:
        record = self._records[3]
        if record.get("record_kind") != "ledger_sealed":
            raise UpdateLedgerError("update ledger seal is invalid")
        body = _body(record, "seal")
        _strict_keys(body, {"completion_sha256", "record_count"}, "seal")
        if body["completion_sha256"] != _sha256(canonical_json(self._records[2])):
            raise ValueError("seal does not bind its completion")
        if body["record_count"] != 4:
            raise ValueError("seal record count is invalid")

    def _acquire_lock(self) -> None:
        try:
            self._lock_descriptor = os.open(
                self._lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as error:
            raise UpdateLedgerError("update ledger already has an active writer") from error

    def _require_open(self) -> None:
        if self._lock_descriptor is None:
            raise UpdateLedgerError("update ledger writer is closed")


def _load_records(root: Path) -> list[dict[str, Any]]:
    records_root = root / "records"
    if not records_root.is_dir():
        raise UpdateLedgerError("update ledger records directory is missing")
    paths = sorted(records_root.glob("*.json"))
    if not paths:
        raise UpdateLedgerError("update ledger has no genesis")
    expected_names = [f"{index:08d}.json" for index in range(len(paths))]
    if [path.name for path in paths] != expected_names:
        raise UpdateLedgerError("update ledger record sequence has gaps")
    records: list[dict[str, Any]] = []
    for path in paths:
        value = path.read_bytes()
        parsed = json.loads(value)
        if not isinstance(parsed, dict) or canonical_json(parsed) != value:
            raise UpdateLedgerError("update ledger record is not canonical JSON")
        records.append(parsed)
    return records


def _body(record: dict[str, Any], label: str) -> dict[str, Any]:
    value = record.get("body")
    if not isinstance(value, dict):
        raise ValueError(f"{label} body must be an object")
    return value


def _strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} body fields differ from the frozen schema")


def _atomic_write(path: Path, value: bytes) -> None:
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
