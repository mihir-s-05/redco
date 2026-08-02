"""Immutable scientific-arm batch contracts for Stage D."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from redco.analysis.stage_d_objective_binding import ArmName, ObjectiveBinding
from redco.contracts import canonical_json

SCHEMA_VERSION = 1
_DOMAIN = "redco-stage-d-three-arm-bridge-v1"
RecordKind = Literal["stock-trajectory", "untargeted-decision", "target-branch"]


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


def _finite(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_from_payload(value: object, name: str) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError(f"{name} must be an exact fraction")
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise ValueError(f"{name} must be an exact fraction")
    return Fraction(numerator, denominator)


@dataclass(frozen=True, slots=True)
class ArmTrainerRecord:
    arm: ArmName
    record_kind: RecordKind
    source_sha256: str
    group_id: str
    rollout_id: str
    decision_id: str | None
    target_id: str | None
    action_slot: int | None
    token_ids: tuple[int, ...]
    mask: tuple[bool, ...]
    behavior_logprobs: tuple[float, ...]
    temperatures: tuple[float, ...]
    advantages: tuple[float, ...]
    rl_weights: tuple[float, ...] | None
    rl_normalizer: Fraction | None

    def __post_init__(self) -> None:
        _require_sha256(self.source_sha256, "record source sha256")
        if self.arm not in {"stock", "branch-global", "local"}:
            raise ValueError("unsupported record arm")
        if self.record_kind not in {
            "stock-trajectory",
            "untargeted-decision",
            "target-branch",
        }:
            raise ValueError("unsupported training record kind")
        if not self.group_id or not self.rollout_id:
            raise ValueError("record identifiers must be nonempty")
        length = len(self.token_ids)
        streams: list[Sequence[object]] = [
            self.mask,
            self.behavior_logprobs,
            self.temperatures,
            self.advantages,
        ]
        if self.rl_weights is not None:
            streams.append(self.rl_weights)
        if length == 0 or any(len(stream) != length for stream in streams):
            raise ValueError("record streams must be nonempty and aligned")
        if any(type(token) is not int or token < 0 for token in self.token_ids):
            raise ValueError("record token IDs must be nonnegative integers")
        if any(type(selected) is not bool for selected in self.mask):
            raise ValueError("record mask values must be exact booleans")
        numeric_streams = [
            self.behavior_logprobs,
            self.temperatures,
            self.advantages,
        ]
        if self.rl_weights is not None:
            numeric_streams.append(self.rl_weights)
        if any(
            type(value) is not float or not math.isfinite(value)
            for stream in numeric_streams
            for value in stream
        ):
            raise ValueError("record numeric streams must contain finite floats")
        if not any(self.mask):
            raise ValueError("record must select tokens")
        if self.rl_normalizer is not None and self.rl_normalizer <= 0:
            raise ValueError("record normalizer must be positive when present")
        weights = self.rl_weights or tuple(1.0 if selected else 0.0 for selected in self.mask)
        for selected, logprob, temperature, advantage, weight in zip(
            self.mask,
            self.behavior_logprobs,
            self.temperatures,
            self.advantages,
            weights,
            strict=True,
        ):
            if temperature <= 0.0:
                raise ValueError("record temperatures must be positive")
            if not selected and (logprob != 0.0 or advantage != 0.0 or weight != 0.0):
                raise ValueError("unselected record positions must carry zero training fields")
            if selected and (logprob > 0.0 or weight <= 0.0):
                raise ValueError("selected record positions have invalid logprobs or weights")
        if self.record_kind == "stock-trajectory":
            if self.arm != "stock" or any(
                value is not None for value in (self.decision_id, self.target_id, self.action_slot)
            ):
                raise ValueError("stock records must be whole trajectories")
            if self.rl_weights is not None or self.rl_normalizer is not None:
                raise ValueError("stock records must use exact Prime token normalization")
        elif self.record_kind == "untargeted-decision":
            if self.arm == "stock" or not self.decision_id or self.target_id is not None:
                raise ValueError("untargeted records require one branch-arm decision")
            if self.action_slot is not None:
                raise ValueError("untargeted records cannot have an action slot")
            if self.rl_weights is None or self.rl_normalizer is None:
                raise ValueError("untargeted records require explicit decision weights")
        elif self.record_kind == "target-branch":
            if (
                self.arm == "stock"
                or not self.decision_id
                or not self.target_id
                or type(self.action_slot) is not int
                or self.action_slot < 0
            ):
                raise ValueError("target records require decision, target, and action slot")
            if self.rl_weights is None or self.rl_normalizer is None:
                raise ValueError("target records require explicit decision weights")
        else:
            raise ValueError("unsupported training record kind")
        if self.record_kind != "stock-trajectory":
            assert self.rl_weights is not None
            assert self.rl_normalizer is not None
            selected_weights = {
                weight
                for selected, weight in zip(self.mask, self.rl_weights, strict=True)
                if selected
            }
            if selected_weights != {float(self.rl_normalizer)}:
                raise ValueError(
                    "branch record numerator weight must equal its decision normalizer"
                )

    def to_payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "record_kind": self.record_kind,
            "source_sha256": self.source_sha256,
            "group_id": self.group_id,
            "rollout_id": self.rollout_id,
            "decision_id": self.decision_id,
            "target_id": self.target_id,
            "action_slot": self.action_slot,
            "token_ids": list(self.token_ids),
            "mask": list(self.mask),
            "behavior_logprobs": list(self.behavior_logprobs),
            "temperatures": list(self.temperatures),
            "advantages": list(self.advantages),
            "rl_weights": None if self.rl_weights is None else list(self.rl_weights),
            "rl_normalizer": (
                None if self.rl_normalizer is None else _fraction_payload(self.rl_normalizer)
            ),
        }


@dataclass(frozen=True, slots=True)
class SealedArmBatch:
    arm: ArmName
    records: tuple[ArmTrainerRecord, ...]
    source_sha256s: tuple[str, ...]
    branch_artifact_sha256s: tuple[str, ...]
    evidence_class: Literal["live", "fixture-only"]
    objective_binding: ObjectiveBinding
    policy_sha256: str
    trainer_step: int
    seq_len: int
    batch_identity: str

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("sealed arm batch must contain records")
        if self.arm not in {"stock", "branch-global", "local"}:
            raise ValueError("unsupported sealed arm")
        if self.evidence_class not in {"live", "fixture-only"}:
            raise ValueError("unsupported sealed evidence class")
        if (
            self.objective_binding.arm != self.arm
            or self.objective_binding.evidence_class != self.evidence_class
        ):
            raise ValueError("sealed batch objective binding differs from its arm")
        if any(record.arm != self.arm for record in self.records):
            raise ValueError("sealed arm batch mixes arms")
        if tuple(sorted(set(self.source_sha256s))) != self.source_sha256s:
            raise ValueError("source hashes must be sorted and unique")
        if tuple(sorted(set(self.branch_artifact_sha256s))) != self.branch_artifact_sha256s:
            raise ValueError("branch artifact hashes must be sorted and unique")
        for digest in (*self.source_sha256s, *self.branch_artifact_sha256s):
            _require_sha256(digest, "sealed source sha256")
        _require_sha256(self.policy_sha256, "sealed policy sha256")
        _require_sha256(self.batch_identity, "sealed batch identity")
        if type(self.trainer_step) is not int or self.trainer_step < 1:
            raise ValueError("trainer step must be positive")
        if type(self.seq_len) is not int or self.seq_len < 1:
            raise ValueError("sequence length must be positive")
        if any(len(record.token_ids) > self.seq_len for record in self.records):
            raise ValueError("sealed arm batch would truncate a record")
        record_sources = tuple(sorted({record.source_sha256 for record in self.records}))
        if record_sources != self.source_sha256s:
            raise ValueError("sealed arm batch source roster differs from its records")
        if self.arm == "stock":
            if self.branch_artifact_sha256s or any(
                record.record_kind != "stock-trajectory" for record in self.records
            ):
                raise ValueError("stock batch cannot contain branch evidence")
        elif not self.branch_artifact_sha256s or any(
            record.record_kind == "stock-trajectory" for record in self.records
        ):
            raise ValueError("branch batch requires branch evidence and decision records")
        if self.batch_identity != _batch_identity(
            self.arm,
            self.records,
            self.source_sha256s,
            self.branch_artifact_sha256s,
            self.evidence_class,
            self.objective_binding,
            self.policy_sha256,
            self.trainer_step,
            self.seq_len,
        ):
            raise ValueError("sealed arm batch identity mismatch")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "domain": _DOMAIN,
            "arm": self.arm,
            "records": [record.to_payload() for record in self.records],
            "source_sha256s": list(self.source_sha256s),
            "branch_artifact_sha256s": list(self.branch_artifact_sha256s),
            "evidence_class": self.evidence_class,
            "objective_binding": self.objective_binding.to_payload(),
            "policy_sha256": self.policy_sha256,
            "trainer_step": self.trainer_step,
            "seq_len": self.seq_len,
            "batch_identity": self.batch_identity,
        }

    def to_bytes(self) -> bytes:
        payload = self.to_payload()
        return canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": _DOMAIN,
                "payload": payload,
                "payload_sha256": _sha256(canonical_json(payload)),
            }
        )

    @classmethod
    def verify_bytes(cls, value: bytes) -> SealedArmBatch:
        if type(value) is not bytes:
            raise ValueError("sealed arm batch must be immutable bytes")
        envelope = json.loads(value)
        if not isinstance(envelope, dict) or canonical_json(envelope) != value:
            raise ValueError("sealed arm batch must be canonical JSON")
        if set(envelope) != {"schema_version", "domain", "payload", "payload_sha256"}:
            raise ValueError("sealed arm batch envelope fields differ")
        if envelope["schema_version"] != SCHEMA_VERSION or envelope["domain"] != _DOMAIN:
            raise ValueError("unsupported sealed arm batch")
        payload = envelope["payload"]
        if not isinstance(payload, dict):
            raise ValueError("sealed arm batch payload must be an object")
        if envelope["payload_sha256"] != _sha256(canonical_json(payload)):
            raise ValueError("sealed arm batch payload digest mismatch")
        batch = _batch_from_payload(payload)
        if batch.to_bytes() != value:
            raise ValueError("sealed arm batch derived fields disagree")
        return batch


@dataclass(frozen=True, slots=True)
class ThreeArmCompilation:
    stock: SealedArmBatch
    branch_global: SealedArmBatch
    local: SealedArmBatch
    common_branch_layout_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.common_branch_layout_sha256,
            "common branch layout sha256",
        )
        if (
            self.stock.arm,
            self.branch_global.arm,
            self.local.arm,
        ) != ("stock", "branch-global", "local"):
            raise ValueError("three-arm compilation has incorrect arm identities")
        if not (
            self.stock.evidence_class
            == self.branch_global.evidence_class
            == self.local.evidence_class
        ):
            raise ValueError("three-arm compilation mixes evidence classes")
        if self.stock.policy_sha256 != self.local.policy_sha256 or (
            self.branch_global.policy_sha256 != self.local.policy_sha256
        ):
            raise ValueError("three-arm compilation mixes policy snapshots")
        if not (
            self.stock.source_sha256s
            == self.branch_global.source_sha256s
            == self.local.source_sha256s
        ):
            raise ValueError("three-arm compilation mixes source rollout sets")
        if self.stock.branch_artifact_sha256s or (
            self.branch_global.branch_artifact_sha256s != self.local.branch_artifact_sha256s
        ):
            raise ValueError("three-arm compilation mixes branch artifact sets")
        if not (
            self.stock.trainer_step == self.branch_global.trainer_step == self.local.trainer_step
        ):
            raise ValueError("three-arm compilation mixes trainer steps")
        if not (self.stock.seq_len == self.branch_global.seq_len == self.local.seq_len):
            raise ValueError("three-arm compilation mixes sequence lengths")
        observed = _common_branch_layout_digest(
            self.branch_global.records,
            self.local.records,
        )
        if observed != self.common_branch_layout_sha256:
            raise ValueError("branch arms do not share one immutable data layout")


def _common_branch_layout_digest(
    branch_global: Sequence[ArmTrainerRecord],
    local: Sequence[ArmTrainerRecord],
) -> str:
    if len(branch_global) != len(local):
        raise ValueError("branch arms have different record counts")
    global_layout = [_branch_layout_payload(record) for record in branch_global]
    local_layout = [_branch_layout_payload(record) for record in local]
    if global_layout != local_layout:
        raise ValueError("branch arms differ outside their target advantages")
    for global_record, local_record in zip(branch_global, local, strict=True):
        if global_record.record_kind != "target-branch" and (
            global_record.advantages != local_record.advantages
        ):
            raise ValueError("untargeted branch-arm advantages must be identical")
    return _sha256(
        canonical_json(
            {
                "domain": "redco-stage-d-common-branch-layout-v1",
                "records": global_layout,
            }
        )
    )


def _branch_layout_payload(record: ArmTrainerRecord) -> dict[str, Any]:
    payload = record.to_payload()
    payload["arm"] = "shared-branch-arm"
    payload["advantages"] = [
        0.0 if selected else value
        for value, selected in zip(record.advantages, record.mask, strict=True)
    ]
    return payload


def _batch_identity(
    arm: ArmName,
    records: Sequence[ArmTrainerRecord],
    source_hashes: tuple[str, ...],
    artifact_hashes: tuple[str, ...],
    evidence_class: Literal["live", "fixture-only"],
    objective_binding: ObjectiveBinding,
    policy_sha256: str,
    trainer_step: int,
    seq_len: int,
) -> str:
    return _sha256(
        canonical_json(
            {
                "domain": "redco-stage-d-three-arm-batch-identity-v1",
                "arm": arm,
                "sources": list(source_hashes),
                "branch_artifacts": list(artifact_hashes),
                "evidence_class": evidence_class,
                "objective_sha256": objective_binding.objective_sha256,
                "policy_sha256": policy_sha256,
                "trainer_step": trainer_step,
                "seq_len": seq_len,
                "records_sha256": _sha256(
                    canonical_json([record.to_payload() for record in records])
                ),
            }
        )
    )


def _record_from_payload(value: object) -> ArmTrainerRecord:
    if not isinstance(value, dict):
        raise ValueError("arm trainer record must be an object")
    expected = {
        "arm",
        "record_kind",
        "source_sha256",
        "group_id",
        "rollout_id",
        "decision_id",
        "target_id",
        "action_slot",
        "token_ids",
        "mask",
        "behavior_logprobs",
        "temperatures",
        "advantages",
        "rl_weights",
        "rl_normalizer",
    }
    if set(value) != expected:
        raise ValueError("arm trainer record fields differ")
    return ArmTrainerRecord(
        value["arm"],
        value["record_kind"],
        value["source_sha256"],
        value["group_id"],
        value["rollout_id"],
        value["decision_id"],
        value["target_id"],
        value["action_slot"],
        tuple(value["token_ids"]),
        tuple(value["mask"]),
        tuple(value["behavior_logprobs"]),
        tuple(value["temperatures"]),
        tuple(value["advantages"]),
        None if value["rl_weights"] is None else tuple(value["rl_weights"]),
        (
            None
            if value["rl_normalizer"] is None
            else _fraction_from_payload(value["rl_normalizer"], "rl_normalizer")
        ),
    )


def _batch_from_payload(payload: Mapping[str, Any]) -> SealedArmBatch:
    expected = {
        "schema_version",
        "domain",
        "arm",
        "records",
        "source_sha256s",
        "branch_artifact_sha256s",
        "evidence_class",
        "objective_binding",
        "policy_sha256",
        "trainer_step",
        "seq_len",
        "batch_identity",
    }
    if set(payload) != expected:
        raise ValueError("sealed arm batch payload fields differ")
    if payload["schema_version"] != SCHEMA_VERSION or payload["domain"] != _DOMAIN:
        raise ValueError("unsupported sealed arm batch payload")
    records_value = payload["records"]
    if not isinstance(records_value, list):
        raise ValueError("sealed arm batch records must be a list")
    batch = SealedArmBatch(
        payload["arm"],
        tuple(_record_from_payload(record) for record in records_value),
        tuple(payload["source_sha256s"]),
        tuple(payload["branch_artifact_sha256s"]),
        payload["evidence_class"],
        ObjectiveBinding.from_bytes(canonical_json(payload["objective_binding"])),
        payload["policy_sha256"],
        payload["trainer_step"],
        payload["seq_len"],
        payload["batch_identity"],
    )
    expected_identity = _batch_identity(
        batch.arm,
        batch.records,
        batch.source_sha256s,
        batch.branch_artifact_sha256s,
        batch.evidence_class,
        batch.objective_binding,
        batch.policy_sha256,
        batch.trainer_step,
        batch.seq_len,
    )
    if batch.batch_identity != expected_identity:
        raise ValueError("sealed arm batch identity mismatch")
    return batch
