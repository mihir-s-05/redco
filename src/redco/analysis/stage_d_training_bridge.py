"""Verified C1 artifact to trainer-record bridge for the clean Stage-D loss."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from redco.analysis.stage_d_exact_action import ExactActionKey
from redco.analysis.stage_d_scientific_branch_group import (
    BranchGroupArtifact,
    ReceiptVerifier,
    behavior_law_digest,
)
from redco.contracts import canonical_json

SCHEMA_VERSION = 1
_DOMAIN = "redco-stage-d-training-bridge-v1"


@dataclass(frozen=True, slots=True)
class ArtifactVerificationContext:
    verifier: ReceiptVerifier
    encode_action: Callable[
        [Mapping[str, Any], Mapping[str, Any]],
        Sequence[int],
    ] | None
    render_prompt: Callable[[Mapping[str, Any]], tuple[int, ...]]
    master_seed: str
    validate_action: Callable[
        [Mapping[str, Any], Mapping[str, Any], Sequence[int]], None
    ] | None = None

    def __post_init__(self) -> None:
        if not self.master_seed:
            raise ValueError("master_seed must be nonempty")


@dataclass(frozen=True, slots=True)
class TrainingBridgeBinding:
    producer_seal_sha256: str
    bridge_source_sha256: str
    prime_runtime_sha256: str
    trainer_config_sha256: str
    expected_policy_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "producer_seal_sha256",
            "bridge_source_sha256",
            "prime_runtime_sha256",
            "trainer_config_sha256",
            "expected_policy_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    def to_payload(self) -> dict[str, str]:
        return {
            "producer_seal_sha256": self.producer_seal_sha256,
            "bridge_source_sha256": self.bridge_source_sha256,
            "prime_runtime_sha256": self.prime_runtime_sha256,
            "trainer_config_sha256": self.trainer_config_sha256,
            "expected_policy_sha256": self.expected_policy_sha256,
        }


@dataclass(frozen=True, slots=True)
class TrainerRecord:
    source_artifact_sha256: str
    behavior_law_sha256: str
    group_id: str
    target_id: str
    action_slot: int
    token_ids: tuple[int, ...]
    mask: tuple[bool, ...]
    behavior_logprobs: tuple[float, ...]
    temperatures: tuple[float, ...]
    advantages: tuple[float, ...]
    rl_weights: tuple[float, ...]
    record_weight: Fraction
    rl_normalizer: Fraction
    env_name: str

    def __post_init__(self) -> None:
        _require_sha256(self.source_artifact_sha256, "source_artifact_sha256")
        _require_sha256(self.behavior_law_sha256, "behavior_law_sha256")
        if not self.group_id or not self.target_id or not self.env_name:
            raise ValueError("trainer record identifiers must be nonempty")
        if type(self.action_slot) is not int or self.action_slot < 0:
            raise ValueError("action_slot must be a nonnegative integer")
        length = len(self.token_ids)
        if length == 0 or any(
            len(stream) != length
            for stream in (
                self.mask,
                self.behavior_logprobs,
                self.temperatures,
                self.advantages,
                self.rl_weights,
            )
        ):
            raise ValueError("trainer record streams must be nonempty and aligned")
        if not any(self.mask) or self.mask == (True,) * length:
            raise ValueError("trainer record must contain prompt and action positions")
        if any(type(token) is not int or token < 0 for token in self.token_ids):
            raise ValueError("trainer token IDs must be nonnegative integers")
        action_start = self.mask.index(True)
        if not all(self.mask[action_start:]):
            raise ValueError("trainer mask must select one suffix action span")
        if any(
            not math.isfinite(value)
            for stream in (
                self.behavior_logprobs,
                self.temperatures,
                self.advantages,
                self.rl_weights,
            )
            for value in stream
        ):
            raise ValueError("trainer record numeric streams must be finite")
        if any(value <= 0 for value in self.temperatures) or len(set(self.temperatures)) != 1:
            raise ValueError("one positive temperature must cover the full sequence")
        for selected, logprob, advantage, weight in zip(
            self.mask,
            self.behavior_logprobs,
            self.advantages,
            self.rl_weights,
            strict=True,
        ):
            if not selected and (logprob != 0.0 or advantage != 0.0 or weight != 0.0):
                raise ValueError("prompt positions must carry zero credit and logprob")
            if selected and logprob > 0.0:
                raise ValueError("action log probabilities cannot be positive")
        if self.record_weight <= 0 or self.rl_normalizer <= 0:
            raise ValueError("record weights and normalizers must be positive")
        selected_weights = {
            value for value, selected in zip(self.rl_weights, self.mask, strict=True) if selected
        }
        if selected_weights != {float(self.record_weight)}:
            raise ValueError("action weights must equal the exact record weight")
        selected_advantages = {
            value for value, selected in zip(self.advantages, self.mask, strict=True) if selected
        }
        if len(selected_advantages) != 1:
            raise ValueError("one action sequence must carry one scalar advantage")

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_artifact_sha256": self.source_artifact_sha256,
            "behavior_law_sha256": self.behavior_law_sha256,
            "group_id": self.group_id,
            "target_id": self.target_id,
            "action_slot": self.action_slot,
            "token_ids": list(self.token_ids),
            "mask": list(self.mask),
            "behavior_logprobs": list(self.behavior_logprobs),
            "temperatures": list(self.temperatures),
            "advantages": list(self.advantages),
            "rl_weights": list(self.rl_weights),
            "record_weight": _fraction_payload(self.record_weight),
            "rl_normalizer": _fraction_payload(self.rl_normalizer),
            "env_name": self.env_name,
        }


@dataclass(frozen=True, slots=True)
class SealedTrainingBatch:
    source_artifact_sha256s: tuple[str, ...]
    binding: TrainingBridgeBinding
    trainer_step: int
    seq_len: int
    records: tuple[TrainerRecord, ...]
    policy_sha256: str
    training_batch_identity: str
    payload_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "domain": _DOMAIN,
            "source_artifact_sha256s": list(self.source_artifact_sha256s),
            "binding": self.binding.to_payload(),
            "trainer_step": self.trainer_step,
            "seq_len": self.seq_len,
            "records": [record.to_payload() for record in self.records],
            "policy_sha256": self.policy_sha256,
            "training_batch_identity": self.training_batch_identity,
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
    def verify_bytes(cls, value: bytes) -> SealedTrainingBatch:
        if type(value) is not bytes:
            raise ValueError("training batch must be immutable bytes")
        envelope = json.loads(value)
        if not isinstance(envelope, dict) or canonical_json(envelope) != value:
            raise ValueError("training batch must be canonical JSON")
        _strict_keys(
            envelope,
            {"schema_version", "domain", "payload", "payload_sha256"},
            "training batch envelope",
        )
        if envelope["schema_version"] != SCHEMA_VERSION or envelope["domain"] != _DOMAIN:
            raise ValueError("unsupported training batch envelope")
        payload = envelope["payload"]
        if not isinstance(payload, dict):
            raise ValueError("training batch payload must be an object")
        payload_bytes = canonical_json(payload)
        if envelope["payload_sha256"] != _sha256(payload_bytes):
            raise ValueError("training batch payload digest mismatch")
        derived = _batch_from_payload(payload)
        if derived.payload_sha256 != envelope["payload_sha256"]:
            raise ValueError("training batch derived digest mismatch")
        if derived.to_bytes() != value:
            raise ValueError("training batch derived fields disagree")
        return derived


def compile_training_batch(
    artifact_bytes: Sequence[bytes],
    *,
    verification_context: ArtifactVerificationContext,
    binding: TrainingBridgeBinding,
    trainer_step: int,
    seq_len: int,
) -> SealedTrainingBatch:
    """Reverify raw C1 bytes and compile exact target-only trainer records."""
    if type(trainer_step) is not int or trainer_step < 1:
        raise ValueError("trainer_step must be a positive integer")
    if type(seq_len) is not int or seq_len < 1:
        raise ValueError("seq_len must be a positive integer")
    raw_values = tuple(artifact_bytes)
    if not raw_values:
        raise ValueError("at least one C1 artifact is required")
    source_hashes = tuple(sorted(_sha256(value) for value in raw_values))
    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError("source C1 artifacts must be unique")
    artifacts = tuple(
        BranchGroupArtifact.verify_bytes(
            value,
            verifier=verification_context.verifier,
            encode_action=verification_context.encode_action,
            validate_action=verification_context.validate_action,
            render_prompt=verification_context.render_prompt,
            master_seed=verification_context.master_seed,
        )
        for value in raw_values
    )
    _validate_target_rosters(artifacts)
    policy_hashes = {_policy_sha256(artifact.recorded_action.key) for artifact in artifacts}
    if len(policy_hashes) != 1:
        raise ValueError("source C1 artifacts mix behavior policies")
    policy_sha256 = policy_hashes.pop()
    if policy_sha256 != binding.expected_policy_sha256:
        raise ValueError("source policy differs from the frozen bridge binding")
    if any(
        behavior_law_digest(artifact.recorded_action.key) != artifact.commitment.behavior_law_sha256
        for artifact in artifacts
    ):
        raise ValueError("source behavior law differs from its verified commitment")

    records: list[TrainerRecord] = []
    seen_groups: set[tuple[str, str]] = set()
    artifact_by_hash = {
        _sha256(value): artifact for value, artifact in zip(raw_values, artifacts, strict=True)
    }
    for source_sha256 in source_hashes:
        artifact = artifact_by_hash[source_sha256]
        group_key = (artifact.commitment.group_id, artifact.commitment.target_id)
        if group_key in seen_groups:
            raise ValueError("source C1 artifacts repeat a scientific group")
        seen_groups.add(group_key)
        arm_count = len(artifact.arms)
        if arm_count != artifact.commitment.branch_count:
            raise ValueError("C1 arm count differs from its commitment")
        for arm in artifact.arms:
            prompt = arm.action.key.prompt_token_ids
            action_tokens = arm.action.action_token_ids
            token_ids = (*prompt, *action_tokens)
            if len(token_ids) > seq_len:
                raise ValueError("Prime seq_len would truncate a scientific action")
            prompt_count = len(prompt)
            action_count = len(action_tokens)
            record_weight = arm.record_weight
            expected_weight = artifact.commitment.outer_weight / arm_count
            if record_weight != expected_weight:
                raise ValueError("C1 record weight differs from the frozen estimator")
            mask = (False,) * prompt_count + (True,) * action_count
            records.append(
                TrainerRecord(
                    source_artifact_sha256=source_sha256,
                    behavior_law_sha256=artifact.commitment.behavior_law_sha256,
                    group_id=artifact.commitment.group_id,
                    target_id=artifact.commitment.target_id,
                    action_slot=arm.action_slot,
                    token_ids=token_ids,
                    mask=mask,
                    behavior_logprobs=(0.0,) * prompt_count + arm.action.behavior_logprobs,
                    temperatures=(arm.action.key.sampler.temperature,) * len(token_ids),
                    advantages=(0.0,) * prompt_count + (arm.advantage,) * action_count,
                    rl_weights=(0.0,) * prompt_count + (float(record_weight),) * action_count,
                    record_weight=record_weight,
                    rl_normalizer=Fraction(1, arm_count),
                    env_name="redco-stage-d-credit",
                )
            )
    identity = _sha256(
        canonical_json(
            {
                "domain": "redco-stage-d-training-batch-identity-v1",
                "sources": list(source_hashes),
                "binding": binding.to_payload(),
                "trainer_step": trainer_step,
                "seq_len": seq_len,
                "policy_sha256": policy_sha256,
            }
        )
    )
    batch = SealedTrainingBatch(
        source_hashes,
        binding,
        trainer_step,
        seq_len,
        tuple(records),
        policy_sha256,
        identity,
        "0" * 64,
    )
    payload_sha256 = _sha256(canonical_json(batch.to_payload()))
    return SealedTrainingBatch(
        batch.source_artifact_sha256s,
        batch.binding,
        batch.trainer_step,
        batch.seq_len,
        batch.records,
        batch.policy_sha256,
        batch.training_batch_identity,
        payload_sha256,
    )


def policy_identity_sha256(key: ExactActionKey) -> str:
    """Public helper used to freeze the expected behavior-policy identity."""
    return _policy_sha256(key)


def _policy_sha256(key: ExactActionKey) -> str:
    sampler = json.loads(key.sampler_config)
    request = json.loads(key.request)
    if not isinstance(sampler, dict) or not isinstance(request, dict):
        raise ValueError("exact action policy inputs must be objects")
    sampler.pop("seed", None)
    for state_field in ("messages", "prompt", "input", "seed", "model", "tools"):
        request.pop(state_field, None)
    extra_body = request.get("extra_body")
    if extra_body is not None:
        if not isinstance(extra_body, dict):
            raise ValueError("request extra_body must be an object")
        extra_body = dict(extra_body)
        extra_body.pop("cache_salt", None)
        request["extra_body"] = extra_body
    return _sha256(
        canonical_json(
            {
                "domain": "redco-stage-d-policy-family-v1",
                "checkpoint_id": key.checkpoint_id,
                "base_model_manifest_sha256": key.base_model_manifest_sha256,
                "adapter_manifest_sha256": key.adapter_manifest_sha256,
                "tokenizer_manifest_sha256": key.tokenizer_manifest_sha256,
                "renderer_manifest_sha256": key.renderer_manifest_sha256,
                "sampler_conformance_manifest_sha256": (key.sampler_conformance_manifest_sha256),
                "tool_schema_sha256": key.tool_schema_sha256,
                "action_selection_policy": key.action_selection_policy,
                "transport_retry_policy": key.transport_retry_policy,
                "resolved_sampler_without_seed": sampler,
                "request_policy_without_state": request,
            }
        )
    )


def _validate_target_rosters(artifacts: Sequence[BranchGroupArtifact]) -> None:
    by_rollout: dict[str, list[BranchGroupArtifact]] = {}
    for artifact in artifacts:
        by_rollout.setdefault(artifact.commitment.rollout_id, []).append(artifact)
    for rollout_id, members in by_rollout.items():
        roster = members[0].commitment.target_roster
        if any(member.commitment.target_roster != roster for member in members):
            raise ValueError(f"rollout {rollout_id} mixes target rosters")
        ordinals = [member.commitment.target_ordinal for member in members]
        if sorted(ordinals) != list(range(len(roster))):
            raise ValueError(f"rollout {rollout_id} has an incomplete target roster")


def _batch_from_payload(payload: Mapping[str, Any]) -> SealedTrainingBatch:
    _strict_keys(
        payload,
        {
            "schema_version",
            "domain",
            "source_artifact_sha256s",
            "binding",
            "trainer_step",
            "seq_len",
            "records",
            "policy_sha256",
            "training_batch_identity",
        },
        "training batch payload",
    )
    if payload["schema_version"] != SCHEMA_VERSION or payload["domain"] != _DOMAIN:
        raise ValueError("unsupported training batch payload")
    binding_value = payload["binding"]
    if not isinstance(binding_value, dict):
        raise ValueError("bridge binding must be an object")
    _strict_keys(
        binding_value,
        {
            "producer_seal_sha256",
            "bridge_source_sha256",
            "prime_runtime_sha256",
            "trainer_config_sha256",
            "expected_policy_sha256",
        },
        "bridge binding",
    )
    binding = TrainingBridgeBinding(**binding_value)
    record_values = payload["records"]
    if not isinstance(record_values, list) or not record_values:
        raise ValueError("training batch records must be a nonempty list")
    records = tuple(_record_from_payload(value) for value in record_values)
    source_values = payload["source_artifact_sha256s"]
    if not isinstance(source_values, list) or not source_values:
        raise ValueError("source artifact hashes must be a nonempty list")
    source_hashes = tuple(source_values)
    if tuple(sorted(set(source_hashes))) != source_hashes:
        raise ValueError("source artifact hashes must be sorted and unique")
    for digest in source_hashes:
        _require_sha256(digest, "source artifact sha256")
    policy_sha256 = _require_sha256(payload["policy_sha256"], "policy_sha256")
    identity = _require_sha256(payload["training_batch_identity"], "batch identity")
    trainer_step = _exact_int(payload["trainer_step"], "trainer_step", minimum=1)
    seq_len = _exact_int(payload["seq_len"], "seq_len", minimum=1)
    if any(len(record.token_ids) > seq_len for record in records):
        raise ValueError("reloaded batch would truncate a record")
    record_sources = {record.source_artifact_sha256 for record in records}
    if record_sources != set(source_hashes):
        raise ValueError("reloaded records do not cover exactly the source artifacts")
    group_slots: dict[tuple[str, str], list[int]] = {}
    group_sources: dict[tuple[str, str], set[str]] = {}
    source_groups: dict[str, set[tuple[str, str]]] = {}
    group_laws: dict[tuple[str, str], set[str]] = {}
    group_normalizers: dict[tuple[str, str], set[Fraction]] = {}
    for record in records:
        group = (record.group_id, record.target_id)
        group_slots.setdefault(group, []).append(record.action_slot)
        group_sources.setdefault(group, set()).add(record.source_artifact_sha256)
        source_groups.setdefault(record.source_artifact_sha256, set()).add(group)
        group_laws.setdefault(group, set()).add(record.behavior_law_sha256)
        group_normalizers.setdefault(group, set()).add(record.rl_normalizer)
    if any(sorted(slots) != list(range(len(slots))) for slots in group_slots.values()):
        raise ValueError("reloaded batch has an incomplete or duplicate action roster")
    if any(len(values) != 1 for values in group_sources.values()) or any(
        len(values) != 1 for values in source_groups.values()
    ):
        raise ValueError("reloaded batch does not map each source to exactly one group")
    if any(len(values) != 1 for values in group_laws.values()):
        raise ValueError("reloaded group mixes behavior laws")
    for group, slots in group_slots.items():
        if group_normalizers[group] != {Fraction(1, len(slots))}:
            raise ValueError("reloaded group has an invalid decision normalizer")
    expected_identity = _sha256(
        canonical_json(
            {
                "domain": "redco-stage-d-training-batch-identity-v1",
                "sources": list(source_hashes),
                "binding": binding.to_payload(),
                "trainer_step": trainer_step,
                "seq_len": seq_len,
                "policy_sha256": policy_sha256,
            }
        )
    )
    if identity != expected_identity or policy_sha256 != binding.expected_policy_sha256:
        raise ValueError("training batch identity or policy binding mismatch")
    result = SealedTrainingBatch(
        source_hashes,
        binding,
        trainer_step,
        seq_len,
        records,
        policy_sha256,
        identity,
        "0" * 64,
    )
    return SealedTrainingBatch(
        result.source_artifact_sha256s,
        result.binding,
        result.trainer_step,
        result.seq_len,
        result.records,
        result.policy_sha256,
        result.training_batch_identity,
        _sha256(canonical_json(result.to_payload())),
    )


def _record_from_payload(value: Any) -> TrainerRecord:
    if not isinstance(value, dict):
        raise ValueError("trainer record must be an object")
    _strict_keys(
        value,
        {
            "source_artifact_sha256",
            "behavior_law_sha256",
            "group_id",
            "target_id",
            "action_slot",
            "token_ids",
            "mask",
            "behavior_logprobs",
            "temperatures",
            "advantages",
            "rl_weights",
            "record_weight",
            "rl_normalizer",
            "env_name",
        },
        "trainer record",
    )
    sequence_fields = (
        "token_ids",
        "mask",
        "behavior_logprobs",
        "temperatures",
        "advantages",
        "rl_weights",
    )
    if any(not isinstance(value[field], list) for field in sequence_fields):
        raise ValueError("trainer record streams must be lists")
    return TrainerRecord(
        source_artifact_sha256=value["source_artifact_sha256"],
        behavior_law_sha256=value["behavior_law_sha256"],
        group_id=value["group_id"],
        target_id=value["target_id"],
        action_slot=_exact_int(value["action_slot"], "action_slot"),
        token_ids=tuple(_exact_int(item, "token_id") for item in value["token_ids"]),
        mask=tuple(_exact_bool(item, "mask") for item in value["mask"]),
        behavior_logprobs=tuple(
            _finite_float(item, "behavior_logprob") for item in value["behavior_logprobs"]
        ),
        temperatures=tuple(_finite_float(item, "temperature") for item in value["temperatures"]),
        advantages=tuple(_finite_float(item, "advantage") for item in value["advantages"]),
        rl_weights=tuple(_finite_float(item, "rl_weight") for item in value["rl_weights"]),
        record_weight=_fraction_from_payload(value["record_weight"]),
        rl_normalizer=_fraction_from_payload(value["rl_normalizer"]),
        env_name=value["env_name"],
    )


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_from_payload(value: Any) -> Fraction:
    if not isinstance(value, dict):
        raise ValueError("fraction must be an object")
    _strict_keys(value, {"numerator", "denominator"}, "fraction")
    return Fraction(
        _exact_int(value["numerator"], "fraction numerator"),
        _exact_int(value["denominator"], "fraction denominator", minimum=1),
    )


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields differ from the frozen schema")


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


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _finite_float(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be numeric")
    resolved = float(cast(int | float, value))
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved
