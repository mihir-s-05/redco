"""Canonical pre-source trust root for one complete Stage-D live campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from redco.contracts import canonical_json

_ARMS = ("stock", "branch-global", "local")
_DOMAIN = "redco-stage-d-protocol-manifest-v1"
_BRANCH_GLOBAL_SCOPE = "within-source-group-all-target-branches-v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _strict_mapping(value: object, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or set(value) != set(_ARMS):
        raise ValueError(f"{name} must contain exactly the three frozen arms")
    return tuple(
        (arm, str(_require_sha256(value[arm], f"{name}.{arm}"))) for arm in _ARMS
    )


@dataclass(frozen=True, slots=True)
class StageDPolicyIdentity:
    checkpoint_id: str
    base_model_manifest_sha256: str
    adapter_manifest_sha256: str | None
    tokenizer_manifest_sha256: str
    renderer_manifest_sha256: str
    sampler_conformance_manifest_sha256: str
    resolved_agent_sampling_law_sha256: str
    resolved_train_client_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.checkpoint_id, str)
            or not self.checkpoint_id
            or len(self.checkpoint_id) > 512
            or not self.checkpoint_id.isprintable()
        ):
            raise ValueError("policy checkpoint_id must be a bounded printable string")
        for name in (
            "base_model_manifest_sha256",
            "tokenizer_manifest_sha256",
            "renderer_manifest_sha256",
            "sampler_conformance_manifest_sha256",
            "resolved_agent_sampling_law_sha256",
            "resolved_train_client_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_sha256(
            self.adapter_manifest_sha256,
            "adapter_manifest_sha256",
            optional=True,
        )

    def to_payload(self) -> dict[str, str | None]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "base_model_manifest_sha256": self.base_model_manifest_sha256,
            "adapter_manifest_sha256": self.adapter_manifest_sha256,
            "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            "renderer_manifest_sha256": self.renderer_manifest_sha256,
            "sampler_conformance_manifest_sha256": (
                self.sampler_conformance_manifest_sha256
            ),
            "resolved_agent_sampling_law_sha256": (
                self.resolved_agent_sampling_law_sha256
            ),
            "resolved_train_client_sha256": self.resolved_train_client_sha256,
        }

    @classmethod
    def from_payload(cls, value: object) -> StageDPolicyIdentity:
        expected = {
            "checkpoint_id",
            "base_model_manifest_sha256",
            "adapter_manifest_sha256",
            "tokenizer_manifest_sha256",
            "renderer_manifest_sha256",
            "sampler_conformance_manifest_sha256",
            "resolved_agent_sampling_law_sha256",
            "resolved_train_client_sha256",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("protocol policy identity fields differ")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class StageDProtocolManifest:
    preregistration_sha256: str
    dependency_stack_sha256: str
    genesis_config_sha256: str
    master_seed_sha256: str
    source_sha256: str
    runtime_sha256: str
    source_eval_config_sha256: str
    scientific_eval_config_sha256: str
    heldout_eval_config_sha256: str
    collection_plan_sha256: str
    evaluation_plan_sha256: str
    decision_rule_sha256: str
    reload_probe_sha256: str
    shared_initialization_sha256: str
    objective_authorization_sha256: str
    objective_binding_sha256s: tuple[tuple[str, str], ...]
    trainer_config_sha256s: tuple[tuple[str, str], ...]
    policy_identity: StageDPolicyIdentity
    arm_order: tuple[str, ...]
    branch_global_scope: str
    trainer_step: int
    seq_len: int

    def __post_init__(self) -> None:
        for name in (
            "preregistration_sha256",
            "dependency_stack_sha256",
            "genesis_config_sha256",
            "master_seed_sha256",
            "source_sha256",
            "runtime_sha256",
            "source_eval_config_sha256",
            "scientific_eval_config_sha256",
            "heldout_eval_config_sha256",
            "collection_plan_sha256",
            "evaluation_plan_sha256",
            "decision_rule_sha256",
            "reload_probe_sha256",
            "shared_initialization_sha256",
            "objective_authorization_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if tuple(name for name, _ in self.objective_binding_sha256s) != _ARMS:
            raise ValueError("objective binding hashes use the wrong arm order")
        if tuple(name for name, _ in self.trainer_config_sha256s) != _ARMS:
            raise ValueError("trainer config hashes use the wrong arm order")
        for name, digest in (*self.objective_binding_sha256s, *self.trainer_config_sha256s):
            _require_sha256(digest, name)
        if self.arm_order != _ARMS:
            raise ValueError("protocol arm_order must equal the frozen three-arm order")
        if self.branch_global_scope != _BRANCH_GLOBAL_SCOPE:
            raise ValueError("protocol branch-global scope differs from the frozen compiler scope")
        if type(self.trainer_step) is not int or self.trainer_step < 1:
            raise ValueError("trainer_step must be a positive integer")
        if type(self.seq_len) is not int or self.seq_len < 1:
            raise ValueError("seq_len must be a positive integer")

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self.to_bytes())

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": _DOMAIN,
                "preregistration_sha256": self.preregistration_sha256,
                "dependency_stack_sha256": self.dependency_stack_sha256,
                "genesis_config_sha256": self.genesis_config_sha256,
                "master_seed_sha256": self.master_seed_sha256,
                "source_sha256": self.source_sha256,
                "runtime_sha256": self.runtime_sha256,
                "source_eval_config_sha256": self.source_eval_config_sha256,
                "scientific_eval_config_sha256": self.scientific_eval_config_sha256,
                "heldout_eval_config_sha256": self.heldout_eval_config_sha256,
                "collection_plan_sha256": self.collection_plan_sha256,
                "evaluation_plan_sha256": self.evaluation_plan_sha256,
                "decision_rule_sha256": self.decision_rule_sha256,
                "reload_probe_sha256": self.reload_probe_sha256,
                "shared_initialization_sha256": self.shared_initialization_sha256,
                "objective_authorization_sha256": self.objective_authorization_sha256,
                "objective_binding_sha256s": dict(self.objective_binding_sha256s),
                "trainer_config_sha256s": dict(self.trainer_config_sha256s),
                "policy_identity": self.policy_identity.to_payload(),
                "arm_order": list(self.arm_order),
                "branch_global_scope": self.branch_global_scope,
                "trainer_step": self.trainer_step,
                "seq_len": self.seq_len,
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDProtocolManifest:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("protocol manifest is not JSON") from error
        expected = {
            "schema_version",
            "domain",
            "preregistration_sha256",
            "dependency_stack_sha256",
            "genesis_config_sha256",
            "master_seed_sha256",
            "source_sha256",
            "runtime_sha256",
            "source_eval_config_sha256",
            "scientific_eval_config_sha256",
            "heldout_eval_config_sha256",
            "collection_plan_sha256",
            "evaluation_plan_sha256",
            "decision_rule_sha256",
            "reload_probe_sha256",
            "shared_initialization_sha256",
            "objective_authorization_sha256",
            "objective_binding_sha256s",
            "trainer_config_sha256s",
            "policy_identity",
            "arm_order",
            "branch_global_scope",
            "trainer_step",
            "seq_len",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload.get("schema_version") != 1
            or payload.get("domain") != _DOMAIN
            or canonical_json(payload) != value
        ):
            raise ValueError("protocol manifest is noncanonical or has different fields")
        return cls(
            preregistration_sha256=payload["preregistration_sha256"],
            dependency_stack_sha256=payload["dependency_stack_sha256"],
            genesis_config_sha256=payload["genesis_config_sha256"],
            master_seed_sha256=payload["master_seed_sha256"],
            source_sha256=payload["source_sha256"],
            runtime_sha256=payload["runtime_sha256"],
            source_eval_config_sha256=payload["source_eval_config_sha256"],
            scientific_eval_config_sha256=payload["scientific_eval_config_sha256"],
            heldout_eval_config_sha256=payload["heldout_eval_config_sha256"],
            collection_plan_sha256=payload["collection_plan_sha256"],
            evaluation_plan_sha256=payload["evaluation_plan_sha256"],
            decision_rule_sha256=payload["decision_rule_sha256"],
            reload_probe_sha256=payload["reload_probe_sha256"],
            shared_initialization_sha256=payload["shared_initialization_sha256"],
            objective_authorization_sha256=payload["objective_authorization_sha256"],
            objective_binding_sha256s=_strict_mapping(
                payload["objective_binding_sha256s"], "objective_binding_sha256s"
            ),
            trainer_config_sha256s=_strict_mapping(
                payload["trainer_config_sha256s"], "trainer_config_sha256s"
            ),
            policy_identity=StageDPolicyIdentity.from_payload(payload["policy_identity"]),
            arm_order=tuple(payload["arm_order"])
            if isinstance(payload["arm_order"], list)
            else (),
            branch_global_scope=payload["branch_global_scope"],
            trainer_step=payload["trainer_step"],
            seq_len=payload["seq_len"],
        )

    @classmethod
    def verify_file(
        cls,
        path: Path,
        expected_sha256: str,
    ) -> StageDProtocolManifest:
        _require_sha256(expected_sha256, "protocol_manifest_sha256")
        value = path.read_bytes()
        if _sha256(value) != expected_sha256:
            raise ValueError("protocol manifest bytes differ from the frozen hash")
        return cls.from_bytes(value)

    def arm_hash(self, field: str, arm: str) -> str:
        if arm not in _ARMS:
            raise ValueError("unknown Stage D arm")
        values: dict[str, str]
        if field == "objective_binding":
            values = dict(self.objective_binding_sha256s)
        elif field == "trainer_config":
            values = dict(self.trainer_config_sha256s)
        else:
            raise ValueError("unknown arm hash field")
        return values[arm]


__all__ = ["StageDPolicyIdentity", "StageDProtocolManifest"]
