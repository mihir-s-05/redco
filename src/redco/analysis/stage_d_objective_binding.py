"""Immutable, independently authorizable Stage-D trainer-objective contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from redco.contracts import canonical_json

ArmName = Literal["stock", "branch-global", "local"]
EvidenceClass = Literal["live", "fixture-only"]

_DOMAIN = "redco-stage-d-objective-binding-v1"
_CONTRACT_FIELDS = {
    "schema_version",
    "domain",
    "arm",
    "evidence_class",
    "effective_argv",
    "trainer_toml_sha256",
    "materialized_trainer_config_sha256",
    "loss_config",
    "exact_categorical",
    "fused_lm_head_token_chunk_size",
    "loss_callable",
    "module_sha256s",
}
_MODULE_KEYS = {
    "prime_rl.configs.trainer",
    "prime_rl.trainer.rl.loss",
    "prime_rl.trainer.rl.train",
    "resolved_loss_callable_module",
}
_AUTH_DOMAIN = "redco-stage-d-objective-authorization-v1"


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


@dataclass(frozen=True, slots=True, init=False)
class ObjectiveBinding:
    """Canonical executable objective, distinct from its frozen authorization."""

    arm: ArmName
    evidence_class: EvidenceClass
    contract: bytes
    objective_sha256: str

    def __new__(cls) -> ObjectiveBinding:
        raise TypeError("ObjectiveBinding requires canonical verification")

    @classmethod
    def from_bytes(cls, value: bytes) -> ObjectiveBinding:
        if type(value) is not bytes:
            raise ValueError("objective contract must be immutable bytes")
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("objective contract must be canonical JSON") from error
        if not isinstance(payload, dict) or canonical_json(payload) != value:
            raise ValueError("objective contract must be canonical JSON")
        if set(payload) != _CONTRACT_FIELDS:
            raise ValueError("objective contract fields differ")
        if payload["schema_version"] != 1 or payload["domain"] != _DOMAIN:
            raise ValueError("unsupported objective contract")
        arm = payload["arm"]
        evidence_class = payload["evidence_class"]
        if arm not in {"stock", "branch-global", "local"}:
            raise ValueError("objective contract arm is invalid")
        if evidence_class not in {"live", "fixture-only"}:
            raise ValueError("objective evidence class is invalid")
        argv = payload["effective_argv"]
        if (
            not isinstance(argv, list)
            or len(argv) != 2
            or argv[0] != "@"
            or not isinstance(argv[1], str)
            or not argv[1]
        ):
            raise ValueError("objective contract requires one TOML and no CLI overrides")
        _require_sha256(payload["trainer_toml_sha256"], "trainer TOML sha256")
        _require_sha256(
            payload["materialized_trainer_config_sha256"],
            "materialized trainer config sha256",
        )
        loss_config = payload["loss_config"]
        if not isinstance(loss_config, dict):
            raise ValueError("objective loss config must be an object")
        callable_payload = payload["loss_callable"]
        if not isinstance(callable_payload, dict) or set(callable_payload) != {
            "kind",
            "import_path",
            "module",
            "qualname",
        }:
            raise ValueError("objective callable fields differ")
        expected = (
            {
                "kind": "function",
                "import_path": "prime_rl.trainer.rl.loss.default_loss_fn",
                "module": "prime_rl.trainer.rl.loss",
                "qualname": "default_loss_fn",
            }
            if arm == "stock"
            else {
                "kind": "function",
                "import_path": "prime_rl.trainer.rl.redco_loss.clean_decision_loss",
                "module": "prime_rl.trainer.rl.redco_loss",
                "qualname": "clean_decision_loss",
            }
        )
        if callable_payload != expected:
            raise ValueError("objective callable is not authorized for its arm")
        if arm == "stock":
            if loss_config.get("type") != "default":
                raise ValueError("stock objective must use Prime default loss")
        elif (
            loss_config.get("type") != "custom"
            or loss_config.get("import_path") != expected["import_path"]
            or loss_config.get("kwargs") != {"kl_tau": 0.0}
        ):
            raise ValueError("branch objective must use exact clean decision loss")
        if payload["exact_categorical"] is not None:
            raise ValueError("Stage D full-vocabulary objective forbids token-group normalization")
        fused_head = payload["fused_lm_head_token_chunk_size"]
        if fused_head != "disabled" and (
            type(fused_head) is not int or fused_head < 1
        ):
            raise ValueError("Stage D fused LM-head setting is invalid")
        module_hashes = payload["module_sha256s"]
        if not isinstance(module_hashes, dict) or set(module_hashes) != _MODULE_KEYS:
            raise ValueError("objective module hash roster differs")
        for name, digest in module_hashes.items():
            _require_sha256(digest, f"{name} sha256")
        self = object.__new__(cls)
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "evidence_class", evidence_class)
        object.__setattr__(self, "contract", value)
        object.__setattr__(self, "objective_sha256", _sha256(value))
        return self

    def to_payload(self) -> dict[str, Any]:
        value = json.loads(self.contract)
        assert isinstance(value, dict)
        return value


@dataclass(frozen=True, slots=True)
class ObjectiveAuthorization:
    """Independent preregistered arm-to-objective allowlist."""

    evidence_class: EvidenceClass
    expected_objective_sha256s: tuple[tuple[ArmName, str], ...]

    def __post_init__(self) -> None:
        if self.evidence_class not in {"live", "fixture-only"}:
            raise ValueError("objective authorization evidence class is invalid")
        expected_arms = ("branch-global", "local", "stock")
        if tuple(arm for arm, _ in self.expected_objective_sha256s) != expected_arms:
            raise ValueError("objective authorization must cover every arm in sorted order")
        for _, digest in self.expected_objective_sha256s:
            _require_sha256(digest, "authorized objective sha256")

    def authorize(self, bindings: tuple[ObjectiveBinding, ...]) -> None:
        actual = tuple(
            sorted((binding.arm, binding.objective_sha256) for binding in bindings)
        )
        if any(binding.evidence_class != self.evidence_class for binding in bindings):
            raise ValueError("objective authorization mixes evidence classes")
        if actual != self.expected_objective_sha256s:
            raise ValueError("objective binding is not independently authorized")

    def authorize_one(self, binding: ObjectiveBinding) -> None:
        expected = dict(self.expected_objective_sha256s).get(binding.arm)
        if (
            binding.evidence_class != self.evidence_class
            or binding.objective_sha256 != expected
        ):
            raise ValueError("objective binding is not independently authorized")

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": _AUTH_DOMAIN,
                "evidence_class": self.evidence_class,
                "expected_objective_sha256s": {
                    arm: digest for arm, digest in self.expected_objective_sha256s
                },
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> ObjectiveAuthorization:
        if type(value) is not bytes:
            raise ValueError("objective authorization must be immutable bytes")
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("objective authorization must be canonical JSON") from error
        if not isinstance(payload, dict) or canonical_json(payload) != value:
            raise ValueError("objective authorization must be canonical JSON")
        if set(payload) != {
            "schema_version",
            "domain",
            "evidence_class",
            "expected_objective_sha256s",
        } or payload.get("schema_version") != 1 or payload.get("domain") != _AUTH_DOMAIN:
            raise ValueError("objective authorization envelope is invalid")
        expected = payload["expected_objective_sha256s"]
        if not isinstance(expected, dict):
            raise ValueError("objective authorization map must be an object")
        return cls(
            payload["evidence_class"],
            tuple(sorted(expected.items())),
        )


def fixture_objective_binding(arm: ArmName) -> ObjectiveBinding:
    """Stable non-live objective identity for CPU compiler tests."""
    custom = arm != "stock"
    module_hashes = {name: "0" * 64 for name in sorted(_MODULE_KEYS)}
    payload = {
        "schema_version": 1,
        "domain": _DOMAIN,
        "arm": arm,
        "evidence_class": "fixture-only",
        "effective_argv": ["@", "fixture-trainer.toml"],
        "trainer_toml_sha256": "1" * 64,
        "materialized_trainer_config_sha256": "2" * 64,
        "loss_config": (
            {
                "type": "custom",
                "import_path": "prime_rl.trainer.rl.redco_loss.clean_decision_loss",
                "kwargs": {"kl_tau": 0.0},
            }
            if custom
            else {
                "type": "default",
                "dppo_mask_low": 0.2,
                "dppo_mask_high": 0.2,
                "adv_tau": 1.0,
                "kl_tau": 0.0,
            }
        ),
        "exact_categorical": None,
        "fused_lm_head_token_chunk_size": "disabled",
        "loss_callable": (
            {
                "kind": "function",
                "import_path": "prime_rl.trainer.rl.redco_loss.clean_decision_loss",
                "module": "prime_rl.trainer.rl.redco_loss",
                "qualname": "clean_decision_loss",
            }
            if custom
            else {
                "kind": "function",
                "import_path": "prime_rl.trainer.rl.loss.default_loss_fn",
                "module": "prime_rl.trainer.rl.loss",
                "qualname": "default_loss_fn",
            }
        ),
        "module_sha256s": module_hashes,
    }
    return ObjectiveBinding.from_bytes(canonical_json(payload))
