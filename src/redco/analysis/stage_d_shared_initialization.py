"""Frozen, directly measurable shared initialization for every Stage-D trainer arm."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from redco.analysis.stage_d_protocol_manifest import StageDProtocolManifest
from redco.contracts import canonical_json

_DOMAIN = "redco-stage-d-shared-initialization-v1"


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


@dataclass(frozen=True, slots=True)
class StageDSharedInitializationManifest:
    initialization_id: str
    checkpoint_id: str
    base_model_manifest_sha256: str
    adapter_manifest_sha256: str | None
    expected_pre_model_sha256: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.initialization_id, "initialization_id"),
            (self.checkpoint_id, "checkpoint_id"),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 512
                or not value.isprintable()
            ):
                raise ValueError(f"{name} must be a bounded printable string")
        _require_sha256(self.base_model_manifest_sha256, "base_model_manifest_sha256")
        if self.adapter_manifest_sha256 is not None:
            _require_sha256(self.adapter_manifest_sha256, "adapter_manifest_sha256")
        _require_sha256(self.expected_pre_model_sha256, "expected_pre_model_sha256")

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self.to_bytes())

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": _DOMAIN,
                "initialization_id": self.initialization_id,
                "checkpoint_id": self.checkpoint_id,
                "base_model_manifest_sha256": self.base_model_manifest_sha256,
                "adapter_manifest_sha256": self.adapter_manifest_sha256,
                "expected_pre_model_sha256": self.expected_pre_model_sha256,
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDSharedInitializationManifest:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("shared initialization manifest must be canonical JSON") from error
        fields = {
            "schema_version",
            "domain",
            "initialization_id",
            "checkpoint_id",
            "base_model_manifest_sha256",
            "adapter_manifest_sha256",
            "expected_pre_model_sha256",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or canonical_json(payload) != value
            or payload["schema_version"] != 1
            or payload["domain"] != _DOMAIN
        ):
            raise ValueError("shared initialization manifest fields differ")
        return cls(
            initialization_id=payload["initialization_id"],
            checkpoint_id=payload["checkpoint_id"],
            base_model_manifest_sha256=payload["base_model_manifest_sha256"],
            adapter_manifest_sha256=payload["adapter_manifest_sha256"],
            expected_pre_model_sha256=payload["expected_pre_model_sha256"],
        )

    @classmethod
    def from_retained_adapter(
        cls,
        *,
        initialization_id: str,
        checkpoint_id: str,
        base_model_manifest_path: Path,
        adapter_manifest_path: Path,
        adapter_path: Path,
    ) -> StageDSharedInitializationManifest:
        """Derive the frozen identity from the actual retained adapter bytes."""
        base_manifest = base_model_manifest_path.read_bytes()
        adapter_manifest = adapter_manifest_path.read_bytes()
        if not base_manifest or not adapter_manifest or not adapter_path.is_file():
            raise ValueError("shared initialization inputs must be nonempty files")
        base_sha256 = _sha256(base_manifest)
        from redco.analysis.stage_d_live_update import adapter_file_state_sha256

        return cls(
            initialization_id=initialization_id,
            checkpoint_id=checkpoint_id,
            base_model_manifest_sha256=base_sha256,
            adapter_manifest_sha256=_sha256(adapter_manifest),
            expected_pre_model_sha256=adapter_file_state_sha256(
                adapter_path,
                base_snapshot_manifest_sha256=base_sha256,
            ),
        )

    def verify_protocol(self, protocol: StageDProtocolManifest) -> None:
        if (
            self.manifest_sha256 != protocol.shared_initialization_sha256
            or self.checkpoint_id != protocol.policy_identity.checkpoint_id
            or self.base_model_manifest_sha256
            != protocol.policy_identity.base_model_manifest_sha256
            or self.adapter_manifest_sha256
            != protocol.policy_identity.adapter_manifest_sha256
        ):
            raise ValueError("shared initialization differs from the Stage D protocol")


__all__ = ["StageDSharedInitializationManifest"]
