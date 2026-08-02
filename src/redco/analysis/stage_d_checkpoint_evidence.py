"""Content-addressed checkpoint and fresh-process reload evidence for Stage D."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from redco.analysis.stage_d_live_update import adapter_file_state_sha256
from redco.analysis.stage_d_objective_binding import ArmName
from redco.contracts import canonical_json

_CHECKPOINT_DOMAIN = "redco-stage-d-checkpoint-manifest-v1"
_RELOAD_DOMAIN = "redco-stage-d-fresh-process-reload-v1"
_METRICS_DOMAIN = "redco-stage-d-trainer-metrics-v1"
_ARMS = {"stock", "branch-global", "local"}
_ADAPTER_ROSTER = {"STABLE", "adapter_config.json", "adapter_model.safetensors"}


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


def _arm(value: object) -> ArmName:
    if value not in _ARMS:
        raise ValueError("checkpoint evidence arm is invalid")
    return cast(ArmName, value)


def _step(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("checkpoint evidence trainer step must be positive")
    return value


@dataclass(frozen=True, slots=True)
class CheckpointMember:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        pure = PurePosixPath(self.path)
        if (
            not self.path
            or pure.is_absolute()
            or "\\" in self.path
            or any(part in {"", ".", ".."} for part in pure.parts)
            or pure.as_posix() != self.path
        ):
            raise ValueError("checkpoint member path is unsafe or noncanonical")
        if type(self.size) is not int or self.size < 0:
            raise ValueError("checkpoint member size is invalid")
        _require_sha256(self.sha256, "checkpoint member sha256")

    def to_payload(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class StageDCheckpointManifest:
    arm: ArmName
    trainer_step: int
    base_model_manifest_sha256: str
    post_model_sha256: str
    members: tuple[CheckpointMember, ...]

    def __post_init__(self) -> None:
        _arm(self.arm)
        _step(self.trainer_step)
        _require_sha256(self.base_model_manifest_sha256, "base model manifest sha256")
        _require_sha256(self.post_model_sha256, "post-model sha256")
        paths = tuple(member.path for member in self.members)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("checkpoint member roster must be sorted and unique")
        if set(paths) != _ADAPTER_ROSTER:
            raise ValueError("checkpoint manifest is not the exact adapter-only roster")

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self.to_bytes())

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": _CHECKPOINT_DOMAIN,
                "arm": self.arm,
                "trainer_step": self.trainer_step,
                "base_model_manifest_sha256": self.base_model_manifest_sha256,
                "post_model_sha256": self.post_model_sha256,
                "members": [member.to_payload() for member in self.members],
            }
        )

    @classmethod
    def build(
        cls,
        *,
        arm: ArmName,
        trainer_step: int,
        checkpoint_root: Path,
        base_model_manifest_sha256: str,
        observed_post_model_sha256: str,
    ) -> StageDCheckpointManifest:
        if not checkpoint_root.is_dir() or not (checkpoint_root / "STABLE").is_file():
            raise ValueError("checkpoint is absent or lacks its stable marker")
        for item in checkpoint_root.iterdir():
            if item.is_symlink() or not item.is_file():
                raise ValueError("checkpoint must contain only regular top-level files")
        members: list[CheckpointMember] = []
        for path in sorted(
            (item for item in checkpoint_root.rglob("*") if item.is_file()),
            key=lambda candidate: candidate.relative_to(checkpoint_root).as_posix(),
        ):
            if path.is_symlink():
                raise ValueError("checkpoint manifest forbids symbolic links")
            value = path.read_bytes()
            members.append(
                CheckpointMember(
                    path.relative_to(checkpoint_root).as_posix(),
                    len(value),
                    _sha256(value),
                )
            )
        manifest = cls(
            arm,
            trainer_step,
            _require_sha256(base_model_manifest_sha256, "base model manifest sha256"),
            _require_sha256(observed_post_model_sha256, "observed post-model sha256"),
            tuple(members),
        )
        semantic = adapter_file_state_sha256(
            checkpoint_root / "adapter_model.safetensors",
            base_snapshot_manifest_sha256=base_model_manifest_sha256,
        )
        if semantic != observed_post_model_sha256:
            raise ValueError("checkpoint adapter differs from the observed post-update state")
        manifest.verify_directory(checkpoint_root)
        return manifest

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDCheckpointManifest:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("checkpoint manifest is not JSON") from error
        expected = {
            "schema_version",
            "domain",
            "arm",
            "trainer_step",
            "base_model_manifest_sha256",
            "post_model_sha256",
            "members",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload.get("schema_version") != 1
            or payload.get("domain") != _CHECKPOINT_DOMAIN
            or canonical_json(payload) != value
            or not isinstance(payload.get("members"), list)
        ):
            raise ValueError("checkpoint manifest is noncanonical or has different fields")
        members = []
        for item in payload["members"]:
            if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
                raise ValueError("checkpoint manifest member fields differ")
            members.append(CheckpointMember(item["path"], item["size"], item["sha256"]))
        return cls(
            _arm(payload["arm"]),
            _step(payload["trainer_step"]),
            _require_sha256(payload["base_model_manifest_sha256"], "base model manifest sha256"),
            _require_sha256(payload["post_model_sha256"], "post-model sha256"),
            tuple(members),
        )

    def verify_directory(self, checkpoint_root: Path, *, verify_semantic: bool = True) -> None:
        if not checkpoint_root.is_dir() or checkpoint_root.is_symlink():
            raise ValueError("checkpoint directory is absent or symbolic")
        for item in checkpoint_root.iterdir():
            if item.is_symlink() or not item.is_file():
                raise ValueError("checkpoint must contain only regular top-level files")
        actual = tuple(
            sorted(
                item.relative_to(checkpoint_root).as_posix()
                for item in checkpoint_root.rglob("*")
                if item.is_file()
            )
        )
        expected = tuple(member.path for member in self.members)
        if actual != expected:
            raise ValueError("checkpoint directory roster differs from its manifest")
        for member in self.members:
            path = checkpoint_root / Path(*PurePosixPath(member.path).parts)
            if path.is_symlink():
                raise ValueError("checkpoint manifest forbids symbolic links")
            value = path.read_bytes()
            if len(value) != member.size or _sha256(value) != member.sha256:
                raise ValueError("checkpoint member bytes differ from their manifest")
        if verify_semantic:
            semantic = adapter_file_state_sha256(
                checkpoint_root / "adapter_model.safetensors",
                base_snapshot_manifest_sha256=self.base_model_manifest_sha256,
            )
            if semantic != self.post_model_sha256:
                raise ValueError("checkpoint semantic state differs from its manifest")

    def verify_member_evidence(self, evidence: dict[str, bytes]) -> None:
        """Verify the exact checkpoint member closure in a content-addressed store."""
        for member in self.members:
            value = evidence.get(member.sha256)
            if value is None or len(value) != member.size or _sha256(value) != member.sha256:
                raise ValueError("checkpoint member evidence differs from its manifest")


def adopt_prime_adapter_checkpoint(
    *,
    source_step_root: Path,
    destination: Path,
    arm: ArmName,
    trainer_step: int,
    base_model_manifest_sha256: str,
    observed_post_model_sha256: str,
) -> StageDCheckpointManifest:
    """Atomically adopt Prime's nested adapter output into the frozen compact roster."""
    source_roster = {item.name for item in source_step_root.iterdir()}
    if source_roster != {"STABLE", "lora_adapters"}:
        raise ValueError("Prime checkpoint step has an unexpected source roster")
    stable = source_step_root / "STABLE"
    adapter_root = source_step_root / "lora_adapters"
    if (
        stable.is_symlink()
        or not stable.is_file()
        or stable.read_bytes() != b""
        or adapter_root.is_symlink()
        or not adapter_root.is_dir()
    ):
        raise ValueError("Prime checkpoint source layout is invalid")
    adapter_roster = {item.name for item in adapter_root.iterdir()}
    if adapter_roster != _ADAPTER_ROSTER - {"STABLE"}:
        raise ValueError("Prime adapter directory has an unexpected source roster")
    sources = {
        "STABLE": stable,
        "adapter_config.json": adapter_root / "adapter_config.json",
        "adapter_model.safetensors": adapter_root / "adapter_model.safetensors",
    }
    if any(path.is_symlink() or not path.is_file() for path in sources.values()):
        raise ValueError("Prime adapter source contains a non-regular member")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(f".{destination.name}.pending")
    if destination.exists():
        if pending.exists():
            raise ValueError("checkpoint adoption has both final and pending directories")
    else:
        pending.mkdir(exist_ok=True)
        for name, source in sources.items():
            _install_checkpoint_member(pending / name, source.read_bytes())
        if {item.name for item in pending.iterdir()} != _ADAPTER_ROSTER:
            raise ValueError("pending checkpoint adoption roster is invalid")
        _fsync_directory(pending)
        os.replace(pending, destination)
        _fsync_directory(destination.parent)
    return StageDCheckpointManifest.build(
        arm=arm,
        trainer_step=trainer_step,
        checkpoint_root=destination,
        base_model_manifest_sha256=base_model_manifest_sha256,
        observed_post_model_sha256=observed_post_model_sha256,
    )


def _install_checkpoint_member(path: Path, value: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != value:
            raise ValueError("pending checkpoint member differs from its source") from None
        return
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class StageDReloadEvidence:
    arm: ArmName
    checkpoint_manifest_sha256: str
    post_model_sha256: str
    reload_probe_sha256: str
    process_identities: tuple[str, str]
    output_sha256s: tuple[str, str]

    def __post_init__(self) -> None:
        _arm(self.arm)
        for name in (
            "checkpoint_manifest_sha256",
            "post_model_sha256",
            "reload_probe_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if len(set(self.process_identities)) != 2 or any(
            not value or not value.isprintable() for value in self.process_identities
        ):
            raise ValueError("reload evidence requires two distinct fresh-process identities")
        for digest in self.output_sha256s:
            _require_sha256(digest, "reload output sha256")
        if self.output_sha256s[0] != self.output_sha256s[1]:
            raise ValueError("fresh-process reload outputs are not exactly reproducible")

    @property
    def evidence_sha256(self) -> str:
        return _sha256(self.to_bytes())

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": _RELOAD_DOMAIN,
                "arm": self.arm,
                "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
                "post_model_sha256": self.post_model_sha256,
                "reload_probe_sha256": self.reload_probe_sha256,
                "process_identities": list(self.process_identities),
                "output_sha256s": list(self.output_sha256s),
            }
        )

    def verify_output_bytes(self, values: tuple[bytes, bytes]) -> None:
        if len(values) != 2 or any(type(value) is not bytes for value in values):
            raise ValueError("reload output evidence must contain exactly two byte payloads")
        observed = tuple(_sha256(value) for value in values)
        if observed != self.output_sha256s or values[0] != values[1]:
            raise ValueError("reload output bytes differ from their frozen evidence")

    def verify_process_result_bytes(self, values: tuple[bytes, bytes]) -> None:
        if len(values) != 2 or any(type(value) is not bytes for value in values):
            raise ValueError("reload process evidence must contain exactly two payloads")
        from redco.analysis.stage_d_reload_supervisor import ReloadWorkerResult

        results = tuple(ReloadWorkerResult.from_bytes(value) for value in values)
        if tuple(result.identity for result in results) != self.process_identities:
            raise ValueError("reload process identities differ from their result bytes")
        for result, output_sha256 in zip(results, self.output_sha256s, strict=True):
            if (
                result.arm != self.arm
                or result.checkpoint_manifest_sha256 != self.checkpoint_manifest_sha256
                or result.post_model_sha256 != self.post_model_sha256
                or result.loaded_model_sha256 != self.post_model_sha256
                or result.reload_probe_sha256 != self.reload_probe_sha256
                or result.output_sha256 != output_sha256
            ):
                raise ValueError("reload process result differs from reload evidence")

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDReloadEvidence:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("reload evidence is not JSON") from error
        expected = {
            "schema_version",
            "domain",
            "arm",
            "checkpoint_manifest_sha256",
            "post_model_sha256",
            "reload_probe_sha256",
            "process_identities",
            "output_sha256s",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload.get("schema_version") != 1
            or payload.get("domain") != _RELOAD_DOMAIN
            or canonical_json(payload) != value
            or not isinstance(payload.get("process_identities"), list)
            or not isinstance(payload.get("output_sha256s"), list)
            or len(payload["process_identities"]) != 2
            or len(payload["output_sha256s"]) != 2
        ):
            raise ValueError("reload evidence is noncanonical or has different fields")
        return cls(
            _arm(payload["arm"]),
            _require_sha256(payload["checkpoint_manifest_sha256"], "checkpoint manifest sha256"),
            _require_sha256(payload["post_model_sha256"], "post-model sha256"),
            _require_sha256(payload["reload_probe_sha256"], "reload probe sha256"),
            tuple(payload["process_identities"]),  # type: ignore[arg-type]
            tuple(payload["output_sha256s"]),  # type: ignore[arg-type]
        )


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class StageDTrainerMetricsEvidence:
    arm: ArmName
    launch_id: str
    batch_identity: str
    trainer_step: int
    pre_model_sha256: str
    post_model_sha256: str
    model_changed: bool
    optimizer_updates: int
    loss: float
    grad_norm: float

    def __post_init__(self) -> None:
        _arm(self.arm)
        if not self.launch_id or not self.launch_id.isprintable():
            raise ValueError("trainer metrics launch identity is invalid")
        _require_sha256(self.batch_identity, "trainer metrics batch identity")
        _step(self.trainer_step)
        _require_sha256(self.pre_model_sha256, "trainer metrics pre-model sha256")
        _require_sha256(self.post_model_sha256, "trainer metrics post-model sha256")
        if type(self.model_changed) is not bool or self.model_changed != (
            self.pre_model_sha256 != self.post_model_sha256
        ):
            raise ValueError("trainer metrics model-changed flag is inconsistent")
        if type(self.optimizer_updates) is not int or self.optimizer_updates != 1:
            raise ValueError("trainer metrics must describe exactly one optimizer update")
        _finite(self.loss, "trainer loss")
        if _finite(self.grad_norm, "trainer gradient norm") < 0:
            raise ValueError("trainer gradient norm must be nonnegative")

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": _METRICS_DOMAIN,
                "arm": self.arm,
                "launch_id": self.launch_id,
                "batch_identity": self.batch_identity,
                "trainer_step": self.trainer_step,
                "pre_model_sha256": self.pre_model_sha256,
                "post_model_sha256": self.post_model_sha256,
                "model_changed": self.model_changed,
                "optimizer_updates": self.optimizer_updates,
                "loss": self.loss,
                "grad_norm": self.grad_norm,
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDTrainerMetricsEvidence:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("trainer metrics is not JSON") from error
        expected = {
            "schema_version",
            "domain",
            "arm",
            "launch_id",
            "batch_identity",
            "trainer_step",
            "pre_model_sha256",
            "post_model_sha256",
            "model_changed",
            "optimizer_updates",
            "loss",
            "grad_norm",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload.get("schema_version") != 1
            or payload.get("domain") != _METRICS_DOMAIN
            or canonical_json(payload) != value
        ):
            raise ValueError("trainer metrics is noncanonical or has different fields")
        return cls(
            _arm(payload["arm"]),
            payload["launch_id"],
            _require_sha256(payload["batch_identity"], "trainer metrics batch identity"),
            _step(payload["trainer_step"]),
            _require_sha256(payload["pre_model_sha256"], "trainer metrics pre-model"),
            _require_sha256(payload["post_model_sha256"], "trainer metrics post-model"),
            payload["model_changed"],
            payload["optimizer_updates"],
            _finite(payload["loss"], "trainer loss"),
            _finite(payload["grad_norm"], "trainer gradient norm"),
        )


def validate_metrics_bytes(
    value: bytes,
    *,
    arm: ArmName,
    launch_id: str,
    batch_identity: str,
    trainer_step: int,
    pre_model_sha256: str,
    post_model_sha256: str,
) -> StageDTrainerMetricsEvidence:
    evidence = StageDTrainerMetricsEvidence.from_bytes(value)
    if (
        evidence.arm != arm
        or evidence.launch_id != launch_id
        or evidence.batch_identity != batch_identity
        or evidence.trainer_step != trainer_step
        or evidence.pre_model_sha256 != pre_model_sha256
        or evidence.post_model_sha256 != post_model_sha256
    ):
        raise ValueError("trainer metrics differs from its frozen training identity")
    return evidence


__all__ = [
    "CheckpointMember",
    "StageDCheckpointManifest",
    "StageDReloadEvidence",
    "StageDTrainerMetricsEvidence",
    "adopt_prime_adapter_checkpoint",
    "validate_metrics_bytes",
]
