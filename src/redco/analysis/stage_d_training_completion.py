"""Canonical terminal summary for the single global Stage-D trainer ledger."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

from redco.analysis.stage_d_checkpoint_evidence import (
    StageDCheckpointManifest,
    StageDReloadEvidence,
    StageDTrainerMetricsEvidence,
)
from redco.analysis.stage_d_objective_binding import ArmName
from redco.analysis.stage_d_reload_supervisor import ReloadWorkerResult
from redco.analysis.stage_d_trainer_supervisor import StageDTrainerRunLedger
from redco.contracts import canonical_json

_DOMAIN = "redco-stage-d-training-completion-v1"
_ARMS: tuple[ArmName, ...] = ("stock", "branch-global", "local")


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
class TrainingArmCompletion:
    arm: ArmName
    post_model_sha256: str
    checkpoint_manifest_sha256: str
    checkpoint_member_sha256s: tuple[str, ...]
    metrics_sha256: str
    reload_evidence_sha256: str
    reload_output_sha256: str
    reload_process_result_sha256s: tuple[str, str]

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise ValueError("training completion arm is invalid")
        for name in (
            "post_model_sha256",
            "checkpoint_manifest_sha256",
            "metrics_sha256",
            "reload_evidence_sha256",
            "reload_output_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not self.checkpoint_member_sha256s:
            raise ValueError("training completion has no checkpoint members")
        for digest in (
            *self.checkpoint_member_sha256s,
            *self.reload_process_result_sha256s,
        ):
            _require_sha256(digest, "training completion evidence sha256")
        if len(set(self.reload_process_result_sha256s)) != 2:
            raise ValueError("training completion requires two process results")

    def to_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "post_model_sha256": self.post_model_sha256,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "checkpoint_member_sha256s": list(self.checkpoint_member_sha256s),
            "metrics_sha256": self.metrics_sha256,
            "reload_evidence_sha256": self.reload_evidence_sha256,
            "reload_output_sha256": self.reload_output_sha256,
            "reload_process_result_sha256s": list(self.reload_process_result_sha256s),
        }

    @classmethod
    def from_payload(cls, value: object) -> TrainingArmCompletion:
        fields = {
            "arm",
            "post_model_sha256",
            "checkpoint_manifest_sha256",
            "checkpoint_member_sha256s",
            "metrics_sha256",
            "reload_evidence_sha256",
            "reload_output_sha256",
            "reload_process_result_sha256s",
        }
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or not isinstance(value.get("checkpoint_member_sha256s"), list)
            or not isinstance(value.get("reload_process_result_sha256s"), list)
        ):
            raise ValueError("training arm completion fields differ")
        return cls(
            value["arm"],
            value["post_model_sha256"],
            value["checkpoint_manifest_sha256"],
            tuple(value["checkpoint_member_sha256s"]),
            value["metrics_sha256"],
            value["reload_evidence_sha256"],
            value["reload_output_sha256"],
            tuple(value["reload_process_result_sha256s"]),
        )


@dataclass(frozen=True, slots=True)
class StageDTrainingCompletion:
    campaign_manifest_sha256: str
    protocol_manifest_sha256: str
    trainer_ledger_head_sha256: str
    trainer_record_count: int
    record_sha256s: tuple[str, ...]
    evidence_sha256s: tuple[str, ...]
    arms: tuple[TrainingArmCompletion, ...]

    def __post_init__(self) -> None:
        for name in (
            "campaign_manifest_sha256",
            "protocol_manifest_sha256",
            "trainer_ledger_head_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            type(self.trainer_record_count) is not int
            or self.trainer_record_count < 1
            or len(self.record_sha256s) != self.trainer_record_count
            or self.record_sha256s[-1] != self.trainer_ledger_head_sha256
        ):
            raise ValueError("training completion ledger chain is inconsistent")
        if tuple(item.arm for item in self.arms) != _ARMS:
            raise ValueError("training completion requires exact three-arm order")
        for roster_name, roster in (
            ("record", self.record_sha256s),
            ("evidence", self.evidence_sha256s),
        ):
            if roster_name == "evidence" and tuple(sorted(set(roster))) != roster:
                raise ValueError("training evidence roster is not sorted and unique")
            for digest in roster:
                _require_sha256(digest, f"training completion {roster_name} sha256")

    @property
    def completion_sha256(self) -> str:
        return _sha256(self.to_bytes())

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": _DOMAIN,
                "campaign_manifest_sha256": self.campaign_manifest_sha256,
                "protocol_manifest_sha256": self.protocol_manifest_sha256,
                "trainer_ledger_head_sha256": self.trainer_ledger_head_sha256,
                "trainer_record_count": self.trainer_record_count,
                "record_sha256s": list(self.record_sha256s),
                "evidence_sha256s": list(self.evidence_sha256s),
                "arms": [item.to_payload() for item in self.arms],
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDTrainingCompletion:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("training completion is not JSON") from error
        fields = {
            "schema_version",
            "domain",
            "campaign_manifest_sha256",
            "protocol_manifest_sha256",
            "trainer_ledger_head_sha256",
            "trainer_record_count",
            "record_sha256s",
            "evidence_sha256s",
            "arms",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or payload.get("schema_version") != 1
            or payload.get("domain") != _DOMAIN
            or canonical_json(payload) != value
            or not isinstance(payload.get("record_sha256s"), list)
            or not isinstance(payload.get("evidence_sha256s"), list)
            or not isinstance(payload.get("arms"), list)
        ):
            raise ValueError("training completion fields differ")
        return cls(
            payload["campaign_manifest_sha256"],
            payload["protocol_manifest_sha256"],
            payload["trainer_ledger_head_sha256"],
            payload["trainer_record_count"],
            tuple(payload["record_sha256s"]),
            tuple(payload["evidence_sha256s"]),
            tuple(TrainingArmCompletion.from_payload(item) for item in payload["arms"]),
        )

    @classmethod
    def build(cls, ledger: StageDTrainerRunLedger) -> StageDTrainingCompletion:
        snapshot = ledger.inspect()
        if any(not snapshot.state(arm).checkpoint_committed for arm in _ARMS):
            raise RuntimeError("training completion requires all three checkpoints")
        record_paths = sorted(ledger.records.glob("*.json"))
        evidence_paths = sorted(ledger.evidence.iterdir())
        if any(
            path.is_symlink() or not path.is_file() for path in (*record_paths, *evidence_paths)
        ):
            raise ValueError("training ledger contains non-regular durable state")
        record_sha256s = tuple(_sha256(path.read_bytes()) for path in record_paths)
        evidence = {path.name: path.read_bytes() for path in evidence_paths}
        if any(_sha256(value) != digest for digest, value in evidence.items()):
            raise ValueError("training evidence filename differs from its bytes")
        worker_results: dict[ArmName, list[tuple[str, ReloadWorkerResult]]] = {
            arm: [] for arm in _ARMS
        }
        for digest, value in evidence.items():
            try:
                result = ReloadWorkerResult.from_bytes(value)
            except ValueError:
                continue
            worker_results[result.arm].append((digest, result))
        arms = []
        for arm in _ARMS:
            state = snapshot.state(arm)
            assert state.checkpoint_sha256 is not None
            assert state.metrics_sha256 is not None
            assert state.reload_evidence_sha256 is not None
            assert state.post_model_sha256 is not None
            manifest = StageDCheckpointManifest.from_bytes(evidence[state.checkpoint_sha256])
            metrics = StageDTrainerMetricsEvidence.from_bytes(evidence[state.metrics_sha256])
            reload_evidence = StageDReloadEvidence.from_bytes(
                evidence[state.reload_evidence_sha256]
            )
            results = sorted(
                (
                    (digest, result)
                    for digest, result in worker_results[arm]
                    if result.identity in reload_evidence.process_identities
                ),
                key=lambda item: reload_evidence.process_identities.index(item[1].identity),
            )
            if (
                manifest.post_model_sha256 != state.post_model_sha256
                or metrics.post_model_sha256 != state.post_model_sha256
                or reload_evidence.post_model_sha256 != state.post_model_sha256
                or len(results) != 2
            ):
                raise ValueError("training arm evidence closure is inconsistent")
            manifest.verify_member_evidence(evidence)
            output_bytes = tuple(evidence[digest] for digest in reload_evidence.output_sha256s)
            process_result_bytes = tuple(evidence[digest] for digest, _ in results)
            reload_evidence.verify_output_bytes(cast(tuple[bytes, bytes], output_bytes))
            reload_evidence.verify_process_result_bytes(
                cast(tuple[bytes, bytes], process_result_bytes)
            )
            process_result_sha256s = cast(
                tuple[str, str], tuple(digest for digest, _ in results)
            )
            arms.append(
                TrainingArmCompletion(
                    arm,
                    state.post_model_sha256,
                    state.checkpoint_sha256,
                    tuple(member.sha256 for member in manifest.members),
                    state.metrics_sha256,
                    state.reload_evidence_sha256,
                    reload_evidence.output_sha256s[0],
                    process_result_sha256s,
                )
            )
        return cls(
            snapshot.campaign_manifest_sha256,
            snapshot.protocol_manifest_sha256,
            snapshot.head_sha256,
            snapshot.record_count,
            record_sha256s,
            tuple(sorted(evidence)),
            tuple(arms),
        )

    def verify_ledger(self, ledger: StageDTrainerRunLedger) -> None:
        if StageDTrainingCompletion.build(ledger) != self:
            raise ValueError("trainer ledger differs from its terminal completion")


__all__ = ["StageDTrainingCompletion", "TrainingArmCompletion"]
