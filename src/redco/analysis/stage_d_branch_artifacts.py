"""Exact target roster and crash-safe branch artifact persistence for Stage D."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_receipt_ledger import inspect_ledger
from redco.analysis.stage_d_scientific_branch_group import BranchGroupArtifact
from redco.analysis.stage_d_source_contracts import SourceRollout
from redco.contracts import canonical_json


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class StageDBranchTarget:
    source_sha256: str
    group_id: str
    rollout_id: str
    decision_id: str
    target_id: str
    target_ordinal: int
    event_address: dict[str, str | int]

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "group_id": self.group_id,
            "rollout_id": self.rollout_id,
            "decision_id": self.decision_id,
            "target_id": self.target_id,
            "target_ordinal": self.target_ordinal,
            "event_address": self.event_address,
        }


@dataclass(frozen=True, slots=True)
class StageDBranchTargetRoster:
    planned_source_count: int
    completed_source_count: int
    eligible_source_count: int
    ineligible_source_count: int
    minimum_eligible_sources: int
    eligibility_passed: bool
    source_sha256s: tuple[str, ...]
    targets: tuple[StageDBranchTarget, ...]

    @classmethod
    def from_sources(
        cls,
        sources: tuple[SourceRollout, ...],
        *,
        planned_source_count: int,
        minimum_eligible_sources: int,
    ) -> StageDBranchTargetRoster:
        if planned_source_count < 1 or len(sources) != planned_source_count:
            raise ValueError("branch target roster requires every planned source disposition")
        if minimum_eligible_sources < 1 or minimum_eligible_sources > planned_source_count:
            raise ValueError("branch eligibility floor is outside the source denominator")
        source_sha256s = tuple(sorted(source.source_sha256 for source in sources))
        if len(set(source_sha256s)) != len(source_sha256s):
            raise ValueError("branch target roster contains duplicate sources")
        eligible: list[SourceRollout] = []
        targets: list[StageDBranchTarget] = []
        for source in sources:
            selected = tuple(
                decision
                for decision in source.decisions
                if decision.node_kind == "child" and decision.provenance.branch_selected
            )
            if selected and not source.branch_eligible:
                raise ValueError("ineligible source cannot contribute a branch target")
            if not selected:
                continue
            eligible.append(source)
            for decision in selected:
                assert decision.target_id is not None
                assert decision.target_ordinal is not None
                targets.append(
                    StageDBranchTarget(
                        source.source_sha256,
                        source.group_id,
                        source.rollout_id,
                        decision.decision_id,
                        decision.target_id,
                        decision.target_ordinal,
                        {
                            **decision.event_address.as_payload(),
                            "turn": decision.event_address.turn,
                        },
                    )
                )
        targets.sort(
            key=lambda target: (
                target.group_id,
                target.rollout_id,
                target.target_ordinal,
                target.decision_id,
            )
        )
        if len({(target.group_id, target.target_id) for target in targets}) != len(targets):
            raise ValueError("branch target roster contains duplicate scientific targets")
        return cls(
            planned_source_count,
            len(sources),
            len(eligible),
            len(sources) - len(eligible),
            minimum_eligible_sources,
            len(eligible) >= minimum_eligible_sources,
            source_sha256s,
            tuple(targets),
        )

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-branch-target-roster-v1",
                "planned_source_count": self.planned_source_count,
                "completed_source_count": self.completed_source_count,
                "eligible_source_count": self.eligible_source_count,
                "ineligible_source_count": self.ineligible_source_count,
                "minimum_eligible_sources": self.minimum_eligible_sources,
                "eligibility_passed": self.eligibility_passed,
                "source_sha256s": list(self.source_sha256s),
                "targets": [target.to_payload() for target in self.targets],
            }
        )

    @property
    def roster_sha256(self) -> str:
        return _sha256(self.to_bytes())


class StageDBranchArtifactStore:
    """Persist target roster before calls and artifacts around completion receipts."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.pending = root / "pending"
        self.completed = root / "completed"
        root.mkdir(parents=True, exist_ok=True)
        self.pending.mkdir(exist_ok=True)
        self.completed.mkdir(exist_ok=True)

    def assert_pristine(self) -> None:
        expected = {self.pending, self.completed}
        if set(self.root.iterdir()) - {self.root / "target-roster.json"} != expected:
            raise RuntimeError("branch artifact store contains unexpected entries")
        if any(self.pending.iterdir()) or any(self.completed.iterdir()):
            raise RuntimeError("branch artifact store contains stale artifacts")

    def persist_target_roster(self, roster: StageDBranchTargetRoster) -> Path:
        path = self.root / "target-roster.json"
        _exclusive_atomic_write(path, roster.to_bytes())
        return path

    def prepare(self, artifact: BranchGroupArtifact) -> str:
        value = artifact.to_bytes()
        digest = _sha256(value)
        _exclusive_atomic_write(self.pending / f"{digest}.json", value)
        return digest

    def commit(self, artifact: BranchGroupArtifact, completion_receipt: bytes) -> Path:
        value = artifact.to_bytes()
        digest = _sha256(value)
        receipt = _completion_receipt(completion_receipt)
        if (
            receipt["artifact_sha256"] != digest
            or receipt["group_id"] != artifact.commitment.group_id
            or receipt["target_id"] != artifact.commitment.target_id
            or receipt["training_batch_identity"] != artifact.training_batch_identity
        ):
            raise ValueError("branch completion receipt differs from its artifact")
        pending = self.pending / f"{digest}.json"
        if pending.read_bytes() != value:
            raise ValueError("branch completion lacks its exact pending artifact")
        destination = self.completed / f"{digest}.json"
        _exclusive_atomic_write(destination, value)
        pending.unlink()
        _fsync_directory(self.pending)
        return destination

    def recover_completed(self, ledger_root: Path) -> tuple[Path, ...]:
        scan = inspect_ledger(ledger_root)
        if scan.status != "active-clean":
            raise RuntimeError("branch artifact recovery requires an active-clean ledger")
        receipts = {
            receipt["artifact_sha256"]: receipt
            for (kind, _), receipt in scan.receipts.items()
            if kind == "branch_group_artifact_completed"
        }
        recovered: list[Path] = []
        for pending in sorted(self.pending.glob("*.json")):
            value = pending.read_bytes()
            digest = _sha256(value)
            if pending.name != f"{digest}.json" or digest not in receipts:
                raise RuntimeError("pending branch artifact lacks a durable completion receipt")
            destination = self.completed / pending.name
            _exclusive_atomic_write(destination, value)
            pending.unlink()
            recovered.append(destination)
        _fsync_directory(self.pending)
        return tuple(recovered)

    def completed_paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self.completed.glob("*.json")))


def _completion_receipt(value: bytes) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict) or canonical_json(payload) != value:
        raise ValueError("branch artifact completion must be canonical JSON")
    expected = {
        "schema_version",
        "receipt_kind",
        "group_id",
        "target_id",
        "artifact_sha256",
        "training_batch_identity",
        "branch_count",
        "continuation_replicates",
    }
    if set(payload) != expected or payload.get("receipt_kind") != "branch_group_artifact_completed":
        raise ValueError("branch artifact completion fields differ")
    return payload


def _exclusive_atomic_write(path: Path, value: bytes) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise FileExistsError(f"branch artifact collision at {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
