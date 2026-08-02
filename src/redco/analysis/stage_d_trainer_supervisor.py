"""Durable single-use state machine for the three Stage-D trainer launches."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from redco.analysis.stage_d_objective_binding import ArmName
from redco.contracts import canonical_json

_DOMAIN = "redco-stage-d-trainer-run-ledger-v1"
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
class ArmRunState:
    arm: ArmName
    launch_attempts: int = 0
    preupdate_failures: int = 0
    active_launch_id: str | None = None
    batch_verified: bool = False
    optimizer_started: bool = False
    optimizer_completed: bool = False
    checkpoint_committed: bool = False
    checkpoint_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class TrainerRunSnapshot:
    campaign_manifest_sha256: str
    initialization_sha256: str
    arm_order: tuple[ArmName, ...]
    batch_identities: tuple[tuple[ArmName, str], ...]
    trainer_config_sha256s: tuple[tuple[ArmName, str], ...]
    states: tuple[ArmRunState, ...]
    head_sha256: str
    record_count: int

    def state(self, arm: ArmName) -> ArmRunState:
        return dict(zip(self.arm_order, self.states, strict=True))[arm]


class StageDTrainerRunLedger:
    """Append-only supervisor ledger; every mutation revalidates the full chain."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.records = root / "records"
        self.lock_path = root / "writer.lock"

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        campaign_manifest_sha256: str,
        initialization_sha256: str,
        batch_identities: Mapping[ArmName, str],
        trainer_config_sha256s: Mapping[ArmName, str],
        arm_order: tuple[ArmName, ...] = _ARMS,
    ) -> StageDTrainerRunLedger:
        if root.exists():
            raise FileExistsError(f"trainer run ledger already exists: {root}")
        if tuple(sorted(arm_order)) != tuple(sorted(_ARMS)) or len(set(arm_order)) != 3:
            raise ValueError("trainer arm order must be one permutation of all three arms")
        if set(batch_identities) != set(_ARMS) or set(trainer_config_sha256s) != set(_ARMS):
            raise ValueError("trainer genesis must bind all three arms")
        _require_sha256(campaign_manifest_sha256, "campaign_manifest_sha256")
        _require_sha256(initialization_sha256, "initialization_sha256")
        for digest in (*batch_identities.values(), *trainer_config_sha256s.values()):
            _require_sha256(digest, "trainer genesis digest")
        root.mkdir(parents=True)
        (root / "records").mkdir()
        ledger = cls(root)
        ledger._append_unlocked(
            "genesis",
            {
                "campaign_manifest_sha256": campaign_manifest_sha256,
                "initialization_sha256": initialization_sha256,
                "arm_order": list(arm_order),
                "batch_identities": dict(sorted(batch_identities.items())),
                "trainer_config_sha256s": dict(sorted(trainer_config_sha256s.items())),
            },
        )
        ledger.inspect()
        return ledger

    def inspect(self) -> TrainerRunSnapshot:
        return _scan(self.records)

    def claim_launch(self, *, arm: ArmName, launch_id: str) -> None:
        if not launch_id:
            raise ValueError("trainer launch ID must be nonempty")
        with _locked(self.lock_path):
            snapshot = self.inspect()
            state = snapshot.state(arm)
            expected = next(
                (
                    item
                    for item in snapshot.arm_order
                    if not snapshot.state(item).checkpoint_committed
                ),
                None,
            )
            if arm != expected:
                raise RuntimeError("trainer launch violates the frozen arm order")
            if state.active_launch_id is not None:
                raise RuntimeError("trainer arm already has an active launch")
            if state.optimizer_started or state.checkpoint_committed:
                raise RuntimeError("trainer arm is permanently single-use after optimizer start")
            if state.launch_attempts > 1 or state.preupdate_failures > 1:
                raise RuntimeError("trainer arm exhausted its one bounded repair")
            self._append_unlocked(
                "launch_intent",
                {
                    "arm": arm,
                    "launch_id": launch_id,
                    "launch_ordinal": state.launch_attempts,
                    "batch_identity": dict(snapshot.batch_identities)[arm],
                    "trainer_config_sha256": dict(snapshot.trainer_config_sha256s)[arm],
                },
            )

    def mark_batch_verified(self, *, arm: ArmName, launch_id: str, batch_identity: str) -> None:
        self._transition(
            "batch_verified",
            arm=arm,
            launch_id=launch_id,
            extra={"batch_identity": _require_sha256(batch_identity, "batch_identity")},
            require="launched",
        )

    def mark_optimizer_started(self, *, arm: ArmName, launch_id: str, trainer_step: int) -> None:
        if type(trainer_step) is not int or trainer_step < 1:
            raise ValueError("trainer step must be positive")
        self._transition(
            "optimizer_started",
            arm=arm,
            launch_id=launch_id,
            extra={"trainer_step": trainer_step},
            require="batch_verified",
        )

    def mark_optimizer_completed(
        self,
        *,
        arm: ArmName,
        launch_id: str,
        trainer_step: int,
    ) -> None:
        if type(trainer_step) is not int or trainer_step < 1:
            raise ValueError("trainer step must be positive")
        self._transition(
            "optimizer_completed",
            arm=arm,
            launch_id=launch_id,
            extra={"trainer_step": trainer_step},
            require="optimizer_started",
        )

    def commit_checkpoint(
        self,
        *,
        arm: ArmName,
        launch_id: str,
        checkpoint_sha256: str,
        metrics_sha256: str,
        reload_evidence_sha256: str,
    ) -> None:
        extra = {
            "checkpoint_sha256": _require_sha256(checkpoint_sha256, "checkpoint_sha256"),
            "metrics_sha256": _require_sha256(metrics_sha256, "metrics_sha256"),
            "reload_evidence_sha256": _require_sha256(
                reload_evidence_sha256, "reload_evidence_sha256"
            ),
        }
        self._transition(
            "checkpoint_committed",
            arm=arm,
            launch_id=launch_id,
            extra=extra,
            require="optimizer_completed",
        )

    def record_preupdate_failure(
        self,
        *,
        arm: ArmName,
        launch_id: str,
        reason: str,
        evidence_sha256: str,
    ) -> None:
        if not reason:
            raise ValueError("pre-update failure reason must be nonempty")
        self._transition(
            "preupdate_failure",
            arm=arm,
            launch_id=launch_id,
            extra={
                "reason": reason,
                "evidence_sha256": _require_sha256(evidence_sha256, "evidence_sha256"),
            },
            require="preupdate",
        )

    def _transition(
        self,
        kind: str,
        *,
        arm: ArmName,
        launch_id: str,
        extra: Mapping[str, Any],
        require: Literal[
            "launched", "batch_verified", "optimizer_started", "optimizer_completed", "preupdate"
        ],
    ) -> None:
        with _locked(self.lock_path):
            state = self.inspect().state(arm)
            if state.active_launch_id != launch_id:
                raise RuntimeError("trainer transition differs from the active launch")
            valid = {
                "launched": not state.batch_verified and not state.optimizer_started,
                "batch_verified": state.batch_verified and not state.optimizer_started,
                "optimizer_started": state.optimizer_started and not state.optimizer_completed,
                "optimizer_completed": state.optimizer_completed and not state.checkpoint_committed,
                "preupdate": not state.optimizer_started and state.preupdate_failures == 0,
            }[require]
            if not valid:
                raise RuntimeError(f"trainer transition {kind} is out of order")
            self._append_unlocked(kind, {"arm": arm, "launch_id": launch_id, **dict(extra)})

    def _append_unlocked(self, kind: str, event: Mapping[str, Any]) -> None:
        existing = sorted(self.records.glob("*.json"))
        offset = len(existing)
        prior = "0" * 64 if not existing else _sha256(existing[-1].read_bytes())
        record = canonical_json(
            {
                "schema_version": 1,
                "domain": _DOMAIN,
                "offset": offset,
                "prior_record_sha256": prior,
                "record_kind": kind,
                "event": dict(event),
            }
        )
        path = self.records / f"{offset:08d}.json"
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(record)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.records)


def _scan(records_root: Path) -> TrainerRunSnapshot:
    paths = sorted(records_root.glob("*.json"))
    if not paths:
        raise ValueError("trainer run ledger is empty")
    records: list[dict[str, Any]] = []
    prior = "0" * 64
    for offset, path in enumerate(paths):
        if path.name != f"{offset:08d}.json":
            raise ValueError("trainer run ledger offsets are not contiguous")
        value = path.read_bytes()
        payload = json.loads(value)
        if not isinstance(payload, dict) or canonical_json(payload) != value:
            raise ValueError("trainer run record must be canonical JSON")
        if (
            payload.get("schema_version") != 1
            or payload.get("domain") != _DOMAIN
            or payload.get("offset") != offset
            or payload.get("prior_record_sha256") != prior
            or not isinstance(payload.get("event"), dict)
        ):
            raise ValueError("trainer run record chain is invalid")
        records.append(payload)
        prior = _sha256(value)
    genesis = records[0]
    if genesis["record_kind"] != "genesis":
        raise ValueError("trainer run ledger lacks genesis")
    event = genesis["event"]
    if set(event) != {
        "campaign_manifest_sha256",
        "initialization_sha256",
        "arm_order",
        "batch_identities",
        "trainer_config_sha256s",
    }:
        raise ValueError("trainer run genesis fields differ")
    order = tuple(event["arm_order"])
    if tuple(sorted(order)) != tuple(sorted(_ARMS)) or len(set(order)) != 3:
        raise ValueError("trainer run arm order is invalid")
    batch_ids = _arm_digests(event["batch_identities"], "batch identities")
    configs = _arm_digests(event["trainer_config_sha256s"], "trainer configs")
    states = {arm: ArmRunState(arm) for arm in order}
    for record in records[1:]:
        _apply_event(states, record["record_kind"], record["event"], batch_ids, configs)
    return TrainerRunSnapshot(
        _require_sha256(event["campaign_manifest_sha256"], "campaign manifest"),
        _require_sha256(event["initialization_sha256"], "initialization"),
        order,
        tuple(sorted(batch_ids.items())),
        tuple(sorted(configs.items())),
        tuple(states[arm] for arm in order),
        prior,
        len(records),
    )


def _apply_event(
    states: dict[ArmName, ArmRunState],
    kind: str,
    event: Mapping[str, Any],
    batch_ids: Mapping[ArmName, str],
    configs: Mapping[ArmName, str],
) -> None:
    arm = event.get("arm")
    if arm not in states or not isinstance(event.get("launch_id"), str):
        raise ValueError("trainer run event has invalid identity")
    state = states[arm]
    launch_id = event["launch_id"]
    from dataclasses import replace

    if kind == "launch_intent":
        if (
            state.active_launch_id is not None
            or state.optimizer_started
            or state.checkpoint_committed
            or event.get("launch_ordinal") != state.launch_attempts
            or event.get("batch_identity") != batch_ids[arm]
            or event.get("trainer_config_sha256") != configs[arm]
        ):
            raise ValueError("trainer launch intent violates single-use state")
        states[arm] = replace(
            state,
            launch_attempts=state.launch_attempts + 1,
            active_launch_id=launch_id,
            batch_verified=False,
        )
    elif kind == "batch_verified":
        if state.active_launch_id != launch_id or state.batch_verified:
            raise ValueError("trainer batch verification is duplicated or unclaimed")
        if event.get("batch_identity") != batch_ids[arm]:
            raise ValueError("trainer verified a different batch identity")
        states[arm] = replace(state, batch_verified=True)
    elif kind == "optimizer_started":
        if (
            state.active_launch_id != launch_id
            or not state.batch_verified
            or state.optimizer_started
        ):
            raise ValueError("trainer optimizer start is out of order")
        states[arm] = replace(state, optimizer_started=True)
    elif kind == "optimizer_completed":
        if (
            state.active_launch_id != launch_id
            or not state.optimizer_started
            or state.optimizer_completed
        ):
            raise ValueError("trainer optimizer completion is out of order")
        states[arm] = replace(state, optimizer_completed=True)
    elif kind == "checkpoint_committed":
        if (
            state.active_launch_id != launch_id
            or not state.optimizer_completed
            or state.checkpoint_committed
        ):
            raise ValueError("trainer checkpoint commit is out of order")
        for name in ("checkpoint_sha256", "metrics_sha256", "reload_evidence_sha256"):
            _require_sha256(event.get(name), name)
        states[arm] = replace(
            state,
            active_launch_id=None,
            checkpoint_committed=True,
            checkpoint_sha256=event["checkpoint_sha256"],
        )
    elif kind == "preupdate_failure":
        if (
            state.active_launch_id != launch_id
            or state.optimizer_started
            or state.preupdate_failures != 0
            or not isinstance(event.get("reason"), str)
            or not event["reason"]
        ):
            raise ValueError("trainer pre-update repair is not eligible")
        _require_sha256(event.get("evidence_sha256"), "failure evidence")
        states[arm] = replace(
            state,
            active_launch_id=None,
            batch_verified=False,
            preupdate_failures=1,
        )
    else:
        raise ValueError(f"unsupported trainer run event: {kind}")


def _arm_digests(value: object, name: str) -> dict[ArmName, str]:
    if not isinstance(value, dict) or set(value) != set(_ARMS):
        raise ValueError(f"trainer run {name} must cover all arms")
    return {arm: _require_sha256(value[arm], f"{name}.{arm}") for arm in _ARMS}


class _locked:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.descriptor: int | None = None

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            _lock_descriptor(descriptor)
        except BaseException:
            os.close(descriptor)
            raise
        self.descriptor = descriptor

    def __exit__(self, *_: object) -> None:
        assert self.descriptor is not None
        _unlock_descriptor(self.descriptor)
        os.close(self.descriptor)
        self.descriptor = None


def _lock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
    else:
        fcntl = importlib.import_module("fcntl")
        fcntl.flock(descriptor, fcntl.LOCK_EX)


def _unlock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        fcntl = importlib.import_module("fcntl")
        fcntl.flock(descriptor, fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
