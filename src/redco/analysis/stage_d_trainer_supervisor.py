"""Durable single-use state machine for the three Stage-D trainer launches."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from redco.analysis.stage_d_checkpoint_evidence import (
    StageDCheckpointManifest,
    StageDReloadEvidence,
    validate_metrics_bytes,
)
from redco.analysis.stage_d_file_lock import exclusive_file_lock, fsync_directory
from redco.analysis.stage_d_objective_binding import ArmName
from redco.analysis.stage_d_process_supervision import TrainerProcessStartReceipt
from redco.contracts import canonical_json

_DOMAIN = "redco-stage-d-trainer-run-ledger-v3"
_ARMS: tuple[ArmName, ...] = ("stock", "branch-global", "local")
FaultHook = Callable[[str, Path], None]
_RECORD_FIELDS = {
    "schema_version",
    "domain",
    "offset",
    "prior_record_sha256",
    "record_kind",
    "event",
}
_EVENT_FIELDS = {
    "launch_intent": {
        "arm",
        "launch_id",
        "launch_ordinal",
        "batch_identity",
        "trainer_config_sha256",
    },
    "initialization_verified": {"arm", "launch_id", "observed_pre_model_sha256"},
    "process_started": {"arm", "launch_id", "process_receipt_sha256"},
    "batch_verified": {"arm", "launch_id", "batch_identity"},
    "optimizer_started": {"arm", "launch_id", "trainer_step"},
    "optimizer_completed": {
        "arm",
        "launch_id",
        "trainer_step",
        "post_model_sha256",
        "model_changed",
    },
    "checkpoint_committed": {
        "arm",
        "launch_id",
        "checkpoint_sha256",
        "metrics_sha256",
        "reload_evidence_sha256",
        "trainer_step",
    },
    "preupdate_failure": {"arm", "launch_id", "reason", "evidence_sha256"},
}


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
    process_started: bool = False
    process_receipt_sha256: str | None = None
    initialization_verified: bool = False
    batch_verified: bool = False
    optimizer_started: bool = False
    optimizer_completed: bool = False
    post_model_sha256: str | None = None
    model_changed: bool | None = None
    checkpoint_committed: bool = False
    checkpoint_sha256: str | None = None
    metrics_sha256: str | None = None
    reload_evidence_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class TrainerRunSnapshot:
    campaign_manifest_sha256: str
    protocol_manifest_sha256: str
    shared_initialization_manifest_sha256: str
    expected_pre_model_sha256: str
    expected_base_model_manifest_sha256: str
    reload_probe_sha256: str
    trainer_step: int
    arm_order: tuple[ArmName, ...]
    batch_identities: tuple[tuple[ArmName, str], ...]
    trainer_config_sha256s: tuple[tuple[ArmName, str], ...]
    process_command_sha256s: tuple[tuple[ArmName, str], ...]
    process_environment_sha256s: tuple[tuple[ArmName, str], ...]
    states: tuple[ArmRunState, ...]
    head_sha256: str
    record_count: int

    def state(self, arm: ArmName) -> ArmRunState:
        return dict(zip(self.arm_order, self.states, strict=True))[arm]


class StageDTrainerRunLedger:
    """Append-only supervisor ledger; every mutation revalidates the full chain."""

    def __init__(self, root: Path, *, fault_hook: FaultHook | None = None) -> None:
        self.root = root
        self.records = root / "records"
        self.evidence = root / "evidence"
        self.lock_path = root / "writer.lock"
        self._fault_hook = fault_hook

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        campaign_manifest_sha256: str,
        protocol_manifest_sha256: str,
        shared_initialization_manifest_sha256: str,
        expected_pre_model_sha256: str,
        expected_base_model_manifest_sha256: str,
        reload_probe_sha256: str,
        trainer_step: int,
        batch_identities: Mapping[ArmName, str],
        trainer_config_sha256s: Mapping[ArmName, str],
        process_command_sha256s: Mapping[ArmName, str],
        process_environment_sha256s: Mapping[ArmName, str],
        arm_order: tuple[ArmName, ...] = _ARMS,
        fault_hook: FaultHook | None = None,
    ) -> StageDTrainerRunLedger:
        if arm_order != _ARMS:
            raise ValueError("trainer arm order must be exactly stock, branch-global, local")
        if any(
            set(mapping) != set(_ARMS)
            for mapping in (
                batch_identities,
                trainer_config_sha256s,
                process_command_sha256s,
                process_environment_sha256s,
            )
        ):
            raise ValueError("trainer genesis must bind all three arms")
        _require_sha256(campaign_manifest_sha256, "campaign_manifest_sha256")
        _require_sha256(protocol_manifest_sha256, "protocol_manifest_sha256")
        _require_sha256(
            shared_initialization_manifest_sha256,
            "shared_initialization_manifest_sha256",
        )
        _require_sha256(expected_pre_model_sha256, "expected_pre_model_sha256")
        _require_sha256(
            expected_base_model_manifest_sha256,
            "expected_base_model_manifest_sha256",
        )
        _require_sha256(reload_probe_sha256, "reload_probe_sha256")
        if type(trainer_step) is not int or trainer_step < 1:
            raise ValueError("trainer_step must be a positive integer")
        for digest in (
            *batch_identities.values(),
            *trainer_config_sha256s.values(),
            *process_command_sha256s.values(),
            *process_environment_sha256s.values(),
        ):
            _require_sha256(digest, "trainer genesis digest")
        root.mkdir(parents=True, exist_ok=True)
        allowed = {"records", "evidence", "writer.lock"}
        if any(item.name not in allowed for item in root.iterdir()):
            raise FileExistsError(f"trainer run ledger has unknown contents: {root}")
        (root / "records").mkdir(exist_ok=True)
        (root / "evidence").mkdir(exist_ok=True)
        ledger = cls(root, fault_hook=fault_hook)
        genesis = {
                "campaign_manifest_sha256": campaign_manifest_sha256,
                "protocol_manifest_sha256": protocol_manifest_sha256,
                "shared_initialization_manifest_sha256": (
                    shared_initialization_manifest_sha256
                ),
                "expected_pre_model_sha256": expected_pre_model_sha256,
                "expected_base_model_manifest_sha256": (
                    expected_base_model_manifest_sha256
                ),
                "reload_probe_sha256": reload_probe_sha256,
                "trainer_step": trainer_step,
                "arm_order": list(arm_order),
                "batch_identities": dict(sorted(batch_identities.items())),
                "trainer_config_sha256s": dict(sorted(trainer_config_sha256s.items())),
                "process_command_sha256s": dict(
                    sorted(process_command_sha256s.items())
                ),
                "process_environment_sha256s": dict(
                    sorted(process_environment_sha256s.items())
                ),
            }
        with exclusive_file_lock(ledger.lock_path):
            if not any(ledger.records.glob("*.json")):
                ledger._append_unlocked("genesis", genesis)
            snapshot = ledger.inspect()
            expected = (
                campaign_manifest_sha256,
                protocol_manifest_sha256,
                shared_initialization_manifest_sha256,
                expected_pre_model_sha256,
                expected_base_model_manifest_sha256,
                reload_probe_sha256,
                trainer_step,
                arm_order,
                tuple(sorted(batch_identities.items())),
                tuple(sorted(trainer_config_sha256s.items())),
                tuple(sorted(process_command_sha256s.items())),
                tuple(sorted(process_environment_sha256s.items())),
            )
            observed = (
                snapshot.campaign_manifest_sha256,
                snapshot.protocol_manifest_sha256,
                snapshot.shared_initialization_manifest_sha256,
                snapshot.expected_pre_model_sha256,
                snapshot.expected_base_model_manifest_sha256,
                snapshot.reload_probe_sha256,
                snapshot.trainer_step,
                snapshot.arm_order,
                snapshot.batch_identities,
                snapshot.trainer_config_sha256s,
                snapshot.process_command_sha256s,
                snapshot.process_environment_sha256s,
            )
            if observed != expected:
                raise FileExistsError("trainer run ledger has a different genesis")
        return ledger

    def inspect(self) -> TrainerRunSnapshot:
        return _scan(self.records)

    def claim_launch(self, *, arm: ArmName, launch_id: str) -> None:
        if not launch_id:
            raise ValueError("trainer launch ID must be nonempty")
        with exclusive_file_lock(self.lock_path):
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
            campaign_failures = sum(item.preupdate_failures for item in snapshot.states)
            if state.launch_attempts > 1 or state.preupdate_failures > 1:
                raise RuntimeError("trainer arm exhausted the campaign's bounded repair")
            if state.launch_attempts == 1 and campaign_failures != 1:
                raise RuntimeError("trainer relaunch lacks the campaign's sole repair receipt")
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

    def mark_initialization_verified(
        self,
        *,
        arm: ArmName,
        launch_id: str,
        observed_pre_model_sha256: str,
    ) -> None:
        self._transition(
            "initialization_verified",
            arm=arm,
            launch_id=launch_id,
            extra={
                "observed_pre_model_sha256": _require_sha256(
                    observed_pre_model_sha256,
                    "observed_pre_model_sha256",
                )
            },
            require="launched",
        )

    def mark_process_started(
        self,
        *,
        arm: ArmName,
        launch_id: str,
        process_receipt_bytes: bytes,
    ) -> None:
        receipt = TrainerProcessStartReceipt.from_bytes(process_receipt_bytes)
        with exclusive_file_lock(self.lock_path):
            snapshot = self.inspect()
            state = snapshot.state(arm)
            digest = _sha256(process_receipt_bytes)
            if state.process_started:
                if (
                    state.active_launch_id != launch_id
                    or state.process_receipt_sha256 != digest
                ):
                    raise RuntimeError("trainer process receipt differs from its launch")
                return
            if (
                state.active_launch_id != launch_id
                or state.process_started
                or state.initialization_verified
                or receipt.arm != arm
                or receipt.launch_id != launch_id
                or receipt.command_sha256
                != dict(snapshot.process_command_sha256s)[arm]
                or receipt.environment_manifest_sha256
                != dict(snapshot.process_environment_sha256s)[arm]
            ):
                raise RuntimeError("trainer process receipt differs from its launch")
            digest = self._put_evidence_unlocked(process_receipt_bytes)
            self._append_unlocked(
                "process_started",
                {
                    "arm": arm,
                    "launch_id": launch_id,
                    "process_receipt_sha256": digest,
                },
            )

    def mark_batch_verified(self, *, arm: ArmName, launch_id: str, batch_identity: str) -> None:
        self._transition(
            "batch_verified",
            arm=arm,
            launch_id=launch_id,
            extra={"batch_identity": _require_sha256(batch_identity, "batch_identity")},
            require="initialization_verified",
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
        post_model_sha256: str,
    ) -> None:
        if type(trainer_step) is not int or trainer_step < 1:
            raise ValueError("trainer step must be positive")
        post_model_sha256 = _require_sha256(post_model_sha256, "post_model_sha256")
        with exclusive_file_lock(self.lock_path):
            snapshot = self.inspect()
            state = snapshot.state(arm)
            if (
                state.active_launch_id != launch_id
                or not state.optimizer_started
                or state.optimizer_completed
                or trainer_step != snapshot.trainer_step
            ):
                raise RuntimeError("trainer optimizer completion is out of order")
            self._append_unlocked(
                "optimizer_completed",
                {
                    "arm": arm,
                    "launch_id": launch_id,
                    "trainer_step": trainer_step,
                    "post_model_sha256": post_model_sha256,
                    "model_changed": (
                        post_model_sha256 != snapshot.expected_pre_model_sha256
                    ),
                },
            )

    def commit_checkpoint(
        self,
        *,
        arm: ArmName,
        launch_id: str,
        checkpoint_root: Path,
        checkpoint_manifest_bytes: bytes,
        metrics_bytes: bytes,
        reload_evidence_bytes: bytes,
        reload_output_bytes: tuple[bytes, bytes],
        reload_process_result_bytes: tuple[bytes, bytes],
        trainer_step: int,
    ) -> None:
        if type(trainer_step) is not int or trainer_step < 1:
            raise ValueError("trainer step must be positive")
        manifest = StageDCheckpointManifest.from_bytes(checkpoint_manifest_bytes)
        manifest.verify_directory(checkpoint_root)
        reload_evidence = StageDReloadEvidence.from_bytes(reload_evidence_bytes)
        reload_evidence.verify_output_bytes(reload_output_bytes)
        reload_evidence.verify_process_result_bytes(reload_process_result_bytes)
        with exclusive_file_lock(self.lock_path):
            snapshot = self.inspect()
            state = snapshot.state(arm)
            if (
                state.active_launch_id != launch_id
                or not state.optimizer_completed
                or state.checkpoint_committed
                or trainer_step != snapshot.trainer_step
                or manifest.arm != arm
                or manifest.trainer_step != trainer_step
                or manifest.base_model_manifest_sha256
                != snapshot.expected_base_model_manifest_sha256
                or manifest.post_model_sha256 != state.post_model_sha256
                or reload_evidence.arm != arm
                or reload_evidence.checkpoint_manifest_sha256
                != manifest.manifest_sha256
                or reload_evidence.post_model_sha256 != state.post_model_sha256
                or reload_evidence.reload_probe_sha256 != snapshot.reload_probe_sha256
            ):
                raise RuntimeError("checkpoint evidence differs from the completed trainer state")
            assert state.post_model_sha256 is not None
            validate_metrics_bytes(
                metrics_bytes,
                arm=arm,
                launch_id=launch_id,
                batch_identity=dict(snapshot.batch_identities)[arm],
                trainer_step=trainer_step,
                pre_model_sha256=snapshot.expected_pre_model_sha256,
                post_model_sha256=state.post_model_sha256,
            )
            checkpoint_sha256 = self._put_evidence_unlocked(checkpoint_manifest_bytes)
            for member in manifest.members:
                member_path = checkpoint_root / member.path
                observed = self._put_evidence_unlocked(member_path.read_bytes())
                if observed != member.sha256:
                    raise RuntimeError("checkpoint member digest changed during installation")
            metrics_sha256 = self._put_evidence_unlocked(metrics_bytes)
            reload_evidence_sha256 = self._put_evidence_unlocked(reload_evidence_bytes)
            for output in reload_output_bytes:
                self._put_evidence_unlocked(output)
            for result in reload_process_result_bytes:
                self._put_evidence_unlocked(result)
            self._append_unlocked(
                "checkpoint_committed",
                {
                    "arm": arm,
                    "launch_id": launch_id,
                    "checkpoint_sha256": checkpoint_sha256,
                    "metrics_sha256": metrics_sha256,
                    "reload_evidence_sha256": reload_evidence_sha256,
                    "trainer_step": trainer_step,
                },
            )

    def record_preupdate_failure(
        self,
        *,
        arm: ArmName,
        launch_id: str,
        reason: str,
        evidence_bytes: bytes,
    ) -> None:
        if not reason:
            raise ValueError("pre-update failure reason must be nonempty")
        if type(evidence_bytes) is not bytes or not evidence_bytes:
            raise ValueError("pre-update failure evidence must be nonempty bytes")
        with exclusive_file_lock(self.lock_path):
            snapshot = self.inspect()
            state = snapshot.state(arm)
            if (
                state.active_launch_id != launch_id
                or state.optimizer_started
                or state.preupdate_failures != 0
                or sum(item.preupdate_failures for item in snapshot.states) != 0
            ):
                raise RuntimeError("pre-update repair is out of order or globally exhausted")
            evidence_sha256 = self._put_evidence_unlocked(evidence_bytes)
            self._append_unlocked(
                "preupdate_failure",
                {
                    "arm": arm,
                    "launch_id": launch_id,
                    "reason": reason,
                    "evidence_sha256": evidence_sha256,
                },
            )

    def _transition(
        self,
        kind: str,
        *,
        arm: ArmName,
        launch_id: str,
        extra: Mapping[str, Any],
        require: Literal[
            "launched",
            "initialization_verified",
            "batch_verified",
            "optimizer_started",
            "optimizer_completed",
            "preupdate",
        ],
    ) -> None:
        with exclusive_file_lock(self.lock_path):
            snapshot = self.inspect()
            state = snapshot.state(arm)
            if state.active_launch_id != launch_id:
                raise RuntimeError("trainer transition differs from the active launch")
            if "trainer_step" in extra and extra["trainer_step"] != snapshot.trainer_step:
                raise RuntimeError("trainer transition changed the frozen trainer step")
            valid = {
                "launched": (
                    state.process_started
                    and not state.initialization_verified
                    and not state.optimizer_started
                ),
                "initialization_verified": (
                    state.initialization_verified
                    and not state.batch_verified
                    and not state.optimizer_started
                ),
                "batch_verified": state.batch_verified and not state.optimizer_started,
                "optimizer_started": (
                    state.optimizer_started
                    and not state.optimizer_completed
                    and not state.checkpoint_committed
                ),
                "optimizer_completed": (
                    state.optimizer_completed and not state.checkpoint_committed
                ),
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
        pending = self.records / f".{path.name}.pending"
        descriptor = os.open(pending, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(record)
            handle.flush()
            os.fsync(handle.fileno())
        self._fault("after-record-temp-fsync", pending)
        os.replace(pending, path)
        fsync_directory(self.records)
        self._fault("after-record-rename", path)

    def _put_evidence_unlocked(self, value: bytes) -> str:
        if type(value) is not bytes:
            raise ValueError("trainer evidence must be immutable bytes")
        digest = _sha256(value)
        path = self.evidence / digest
        if path.exists():
            if path.read_bytes() != value:
                raise ValueError("trainer evidence digest collision")
            return digest
        pending = self.evidence / f".{digest}.pending"
        descriptor = os.open(pending, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        self._fault("after-evidence-temp-fsync", pending)
        os.replace(pending, path)
        fsync_directory(self.evidence)
        self._fault("after-evidence-rename", path)
        return digest

    def _fault(self, stage: str, path: Path) -> None:
        if self._fault_hook is not None:
            self._fault_hook(stage, path)


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
            set(payload) != _RECORD_FIELDS
            or payload.get("schema_version") != 1
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
        "protocol_manifest_sha256",
        "shared_initialization_manifest_sha256",
        "expected_pre_model_sha256",
        "expected_base_model_manifest_sha256",
        "reload_probe_sha256",
        "trainer_step",
        "arm_order",
        "batch_identities",
        "trainer_config_sha256s",
        "process_command_sha256s",
        "process_environment_sha256s",
    }:
        raise ValueError("trainer run genesis fields differ")
    order = tuple(event["arm_order"])
    if order != _ARMS:
        raise ValueError("trainer run arm order is invalid")
    batch_ids = _arm_digests(event["batch_identities"], "batch identities")
    configs = _arm_digests(event["trainer_config_sha256s"], "trainer configs")
    process_commands = _arm_digests(
        event["process_command_sha256s"], "process commands"
    )
    process_environments = _arm_digests(
        event["process_environment_sha256s"], "process environments"
    )
    states = {arm: ArmRunState(arm) for arm in order}
    expected_pre_model_sha256 = _require_sha256(
        event["expected_pre_model_sha256"], "expected pre-model"
    )
    expected_base_model_manifest_sha256 = _require_sha256(
        event["expected_base_model_manifest_sha256"],
        "expected base-model manifest",
    )
    reload_probe_sha256 = _require_sha256(
        event["reload_probe_sha256"],
        "reload probe",
    )
    trainer_step = event["trainer_step"]
    if type(trainer_step) is not int or trainer_step < 1:
        raise ValueError("trainer run genesis has invalid trainer step")
    for record in records[1:]:
        _apply_event(
            states,
            record["record_kind"],
            record["event"],
            batch_ids,
            configs,
            process_commands,
            process_environments,
            expected_pre_model_sha256,
            expected_base_model_manifest_sha256,
            reload_probe_sha256,
            trainer_step,
            records_root.parent / "evidence",
        )
    return TrainerRunSnapshot(
        _require_sha256(event["campaign_manifest_sha256"], "campaign manifest"),
        _require_sha256(event["protocol_manifest_sha256"], "protocol manifest"),
        _require_sha256(
            event["shared_initialization_manifest_sha256"],
            "shared initialization manifest",
        ),
        expected_pre_model_sha256,
        expected_base_model_manifest_sha256,
        reload_probe_sha256,
        trainer_step,
        order,
        tuple(sorted(batch_ids.items())),
        tuple(sorted(configs.items())),
        tuple(sorted(process_commands.items())),
        tuple(sorted(process_environments.items())),
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
    process_commands: Mapping[ArmName, str],
    process_environments: Mapping[ArmName, str],
    expected_pre_model_sha256: str,
    expected_base_model_manifest_sha256: str,
    reload_probe_sha256: str,
    expected_trainer_step: int,
    evidence_root: Path,
) -> None:
    expected_fields = _EVENT_FIELDS.get(kind)
    if expected_fields is None or set(event) != expected_fields:
        raise ValueError("trainer run event fields differ from schema")
    arm = event.get("arm")
    if (
        arm not in states
        or not isinstance(event.get("launch_id"), str)
        or not event["launch_id"]
    ):
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
            initialization_verified=False,
            batch_verified=False,
            process_started=False,
            process_receipt_sha256=None,
        )
    elif kind == "process_started":
        digest = _require_sha256(
            event.get("process_receipt_sha256"),
            "process receipt sha256",
        )
        receipt = TrainerProcessStartReceipt.from_bytes(
            _verified_evidence(evidence_root, digest)
        )
        if (
            state.active_launch_id != launch_id
            or state.process_started
            or state.initialization_verified
            or receipt.arm != arm
            or receipt.launch_id != launch_id
            or receipt.command_sha256 != process_commands[arm]
            or receipt.environment_manifest_sha256 != process_environments[arm]
        ):
            raise ValueError("trainer process start is invalid")
        states[arm] = replace(
            state,
            process_started=True,
            process_receipt_sha256=digest,
        )
    elif kind == "initialization_verified":
        observed = _require_sha256(
            event.get("observed_pre_model_sha256"),
            "observed_pre_model_sha256",
        )
        if (
            state.active_launch_id != launch_id
            or not state.process_started
            or state.initialization_verified
        ):
            raise ValueError("trainer initialization verification is invalid")
        if observed != expected_pre_model_sha256:
            raise ValueError("trainer initialization differs from the frozen shared state")
        states[arm] = replace(state, initialization_verified=True)
    elif kind == "batch_verified":
        if (
            state.active_launch_id != launch_id
            or not state.initialization_verified
            or state.batch_verified
        ):
            raise ValueError("trainer batch verification is duplicated or unclaimed")
        if event.get("batch_identity") != batch_ids[arm]:
            raise ValueError("trainer verified a different batch identity")
        states[arm] = replace(state, batch_verified=True)
    elif kind == "optimizer_started":
        if (
            state.active_launch_id != launch_id
            or not state.batch_verified
            or state.optimizer_started
            or event.get("trainer_step") != expected_trainer_step
        ):
            raise ValueError("trainer optimizer start is out of order")
        states[arm] = replace(state, optimizer_started=True)
    elif kind == "optimizer_completed":
        post_model_sha256 = _require_sha256(
            event.get("post_model_sha256"),
            "post_model_sha256",
        )
        if (
            state.active_launch_id != launch_id
            or not state.optimizer_started
            or state.optimizer_completed
            or type(event.get("trainer_step")) is not int
            or event["trainer_step"] < 1
            or event["trainer_step"] != expected_trainer_step
            or type(event.get("model_changed")) is not bool
            or event["model_changed"]
            != (post_model_sha256 != expected_pre_model_sha256)
        ):
            raise ValueError("trainer optimizer completion is out of order")
        states[arm] = replace(
            state,
            optimizer_completed=True,
            post_model_sha256=post_model_sha256,
            model_changed=event["model_changed"],
        )
    elif kind == "checkpoint_committed":
        if (
            state.active_launch_id != launch_id
            or not state.optimizer_started
            or not state.optimizer_completed
            or state.checkpoint_committed
        ):
            raise ValueError("trainer checkpoint commit is out of order")
        for name in ("checkpoint_sha256", "metrics_sha256", "reload_evidence_sha256"):
            _require_sha256(event.get(name), name)
        if type(event.get("trainer_step")) is not int or event["trainer_step"] < 1:
            raise ValueError("trainer checkpoint has invalid step")
        if event["trainer_step"] != expected_trainer_step:
            raise ValueError("trainer checkpoint changed the frozen step")
        checkpoint_bytes = _verified_evidence(
            evidence_root,
            event["checkpoint_sha256"],
        )
        metrics_bytes = _verified_evidence(evidence_root, event["metrics_sha256"])
        reload_bytes = _verified_evidence(
            evidence_root,
            event["reload_evidence_sha256"],
        )
        checkpoint = StageDCheckpointManifest.from_bytes(checkpoint_bytes)
        for member in checkpoint.members:
            member_bytes = _verified_evidence(evidence_root, member.sha256)
            if len(member_bytes) != member.size:
                raise ValueError("retained checkpoint member size differs from its manifest")
        reload_evidence = StageDReloadEvidence.from_bytes(reload_bytes)
        reload_outputs = tuple(
            _verified_evidence(evidence_root, digest)
            for digest in reload_evidence.output_sha256s
        )
        reload_process_results = tuple(
            _verified_evidence(evidence_root, digest)
            for digest in reload_evidence.process_identities
        )
        reload_evidence.verify_output_bytes(reload_outputs)  # type: ignore[arg-type]
        reload_evidence.verify_process_result_bytes(  # type: ignore[arg-type]
            reload_process_results
        )
        assert state.post_model_sha256 is not None
        validate_metrics_bytes(
            metrics_bytes,
            arm=arm,
            launch_id=launch_id,
            batch_identity=batch_ids[arm],
            trainer_step=expected_trainer_step,
            pre_model_sha256=expected_pre_model_sha256,
            post_model_sha256=state.post_model_sha256,
        )
        if (
            checkpoint.arm != arm
            or checkpoint.trainer_step != expected_trainer_step
            or checkpoint.base_model_manifest_sha256
            != expected_base_model_manifest_sha256
            or checkpoint.post_model_sha256 != state.post_model_sha256
            or reload_evidence.arm != arm
            or reload_evidence.checkpoint_manifest_sha256
            != event["checkpoint_sha256"]
            or reload_evidence.post_model_sha256 != state.post_model_sha256
            or reload_evidence.reload_probe_sha256 != reload_probe_sha256
        ):
            raise ValueError("trainer checkpoint evidence differs from the completed state")
        states[arm] = replace(
            state,
            active_launch_id=None,
            checkpoint_committed=True,
            checkpoint_sha256=event["checkpoint_sha256"],
            metrics_sha256=event["metrics_sha256"],
            reload_evidence_sha256=event["reload_evidence_sha256"],
        )
    elif kind == "preupdate_failure":
        if (
            state.active_launch_id != launch_id
            or state.optimizer_started
            or state.preupdate_failures != 0
            or sum(item.preupdate_failures for item in states.values()) != 0
            or not isinstance(event.get("reason"), str)
            or not event["reason"]
        ):
            raise ValueError("trainer pre-update repair is not eligible")
        _verified_evidence(evidence_root, event["evidence_sha256"])
        states[arm] = replace(
            state,
            active_launch_id=None,
            initialization_verified=False,
            batch_verified=False,
            process_started=False,
            process_receipt_sha256=None,
            preupdate_failures=1,
        )
    else:
        raise ValueError(f"unsupported trainer run event: {kind}")


def _arm_digests(value: object, name: str) -> dict[ArmName, str]:
    if not isinstance(value, dict) or set(value) != set(_ARMS):
        raise ValueError(f"trainer run {name} must cover all arms")
    return {arm: _require_sha256(value[arm], f"{name}.{arm}") for arm in _ARMS}


def _verified_evidence(root: Path, digest: str) -> bytes:
    path = root / _require_sha256(digest, "trainer evidence sha256")
    if not path.is_file():
        raise ValueError("trainer ledger references missing evidence")
    value = path.read_bytes()
    if _sha256(value) != digest:
        raise ValueError("trainer ledger evidence bytes differ from their digest")
    return value
