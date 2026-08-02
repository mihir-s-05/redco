"""Durable process-launch transactions for held-out Stage-D evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from redco.analysis.stage_d_evaluation_actuation import ActuatedProcessReceipt
from redco.analysis.stage_d_evaluation_capabilities import (
    EvaluationClientLaunch,
    EvaluationClientSession,
    EvaluationServerLaunch,
)
from redco.analysis.stage_d_evaluation_codec import EvaluationEvidenceStore, exclusive_lock, sha256
from redco.analysis.stage_d_evaluation_contracts import (
    EvaluationProgramBinding,
    StageDEvaluationExecutionManifest,
)
from redco.analysis.stage_d_evaluation_server import EvaluationServerProcessObservation
from redco.analysis.stage_d_evaluation_state import EvaluationLedgerSnapshot
from redco.analysis.stage_d_objective_binding import ArmName
from redco.contracts import canonical_json


class EvaluationProcessTransactionLedger(Protocol):
    """Minimal storage surface required by process lifecycle transactions."""

    lock_path: Path
    evidence: EvaluationEvidenceStore

    @property
    def manifest(self) -> StageDEvaluationExecutionManifest: ...

    def inspect(self) -> EvaluationLedgerSnapshot: ...

    def _append_unlocked(self, kind: str, event: dict[str, Any]) -> str: ...

    def _verify_process_receipt(
        self,
        receipt: ActuatedProcessReceipt,
        program: EvaluationProgramBinding,
        *,
        require_current: bool,
    ) -> None: ...


def reserve_server_launch(
    ledger: EvaluationProcessTransactionLedger,
    arm: ArmName,
) -> EvaluationServerLaunch:
    with exclusive_lock(ledger.lock_path):
        snapshot = ledger.inspect()
        if snapshot.sealed or snapshot.terminal_status == "ambiguous-dispatch":
            raise RuntimeError("evaluation server launch is terminally forbidden")
        current = snapshot.current_task
        expected_arm = (
            current.unit.arm
            if current is not None
            else (
                ledger.manifest.schedule[len(snapshot.tasks)].arm
                if len(snapshot.tasks) < len(ledger.manifest.schedule)
                else None
            )
        )
        if arm != expected_arm:
            raise RuntimeError("evaluation server launch is outside the frozen arm block")
        previous = snapshot.latest_server_epoch(arm)
        if previous is not None and previous.process_receipt_sha256 is None:
            return EvaluationServerLaunch(arm, previous.epoch, previous.launch_record_sha256)
        prior_digest = None
        dead_digest = None
        if previous is not None:
            assert previous.process_receipt_sha256 is not None
            prior_digest = previous.process_receipt_sha256
            receipt = ActuatedProcessReceipt.from_bytes(ledger.evidence.get(prior_digest))
            if receipt.is_same_live_process():
                raise RuntimeError("evaluation server process is still live")
            if any(
                call.dispatch_receipt_sha256 is not None
                for task in snapshot.tasks
                if task.unit.arm == arm
                for call in task.calls
            ):
                raise RuntimeError("evaluation server replacement is past its cutoff")
            dead_digest = ledger.evidence.put(
                canonical_json(
                    {
                        "schema_version": 1,
                        "domain": "redco-stage-d-evaluation-dead-server-v1",
                        "arm": arm,
                        "prior_process_receipt_sha256": prior_digest,
                        "prior_server_attestation_sha256": previous.server_attestation_sha256,
                    }
                )
            )
        epoch = 0 if previous is None else previous.epoch + 1
        if epoch >= ledger.manifest.max_server_launches_per_arm:
            raise RuntimeError("evaluation server replacement budget is exhausted")
        record_sha256 = ledger._append_unlocked(
            "server_launch_reserved",
            {
                "arm": arm,
                "epoch": epoch,
                "prior_process_receipt_sha256": prior_digest,
                "dead_process_evidence_sha256": dead_digest,
            },
        )
        return EvaluationServerLaunch(arm, epoch, record_sha256)


def claim_server(
    ledger: EvaluationProcessTransactionLedger,
    launch: EvaluationServerLaunch,
    process_receipt_bytes: bytes,
) -> str:
    receipt = ActuatedProcessReceipt.from_bytes(process_receipt_bytes)
    receipt.verify_program(
        arm=launch.arm,
        role="server",
        epoch=launch.epoch,
        launch_capability_sha256=launch.launch_record_sha256,
        argv=ledger.manifest.program(launch.arm, "server").argv,
        environment=ledger.manifest.program(launch.arm, "server").environment,
        cgroup_root=ledger.manifest.supervisor_limits.cgroup_root,
        control_root=ledger.manifest.supervisor_limits.control_root,
        evaluation_ledger_id=ledger.manifest.evaluation_ledger_id,
        require_current=True,
    )
    program = ledger.manifest.program(launch.arm, "server")
    ledger._verify_process_receipt(receipt, program, require_current=True)
    with exclusive_lock(ledger.lock_path):
        snapshot = ledger.inspect()
        epoch = snapshot.latest_server_epoch(launch.arm)
        if epoch is None or launch != EvaluationServerLaunch(
            epoch.arm,
            epoch.epoch,
            epoch.launch_record_sha256,
        ):
            raise RuntimeError("evaluation server claim lacks its launch capability")
        if epoch.process_receipt_sha256 is not None:
            if epoch.process_receipt_sha256 == receipt.receipt_sha256:
                return epoch.process_receipt_sha256
            raise RuntimeError("evaluation server launch was already claimed")
        _require_reserved_attempt(
            snapshot,
            receipt,
            arm=launch.arm,
            role="server",
            epoch=launch.epoch,
            launch_record_sha256=launch.launch_record_sha256,
        )
        digest = ledger.evidence.put(process_receipt_bytes)
        ledger._append_unlocked(
            "server_claimed",
            {
                "arm": launch.arm,
                "epoch": launch.epoch,
                "launch_record_sha256": launch.launch_record_sha256,
                "process_receipt_sha256": digest,
            },
        )
        return digest


def attest_server(
    ledger: EvaluationProcessTransactionLedger,
    *,
    launch: EvaluationServerLaunch,
    process_receipt_bytes: bytes,
    process_observation_bytes: bytes,
    probe_response_bytes: bytes,
) -> str:
    program = ledger.manifest.program(launch.arm, "server")
    receipt = ActuatedProcessReceipt.from_bytes(process_receipt_bytes)
    receipt.verify_program(
        arm=launch.arm,
        role="server",
        epoch=launch.epoch,
        launch_capability_sha256=launch.launch_record_sha256,
        argv=program.argv,
        environment=program.environment,
        cgroup_root=ledger.manifest.supervisor_limits.cgroup_root,
        control_root=ledger.manifest.supervisor_limits.control_root,
        evaluation_ledger_id=ledger.manifest.evaluation_ledger_id,
        require_current=False,
    )
    ledger._verify_process_receipt(receipt, program, require_current=False)
    observation = EvaluationServerProcessObservation.from_bytes(process_observation_bytes)
    observation.verify(launch=launch, receipt=receipt, program=program)
    process_receipt_sha256 = sha256(process_receipt_bytes)
    probe_sha256 = ledger.evidence.put(probe_response_bytes)
    observation_sha256 = ledger.evidence.put(process_observation_bytes)
    attestation = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-evaluation-server-attestation-v1",
            "arm": launch.arm,
            "server_epoch": launch.epoch,
            "launch_record_sha256": launch.launch_record_sha256,
            "process_receipt_sha256": process_receipt_sha256,
            "process_observation_sha256": observation_sha256,
            "program_binding_sha256": program.binding_sha256,
            "checkpoint_manifest_sha256": program.checkpoint_manifest_sha256,
            "post_model_sha256": program.post_model_sha256,
            "checkpoint_state_sha256": program.post_model_sha256,
            "reload_evidence_sha256": program.reload_evidence_sha256,
            "endpoint": program.endpoint,
            "cache_namespace": program.cache_namespace,
            "probe_response_sha256": probe_sha256,
        }
    )
    with exclusive_lock(ledger.lock_path):
        snapshot = ledger.inspect()
        epoch = snapshot.latest_server_epoch(launch.arm)
        if (
            epoch is None
            or launch != EvaluationServerLaunch(epoch.arm, epoch.epoch, epoch.launch_record_sha256)
            or epoch.process_receipt_sha256 != process_receipt_sha256
        ):
            raise RuntimeError("evaluation server claim differs")
        digest = ledger.evidence.put(attestation)
        ledger._append_unlocked(
            "server_attested",
            {
                "arm": launch.arm,
                "epoch": launch.epoch,
                "launch_record_sha256": launch.launch_record_sha256,
                "process_receipt_sha256": process_receipt_sha256,
                "server_attestation_sha256": digest,
            },
        )
        return digest


def reserve_client_launch(
    ledger: EvaluationProcessTransactionLedger,
    arm: ArmName,
) -> EvaluationClientLaunch:
    with exclusive_lock(ledger.lock_path):
        snapshot = ledger.inspect()
        previous = snapshot.latest_epoch(arm)
        if previous is not None and previous.process_receipt_sha256 is None:
            return EvaluationClientLaunch(
                arm,
                previous.epoch,
                previous.launch_record_sha256,
                previous.resume_task_attempt_id,
            )
        current = snapshot.current_task
        expected_arm = (
            current.unit.arm
            if current is not None
            else (
                ledger.manifest.schedule[len(snapshot.tasks)].arm
                if len(snapshot.tasks) < len(ledger.manifest.schedule)
                else None
            )
        )
        if arm != expected_arm or dict(snapshot.server_attestations).get(arm) is None:
            raise RuntimeError("evaluation client launch is outside its authorized arm")
        prior_digest = None
        dead_digest = None
        if previous is not None:
            assert previous.process_receipt_sha256 is not None
            previous_bytes = ledger.evidence.get(previous.process_receipt_sha256)
            previous_receipt = ActuatedProcessReceipt.from_bytes(previous_bytes)
            if previous_receipt.is_same_live_process():
                raise RuntimeError("prior evaluation client process is still live")
            prior_digest = previous.process_receipt_sha256
            if current is not None and any(
                call.dispatch_receipt_sha256 is not None and call.response_envelope_sha256 is None
                for call in current.calls
            ):
                raise RuntimeError("evaluation client replacement crosses an ambiguous call")
            dead_digest = ledger.evidence.put(
                canonical_json(
                    {
                        "schema_version": 1,
                        "domain": "redco-stage-d-evaluation-dead-client-v1",
                        "arm": arm,
                        "prior_process_receipt_sha256": prior_digest,
                        "resume_task_attempt_id": (
                            None if current is None else current.task_attempt_id
                        ),
                    }
                )
            )
        epoch = 0 if previous is None else previous.epoch + 1
        if epoch >= ledger.manifest.max_client_launches_per_arm:
            raise RuntimeError("evaluation client replacement budget is exhausted")
        resume_id = None if current is None else current.task_attempt_id
        record_sha256 = ledger._append_unlocked(
            "client_launch_reserved",
            {
                "arm": arm,
                "epoch": epoch,
                "resume_task_attempt_id": resume_id,
                "prior_process_receipt_sha256": prior_digest,
                "dead_process_evidence_sha256": dead_digest,
            },
        )
        return EvaluationClientLaunch(arm, epoch, record_sha256, resume_id)


def claim_client(
    ledger: EvaluationProcessTransactionLedger,
    launch: EvaluationClientLaunch,
    process_receipt_bytes: bytes,
) -> EvaluationClientSession:
    receipt = ActuatedProcessReceipt.from_bytes(process_receipt_bytes)
    receipt.verify_program(
        arm=launch.arm,
        role="client",
        epoch=launch.epoch,
        launch_capability_sha256=launch.launch_record_sha256,
        argv=ledger.manifest.program(launch.arm, "client").argv,
        environment=ledger.manifest.program(launch.arm, "client").environment,
        cgroup_root=ledger.manifest.supervisor_limits.cgroup_root,
        control_root=ledger.manifest.supervisor_limits.control_root,
        evaluation_ledger_id=ledger.manifest.evaluation_ledger_id,
        require_current=True,
    )
    program = ledger.manifest.program(launch.arm, "client")
    ledger._verify_process_receipt(receipt, program, require_current=True)
    with exclusive_lock(ledger.lock_path):
        snapshot = ledger.inspect()
        epoch = snapshot.latest_epoch(launch.arm)
        if epoch is None or launch != EvaluationClientLaunch(
            epoch.arm,
            epoch.epoch,
            epoch.launch_record_sha256,
            epoch.resume_task_attempt_id,
        ):
            raise RuntimeError("evaluation client claim lacks its launch capability")
        receipt_digest = ledger.evidence.put(process_receipt_bytes)
        if epoch.process_receipt_sha256 is not None:
            if epoch.process_receipt_sha256 == receipt_digest:
                return EvaluationClientSession(launch.arm, launch.epoch, process_receipt_bytes)
            raise RuntimeError("evaluation client launch was already claimed")
        _require_reserved_attempt(
            snapshot,
            receipt,
            arm=launch.arm,
            role="client",
            epoch=launch.epoch,
            launch_record_sha256=launch.launch_record_sha256,
        )
        ledger._append_unlocked(
            "client_claimed",
            {
                "arm": launch.arm,
                "epoch": launch.epoch,
                "launch_record_sha256": launch.launch_record_sha256,
                "resume_task_attempt_id": launch.resume_task_attempt_id,
                "process_receipt_sha256": receipt_digest,
            },
        )
        return EvaluationClientSession(launch.arm, launch.epoch, process_receipt_bytes)


def _require_reserved_attempt(
    snapshot: EvaluationLedgerSnapshot,
    receipt: ActuatedProcessReceipt,
    *,
    arm: ArmName,
    role: str,
    epoch: int,
    launch_record_sha256: str,
) -> None:
    matches = [
        item
        for item in snapshot.actuation_attempts
        if item.actuation_attempt_id == receipt.actuation_attempt_id
    ]
    if (
        len(matches) != 1
        or matches[0].disposition is not None
        or (
            matches[0].arm,
            matches[0].role,
            matches[0].epoch,
            matches[0].launch_record_sha256,
        )
        != (arm, role, epoch, launch_record_sha256)
    ):
        raise RuntimeError("evaluation process lacks its reserved actuation attempt")


__all__ = [
    "EvaluationProcessTransactionLedger",
    "attest_server",
    "claim_client",
    "claim_server",
    "reserve_client_launch",
    "reserve_server_launch",
]
