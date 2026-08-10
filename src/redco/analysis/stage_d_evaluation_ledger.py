"""Thin transactional API over the Stage-D held-out evaluation reducer."""

from __future__ import annotations

import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

from redco.analysis import stage_d_evaluation_attempts as evaluation_attempts
from redco.analysis import stage_d_evaluation_completion as evaluation_completion
from redco.analysis import stage_d_evaluation_process_transactions as process_transactions
from redco.analysis.stage_d_evaluation_actuation import (
    ActuatedProcessReceipt,
    EvaluationSupervisorIdentity,
)
from redco.analysis.stage_d_evaluation_attempts import ActuationDisposition
from redco.analysis.stage_d_evaluation_barrier import (
    StageDEvaluationAuthorization,
    StageDEvaluationPlan,
)
from redco.analysis.stage_d_evaluation_capabilities import (
    EvaluationCallAuthorization,
    EvaluationClientLaunch,
    EvaluationClientSession,
    EvaluationDispatchAuthorization,
    EvaluationServerLaunch,
    EvaluationTaskAttempt,
)
from redco.analysis.stage_d_evaluation_codec import (
    EvaluationEvidenceStore,
    FaultHook,
    atomic_publish,
    canonical_object,
    decode_record,
    encode_record,
    exclusive_lock,
    sha256,
)
from redco.analysis.stage_d_evaluation_contracts import (
    EvaluationProgramBinding,
    StageDEvaluationExecutionManifest,
    evaluation_environment_sha256,
)
from redco.analysis.stage_d_evaluation_evidence import (
    EVIDENCE_FIELDS,
    reachable_evidence,
    verify_evidence_closure,
)
from redco.analysis.stage_d_evaluation_reducer import reduce_evaluation_records
from redco.analysis.stage_d_evaluation_state import (
    EvaluationActuationAttempt,
    EvaluationCallState,
    EvaluationLedgerSnapshot,
)
from redco.analysis.stage_d_evaluation_transport import (
    verify_nonsecret_headers,
    verify_transport_request,
)
from redco.analysis.stage_d_objective_binding import ArmName
from redco.analysis.stage_d_process_supervision import command_sha256
from redco.contracts import canonical_json

_ARMS: tuple[ArmName, ...] = ("stock", "branch-global", "local")


class StageDEvaluationLedger:
    """Exactly-once task/call ledger; process replacement is only a clean-boundary tool."""

    def __init__(self, root: Path, *, fault_hook: FaultHook | None = None) -> None:
        self.root = root
        self.records = root / "records"
        self.inputs = root / "inputs"
        self.responses = root / "responses"
        self.evidence = EvaluationEvidenceStore(root / "evidence")
        self.lock_path = root / "writer.lock"
        self._fault_hook = fault_hook

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        authorization_bytes: bytes,
        execution_manifest_bytes: bytes,
        evaluation_plan_bytes: bytes,
        runtime_bundle_bytes: bytes,
        fault_hook: FaultHook | None = None,
    ) -> StageDEvaluationLedger:
        manifest = StageDEvaluationExecutionManifest.from_bytes(execution_manifest_bytes)
        if root.name != manifest.evaluation_ledger_id:
            raise ValueError("evaluation ledger root differs from its frozen identity")
        plan = StageDEvaluationPlan.from_bytes(evaluation_plan_bytes)
        authorization = StageDEvaluationAuthorization.from_bytes(authorization_bytes)
        manifest_checkpoints = tuple(
            (
                program.arm,
                program.checkpoint_manifest_sha256,
                program.post_model_sha256,
                program.reload_evidence_sha256,
            )
            for program in manifest.programs
            if program.role == "server"
        )
        authorization_checkpoints = tuple(
            (
                item.arm,
                item.checkpoint_manifest_sha256,
                item.post_model_sha256,
                item.reload_evidence_sha256,
            )
            for item in authorization.checkpoints
        )
        if (
            authorization.execution_manifest_sha256 != manifest.manifest_sha256
            or authorization.evaluation_plan_sha256 != manifest.evaluation_plan_sha256
            or authorization.protocol_manifest_sha256 != manifest.protocol_manifest_sha256
            or authorization.trainer_ledger_head_sha256 != manifest.trainer_ledger_head_sha256
            or authorization.trainer_record_count != manifest.trainer_record_count
            or authorization.heldout_eval_config_sha256 != manifest.heldout_eval_config_sha256
            or authorization_checkpoints != manifest_checkpoints
            or sha256(evaluation_plan_bytes) != manifest.evaluation_plan_sha256
            or sha256(runtime_bundle_bytes) != manifest.runtime_bundle_sha256
        ):
            raise ValueError("evaluation ledger inputs are not mutually bound")
        expected_tasks = tuple((item.task_id, item.seed) for item in plan.tasks)
        for arm in _ARMS:
            observed = tuple(
                (item.task_id, item.seed) for item in manifest.schedule if item.arm == arm
            )
            if observed != expected_tasks:
                raise ValueError("evaluation manifest schedule differs from the plan")
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise FileExistsError("evaluation ledger root is not a regular directory")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        ledger = cls(root, fault_hook=fault_hook)
        allowed = {"records", "inputs", "responses", "evidence", "writer.lock"}
        if any(path.name not in allowed for path in root.iterdir()):
            raise FileExistsError("evaluation ledger root contains unknown state")
        for directory in (
            ledger.records,
            ledger.inputs,
            ledger.responses,
            ledger.evidence.root,
        ):
            if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
                raise FileExistsError("evaluation ledger storage path is invalid")
            directory.mkdir(exist_ok=True)
        atomic_publish(ledger.inputs / "authorization.json", authorization_bytes)
        atomic_publish(ledger.inputs / "execution-manifest.json", execution_manifest_bytes)
        atomic_publish(ledger.inputs / "evaluation-plan.json", evaluation_plan_bytes)
        atomic_publish(ledger.inputs / "runtime-bundle.zip", runtime_bundle_bytes)
        with exclusive_lock(ledger.lock_path):
            if not any(ledger.records.glob("*.json")):
                ledger._append_unlocked(
                    "genesis",
                    {
                        "authorization_sha256": sha256(authorization_bytes),
                        "execution_manifest_sha256": manifest.manifest_sha256,
                        "evaluation_plan_sha256": manifest.evaluation_plan_sha256,
                        "created_at_unix_ns": time.time_ns(),
                    },
                )
        ledger.inspect()
        return ledger

    @property
    def manifest(self) -> StageDEvaluationExecutionManifest:
        return StageDEvaluationExecutionManifest.from_bytes(
            (self.inputs / "execution-manifest.json").read_bytes()
        )

    @property
    def plan(self) -> StageDEvaluationPlan:
        return StageDEvaluationPlan.from_bytes((self.inputs / "evaluation-plan.json").read_bytes())

    @property
    def authorization_bytes(self) -> bytes:
        return (self.inputs / "authorization.json").read_bytes()

    def inspect(self) -> EvaluationLedgerSnapshot:
        input_names = {path.name for path in self.inputs.iterdir() if path.is_file()}
        if input_names != {
            "authorization.json",
            "evaluation-plan.json",
            "execution-manifest.json",
            "runtime-bundle.zip",
        }:
            raise ValueError("evaluation ledger input roster differs")
        manifest_bytes = (self.inputs / "execution-manifest.json").read_bytes()
        manifest = StageDEvaluationExecutionManifest.from_bytes(manifest_bytes)
        if sha256(manifest_bytes) != manifest.manifest_sha256:
            raise ValueError("evaluation execution manifest digest differs")
        plan_bytes = (self.inputs / "evaluation-plan.json").read_bytes()
        if sha256(plan_bytes) != manifest.evaluation_plan_sha256:
            raise ValueError("evaluation plan differs from execution manifest")
        if sha256((self.inputs / "runtime-bundle.zip").read_bytes()) != (
            manifest.runtime_bundle_sha256
        ):
            raise ValueError("evaluation runtime bundle differs from execution manifest")
        auth_bytes = self.authorization_bytes
        auth = StageDEvaluationAuthorization.from_bytes(auth_bytes)
        if auth.execution_manifest_sha256 != manifest.manifest_sha256:
            raise ValueError("evaluation authorization differs from execution manifest")
        paths = sorted(self.records.glob("*.json"))
        if [path.name for path in paths] != [f"{index:08d}.json" for index in range(len(paths))]:
            raise ValueError("evaluation ledger record roster is noncontiguous")
        values = []
        records = []
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError("evaluation ledger record is absent or symbolic")
            value = path.read_bytes()
            values.append(value)
            records.append(decode_record(value))
        snapshot = reduce_evaluation_records(
            records,
            [sha256(value) for value in values],
            manifest=manifest,
            expected_authorization_sha256=auth.authorization_sha256,
        )
        evidence_roots: list[str] = []
        for record in records:
            for name, digest in record["event"].items():
                if name in EVIDENCE_FIELDS and digest is not None:
                    evidence_roots.append(digest)
        self._verify_evidence_closure(evidence_roots)
        for task in snapshot.tasks:
            for call in task.calls:
                if call.response_envelope_sha256 is not None:
                    pointer = self.responses / f"{call.call_id}.json"
                    if (
                        pointer.is_symlink()
                        or not pointer.is_file()
                        or sha256(pointer.read_bytes()) != call.response_envelope_sha256
                    ):
                        raise ValueError("evaluation response pointer differs")
        current = snapshot.current_task
        if snapshot.terminal_status == "active" and current is not None:
            epoch = next(
                (
                    item
                    for item in snapshot.client_epochs
                    if item.arm == current.unit.arm and item.epoch == current.client_epoch
                ),
                None,
            )
            if epoch is None or epoch.process_receipt_sha256 is None:
                raise ValueError("open evaluation task lacks a client epoch")
            receipt = ActuatedProcessReceipt.from_bytes(
                self.evidence.get(epoch.process_receipt_sha256)
            )
            if not receipt.is_same_live_process():
                snapshot = replace(snapshot, terminal_status="orphaned-open-task")
        if snapshot.terminal_status == "active":
            current_task = snapshot.current_task
            server_arm = (
                current_task.unit.arm
                if current_task is not None
                else (
                    manifest.schedule[len(snapshot.tasks)].arm
                    if len(snapshot.tasks) < len(manifest.schedule)
                    else None
                )
            )
            claim_digest = (
                None if server_arm is None else dict(snapshot.server_claims).get(server_arm)
            )
            if claim_digest is not None:
                server_receipt = ActuatedProcessReceipt.from_bytes(self.evidence.get(claim_digest))
                if not server_receipt.is_same_live_process():
                    snapshot = replace(snapshot, terminal_status="orphaned-server")
        return snapshot

    def reachable_evidence_sha256s(self) -> tuple[str, ...]:
        """Return the verified transitive evidence closure of the durable ledger."""
        self.inspect()
        return reachable_evidence(self.records, self.evidence)

    def _verify_evidence_closure(self, roots: list[str]) -> set[str]:
        return verify_evidence_closure(self.evidence, roots)

    def reserve_actuation_attempt(
        self,
        *,
        actuation_attempt_id: str,
        arm: ArmName,
        role: Literal["client", "server"],
        epoch: int,
        launch_record_sha256: str,
        supervisor: EvaluationSupervisorIdentity,
    ) -> EvaluationActuationAttempt:
        return evaluation_attempts.reserve_actuation_attempt(
            self,
            actuation_attempt_id=actuation_attempt_id,
            arm=arm,
            role=role,
            epoch=epoch,
            launch_record_sha256=launch_record_sha256,
            supervisor=supervisor,
        )

    def finish_actuation_attempt(
        self,
        *,
        actuation_attempt_id: str,
        disposition: ActuationDisposition,
        process_receipt_bytes: bytes | None = None,
        error_evidence_bytes: bytes | None = None,
        cleanup_evidence_bytes: bytes | None = None,
    ) -> EvaluationActuationAttempt:
        return evaluation_attempts.finish_actuation_attempt(
            self,
            actuation_attempt_id=actuation_attempt_id,
            disposition=disposition,
            process_receipt_bytes=process_receipt_bytes,
            error_evidence_bytes=error_evidence_bytes,
            cleanup_evidence_bytes=cleanup_evidence_bytes,
        )

    def reserve_server_launch(self, arm: ArmName) -> EvaluationServerLaunch:
        return process_transactions.reserve_server_launch(self, arm)

    def claim_server(
        self,
        launch: EvaluationServerLaunch,
        process_receipt_bytes: bytes,
    ) -> str:
        return process_transactions.claim_server(self, launch, process_receipt_bytes)

    def attest_server(
        self,
        *,
        launch: EvaluationServerLaunch,
        process_receipt_bytes: bytes,
        process_observation_bytes: bytes,
        probe_response_bytes: bytes,
    ) -> str:
        return process_transactions.attest_server(
            self,
            launch=launch,
            process_receipt_bytes=process_receipt_bytes,
            process_observation_bytes=process_observation_bytes,
            probe_response_bytes=probe_response_bytes,
        )

    def reserve_client_launch(self, arm: ArmName) -> EvaluationClientLaunch:
        return process_transactions.reserve_client_launch(self, arm)

    def claim_client(
        self,
        launch: EvaluationClientLaunch,
        process_receipt_bytes: bytes,
    ) -> EvaluationClientSession:
        return process_transactions.claim_client(self, launch, process_receipt_bytes)

    def resume_open_task(self, *, session: EvaluationClientSession) -> EvaluationTaskAttempt:
        snapshot = self.inspect()
        self._verify_client_session(snapshot, session)
        self._verify_server_session(snapshot, session.arm)
        current = snapshot.current_task
        if (
            snapshot.terminal_status != "active"
            or current is None
            or current.unit.arm != session.arm
            or current.client_epoch != session.epoch
        ):
            raise RuntimeError("evaluation client does not own an open task")
        return EvaluationTaskAttempt(
            current.task_attempt_id,
            current.unit,
            current.client_epoch,
        )

    def resume_reserved_call(
        self,
        task: EvaluationTaskAttempt,
        *,
        session: EvaluationClientSession,
    ) -> EvaluationCallAuthorization:
        resumed = self.resume_open_task(session=session)
        snapshot = self.inspect()
        current = snapshot.current_task
        if resumed != task or current is None or not current.calls:
            raise RuntimeError("evaluation task lacks a reserved call")
        call = current.calls[-1]
        if call.dispatch_receipt_sha256 is not None or call.outcome_sha256 is not None:
            raise RuntimeError("evaluation call is not safely resumable")
        return EvaluationCallAuthorization(
            call.call_id,
            call.task_attempt_id,
            call.call_ordinal,
            call.request_sha256,
            call.transport_sha256,
        )

    def resume_client_session(
        self,
        arm: ArmName,
        process_receipt_bytes: bytes,
    ) -> EvaluationClientSession:
        receipt = ActuatedProcessReceipt.from_bytes(process_receipt_bytes)
        program = self.manifest.program(arm, "client")
        self._verify_process_receipt(receipt, program, require_current=True)
        snapshot = self.inspect()
        epoch = snapshot.latest_epoch(arm)
        if epoch is None or epoch.process_receipt_sha256 != sha256(process_receipt_bytes):
            raise RuntimeError("evaluation client receipt does not own the current epoch")
        return EvaluationClientSession(arm, epoch.epoch, process_receipt_bytes)

    def resume_current_client_session(self, arm: ArmName) -> EvaluationClientSession:
        snapshot = self.inspect()
        epoch = snapshot.latest_epoch(arm)
        if epoch is None or epoch.process_receipt_sha256 is None:
            raise RuntimeError("evaluation arm lacks a client epoch")
        receipt_bytes = self.evidence.get(epoch.process_receipt_sha256)
        receipt = ActuatedProcessReceipt.from_bytes(receipt_bytes)
        self._verify_process_receipt(
            receipt,
            self.manifest.program(arm, "client"),
            require_current=True,
        )
        return EvaluationClientSession(arm, epoch.epoch, receipt_bytes)

    def reserve_next_task(self, *, session: EvaluationClientSession) -> EvaluationTaskAttempt:
        with exclusive_lock(self.lock_path):
            snapshot = self.inspect()
            self._verify_client_session(snapshot, session)
            self._verify_server_session(snapshot, session.arm)
            if snapshot.terminal_status != "active" or snapshot.current_task is not None:
                raise RuntimeError("evaluation cannot reserve another task")
            ordinal = len(snapshot.tasks)
            if ordinal >= len(self.manifest.schedule):
                raise RuntimeError("evaluation schedule is already complete")
            unit = self.manifest.schedule[ordinal]
            epoch = snapshot.latest_epoch(session.arm)
            if unit.arm != session.arm or epoch is None or epoch.epoch != session.epoch:
                raise RuntimeError("evaluation task reservation is out of frozen order")
            server_attestation = dict(snapshot.server_attestations).get(session.arm)
            if server_attestation is None:
                raise RuntimeError("evaluation task lacks a server attestation")
            attempt_id = sha256(
                canonical_json(
                    {
                        "domain": "redco-stage-d-evaluation-task-attempt-v1",
                        "execution_manifest_sha256": self.manifest.manifest_sha256,
                        **unit.to_payload(),
                    }
                )
            )
            self._append_unlocked(
                "task_reserved",
                {
                    **unit.to_payload(),
                    "task_attempt_id": attempt_id,
                    "client_epoch": session.epoch,
                    "server_attestation_sha256": server_attestation,
                },
            )
            return EvaluationTaskAttempt(attempt_id, unit, session.epoch)

    def reserve_call(
        self,
        task: EvaluationTaskAttempt,
        *,
        session: EvaluationClientSession,
        event_address_bytes: bytes,
        seed: int,
        cache_salt: str,
        request_body_bytes: bytes,
        transport_bytes: bytes,
        call_ordinal: int | None = None,
    ) -> EvaluationCallAuthorization:
        address_sha256 = self.evidence.put(event_address_bytes)
        body_sha256 = self.evidence.put(request_body_bytes)
        verify_transport_request(
            transport_bytes,
            expected_endpoint=self.manifest.program(task.unit.arm, "server").endpoint,
            expected_body_sha256=body_sha256,
        )
        transport_sha256 = self.evidence.put(transport_bytes)
        with exclusive_lock(self.lock_path):
            snapshot = self.inspect()
            self._verify_client_session(snapshot, session)
            self._verify_server_session(snapshot, session.arm)
            current = snapshot.current_task
            expected_task = (
                None
                if current is None
                else EvaluationTaskAttempt(
                    current.task_attempt_id,
                    current.unit,
                    current.client_epoch,
                )
            )
            if (
                current is None
                or task != expected_task
                or task.unit.arm != session.arm
                or task.client_epoch != session.epoch
            ):
                raise RuntimeError("evaluation call does not belong to the open task")
            expected_ordinal = len(current.calls) if call_ordinal is None else call_ordinal
            if type(expected_ordinal) is not int or not 0 <= expected_ordinal <= len(current.calls):
                raise RuntimeError("evaluation call ordinal is out of order")
            if expected_ordinal < len(current.calls):
                observed = current.calls[expected_ordinal]
                expected = (
                    task.task_attempt_id,
                    expected_ordinal,
                    address_sha256,
                    seed,
                    cache_salt,
                    body_sha256,
                    transport_sha256,
                )
                actual = (
                    observed.task_attempt_id,
                    observed.call_ordinal,
                    observed.event_address_sha256,
                    observed.seed,
                    observed.cache_salt,
                    observed.request_sha256,
                    observed.transport_sha256,
                )
                if actual != expected:
                    raise RuntimeError("evaluation replay call differs from its transcript")
                return EvaluationCallAuthorization(
                    observed.call_id,
                    observed.task_attempt_id,
                    observed.call_ordinal,
                    observed.request_sha256,
                    observed.transport_sha256,
                )
            if snapshot.terminal_status != "active":
                raise RuntimeError("evaluation call does not belong to the open task")
            if current.calls and current.calls[-1].outcome_sha256 is None:
                raise RuntimeError("evaluation task has an unfinished policy call")
            call_id = sha256(
                canonical_json(
                    {
                        "domain": "redco-stage-d-evaluation-call-v1",
                        "task_attempt_id": task.task_attempt_id,
                        "call_ordinal": expected_ordinal,
                        "event_address_sha256": address_sha256,
                        "seed": seed,
                        "cache_salt": cache_salt,
                    }
                )
            )
            self._append_unlocked(
                "call_reserved",
                {
                    "task_attempt_id": task.task_attempt_id,
                    "call_id": call_id,
                    "call_ordinal": expected_ordinal,
                    "event_address_sha256": address_sha256,
                    "seed": seed,
                    "cache_salt": cache_salt,
                    "request_sha256": body_sha256,
                    "transport_sha256": transport_sha256,
                },
            )
            return EvaluationCallAuthorization(
                call_id,
                task.task_attempt_id,
                expected_ordinal,
                body_sha256,
                transport_sha256,
            )

    def call_state(self, call: EvaluationCallAuthorization) -> EvaluationCallState:
        return _verified_call(self.inspect(), call, "transcript")

    def finalized_response_bytes(self, call: EvaluationCallAuthorization) -> bytes | None:
        observed = self.call_state(call)
        if observed.outcome_sha256 is None:
            return None
        outcome = canonical_object(
            self.evidence.get(observed.outcome_sha256),
            "evaluation call outcome",
        )
        return self.evidence.get(outcome["parsed_response_sha256"])

    def witnessed_raw_response_bytes(
        self,
        call: EvaluationCallAuthorization,
    ) -> bytes | None:
        observed = self.call_state(call)
        if observed.response_envelope_sha256 is None:
            return None
        return self.evidence.get(cast(str, observed.raw_response_sha256))

    def authorize_dispatch(
        self,
        call: EvaluationCallAuthorization,
        *,
        session: EvaluationClientSession,
    ) -> EvaluationDispatchAuthorization:
        receipt = canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-evaluation-dispatch-v1",
                "call_id": call.call_id,
                "request_sha256": call.request_sha256,
                "transport_sha256": call.transport_sha256,
            }
        )
        with exclusive_lock(self.lock_path):
            snapshot = self.inspect()
            self._verify_client_session(snapshot, session)
            self._verify_server_session(snapshot, session.arm)
            observed = _verified_call(snapshot, call, "reservation")
            if observed.dispatch_receipt_sha256 is not None:
                raise RuntimeError("evaluation call was already dispatch-authorized")
            digest = self.evidence.put(receipt)
            self._append_unlocked(
                "call_dispatch_authorized",
                {"call_id": call.call_id, "dispatch_receipt_sha256": digest},
            )
            return EvaluationDispatchAuthorization(call, digest)

    def record_response(
        self,
        dispatch: EvaluationDispatchAuthorization,
        *,
        session: EvaluationClientSession,
        status_code: int,
        headers: tuple[tuple[str, str], ...],
        raw_response_bytes: bytes,
        wall_seconds: float = 0.0,
    ) -> bytes:
        if type(status_code) is not int or not 100 <= status_code <= 599:
            raise ValueError("evaluation response status is invalid")
        verify_nonsecret_headers(headers)
        if not math.isfinite(wall_seconds) or wall_seconds < 0:
            raise ValueError("evaluation response wall time is invalid")
        with exclusive_lock(self.lock_path):
            snapshot = self.inspect()
            self._verify_client_session(snapshot, session)
            observed = _snapshot_call(snapshot, dispatch.call.call_id)
            expected_call = EvaluationCallAuthorization(
                observed.call_id,
                observed.task_attempt_id,
                observed.call_ordinal,
                observed.request_sha256,
                observed.transport_sha256,
            )
            if (
                dispatch.call != expected_call
                or observed.dispatch_receipt_sha256 != dispatch.dispatch_receipt_sha256
            ):
                raise RuntimeError("evaluation response lacks its durable dispatch")
        raw_sha256 = self.evidence.put(raw_response_bytes)
        envelope = canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-evaluation-response-envelope-v1",
                "call_id": dispatch.call.call_id,
                "dispatch_receipt_sha256": dispatch.dispatch_receipt_sha256,
                "status_code": status_code,
                "headers": dict(headers),
                "raw_response_sha256": raw_sha256,
                "wall_seconds": wall_seconds,
            }
        )
        pointer = self.responses / f"{dispatch.call.call_id}.json"
        atomic_publish(pointer, envelope, fault_hook=self._fault_hook)
        envelope_sha256 = self.evidence.put(envelope)
        with exclusive_lock(self.lock_path):
            snapshot = self.inspect()
            self._verify_client_session(snapshot, session)
            observed = _snapshot_call(snapshot, dispatch.call.call_id)
            if observed.response_envelope_sha256 is None:
                self._append_unlocked(
                    "call_response_witnessed",
                    {
                        "call_id": dispatch.call.call_id,
                        "dispatch_receipt_sha256": dispatch.dispatch_receipt_sha256,
                        "response_envelope_sha256": envelope_sha256,
                        "raw_response_sha256": raw_sha256,
                    },
                )
            elif observed.response_envelope_sha256 != envelope_sha256:
                raise RuntimeError("evaluation call response differs from its witness")
        return envelope

    def finalize_call(
        self,
        call: EvaluationCallAuthorization,
        *,
        session: EvaluationClientSession,
        parsed_response_bytes: bytes,
        prompt_tokens: int,
        completion_tokens: int,
        wall_seconds: float,
        gpu_seconds: float,
        finish_kind: str,
    ) -> bytes:
        for name, value in (
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"evaluation {name} is invalid")
        for name, numeric_value in (
            ("wall_seconds", wall_seconds),
            ("gpu_seconds", gpu_seconds),
        ):
            if not math.isfinite(numeric_value) or numeric_value < 0:
                raise ValueError(f"evaluation {name} is invalid")
        if not finish_kind or not finish_kind.isprintable():
            raise ValueError("evaluation call finish kind is invalid")
        parsed_sha256 = self.evidence.put(parsed_response_bytes)
        with exclusive_lock(self.lock_path):
            snapshot = self.inspect()
            self._verify_client_session(snapshot, session)
            observed = _verified_call(snapshot, call, "reservation")
            if observed.response_envelope_sha256 is None:
                raise RuntimeError("evaluation call lacks a witnessed response")
            outcome = canonical_json(
                {
                    "schema_version": 1,
                    "domain": "redco-stage-d-evaluation-call-outcome-v1",
                    "call_id": call.call_id,
                    "response_envelope_sha256": observed.response_envelope_sha256,
                    "parsed_response_sha256": parsed_sha256,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "wall_seconds": wall_seconds,
                    "gpu_seconds": gpu_seconds,
                    "finish_kind": finish_kind,
                }
            )
            if observed.outcome_sha256 is not None:
                existing = self.evidence.get(observed.outcome_sha256)
                if existing != outcome:
                    raise RuntimeError("evaluation call outcome differs from its durable result")
                return existing
            digest = self.evidence.put(outcome)
            self._append_unlocked(
                "call_outcome_finalized",
                {
                    "call_id": call.call_id,
                    "response_envelope_sha256": observed.response_envelope_sha256,
                    "outcome_sha256": digest,
                },
            )
            return outcome

    def complete_task(
        self,
        task: EvaluationTaskAttempt,
        *,
        session: EvaluationClientSession,
        terminal_result_bytes: bytes,
        scorer_evidence_bytes: bytes,
        reward: float,
        overhead_wall_seconds: float = 0.0,
        overhead_gpu_seconds: float = 0.0,
    ) -> bytes:
        plan = self.plan
        if not math.isfinite(reward) or not plan.reward_min <= reward <= plan.reward_max:
            raise ValueError("evaluation task reward is outside the frozen bounds")
        for value in (overhead_wall_seconds, overhead_gpu_seconds):
            if not math.isfinite(value) or value < 0:
                raise ValueError("evaluation task overhead is invalid")
        terminal_sha256 = self.evidence.put(terminal_result_bytes)
        scorer_sha256 = self.evidence.put(scorer_evidence_bytes)
        with exclusive_lock(self.lock_path):
            snapshot = self.inspect()
            self._verify_client_session(snapshot, session)
            current = snapshot.current_task
            expected_task = (
                None
                if current is None
                else EvaluationTaskAttempt(
                    current.task_attempt_id,
                    current.unit,
                    current.client_epoch,
                )
            )
            if current is None or task != expected_task:
                raise RuntimeError("evaluation task is not the open task")
            if not current.calls or any(call.outcome_sha256 is None for call in current.calls):
                raise RuntimeError("evaluation task has unfinished policy calls")
            outcomes = [
                canonical_object(self.evidence.get(cast(str, call.outcome_sha256)), "call outcome")
                for call in current.calls
            ]
            metrics = canonical_json(
                {
                    "schema_version": 1,
                    "domain": "redco-stage-d-evaluation-task-metrics-v1",
                    "task_attempt_id": task.task_attempt_id,
                    **task.unit.to_payload(),
                    "call_ids": [call.call_id for call in current.calls],
                    "terminal_result_sha256": terminal_sha256,
                    "scorer_evidence_sha256": scorer_sha256,
                    "reward": reward,
                    "success": reward >= plan.success_reward_threshold,
                    "policy_calls": len(current.calls),
                    "prompt_tokens": sum(item["prompt_tokens"] for item in outcomes),
                    "completion_tokens": sum(item["completion_tokens"] for item in outcomes),
                    "wall_seconds": overhead_wall_seconds
                    + sum(item["wall_seconds"] for item in outcomes),
                    "gpu_seconds": overhead_gpu_seconds
                    + sum(item["gpu_seconds"] for item in outcomes),
                }
            )
            metrics_sha256 = self.evidence.put(metrics)
            self._append_unlocked(
                "task_completed",
                {
                    "task_attempt_id": task.task_attempt_id,
                    "terminal_result_sha256": terminal_sha256,
                    "task_metrics_sha256": metrics_sha256,
                    "call_ids": [call.call_id for call in current.calls],
                },
            )
            return metrics

    def _verify_client_session(
        self,
        snapshot: EvaluationLedgerSnapshot,
        session: EvaluationClientSession,
    ) -> None:
        program = self.manifest.program(session.arm, "client")
        receipt = ActuatedProcessReceipt.from_bytes(session.process_receipt_bytes)
        self._verify_process_receipt(receipt, program, require_current=True)
        epoch = snapshot.latest_epoch(session.arm)
        if (
            epoch is None
            or epoch.epoch != session.epoch
            or epoch.process_receipt_sha256 != session.process_receipt_sha256
        ):
            raise RuntimeError("evaluation client session does not own the current epoch")

    def _verify_server_session(
        self,
        snapshot: EvaluationLedgerSnapshot,
        arm: ArmName,
    ) -> None:
        claim = dict(snapshot.server_claims).get(arm)
        attestation = dict(snapshot.server_attestations).get(arm)
        if claim is None or attestation is None:
            raise RuntimeError("evaluation arm lacks a claimed and attested server")
        receipt = ActuatedProcessReceipt.from_bytes(self.evidence.get(claim))
        self._verify_process_receipt(
            receipt,
            self.manifest.program(arm, "server"),
            require_current=False,
        )

    def complete_arm(self, arm: ArmName) -> bytes:
        return evaluation_completion.complete_arm(self, arm)

    def seal(self) -> bytes:
        with exclusive_lock(self.lock_path):
            snapshot = self.inspect()
            if snapshot.sealed:
                return self.records.joinpath(f"{snapshot.record_count - 1:08d}.json").read_bytes()
            if tuple(arm for arm, _ in snapshot.arm_completions) != _ARMS:
                raise RuntimeError("evaluation cannot seal before all arm completions")
            self._append_unlocked(
                "seal",
                {
                    "arm_completion_sha256s": [
                        [arm, digest] for arm, digest in snapshot.arm_completions
                    ]
                },
            )
            result = self.inspect()
            if not result.sealed:
                raise RuntimeError("evaluation seal did not produce terminal state")
            return self.records.joinpath(f"{result.record_count - 1:08d}.json").read_bytes()

    def _verify_process_receipt(
        self,
        receipt: ActuatedProcessReceipt,
        program: EvaluationProgramBinding,
        *,
        require_current: bool,
    ) -> None:
        if receipt.arm != program.arm or receipt.role != program.role:
            raise ValueError("evaluation process receipt identity differs")
        if receipt.command_sha256 != command_sha256(program.argv) or (
            receipt.environment_manifest_sha256
            != evaluation_environment_sha256(program.environment)
        ):
            raise ValueError("evaluation process receipt command or environment differs")
        if not receipt.is_same_live_process() or (require_current and receipt.pid != os.getpid()):
            raise ValueError("evaluation process receipt is stale or belongs elsewhere")

    def _append(self, kind: str, event: dict[str, Any]) -> None:
        with exclusive_lock(self.lock_path):
            self._append_unlocked(kind, event)

    def _append_unlocked(self, kind: str, event: dict[str, Any]) -> str:
        paths = sorted(self.records.glob("*.json"))
        offset = len(paths)
        prior = None if offset == 0 else sha256(paths[-1].read_bytes())
        value = encode_record(
            offset=offset,
            prior_record_sha256=prior,
            record_kind=kind,
            event=event,
        )
        atomic_publish(
            self.records / f"{offset:08d}.json",
            value,
            fault_hook=self._fault_hook,
        )
        return sha256(value)


def _snapshot_call(snapshot: EvaluationLedgerSnapshot, call_id: str) -> EvaluationCallState:
    matches = [call for task in snapshot.tasks for call in task.calls if call.call_id == call_id]
    if len(matches) != 1:
        raise RuntimeError("evaluation call authorization is unknown")
    return matches[0]


def _verified_call(
    snapshot: EvaluationLedgerSnapshot,
    call: EvaluationCallAuthorization,
    source: str,
) -> EvaluationCallState:
    observed = _snapshot_call(snapshot, call.call_id)
    expected = EvaluationCallAuthorization(
        observed.call_id, observed.task_attempt_id, observed.call_ordinal,
        observed.request_sha256, observed.transport_sha256,
    )
    if call != expected:
        raise RuntimeError(f"evaluation call capability differs from its {source}")
    return observed


__all__ = [
    "EvaluationCallAuthorization",
    "EvaluationClientLaunch",
    "EvaluationClientSession",
    "EvaluationDispatchAuthorization",
    "EvaluationServerLaunch",
    "EvaluationTaskAttempt",
    "StageDEvaluationLedger",
]
