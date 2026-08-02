"""Pure reducer for the append-only Stage-D evaluation ledger."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from redco.analysis.stage_d_evaluation_contracts import (
    EvaluationScheduleUnit,
    StageDEvaluationExecutionManifest,
)
from redco.analysis.stage_d_objective_binding import ArmName

_ARMS: tuple[ArmName, ...] = ("stock", "branch-global", "local")
_SHA_FIELDS = {
    "authorization_sha256",
    "execution_manifest_sha256",
    "evaluation_plan_sha256",
    "process_receipt_sha256",
    "prior_process_receipt_sha256",
    "dead_process_evidence_sha256",
    "server_attestation_sha256",
    "launch_record_sha256",
    "resume_task_attempt_id",
    "request_sha256",
    "transport_sha256",
    "dispatch_receipt_sha256",
    "response_envelope_sha256",
    "raw_response_sha256",
    "outcome_sha256",
    "terminal_result_sha256",
    "task_metrics_sha256",
    "arm_metrics_sha256",
    "supervisor_identity_sha256",
    "error_evidence_sha256",
    "cleanup_evidence_sha256",
}
_EVENT_FIELDS = {
    "genesis": {
        "authorization_sha256",
        "execution_manifest_sha256",
        "evaluation_plan_sha256",
        "created_at_unix_ns",
    },
    "server_launch_reserved": {
        "arm",
        "epoch",
        "prior_process_receipt_sha256",
        "dead_process_evidence_sha256",
    },
    "server_claimed": {
        "arm",
        "epoch",
        "launch_record_sha256",
        "process_receipt_sha256",
    },
    "server_attested": {
        "arm",
        "epoch",
        "launch_record_sha256",
        "process_receipt_sha256",
        "server_attestation_sha256",
    },
    "client_launch_reserved": {
        "arm",
        "epoch",
        "resume_task_attempt_id",
        "prior_process_receipt_sha256",
        "dead_process_evidence_sha256",
    },
    "client_claimed": {
        "arm",
        "epoch",
        "launch_record_sha256",
        "resume_task_attempt_id",
        "process_receipt_sha256",
    },
    "actuation_attempt_reserved": {
        "actuation_attempt_id",
        "arm",
        "role",
        "epoch",
        "launch_record_sha256",
        "supervisor_identity_sha256",
    },
    "actuation_attempt_disposition": {
        "actuation_attempt_id",
        "disposition",
        "process_receipt_sha256",
        "error_evidence_sha256",
        "cleanup_evidence_sha256",
    },
    "task_reserved": {
        "ordinal",
        "arm",
        "task_index",
        "task_id",
        "seed",
        "task_attempt_id",
        "client_epoch",
        "server_attestation_sha256",
    },
    "call_reserved": {
        "task_attempt_id",
        "call_id",
        "call_ordinal",
        "event_address_sha256",
        "seed",
        "cache_salt",
        "request_sha256",
        "transport_sha256",
    },
    "call_dispatch_authorized": {"call_id", "dispatch_receipt_sha256"},
    "call_response_witnessed": {
        "call_id",
        "dispatch_receipt_sha256",
        "response_envelope_sha256",
        "raw_response_sha256",
    },
    "call_outcome_finalized": {
        "call_id",
        "response_envelope_sha256",
        "outcome_sha256",
    },
    "task_completed": {
        "task_attempt_id",
        "terminal_result_sha256",
        "task_metrics_sha256",
        "call_ids",
    },
    "arm_completed": {"arm", "arm_metrics_sha256", "task_attempt_ids"},
    "seal": {"arm_completion_sha256s"},
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class EvaluationCallState:
    call_id: str
    task_attempt_id: str
    call_ordinal: int
    event_address_sha256: str
    seed: int
    cache_salt: str
    request_sha256: str
    transport_sha256: str
    dispatch_receipt_sha256: str | None = None
    response_envelope_sha256: str | None = None
    raw_response_sha256: str | None = None
    outcome_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationTaskState:
    unit: EvaluationScheduleUnit
    task_attempt_id: str
    client_epoch: int
    server_attestation_sha256: str
    calls: tuple[EvaluationCallState, ...] = ()
    terminal_result_sha256: str | None = None
    task_metrics_sha256: str | None = None

    @property
    def completed(self) -> bool:
        return self.terminal_result_sha256 is not None


@dataclass(frozen=True, slots=True)
class EvaluationProcessEpoch:
    arm: ArmName
    epoch: int
    launch_record_sha256: str
    resume_task_attempt_id: str | None
    process_receipt_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationServerEpoch:
    arm: ArmName
    epoch: int
    launch_record_sha256: str
    process_receipt_sha256: str | None = None
    server_attestation_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationActuationAttempt:
    actuation_attempt_id: str
    arm: ArmName
    role: Literal["client", "server"]
    epoch: int
    launch_record_sha256: str
    supervisor_identity_sha256: str
    disposition: Literal[
        "claimed",
        "lost-claim-and-drained",
        "spawn-failed",
        "not-observed-after-crash",
        "cleanup-failed",
    ] | None = None
    process_receipt_sha256: str | None = None
    error_evidence_sha256: str | None = None
    cleanup_evidence_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationLedgerSnapshot:
    authorization_sha256: str
    execution_manifest_sha256: str
    evaluation_plan_sha256: str
    created_at_unix_ns: int
    server_claims: tuple[tuple[ArmName, str], ...]
    server_attestations: tuple[tuple[ArmName, str], ...]
    server_epochs: tuple[EvaluationServerEpoch, ...]
    client_epochs: tuple[EvaluationProcessEpoch, ...]
    actuation_attempts: tuple[EvaluationActuationAttempt, ...]
    tasks: tuple[EvaluationTaskState, ...]
    arm_completions: tuple[tuple[ArmName, str], ...]
    arm_metrics: tuple[tuple[ArmName, str], ...]
    sealed: bool
    terminal_status: Literal[
        "active",
        "ambiguous-dispatch",
        "orphaned-open-task",
        "orphaned-server",
        "sealed",
    ]
    head_sha256: str
    record_count: int

    @property
    def current_task(self) -> EvaluationTaskState | None:
        return self.tasks[-1] if self.tasks and not self.tasks[-1].completed else None

    def latest_epoch(self, arm: ArmName) -> EvaluationProcessEpoch | None:
        matches = [item for item in self.client_epochs if item.arm == arm]
        return matches[-1] if matches else None

    def latest_server_epoch(self, arm: ArmName) -> EvaluationServerEpoch | None:
        matches = [item for item in self.server_epochs if item.arm == arm]
        return matches[-1] if matches else None


def reduce_evaluation_records(
    records: Sequence[Mapping[str, Any]],
    record_sha256s: Sequence[str],
    *,
    manifest: StageDEvaluationExecutionManifest,
) -> EvaluationLedgerSnapshot:
    if not records or len(records) != len(record_sha256s):
        raise ValueError("evaluation ledger is empty or has inconsistent hashes")
    for offset, (record, digest) in enumerate(zip(records, record_sha256s, strict=True)):
        if record.get("offset") != offset or not _is_sha256(digest):
            raise ValueError("evaluation ledger offset or digest differs")
        expected_prior = None if offset == 0 else record_sha256s[offset - 1]
        if record.get("prior_record_sha256") != expected_prior:
            raise ValueError("evaluation ledger hash chain differs")
        kind = record.get("record_kind")
        event = record.get("event")
        if (
            kind not in _EVENT_FIELDS
            or not isinstance(event, dict)
            or set(event) != _EVENT_FIELDS[kind]
        ):
            raise ValueError("evaluation ledger event schema differs")
        for name, value in event.items():
            if name in _SHA_FIELDS and value is not None and not _is_sha256(value):
                raise ValueError(f"evaluation ledger {name} is invalid")
    genesis = records[0]
    if genesis["record_kind"] != "genesis":
        raise ValueError("evaluation ledger must start with genesis")
    body = genesis["event"]
    if (
        body["execution_manifest_sha256"] != manifest.manifest_sha256
        or body["evaluation_plan_sha256"] != manifest.evaluation_plan_sha256
        or type(body["created_at_unix_ns"]) is not int
        or body["created_at_unix_ns"] < 1
    ):
        raise ValueError("evaluation ledger genesis differs from execution manifest")

    server_claims: dict[ArmName, str] = {}
    server_attestations: dict[ArmName, str] = {}
    server_epochs: list[EvaluationServerEpoch] = []
    client_epochs: list[EvaluationProcessEpoch] = []
    actuation_attempts: list[EvaluationActuationAttempt] = []
    attempt_by_id: dict[str, int] = {}
    tasks: list[EvaluationTaskState] = []
    task_by_id: dict[str, int] = {}
    call_to_task: dict[str, int] = {}
    arm_completions: dict[ArmName, str] = {}
    arm_metrics: dict[ArmName, str] = {}
    sealed = False

    for record, record_digest in zip(records[1:], record_sha256s[1:], strict=True):
        kind = record["record_kind"]
        event = record["event"]
        if sealed:
            raise ValueError("evaluation ledger appends after its seal")
        if kind == "server_launch_reserved":
            arm = _arm(event["arm"])
            previous = next(
                (item for item in reversed(server_epochs) if item.arm == arm),
                None,
            )
            expected_epoch = 0 if previous is None else previous.epoch + 1
            if (
                event["epoch"] != expected_epoch
                or expected_epoch >= manifest.max_server_launches_per_arm
            ):
                raise ValueError("evaluation server launch epoch is out of order")
            if previous is None:
                if (
                    event["prior_process_receipt_sha256"] is not None
                    or event["dead_process_evidence_sha256"] is not None
                ):
                    raise ValueError("first evaluation server has replacement evidence")
            elif (
                previous.process_receipt_sha256 is None
                or event["prior_process_receipt_sha256"] != previous.process_receipt_sha256
                or event["dead_process_evidence_sha256"] is None
                or any(
                    call.dispatch_receipt_sha256 is not None
                    for task in tasks
                    if task.unit.arm == arm
                    for call in task.calls
                )
            ):
                raise ValueError("evaluation server replacement is invalid")
            server_epochs.append(EvaluationServerEpoch(arm, expected_epoch, record_digest))
            server_claims.pop(arm, None)
            server_attestations.pop(arm, None)
        elif kind == "server_claimed":
            arm = _arm(event["arm"])
            epoch_index = next(
                (
                    index
                    for index in range(len(server_epochs) - 1, -1, -1)
                    if server_epochs[index].arm == arm
                ),
                None,
            )
            epoch = None if epoch_index is None else server_epochs[epoch_index]
            if (
                epoch is None
                or epoch.epoch != event["epoch"]
                or epoch.launch_record_sha256 != event["launch_record_sha256"]
                or epoch.process_receipt_sha256 is not None
            ):
                raise ValueError("evaluation server claim lacks its launch reservation")
            server_claims[arm] = event["process_receipt_sha256"]
            assert epoch_index is not None
            server_epochs[epoch_index] = replace(
                epoch,
                process_receipt_sha256=event["process_receipt_sha256"],
            )
        elif kind == "server_attested":
            arm = _arm(event["arm"])
            epoch_index = next(
                (
                    index
                    for index in range(len(server_epochs) - 1, -1, -1)
                    if server_epochs[index].arm == arm
                ),
                None,
            )
            epoch = None if epoch_index is None else server_epochs[epoch_index]
            if (
                server_claims.get(arm) != event["process_receipt_sha256"]
                or epoch is None
                or epoch.epoch != event["epoch"]
                or epoch.launch_record_sha256 != event["launch_record_sha256"]
                or epoch.server_attestation_sha256 is not None
            ):
                raise ValueError("evaluation server attestation is out of order")
            server_attestations[arm] = event["server_attestation_sha256"]
            assert epoch_index is not None
            server_epochs[epoch_index] = replace(
                epoch,
                server_attestation_sha256=event["server_attestation_sha256"],
            )
        elif kind == "client_launch_reserved":
            arm = _arm(event["arm"])
            previous_client = next(
                (item for item in reversed(client_epochs) if item.arm == arm), None
            )
            expected_epoch = 0 if previous_client is None else previous_client.epoch + 1
            current = tasks[-1] if tasks and not tasks[-1].completed else None
            expected_resume = None if current is None else current.task_attempt_id
            expected_arm = (
                current.unit.arm
                if current is not None
                else (
                    manifest.schedule[len(tasks)].arm
                    if len(tasks) < len(manifest.schedule)
                    else None
                )
            )
            if (
                arm != expected_arm
                or event["epoch"] != expected_epoch
                or expected_epoch >= manifest.max_client_launches_per_arm
                or event["resume_task_attempt_id"] != expected_resume
            ):
                raise ValueError("evaluation client epoch is out of order")
            if previous_client is None:
                if (
                    event["prior_process_receipt_sha256"] is not None
                    or event["dead_process_evidence_sha256"] is not None
                ):
                    raise ValueError("first evaluation client has replacement evidence")
            elif (
                previous_client.process_receipt_sha256 is None
                or event["prior_process_receipt_sha256"] != previous_client.process_receipt_sha256
                or event["dead_process_evidence_sha256"] is None
                or (
                    current is not None
                    and (
                        current.client_epoch != previous_client.epoch
                        or any(
                            call.dispatch_receipt_sha256 is not None
                            and call.response_envelope_sha256 is None
                            for call in current.calls
                        )
                    )
                )
            ):
                raise ValueError("replacement evaluation client lacks dead-process evidence")
            client_epochs.append(
                EvaluationProcessEpoch(
                    arm,
                    expected_epoch,
                    record_digest,
                    event["resume_task_attempt_id"],
                )
            )
        elif kind == "client_claimed":
            arm = _arm(event["arm"])
            epoch_index = next(
                (
                    index
                    for index in range(len(client_epochs) - 1, -1, -1)
                    if client_epochs[index].arm == arm
                ),
                None,
            )
            client_epoch = None if epoch_index is None else client_epochs[epoch_index]
            if (
                client_epoch is None
                or client_epoch.epoch != event["epoch"]
                or client_epoch.launch_record_sha256 != event["launch_record_sha256"]
                or client_epoch.resume_task_attempt_id != event["resume_task_attempt_id"]
                or client_epoch.process_receipt_sha256 is not None
            ):
                raise ValueError("evaluation client claim lacks its launch reservation")
            assert epoch_index is not None
            client_epochs[epoch_index] = replace(
                client_epoch,
                process_receipt_sha256=event["process_receipt_sha256"],
            )
            if client_epoch.resume_task_attempt_id is not None:
                if not tasks or tasks[-1].completed:
                    raise ValueError("evaluation client reclaim lacks an open task")
                task = tasks[-1]
                if (
                    task.unit.arm != arm
                    or task.task_attempt_id != client_epoch.resume_task_attempt_id
                ):
                    raise ValueError("evaluation client reclaim task changed before claim")
                tasks[-1] = replace(task, client_epoch=client_epoch.epoch)
        elif kind == "actuation_attempt_reserved":
            attempt_id = event["actuation_attempt_id"]
            arm = _arm(event["arm"])
            role = event["role"]
            attempt_epoch = _latest_process_epoch(
                server_epochs,
                client_epochs,
                arm=arm,
                role=role,
            )
            if (
                not isinstance(attempt_id, str)
                or len(attempt_id) != 32
                or any(character not in "0123456789abcdef" for character in attempt_id)
                or attempt_id in attempt_by_id
                or role not in {"client", "server"}
                or attempt_epoch is None
                or (attempt_epoch.epoch, attempt_epoch.launch_record_sha256)
                != (event["epoch"], event["launch_record_sha256"])
            ):
                raise ValueError("evaluation actuation attempt reservation is invalid")
            attempt_by_id[attempt_id] = len(actuation_attempts)
            actuation_attempts.append(
                EvaluationActuationAttempt(
                    attempt_id,
                    arm,
                    role,
                    event["epoch"],
                    event["launch_record_sha256"],
                    event["supervisor_identity_sha256"],
                )
            )
        elif kind == "actuation_attempt_disposition":
            attempt_id = event["actuation_attempt_id"]
            disposition = event["disposition"]
            index = attempt_by_id.get(attempt_id)
            attempt = None if index is None else actuation_attempts[index]
            process_receipt = event["process_receipt_sha256"]
            error_evidence = event["error_evidence_sha256"]
            cleanup_evidence = event["cleanup_evidence_sha256"]
            attempt_epoch = (
                None
                if attempt is None
                else _latest_process_epoch(
                    server_epochs,
                    client_epochs,
                    arm=attempt.arm,
                    role=attempt.role,
                )
            )
            if (
                attempt is None
                or attempt.disposition is not None
                or disposition
                not in {
                    "claimed",
                    "lost-claim-and-drained",
                    "spawn-failed",
                    "not-observed-after-crash",
                    "cleanup-failed",
                }
                or (disposition == "claimed") != (process_receipt is not None)
                or (
                    disposition == "claimed"
                    and (
                        attempt_epoch is None
                        or attempt_epoch.epoch != attempt.epoch
                        or attempt_epoch.process_receipt_sha256 != process_receipt
                    )
                )
                or (disposition == "spawn-failed") != (error_evidence is not None)
                or (disposition in {"lost-claim-and-drained", "cleanup-failed"})
                != (cleanup_evidence is not None)
            ):
                raise ValueError("evaluation actuation attempt disposition is invalid")
            assert index is not None
            actuation_attempts[index] = replace(
                attempt,
                disposition=disposition,
                process_receipt_sha256=process_receipt,
                error_evidence_sha256=error_evidence,
                cleanup_evidence_sha256=cleanup_evidence,
            )
        elif kind == "task_reserved":
            if any(not task.completed for task in tasks):
                raise ValueError("evaluation reserved a task while another task was open")
            next_ordinal = len(tasks)
            if next_ordinal >= len(manifest.schedule):
                raise ValueError("evaluation reserved a task beyond the frozen schedule")
            unit = manifest.schedule[next_ordinal]
            if (
                event["ordinal"],
                event["arm"],
                event["task_index"],
                event["task_id"],
                event["seed"],
            ) != (
                unit.ordinal,
                unit.arm,
                unit.task_index,
                unit.task_id,
                unit.seed,
            ):
                raise ValueError("evaluation task differs from the next frozen unit")
            client_epoch_state = next(
                (item for item in reversed(client_epochs) if item.arm == unit.arm), None
            )
            if (
                client_epoch_state is None
                or client_epoch_state.process_receipt_sha256 is None
                or event["client_epoch"] != client_epoch_state.epoch
                or server_attestations.get(unit.arm) != event["server_attestation_sha256"]
            ):
                raise ValueError("evaluation task lacks its client/server authorization")
            attempt_id = event["task_attempt_id"]
            if not _is_sha256(attempt_id) or attempt_id in task_by_id:
                raise ValueError("evaluation task attempt identity is invalid or duplicated")
            task_by_id[attempt_id] = len(tasks)
            tasks.append(
                EvaluationTaskState(
                    unit,
                    attempt_id,
                    client_epoch_state.epoch,
                    event["server_attestation_sha256"],
                )
            )
        elif kind == "call_reserved":
            task_index = _open_task_index(tasks, event["task_attempt_id"])
            task = tasks[task_index]
            call_id = event["call_id"]
            if (
                not _is_sha256(call_id)
                or call_id in call_to_task
                or event["call_ordinal"] != len(task.calls)
                or type(event["seed"]) is not int
                or event["seed"] < 0
                or not isinstance(event["cache_salt"], str)
                or not event["cache_salt"]
                or not _is_sha256(event["event_address_sha256"])
            ):
                raise ValueError("evaluation call reservation is invalid")
            call_to_task[call_id] = task_index
            tasks[task_index] = replace(
                task,
                calls=(
                    *task.calls,
                    EvaluationCallState(
                        call_id,
                        task.task_attempt_id,
                        event["call_ordinal"],
                        event["event_address_sha256"],
                        event["seed"],
                        event["cache_salt"],
                        event["request_sha256"],
                        event["transport_sha256"],
                    ),
                ),
            )
        elif kind == "call_dispatch_authorized":
            _update_call(
                tasks,
                call_to_task,
                event["call_id"],
                require="reserved",
                dispatch_receipt_sha256=event["dispatch_receipt_sha256"],
            )
        elif kind == "call_response_witnessed":
            call = _call(tasks, call_to_task, event["call_id"])
            if call.dispatch_receipt_sha256 != event["dispatch_receipt_sha256"]:
                raise ValueError("evaluation response differs from dispatch authorization")
            _update_call(
                tasks,
                call_to_task,
                event["call_id"],
                require="dispatched",
                response_envelope_sha256=event["response_envelope_sha256"],
                raw_response_sha256=event["raw_response_sha256"],
            )
        elif kind == "call_outcome_finalized":
            call = _call(tasks, call_to_task, event["call_id"])
            if call.response_envelope_sha256 != event["response_envelope_sha256"]:
                raise ValueError("evaluation call outcome differs from witnessed response")
            _update_call(
                tasks,
                call_to_task,
                event["call_id"],
                require="responded",
                outcome_sha256=event["outcome_sha256"],
            )
        elif kind == "task_completed":
            task_index = _open_task_index(tasks, event["task_attempt_id"])
            task = tasks[task_index]
            if (
                not task.calls
                or any(call.outcome_sha256 is None for call in task.calls)
                or event["call_ids"] != [call.call_id for call in task.calls]
            ):
                raise ValueError("evaluation task completion has open or different calls")
            tasks[task_index] = replace(
                task,
                terminal_result_sha256=event["terminal_result_sha256"],
                task_metrics_sha256=event["task_metrics_sha256"],
            )
        elif kind == "arm_completed":
            arm = _arm(event["arm"])
            expected = [
                task.task_attempt_id for task in tasks if task.unit.arm == arm and task.completed
            ]
            frozen_count = sum(item.arm == arm for item in manifest.schedule)
            if (
                arm in arm_completions
                or len(expected) != frozen_count
                or event["task_attempt_ids"] != expected
            ):
                raise ValueError("evaluation arm completion roster differs")
            arm_metrics[arm] = event["arm_metrics_sha256"]
            arm_completions[arm] = record_digest
        elif kind == "seal":
            expected_completion_pairs = [[arm, arm_completions.get(arm)] for arm in _ARMS]
            if event["arm_completion_sha256s"] != expected_completion_pairs or any(
                value is None for _, value in expected_completion_pairs
            ):
                raise ValueError("evaluation seal lacks exact arm completions")
            if len(tasks) != len(manifest.schedule) or any(not task.completed for task in tasks):
                raise ValueError("evaluation seal precedes the frozen task schedule")
            sealed = True
        else:
            raise ValueError(f"unsupported evaluation record kind: {kind}")

    ambiguous = any(
        call.dispatch_receipt_sha256 is not None and call.response_envelope_sha256 is None
        for task in tasks
        for call in task.calls
    )
    return EvaluationLedgerSnapshot(
        authorization_sha256=body["authorization_sha256"],
        execution_manifest_sha256=body["execution_manifest_sha256"],
        evaluation_plan_sha256=body["evaluation_plan_sha256"],
        created_at_unix_ns=body["created_at_unix_ns"],
        server_claims=tuple((arm, server_claims[arm]) for arm in _ARMS if arm in server_claims),
        server_attestations=tuple(
            (arm, server_attestations[arm]) for arm in _ARMS if arm in server_attestations
        ),
        server_epochs=tuple(server_epochs),
        client_epochs=tuple(client_epochs),
        actuation_attempts=tuple(actuation_attempts),
        tasks=tuple(tasks),
        arm_completions=tuple(
            (arm, arm_completions[arm]) for arm in _ARMS if arm in arm_completions
        ),
        arm_metrics=tuple((arm, arm_metrics[arm]) for arm in _ARMS if arm in arm_metrics),
        sealed=sealed,
        terminal_status="sealed" if sealed else "ambiguous-dispatch" if ambiguous else "active",
        head_sha256=record_sha256s[-1],
        record_count=len(records),
    )


def _arm(value: object) -> ArmName:
    if value not in _ARMS:
        raise ValueError("evaluation record arm is invalid")
    return value


def _latest_process_epoch(
    server_epochs: list[EvaluationServerEpoch],
    client_epochs: list[EvaluationProcessEpoch],
    *,
    arm: ArmName,
    role: object,
) -> EvaluationServerEpoch | EvaluationProcessEpoch | None:
    if role == "server":
        return next((item for item in reversed(server_epochs) if item.arm == arm), None)
    if role == "client":
        return next((item for item in reversed(client_epochs) if item.arm == arm), None)
    return None


def _open_task_index(tasks: list[EvaluationTaskState], attempt_id: object) -> int:
    if not tasks or tasks[-1].task_attempt_id != attempt_id or tasks[-1].completed:
        raise ValueError("evaluation record does not name the open task")
    return len(tasks) - 1


def _call(
    tasks: list[EvaluationTaskState],
    call_to_task: dict[str, int],
    call_id: object,
) -> EvaluationCallState:
    if not isinstance(call_id, str) or call_id not in call_to_task:
        raise ValueError("evaluation record names an unknown call")
    task = tasks[call_to_task[call_id]]
    return next(item for item in task.calls if item.call_id == call_id)


def _update_call(
    tasks: list[EvaluationTaskState],
    call_to_task: dict[str, int],
    call_id: object,
    *,
    require: Literal["reserved", "dispatched", "responded"],
    dispatch_receipt_sha256: str | None = None,
    response_envelope_sha256: str | None = None,
    raw_response_sha256: str | None = None,
    outcome_sha256: str | None = None,
) -> None:
    call = _call(tasks, call_to_task, call_id)
    conditions = {
        "reserved": call.dispatch_receipt_sha256 is None,
        "dispatched": call.dispatch_receipt_sha256 is not None
        and call.response_envelope_sha256 is None,
        "responded": call.response_envelope_sha256 is not None and call.outcome_sha256 is None,
    }
    if not conditions[require]:
        raise ValueError(f"evaluation call transition requires {require}")
    task_index = call_to_task[call.call_id]
    task = tasks[task_index]
    if require == "reserved":
        updated = replace(call, dispatch_receipt_sha256=dispatch_receipt_sha256)
    elif require == "dispatched":
        updated = replace(
            call,
            response_envelope_sha256=response_envelope_sha256,
            raw_response_sha256=raw_response_sha256,
        )
    else:
        updated = replace(call, outcome_sha256=outcome_sha256)
    calls = tuple(updated if item.call_id == call.call_id else item for item in task.calls)
    tasks[task_index] = replace(task, calls=calls)


__all__ = [
    "EvaluationActuationAttempt",
    "EvaluationCallState",
    "EvaluationLedgerSnapshot",
    "EvaluationProcessEpoch",
    "EvaluationServerEpoch",
    "EvaluationTaskState",
    "reduce_evaluation_records",
]
