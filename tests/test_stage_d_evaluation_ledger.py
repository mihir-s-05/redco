from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from redco.analysis.stage_d_evaluation_actuation import (
    ActuatedProcessReceipt,
    EvaluationSupervisorIdentity,
)
from redco.analysis.stage_d_evaluation_barrier import (
    EvaluationCheckpointBinding,
    StageDEvaluationAuthorization,
    StageDEvaluationPlan,
    StageDEvaluationTask,
    StageDSealedEvaluationCompletion,
    commit_sealed_heldout_evaluation,
)
from redco.analysis.stage_d_evaluation_codec import atomic_publish
from redco.analysis.stage_d_evaluation_contracts import (
    EvaluationProgramBinding,
    EvaluationRuntimeEntrypoint,
    EvaluationScheduleUnit,
    EvaluationSupervisorLimits,
    StageDEvaluationExecutionManifest,
    evaluation_environment_sha256,
)
from redco.analysis.stage_d_evaluation_http import dispatch_local_http_once
from redco.analysis.stage_d_evaluation_ledger import StageDEvaluationLedger
from redco.analysis.stage_d_evaluation_server import (
    EvaluationServerProcessObservation,
)
from redco.analysis.stage_d_process_supervision import (
    TrainerProcessStartReceipt,
    command_sha256,
    linux_process_identity,
)
from redco.contracts import canonical_json

_CONFIG = b"frozen=true\n"
_RUNTIME = b"frozen evaluation runtime"


def _sha(character: str) -> str:
    return character * 64


def _frozen_inputs(
    *,
    endpoints: tuple[str, str, str] | None = None,
) -> tuple[bytes, bytes, bytes]:
    plan = StageDEvaluationPlan(
        tasks=(StageDEvaluationTask("heldout-1", 9101),),
        reward_min=0.0,
        reward_max=1.0,
        success_reward_threshold=0.5,
    ).to_bytes()
    checkpoints = {
        arm: EvaluationCheckpointBinding(arm, _sha(str(index + 1)), _sha("a"), _sha("b"))
        for index, arm in enumerate(("stock", "branch-global", "local"))
    }
    programs = tuple(
        EvaluationProgramBinding(
            arm=arm,
            role=role,
            absolute_executable="/opt/redco/.venv/bin/python",
            executable_sha256=_sha("c"),
            argv=(
                "/opt/redco/.venv/bin/python",
                f"{role}.py",
                arm,
                *([f"/opt/redco/checkpoints/{arm}"] if role == "server" else []),
            ),
            working_directory="/opt/redco",
            checkpoint_root=f"/opt/redco/checkpoints/{arm}",
            environment=(("PYTHONHASHSEED", "0"),),
            source_sha256s=((f"{role}.py", _sha("d")),),
            checkpoint_manifest_sha256=checkpoints[arm].checkpoint_manifest_sha256,
            post_model_sha256=checkpoints[arm].post_model_sha256,
            reload_evidence_sha256=checkpoints[arm].reload_evidence_sha256,
            endpoint=(
                f"http://127.0.0.1:{8100 + arm_index}"
                if endpoints is None
                else endpoints[arm_index]
            ),
            gpu_assignment=(arm_index,),
            cache_namespace=f"heldout-{arm}",
        )
        for arm_index, arm in enumerate(("stock", "branch-global", "local"))
        for role in ("server", "client")
    )
    manifest = StageDEvaluationExecutionManifest(
        evaluation_ledger_id=_sha("6"),
        protocol_manifest_sha256=_sha("e"),
        trainer_ledger_head_sha256=_sha("f"),
        trainer_record_count=41,
        heldout_eval_config_sha256=hashlib.sha256(_CONFIG).hexdigest(),
        evaluation_plan_sha256=hashlib.sha256(plan).hexdigest(),
        decision_rule_sha256=_sha("1"),
        runtime_entrypoints=(
            EvaluationRuntimeEntrypoint(
                "task_runner",
                "task_runtime.py",
                "task_runtime",
                "run_task",
                "redco-stage-d-worker-ipc-v1",
                _sha("2"),
            ),
            EvaluationRuntimeEntrypoint(
                "scorer", "scorer.py", "scorer", "score", "redco-stage-d-scorer-v1", _sha("3")
            ),
            EvaluationRuntimeEntrypoint(
                "request_serializer",
                "serializer.py",
                "serializer",
                "serialize",
                "redco-stage-d-request-serializer-v1",
                _sha("4"),
            ),
        ),
        runtime_worker_image="python@sha256:" + "5" * 64,
        runtime_bundle_path="/opt/redco/runtime.zip",
        runtime_bundle_sha256=hashlib.sha256(_RUNTIME).hexdigest(),
        container_runtime_executable="/usr/bin/docker",
        container_runtime_executable_sha256=_sha("7"),
        supervisor_limits=EvaluationSupervisorLimits(
            "/opt/redco/evaluation-control",
            "/opt/redco/evaluation-logs",
            "/sys/fs/cgroup",
            7200,
            30,
            30,
            5,
            50,
            1048576,
        ),
        max_server_launches_per_arm=2,
        max_client_launches_per_arm=2,
        server_replacement_policy="before-first-dispatch-only-v1",
        programs=programs,
        schedule=tuple(
            EvaluationScheduleUnit(index, arm, 0, "heldout-1", 9101)
            for index, arm in enumerate(("stock", "branch-global", "local"))
        ),
    )
    authorization = StageDEvaluationAuthorization(
        handoff_training_adoption_record_sha256=_sha("7"),
        campaign_manifest_sha256=_sha("5"),
        protocol_manifest_sha256=manifest.protocol_manifest_sha256,
        trainer_ledger_head_sha256=manifest.trainer_ledger_head_sha256,
        trainer_record_count=manifest.trainer_record_count,
        heldout_eval_config_sha256=manifest.heldout_eval_config_sha256,
        evaluation_plan_sha256=manifest.evaluation_plan_sha256,
        execution_manifest_sha256=manifest.manifest_sha256,
        checkpoints=tuple(checkpoints.values()),
    )
    return authorization.to_bytes(), manifest.to_bytes(), plan


def _receipt(
    arm: str,
    role: str,
    launch_record_sha256: str,
    *,
    epoch: int = 0,
    suffix: str = "test",
    ledger: StageDEvaluationLedger | None = None,
) -> bytes:
    actuator_pid = os.getpid() + (2 if suffix == "different" else 1)
    actuation_attempt_id = hashlib.sha256(f"{arm}:{role}:{epoch}:{suffix}".encode()).hexdigest()[
        :32
    ]
    cgroup_root = Path(
        "/sys/fs/cgroup" if ledger is None else ledger.manifest.supervisor_limits.cgroup_root
    )
    cgroup = cgroup_root / (f"redco-{launch_record_sha256[:24]}-{actuator_pid}")
    ledger_id = _sha("6") if ledger is None else ledger.manifest.evaluation_ledger_id
    control_root = Path(
        "/opt/redco/evaluation-control"
        if ledger is None
        else ledger.manifest.supervisor_limits.control_root
    )
    stop_path = control_root / ledger_id / actuation_attempt_id / "stop-request.json"
    return ActuatedProcessReceipt(
        arm=arm,
        role=role,
        epoch=epoch,
        launch_capability_sha256=launch_record_sha256,
        actuation_attempt_id=actuation_attempt_id,
        pid=os.getpid(),
        boot_id="test-boot",
        process_start_ticks="1",
        process_group_id=os.getpid(),
        process_session_id=actuator_pid,
        cgroup_path=cgroup.as_posix(),
        cgroup_device_id=0,
        cgroup_inode=1,
        cgroup_lines=("0::/redco-test",),
        supervisor=EvaluationSupervisorIdentity(os.getpid(), "test-boot", "1"),
        actuator_pid=actuator_pid,
        actuator_boot_id="test-boot",
        actuator_start_ticks="1",
        command_sha256=command_sha256(
            (
                "/opt/redco/.venv/bin/python",
                f"{role}.py",
                arm,
                *([f"/opt/redco/checkpoints/{arm}"] if role == "server" else []),
            )
        ),
        environment_manifest_sha256=evaluation_environment_sha256((("PYTHONHASHSEED", "0"),)),
        stop_request_path=stop_path.as_posix(),
    ).to_bytes()


def _reserve_receipt_attempt(
    ledger: StageDEvaluationLedger,
    receipt_bytes: bytes,
) -> ActuatedProcessReceipt:
    receipt = ActuatedProcessReceipt.from_bytes(receipt_bytes)
    ledger.reserve_actuation_attempt(
        actuation_attempt_id=receipt.actuation_attempt_id,
        arm=receipt.arm,
        role=receipt.role,
        epoch=receipt.epoch,
        launch_record_sha256=receipt.launch_capability_sha256,
        supervisor=receipt.supervisor,
    )
    return receipt


def _finish_claimed_attempt(
    ledger: StageDEvaluationLedger,
    receipt_bytes: bytes,
) -> None:
    receipt = ActuatedProcessReceipt.from_bytes(receipt_bytes)
    ledger.finish_actuation_attempt(
        actuation_attempt_id=receipt.actuation_attempt_id,
        disposition="claimed",
        process_receipt_bytes=receipt_bytes,
    )


def _claim_server(
    ledger: StageDEvaluationLedger,
    launch,
    receipt_bytes: bytes,
) -> str:
    _reserve_receipt_attempt(ledger, receipt_bytes)
    result = ledger.claim_server(launch, receipt_bytes)
    _finish_claimed_attempt(ledger, receipt_bytes)
    return result


def _claim_client(
    ledger: StageDEvaluationLedger,
    launch,
    receipt_bytes: bytes,
):
    _reserve_receipt_attempt(ledger, receipt_bytes)
    result = ledger.claim_client(launch, receipt_bytes)
    _finish_claimed_attempt(ledger, receipt_bytes)
    return result


def _server_observation(
    ledger: StageDEvaluationLedger,
    launch,
    receipt_bytes: bytes,
) -> bytes:
    receipt = ActuatedProcessReceipt.from_bytes(receipt_bytes)
    program = ledger.manifest.program(launch.arm, "server")
    return EvaluationServerProcessObservation(
        arm=launch.arm,
        server_epoch=launch.epoch,
        launch_record_sha256=launch.launch_record_sha256,
        process_receipt_sha256=receipt.receipt_sha256,
        program_binding_sha256=program.binding_sha256,
        pid=receipt.pid,
        boot_id=receipt.boot_id,
        process_start_ticks=receipt.process_start_ticks,
        executable_path=program.absolute_executable,
        executable_sha256=program.executable_sha256,
        argv=program.argv,
        working_directory=program.working_directory,
        environment_manifest_sha256=evaluation_environment_sha256(program.environment),
        cgroup_lines=receipt.cgroup_lines,
        checkpoint_root=f"/opt/redco/checkpoints/{launch.arm}",
        checkpoint_manifest_sha256=program.checkpoint_manifest_sha256,
        endpoint=program.endpoint,
        cache_namespace=program.cache_namespace,
    ).to_bytes()


@pytest.mark.skipif(os.name == "nt", reason="Linux process identity requires /proc")
def test_generic_trainer_receipt_cannot_claim_an_evaluation_launch(
    tmp_path: Path,
) -> None:
    authorization, manifest_bytes, plan = _frozen_inputs()
    ledger_id = StageDEvaluationExecutionManifest.from_bytes(manifest_bytes).evaluation_ledger_id
    ledger = StageDEvaluationLedger.create(
        tmp_path / ledger_id,
        authorization_bytes=authorization,
        execution_manifest_bytes=manifest_bytes,
        evaluation_plan_bytes=plan,
        runtime_bundle_bytes=_RUNTIME,
    )
    boot_id, start_ticks = linux_process_identity(os.getpid())
    server_launch = ledger.reserve_server_launch("stock")
    server_program = ledger.manifest.program("stock", "server")
    server_receipt = TrainerProcessStartReceipt(
        arm="stock",
        launch_id="stage-d-evaluation-server-stock-real",
        pid=os.getpid(),
        boot_id=boot_id,
        process_start_ticks=start_ticks,
        command_sha256=command_sha256(server_program.argv),
        environment_manifest_sha256=evaluation_environment_sha256(server_program.environment),
    ).to_bytes()
    with pytest.raises(ValueError, match="actuated process receipt fields differ"):
        ledger.claim_server(server_launch, server_receipt)


def _transport(endpoint: str, body_sha256: str) -> bytes:
    return canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-evaluation-transport-v1",
            "method": "POST",
            "url": f"{endpoint}/v1/chat/completions",
            "headers": {"content-type": "application/json"},
            "body_sha256": body_sha256,
            "timeout_seconds": 30.0,
            "transport_retries": 0,
        }
    )


def _new_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    endpoints: tuple[str, str, str] | None = None,
) -> StageDEvaluationLedger:
    authorization, manifest, plan = _frozen_inputs(endpoints=endpoints)
    ledger_id = StageDEvaluationExecutionManifest.from_bytes(manifest).evaluation_ledger_id
    monkeypatch.setattr(
        StageDEvaluationLedger,
        "_verify_process_receipt",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(ActuatedProcessReceipt, "is_same_live_process", lambda _self: True)
    monkeypatch.setattr(ActuatedProcessReceipt, "is_same_live_tree", lambda _self: True)
    return StageDEvaluationLedger.create(
        tmp_path / ledger_id,
        authorization_bytes=authorization,
        execution_manifest_bytes=manifest,
        evaluation_plan_bytes=plan,
        runtime_bundle_bytes=_RUNTIME,
    )


def test_creation_is_idempotent_only_for_exact_frozen_inputs(tmp_path: Path) -> None:
    authorization, manifest, plan = _frozen_inputs()
    root = tmp_path / StageDEvaluationExecutionManifest.from_bytes(manifest).evaluation_ledger_id
    first = StageDEvaluationLedger.create(
        root,
        authorization_bytes=authorization,
        execution_manifest_bytes=manifest,
        evaluation_plan_bytes=plan,
        runtime_bundle_bytes=_RUNTIME,
    )
    first_snapshot = first.inspect()
    assert first_snapshot.authorization_sha256 == hashlib.sha256(authorization).hexdigest()
    assert StageDEvaluationLedger(root).inspect() == first_snapshot
    second = StageDEvaluationLedger.create(
        root,
        authorization_bytes=authorization,
        execution_manifest_bytes=manifest,
        evaluation_plan_bytes=plan,
        runtime_bundle_bytes=_RUNTIME,
    )
    assert second.inspect() == first_snapshot
    with pytest.raises(ValueError):
        StageDEvaluationLedger.create(
            root,
            authorization_bytes=authorization,
            execution_manifest_bytes=manifest,
            evaluation_plan_bytes=plan + b"\n",
            runtime_bundle_bytes=_RUNTIME,
        )


def _start_server(ledger: StageDEvaluationLedger, arm: str) -> None:
    launch = ledger.reserve_server_launch(arm)
    server_receipt = _receipt(
        arm,
        "server",
        launch.launch_record_sha256,
        epoch=launch.epoch,
        ledger=ledger,
    )
    _claim_server(ledger, launch, server_receipt)
    ledger.attest_server(
        launch=launch,
        process_receipt_bytes=server_receipt,
        process_observation_bytes=_server_observation(ledger, launch, server_receipt),
        probe_response_bytes=f"probe-{arm}".encode(),
    )


def _start_arm(ledger: StageDEvaluationLedger, arm: str):
    _start_server(ledger, arm)
    client_launch = ledger.reserve_client_launch(arm)
    receipt = _receipt(
        arm,
        "client",
        client_launch.launch_record_sha256,
        epoch=client_launch.epoch,
        ledger=ledger,
    )
    session = _claim_client(
        ledger,
        client_launch,
        receipt,
    )
    return ledger.reserve_next_task(session=session), session


def _reserve_call(
    ledger: StageDEvaluationLedger,
    task,
    session,
    *,
    call_ordinal: int | None = None,
    content: str = "frozen",
):
    body = canonical_json({"messages": [{"role": "user", "content": content}]})
    endpoint = ledger.manifest.program(task.unit.arm, "server").endpoint
    return ledger.reserve_call(
        task,
        session=session,
        event_address_bytes=b"root/0",
        seed=9201,
        cache_salt=f"{task.unit.arm}-0",
        request_body_bytes=body,
        transport_bytes=_transport(endpoint, hashlib.sha256(body).hexdigest()),
        call_ordinal=call_ordinal,
    )


def _finish_task(ledger: StageDEvaluationLedger, task, session) -> None:
    call = _reserve_call(ledger, task, session)
    dispatch = ledger.authorize_dispatch(call, session=session)
    ledger.record_response(
        dispatch,
        session=session,
        status_code=200,
        headers=(("content-type", "application/json"),),
        raw_response_bytes=b'{"choices":[]}',
    )
    ledger.finalize_call(
        call,
        session=session,
        parsed_response_bytes=b"answer",
        prompt_tokens=7,
        completion_tokens=2,
        wall_seconds=0.2,
        gpu_seconds=0.1,
        finish_kind="stop",
    )
    ledger.complete_task(
        task,
        session=session,
        terminal_result_bytes=b"terminal answer",
        scorer_evidence_bytes=b"exact score",
        reward=1.0,
    )


def test_full_evaluation_ledger_seals_exact_three_arm_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    for arm in ("stock", "branch-global", "local"):
        task, session = _start_arm(ledger, arm)
        _finish_task(ledger, task, session)
        ledger.complete_arm(arm)
    ledger.seal()
    snapshot = ledger.inspect()
    assert snapshot.sealed
    assert snapshot.terminal_status == "sealed"
    assert len(snapshot.tasks) == 3
    assert tuple(arm for arm, _ in snapshot.arm_completions) == (
        "stock",
        "branch-global",
        "local",
    )
    assert dict(snapshot.arm_completions) != dict(snapshot.arm_metrics)


def test_production_completion_requires_and_binds_sealed_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    authorization = StageDEvaluationAuthorization.from_bytes(ledger.authorization_bytes)
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_bytes(ledger.authorization_bytes)
    config_path = tmp_path / "eval.toml"
    config_path.write_bytes(_CONFIG)
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes((ledger.inputs / "evaluation-plan.json").read_bytes())

    class _State:
        def __init__(self, binding: EvaluationCheckpointBinding) -> None:
            self.checkpoint_sha256 = binding.checkpoint_manifest_sha256
            self.post_model_sha256 = binding.post_model_sha256
            self.reload_evidence_sha256 = binding.reload_evidence_sha256

    class _Snapshot:
        campaign_manifest_sha256 = authorization.campaign_manifest_sha256
        protocol_manifest_sha256 = authorization.protocol_manifest_sha256
        head_sha256 = authorization.trainer_ledger_head_sha256
        record_count = authorization.trainer_record_count

        @staticmethod
        def state(arm: str):
            return _State(next(item for item in authorization.checkpoints if item.arm == arm))

    class _TrainerLedger:
        @staticmethod
        def inspect():
            return _Snapshot()

    with pytest.raises(RuntimeError, match="not sealed"):
        commit_sealed_heldout_evaluation(
            authorization_path=authorization_path,
            trainer_ledger=_TrainerLedger(),  # type: ignore[arg-type]
            evaluation_ledger=ledger,
            heldout_eval_config_path=config_path,
            evaluation_plan_path=plan_path,
            retained_evidence_root=tmp_path / "retained",
            destination=tmp_path / "completion.json",
        )
    for arm in ("stock", "branch-global", "local"):
        task, session = _start_arm(ledger, arm)
        _finish_task(ledger, task, session)
        ledger.complete_arm(arm)
    ledger.seal()
    completion_path = tmp_path / "completion.json"
    completion = commit_sealed_heldout_evaluation(
        authorization_path=authorization_path,
        trainer_ledger=_TrainerLedger(),  # type: ignore[arg-type]
        evaluation_ledger=ledger,
        heldout_eval_config_path=config_path,
        evaluation_plan_path=plan_path,
        retained_evidence_root=tmp_path / "retained",
        destination=completion_path,
    )
    assert StageDSealedEvaluationCompletion.from_bytes(completion_path.read_bytes()) == completion
    completion.verify_ledger(ledger)


def test_dispatched_without_response_is_ambiguous_and_never_redispatched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    task, session = _start_arm(ledger, "stock")
    call = _reserve_call(ledger, task, session)
    dispatch = ledger.authorize_dispatch(call, session=session)
    assert ledger.inspect().terminal_status == "ambiguous-dispatch"
    with pytest.raises(RuntimeError, match="already dispatch-authorized"):
        ledger.authorize_dispatch(call, session=session)
    ledger.record_response(
        dispatch,
        session=session,
        status_code=200,
        headers=(),
        raw_response_bytes=b"witnessed",
    )
    assert ledger.inspect().terminal_status == "active"


def test_finalized_call_replays_without_another_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    task, session = _start_arm(ledger, "stock")
    call = _reserve_call(ledger, task, session)
    dispatch = ledger.authorize_dispatch(call, session=session)
    ledger.record_response(
        dispatch,
        session=session,
        status_code=200,
        headers=(),
        raw_response_bytes=b"raw",
    )
    ledger.finalize_call(
        call,
        session=session,
        parsed_response_bytes=b"parsed",
        prompt_tokens=1,
        completion_tokens=1,
        wall_seconds=0.1,
        gpu_seconds=0.0,
        finish_kind="stop",
    )
    replay = _reserve_call(ledger, task, session, call_ordinal=0)
    assert replay == call
    assert ledger.finalized_response_bytes(replay) == b"parsed"
    with pytest.raises(RuntimeError, match="differs from its transcript"):
        _reserve_call(ledger, task, session, call_ordinal=0, content="changed")


def test_dead_client_reclaims_open_task_before_any_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    task, old_session = _start_arm(ledger, "stock")
    call = _reserve_call(ledger, task, old_session)

    def liveness(receipt: ActuatedProcessReceipt) -> bool:
        return receipt.epoch == 1 or receipt.role == "server"

    monkeypatch.setattr(
        ActuatedProcessReceipt,
        "is_same_live_process",
        liveness,
    )
    assert ledger.inspect().terminal_status == "orphaned-open-task"
    launch = ledger.reserve_client_launch("stock")
    new_session = _claim_client(
        ledger,
        launch,
        _receipt(
            "stock",
            "client",
            launch.launch_record_sha256,
            epoch=launch.epoch,
            suffix="replacement",
            ledger=ledger,
        ),
    )
    resumed_task = ledger.resume_open_task(session=new_session)
    resumed_call = ledger.resume_reserved_call(resumed_task, session=new_session)
    assert resumed_task == replace(task, client_epoch=1)
    assert resumed_call == call
    assert ledger.authorize_dispatch(resumed_call, session=new_session).call == call


def test_dead_client_cannot_reclaim_after_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    task, old_session = _start_arm(ledger, "stock")
    call = _reserve_call(ledger, task, old_session)
    ledger.authorize_dispatch(call, session=old_session)
    monkeypatch.setattr(
        ActuatedProcessReceipt,
        "is_same_live_process",
        lambda receipt: receipt.epoch == 1 or receipt.role == "server",
    )
    assert ledger.inspect().terminal_status == "ambiguous-dispatch"
    with pytest.raises(RuntimeError, match="ambiguous"):
        ledger.reserve_client_launch("stock")


def test_dead_client_reclaims_finalized_transcript_without_redispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    task, old_session = _start_arm(ledger, "stock")
    call = _reserve_call(ledger, task, old_session)
    dispatch = ledger.authorize_dispatch(call, session=old_session)
    ledger.record_response(
        dispatch,
        session=old_session,
        status_code=200,
        headers=(),
        raw_response_bytes=b"raw",
    )
    ledger.finalize_call(
        call,
        session=old_session,
        parsed_response_bytes=b"parsed",
        prompt_tokens=1,
        completion_tokens=1,
        wall_seconds=0.1,
        gpu_seconds=0.0,
        finish_kind="stop",
    )
    monkeypatch.setattr(
        ActuatedProcessReceipt,
        "is_same_live_process",
        lambda receipt: receipt.epoch == 1 or receipt.role == "server",
    )
    launch = ledger.reserve_client_launch("stock")
    new_session = _claim_client(
        ledger,
        launch,
        _receipt(
            "stock",
            "client",
            launch.launch_record_sha256,
            epoch=launch.epoch,
            suffix="replacement",
            ledger=ledger,
        ),
    )
    resumed = ledger.resume_open_task(session=new_session)
    replay = _reserve_call(ledger, resumed, new_session, call_ordinal=0)
    assert replay == call
    assert ledger.finalized_response_bytes(replay) == b"parsed"


def test_dead_server_cannot_authorize_a_policy_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    task, session = _start_arm(ledger, "stock")

    def reject_dead_server(
        _self: StageDEvaluationLedger,
        _receipt: ActuatedProcessReceipt,
        program: EvaluationProgramBinding,
        *,
        require_current: bool,
    ) -> None:
        assert require_current is (program.role == "client")
        if program.role == "server":
            raise ValueError("evaluation process receipt is stale or belongs elsewhere")

    monkeypatch.setattr(
        StageDEvaluationLedger,
        "_verify_process_receipt",
        reject_dead_server,
    )
    with pytest.raises(ValueError, match="stale or belongs elsewhere"):
        _reserve_call(ledger, task, session)
    assert not ledger.inspect().tasks[0].calls


def test_inspection_marks_current_task_orphaned_when_its_server_dies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    _start_arm(ledger, "stock")

    def liveness(receipt: ActuatedProcessReceipt) -> bool:
        return receipt.role != "server"

    monkeypatch.setattr(ActuatedProcessReceipt, "is_same_live_process", liveness)
    assert ledger.inspect().terminal_status == "orphaned-server"


def test_dead_server_is_replaced_once_before_first_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    task, session = _start_arm(ledger, "stock")
    monkeypatch.setattr(
        ActuatedProcessReceipt,
        "is_same_live_process",
        lambda receipt: receipt.epoch == 1 or receipt.role == "client",
    )
    assert ledger.inspect().terminal_status == "orphaned-server"
    replacement = ledger.reserve_server_launch("stock")
    assert replacement.epoch == 1
    replacement_receipt = _receipt(
        "stock",
        "server",
        replacement.launch_record_sha256,
        epoch=replacement.epoch,
        suffix="replacement",
        ledger=ledger,
    )
    _claim_server(ledger, replacement, replacement_receipt)
    ledger.attest_server(
        launch=replacement,
        process_receipt_bytes=replacement_receipt,
        process_observation_bytes=_server_observation(ledger, replacement, replacement_receipt),
        probe_response_bytes=b"replacement probe",
    )
    call = _reserve_call(ledger, task, session)
    assert ledger.authorize_dispatch(call, session=session).call == call
    monkeypatch.setattr(
        ActuatedProcessReceipt,
        "is_same_live_process",
        lambda receipt: receipt.role == "client",
    )
    with pytest.raises(RuntimeError, match=r"terminally forbidden|past its cutoff"):
        ledger.reserve_server_launch("stock")


def test_unclaimed_server_launch_reservation_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    first = ledger.reserve_server_launch("stock")
    assert ledger.reserve_server_launch("stock") == first


def test_client_launch_reservation_is_idempotent_and_single_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    _start_server(ledger, "stock")
    launch = ledger.reserve_client_launch("stock")
    assert ledger.reserve_client_launch("stock") == launch
    receipt = _receipt("stock", "client", launch.launch_record_sha256, ledger=ledger)
    session = _claim_client(ledger, launch, receipt)
    assert ledger.claim_client(launch, receipt) == session
    with pytest.raises(RuntimeError, match="already claimed"):
        ledger.claim_client(
            launch,
            _receipt(
                "stock",
                "client",
                launch.launch_record_sha256,
                suffix="different",
                ledger=ledger,
            ),
        )


def test_response_witness_is_idempotent_only_for_identical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    task, session = _start_arm(ledger, "stock")
    call = _reserve_call(ledger, task, session)
    dispatch = ledger.authorize_dispatch(call, session=session)
    arguments = {
        "status_code": 200,
        "headers": (),
        "raw_response_bytes": b"same response",
    }
    first = ledger.record_response(dispatch, session=session, **arguments)
    assert ledger.record_response(dispatch, session=session, **arguments) == first
    with pytest.raises(FileExistsError, match="durable evaluation state differs"):
        ledger.record_response(
            dispatch,
            session=session,
            **(arguments | {"raw_response_bytes": b"different"}),
        )


def test_exactly_one_concurrent_dispatch_authorization_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    task, session = _start_arm(ledger, "stock")
    call = _reserve_call(ledger, task, session)

    def authorize() -> str:
        try:
            return ledger.authorize_dispatch(call, session=session).dispatch_receipt_sha256
        except RuntimeError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: authorize(), range(2)))
    assert results.count("rejected") == 1
    assert sum(value != "rejected" for value in results) == 1


def test_forged_task_call_and_dispatch_capabilities_fail_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    task, session = _start_arm(ledger, "stock")
    with pytest.raises(RuntimeError, match="does not belong"):
        _reserve_call(ledger, replace(task, client_epoch=99), session)
    call = _reserve_call(ledger, task, session)
    forged = replace(call, request_sha256=_sha("9"))
    with pytest.raises(RuntimeError, match="capability differs"):
        ledger.authorize_dispatch(forged, session=session)
    assert not tuple(ledger.responses.iterdir())
    dispatch = ledger.authorize_dispatch(call, session=session)
    forged_dispatch = replace(dispatch, dispatch_receipt_sha256=_sha("8"))
    with pytest.raises(RuntimeError, match="durable dispatch"):
        ledger.record_response(
            forged_dispatch,
            session=session,
            status_code=200,
            headers=(),
            raw_response_bytes=b"must not publish",
        )
    assert not tuple(ledger.responses.iterdir())


def test_inspection_reopens_nested_evidence_and_event_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    task, session = _start_arm(ledger, "stock")
    call = _reserve_call(ledger, task, session)
    record = next(
        path for path in ledger.records.glob("*.json") if b"call_reserved" in path.read_bytes()
    )
    import json

    event = json.loads(record.read_bytes())["event"]
    (ledger.evidence.root / event["event_address_sha256"]).unlink()
    with pytest.raises(ValueError, match="evidence is absent"):
        ledger.inspect()
    assert call.call_id


def test_transport_rejects_secret_headers_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    task, session = _start_arm(ledger, "stock")
    body = b"{}"
    endpoint = ledger.manifest.program("stock", "server").endpoint
    transport = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-evaluation-transport-v1",
            "method": "POST",
            "url": f"{endpoint}/v1/chat/completions",
            "headers": {"authorization": "secret"},
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "timeout_seconds": 30.0,
            "transport_retries": 1,
        }
    )
    with pytest.raises(ValueError, match="frozen local POST"):
        ledger.reserve_call(
            task,
            session=session,
            event_address_bytes=b"root/0",
            seed=1,
            cache_salt="stock-0",
            request_body_bytes=body,
            transport_bytes=transport,
        )


class _CountingHandler(BaseHTTPRequestHandler):
    calls = 0
    bodies: ClassVar[list[bytes]] = []

    def do_POST(self) -> None:
        type(self).calls += 1
        body = self.rfile.read(int(self.headers["content-length"]))
        type(self).bodies.append(body)
        response = b'{"choices":[{"message":{"content":"answer"}}]}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, _format: str, *_arguments: object) -> None:
        return


@pytest.mark.parametrize("event_address", [b"root/0", b"root/0/child/0"])
@pytest.mark.parametrize(
    "crash_stage",
    [
        None,
        "after-call-reserved",
        "after-dispatch-durable",
        "after-response-read",
        "after-response-witnessed",
    ],
)
def test_real_http_dispatch_is_never_duplicated_across_crash_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    event_address: bytes,
    crash_stage: str | None,
) -> None:
    _CountingHandler.calls = 0
    _CountingHandler.bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CountingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    endpoints = (
        f"http://127.0.0.1:{port}",
        "http://127.0.0.1:61001",
        "http://127.0.0.1:61002",
    )
    ledger = _new_ledger(tmp_path, monkeypatch, endpoints=endpoints)
    task, session = _start_arm(ledger, "stock")
    request = b'{"messages":[{"role":"user","content":"frozen"}]}'
    dispatch = partial(
        dispatch_local_http_once,
        ledger=ledger,
        task=task,
        session=session,
        event_address_bytes=event_address,
        seed=9201,
        cache_salt="stock-0",
        request_body_bytes=request,
        timeout_seconds=5.0,
    )

    def crash(stage: str) -> None:
        if stage == crash_stage:
            raise RuntimeError(f"simulated {stage}")

    try:
        if crash_stage is None:
            result = dispatch()
            assert result.status_code == 200
        else:
            with pytest.raises(RuntimeError, match=f"simulated {crash_stage}"):
                dispatch(fault_hook=crash)
        first_count = _CountingHandler.calls
        with pytest.raises(
            RuntimeError,
            match=r"unfinished policy call|does not belong to the open task",
        ):
            dispatch()
        assert _CountingHandler.calls == first_count
        before_network = {"after-call-reserved", "after-dispatch-durable"}
        assert first_count == (0 if crash_stage in before_network else 1)
        assert _CountingHandler.bodies in ([], [request])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    ("stage", "published"),
    (
        ("after-evaluation-ledger-pending-fsync", False),
        ("after-evaluation-ledger-link", True),
    ),
)
def test_atomic_publication_is_exactly_restartable_after_each_crash_point(
    tmp_path: Path,
    stage: str,
    published: bool,
) -> None:
    destination = tmp_path / "durable.json"
    value = b"durable evaluation evidence"

    def crash(observed: str, _path: Path) -> None:
        if observed == stage:
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        atomic_publish(destination, value, fault_hook=crash)
    assert destination.exists() is published
    if published:
        assert destination.read_bytes() == value
    atomic_publish(destination, value)
    assert destination.read_bytes() == value
