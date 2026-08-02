from __future__ import annotations

from dataclasses import replace

import pytest

from redco.analysis.stage_d_evaluation_contracts import (
    EvaluationProgramBinding,
    EvaluationRuntimeEntrypoint,
    EvaluationScheduleUnit,
    EvaluationSupervisorLimits,
    StageDEvaluationExecutionManifest,
)


def _program(arm: str, role: str) -> EvaluationProgramBinding:
    return EvaluationProgramBinding(
        arm=arm,  # type: ignore[arg-type]
        role=role,  # type: ignore[arg-type]
        absolute_executable="/opt/redco/.venv/bin/python",
        executable_sha256="1" * 64,
        argv=(
            "/opt/redco/.venv/bin/python",
            "/opt/redco/evaluator.py",
            arm,
            *([f"/opt/redco/checkpoints/{arm}"] if role == "server" else []),
        ),
        working_directory="/opt/redco",
        checkpoint_root=f"/opt/redco/checkpoints/{arm}",
        environment=(
            ("CUDA_VISIBLE_DEVICES", "0,1"),
            ("PYTHONHASHSEED", "0"),
            ("PYTHONNOUSERSITE", "1"),
        ),
        source_sha256s=(("evaluator.py", "2" * 64), ("runtime.tar", "3" * 64)),
        checkpoint_manifest_sha256={
            "stock": "4" * 64,
            "branch-global": "5" * 64,
            "local": "6" * 64,
        }[arm],
        post_model_sha256={
            "stock": "7" * 64,
            "branch-global": "8" * 64,
            "local": "9" * 64,
        }[arm],
        reload_evidence_sha256={
            "stock": "a" * 64,
            "branch-global": "b" * 64,
            "local": "c" * 64,
        }[arm],
        endpoint=f"http://127.0.0.1:{8100 + ('stock', 'branch-global', 'local').index(arm)}",
        gpu_assignment=(0, 1),
        cache_namespace=f"stage-d-{arm}",
    )


def _manifest() -> StageDEvaluationExecutionManifest:
    programs = []
    for arm in ("stock", "branch-global", "local"):
        server = _program(arm, "server")
        programs.extend((server, replace(server, role="client")))
    schedule = tuple(
        EvaluationScheduleUnit(ordinal, arm, task_index, task_id, seed)
        for ordinal, (task_index, task_id, seed, arm) in enumerate(
            (
                (0, "task-a", 9101, "stock"),
                (1, "task-b", 9102, "stock"),
                (0, "task-a", 9101, "branch-global"),
                (1, "task-b", 9102, "branch-global"),
                (0, "task-a", 9101, "local"),
                (1, "task-b", 9102, "local"),
            )
        )
    )
    return StageDEvaluationExecutionManifest(
        evaluation_ledger_id="0" * 64,
        protocol_manifest_sha256="d" * 64,
        trainer_ledger_head_sha256="e" * 64,
        trainer_record_count=16,
        heldout_eval_config_sha256="f" * 64,
        evaluation_plan_sha256="1" * 64,
        decision_rule_sha256="2" * 64,
        runtime_entrypoints=(
            EvaluationRuntimeEntrypoint(
                "task_runner",
                "task_runtime.py",
                "task_runtime",
                "run_task",
                "redco-stage-d-worker-ipc-v1",
                "3" * 64,
            ),
            EvaluationRuntimeEntrypoint(
                "scorer", "scorer.py", "scorer", "score", "redco-stage-d-scorer-v1", "4" * 64
            ),
            EvaluationRuntimeEntrypoint(
                "request_serializer",
                "serializer.py",
                "serializer",
                "serialize",
                "redco-stage-d-request-serializer-v1",
                "5" * 64,
            ),
        ),
        runtime_worker_image="python@sha256:" + "7" * 64,
        runtime_bundle_path="/opt/redco/runtime.zip",
        runtime_bundle_sha256="6" * 64,
        container_runtime_executable="/usr/bin/docker",
        container_runtime_executable_sha256="7" * 64,
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
        programs=tuple(programs),
        schedule=schedule,
    )


def test_execution_manifest_roundtrips_and_binds_exact_cross_arm_roster() -> None:
    manifest = _manifest()
    assert StageDEvaluationExecutionManifest.from_bytes(manifest.to_bytes()) == manifest
    changed = list(manifest.schedule)
    changed[-1] = replace(changed[-1], seed=9999)
    with pytest.raises(ValueError, match="exact task roster"):
        replace(manifest, schedule=tuple(changed))


@pytest.mark.parametrize("name", ["HF_TOKEN", "OPENAI_API_KEY", "AUTH_TOKEN"])
def test_execution_manifest_rejects_secret_environment_names(name: str) -> None:
    program = _program("stock", "server")
    with pytest.raises(ValueError, match="forbidden"):
        replace(program, environment=((name, "secret"),))


def test_execution_manifest_requires_direct_absolute_executable() -> None:
    program = _program("stock", "server")
    with pytest.raises(ValueError, match="absolute path"):
        replace(program, absolute_executable="uv", argv=("uv", "run", "evaluator"))


def test_execution_manifest_binds_one_checkpoint_root_per_arm() -> None:
    manifest = _manifest()
    programs = list(manifest.programs)
    client_index = next(
        index
        for index, program in enumerate(programs)
        if (program.arm, program.role) == ("stock", "client")
    )
    programs[client_index] = replace(
        programs[client_index],
        checkpoint_root="/opt/redco/checkpoints/other",
    )
    with pytest.raises(ValueError, match="policy bindings differ"):
        replace(manifest, programs=tuple(programs))


def test_server_program_requires_checkpoint_in_exact_command() -> None:
    program = _program("stock", "server")
    with pytest.raises(ValueError, match="checkpoint exactly once"):
        replace(program, argv=program.argv[:-1])
