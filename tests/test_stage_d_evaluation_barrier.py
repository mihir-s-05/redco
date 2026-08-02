from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from redco.analysis.stage_d_evaluation_barrier import (
    StageDEvaluationAuthorization,
    StageDEvaluationCompletion,
    StageDEvaluationPlan,
    StageDEvaluationTask,
    StageDHeldoutMetrics,
    _authorize_heldout_evaluation_from_live_ledger,
    commit_heldout_evaluation,
)
from redco.analysis.stage_d_evaluation_contracts import (
    EvaluationProgramBinding,
    EvaluationRuntimeEntrypoint,
    EvaluationScheduleUnit,
    EvaluationSupervisorLimits,
    StageDEvaluationExecutionManifest,
)
from redco.analysis.stage_d_protocol_manifest import (
    StageDPolicyIdentity,
    StageDProtocolManifest,
)
from redco.analysis.stage_d_trainer_supervisor import ArmRunState, TrainerRunSnapshot
from redco.contracts import canonical_json


def _sha(character: str) -> str:
    return character * 64


def test_legacy_v3_authorization_is_rejected() -> None:
    payload = {
        "schema_version": 3,
        "domain": "redco-stage-d-evaluation-authorization-v3",
        "handoff_evaluation_grant_sha256": _sha("1"),
        "campaign_manifest_sha256": _sha("2"),
        "protocol_manifest_sha256": _sha("3"),
        "trainer_ledger_head_sha256": _sha("4"),
        "trainer_record_count": 1,
        "heldout_eval_config_sha256": _sha("5"),
        "evaluation_plan_sha256": _sha("6"),
        "execution_manifest_sha256": _sha("7"),
        "checkpoints": [],
    }
    with pytest.raises(ValueError, match="fields differ"):
        StageDEvaluationAuthorization.from_bytes(canonical_json(payload))


class _Ledger:
    def __init__(self, snapshot: TrainerRunSnapshot) -> None:
        self.snapshot = snapshot

    def inspect(self) -> TrainerRunSnapshot:
        return self.snapshot


def _snapshot(
    *, complete: bool = True, protocol_manifest_sha256: str | None = None
) -> TrainerRunSnapshot:
    states = tuple(
        ArmRunState(
            arm=arm,
            optimizer_started=complete,
            optimizer_completed=complete,
            post_model_sha256=_sha(str(index + 1)) if complete else None,
            model_changed=True if complete else None,
            checkpoint_committed=complete,
            checkpoint_sha256=_sha(str(index + 4)) if complete else None,
            metrics_sha256=_sha(str(index + 7)) if complete else None,
            reload_evidence_sha256=_sha(chr(ord("a") + index)) if complete else None,
        )
        for index, arm in enumerate(("stock", "branch-global", "local"))
    )
    return TrainerRunSnapshot(
        campaign_manifest_sha256=_sha("a"),
        protocol_manifest_sha256=protocol_manifest_sha256 or _sha("b"),
        shared_initialization_manifest_sha256=_sha("c"),
        expected_pre_model_sha256=_sha("d"),
        expected_base_model_manifest_sha256=_sha("e"),
        reload_probe_sha256=_sha("f"),
        trainer_step=1,
        arm_order=("stock", "branch-global", "local"),
        batch_identities=(
            ("branch-global", _sha("2")),
            ("local", _sha("3")),
            ("stock", _sha("1")),
        ),
        trainer_config_sha256s=(
            ("branch-global", _sha("5")),
            ("local", _sha("6")),
            ("stock", _sha("4")),
        ),
        process_command_sha256s=(
            ("branch-global", _sha("b")),
            ("local", _sha("c")),
            ("stock", _sha("a")),
        ),
        process_environment_sha256s=(
            ("branch-global", _sha("8")),
            ("local", _sha("9")),
            ("stock", _sha("7")),
        ),
        states=states,
        head_sha256=_sha("9"),
        record_count=16,
    )


def _plan_bytes() -> bytes:
    return StageDEvaluationPlan(
        tasks=(StageDEvaluationTask("task-1", 9101),),
        reward_min=0.0,
        reward_max=1.0,
        success_reward_threshold=1.0,
    ).to_bytes()


def _protocol_bytes(config: bytes, plan: bytes) -> bytes:
    protocol = StageDProtocolManifest(
        preregistration_sha256=_sha("1"),
        dependency_stack_sha256=_sha("2"),
        genesis_config_sha256=_sha("3"),
        master_seed_sha256=_sha("4"),
        source_sha256=_sha("5"),
        runtime_sha256=_sha("6"),
        source_eval_config_sha256=_sha("7"),
        scientific_eval_config_sha256=_sha("8"),
        heldout_eval_config_sha256=hashlib.sha256(config).hexdigest(),
        collection_plan_sha256=_sha("9"),
        evaluation_plan_sha256=hashlib.sha256(plan).hexdigest(),
        decision_rule_sha256=_sha("a"),
        support_rules_sha256=_sha("support"),
        reload_probe_sha256=_sha("b"),
        shared_initialization_sha256=_sha("c"),
        objective_authorization_sha256=_sha("d"),
        objective_binding_sha256s=(
            ("stock", _sha("1")),
            ("branch-global", _sha("2")),
            ("local", _sha("3")),
        ),
        trainer_config_sha256s=(
            ("stock", _sha("4")),
            ("branch-global", _sha("5")),
            ("local", _sha("6")),
        ),
        policy_identity=StageDPolicyIdentity(
            checkpoint_id="checkpoint",
            base_model_manifest_sha256=_sha("1"),
            adapter_manifest_sha256=None,
            tokenizer_manifest_sha256=_sha("2"),
            renderer_manifest_sha256=_sha("3"),
            sampler_conformance_manifest_sha256=_sha("4"),
            resolved_agent_sampling_law_sha256=_sha("5"),
            resolved_train_client_sha256=_sha("6"),
        ),
        arm_order=("stock", "branch-global", "local"),
        branch_global_scope="within-source-group-all-target-branches-v1",
        trainer_step=1,
        seq_len=64,
    )
    return protocol.to_bytes()


def _frozen_files(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    config = tmp_path / "eval.toml"
    config.write_bytes(b"frozen=true\n")
    plan = tmp_path / "plan.json"
    plan.write_bytes(_plan_bytes())
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes(_protocol_bytes(config.read_bytes(), plan.read_bytes()))
    return config, plan, protocol, hashlib.sha256(protocol.read_bytes()).hexdigest()


def _execution_manifest_bytes(
    snapshot: TrainerRunSnapshot,
    *,
    protocol_sha256: str,
    config: Path,
    plan: Path,
) -> bytes:
    programs = []
    for arm in ("stock", "branch-global", "local"):
        state = snapshot.state(arm)
        for role in ("server", "client"):
            programs.append(
                EvaluationProgramBinding(
                    arm=arm,
                    role=role,
                    absolute_executable="/opt/redco/.venv/bin/python",
                    executable_sha256=_sha("1"),
                    argv=(
                        "/opt/redco/.venv/bin/python",
                        f"{role}.py",
                        arm,
                        *([f"/opt/redco/checkpoints/{arm}"] if role == "server" else []),
                    ),
                    working_directory="/opt/redco",
                    checkpoint_root=f"/opt/redco/checkpoints/{arm}",
                    environment=(("PYTHONHASHSEED", "0"),),
                    source_sha256s=((f"{role}.py", _sha("2")),),
                    checkpoint_manifest_sha256=state.checkpoint_sha256 or _sha("3"),
                    post_model_sha256=state.post_model_sha256 or _sha("4"),
                    reload_evidence_sha256=state.reload_evidence_sha256 or _sha("5"),
                    endpoint=f"http://127.0.0.1:{8100 + len(programs) // 2}",
                    gpu_assignment=(len(programs) // 2,),
                    cache_namespace=f"evaluation-{arm}",
                )
            )
    schedule = tuple(
        EvaluationScheduleUnit(ordinal, arm, 0, "task-1", 9101)
        for ordinal, arm in enumerate(("stock", "branch-global", "local"))
    )
    return StageDEvaluationExecutionManifest(
        evaluation_ledger_id=_sha("0"),
        protocol_manifest_sha256=protocol_sha256,
        trainer_ledger_head_sha256=snapshot.head_sha256,
        trainer_record_count=snapshot.record_count,
        heldout_eval_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        evaluation_plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
        decision_rule_sha256=_sha("a"),
        runtime_entrypoints=(
            EvaluationRuntimeEntrypoint(
                "task_runner",
                "task_runtime.py",
                "task_runtime",
                "run_task",
                "redco-stage-d-worker-ipc-v1",
                _sha("b"),
            ),
            EvaluationRuntimeEntrypoint(
                "scorer", "scorer.py", "scorer", "score", "redco-stage-d-scorer-v1", _sha("c")
            ),
            EvaluationRuntimeEntrypoint(
                "request_serializer",
                "serializer.py",
                "serializer",
                "serialize",
                "redco-stage-d-request-serializer-v1",
                _sha("d"),
            ),
        ),
        runtime_worker_image="python@sha256:" + "f" * 64,
        runtime_bundle_path="/opt/redco/runtime.zip",
        runtime_bundle_sha256=_sha("e"),
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
        programs=tuple(programs),
        schedule=schedule,
    ).to_bytes()


def _write_execution_manifest(
    tmp_path: Path,
    snapshot: TrainerRunSnapshot,
    *,
    protocol_sha256: str,
    config: Path,
    plan: Path,
) -> Path:
    path = tmp_path / "execution-manifest.json"
    path.write_bytes(
        _execution_manifest_bytes(
            snapshot,
            protocol_sha256=protocol_sha256,
            config=config,
            plan=plan,
        )
    )
    return path


def _metrics_bytes(
    *,
    arm: str,
    checkpoint_sha256: str,
    authorization_sha256: str,
    raw_output_sha256: str,
) -> bytes:
    return canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-heldout-metrics-v1",
            "arm": arm,
            "checkpoint_manifest_sha256": checkpoint_sha256,
            "evaluation_authorization_sha256": authorization_sha256,
            "task_order": ["task-1"],
            "examples": [
                {
                    "task_id": "task-1",
                    "seed": 9101,
                    "reward": 0.5,
                    "raw_output_sha256": raw_output_sha256,
                    "policy_calls": 1,
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "wall_seconds": 0.1,
                    "gpu_seconds": 0.1,
                }
            ],
            "aggregate": {
                "mean_reward": 0.5,
                "success_count": 0,
                "example_count": 1,
                "policy_calls": 1,
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "wall_seconds": 0.1,
                "gpu_seconds": 0.1,
            },
        }
    )


def test_evaluation_is_forbidden_until_all_three_training_commits(tmp_path: Path) -> None:
    config, plan, protocol, protocol_sha256 = _frozen_files(tmp_path)
    snapshot = _snapshot(complete=False, protocol_manifest_sha256=protocol_sha256)
    execution_manifest = _write_execution_manifest(
        tmp_path,
        snapshot,
        protocol_sha256=protocol_sha256,
        config=config,
        plan=plan,
    )
    with pytest.raises(RuntimeError, match="before all training commits"):
        _authorize_heldout_evaluation_from_live_ledger(
            ledger=_Ledger(snapshot),  # type: ignore[arg-type]
            protocol_manifest_path=protocol,
            heldout_eval_config_path=config,
            evaluation_plan_path=plan,
            execution_manifest_path=execution_manifest,
            handoff_training_adoption_record_sha256=_sha("f"),
            destination=tmp_path / "authorization.json",
        )
    assert not (tmp_path / "authorization.json").exists()


def test_authorization_binds_ledger_head_and_exact_metrics_roster(tmp_path: Path) -> None:
    config, plan, protocol, protocol_sha256 = _frozen_files(tmp_path)
    ledger = _Ledger(_snapshot(protocol_manifest_sha256=protocol_sha256))
    execution_manifest = _write_execution_manifest(
        tmp_path,
        ledger.snapshot,
        protocol_sha256=protocol_sha256,
        config=config,
        plan=plan,
    )
    authorization_path = tmp_path / "authorization.json"
    authorization = _authorize_heldout_evaluation_from_live_ledger(
        ledger=ledger,  # type: ignore[arg-type]
        protocol_manifest_path=protocol,
        heldout_eval_config_path=config,
        evaluation_plan_path=plan,
        execution_manifest_path=execution_manifest,
        handoff_training_adoption_record_sha256=_sha("f"),
        destination=authorization_path,
    )
    assert (
        StageDEvaluationAuthorization.from_bytes(authorization_path.read_bytes()) == authorization
    )
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    retained = tmp_path / "retained"
    raw_value = b"raw held-out output"
    raw_digest = hashlib.sha256(raw_value).hexdigest()
    (raw / raw_digest).write_bytes(raw_value)
    checkpoint_by_arm = {
        item.arm: item.checkpoint_manifest_sha256 for item in authorization.checkpoints
    }
    for arm in ("stock", "branch-global", "local"):
        (metrics / f"{arm}.json").write_bytes(
            _metrics_bytes(
                arm=arm,
                checkpoint_sha256=checkpoint_by_arm[arm],
                authorization_sha256=authorization.authorization_sha256,
                raw_output_sha256=raw_digest,
            )
        )
    completion_path = tmp_path / "completion.json"
    completion = commit_heldout_evaluation(
        authorization_path=authorization_path,
        ledger=ledger,  # type: ignore[arg-type]
        heldout_eval_config_path=config,
        evaluation_plan_path=plan,
        metrics_root=metrics,
        raw_evidence_root=raw,
        retained_evidence_root=retained,
        destination=completion_path,
    )
    assert StageDEvaluationCompletion.from_bytes(completion_path.read_bytes()) == completion
    completion.verify_evidence(retained)
    assert (retained / raw_digest).read_bytes() == raw_value
    (retained / raw_digest).unlink()
    with pytest.raises(ValueError, match="raw evidence"):
        completion.verify_evidence(retained)
    (retained / raw_digest).write_bytes(raw_value)
    ledger.snapshot = replace(ledger.snapshot, head_sha256=_sha("8"))
    with pytest.raises(ValueError, match="changed after"):
        authorization.verify_trainer_ledger(ledger)  # type: ignore[arg-type]


def test_authorization_rejects_protocol_or_plan_mismatch(tmp_path: Path) -> None:
    config, plan, protocol, protocol_sha256 = _frozen_files(tmp_path)
    ledger = _Ledger(_snapshot(protocol_manifest_sha256=protocol_sha256))
    plan.write_bytes(
        StageDEvaluationPlan(
            tasks=(StageDEvaluationTask("different", 9102),),
            reward_min=0.0,
            reward_max=1.0,
            success_reward_threshold=1.0,
        ).to_bytes()
    )
    execution_manifest = _write_execution_manifest(
        tmp_path,
        ledger.snapshot,
        protocol_sha256=protocol_sha256,
        config=config,
        plan=plan,
    )
    with pytest.raises(ValueError, match="plan differs"):
        _authorize_heldout_evaluation_from_live_ledger(
            ledger=ledger,  # type: ignore[arg-type]
            protocol_manifest_path=protocol,
            heldout_eval_config_path=config,
            evaluation_plan_path=plan,
            execution_manifest_path=execution_manifest,
            handoff_training_adoption_record_sha256=_sha("f"),
            destination=tmp_path / "authorization.json",
        )


def test_completion_rejects_extra_or_mislabeled_metrics(tmp_path: Path) -> None:
    config, plan, protocol, protocol_sha256 = _frozen_files(tmp_path)
    ledger = _Ledger(_snapshot(protocol_manifest_sha256=protocol_sha256))
    execution_manifest = _write_execution_manifest(
        tmp_path,
        ledger.snapshot,
        protocol_sha256=protocol_sha256,
        config=config,
        plan=plan,
    )
    authorization_path = tmp_path / "authorization.json"
    _authorize_heldout_evaluation_from_live_ledger(
        ledger=ledger,  # type: ignore[arg-type]
        protocol_manifest_path=protocol,
        heldout_eval_config_path=config,
        evaluation_plan_path=plan,
        execution_manifest_path=execution_manifest,
        handoff_training_adoption_record_sha256=_sha("f"),
        destination=authorization_path,
    )
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    retained = tmp_path / "retained"
    for arm in ("stock", "branch-global", "local"):
        (metrics / f"{arm}.json").write_bytes(canonical_json({"arm": arm}))
    (metrics / "stale.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="exact three-arm roster"):
        commit_heldout_evaluation(
            authorization_path=authorization_path,
            ledger=ledger,  # type: ignore[arg-type]
            heldout_eval_config_path=config,
            evaluation_plan_path=plan,
            metrics_root=metrics,
            raw_evidence_root=raw,
            retained_evidence_root=retained,
            destination=tmp_path / "completion.json",
        )


def test_metrics_reject_negative_costs_and_wrong_frozen_seed() -> None:
    raw = hashlib.sha256(b"raw").hexdigest()
    value = _metrics_bytes(
        arm="stock",
        checkpoint_sha256=_sha("1"),
        authorization_sha256=_sha("2"),
        raw_output_sha256=raw,
    )
    payload = json.loads(value)
    payload["examples"][0]["wall_seconds"] = -0.1
    payload["aggregate"]["wall_seconds"] = -0.1
    with pytest.raises(ValueError, match="nonnegative"):
        StageDHeldoutMetrics.from_bytes(canonical_json(payload))

    parsed = StageDHeldoutMetrics.from_bytes(value)
    wrong_seed_plan = StageDEvaluationPlan(
        tasks=(StageDEvaluationTask("task-1", 9999),),
        reward_min=0.0,
        reward_max=1.0,
        success_reward_threshold=1.0,
    )
    with pytest.raises(ValueError, match="frozen evaluation plan"):
        parsed.verify_plan(wrong_seed_plan)


@pytest.mark.parametrize(
    ("fault_stage", "expected_final"),
    (
        ("after-evaluation-temp-fsync", False),
        ("after-evaluation-rename", True),
    ),
)
def test_durable_receipt_hard_kill_is_exactly_restartable(
    tmp_path: Path,
    fault_stage: str,
    expected_final: bool,
) -> None:
    path = tmp_path / "authorization.json"
    value = canonical_json({"authorization": "frozen"})
    code = r"""
import os
import sys
from pathlib import Path
from redco.analysis.stage_d_evaluation_barrier import _exclusive_write

def crash(stage, _path):
    if stage == sys.argv[3]:
        os._exit(97)

_exclusive_write(Path(sys.argv[1]), bytes.fromhex(sys.argv[2]), fault_hook=crash)
"""
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), environment.get("PYTHONPATH", "")))
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), value.hex(), fault_stage],
        check=False,
        capture_output=True,
        env=environment,
    )
    assert result.returncode == 97
    assert path.exists() is expected_final
    from redco.analysis.stage_d_evaluation_barrier import _exclusive_write

    _exclusive_write(path, value)
    assert path.read_bytes() == value
    assert not path.with_name(f".{path.name}.pending").exists()
