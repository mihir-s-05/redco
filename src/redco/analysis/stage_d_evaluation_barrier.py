"""Durable all-training-before-evaluation barrier for Stage D."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from redco.analysis.stage_d_evaluation_contracts import (
    StageDEvaluationExecutionManifest,
)
from redco.analysis.stage_d_objective_binding import ArmName
from redco.analysis.stage_d_protocol_manifest import StageDProtocolManifest
from redco.analysis.stage_d_trainer_supervisor import StageDTrainerRunLedger
from redco.contracts import canonical_json
from redco.integrity import require_sha256_hex as _require_sha256
from redco.integrity import sha256_bytes as _sha256

if TYPE_CHECKING:
    from redco.analysis.stage_d_evaluation_ledger import StageDEvaluationLedger

_AUTHORIZATION_DOMAIN = "redco-stage-d-evaluation-authorization-v4"
_COMPLETION_DOMAIN = "redco-stage-d-evaluation-completion-v1"
_SEALED_COMPLETION_DOMAIN = "redco-stage-d-sealed-evaluation-completion-v1"
_METRICS_DOMAIN = "redco-stage-d-heldout-metrics-v1"
_PLAN_DOMAIN = "redco-stage-d-evaluation-plan-v1"
_ARMS: tuple[ArmName, ...] = ("stock", "branch-global", "local")
FaultHook = Callable[[str, Path], None]


@dataclass(frozen=True, slots=True)
class EvaluationCheckpointBinding:
    arm: ArmName
    checkpoint_manifest_sha256: str
    post_model_sha256: str
    reload_evidence_sha256: str

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise ValueError("evaluation checkpoint arm is invalid")
        for name in (
            "checkpoint_manifest_sha256",
            "post_model_sha256",
            "reload_evidence_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    def to_payload(self) -> dict[str, str]:
        return {
            "arm": self.arm,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "post_model_sha256": self.post_model_sha256,
            "reload_evidence_sha256": self.reload_evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class StageDEvaluationTask:
    task_id: str
    seed: int

    def __post_init__(self) -> None:
        if not self.task_id or not self.task_id.isprintable():
            raise ValueError("evaluation task ID is invalid")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("evaluation task seed is invalid")


@dataclass(frozen=True, slots=True)
class StageDEvaluationPlan:
    tasks: tuple[StageDEvaluationTask, ...]
    reward_min: float
    reward_max: float
    success_reward_threshold: float

    def __post_init__(self) -> None:
        if not self.tasks or len({item.task_id for item in self.tasks}) != len(self.tasks):
            raise ValueError("evaluation plan task roster is empty or duplicated")
        if len({(item.task_id, item.seed) for item in self.tasks}) != len(self.tasks):
            raise ValueError("evaluation plan task identities are duplicated")
        for name in ("reward_min", "reward_max", "success_reward_threshold"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"evaluation plan {name} must be finite")
        if not self.reward_min <= self.success_reward_threshold <= self.reward_max:
            raise ValueError("evaluation success threshold is outside reward bounds")

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": _PLAN_DOMAIN,
                "tasks": [{"task_id": item.task_id, "seed": item.seed} for item in self.tasks],
                "reward_min": self.reward_min,
                "reward_max": self.reward_max,
                "success_reward_threshold": self.success_reward_threshold,
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDEvaluationPlan:
        payload = _canonical_object(value, "evaluation plan")
        if (
            set(payload)
            != {
                "schema_version",
                "domain",
                "tasks",
                "reward_min",
                "reward_max",
                "success_reward_threshold",
            }
            or payload.get("schema_version") != 1
            or payload.get("domain") != _PLAN_DOMAIN
            or not isinstance(payload.get("tasks"), list)
        ):
            raise ValueError("evaluation plan fields differ")
        tasks = []
        for item in payload["tasks"]:
            if not isinstance(item, dict) or set(item) != {"task_id", "seed"}:
                raise ValueError("evaluation task fields differ")
            tasks.append(StageDEvaluationTask(item["task_id"], item["seed"]))
        return cls(
            tuple(tasks),
            _finite(payload["reward_min"], "evaluation reward minimum"),
            _finite(payload["reward_max"], "evaluation reward maximum"),
            _finite(payload["success_reward_threshold"], "success reward threshold"),
        )


@dataclass(frozen=True, slots=True)
class StageDEvaluationAuthorization:
    handoff_training_adoption_record_sha256: str
    campaign_manifest_sha256: str
    protocol_manifest_sha256: str
    trainer_ledger_head_sha256: str
    trainer_record_count: int
    heldout_eval_config_sha256: str
    evaluation_plan_sha256: str
    execution_manifest_sha256: str
    checkpoints: tuple[EvaluationCheckpointBinding, ...]

    def __post_init__(self) -> None:
        for name in (
            "handoff_training_adoption_record_sha256",
            "campaign_manifest_sha256",
            "protocol_manifest_sha256",
            "trainer_ledger_head_sha256",
            "heldout_eval_config_sha256",
            "evaluation_plan_sha256",
            "execution_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.trainer_record_count) is not int or self.trainer_record_count < 1:
            raise ValueError("evaluation authorization trainer record count is invalid")
        if tuple(item.arm for item in self.checkpoints) != _ARMS:
            raise ValueError("evaluation authorization requires the exact three-arm order")

    @property
    def authorization_sha256(self) -> str:
        return _sha256(self.to_bytes())

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 4,
                "domain": _AUTHORIZATION_DOMAIN,
                "handoff_training_adoption_record_sha256": (
                    self.handoff_training_adoption_record_sha256
                ),
                "campaign_manifest_sha256": self.campaign_manifest_sha256,
                "protocol_manifest_sha256": self.protocol_manifest_sha256,
                "trainer_ledger_head_sha256": self.trainer_ledger_head_sha256,
                "trainer_record_count": self.trainer_record_count,
                "heldout_eval_config_sha256": self.heldout_eval_config_sha256,
                "evaluation_plan_sha256": self.evaluation_plan_sha256,
                "execution_manifest_sha256": self.execution_manifest_sha256,
                "checkpoints": [item.to_payload() for item in self.checkpoints],
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDEvaluationAuthorization:
        payload = _canonical_object(value, "evaluation authorization")
        fields = {
            "schema_version",
            "domain",
            "handoff_training_adoption_record_sha256",
            "campaign_manifest_sha256",
            "protocol_manifest_sha256",
            "trainer_ledger_head_sha256",
            "trainer_record_count",
            "heldout_eval_config_sha256",
            "evaluation_plan_sha256",
            "execution_manifest_sha256",
            "checkpoints",
        }
        if (
            set(payload) != fields
            or payload.get("schema_version") != 4
            or payload.get("domain") != _AUTHORIZATION_DOMAIN
            or not isinstance(payload.get("checkpoints"), list)
        ):
            raise ValueError("evaluation authorization fields differ")
        checkpoints = []
        for item in payload["checkpoints"]:
            if not isinstance(item, dict) or set(item) != {
                "arm",
                "checkpoint_manifest_sha256",
                "post_model_sha256",
                "reload_evidence_sha256",
            }:
                raise ValueError("evaluation checkpoint binding fields differ")
            checkpoints.append(EvaluationCheckpointBinding(**item))
        return cls(
            handoff_training_adoption_record_sha256=(
                payload["handoff_training_adoption_record_sha256"]
            ),
            campaign_manifest_sha256=payload["campaign_manifest_sha256"],
            protocol_manifest_sha256=payload["protocol_manifest_sha256"],
            trainer_ledger_head_sha256=payload["trainer_ledger_head_sha256"],
            trainer_record_count=payload["trainer_record_count"],
            heldout_eval_config_sha256=payload["heldout_eval_config_sha256"],
            evaluation_plan_sha256=payload["evaluation_plan_sha256"],
            execution_manifest_sha256=payload["execution_manifest_sha256"],
            checkpoints=tuple(checkpoints),
        )

    def verify_trainer_ledger(self, ledger: StageDTrainerRunLedger) -> None:
        snapshot = ledger.inspect()
        if (
            snapshot.campaign_manifest_sha256 != self.campaign_manifest_sha256
            or snapshot.protocol_manifest_sha256 != self.protocol_manifest_sha256
            or snapshot.head_sha256 != self.trainer_ledger_head_sha256
            or snapshot.record_count != self.trainer_record_count
        ):
            raise ValueError("trainer ledger changed after evaluation authorization")
        observed = tuple(
            EvaluationCheckpointBinding(
                arm=arm,
                checkpoint_manifest_sha256=_required_state(state.checkpoint_sha256),
                post_model_sha256=_required_state(state.post_model_sha256),
                reload_evidence_sha256=_required_state(state.reload_evidence_sha256),
            )
            for arm in _ARMS
            for state in (snapshot.state(arm),)
        )
        if observed != self.checkpoints:
            raise ValueError("trainer checkpoints differ from evaluation authorization")


@dataclass(frozen=True, slots=True)
class StageDEvaluationCompletion:
    evaluation_authorization_sha256: str
    heldout_eval_config_sha256: str
    evaluation_plan_sha256: str
    metrics_sha256s: tuple[tuple[ArmName, str], ...]

    def __post_init__(self) -> None:
        _require_sha256(
            self.evaluation_authorization_sha256,
            "evaluation authorization sha256",
        )
        _require_sha256(self.heldout_eval_config_sha256, "heldout eval config sha256")
        _require_sha256(self.evaluation_plan_sha256, "evaluation plan sha256")
        if tuple(arm for arm, _ in self.metrics_sha256s) != _ARMS:
            raise ValueError("evaluation completion requires exact three-arm metrics")
        for _, digest in self.metrics_sha256s:
            _require_sha256(digest, "evaluation metrics sha256")

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": _COMPLETION_DOMAIN,
                "evaluation_authorization_sha256": (self.evaluation_authorization_sha256),
                "heldout_eval_config_sha256": self.heldout_eval_config_sha256,
                "evaluation_plan_sha256": self.evaluation_plan_sha256,
                "metrics_sha256s": [
                    {"arm": arm, "sha256": digest} for arm, digest in self.metrics_sha256s
                ],
            }
        )

    def verify_evidence(self, evidence_root: Path) -> None:
        for arm, digest in self.metrics_sha256s:
            path = evidence_root / digest
            if path.is_symlink() or not path.is_file():
                raise ValueError("retained evaluation metrics evidence differs")
            value = path.read_bytes()
            if _sha256(value) != digest:
                raise ValueError("retained evaluation metrics evidence differs")
            metrics = StageDHeldoutMetrics.from_bytes(value)
            if (
                metrics.arm != arm
                or metrics.evaluation_authorization_sha256 != self.evaluation_authorization_sha256
            ):
                raise ValueError("retained evaluation metrics binding differs")
            for example in metrics.examples:
                raw_path = evidence_root / example.raw_output_sha256
                if (
                    raw_path.is_symlink()
                    or not raw_path.is_file()
                    or _sha256(raw_path.read_bytes()) != example.raw_output_sha256
                ):
                    raise ValueError("retained evaluation raw evidence differs")

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDEvaluationCompletion:
        payload = _canonical_object(value, "evaluation completion")
        if (
            set(payload)
            != {
                "schema_version",
                "domain",
                "evaluation_authorization_sha256",
                "heldout_eval_config_sha256",
                "evaluation_plan_sha256",
                "metrics_sha256s",
            }
            or payload.get("schema_version") != 1
            or payload.get("domain") != _COMPLETION_DOMAIN
            or not isinstance(payload.get("metrics_sha256s"), list)
        ):
            raise ValueError("evaluation completion fields differ")
        metrics = []
        for item in payload["metrics_sha256s"]:
            if not isinstance(item, dict) or set(item) != {"arm", "sha256"}:
                raise ValueError("evaluation metrics binding fields differ")
            metrics.append((item["arm"], item["sha256"]))
        return cls(
            payload["evaluation_authorization_sha256"],
            payload["heldout_eval_config_sha256"],
            payload["evaluation_plan_sha256"],
            tuple(metrics),
        )


@dataclass(frozen=True, slots=True)
class StageDSealedEvaluationCompletion:
    evaluation_authorization_sha256: str
    execution_manifest_sha256: str
    evaluation_ledger_head_sha256: str
    evaluation_record_count: int
    metrics_sha256s: tuple[tuple[ArmName, str], ...]

    def __post_init__(self) -> None:
        for name in (
            "evaluation_authorization_sha256",
            "execution_manifest_sha256",
            "evaluation_ledger_head_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.evaluation_record_count) is not int or self.evaluation_record_count < 1:
            raise ValueError("sealed evaluation record count is invalid")
        if tuple(arm for arm, _ in self.metrics_sha256s) != _ARMS:
            raise ValueError("sealed evaluation completion requires exact three-arm metrics")
        for _, digest in self.metrics_sha256s:
            _require_sha256(digest, "sealed evaluation metrics sha256")

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": _SEALED_COMPLETION_DOMAIN,
                "evaluation_authorization_sha256": self.evaluation_authorization_sha256,
                "execution_manifest_sha256": self.execution_manifest_sha256,
                "evaluation_ledger_head_sha256": self.evaluation_ledger_head_sha256,
                "evaluation_record_count": self.evaluation_record_count,
                "metrics_sha256s": [
                    {"arm": arm, "sha256": digest} for arm, digest in self.metrics_sha256s
                ],
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDSealedEvaluationCompletion:
        payload = _canonical_object(value, "sealed evaluation completion")
        if (
            set(payload)
            != {
                "schema_version",
                "domain",
                "evaluation_authorization_sha256",
                "execution_manifest_sha256",
                "evaluation_ledger_head_sha256",
                "evaluation_record_count",
                "metrics_sha256s",
            }
            or payload.get("schema_version") != 1
            or payload.get("domain") != _SEALED_COMPLETION_DOMAIN
            or not isinstance(payload.get("metrics_sha256s"), list)
        ):
            raise ValueError("sealed evaluation completion fields differ")
        metrics = []
        for item in payload["metrics_sha256s"]:
            if not isinstance(item, dict) or set(item) != {"arm", "sha256"}:
                raise ValueError("sealed evaluation metrics binding fields differ")
            metrics.append((item["arm"], item["sha256"]))
        return cls(
            payload["evaluation_authorization_sha256"],
            payload["execution_manifest_sha256"],
            payload["evaluation_ledger_head_sha256"],
            payload["evaluation_record_count"],
            tuple(metrics),
        )

    def verify_ledger(self, ledger: StageDEvaluationLedger) -> None:
        snapshot = ledger.inspect()
        if (
            not snapshot.sealed
            or snapshot.authorization_sha256 != self.evaluation_authorization_sha256
            or snapshot.execution_manifest_sha256 != self.execution_manifest_sha256
            or snapshot.head_sha256 != self.evaluation_ledger_head_sha256
            or snapshot.record_count != self.evaluation_record_count
            or snapshot.arm_metrics != self.metrics_sha256s
        ):
            raise ValueError("sealed evaluation ledger differs from completion")


@dataclass(frozen=True, slots=True)
class StageDHeldoutExample:
    task_id: str
    seed: int
    reward: float
    raw_output_sha256: str
    policy_calls: int
    prompt_tokens: int
    completion_tokens: int
    wall_seconds: float
    gpu_seconds: float


@dataclass(frozen=True, slots=True)
class StageDHeldoutMetrics:
    arm: ArmName
    checkpoint_manifest_sha256: str
    evaluation_authorization_sha256: str
    task_order: tuple[str, ...]
    examples: tuple[StageDHeldoutExample, ...]
    mean_reward: float
    success_count: int
    policy_calls: int
    prompt_tokens: int
    completion_tokens: int
    wall_seconds: float
    gpu_seconds: float

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDHeldoutMetrics:
        payload = _canonical_object(value, "held-out metrics")
        fields = {
            "schema_version",
            "domain",
            "arm",
            "checkpoint_manifest_sha256",
            "evaluation_authorization_sha256",
            "task_order",
            "examples",
            "aggregate",
        }
        if (
            set(payload) != fields
            or payload.get("schema_version") != 1
            or payload.get("domain") != _METRICS_DOMAIN
            or payload.get("arm") not in _ARMS
            or not isinstance(payload.get("task_order"), list)
            or not isinstance(payload.get("examples"), list)
            or not isinstance(payload.get("aggregate"), dict)
        ):
            raise ValueError("held-out metrics fields differ")
        task_order = tuple(payload["task_order"])
        if (
            not task_order
            or len(set(task_order)) != len(task_order)
            or any(not isinstance(item, str) or not item for item in task_order)
        ):
            raise ValueError("held-out task order is invalid")
        examples = []
        for item in payload["examples"]:
            expected_example_fields = {
                "task_id",
                "seed",
                "reward",
                "raw_output_sha256",
                "policy_calls",
                "prompt_tokens",
                "completion_tokens",
                "wall_seconds",
                "gpu_seconds",
            }
            if not isinstance(item, dict) or set(item) != expected_example_fields:
                raise ValueError("held-out example fields differ")
            if not isinstance(item["task_id"], str) or not item["task_id"]:
                raise ValueError("held-out example task ID is invalid")
            if type(item["seed"]) is not int or item["seed"] < 0:
                raise ValueError("held-out example seed is invalid")
            for name in ("policy_calls", "prompt_tokens", "completion_tokens"):
                if type(item[name]) is not int or item[name] < 0:
                    raise ValueError(f"held-out example {name} is invalid")
            examples.append(
                StageDHeldoutExample(
                    task_id=item["task_id"],
                    seed=item["seed"],
                    reward=_finite(item["reward"], "held-out reward"),
                    raw_output_sha256=_require_sha256(
                        item["raw_output_sha256"], "raw output sha256"
                    ),
                    policy_calls=item["policy_calls"],
                    prompt_tokens=item["prompt_tokens"],
                    completion_tokens=item["completion_tokens"],
                    wall_seconds=_nonnegative_finite(
                        item["wall_seconds"], "held-out example wall seconds"
                    ),
                    gpu_seconds=_nonnegative_finite(
                        item["gpu_seconds"], "held-out example GPU seconds"
                    ),
                )
            )
        if tuple(item.task_id for item in examples) != task_order:
            raise ValueError("held-out examples differ from the frozen task order")
        aggregate = payload["aggregate"]
        aggregate_fields = {
            "mean_reward",
            "success_count",
            "example_count",
            "policy_calls",
            "prompt_tokens",
            "completion_tokens",
            "wall_seconds",
            "gpu_seconds",
        }
        if set(aggregate) != aggregate_fields:
            raise ValueError("held-out aggregate fields differ")
        for name in (
            "success_count",
            "example_count",
            "policy_calls",
            "prompt_tokens",
            "completion_tokens",
        ):
            if type(aggregate[name]) is not int or aggregate[name] < 0:
                raise ValueError(f"held-out {name} is invalid")
        mean_reward = _finite(aggregate["mean_reward"], "mean reward")
        if aggregate["example_count"] != len(examples):
            raise ValueError("held-out example count differs")
        observed_mean = sum(item.reward for item in examples) / len(examples)
        if not math.isclose(mean_reward, observed_mean, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("held-out mean reward differs from examples")
        for name in ("policy_calls", "prompt_tokens", "completion_tokens"):
            if aggregate[name] != sum(getattr(item, name) for item in examples):
                raise ValueError(f"held-out {name} differs from examples")
        wall_seconds = _nonnegative_finite(aggregate["wall_seconds"], "wall seconds")
        gpu_seconds = _nonnegative_finite(aggregate["gpu_seconds"], "GPU seconds")
        for name, observed in (("wall_seconds", wall_seconds), ("gpu_seconds", gpu_seconds)):
            expected = sum(getattr(item, name) for item in examples)
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"held-out {name} differs from examples")
        return cls(
            arm=payload["arm"],
            checkpoint_manifest_sha256=_require_sha256(
                payload["checkpoint_manifest_sha256"],
                "held-out checkpoint manifest sha256",
            ),
            evaluation_authorization_sha256=_require_sha256(
                payload["evaluation_authorization_sha256"],
                "held-out authorization sha256",
            ),
            task_order=task_order,
            examples=tuple(examples),
            mean_reward=mean_reward,
            success_count=aggregate["success_count"],
            policy_calls=aggregate["policy_calls"],
            prompt_tokens=aggregate["prompt_tokens"],
            completion_tokens=aggregate["completion_tokens"],
            wall_seconds=wall_seconds,
            gpu_seconds=gpu_seconds,
        )

    def verify_plan(self, plan: StageDEvaluationPlan) -> None:
        expected = tuple((item.task_id, item.seed) for item in plan.tasks)
        observed = tuple((item.task_id, item.seed) for item in self.examples)
        if observed != expected or self.task_order != tuple(item[0] for item in expected):
            raise ValueError("held-out metrics differ from the frozen evaluation plan")
        if any(not plan.reward_min <= item.reward <= plan.reward_max for item in self.examples):
            raise ValueError("held-out reward is outside the frozen bounds")
        success_count = sum(item.reward >= plan.success_reward_threshold for item in self.examples)
        if self.success_count != success_count:
            raise ValueError("held-out success count differs from the frozen criterion")


def _authorize_heldout_evaluation_from_live_ledger(
    *,
    ledger: StageDTrainerRunLedger,
    protocol_manifest_path: Path,
    heldout_eval_config_path: Path,
    evaluation_plan_path: Path,
    execution_manifest_path: Path,
    handoff_training_adoption_record_sha256: str,
    destination: Path,
    fault_hook: FaultHook | None = None,
) -> StageDEvaluationAuthorization:
    """Atomically authorize evaluation only after all three checkpoints validate."""
    snapshot = ledger.inspect()
    protocol_bytes = protocol_manifest_path.read_bytes()
    protocol = StageDProtocolManifest.from_bytes(protocol_bytes)
    if protocol.manifest_sha256 != snapshot.protocol_manifest_sha256:
        raise ValueError("evaluation protocol differs from the trainer ledger")
    config_sha256 = _sha256(heldout_eval_config_path.read_bytes())
    if config_sha256 != protocol.heldout_eval_config_sha256:
        raise ValueError("held-out evaluation config differs from the protocol")
    plan_bytes = evaluation_plan_path.read_bytes()
    StageDEvaluationPlan.from_bytes(plan_bytes)
    plan_sha256 = _sha256(plan_bytes)
    if plan_sha256 != protocol.evaluation_plan_sha256:
        raise ValueError("held-out evaluation plan differs from the protocol")
    execution_manifest_bytes = execution_manifest_path.read_bytes()
    execution_manifest = StageDEvaluationExecutionManifest.from_bytes(execution_manifest_bytes)
    if (
        execution_manifest.protocol_manifest_sha256 != protocol.manifest_sha256
        or execution_manifest.trainer_ledger_head_sha256 != snapshot.head_sha256
        or execution_manifest.trainer_record_count != snapshot.record_count
        or execution_manifest.heldout_eval_config_sha256 != config_sha256
        or execution_manifest.evaluation_plan_sha256 != plan_sha256
        or execution_manifest.decision_rule_sha256 != protocol.decision_rule_sha256
    ):
        raise ValueError("evaluation execution manifest differs from frozen inputs")
    checkpoints = []
    for arm in _ARMS:
        state = snapshot.state(arm)
        if not state.checkpoint_committed:
            raise RuntimeError("held-out evaluation is forbidden before all training commits")
        checkpoints.append(
            EvaluationCheckpointBinding(
                arm,
                _required_state(state.checkpoint_sha256),
                _required_state(state.post_model_sha256),
                _required_state(state.reload_evidence_sha256),
            )
        )
    program_checkpoints = tuple(
        EvaluationCheckpointBinding(
            arm,
            execution_manifest.program(arm, "server").checkpoint_manifest_sha256,
            execution_manifest.program(arm, "server").post_model_sha256,
            execution_manifest.program(arm, "server").reload_evidence_sha256,
        )
        for arm in _ARMS
    )
    if tuple(checkpoints) != program_checkpoints:
        raise ValueError("evaluation execution manifest checkpoint bindings differ")
    authorization = StageDEvaluationAuthorization(
        _require_sha256(
            handoff_training_adoption_record_sha256,
            "handoff training adoption record sha256",
        ),
        snapshot.campaign_manifest_sha256,
        snapshot.protocol_manifest_sha256,
        snapshot.head_sha256,
        snapshot.record_count,
        config_sha256,
        plan_sha256,
        execution_manifest.manifest_sha256,
        tuple(checkpoints),
    )
    _exclusive_write(destination, authorization.to_bytes(), fault_hook=fault_hook)
    return authorization


def commit_heldout_evaluation(
    *,
    authorization_path: Path,
    ledger: StageDTrainerRunLedger,
    heldout_eval_config_path: Path,
    evaluation_plan_path: Path,
    metrics_root: Path,
    raw_evidence_root: Path,
    retained_evidence_root: Path,
    destination: Path,
    fault_hook: FaultHook | None = None,
) -> StageDEvaluationCompletion:
    authorization_bytes = authorization_path.read_bytes()
    authorization = StageDEvaluationAuthorization.from_bytes(authorization_bytes)
    authorization.verify_trainer_ledger(ledger)
    if _sha256(heldout_eval_config_path.read_bytes()) != (authorization.heldout_eval_config_sha256):
        raise ValueError("held-out evaluation config changed after authorization")
    plan_bytes = evaluation_plan_path.read_bytes()
    if _sha256(plan_bytes) != authorization.evaluation_plan_sha256:
        raise ValueError("held-out evaluation plan changed after authorization")
    plan = StageDEvaluationPlan.from_bytes(plan_bytes)
    expected = {f"{arm}.json" for arm in _ARMS}
    if not metrics_root.is_dir() or {path.name for path in metrics_root.iterdir()} != expected:
        raise ValueError("held-out metrics do not have the exact three-arm roster")
    metrics: list[tuple[ArmName, str]] = []
    retained_evidence_root.mkdir(parents=True, exist_ok=True)
    authorization_sha256 = _sha256(authorization_bytes)
    checkpoint_by_arm = {
        item.arm: item.checkpoint_manifest_sha256 for item in authorization.checkpoints
    }
    for arm in _ARMS:
        metrics_path = metrics_root / f"{arm}.json"
        if metrics_path.is_symlink() or not metrics_path.is_file():
            raise ValueError("held-out metrics must be regular non-symbolic files")
        value = metrics_path.read_bytes()
        parsed = StageDHeldoutMetrics.from_bytes(value)
        if (
            parsed.arm != arm
            or parsed.checkpoint_manifest_sha256 != checkpoint_by_arm[arm]
            or parsed.evaluation_authorization_sha256 != authorization_sha256
        ):
            raise ValueError("held-out metrics differ from their authorization")
        parsed.verify_plan(plan)
        for example in parsed.examples:
            raw_digest = example.raw_output_sha256
            raw_path = raw_evidence_root / raw_digest
            if raw_path.is_symlink() or not raw_path.is_file():
                raise ValueError("held-out raw response evidence differs")
            raw_value = raw_path.read_bytes()
            if _sha256(raw_value) != raw_digest:
                raise ValueError("held-out raw response evidence differs")
            if _write_content_addressed(retained_evidence_root, raw_value) != raw_digest:
                raise RuntimeError("retained raw evidence digest differs")
        digest = _write_content_addressed(retained_evidence_root, value)
        metrics.append((arm, digest))
    completion = StageDEvaluationCompletion(
        _sha256(authorization_bytes),
        authorization.heldout_eval_config_sha256,
        authorization.evaluation_plan_sha256,
        tuple(metrics),
    )
    _exclusive_write(destination, completion.to_bytes(), fault_hook=fault_hook)
    completion.verify_evidence(retained_evidence_root)
    return completion


def commit_sealed_heldout_evaluation(
    *,
    authorization_path: Path,
    trainer_ledger: StageDTrainerRunLedger,
    evaluation_ledger: StageDEvaluationLedger,
    heldout_eval_config_path: Path,
    evaluation_plan_path: Path,
    retained_evidence_root: Path,
    destination: Path,
    fault_hook: FaultHook | None = None,
) -> StageDSealedEvaluationCompletion:
    """Commit production metrics only from a sealed, transitively verified ledger."""
    authorization_bytes = authorization_path.read_bytes()
    if evaluation_ledger.authorization_bytes != authorization_bytes:
        raise ValueError("evaluation ledger authorization bytes differ")
    authorization = StageDEvaluationAuthorization.from_bytes(authorization_bytes)
    authorization.verify_trainer_ledger(trainer_ledger)
    if _sha256(heldout_eval_config_path.read_bytes()) != (authorization.heldout_eval_config_sha256):
        raise ValueError("held-out evaluation config changed after authorization")
    if _sha256(evaluation_plan_path.read_bytes()) != authorization.evaluation_plan_sha256:
        raise ValueError("held-out evaluation plan changed after authorization")
    snapshot = evaluation_ledger.inspect()
    if not snapshot.sealed:
        raise RuntimeError("held-out evaluation ledger is not sealed")
    if (
        snapshot.authorization_sha256 != authorization.authorization_sha256
        or snapshot.execution_manifest_sha256 != authorization.execution_manifest_sha256
        or tuple(arm for arm, _ in snapshot.arm_metrics) != _ARMS
    ):
        raise ValueError("sealed evaluation ledger differs from its authorization")
    retained_evidence_root.mkdir(parents=True, exist_ok=True)
    reachable = evaluation_ledger.reachable_evidence_sha256s()
    for digest in reachable:
        value = evaluation_ledger.evidence.get(digest)
        if _write_content_addressed(retained_evidence_root, value) != digest:
            raise RuntimeError("retained evaluation evidence digest differs")
    completion = StageDSealedEvaluationCompletion(
        evaluation_authorization_sha256=authorization.authorization_sha256,
        execution_manifest_sha256=authorization.execution_manifest_sha256,
        evaluation_ledger_head_sha256=snapshot.head_sha256,
        evaluation_record_count=snapshot.record_count,
        metrics_sha256s=snapshot.arm_metrics,
    )
    _exclusive_write(destination, completion.to_bytes(), fault_hook=fault_hook)
    completion.verify_ledger(evaluation_ledger)
    for _, digest in completion.metrics_sha256s:
        retained = retained_evidence_root / digest
        if (
            retained.is_symlink()
            or not retained.is_file()
            or _sha256(retained.read_bytes()) != digest
        ):
            raise ValueError("retained sealed evaluation metrics differ")
    return completion


def _required_state(value: str | None) -> str:
    if value is None:
        raise RuntimeError("trainer state lacks committed checkpoint evidence")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_finite(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _canonical_object(value: bytes, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(payload, dict) or canonical_json(payload) != value:
        raise ValueError(f"{name} is not canonical JSON")
    return payload


def _exclusive_write(
    path: Path,
    value: bytes,
    *,
    fault_hook: FaultHook | None = None,
) -> None:
    pending = path.with_name(f".{path.name}.pending")
    if path.exists():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != value
            or pending.exists()
        ):
            raise FileExistsError(f"durable evaluation receipt differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if pending.exists():
        if pending.is_symlink() or not pending.is_file() or pending.read_bytes() != value:
            raise FileExistsError(f"pending evaluation receipt differs: {pending}")
    else:
        descriptor = os.open(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    if fault_hook is not None:
        fault_hook("after-evaluation-temp-fsync", pending)
    os.replace(pending, path)
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    if fault_hook is not None:
        fault_hook("after-evaluation-rename", path)


def _write_content_addressed(root: Path, value: bytes) -> str:
    digest = _sha256(value)
    path = root / digest
    _exclusive_write(path, value)
    return digest


__all__ = [
    "EvaluationCheckpointBinding",
    "StageDEvaluationAuthorization",
    "StageDEvaluationCompletion",
    "StageDEvaluationPlan",
    "StageDEvaluationTask",
    "StageDHeldoutExample",
    "StageDHeldoutMetrics",
    "commit_heldout_evaluation",
]
