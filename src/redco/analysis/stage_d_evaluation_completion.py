"""Arm-level completion transaction for held-out Stage-D evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, cast

from redco.analysis.stage_d_evaluation_codec import (
    EvaluationEvidenceStore,
    canonical_object,
    exclusive_lock,
    sha256,
)
from redco.analysis.stage_d_evaluation_contracts import StageDEvaluationExecutionManifest
from redco.analysis.stage_d_evaluation_state import EvaluationLedgerSnapshot
from redco.analysis.stage_d_objective_binding import ArmName
from redco.contracts import canonical_json


class EvaluationCompletionLedger(Protocol):
    lock_path: Path
    evidence: EvaluationEvidenceStore

    @property
    def manifest(self) -> StageDEvaluationExecutionManifest: ...

    @property
    def authorization_bytes(self) -> bytes: ...

    def inspect(self) -> EvaluationLedgerSnapshot: ...

    def _append_unlocked(self, kind: str, event: dict[str, Any]) -> str: ...


def complete_arm(ledger: EvaluationCompletionLedger, arm: ArmName) -> bytes:
    """Aggregate one complete frozen arm and append its immutable metrics record."""
    with exclusive_lock(ledger.lock_path):
        snapshot = ledger.inspect()
        if arm in dict(snapshot.arm_completions):
            return ledger.evidence.get(dict(snapshot.arm_metrics)[arm])
        tasks = [task for task in snapshot.tasks if task.unit.arm == arm]
        frozen_count = sum(item.arm == arm for item in ledger.manifest.schedule)
        if len(tasks) != frozen_count or any(not task.completed for task in tasks):
            raise RuntimeError("evaluation arm has not completed its frozen tasks")
        task_metrics = [
            canonical_object(
                ledger.evidence.get(cast(str, task.task_metrics_sha256)),
                "evaluation task metrics",
            )
            for task in tasks
        ]
        checkpoint = ledger.manifest.program(arm, "server").checkpoint_manifest_sha256
        examples = [
            {
                "task_id": item["task_id"],
                "seed": item["seed"],
                "reward": item["reward"],
                "raw_output_sha256": item["terminal_result_sha256"],
                "policy_calls": item["policy_calls"],
                "prompt_tokens": item["prompt_tokens"],
                "completion_tokens": item["completion_tokens"],
                "wall_seconds": item["wall_seconds"],
                "gpu_seconds": item["gpu_seconds"],
            }
            for item in task_metrics
        ]
        metrics = canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-heldout-metrics-v1",
                "arm": arm,
                "checkpoint_manifest_sha256": checkpoint,
                "evaluation_authorization_sha256": sha256(ledger.authorization_bytes),
                "task_order": [item["task_id"] for item in task_metrics],
                "examples": examples,
                "aggregate": {
                    "mean_reward": sum(item["reward"] for item in task_metrics) / len(task_metrics),
                    "success_count": sum(bool(item["success"]) for item in task_metrics),
                    "example_count": len(task_metrics),
                    "policy_calls": sum(item["policy_calls"] for item in task_metrics),
                    "prompt_tokens": sum(item["prompt_tokens"] for item in task_metrics),
                    "completion_tokens": sum(item["completion_tokens"] for item in task_metrics),
                    "wall_seconds": sum(item["wall_seconds"] for item in task_metrics),
                    "gpu_seconds": sum(item["gpu_seconds"] for item in task_metrics),
                },
            }
        )
        metrics_sha256 = ledger.evidence.put(metrics)
        ledger._append_unlocked(
            "arm_completed",
            {
                "arm": arm,
                "arm_metrics_sha256": metrics_sha256,
                "task_attempt_ids": [task.task_attempt_id for task in tasks],
            },
        )
        return metrics


__all__ = ["EvaluationCompletionLedger", "complete_arm"]
