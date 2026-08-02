from __future__ import annotations

import hashlib
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_stage_d_evaluation_ledger import _frozen_inputs

import redco.analysis.stage_d_evaluation_driver as driver_module
from redco.analysis.stage_d_evaluation_capabilities import EvaluationTaskAttempt
from redco.analysis.stage_d_evaluation_contracts import (
    EvaluationRuntimeEntrypoint,
    StageDEvaluationExecutionManifest,
)
from redco.analysis.stage_d_evaluation_driver import (
    EvaluationDriverLimits,
    run_evaluation_arm,
)
from redco.analysis.stage_d_evaluation_worker import RuntimeTaskOutput

_TASK_SOURCE = b"def run_task():\n    raise AssertionError('worker is isolated')\n"
_SERIALIZER_SOURCE = (
    b"from redco.contracts import canonical_json\n"
    b"def serialize(payload, *, seed, cache_salt):\n"
    b"    return canonical_json({**payload, 'seed': seed, "
    b"'extra_body': {'cache_salt': cache_salt}})\n"
)
_SCORER_SOURCE = (
    b"import hashlib\n"
    b"from redco.contracts import canonical_json\n"
    b"def score(*, task_attempt_id, task_id, seed, terminal_output_bytes, "
    b"task_evidence_bytes):\n"
    b"    return canonical_json({'schema_version': 1, "
    b"'domain': 'redco-stage-d-heldout-score-v1', "
    b"'task_attempt_id': task_attempt_id, 'task_id': task_id, 'seed': seed, "
    b"'terminal_output_sha256': hashlib.sha256(terminal_output_bytes).hexdigest(), "
    b"'task_evidence_sha256': hashlib.sha256(task_evidence_bytes).hexdigest(), "
    b"'reward': 0.75, 'details': {}})\n"
)


def _manifest(tmp_path: Path) -> StageDEvaluationExecutionManifest:
    _, base_bytes, _ = _frozen_inputs()
    bundle = tmp_path / "runtime.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("task_runtime.py", _TASK_SOURCE)
        archive.writestr("serializer.py", _SERIALIZER_SOURCE)
        archive.writestr("scorer.py", _SCORER_SOURCE)
    entrypoints = (
        EvaluationRuntimeEntrypoint(
            "task_runner",
            "task_runtime.py",
            "task_runtime",
            "run_task",
            "redco-stage-d-worker-ipc-v1",
            hashlib.sha256(_TASK_SOURCE).hexdigest(),
        ),
        EvaluationRuntimeEntrypoint(
            "scorer",
            "scorer.py",
            "scorer",
            "score",
            "redco-stage-d-scorer-v1",
            hashlib.sha256(_SCORER_SOURCE).hexdigest(),
        ),
        EvaluationRuntimeEntrypoint(
            "request_serializer",
            "serializer.py",
            "serializer",
            "serialize",
            "redco-stage-d-request-serializer-v1",
            hashlib.sha256(_SERIALIZER_SOURCE).hexdigest(),
        ),
    )
    return replace(
        StageDEvaluationExecutionManifest.from_bytes(base_bytes),
        runtime_entrypoints=entrypoints,
        runtime_bundle_path=str(bundle.resolve()),
        runtime_bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
    )


class _Ledger:
    def __init__(self, manifest: StageDEvaluationExecutionManifest) -> None:
        self.manifest = manifest
        self.tasks: list[Any] = []
        self.current: Any = None
        self.completed_reward: float | None = None

    def inspect(self) -> Any:
        return SimpleNamespace(tasks=tuple(self.tasks), current_task=self.current)

    def resume_current_client_session(self, _arm: str) -> object:
        return object()

    def reserve_next_task(self, *, session: object) -> EvaluationTaskAttempt:
        del session
        unit = self.manifest.schedule[0]
        task = EvaluationTaskAttempt("a" * 64, unit, 0)
        self.current = SimpleNamespace(calls=())
        return task

    def resume_open_task(self, *, session: object) -> EvaluationTaskAttempt:
        raise AssertionError(f"unexpected resume for {session!r}")

    def complete_task(self, task: EvaluationTaskAttempt, **kwargs: Any) -> None:
        self.completed_reward = kwargs["reward"]
        self.tasks.append(SimpleNamespace(unit=task.unit))
        self.current = None

    def complete_arm(self, arm: str) -> bytes:
        assert arm == "stock"
        return b"stock metrics"


class _Runtime:
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def run_task(self, **_kwargs: Any) -> RuntimeTaskOutput:
        return RuntimeTaskOutput(b'{"answer":"heldout"}', b'{"trace":"frozen"}')


def test_driver_uses_hash_bound_parent_scorer_and_completes_contiguous_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _Ledger(_manifest(tmp_path))
    monkeypatch.setattr(driver_module, "DockerEvaluationRuntime", _Runtime)
    result = run_evaluation_arm(
        ledger=ledger,  # type: ignore[arg-type]
        arm="stock",
        docker_executable=tmp_path / "unused-docker",
        docker_executable_sha256="0" * 64,
        limits=EvaluationDriverLimits(1.0, 2.0, 4, 128),
    )
    assert result == b"stock metrics"
    assert ledger.completed_reward == 0.75


def test_driver_rejects_changed_frozen_scorer_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    with zipfile.ZipFile(manifest.runtime_bundle_path, "w") as archive:
        archive.writestr("task_runtime.py", _TASK_SOURCE)
        archive.writestr("serializer.py", _SERIALIZER_SOURCE)
        archive.writestr("scorer.py", b"def score(**kwargs): return b'{}'\n")
    ledger = _Ledger(manifest)
    monkeypatch.setattr(driver_module, "DockerEvaluationRuntime", _Runtime)
    with pytest.raises(ValueError, match="entrypoint changed"):
        run_evaluation_arm(
            ledger=ledger,  # type: ignore[arg-type]
            arm="stock",
            docker_executable=tmp_path / "unused-docker",
            docker_executable_sha256="0" * 64,
            limits=EvaluationDriverLimits(1.0, 2.0, 4, 128),
        )
