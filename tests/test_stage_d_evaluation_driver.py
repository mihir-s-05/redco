import hashlib
import zipfile
from collections.abc import Callable
from dataclasses import replace
from functools import partial
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
from redco.analysis.stage_d_evaluation_driver import EvaluationDriverLimits, run_evaluation_arm
from redco.analysis.stage_d_evaluation_worker import RuntimeTaskOutput

_TASK_SOURCE = b"def run_task():\n    raise AssertionError('worker is isolated')\n"
_SERIALIZER_SOURCE = (
    b"from redco.contracts import canonical_json\ndef serialize(payload, *, seed, cache_salt):\n"
    b"    return canonical_json({**payload, 'seed': seed, "
    b"'extra_body': {'cache_salt': cache_salt}})\n"
)
_SCORER_SOURCE = (
    b"import hashlib\nfrom redco.contracts import canonical_json\n"
    b"def score(*, task_attempt_id, task_id, seed, terminal_output_bytes, task_evidence_bytes):\n"
    b"    return canonical_json({'schema_version': 1, "
    b"'domain': 'redco-stage-d-heldout-score-v1', "
    b"'task_attempt_id': task_attempt_id, 'task_id': task_id, 'seed': seed, "
    b"'terminal_output_sha256': hashlib.sha256(terminal_output_bytes).hexdigest(), "
    b"'task_evidence_sha256': hashlib.sha256(task_evidence_bytes).hexdigest(), "
    b"'reward': 0.75, 'details': {}})\n"
)
_SERIALIZER_DECL = b"def serialize(payload, *, seed, cache_salt): "
_SCORER_DECL = (
    b"def score(*, task_attempt_id, task_id, seed, terminal_output_bytes, task_evidence_bytes): "
)
_SERIALIZER_ADAPTER = driver_module._RequestSerializerAdapter
_SCORER_ADAPTER = driver_module._TaskScorerAdapter


def _archive(tmp_path: Path, **modules: bytes) -> Path:
    path = tmp_path / "runtime.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for module, source in modules.items():
            archive.writestr(f"{module}.py", source)
    return path


def _entrypoint(
    component: str, module: str, callable_name: str, protocol: str, source: bytes
) -> EvaluationRuntimeEntrypoint:
    return EvaluationRuntimeEntrypoint(
        component,
        f"{module}.py",
        module,
        callable_name,
        protocol,
        hashlib.sha256(source).hexdigest(),
    )


def _manifest(
    tmp_path: Path,
    *,
    serializer_source: bytes = _SERIALIZER_SOURCE,
    scorer_source: bytes = _SCORER_SOURCE,
) -> StageDEvaluationExecutionManifest:
    _, base_bytes, _ = _frozen_inputs()
    bundle = _archive(
        tmp_path, task_runtime=_TASK_SOURCE, serializer=serializer_source, scorer=scorer_source
    )
    entrypoints = (
        _entrypoint(
            "task_runner", "task_runtime", "run_task", "redco-stage-d-worker-ipc-v1", _TASK_SOURCE
        ),
        _entrypoint("scorer", "scorer", "score", "redco-stage-d-scorer-v1", scorer_source),
        _entrypoint(
            "request_serializer",
            "serializer",
            "serialize",
            "redco-stage-d-request-serializer-v1",
            serializer_source,
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
        self.reserve_calls = 0

    def inspect(self) -> Any:
        return SimpleNamespace(tasks=tuple(self.tasks), current_task=self.current)

    def resume_current_client_session(self, _arm: str) -> object:
        return object()

    def reserve_next_task(self, *, session: object) -> EvaluationTaskAttempt:
        del session
        self.reserve_calls += 1
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


def _run(ledger: _Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> bytes:
    monkeypatch.setattr(driver_module, "DockerEvaluationRuntime", _Runtime)
    return run_evaluation_arm(
        ledger=ledger,  # type: ignore[arg-type]
        arm="stock",
        docker_executable=tmp_path / "unused-docker",
        docker_executable_sha256="0" * 64,
        limits=EvaluationDriverLimits(1.0, 2.0, 4, 128),
    )


def test_driver_uses_hash_bound_parent_scorer_and_completes_contiguous_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _Ledger(_manifest(tmp_path))
    assert _run(ledger, tmp_path, monkeypatch) == b"stock metrics"
    assert ledger.completed_reward == 0.75


def test_driver_rejects_changed_frozen_scorer_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    _archive(
        tmp_path,
        task_runtime=_TASK_SOURCE,
        serializer=_SERIALIZER_SOURCE,
        scorer=b"def score(**kwargs): return b'{}'\n",
    )
    ledger = _Ledger(manifest)
    with pytest.raises(ValueError, match="entrypoint changed"):
        _run(ledger, tmp_path, monkeypatch)


def test_authenticated_exec_boundary_preserves_namespace(tmp_path: Path) -> None:
    source = b"def inspect():\n    return __file__, __name__\n"
    archive_path = _archive(tmp_path, inspector=source)
    entrypoint = _entrypoint("scorer", "inspector", "inspect", "redco-stage-d-scorer-v1", source)
    loaded = driver_module._load_callable(archive_path, entrypoint)
    assert loaded() == (
        f"{archive_path}!/inspector.py",
        "_redco_frozen_scorer",
    )


def test_authenticated_exec_boundary_rejects_noncallable(tmp_path: Path) -> None:
    source = b"score = 1\n"
    archive_path = _archive(tmp_path, scorer=source)
    entrypoint = _entrypoint("scorer", "scorer", "score", "redco-stage-d-scorer-v1", source)
    with pytest.raises(ValueError, match="evaluation runtime entrypoint callable is absent"):
        driver_module._load_callable(archive_path, entrypoint)


def _call_adapter(component: str, value: object) -> bytes:
    if component == "serializer":
        entrypoint = _SERIALIZER_ADAPTER(
            lambda payload, *, seed, cache_salt: value
        )
        return entrypoint({}, seed=7, cache_salt="salt")
    entrypoint = _SCORER_ADAPTER(
        lambda *, task_attempt_id, task_id, seed, terminal_output_bytes, task_evidence_bytes: value
    )
    return entrypoint(
        task_attempt_id="attempt",
        task_id="task",
        seed=7,
        terminal_output_bytes=b"output",
        task_evidence_bytes=b"evidence",
    )


@pytest.mark.parametrize("component", ("serializer", "scorer"))
@pytest.mark.parametrize("value", [b"frozen", "text", bytearray(b"mutable"), None])
def test_dynamic_entrypoint_adapters_require_immutable_bytes(
    component: str, value: object
) -> None:
    if type(value) is bytes:
        assert _call_adapter(component, value) is value
    else:
        with pytest.raises(TypeError, match="must return immutable bytes"):
            _call_adapter(component, value)


@pytest.mark.parametrize(
    ("adapter", "entrypoint"),
    [
        (_SERIALIZER_ADAPTER, lambda: b"unused"),
        (_SERIALIZER_ADAPTER, lambda _payload, seed, cache_salt: b"unused"),
        (_SERIALIZER_ADAPTER, lambda payload, *, seed, cache_salt, extra: b"unused"),
        (_SERIALIZER_ADAPTER, lambda payload, *, seed, cache_salt="default": b"unused"),
        (_SERIALIZER_ADAPTER, lambda *_args: b"unused"),
        (_SERIALIZER_ADAPTER, partial(lambda payload, *, seed, cache_salt: b"unused")),
        (_SCORER_ADAPTER, lambda **_kwargs: b"unused"),
        (_SCORER_ADAPTER, lambda *, task_id: b"unused"),
    ],
)
def test_dynamic_entrypoint_adapter_rejects_wrong_signature(
    adapter: Callable[[Callable[..., object]], object],
    entrypoint: Callable[..., object],
) -> None:
    with pytest.raises(TypeError, match="signature differs from the frozen API"):
        adapter(entrypoint)


_INVALID_ENTRYPOINT_CASES = (
    ("serializer", "no_arguments"),
    ("serializer", "default_argument"),
    ("scorer", "variadic_keywords"),
    *(
        (component, variant)
        for component in ("serializer", "scorer")
        for variant in (
            "signature_spoof", "wrapped_spoof", "callable_class",
            "coroutine", "generator", "async_generator",
        )
    ),
)


def _invalid_entrypoint_source(component: str, variant: str) -> tuple[bytes, str]:
    direct = {
        "no_arguments": b"def serialize(): return b'{}'\n",
        "default_argument": (
            b"def serialize(payload, *, seed, cache_salt='default'): return b'{}'\n"
        ),
        "variadic_keywords": b"def score(**kwargs): return b'{}'\n",
    }
    if variant in direct:
        return direct[variant], "signature differs from the frozen API"

    name = "serialize" if component == "serializer" else "score"
    declaration = _SERIALIZER_DECL if component == "serializer" else _SCORER_DECL
    valid = declaration.replace(f"def {name}".encode(), b"def _valid") + b"return b'{}'\n"
    if variant == "callable_class":
        constructor = declaration.replace(f"def {name}(".encode(), b"def __init__(self, ", 1)
        return (
            f"class {name}:\n    ".encode()
            + constructor
            + b"pass\n    async def __call__(self, *args, **kwargs): return b'{}'\n"
        ), "signature differs from the frozen API"
    elif variant.endswith("_spoof"):
        source = valid + f"def {name}(*args, **kwargs): return b'{{}}'\n".encode()
        attribute = (
            "__signature__ = __import__('inspect').signature(_valid)"
            if variant == "signature_spoof"
            else "__wrapped__ = _valid"
        )
        source += f"{name}.{attribute}\n".encode()
        return source, "signature differs from the frozen API"
    prefix, body = {
        "coroutine": (b"async ", b"return b'{}'\n"),
        "generator": (b"", b"yield b'{}'\n"),
        "async_generator": (b"async ", b"yield b'{}'\n"),
    }[variant]
    return prefix + declaration + body, "must be synchronous"


@pytest.mark.parametrize(("component", "variant"), _INVALID_ENTRYPOINT_CASES)
def test_driver_rejects_invalid_authenticated_entrypoint_before_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    variant: str,
) -> None:
    source, message = _invalid_entrypoint_source(component, variant)
    sources = (
        (source, _SCORER_SOURCE) if component == "serializer" else (_SERIALIZER_SOURCE, source)
    )
    ledger = _Ledger(_manifest(tmp_path, serializer_source=sources[0], scorer_source=sources[1]))
    with pytest.raises(TypeError, match=message):
        _run(ledger, tmp_path, monkeypatch)
    assert (ledger.reserve_calls, ledger.current, ledger.tasks) == (0, None, [])
