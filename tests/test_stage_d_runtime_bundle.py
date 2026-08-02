from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from dataclasses import replace

import pytest
from test_stage_d_evaluation_ledger import _frozen_inputs

from redco.analysis.stage_d_evaluation_contracts import (
    EvaluationRuntimeEntrypoint,
    StageDEvaluationExecutionManifest,
)
from redco.analysis.stage_d_runtime_bundle import verify_evaluation_runtime_bundle


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _archive(entries: tuple[tuple[str, bytes], ...]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries:
            archive.writestr(name, value)
    return output.getvalue()


def _bound_manifest(runtime: bytes) -> StageDEvaluationExecutionManifest:
    _, manifest_bytes, _ = _frozen_inputs()
    manifest = StageDEvaluationExecutionManifest.from_bytes(manifest_bytes)
    sources = {"server": b"server", "client": b"client"}
    programs = tuple(
        replace(program, source_sha256s=((f"{program.role}.py", _sha(sources[program.role])),))
        for program in manifest.programs
    )
    return replace(
        manifest,
        runtime_entrypoints=(
            EvaluationRuntimeEntrypoint(
                "task_runner",
                "task_runtime.py",
                "task_runtime",
                "run_task",
                "redco-stage-d-worker-ipc-v1",
                _sha(b"task runtime"),
            ),
            EvaluationRuntimeEntrypoint(
                "scorer", "scorer.py", "scorer", "score", "redco-stage-d-scorer-v1", _sha(b"scorer")
            ),
            EvaluationRuntimeEntrypoint(
                "request_serializer",
                "serializer.py",
                "serializer",
                "serialize",
                "redco-stage-d-request-serializer-v1",
                _sha(b"serializer"),
            ),
        ),
        runtime_bundle_sha256=_sha(runtime),
        programs=programs,
    )


def test_runtime_bundle_verifies_exact_source_roster() -> None:
    runtime = _archive(
        (
            ("client.py", b"client"),
            ("scorer.py", b"scorer"),
            ("serializer.py", b"serializer"),
            ("server.py", b"server"),
            ("task_runtime.py", b"task runtime"),
        )
    )
    observed = dict(verify_evaluation_runtime_bundle(runtime, manifest=_bound_manifest(runtime)))
    assert observed["scorer.py"] == _sha(b"scorer")


@pytest.mark.parametrize("unsafe_name", ["../escape.py", "/absolute.py"])
def test_runtime_bundle_rejects_unsafe_members(unsafe_name: str) -> None:
    runtime = _archive(((unsafe_name, b"bad"),))
    with pytest.raises(ValueError, match="unsafe"):
        verify_evaluation_runtime_bundle(runtime, manifest=_bound_manifest(runtime))


def test_runtime_bundle_rejects_symlink_member() -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        member = zipfile.ZipInfo("link.py")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(member, "target.py")
    runtime = output.getvalue()
    with pytest.raises(ValueError, match="special"):
        verify_evaluation_runtime_bundle(runtime, manifest=_bound_manifest(runtime))
