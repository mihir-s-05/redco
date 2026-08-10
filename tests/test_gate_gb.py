from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import redco.analysis.gate_gb as gate_gb

_NAME_ERROR = "trace file name must be a single path component"
_INVALID_NAMES = ("", "../report.json", "nested/report.json") + (
    (r"nested\report.json",) if os.name == "nt" else ()
)


class _StubGate:
    def __init__(self, passed: bool) -> None:
        self.passed_cpu_gate, self.signed_dict_calls = passed, 0

    def signed_dict(self) -> dict[str, object]:
        self.signed_dict_calls += 1
        return {"z": 1, "a": 2}

    def __call__(
        self, *, seed: int, programs: int, interventions_per_program: int, events: int
    ) -> _StubGate:
        assert (seed, programs, interventions_per_program, events) == (20260725, 500, 20, 16)
        return self


def test_cpu_gate_exercises_all_dependencies_and_passes() -> None:
    report = gate_gb.run_gate(seed=19, programs=100, interventions_per_program=10, events=12)
    assert report.passed_cpu_gate and report.snapshot_roundtrip_exact
    assert report.interventions == 1_000
    assert report.deterministic_failures == 0
    assert all(count > 0 for count in report.dependency_edge_counts.values())
    assert report.event_raf > 1.0


def test_snapshot_roundtrip_uses_verified_content_addressed_bytes() -> None:
    assert gate_gb._audit_snapshot_roundtrip() is True


@pytest.mark.parametrize(
    ("existing", "payload", "expected"),
    [
        (None, {"z": 1, "a": 2}, b'{"a":2,"z":1}\n'),
        (b"stale", {"current": True}, b'{"current":true}\n'),
    ],
)
def test_gate_report_writer_publishes_one_canonical_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    existing: bytes | None,
    payload: dict[str, object],
    expected: bytes,
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("reports")
    if existing is not None:
        root.mkdir()
        (root / "report.json").write_bytes(existing)
    path = gate_gb._write_gate_report(root, "report.json", payload)
    data = path.read_bytes()
    assert path.is_absolute() and path == (tmp_path / "reports/report.json").resolve()
    assert data == expected
    assert data.count(b"\n") == 1 and b"\r" not in data
    assert {item.name for item in path.parent.iterdir()} == {"report.json"}


@pytest.mark.parametrize("name", _INVALID_NAMES)
def test_gate_report_writer_rejects_name_after_creating_root(tmp_path: Path, name: str) -> None:
    root = tmp_path / "created-before-failure"
    with pytest.raises(ValueError) as captured:
        gate_gb._write_gate_report(root, name, {"valid": True})
    assert str(captured.value) == _NAME_ERROR
    assert root.is_dir() and not tuple(root.iterdir())


def test_gate_report_writer_serializes_after_creating_root(tmp_path: Path) -> None:
    root = tmp_path / "created-before-failure"
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant: nan"):
        gate_gb._write_gate_report(root, "", {"invalid": float("nan")})
    assert root.is_dir() and not tuple(root.iterdir())


def test_gate_report_writer_preserves_files_on_replace_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    target = root / "report.json"
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    target.write_bytes(b"existing")

    def fail_replace(source: Path, destination: Path) -> None:
        assert (source, destination) == (temporary, target)
        raise PermissionError("replace blocked")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="replace blocked"):
        gate_gb._write_gate_report(root, target.name, {"complete": True})
    assert target.read_bytes() == b"existing"
    assert temporary.read_bytes() == b'{"complete":true}\n'


@pytest.mark.parametrize(("passed", "expected_exit"), [(True, 0), (False, 1)])
def test_main_preserves_cheap_cli_contract(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    passed: bool,
    expected_exit: int,
) -> None:
    stub = _StubGate(passed)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gate_gb, "run_gate", stub)
    monkeypatch.setattr(sys, "argv", ["gate-gb"])
    assert gate_gb.main() == expected_exit

    output = (tmp_path / "runs/stage-b/gate-gb-cpu/report.json").resolve()
    assert stub.signed_dict_calls == 2
    assert output.read_bytes() == b'{"a":2,"z":1}\n'
    expected_stdout = '{\n  "a": 2,\n  "z": 1\n}\n' + f"wrote {output}\n"
    assert capsys.readouterr().out == expected_stdout
