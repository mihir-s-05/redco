from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from redco.analysis import stage_d_branch_group
from redco.analysis.empirical_branch_replay import (
    DeterministicReplayIneligibility,
)
from redco.integrations.signed_subprocess import verify_signed_payload


def _run(tmp_path: Path) -> tuple[dict[str, object], bool]:
    return stage_d_branch_group.run_group(
        trace_path=tmp_path / "trace.jsonl",
        client=object(),  # type: ignore[arg-type]
        master_seed="master",
        temperature=0.7,
        candidate_max_tokens=512,
        continuation_max_tokens=768,
    )


def test_deterministic_replay_negative_is_a_signed_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        stage_d_branch_group,
        "load_trace_records",
        lambda _: [{"id": "trace-1"}],
    )
    monkeypatch.setattr(
        stage_d_branch_group,
        "audit_trace_file",
        lambda _: SimpleNamespace(calls=[object()]),
    )

    def fail(**_: object) -> None:
        raise DeterministicReplayIneligibility(
            "fixed-topology splice is unavailable"
        )

    monkeypatch.setattr(stage_d_branch_group, "run_empirical_replay", fail)
    report, replayable = _run(tmp_path)

    assert replayable is False
    assert report["status"] == "deterministic_ineligible"
    assert report["trace_id"] == "trace-1"
    verify_signed_payload(report)


def test_infrastructure_error_is_not_converted_to_negative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        stage_d_branch_group,
        "load_trace_records",
        lambda _: [{"id": "trace-1"}],
    )
    monkeypatch.setattr(
        stage_d_branch_group,
        "audit_trace_file",
        lambda _: SimpleNamespace(calls=[object()]),
    )

    def fail(**_: object) -> None:
        raise OSError("network unavailable")

    monkeypatch.setattr(stage_d_branch_group, "run_empirical_replay", fail)
    with pytest.raises(OSError, match="network unavailable"):
        _run(tmp_path)


def test_malformed_response_is_not_converted_to_negative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        stage_d_branch_group,
        "load_trace_records",
        lambda _: [{"id": "trace-1"}],
    )
    monkeypatch.setattr(
        stage_d_branch_group,
        "audit_trace_file",
        lambda _: SimpleNamespace(calls=[object()]),
    )

    def fail(**_: object) -> None:
        raise ValueError("malformed inference JSON")

    monkeypatch.setattr(stage_d_branch_group, "run_empirical_replay", fail)
    with pytest.raises(ValueError, match="malformed inference JSON"):
        _run(tmp_path)
