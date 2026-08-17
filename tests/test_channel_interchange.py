from __future__ import annotations

import math
from typing import cast

import pytest

from redco.analysis.channel_interchange import (
    InterchangeEffects,
    ProbeKind,
    Representation,
    _observe_ambient_context,
    _read_declared_artifact,
    build_report,
    compact_result,
    evaluate_probe,
)


@pytest.mark.parametrize(
    ("kind", "cells", "allocations", "interaction"),
    [
        (ProbeKind.ARTIFACT_ONLY, (0.0, 1.0, 0.0, 1.0), (1.0, 0.0), 0.0),
        (ProbeKind.CONTEXT_ONLY, (0.0, 0.0, 1.0, 1.0), (0.0, 1.0), 0.0),
        (ProbeKind.REDUNDANT, (0.0, 1.0, 1.0, 1.0), (0.5, 0.5), -1.0),
        (ProbeKind.SYNERGISTIC, (0.0, 0.0, 0.0, 1.0), (0.5, 0.5), 1.0),
    ],
)
def test_planted_probe_recovers_channel_effects(
    kind: ProbeKind,
    cells: tuple[float, float, float, float],
    allocations: tuple[float, float],
    interaction: float,
) -> None:
    effects, executions = evaluate_probe(kind, Representation.CANONICAL)

    assert effects.cells == cells
    assert (effects.artifact_allocation, effects.context_allocation) == allocations
    assert effects.interaction == interaction
    assert effects.conservation_error == 0.0
    assert all(item.artifact_read and item.ambient_included for item in executions.values())


@pytest.mark.parametrize("kind", list(ProbeKind))
def test_effects_are_stable_under_equivalent_representation(kind: ProbeKind) -> None:
    canonical, canonical_cells = evaluate_probe(kind, Representation.CANONICAL)
    equivalent, equivalent_cells = evaluate_probe(kind, Representation.EQUIVALENT)

    assert canonical == equivalent
    assert {cell.artifact_sha256 for cell in canonical_cells.values()} != {
        cell.artifact_sha256 for cell in equivalent_cells.values()
    }
    assert {cell.ambient_sha256 for cell in canonical_cells.values()} != {
        cell.ambient_sha256 for cell in equivalent_cells.values()
    }


def test_mixed_cells_record_prompt_geometry_without_claiming_token_counts() -> None:
    _, cells = evaluate_probe(ProbeKind.REDUNDANT, Representation.CANONICAL)
    geometry = cells["r00"].prompt_geometry
    assert geometry["tokenizer_used"] is False
    assert geometry["unrelated_tokens_displaced"] is None
    assert geometry["contract_equivalent"] is True
    assert geometry["utf8_byte_delta"] == 2


def test_report_passes_only_the_deterministic_measurement_gate() -> None:
    report = build_report()
    payload = cast(dict[str, object], report["payload"])

    assert payload["passed"] is True
    assert payload["training_performed"] is False
    assert "not evidence that an LLM learns" in cast(str, payload["interpretation"])
    assert set(cast(dict[str, bool], payload["gates"]).values()) == {True}

    compact = compact_result(
        report,
        source_path="runs/channel-interchange-kill-test-v1/report.json",
        source_raw_sha256="a" * 64,
    )
    compact_payload = cast(dict[str, object], compact["payload"])
    assert compact_payload["passed"] is True
    probes = cast(list[dict[str, object]], compact_payload["probes"])
    assert len(probes) == 4
    assert all("cells" not in probe for probe in probes)


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"[]",
        b'{"schema_version":1}',
        b'{"schema_version":1,"signal":1}',
        b'{"schema_version":1,"signal":true,"extra":false}',
    ],
)
def test_declared_artifact_contract_rejects_malformed_values(raw: bytes) -> None:
    with pytest.raises(ValueError, match="declared artifact"):
        _read_declared_artifact(raw)


@pytest.mark.parametrize("text", ["true", "ambient=true", "ambient_signal=1", ""])
def test_ambient_contract_rejects_malformed_values(text: str) -> None:
    with pytest.raises(ValueError, match="ambient context"):
        _observe_ambient_context(text)


def test_effects_reject_nonfinite_rewards() -> None:
    with pytest.raises(ValueError, match="finite"):
        InterchangeEffects(0.0, 1.0, math.nan, 1.0)
