from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from redco.analysis.gate_gb_aggregate import (
    _gpu_resource_metrics,
    _paired_equivalence_interval,
    _verify_signed_report,
)
from redco.contracts import canonical_json


def test_paired_live_interval_matches_frozen_4096_pair_design() -> None:
    interval = _paired_equivalence_interval(
        [0.0] * 4096,
        confidence=0.9,
        margin=0.04,
    )

    assert interval.mean_difference == 0
    assert interval.half_width < 0.04
    assert interval.passed

    target = _paired_equivalence_interval(
        [0.0] * 1024,
        confidence=0.9,
        margin=0.08,
    )
    assert target.half_width < 0.08
    assert target.passed


def test_gpu_resource_metrics_integrate_left_endpoint_samples(
    tmp_path: Path,
) -> None:
    samples = tmp_path / "gpu.csv"
    samples.write_text(
        "100.0, 50, 1000, 200\n"
        "102.0, 100, 2000, 300\n"
        "104.0, 0, 1500, 100\n",
        encoding="utf-8",
    )

    metrics = _gpu_resource_metrics(samples, minimum_samples=3)

    assert metrics.elapsed_seconds == 4
    assert metrics.utilization_weighted_gpu_seconds == 3
    assert metrics.mean_utilization_fraction == 0.75
    assert metrics.peak_memory_mib == 2000
    assert metrics.mean_power_watts == 250
    assert metrics.energy_watt_hours == pytest.approx(1000 / 3600)
    assert metrics.passed_minimum_samples


def test_signed_report_verification_detects_mutation() -> None:
    payload: dict[str, object] = {"schema_version": 1, "passed": True}
    payload["report_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()

    assert _verify_signed_report(payload)
    payload["passed"] = False
    assert not _verify_signed_report(payload)
