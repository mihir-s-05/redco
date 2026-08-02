from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from redco.analysis.stage_d_provider_billing import (
    ProviderDeploymentBilling,
    StageDProviderBilling,
    rate_duration_estimate_micro_usd,
)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _entry() -> ProviderDeploymentBilling:
    duration = 1_800_001
    rate = 2_000_000
    return ProviderDeploymentBilling(
        attempt_id="stage-d-attempt-001",
        phase="training",
        provider="prime-intellect",
        resource_id="resource-1",
        location="us-west",
        gpu_type="L40S",
        gpu_count=2,
        pricing_type="on-demand",
        started_unix_milliseconds=1_000_000,
        ended_unix_milliseconds=2_800_000,
        billed_duration_milliseconds=duration,
        rate_micro_usd_per_hour=rate,
        rate_duration_estimate_micro_usd=rate_duration_estimate_micro_usd(
            rate_micro_usd_per_hour=rate,
            billed_duration_milliseconds=duration,
        ),
        provider_charge_micro_usd=1_111_111,
        provider_receipt_sha256=_digest(b"deployment receipt"),
    )


def _billing() -> tuple[StageDProviderBilling, dict[str, bytes]]:
    entry = _entry()
    before = b"wallet before"
    after = b"wallet after"
    deployment = b"deployment receipt"
    billing = StageDProviderBilling(
        "USD",
        (entry,),
        entry.provider_charge_micro_usd,
        entry.rate_duration_estimate_micro_usd,
        40_000_000,
        38_999_999,
        _digest(before),
        _digest(after),
    )
    return billing, {
        _digest(before): before,
        _digest(after): after,
        _digest(deployment): deployment,
    }


def test_provider_billing_roundtrips_and_verifies_exact_receipts() -> None:
    billing, receipts = _billing()
    assert StageDProviderBilling.from_bytes(billing.to_bytes()) == billing
    billing.verify_receipts(receipts)
    with pytest.raises(ValueError, match="receipt roster"):
        billing.verify_receipts({})


def test_rate_estimate_rounds_up_without_float_arithmetic() -> None:
    assert (
        rate_duration_estimate_micro_usd(
            rate_micro_usd_per_hour=1,
            billed_duration_milliseconds=1,
        )
        == 1
    )
    entry = _entry()
    with pytest.raises(ValueError, match="estimate differs"):
        replace(entry, rate_duration_estimate_micro_usd=0)


def test_provider_charge_is_not_forced_to_equal_rate_estimate() -> None:
    entry = _entry()
    assert entry.provider_charge_micro_usd != entry.rate_duration_estimate_micro_usd


def test_billing_rejects_duplicate_attempts_and_float_currency() -> None:
    billing, _ = _billing()
    with pytest.raises(ValueError, match="duplicated"):
        replace(billing, deployments=(billing.deployments[0], billing.deployments[0]))
    with pytest.raises(ValueError, match="integer"):
        replace(billing.deployments[0], provider_charge_micro_usd=1.5)  # type: ignore[arg-type]
