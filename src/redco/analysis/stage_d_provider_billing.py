"""Canonical provider billing input for a terminal Stage-D report."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from redco.contracts import canonical_json

BillingPhase = Literal[
    "setup",
    "source",
    "scientific-replay",
    "training",
    "evaluation",
    "repair",
]
PricingType = Literal["on-demand", "spot"]
_PHASES = {
    "setup",
    "source",
    "scientific-replay",
    "training",
    "evaluation",
    "repair",
}
_PRICING_TYPES = {"on-demand", "spot"}


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.isprintable():
        raise ValueError(f"{name} must be a nonempty printable string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def rate_duration_estimate_micro_usd(
    *,
    rate_micro_usd_per_hour: int,
    billed_duration_milliseconds: int,
) -> int:
    """Round a rate-duration estimate upward to the nearest micro-USD."""
    rate = _integer(rate_micro_usd_per_hour, "hourly rate", minimum=1)
    duration = _integer(billed_duration_milliseconds, "billed duration")
    numerator = rate * duration
    denominator = 3_600_000
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True, slots=True)
class ProviderDeploymentBilling:
    attempt_id: str
    phase: BillingPhase
    provider: str
    resource_id: str
    location: str
    gpu_type: str
    gpu_count: int
    pricing_type: PricingType
    started_unix_milliseconds: int
    ended_unix_milliseconds: int
    billed_duration_milliseconds: int
    rate_micro_usd_per_hour: int
    rate_duration_estimate_micro_usd: int
    provider_charge_micro_usd: int
    provider_receipt_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "attempt_id",
            "provider",
            "resource_id",
            "location",
            "gpu_type",
        ):
            _text(getattr(self, name), name)
        if self.phase not in _PHASES or self.pricing_type not in _PRICING_TYPES:
            raise ValueError("provider billing phase or pricing type is invalid")
        _integer(self.gpu_count, "GPU count", minimum=1)
        start = _integer(self.started_unix_milliseconds, "deployment start")
        end = _integer(self.ended_unix_milliseconds, "deployment end")
        duration = _integer(self.billed_duration_milliseconds, "billed duration")
        if end < start or duration < end - start:
            raise ValueError("provider billing interval or duration is inconsistent")
        rate = _integer(self.rate_micro_usd_per_hour, "hourly rate", minimum=1)
        expected_estimate = rate_duration_estimate_micro_usd(
            rate_micro_usd_per_hour=rate,
            billed_duration_milliseconds=duration,
        )
        if self.rate_duration_estimate_micro_usd != expected_estimate:
            raise ValueError("provider rate-duration estimate differs")
        _integer(self.provider_charge_micro_usd, "provider charge")
        _sha256(self.provider_receipt_sha256, "provider receipt")

    def to_payload(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "phase": self.phase,
            "provider": self.provider,
            "resource_id": self.resource_id,
            "location": self.location,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "pricing_type": self.pricing_type,
            "started_unix_milliseconds": self.started_unix_milliseconds,
            "ended_unix_milliseconds": self.ended_unix_milliseconds,
            "billed_duration_milliseconds": self.billed_duration_milliseconds,
            "rate_micro_usd_per_hour": self.rate_micro_usd_per_hour,
            "rate_duration_estimate_micro_usd": self.rate_duration_estimate_micro_usd,
            "provider_charge_micro_usd": self.provider_charge_micro_usd,
            "provider_receipt_sha256": self.provider_receipt_sha256,
        }

    @classmethod
    def from_payload(cls, value: object) -> ProviderDeploymentBilling:
        fields = {
            "attempt_id",
            "phase",
            "provider",
            "resource_id",
            "location",
            "gpu_type",
            "gpu_count",
            "pricing_type",
            "started_unix_milliseconds",
            "ended_unix_milliseconds",
            "billed_duration_milliseconds",
            "rate_micro_usd_per_hour",
            "rate_duration_estimate_micro_usd",
            "provider_charge_micro_usd",
            "provider_receipt_sha256",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("provider deployment billing fields differ")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class StageDProviderBilling:
    currency: Literal["USD"]
    deployments: tuple[ProviderDeploymentBilling, ...]
    total_provider_charge_micro_usd: int
    total_rate_duration_estimate_micro_usd: int
    wallet_before_micro_usd: int
    wallet_after_micro_usd: int
    wallet_before_receipt_sha256: str
    wallet_after_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.currency != "USD" or not self.deployments:
            raise ValueError("provider billing currency or deployment roster differs")
        attempt_ids = tuple(item.attempt_id for item in self.deployments)
        if len(set(attempt_ids)) != len(attempt_ids):
            raise ValueError("provider billing deployment attempts are duplicated")
        if attempt_ids != tuple(sorted(attempt_ids)):
            raise ValueError("provider billing attempts must use deterministic order")
        expected_charge = sum(item.provider_charge_micro_usd for item in self.deployments)
        expected_estimate = sum(item.rate_duration_estimate_micro_usd for item in self.deployments)
        if (
            self.total_provider_charge_micro_usd != expected_charge
            or self.total_rate_duration_estimate_micro_usd != expected_estimate
        ):
            raise ValueError("provider billing totals differ from deployment entries")
        _integer(self.wallet_before_micro_usd, "wallet before")
        _integer(self.wallet_after_micro_usd, "wallet after")
        _sha256(self.wallet_before_receipt_sha256, "wallet-before receipt")
        _sha256(self.wallet_after_receipt_sha256, "wallet-after receipt")

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-provider-billing-v1",
                "currency": self.currency,
                "deployments": [item.to_payload() for item in self.deployments],
                "total_provider_charge_micro_usd": self.total_provider_charge_micro_usd,
                "total_rate_duration_estimate_micro_usd": (
                    self.total_rate_duration_estimate_micro_usd
                ),
                "wallet_before_micro_usd": self.wallet_before_micro_usd,
                "wallet_after_micro_usd": self.wallet_after_micro_usd,
                "wallet_before_receipt_sha256": self.wallet_before_receipt_sha256,
                "wallet_after_receipt_sha256": self.wallet_after_receipt_sha256,
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDProviderBilling:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("provider billing is not JSON") from error
        fields = {
            "schema_version",
            "domain",
            "currency",
            "deployments",
            "total_provider_charge_micro_usd",
            "total_rate_duration_estimate_micro_usd",
            "wallet_before_micro_usd",
            "wallet_after_micro_usd",
            "wallet_before_receipt_sha256",
            "wallet_after_receipt_sha256",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or payload.get("schema_version") != 1
            or payload.get("domain") != "redco-stage-d-provider-billing-v1"
            or not isinstance(payload.get("deployments"), list)
            or canonical_json(payload) != value
        ):
            raise ValueError("provider billing fields differ")
        return cls(
            payload["currency"],
            tuple(ProviderDeploymentBilling.from_payload(item) for item in payload["deployments"]),
            payload["total_provider_charge_micro_usd"],
            payload["total_rate_duration_estimate_micro_usd"],
            payload["wallet_before_micro_usd"],
            payload["wallet_after_micro_usd"],
            payload["wallet_before_receipt_sha256"],
            payload["wallet_after_receipt_sha256"],
        )

    def verify_receipts(self, receipt_bytes: dict[str, bytes]) -> None:
        expected = {
            self.wallet_before_receipt_sha256,
            self.wallet_after_receipt_sha256,
            *(item.provider_receipt_sha256 for item in self.deployments),
        }
        if set(receipt_bytes) != expected:
            raise ValueError("provider billing receipt roster differs")
        for digest, value in receipt_bytes.items():
            if hashlib.sha256(value).hexdigest() != digest:
                raise ValueError("provider billing receipt bytes differ")


__all__ = [
    "ProviderDeploymentBilling",
    "StageDProviderBilling",
    "rate_duration_estimate_micro_usd",
]
