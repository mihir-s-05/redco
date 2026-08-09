"""Source-free tests for the non-authorizing Prime inventory evidence v2 owner."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes
from redco.analysis.stage_d_v13_prime_inventory_v2 import (
    AUDIT_RELATIVE,
    CONTRACT_RELATIVE,
    PRIME_READ_ONLY_COMMANDS,
    V1_AUDIT_RELATIVE,
    V1_AUDIT_SHA256,
    V1_DEPENDENCY_RELATIVE,
    V1_DEPENDENCY_SHA256,
    V1_OWNER_RELATIVE,
    V1_OWNER_SHA256,
    V1_READINESS_RELATIVE,
    V1_READINESS_SHA256,
    V12_DEPENDENCY_RELATIVE,
    V12_DEPENDENCY_SHA256,
    PrimeInventoryObservationProducerV2,
    build_prime_inventory_v2_artifacts,
    validate_future_prime_readiness_v2,
    validate_prime_inventory_observation_v2,
)
from redco.analysis.stage_d_v13_support_readiness import ReadinessBlocked

ROOT = Path(__file__).parents[1].resolve()


def _resource(
    *,
    resource_id: str = "resource-001",
    provider: str = "massedcompute",
    gpu_type: str = "L40S 48GB",
    gpu_count: int = 2,
    gpu_memory: int = 96,
    price: object = "$1.50",
    price_value: object = 1.5,
    stock: str = "Available",
    is_spot: object = False,
) -> dict[str, object]:
    return {
        "id": resource_id,
        "cloud_id": "cloud-001",
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "socket": "socket-0",
        "provider": provider,
        "location": "us-test-1",
        "stock_status": stock,
        "price_per_hour": price,
        "price_value": price_value,
        "security": {"ssh": True},
        "vcpus": 16,
        "memory_gb": 128,
        "disk_gb": 128,
        "gpu_memory": gpu_memory,
        "is_spot": is_spot,
    }


def _outputs(resources: list[dict[str, object]]) -> dict[str, bytes]:
    return {
        "version": b"Prime CLI version: 0.6.20\n",
        "wallet": canonical_json_bytes(
            {
                "wallet_id": "wallet-1",
                "team_id": "team-1",
                "balance_usd": 30,
                "currency": "USD",
                "total_billings": 0,
                "recent_billings": [],
            }
        ),
        "pods": canonical_json_bytes({"pods": [], "total_count": 0, "offset": 0, "limit": 100}),
        "disks": canonical_json_bytes({"disks": [], "total_count": 0, "offset": 0, "limit": 100}),
        "availability": canonical_json_bytes(
            {
                "gpu_resources": resources,
                "total_count": len(resources),
                "filters": {"gpu_count": 2, "gpu_type": "L40S_48GB"},
            }
        ),
    }


def _capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resources: list[dict[str, object]],
    *,
    captured_at: int = 1_800_000_000,
) -> tuple[Path, dict[str, Any]]:
    outputs = _outputs(resources)

    def run_command(
        _self: PrimeInventoryObservationProducerV2,
        argv: tuple[str, ...],
    ) -> subprocess.CompletedProcess[bytes]:
        name = next(name for name, command in PRIME_READ_ONLY_COMMANDS.items() if command == argv)
        return subprocess.CompletedProcess(argv, 0, outputs[name], b"")

    monkeypatch.setattr(PrimeInventoryObservationProducerV2, "_run_command", run_command)
    path = tmp_path / "observation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        PrimeInventoryObservationProducerV2(ROOT).capture(captured_at_epoch=captured_at)
    )
    return path, validate_prime_inventory_observation_v2(ROOT, path, now_epoch=captured_at)


def test_exact_live_shape_is_qualifying_only_with_false_non_spot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path, value = _capture(monkeypatch, tmp_path, [_resource()])
    assert value["state"] == "observed_qualifying_resource"
    assert value["resource"]["provider"] == "massedcompute"
    assert value["resource"]["hardware"] == "L40S 48GB"
    assert value["resource"]["aggregate_gpu_memory_gb"] == 96.0
    assert value["resource"]["hourly_rate_usd"] == 1.5
    assert value["resource"]["price_value_usd"] == 1.5
    assert value["resource"]["hourly_rate_cents"] == 150
    assert value["resource"]["raw"]["price_per_hour"] == "$1.50"
    assert value["resource"]["raw"]["price_value"] == 1.5
    assert path.read_bytes() == canonical_json_bytes(value)


def test_null_non_spot_is_ambiguous_and_never_selected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, value = _capture(monkeypatch, tmp_path, [_resource(is_spot=None)])
    assert value["state"] == "observed_ambiguous_resources"
    assert value["resource"] is None
    assert value["availability"]["eligible_count"] == 0
    assert value["availability"]["unknown_non_spot_count"] == 1


def test_zero_and_multiple_capacity_are_explicitly_non_authorizing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, zero = _capture(
        monkeypatch,
        tmp_path / "zero",
        [_resource(price="$2.01", price_value=2.01)],
    )
    assert zero["state"] == "observed_no_qualifying_resource"
    assert zero["resource"] is None
    _, multiple = _capture(
        monkeypatch,
        tmp_path / "multiple",
        [_resource(), _resource(resource_id="resource-002")],
    )
    assert multiple["state"] == "observed_ambiguous_resources"
    assert multiple["resource"] is None
    assert all(value is False for value in multiple["authorization"].values())
    _, duplicate = _capture(
        monkeypatch,
        tmp_path / "duplicate",
        [_resource(), _resource()],
    )
    assert duplicate["state"] == "observed_ambiguous_resources"
    assert duplicate["resource"] is None


def test_strict_numeric_and_dollar_prices_are_accepted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    price_pairs: tuple[tuple[object, object], ...] = (
        (1.5, 1.5),
        (2, 2),
        ("$0.00", 0),
        ("$2.00", 2.0),
    )
    for index, (price, price_value) in enumerate(price_pairs):
        _, value = _capture(
            monkeypatch,
            tmp_path / str(index),
            [_resource(price=price, price_value=price_value)],
        )
        assert value["state"] == "observed_qualifying_resource"


def test_malformed_prices_fail_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad_pairs: tuple[tuple[object, object], ...] = (
        (True, 1.5),
        ("$1.5", 1.5),
        ("1.50", 1.5),
        ("$01.50", 1.5),
        ("nan", 1.5),
        ("$1.501", 1.501),
        ("$1.51", 1.5),
        ("$1.50", 1.51),
        ("$1.50", 999.0),
        ("$1.50", True),
        ("$1.50", "1.50"),
    )
    for index, (price, price_value) in enumerate(bad_pairs):
        output = tmp_path / str(index) / "observation.json"
        outputs = _outputs([_resource(price=price, price_value=price_value)])

        def run_command(
            _self: PrimeInventoryObservationProducerV2,
            argv: tuple[str, ...],
            output_map: dict[str, bytes] = outputs,
        ) -> subprocess.CompletedProcess[bytes]:
            name = next(
                name for name, command in PRIME_READ_ONLY_COMMANDS.items() if command == argv
            )
            return subprocess.CompletedProcess(argv, 0, output_map[name], b"")

        monkeypatch.setattr(PrimeInventoryObservationProducerV2, "_run_command", run_command)
        with pytest.raises(ValueError, match="price"):
            PrimeInventoryObservationProducerV2(ROOT).capture(captured_at_epoch=1_800_000_000)
        assert not output.exists()


def test_malformed_resources_fail_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"provider": ""}, "provider"),
        ({"gpu_memory": 48}, "memory"),
        ({"is_spot": True}, "spot"),
        ({"gpu_count": 2, "gpu_memory": 95}, "memory"),
        ({"unknown": None}, "schema"),
    )
    for index, (mutation, match) in enumerate(cases):
        resource = _resource()
        resource.update(mutation)
        output = tmp_path / str(index) / "observation.json"
        outputs = _outputs([resource])

        def run_command(
            _self: PrimeInventoryObservationProducerV2,
            argv: tuple[str, ...],
            output_map: dict[str, bytes] = outputs,
        ) -> subprocess.CompletedProcess[bytes]:
            name = next(
                name for name, command in PRIME_READ_ONLY_COMMANDS.items() if command == argv
            )
            return subprocess.CompletedProcess(argv, 0, output_map[name], b"")

        monkeypatch.setattr(PrimeInventoryObservationProducerV2, "_run_command", run_command)
        with pytest.raises(ValueError, match=match):
            PrimeInventoryObservationProducerV2(ROOT).capture(captured_at_epoch=1_800_000_000)
        assert not output.exists()


def test_canonical_and_raw_binding_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mutations = (
        ("state", "observed_no_qualifying_resource"),
        ("resource", None),
        ("authorization", {"prime_authorized": True}),
        ("commands.wallet.stdout_sha256", "0" * 64),
        ("commands.availability.argv", ["prime", "availability"]),
    )
    for index, (key, replacement) in enumerate(mutations):
        path, value = _capture(monkeypatch, tmp_path / str(index), [_resource()])
        changed = deepcopy(value)
        cursor: dict[str, Any] = changed
        pieces = key.split(".")
        for piece in pieces[:-1]:
            cursor = cursor[piece]
        cursor[pieces[-1]] = replacement
        path.write_bytes(canonical_json_bytes(changed))
        with pytest.raises(ValueError):
            validate_prime_inventory_observation_v2(ROOT, path, now_epoch=1_800_000_000)


def test_checkout_ttl_and_unknown_field_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path, value = _capture(monkeypatch, tmp_path, [_resource()])
    for mutate, now in (
        (lambda item: item.update({"unknown": None}), 1_800_000_000),
        (lambda item: item["bundle"].update({"commit": "0" * 40}), 1_800_000_000),
        (lambda _item: None, 1_800_000_901),
    ):
        changed = deepcopy(value)
        mutate(changed)
        path.write_bytes(canonical_json_bytes(changed))
        with pytest.raises(ValueError):
            validate_prime_inventory_observation_v2(ROOT, path, now_epoch=now)


def test_future_gate_is_fixed_no_argument_and_missing_root_blocks() -> None:
    assert validate_future_prime_readiness_v2.__code__.co_argcount == 0
    with pytest.raises((ValueError, FileNotFoundError, ReadinessBlocked)):
        validate_future_prime_readiness_v2()


def test_historical_v1_and_v12_bindings_are_unchanged() -> None:
    expected = {
        V1_OWNER_RELATIVE: V1_OWNER_SHA256,
        V12_DEPENDENCY_RELATIVE: V12_DEPENDENCY_SHA256,
        V1_READINESS_RELATIVE: V1_READINESS_SHA256,
        V1_DEPENDENCY_RELATIVE: V1_DEPENDENCY_SHA256,
        V1_AUDIT_RELATIVE: V1_AUDIT_SHA256,
    }
    assert {
        relative: sha256_bytes((ROOT / relative).read_bytes()) for relative in expected
    } == expected


def test_artifact_build_is_deterministic_and_non_authorizing() -> None:
    first = build_prime_inventory_v2_artifacts(ROOT)
    second = build_prime_inventory_v2_artifacts(ROOT)
    assert first == second
    assert set(first) == {CONTRACT_RELATIVE, AUDIT_RELATIVE}
    contract = json.loads(first[CONTRACT_RELATIVE])
    audit = json.loads(first[AUDIT_RELATIVE])
    assert contract["parent"]["commit"] == ("d3884673faba6dc63916b74960d6a4b5cb691406")
    assert all(value is False for value in contract["authorization"].values())
    assert all(value is False for value in audit["authorization"].values())
    assert audit["raw_wallet_or_billing_tracked"] is False
