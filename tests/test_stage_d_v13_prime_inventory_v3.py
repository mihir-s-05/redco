"""Source-free tests for immutable Prime raw receipts and v3 assessment."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from redco.analysis import stage_d_v13_prime_inventory_v2 as inventory_v2
from redco.analysis import stage_d_v13_prime_inventory_v3 as inventory
from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes

ROOT = Path(__file__).parents[1].resolve()
NOW = 1_800_000_000


def _resource(
    *,
    resource_id: str = "resource-001",
    disk: object = "128",
    price: object = "$1.50",
    price_value: object = 1.5,
    is_spot: object = False,
) -> dict[str, object]:
    return {
        "id": resource_id,
        "cloud_id": "cloud-001",
        "gpu_type": "L40S 48GB",
        "gpu_count": 2,
        "socket": "PCIe",
        "provider": "massedcompute",
        "location": "united_states",
        "stock_status": "Available",
        "price_per_hour": price,
        "price_value": price_value,
        "security": "datacenter",
        "vcpus": "16",
        "memory_gb": "128",
        "disk_gb": disk,
        "gpu_memory": 96,
        "is_spot": is_spot,
    }


def _outputs(
    resources: list[dict[str, object]],
    *,
    availability: bytes | None = None,
    wallet_balance: float = 30.01,
) -> dict[str, bytes]:
    live_availability = canonical_json_bytes(
        {
            "gpu_resources": resources,
            "total_count": len(resources),
            "filters": {
                "gpu_count": 2,
                "gpu_type": "L40S_48GB",
                "regions": None,
                "socket": None,
                "provider": None,
                "group_similar": True,
            },
        }
    )
    return {
        "version": b"Prime CLI version: 0.6.20\n",
        "wallet": canonical_json_bytes(
            {
                "wallet_id": "wallet-1",
                "team_id": "team-1",
                "balance_usd": wallet_balance,
                "currency": "USD",
                "total_billings": 0,
                "recent_billings": [],
            }
        ),
        "pods": canonical_json_bytes({"pods": [], "total_count": 0, "offset": 0, "limit": 100}),
        "disks": canonical_json_bytes({"disks": [], "total_count": 0, "offset": 0, "limit": 100}),
        "availability": live_availability if availability is None else availability,
    }


def _bind_root(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    outputs: dict[str, bytes],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(inventory, "ROOT", root)
    monkeypatch.setattr(
        "redco.analysis.stage_d_v13_prime_inventory_v3.time.time",
        lambda: float(NOW),
    )
    monkeypatch.setattr(
        inventory,
        "_git",
        lambda _root, *args: "1" * 40 if args[-1] == "HEAD" else "2" * 40,
    )

    def run_command(argv: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
        logical_argv = ("prime", *argv[1:])
        name = next(
            name
            for name, command in inventory_v2.PRIME_READ_ONLY_COMMANDS.items()
            if command == logical_argv
        )
        return subprocess.CompletedProcess(argv, 0, outputs[name], b"")

    monkeypatch.setattr(inventory, "_run_prime_command", run_command)


def _capture_and_assess(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    resources: list[dict[str, object]],
) -> tuple[bytes, dict[str, Any]]:
    _bind_root(monkeypatch, root, _outputs(resources))
    raw = inventory.capture_prime_inventory_raw_v3()
    inventory.assess_prime_inventory_v3()
    assessment = inventory.validate_prime_inventory_assessment_v3(now_epoch=NOW)
    return raw, assessment


def test_raw_receipt_survives_semantic_unknown_and_assessment_is_separate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bind_root(monkeypatch, tmp_path, _outputs([], availability=b"not-json"))
    raw = inventory.capture_prime_inventory_raw_v3()
    raw_path = tmp_path / inventory.RAW_RELATIVE
    assert raw_path.read_bytes() == raw
    receipt = json.loads(raw)
    executable = inventory.authenticate_installed_prime_executable()
    assert receipt["cli"]["installed_executable"] == executable
    assert all(
        capture["executed_argv"][0] == executable["canonical_path"]
        for capture in receipt["commands"].values()
    )
    assessment = json.loads(inventory.assess_prime_inventory_v3())
    assert assessment["state"] == "observed_schema_unknown"
    assert assessment["resource"] is None
    assert all(value is False for value in assessment["authorization"].values())
    assert raw_path.read_bytes() == raw
    assert (tmp_path / inventory.ASSESSMENT_RELATIVE).is_file()


def test_only_exact_installed_disk_string_grammar_can_qualify(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for index, disk in enumerate(("128", "128+")):
        _raw, assessment = _capture_and_assess(
            monkeypatch, tmp_path / str(index), [_resource(disk=disk)]
        )
        assert assessment["state"] == "observed_qualifying_resource"
        capability = assessment["resource"]["disk_capability"]
        assert capability["raw"] == disk
        assert capability["minimum_ephemeral_gb"] == 128
        assert capability["expandable"] is disk.endswith("+")


def test_numeric_null_malformed_unknown_and_oversized_disk_never_qualify(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    values: tuple[object, ...] = (
        128,
        None,
        "unknown",
        "-1",
        "01",
        "1.0",
        "1000000",
        "9" * 33,
        {"default_count": 128},
    )
    for index, disk in enumerate(values):
        raw, assessment = _capture_and_assess(
            monkeypatch, tmp_path / str(index), [_resource(disk=disk)]
        )
        assert assessment["state"] == "observed_schema_unknown"
        assert assessment["resource"] is None
        recorded = assessment["semantic"]["availability"]["resources"][0]
        assert recorded["disk_capability"]["raw"] == disk
        assert all(value is False for value in assessment["authorization"].values())
        assert (tmp_path / str(index) / inventory.RAW_RELATIVE).read_bytes() == raw


def test_null_spot_and_capacity_states_remain_non_authorizing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = (
        ([_resource(is_spot=None)], "observed_ambiguous_resources"),
        ([_resource(price="$2.01", price_value=2.01)], "observed_no_qualifying_resource"),
        (
            [_resource(), _resource(resource_id="resource-002")],
            "observed_ambiguous_resources",
        ),
    )
    for index, (resources, expected) in enumerate(cases):
        _raw, assessment = _capture_and_assess(monkeypatch, tmp_path / str(index), resources)
        assert assessment["state"] == expected
        assert assessment["resource"] is None
        assert all(value is False for value in assessment["authorization"].values())
    wallet_root = tmp_path / "wallet-boundary"
    _bind_root(
        monkeypatch,
        wallet_root,
        _outputs([_resource()], wallet_balance=30.0),
    )
    inventory.capture_prime_inventory_raw_v3()
    inventory.assess_prime_inventory_v3()
    wallet_assessment = inventory.validate_prime_inventory_assessment_v3(now_epoch=NOW)
    assert wallet_assessment["state"] == "observed_no_qualifying_resource"
    assert wallet_assessment["resource"] is None


def test_assessment_never_invokes_prime_and_both_outputs_are_no_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bind_root(monkeypatch, tmp_path, _outputs([_resource()]))
    raw = inventory.capture_prime_inventory_raw_v3()
    monkeypatch.setattr(
        inventory,
        "_run_prime_command",
        lambda _argv: (_ for _ in ()).throw(AssertionError("Prime invoked by assessor")),
    )
    assessment = inventory.assess_prime_inventory_v3()
    with pytest.raises(FileExistsError):
        inventory.capture_prime_inventory_raw_v3()
    with pytest.raises(FileExistsError):
        inventory.assess_prime_inventory_v3()
    assert (tmp_path / inventory.RAW_RELATIVE).read_bytes() == raw
    assert (tmp_path / inventory.ASSESSMENT_RELATIVE).read_bytes() == assessment


def test_internal_assessment_failure_cannot_mutate_raw_or_create_assessment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bind_root(monkeypatch, tmp_path, _outputs([_resource()]))
    raw = inventory.capture_prime_inventory_raw_v3()
    monkeypatch.setattr(
        inventory,
        "_semantic",
        lambda _receipt: (_ for _ in ()).throw(RuntimeError("internal failure")),
    )
    with pytest.raises(RuntimeError, match="internal failure"):
        inventory.assess_prime_inventory_v3()
    assert (tmp_path / inventory.RAW_RELATIVE).read_bytes() == raw
    assert not (tmp_path / inventory.ASSESSMENT_RELATIVE).exists()


def test_raw_tamper_ttl_checkout_argv_and_hash_fail_before_assessment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mutations: tuple[tuple[str, object], ...] = (
        ("unknown", None),
        ("captured_at_epoch", NOW - 901),
        ("checkout.commit", "0" * 40),
        ("commands.wallet.argv", ["prime", "wallet"]),
        ("commands.wallet.executed_argv", ["shadow-prime", "wallet"]),
        ("cli.installed_executable.sha256", "0" * 64),
        ("commands.availability.stdout_sha256", "0" * 64),
    )
    for index, (field, replacement) in enumerate(mutations):
        root = tmp_path / str(index)
        _bind_root(monkeypatch, root, _outputs([_resource()]))
        inventory.capture_prime_inventory_raw_v3()
        path = root / inventory.RAW_RELATIVE
        value = json.loads(path.read_bytes())
        cursor: dict[str, Any] = value
        parts = field.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = replacement
        path.write_bytes(canonical_json_bytes(value))
        with pytest.raises(ValueError):
            inventory.validate_prime_inventory_raw_v3(now_epoch=NOW)
        assert not (root / inventory.ASSESSMENT_RELATIVE).exists()


def test_fixed_paths_reject_existing_alias_and_linked_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "alias"
    _bind_root(monkeypatch, root, _outputs([_resource()]))
    target = root / inventory.RAW_RELATIVE
    target.parent.mkdir(parents=True)
    victim = root / "victim.json"
    victim.write_bytes(b"victim")
    os.link(victim, target)
    with pytest.raises(FileExistsError):
        inventory.capture_prime_inventory_raw_v3()
    assert victim.read_bytes() == b"victim"

    linked_root = tmp_path / "linked"
    _bind_root(monkeypatch, linked_root, _outputs([_resource()]))
    output_parent = linked_root / Path(inventory.RAW_RELATIVE).parent
    original = inventory._is_link_or_reparse
    monkeypatch.setattr(
        inventory,
        "_is_link_or_reparse",
        lambda path: path == output_parent or original(path),
    )
    with pytest.raises(ValueError, match="link or reparse"):
        inventory.capture_prime_inventory_raw_v3()
    assert not (linked_root / inventory.RAW_RELATIVE).exists()

    root_link = tmp_path / "root-link"
    _bind_root(monkeypatch, root_link, _outputs([_resource()]))
    monkeypatch.setattr(
        inventory,
        "_is_link_or_reparse",
        lambda path: path == root_link or original(path),
    )
    with pytest.raises(ValueError, match="repository ancestor"):
        inventory.capture_prime_inventory_raw_v3()
    assert not (root_link / inventory.RAW_RELATIVE).exists()


def test_installed_prime_owner_proves_exact_disk_serialization() -> None:
    binding = inventory.authenticate_installed_prime_source()
    assert binding == {
        "distribution": "prime",
        "version": "0.6.20",
        "module_path": "prime_cli/commands/availability.py",
        "sha256": "9b9e72810b138d66278e9375988db1a9ae847d1d0fbee58424cc5c63554a83fa",
        "bytes": 16_169,
    }
    text = inventory._prime_source_path().read_text(encoding="utf-8")
    assert all(line in text for line in inventory.PRIME_SOURCE_REQUIRED_LINES)


def test_prime_executable_accepts_only_disposable_hardlink_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    installed_executable, _installed_entrypoint, installed_receipt = (
        inventory._prime_tool_paths()
    )
    disposable_source = tmp_path / "disposable-prime.exe"
    disposable_source.write_bytes(installed_executable.read_bytes())
    assert not disposable_source.samefile(installed_executable)
    hardlink_executable = tmp_path / "uv" / "tools" / "prime" / "Scripts" / "prime.exe"
    hardlink_entrypoint = tmp_path / "bin" / "prime.exe"
    hardlink_receipt = tmp_path / "uv" / "tools" / "prime" / "uv-receipt.toml"
    hardlink_executable.parent.mkdir(parents=True)
    hardlink_entrypoint.parent.mkdir(parents=True)
    os.link(disposable_source, hardlink_executable)
    os.link(disposable_source, hardlink_entrypoint)
    hardlink_receipt.write_bytes(installed_receipt.read_bytes())
    monkeypatch.setattr(
        inventory,
        "_prime_tool_paths",
        lambda: (hardlink_executable, hardlink_entrypoint, hardlink_receipt),
    )
    monkeypatch.setattr(
        "redco.analysis.stage_d_v13_prime_inventory_v3.shutil.which",
        lambda _name: str(hardlink_entrypoint),
    )
    assert inventory.authenticate_installed_prime_executable()["identity"] == "samefile"


def test_prime_executable_accepts_independent_copy_and_rejects_path_shadow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    installed_executable, _installed_entrypoint, installed_receipt = (
        inventory._prime_tool_paths()
    )
    copied_executable = tmp_path / "uv" / "tools" / "prime" / "Scripts" / "prime.exe"
    copied_entrypoint = tmp_path / "bin" / "prime.exe"
    copied_receipt = tmp_path / "uv" / "tools" / "prime" / "uv-receipt.toml"
    copied_executable.parent.mkdir(parents=True)
    copied_entrypoint.parent.mkdir(parents=True)
    copied_executable.write_bytes(installed_executable.read_bytes())
    copied_entrypoint.write_bytes(installed_executable.read_bytes())
    copied_receipt.write_bytes(installed_receipt.read_bytes())
    assert not copied_executable.samefile(copied_entrypoint)
    assert not copied_executable.samefile(installed_executable)
    monkeypatch.setattr(
        inventory,
        "_prime_tool_paths",
        lambda: (copied_executable, copied_entrypoint, copied_receipt),
    )
    monkeypatch.setattr(
        "redco.analysis.stage_d_v13_prime_inventory_v3.shutil.which",
        lambda _name: str(copied_entrypoint),
    )
    binding = inventory.authenticate_installed_prime_executable()
    assert binding["identity"] == "uv_receipt_bound_hash_equivalent_entrypoint"
    assert binding["sha256"] == inventory.PRIME_EXECUTABLE_SHA256

    root = tmp_path / "shadowed"
    root.mkdir()
    monkeypatch.setattr(inventory, "ROOT", root)
    shadow = tmp_path / "shadow" / "prime.exe"
    shadow.parent.mkdir()
    shadow.write_bytes(b"not the authenticated Prime executable")
    monkeypatch.setattr(
        "redco.analysis.stage_d_v13_prime_inventory_v3.shutil.which",
        lambda _name: str(shadow),
    )
    command_calls = 0

    def command_must_not_run(
        _argv: tuple[str, ...],
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal command_calls
        command_calls += 1
        raise AssertionError("shadowed Prime executable reached command execution")

    monkeypatch.setattr(inventory, "_run_prime_command", command_must_not_run)
    with pytest.raises(ValueError, match="PATH entrypoint is shadowed"):
        inventory.capture_prime_inventory_raw_v3()
    assert command_calls == 0
    assert not (root / inventory.RAW_RELATIVE).exists()


def test_v1_v2_are_immutable_and_v3_build_is_deterministic() -> None:
    assert {
        relative: sha256_bytes((ROOT / relative).read_bytes())
        for relative in inventory.HISTORICAL_BINDINGS
    } == inventory.HISTORICAL_BINDINGS
    first = inventory.build_prime_inventory_v3_artifacts(ROOT)
    second = inventory.build_prime_inventory_v3_artifacts(ROOT)
    assert first == second
    contract = json.loads(first[inventory.CONTRACT_RELATIVE])
    audit = json.loads(first[inventory.AUDIT_RELATIVE])
    assert contract["parent"]["commit"] == inventory.PARENT_COMMIT
    assert contract["installed_prime_owner"]["sha256"] == inventory.PRIME_SOURCE_SHA256
    assert (
        contract["installed_prime_executable"]["sha256"]
        == inventory.PRIME_EXECUTABLE_SHA256
    )
    assert contract["raw_contract"]["absolute_authenticated_execution"] is True
    assert contract["raw_contract"]["path_shadow_rejected_before_execution"] is True
    assert audit["installed_prime_executable"] == contract["installed_prime_executable"]
    assert contract["fixed_artifact_root"]["absent_is_fail_closed"] is True
    assert audit["raw_wallet_or_billing_tracked"] is False
    assert all(value is False for value in contract["authorization"].values())
    assert all(value is False for value in audit["authorization"].values())


def test_production_capture_and_assessment_have_no_path_or_fact_arguments() -> None:
    assert inventory.capture_prime_inventory_raw_v3.__code__.co_argcount == 0
    assert inventory.assess_prime_inventory_v3.__code__.co_argcount == 0
