"""Source-free tests for the non-authorizing Prime inventory v4 correction."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from redco.analysis import stage_d_v13_prime_inventory_v3 as inventory_v3
from redco.analysis import stage_d_v13_prime_inventory_v4 as inventory
from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes

ROOT = Path(__file__).parents[1].resolve()


def _resource(
    *,
    label: object = "L40S 48GB",
    count: object = 2,
    memory: object = 96,
    spot: object = False,
    resource_id: str = "fixture-resource",
) -> dict[str, object]:
    return {
        "id": resource_id,
        "cloud_id": "fixture-cloud",
        "gpu_type": label,
        "gpu_count": count,
        "socket": "PCIe",
        "provider": "fixture-provider",
        "location": "fixture-location",
        "stock_status": "Available",
        "price_per_hour": "$1.64",
        "price_value": 1.64,
        "security": "datacenter",
        "vcpus": "16",
        "memory_gb": "128",
        "disk_gb": "1250",
        "gpu_memory": memory,
        "is_spot": spot,
    }


def _availability(
    resources: list[dict[str, object]],
    *,
    filter_label: object = "L40S_48GB",
) -> dict[str, object]:
    return {
        "gpu_resources": resources,
        "total_count": len(resources),
        "filters": {
            "gpu_type": filter_label,
            "gpu_count": 2,
            "regions": None,
            "socket": None,
            "provider": None,
            "group_similar": True,
        },
    }


def _state(resources: list[dict[str, object]]) -> dict[str, Any]:
    return inventory.assess_historical_availability_v4(_availability(resources))


def test_terminal_v3_reassessment_is_exact_sanitized_and_non_authorizing() -> None:
    raw_path = ROOT / inventory_v3.RAW_RELATIVE
    raw = raw_path.read_bytes()
    assert len(raw) == inventory.V3_RAW_BYTES
    assert sha256_bytes(raw) == inventory.V3_RAW_SHA256
    assessed = inventory._assessment_value(raw)
    assert assessed["state"] == "historical_observed_ambiguous_resources"
    assert assessed["reason"] == "non_spot_status_unknown"
    assert assessed["resource"] is None
    semantic = cast(dict[str, Any], assessed["semantic"])
    assert semantic["captured_row_count"] == 9
    assert semantic["target_row_count"] == 1
    assert semantic["eligible_target_row_count"] == 0
    assert semantic["unknown_non_spot_row_count"] == 1
    assert semantic["historical_non_exhaustive"] is True
    assert semantic["current_availability_claimed"] is False
    authorization = cast(dict[str, bool], assessed["authorization"])
    assert all(value is False for value in authorization.values())

    receipt = json.loads(raw)
    availability = json.loads(
        base64.b64decode(receipt["commands"]["availability"]["stdout_b64"])
    )
    wallet = json.loads(base64.b64decode(receipt["commands"]["wallet"]["stdout_b64"]))
    sensitive = {wallet["wallet_id"]}
    sensitive.update(row["id"] for row in availability["gpu_resources"])
    assessment_bytes = canonical_json_bytes(assessed)
    assert all(value.encode() not in assessment_bytes for value in sensitive)


def test_installed_semantic_owners_bind_optional_spot_and_direct_memory() -> None:
    owners = inventory.authenticate_installed_semantic_owners()
    api = cast(dict[str, object], owners["availability_api"])
    formatter = cast(dict[str, object], owners["availability_formatter"])
    assert api["sha256"] == inventory.API_SOURCE_SHA256
    assert formatter["sha256"] == inventory_v3.PRIME_SOURCE_SHA256
    assert api["gpu_memory"] == (
        "required_integer_forwarded_without_conversion"
    )
    assert api["is_spot"] == "optional_boolean"
    assert owners["grouping_key_includes_is_spot"] is False
    assert owners["bundled_openapi_or_swagger"] is False
    assert owners["stronger_preprovision_spot_proof"] is False


def test_gpu_counts_use_exact_48gb_per_device_but_only_two_is_eligible() -> None:
    cases = (
        (1, "historical_observed_no_qualifying_resource", 0),
        (2, "historical_observed_non_authorizing_resource", 1),
        (4, "historical_observed_no_qualifying_resource", 0),
    )
    for count, expected_state, eligible in cases:
        assessed = _state([_resource(count=count, memory=count * 48)])
        assert assessed["state"] == expected_state
        availability = cast(dict[str, Any], assessed["availability"])
        assert availability["eligible_target_row_count"] == eligible
        assert assessed["resource"] is None


def test_target_memory_mutations_are_schema_unknown() -> None:
    for memory in (48, 95, 97, "96", 96.0, True, None, -1, 1_000_000_001):
        assessed = _state([_resource(memory=memory)])
        assert assessed["state"] == "historical_observed_schema_unknown"
        assert assessed["resource"] is None
        assert assessed["availability"] is None


def test_target_label_variants_and_other_allowed_labels_fail_closed() -> None:
    labels = (
        "L40S_48GB",
        "l40s 48gb",
        " L40S 48GB",
        "L40S 48GB ",
        "L40S 48GB (Spot)",
        "L40 48GB",
        "RTX6000Ada 48GB",
    )
    for label in labels:
        assessed = _state([_resource(label=label)])
        assert assessed["state"] == "historical_observed_schema_unknown"
        assert assessed["resource"] is None


def test_exact_filter_and_target_labels_are_not_generically_derived() -> None:
    assessed = _state([_resource(label="B300 288GB", memory=288)])
    assert assessed["state"] == "historical_observed_no_qualifying_resource"
    assert assessed["availability"]["target_row_count"] == 0
    bad_filter = inventory.assess_historical_availability_v4(
        _availability([_resource()], filter_label="B300_288GB")
    )
    assert bad_filter["state"] == "historical_observed_schema_unknown"


def test_only_literal_false_is_row_eligible() -> None:
    cases: tuple[tuple[object, str, str], ...] = (
        (False, "historical_observed_non_authorizing_resource", "grouped_capture_non_exhaustive"),
        (True, "historical_observed_no_qualifying_resource", "spot_resource"),
        (None, "historical_observed_ambiguous_resources", "non_spot_status_unknown"),
    )
    for spot, expected_state, expected_reason in cases:
        assessed = _state([_resource(spot=spot)])
        assert assessed["state"] == expected_state
        assert assessed["reason"] == expected_reason
        assert assessed["resource"] is None


def test_wrong_spot_types_are_schema_unknown() -> None:
    spots: tuple[object, ...] = ("false", 0, 1, [], {})
    for spot in spots:
        assessed = _state([_resource(spot=spot)])
        assert assessed["state"] == "historical_observed_schema_unknown"


def test_absent_spot_and_display_suffix_contradictions_fail_closed() -> None:
    absent = _resource()
    absent.pop("is_spot")
    assert _state([absent])["state"] == "historical_observed_schema_unknown"
    for spot in (False, True, None):
        assessed = _state([_resource(label="L40S 48GB (Spot)", spot=spot)])
        assert assessed["state"] == "historical_observed_schema_unknown"


def test_duplicates_are_ambiguous_and_unknown_keys_fail_closed() -> None:
    duplicate = _state(
        [_resource(resource_id="first"), _resource(resource_id="second")]
    )
    assert duplicate["state"] == "historical_observed_ambiguous_resources"
    assert duplicate["reason"] == "duplicate_or_grouped_target_rows"
    assert duplicate["resource"] is None

    extra = _resource()
    extra["unknown"] = None
    assert _state([extra])["state"] == "historical_observed_schema_unknown"
    value = _availability([_resource()])
    cast_filters = value["filters"]
    assert isinstance(cast_filters, dict)
    cast_filters["unknown"] = None
    assert inventory.assess_historical_availability_v4(value)["state"] == (
        "historical_observed_schema_unknown"
    )


def test_unrelated_row_memory_cannot_poison_valid_target_but_type_bounds_apply() -> None:
    unrelated = _resource(label="B300 288GB", memory="not-a-target-memory")
    assessed = _state([unrelated, _resource()])
    assert assessed["state"] == "historical_observed_non_authorizing_resource"
    assert assessed["availability"]["eligible_target_row_count"] == 1
    assert assessed["availability"]["unrelated_row_count"] == 1

    unsafe = _resource(label="B300 288GB", memory={"nested": "value"})
    assert _state([unsafe, _resource()])["state"] == "historical_observed_schema_unknown"
    unsafe["unknown"] = None
    assert _state([unsafe, _resource()])["state"] == "historical_observed_schema_unknown"


def test_history_hash_mutation_is_rejected_before_semantics() -> None:
    raw = bytearray((ROOT / inventory_v3.RAW_RELATIVE).read_bytes())
    raw[-2] = raw[-2] ^ 1
    with pytest.raises(ValueError, match="raw receipt binding differs"):
        inventory._authenticate_v3_raw_bytes(bytes(raw))
    assessment = bytearray((ROOT / inventory_v3.ASSESSMENT_RELATIVE).read_bytes())
    assessment[-2] = assessment[-2] ^ 1
    with pytest.raises(ValueError, match="assessment binding differs"):
        inventory._authenticate_v3_assessment_bytes(bytes(assessment))


def _bind_temp_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> bytes:
    raw = (ROOT / inventory_v3.RAW_RELATIVE).read_bytes()
    path = root / inventory_v3.RAW_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    assessment = root / inventory_v3.ASSESSMENT_RELATIVE
    assessment.parent.mkdir(parents=True, exist_ok=True)
    assessment.write_bytes((ROOT / inventory_v3.ASSESSMENT_RELATIVE).read_bytes())
    monkeypatch.setattr(inventory, "ROOT", root)
    monkeypatch.setattr(inventory_v3, "ROOT", root)
    return cast(bytes, raw)


def test_offline_assessment_is_fixed_no_overwrite_and_tamper_evident(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bind_temp_root(monkeypatch, tmp_path)
    assessment = inventory.assess_prime_inventory_v4()
    output = tmp_path / inventory.ASSESSMENT_RELATIVE
    assert output.read_bytes() == assessment
    validated = inventory.validate_prime_inventory_assessment_v4()
    assert validated["state"] == "historical_observed_ambiguous_resources"
    with pytest.raises(FileExistsError):
        inventory.assess_prime_inventory_v4()
    output.write_bytes(assessment[:-1] + b" ")
    with pytest.raises(ValueError, match="binding differs"):
        inventory.validate_prime_inventory_assessment_v4()


def test_raw_alias_linked_ancestor_and_output_alias_fail_before_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    aliased_root = tmp_path / "aliased"
    raw_source = aliased_root / "raw-source.json"
    raw_source.parent.mkdir(parents=True)
    raw_source.write_bytes((ROOT / inventory_v3.RAW_RELATIVE).read_bytes())
    raw_path = aliased_root / inventory_v3.RAW_RELATIVE
    raw_path.parent.mkdir(parents=True)
    os.link(raw_source, raw_path)
    monkeypatch.setattr(inventory, "ROOT", aliased_root)
    monkeypatch.setattr(inventory_v3, "ROOT", aliased_root)
    with pytest.raises(ValueError, match="input is aliased"):
        inventory.assess_prime_inventory_v4()
    assert not (aliased_root / inventory.ASSESSMENT_RELATIVE).exists()

    output_root = tmp_path / "output-alias"
    _bind_temp_root(monkeypatch, output_root)
    victim = output_root / "victim.json"
    victim.write_bytes(b"victim")
    output = output_root / inventory.ASSESSMENT_RELATIVE
    output.parent.mkdir(parents=True, exist_ok=True)
    os.link(victim, output)
    with pytest.raises(FileExistsError):
        inventory.assess_prime_inventory_v4()
    assert victim.read_bytes() == b"victim"

    linked_root = tmp_path / "linked"
    _bind_temp_root(monkeypatch, linked_root)
    linked_parent = linked_root / Path(inventory_v3.RAW_RELATIVE).parent
    original = inventory_v3._is_link_or_reparse
    monkeypatch.setattr(
        inventory_v3,
        "_is_link_or_reparse",
        lambda path: path == linked_parent or original(path),
    )
    with pytest.raises(ValueError, match="ancestor"):
        inventory.assess_prime_inventory_v4()
    assert not (linked_root / inventory.ASSESSMENT_RELATIVE).exists()


def test_v1_through_v3_are_immutable_and_v4_build_is_deterministic() -> None:
    assert {
        relative: sha256_bytes((ROOT / relative).read_bytes())
        for relative in inventory.HISTORICAL_BINDINGS
    } == inventory.HISTORICAL_BINDINGS
    first = inventory.build_prime_inventory_v4_artifacts(ROOT)
    second = inventory.build_prime_inventory_v4_artifacts(ROOT)
    assert first == second
    contract = json.loads(first[inventory.CONTRACT_RELATIVE])
    audit = json.loads(first[inventory.AUDIT_RELATIVE])
    assert contract["parent"] == {
        "commit": inventory.PARENT_COMMIT,
        "tree": inventory.PARENT_TREE,
    }
    assert contract["offline_assessment"]["state"] == (
        "historical_observed_ambiguous_resources"
    )
    assert contract["offline_assessment"]["reason"] == "non_spot_status_unknown"
    assert audit["sanitized_counts"] == {
        "captured_rows": 9,
        "target_rows": 1,
        "eligible_target_rows": 0,
        "unknown_non_spot_rows": 1,
    }
    serialized = first[inventory.CONTRACT_RELATIVE] + first[inventory.AUDIT_RELATIVE]
    assert b"wallet_id" not in serialized
    assert b"stdout_b64" not in serialized
    assert b"resource_id" not in serialized
    assert all(value is False for value in contract["authorization"].values())
    assert all(value is False for value in audit["authorization"].values())


def test_v4_has_no_capture_or_authorizing_entrypoint() -> None:
    assert not hasattr(inventory, "capture_prime_inventory_v4")
    assert inventory.assess_prime_inventory_v4.__code__.co_argcount == 0
    assert all(value is False for value in inventory.AUTHORIZATION_FALSE.values())
