"""Pure historical reassessment of the terminal Prime inventory v3 receipt."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

from redco.analysis import stage_d_v13_prime_inventory_v2 as v2
from redco.analysis import stage_d_v13_prime_inventory_v3 as v3
from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes
from redco.analysis.stage_d_v13_support_readiness import _status_paths

ROOT = Path(__file__).parents[3].resolve()
PARENT_COMMIT = "94ecbf77c4e9584d30d974a26cfb45edd574b638"
PARENT_TREE = "042ceb8628a049197723b3a04411cce44d2e3b66"
SCHEMA_VERSION = 4
ASSESSMENT_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-assessment-v4"
CONTRACT_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-contract-v4"
AUDIT_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-audit-v4"

ASSESSMENT_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-readiness/prime-inventory-assessment-v4.json"
)
OWNER_RELATIVE = "src/redco/analysis/stage_d_v13_prime_inventory_v4.py"
BUILDER_RELATIVE = "scripts/build_stage_d_v13_prime_inventory_v4.py"
TEST_RELATIVE = "tests/test_stage_d_v13_prime_inventory_v4.py"
CONTRACT_RELATIVE = "configs/stage-d/stage-d1-support-prime-inventory-contract-v4.json"
AUDIT_RELATIVE = "reports/stage-d1-support-prime-inventory-audit-v4.json"
CHECKPOINT_PATHS = frozenset(
    {OWNER_RELATIVE, BUILDER_RELATIVE, TEST_RELATIVE, CONTRACT_RELATIVE, AUDIT_RELATIVE}
)

V3_RAW_SHA256 = "921bb07a5bfcb62f7fd63b7498ba9c2dcdb04146b0297e1515f66b318cff65ff"
V3_RAW_BYTES = 48_624
V3_ASSESSMENT_SHA256 = "c506dd2e19a660b0f45f4a0cb607c0f104deb8d22da132be41d95c1ec4e958b6"
V3_ASSESSMENT_BYTES = 742
V3_TRACKED_BINDINGS = {
    v3.OWNER_RELATIVE: "d21eb09221075526797370e752a12b64d60fca2599937ad330f577baf9660734",
    v3.BUILDER_RELATIVE: "0ab21ba352a2e76029d59facc3f61de302ac7ff0a73243a76948e942e9b90723",
    v3.TEST_RELATIVE: "628df14003864532719c586becf58a75f5e2337ebeb3f79f1fa0c7f9759d4502",
    v3.CONTRACT_RELATIVE: "bf3b4cbf9e01aedc18b602722d6e882ff90e45b48a253f7958e78e147176b905",
    v3.AUDIT_RELATIVE: "ae1ff766704eab6f203d3e567effc81baa99ab9a2b3f35c7e4d014f6a5eeeaf5",
}
HISTORICAL_BINDINGS = {**v3.HISTORICAL_BINDINGS, **V3_TRACKED_BINDINGS}

API_SOURCE_RELATIVE = "prime_cli/api/availability.py"
API_SOURCE_SHA256 = "fe366aea5b501ae278902e55a4d1d3059e2fbbcde48d0beeffe980d36603e938"
API_SOURCE_BYTES = 7_439
ALLOWED_FILTERS = {
    "L40_48GB": "L40 48GB",
    "L40S_48GB": "L40S 48GB",
    "RTX6000Ada_48GB": "RTX6000Ada 48GB",
}
ALLOWED_BASE_LABELS = frozenset(ALLOWED_FILTERS.values())
MEMORY_GB_PER_DEVICE = 48
MAX_TEXT_BYTES = 512
MAX_NUMBER = 1_000_000_000

AUTHORIZATION_FALSE = dict(v3.AUTHORIZATION_FALSE)


def _api_source_path() -> Path:
    return cast(Path, v3._prime_source_path().parents[1] / "api" / "availability.py")


def authenticate_installed_semantic_owners() -> dict[str, object]:
    """Bind the exact API and formatter owners that define captured semantics."""

    api_path = _api_source_path()
    if v3._is_link_or_reparse(api_path) or not api_path.is_file():
        raise ValueError("installed Prime availability API owner is unavailable")
    api_raw = api_path.read_bytes()
    if len(api_raw) != API_SOURCE_BYTES or sha256_bytes(api_raw) != API_SOURCE_SHA256:
        raise ValueError("installed Prime availability API owner binding differs")
    api_text = api_raw.decode("utf-8")
    api_lines = (
        'gpu_memory: int = Field(..., alias="gpuMemory")',
        'is_spot: Optional[bool] = Field(None, alias="isSpot")',
    )
    if any(line not in api_text for line in api_lines):
        raise ValueError("installed Prime availability API semantics differ")

    formatter = v3.authenticate_installed_prime_source()
    formatter_path = v3._prime_source_path()
    formatter_text = formatter_path.read_text(encoding="utf-8")
    formatter_lines = (
        '"gpu_memory": gpu.gpu_memory,',
        '"is_spot": gpu.is_spot,',
        '"gpu_memory": gpu_entry["gpu_memory"],',
        '"is_spot": gpu_entry["is_spot"],',
    )
    if any(line not in formatter_text for line in formatter_lines):
        raise ValueError("installed Prime formatter forwarding semantics differ")
    grouping = formatter_text[formatter_text.index("key = (") : formatter_text.index(
        ")\n                if key", formatter_text.index("key = (")
    )]
    if "is_spot" in grouping:
        raise ValueError("installed Prime grouping unexpectedly binds spot state")

    package_root = api_path.parents[1]
    bundled_schema = sorted(
        str(path.relative_to(package_root)).replace("\\", "/")
        for path in package_root.rglob("*")
        if path.is_file()
        and any(token in path.name.casefold() for token in ("openapi", "swagger"))
    )
    if bundled_schema:
        raise ValueError("installed Prime package unexpectedly bundles an API schema")
    return {
        "availability_api": {
            "path": API_SOURCE_RELATIVE,
            "sha256": API_SOURCE_SHA256,
            "bytes": API_SOURCE_BYTES,
            "gpu_memory": "required_integer_forwarded_without_conversion",
            "is_spot": "optional_boolean",
        },
        "availability_formatter": formatter,
        "grouping_key_fields": [
            "provider",
            "gpu_type",
            "gpu_count",
            "socket",
            "location",
            "security",
            "price",
        ],
        "grouping_key_includes_is_spot": False,
        "bundled_openapi_or_swagger": False,
        "stronger_preprovision_spot_proof": False,
    }


def _fixed_input(root: Path, relative: str, *, expected_hash: str, expected_bytes: int) -> bytes:
    root = root.absolute()
    if v3._is_link_or_reparse(root):
        raise ValueError("Prime v4 repository root is a link or reparse point")
    path = root / relative
    current = root
    for part in Path(relative).parts[:-1]:
        current /= part
        if current.exists() and v3._is_link_or_reparse(current):
            raise ValueError("Prime v4 input ancestor is a link or reparse point")
    if v3._is_link_or_reparse(path) or not path.is_file():
        raise ValueError(f"Prime v4 fixed input is unavailable: {relative}")
    if path.stat().st_nlink != 1:
        raise ValueError(f"Prime v4 fixed input is aliased: {relative}")
    raw = path.read_bytes()
    if len(raw) != expected_bytes or sha256_bytes(raw) != expected_hash:
        raise ValueError(f"Prime v4 fixed input binding differs: {relative}")
    return raw


def _authenticate_v3_raw_bytes(raw: bytes) -> dict[str, Any]:
    if len(raw) != V3_RAW_BYTES or sha256_bytes(raw) != V3_RAW_SHA256:
        raise ValueError("terminal Prime v3 raw receipt binding differs")
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("terminal Prime v3 raw receipt is not JSON") from error
    expected = {
        "schema_version",
        "domain",
        "state",
        "captured_at_epoch",
        "expires_at_epoch",
        "checkout",
        "cli",
        "commands",
        "authorization",
    }
    if (
        not isinstance(parsed, dict)
        or canonical_json_bytes(parsed) != raw
        or set(parsed) != expected
        or parsed["schema_version"] != v3.SCHEMA_VERSION
        or parsed["domain"] != v3.RAW_DOMAIN
        or parsed["state"] != "captured_raw_terminal"
        or parsed["checkout"] != {"commit": PARENT_COMMIT, "tree": PARENT_TREE}
        or parsed["authorization"] != AUTHORIZATION_FALSE
    ):
        raise ValueError("terminal Prime v3 raw receipt fields differ")
    captured = parsed["captured_at_epoch"]
    expires = parsed["expires_at_epoch"]
    if (
        type(captured) is not int
        or type(expires) is not int
        or expires != captured + v3.TTL_SECONDS
    ):
        raise ValueError("terminal Prime v3 raw receipt time binding differs")
    installed_owner = v3.authenticate_installed_prime_source()
    executable = v3.authenticate_installed_prime_executable()
    if parsed["cli"] != {
        "version": v2.PRIME_CLI_VERSION,
        "banner": v2.PRIME_VERSION_BANNER,
        "installed_owner": installed_owner,
        "installed_executable": executable,
    }:
        raise ValueError("terminal Prime v3 CLI binding differs")
    captures = parsed["commands"]
    if not isinstance(captures, dict) or set(captures) != set(v2.PRIME_READ_ONLY_COMMANDS):
        raise ValueError("terminal Prime v3 command set differs")
    for name in v2.PRIME_READ_ONLY_COMMANDS:
        v3._decode_capture(captures, name, cast(str, executable["canonical_path"]))
    return cast(dict[str, Any], parsed)


def _authenticate_v3_assessment_bytes(raw: bytes) -> dict[str, Any]:
    if len(raw) != V3_ASSESSMENT_BYTES or sha256_bytes(raw) != V3_ASSESSMENT_SHA256:
        raise ValueError("terminal Prime v3 assessment binding differs")
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("terminal Prime v3 assessment is not JSON") from error
    expected = {
        "schema_version",
        "domain",
        "state",
        "raw_receipt",
        "semantic",
        "resource",
        "diagnostic",
        "authorization",
    }
    if (
        not isinstance(parsed, dict)
        or canonical_json_bytes(parsed) != raw
        or set(parsed) != expected
        or parsed.get("schema_version") != v3.SCHEMA_VERSION
        or parsed.get("domain") != v3.ASSESSMENT_DOMAIN
        or parsed.get("state") != "observed_schema_unknown"
        or parsed.get("raw_receipt")
        != {"path": v3.RAW_RELATIVE, "sha256": V3_RAW_SHA256}
        or parsed.get("authorization") != AUTHORIZATION_FALSE
    ):
        raise ValueError("terminal Prime v3 assessment fields differ")
    return cast(dict[str, Any], parsed)


def _bounded_scalar(value: object, label: str) -> None:
    if value is None or type(value) is bool:
        return
    if isinstance(value, str):
        if not value or len(value.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError(f"{label} text is malformed")
        return
    if type(value) in (int, float):
        number = float(cast(int | float, value))
        if not math.isfinite(number) or abs(number) > MAX_NUMBER:
            raise ValueError(f"{label} number is malformed")
        return
    raise ValueError(f"{label} value type is malformed")


def _row(value: object) -> dict[str, Any]:
    row = cast(dict[str, Any], v2._object(value, "Prime availability resource"))
    if set(row) != v2.RESOURCE_KEYS:
        raise ValueError("Prime availability resource schema differs from 0.6.20")
    label = row["gpu_type"]
    if not isinstance(label, str) or not label or len(label.encode("utf-8")) > 128:
        raise ValueError("Prime GPU label is malformed")
    return row


def _looks_like_expected_label(label: str, expected: str) -> bool:
    normalized = label.replace("_", " ").strip().casefold()
    if normalized.endswith(" (spot)"):
        normalized = normalized.removesuffix(" (spot)")
    return normalized == expected.casefold()


def _target_row(row: dict[str, Any], expected_label: str) -> dict[str, object]:
    for key, value in row.items():
        _bounded_scalar(value, f"Prime target {key}")
    if row["gpu_type"] != expected_label:
        raise ValueError("Prime target GPU label differs from the exact frozen label")
    count = row["gpu_count"]
    memory = row["gpu_memory"]
    if type(count) is not int or count <= 0 or count > 1_024:
        raise ValueError("Prime target GPU count is malformed")
    if type(memory) is not int or memory < 0 or memory > MAX_NUMBER:
        raise ValueError("Prime target aggregate GPU memory is malformed")
    if memory != count * MEMORY_GB_PER_DEVICE:
        raise ValueError("Prime target aggregate GPU memory differs from count times 48GB")
    display_amount, display_rate = v2._price(row["price_per_hour"])
    numeric_amount, numeric_rate = v2._price(row["price_value"])
    if display_amount != numeric_amount:
        raise ValueError("Prime target hourly price fields disagree")
    spot = row["is_spot"]
    if spot is not None and type(spot) is not bool:
        raise ValueError("Prime target spot state is malformed")
    stock_raw = row["stock_status"]
    if not isinstance(stock_raw, str):
        raise ValueError("Prime target stock status is malformed")
    stock = stock_raw.strip().lower()
    capability = v3._disk_capability(row["disk_gb"])
    reasons: list[str] = []
    if count != 2:
        reasons.append("gpu_count_not_two")
    if stock not in v2.AVAILABLE_STATES:
        reasons.append("stock_not_available")
    if display_amount > v2.QUALIFYING_RATE_USD:
        reasons.append("hourly_rate_above_cap")
    if spot is True:
        reasons.append("spot_resource")
    elif spot is None:
        reasons.append("non_spot_status_unknown")
    if capability["status"] != "proven":
        reasons.append("disk_capability_unknown")
    elif capability["minimum_ephemeral_gb"] == 0:
        reasons.append("disk_capability_not_positive")
    return {
        "gpu_type": expected_label,
        "gpu_count": count,
        "aggregate_gpu_memory_gb": memory,
        "hourly_rate_usd": display_rate,
        "price_value_usd": numeric_rate,
        "hourly_rate_cents": int(display_amount * 100),
        "spot_state": (
            "proven_non_spot" if spot is False else "spot" if spot is True else "unknown"
        ),
        "disk_capability": capability,
        "eligible_by_row": not reasons and spot is False,
        "reasons": reasons,
    }


def _assess_historical_availability_v4(value: object) -> dict[str, object]:
    availability = v2._object(value, "Prime availability")
    if set(availability) != {"gpu_resources", "total_count", "filters"}:
        raise ValueError("Prime availability schema differs")
    resources = availability["gpu_resources"]
    total = availability["total_count"]
    if not isinstance(resources, list) or type(total) is not int or total != len(resources):
        raise ValueError("Prime availability resource count differs")
    filters = v3._filters(availability["filters"])
    filter_label = filters["gpu_type"]
    if filter_label not in ALLOWED_FILTERS:
        raise ValueError("Prime availability filter is outside the exact v4 hardware law")
    expected_label = ALLOWED_FILTERS[cast(str, filter_label)]
    targets: list[dict[str, object]] = []
    unrelated = 0
    for raw_row in resources:
        row = _row(raw_row)
        label = cast(str, row["gpu_type"])
        if label == expected_label:
            targets.append(_target_row(row, expected_label))
        elif label in ALLOWED_BASE_LABELS or _looks_like_expected_label(label, expected_label):
            raise ValueError("Prime target GPU label differs from the exact frozen label")
        else:
            for key, item in row.items():
                _bounded_scalar(item, f"Prime unrelated {key}")
            unrelated += 1

    eligible = sum(target["eligible_by_row"] is True for target in targets)
    unknown_spot = sum(target["spot_state"] == "unknown" for target in targets)
    if not targets:
        state = "historical_observed_no_qualifying_resource"
        reason = "target_not_observed"
    elif unknown_spot:
        state = "historical_observed_ambiguous_resources"
        reason = "non_spot_status_unknown"
    elif len(targets) > 1:
        state = "historical_observed_ambiguous_resources"
        reason = "duplicate_or_grouped_target_rows"
    elif eligible == 1:
        state = "historical_observed_non_authorizing_resource"
        reason = "grouped_capture_non_exhaustive"
    else:
        state = "historical_observed_no_qualifying_resource"
        reasons = cast(list[str], targets[0]["reasons"])
        reason = reasons[0] if reasons else "not_eligible"
    return {
        "state": state,
        "reason": reason,
        "diagnostic": None,
        "resource": None,
        "availability": {
            "raw_filters": filters,
            "captured_row_count": total,
            "target_row_count": len(targets),
            "unrelated_row_count": unrelated,
            "eligible_target_row_count": eligible,
            "unknown_non_spot_row_count": unknown_spot,
            "target_rows": targets,
            "grouped_capture": True,
            "grouping_key_includes_is_spot": False,
            "historical_non_exhaustive": True,
            "current_availability_claimed": False,
        },
    }


def assess_historical_availability_v4(value: object) -> dict[str, object]:
    """Apply the v4 row-scoped law without claiming current availability."""

    try:
        return _assess_historical_availability_v4(value)
    except (KeyError, TypeError, ValueError) as error:
        return {
            "state": "historical_observed_schema_unknown",
            "reason": "semantic_schema_unknown",
            "diagnostic": str(error),
            "resource": None,
            "availability": None,
        }


def _assessment_value(raw: bytes) -> dict[str, object]:
    receipt = _authenticate_v3_raw_bytes(raw)
    captures = cast(dict[str, object], receipt["commands"])
    executable = cast(str, receipt["cli"]["installed_executable"]["canonical_path"])
    streams = {
        name: v3._decode_capture(captures, name, executable)
        for name in v2.PRIME_READ_ONLY_COMMANDS
    }
    if any(code != 0 for code, _stdout, _stderr in streams.values()):
        raise ValueError("terminal Prime v3 receipt contains a failed command")
    if streams["version"][1].decode("utf-8").strip() != v2.PRIME_VERSION_BANNER:
        raise ValueError("terminal Prime v3 version banner differs")
    wallet = v2._wallet(streams["wallet"][1])
    pods = v2._inventory_rows(
        v2._json_object(streams["pods"][1], "Prime pods"),
        key="pods",
        label="Prime pods",
        item_keys={"id", "name", "gpu", "status", "created_at"},
    )
    disks = v2._inventory_rows(
        v2._json_object(streams["disks"][1], "Prime disks"),
        key="disks",
        label="Prime disks",
        item_keys={
            "id",
            "name",
            "size",
            "status",
            "provider",
            "location",
            "created_at",
            "price_hr",
        },
    )
    semantic = assess_historical_availability_v4(
        v2._json_object(streams["availability"][1], "Prime availability")
    )
    preconditions = {
        "wallet_balance_over_30_at_capture": float(wallet["balance_usd"]) > 30,
        "pod_inventory_empty_at_capture": not pods,
        "disk_inventory_empty_at_capture": not disks,
    }
    if not all(preconditions.values()):
        semantic = {
            **semantic,
            "state": "historical_observed_no_qualifying_resource",
            "reason": "historical_readiness_precondition_failed",
            "resource": None,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "domain": ASSESSMENT_DOMAIN,
        "state": semantic["state"],
        "reason": semantic["reason"],
        "source_receipt": {
            "path": v3.RAW_RELATIVE,
            "sha256": V3_RAW_SHA256,
            "bytes": V3_RAW_BYTES,
            "checkout_commit": PARENT_COMMIT,
            "checkout_tree": PARENT_TREE,
            "terminal_non_reusable": True,
        },
        "source_assessment": {
            "path": v3.ASSESSMENT_RELATIVE,
            "sha256": V3_ASSESSMENT_SHA256,
            "bytes": V3_ASSESSMENT_BYTES,
            "original_state": "observed_schema_unknown",
        },
        "historical_preconditions": preconditions,
        "semantic": semantic["availability"],
        "resource": None,
        "authorization": AUTHORIZATION_FALSE,
    }


def assess_prime_inventory_v4() -> bytes:
    """Publish the one fixed historical v4 assessment without invoking Prime."""

    raw = _fixed_input(
        ROOT,
        v3.RAW_RELATIVE,
        expected_hash=V3_RAW_SHA256,
        expected_bytes=V3_RAW_BYTES,
    )
    _authenticate_v3_assessment_bytes(
        _fixed_input(
            ROOT,
            v3.ASSESSMENT_RELATIVE,
            expected_hash=V3_ASSESSMENT_SHA256,
            expected_bytes=V3_ASSESSMENT_BYTES,
        )
    )
    assessment = cast(bytes, canonical_json_bytes(_assessment_value(raw)))
    v3._publish_fixed(ASSESSMENT_RELATIVE, assessment)
    return assessment


def validate_prime_inventory_assessment_v4() -> dict[str, Any]:
    raw = _fixed_input(
        ROOT,
        v3.RAW_RELATIVE,
        expected_hash=V3_RAW_SHA256,
        expected_bytes=V3_RAW_BYTES,
    )
    _authenticate_v3_assessment_bytes(
        _fixed_input(
            ROOT,
            v3.ASSESSMENT_RELATIVE,
            expected_hash=V3_ASSESSMENT_SHA256,
            expected_bytes=V3_ASSESSMENT_BYTES,
        )
    )
    assessment = _fixed_input(
        ROOT,
        ASSESSMENT_RELATIVE,
        expected_hash=sha256_bytes(canonical_json_bytes(_assessment_value(raw))),
        expected_bytes=len(canonical_json_bytes(_assessment_value(raw))),
    )
    expected = cast(bytes, canonical_json_bytes(_assessment_value(raw)))
    if assessment != expected:
        raise ValueError("Prime inventory v4 assessment differs from its authenticated history")
    return cast(dict[str, Any], json.loads(assessment))


def _bound_file(root: Path, relative: str, expected: str) -> None:
    path = root / relative
    if path.is_symlink() or not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
        raise ValueError(f"historical Prime inventory binding differs: {relative}")


def _authenticate_precommit(root: Path) -> None:
    if v3._git(root, "rev-parse", "HEAD") != PARENT_COMMIT:
        raise ValueError("Prime inventory v4 build requires exact parent 94ecbf7")
    if v3._git(root, "rev-parse", "HEAD^{tree}") != PARENT_TREE:
        raise ValueError("Prime inventory v4 parent tree differs")
    unexpected = _status_paths(root).difference(CHECKPOINT_PATHS)
    if unexpected:
        raise ValueError(
            "Prime inventory v4 worktree exceeds its exact allowlist: "
            + ", ".join(sorted(unexpected))
        )


def build_prime_inventory_v4_artifacts(root: Path) -> dict[str, bytes]:
    root = root.resolve()
    _authenticate_precommit(root)
    for relative, expected in HISTORICAL_BINDINGS.items():
        _bound_file(root, relative, expected)
    owners = authenticate_installed_semantic_owners()
    raw = _fixed_input(
        root,
        v3.RAW_RELATIVE,
        expected_hash=V3_RAW_SHA256,
        expected_bytes=V3_RAW_BYTES,
    )
    _authenticate_v3_assessment_bytes(
        _fixed_input(
            root,
            v3.ASSESSMENT_RELATIVE,
            expected_hash=V3_ASSESSMENT_SHA256,
            expected_bytes=V3_ASSESSMENT_BYTES,
        )
    )
    expected_assessment = cast(bytes, canonical_json_bytes(_assessment_value(raw)))
    owner_hash = sha256_bytes((root / OWNER_RELATIVE).read_bytes())
    builder_hash = sha256_bytes((root / BUILDER_RELATIVE).read_bytes())
    test_hash = sha256_bytes((root / TEST_RELATIVE).read_bytes())
    contract = cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": CONTRACT_DOMAIN,
                "state": "non_authorizing_cpu_semantic_correction",
                "parent": {"commit": PARENT_COMMIT, "tree": PARENT_TREE},
                "historical": {
                    relative: {"sha256": expected, "immutable": True}
                    for relative, expected in sorted(HISTORICAL_BINDINGS.items())
                },
                "terminal_v3": {
                    "raw": {
                        "path": v3.RAW_RELATIVE,
                        "sha256": V3_RAW_SHA256,
                        "bytes": V3_RAW_BYTES,
                    },
                    "assessment": {
                        "path": v3.ASSESSMENT_RELATIVE,
                        "sha256": V3_ASSESSMENT_SHA256,
                        "bytes": V3_ASSESSMENT_BYTES,
                        "state": "observed_schema_unknown",
                    },
                    "captured_rows": 9,
                    "group_similar": True,
                    "historical_non_exhaustive": True,
                    "terminal_non_reusable": True,
                },
                "installed_semantic_owners": owners,
                "semantic_law": {
                    "exact_filter_to_base_label": ALLOWED_FILTERS,
                    "memory_gb_per_device": MEMORY_GB_PER_DEVICE,
                    "target_memory": "integer_gpu_count_times_48",
                    "eligibility_gpu_count": 2,
                    "literal_false_is_spot_required": True,
                    "spot_inference_forbidden": True,
                    "unrelated_rows_cannot_poison_target_memory": True,
                    "current_or_exhaustive_availability_claimed": False,
                },
                "offline_assessment": {
                    "path": ASSESSMENT_RELATIVE,
                    "sha256": sha256_bytes(expected_assessment),
                    "bytes": len(expected_assessment),
                    "state": "historical_observed_ambiguous_resources",
                    "reason": "non_spot_status_unknown",
                    "atomic_no_overwrite": True,
                },
                "authorization": AUTHORIZATION_FALSE,
            }
        ),
    )
    bindings = {
        OWNER_RELATIVE: owner_hash,
        BUILDER_RELATIVE: builder_hash,
        TEST_RELATIVE: test_hash,
        CONTRACT_RELATIVE: sha256_bytes(contract),
    }
    assessed = _assessment_value(raw)
    semantic = cast(dict[str, Any], assessed["semantic"])
    audit = cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": AUDIT_DOMAIN,
                "state": "non_authorizing_cpu_semantic_correction",
                "parent": {"commit": PARENT_COMMIT, "tree": PARENT_TREE},
                "allowlist": sorted(CHECKPOINT_PATHS),
                "file_bindings": dict(sorted(bindings.items())),
                "installed_source_bindings": owners,
                "terminal_v3_bindings": {
                    "raw_sha256": V3_RAW_SHA256,
                    "assessment_sha256": V3_ASSESSMENT_SHA256,
                },
                "sanitized_counts": {
                    "captured_rows": semantic["captured_row_count"],
                    "target_rows": semantic["target_row_count"],
                    "eligible_target_rows": semantic["eligible_target_row_count"],
                    "unknown_non_spot_rows": semantic["unknown_non_spot_row_count"],
                },
                "expected_disposition": {
                    "state": assessed["state"],
                    "reason": assessed["reason"],
                    "resource": None,
                    "historical_non_exhaustive": True,
                },
                "offline_assessment": {
                    "path": ASSESSMENT_RELATIVE,
                    "sha256": sha256_bytes(expected_assessment),
                    "bytes": len(expected_assessment),
                    "tracked": False,
                },
                "sensitive_raw_values_tracked": False,
                "external_activity": {
                    "prime_calls": 0,
                    "network_calls": 0,
                    "provider_calls": 0,
                    "model_calls": 0,
                    "gpu_calls": 0,
                    "wallet_calls": 0,
                    "source_or_parquet_reads": 0,
                },
                "authorization": AUTHORIZATION_FALSE,
            }
        ),
    )
    return {CONTRACT_RELATIVE: contract, AUDIT_RELATIVE: audit}


def verify_prime_inventory_v4_artifacts(root: Path, output_root: Path) -> dict[str, str]:
    expected = build_prime_inventory_v4_artifacts(root)
    hashes: dict[str, str] = {}
    for relative, raw in expected.items():
        path = output_root / relative
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise ValueError(f"Prime inventory v4 artifact differs: {relative}")
        hashes[relative] = sha256_bytes(raw)
    return hashes


__all__ = [
    "ASSESSMENT_RELATIVE",
    "AUDIT_RELATIVE",
    "CONTRACT_RELATIVE",
    "PARENT_COMMIT",
    "PARENT_TREE",
    "assess_historical_availability_v4",
    "assess_prime_inventory_v4",
    "authenticate_installed_semantic_owners",
    "build_prime_inventory_v4_artifacts",
    "validate_prime_inventory_assessment_v4",
    "verify_prime_inventory_v4_artifacts",
]
