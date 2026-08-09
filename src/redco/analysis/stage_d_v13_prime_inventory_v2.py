"""Strict, non-authorizing Prime CLI 0.6.20 inventory evidence v2."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes
from redco.analysis.stage_d_v13_launch_observations import (
    PRIME_CLI_VERSION,
    PRIME_READ_ONLY_COMMANDS,
    PRIME_VERSION_BANNER,
)
from redco.analysis.stage_d_v13_source_phase_a_decoder import hardened_git
from redco.analysis.stage_d_v13_support_readiness import (
    FIXED_LOCAL_ARTIFACT_ROOT,
    FIXED_PRIME_OBSERVATION_RELATIVE,
    ReadinessBlocked,
    _status_paths,
    validate_local_artifacts,
)

ROOT = Path(__file__).parents[3].resolve()
PARENT_COMMIT = "d3884673faba6dc63916b74960d6a4b5cb691406"
PARENT_TREE = "e6468e9a857dfbc001e4eb6ad81936de234bdd2a"
OBSERVATION_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-observation-v2"
OBSERVATION_SCHEMA_VERSION = 2
CONTRACT_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-contract-v2"
AUDIT_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-audit-v2"
OBSERVATION_TTL_SECONDS = 900
MAX_CAPTURE_BYTES = 256 * 1024
MAX_HOURLY_RATE_USD = Decimal("10000.00")
QUALIFYING_RATE_USD = Decimal("2.00")
FIXED_OBSERVATION_RELATIVE = FIXED_PRIME_OBSERVATION_RELATIVE
PRICE_PATTERN = re.compile(r"^\$(0|[1-9][0-9]*)\.([0-9]{2})$")

CONTRACT_RELATIVE = "configs/stage-d/stage-d1-support-prime-inventory-contract-v2.json"
AUDIT_RELATIVE = "reports/stage-d1-support-prime-inventory-audit-v2.json"
BUILDER_RELATIVE = "scripts/build_stage_d_v13_prime_inventory_v2.py"
OWNER_RELATIVE = "src/redco/analysis/stage_d_v13_prime_inventory_v2.py"
TEST_RELATIVE = "tests/test_stage_d_v13_prime_inventory_v2.py"
CHECKPOINT_PATHS = frozenset(
    {CONTRACT_RELATIVE, AUDIT_RELATIVE, BUILDER_RELATIVE, OWNER_RELATIVE, TEST_RELATIVE}
)

V1_OWNER_RELATIVE = "src/redco/analysis/stage_d_v13_launch_observations.py"
V1_OWNER_SHA256 = "812d2d00c8aa3a71d507e55ddc85d458571a0541560ff6158f646cdf5cebc06f"
V12_DEPENDENCY_RELATIVE = "configs/stage-d/stage-d1-dependency-stack-v12.json"
V12_DEPENDENCY_SHA256 = "cda524c6ecea9821b1e36290da64df465aa46fad9ec174881c24d3dc895b2831"
V1_READINESS_RELATIVE = "configs/stage-d/v13-draft/stage-d1-support-v13-readiness-repair-v1.json"
V1_READINESS_SHA256 = "dab44ad8f48a2c6d5d1264e6e3ec40e28e8dd9e00c28eaade5d34175c8c5f0a2"
V1_DEPENDENCY_RELATIVE = "configs/stage-d/stage-d1-support-readiness-dependency-manifest-v1.json"
V1_DEPENDENCY_SHA256 = "76026b45e2f52b919c43efdf28deabfe13ea0fb913bb6d93611cde3819f63a1e"
V1_AUDIT_RELATIVE = "reports/stage-d1-support-v13-readiness-audit-v1.json"
V1_AUDIT_SHA256 = "6d73f78fadea281fba920867c43f02aee20ef801839c2bad0188e83c792d5926"

RESOURCE_KEYS = {
    "id",
    "cloud_id",
    "gpu_type",
    "gpu_count",
    "socket",
    "provider",
    "location",
    "stock_status",
    "price_per_hour",
    "price_value",
    "security",
    "vcpus",
    "memory_gb",
    "disk_gb",
    "gpu_memory",
    "is_spot",
}
HARDWARE_LABELS = {
    "L40 48GB": "L40 48GB",
    "L40S 48GB": "L40S 48GB",
    "RTX6000Ada 48GB": "RTX 6000 Ada 48GB",
}
FILTER_GPU_TYPES = {"L40_48GB", "L40S_48GB", "RTX6000Ada_48GB"}
AVAILABLE_STATES = {"available", "ready", "in_stock"}
EVIDENCE_STATES = {
    "observed_qualifying_resource",
    "observed_no_qualifying_resource",
    "observed_ambiguous_resources",
}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        return _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error


def _string(value: Mapping[str, Any], key: str, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{label}.{key} must be a nonempty string")
    return result


def _number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _price(value: object) -> tuple[Decimal, float]:
    if isinstance(value, str):
        match = PRICE_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError("Prime hourly price string is malformed")
        amount = Decimal(f"{match.group(1)}.{match.group(2)}")
    elif (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    ):
        try:
            amount = Decimal(str(value))
        except InvalidOperation as error:
            raise ValueError("Prime hourly price number is malformed") from error
    else:
        raise ValueError("Prime hourly price has the wrong type")
    if amount < 0 or amount > MAX_HOURLY_RATE_USD:
        raise ValueError("Prime hourly price exceeds the bounded numeric domain")
    if amount != amount.quantize(Decimal("0.01")):
        raise ValueError("Prime hourly price is not an exact cent value")
    return amount, float(amount)


def _inventory_rows(
    value: Mapping[str, Any], *, key: str, label: str, item_keys: set[str]
) -> list[dict[str, Any]]:
    if set(value) != {key, "total_count", "offset", "limit"}:
        raise ValueError(f"{label} schema differs from Prime CLI 0.6.20")
    rows = value.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"{label}.{key} must be a list")
    if (
        type(value["total_count"]) is not int
        or type(value["offset"]) is not int
        or type(value["limit"]) is not int
        or value["total_count"] != len(rows)
        or value["offset"] < 0
        or value["limit"] < 0
    ):
        raise ValueError(f"{label} pagination is invalid")
    parsed: list[dict[str, Any]] = []
    for raw in rows:
        row = _object(raw, f"{label} row")
        if set(row) != item_keys:
            raise ValueError(f"{label} row schema differs from Prime CLI 0.6.20")
        parsed.append(row)
    return parsed


def _wallet(raw: bytes) -> dict[str, Any]:
    wallet = _json_object(raw, "Prime wallet")
    allowed = {
        "wallet_id",
        "team_id",
        "balance_usd",
        "currency",
        "total_billings",
        "recent_billings",
    }
    if set(wallet) - allowed or not {"wallet_id", "balance_usd", "currency"}.issubset(wallet):
        raise ValueError("Prime wallet schema differs from 0.6.20")
    _string(wallet, "wallet_id", "wallet")
    _string(wallet, "currency", "wallet")
    if wallet.get("team_id") is not None and not isinstance(wallet["team_id"], str):
        raise ValueError("Prime wallet team identity is malformed")
    _number(wallet.get("balance_usd"), "Prime wallet balance")
    if type(wallet.get("total_billings")) is not int or wallet["total_billings"] < 0:
        raise ValueError("Prime wallet billing count is malformed")
    billings = wallet.get("recent_billings")
    if not isinstance(billings, list):
        raise ValueError("Prime wallet billing rows are malformed")
    required = {"id", "created_at", "updated_at", "amount_usd", "currency", "resource_type"}
    allowed_billing = required | {"last_billed_at", "resource_id"}
    for raw_row in billings:
        row = _object(raw_row, "Prime billing row")
        if set(row) - allowed_billing or not required.issubset(row):
            raise ValueError("Prime billing row schema differs from 0.6.20")
        _number(row["amount_usd"], "Prime billing amount")
        if row["currency"] != wallet["currency"]:
            raise ValueError("Prime billing currency differs")
    return wallet


def _parse_resource(raw: object) -> dict[str, Any]:
    resource = _object(raw, "Prime availability resource")
    if set(resource) != RESOURCE_KEYS:
        raise ValueError("Prime availability resource schema differs from 0.6.20")
    for key in ("id", "cloud_id", "gpu_type", "socket", "provider", "location", "stock_status"):
        _string(resource, key, "resource")
    if type(resource["gpu_count"]) is not int or resource["gpu_count"] <= 0:
        raise ValueError("Prime GPU count is malformed")
    aggregate_memory = _number(resource["gpu_memory"], "Prime aggregate GPU memory")
    if aggregate_memory != 48 * resource["gpu_count"]:
        raise ValueError("Prime aggregate GPU memory differs from 48GB per GPU")
    hourly_decimal, hourly_rate = _price(resource["price_per_hour"])
    price_value_decimal, price_value_rate = _price(resource["price_value"])
    if hourly_decimal != price_value_decimal:
        raise ValueError("Prime hourly price fields disagree")
    disk_gb = _number(resource["disk_gb"], "Prime ephemeral disk size")
    if disk_gb < 0:
        raise ValueError("Prime ephemeral disk size is negative")
    if not isinstance(resource["security"], (dict, str)) or not resource["security"]:
        raise ValueError("Prime resource security is missing")
    if resource["is_spot"] is not False and resource["is_spot"] is not None:
        raise ValueError("Prime spot state must be false or null")
    hardware = HARDWARE_LABELS.get(cast(str, resource["gpu_type"]))
    stock = cast(str, resource["stock_status"]).strip().lower()
    stock_normalized = stock if stock in AVAILABLE_STATES else None
    reasons: list[str] = []
    if hardware is None:
        reasons.append("hardware_not_allowed")
    if resource["gpu_count"] != 2:
        reasons.append("gpu_count_not_two")
    if stock_normalized is None:
        reasons.append("stock_not_available")
    if hourly_decimal > QUALIFYING_RATE_USD:
        reasons.append("hourly_rate_above_cap")
    if resource["is_spot"] is None:
        reasons.append("non_spot_status_unknown")
    eligible = not reasons and resource["is_spot"] is False
    return {
        "raw": resource,
        "resource_id": resource["id"],
        "provider": resource["provider"],
        "hardware": hardware,
        "gpu_count": resource["gpu_count"],
        "aggregate_gpu_memory_gb": aggregate_memory,
        "memory_per_gpu_gb": 48,
        "hourly_rate_usd": hourly_rate,
        "price_value_usd": price_value_rate,
        "hourly_rate_cents": int(hourly_decimal * 100),
        "stock_status": stock_normalized,
        "non_spot_status": ("proven_non_spot" if resource["is_spot"] is False else "unknown"),
        "persistent_storage": False,
        "ephemeral_storage_gb": disk_gb,
        "eligible": eligible,
        "ineligibility_reasons": reasons,
    }


def _availability(raw: bytes) -> dict[str, Any]:
    value = _json_object(raw, "Prime availability")
    if set(value) != {"gpu_resources", "total_count", "filters"}:
        raise ValueError("Prime availability schema differs from 0.6.20")
    resources = value.get("gpu_resources")
    if not isinstance(resources, list) or type(value.get("total_count")) is not int:
        raise ValueError("Prime availability resources are malformed")
    if value["total_count"] != len(resources):
        raise ValueError("Prime availability count differs")
    filters = _object(value.get("filters"), "Prime availability filters")
    if set(filters) != {"gpu_count", "gpu_type"}:
        raise ValueError("Prime availability filter schema differs")
    if type(filters["gpu_count"]) is not int or filters["gpu_count"] != 2:
        raise ValueError("Prime availability GPU-count filter differs")
    if filters["gpu_type"] not in FILTER_GPU_TYPES:
        raise ValueError("Prime availability GPU-type filter differs")
    parsed = [_parse_resource(resource) for resource in resources]
    eligible = [resource for resource in parsed if resource["eligible"] is True]
    unknown_nonspot = [
        resource
        for resource in parsed
        if resource["non_spot_status"] == "unknown"
        and set(resource["ineligibility_reasons"]) == {"non_spot_status_unknown"}
    ]
    if len(eligible) == 1 and not unknown_nonspot:
        state = "observed_qualifying_resource"
        selected: dict[str, Any] | None = eligible[0]
    elif len(eligible) > 1 or unknown_nonspot:
        state = "observed_ambiguous_resources"
        selected = None
    else:
        state = "observed_no_qualifying_resource"
        selected = None
    return {
        "raw_filters": filters,
        "total_count": value["total_count"],
        "resources": parsed,
        "eligible_count": len(eligible),
        "unknown_non_spot_count": len(unknown_nonspot),
        "state": state,
        "selected": selected,
    }


def _capture_value(capture: Mapping[str, Any], name: str) -> bytes:
    expected = {
        "argv",
        "returncode",
        "stdout_b64",
        "stderr_b64",
        "stdout_sha256",
        "stderr_sha256",
    }
    if set(capture) != expected or capture["argv"] != list(PRIME_READ_ONLY_COMMANDS[name]):
        raise ValueError(f"Prime {name} capture fields differ")
    if type(capture["returncode"]) is not int or capture["returncode"] != 0:
        raise ValueError(f"Prime {name} command did not succeed")
    for stream in ("stdout", "stderr"):
        encoded = capture[f"{stream}_b64"]
        if not isinstance(encoded, str):
            raise ValueError(f"Prime {name} {stream} is not base64 text")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise ValueError(f"Prime {name} {stream} is not valid base64") from error
        if len(raw) > MAX_CAPTURE_BYTES or capture[f"{stream}_sha256"] != sha256_bytes(raw):
            raise ValueError(f"Prime {name} {stream} binding differs")
        if stream == "stdout":
            stdout = raw
    return stdout


def _facts(commands: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(commands) != set(PRIME_READ_ONLY_COMMANDS):
        raise ValueError("Prime read-only command set differs")
    stdout = {name: _capture_value(commands[name], name) for name in PRIME_READ_ONLY_COMMANDS}
    if stdout["version"].decode("utf-8").strip() != PRIME_VERSION_BANNER:
        raise ValueError("Prime CLI version differs from 0.6.20")
    wallet = _wallet(stdout["wallet"])
    pods = _inventory_rows(
        _json_object(stdout["pods"], "Prime pods"),
        key="pods",
        label="Prime pods",
        item_keys={"id", "name", "gpu", "status", "created_at"},
    )
    disks = _inventory_rows(
        _json_object(stdout["disks"], "Prime disks"),
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
    availability = _availability(stdout["availability"])
    return {"wallet": wallet, "pods": pods, "disks": disks, "availability": availability}


def _git(root: Path, *args: str) -> str:
    result = hardened_git(root, "-c", "core.autocrlf=true", *args, text=True)
    if result.returncode != 0 or not isinstance(result.stdout, str):
        raise ValueError("Prime v2 Git binding failed")
    return result.stdout.strip()


def _payload(root: Path, commands: dict[str, dict[str, Any]], captured_at: int) -> bytes:
    if type(captured_at) is not int:
        raise TypeError("Prime capture time must be an integer epoch")
    facts = _facts(commands)
    availability = cast(dict[str, Any], facts["availability"])
    wallet = cast(dict[str, Any], facts["wallet"])
    value = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "domain": OBSERVATION_DOMAIN,
        "state": availability["state"],
        "captured_at_epoch": captured_at,
        "expires_at_epoch": captured_at + OBSERVATION_TTL_SECONDS,
        "bundle": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        },
        "cli": {"version": PRIME_CLI_VERSION, "banner": PRIME_VERSION_BANNER},
        "wallet": wallet,
        "inventory": {"pods": facts["pods"], "disks": facts["disks"]},
        "availability": {
            key: availability[key]
            for key in (
                "raw_filters",
                "total_count",
                "resources",
                "eligible_count",
                "unknown_non_spot_count",
            )
        },
        "resource": availability["selected"],
        "commands": commands,
        "authorization": {
            "candidate_authorized": False,
            "live_authorized": False,
            "prime_authorized": False,
            "provider_calls_authorized": False,
            "model_calls_authorized": False,
            "support_launch_authorized": False,
            "science_authorized": False,
        },
    }
    return cast(bytes, canonical_json_bytes(value))


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class PrimeInventoryObservationProducerV2:
    """Capture the five frozen subprocess commands; callers supply no facts."""

    root: Path

    def _run_command(self, argv: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(list(argv), cwd=self.root, check=False, capture_output=True)

    def capture(self, *, captured_at_epoch: int | None = None) -> bytes:
        commands: dict[str, dict[str, Any]] = {}
        for name, argv in PRIME_READ_ONLY_COMMANDS.items():
            result = self._run_command(argv)
            stdout = bytes(result.stdout)
            stderr = bytes(result.stderr)
            if len(stdout) > MAX_CAPTURE_BYTES or len(stderr) > MAX_CAPTURE_BYTES:
                raise ValueError(f"Prime {name} output exceeds the bounded capture")
            commands[name] = {
                "argv": list(argv),
                "returncode": result.returncode,
                "stdout_b64": base64.b64encode(stdout).decode("ascii"),
                "stderr_b64": base64.b64encode(stderr).decode("ascii"),
                "stdout_sha256": sha256_bytes(stdout),
                "stderr_sha256": sha256_bytes(stderr),
            }
        captured = int(time.time()) if captured_at_epoch is None else captured_at_epoch
        return _payload(self.root.resolve(), commands, captured)

    def capture_to(self, path: Path, *, captured_at_epoch: int | None = None) -> bytes:
        value = self.capture(captured_at_epoch=captured_at_epoch)
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError as error:
            raise ValueError("Prime v2 observation path escapes the repository") from error
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_parent(path)
        return value


def validate_prime_inventory_observation_v2(
    root: Path, path: Path, *, now_epoch: int | None = None
) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Prime v2 observation is not JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError("Prime v2 observation is not canonical")
    expected_keys = {
        "schema_version",
        "domain",
        "state",
        "captured_at_epoch",
        "expires_at_epoch",
        "bundle",
        "cli",
        "wallet",
        "inventory",
        "availability",
        "resource",
        "commands",
        "authorization",
    }
    if (
        set(value) != expected_keys
        or value["schema_version"] != 2
        or value["domain"] != OBSERVATION_DOMAIN
    ):
        raise ValueError("Prime v2 observation fields differ")
    captured = value["captured_at_epoch"]
    expires = value["expires_at_epoch"]
    now = int(time.time()) if now_epoch is None else now_epoch
    if (
        type(captured) is not int
        or type(expires) is not int
        or expires != captured + OBSERVATION_TTL_SECONDS
        or captured > now
        or expires < now
    ):
        raise ValueError("Prime v2 observation TTL is invalid")
    if value["bundle"] != {
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
    }:
        raise ValueError("Prime v2 observation checkout binding differs")
    commands = _object(value["commands"], "Prime v2 commands")
    facts = _facts(cast(dict[str, dict[str, Any]], commands))
    availability = cast(dict[str, Any], facts["availability"])
    expected_availability = {
        key: availability[key]
        for key in (
            "raw_filters",
            "total_count",
            "resources",
            "eligible_count",
            "unknown_non_spot_count",
        )
    }
    if value["cli"] != {"version": PRIME_CLI_VERSION, "banner": PRIME_VERSION_BANNER}:
        raise ValueError("Prime v2 CLI binding differs")
    if value["wallet"] != facts["wallet"] or value["inventory"] != {
        "pods": facts["pods"],
        "disks": facts["disks"],
    }:
        raise ValueError("Prime v2 wallet or inventory differs from raw captures")
    if value["availability"] != expected_availability:
        raise ValueError("Prime v2 availability differs from raw capture")
    if value["state"] != availability["state"] or value["state"] not in EVIDENCE_STATES:
        raise ValueError("Prime v2 capacity state differs")
    if value["resource"] != availability["selected"]:
        raise ValueError("Prime v2 selected-resource relationship differs")
    if value["authorization"] != {
        "candidate_authorized": False,
        "live_authorized": False,
        "prime_authorized": False,
        "provider_calls_authorized": False,
        "model_calls_authorized": False,
        "support_launch_authorized": False,
        "science_authorized": False,
    }:
        raise ValueError("Prime v2 evidence is authorizing")
    return cast(dict[str, Any], value)


def validate_future_prime_readiness_v2() -> dict[str, Any]:
    """No-argument fixed-path gate; evidence never grants launch authority."""

    validate_local_artifacts()
    value = validate_prime_inventory_observation_v2(ROOT, ROOT / FIXED_OBSERVATION_RELATIVE)
    if value["state"] != "observed_qualifying_resource":
        raise ReadinessBlocked("Prime v2 capacity evidence is non-qualifying or ambiguous")
    if value["wallet"]["balance_usd"] < 30:
        raise ReadinessBlocked("Prime wallet does not cover frozen reserves")
    if value["inventory"] != {"pods": [], "disks": []}:
        raise ReadinessBlocked("Prime inventory is not empty")
    return value


def _bound_file(root: Path, relative: str, expected: str) -> None:
    path = root / relative
    if path.is_symlink() or not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
        raise ValueError(f"immutable v1/v12 binding differs: {relative}")


def _authenticate_precommit(root: Path) -> None:
    if _git(root, "rev-parse", "HEAD") != PARENT_COMMIT:
        raise ValueError("Prime inventory v2 build requires exact parent d3884673")
    if _git(root, "rev-parse", "HEAD^{tree}") != PARENT_TREE:
        raise ValueError("Prime inventory v2 parent tree differs")
    unexpected = _status_paths(root).difference(CHECKPOINT_PATHS)
    if unexpected:
        raise ValueError(
            "Prime inventory v2 worktree exceeds its exact allowlist: "
            + ", ".join(sorted(unexpected))
        )


def build_prime_inventory_v2_artifacts(root: Path) -> dict[str, bytes]:
    root = root.resolve()
    _authenticate_precommit(root)
    immutable = {
        V1_OWNER_RELATIVE: V1_OWNER_SHA256,
        V12_DEPENDENCY_RELATIVE: V12_DEPENDENCY_SHA256,
        V1_READINESS_RELATIVE: V1_READINESS_SHA256,
        V1_DEPENDENCY_RELATIVE: V1_DEPENDENCY_SHA256,
        V1_AUDIT_RELATIVE: V1_AUDIT_SHA256,
    }
    for relative, expected in immutable.items():
        _bound_file(root, relative, expected)
    owner_sha = sha256_bytes((root / OWNER_RELATIVE).read_bytes())
    builder_sha = sha256_bytes((root / BUILDER_RELATIVE).read_bytes())
    test_sha = sha256_bytes((root / TEST_RELATIVE).read_bytes())
    contract = cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": 2,
                "domain": CONTRACT_DOMAIN,
                "state": "non_authorizing_cpu_inventory_contract",
                "parent": {"commit": PARENT_COMMIT, "tree": PARENT_TREE},
                "historical": {
                    relative: {"sha256": expected, "immutable": True}
                    for relative, expected in sorted(immutable.items())
                },
                "owner": {"path": OWNER_RELATIVE, "sha256": owner_sha},
                "fixed_observation_path": FIXED_OBSERVATION_RELATIVE,
                "fixed_artifact_root": FIXED_LOCAL_ARTIFACT_ROOT,
                "prime_cli": {
                    "version": PRIME_CLI_VERSION,
                    "banner": PRIME_VERSION_BANNER,
                    "commands": {
                        name: list(argv) for name, argv in PRIME_READ_ONLY_COMMANDS.items()
                    },
                    "ttl_seconds": OBSERVATION_TTL_SECONDS,
                },
                "capacity": {
                    "states": sorted(EVIDENCE_STATES),
                    "hardware_labels": dict(sorted(HARDWARE_LABELS.items())),
                    "gpu_count": 2,
                    "memory_per_gpu_gb": 48,
                    "maximum_hourly_rate_usd": 2,
                    "price_binding": (
                        "price_per_hour and price_value independently parse to exact cents "
                        "and must agree"
                    ),
                    "non_spot_proof": "is_spot must be exact false; null is unknown",
                    "multiple_or_unknown": "observed_ambiguous_resources",
                    "persistent_storage": False,
                },
                "authorization": {
                    "candidate_authorized": False,
                    "live_authorized": False,
                    "prime_authorized": False,
                    "provider_calls_authorized": False,
                    "model_calls_authorized": False,
                    "support_launch_authorized": False,
                    "science_authorized": False,
                },
            }
        ),
    )
    bindings = {
        OWNER_RELATIVE: owner_sha,
        BUILDER_RELATIVE: builder_sha,
        TEST_RELATIVE: test_sha,
        CONTRACT_RELATIVE: sha256_bytes(contract),
    }
    audit = cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": 2,
                "domain": AUDIT_DOMAIN,
                "state": "non_authorizing_cpu_evidence_repair",
                "parent": {"commit": PARENT_COMMIT, "tree": PARENT_TREE},
                "allowlist": sorted(CHECKPOINT_PATHS),
                "file_bindings": dict(sorted(bindings.items())),
                "self_hash": "excluded_to_avoid_circular_binding",
                "raw_wallet_or_billing_tracked": False,
                "external_activity": {
                    "prime_calls": 0,
                    "network_calls": 0,
                    "provider_calls": 0,
                    "model_calls": 0,
                    "gpu_calls": 0,
                    "wallet_calls": 0,
                    "source_or_parquet_reads": 0,
                },
                "authorization": {
                    "candidate_authorized": False,
                    "live_authorized": False,
                    "prime_authorized": False,
                    "support_launch_authorized": False,
                    "science_authorized": False,
                },
            }
        ),
    )
    return {CONTRACT_RELATIVE: contract, AUDIT_RELATIVE: audit}


def verify_prime_inventory_v2_artifacts(root: Path, output_root: Path) -> dict[str, str]:
    expected = build_prime_inventory_v2_artifacts(root)
    hashes: dict[str, str] = {}
    for relative, value in expected.items():
        path = output_root / relative
        if path.is_symlink() or not path.is_file() or path.read_bytes() != value:
            raise ValueError(f"Prime inventory v2 artifact differs: {relative}")
        hashes[relative] = sha256_bytes(value)
    return hashes


__all__ = [
    "AUDIT_RELATIVE",
    "CHECKPOINT_PATHS",
    "CONTRACT_RELATIVE",
    "FIXED_OBSERVATION_RELATIVE",
    "OBSERVATION_DOMAIN",
    "PARENT_COMMIT",
    "PARENT_TREE",
    "PRIME_READ_ONLY_COMMANDS",
    "PrimeInventoryObservationProducerV2",
    "build_prime_inventory_v2_artifacts",
    "validate_future_prime_readiness_v2",
    "validate_prime_inventory_observation_v2",
    "verify_prime_inventory_v2_artifacts",
]
