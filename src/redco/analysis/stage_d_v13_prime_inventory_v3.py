"""Immutable raw Prime receipt and pure non-authorizing inventory assessment v3."""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from redco.analysis import stage_d_v13_prime_inventory_v2 as v2
from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes
from redco.analysis.stage_d_v13_source_phase_a_decoder import hardened_git
from redco.analysis.stage_d_v13_support_readiness import (
    FIXED_LOCAL_ARTIFACT_ROOT,
    _status_paths,
)

ROOT = Path(__file__).parents[3].resolve()
PARENT_COMMIT = "267ee8dc6e40363be5137c37bc7839fb45d985b9"
PARENT_TREE = "84009c252b38dd987682432f1ebfd21e322a3933"
RAW_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-raw-v3"
ASSESSMENT_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-assessment-v3"
CONTRACT_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-contract-v3"
AUDIT_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-audit-v3"
SCHEMA_VERSION = 3
TTL_SECONDS = 900
MAX_CAPTURE_BYTES = 256 * 1024
MAX_DISK_TEXT_BYTES = 32
MAX_DISK_GB = 1_000_000
DISK_PATTERN = re.compile(r"^(0|[1-9][0-9]{0,5})(\+)?$")

RAW_RELATIVE = "runs/stage-d/stage-d1-support-v13-readiness/prime-inventory-raw-v3.json"
ASSESSMENT_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-readiness/prime-inventory-assessment-v3.json"
)
OWNER_RELATIVE = "src/redco/analysis/stage_d_v13_prime_inventory_v3.py"
BUILDER_RELATIVE = "scripts/build_stage_d_v13_prime_inventory_v3.py"
TEST_RELATIVE = "tests/test_stage_d_v13_prime_inventory_v3.py"
CONTRACT_RELATIVE = "configs/stage-d/stage-d1-support-prime-inventory-contract-v3.json"
AUDIT_RELATIVE = "reports/stage-d1-support-prime-inventory-audit-v3.json"
CHECKPOINT_PATHS = frozenset(
    {OWNER_RELATIVE, BUILDER_RELATIVE, TEST_RELATIVE, CONTRACT_RELATIVE, AUDIT_RELATIVE}
)

PRIME_SOURCE_RELATIVE = "prime_cli/commands/availability.py"
PRIME_SOURCE_SHA256 = "9b9e72810b138d66278e9375988db1a9ae847d1d0fbee58424cc5c63554a83fa"
PRIME_SOURCE_SIZE = 16_169
PRIME_EXECUTABLE_SHA256 = "67ea90a0fc17020a26e7be97e4544baead1b1b3ef2246d9c378ba03904c6677d"
PRIME_EXECUTABLE_SIZE = 40_982
PRIME_UV_RECEIPT_SHA256 = "2c0950773cb2d5b76c076f3d45e3c17d625fed406b32c3364b2e52e940dd5895"
PRIME_UV_RECEIPT_SIZE = 143
PRIME_EXECUTABLE_LOCATOR = "%APPDATA%/uv/tools/prime/Scripts/prime.exe"
PRIME_ENTRYPOINT_LOCATOR = "%USERPROFILE%/.local/bin/prime.exe"
PRIME_SOURCE_REQUIRED_LINES = (
    "disk_info: str = str(gpu.disk.default_count)",
    'disk_info = f"{gpu.disk.default_count}+"',
    '"disk_gb": str(gpu_entry["disk"]),',
)

HISTORICAL_BINDINGS = {
    v2.V1_OWNER_RELATIVE: v2.V1_OWNER_SHA256,
    v2.V12_DEPENDENCY_RELATIVE: v2.V12_DEPENDENCY_SHA256,
    v2.V1_READINESS_RELATIVE: v2.V1_READINESS_SHA256,
    v2.V1_DEPENDENCY_RELATIVE: v2.V1_DEPENDENCY_SHA256,
    v2.V1_AUDIT_RELATIVE: v2.V1_AUDIT_SHA256,
    v2.OWNER_RELATIVE: "5289e09d926976ac67a9894a50f255db233b83d9ab06fb1c706412d771f3dedf",
    v2.BUILDER_RELATIVE: "5409512bb523b0b73a2c7f19523109cff9f48d9211882a2049ba84694127a8a6",
    v2.TEST_RELATIVE: "f3e68d043544a9c02268ddade4f0448590a55bee671da67b2ce58abb0c60ffba",
    v2.CONTRACT_RELATIVE: "521f601fe4422b7c0c5d736c5703cb0c0558471bf66af517650da8427d14ba97",
    v2.AUDIT_RELATIVE: "58f84d944f024571bb3dd6bf70d3034a5ccda300f2102f2306c3b158dd792487",
}

AUTHORIZATION_FALSE = {
    "candidate_authorized": False,
    "live_authorized": False,
    "prime_authorized": False,
    "provider_calls_authorized": False,
    "model_calls_authorized": False,
    "provisioning_authorized": False,
    "support_launch_authorized": False,
    "science_authorized": False,
}


def _prime_source_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise ValueError("the fixed local uv tool root is unavailable")
    return (
        Path(appdata)
        / "uv"
        / "tools"
        / "prime"
        / "Lib"
        / "site-packages"
        / "prime_cli"
        / "commands"
        / "availability.py"
    )


def _prime_tool_paths() -> tuple[Path, Path, Path]:
    appdata = os.environ.get("APPDATA")
    profile = os.environ.get("USERPROFILE")
    if not appdata or not profile:
        raise ValueError("the fixed local uv tool identity is unavailable")
    tool_root = Path(appdata) / "uv" / "tools" / "prime"
    return (
        tool_root / "Scripts" / "prime.exe",
        Path(profile) / ".local" / "bin" / "prime.exe",
        tool_root / "uv-receipt.toml",
    )


def authenticate_installed_prime_source() -> dict[str, object]:
    path = _prime_source_path()
    if path.is_symlink() or not path.is_file():
        raise ValueError("installed Prime availability owner is unavailable")
    raw = path.read_bytes()
    if len(raw) != PRIME_SOURCE_SIZE or sha256_bytes(raw) != PRIME_SOURCE_SHA256:
        raise ValueError("installed Prime availability owner binding differs")
    text = raw.decode("utf-8")
    if any(line not in text for line in PRIME_SOURCE_REQUIRED_LINES):
        raise ValueError("installed Prime disk serialization law differs")
    return {
        "distribution": "prime",
        "version": v2.PRIME_CLI_VERSION,
        "module_path": PRIME_SOURCE_RELATIVE,
        "sha256": PRIME_SOURCE_SHA256,
        "bytes": PRIME_SOURCE_SIZE,
    }


def authenticate_installed_prime_executable() -> dict[str, object]:
    executable, entrypoint, receipt = _prime_tool_paths()
    bindings = (
        (executable, "tool executable", PRIME_EXECUTABLE_SHA256, PRIME_EXECUTABLE_SIZE),
        (entrypoint, "uv entrypoint", PRIME_EXECUTABLE_SHA256, PRIME_EXECUTABLE_SIZE),
        (receipt, "uv receipt", PRIME_UV_RECEIPT_SHA256, PRIME_UV_RECEIPT_SIZE),
    )
    for path, label, expected_hash, expected_size in bindings:
        if _is_link_or_reparse(path) or not path.is_file():
            raise ValueError(f"installed Prime {label} is unavailable")
        raw = path.read_bytes()
        if len(raw) != expected_size or sha256_bytes(raw) != expected_hash:
            raise ValueError(f"installed Prime {label} binding differs")
    expected_receipt = (
        b'[tool]\nrequirements = [{ name = "prime" }]\nentrypoints = [\n'
        b'    { name = "prime.exe", install-path = "C:/Users/mihir/.local/bin/prime.exe" },\n'
        b"]\n"
    )
    if receipt.read_bytes() != expected_receipt:
        raise ValueError("installed Prime uv receipt content differs")
    resolved = shutil.which("prime")
    if resolved is None:
        raise ValueError("Prime PATH entrypoint is unavailable")
    resolved_path = Path(resolved)
    if resolved_path.resolve() != entrypoint.resolve():
        raise ValueError("Prime PATH entrypoint is shadowed")
    if (
        resolved_path.stat().st_size != PRIME_EXECUTABLE_SIZE
        or sha256_bytes(resolved_path.read_bytes()) != PRIME_EXECUTABLE_SHA256
    ):
        raise ValueError("Prime PATH entrypoint bytes differ")
    return {
        "canonical_locator": PRIME_EXECUTABLE_LOCATOR,
        "canonical_path": str(executable.resolve()),
        "entrypoint_locator": PRIME_ENTRYPOINT_LOCATOR,
        "entrypoint_path": str(entrypoint.resolve()),
        "identity": (
            "samefile"
            if executable.samefile(entrypoint)
            else "uv_receipt_bound_hash_equivalent_entrypoint"
        ),
        "sha256": PRIME_EXECUTABLE_SHA256,
        "bytes": PRIME_EXECUTABLE_SIZE,
        "uv_receipt_sha256": PRIME_UV_RECEIPT_SHA256,
    }


def _git(root: Path, *arguments: str) -> str:
    result = hardened_git(
        root,
        "-c",
        "core.autocrlf=true",
        *arguments,
        text=True,
    )
    if result.returncode != 0 or not isinstance(result.stdout, str):
        raise ValueError("Prime inventory v3 Git binding failed")
    return result.stdout.strip()


def _run_prime_command(argv: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(list(argv), cwd=ROOT, check=False, capture_output=True)


def _capture_record(
    argv: tuple[str, ...],
    executed_argv: tuple[str, ...],
    result: subprocess.CompletedProcess[bytes],
) -> dict[str, object]:
    stdout = bytes(result.stdout)
    stderr = bytes(result.stderr)
    if len(stdout) > MAX_CAPTURE_BYTES or len(stderr) > MAX_CAPTURE_BYTES:
        raise ValueError("Prime raw command output exceeds the bounded capture")
    return {
        "argv": list(argv),
        "executed_argv": list(executed_argv),
        "returncode": result.returncode,
        "stdout_b64": base64.b64encode(stdout).decode("ascii"),
        "stderr_b64": base64.b64encode(stderr).decode("ascii"),
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
    }


def _is_link_or_reparse(path: Path) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    reparse = getattr(status, "st_file_attributes", 0) & 0x400
    return path.is_symlink() or bool(reparse)


def _fixed_path(relative: str) -> Path:
    original_root = ROOT.absolute()
    current_root = original_root
    while current_root != current_root.parent:
        if _is_link_or_reparse(current_root):
            raise ValueError("Prime inventory v3 repository ancestor is a link or reparse point")
        current_root = current_root.parent
    root = ROOT.resolve()
    path = root / relative
    if path.is_absolute() and not path.resolve(strict=False).is_relative_to(root):
        raise ValueError("Prime inventory v3 fixed path escapes the repository")
    current = path.parent
    while current != root.parent:
        if _is_link_or_reparse(current):
            raise ValueError("Prime inventory v3 output ancestor is a link or reparse point")
        if current == root:
            break
        current = current.parent
    return path


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_fixed(relative: str, raw: bytes) -> Path:
    path = _fixed_path(relative)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Prime inventory v3 evidence already exists: {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(path.parent):
        raise ValueError("Prime inventory v3 output parent is a link or reparse point")
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def capture_prime_inventory_raw_v3() -> bytes:
    """Run the five frozen commands and durably publish their raw receipt once."""

    raw_path = _fixed_path(RAW_RELATIVE)
    if raw_path.exists() or raw_path.is_symlink():
        raise FileExistsError(
            f"Prime inventory v3 evidence already exists: {RAW_RELATIVE}"
        )
    installed = authenticate_installed_prime_source()
    executable = authenticate_installed_prime_executable()
    executable_path = cast(str, executable["canonical_path"])
    commands = {
        name: _capture_record(
            argv,
            (executable_path, *argv[1:]),
            _run_prime_command((executable_path, *argv[1:])),
        )
        for name, argv in v2.PRIME_READ_ONLY_COMMANDS.items()
    }
    captured = int(time.time())
    raw = cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": RAW_DOMAIN,
                "state": "captured_raw_terminal",
                "captured_at_epoch": captured,
                "expires_at_epoch": captured + TTL_SECONDS,
                "checkout": {
                    "commit": _git(ROOT, "rev-parse", "HEAD"),
                    "tree": _git(ROOT, "rev-parse", "HEAD^{tree}"),
                },
                "cli": {
                    "version": v2.PRIME_CLI_VERSION,
                    "banner": v2.PRIME_VERSION_BANNER,
                    "installed_owner": installed,
                    "installed_executable": executable,
                },
                "commands": commands,
                "authorization": AUTHORIZATION_FALSE,
            }
        ),
    )
    _publish_fixed(RAW_RELATIVE, raw)
    return raw


def _decode_capture(
    captures: Mapping[str, object], name: str, executable_path: str
) -> tuple[int, bytes, bytes]:
    raw_capture = captures.get(name)
    if not isinstance(raw_capture, dict):
        raise ValueError(f"Prime raw {name} capture is missing")
    capture = cast(dict[str, object], raw_capture)
    expected = {
        "argv",
        "executed_argv",
        "returncode",
        "stdout_b64",
        "stderr_b64",
        "stdout_sha256",
        "stderr_sha256",
    }
    logical_argv = v2.PRIME_READ_ONLY_COMMANDS[name]
    if (
        set(capture) != expected
        or capture["argv"] != list(logical_argv)
        or capture["executed_argv"] != [executable_path, *logical_argv[1:]]
    ):
        raise ValueError(f"Prime raw {name} command binding differs")
    if type(capture["returncode"]) is not int:
        raise ValueError(f"Prime raw {name} return code is malformed")
    decoded: dict[str, bytes] = {}
    for stream in ("stdout", "stderr"):
        encoded = capture[f"{stream}_b64"]
        if not isinstance(encoded, str):
            raise ValueError(f"Prime raw {name} {stream} is malformed")
        try:
            value = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise ValueError(f"Prime raw {name} {stream} is malformed") from error
        if len(value) > MAX_CAPTURE_BYTES or capture[f"{stream}_sha256"] != sha256_bytes(value):
            raise ValueError(f"Prime raw {name} {stream} hash differs")
        decoded[stream] = value
    return capture["returncode"], decoded["stdout"], decoded["stderr"]


def _validate_raw_bytes(raw: bytes, *, now_epoch: int) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Prime raw v3 receipt is not JSON") from error
    if not isinstance(parsed, dict) or canonical_json_bytes(parsed) != raw:
        raise ValueError("Prime raw v3 receipt is not canonical")
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
        set(parsed) != expected
        or parsed["schema_version"] != SCHEMA_VERSION
        or parsed["domain"] != RAW_DOMAIN
        or parsed["state"] != "captured_raw_terminal"
        or parsed["authorization"] != AUTHORIZATION_FALSE
    ):
        raise ValueError("Prime raw v3 receipt fields differ")
    captured = parsed["captured_at_epoch"]
    expires = parsed["expires_at_epoch"]
    if (
        type(captured) is not int
        or type(expires) is not int
        or expires != captured + TTL_SECONDS
        or captured > now_epoch
        or expires < now_epoch
    ):
        raise ValueError("Prime raw v3 receipt TTL differs")
    if parsed["checkout"] != {
        "commit": _git(ROOT, "rev-parse", "HEAD"),
        "tree": _git(ROOT, "rev-parse", "HEAD^{tree}"),
    }:
        raise ValueError("Prime raw v3 checkout binding differs")
    installed = authenticate_installed_prime_source()
    executable = authenticate_installed_prime_executable()
    if parsed["cli"] != {
        "version": v2.PRIME_CLI_VERSION,
        "banner": v2.PRIME_VERSION_BANNER,
        "installed_owner": installed,
        "installed_executable": executable,
    }:
        raise ValueError("Prime raw v3 CLI binding differs")
    captures = parsed["commands"]
    if not isinstance(captures, dict) or set(captures) != set(v2.PRIME_READ_ONLY_COMMANDS):
        raise ValueError("Prime raw v3 command set differs")
    for name in v2.PRIME_READ_ONLY_COMMANDS:
        _decode_capture(
            cast(dict[str, object], captures),
            name,
            cast(str, executable["canonical_path"]),
        )
    return cast(dict[str, Any], parsed)


def validate_prime_inventory_raw_v3(*, now_epoch: int | None = None) -> dict[str, Any]:
    now = int(time.time()) if now_epoch is None else now_epoch
    return _validate_raw_bytes(_fixed_path(RAW_RELATIVE).read_bytes(), now_epoch=now)


def _disk_capability(raw: object) -> dict[str, object]:
    if not isinstance(raw, str):
        return {"raw": raw, "status": "unknown", "reason": "wrong_type"}
    if len(raw.encode("utf-8")) > MAX_DISK_TEXT_BYTES:
        return {"raw": raw, "status": "unknown", "reason": "oversized"}
    match = DISK_PATTERN.fullmatch(raw)
    if match is None:
        return {"raw": raw, "status": "unknown", "reason": "malformed"}
    minimum = int(match.group(1))
    if minimum > MAX_DISK_GB:
        return {"raw": raw, "status": "unknown", "reason": "out_of_range"}
    return {
        "raw": raw,
        "status": "proven",
        "minimum_ephemeral_gb": minimum,
        "expandable": match.group(2) == "+",
    }


def _parse_v3_resource(raw: object) -> dict[str, Any]:
    resource = v2._object(raw, "Prime availability resource")
    if set(resource) != v2.RESOURCE_KEYS:
        raise ValueError("Prime availability resource schema differs from 0.6.20")
    for key in (
        "id",
        "cloud_id",
        "gpu_type",
        "socket",
        "provider",
        "location",
        "stock_status",
    ):
        v2._string(resource, key, "resource")
    if type(resource["gpu_count"]) is not int or resource["gpu_count"] <= 0:
        raise ValueError("Prime GPU count is malformed")
    memory = v2._number(resource["gpu_memory"], "Prime aggregate GPU memory")
    if memory != 48 * resource["gpu_count"]:
        raise ValueError("Prime aggregate GPU memory differs from 48GB per GPU")
    display_price, hourly_rate = v2._price(resource["price_per_hour"])
    numeric_price, numeric_rate = v2._price(resource["price_value"])
    if display_price != numeric_price:
        raise ValueError("Prime hourly price fields disagree")
    if resource["is_spot"] is not False and resource["is_spot"] is not None:
        raise ValueError("Prime spot state must be false or null")
    capability = _disk_capability(resource["disk_gb"])
    hardware = v2.HARDWARE_LABELS.get(cast(str, resource["gpu_type"]))
    stock = cast(str, resource["stock_status"]).strip().lower()
    normalized_stock = stock if stock in v2.AVAILABLE_STATES else None
    reasons: list[str] = []
    if hardware is None:
        reasons.append("hardware_not_allowed")
    if resource["gpu_count"] != 2:
        reasons.append("gpu_count_not_two")
    if normalized_stock is None:
        reasons.append("stock_not_available")
    if display_price > v2.QUALIFYING_RATE_USD:
        reasons.append("hourly_rate_above_cap")
    if resource["is_spot"] is None:
        reasons.append("non_spot_status_unknown")
    if capability["status"] != "proven":
        reasons.append("disk_capability_unknown")
    elif capability["minimum_ephemeral_gb"] == 0:
        reasons.append("disk_capability_not_positive")
    return {
        "raw": resource,
        "resource_id": resource["id"],
        "provider": resource["provider"],
        "hardware": hardware,
        "gpu_count": resource["gpu_count"],
        "aggregate_gpu_memory_gb": memory,
        "hourly_rate_usd": hourly_rate,
        "price_value_usd": numeric_rate,
        "hourly_rate_cents": int(display_price * 100),
        "stock_status": normalized_stock,
        "non_spot_status": ("proven_non_spot" if resource["is_spot"] is False else "unknown"),
        "persistent_storage": False,
        "disk_capability": capability,
        "eligible": not reasons and resource["is_spot"] is False,
        "ineligibility_reasons": reasons,
    }


def _filters(value: object) -> dict[str, Any]:
    filters = v2._object(value, "Prime availability filters")
    expected = {"gpu_count", "gpu_type", "regions", "socket", "provider", "group_similar"}
    if set(filters) != expected:
        raise ValueError("Prime availability filter schema differs")
    if type(filters["gpu_count"]) is not int or filters["gpu_count"] != 2:
        raise ValueError("Prime availability GPU-count filter differs")
    if filters["gpu_type"] not in v2.FILTER_GPU_TYPES:
        raise ValueError("Prime availability GPU-type filter differs")
    regions = filters["regions"]
    if regions is not None and (
        not isinstance(regions, list)
        or not regions
        or any(not isinstance(region, str) or not region.strip() for region in regions)
        or len(set(regions)) != len(regions)
    ):
        raise ValueError("Prime availability regions filter is malformed")
    for key in ("socket", "provider"):
        field = filters[key]
        if field is not None and (not isinstance(field, str) or not field.strip()):
            raise ValueError(f"Prime availability {key} filter is malformed")
    if type(filters["group_similar"]) is not bool or filters["group_similar"] is not True:
        raise ValueError("Prime availability grouping filter differs")
    return cast(dict[str, Any], filters)


def _semantic(raw_receipt: dict[str, Any]) -> dict[str, Any]:
    captures = cast(dict[str, object], raw_receipt["commands"])
    executable_path = cast(
        str, raw_receipt["cli"]["installed_executable"]["canonical_path"]
    )
    streams = {
        name: _decode_capture(captures, name, executable_path)
        for name in v2.PRIME_READ_ONLY_COMMANDS
    }
    if any(returncode != 0 for returncode, _stdout, _stderr in streams.values()):
        raise ValueError("one or more Prime read-only commands failed")
    if streams["version"][1].decode("utf-8").strip() != v2.PRIME_VERSION_BANNER:
        raise ValueError("Prime CLI banner differs")
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
    availability = v2._json_object(streams["availability"][1], "Prime availability")
    if set(availability) != {"gpu_resources", "total_count", "filters"}:
        raise ValueError("Prime availability schema differs")
    raw_resources = availability["gpu_resources"]
    if not isinstance(raw_resources, list) or type(availability["total_count"]) is not int:
        raise ValueError("Prime availability resources are malformed")
    if availability["total_count"] != len(raw_resources):
        raise ValueError("Prime availability count differs")
    parsed = [_parse_v3_resource(resource) for resource in raw_resources]
    eligible = [resource for resource in parsed if resource["eligible"] is True]
    disk_unknown = [
        resource for resource in parsed if resource["disk_capability"]["status"] == "unknown"
    ]
    otherwise_unknown_spot = [
        resource
        for resource in parsed
        if resource["non_spot_status"] == "unknown"
        and set(resource["ineligibility_reasons"]) == {"non_spot_status_unknown"}
    ]
    wallet_ok = float(wallet["balance_usd"]) > 30
    inventory_empty = not pods and not disks
    selected: dict[str, Any] | None = None
    if disk_unknown:
        state = "observed_schema_unknown"
    elif len(eligible) > 1 or otherwise_unknown_spot:
        state = "observed_ambiguous_resources"
    elif len(eligible) == 1 and wallet_ok and inventory_empty:
        state = "observed_qualifying_resource"
        selected = eligible[0]
    else:
        state = "observed_no_qualifying_resource"
    return {
        "state": state,
        "wallet": wallet,
        "inventory": {"pods": pods, "disks": disks},
        "availability": {
            "raw_filters": _filters(availability["filters"]),
            "total_count": availability["total_count"],
            "resources": parsed,
            "eligible_count": len(eligible),
            "unknown_disk_count": len(disk_unknown),
            "unknown_non_spot_count": len(otherwise_unknown_spot),
        },
        "resource": selected,
    }


def assess_prime_inventory_v3() -> bytes:
    """Authenticate the fixed raw receipt and publish one pure semantic assessment."""

    raw_path = _fixed_path(RAW_RELATIVE)
    raw = raw_path.read_bytes()
    receipt = _validate_raw_bytes(raw, now_epoch=int(time.time()))
    try:
        semantic = _semantic(receipt)
        diagnostic = None
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        semantic = {
            "state": "observed_schema_unknown",
            "wallet": None,
            "inventory": None,
            "availability": None,
            "resource": None,
        }
        diagnostic = {"code": "semantic_schema_unknown", "message": str(error)}
    assessment = cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": ASSESSMENT_DOMAIN,
                "state": semantic["state"],
                "raw_receipt": {"path": RAW_RELATIVE, "sha256": sha256_bytes(raw)},
                "semantic": {
                    "wallet": semantic["wallet"],
                    "inventory": semantic["inventory"],
                    "availability": semantic["availability"],
                },
                "resource": semantic["resource"],
                "diagnostic": diagnostic,
                "authorization": AUTHORIZATION_FALSE,
            }
        ),
    )
    _publish_fixed(ASSESSMENT_RELATIVE, assessment)
    return assessment


def validate_prime_inventory_assessment_v3(*, now_epoch: int | None = None) -> dict[str, Any]:
    now = int(time.time()) if now_epoch is None else now_epoch
    raw = _fixed_path(RAW_RELATIVE).read_bytes()
    receipt = _validate_raw_bytes(raw, now_epoch=now)
    assessment_raw = _fixed_path(ASSESSMENT_RELATIVE).read_bytes()
    try:
        assessment = json.loads(assessment_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Prime v3 assessment is not JSON") from error
    if not isinstance(assessment, dict) or canonical_json_bytes(assessment) != assessment_raw:
        raise ValueError("Prime v3 assessment is not canonical")
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
        set(assessment) != expected
        or assessment["schema_version"] != SCHEMA_VERSION
        or assessment["domain"] != ASSESSMENT_DOMAIN
        or assessment["raw_receipt"] != {"path": RAW_RELATIVE, "sha256": sha256_bytes(raw)}
        or assessment["authorization"] != AUTHORIZATION_FALSE
    ):
        raise ValueError("Prime v3 assessment bindings differ")
    try:
        expected_semantic = _semantic(receipt)
        expected_diagnostic = None
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        expected_semantic = {
            "state": "observed_schema_unknown",
            "wallet": None,
            "inventory": None,
            "availability": None,
            "resource": None,
        }
        expected_diagnostic = {"code": "semantic_schema_unknown", "message": str(error)}
    if assessment != {
        "schema_version": SCHEMA_VERSION,
        "domain": ASSESSMENT_DOMAIN,
        "state": expected_semantic["state"],
        "raw_receipt": {"path": RAW_RELATIVE, "sha256": sha256_bytes(raw)},
        "semantic": {
            "wallet": expected_semantic["wallet"],
            "inventory": expected_semantic["inventory"],
            "availability": expected_semantic["availability"],
        },
        "resource": expected_semantic["resource"],
        "diagnostic": expected_diagnostic,
        "authorization": AUTHORIZATION_FALSE,
    }:
        raise ValueError("Prime v3 assessment differs from its authenticated raw receipt")
    return cast(dict[str, Any], assessment)


def _bound_file(root: Path, relative: str, expected: str) -> None:
    path = root / relative
    if path.is_symlink() or not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
        raise ValueError(f"historical Prime inventory binding differs: {relative}")


def _authenticate_precommit(root: Path) -> None:
    if _git(root, "rev-parse", "HEAD") != PARENT_COMMIT:
        raise ValueError("Prime inventory v3 build requires exact parent 267ee8d")
    if _git(root, "rev-parse", "HEAD^{tree}") != PARENT_TREE:
        raise ValueError("Prime inventory v3 parent tree differs")
    unexpected = _status_paths(root).difference(CHECKPOINT_PATHS)
    if unexpected:
        raise ValueError(
            "Prime inventory v3 worktree exceeds its exact allowlist: "
            + ", ".join(sorted(unexpected))
        )


def build_prime_inventory_v3_artifacts(root: Path) -> dict[str, bytes]:
    root = root.resolve()
    _authenticate_precommit(root)
    for relative, expected in HISTORICAL_BINDINGS.items():
        _bound_file(root, relative, expected)
    installed = authenticate_installed_prime_source()
    executable = authenticate_installed_prime_executable()
    owner_hash = sha256_bytes((root / OWNER_RELATIVE).read_bytes())
    builder_hash = sha256_bytes((root / BUILDER_RELATIVE).read_bytes())
    test_hash = sha256_bytes((root / TEST_RELATIVE).read_bytes())
    contract = cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": CONTRACT_DOMAIN,
                "state": "non_authorizing_cpu_inventory_contract",
                "parent": {"commit": PARENT_COMMIT, "tree": PARENT_TREE},
                "historical": {
                    relative: {"sha256": expected, "immutable": True}
                    for relative, expected in sorted(HISTORICAL_BINDINGS.items())
                },
                "installed_prime_owner": installed,
                "installed_prime_executable": executable,
                "paths": {"raw_receipt": RAW_RELATIVE, "assessment": ASSESSMENT_RELATIVE},
                "raw_contract": {
                    "commands": {
                        name: list(argv) for name, argv in v2.PRIME_READ_ONLY_COMMANDS.items()
                    },
                    "absolute_authenticated_execution": True,
                    "path_shadow_rejected_before_execution": True,
                    "ttl_seconds": TTL_SECONDS,
                    "atomic_no_overwrite": True,
                    "semantic_failure_preserves_raw": True,
                },
                "disk_contract": {
                    "source_grammar": "^(0|[1-9][0-9]{0,5})(\\+)?$",
                    "maximum_text_bytes": MAX_DISK_TEXT_BYTES,
                    "maximum_gb": MAX_DISK_GB,
                    "numeric_null_malformed": "schema_unknown_non_authorizing",
                    "positive_proven_string_required": True,
                },
                "readiness_guards": {
                    "wallet_usd_strictly_greater_than": 30,
                    "persistent_disk_count": 0,
                    "literal_non_spot_required": True,
                },
                "fixed_artifact_root": {
                    "path": FIXED_LOCAL_ARTIFACT_ROOT,
                    "absent_is_fail_closed": True,
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
    audit = cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": AUDIT_DOMAIN,
                "state": "non_authorizing_cpu_diagnosis",
                "parent": {"commit": PARENT_COMMIT, "tree": PARENT_TREE},
                "allowlist": sorted(CHECKPOINT_PATHS),
                "file_bindings": dict(sorted(bindings.items())),
                "installed_prime_executable": executable,
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
                "authorization": AUTHORIZATION_FALSE,
            }
        ),
    )
    return {CONTRACT_RELATIVE: contract, AUDIT_RELATIVE: audit}


def verify_prime_inventory_v3_artifacts(root: Path, output_root: Path) -> dict[str, str]:
    expected = build_prime_inventory_v3_artifacts(root)
    hashes: dict[str, str] = {}
    for relative, raw in expected.items():
        path = output_root / relative
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise ValueError(f"Prime inventory v3 artifact differs: {relative}")
        hashes[relative] = sha256_bytes(raw)
    return hashes


__all__ = [
    "ASSESSMENT_RELATIVE",
    "AUDIT_RELATIVE",
    "CONTRACT_RELATIVE",
    "PARENT_COMMIT",
    "PARENT_TREE",
    "RAW_RELATIVE",
    "assess_prime_inventory_v3",
    "authenticate_installed_prime_executable",
    "authenticate_installed_prime_source",
    "build_prime_inventory_v3_artifacts",
    "capture_prime_inventory_raw_v3",
    "validate_prime_inventory_assessment_v3",
    "validate_prime_inventory_raw_v3",
    "verify_prime_inventory_v3_artifacts",
]
