"""Raw launch-time observation owners for the support bundle.

The producer in this module is intentionally the only authority for Prime
preflight facts.  It records the bytes returned by the read-only CLI commands
and derives the selected resource, account, inventory, and billing facts from
those bytes.  A caller cannot pass an already-approved resource dictionary to
the producer.  The synthetic builder is explicitly non-authorizing and exists
only for source-free tests.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes

OBSERVATION_DOMAIN = "redco-stage-d1-support-v13-launch-prime-observation-v1"
OBSERVATION_SCHEMA_VERSION = 1
SYNTHETIC_OBSERVATION_DOMAIN = "redco-stage-d1-support-v13-launch-prime-observation-synthetic-v1"
POD_OBSERVATION_DOMAIN = "redco-stage-d1-support-v13-launch-pod-observation-v1"
POD_OBSERVATION_SCHEMA_VERSION = 1
OBSERVATION_TTL_SECONDS = 900
MAX_CAPTURE_BYTES = 256 * 1024
REQUIRED_RESOURCE_TYPE = "2x48GB L40/L40S/RTX 6000 Ada"
PRIME_CLI_VERSION = "0.6.20"
PRIME_VERSION_BANNER = f"Prime CLI version: {PRIME_CLI_VERSION}"
PRIME_WALLET_LIMIT = 100

PRIME_READ_ONLY_COMMANDS: dict[str, tuple[str, ...]] = {
    "version": ("prime", "--version"),
    "wallet": (
        "prime",
        "--plain",
        "wallet",
        "--limit",
        str(PRIME_WALLET_LIMIT),
        "--output",
        "json",
    ),
    "pods": ("prime", "--plain", "pods", "list", "--output", "json"),
    "disks": ("prime", "--plain", "disks", "list", "--output", "json"),
    "availability": (
        "prime",
        "--plain",
        "availability",
        "list",
        "--gpu-count",
        "2",
        "--output",
        "json",
    ),
}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        return _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error


def _required_string(value: Mapping[str, Any], key: str, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{label}.{key} is missing")
    return result


def _inventory_rows(
    value: Mapping[str, Any],
    *,
    key: str,
    label: str,
    item_keys: set[str],
) -> list[Any]:
    if set(value) != {key, "total_count", "offset", "limit"}:
        raise ValueError(f"{label} schema differs from 0.6.20")
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
    for row in rows:
        item = _object(row, label)
        if set(item) != item_keys:
            raise ValueError(f"{label} item schema differs from 0.6.20")
    return rows


def _resource_from_availability(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {"gpu_resources", "total_count", "filters"}:
        raise ValueError("Prime availability schema differs from 0.6.20")
    raw_resources = value.get("gpu_resources")
    resources = raw_resources if isinstance(raw_resources, list) else None
    if (
        resources is None
        or type(value["total_count"]) is not int
        or not isinstance(value["filters"], dict)
    ):
        raise ValueError("Prime availability resources are malformed")
    candidates: list[dict[str, Any]] = []
    for item in resources:
        resource = _object(item, "availability resource")
        if resource.get("provider") != "prime":
            continue
        if resource.get("stock_status") not in {"available", "ready", "in_stock"}:
            continue
        if type(resource.get("gpu_count")) is not int or resource["gpu_count"] != 2:
            continue
        memory = resource.get("gpu_memory")
        if not (
            (type(memory) in {int, float} and memory == 48)
            or (isinstance(memory, str) and memory.rstrip("GB").strip() == "48")
        ):
            continue
        if resource.get("is_spot") is not False:
            continue
        if type(resource.get("price_per_hour")) not in {int, float}:
            continue
        if resource["price_per_hour"] > 2:
            continue
        if not isinstance(resource.get("id"), str) or not resource["id"]:
            continue
        candidates.append(resource)
    if not candidates:
        raise ValueError("Prime availability has no eligible non-spot resource")
    selected_source = min(
        candidates,
        key=lambda item: _required_string(item, "id", "resource"),
    )
    required = {
        "id",
        "cloud_id",
        "gpu_type",
        "provider",
        "location",
        "gpu_count",
        "socket",
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
    if not required.issubset(selected_source):
        raise ValueError("selected Prime resource fields differ")
    selected = {key: selected_source[key] for key in required}
    if not isinstance(selected["security"], (dict, str)) or not selected["security"]:
        raise ValueError("selected Prime resource security is missing")
    if (
        type(selected["disk_gb"]) not in {int, float}
        or selected["disk_gb"] < 0
    ):
        raise ValueError("selected Prime resource lacks ephemeral storage")
    selected["resource_id"] = selected["id"]
    selected["resource_type"] = REQUIRED_RESOURCE_TYPE
    selected["hourly_rate_usd"] = selected["price_per_hour"]
    selected["spot"] = selected["is_spot"]
    selected["gpu_memory_gb"] = 48
    selected["persistent_storage"] = False
    selected["ephemeral_storage_gb"] = selected["disk_gb"]
    return selected


def _parsed_facts(captures: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    wallet = _json_object(base64.b64decode(captures["wallet"]["stdout_b64"]), "Prime wallet")
    pods_raw = _json_object(base64.b64decode(captures["pods"]["stdout_b64"]), "Prime pods")
    disks_raw = _json_object(base64.b64decode(captures["disks"]["stdout_b64"]), "Prime disks")
    availability = _json_object(
        base64.b64decode(captures["availability"]["stdout_b64"]),
        "Prime availability",
    )
    if set(wallet) - {
        "wallet_id",
        "team_id",
        "balance_usd",
        "currency",
        "total_billings",
        "recent_billings",
    } or "wallet_id" not in wallet or "balance_usd" not in wallet:
        raise ValueError("Prime wallet schema differs from 0.6.20")
    wallet_balance = wallet.get("balance_usd")
    if type(wallet_balance) not in {int, float}:
        raise ValueError("Prime wallet balance is missing")
    wallet_identity = _required_string(wallet, "wallet_id", "wallet")
    currency = _required_string(wallet, "currency", "wallet")
    if type(wallet.get("total_billings")) is not int or wallet["total_billings"] < 0:
        raise ValueError("Prime wallet billing count is missing")
    recent_billings = wallet.get("recent_billings")
    if not isinstance(recent_billings, list):
        raise ValueError("Prime wallet billing rows are missing")
    for row in recent_billings:
        billing = _object(row, "Prime billing row")
        required_billing = {
            "id",
            "created_at",
            "updated_at",
            "amount_usd",
            "currency",
            "resource_type",
        }
        if (
            not required_billing.issubset(billing)
            or type(billing["amount_usd"]) not in {int, float}
        ):
            raise ValueError("Prime billing row schema differs")
        if billing["currency"] != currency:
            raise ValueError("Prime billing currency differs from wallet")
    pods = _inventory_rows(
        pods_raw,
        key="pods",
        label="Prime pods",
        item_keys={"id", "name", "gpu", "status", "created_at"},
    )
    disks = _inventory_rows(
        disks_raw,
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
    if pods or disks:
        raise ValueError("Prime has active pods or persistent disks")
    resource = _resource_from_availability(availability)
    billing_cursor = sha256_bytes(
        canonical_json_bytes(
            {
                "total_billings": wallet["total_billings"],
                "recent_billings": recent_billings,
            }
        )
    )
    return {
        "account_id": wallet_identity,
        "wallet_id": wallet_identity,
        "team_id": wallet.get("team_id"),
        "currency": currency,
        "total_billings": wallet["total_billings"],
        "recent_billings": recent_billings,
        "billing_cursor": billing_cursor,
        "wallet_usd": wallet_balance,
        "pods": pods,
        "disks": disks,
        "resource": resource,
    }


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8").strip()


@dataclass(frozen=True, slots=True)
class PrimeObservationProducer:
    """Capture raw read-only Prime observations; no approved facts are inputs."""

    root: Path

    def _run_command(self, argv: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            list(argv),
            cwd=self.root,
            check=False,
            capture_output=True,
        )

    def capture(self, *, captured_at_epoch: int | None = None) -> bytes:
        root = self.root.resolve()
        captures: dict[str, dict[str, Any]] = {}
        for name, argv in PRIME_READ_ONLY_COMMANDS.items():
            result = self._run_command(argv)
            stdout = bytes(result.stdout)
            stderr = bytes(result.stderr)
            if len(stdout) > MAX_CAPTURE_BYTES or len(stderr) > MAX_CAPTURE_BYTES:
                raise ValueError(f"Prime {name} output exceeds the bounded capture")
            if result.returncode != 0:
                raise RuntimeError(f"Prime read-only command failed: {name}")
            captures[name] = {
                "argv": list(argv),
                "returncode": result.returncode,
                "stdout_b64": base64.b64encode(stdout).decode("ascii"),
                "stderr_b64": base64.b64encode(stderr).decode("ascii"),
                "stdout_sha256": sha256_bytes(stdout),
                "stderr_sha256": sha256_bytes(stderr),
            }
        facts = _parsed_facts(captures)
        version_raw = base64.b64decode(captures["version"]["stdout_b64"])
        version = version_raw.decode("utf-8").strip()
        if version != PRIME_VERSION_BANNER:
            raise ValueError("Prime CLI version is not the pinned 0.6.20 contract")
        captured = int(time.time()) if captured_at_epoch is None else captured_at_epoch
        if type(captured) is not int:
            raise TypeError("capture time must be an integer epoch")
        payload = {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "domain": OBSERVATION_DOMAIN,
            "state": "observed",
            "captured_at_epoch": captured,
            "expires_at_epoch": captured + OBSERVATION_TTL_SECONDS,
            "bundle": {
                "commit": _git_value(root, "rev-parse", "HEAD"),
                "tree": _git_value(root, "rev-parse", "HEAD^{tree}"),
            },
            "cli": {
                "version": version,
                "account_id": facts["account_id"],
                "wallet_id": facts["wallet_id"],
                "team_id": facts["team_id"],
            },
            "wallet": {
                "account_id": facts["account_id"],
                "wallet_id": facts["wallet_id"],
                "team_id": facts["team_id"],
                "currency": facts["currency"],
                "total_billings": facts["total_billings"],
                "recent_billings": facts["recent_billings"],
                "billing_cursor": facts["billing_cursor"],
                "wallet_usd": facts["wallet_usd"],
            },
            "inventory": {"pods": facts["pods"], "disks": facts["disks"]},
            "resource": facts["resource"],
            "commands": captures,
        }
        return cast(bytes, canonical_json_bytes(payload))

    def capture_to(self, path: Path, *, captured_at_epoch: int | None = None) -> bytes:
        value = self.capture(captured_at_epoch=captured_at_epoch)
        root = self.root.resolve()
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError("Prime observation output escapes the repository") from error
        ancestor = path.parent
        while ancestor != ancestor.parent:
            if ancestor.is_symlink():
                raise ValueError("Prime observation output parent is a symlink")
            if ancestor.exists():
                break
            ancestor = ancestor.parent
        path = resolved
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        with path.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_parent(path)
        return value


def _validate_capture(name: str, value: object) -> dict[str, Any]:
    capture = _object(value, f"Prime command {name}")
    expected = {
        "argv",
        "returncode",
        "stdout_b64",
        "stderr_b64",
        "stdout_sha256",
        "stderr_sha256",
    }
    if set(capture) != expected:
        raise ValueError(f"Prime command {name} fields differ")
    argv = capture["argv"]
    if argv != list(PRIME_READ_ONLY_COMMANDS[name]) or type(capture["returncode"]) is not int:
        raise ValueError(f"Prime command {name} is not the frozen read-only command")
    for stream in ("stdout_b64", "stderr_b64"):
        if not isinstance(capture[stream], str):
            raise ValueError(f"Prime command {name} stream is invalid")
        try:
            raw = base64.b64decode(capture[stream], validate=True)
        except ValueError as error:
            raise ValueError(f"Prime command {name} stream is not base64") from error
        if len(raw) > MAX_CAPTURE_BYTES:
            raise ValueError(f"Prime command {name} stream is too large")
        hash_key = stream.replace("_b64", "_sha256")
        if capture[hash_key] != sha256_bytes(raw):
            raise ValueError(f"Prime command {name} stream hash differs")
    if capture["returncode"] != 0:
        raise ValueError(f"Prime command {name} did not succeed")
    return capture


def validate_prime_observation(
    root: Path,
    path: Path,
    *,
    now_epoch: int | None = None,
) -> dict[str, Any]:
    """Validate raw Prime evidence against the current authenticated checkout."""

    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Prime observation is not JSON") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise ValueError("Prime observation is not canonical JSON")
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
        "resource",
        "commands",
    }
    if set(payload) != expected_keys or payload["schema_version"] != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("Prime observation fields differ")
    if payload["domain"] != OBSERVATION_DOMAIN or payload["state"] != "observed":
        raise ValueError("Prime observation is not an authoritative launch observation")
    captured = payload["captured_at_epoch"]
    expires = payload["expires_at_epoch"]
    if (
        type(captured) is not int
        or type(expires) is not int
        or expires != captured + OBSERVATION_TTL_SECONDS
    ):
        raise ValueError("Prime observation TTL differs")
    now = int(time.time()) if now_epoch is None else now_epoch
    if captured > now or expires < now:
        raise ValueError("Prime observation is future-dated or stale")
    bundle = _object(payload["bundle"], "Prime observation bundle")
    commit = _git_value(root, "rev-parse", "HEAD")
    tree = _git_value(root, "rev-parse", "HEAD^{tree}")
    if bundle != {"commit": commit, "tree": tree}:
        raise ValueError("Prime observation bundle binding differs")
    commands = _object(payload["commands"], "Prime observation commands")
    if set(commands) != set(PRIME_READ_ONLY_COMMANDS):
        raise ValueError("Prime observation command set differs")
    captures = {name: _validate_capture(name, commands[name]) for name in PRIME_READ_ONLY_COMMANDS}
    facts = _parsed_facts(captures)
    version = base64.b64decode(captures["version"]["stdout_b64"]).decode("utf-8").strip()
    if version != PRIME_VERSION_BANNER:
        raise ValueError("Prime CLI version is not the pinned 0.6.20 contract")
    cli = _object(payload["cli"], "Prime observation CLI")
    wallet = _object(payload["wallet"], "Prime observation wallet")
    inventory = _object(payload["inventory"], "Prime observation inventory")
    if cli != {
        "version": base64.b64decode(captures["version"]["stdout_b64"]).decode("utf-8").strip(),
        "account_id": facts["account_id"],
        "wallet_id": facts["wallet_id"],
        "team_id": facts["team_id"],
    }:
        raise ValueError("Prime CLI identity differs from raw output")
    if wallet != {
        "account_id": facts["account_id"],
        "wallet_id": facts["wallet_id"],
        "team_id": facts["team_id"],
        "currency": facts["currency"],
        "total_billings": facts["total_billings"],
        "recent_billings": facts["recent_billings"],
        "billing_cursor": facts["billing_cursor"],
        "wallet_usd": facts["wallet_usd"],
    }:
        raise ValueError("Prime billing witness differs from raw output")
    if inventory != {"pods": facts["pods"], "disks": facts["disks"]}:
        raise ValueError("Prime inventory differs from raw output")
    resource = _object(payload["resource"], "Prime observation resource")
    if resource != facts["resource"]:
        raise ValueError("Prime selected resource differs from raw output")
    if facts["wallet_usd"] < 30:
        raise ValueError("Prime wallet does not cover frozen reserves")
    return cast(dict[str, Any], payload)


def validate_current_prime_observation(root: Path, path: Path) -> dict[str, Any]:
    """Require the persisted receipt to match a fresh read-only Prime probe."""

    persisted = validate_prime_observation(root, path)
    fresh = PrimeObservationProducer(root).capture()
    try:
        fresh_payload = json.loads(fresh)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("fresh Prime observation is not canonical JSON") from error
    if not isinstance(fresh_payload, dict):
        raise ValueError("fresh Prime observation is not an object")
    for field in ("cli", "wallet", "inventory", "resource"):
        if persisted[field] != fresh_payload.get(field):
            raise ValueError(f"Prime observation changed since capture: {field}")
    persisted_commands = _object(persisted["commands"], "persisted Prime commands")
    fresh_commands = _object(fresh_payload.get("commands"), "fresh Prime commands")
    for name in PRIME_READ_ONLY_COMMANDS:
        p = _object(persisted_commands[name], "persisted Prime command")
        f = _object(fresh_commands[name], "fresh Prime command")
        if (p.get("argv"), p.get("stdout_sha256"), p.get("stderr_sha256")) != (
            f.get("argv"),
            f.get("stdout_sha256"),
            f.get("stderr_sha256"),
        ):
            raise ValueError(f"Prime command changed since capture: {name}")
    return persisted


def build_synthetic_observation(
    root: Path,
    *,
    captured_at_epoch: int,
    expires_at_epoch: int,
    location: str,
) -> bytes:
    """Build an explicitly non-authorizing source-free fixture."""

    bundle = {
        "commit": "synthetic-precommit",
        "tree": "synthetic-tree",
    }
    payload = {
        "schema_version": 1,
        "domain": SYNTHETIC_OBSERVATION_DOMAIN,
        "state": "synthetic_non_authorizing",
        "captured_at_epoch": captured_at_epoch,
        "expires_at_epoch": expires_at_epoch,
        "bundle": bundle,
        "resource": {
            "provider": "prime",
            "resource_id": "synthetic-resource",
            "location": location,
            "resource_type": REQUIRED_RESOURCE_TYPE,
            "gpu_count": 2,
            "gpu_memory_gb": 48,
            "spot": False,
            "security": {"persistent_storage": False},
            "hourly_rate_usd": 2,
            "persistent_storage": False,
            "ephemeral_storage_gb": 1,
        },
        "wallet": {
            "account_id": "synthetic-account",
            "context": "synthetic-context",
            "billing_cursor": "synthetic-cursor",
            "wallet_usd": 30,
        },
        "inventory": {"pods": [], "disks": []},
        "runtime": {"python": "3.12.3", "pyarrow": "25.0.0", "datasets": "5.0.0"},
        "assets": {},
        "vllm": {"health": "ready", "model_list": ["/workspace/models/stage-d1-merged"]},
    }
    del root
    return cast(bytes, canonical_json_bytes(payload))


def _read_response(response: Any) -> tuple[int, bytes]:
    status_value = getattr(response, "status", None)
    if status_value is None:
        status_value = response.getcode()
    status = int(status_value)
    body = bytes(response.read())
    return status, body


def capture_pod_runtime_observation(
    *,
    base_url: str,
    asset_paths: Mapping[str, tuple[Path, str]],
    opener: Callable[[urllib.request.Request], Any] | None = None,
    expected_model_ids: tuple[str, ...] = ("/workspace/models/stage-d1-merged",),
    runtime_probe: bytes | None = None,
) -> bytes:
    """Hash local assets and observe only vLLM health/model endpoints."""

    probe = runtime_probe or b""
    if len(probe) > MAX_CAPTURE_BYTES:
        raise ValueError("pod runtime version probe exceeds the bounded capture")
    open_url = opener or urllib.request.urlopen
    endpoints = {"health": "/health", "models": "/v1/models"}
    http: dict[str, dict[str, Any]] = {}
    for name, suffix in endpoints.items():
        request = urllib.request.Request(base_url.rstrip("/") + suffix, method="GET")
        response = open_url(request)
        status, body = _read_response(response)
        if status != 200 or len(body) > MAX_CAPTURE_BYTES:
            raise ValueError(f"vLLM {name} observation failed")
        http[name] = {
            "path": suffix,
            "status": status,
            "body_b64": base64.b64encode(body).decode("ascii"),
            "body_sha256": sha256_bytes(body),
        }
    health = _json_object(base64.b64decode(http["health"]["body_b64"]), "vLLM health")
    models = _json_object(base64.b64decode(http["models"]["body_b64"]), "vLLM models")
    if health.get("status") not in {"ok", "ready", "healthy"}:
        raise ValueError("vLLM health is not ready")
    model_data = models.get("data")
    if not isinstance(model_data, list) or not model_data:
        raise ValueError("vLLM model list is empty")
    model_ids = tuple(
        item.get("id")
        for item in model_data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    )
    if model_ids != expected_model_ids:
        raise ValueError("vLLM model list differs from the frozen model identity")
    assets: dict[str, dict[str, Any]] = {}
    for name, (path, expected) in sorted(asset_paths.items()):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"runtime asset is missing: {name}")
        value = path.read_bytes()
        digest = sha256_bytes(value)
        if digest != expected:
            raise ValueError(f"runtime asset differs: {name}")
        assets[name] = {"path": name, "sha256": digest, "bytes": len(value)}
    return cast(
        bytes,
        canonical_json_bytes(
        {
            "schema_version": POD_OBSERVATION_SCHEMA_VERSION,
            "domain": POD_OBSERVATION_DOMAIN,
            "state": "observed",
            "http": http,
            "assets": assets,
            "runtime": {
                "probe_b64": base64.b64encode(probe).decode("ascii"),
                "probe_sha256": sha256_bytes(probe),
            },
            "completion_requests": 0,
        }
        ),
    )


def validate_pod_runtime_observation(
    value: bytes,
    *,
    expected_asset_hashes: Mapping[str, str] | None = None,
    expected_model_ids: tuple[str, ...] = ("/workspace/models/stage-d1-merged",),
    expected_runtime: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("pod runtime observation is not JSON") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != value:
        raise ValueError("pod runtime observation is not canonical")
    if set(payload) != {
        "schema_version",
        "domain",
        "state",
        "http",
        "assets",
        "runtime",
        "completion_requests",
    }:
        raise ValueError("pod runtime observation fields differ")
    if (
        payload["schema_version"] != POD_OBSERVATION_SCHEMA_VERSION
        or payload["domain"] != POD_OBSERVATION_DOMAIN
        or payload["state"] != "observed"
        or payload["completion_requests"] != 0
    ):
        raise ValueError("pod runtime observation is not a no-generation witness")
    assets = _object(payload["assets"], "pod runtime assets")
    if not assets:
        raise ValueError("pod runtime observation has no authenticated assets")
    for name, item in assets.items():
        asset = _object(item, "pod runtime asset")
        if set(asset) != {"path", "sha256", "bytes"}:
            raise ValueError("pod runtime asset fields differ")
        if asset["path"] != name or not isinstance(asset["sha256"], str):
            raise ValueError("pod runtime asset identity differs")
        if len(asset["sha256"]) != 64 or any(
            character not in "0123456789abcdef" for character in asset["sha256"]
        ):
            raise ValueError("pod runtime asset hash is not SHA-256")
        if type(asset["bytes"]) is not int or asset["bytes"] < 0:
            raise ValueError("pod runtime asset byte length is invalid")
    if expected_asset_hashes is not None:
        if set(assets) != set(expected_asset_hashes):
            raise ValueError("pod runtime asset set differs from the frozen bundle")
        if any(
            _object(assets[name], "pod runtime asset")["sha256"] != expected
            for name, expected in expected_asset_hashes.items()
        ):
            raise ValueError("pod runtime asset hash differs from the frozen bundle")
    runtime = _object(payload["runtime"], "pod runtime versions")
    if set(runtime) != {"probe_b64", "probe_sha256"}:
        raise ValueError("pod runtime version witness fields differ")
    try:
        runtime_probe = base64.b64decode(cast(str, runtime["probe_b64"]), validate=True)
    except ValueError as error:
        raise ValueError("pod runtime version witness is not base64") from error
    if runtime["probe_sha256"] != sha256_bytes(runtime_probe):
        raise ValueError("pod runtime version witness hash differs")
    if expected_runtime is not None:
        if not runtime_probe:
            raise ValueError("pod runtime version witness is missing")
        parsed_runtime = _json_object(runtime_probe, "pod runtime versions")
        if parsed_runtime != dict(expected_runtime):
            raise ValueError("pod runtime versions differ from the frozen stack")
    for name, path in (("health", "/health"), ("models", "/v1/models")):
        item = _object(payload["http"], "pod HTTP").get(name)
        if not isinstance(item, dict) or set(item) != {"path", "status", "body_b64", "body_sha256"}:
            raise ValueError("pod HTTP observation fields differ")
        if item["path"] != path or item["status"] != 200:
            raise ValueError("pod HTTP status differs")
        body = base64.b64decode(cast(str, item["body_b64"]))
        if item["body_sha256"] != sha256_bytes(body):
            raise ValueError("pod HTTP body hash differs")
    health = _json_object(
        base64.b64decode(cast(str, _object(payload["http"], "pod HTTP")["health"]["body_b64"])),
        "vLLM health",
    )
    models = _json_object(
        base64.b64decode(cast(str, _object(payload["http"], "pod HTTP")["models"]["body_b64"])),
        "vLLM models",
    )
    if health.get("status") not in {"ok", "ready", "healthy"}:
        raise ValueError("vLLM health is not ready")
    model_data = models.get("data")
    model_ids = tuple(
        item.get("id")
        for item in model_data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ) if isinstance(model_data, list) else ()
    if model_ids != expected_model_ids:
        raise ValueError("vLLM model list differs from the frozen model identity")
    return cast(dict[str, Any], payload)


__all__ = [
    "MAX_CAPTURE_BYTES",
    "OBSERVATION_DOMAIN",
    "OBSERVATION_SCHEMA_VERSION",
    "OBSERVATION_TTL_SECONDS",
    "POD_OBSERVATION_DOMAIN",
    "PRIME_CLI_VERSION",
    "PRIME_READ_ONLY_COMMANDS",
    "PRIME_VERSION_BANNER",
    "PRIME_WALLET_LIMIT",
    "PrimeObservationProducer",
    "build_synthetic_observation",
    "capture_pod_runtime_observation",
    "validate_current_prime_observation",
    "validate_pod_runtime_observation",
    "validate_prime_observation",
]
