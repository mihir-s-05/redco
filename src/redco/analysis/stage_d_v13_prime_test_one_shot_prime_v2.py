"""Prime transport, resource, finance, and cleanup owners for the test one-shot."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol, cast

from redco.analysis import stage_d_v13_prime_inventory_v5 as v5
from redco.analysis.stage_d_v13_prime_test_one_shot_contract_v2 import (
    ASSESSMENT_DOMAIN,
    ASSESSMENT_TTL_SECONDS,
    CLEANUP_BILLING_TIMEOUT_SECONDS,
    CLEANUP_DISK_TIMEOUT_SECONDS,
    CLEANUP_POD_TIMEOUT_SECONDS,
    CLEANUP_TIMEOUT_SECONDS,
    COMMAND_TIMEOUT_SECONDS,
    MAX_BILLING_POLLS,
    MAX_CLEANUP_PRIME_CLI_CALLS,
    MAX_COMMAND_OUTPUT_BYTES,
    MAX_OPERATIONAL_PRIME_CLI_CALLS,
    MAX_OWNED_POD_IDENTITIES,
    MAX_PRIME_CLI_CALLS,
    MAX_TERMINATION_POLLS,
    MAXIMUM_POD_SECONDS,
    OPENSSH_EXECUTABLES,
    POD_NAME_PREFIX,
    PODS_API_OWNER,
    PODS_API_OWNER_SHA256,
    PODS_COMMAND_OWNER,
    PODS_COMMAND_OWNER_SHA256,
    PODS_CREATE_ENDPOINT,
    POLL_INTERVAL_SECONDS,
    PRIME_CLIENT_OWNER,
    PRIME_CLIENT_OWNER_SHA256,
    PRIME_CONFIG_OWNER,
    PRIME_CONFIG_OWNER_SHA256,
    READINESS_AUTHORITY,
    ROOT,
    RUNTIME_AUTHORITY,
    V5_CONTRACT_PATH,
    V5_CONTRACT_SHA256,
    V5_OWNER_PATH,
    V5_OWNER_SHA256,
    WALLET_API_ENDPOINT,
    WALLET_API_OWNER,
    WALLET_API_OWNER_SHA256,
    CommandJournalSummary,
    CommandResult,
    CreateDispatchSummary,
    CreateResultSummary,
    SigningIdentity,
    authenticate_authorization,
    authority_value,
    canonical_json,
    publish_once,
    sha256_bytes,
    strict_object,
)
from redco.analysis.stage_d_v13_prime_test_one_shot_remote_v2 import (
    LINUX_UV_BYTES,
    LINUX_UV_SHA256,
    validate_command_journal_details,
)
from redco.analysis.stage_d_v13_prime_test_one_shot_wallet_v2 import (
    MAX_RESPONSE_BYTES,
    WalletSnapshot,
    capture_wallet_snapshot,
    reconcile_wallet_snapshots,
    validate_wallet_journal_details,
)

POD_IMAGE = "ubuntu_22_cuda_12"
MAX_OUTPUT_BYTES = MAX_COMMAND_OUTPUT_BYTES


class APIClient(Protocol):
    base_url: str
    api_key: str
    client: Any
    config: Any


@dataclass(slots=True)
class RuntimeContext:
    repository: Path
    authorization: Mapping[str, str]
    client: APIClient
    wallet_team_id: str | None
    transport_errors: tuple[type[BaseException], ...]
    prime_executable: Path
    openssh: Mapping[str, Path]
    keygen_executable: Path
    signing_key: Path
    identity: SigningIdentity
    linux_uv: Path
    run: Callable[[Sequence[str], bytes | None, float], CommandResult]
    now: Callable[[], int]
    monotonic: Callable[[], float]
    sleep: Callable[[float], None]


def sha_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(raw: bytes, label: str) -> dict[str, Any]:
    if len(raw) > MAX_OUTPUT_BYTES:
        raise ValueError(f"{label} exceeds its byte bound")
    try:
        value = json.loads(raw, parse_float=Decimal, parse_int=int)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return cast(dict[str, Any], value)


def inventory(raw: bytes, key: str) -> list[dict[str, Any]]:
    value = load_json(raw, f"Prime {key}")
    if set(value) != {key, "total_count", "offset", "limit"}:
        raise ValueError(f"Prime {key} schema differs")
    rows = value[key]
    if (
        not isinstance(rows, list)
        or type(value["total_count"]) is not int
        or value["total_count"] != len(rows)
    ):
        raise ValueError(f"Prime {key} rows differ")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Prime {key} row differs")
    return cast(list[dict[str, Any]], rows)


def assess_pages(
    pages: list[dict[str, object]], checkout: Mapping[str, str], captured: int
) -> tuple[bytes, dict[str, Any] | None]:
    replay = v5._replay_transcript(pages, None, None, len(pages))
    rows: list[dict[str, object]] = []
    for record in pages:
        body = base64.b64decode(cast(str, record["decoded_application_body_b64"]), validate=True)
        _total, items = v5._pagination(body)
        rows.extend(
            v5._assess_item(
                item,
                {
                    "endpoint": record["endpoint"],
                    "page": record["page_ordinal"],
                    "item_ordinal": ordinal,
                },
            )
            for ordinal, item in enumerate(items)
        )
    identities = [row["cloud_id"] for row in rows if isinstance(row["cloud_id"], str)]
    duplicate = len(identities) != len(set(identities))
    eligible = [row for row in rows if row["eligible"] is True]
    ordered = sorted(
        eligible,
        key=lambda row: (
            cast(float, row["hourly_rate_usd"]),
            cast(dict[str, Any], row["raw_item"])["gpuType"],
            sha256_bytes(canonical_json(row["raw_item"])),
        ),
    )
    selected = None if duplicate or not ordered else cast(dict[str, Any], ordered[0]["raw_item"])
    state, reason = (
        ("ambiguous_capacity", "duplicate_resource_identity")
        if duplicate
        else ("no_qualifying_capacity", "no_eligible_resource")
        if selected is None
        else ("qualifying_capacity", "deterministic_first_eligible")
    )
    facts = (
        None
        if selected is None
        else {
            "gpu_type": selected["gpuType"],
            "gpu_count": selected["gpuCount"],
            "gpu_memory_gb": selected["gpuMemory"],
            "is_spot": selected["isSpot"],
            "hourly_rate_usd": ordered[0]["hourly_rate_usd"],
            "disk_size": 0,
        }
    )
    return canonical_json(
        {
            "schema_version": 2,
            "domain": ASSESSMENT_DOMAIN,
            "state": state,
            "reason": reason,
            "captured_at_epoch": captured,
            "expires_at_epoch": captured + ASSESSMENT_TTL_SECONDS,
            "checkout": dict(checkout),
            "transcript_payload_sha256": replay["payload_sha256"],
            "row_count": len(rows),
            "eligible_count": len(eligible),
            "duplicate_identity": duplicate,
            "selection_order": "hourly_rate,gpu_label,canonical_resource_sha256",
            "selected_resource_sha256": None
            if selected is None
            else sha256_bytes(canonical_json(selected)),
            "selected_facts": facts,
            "attempt_consumed": True,
            "retry": False,
            "authority": RUNTIME_AUTHORITY,
        }
    ), selected


def create_payload(
    resource: Mapping[str, Any], pod_name: str, team_id: object
) -> dict[str, object]:
    required = {"cloudId", "gpuType", "socket", "gpuCount", "provider"}
    if not required.issubset(resource) or resource["gpuCount"] != 2:
        raise ValueError("Prime selected resource lacks create fields")
    payload: dict[str, object] = {
        "pod": {
            "name": pod_name,
            "cloudId": resource["cloudId"],
            "gpuType": resource["gpuType"],
            "socket": resource["socket"],
            "gpuCount": 2,
            "diskSize": 0,
            "vcpus": None,
            "memory": None,
            "image": POD_IMAGE,
            "dataCenterId": resource.get("dataCenter"),
            "maxPrice": None,
            "country": None,
            "security": None,
            "jupyterPassword": None,
            "autoRestart": False,
            "customTemplateId": None,
            "envVars": None,
        },
        "provider": {"type": resource["provider"]},
        "disks": [],
        "team": {"teamId": team_id} if isinstance(team_id, str) else None,
    }
    return payload


def _digest(value: object, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Prime one-shot {label} differs")
    return value


def validate_create_dispatch(raw: bytes) -> CreateDispatchSummary:
    value = strict_object(
        raw,
        {
            "schema_version", "state", "endpoint", "payload_sha256", "resource_sha256",
            "disk_size", "attempt_limit", "retry", "authority",
        },
        "Prime one-shot create dispatch",
    )
    if (
        value["schema_version"] != 2
        or value["state"] != "create_dispatched_attempt_consumed"
        or value["endpoint"] != PODS_CREATE_ENDPOINT
        or type(value["disk_size"]) is not int
        or value["disk_size"] != 0
        or type(value["attempt_limit"]) is not int
        or value["attempt_limit"] != 1
        or value["retry"] is not False
    ):
        raise ValueError("Prime one-shot create dispatch binding differs")
    authority_value(value["authority"], READINESS_AUTHORITY, "create dispatch")
    return CreateDispatchSummary(
        _digest(value["resource_sha256"], "create resource digest"),
        _digest(value["payload_sha256"], "create payload digest"),
    )


def create_dispatch_bytes(resource: Mapping[str, Any], payload: object) -> bytes:
    raw = canonical_json(
        {
            "schema_version": 2,
            "state": "create_dispatched_attempt_consumed",
            "endpoint": PODS_CREATE_ENDPOINT,
            "payload_sha256": sha256_bytes(canonical_json(payload)),
            "resource_sha256": sha256_bytes(canonical_json(resource)),
            "disk_size": 0,
            "attempt_limit": 1,
            "retry": False,
            "authority": READINESS_AUTHORITY,
        }
    )
    validate_create_dispatch(raw)
    return raw


def validate_create_result(
    raw: bytes, authorization: Mapping[str, str]
) -> CreateResultSummary:
    value = strict_object(
        raw,
        {
            "schema_version", "state", "status_code", "response_bytes", "response_sha256",
            "pod_identity_sha256", "pod_name", "authority",
        },
        "Prime one-shot create result",
    )
    response_bytes = value["response_bytes"]
    if (
        value["schema_version"] != 2
        or value["state"] != "create_response_authenticated"
        or type(value["status_code"]) is not int
        or value["status_code"] not in {200, 201}
        or type(response_bytes) is not int
        or not 0 <= response_bytes <= MAX_RESPONSE_BYTES
        or value["pod_name"] != f"{POD_NAME_PREFIX}-{authorization['commit'][:12]}"
    ):
        raise ValueError("Prime one-shot create result binding differs")
    authority_value(value["authority"], READINESS_AUTHORITY, "create result")
    return CreateResultSummary(
        value["status_code"],
        _digest(value["response_sha256"], "create response digest"),
        response_bytes,
        _digest(value["pod_identity_sha256"], "create pod identity"),
        cast(str, value["pod_name"]),
    )


def create_result_bytes(
    status_code: int, response: bytes, pod_identity: str, authorization: Mapping[str, str]
) -> bytes:
    raw = canonical_json(
        {
            "schema_version": 2,
            "state": "create_response_authenticated",
            "status_code": status_code,
            "response_bytes": len(response),
            "response_sha256": sha256_bytes(response),
            "pod_identity_sha256": sha256_bytes(pod_identity.encode()),
            "pod_name": f"{POD_NAME_PREFIX}-{authorization['commit'][:12]}",
            "authority": READINESS_AUTHORITY,
        }
    )
    validate_create_result(raw, authorization)
    return raw


def command_binding(result: CommandResult) -> dict[str, object]:
    return {
        "operation": list(result.argv[1:3]),
        "argv_sha256": sha256_bytes(canonical_json(list(result.argv))),
        "returncode": result.returncode,
        "stdout_sha256": sha256_bytes(result.stdout),
        "stdout_bytes": len(result.stdout),
        "stderr_sha256": sha256_bytes(result.stderr),
        "stderr_bytes": len(result.stderr),
    }


def _create_journal_details(
    phase: str, value: object
) -> dict[str, object] | None:
    if type(value) is not dict:
        raise ValueError("Prime create journal details differ")
    details = cast(dict[str, object], value)
    if phase == "dispatch":
        if set(details) != {"endpoint", "payload_sha256"}:
            raise ValueError("Prime create journal dispatch schema differs")
        if details["endpoint"] != PODS_CREATE_ENDPOINT:
            raise ValueError("Prime create journal endpoint differs")
        _digest(details["payload_sha256"], "create journal payload")
        return details
    if set(details) == {"error"}:
        if details["error"] not in {"transport_failure", "schema_or_binding_failure"}:
            raise ValueError("Prime create journal error differs")
        return None
    if set(details) == {"status_code"}:
        status = details["status_code"]
        if not (status is None or (type(status) is int and 100 <= status <= 599)):
            raise ValueError("Prime create journal status differs")
        return None
    if set(details) != {
        "status_code", "pod_identity_sha256", "response_sha256", "response_bytes"
    }:
        raise ValueError("Prime create journal outcome schema differs")
    if details["status_code"] not in {200, 201}:
        raise ValueError("Prime create journal success status differs")
    _digest(details["pod_identity_sha256"], "create journal pod identity")
    _digest(details["response_sha256"], "create journal response")
    if type(details["response_bytes"]) is not int or not 0 <= details[
        "response_bytes"
    ] <= MAX_RESPONSE_BYTES:
        raise ValueError("Prime create journal response length differs")
    return details


def replay_command_journal(records_raw: bytes, journal_raw: bytes) -> CommandJournalSummary:
    try:
        records = json.loads(records_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Prime command records are not JSON") from error
    if type(records) is not list or canonical_json(records) != records_raw:
        raise ValueError("Prime command records differ")
    if not journal_raw.endswith(b"\n") or len(journal_raw) > MAX_COMMAND_OUTPUT_BYTES:
        raise ValueError("Prime command journal framing differs")
    pending: tuple[str, str, dict[str, object] | None] | None = None
    successes: list[dict[str, object]] = []
    prime_count = wallet_count = 0
    create_payload_hash: str | None = None
    create_status: int | None = None
    create_pod_hash: str | None = None
    create_response_hash: str | None = None
    create_response_bytes: int | None = None
    wallet_outcomes: list[Mapping[str, object]] = []
    keyscan_dispatches: list[int] = []
    keyscan_outcomes: list[int] = []
    keyscan_stdout_hashes: list[str] = []
    lines = journal_raw.splitlines()
    for ordinal, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Prime command journal is not JSON") from error
        if type(record) is not dict or canonical_json(record) != line:
            raise ValueError("Prime command journal record is not canonical")
        item = cast(dict[str, object], record)
        if set(item) != {"schema_version", "phase", "operation", "ordinal", "details"}:
            raise ValueError("Prime command journal schema differs")
        phase, operation = item["phase"], item["operation"]
        if (
            item["schema_version"] != 2
            or phase not in {"dispatch", "outcome"}
            or type(operation) is not str
            or type(item["ordinal"]) is not int
            or item["ordinal"] != ordinal
        ):
            raise ValueError("Prime command journal binding differs")
        operation_text = operation
        if operation_text == "prime-api-wallet":
            kind = "wallet"
            details = validate_wallet_journal_details(phase, item["details"])
        elif operation_text == "prime-api-create":
            kind = "create"
            details = _create_journal_details(phase, item["details"])
        else:
            kind = "command"
            details = validate_command_journal_details(
                phase, operation_text, item["details"]
            )
        if phase == "dispatch":
            if pending is not None or details is None:
                raise ValueError("Prime command journal has an unpaired dispatch")
            pending = operation_text, kind, details
            if kind == "wallet":
                wallet_count += 1
            elif kind == "command" and operation_text.lower().startswith("prime.exe --plain "):
                prime_count += 1
            elif kind == "command" and operation_text.lower().startswith("ssh-keyscan.exe "):
                keyscan_dispatches.append(ordinal)
            elif kind == "create":
                if create_payload_hash is not None:
                    raise ValueError("Prime create dispatch is duplicated")
                create_payload_hash = cast(str, details["payload_sha256"])
        else:
            if pending is None or pending[:2] != (operation_text, kind):
                raise ValueError("Prime command journal outcome is out of order")
            dispatch_details = pending[2]
            if kind == "wallet" and details is not None:
                for key in (
                    "endpoint", "method", "limit", "offset", "team_id_sha256",
                    "request_ordinal",
                ):
                    if details[key] != cast(dict[str, object], dispatch_details)[key]:
                        raise ValueError("Prime wallet dispatch and outcome differ")
                wallet_outcomes.append(details)
            elif kind == "command" and details is not None:
                if details["argv_sha256"] != cast(dict[str, object], dispatch_details)[
                    "argv_sha256"
                ]:
                    raise ValueError("Prime command dispatch and outcome differ")
                successes.append(details)
                if operation_text.lower().startswith("ssh-keyscan.exe "):
                    keyscan_stdout_hashes.append(cast(str, details["stdout_sha256"]))
            elif kind == "create" and details is not None:
                create_status = cast(int, details["status_code"])
                create_pod_hash = cast(str, details["pod_identity_sha256"])
                create_response_hash = cast(str, details["response_sha256"])
                create_response_bytes = cast(int, details["response_bytes"])
            if kind == "command" and operation_text.lower().startswith("ssh-keyscan.exe "):
                keyscan_outcomes.append(ordinal)
            pending = None
    if pending is not None:
        raise ValueError("Prime command journal ends with a dispatch")
    if records != successes:
        raise ValueError("Prime command records differ from journal outcomes")
    return CommandJournalSummary(
        len(successes), prime_count, wallet_count,
        create_payload_hash, create_status, create_pod_hash,
        create_response_hash, create_response_bytes, tuple(wallet_outcomes),
        tuple(keyscan_dispatches), tuple(keyscan_outcomes), tuple(keyscan_stdout_hashes),
    )


class Lifecycle:
    def __init__(self, context: RuntimeContext, root: Path, started: float) -> None:
        self.context, self.root, self.started = context, root, started
        self.commands: list[dict[str, object]] = []
        self.cli_calls = 0
        self.operational_cli_calls = 0
        self.cleanup_cli_calls = 0
        self.wallet_api_calls = 0
        self.cleanup_deadline: float | None = None
        self.cleanup_phase_deadline: float | None = None
        self.journal_records = 0
        self.wallet_before_epoch = context.now()
        self.create_dispatch_epoch: int | None = None
        self.create_dispatched = False
        self.known_pod_ids: set[str] = set()
        self.trusted_pod_id: str | None = None
        self.terminated_pod_ids: set[str] = set()
        self.pod_name = f"{POD_NAME_PREFIX}-{context.authorization['commit'][:12]}"
        self.journal_path = root / "command-journal.jsonl"

    def begin_cleanup(self) -> None:
        if self.cleanup_deadline is None:
            self.cleanup_deadline = self.context.monotonic() + CLEANUP_TIMEOUT_SECONDS

    def begin_cleanup_phase(self, seconds: int) -> None:
        if self.cleanup_deadline is None:
            raise RuntimeError("Prime one-shot cleanup owner was not started")
        self.cleanup_phase_deadline = min(self.cleanup_deadline, self.context.monotonic() + seconds)

    def remaining(self, *, cleanup: bool) -> float:
        if cleanup:
            if self.cleanup_deadline is None:
                raise RuntimeError("Prime one-shot cleanup owner was not started")
            deadline = self.cleanup_deadline
            if self.cleanup_phase_deadline is not None:
                deadline = min(deadline, self.cleanup_phase_deadline)
            return deadline - self.context.monotonic()
        return MAXIMUM_POD_SECONDS - (self.context.monotonic() - self.started)

    def sleep_bounded(self, *, cleanup: bool) -> None:
        remaining = self.remaining(cleanup=cleanup)
        if remaining <= 0:
            message = (
                "Prime one-shot cleanup deadline elapsed"
                if cleanup
                else "Prime one-shot pod-lifetime deadline elapsed"
            )
            raise TimeoutError(message)
        self.context.sleep(min(POLL_INTERVAL_SECONDS, remaining))

    def journal(self, phase: str, operation: str, details: Mapping[str, object]) -> None:
        self.journal_records += 1
        record = (
            canonical_json(
                {
                    "schema_version": 2,
                    "phase": phase,
                    "operation": operation,
                    "ordinal": self.journal_records,
                    "details": dict(details),
                }
            )
            + b"\n"
        )
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        descriptor = os.open(self.journal_path, flags, 0o600)
        with os.fdopen(descriptor, "ab", closefd=True) as handle:
            handle.write(record)
            handle.flush()
            os.fsync(handle.fileno())

    def command(
        self,
        argv: Sequence[str],
        *,
        input_bytes: bytes | None = None,
        timeout: float,
        allow_failure: bool = False,
        cleanup: bool = False,
    ) -> CommandResult:
        remaining = self.remaining(cleanup=cleanup)
        if remaining <= 0:
            raise TimeoutError(
                "Prime one-shot cleanup deadline elapsed"
                if cleanup
                else "Prime one-shot pod-lifetime deadline elapsed"
            )
        operation = " ".join(
            Path(item).name if index == 0 else item for index, item in enumerate(argv[:3])
        )
        self.journal(
            "dispatch",
            operation,
            {
                "argv_sha256": sha256_bytes(canonical_json(list(argv))),
                "timeout": min(timeout, remaining),
            },
        )
        try:
            result = self.context.run(argv, input_bytes, min(timeout, remaining))
        except Exception as error:
            self.journal("outcome", operation, {"error": type(error).__name__})
            raise
        self.commands.append(command_binding(result))
        self.journal("outcome", operation, command_binding(result))
        if self.remaining(cleanup=cleanup) < 0:
            raise TimeoutError(
                "Prime one-shot cleanup deadline elapsed"
                if cleanup
                else "Prime one-shot pod-lifetime deadline elapsed"
            )
        if (
            result.argv != tuple(argv)
            or len(result.stdout) > MAX_OUTPUT_BYTES
            or len(result.stderr) > MAX_OUTPUT_BYTES
            or (result.returncode and not allow_failure)
        ):
            raise RuntimeError("Prime one-shot command failed or differs")
        return result

    def prime(self, *arguments: str, cleanup: bool = False) -> CommandResult:
        if cleanup:
            allowed = (
                arguments == ("pods", "list", "--output", "json")
                or arguments == ("disks", "list", "--output", "json")
                or (
                    len(arguments) == 4
                    and arguments[:2] == ("pods", "terminate")
                    and arguments[2] in self.known_pod_ids
                    and arguments[3] == "--yes"
                )
            )
            if not allowed:
                raise RuntimeError("Prime one-shot cleanup command is not allowlisted")
            self.cleanup_cli_calls += 1
            if self.cleanup_cli_calls > MAX_CLEANUP_PRIME_CLI_CALLS:
                raise RuntimeError("Prime one-shot cleanup CLI call budget exceeded")
        else:
            self.operational_cli_calls += 1
            if self.operational_cli_calls > MAX_OPERATIONAL_PRIME_CLI_CALLS:
                raise RuntimeError("Prime one-shot operational CLI call budget exceeded")
        self.cli_calls += 1
        if self.cli_calls > MAX_PRIME_CLI_CALLS:
            raise RuntimeError("Prime one-shot CLI call budget exceeded")
        return self.command(
            (str(self.context.prime_executable), "--plain", *arguments),
            timeout=COMMAND_TIMEOUT_SECONDS,
            cleanup=cleanup,
        )

    def list_pods(self, *, cleanup: bool = False) -> tuple[list[dict[str, Any]], CommandResult]:
        result = self.prime("pods", "list", "--output", "json", cleanup=cleanup)
        return inventory(result.stdout, "pods"), result

    def list_disks(self, *, cleanup: bool = False) -> tuple[list[dict[str, Any]], CommandResult]:
        result = self.prime("disks", "list", "--output", "json", cleanup=cleanup)
        return inventory(result.stdout, "disks"), result

    def wallet(self, *, cleanup: bool = False) -> WalletSnapshot:
        snapshot = capture_wallet_snapshot(
            self, phase="postcleanup" if cleanup else "precreate", cleanup=cleanup
        )
        if not cleanup:
            self.wallet_before_epoch = self.context.now()
        return snapshot


def direct_create(
    owner: Lifecycle,
    resource: Mapping[str, Any],
    team_id: object,
    dispatch: Path,
    result_path: Path,
) -> dict[str, Any]:
    remaining = owner.remaining(cleanup=False)
    if remaining <= 0:
        raise TimeoutError("Prime one-shot pod-lifetime deadline elapsed before create")
    payload = create_payload(resource, owner.pod_name, team_id)
    publish_once(dispatch, create_dispatch_bytes(resource, payload))
    owner.create_dispatched = True
    owner.create_dispatch_epoch = owner.context.now()
    owner.journal(
        "dispatch",
        "prime-api-create",
        {"endpoint": PODS_CREATE_ENDPOINT, "payload_sha256": sha256_bytes(canonical_json(payload))},
    )
    try:
        response = owner.context.client.client.request(
            "POST",
            PODS_CREATE_ENDPOINT,
            json=payload,
            follow_redirects=False,
            timeout=min(30.0, remaining),
        )
    except owner.context.transport_errors as error:
        owner.journal("outcome", "prime-api-create", {"error": "transport_failure"})
        raise RuntimeError("Prime one-shot create transport outcome is ambiguous") from error
    if getattr(response, "status_code", None) not in {200, 201}:
        owner.journal(
            "outcome", "prime-api-create", {"status_code": getattr(response, "status_code", None)}
        )
        raise RuntimeError("Prime one-shot create status is ambiguous")
    if owner.remaining(cleanup=False) < 0:
        raise TimeoutError("Prime one-shot pod-lifetime deadline elapsed during create")
    raw = bytes(getattr(response, "content", b""))
    value = load_json(raw, "Prime create response")
    identifier = value.get("id")
    if isinstance(identifier, str) and identifier:
        owner.known_pod_ids.add(identifier)
    required = {"id", "name", "gpuName", "gpuCount", "status", "createdAt", "providerType"}
    if (
        not required.issubset(value)
        or not isinstance(identifier, str)
        or not identifier
        or value["name"] != owner.pod_name
        or value["gpuName"] != resource["gpuType"]
        or value["gpuCount"] != resource["gpuCount"]
        or value["providerType"] != resource["provider"]
        or not isinstance(value["status"], str)
        or value["status"].upper() not in {"INSTALLING", "PENDING", "ACTIVE"}
        or not isinstance(value["createdAt"], str)
        or not value["createdAt"]
    ):
        owner.journal("outcome", "prime-api-create", {"error": "schema_or_binding_failure"})
        raise RuntimeError("Prime one-shot create response schema is ambiguous")
    owner.trusted_pod_id = identifier
    owner.journal(
        "outcome",
        "prime-api-create",
        {
            "status_code": response.status_code,
            "pod_identity_sha256": sha256_bytes(identifier.encode()),
            "response_sha256": sha256_bytes(raw),
            "response_bytes": len(raw),
        },
    )
    publish_once(
        result_path,
        create_result_bytes(response.status_code, raw, value["id"], owner.context.authorization),
    )
    return value


def cleanup(
    owner: Lifecycle, before: WalletSnapshot | None
) -> tuple[bool, dict[str, object], list[str]]:
    errors: list[str] = []
    owner.begin_cleanup()
    owner.begin_cleanup_phase(CLEANUP_POD_TIMEOUT_SECONDS)
    terminated: list[str] = []
    pods: list[dict[str, Any]] | None = None
    try:
        terminate_unattempted_known_ids(owner, terminated, errors)
    except Exception as error:
        errors.append(f"terminate:{type(error).__name__}")
    for poll in range(MAX_TERMINATION_POLLS):
        try:
            pods, _result = owner.list_pods(cleanup=True)
            exact = [row for row in pods if row.get("name") == owner.pod_name]
            identifiers = {cast(str, row["id"]) for row in exact if isinstance(row.get("id"), str)}
            owner.known_pod_ids.update(identifiers)
            if len(owner.known_pod_ids) > MAX_OWNED_POD_IDENTITIES:
                raise RuntimeError("Prime cleanup identity bound exceeded")
            terminate_unattempted_known_ids(owner, terminated, errors)
        except Exception as error:
            errors.append(f"pods:{type(error).__name__}")
        if poll + 1 < MAX_TERMINATION_POLLS:
            try:
                owner.sleep_bounded(cleanup=True)
            except TimeoutError as error:
                errors.append(f"pods:{type(error).__name__}")
                break
    if pods:
        errors.append("pods:global_inventory_not_empty")
    disks: list[dict[str, Any]] | None = None
    owner.begin_cleanup_phase(CLEANUP_DISK_TIMEOUT_SECONDS)
    try:
        disks, _ = owner.list_disks(cleanup=True)
        if disks:
            raise RuntimeError("global disk inventory is not empty")
    except Exception as error:
        errors.append(f"disks:{type(error).__name__}")
    wallet_projection: dict[str, object] | None = None
    owner.begin_cleanup_phase(CLEANUP_BILLING_TIMEOUT_SECONDS)
    billing_error: Exception | None = None
    for poll in range(MAX_BILLING_POLLS):
        try:
            snapshot = owner.wallet(cleanup=True)
            observed_epoch = owner.context.now()
            if before is None:
                raise RuntimeError("wallet pre-create snapshot is absent")
            wallet_projection = reconcile_wallet_snapshots(
                before,
                snapshot,
                owned_pod_ids=owner.known_pod_ids,
                dispatch_epoch=owner.create_dispatch_epoch,
                observed_epoch=observed_epoch,
            )
            billing_error = None
            break
        except Exception as error:
            billing_error = error
            if poll + 1 < MAX_BILLING_POLLS:
                try:
                    owner.sleep_bounded(cleanup=True)
                except TimeoutError as timeout:
                    billing_error = timeout
                    break
    if billing_error:
        errors.append(f"wallet:{type(billing_error).__name__}")
    evidence: dict[str, object] = {
        "owned_identity_sha256s": sorted(
            sha256_bytes(identifier.encode()) for identifier in owner.known_pod_ids
        ),
        "terminated_identity_sha256s": sorted(terminated),
        "pods_after_count": None if pods is None else len(pods),
        "disks_after_count": None if disks is None else len(disks),
        "wallet_after": wallet_projection,
        "errors": errors,
    }
    return not errors, evidence, errors


def terminate_unattempted_known_ids(
    owner: Lifecycle, terminated: list[str], errors: list[str]
) -> None:
    if len(owner.known_pod_ids) > MAX_OWNED_POD_IDENTITIES:
        raise RuntimeError("Prime cleanup identity bound exceeded")
    for identifier in sorted(owner.known_pod_ids - owner.terminated_pod_ids):
        owner.terminated_pod_ids.add(identifier)
        try:
            owner.prime("pods", "terminate", identifier, "--yes", cleanup=True)
            terminated.append(sha256_bytes(identifier.encode()))
        except Exception as error:
            errors.append(f"terminate:{type(error).__name__}")


def _authenticate_source(path: str, expected: str) -> None:
    spec = importlib.util.find_spec(path.replace("/", ".").removesuffix(".py"))
    if spec is None or spec.origin is None or sha_file(Path(spec.origin)) != expected:
        raise ValueError(f"Prime one-shot installed owner differs: {path}")


def _production_run(
    argv: Sequence[str], input_bytes: bytes | None, timeout: float
) -> CommandResult:
    environment = None
    if Path(argv[0]).name.lower() == "ssh-keygen.exe":
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith(("SSH_", "PRIME_", "GIT_SSH"))
        }
        environment["PATH"] = r"C:\Windows\System32\OpenSSH;C:\Windows\System32;C:\Windows"
    result = subprocess.run(
        argv,
        input=input_bytes,
        capture_output=True,
        check=False,
        timeout=timeout,
        env=environment,
    )
    return CommandResult(tuple(argv), result.returncode, result.stdout, result.stderr)


def _production_context(signing_key: Path) -> RuntimeContext:
    if sys.version_info[:3] != (3, 13, 2):
        raise ValueError("Prime one-shot requires CPython 3.13.2")
    authorization = cast(dict[str, str], authenticate_authorization(ROOT))
    for relative, expected in (
        (V5_OWNER_PATH, V5_OWNER_SHA256),
        (V5_CONTRACT_PATH, V5_CONTRACT_SHA256),
    ):
        if sha_file(ROOT / relative) != expected:
            raise ValueError("historical v5 binding differs")
    tool = cast(dict[str, object], v5.authenticate_installed_capture_owners()["prime_uv_tool"])
    prime = Path(cast(str, tool["canonical_path"]))
    if sha_file(prime) != tool["sha256"]:
        raise ValueError("Prime executable differs")
    _authenticate_source(PODS_API_OWNER, PODS_API_OWNER_SHA256)
    _authenticate_source(PODS_COMMAND_OWNER, PODS_COMMAND_OWNER_SHA256)
    _authenticate_source(WALLET_API_OWNER, WALLET_API_OWNER_SHA256)
    _authenticate_source(PRIME_CLIENT_OWNER, PRIME_CLIENT_OWNER_SHA256)
    _authenticate_source(PRIME_CONFIG_OWNER, PRIME_CONFIG_OWNER_SHA256)
    openssh = {
        name: Path(cast(str, binding["path"])) for name, binding in OPENSSH_EXECUTABLES.items()
    }
    for name, path in openssh.items():
        binding = OPENSSH_EXECUTABLES[name]
        info = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_size != binding["bytes"]
            or sha_file(path) != binding["sha256"]
        ):
            raise ValueError("OpenSSH executable differs")
    keygen = v5.authenticate_approved_openssh_executable()
    raw_identity = v5._load_terminal_signing_identity()
    v5._authenticate_operator_key(signing_key, raw_identity)
    identity = SigningIdentity(
        raw_identity.principal,
        raw_identity.key_type,
        raw_identity.public_key_base64,
        raw_identity.fingerprint_sha256,
        raw_identity.allowed_signers_sha256,
    )
    linux_uv = Path(r"\\wsl.localhost\Ubuntu\home\mihir\.local\uv-latest\uv")
    if (
        not linux_uv.is_file()
        or linux_uv.stat().st_size != LINUX_UV_BYTES
        or sha_file(linux_uv) != LINUX_UV_SHA256
    ):
        raise ValueError("Linux uv asset differs")
    client = cast(APIClient, v5._construct_api_client())
    configured_team = getattr(client.config, "team_id", None)
    if configured_team is not None and (
        not isinstance(configured_team, str) or not configured_team
    ):
        raise ValueError("Prime configured team identity differs")
    return RuntimeContext(
        ROOT,
        authorization,
        client,
        configured_team,
        v5._httpx_request_error_types(),
        prime,
        openssh,
        Path(cast(str, keygen["path"])),
        signing_key,
        identity,
        linux_uv,
        _production_run,
        lambda: int(time.time()),
        time.monotonic,
        time.sleep,
    )


def production_context() -> RuntimeContext:
    return _production_context(Path.home() / ".ssh" / "id_rsa")


__all__ = [
    "PODS_COMMAND_OWNER",
    "WALLET_API_ENDPOINT",
    "CommandJournalSummary",
    "CommandResult",
    "CreateDispatchSummary",
    "CreateResultSummary",
    "Lifecycle",
    "RuntimeContext",
    "SigningIdentity",
    "assess_pages",
    "cleanup",
    "create_dispatch_bytes",
    "create_result_bytes",
    "direct_create",
    "load_json",
    "production_context",
    "replay_command_journal",
    "sha_file",
    "validate_create_dispatch",
    "validate_create_result",
]
