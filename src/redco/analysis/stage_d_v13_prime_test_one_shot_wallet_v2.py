"""Typed, authenticated wallet pagination and reconciliation for the Prime test one-shot."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import cast

from redco.analysis.stage_d_v13_prime_test_one_shot_contract_v2 import (
    BILLING_RESOURCE_ID_NULL_ALLOWED,
    BILLING_RESOURCE_TYPE,
    MAX_NEW_BILLING_ROWS,
    MAX_POST_WALLET_PAGES,
    MAX_POST_WALLET_REQUESTS,
    MAX_POST_WALLET_ROWS,
    MAX_PRE_WALLET_PAGES,
    MAX_PRE_WALLET_REQUESTS,
    MAX_PREEXISTING_WALLET_ROWS,
    MAX_WALLET_API_CALLS,
    RESERVE_USD,
    SUPPORT_CAP_USD,
    WALLET_API_ENDPOINT,
    WALLET_PAGE_LIMIT,
    WALLET_RECONCILIATION_DOMAIN,
    WALLET_ROW_DOMAIN,
    WALLET_SNAPSHOT_DOMAIN,
    WalletOwner,
    canonical_json,
    sha256_bytes,
)

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_WALLET_PROJECTION_BYTES = 32 * 1024 * 1024
_HEX64 = re.compile(r"[0-9a-f]{64}")
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
BILLING_ROW_KEYS = frozenset(
    {
        "id",
        "created_at",
        "updated_at",
        "last_billed_at",
        "amount_usd",
        "currency",
        "resource_type",
        "resource_id",
    }
)


@dataclass(frozen=True, slots=True)
class BillingRow:
    identifier: str
    amount: Decimal
    amount_text: str
    currency: str
    resource_type: str
    resource_id: str | None
    created_epoch: float
    updated_epoch: float
    last_billed_epoch: float | None
    canonical: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class WalletPage:
    wallet_id: str
    team_id: str | None
    balance: Decimal
    currency: str
    total_billings: int
    rows: tuple[BillingRow, ...]

    @property
    def identity(self) -> tuple[str, str | None, str, Decimal, int]:
        return self.wallet_id, self.team_id, self.currency, self.balance, self.total_billings


@dataclass(frozen=True, slots=True)
class WalletSnapshot:
    wallet_id: str
    team_id: str | None
    balance: Decimal
    currency: str
    total_billings: int
    rows: tuple[BillingRow, ...]
    evidence: dict[str, object]


@dataclass(frozen=True, slots=True)
class SanitizedWalletSnapshot:
    phase: str
    wallet_identity_sha256: str
    team_identity_sha256: str | None
    currency: str
    balance: Decimal
    total: int
    rows: tuple[SanitizedBillingRow, ...]


@dataclass(frozen=True, slots=True)
class SanitizedBillingRow:
    billing_id_sha256: str
    semantic_row_sha256: str
    amount: Decimal
    amount_text: str
    currency: str
    resource_type: str
    resource_id_sha256: str | None
    created_at_epoch: float
    updated_at_epoch: float
    last_billed_at_epoch: float | None


def _strict(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"Prime wallet {label} schema differs")
    return cast(dict[str, object], value)


def _hash(value: object, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ValueError(f"Prime wallet {label} digest differs")
    return value


def _decimal_text(value: object, label: str) -> Decimal:
    if type(value) is not str or _DECIMAL.fullmatch(value) is None:
        raise ValueError(f"Prime wallet {label} decimal differs")
    number = Decimal(value)
    if not number.is_finite() or number < 0:
        raise ValueError(f"Prime wallet {label} decimal differs")
    return number


def _epoch(value: object, label: str) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise ValueError(f"Prime wallet {label} epoch differs")
    result = float(cast(int | float, value))
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Prime wallet {label} epoch differs")
    return result


def _amount_text(value: object, label: str) -> Decimal:
    if type(value) is not str or not value or len(value) > 128:
        raise ValueError(f"Prime wallet {label} amount representation differs")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Prime wallet {label} amount representation differs") from error
    if (
        type(parsed) not in {int, float}
        or canonical_json(parsed).decode("utf-8") != value
    ):
        raise ValueError(f"Prime wallet {label} amount representation differs")
    return decimal_value(parsed, f"Prime wallet {label} amount")


def decimal_value(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or type(value) not in {int, float, Decimal}:
        raise ValueError(f"{label} is not numeric")
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{label} is malformed") from error
    if not number.is_finite() or number < 0:
        raise ValueError(f"{label} is negative or nonfinite")
    return number


def billing_timestamp(value: object, label: str) -> float:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} differs")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} differs") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label} lacks timezone authority")
    return parsed.timestamp()


def _row(value: object) -> BillingRow:
    if not isinstance(value, dict) or set(value) != BILLING_ROW_KEYS:
        raise ValueError("Prime wallet billing row differs")
    row = cast(dict[str, object], value)
    identifier = row["id"]
    resource_id = row["resource_id"]
    if (
        not isinstance(identifier, str)
        or not identifier
        or row["currency"] != "USD"
        or not isinstance(row["resource_type"], str)
        or not row["resource_type"]
        or not (resource_id is None or (isinstance(resource_id, str) and bool(resource_id)))
    ):
        raise ValueError("Prime wallet billing row differs")
    created = billing_timestamp(row["created_at"], "Prime billing created_at")
    updated = billing_timestamp(row["updated_at"], "Prime billing updated_at")
    last_raw = row["last_billed_at"]
    last = None if last_raw is None else billing_timestamp(last_raw, "Prime billing last_billed_at")
    if updated < created or (last is not None and last < created):
        raise ValueError("Prime wallet billing timestamps differ")
    amount = decimal_value(row["amount_usd"], "Prime billing amount")
    amount_text = canonical_json(row["amount_usd"]).decode("utf-8")
    canonical = canonical_json(row)
    return BillingRow(
        identifier,
        amount,
        amount_text,
        row["currency"],
        row["resource_type"],
        resource_id,
        created,
        updated,
        last,
        canonical,
        sha256_bytes(canonical),
    )


def parse_wallet_page(raw: bytes, *, maximum_total: int) -> WalletPage:
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Prime wallet response exceeds its byte bound")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Prime wallet is not JSON") from error
    keys = {
        "wallet_id",
        "team_id",
        "balance_usd",
        "currency",
        "total_billings",
        "recent_billings",
    }
    if not isinstance(value, dict) or (set(value) != keys and set(value) != keys - {"team_id"}):
        raise ValueError("Prime wallet schema differs")
    wallet_id = value["wallet_id"]
    team_id = value.get("team_id")
    total = value["total_billings"]
    raw_rows = value["recent_billings"]
    if (
        not isinstance(wallet_id, str)
        or not wallet_id
        or (team_id is not None and (not isinstance(team_id, str) or not team_id))
        or value["currency"] != "USD"
        or type(total) is not int
        or not 0 <= total <= maximum_total
        or not isinstance(raw_rows, list)
        or len(raw_rows) > WALLET_PAGE_LIMIT
    ):
        raise ValueError("Prime wallet identity or count differs")
    rows = tuple(_row(item) for item in raw_rows)
    identifiers = [row.identifier for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Prime wallet page contains duplicate billing identities")
    return WalletPage(
        wallet_id,
        team_id,
        decimal_value(value["balance_usd"], "Prime wallet balance"),
        "USD",
        total,
        rows,
    )


def _request_page(
    owner: WalletOwner, offset: int, *, maximum_total: int, cleanup: bool
) -> tuple[WalletPage, dict[str, object], bytes]:
    remaining = owner.remaining(cleanup=cleanup)
    if remaining <= 0:
        raise TimeoutError("Prime one-shot wallet deadline elapsed")
    owner.wallet_api_calls += 1
    if owner.wallet_api_calls > MAX_WALLET_API_CALLS:
        raise RuntimeError("Prime one-shot wallet API call budget exceeded")
    team_id = owner.context.wallet_team_id
    params: dict[str, object] = {"limit": WALLET_PAGE_LIMIT, "offset": offset}
    if team_id is not None:
        params["teamId"] = team_id
    request = {
        "endpoint": WALLET_API_ENDPOINT,
        "method": "GET",
        "limit": WALLET_PAGE_LIMIT,
        "offset": offset,
        "team_id_sha256": None if team_id is None else sha256_bytes(team_id.encode()),
        "request_ordinal": owner.wallet_api_calls,
    }
    owner.journal("dispatch", "prime-api-wallet", request)
    try:
        response = owner.context.client.client.request(
            "GET",
            WALLET_API_ENDPOINT,
            params=params,
            follow_redirects=False,
            timeout=min(30.0, remaining),
        )
    except owner.context.transport_errors as error:
        owner.journal("outcome", "prime-api-wallet", {"error": "transport_failure"})
        raise RuntimeError("Prime wallet transport failed") from error
    if owner.remaining(cleanup=cleanup) < 0:
        raise TimeoutError("Prime one-shot wallet deadline elapsed")
    status = getattr(response, "status_code", None)
    raw = bytes(getattr(response, "content", b""))
    headers = getattr(response, "headers", {})
    content_type = str(headers.get("content-type", "")).split(";", 1)[0].strip().lower()
    if status != 200 or content_type != "application/json" or len(raw) > MAX_RESPONSE_BYTES:
        owner.journal(
            "outcome",
            "prime-api-wallet",
            {"status_code": status, "content_type": content_type, "bytes": len(raw)},
        )
        raise RuntimeError("Prime wallet response differs")
    page = parse_wallet_page(raw, maximum_total=maximum_total)
    if team_id is not None and page.team_id != team_id:
        raise RuntimeError("Prime wallet team identity differs")
    receipt = {
        **request,
        "status_code": 200,
        "content_type": content_type,
        "body_sha256": sha256_bytes(raw),
        "body_bytes": len(raw),
        "total_billings": page.total_billings,
        "row_count": len(page.rows),
        "team_id_sha256": None
        if page.team_id is None
        else sha256_bytes(page.team_id.encode()),
        "wallet_identity_sha256": sha256_bytes(page.wallet_id.encode()),
        "currency": page.currency,
        "balance_usd": str(page.balance),
        "row_semantic_sha256s": [
            _sanitized_row_projection(row)["semantic_row_sha256"] for row in page.rows
        ],
    }
    owner.journal("outcome", "prime-api-wallet", receipt)
    return page, receipt, raw


def capture_wallet_snapshot(owner: WalletOwner, *, phase: str, cleanup: bool) -> WalletSnapshot:
    if phase == "precreate":
        maximum_total = MAX_PREEXISTING_WALLET_ROWS
        maximum_pages = MAX_PRE_WALLET_PAGES
        maximum_requests = MAX_PRE_WALLET_REQUESTS
    elif phase == "postcleanup":
        maximum_total = MAX_POST_WALLET_ROWS
        maximum_pages = MAX_POST_WALLET_PAGES
        maximum_requests = MAX_POST_WALLET_REQUESTS
    else:
        raise ValueError("Prime wallet snapshot phase differs")
    start_calls = owner.wallet_api_calls
    first, first_receipt, first_raw = _request_page(
        owner, 0, maximum_total=maximum_total, cleanup=cleanup
    )
    total = first.total_billings
    pages = max(1, (total + WALLET_PAGE_LIMIT - 1) // WALLET_PAGE_LIMIT)
    if pages > maximum_pages:
        raise RuntimeError("Prime wallet pagination exceeds its bound")
    rows: list[BillingRow] = []
    receipts = [first_receipt]
    for page_ordinal in range(pages):
        offset = page_ordinal * WALLET_PAGE_LIMIT
        if page_ordinal == 0:
            page = first
        else:
            page, receipt, _raw = _request_page(
                owner, offset, maximum_total=maximum_total, cleanup=cleanup
            )
            receipts.append(receipt)
        expected_rows = min(WALLET_PAGE_LIMIT, total - offset)
        if page.identity != first.identity or len(page.rows) != expected_rows:
            raise RuntimeError("Prime wallet pagination changed or is incomplete")
        rows.extend(page.rows)
    reread, reread_receipt, reread_raw = _request_page(
        owner, 0, maximum_total=maximum_total, cleanup=cleanup
    )
    receipts.append(reread_receipt)
    if reread != first or reread_raw != first_raw:
        raise RuntimeError("Prime wallet leading page changed during pagination")
    identifiers = [row.identifier for row in rows]
    if len(rows) != total or len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Prime wallet pagination overlaps or is incomplete")
    request_count = owner.wallet_api_calls - start_calls
    if request_count != pages + 1 or request_count > maximum_requests:
        raise RuntimeError("Prime wallet request topology differs")
    evidence: dict[str, object] = {
        "schema_version": 2,
        "domain": WALLET_SNAPSHOT_DOMAIN,
        "endpoint": WALLET_API_ENDPOINT,
        "phase": phase,
        "wallet_identity_sha256": sha256_bytes(first.wallet_id.encode()),
        "team_identity_sha256": None
        if first.team_id is None
        else sha256_bytes(first.team_id.encode()),
        "currency": first.currency,
        "balance_usd": str(first.balance),
        "total_billings": total,
        "rows": [_sanitized_row_projection(row) for row in rows],
        "pagination": {
            "page_limit": WALLET_PAGE_LIMIT,
            "page_count": pages,
            "request_count": request_count,
            "pages": receipts,
            "transcript_sha256": sha256_bytes(canonical_json(receipts)),
        },
    }
    return WalletSnapshot(
        first.wallet_id,
        first.team_id,
        first.balance,
        first.currency,
        total,
        tuple(rows),
        evidence,
    )


def validate_wallet_snapshot_projection(
    value: object, *, expected_phase: str
) -> SanitizedWalletSnapshot:
    snapshot = _strict(
        value,
        {
            "schema_version",
            "domain",
            "endpoint",
            "phase",
            "wallet_identity_sha256",
            "team_identity_sha256",
            "currency",
            "balance_usd",
            "total_billings",
            "rows",
            "pagination",
        },
        "snapshot",
    )
    total_limit = (
        MAX_PREEXISTING_WALLET_ROWS
        if expected_phase == "precreate"
        else MAX_POST_WALLET_ROWS
    )
    page_limit = (
        MAX_PRE_WALLET_PAGES if expected_phase == "precreate" else MAX_POST_WALLET_PAGES
    )
    request_limit = (
        MAX_PRE_WALLET_REQUESTS
        if expected_phase == "precreate"
        else MAX_POST_WALLET_REQUESTS
    )
    total = snapshot["total_billings"]
    team_hash = snapshot["team_identity_sha256"]
    raw_rows = snapshot["rows"]
    if (
        snapshot["schema_version"] != 2
        or snapshot["domain"] != WALLET_SNAPSHOT_DOMAIN
        or snapshot["endpoint"] != WALLET_API_ENDPOINT
        or snapshot["phase"] != expected_phase
        or snapshot["currency"] != "USD"
        or type(total) is not int
        or not 0 <= total <= total_limit
        or not isinstance(raw_rows, list)
        or len(raw_rows) != total
        or not (team_hash is None or type(team_hash) is str)
    ):
        raise ValueError("Prime wallet snapshot binding differs")
    wallet_hash = _hash(snapshot["wallet_identity_sha256"], "wallet identity")
    normalized_team = None if team_hash is None else _hash(team_hash, "team identity")
    rows: list[SanitizedBillingRow] = []
    for item in raw_rows:
        rows.append(_validate_sanitized_row(item, "snapshot row"))
    if len({row.billing_id_sha256 for row in rows}) != len(rows):
        raise ValueError("Prime wallet snapshot billing identities overlap")
    pagination = _strict(
        snapshot["pagination"],
        {"page_limit", "page_count", "request_count", "pages", "transcript_sha256"},
        "pagination",
    )
    page_count = pagination["page_count"]
    request_count = pagination["request_count"]
    raw_pages = pagination["pages"]
    expected_page_count = max(1, (total + WALLET_PAGE_LIMIT - 1) // WALLET_PAGE_LIMIT)
    if (
        pagination["page_limit"] != WALLET_PAGE_LIMIT
        or type(page_count) is not int
        or page_count != expected_page_count
        or page_count > page_limit
        or type(request_count) is not int
        or request_count != page_count + 1
        or request_count > request_limit
        or not isinstance(raw_pages, list)
        or len(raw_pages) != request_count
        or _hash(pagination["transcript_sha256"], "pagination transcript")
        != sha256_bytes(canonical_json(raw_pages))
    ):
        raise ValueError("Prime wallet pagination topology differs")
    expected_offsets = [index * WALLET_PAGE_LIMIT for index in range(page_count)] + [0]
    first_page: dict[str, object] | None = None
    previous_ordinal: int | None = None
    for index, item in enumerate(raw_pages):
        page = _strict(
            item,
            {
                "endpoint",
                "method",
                "limit",
                "offset",
                "team_id_sha256",
                "request_ordinal",
                "status_code",
                "content_type",
                "body_sha256",
                "body_bytes",
                "total_billings",
                "row_count",
                "wallet_identity_sha256",
                "currency",
                "balance_usd",
                "row_semantic_sha256s",
            },
            "pagination page",
        )
        ordinal = page["request_ordinal"]
        body_bytes = page["body_bytes"]
        row_count = page["row_count"]
        row_hashes = page["row_semantic_sha256s"]
        _hash(page["body_sha256"], "page body")
        expected_rows = (
            min(WALLET_PAGE_LIMIT, total - expected_offsets[index])
            if index < page_count
            else min(WALLET_PAGE_LIMIT, total)
        )
        row_start = expected_offsets[index] if index < page_count else 0
        expected_row_hashes = [
            row.semantic_row_sha256 for row in rows[row_start : row_start + expected_rows]
        ]
        if (
            page["endpoint"] != WALLET_API_ENDPOINT
            or page["method"] != "GET"
            or page["limit"] != WALLET_PAGE_LIMIT
            or page["offset"] != expected_offsets[index]
            or page["team_id_sha256"] != normalized_team
            or type(ordinal) is not int
            or (previous_ordinal is not None and ordinal != previous_ordinal + 1)
            or page["status_code"] != 200
            or page["content_type"] != "application/json"
            or type(body_bytes) is not int
            or not 0 <= body_bytes <= MAX_RESPONSE_BYTES
            or page["total_billings"] != total
            or type(row_count) is not int
            or row_count != expected_rows
            or _hash(page["wallet_identity_sha256"], "page wallet identity")
            != wallet_hash
            or page["currency"] != "USD"
            or _decimal_text(page["balance_usd"], "page balance")
            != _decimal_text(snapshot["balance_usd"], "balance")
            or type(row_hashes) is not list
            or row_hashes != expected_row_hashes
        ):
            raise ValueError("Prime wallet pagination page differs")
        previous_ordinal = ordinal
        if index == 0:
            first_page = page
        elif index == page_count:
            assert first_page is not None
            for key in (
                "endpoint",
                "method",
                "limit",
                "offset",
                "team_id_sha256",
                "status_code",
                "content_type",
                "body_sha256",
                "body_bytes",
                "total_billings",
                "row_count",
                "wallet_identity_sha256",
                "currency",
                "balance_usd",
                "row_semantic_sha256s",
            ):
                if page[key] != first_page[key]:
                    raise ValueError("Prime wallet page-zero reread differs")
    return SanitizedWalletSnapshot(
        expected_phase,
        wallet_hash,
        normalized_team,
        "USD",
        _decimal_text(snapshot["balance_usd"], "balance"),
        total,
        tuple(rows),
    )


def validate_wallet_snapshot_bytes(
    raw: bytes, *, expected_phase: str
) -> SanitizedWalletSnapshot:
    if len(raw) > MAX_WALLET_PROJECTION_BYTES:
        raise ValueError("Prime wallet snapshot exceeds its byte bound")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Prime wallet snapshot is not JSON") from error
    if canonical_json(value) != raw:
        raise ValueError("Prime wallet snapshot is not canonical")
    return validate_wallet_snapshot_projection(value, expected_phase=expected_phase)


def validate_wallet_snapshot_journal(
    value: object, *, expected_phase: str, outcomes: Sequence[Mapping[str, object]]
) -> SanitizedWalletSnapshot:
    summary = validate_wallet_snapshot_projection(value, expected_phase=expected_phase)
    snapshot = cast(dict[str, object], value)
    pagination = cast(dict[str, object], snapshot["pagination"])
    if list(outcomes) != pagination["pages"]:
        raise ValueError("Prime wallet snapshot differs from command journal")
    return summary


def validate_wallet_journal_details(
    phase: str, value: object
) -> dict[str, object] | None:
    if type(value) is not dict:
        raise ValueError("Prime wallet journal details differ")
    details = cast(dict[str, object], value)
    request_keys = {
        "endpoint", "method", "limit", "offset", "team_id_sha256", "request_ordinal"
    }
    success_keys = request_keys | {
        "status_code", "content_type", "body_sha256", "body_bytes",
        "total_billings", "row_count",
        "wallet_identity_sha256", "currency", "balance_usd",
        "row_semantic_sha256s",
    }
    if phase == "outcome" and set(details) == {"error"}:
        if details["error"] != "transport_failure":
            raise ValueError("Prime wallet journal error differs")
        return None
    if phase == "outcome" and set(details) == {"status_code", "content_type", "bytes"}:
        status = details["status_code"]
        if (
            not (status is None or (type(status) is int and 100 <= status <= 599))
            or type(details["content_type"]) is not str
            or len(details["content_type"]) > 128
            or type(details["bytes"]) is not int
            or not 0 <= details["bytes"] <= MAX_RESPONSE_BYTES
        ):
            raise ValueError("Prime wallet failure journal differs")
        return None
    if set(details) != (request_keys if phase == "dispatch" else success_keys):
        raise ValueError("Prime wallet journal schema differs")
    team_hash = details["team_id_sha256"]
    if (
        details["endpoint"] != WALLET_API_ENDPOINT
        or details["method"] != "GET"
        or details["limit"] != WALLET_PAGE_LIMIT
        or type(details["offset"]) is not int
        or details["offset"] < 0
        or details["offset"] % WALLET_PAGE_LIMIT
        or not (team_hash is None or type(team_hash) is str)
        or (type(team_hash) is str and _hash(team_hash, "team identity") != team_hash)
        or type(details["request_ordinal"]) is not int
        or not 1 <= details["request_ordinal"] <= MAX_WALLET_API_CALLS
    ):
        raise ValueError("Prime wallet journal binding differs")
    if phase == "outcome" and (
        details["status_code"] != 200
        or details["content_type"] != "application/json"
        or _hash(details["body_sha256"], "wallet body") != details["body_sha256"]
        or type(details["body_bytes"]) is not int
        or not 0 <= details["body_bytes"] <= MAX_RESPONSE_BYTES
        or type(details["total_billings"]) is not int
        or not 0 <= details["total_billings"] <= MAX_POST_WALLET_ROWS
        or type(details["row_count"]) is not int
        or not 0 <= details["row_count"] <= WALLET_PAGE_LIMIT
        or _hash(details["wallet_identity_sha256"], "wallet outcome identity")
        != details["wallet_identity_sha256"]
        or details["currency"] != "USD"
        or _decimal_text(details["balance_usd"], "wallet outcome balance") < 0
        or type(details["row_semantic_sha256s"]) is not list
        or len(cast(list[object], details["row_semantic_sha256s"]))
        != details["row_count"]
        or any(
            type(item) is not str or _HEX64.fullmatch(item) is None
            for item in cast(list[object], details["row_semantic_sha256s"])
        )
    ):
        raise ValueError("Prime wallet outcome differs")
    return details


def _sanitized_row_projection(row: BillingRow) -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": 2,
        "domain": WALLET_ROW_DOMAIN,
        "billing_id_sha256": sha256_bytes(row.identifier.encode()),
        "amount_usd": row.amount_text,
        "currency": row.currency,
        "resource_type": row.resource_type,
        "resource_id_sha256": None
        if row.resource_id is None
        else sha256_bytes(row.resource_id.encode()),
        "created_at_epoch": row.created_epoch,
        "updated_at_epoch": row.updated_epoch,
        "last_billed_at_epoch": row.last_billed_epoch,
    }
    return {**core, "semantic_row_sha256": sha256_bytes(canonical_json(core))}


def _validate_sanitized_row(value: object, label: str) -> SanitizedBillingRow:
    row = _strict(
        value,
        {
            "schema_version",
            "domain",
            "billing_id_sha256",
            "semantic_row_sha256",
            "amount_usd",
            "currency",
            "resource_type",
            "resource_id_sha256",
            "created_at_epoch",
            "updated_at_epoch",
            "last_billed_at_epoch",
        },
        label,
    )
    semantic_digest = _hash(row["semantic_row_sha256"], f"{label} semantic")
    core = {key: item for key, item in row.items() if key != "semantic_row_sha256"}
    resource_hash = row["resource_id_sha256"]
    normalized_resource = (
        None
        if resource_hash is None
        else _hash(resource_hash, f"{label} resource")
        if type(resource_hash) is str
        else ""
    )
    created = _epoch(row["created_at_epoch"], f"{label} created")
    updated = _epoch(row["updated_at_epoch"], f"{label} updated")
    last_raw = row["last_billed_at_epoch"]
    last = None if last_raw is None else _epoch(last_raw, f"{label} billed")
    amount_text = row["amount_usd"]
    if type(amount_text) is not str:
        raise ValueError(f"Prime wallet {label} amount representation differs")
    amount = _amount_text(amount_text, label)
    if (
        row["schema_version"] != 2
        or row["domain"] != WALLET_ROW_DOMAIN
        or row["currency"] != "USD"
        or type(row["resource_type"]) is not str
        or not row["resource_type"]
        or not (resource_hash is None or normalized_resource)
        or updated < created
        or (last is not None and last < created)
        or semantic_digest != sha256_bytes(canonical_json(core))
    ):
        raise ValueError(f"Prime wallet {label} semantic commitment differs")
    return SanitizedBillingRow(
        _hash(row["billing_id_sha256"], f"{label} identity"),
        semantic_digest,
        amount,
        amount_text,
        "USD",
        row["resource_type"],
        normalized_resource,
        created,
        updated,
        last,
    )


def reconcile_wallet_snapshots(
    before: WalletSnapshot,
    after: WalletSnapshot,
    *,
    owned_pod_ids: set[str],
    dispatch_epoch: int | None,
    observed_epoch: int,
) -> dict[str, object]:
    if (
        before.total_billings > MAX_PREEXISTING_WALLET_ROWS
        or after.total_billings > MAX_POST_WALLET_ROWS
        or before.wallet_id != after.wallet_id
        or before.team_id != after.team_id
        or before.currency != after.currency
    ):
        raise RuntimeError("wallet identity or history bound changed")
    old_by_id = {row.identifier: row for row in before.rows}
    post_old = [row for row in after.rows if row.identifier in old_by_id]
    if [row.identifier for row in post_old] != [row.identifier for row in before.rows]:
        raise RuntimeError("wallet historical billing order changed")
    if any(
        row.canonical != old_by_id[row.identifier].canonical
        or row.sha256 != old_by_id[row.identifier].sha256
        for row in post_old
    ):
        raise RuntimeError("wallet historical billing content changed")
    new = [row for row in after.rows if row.identifier not in old_by_id]
    if (
        after.total_billings != before.total_billings + len(new)
        or not 1 <= len(new) <= MAX_NEW_BILLING_ROWS
    ):
        raise RuntimeError("wallet billing cardinality is not reconciled")
    delta = before.balance - after.balance
    billed = sum((row.amount for row in new), Decimal())
    if (
        dispatch_epoch is None
        or delta <= 0
        or delta != billed
        or delta > Decimal(str(SUPPORT_CAP_USD))
        or after.balance < Decimal(str(RESERVE_USD))
        or any(
            row.resource_type != BILLING_RESOURCE_TYPE
            or (row.resource_id is None and not BILLING_RESOURCE_ID_NULL_ALLOWED)
            or row.resource_id not in owned_pod_ids
            or not dispatch_epoch <= row.created_epoch <= observed_epoch
            or not dispatch_epoch <= row.updated_epoch <= observed_epoch
            or (
                row.last_billed_epoch is not None
                and not dispatch_epoch <= row.last_billed_epoch <= observed_epoch
            )
            for row in new
        )
    ):
        raise RuntimeError("wallet billing is not reconciled")
    projection: dict[str, object] = {
        "schema_version": 2,
        "domain": WALLET_RECONCILIATION_DOMAIN,
        "before_snapshot_sha256": sha256_bytes(canonical_json(before.evidence)),
        "after_snapshot": after.evidence,
        "delta_usd": str(delta),
        "support_cap_usd": str(Decimal(str(SUPPORT_CAP_USD))),
        "reserve_usd": str(Decimal(str(RESERVE_USD))),
        "new_rows": [_sanitized_row_projection(row) for row in new],
        "owned_pod_identity_sha256s": sorted(
            sha256_bytes(identifier.encode()) for identifier in owned_pod_ids
        ),
        "interval_start_epoch": dispatch_epoch,
        "interval_end_epoch": observed_epoch,
    }
    return projection


def replay_wallet_reconciliation(
    before_value: object, reconciliation_value: object
) -> dict[str, object]:
    before = validate_wallet_snapshot_projection(before_value, expected_phase="precreate")
    reconciliation = _strict(
        reconciliation_value,
        {
            "schema_version",
            "domain",
            "before_snapshot_sha256",
            "after_snapshot",
            "delta_usd",
            "support_cap_usd",
            "reserve_usd",
            "new_rows",
            "owned_pod_identity_sha256s",
            "interval_start_epoch",
            "interval_end_epoch",
        },
        "reconciliation",
    )
    if (
        reconciliation["schema_version"] != 2
        or reconciliation["domain"] != WALLET_RECONCILIATION_DOMAIN
        or _hash(reconciliation["before_snapshot_sha256"], "before snapshot")
        != sha256_bytes(canonical_json(before_value))
    ):
        raise ValueError("Prime wallet reconciliation binding differs")
    after = validate_wallet_snapshot_projection(
        reconciliation["after_snapshot"], expected_phase="postcleanup"
    )
    if (
        before.wallet_identity_sha256 != after.wallet_identity_sha256
        or before.team_identity_sha256 != after.team_identity_sha256
        or before.currency != after.currency
    ):
        raise ValueError("Prime wallet reconciliation identity differs")
    before_by_id = {row.billing_id_sha256: row for row in before.rows}
    after_old = [row for row in after.rows if row.billing_id_sha256 in before_by_id]
    if after_old != list(before.rows):
        raise ValueError("Prime wallet historical billing order or content differs")
    after_new = [row for row in after.rows if row.billing_id_sha256 not in before_by_id]
    raw_new = reconciliation["new_rows"]
    raw_owned = reconciliation["owned_pod_identity_sha256s"]
    if (
        not isinstance(raw_new, list)
        or not 1 <= len(raw_new) <= MAX_NEW_BILLING_ROWS
        or after.total != before.total + len(raw_new)
        or not isinstance(raw_owned, list)
    ):
        raise ValueError("Prime wallet reconciliation cardinality differs")
    owned = [_hash(item, "owned pod identity") for item in raw_owned]
    if owned != sorted(set(owned)):
        raise ValueError("Prime wallet owned pod identities differ")
    start = _epoch(reconciliation["interval_start_epoch"], "interval start")
    end = _epoch(reconciliation["interval_end_epoch"], "interval end")
    if end < start:
        raise ValueError("Prime wallet reconciliation interval differs")
    sanitized_new: list[SanitizedBillingRow] = []
    billed = Decimal()
    for item in raw_new:
        row = _validate_sanitized_row(item, "new billing row")
        if (
            row.resource_type != BILLING_RESOURCE_TYPE
            or row.resource_id_sha256 is None
            or row.resource_id_sha256 not in owned
            or not start <= row.created_at_epoch <= end
            or not start <= row.updated_at_epoch <= end
            or (
                row.last_billed_at_epoch is not None
                and not start <= row.last_billed_at_epoch <= end
            )
        ):
            raise ValueError("Prime wallet new billing row differs")
        sanitized_new.append(row)
        billed += row.amount
    if sanitized_new != after_new or len(
        {row.billing_id_sha256 for row in sanitized_new}
    ) != len(sanitized_new):
        raise ValueError("Prime wallet new billing set differs")
    delta = _decimal_text(reconciliation["delta_usd"], "delta")
    cap = _decimal_text(reconciliation["support_cap_usd"], "support cap")
    reserve = _decimal_text(reconciliation["reserve_usd"], "reserve")
    if (
        cap != Decimal(str(SUPPORT_CAP_USD))
        or reserve != Decimal(str(RESERVE_USD))
        or delta <= 0
        or delta != before.balance - after.balance
        or delta != billed
        or delta > cap
        or after.balance < reserve
    ):
        raise ValueError("Prime wallet reconciliation financial law differs")
    return {
        "before_total": before.total,
        "after_total": after.total,
        "new_rows": len(sanitized_new),
        "delta_usd": str(delta),
        "wallet_identity_sha256": before.wallet_identity_sha256,
        "team_identity_sha256": before.team_identity_sha256,
        "owned_pod_identity_sha256s": owned,
        "new_resource_identity_sha256s": [
            cast(str, row.resource_id_sha256) for row in sanitized_new
        ],
    }


__all__ = [
    "BillingRow",
    "WalletSnapshot",
    "billing_timestamp",
    "capture_wallet_snapshot",
    "decimal_value",
    "parse_wallet_page",
    "reconcile_wallet_snapshots",
    "replay_wallet_reconciliation",
    "validate_wallet_journal_details",
    "validate_wallet_snapshot_bytes",
    "validate_wallet_snapshot_journal",
    "validate_wallet_snapshot_projection",
]
