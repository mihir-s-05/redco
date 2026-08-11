"""Adversarial tests for the isolated Prime one-shot wallet trust boundary."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import ParamSpec, TypeVar, cast

import pytest

from redco.analysis import stage_d_v13_prime_test_one_shot_wallet_v2 as wallet_owner
from redco.analysis.stage_d_v13_prime_test_one_shot_contract_v2 import (
    MAX_POST_WALLET_REQUESTS,
    MAX_PRE_WALLET_REQUESTS,
    MAX_WALLET_API_CALLS,
    WALLET_API_ENDPOINT,
    canonical_json,
    sha256_bytes,
)

P = ParamSpec("P")
R = TypeVar("R")
_TEAM_SENTINEL = "team-sentinel-do-not-persist"
_WALLET_SENTINEL = "wallet-sentinel-do-not-persist"


def _parametrize(
    argnames: str | Sequence[str], argvalues: Sequence[object]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    return cast(
        Callable[[Callable[P, R]], Callable[P, R]],
        pytest.mark.parametrize(argnames, argvalues),
    )


def _billing_row(
    identifier: str,
    *,
    amount: int | float = 1.0,
    resource_type: str = "pod",
    resource_id: str | None = "owned",
) -> dict[str, object]:
    return {
        "id": identifier,
        "created_at": "1970-01-01T00:16:40+00:00",
        "updated_at": "1970-01-01T00:16:40+00:00",
        "last_billed_at": "1970-01-01T00:16:40+00:00",
        "amount_usd": amount,
        "currency": "USD",
        "resource_type": resource_type,
        "resource_id": resource_id,
    }


def _wallet_page(balance: float, total: int, rows: list[dict[str, object]]) -> bytes:
    return canonical_json(
        {
            "wallet_id": _WALLET_SENTINEL,
            "team_id": _TEAM_SENTINEL,
            "balance_usd": balance,
            "currency": "USD",
            "total_billings": total,
            "recent_billings": rows,
        }
    )


@dataclass(frozen=True)
class _Response:
    status_code: int
    content: bytes
    headers: Mapping[str, str]


class _Transport:
    def __init__(self, handler: Callable[[int, int], bytes]) -> None:
        self.handler = handler
        self.calls: list[dict[str, object]] = []

    def request(self, method: str, url: str, **kwargs: object) -> _Response:
        params = cast(dict[str, object], kwargs["params"])
        assert method == "GET" and url == WALLET_API_ENDPOINT
        assert kwargs["follow_redirects"] is False
        assert params["limit"] == 100 and params["teamId"] == _TEAM_SENTINEL
        self.calls.append(dict(params))
        return _Response(
            200,
            self.handler(cast(int, params["offset"]), len(self.calls)),
            {"content-type": "application/json"},
        )


class _Client:
    def __init__(self, transport: _Transport) -> None:
        self.client = transport


class _Context:
    def __init__(self, transport: _Transport) -> None:
        self.client = _Client(transport)
        self.wallet_team_id = _TEAM_SENTINEL
        self.transport_errors = (OSError,)


class _Owner:
    def __init__(self, handler: Callable[[int, int], bytes]) -> None:
        self.transport = _Transport(handler)
        self.context = _Context(self.transport)
        self.wallet_api_calls = 0
        self.create_dispatch_epoch: int | None = 1_000
        self.known_pod_ids = {"owned"}
        self.records: list[tuple[str, str, Mapping[str, object]]] = []

    def remaining(self, *, cleanup: bool) -> float:
        del cleanup
        return 600.0

    def journal(self, phase: str, operation: str, details: Mapping[str, object]) -> None:
        self.records.append((phase, operation, details))


def _handler(balance: float, rows: list[dict[str, object]]) -> Callable[[int, int], bytes]:
    def page(offset: int, _ordinal: int) -> bytes:
        return _wallet_page(balance, len(rows), rows[offset : offset + 100])

    return page


def _snapshot(
    balance: float, rows: list[dict[str, object]], *, phase: str
) -> wallet_owner.WalletSnapshot:
    owner = _Owner(_handler(balance, rows))
    return wallet_owner.capture_wallet_snapshot(owner, phase=phase, cleanup=phase == "postcleanup")


def test_installed_number_and_row_schema_are_strict() -> None:
    for value in (True, False, float("nan"), float("inf"), -1, "1"):
        with pytest.raises(ValueError):
            wallet_owner.decimal_value(value, "probe")
    row = _billing_row("old")
    for mutation in (
        {**row, "unknown": None},
        {key: value for key, value in row.items() if key != "updated_at"},
        {**row, "resource_id": ""},
    ):
        with pytest.raises(ValueError):
            wallet_owner.parse_wallet_page(_wallet_page(30.0, 1, [mutation]), maximum_total=4_096)


def test_complete_100_to_101_snapshot_and_exact_offsets() -> None:
    before_rows = [_billing_row(f"old-{index:03d}", amount=0.1) for index in range(100)]
    after_rows = [_billing_row("new", amount=0.5), *before_rows]
    before_owner = _Owner(_handler(30.0, before_rows))
    before = wallet_owner.capture_wallet_snapshot(before_owner, phase="precreate", cleanup=False)
    after_owner = _Owner(_handler(29.5, after_rows))
    after = wallet_owner.capture_wallet_snapshot(after_owner, phase="postcleanup", cleanup=True)
    result = wallet_owner.reconcile_wallet_snapshots(
        before,
        after,
        owned_pod_ids={"owned"},
        dispatch_epoch=1_000,
        observed_epoch=1_000,
    )
    replayed = wallet_owner.replay_wallet_reconciliation(before.evidence, result)
    assert replayed["before_total"] == 100 and replayed["after_total"] == 101
    assert len(cast(list[object], result["new_rows"])) == 1
    assert [call["offset"] for call in before_owner.transport.calls] == [0, 0]
    assert [call["offset"] for call in after_owner.transport.calls] == [0, 100, 0]


def test_journal_and_wallet_projection_are_replayable_and_identifier_free() -> None:
    before_owner = _Owner(_handler(30.0, [_billing_row("old-secret", amount=1)]))
    before = wallet_owner.capture_wallet_snapshot(
        before_owner, phase="precreate", cleanup=False
    )
    owned_pod_id = "pod-sentinel-do-not-persist"
    after_owner = _Owner(
        _handler(
            29.5,
            [
                _billing_row("new-secret", amount=0.5, resource_id=owned_pod_id),
                _billing_row("old-secret", amount=1),
            ],
        )
    )
    after = wallet_owner.capture_wallet_snapshot(
        after_owner, phase="postcleanup", cleanup=True
    )
    projection = wallet_owner.reconcile_wallet_snapshots(
        before,
        after,
        owned_pod_ids={owned_pod_id},
        dispatch_epoch=1_000,
        observed_epoch=1_000,
    )
    wallet_owner.replay_wallet_reconciliation(before.evidence, projection)
    durable = canonical_json(
        {
            "before_journal": before_owner.records,
            "after_journal": after_owner.records,
            "before": before.evidence,
            "reconciliation": projection,
        }
    )
    for secret in (
        _TEAM_SENTINEL,
        _WALLET_SENTINEL,
        "old-secret",
        "new-secret",
        owned_pod_id,
    ):
        assert secret.encode() not in durable
    assert sha256_bytes(_TEAM_SENTINEL.encode()).encode() in durable
    assert sha256_bytes(owned_pod_id.encode()).encode() in durable


@_parametrize("case", ["missing", "duplicate", "changed_total", "changed_identity"])
def test_pagination_mutation_is_fail_closed(case: str) -> None:
    rows = [_billing_row(f"old-{index:03d}", amount=0.1) for index in range(101)]

    def malformed(offset: int, ordinal: int) -> bytes:
        if case == "changed_total" and ordinal == 3:
            return _wallet_page(30.0, 102, rows[:100])
        if case == "changed_identity" and ordinal == 3:
            value = json.loads(_wallet_page(30.0, 101, rows[:100]))
            value["team_id"] = "other"
            return canonical_json(value)
        if offset == 0:
            return _wallet_page(30.0, 101, rows[:100])
        if case == "missing":
            return _wallet_page(30.0, 101, [])
        return _wallet_page(30.0, 101, [rows[0]])

    with pytest.raises((RuntimeError, ValueError)):
        wallet_owner.capture_wallet_snapshot(_Owner(malformed), phase="postcleanup", cleanup=True)


def test_protocol_headroom_accepts_pre4096_plus_bounded_new_and_rejects_edges() -> None:
    old = [_billing_row(f"old-{index:04d}", amount=0.01) for index in range(4_096)]
    new = [_billing_row("new", amount=0.5), *old]
    before_owner = _Owner(_handler(30.0, old))
    before = wallet_owner.capture_wallet_snapshot(before_owner, phase="precreate", cleanup=False)
    after_owner = _Owner(_handler(29.5, new))
    after = wallet_owner.capture_wallet_snapshot(after_owner, phase="postcleanup", cleanup=True)
    before_pagination = cast(dict[str, object], before.evidence["pagination"])
    after_pagination = cast(dict[str, object], after.evidence["pagination"])
    assert before_pagination["request_count"] == MAX_PRE_WALLET_REQUESTS == 42
    assert after_pagination["request_count"] == 42
    assert (
        len(cast(list[object], wallet_owner.reconcile_wallet_snapshots(
            before,
            after,
            owned_pod_ids={"owned"},
            dispatch_epoch=1_000,
            observed_epoch=1_000,
        )["new_rows"]))
        == 1
    )

    pre_over = _Owner(lambda _offset, _ordinal: _wallet_page(30.0, 4_097, []))
    with pytest.raises(ValueError):
        wallet_owner.capture_wallet_snapshot(pre_over, phase="precreate", cleanup=False)
    assert len(pre_over.transport.calls) == 1

    post_over = _Owner(lambda _offset, _ordinal: _wallet_page(29.5, 8_193, []))
    with pytest.raises(ValueError):
        wallet_owner.capture_wallet_snapshot(post_over, phase="postcleanup", cleanup=True)
    assert len(post_over.transport.calls) == 1
    assert MAX_POST_WALLET_REQUESTS == 83 and MAX_WALLET_API_CALLS == 1_038


_HISTORICAL_MUTATIONS = (
    "amount_usd",
    "resource_id",
    "resource_type",
    "currency",
    "created_at",
    "updated_at",
    "last_billed_at",
    "remove",
    "replace",
    "reorder",
    "numeric_representation",
)


@_parametrize("mutation", _HISTORICAL_MUTATIONS)
def test_exact_historical_rows_and_order_reject_every_mutation(mutation: str) -> None:
    old_a = _billing_row("old-a", amount=1)
    old_b = _billing_row("old-b", amount=2.0)
    before = _snapshot(30.0, [old_a, old_b], phase="precreate")
    post_old = [dict(old_a), dict(old_b)]
    if mutation == "amount_usd":
        post_old[0]["amount_usd"] = 9
    elif mutation == "resource_id":
        post_old[0]["resource_id"] = "other"
    elif mutation == "resource_type":
        post_old[0]["resource_type"] = "disk"
    elif mutation == "currency":
        post_old[0]["currency"] = "EUR"
    elif mutation in {"created_at", "updated_at", "last_billed_at"}:
        post_old[0][mutation] = "1970-01-01T00:16:41+00:00"
    elif mutation == "remove":
        post_old.pop(0)
    elif mutation == "replace":
        post_old[0] = _billing_row("replacement")
    elif mutation == "reorder":
        post_old.reverse()
    else:
        post_old[0]["amount_usd"] = 1.0
    rows = [*post_old, _billing_row("new", amount=0.5)]
    with pytest.raises((RuntimeError, ValueError)):
        after = _snapshot(29.5, rows, phase="postcleanup")
        wallet_owner.reconcile_wallet_snapshots(
            before,
            after,
            owned_pod_ids={"owned"},
            dispatch_epoch=1_000,
            observed_epoch=1_001,
        )


@_parametrize(
    "case",
    ["unrelated_type", "null_id", "wrong_id", "early_time", "mixed", "balance_sum"],
)
def test_new_rows_require_owned_pod_exact_sum_and_dispatch_interval(case: str) -> None:
    before = _snapshot(30.0, [], phase="precreate")
    row = _billing_row("new", amount=0.5)
    rows = [row]
    balance = 29.5
    if case == "unrelated_type":
        row["resource_type"] = "disk"
    elif case == "null_id":
        row["resource_id"] = None
    elif case == "wrong_id":
        row["resource_id"] = "other"
    elif case == "early_time":
        for key in ("created_at", "updated_at", "last_billed_at"):
            row[key] = "1970-01-01T00:16:39+00:00"
    elif case == "mixed":
        rows.append(_billing_row("other", amount=0.25, resource_type="disk"))
        balance = 29.25
    else:
        row["amount_usd"] = 0.4
    after = _snapshot(balance, rows, phase="postcleanup")
    with pytest.raises(RuntimeError):
        wallet_owner.reconcile_wallet_snapshots(
            before,
            after,
            owned_pod_ids={"owned"},
            dispatch_epoch=1_000,
            observed_epoch=1_000,
        )


_PROJECTION_MUTATIONS = (
    "old_order",
    "old_hash",
    "new_set",
    "owned_pod",
    "resource_id",
    "amount",
    "timestamp",
    "total",
    "balance",
    "cap",
    "reserve",
    "page_offset",
    "page_cardinality",
    "page_hash",
    "wallet_identity",
    "team_identity",
)


@_parametrize("mutation", _PROJECTION_MUTATIONS)
def test_sanitized_reconciliation_projection_rejects_every_mutation(
    mutation: str,
) -> None:
    old_a = _billing_row("old-a", amount=1)
    old_b = _billing_row("old-b", amount=2)
    before = _snapshot(30.0, [old_a, old_b], phase="precreate")
    after = _snapshot(
        29.5,
        [_billing_row("new", amount=0.5), old_a, old_b],
        phase="postcleanup",
    )
    projection = wallet_owner.reconcile_wallet_snapshots(
        before,
        after,
        owned_pod_ids={"owned"},
        dispatch_epoch=1_000,
        observed_epoch=1_000,
    )
    mutated = deepcopy(projection)
    after_value = cast(dict[str, object], mutated["after_snapshot"])
    after_rows = cast(list[dict[str, object]], after_value["rows"])
    new_rows = cast(list[dict[str, object]], mutated["new_rows"])
    pagination = cast(dict[str, object], after_value["pagination"])
    pages = cast(list[dict[str, object]], pagination["pages"])
    if mutation == "old_order":
        after_rows[1], after_rows[2] = after_rows[2], after_rows[1]
    elif mutation == "old_hash":
        after_rows[1]["canonical_row_sha256"] = "0" * 64
    elif mutation == "new_set":
        new_rows[0]["billing_id_sha256"] = "1" * 64
    elif mutation == "owned_pod":
        mutated["owned_pod_identity_sha256s"] = ["2" * 64]
    elif mutation == "resource_id":
        new_rows[0]["resource_id_sha256"] = "3" * 64
    elif mutation == "amount":
        new_rows[0]["amount_usd"] = "0.4"
    elif mutation == "timestamp":
        new_rows[0]["created_at_epoch"] = 999
    elif mutation == "total":
        after_value["total_billings"] = 4
    elif mutation == "balance":
        after_value["balance_usd"] = "29.4"
    elif mutation == "cap":
        mutated["support_cap_usd"] = "11.0"
    elif mutation == "reserve":
        mutated["reserve_usd"] = "19.0"
    elif mutation == "page_offset":
        pages[0]["offset"] = 100
    elif mutation == "page_cardinality":
        pagination["page_count"] = 2
    elif mutation == "page_hash":
        pages[0]["body_sha256"] = "4" * 64
    elif mutation == "wallet_identity":
        after_value["wallet_identity_sha256"] = "5" * 64
    else:
        after_value["team_identity_sha256"] = "6" * 64
    with pytest.raises(ValueError):
        wallet_owner.replay_wallet_reconciliation(before.evidence, mutated)


def test_wallet_owner_has_no_provisioning_or_availability_authority() -> None:
    source = Path(wallet_owner.__file__).read_text(encoding="utf-8")
    for forbidden in ("direct_create", "terminate", "availability", "provision"):
        assert forbidden not in source
