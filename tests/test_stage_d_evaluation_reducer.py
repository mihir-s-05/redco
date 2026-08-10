from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from redco.analysis.stage_d_evaluation_contracts import (
    EvaluationScheduleUnit,
    StageDEvaluationExecutionManifest,
)
from redco.analysis.stage_d_evaluation_reducer import reduce_evaluation_records
from redco.analysis.stage_d_evaluation_state import EvaluationLedgerSnapshot

_AUTHORIZATION = "a" * 64
_MANIFEST = "b" * 64
_PLAN = "c" * 64
_DIGESTS = tuple(f"{index:x}" * 64 for index in range(8))
_MISSING = object()
_EXPECTED_AUTH_ERROR = "expected evaluation authorization_sha256 must be a lowercase SHA-256"
_EVENT_AUTH_ERROR = "evaluation ledger authorization_sha256 is invalid"
_AUTH_MISMATCH_ERROR = "evaluation ledger authorization differs from authenticated input"
_EVENT_SCHEMA_ERROR = "evaluation ledger event schema differs"
_GENESIS_MANIFEST_ERROR = "evaluation ledger genesis differs from execution manifest"
_MANIFEST_SHA_ERROR = "evaluation ledger execution_manifest_sha256 is invalid"


class _ForgedDigest(str):
    def __eq__(self, _other: object) -> bool:
        return True

    def __ne__(self, _other: object) -> bool:
        return False


class _FlipOnRead(dict[str, Any]):
    def __init__(self, value: dict[str, Any], *, field: str, replacement: object) -> None:
        super().__init__(value)
        self._field = field
        self._replacement = replacement

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        if key == self._field:
            super().__setitem__(key, self._replacement)
        return value


@dataclass(frozen=True)
class _ManifestStub:
    manifest_sha256: str = _MANIFEST
    evaluation_plan_sha256: str = _PLAN
    max_server_launches_per_arm: int = 2
    max_client_launches_per_arm: int = 2
    schedule: tuple[EvaluationScheduleUnit, ...] = ()


def _manifest() -> StageDEvaluationExecutionManifest:
    return cast(StageDEvaluationExecutionManifest, _ManifestStub())


def _record(
    offset: int,
    prior: str | None,
    kind: str,
    event: dict[str, Any],
) -> dict[str, Any]:
    return {
        "offset": offset,
        "prior_record_sha256": prior,
        "record_kind": kind,
        "event": event,
    }


def _genesis() -> dict[str, Any]:
    return _record(
        0,
        None,
        "genesis",
        {
            "authorization_sha256": _AUTHORIZATION,
            "execution_manifest_sha256": _MANIFEST,
            "evaluation_plan_sha256": _PLAN,
            "created_at_unix_ns": 1,
        },
    )


def _server_launch() -> dict[str, Any]:
    return _record(
        1,
        _DIGESTS[0],
        "server_launch_reserved",
        {
            "arm": "stock",
            "epoch": 0,
            "prior_process_receipt_sha256": None,
            "dead_process_evidence_sha256": None,
        },
    )


def _reduce(
    records: list[dict[str, Any]],
    *,
    expected_authorization: object = _AUTHORIZATION,
) -> EvaluationLedgerSnapshot:
    return reduce_evaluation_records(
        records,
        _DIGESTS[: len(records)],
        manifest=_manifest(),
        expected_authorization_sha256=cast(str, expected_authorization),
    )


def test_authenticated_genesis_reduces_to_public_snapshot() -> None:
    assert _reduce([_genesis()]) == EvaluationLedgerSnapshot(
        authorization_sha256=_AUTHORIZATION,
        execution_manifest_sha256=_MANIFEST,
        evaluation_plan_sha256=_PLAN,
        created_at_unix_ns=1,
        server_claims=(),
        server_attestations=(),
        server_epochs=(),
        client_epochs=(),
        actuation_attempts=(),
        tasks=(),
        arm_completions=(),
        arm_metrics=(),
        sealed=False,
        terminal_status="active",
        head_sha256=_DIGESTS[0],
        record_count=1,
    )


@pytest.mark.parametrize(
    ("trusted", "error_type", "expected_error"),
    [
        (_MISSING, TypeError, "expected_authorization_sha256"),
        (None, ValueError, _EXPECTED_AUTH_ERROR),
        ("A" * 64, ValueError, _EXPECTED_AUTH_ERROR),
        ("d" * 64, ValueError, _AUTH_MISMATCH_ERROR),
        (_ForgedDigest("d" * 64), ValueError, _EXPECTED_AUTH_ERROR),
    ],
)
def test_reducer_requires_authenticated_authorization_input(
    trusted: object,
    error_type: type[Exception],
    expected_error: str,
) -> None:
    kwargs: dict[str, Any] = {"manifest": _manifest()}
    if trusted is not _MISSING:
        kwargs["expected_authorization_sha256"] = trusted

    with pytest.raises(error_type, match=expected_error):
        cast(Any, reduce_evaluation_records)(
            [_genesis()],
            [_DIGESTS[0]],
            **kwargs,
        )


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("authorization_sha256", _MISSING, _EVENT_SCHEMA_ERROR),
        ("authorization_sha256", None, _EVENT_AUTH_ERROR),
        ("authorization_sha256", "A" * 64, _EVENT_AUTH_ERROR),
        ("authorization_sha256", "d" * 64, _AUTH_MISMATCH_ERROR),
        ("authorization_sha256", _ForgedDigest("d" * 64), _EVENT_AUTH_ERROR),
        ("execution_manifest_sha256", _MISSING, _EVENT_SCHEMA_ERROR),
        ("execution_manifest_sha256", None, _GENESIS_MANIFEST_ERROR),
        ("execution_manifest_sha256", "B" * 64, _MANIFEST_SHA_ERROR),
        ("execution_manifest_sha256", "d" * 64, _GENESIS_MANIFEST_ERROR),
    ],
)
def test_genesis_trust_bindings_fail_closed(
    field: str,
    value: object,
    expected_error: str,
) -> None:
    record = _genesis()
    if value is _MISSING:
        record["event"].pop(field)
    else:
        record["event"][field] = value

    with pytest.raises(ValueError, match=f"^{expected_error}$"):
        _reduce([record])


def test_authenticated_replay_reduces_core_transition_exactly() -> None:
    records = [_genesis(), _server_launch()]
    snapshot = _reduce(records)

    assert snapshot == _reduce(records)
    assert snapshot.record_count == 2
    assert snapshot.head_sha256 == _DIGESTS[1]
    assert snapshot.server_epochs[0].arm == "stock"
    assert snapshot.server_epochs[0].epoch == 0
    assert snapshot.server_epochs[0].launch_record_sha256 == _DIGESTS[1]


@pytest.mark.parametrize(
    ("kind", "field", "replacement"),
    [
        ("genesis", "authorization_sha256", None),
        ("server_launch_reserved", "epoch", 1),
    ],
)
def test_reducer_rejects_stateful_event_mappings_before_any_transition(
    kind: str,
    field: str,
    replacement: object,
) -> None:
    records = [_genesis()]
    if kind == "server_launch_reserved":
        records.append(_server_launch())
    target = records[-1]
    target["event"] = _FlipOnRead(target["event"], field=field, replacement=replacement)

    with pytest.raises(ValueError, match=r"^evaluation ledger record representation differs$"):
        _reduce(records)


@pytest.mark.parametrize(
    "field",
    ["prior_process_receipt_sha256", "dead_process_evidence_sha256"],
)
def test_nullable_sha_fields_reject_noncanonical_values(field: str) -> None:
    launch = _server_launch()
    launch["event"][field] = "A" * 64

    with pytest.raises(ValueError, match=rf"^evaluation ledger {field} is invalid$"):
        _reduce([_genesis(), launch])


def test_post_seal_record_is_rejected() -> None:
    arms = ("stock", "branch-global", "local")
    records = [_genesis()]
    records.extend(
        _record(
            offset,
            _DIGESTS[offset - 1],
            "arm_completed",
            {
                "arm": arm,
                "arm_metrics_sha256": "d" * 64,
                "task_attempt_ids": [],
            },
        )
        for offset, arm in enumerate(arms, start=1)
    )
    records.append(
        _record(
            4,
            _DIGESTS[3],
            "seal",
            {
                "arm_completion_sha256s": [
                    [arm, _DIGESTS[offset]] for offset, arm in enumerate(arms, start=1)
                ]
            },
        )
    )

    sealed = _reduce(records)
    assert sealed.sealed
    assert sealed.terminal_status == "sealed"
    assert sealed.arm_completions == tuple(zip(arms, _DIGESTS[1:4], strict=True))

    records.append(
        _record(
            5,
            _DIGESTS[4],
            "arm_completed",
            {
                "arm": "stock",
                "arm_metrics_sha256": "e" * 64,
                "task_attempt_ids": [],
            },
        )
    )
    with pytest.raises(ValueError, match=r"^evaluation ledger appends after its seal$"):
        _reduce(records)
