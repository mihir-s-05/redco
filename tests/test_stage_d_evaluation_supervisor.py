from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from redco.analysis.stage_d_evaluation_contracts import EvaluationScheduleUnit
from redco.analysis.stage_d_evaluation_state import (
    EvaluationCallState,
    EvaluationLedgerSnapshot,
    EvaluationProcessEpoch,
    EvaluationServerEpoch,
    EvaluationTaskState,
)
from redco.analysis.stage_d_evaluation_supervisor import (
    EvaluationBlockCode,
    EvaluationSupervisorAction,
    next_evaluation_supervisor_action,
)


def _sha(value: str) -> str:
    return value * 64


_SCHEDULE = tuple(
    EvaluationScheduleUnit(index, arm, 0, f"task-{arm}", 9101 + index)
    for index, arm in enumerate(("stock", "branch-global", "local"))
)
_MANIFEST = SimpleNamespace(manifest_sha256=_sha("a"), schedule=_SCHEDULE)


def _snapshot(**changes: object) -> EvaluationLedgerSnapshot:
    base = EvaluationLedgerSnapshot(
        authorization_sha256=_sha("b"),
        execution_manifest_sha256=_MANIFEST.manifest_sha256,
        evaluation_plan_sha256=_sha("c"),
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
        head_sha256=_sha("d"),
        record_count=1,
    )
    return replace(base, **changes)


def _next(snapshot: EvaluationLedgerSnapshot, **liveness: str):
    return next_evaluation_supervisor_action(
        manifest=_MANIFEST,  # type: ignore[arg-type]
        snapshot=snapshot,
        process_liveness=liveness,  # type: ignore[arg-type]
    )


def test_supervisor_orders_server_attestation_before_client() -> None:
    assert _next(_snapshot()).kind == "reserve-server"
    reserved = EvaluationServerEpoch("stock", 0, _sha("1"))
    action = _next(_snapshot(server_epochs=(reserved,)))
    assert action.kind == "spawn-server"
    assert action.server_launch is not None
    assert (action.server_launch.arm, action.server_launch.epoch) == ("stock", 0)
    claimed = replace(reserved, process_receipt_sha256=_sha("2"))
    action = _next(_snapshot(server_epochs=(claimed,)), **{_sha("2"): "live"})
    assert action.kind == "attest-server"
    attested = replace(claimed, server_attestation_sha256=_sha("3"))
    action = _next(
        _snapshot(server_epochs=(attested,)),
        **{_sha("2"): "live"},
    )
    assert action.kind == "reserve-client"
    reserved_client = EvaluationProcessEpoch("stock", 0, _sha("5"), None)
    action = _next(
        _snapshot(server_epochs=(attested,), client_epochs=(reserved_client,)),
        **{_sha("2"): "live"},
    )
    assert action.kind == "spawn-client"
    assert action.client_launch is not None
    client = EvaluationProcessEpoch("stock", 0, _sha("5"), None, _sha("4"))
    action = _next(
        _snapshot(server_epochs=(attested,), client_epochs=(client,)),
        **{_sha("2"): "live", _sha("4"): "live"},
    )
    assert action.kind == "wait-client"


def test_supervisor_completes_and_cleans_each_contiguous_arm() -> None:
    task = EvaluationTaskState(_SCHEDULE[0], _sha("5"), 0, _sha("3"), ())
    task = replace(task, terminal_result_sha256=_sha("6"), task_metrics_sha256=_sha("7"))
    assert _next(_snapshot(tasks=(task,))).kind == "complete-arm"

    server = EvaluationServerEpoch("stock", 0, _sha("1"), _sha("2"), _sha("3"))
    client = EvaluationProcessEpoch("stock", 0, _sha("5"), None, _sha("4"))
    completed = _snapshot(
        server_epochs=(server,),
        client_epochs=(client,),
        tasks=(task,),
        arm_completions=(("stock", _sha("8")),),
        arm_metrics=(("stock", _sha("9")),),
    )
    assert (
        _next(
            completed,
            **{_sha("2"): "live", _sha("4"): "live"},
        ).kind
        == "wait-client"
    )
    assert (
        _next(
            completed,
            **{_sha("2"): "live", _sha("4"): "dead"},
        ).kind
        == "stop-server"
    )
    assert (
        _next(
            completed,
            **{_sha("2"): "dead", _sha("4"): "dead"},
        ).kind
        == "reserve-server"
    )


def test_supervisor_forbids_server_replacement_after_dispatch() -> None:
    server = EvaluationServerEpoch("stock", 0, _sha("1"), _sha("2"), _sha("3"))
    task = EvaluationTaskState(
        _SCHEDULE[0],
        _sha("5"),
        0,
        _sha("3"),
        (
            EvaluationCallState(
                _sha("6"),
                _sha("5"),
                0,
                _sha("7"),
                9201,
                "cache",
                _sha("8"),
                _sha("9"),
                dispatch_receipt_sha256=_sha("e"),
                response_envelope_sha256=_sha("f"),
                raw_response_sha256=_sha("0"),
                outcome_sha256=_sha("1"),
            ),
        ),
    )
    action = _next(
        _snapshot(server_epochs=(server,), tasks=(task,)),
        **{_sha("2"): "dead"},
    )
    assert action.kind == "blocked"
    assert action.block_code == EvaluationBlockCode.SERVER_DIED_AFTER_DISPATCH


def test_supervisor_cleans_contained_orphans_before_replacement() -> None:
    server = EvaluationServerEpoch("stock", 0, _sha("1"), _sha("2"))
    action = _next(
        _snapshot(server_epochs=(server,)),
        **{_sha("2"): "contained-orphan"},
    )
    assert action.kind == "cleanup-process"
    assert action.process_receipt_sha256 == _sha("2")

    attested = replace(server, server_attestation_sha256=_sha("3"))
    client = EvaluationProcessEpoch("stock", 0, _sha("5"), None, _sha("4"))
    action = _next(
        _snapshot(server_epochs=(attested,), client_epochs=(client,)),
        **{_sha("2"): "live", _sha("4"): "contained-orphan"},
    )
    assert action.kind == "cleanup-process"
    assert action.process_receipt_sha256 == _sha("4")


def test_supervisor_seals_exact_three_arm_completion() -> None:
    completed = tuple(
        (arm, _sha(str(index + 1))) for index, arm in enumerate(("stock", "branch-global", "local"))
    )
    assert _next(_snapshot(arm_completions=completed)).kind == "seal"
    assert _next(_snapshot(sealed=True, terminal_status="sealed")).kind == "done"


def test_supervisor_stops_on_ambiguous_dispatch() -> None:
    action = _next(_snapshot(terminal_status="ambiguous-dispatch"))
    assert action.kind == "blocked"
    assert action.block_code == EvaluationBlockCode.AMBIGUOUS_DISPATCH


def test_supervisor_action_rejects_impossible_optional_field_combinations() -> None:
    with pytest.raises(ValueError, match="fields differ"):
        EvaluationSupervisorAction("spawn-server", arm="stock")
    with pytest.raises(ValueError, match="malformed"):
        EvaluationSupervisorAction("blocked", arm="stock")
