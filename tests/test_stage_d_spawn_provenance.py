from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from redco.analysis.stage_d_spawn_provenance import (
    CausalProvenanceGraph,
    CouplingMode,
    EventCompletionSnapshot,
    EventSeedScheduler,
    PolicyEventAddress,
    SpawnLedger,
    SpawnScope,
    derive_child_lineage,
)
from redco.integrations.verifiers_trace_v2 import (
    extract_v2_rlm_provenance,
    validate_v2_spawn_ticket_ledger,
)


def _scope(*, parent_turn: int = 3, parent_lineage: str = "root") -> SpawnScope:
    return SpawnScope(
        depth=1,
        parent_lineage=parent_lineage,
        parent_call_ordinal=2,
        parent_tool_call_slot=0,
        parent_turn=parent_turn,
    )


def _event(*, lineage: str = "root/child", call: int = 0) -> PolicyEventAddress:
    return PolicyEventAddress(
        depth=1,
        lineage=lineage,
        session_call_ordinal=call,
        turn=call,
    )


def _scheduler(
    *,
    rollout: str = "trace-1",
    target: str = "target-1",
    replicate: int = 1,
) -> EventSeedScheduler:
    return EventSeedScheduler(
        master_seed="master",
        rollout_id=rollout,
        target_id=target,
        continuation_replicate=replicate,
    )


def _completion_snapshots(
    events: tuple[PolicyEventAddress, ...],
    *,
    completed: dict[PolicyEventAddress, tuple[int, ...]] | None = None,
) -> tuple[EventCompletionSnapshot, ...]:
    completed = completed or {}
    return tuple(EventCompletionSnapshot(event, completed.get(event, ())) for event in events)


def _root_provenance(*, call: int = 0) -> dict[str, object]:
    return {
        "provenance_version": 2,
        "depth": 0,
        "session_id": "generated-root",
        "turn": call,
        "call_kind": "policy",
        "lineage": "root",
        "session_call_ordinal": call,
        "completed_episode_spawn_ordinals": [],
    }


def _trace(
    calls: list[dict[str, object]],
    *,
    trace_id: object = "trace",
) -> dict[str, object]:
    node_indexes = [
        cast(int, call["node"]) for call in calls if type(call.get("node")) is int
    ]
    node_count = max(node_indexes, default=-1) + 1
    return {
        "id": trace_id,
        "nodes": [{} for _ in range(node_count)],
        "calls": calls,
    }


def test_spawn_ordinals_ignore_request_and_completion_order() -> None:
    ledger = SpawnLedger()
    first = ledger.reserve(_scope())
    second = ledger.reserve(_scope())

    ledger.complete(second)
    ledger.complete(first)

    assert (first.episode_spawn_ordinal, second.episode_spawn_ordinal) == (0, 1)
    assert (first.spawn_ordinal, second.spawn_ordinal) == (0, 1)
    assert first.completed_predecessor_spawn_ordinals == ()
    assert second.completed_predecessor_spawn_ordinals == ()


def test_two_child_local_turn_zero_addresses_never_collide() -> None:
    ledger = SpawnLedger()
    first = ledger.reserve(_scope())
    second = ledger.reserve(_scope())

    assert _event(lineage=first.lineage) != _event(lineage=second.lineage)


def test_sequential_and_concurrent_same_turn_siblings_have_distinct_dependencies() -> None:
    concurrent = SpawnLedger()
    concurrent.reserve(_scope())
    concurrent_second = concurrent.reserve(_scope())
    assert concurrent_second.completed_predecessor_spawn_ordinals == ()

    sequential = SpawnLedger()
    sequential_first = sequential.reserve(_scope())
    sequential.complete(sequential_first)
    sequential_second = sequential.reserve(_scope())
    assert sequential_second.completed_predecessor_spawn_ordinals == (
        sequential_first.spawn_ordinal,
    )


def test_later_target_child_turn_and_nested_lineages_are_distinct() -> None:
    reservation = SpawnLedger().reserve(_scope())
    target = _event(lineage=reservation.lineage, call=0)
    later = _event(lineage=reservation.lineage, call=1)
    nested_a = derive_child_lineage(
        SpawnScope(2, reservation.lineage, 1, 0, 1),
        spawn_ordinal=0,
    )
    nested_b = derive_child_lineage(
        SpawnScope(2, "root/other-parent", 1, 0, 1),
        spawn_ordinal=0,
    )

    assert later.session_call_ordinal > target.session_call_ordinal
    assert nested_a != nested_b


def test_causal_graph_separates_gather_siblings_and_propagates_descendants() -> None:
    ledger = SpawnLedger()
    scope = _scope()
    target_spawn = ledger.reserve(scope)
    sibling_spawn = ledger.reserve(scope)
    target_nested = ledger.reserve(SpawnScope(2, target_spawn.lineage, 1, 0, 1))
    sibling_nested = ledger.reserve(SpawnScope(2, sibling_spawn.lineage, 1, 0, 1))
    root_0 = _event(lineage="root", call=2)
    root_1 = _event(lineage="root", call=3)
    target_0 = _event(lineage=target_spawn.lineage, call=0)
    target_1 = _event(lineage=target_spawn.lineage, call=1)
    target_grandchild = PolicyEventAddress(2, target_nested.lineage, 0, 0)
    sibling_0 = _event(lineage=sibling_spawn.lineage, call=0)
    sibling_1 = _event(lineage=sibling_spawn.lineage, call=1)
    sibling_grandchild = PolicyEventAddress(2, sibling_nested.lineage, 0, 0)
    events = (
        root_0,
        root_1,
        target_0,
        target_1,
        target_grandchild,
        sibling_0,
        sibling_1,
        sibling_grandchild,
    )
    graph = CausalProvenanceGraph(
        events=events,
        spawns=(target_spawn, sibling_spawn, target_nested, sibling_nested),
        completion_snapshots=_completion_snapshots(
            events,
            completed={root_1: (0, 1, 2, 3)},
        ),
    )

    assert graph.is_downstream(target_1, target=target_0)
    assert graph.is_downstream(target_grandchild, target=target_0)
    assert graph.is_downstream(root_1, target=target_0)
    assert not graph.is_downstream(sibling_0, target=target_0)
    assert not graph.is_downstream(sibling_1, target=target_0)
    assert not graph.is_downstream(sibling_grandchild, target=target_0)


def test_causal_graph_marks_sequential_same_scope_sibling_downstream() -> None:
    ledger = SpawnLedger()
    scope = _scope()
    first = ledger.reserve(scope)
    ledger.complete(first)
    second = ledger.reserve(scope)
    root = _event(lineage="root", call=2)
    first_event = _event(lineage=first.lineage)
    second_event = _event(lineage=second.lineage)
    events = (root, first_event, second_event)
    graph = CausalProvenanceGraph(
        events=events,
        spawns=(first, second),
        completion_snapshots=_completion_snapshots(events),
    )

    assert graph.is_downstream(second_event, target=first_event)


def test_causal_graph_waits_for_observed_child_completion_before_return_edge() -> None:
    ledger = SpawnLedger()
    spawn = ledger.reserve(_scope())
    root_parent = _event(lineage="root", call=2)
    root_overlapping = _event(lineage="root", call=3)
    root_after_join = _event(lineage="root", call=4)
    child = _event(lineage=spawn.lineage)
    events = (root_parent, root_overlapping, root_after_join, child)
    graph = CausalProvenanceGraph(
        events=events,
        spawns=(spawn,),
        completion_snapshots=_completion_snapshots(
            events,
            completed={root_after_join: (spawn.episode_spawn_ordinal,)},
        ),
    )

    assert not graph.is_downstream(root_overlapping, target=child)
    assert graph.is_downstream(root_after_join, target=child)


def test_lineage_excludes_diagnostic_turn_and_generated_transport_ids() -> None:
    expected = derive_child_lineage(_scope(parent_turn=3), spawn_ordinal=0)
    transport_a = ("session-a", "tool-call-a", "invocation-a")
    transport_b = ("session-b", "tool-call-b", "invocation-b")

    assert transport_a != transport_b
    assert derive_child_lineage(_scope(parent_turn=99), spawn_ordinal=0) == expected

    ledger = SpawnLedger()
    first = ledger.reserve(_scope(parent_turn=3))
    second = ledger.reserve(_scope(parent_turn=99))
    assert (first.spawn_ordinal, second.spawn_ordinal) == (0, 1)
    assert first.lineage != second.lineage


def test_paired_seed_is_shared_across_action_arms() -> None:
    address = _event(call=2)
    scheduler = _scheduler()

    seeds = {
        scheduler.paired_continuation_seed(
            address,
            committed_address=address,
        ).seed
        for _action_arm in ("recorded", "alternative-1", "alternative-2", "alternative-3")
    }

    assert len(seeds) == 1


def test_trace_target_replicate_and_action_namespaces_are_distinct() -> None:
    address = _event()
    base = _scheduler()
    base_seed = base.paired_continuation_seed(address, committed_address=address).seed

    alternatives = {
        _scheduler(rollout="trace-2")
        .paired_continuation_seed(address, committed_address=address)
        .seed,
        _scheduler(target="target-2")
        .paired_continuation_seed(address, committed_address=address)
        .seed,
        _scheduler(replicate=2).paired_continuation_seed(address, committed_address=address).seed,
    }
    assert base_seed not in alternatives
    assert len({base.action_seed(action_slot=index) for index in range(4)}) == 4
    assert base_seed not in {base.action_seed(action_slot=index) for index in range(4)}
    assert base.action_seed(action_slot=0) == _scheduler(replicate=2).action_seed(action_slot=0)


def test_dynamic_topology_is_exogenous_and_collision_free() -> None:
    scheduler = _scheduler()
    inserted = PolicyEventAddress(2, "root/inserted", 0, 0)
    shifted = _event(lineage="root/existing", call=2)

    inserted_a = scheduler.exogenous_continuation_seed(inserted, action_arm="alternative-a")
    inserted_b = scheduler.exogenous_continuation_seed(inserted, action_arm="alternative-b")
    shifted_a = scheduler.exogenous_continuation_seed(shifted, action_arm="alternative-a")

    assert inserted_a.coupling_mode is CouplingMode.EXOGENOUS
    assert len({inserted_a.seed, inserted_b.seed, shifted_a.seed}) == 3
    with pytest.raises(ValueError, match="match the committed"):
        scheduler.paired_continuation_seed(inserted, committed_address=shifted)


def test_invalid_spawn_and_seed_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive"):
        SpawnScope(0, "root", 0, 0, 0)
    with pytest.raises(ValueError, match="nonempty"):
        SpawnScope(1, "", 0, 0, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        derive_child_lineage(_scope(), spawn_ordinal=-1)
    with pytest.raises(ValueError, match="nonempty"):
        _scheduler().exogenous_continuation_seed(_event(), action_arm="")


def test_v2_trace_provenance_round_trips_strict_recursive_fields() -> None:
    scope = SpawnScope(1, "root", 4, 0, 4)
    payload = {
        "provenance_version": 2,
        "depth": 1,
        "session_id": "generated-child",
        "turn": 0,
        "call_kind": "policy",
        "lineage": derive_child_lineage(scope, spawn_ordinal=2),
        "session_call_ordinal": 0,
        "parent_session_id": "generated-root",
        "parent_turn": 4,
        "parent_tool_call_id": "generated-tool",
        "invocation_id": "diagnostic-label",
        "parent_lineage": "root",
        "parent_call_ordinal": 4,
        "parent_tool_call_slot": 0,
        "spawn_ordinal": 2,
        "episode_spawn_ordinal": 2,
        "completed_predecessor_spawn_ordinals": [0, 1],
        "completed_episode_spawn_ordinals": [0, 1],
    }

    predecessor_calls: list[dict[str, object]] = []
    for spawn_ordinal in (0, 1):
        predecessor = dict(payload)
        predecessor["lineage"] = derive_child_lineage(scope, spawn_ordinal=spawn_ordinal)
        predecessor["spawn_ordinal"] = spawn_ordinal
        predecessor["episode_spawn_ordinal"] = spawn_ordinal
        predecessor["completed_predecessor_spawn_ordinals"] = list(range(spawn_ordinal))
        predecessor["completed_episode_spawn_ordinals"] = list(range(spawn_ordinal))
        predecessor_calls.append({"node": spawn_ordinal + 5, "error": None, "rlm": predecessor})
    root_calls: list[dict[str, object]] = [
        {
            "node": call,
            "error": None,
            "rlm": _root_provenance(call=call),
        }
        for call in range(5)
    ]
    parsed = extract_v2_rlm_provenance(
        _trace(
            [
                *root_calls,
                *predecessor_calls,
                {"node": 7, "error": None, "rlm": payload},
            ]
        )
    )[-1]

    assert parsed.lineage == derive_child_lineage(scope, spawn_ordinal=2)
    assert parsed.spawn_ordinal == 2
    assert parsed.completed_predecessor_spawn_ordinals == (0, 1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lineage", None),
        ("lineage", "root/not-the-structural-lineage"),
        ("session_call_ordinal", True),
        ("completed_predecessor_spawn_ordinals", [0, 0]),
        ("completed_predecessor_spawn_ordinals", [0, 2]),
    ],
)
def test_v2_trace_provenance_rejects_partial_or_malformed_fields(
    field: str,
    value: object,
) -> None:
    scope = SpawnScope(1, "root", 4, 0, 4)
    payload = {
        "provenance_version": 2,
        "depth": 1,
        "session_id": "generated-child",
        "turn": 0,
        "call_kind": "policy",
        "lineage": derive_child_lineage(scope, spawn_ordinal=2),
        "session_call_ordinal": 0,
        "parent_session_id": "generated-root",
        "parent_turn": 4,
        "parent_tool_call_id": "generated-tool",
        "invocation_id": "diagnostic-label",
        "parent_lineage": "root",
        "parent_call_ordinal": 4,
        "parent_tool_call_slot": 0,
        "spawn_ordinal": 2,
        "episode_spawn_ordinal": 8,
        "completed_predecessor_spawn_ordinals": [0, 1],
        "completed_episode_spawn_ordinals": [],
    }
    payload[field] = value

    with pytest.raises(ValueError, match="RLM v2"):
        extract_v2_rlm_provenance(_trace([{"node": 0, "rlm": payload}]))


@pytest.mark.parametrize("trace_id", [None, "", "bad\ntrace"])
def test_v2_trace_provenance_rejects_missing_or_malformed_trace_id(
    trace_id: object,
) -> None:
    with pytest.raises(ValueError, match="trace id"):
        extract_v2_rlm_provenance(_trace([], trace_id=trace_id))


def test_v2_trace_provenance_rejects_failed_unlinked_or_duplicate_calls() -> None:
    root = _root_provenance()
    with pytest.raises(ValueError, match="committed node"):
        extract_v2_rlm_provenance(_trace([{"node": None, "rlm": root}]))
    with pytest.raises(ValueError, match="out of bounds"):
        extract_v2_rlm_provenance({"id": "trace", "nodes": [], "calls": [{"node": 0, "rlm": root}]})
    with pytest.raises(ValueError, match="failed model exchange"):
        extract_v2_rlm_provenance(
            _trace([{"node": 0, "error": {"type": "transport"}, "rlm": root}])
        )
    with pytest.raises(ValueError, match="repeats a scientific event"):
        extract_v2_rlm_provenance(
            _trace(
                [
                    {"node": 0, "rlm": root},
                    {"node": 1, "rlm": root},
                ]
            )
        )


def test_v2_trace_provenance_rejects_empty_trace() -> None:
    with pytest.raises(ValueError, match="no committed"):
        extract_v2_rlm_provenance(_trace([]))


def test_v2_trace_provenance_requires_contiguous_calls_and_causal_parent() -> None:
    with pytest.raises(ValueError, match="call ordinals"):
        extract_v2_rlm_provenance(
            _trace(
                [
                    {"node": 0, "rlm": _root_provenance(call=0)},
                    {"node": 1, "rlm": _root_provenance(call=2)},
                ]
            )
        )

    scope = SpawnScope(1, "root", 1, 0, 0)
    child = {
        "provenance_version": 2,
        "depth": 1,
        "session_id": "child",
        "turn": 0,
        "call_kind": "policy",
        "lineage": derive_child_lineage(scope, spawn_ordinal=0),
        "session_call_ordinal": 0,
        "parent_session_id": "root-session",
        "parent_turn": 0,
        "parent_tool_call_id": "tool",
        "invocation_id": None,
        "parent_lineage": "root",
        "parent_call_ordinal": 1,
        "parent_tool_call_slot": 0,
        "spawn_ordinal": 0,
        "episode_spawn_ordinal": 0,
        "completed_predecessor_spawn_ordinals": [],
        "completed_episode_spawn_ordinals": [],
    }
    with pytest.raises(ValueError, match="causal parent"):
        extract_v2_rlm_provenance(
            _trace(
                [
                    {"node": 0, "rlm": _root_provenance()},
                    {"node": 1, "rlm": child},
                ]
            )
        )


def test_v2_spawn_ticket_ledger_detects_trailing_orphan(
    tmp_path: Path,
) -> None:
    records = extract_v2_rlm_provenance(_trace([{"node": 0, "rlm": _root_provenance()}]))
    ledger = tmp_path / ".redco-spawn-ordinals"
    ledger.mkdir()
    (ledger / "000000000000.reserved").write_text("reserved\n", encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        validate_v2_spawn_ticket_ledger(records, ledger)


def test_v2_spawn_ticket_ledger_matches_committed_children(
    tmp_path: Path,
) -> None:
    root = _root_provenance()
    scope = SpawnScope(1, "root", 0, 0, 0)
    child = {
        "provenance_version": 2,
        "depth": 1,
        "session_id": "child",
        "turn": 0,
        "call_kind": "policy",
        "lineage": derive_child_lineage(scope, spawn_ordinal=0),
        "session_call_ordinal": 0,
        "parent_session_id": "root-session",
        "parent_turn": 0,
        "parent_tool_call_id": "tool",
        "invocation_id": None,
        "parent_lineage": "root",
        "parent_call_ordinal": 0,
        "parent_tool_call_slot": 0,
        "spawn_ordinal": 0,
        "episode_spawn_ordinal": 0,
        "completed_predecessor_spawn_ordinals": [],
        "completed_episode_spawn_ordinals": [],
    }
    records = extract_v2_rlm_provenance(
        _trace(
            [
                {"node": 0, "rlm": root},
                {"node": 1, "rlm": child},
            ]
        )
    )
    ledger = tmp_path / ".redco-spawn-ordinals"
    ledger.mkdir()
    (ledger / "000000000000.reserved").write_text("reserved\n", encoding="utf-8")
    (ledger / "000000000000.terminal").write_text("terminal\n", encoding="utf-8")
    validate_v2_spawn_ticket_ledger(records, ledger)
