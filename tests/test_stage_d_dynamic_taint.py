from __future__ import annotations

import itertools

import pytest

from redco.analysis.stage_d_dynamic_taint import (
    DynamicCausalTaintTracker,
    ReplayDisposition,
)
from redco.analysis.stage_d_spawn_provenance import (
    CausalProvenanceGraph,
    EventCompletionSnapshot,
    PolicyEventAddress,
    SpawnReservation,
    SpawnScope,
    derive_child_lineage,
)
from redco.integrations.verifiers_trace_v2 import RecordedRLMProvenanceV2

type SourceFixture = tuple[
    PolicyEventAddress,
    PolicyEventAddress,
    tuple[PolicyEventAddress, ...],
    tuple[RecordedRLMProvenanceV2, ...],
    CausalProvenanceGraph,
]


def _record(
    address: PolicyEventAddress,
    *,
    completed: tuple[int, ...] = (),
    parent: PolicyEventAddress | None = None,
    spawn_ordinal: int | None = None,
    episode_spawn_ordinal: int | None = None,
) -> RecordedRLMProvenanceV2:
    child = parent is not None
    parent_turn = parent.turn if parent is not None else None
    parent_lineage = parent.lineage if parent is not None else None
    parent_call_ordinal = (
        parent.session_call_ordinal if parent is not None else None
    )
    return RecordedRLMProvenanceV2(
        trace_id="trace",
        call_index=0,
        node_index=0,
        depth=address.depth,
        session_id=f"session-{address.lineage}-{address.session_call_ordinal}",
        turn=address.turn,
        call_kind=address.call_kind,
        lineage=address.lineage,
        session_call_ordinal=address.session_call_ordinal,
        parent_session_id="parent-session" if child else None,
        parent_turn=parent_turn,
        parent_tool_call_id="tool-call" if child else None,
        invocation_id="invocation" if child else None,
        parent_lineage=parent_lineage,
        parent_call_ordinal=parent_call_ordinal,
        parent_tool_call_slot=0 if child else None,
        spawn_ordinal=spawn_ordinal,
        episode_spawn_ordinal=episode_spawn_ordinal,
        completed_predecessor_spawn_ordinals=() if child else None,
        completed_episode_spawn_ordinals=completed,
    )


def _source_fixture() -> SourceFixture:
    root_0 = PolicyEventAddress(0, "root", 0, 0)
    root_1 = PolicyEventAddress(0, "root", 1, 1)
    scope = SpawnScope(1, "root", 0, 0, 0)
    spawns = tuple(
        SpawnReservation(
            scope,
            slot,
            slot,
            (),
            derive_child_lineage(scope, spawn_ordinal=slot),
        )
        for slot in range(4)
    )
    children = tuple(
        PolicyEventAddress(1, spawn.lineage, 0, 0) for spawn in spawns
    )
    records = (
        _record(root_0),
        *(
            _record(
                child,
                parent=root_0,
                spawn_ordinal=slot,
                episode_spawn_ordinal=slot,
            )
            for slot, child in enumerate(children)
        ),
        _record(root_1, completed=(0, 1, 2, 3)),
    )
    graph = CausalProvenanceGraph(
        events=(root_0, *children, root_1),
        spawns=spawns,
        completion_snapshots=(
            EventCompletionSnapshot(root_0, ()),
            *(EventCompletionSnapshot(child, ()) for child in children),
            EventCompletionSnapshot(root_1, (0, 1, 2, 3)),
        ),
    )
    return root_0, root_1, children, records, graph


def test_four_sibling_completion_permutations_never_taint_independent_siblings() -> None:
    root_0, root_1, children, records, graph = _source_fixture()
    by_address = {record.scientific_address: record for record in records}

    for order in itertools.permutations(children):
        tracker = DynamicCausalTaintTracker(
            target=children[0],
            source_records=records,
            source_graph=graph,
        )
        root_decision = tracker.observe(by_address[root_0])
        assert root_decision.disposition is ReplayDisposition.REUSE
        assert root_decision.counts_toward_logical_continuation is False
        decisions = {
            child: tracker.observe(by_address[child]) for child in order
        }
        assert decisions[children[0]].disposition is ReplayDisposition.INJECT
        assert (
            decisions[children[0]].counts_toward_logical_continuation is False
        )
        assert all(
            decisions[child].disposition is ReplayDisposition.REUSE
            for child in children[1:]
        )
        assert all(
            decisions[child].counts_toward_logical_continuation
            for child in children[1:]
        )
        assert tracker.observe(by_address[root_1]).disposition is ReplayDisposition.GENERATE
        tracker.finalize()


def test_dynamic_children_and_later_lineage_events_inherit_taint() -> None:
    root_0, root_1, children, records, graph = _source_fixture()
    by_address = {record.scientific_address: record for record in records}
    tracker = DynamicCausalTaintTracker(
        target=children[0],
        source_records=records,
        source_graph=graph,
    )
    tracker.observe(by_address[root_0])
    tracker.observe(by_address[children[0]])
    tracker.observe(by_address[root_1])

    dynamic_scope = SpawnScope(1, "root", 1, 0, 1)
    dynamic_address = PolicyEventAddress(
        1,
        derive_child_lineage(dynamic_scope, spawn_ordinal=4),
        0,
        0,
    )
    dynamic = _record(
        dynamic_address,
        parent=root_1,
        spawn_ordinal=4,
        episode_spawn_ordinal=4,
    )
    assert tracker.observe(dynamic).disposition is ReplayDisposition.GENERATE

    later = _record(
        PolicyEventAddress(1, dynamic_address.lineage, 1, 1),
        parent=root_1,
        spawn_ordinal=4,
        episode_spawn_ordinal=4,
    )
    assert tracker.observe(later).disposition is ReplayDisposition.GENERATE


def test_unexpected_untainted_topology_and_duplicate_addresses_fail_closed() -> None:
    root_0, _, children, records, graph = _source_fixture()
    by_address = {record.scientific_address: record for record in records}
    tracker = DynamicCausalTaintTracker(
        target=children[0],
        source_records=records,
        source_graph=graph,
    )
    tracker.observe(by_address[root_0])
    with pytest.raises(ValueError, match="not target-descended"):
        scope = SpawnScope(1, "root", 0, 0, 0)
        tracker.observe(
            _record(
                PolicyEventAddress(
                    1,
                    derive_child_lineage(scope, spawn_ordinal=9),
                    0,
                    0,
                ),
                parent=root_0,
                spawn_ordinal=9,
                episode_spawn_ordinal=9,
            )
        )

    with pytest.raises(ValueError, match="address twice"):
        tracker.observe(by_address[root_0])


def test_completed_target_does_not_taint_dynamic_independent_child() -> None:
    root_0, _, children, records, graph = _source_fixture()
    by_address = {record.scientific_address: record for record in records}
    tracker = DynamicCausalTaintTracker(
        target=children[0],
        source_records=records,
        source_graph=graph,
    )
    tracker.observe(by_address[root_0])
    tracker.observe(by_address[children[0]])
    dynamic = _record(
        PolicyEventAddress(2, f"{children[1].lineage}/dynamic", 0, 0),
        parent=children[1],
        spawn_ordinal=8,
        episode_spawn_ordinal=8,
        completed=(0,),
    )
    with pytest.raises(ValueError, match="not target-descended"):
        tracker.observe(dynamic)


def test_upstream_turn_in_target_child_is_not_rebilled_as_continuation() -> None:
    root = PolicyEventAddress(0, "root", 0, 0)
    scope = SpawnScope(1, "root", 0, 0, 0)
    spawn = SpawnReservation(scope, 0, 0, (), derive_child_lineage(scope, spawn_ordinal=0))
    child_0 = PolicyEventAddress(1, spawn.lineage, 0, 0)
    target = PolicyEventAddress(1, spawn.lineage, 1, 1)
    records = (
        _record(root),
        _record(child_0, parent=root, spawn_ordinal=0, episode_spawn_ordinal=0),
        _record(target, parent=root, spawn_ordinal=0, episode_spawn_ordinal=0),
    )
    graph = CausalProvenanceGraph(
        events=(root, child_0, target),
        spawns=(spawn,),
        completion_snapshots=tuple(
            EventCompletionSnapshot(address, ()) for address in (root, child_0, target)
        ),
    )
    tracker = DynamicCausalTaintTracker(
        target=target,
        source_records=records,
        source_graph=graph,
    )
    tracker.observe(records[0])
    upstream = tracker.observe(records[1])
    assert upstream.disposition is ReplayDisposition.REUSE
    assert upstream.counts_toward_logical_continuation is False
    assert tracker.observe(records[2]).disposition is ReplayDisposition.INJECT
    tracker.finalize()


@pytest.mark.parametrize("omit", ["target", "independent"])
def test_finalization_rejects_missing_required_source_events(omit: str) -> None:
    root_0, root_1, children, records, graph = _source_fixture()
    by_address = {record.scientific_address: record for record in records}
    tracker = DynamicCausalTaintTracker(
        target=children[0],
        source_records=records,
        source_graph=graph,
    )
    required = (root_0, *children)
    omitted = children[0] if omit == "target" else children[1]
    for address in required:
        if address != omitted:
            tracker.observe(by_address[address])
    tracker.observe(by_address[root_1])

    with pytest.raises(ValueError, match="omitted target or non-downstream"):
        tracker.finalize()


def test_terminal_truncation_allows_only_required_events_after_the_target() -> None:
    root_0, _, children, records, graph = _source_fixture()
    by_address = {record.scientific_address: record for record in records}
    tracker = DynamicCausalTaintTracker(
        target=children[0],
        source_records=records,
        source_graph=graph,
    )
    tracker.observe(by_address[root_0])
    tracker.observe(by_address[children[0]])
    tracker.finalize_terminal_truncation()

    missing_upstream = DynamicCausalTaintTracker(
        target=children[0],
        source_records=records,
        source_graph=graph,
    )
    missing_upstream.observe(by_address[children[0]])
    with pytest.raises(ValueError, match="required pre-target"):
        missing_upstream.finalize_terminal_truncation()


def test_terminal_truncation_is_invariant_to_parallel_completion_order() -> None:
    root_0, root_1, children, records, graph = _source_fixture()
    by_address = {record.scientific_address: record for record in records}
    reordered = (
        by_address[root_0],
        by_address[children[1]],
        by_address[children[0]],
        *(by_address[child] for child in children[2:]),
        by_address[root_1],
    )
    for source_records in (records, reordered):
        tracker = DynamicCausalTaintTracker(
            target=children[0],
            source_records=source_records,
            source_graph=graph,
        )
        tracker.observe(by_address[root_0])
        tracker.observe(by_address[children[0]])
        tracker.finalize_terminal_truncation()


def test_same_scientific_key_with_changed_turn_is_not_dynamic_topology() -> None:
    root_0, _, children, records, graph = _source_fixture()
    by_address = {record.scientific_address: record for record in records}
    tracker = DynamicCausalTaintTracker(
        target=children[0],
        source_records=records,
        source_graph=graph,
    )
    changed = PolicyEventAddress(
        root_0.depth,
        root_0.lineage,
        root_0.session_call_ordinal,
        root_0.turn + 1,
    )

    with pytest.raises(ValueError, match="structural parentage"):
        tracker.observe(_record(changed))

    with pytest.raises(ValueError, match="address twice"):
        tracker.observe(by_address[root_0])
