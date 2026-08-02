"""Strict reader for versioned RLM provenance in Verifiers model calls."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_spawn_provenance import (
    EventCompletionSnapshot,
    PolicyEventAddress,
    SpawnScope,
    derive_child_lineage,
)


@dataclass(frozen=True, slots=True)
class RecordedRLMProvenanceV2:
    """One fail-closed v2 provenance record, including transport diagnostics."""

    trace_id: str
    call_index: int
    node_index: int
    depth: int
    session_id: str
    turn: int
    call_kind: str
    lineage: str
    session_call_ordinal: int
    parent_session_id: str | None
    parent_turn: int | None
    parent_tool_call_id: str | None
    invocation_id: str | None
    parent_lineage: str | None
    parent_call_ordinal: int | None
    parent_tool_call_slot: int | None
    spawn_ordinal: int | None
    episode_spawn_ordinal: int | None
    completed_predecessor_spawn_ordinals: tuple[int, ...] | None
    completed_episode_spawn_ordinals: tuple[int, ...]

    @property
    def scientific_address(self) -> PolicyEventAddress:
        return PolicyEventAddress(
            depth=self.depth,
            lineage=self.lineage,
            session_call_ordinal=self.session_call_ordinal,
            turn=self.turn,
            call_kind=self.call_kind,
        )

    @property
    def completion_snapshot(self) -> EventCompletionSnapshot:
        return EventCompletionSnapshot(
            self.scientific_address,
            self.completed_episode_spawn_ordinals,
        )


def extract_v2_rlm_provenance(
    trace: dict[str, Any],
) -> tuple[RecordedRLMProvenanceV2, ...]:
    """Return a committed, causally consistent, trace-internally complete topology."""
    trace_id_value = trace.get("id")
    if (
        not isinstance(trace_id_value, str)
        or not trace_id_value
        or len(trace_id_value) > 512
        or not trace_id_value.isprintable()
    ):
        raise ValueError("trace id must be a nonempty printable string")
    trace_id = trace_id_value
    calls = trace.get("calls")
    if not isinstance(calls, list):
        raise TypeError("trace calls must be a list")
    nodes = trace.get("nodes")
    if not isinstance(nodes, list) or any(not isinstance(node, dict) for node in nodes):
        raise TypeError("trace nodes must be a list of objects")
    records = []
    for call_index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise TypeError("trace calls must contain objects")
        rlm = call.get("rlm")
        if not isinstance(rlm, dict):
            raise ValueError(f"call {call_index} lacks RLM v2 provenance")
        node_index = call.get("node")
        if type(node_index) is not int or node_index < 0:
            raise ValueError(f"call {call_index} is not linked to a committed node")
        if node_index >= len(nodes):
            raise ValueError(f"call {call_index} node index is out of bounds")
        if call.get("error") is not None:
            raise ValueError(f"call {call_index} records a failed model exchange")
        records.append(
            _parse_v2(
                trace_id=trace_id,
                call_index=call_index,
                node_index=node_index,
                payload=rlm,
            )
        )
    result = tuple(records)
    _validate_trace_topology(result)
    return result


def parse_v2_rlm_provenance_payload(
    *,
    trace_id: str,
    payload: dict[str, Any],
) -> RecordedRLMProvenanceV2:
    """Parse one prepared-call provenance payload before its trace node exists."""
    if (
        not isinstance(trace_id, str)
        or not trace_id
        or len(trace_id) > 512
        or not trace_id.isprintable()
    ):
        raise ValueError("trace id must be a nonempty printable string")
    if not isinstance(payload, dict):
        raise TypeError("RLM v2 provenance payload must be an object")
    return _parse_v2(
        trace_id=trace_id,
        call_index=0,
        node_index=0,
        payload=payload,
    )


def validate_v2_spawn_ticket_ledger(
    records: tuple[RecordedRLMProvenanceV2, ...],
    ledger_dir: Path,
) -> None:
    """Reconcile durable tickets so trailing orphan reservations cannot be hidden."""
    recorded = {
        record.episode_spawn_ordinal
        for record in records
        if record.episode_spawn_ordinal is not None
    }
    if not ledger_dir.exists():
        if recorded:
            raise ValueError("RLM v2 spawn ticket ledger is missing")
        return
    if not ledger_dir.is_dir():
        raise ValueError("RLM v2 spawn ticket ledger must be a directory")
    reserved: set[int] = set()
    terminal: set[int] = set()
    pattern = re.compile(r"([0-9]{12})\.(reserved|terminal)")
    for path in ledger_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None or not path.is_file():
            raise ValueError("RLM v2 spawn ticket ledger contains an unknown entry")
        expected_content = f"{match.group(2)}\n"
        if path.read_text(encoding="utf-8") != expected_content:
            raise ValueError("RLM v2 spawn ticket ledger contains malformed bytes")
        ordinal = int(match.group(1))
        target = reserved if match.group(2) == "reserved" else terminal
        target.add(ordinal)
    if reserved != terminal or reserved != recorded:
        raise ValueError("RLM v2 spawn tickets do not match committed child lineages")


def _parse_v2(
    *,
    trace_id: str,
    call_index: int,
    node_index: int,
    payload: dict[str, Any],
) -> RecordedRLMProvenanceV2:
    allowed = {
        "provenance_version",
        "depth",
        "session_id",
        "turn",
        "call_kind",
        "parent_session_id",
        "parent_turn",
        "parent_tool_call_id",
        "invocation_id",
        "lineage",
        "parent_lineage",
        "session_call_ordinal",
        "parent_call_ordinal",
        "parent_tool_call_slot",
        "spawn_ordinal",
        "episode_spawn_ordinal",
        "completed_predecessor_spawn_ordinals",
        "completed_episode_spawn_ordinals",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown RLM v2 provenance fields: {sorted(unknown)}")
    if type(payload.get("provenance_version")) is not int or payload.get("provenance_version") != 2:
        raise ValueError("RLM provenance_version must be the integer 2")

    def integer(name: str, *, required: bool = True) -> int | None:
        value = payload.get(name)
        if value is None and not required:
            return None
        if type(value) is not int or value < 0:
            raise ValueError(f"RLM v2 {name} must be a nonnegative integer")
        return value

    def text(
        name: str,
        *,
        required: bool = True,
        max_length: int = 512,
    ) -> str | None:
        value = payload.get(name)
        if value is None and not required:
            return None
        if (
            not isinstance(value, str)
            or not value
            or len(value) > max_length
            or not value.isprintable()
        ):
            raise ValueError(f"RLM v2 {name} must be a nonempty printable string")
        return value

    depth = integer("depth")
    session_id = text("session_id", max_length=128)
    turn = integer("turn")
    call_kind = text("call_kind")
    lineage = text("lineage")
    session_call_ordinal = integer("session_call_ordinal")
    completed_episode_spawn_ordinals = _integer_tuple(
        payload,
        "completed_episode_spawn_ordinals",
    )
    assert depth is not None
    assert session_id is not None
    assert turn is not None
    assert call_kind is not None
    assert lineage is not None
    assert session_call_ordinal is not None
    if call_kind not in {"policy", "compaction"}:
        raise ValueError("RLM v2 call_kind must be policy or compaction")

    recursive_names = (
        "parent_session_id",
        "parent_turn",
        "parent_tool_call_id",
        "invocation_id",
        "parent_lineage",
        "parent_call_ordinal",
        "parent_tool_call_slot",
        "spawn_ordinal",
        "episode_spawn_ordinal",
        "completed_predecessor_spawn_ordinals",
    )
    if depth == 0:
        if lineage != "root":
            raise ValueError("root RLM v2 lineage must be root")
        if any(payload.get(name) is not None for name in recursive_names):
            raise ValueError("root RLM v2 provenance contains recursive fields")
        return RecordedRLMProvenanceV2(
            trace_id=trace_id,
            call_index=call_index,
            node_index=node_index,
            depth=depth,
            session_id=session_id,
            turn=turn,
            call_kind=call_kind,
            lineage=lineage,
            session_call_ordinal=session_call_ordinal,
            parent_session_id=None,
            parent_turn=None,
            parent_tool_call_id=None,
            invocation_id=None,
            parent_lineage=None,
            parent_call_ordinal=None,
            parent_tool_call_slot=None,
            spawn_ordinal=None,
            episode_spawn_ordinal=None,
            completed_predecessor_spawn_ordinals=None,
            completed_episode_spawn_ordinals=completed_episode_spawn_ordinals,
        )

    predecessors_value = payload.get("completed_predecessor_spawn_ordinals")
    if not isinstance(predecessors_value, list) or any(
        type(value) is not int or value < 0 for value in predecessors_value
    ):
        raise ValueError(
            "RLM v2 completed_predecessor_spawn_ordinals must be a nonnegative integer list"
        )
    predecessors = tuple(predecessors_value)
    if predecessors != tuple(sorted(set(predecessors))):
        raise ValueError("RLM v2 predecessor spawn ordinals must be sorted and unique")
    spawn_ordinal = integer("spawn_ordinal")
    assert spawn_ordinal is not None
    if any(value >= spawn_ordinal for value in predecessors):
        raise ValueError("RLM v2 predecessor spawn ordinals must identify earlier siblings")

    parent_session_id = text("parent_session_id", max_length=128)
    parent_turn = integer("parent_turn")
    parent_tool_call_id = text("parent_tool_call_id", max_length=128)
    invocation_id = text("invocation_id", required=False, max_length=128)
    parent_lineage = text("parent_lineage")
    parent_call_ordinal = integer("parent_call_ordinal")
    parent_tool_call_slot = integer("parent_tool_call_slot")
    episode_spawn_ordinal = integer("episode_spawn_ordinal")
    assert parent_session_id is not None
    assert parent_turn is not None
    assert parent_tool_call_id is not None
    assert parent_lineage is not None
    assert parent_call_ordinal is not None
    assert parent_tool_call_slot is not None
    assert episode_spawn_ordinal is not None
    expected_lineage = derive_child_lineage(
        SpawnScope(
            depth=depth,
            parent_lineage=parent_lineage,
            parent_call_ordinal=parent_call_ordinal,
            parent_tool_call_slot=parent_tool_call_slot,
            parent_turn=parent_turn,
        ),
        spawn_ordinal=spawn_ordinal,
    )
    if lineage != expected_lineage:
        raise ValueError("RLM v2 lineage does not match its structural coordinates")
    return RecordedRLMProvenanceV2(
        trace_id=trace_id,
        call_index=call_index,
        node_index=node_index,
        depth=depth,
        session_id=session_id,
        turn=turn,
        call_kind=call_kind,
        lineage=lineage,
        session_call_ordinal=session_call_ordinal,
        parent_session_id=parent_session_id,
        parent_turn=parent_turn,
        parent_tool_call_id=parent_tool_call_id,
        invocation_id=invocation_id,
        parent_lineage=parent_lineage,
        parent_call_ordinal=parent_call_ordinal,
        parent_tool_call_slot=parent_tool_call_slot,
        spawn_ordinal=spawn_ordinal,
        episode_spawn_ordinal=episode_spawn_ordinal,
        completed_predecessor_spawn_ordinals=predecessors,
        completed_episode_spawn_ordinals=completed_episode_spawn_ordinals,
    )


def _validate_trace_topology(
    records: tuple[RecordedRLMProvenanceV2, ...],
) -> None:
    if not records:
        raise ValueError("RLM v2 trace contains no committed model calls")
    by_event: dict[tuple[str, int], RecordedRLMProvenanceV2] = {}
    by_lineage: dict[str, list[RecordedRLMProvenanceV2]] = {}
    node_indexes: set[int] = set()
    for record in records:
        if record.node_index in node_indexes:
            raise ValueError("RLM v2 trace reuses a committed node index")
        node_indexes.add(record.node_index)
        event_key = (record.lineage, record.session_call_ordinal)
        if event_key in by_event:
            raise ValueError("RLM v2 trace repeats a scientific event address")
        by_event[event_key] = record
        by_lineage.setdefault(record.lineage, []).append(record)

    child_roots: dict[str, RecordedRLMProvenanceV2] = {}
    episode_lineages: dict[int, str] = {}
    scope_spawns: dict[tuple[str, int, int], dict[int, str]] = {}
    for lineage, lineage_records in by_lineage.items():
        depths = {record.depth for record in lineage_records}
        session_ids = {record.session_id for record in lineage_records}
        ordinals = sorted(record.session_call_ordinal for record in lineage_records)
        if len(depths) != 1 or len(session_ids) != 1 or ordinals != list(range(len(ordinals))):
            raise ValueError("RLM v2 lineage has inconsistent session, depth, or call ordinals")
        first = lineage_records[0]
        if lineage == "root":
            if first.depth != 0:
                raise ValueError("RLM v2 root lineage must have depth zero")
            continue
        if first.depth == 0:
            raise ValueError("non-root RLM v2 lineage cannot have depth zero")
        assert first.parent_lineage is not None
        assert first.parent_call_ordinal is not None
        assert first.parent_tool_call_slot is not None
        assert first.spawn_ordinal is not None
        assert first.episode_spawn_ordinal is not None
        assert first.completed_predecessor_spawn_ordinals is not None
        if any(_spawn_signature(record) != _spawn_signature(first) for record in lineage_records):
            raise ValueError("RLM v2 child lineage changes spawn metadata")
        child_roots[lineage] = first
        prior_lineage = episode_lineages.setdefault(first.episode_spawn_ordinal, lineage)
        if prior_lineage != lineage:
            raise ValueError("RLM v2 episode spawn ordinal is reused")
        scope = (
            first.parent_lineage,
            first.parent_call_ordinal,
            first.parent_tool_call_slot,
        )
        spawn_lineages = scope_spawns.setdefault(scope, {})
        prior_spawn_lineage = spawn_lineages.setdefault(first.spawn_ordinal, lineage)
        if prior_spawn_lineage != lineage:
            raise ValueError("RLM v2 parent scope reuses a spawn ordinal")

    root_event = by_event.get(("root", 0))
    if root_event is None or root_event.call_kind != "policy":
        raise ValueError("RLM v2 trace lacks its initial root policy event")
    if episode_lineages and sorted(episode_lineages) != list(range(len(episode_lineages))):
        raise ValueError("RLM v2 episode spawn ordinals are not contiguous")
    if any(
        ordinal not in episode_lineages
        for record in records
        for ordinal in record.completed_episode_spawn_ordinals
    ):
        raise ValueError("RLM v2 event names an absent completed episode spawn")
    if any(
        record.episode_spawn_ordinal in record.completed_episode_spawn_ordinals
        for record in records
        if record.episode_spawn_ordinal is not None
    ):
        raise ValueError("RLM v2 child event cannot observe its own terminal marker")
    if any(sorted(spawns) != list(range(len(spawns))) for spawns in scope_spawns.values()):
        raise ValueError("RLM v2 local spawn ordinals are not contiguous")

    for first in child_roots.values():
        assert first.parent_lineage is not None
        assert first.parent_call_ordinal is not None
        assert first.parent_tool_call_slot is not None
        assert first.completed_predecessor_spawn_ordinals is not None
        parent = by_event.get((first.parent_lineage, first.parent_call_ordinal))
        if parent is None or parent.call_kind != "policy" or parent.depth != first.depth - 1:
            raise ValueError("RLM v2 child lacks its causal parent policy event")
        siblings = scope_spawns[
            (
                first.parent_lineage,
                first.parent_call_ordinal,
                first.parent_tool_call_slot,
            )
        ]
        if any(
            predecessor not in siblings
            for predecessor in first.completed_predecessor_spawn_ordinals
        ):
            raise ValueError("RLM v2 child names an absent predecessor sibling")


def _spawn_signature(record: RecordedRLMProvenanceV2) -> tuple[object, ...]:
    """Fields that must remain constant for every call in one child session."""
    return (
        record.depth,
        record.session_id,
        record.parent_session_id,
        record.parent_turn,
        record.parent_tool_call_id,
        record.invocation_id,
        record.parent_lineage,
        record.parent_call_ordinal,
        record.parent_tool_call_slot,
        record.spawn_ordinal,
        record.episode_spawn_ordinal,
        record.completed_predecessor_spawn_ordinals,
    )


def _integer_tuple(payload: dict[str, Any], name: str) -> tuple[int, ...]:
    value = payload.get(name)
    if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"RLM v2 {name} must be a nonnegative integer list")
    result = tuple(value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"RLM v2 {name} must be sorted and unique")
    return result
