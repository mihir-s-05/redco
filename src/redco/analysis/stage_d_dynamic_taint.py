"""Structural causal tainting for live Stage-D branch replay."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from redco.analysis.stage_d_spawn_provenance import (
    CausalProvenanceGraph,
    PolicyEventAddress,
    SpawnReservation,
    SpawnScope,
)
from redco.contracts import canonical_json
from redco.integrations.verifiers_trace_v2 import RecordedRLMProvenanceV2


class ReplayDisposition(StrEnum):
    REUSE = "reuse"
    INJECT = "inject"
    GENERATE = "generate"


@dataclass(frozen=True, slots=True)
class ReplayEventDecision:
    address: PolicyEventAddress
    disposition: ReplayDisposition
    source_matched: bool
    counts_toward_logical_continuation: bool


class DynamicCausalTaintTracker:
    """Classify one live replay event without using request arrival chronology."""

    def __init__(
        self,
        *,
        target: PolicyEventAddress,
        source_records: Iterable[RecordedRLMProvenanceV2],
        source_graph: CausalProvenanceGraph,
    ) -> None:
        records = tuple(source_records)
        by_address = {_address_key(record.scientific_address): record for record in records}
        if len(by_address) != len(records):
            raise ValueError("source replay records reuse a structural policy address")
        target_key = _address_key(target)
        if target_key not in by_address:
            raise ValueError("replay target is absent from the frozen source topology")
        if by_address[target_key].scientific_address != target:
            raise ValueError("replay target changed its diagnostic turn")
        self._target = target
        self._target_key = target_key
        self._source = by_address
        self._completed_before_target = set(
            by_address[target_key].completed_episode_spawn_ordinals
        )
        self._graph = source_graph
        self._required_source_keys = {
            key
            for key, source in by_address.items()
            if key == target_key
            or not source_graph.is_downstream(source.scientific_address, target=target)
        }
        target_record = by_address[target_key]
        self._required_through_target_keys = {
            _address_key(record.scientific_address)
            for record in records
            if _address_key(record.scientific_address) in self._required_source_keys
            and (
                _address_key(record.scientific_address) == target_key
                or source_graph.is_downstream(
                    target,
                    target=record.scientific_address,
                )
                or (
                    record.episode_spawn_ordinal is not None
                    and record.episode_spawn_ordinal in self._completed_before_target
                )
                or (
                    record.lineage == target_record.lineage
                    and record.session_call_ordinal
                    < target_record.session_call_ordinal
                )
            )
        }
        self._seen: set[bytes] = set()
        self._tainted: set[bytes] = set()
        self._tainted_lineage_start: dict[str, int] = {}
        self._tainted_episode_spawns: set[int] = set()

    def observe(self, record: RecordedRLMProvenanceV2) -> ReplayEventDecision:
        if type(record) is not RecordedRLMProvenanceV2:
            raise ValueError("replay event must carry parsed v2 provenance")
        address = record.scientific_address
        address_key = _address_key(address)
        if address_key in self._seen:
            raise ValueError("replay attempted one structural policy address twice")
        self._seen.add(address_key)
        source = self._source.get(address_key)
        if source is not None:
            _require_same_structural_parent(source, record)
            if address_key == self._target_key:
                disposition = ReplayDisposition.INJECT
                tainted = True
            elif self._graph.is_downstream(
                source.scientific_address,
                target=self._target,
            ):
                disposition = ReplayDisposition.GENERATE
                tainted = True
            else:
                _require_same_causal_snapshot(source, record)
                disposition = ReplayDisposition.REUSE
                tainted = False
        else:
            tainted = self._is_dynamically_tainted(record)
            if not tainted:
                raise ValueError("unexpected dynamic policy event is not target-descended")
            disposition = ReplayDisposition.GENERATE
        if tainted:
            self._mark_tainted(record)
        logical_continuation = bool(
            disposition is ReplayDisposition.REUSE
            and source is not None
            and source.episode_spawn_ordinal is not None
            and source.episode_spawn_ordinal not in self._completed_before_target
            and not self._graph.is_downstream(
                self._target,
                target=source.scientific_address,
            )
        )
        return ReplayEventDecision(
            address,
            disposition,
            source is not None,
            logical_continuation,
        )

    def finalize(self) -> None:
        """Require target and every non-downstream frozen source event exactly once."""
        missing = self._required_source_keys - self._seen
        if missing:
            raise ValueError("replay omitted target or non-downstream frozen source events")

    def finalize_terminal_truncation(self) -> None:
        """Allow only source events after a clean, target-delivered terminal truncation."""
        missing = self._required_through_target_keys - self._seen
        if missing or self._target_key not in self._seen:
            raise ValueError(
                "truncated replay omitted the target or a required pre-target event"
            )

    def _is_dynamically_tainted(self, record: RecordedRLMProvenanceV2) -> bool:
        address = record.scientific_address
        lineage_start = self._tainted_lineage_start.get(address.lineage)
        if lineage_start is not None and address.session_call_ordinal > lineage_start:
            return True
        if record.depth > 0:
            assert record.parent_lineage is not None
            assert record.parent_call_ordinal is not None
            assert record.parent_turn is not None
            parent = PolicyEventAddress(
                record.depth - 1,
                record.parent_lineage,
                record.parent_call_ordinal,
                record.parent_turn,
            )
            if _address_key(parent) in self._tainted:
                return True
        return bool(
            record.depth == 0
            and set(record.completed_episode_spawn_ordinals)
            & self._tainted_episode_spawns
        )

    def _mark_tainted(self, record: RecordedRLMProvenanceV2) -> None:
        address = record.scientific_address
        self._tainted.add(_address_key(address))
        prior = self._tainted_lineage_start.get(address.lineage)
        if prior is None or address.session_call_ordinal < prior:
            self._tainted_lineage_start[address.lineage] = address.session_call_ordinal
        if record.episode_spawn_ordinal is not None:
            self._tainted_episode_spawns.add(record.episode_spawn_ordinal)


def build_source_causal_graph(
    records: Iterable[RecordedRLMProvenanceV2],
) -> CausalProvenanceGraph:
    """Rebuild the exact causal graph carried by a verified live source trace."""
    frozen = tuple(records)
    if not frozen:
        raise ValueError("source causal graph requires policy events")
    first_by_lineage: dict[str, RecordedRLMProvenanceV2] = {}
    for record in frozen:
        first_by_lineage.setdefault(record.lineage, record)
    spawns: list[SpawnReservation] = []
    for record in first_by_lineage.values():
        if record.depth == 0:
            continue
        if any(
            value is None
            for value in (
                record.parent_lineage,
                record.parent_call_ordinal,
                record.parent_tool_call_slot,
                record.parent_turn,
                record.spawn_ordinal,
                record.episode_spawn_ordinal,
                record.completed_predecessor_spawn_ordinals,
            )
        ):
            raise ValueError("child source event lacks complete spawn provenance")
        assert record.parent_lineage is not None
        assert record.parent_call_ordinal is not None
        assert record.parent_tool_call_slot is not None
        assert record.parent_turn is not None
        assert record.spawn_ordinal is not None
        assert record.episode_spawn_ordinal is not None
        assert record.completed_predecessor_spawn_ordinals is not None
        spawns.append(
            SpawnReservation(
                SpawnScope(
                    record.depth,
                    record.parent_lineage,
                    record.parent_call_ordinal,
                    record.parent_tool_call_slot,
                    record.parent_turn,
                ),
                record.spawn_ordinal,
                record.episode_spawn_ordinal,
                record.completed_predecessor_spawn_ordinals,
                record.lineage,
            )
        )
    return CausalProvenanceGraph(
        events=(record.scientific_address for record in frozen),
        spawns=spawns,
        completion_snapshots=(record.completion_snapshot for record in frozen),
    )


def _address_key(address: PolicyEventAddress) -> bytes:
    return canonical_json(address.as_payload())


def _require_same_structural_parent(
    source: RecordedRLMProvenanceV2,
    replay: RecordedRLMProvenanceV2,
) -> None:
    fields = (
        "depth",
        "lineage",
        "session_call_ordinal",
        "turn",
        "call_kind",
        "parent_lineage",
        "parent_call_ordinal",
        "parent_tool_call_slot",
        "spawn_ordinal",
        "episode_spawn_ordinal",
    )
    if any(getattr(source, name) != getattr(replay, name) for name in fields):
        raise ValueError("source-matched replay event changed structural parentage")


def _require_same_causal_snapshot(
    source: RecordedRLMProvenanceV2,
    replay: RecordedRLMProvenanceV2,
) -> None:
    if (
        source.completed_predecessor_spawn_ordinals
        != replay.completed_predecessor_spawn_ordinals
        or source.completed_episode_spawn_ordinals
        != replay.completed_episode_spawn_ordinals
    ):
        raise ValueError("reused replay event changed its causal completion snapshot")
