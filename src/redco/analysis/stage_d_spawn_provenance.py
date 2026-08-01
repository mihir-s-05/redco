"""Outcome-independent child-spawn provenance and event-keyed seed contracts."""

from __future__ import annotations

import hashlib
import hmac
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from redco.contracts import canonical_json

_LINEAGE_DOMAIN = "redco.stage-d.spawn-lineage.v2"
_SEED_DOMAIN = "redco.stage-d.event-seed.v2"


class CouplingMode(StrEnum):
    """Whether a continuation seed is shared or intentionally branch-specific."""

    PAIRED = "paired"
    EXOGENOUS = "exogenous"


@dataclass(frozen=True, slots=True)
class SpawnScope:
    """Stable parent coordinates for child calls created in one tool execution."""

    depth: int
    parent_lineage: str
    parent_call_ordinal: int
    parent_tool_call_slot: int
    parent_turn: int

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("child depth must be positive")
        if not self.parent_lineage:
            raise ValueError("parent_lineage must be nonempty")
        for name in ("parent_call_ordinal", "parent_tool_call_slot", "parent_turn"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")

    def scientific_key(self) -> tuple[int, str, int, int]:
        """Return the identity coordinates, excluding diagnostic ``parent_turn``."""
        return (
            self.depth,
            self.parent_lineage,
            self.parent_call_ordinal,
            self.parent_tool_call_slot,
        )


def derive_child_lineage(scope: SpawnScope, *, spawn_ordinal: int) -> str:
    """Derive hierarchical identity without transport-generated identifiers."""
    if spawn_ordinal < 0:
        raise ValueError("spawn_ordinal must be nonnegative")
    digest = hashlib.sha256(
        canonical_json(
            {
                "domain": _LINEAGE_DOMAIN,
                "depth": scope.depth,
                "parent_lineage": scope.parent_lineage,
                "parent_call_ordinal": scope.parent_call_ordinal,
                "parent_tool_call_slot": scope.parent_tool_call_slot,
                "spawn_ordinal": spawn_ordinal,
            }
        )
    ).hexdigest()[:24]
    return f"{scope.parent_lineage}/{digest}"


@dataclass(frozen=True, slots=True)
class SpawnReservation:
    """A child identity fixed when ``rlm(...)`` is called, before scheduling."""

    scope: SpawnScope
    spawn_ordinal: int
    episode_spawn_ordinal: int
    completed_predecessor_spawn_ordinals: tuple[int, ...]
    lineage: str

    def __post_init__(self) -> None:
        if self.spawn_ordinal < 0 or self.episode_spawn_ordinal < 0:
            raise ValueError("spawn ordinals must be nonnegative")
        predecessors = self.completed_predecessor_spawn_ordinals
        if tuple(sorted(set(predecessors))) != predecessors:
            raise ValueError("predecessor ordinals must be sorted and unique")
        if any(value < 0 or value >= self.spawn_ordinal for value in predecessors):
            raise ValueError("predecessors must identify earlier sibling spawns")
        expected = derive_child_lineage(self.scope, spawn_ordinal=self.spawn_ordinal)
        if self.lineage != expected:
            raise ValueError("lineage does not match the structural spawn coordinates")


class SpawnLedger:
    """Thread-safe reference allocator used to pin spawn-time semantics in tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_episode_ordinal = 0
        self._scope_reservations: dict[tuple[int, str, int, int], list[SpawnReservation]] = {}
        self._completed_episode_ordinals: set[int] = set()

    def reserve(self, scope: SpawnScope) -> SpawnReservation:
        """Reserve identity synchronously; request and completion order are irrelevant."""
        with self._lock:
            siblings = self._scope_reservations.setdefault(scope.scientific_key(), [])
            spawn_ordinal = len(siblings)
            episode_ordinal = self._next_episode_ordinal
            predecessors = tuple(
                reservation.spawn_ordinal
                for reservation in siblings
                if reservation.episode_spawn_ordinal in self._completed_episode_ordinals
            )
            reservation = SpawnReservation(
                scope=scope,
                spawn_ordinal=spawn_ordinal,
                episode_spawn_ordinal=episode_ordinal,
                completed_predecessor_spawn_ordinals=predecessors,
                lineage=derive_child_lineage(scope, spawn_ordinal=spawn_ordinal),
            )
            siblings.append(reservation)
            self._next_episode_ordinal = episode_ordinal + 1
            return reservation

    def complete(self, reservation: SpawnReservation) -> None:
        """Mark a child terminal; failures count because later control may observe them."""
        with self._lock:
            known = self._scope_reservations.get(reservation.scope.scientific_key(), [])
            if reservation not in known:
                raise ValueError("cannot complete an unknown spawn reservation")
            episode_ordinal = reservation.episode_spawn_ordinal
            if episode_ordinal in self._completed_episode_ordinals:
                raise ValueError("spawn reservation is already complete")
            self._completed_episode_ordinals.add(episode_ordinal)


@dataclass(frozen=True, slots=True)
class EventCompletionSnapshot:
    """Durable child completions visible before one outbound policy event."""

    address: PolicyEventAddress
    completed_episode_spawn_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        completed = self.completed_episode_spawn_ordinals
        if completed != tuple(sorted(set(completed))) or any(
            type(value) is not int or value < 0 for value in completed
        ):
            raise ValueError("completed episode spawn ordinals must be sorted and unique")


class CausalProvenanceGraph:
    """Exact reachability over session order, child calls, and sibling dependencies."""

    def __init__(
        self,
        *,
        events: Iterable[PolicyEventAddress],
        spawns: Iterable[SpawnReservation],
        completion_snapshots: Iterable[EventCompletionSnapshot],
    ) -> None:
        event_tuple = tuple(events)
        if len(set(event_tuple)) != len(event_tuple):
            raise ValueError("policy event addresses must be unique")
        self._events = set(event_tuple)
        self._edges: dict[PolicyEventAddress, set[PolicyEventAddress]] = {
            event: set() for event in event_tuple
        }
        snapshot_tuple = tuple(completion_snapshots)
        snapshot_by_event = {
            snapshot.address: snapshot.completed_episode_spawn_ordinals
            for snapshot in snapshot_tuple
        }
        if len(snapshot_by_event) != len(snapshot_tuple) or set(snapshot_by_event) != self._events:
            raise ValueError("every policy event needs exactly one completion snapshot")
        by_lineage: dict[str, list[PolicyEventAddress]] = {}
        for event in event_tuple:
            by_lineage.setdefault(event.lineage, []).append(event)
        for lineage_events in by_lineage.values():
            lineage_events.sort(key=lambda event: event.session_call_ordinal)
            ordinals = [event.session_call_ordinal for event in lineage_events]
            if len(set(ordinals)) != len(ordinals):
                raise ValueError("one lineage cannot reuse a session call ordinal")
            for earlier, later in pairwise(lineage_events):
                self._edges[earlier].add(later)

        spawn_tuple = tuple(spawns)
        by_child_lineage: dict[str, SpawnReservation] = {}
        by_scope_slot: dict[tuple[tuple[int, str, int, int], int], SpawnReservation] = {}
        for spawn in spawn_tuple:
            if spawn.lineage in by_child_lineage:
                raise ValueError("child lineages must be unique")
            by_child_lineage[spawn.lineage] = spawn
            by_scope_slot[(spawn.scope.scientific_key(), spawn.spawn_ordinal)] = spawn

        for spawn in spawn_tuple:
            parent_events = by_lineage.get(spawn.scope.parent_lineage, [])
            child_events = by_lineage.get(spawn.lineage, [])
            parent = next(
                (
                    event
                    for event in parent_events
                    if event.session_call_ordinal == spawn.scope.parent_call_ordinal
                    and event.call_kind == "policy"
                ),
                None,
            )
            if parent is None or not child_events:
                raise ValueError("spawn provenance lacks its parent or child policy event")
            first_child = child_events[0]
            last_child = child_events[-1]
            self._edges[parent].add(first_child)
            later_parent = next(
                (
                    event
                    for event in parent_events
                    if event.session_call_ordinal > spawn.scope.parent_call_ordinal
                    and spawn.episode_spawn_ordinal in snapshot_by_event[event]
                ),
                None,
            )
            if later_parent is not None:
                self._edges[last_child].add(later_parent)
            for predecessor_slot in spawn.completed_predecessor_spawn_ordinals:
                predecessor = by_scope_slot.get((spawn.scope.scientific_key(), predecessor_slot))
                if predecessor is None:
                    raise ValueError("spawn predecessor is absent from the same parent scope")
                predecessor_events = by_lineage.get(predecessor.lineage, [])
                if not predecessor_events:
                    raise ValueError("spawn predecessor has no policy event")
                self._edges[predecessor_events[-1]].add(first_child)

    def is_downstream(
        self,
        candidate: PolicyEventAddress,
        *,
        target: PolicyEventAddress,
    ) -> bool:
        """Return strict graph reachability; an event is not downstream of itself."""
        if candidate not in self._events or target not in self._events:
            raise ValueError("causal query contains an unknown policy event")
        queue = deque([target])
        visited = {target}
        while queue:
            current = queue.popleft()
            for successor in self._edges[current]:
                if successor == candidate:
                    return True
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)
        return False


@dataclass(frozen=True, slots=True)
class PolicyEventAddress:
    """Structural stochastic-event address independent of generated transport IDs."""

    depth: int
    lineage: str
    session_call_ordinal: int
    turn: int
    call_kind: str = "policy"

    def __post_init__(self) -> None:
        if self.depth < 0 or self.session_call_ordinal < 0 or self.turn < 0:
            raise ValueError("depth, session_call_ordinal, and turn must be nonnegative")
        if not self.lineage:
            raise ValueError("lineage must be nonempty")
        if self.call_kind not in {"policy", "compaction"}:
            raise ValueError("call_kind must be policy or compaction")

    def as_payload(self) -> dict[str, str | int]:
        """Return only scientific coordinates; turn is a diagnostic assertion."""
        return {
            "depth": self.depth,
            "lineage": self.lineage,
            "session_call_ordinal": self.session_call_ordinal,
            "call_kind": self.call_kind,
        }


@dataclass(frozen=True, slots=True)
class ScheduledSeed:
    seed: int
    coupling_mode: CouplingMode
    address: PolicyEventAddress


@dataclass(frozen=True, slots=True)
class EventSeedScheduler:
    """Counter-based CRN scheduler with a separate dynamic-topology namespace."""

    master_seed: str
    rollout_id: str
    target_id: str
    continuation_replicate: int

    def __post_init__(self) -> None:
        if not self.master_seed or not self.rollout_id or not self.target_id:
            raise ValueError("seed namespace identifiers must be nonempty")
        if self.continuation_replicate < 1:
            raise ValueError("continuation_replicate must be one-indexed")

    def paired_continuation_seed(
        self,
        address: PolicyEventAddress,
        *,
        committed_address: PolicyEventAddress,
    ) -> ScheduledSeed:
        """Share randomness only after exact structural correspondence is established."""
        if address.as_payload() != committed_address.as_payload():
            raise ValueError("paired events must match the committed structural address")
        return ScheduledSeed(
            seed=self._derive(
                purpose="paired_continuation",
                payload={"event_address": committed_address.as_payload()},
            ),
            coupling_mode=CouplingMode.PAIRED,
            address=address,
        )

    def exogenous_continuation_seed(
        self,
        address: PolicyEventAddress,
        *,
        action_arm: str,
    ) -> ScheduledSeed:
        """Give an unmatched dynamic event a deterministic, non-paired seed."""
        if not action_arm:
            raise ValueError("action_arm must be nonempty for unmatched topology")
        return ScheduledSeed(
            seed=self._derive(
                purpose="exogenous_continuation",
                payload={
                    "action_arm": action_arm,
                    "event_address": address.as_payload(),
                },
            ),
            coupling_mode=CouplingMode.EXOGENOUS,
            address=address,
        )

    def action_seed(self, *, action_slot: int) -> int:
        """Derive target-action randomness outside every continuation namespace."""
        if action_slot < 0:
            raise ValueError("action_slot must be nonnegative")
        return self._derive_payload(
            {
                "domain": _SEED_DOMAIN,
                "rollout_id": self.rollout_id,
                "target_id": self.target_id,
                "purpose": "candidate_action",
                "action_slot": action_slot,
            }
        )

    def _derive(self, *, purpose: str, payload: dict[str, object]) -> int:
        return self._derive_payload(
            {
                "domain": _SEED_DOMAIN,
                "rollout_id": self.rollout_id,
                "target_id": self.target_id,
                "continuation_replicate": self.continuation_replicate,
                "purpose": purpose,
                **payload,
            }
        )

    def _derive_payload(self, payload: dict[str, object]) -> int:
        namespace = canonical_json(payload)
        digest = hmac.new(self.master_seed.encode("utf-8"), namespace, hashlib.sha256).digest()
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
