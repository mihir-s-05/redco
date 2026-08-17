"""Event-level causal trace with dependency closure."""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from redco.contracts import PolicyNodeKind, canonical_json


class EventNodeKind(StrEnum):
    POLICY = "policy"
    OPERATION = "operation"
    ARTIFACT = "artifact"
    REWARD_RESOURCE = "reward_resource"


class EdgeKind(StrEnum):
    DATAFLOW = "dataflow"
    CONTROL = "control"
    CALL = "call"
    SIDE_EFFECT = "side_effect"
    OBSERVATION = "observation"
    RESOURCE = "resource"


class InformationChannelKind(StrEnum):
    """How information became visible to a policy observation."""

    DECLARED_READ = "declared_read"
    AMBIENT_STDOUT = "ambient_stdout"
    AMBIENT_HISTORY = "ambient_history"
    INHERITED_CONTEXT = "inherited_context"


@dataclass(frozen=True, slots=True)
class EventNode:
    node_id: str
    kind: EventNodeKind
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EventEdge:
    source: str
    target: str
    kind: EdgeKind
    via: str | None = None


@dataclass(frozen=True, slots=True)
class InformationProvenanceSpan:
    """One actual-inclusion span with a declared or ambient access path."""

    source_event_id: str
    channel_kind: InformationChannelKind
    token_start: int
    token_end: int
    serializer_id: str
    raw_content_hash: str
    semantic_content_hash: str
    visibility_scope: str
    artifact_id: str | None = None
    artifact_version: int | None = None

    def __post_init__(self) -> None:
        if not self.source_event_id:
            raise ValueError("source event must be non-empty")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("prompt token span must be non-empty and ordered")
        if not self.serializer_id or not self.visibility_scope:
            raise ValueError("serializer and visibility scope must be non-empty")
        for name, value in (
            ("raw_content_hash", self.raw_content_hash),
            ("semantic_content_hash", self.semantic_content_hash),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (self.artifact_id is None) != (self.artifact_version is None):
            raise ValueError("artifact ID and version must be present together")
        if self.artifact_version is not None and self.artifact_version < 0:
            raise ValueError("artifact version must be non-negative")
        if self.channel_kind is InformationChannelKind.DECLARED_READ and not self.artifact_id:
            raise ValueError("declared reads require an artifact identity")


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    """Exact rendered token IDs and access-path-aware actual-inclusion spans."""

    prompt_token_ids: tuple[int, ...]
    provenance_spans: tuple[InformationProvenanceSpan, ...] = ()

    def __post_init__(self) -> None:
        if any(token_id < 0 for token_id in self.prompt_token_ids):
            raise ValueError("token IDs must be non-negative")
        for span in self.provenance_spans:
            if span.token_end > len(self.prompt_token_ids):
                raise ValueError("provenance span exceeds rendered prompt")

    @property
    def token_ids_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.prompt_token_ids)).hexdigest()


@dataclass(frozen=True, slots=True)
class PolicyNodeRecord:
    node_id: str
    kind: PolicyNodeKind
    observation: PolicyObservation
    action_token_start: int
    action_token_end: int
    behavior_logprobs_ref: str
    checkpoint_id: str
    decoding_config_hash: str
    state_snapshot_ref: str
    event_seed: int
    target_status: str

    def __post_init__(self) -> None:
        references = (
            self.node_id,
            self.behavior_logprobs_ref,
            self.checkpoint_id,
            self.decoding_config_hash,
            self.state_snapshot_ref,
            self.target_status,
        )
        if any(not reference for reference in references):
            raise ValueError("policy-node identifiers and references must be non-empty")
        if self.action_token_start < 0 or self.action_token_end <= self.action_token_start:
            raise ValueError("action token span must be non-empty and ordered")
        if self.event_seed < 0:
            raise ValueError("event_seed must be non-negative")


class EventGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, EventNode] = {}
        self.edges: list[EventEdge] = []
        self._outgoing: dict[str, list[EventEdge]] = {}

    def add_node(self, node: EventNode) -> None:
        if not node.node_id:
            raise ValueError("node_id must be non-empty")
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node: {node.node_id}")
        self.nodes[node.node_id] = node
        self._outgoing[node.node_id] = []

    def add_edge(self, edge: EventEdge) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError("edge endpoints must exist before the edge")
        if edge.source == edge.target:
            raise ValueError("self edges are not allowed")
        self.edges.append(edge)
        self._outgoing[edge.source].append(edge)
        try:
            self.assert_acyclic()
        except ValueError:
            self.edges.pop()
            self._outgoing[edge.source].pop()
            raise

    def descendants(
        self,
        node_id: str,
        *,
        edge_kinds: frozenset[EdgeKind] | None = None,
    ) -> frozenset[str]:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        seen: set[str] = set()
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            for edge in self._outgoing[current]:
                if edge_kinds is not None and edge.kind not in edge_kinds:
                    continue
                if edge.target not in seen:
                    seen.add(edge.target)
                    queue.append(edge.target)
        return frozenset(seen)

    def assert_acyclic(self) -> None:
        indegree = dict.fromkeys(self.nodes, 0)
        for edge in self.edges:
            indegree[edge.target] += 1
        queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for edge in self._outgoing[current]:
                indegree[edge.target] -= 1
                if indegree[edge.target] == 0:
                    queue.append(edge.target)
        if visited != len(self.nodes):
            raise ValueError("event graph must remain acyclic")

    def as_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible trace data."""
        nodes = [
            {
                "node_id": node.node_id,
                "kind": node.kind.value,
                "metadata": node.metadata,
            }
            for node in sorted(self.nodes.values(), key=lambda item: item.node_id)
        ]
        edges = [
            {
                "source": edge.source,
                "target": edge.target,
                "kind": edge.kind.value,
                "via": edge.via,
            }
            for edge in sorted(
                self.edges,
                key=lambda item: (item.source, item.target, item.kind.value, item.via or ""),
            )
        ]
        return {"nodes": nodes, "edges": edges}
