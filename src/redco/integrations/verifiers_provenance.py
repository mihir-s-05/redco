"""Map native verifiers traces into ReDCO event graphs and measured ledgers."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from redco.contracts import PolicyNodeKind, canonical_json
from redco.env.tracer import (
    EdgeKind,
    EventEdge,
    EventGraph,
    EventNode,
    EventNodeKind,
    PolicyNodeRecord,
    PolicyObservation,
    PromptProvenanceSpan,
)
from redco.integrations.verifiers_trace import (
    RecordedPolicyCall,
    extract_policy_calls,
    load_trace_records,
    path_to_node,
)


@dataclass(frozen=True, slots=True)
class TraceCostLedger:
    """Meters that are directly supported by one native trace."""

    prompt_tokens: int
    generated_tokens: int
    model_calls: int
    model_call_wall_seconds: float
    trace_generation_wall_seconds: float
    judge_calls: int
    storage_bytes: int
    unavailable_meters: tuple[str, ...] = ("cpu_seconds", "gpu_seconds")


@dataclass(frozen=True, slots=True)
class TraceProvenanceResult:
    trace_id: str
    graph: EventGraph
    policy_records: tuple[PolicyNodeRecord, ...]
    cost: TraceCostLedger
    rendered_prompt_tokens: int
    mapped_prompt_tokens: int
    exact_observation_edges: int
    connected_components: int
    recursive_components: int
    structurally_identified_model_calls: int
    recursive_model_calls: int
    exactly_parented_recursive_model_calls: int
    exact_cross_component_links: int
    cross_component_fallbacks: int
    unresolved_cross_component_links: int

    @property
    def exact_prompt_provenance_coverage(self) -> float:
        if not self.rendered_prompt_tokens:
            return 0.0
        return self.mapped_prompt_tokens / self.rendered_prompt_tokens

    @property
    def cross_component_fallback_rate(self) -> float:
        if not self.recursive_components:
            return 0.0
        return self.cross_component_fallbacks / self.recursive_components

    def as_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "event_graph": self.graph.as_dict(),
            "policy_records": [
                {
                    **asdict(record),
                    "kind": record.kind.value,
                }
                for record in self.policy_records
            ],
            "cost": asdict(self.cost),
            "rendered_prompt_tokens": self.rendered_prompt_tokens,
            "mapped_prompt_tokens": self.mapped_prompt_tokens,
            "exact_observation_edges": self.exact_observation_edges,
            "exact_prompt_provenance_coverage": (
                self.exact_prompt_provenance_coverage
            ),
            "connected_components": self.connected_components,
            "recursive_components": self.recursive_components,
            "structurally_identified_model_calls": (
                self.structurally_identified_model_calls
            ),
            "recursive_model_calls": self.recursive_model_calls,
            "exactly_parented_recursive_model_calls": (
                self.exactly_parented_recursive_model_calls
            ),
            "exact_cross_component_links": self.exact_cross_component_links,
            "cross_component_fallbacks": self.cross_component_fallbacks,
            "cross_component_fallback_rate": (
                self.cross_component_fallback_rate
            ),
            "unresolved_cross_component_links": (
                self.unresolved_cross_component_links
            ),
        }


@dataclass(frozen=True, slots=True)
class ProvenanceFileReport:
    schema_version: int
    source: str
    source_sha256: str
    traces: tuple[TraceProvenanceResult, ...]

    @property
    def rendered_prompt_tokens(self) -> int:
        return sum(trace.rendered_prompt_tokens for trace in self.traces)

    @property
    def mapped_prompt_tokens(self) -> int:
        return sum(trace.mapped_prompt_tokens for trace in self.traces)

    @property
    def exact_prompt_provenance_coverage(self) -> float:
        if not self.rendered_prompt_tokens:
            return 0.0
        return self.mapped_prompt_tokens / self.rendered_prompt_tokens

    @property
    def recursive_components(self) -> int:
        return sum(trace.recursive_components for trace in self.traces)

    @property
    def cross_component_fallbacks(self) -> int:
        return sum(trace.cross_component_fallbacks for trace in self.traces)

    @property
    def exact_cross_component_links(self) -> int:
        return sum(trace.exact_cross_component_links for trace in self.traces)

    @property
    def model_calls(self) -> int:
        return sum(trace.cost.model_calls for trace in self.traces)

    @property
    def structurally_identified_model_calls(self) -> int:
        return sum(
            trace.structurally_identified_model_calls for trace in self.traces
        )

    @property
    def structural_model_call_coverage(self) -> float:
        if not self.model_calls:
            return 0.0
        return self.structurally_identified_model_calls / self.model_calls

    @property
    def recursive_model_calls(self) -> int:
        return sum(trace.recursive_model_calls for trace in self.traces)

    @property
    def exactly_parented_recursive_model_calls(self) -> int:
        return sum(
            trace.exactly_parented_recursive_model_calls for trace in self.traces
        )

    @property
    def exact_recursive_parent_coverage(self) -> float:
        if not self.recursive_model_calls:
            return 0.0
        return (
            self.exactly_parented_recursive_model_calls
            / self.recursive_model_calls
        )

    @property
    def cross_component_fallback_rate(self) -> float:
        if not self.recursive_components:
            return 0.0
        return self.cross_component_fallbacks / self.recursive_components

    @property
    def ready_for_representative_raf(self) -> bool:
        return (
            bool(self.traces)
            and self.exact_prompt_provenance_coverage == 1.0
            and self.structural_model_call_coverage == 1.0
            and self.exact_recursive_parent_coverage == 1.0
            and self.cross_component_fallbacks == 0
            and all(
                not trace.unresolved_cross_component_links
                for trace in self.traces
            )
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "trace_count": len(self.traces),
            "rendered_prompt_tokens": self.rendered_prompt_tokens,
            "mapped_prompt_tokens": self.mapped_prompt_tokens,
            "exact_prompt_provenance_coverage": (
                self.exact_prompt_provenance_coverage
            ),
            "recursive_components": self.recursive_components,
            "model_calls": self.model_calls,
            "structurally_identified_model_calls": (
                self.structurally_identified_model_calls
            ),
            "structural_model_call_coverage": (
                self.structural_model_call_coverage
            ),
            "recursive_model_calls": self.recursive_model_calls,
            "exactly_parented_recursive_model_calls": (
                self.exactly_parented_recursive_model_calls
            ),
            "exact_recursive_parent_coverage": (
                self.exact_recursive_parent_coverage
            ),
            "exact_cross_component_links": self.exact_cross_component_links,
            "cross_component_fallbacks": self.cross_component_fallbacks,
            "cross_component_fallback_rate": (
                self.cross_component_fallback_rate
            ),
            "ready_for_representative_raf": self.ready_for_representative_raf,
            "traces": [trace.as_dict() for trace in self.traces],
        }


def import_trace_file(path: Path) -> ProvenanceFileReport:
    """Import every trace while retaining only locally verifiable claims."""
    traces = tuple(
        import_trace(trace, source_bytes=len(canonical_json(trace)))
        for trace in load_trace_records(path)
    )
    return ProvenanceFileReport(
        schema_version=1,
        source=path.as_posix(),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        traces=traces,
    )


def import_trace(
    trace: dict[str, Any],
    *,
    source_bytes: int = 0,
) -> TraceProvenanceResult:
    """Build exact within-component provenance and conservative call fallbacks."""
    trace_id = str(trace.get("id") or "unknown")
    nodes = _object_list(trace.get("nodes"), "nodes")
    raw_calls = _object_list(trace.get("calls"), "calls")
    calls = extract_policy_calls(trace)
    if len(calls) != len(raw_calls):
        raise ValueError("every model call must link to a message node")

    graph = EventGraph()
    roots = _component_roots(nodes)
    primary_root = min(roots) if roots else -1
    component_by_node = {
        node_index: path_to_node(nodes, node_index)[0]
        for node_index in range(len(nodes))
    }
    for node_index, node in enumerate(nodes):
        tokens = _integer_list(node.get("token_ids"), f"node {node_index}")
        message = node.get("message")
        role = (
            str(message.get("role"))
            if isinstance(message, dict) and message.get("role")
            else "unknown"
        )
        graph.add_node(
            EventNode(
                _message_id(trace_id, node_index),
                EventNodeKind.ARTIFACT,
                {
                    "artifact_type": "rendered_message",
                    "component_root": component_by_node[node_index],
                    "role": role,
                    "sampled": node.get("sampled") is True,
                    "token_count": len(tokens),
                    "token_sha256": hashlib.sha256(
                        canonical_json(tokens)
                    ).hexdigest(),
                    "timestamp": _number_or_none(node.get("timestamp")),
                },
            )
        )
    for node_index, node in enumerate(nodes):
        parent = node.get("parent")
        if type(parent) is int:
            graph.add_edge(
                EventEdge(
                    _message_id(trace_id, parent),
                    _message_id(trace_id, node_index),
                    EdgeKind.CONTROL,
                    via="message_parent",
                )
            )

    policy_records: list[PolicyNodeRecord] = []
    rendered_prompt_tokens = 0
    mapped_prompt_tokens = 0
    exact_observation_edges = 0
    for call, raw_call in zip(calls, raw_calls, strict=True):
        policy_id = _policy_id(trace_id, call.call_index)
        component_root = component_by_node[call.node_index]
        kind = (
            PolicyNodeKind.ROOT_TURN
            if component_root == primary_root
            else PolicyNodeKind.SUBCALL_OUTPUT
        )
        graph.add_node(
            EventNode(
                policy_id,
                EventNodeKind.POLICY,
                {
                    "policy_kind": kind.value,
                    "call_index": call.call_index,
                    "node_index": call.node_index,
                    "checkpoint_id": call.checkpoint_id,
                    "decoding_config_hash": call.decoding_config_hash,
                    "event_seed": call.event_seed,
                    "agent_depth": call.agent_depth,
                    "session_id": call.session_id,
                    "turn_index": call.turn_index,
                    "call_kind": call.call_kind,
                    "parent_session_id": call.parent_session_id,
                    "parent_turn_index": call.parent_turn_index,
                    "prompt_tokens": call.prompt_tokens_reported,
                    "completion_tokens": call.completion_tokens_reported,
                    "wall_seconds": call.wall_seconds,
                    "start": _call_time(raw_call, "start"),
                    "end": _call_time(raw_call, "end"),
                },
            )
        )
        path = path_to_node(nodes, call.node_index)
        spans: list[PromptProvenanceSpan] = []
        offset = 0
        reconstructed: list[int] = []
        for ancestor_index in path[:-1]:
            ancestor_tokens = _integer_list(
                nodes[ancestor_index].get("token_ids"),
                f"node {ancestor_index}",
            )
            end = offset + len(ancestor_tokens)
            artifact_id = _message_id(trace_id, ancestor_index)
            if ancestor_tokens:
                spans.append(PromptProvenanceSpan(artifact_id, 0, offset, end))
                graph.add_edge(
                    EventEdge(
                        artifact_id,
                        policy_id,
                        EdgeKind.OBSERVATION,
                        via=f"prompt_tokens:{offset}:{end}",
                    )
                )
                exact_observation_edges += 1
            reconstructed.extend(ancestor_tokens)
            offset = end

        current = nodes[call.node_index]
        current_tokens = _integer_list(
            current.get("token_ids"),
            f"node {call.node_index}",
        )
        mask = _boolean_list(current.get("mask"), f"node {call.node_index}")
        prefix_end = next(
            (index for index, sampled in enumerate(mask) if sampled),
            len(mask),
        )
        prefix_tokens = current_tokens[:prefix_end]
        renderer_id = _renderer_id(trace_id, call.call_index)
        graph.add_node(
            EventNode(
                renderer_id,
                EventNodeKind.ARTIFACT,
                {
                    "artifact_type": "renderer_scaffold",
                    "token_count": len(prefix_tokens),
                    "token_sha256": hashlib.sha256(
                        canonical_json(prefix_tokens)
                    ).hexdigest(),
                },
            )
        )
        prefix_span_end = offset + len(prefix_tokens)
        if prefix_tokens:
            spans.append(
                PromptProvenanceSpan(
                    renderer_id,
                    0,
                    offset,
                    prefix_span_end,
                )
            )
            graph.add_edge(
                EventEdge(
                    renderer_id,
                    policy_id,
                    EdgeKind.OBSERVATION,
                    via=f"prompt_tokens:{offset}:{prefix_span_end}",
                )
            )
            exact_observation_edges += 1
        reconstructed.extend(prefix_tokens)
        if tuple(reconstructed) != call.prompt_token_ids:
            raise ValueError(
                f"call {trace_id}:{call.call_index} prompt reconstruction drift"
            )
        rendered_prompt_tokens += len(call.prompt_token_ids)
        mapped_prompt_tokens += sum(
            span.token_end - span.token_start for span in spans
        )

        if call.event_seed is None:
            raise ValueError(
                f"call {trace_id}:{call.call_index} lacks an event seed"
            )
        policy_records.append(
            PolicyNodeRecord(
                node_id=policy_id,
                kind=kind,
                observation=PolicyObservation(
                    call.prompt_token_ids,
                    tuple(spans),
                ),
                action_token_start=len(call.prompt_token_ids),
                action_token_end=(
                    len(call.prompt_token_ids) + len(call.action_token_ids)
                ),
                behavior_logprobs_ref=(
                    f"trace:{trace_id}:node:{call.node_index}:logprobs"
                ),
                checkpoint_id=call.checkpoint_id,
                decoding_config_hash=call.decoding_config_hash,
                state_snapshot_ref=f"trace:{trace_id}:recorded",
                event_seed=call.event_seed,
                target_status="recorded_not_committed",
            )
        )
        graph.add_edge(
            EventEdge(
                policy_id,
                _message_id(trace_id, call.node_index),
                EdgeKind.DATAFLOW,
                via="sampled_action_tokens",
            )
        )

    reward_id = f"{trace_id}:reward"
    graph.add_node(
        EventNode(
            reward_id,
            EventNodeKind.REWARD_RESOURCE,
            {
                "rewards": trace.get("rewards", {}),
                "trace_ok": trace.get("ok") is True,
            },
        )
    )
    for leaf_index in _leaf_indices(nodes):
        graph.add_edge(
            EventEdge(
                _message_id(trace_id, leaf_index),
                reward_id,
                EdgeKind.RESOURCE,
                via="conservative_terminal_component",
            )
        )

    exact_links, fallbacks, unresolved = _add_cross_component_links(
        graph,
        trace_id=trace_id,
        nodes=nodes,
        calls=calls,
        raw_calls=raw_calls,
        primary_root=primary_root,
        component_by_node=component_by_node,
    )
    trace_generation_seconds = _trace_generation_seconds(trace)
    return TraceProvenanceResult(
        trace_id=trace_id,
        graph=graph,
        policy_records=tuple(policy_records),
        cost=TraceCostLedger(
            prompt_tokens=sum(call.prompt_tokens_reported or 0 for call in calls),
            generated_tokens=sum(
                call.completion_tokens_reported or 0 for call in calls
            ),
            model_calls=len(calls),
            model_call_wall_seconds=sum(call.wall_seconds for call in calls),
            trace_generation_wall_seconds=trace_generation_seconds,
            judge_calls=0,
            storage_bytes=source_bytes,
        ),
        rendered_prompt_tokens=rendered_prompt_tokens,
        mapped_prompt_tokens=mapped_prompt_tokens,
        exact_observation_edges=exact_observation_edges,
        connected_components=len(roots),
        recursive_components=max(0, len(roots) - 1),
        structurally_identified_model_calls=sum(
            call.agent_depth is not None
            and call.session_id is not None
            and call.turn_index is not None
            and call.call_kind is not None
            for call in calls
        ),
        recursive_model_calls=sum(
            call.agent_depth is not None and call.agent_depth > 0
            for call in calls
        ),
        exactly_parented_recursive_model_calls=sum(
            call.agent_depth is not None
            and call.agent_depth > 0
            and call.parent_session_id is not None
            and call.parent_turn_index is not None
            for call in calls
        ),
        exact_cross_component_links=exact_links,
        cross_component_fallbacks=fallbacks,
        unresolved_cross_component_links=unresolved,
    )


def _add_cross_component_links(
    graph: EventGraph,
    *,
    trace_id: str,
    nodes: list[dict[str, Any]],
    calls: tuple[RecordedPolicyCall, ...],
    raw_calls: list[dict[str, Any]],
    primary_root: int,
    component_by_node: dict[int, int],
) -> tuple[int, int, int]:
    """Prefer exact session/turn edges; fall back conservatively when absent."""
    exact_components = 0
    fallback_components = 0
    unresolved_components = 0
    roots = sorted(set(component_by_node.values()))
    for root in roots:
        if root == primary_root:
            continue
        component_calls = [
            call
            for call in calls
            if component_by_node[call.node_index] == root
        ]
        exact_triplet = _exact_structural_triplet(calls, component_calls)
        if exact_triplet is not None:
            parent_call, child_return_call, return_call = exact_triplet
            exact_components += 1
            graph.add_edge(
                EventEdge(
                    _policy_id(trace_id, parent_call.call_index),
                    _message_id(trace_id, root),
                    EdgeKind.CALL,
                    via="cross_component_parent_session_turn",
                )
            )
            graph.add_edge(
                EventEdge(
                    _message_id(trace_id, child_return_call.node_index),
                    _policy_id(trace_id, return_call.call_index),
                    EdgeKind.CALL,
                    via="cross_component_return_session_turn",
                )
            )
            continue
        fallback_components += 1
        root_time = _number_or_none(nodes[root].get("timestamp"))
        component_nodes = {
            index
            for index, component in component_by_node.items()
            if component == root
        }
        component_leaves = [
            index
            for index in _leaf_indices(nodes)
            if index in component_nodes
        ]
        leaf_times = [
            timestamp
            for index in component_leaves
            if (
                timestamp := _number_or_none(nodes[index].get("timestamp"))
            )
            is not None
        ]
        component_end = max(leaf_times) if leaf_times else None
        prior_calls = [
            call
            for call, raw_call in zip(calls, raw_calls, strict=True)
            if (
                end := _call_time(raw_call, "end")
            )
            is not None
            and root_time is not None
            and end <= root_time
            and component_by_node[call.node_index] != root
        ]
        later_calls = [
            call
            for call, raw_call in zip(calls, raw_calls, strict=True)
            if (
                start := _call_time(raw_call, "start")
            )
            is not None
            and component_end is not None
            and start >= component_end
            and component_by_node[call.node_index] != root
        ]
        if not prior_calls or not later_calls or not component_leaves:
            unresolved_components += 1
            continue
        for call in prior_calls:
            graph.add_edge(
                EventEdge(
                    _policy_id(trace_id, call.call_index),
                    _message_id(trace_id, root),
                    EdgeKind.CALL,
                    via="cross_component_all_prior_temporal_fallback",
                )
            )
        for leaf in component_leaves:
            for call in later_calls:
                graph.add_edge(
                    EventEdge(
                        _message_id(trace_id, leaf),
                        _policy_id(trace_id, call.call_index),
                        EdgeKind.CALL,
                        via="cross_component_all_later_temporal_fallback",
                    )
                )
    return exact_components, fallback_components, unresolved_components


def _exact_structural_triplet(
    calls: tuple[RecordedPolicyCall, ...],
    component_calls: list[RecordedPolicyCall],
) -> tuple[
    RecordedPolicyCall,
    RecordedPolicyCall,
    RecordedPolicyCall,
] | None:
    if not component_calls:
        return None
    session_ids = {call.session_id for call in component_calls}
    if len(session_ids) != 1 or None in session_ids:
        return None
    parent_pairs = {
        (call.parent_session_id, call.parent_turn_index)
        for call in component_calls
    }
    if len(parent_pairs) != 1:
        return None
    parent_session_id, parent_turn = next(iter(parent_pairs))
    if parent_session_id is None or parent_turn is None:
        return None
    parent_matches = [
        call
        for call in calls
        if call.session_id == parent_session_id
        and call.turn_index == parent_turn
        and call.call_kind == "policy"
    ]
    return_matches = [
        call
        for call in calls
        if call.session_id == parent_session_id
        and call.turn_index == parent_turn + 1
        and call.call_kind == "policy"
    ]
    child_policy_calls = [
        call
        for call in component_calls
        if call.turn_index is not None and call.call_kind == "policy"
    ]
    if not child_policy_calls:
        return None
    last_child_turn = max(
        call.turn_index
        for call in child_policy_calls
        if call.turn_index is not None
    )
    child_return_matches = [
        call
        for call in child_policy_calls
        if call.turn_index == last_child_turn
    ]
    if (
        len(parent_matches) != 1
        or len(child_return_matches) != 1
        or len(return_matches) != 1
    ):
        return None
    return parent_matches[0], child_return_matches[0], return_matches[0]


def _component_roots(nodes: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, node in enumerate(nodes)
        if node.get("parent") is None
    ]


def _leaf_indices(nodes: list[dict[str, Any]]) -> list[int]:
    parents = {
        parent
        for node in nodes
        if type(parent := node.get("parent")) is int
    }
    return [index for index in range(len(nodes)) if index not in parents]


def _trace_generation_seconds(trace: dict[str, Any]) -> float:
    timing = trace.get("timing")
    if not isinstance(timing, dict):
        return 0.0
    generation = timing.get("generation")
    if not isinstance(generation, dict):
        return 0.0
    start = _number_or_none(generation.get("start"))
    end = _number_or_none(generation.get("end"))
    if start is None or end is None:
        return 0.0
    return max(0.0, end - start)


def _call_time(call: dict[str, Any], field: str) -> float | None:
    timing = call.get("time")
    if not isinstance(timing, dict):
        return None
    return _number_or_none(timing.get(field))


def _message_id(trace_id: str, node_index: int) -> str:
    return f"{trace_id}:message:{node_index}"


def _renderer_id(trace_id: str, call_index: int) -> str:
    return f"{trace_id}:renderer:{call_index}"


def _policy_id(trace_id: str, call_index: int) -> str:
    return f"{trace_id}:policy:{call_index}"


def _object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(
        not isinstance(item, dict) for item in value
    ):
        raise TypeError(f"{label} must be a list of objects")
    return value


def _integer_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise TypeError(f"{label} token_ids must be integers")
    return value


def _boolean_list(value: Any, label: str) -> list[bool]:
    if not isinstance(value, list) or any(type(item) is not bool for item in value):
        raise TypeError(f"{label} mask must be booleans")
    return value


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
