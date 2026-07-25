from __future__ import annotations

import pytest

from redco.contracts import PolicyNodeKind
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


def test_event_graph_tracks_typed_dependency_closure() -> None:
    graph = EventGraph()
    for node_id, kind in (
        ("policy", EventNodeKind.POLICY),
        ("artifact", EventNodeKind.ARTIFACT),
        ("operation", EventNodeKind.OPERATION),
        ("reward", EventNodeKind.REWARD_RESOURCE),
    ):
        graph.add_node(EventNode(node_id, kind))

    graph.add_edge(EventEdge("policy", "artifact", EdgeKind.DATAFLOW))
    graph.add_edge(EventEdge("artifact", "operation", EdgeKind.OBSERVATION))
    graph.add_edge(EventEdge("operation", "reward", EdgeKind.RESOURCE))

    assert graph.descendants("policy") == frozenset({"artifact", "operation", "reward"})
    assert graph.descendants(
        "policy",
        edge_kinds=frozenset({EdgeKind.DATAFLOW}),
    ) == frozenset({"artifact"})


def test_event_graph_rejects_cycles_without_mutating_graph() -> None:
    graph = EventGraph()
    graph.add_node(EventNode("a", EventNodeKind.OPERATION))
    graph.add_node(EventNode("b", EventNodeKind.OPERATION))
    graph.add_edge(EventEdge("a", "b", EdgeKind.CONTROL))

    with pytest.raises(ValueError, match="acyclic"):
        graph.add_edge(EventEdge("b", "a", EdgeKind.CONTROL))

    assert graph.edges == [EventEdge("a", "b", EdgeKind.CONTROL)]


def test_policy_observation_hashes_exact_tokens_and_checks_provenance() -> None:
    observation = PolicyObservation(
        prompt_token_ids=(1, 2, 3, 4),
        provenance_spans=(PromptProvenanceSpan("context", 2, 1, 3),),
    )
    record = PolicyNodeRecord(
        node_id="policy-1",
        kind=PolicyNodeKind.SUBCALL_OUTPUT,
        observation=observation,
        action_token_start=4,
        action_token_end=6,
        behavior_logprobs_ref="logprobs:1",
        checkpoint_id="theta-0",
        decoding_config_hash="decode:1",
        state_snapshot_ref="snapshot:1",
        event_seed=7,
        target_status="committed",
    )

    assert len(observation.token_ids_hash) == 64
    assert record.observation == observation
    assert observation.token_ids_hash != PolicyObservation((1, 2, 3, 5)).token_ids_hash


def test_prompt_provenance_rejects_non_actual_span() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        PolicyObservation(
            prompt_token_ids=(1, 2),
            provenance_spans=(PromptProvenanceSpan("context", 0, 1, 3),),
        )


def test_event_graph_serialization_is_stable() -> None:
    graph = EventGraph()
    graph.add_node(EventNode("b", EventNodeKind.ARTIFACT))
    graph.add_node(EventNode("a", EventNodeKind.POLICY))
    graph.add_edge(EventEdge("a", "b", EdgeKind.DATAFLOW, via="answer"))

    assert graph.as_dict() == {
        "nodes": [
            {"node_id": "a", "kind": "policy", "metadata": {}},
            {"node_id": "b", "kind": "artifact", "metadata": {}},
        ],
        "edges": [
            {
                "source": "a",
                "target": "b",
                "kind": "dataflow",
                "via": "answer",
            }
        ],
    }
