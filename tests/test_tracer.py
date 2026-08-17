from __future__ import annotations

import pytest

from redco.contracts import PolicyNodeKind
from redco.env.tracer import (
    EdgeKind,
    EventEdge,
    EventGraph,
    EventNode,
    EventNodeKind,
    InformationChannelKind,
    InformationProvenanceSpan,
    PolicyNodeRecord,
    PolicyObservation,
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
    digest = "a" * 64
    observation = PolicyObservation(
        prompt_token_ids=(1, 2, 3, 4),
        provenance_spans=(
            InformationProvenanceSpan(
                "artifact-read",
                InformationChannelKind.DECLARED_READ,
                1,
                3,
                "json-v1",
                digest,
                digest,
                "explicit_projection",
                "context",
                2,
            ),
            InformationProvenanceSpan(
                "parent-turn",
                InformationChannelKind.AMBIENT_HISTORY,
                0,
                1,
                "chat-template-v1",
                digest,
                digest,
                "automatic_history",
            ),
        ),
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
            provenance_spans=(
                InformationProvenanceSpan(
                    "read",
                    InformationChannelKind.DECLARED_READ,
                    1,
                    3,
                    "json-v1",
                    "a" * 64,
                    "b" * 64,
                    "explicit_projection",
                    "context",
                    0,
                ),
            ),
        )

    with pytest.raises(ValueError, match="exceeds"):
        PolicyObservation(
            prompt_token_ids=(1, 2),
            provenance_spans=(
                InformationProvenanceSpan(
                    "stdout-event",
                    InformationChannelKind.AMBIENT_STDOUT,
                    1,
                    3,
                    "stdout-v1",
                    "a" * 64,
                    "b" * 64,
                    "automatic_stdout",
                ),
            ),
        )


def test_information_provenance_requires_declared_artifact_and_exact_hashes() -> None:
    with pytest.raises(ValueError, match="artifact identity"):
        InformationProvenanceSpan(
            "read",
            InformationChannelKind.DECLARED_READ,
            0,
            1,
            "json-v1",
            "a" * 64,
            "b" * 64,
            "explicit_projection",
        )
    with pytest.raises(ValueError, match="raw_content_hash"):
        InformationProvenanceSpan(
            "history",
            InformationChannelKind.AMBIENT_HISTORY,
            0,
            1,
            "chat-v1",
            "not-a-digest",
            "b" * 64,
            "automatic_history",
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
