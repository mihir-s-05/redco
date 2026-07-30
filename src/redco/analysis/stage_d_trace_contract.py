from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RLMTraceContract:
    model_calls: int
    root_calls: int
    child_calls: int
    linked_child_calls: int
    sampled_call_nodes: int
    exact_token_call_nodes: int
    behavior_seeded_calls: int
    checkpoint_stamped: bool

    @property
    def trace_contract_passes(self) -> bool:
        return (
            self.root_calls >= 2
            and self.child_calls >= 2
            and self.linked_child_calls == self.child_calls
            and self.sampled_call_nodes == self.model_calls
            and self.exact_token_call_nodes == self.model_calls
            and self.behavior_seeded_calls == self.model_calls
        )

    @property
    def stage_d_science_ready(self) -> bool:
        return self.trace_contract_passes and self.checkpoint_stamped

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "trace_contract_passes": self.trace_contract_passes,
            "stage_d_science_ready": self.stage_d_science_ready,
        }


def audit_rlm_trace(trace: Mapping[str, Any]) -> RLMTraceContract:
    nodes = trace.get("nodes") or []
    calls = trace.get("calls") or []
    root_sessions = {
        structure.get("session_id")
        for call in calls
        if isinstance((structure := call.get("rlm") or {}).get("depth"), int)
        and structure["depth"] == 0
        and structure.get("session_id")
    }
    root_calls = child_calls = linked_children = 0
    sampled_nodes = exact_nodes = seeded = 0
    for call in calls:
        structure = call.get("rlm") or {}
        depth = structure.get("depth")
        if depth == 0:
            root_calls += 1
        elif isinstance(depth, int) and depth > 0:
            child_calls += 1
            if structure.get("parent_session_id") in root_sessions:
                linked_children += 1
        node_index = call.get("node")
        if isinstance(node_index, int) and 0 <= node_index < len(nodes):
            node = nodes[node_index]
            if node.get("sampled") is True:
                sampled_nodes += 1
            token_ids = node.get("token_ids") or []
            mask = node.get("mask") or []
            logprobs = node.get("logprobs") or []
            sampled_count = sum(value is True for value in mask)
            if (
                token_ids
                and len(token_ids) == len(mask)
                and sampled_count > 0
                and sampled_count == len(logprobs)
            ):
                exact_nodes += 1
        sampling = call.get("sampling") or {}
        seed = sampling.get("seed")
        if isinstance(seed, int) and seed >= 0:
            seeded += 1
    info = trace.get("info") or {}
    policy_version = info.get("policy_version")
    checkpoint = info.get("checkpoint_id")
    return RLMTraceContract(
        model_calls=len(calls),
        root_calls=root_calls,
        child_calls=child_calls,
        linked_child_calls=linked_children,
        sampled_call_nodes=sampled_nodes,
        exact_token_call_nodes=exact_nodes,
        behavior_seeded_calls=seeded,
        checkpoint_stamped=(
            isinstance(policy_version, int)
            or (isinstance(checkpoint, str)
            and bool(checkpoint))
        ),
    )
