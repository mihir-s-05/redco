"""Exact Stage D fixed-topology target eligibility and informativeness audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json
from redco.env.replay import ReplayMode
from redco.env.tracer import EventNodeKind
from redco.integrations.signed_subprocess import sign_payload
from redco.integrations.verifiers_provenance import import_trace_file
from redco.integrations.verifiers_trace import (
    RecordedPolicyCall,
    audit_trace_file,
    build_policy_cache,
    load_trace_records,
)

from .empirical_branch_replay import (
    build_replay_indices,
    execute_cached_arm,
)

MINIMUM_REWARD_RANGE = 0.05


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _policy_node_ids(trace: Any) -> dict[int, str]:
    result: dict[int, str] = {}
    for node in trace.graph.nodes.values():
        if node.kind is EventNodeKind.POLICY:
            call_index = node.metadata.get("call_index")
            if type(call_index) is int:
                result[call_index] = node.node_id
    return result


def _call_key_record(call: RecordedPolicyCall) -> dict[str, Any]:
    return {
        "call_index": call.call_index,
        "prompt_token_ids": list(call.prompt_token_ids),
        "action_token_ids": list(call.action_token_ids),
        "checkpoint_id": call.checkpoint_id,
        "decoding_config_hash": call.decoding_config_hash,
        "event_seed": call.event_seed,
    }


def _target_behavior_logprobs(
    raw_trace: dict[str, Any], target: RecordedPolicyCall
) -> list[float]:
    nodes = raw_trace.get("nodes")
    if not isinstance(nodes, list):
        raise TypeError("trace.nodes must be a list")
    node = nodes[target.node_index]
    values = node.get("logprobs")
    if not isinstance(values, list) or not values:
        raise ValueError("target node has no behavior logprobs")
    if any(not isinstance(value, (int, float)) for value in values):
        raise TypeError("target behavior logprobs must be numeric")
    return [float(value) for value in values]


def _snapshot(
    *,
    trace_path: Path,
    raw_trace: dict[str, Any],
    target: RecordedPolicyCall,
    target_node_id: str,
    calls: tuple[RecordedPolicyCall, ...],
) -> dict[str, Any]:
    task = (raw_trace.get("task") or {}).get("data") or {}
    paper = task.get("paper")
    if not isinstance(paper, str):
        raise TypeError("task paper must be present for exact restore")
    dataset_sha256 = task.get("snapshot_sha256")
    if not isinstance(dataset_sha256, str) or len(dataset_sha256) != 64:
        raise ValueError("task snapshot hash is missing")
    prefix = [
        _call_key_record(call)
        for call in calls
        if call.call_index < target.call_index
    ]
    snapshot = {
        "schema_version": 1,
        "restore_scope": (
            "fixed-topology policy-call cache plus content-addressed QASPER "
            "workspace; not arbitrary IPython heap serialization"
        ),
        "source_trace": {
            "path": trace_path.as_posix(),
            "sha256": _sha256(trace_path.read_bytes()),
            "trace_id": target.trace_id,
        },
        "task_workspace": {
            "dataset_sha256": dataset_sha256,
            "example_id": task.get("example_id"),
            "paper_id": task.get("paper_id"),
            "context_path": "/workspace/evidence_context.txt",
            "context_sha256": _sha256(paper.encode("utf-8")),
        },
        "selector": {
            "version": "first-depth-one-native-call-v1",
            "uses_only": [
                "agent_depth",
                "native_call_index",
                "parent_session_id",
                "parent_turn_index",
            ],
            "forbidden_inputs": [
                "action_token_ids",
                "behavior_logprobs",
                "reward",
                "reference_evidence",
            ],
            "maximum_targets_per_rollout": 1,
        },
        "target": {
            "structural_event_address": target_node_id,
            "native_call_index": target.call_index,
            "agent_depth": target.agent_depth,
            "parent_session_id": target.parent_session_id,
            "parent_turn_index": target.parent_turn_index,
            "prompt_token_ids": list(target.prompt_token_ids),
            "checkpoint_id": target.checkpoint_id,
            "decoding_config_hash": target.decoding_config_hash,
            "event_seed": target.event_seed,
        },
        "exact_prefix_policy_cache": prefix,
    }
    snapshot_bytes = canonical_json(snapshot)
    return {
        "encoding": "redco-canonical-json",
        "bytes": len(snapshot_bytes),
        "sha256": _sha256(snapshot_bytes),
        "payload": snapshot,
    }


def _original_replay(
    *,
    calls: tuple[RecordedPolicyCall, ...],
    trace: Any,
    target: RecordedPolicyCall,
    target_node_id: str,
    original_reward: float,
) -> dict[str, Any]:
    calls_by_index = {call.call_index: call for call in calls}
    roots = [
        call
        for call in calls
        if call.agent_depth == 0 and call.turn_index is not None
    ]
    if not roots:
        raise ValueError("trace has no returning root call")
    final = max(roots, key=lambda call: call.turn_index or 0)
    if final.event_seed is None:
        raise ValueError("final root call has no event seed")
    node_ids = _policy_node_ids(trace)
    full_indices, sliced_indices = build_replay_indices(
        target_call_index=target.call_index,
        target_node_id=target_node_id,
        policy_node_ids_by_call=node_ids,
        descendants=trace.graph.descendants(target_node_id),
    )
    cache = build_policy_cache(calls)
    kwargs = {
        "calls_by_index": calls_by_index,
        "final_call_index": final.call_index,
        "branch_final_prompt": final.prompt_token_ids,
        "branch_final_seed": final.event_seed,
        "branch_final_decoding_config_hash": final.decoding_config_hash,
        "branch_final_action": final.action_token_ids,
        "reward": original_reward,
    }
    full = execute_cached_arm(
        mode=ReplayMode.FULL_SUFFIX,
        visited_call_indices=full_indices,
        cache=cache.fork(),
        **kwargs,
    )
    sliced = execute_cached_arm(
        mode=ReplayMode.SLICED,
        visited_call_indices=sliced_indices,
        cache=cache.fork(),
        **kwargs,
    )
    exact = (
        full.terminal_action_sha256 == sliced.terminal_action_sha256
        and full.reward == sliced.reward == original_reward
        and len(full.exact_key_reused_call_indices)
        == len(full.visited_call_indices)
        and len(sliced.exact_key_reused_call_indices)
        == len(sliced.visited_call_indices)
    )
    return {
        "full": asdict(full),
        "sliced": asdict(sliced),
        "terminal_and_reward_exact": exact,
    }


def evaluate_target(
    *,
    trace_path: Path,
    replay_path: Path,
    scorer_path: Path,
) -> dict[str, Any]:
    native = load_trace_records(trace_path)
    audit = audit_trace_file(trace_path)
    provenance = import_trace_file(trace_path)
    if len(native) != 1 or len(provenance.traces) != 1:
        raise ValueError("target audit requires exactly one native trace")
    raw_trace = native[0]
    trace = provenance.traces[0]
    calls = tuple(sorted(audit.calls, key=lambda call: call.call_index))
    targets = [call for call in calls if call.agent_depth == 1]
    if not targets:
        return sign_payload(
            {
                "schema_version": 1,
                "analysis": "stage-d-target-eligibility",
                "trace_id": raw_trace.get("id"),
                "eligible": False,
                "informative": False,
                "joint_eligible_and_informative": False,
                "reason": "zero_depth_one_calls",
            }
        )
    # This is the committed structural selector. It does not inspect the target
    # action, score, or reference answer.
    target = targets[0]
    node_ids = _policy_node_ids(trace)
    target_node_id = node_ids[target.call_index]
    score = (
        ((raw_trace.get("info") or {}).get("evidence_selection") or {})
        .get("score")
        or {}
    )
    original_f1 = float(score["f1"])
    snapshot = _snapshot(
        trace_path=trace_path,
        raw_trace=raw_trace,
        target=target,
        target_node_id=target_node_id,
        calls=calls,
    )
    original_replay = _original_replay(
        calls=calls,
        trace=trace,
        target=target,
        target_node_id=target_node_id,
        original_reward=original_f1,
    )

    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    scorer = json.loads(scorer_path.read_text(encoding="utf-8"))
    pairs = [
        pair
        for pair in replay.get("pairs") or []
        if pair.get("target_call_index") == target.call_index
    ]
    scored = [
        pair
        for pair in scorer.get("pairs") or []
        if pair.get("target_call_index") == target.call_index
    ]
    scored_by_index = {
        int(pair["alternative_index"]): pair for pair in scored
    }
    alternative_f1 = [
        float(scored_by_index[int(pair["alternative_index"])]["f1"])
        for pair in pairs
        if int(pair["alternative_index"]) in scored_by_index
    ]
    rewards = [original_f1, *alternative_f1]
    reward_range = max(rewards) - min(rewards) if rewards else 0.0

    descendants = trace.graph.descendants(target_node_id)
    downstream_root = any(
        call.agent_depth == 0
        and call.call_index > target.call_index
        and node_ids[call.call_index] in descendants
        for call in calls
    )
    alternative_replay_exact = (
        len(pairs) == 3
        and len(scored) == 3
        and all(
            pair.get("terminal_artifacts_exact") is True
            and pair.get("rewards_exact") is True
            and pair.get("cached_actions_exact") is True
            for pair in pairs
        )
        and all(
            pair.get("alternative_distinct_from_original") is True
            and pair.get("downstream_prompt_changed") is True
            for pair in scored
        )
    )
    exact_fields = {
        "structural_event_address": bool(target_node_id),
        "prompt_token_ids": bool(target.prompt_token_ids),
        "checkpoint_id": target.checkpoint_id != "unknown",
        "decoding_config_hash": bool(target.decoding_config_hash),
        "event_seed": target.event_seed is not None,
        "behavior_logprobs": bool(
            _target_behavior_logprobs(raw_trace, target)
        ),
        "pre_call_snapshot_bytes_and_sha256": (
            snapshot["bytes"] > 0 and len(snapshot["sha256"]) == 64
        ),
        "downstream_provenance": downstream_root,
        "exact_original_full_and_sliced_replay": original_replay[
            "terminal_and_reward_exact"
        ],
        "exact_alternative_full_and_sliced_replay": alternative_replay_exact,
    }
    eligible = all(exact_fields.values())
    informative = (
        eligible
        and len(rewards) == 4
        and reward_range >= MINIMUM_REWARD_RANGE
        and len(set(rewards)) >= 2
    )
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-target-eligibility",
            "trace_id": raw_trace.get("id"),
            "example_id": (
                ((raw_trace.get("task") or {}).get("data") or {}).get(
                    "example_id"
                )
            ),
            "target_call_index": target.call_index,
            "target_node_id": target_node_id,
            "selector": "first-depth-one-native-call-v1",
            "snapshot": snapshot,
            "behavior_logprobs": _target_behavior_logprobs(
                raw_trace, target
            ),
            "original_replay": original_replay,
            "alternative_replay": {
                "pairs": len(pairs),
                "scored_pairs": len(scored),
                "exact": alternative_replay_exact,
            },
            "reward_informativeness": {
                "original_f1": original_f1,
                "alternative_f1": alternative_f1,
                "range": reward_range,
                "minimum_range": MINIMUM_REWARD_RANGE,
            },
            "exact_field_checks": exact_fields,
            "eligible": eligible,
            "informative": informative,
            "joint_eligible_and_informative": eligible and informative,
            "limitations": [
                (
                    "Restore is exact for the frozen fixed-topology policy-call "
                    "cache and content-addressed QASPER workspace; it is not a "
                    "generic serialized IPython heap."
                ),
                (
                    "Eligibility is therefore valid only while the shared "
                    "scaffold preserves the audited root-child-root topology."
                ),
            ],
        }
    )


def aggregate_support(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 64:
        raise ValueError("support aggregation requires exactly 64 rollouts")
    ids = [str(record.get("trace_id")) for record in records]
    if len(set(ids)) != 64:
        raise ValueError("support trace IDs must be unique")
    eligible = sum(record.get("eligible") is True for record in records)
    informative = sum(
        record.get("informative") is True for record in records
    )
    joint = sum(
        record.get("joint_eligible_and_informative") is True
        for record in records
    )
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-target-support-aggregate",
            "rollouts": 64,
            "eligible": eligible,
            "informative": informative,
            "joint_eligible_and_informative": joint,
            "required_joint_successes": 58,
            "eligible_probability": eligible / 64,
            "informative_probability_unconditional": informative / 64,
            "informative_probability_conditional_on_eligible": (
                informative / eligible if eligible else 0.0
            ),
            "joint_probability": joint / 64,
            "passes": joint >= 58,
        }
    )
