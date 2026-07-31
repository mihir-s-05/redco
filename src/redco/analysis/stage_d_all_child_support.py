"""Estimator-clean all-child support audit for the Stage D successor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json
from redco.env.tracer import EventNodeKind
from redco.integrations.signed_subprocess import (
    sign_payload,
    verify_signed_payload,
)
from redco.integrations.verifiers_provenance import import_trace_file
from redco.integrations.verifiers_trace import audit_trace_file, load_trace_records

from .empirical_branch_replay import derive_branch_group_seeds
from .stage_d_target_eligibility import (
    MINIMUM_REWARD_RANGE,
    _call_key_record,
    _original_replay,
    _policy_node_ids,
    _target_behavior_logprobs,
)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _candidate_fields(call: Any, node_id: str) -> dict[str, Any]:
    """Return only fields available before the child action is sampled."""
    return {
        "structural_event_address": node_id,
        "native_call_index_diagnostic_only": call.call_index,
        "agent_depth": call.agent_depth,
        "session_id": call.session_id,
        "parent_session_id": call.parent_session_id,
        "parent_turn_index": call.parent_turn_index,
        "prompt_token_ids": list(call.prompt_token_ids),
        "checkpoint_id": call.checkpoint_id,
        "decoding_config_hash": call.decoding_config_hash,
        "event_seed": call.event_seed,
    }


def precommit_all_depth_one_targets(trace_path: Path) -> dict[str, Any]:
    """Commit the complete depth-one set before any branch alternatives exist."""
    native = load_trace_records(trace_path)
    audit = audit_trace_file(trace_path)
    provenance = import_trace_file(trace_path)
    if len(native) != 1 or len(provenance.traces) != 1:
        raise ValueError("all-child precommit requires exactly one trace")
    trace = provenance.traces[0]
    node_ids: dict[int, str] = {}
    for node in trace.graph.nodes.values():
        if node.kind is EventNodeKind.POLICY:
            call_index = node.metadata.get("call_index")
            if type(call_index) is int:
                node_ids[call_index] = node.node_id
    candidates = [
        _candidate_fields(call, node_ids[call.call_index])
        for call in audit.calls
        if call.agent_depth == 1
    ]
    for candidate in candidates:
        candidate["pre_action_rank_sha256"] = _sha256(
            {
                key: value
                for key, value in candidate.items()
                if key != "native_call_index_diagnostic_only"
            }
        )
    candidates.sort(key=lambda row: row["pre_action_rank_sha256"])
    count = len(candidates)
    for candidate in candidates:
        candidate["decision_unit_weight"] = {
            "numerator": 1,
            "denominator": count,
        }
    task = (native[0].get("task") or {}).get("data") or {}
    source_trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    candidate_set_sha256 = _sha256(candidates)
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-all-child-precommit",
            "trace_id": native[0].get("id"),
            "source_trace_sha256": source_trace_sha256,
            "paper_id": task.get("paper_id"),
            "selector": "all-depth-one-precommitted-v1",
            "selection_time": "after recorded rollout, before any branch alternative",
            "forbidden_selector_inputs": [
                "child action tokens or text",
                "behavior logprobs",
                "reference evidence",
                "reward",
                "branch alternatives",
                "downstream scores",
            ],
            "native_call_order_used_for_selection": False,
            "candidate_count": count,
            "candidate_set_sha256": candidate_set_sha256,
            "outer_decision_unit_weight_sum": {
                "numerator": count,
                "denominator": count if count else 1,
            },
            "candidates": candidates,
        }
    )


def verify_canonical_precommit(trace_path: Path, committed: dict[str, Any]) -> dict[str, Any]:
    """Verify that a precommit is the complete canonical set for this trace."""
    verify_signed_payload(committed)
    expected = precommit_all_depth_one_targets(trace_path)
    if canonical_json(committed) != canonical_json(expected):
        raise ValueError("precommit is not the canonical complete trace target set")
    if not 2 <= int(committed["candidate_count"]) <= 4:
        raise ValueError("precommit must contain two to four depth-one targets")
    return expected


def verify_replay_chain(
    *,
    trace_path: Path,
    committed: dict[str, Any],
    replay: dict[str, Any],
    master_seed: str,
) -> None:
    """Verify trace, target-set, index, and HMAC seed bindings."""
    verify_canonical_precommit(trace_path, committed)
    verify_signed_payload(replay)
    trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    if replay.get("source_trace_sha256") != trace_sha256:
        raise ValueError("replay source trace hash mismatch")
    if replay.get("precommit_signed_payload_sha256") != committed.get("signed_payload_sha256"):
        raise ValueError("replay precommit hash mismatch")
    if replay.get("candidate_set_sha256") != committed.get("candidate_set_sha256"):
        raise ValueError("replay candidate-set hash mismatch")
    if replay.get("master_seed_sha256") != hashlib.sha256(master_seed.encode("utf-8")).hexdigest():
        raise ValueError("replay master-seed commitment mismatch")
    expected_nodes = {
        str(candidate["structural_event_address"]) for candidate in committed["candidates"]
    }
    if set(replay.get("target_node_ids") or []) != expected_nodes:
        raise ValueError("replay target node set mismatch")
    expected_weights = [
        {
            "target_node_id": candidate["structural_event_address"],
            "weight": candidate["decision_unit_weight"],
        }
        for candidate in committed["candidates"]
    ]
    if replay.get("decision_unit_weights") != expected_weights:
        raise ValueError("replay decision-unit weights mismatch")
    if replay.get("status") == "deterministic_ineligible":
        return
    if int(replay.get("target_count", -1)) != len(expected_nodes):
        raise ValueError("replay target count mismatch")
    if int(replay.get("alternatives_per_target", -1)) != 3:
        raise ValueError("replay must use exactly three alternatives")
    originals = replay.get("regenerated_originals") or []
    pairs = replay.get("pairs") or []
    original_keys = [
        (int(row["target_call_index"]), str(row["target_node_id"])) for row in originals
    ]
    expected_keys = {
        (
            int(candidate["native_call_index_diagnostic_only"]),
            str(candidate["structural_event_address"]),
        )
        for candidate in committed["candidates"]
    }
    if len(original_keys) != len(expected_keys) or set(original_keys) != expected_keys:
        raise ValueError("replay regenerated-original target set mismatch")
    pair_keys = [
        (
            int(row["target_call_index"]),
            str(row["target_node_id"]),
            int(row["alternative_index"]),
        )
        for row in pairs
    ]
    expected_pair_keys = {
        (call_index, node_id, alternative_index)
        for call_index, node_id in expected_keys
        for alternative_index in (1, 2, 3)
    }
    if len(pair_keys) != len(expected_pair_keys) or set(pair_keys) != expected_pair_keys:
        raise ValueError("replay alternative indices or target set mismatch")
    raw = load_trace_records(trace_path)
    audit = audit_trace_file(trace_path)
    if len(raw) != 1:
        raise ValueError("seed audit requires exactly one trace")
    roots = [call for call in audit.calls if call.agent_depth == 0 and call.turn_index is not None]
    if not roots:
        raise ValueError("seed audit requires a returning root")
    final_root = max(roots, key=lambda call: call.turn_index or 0)
    for call_index, node_id in expected_keys:
        continuation, actions = derive_branch_group_seeds(
            master_seed=master_seed,
            rollout_id=str(raw[0]["id"]),
            target_node_id=node_id,
            final_turn_index=final_root.turn_index or 0,
            alternatives=3,
        )
        original = next(row for row in originals if int(row["target_call_index"]) == call_index)
        if int(original["continuation_seed"]) != continuation:
            raise ValueError("regenerated-original continuation seed mismatch")
        target_pairs = sorted(
            (row for row in pairs if int(row["target_call_index"]) == call_index),
            key=lambda row: int(row["alternative_index"]),
        )
        if [int(row["action_seed"]) for row in target_pairs] != list(actions):
            raise ValueError("alternative action seed derivation mismatch")
        if any(int(row["continuation_seed"]) != continuation for row in target_pairs):
            raise ValueError("alternative continuation seed mismatch")


def verify_scorer_chain(
    *,
    committed: dict[str, Any],
    replay: dict[str, Any],
    scorer: dict[str, Any],
) -> None:
    """Verify scorer provenance and exact one-to-one replay coverage."""
    verify_signed_payload(scorer)
    for key in (
        "source_trace_sha256",
        "precommit_signed_payload_sha256",
        "candidate_set_sha256",
    ):
        expected = (
            committed.get("signed_payload_sha256")
            if key == "precommit_signed_payload_sha256"
            else replay.get(key)
        )
        if scorer.get(key) != expected:
            raise ValueError(f"scorer {key} mismatch")
    if scorer.get("decision_unit_weights") != replay.get("decision_unit_weights"):
        raise ValueError("scorer decision-unit weights mismatch")
    if scorer.get("replay_signed_payload_sha256") != replay.get("signed_payload_sha256"):
        raise ValueError("scorer replay hash mismatch")
    if scorer.get("status") == "deterministic_ineligible":
        if replay.get("status") != "deterministic_ineligible":
            raise ValueError("scorer/replay status mismatch")
        return
    expected_originals = {
        (int(row["target_call_index"]), str(row["target_node_id"]))
        for row in replay.get("regenerated_originals") or []
    }
    actual_originals = [
        (int(row["target_call_index"]), str(row["target_node_id"]))
        for row in scorer.get("regenerated_originals") or []
    ]
    if (
        len(actual_originals) != len(expected_originals)
        or set(actual_originals) != expected_originals
    ):
        raise ValueError("scorer regenerated-original coverage mismatch")
    expected_pairs = {
        (
            int(row["target_call_index"]),
            str(row["target_node_id"]),
            int(row["alternative_index"]),
        )
        for row in replay.get("pairs") or []
    }
    actual_pairs = [
        (
            int(row["target_call_index"]),
            str(row["target_node_id"]),
            int(row["alternative_index"]),
        )
        for row in scorer.get("pairs") or []
    ]
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        raise ValueError("scorer alternative coverage mismatch")


def _snapshot_all_child(
    *,
    trace_path: Path,
    raw_trace: dict[str, Any],
    target: Any,
    target_node_id: str,
    calls: tuple[Any, ...],
) -> dict[str, Any]:
    task = (raw_trace.get("task") or {}).get("data") or {}
    paper = task.get("paper")
    if not isinstance(paper, str):
        raise TypeError("task paper must be present for exact restore")
    dataset_sha256 = task.get("snapshot_sha256")
    if not isinstance(dataset_sha256, str) or len(dataset_sha256) != 64:
        raise ValueError("task snapshot hash is missing")
    prefix = [_call_key_record(call) for call in calls if call.call_index < target.call_index]
    snapshot = {
        "schema_version": 1,
        "scope": (
            "trace-indexed fixed-topology prompt-splice and exact-key cache "
            "capsule; this is provenance metadata, not an environment snapshot"
        ),
        "source_trace": {
            "path": trace_path.as_posix(),
            "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "trace_id": target.trace_id,
        },
        "task_workspace": {
            "dataset_sha256": dataset_sha256,
            "example_id": task.get("example_id"),
            "paper_id": task.get("paper_id"),
            "context_path": "/workspace/evidence_context.txt",
            "context_sha256": hashlib.sha256(paper.encode("utf-8")).hexdigest(),
        },
        "selector": {
            "version": "all-depth-one-precommitted-v1",
            "uses_only": [
                "agent_depth",
                "session_id",
                "parent_session_id",
                "parent_turn_index",
                "prompt_token_ids",
                "checkpoint_id",
                "decoding_config_hash",
                "event_seed",
            ],
            "forbidden_inputs": [
                "action_token_ids",
                "behavior_logprobs",
                "reward",
                "reference_evidence",
                "branch_alternatives",
                "downstream_scores",
            ],
            "maximum_targets_per_rollout": None,
        },
        "target": {
            "structural_event_address": target_node_id,
            "native_call_index_diagnostic_only": target.call_index,
            "agent_depth": target.agent_depth,
            "session_id": target.session_id,
            "parent_session_id": target.parent_session_id,
            "parent_turn_index": target.parent_turn_index,
            "prompt_token_ids": list(target.prompt_token_ids),
            "checkpoint_id": target.checkpoint_id,
            "decoding_config_hash": target.decoding_config_hash,
            "event_seed": target.event_seed,
        },
        "exact_prefix_policy_cache": prefix,
    }
    encoded = canonical_json(snapshot)
    return {
        "encoding": "redco-canonical-json",
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "payload": snapshot,
    }


def _evaluate_precommitted_target(
    *,
    trace_path: Path,
    replay_path: Path,
    scorer_path: Path,
    target_call_index: int,
    decision_unit_weight: dict[str, int],
) -> dict[str, Any]:
    native = load_trace_records(trace_path)
    audit = audit_trace_file(trace_path)
    provenance = import_trace_file(trace_path)
    if len(native) != 1 or len(provenance.traces) != 1:
        raise ValueError("all-child target audit requires exactly one trace")
    raw_trace = native[0]
    trace = provenance.traces[0]
    calls = tuple(sorted(audit.calls, key=lambda call: call.call_index))
    matches = [
        call for call in calls if call.agent_depth == 1 and call.call_index == target_call_index
    ]
    if len(matches) != 1:
        raise ValueError("committed target is not a unique depth-one call")
    target = matches[0]
    roots = [call for call in calls if call.agent_depth == 0]
    children = [call for call in calls if call.agent_depth == 1]
    task = (raw_trace.get("task") or {}).get("data") or {}
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    scorer = json.loads(scorer_path.read_text(encoding="utf-8"))
    verify_signed_payload(replay)
    verify_signed_payload(scorer)
    if replay.get("status") == "deterministic_ineligible":
        return sign_payload(
            {
                "schema_version": 1,
                "analysis": "stage-d-all-child-target-eligibility",
                "trace_id": raw_trace.get("id"),
                "paper_id": task.get("paper_id"),
                "target_call_index": target_call_index,
                "decision_unit_weight": decision_unit_weight,
                "eligible": False,
                "informative": False,
                "joint_eligible_and_informative": False,
                "reason": replay.get("reason"),
            }
        )
    node_ids = _policy_node_ids(trace)
    target_node_id = node_ids[target.call_index]
    score = ((raw_trace.get("info") or {}).get("evidence_selection") or {}).get("score") or {}
    original_f1 = float(score["f1"])
    snapshot = _snapshot_all_child(
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
    scored_by_index = {int(pair["alternative_index"]): pair for pair in scored}
    original_arms = [
        arm
        for arm in replay.get("regenerated_originals") or []
        if arm.get("target_call_index") == target.call_index
    ]
    scored_originals = [
        arm
        for arm in scorer.get("regenerated_originals") or []
        if arm.get("target_call_index") == target.call_index
    ]
    alternative_f1 = [
        float(scored_by_index[int(pair["alternative_index"])]["f1"])
        for pair in pairs
        if int(pair["alternative_index"]) in scored_by_index
    ]
    regenerated_original_f1 = (
        float(scored_originals[0]["f1"]) if len(scored_originals) == 1 else None
    )
    rewards = (
        [regenerated_original_f1, *alternative_f1]
        if regenerated_original_f1 is not None
        else alternative_f1
    )
    reward_range = max(rewards) - min(rewards) if rewards else 0.0
    descendants = trace.graph.descendants(target_node_id)
    downstream_root = any(
        call.agent_depth == 0
        and call.call_index > target.call_index
        and node_ids[call.call_index] in descendants
        for call in calls
    )
    common_seed_exact = (
        len(original_arms) == 1
        and len(pairs) == 3
        and len(
            {
                int(original_arms[0]["continuation_seed"]),
                *(int(pair["continuation_seed"]) for pair in pairs),
            }
        )
        == 1
    )
    action_seeds_unique = len(pairs) == 3 and len({int(pair["action_seed"]) for pair in pairs}) == 3
    regenerated_original_exact = (
        len(original_arms) == 1
        and original_arms[0].get("terminal_artifacts_exact") is True
        and original_arms[0].get("rewards_exact") is True
        and original_arms[0].get("cached_actions_exact") is True
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
    )
    original_output_valid = (
        len(scored_originals) == 1
        and scored_originals[0].get("parseable") is True
        and scored_originals[0].get("all_predicted_spans_verbatim") == 1.0
    )
    alternatives_output_valid = len(scored) == 3 and all(
        pair.get("parseable") is True and pair.get("all_predicted_spans_verbatim") == 1.0
        for pair in scored
    )
    trace_output_valid = (
        score.get("parseable") == 1.0 and score.get("all_predicted_spans_verbatim") == 1.0
    )
    exact_fields = {
        "trace_completed_ok": raw_trace.get("ok") is True,
        "at_least_two_root_calls": len(roots) >= 2,
        "at_least_two_depth_one_children": len(children) >= 2,
        "at_most_four_depth_one_children": len(children) <= 4,
        "at_most_eight_policy_calls": len(calls) <= 8,
        "maximum_agent_depth_one": all(
            call.agent_depth is not None and call.agent_depth <= 1 for call in calls
        ),
        "completion_tokens_at_most_4096": sum(call.completion_tokens_reported for call in calls)
        <= 4096,
        "structural_event_address": bool(target_node_id),
        "prompt_token_ids": bool(target.prompt_token_ids),
        "checkpoint_id": target.checkpoint_id != "unknown",
        "decoding_config_hash": bool(target.decoding_config_hash),
        "event_seed": target.event_seed is not None,
        "behavior_logprobs": bool(_target_behavior_logprobs(raw_trace, target)),
        "trace_replay_capsule_bytes_and_sha256": (
            snapshot["bytes"] > 0 and len(snapshot["sha256"]) == 64
        ),
        "downstream_provenance": downstream_root,
        "recorded_trace_cache_pairing": original_replay["terminal_and_reward_exact"],
        "regenerated_original_full_and_sliced_replay": regenerated_original_exact,
        "exact_alternative_full_and_sliced_replay": alternative_replay_exact,
        "common_downstream_seed_across_K4": common_seed_exact,
        "three_unique_action_seeds": action_seeds_unique,
        "recorded_trace_output_valid": trace_output_valid,
        "regenerated_original_output_valid": original_output_valid,
        "all_alternative_outputs_valid": alternatives_output_valid,
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
            "analysis": "stage-d-all-child-target-eligibility",
            "trace_id": raw_trace.get("id"),
            "example_id": task.get("example_id"),
            "paper_id": task.get("paper_id"),
            "answer_type": task.get("answer_type"),
            "root_calls": len(roots),
            "child_calls": len(children),
            "target_call_index": target.call_index,
            "target_node_id": target_node_id,
            "decision_unit_weight": decision_unit_weight,
            "selector": "all-depth-one-precommitted-v1",
            "topology_profile": "bounded-variable-root-v1",
            "trace_replay_capsule": snapshot,
            "behavior_logprobs": _target_behavior_logprobs(raw_trace, target),
            "recorded_trace_cache_pairing": original_replay,
            "regenerated_original": {
                "arm": original_arms[0] if len(original_arms) == 1 else None,
                "score": (scored_originals[0] if len(scored_originals) == 1 else None),
            },
            "alternative_replay": {
                "pairs": len(pairs),
                "scored_pairs": len(scored),
                "exact": alternative_replay_exact,
            },
            "reward_informativeness": {
                "recorded_original_f1_diagnostic_only": original_f1,
                "regenerated_original_f1": regenerated_original_f1,
                "alternative_f1": alternative_f1,
                "range": reward_range,
                "minimum_range": MINIMUM_REWARD_RANGE,
            },
            "exact_field_checks": exact_fields,
            "eligible": eligible,
            "informative": informative,
            "joint_eligible_and_informative": eligible and informative,
        }
    )


def evaluate_all_precommitted_targets(
    *,
    trace_path: Path,
    replay_path: Path,
    scorer_path: Path,
    precommit: dict[str, Any] | None = None,
    master_seed: str,
) -> dict[str, Any]:
    """Evaluate every committed target; never filter on informativeness."""
    committed = precommit or precommit_all_depth_one_targets(trace_path)
    verify_canonical_precommit(trace_path, committed)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    scorer = json.loads(scorer_path.read_text(encoding="utf-8"))
    verify_replay_chain(
        trace_path=trace_path,
        committed=committed,
        replay=replay,
        master_seed=master_seed,
    )
    verify_scorer_chain(
        committed=committed,
        replay=replay,
        scorer=scorer,
    )
    reports = [
        _evaluate_precommitted_target(
            trace_path=trace_path,
            replay_path=replay_path,
            scorer_path=scorer_path,
            target_call_index=int(candidate["native_call_index_diagnostic_only"]),
            decision_unit_weight=dict(candidate["decision_unit_weight"]),
        )
        for candidate in committed["candidates"]
    ]
    for report in reports:
        verify_signed_payload(report)
    eligible = sum(report.get("eligible") is True for report in reports)
    informative = sum(report.get("informative") is True for report in reports)
    joint = sum(report.get("joint_eligible_and_informative") is True for report in reports)
    candidate_count = len(reports)
    weights = [report["decision_unit_weight"] for report in reports]
    exact_weight_contract = all(
        int(weight["numerator"]) == 1 and int(weight["denominator"]) == candidate_count
        for weight in weights
    )
    outer_weight_sum = {
        "numerator": sum(int(weight["numerator"]) for weight in weights),
        "denominator": candidate_count,
    }
    all_committed_targets_eligible = eligible == candidate_count
    paper_joint_pass = (
        2 <= candidate_count <= 4
        and all_committed_targets_eligible
        and joint >= 1
        and exact_weight_contract
        and outer_weight_sum["numerator"] == outer_weight_sum["denominator"]
    )
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-all-child-target-audit",
            "trace_id": committed.get("trace_id"),
            "paper_id": committed.get("paper_id"),
            "source_trace_sha256": committed["source_trace_sha256"],
            "precommit_signed_payload_sha256": committed["signed_payload_sha256"],
            "candidate_set_sha256": committed["candidate_set_sha256"],
            "replay_signed_payload_sha256": replay["signed_payload_sha256"],
            "scorer_signed_payload_sha256": scorer["signed_payload_sha256"],
            "candidate_count": candidate_count,
            "eligible_target_count": eligible,
            "informative_target_count": informative,
            "joint_target_count": joint,
            "all_committed_targets_eligible": all_committed_targets_eligible,
            "paper_has_two_eligible_targets": eligible >= 2,
            "paper_has_any_joint_target": joint >= 1,
            "exact_decision_unit_weight_contract": exact_weight_contract,
            "outer_decision_unit_weight_sum": outer_weight_sum,
            "paper_joint_pass": paper_joint_pass,
            "flat_groups_retained": informative < eligible,
            "target_reports": reports,
        }
    )


def aggregate_paper_support(
    records: list[dict[str, Any]],
    *,
    required_paper_successes: int = 58,
) -> dict[str, Any]:
    """Aggregate nested target observations at the independent paper unit."""
    if len(records) != 64:
        raise ValueError("paper support requires exactly 64 records")
    if not 1 <= required_paper_successes <= 64:
        raise ValueError("required paper successes must be in [1, 64]")
    for record in records:
        verify_signed_payload(record)
    paper_ids = [str(record.get("paper_id")) for record in records]
    if len(set(paper_ids)) != 64:
        raise ValueError("paper support requires 64 unique paper IDs")
    candidate_counts = [int(record.get("candidate_count", 0)) for record in records]
    if not all(
        record.get("exact_decision_unit_weight_contract") is True
        and record.get("outer_decision_unit_weight_sum")
        == {
            "numerator": int(record.get("candidate_count", 0)),
            "denominator": int(record.get("candidate_count", 0)),
        }
        and not (
            record.get("paper_joint_pass") is True
            and record.get("all_committed_targets_eligible") is not True
        )
        for record in records
    ):
        raise ValueError("paper support contains broken target weights or skips")
    eligible_targets = sum(int(record.get("eligible_target_count", 0)) for record in records)
    informative_targets = sum(int(record.get("informative_target_count", 0)) for record in records)
    paper_successes = sum(record.get("paper_joint_pass") is True for record in records)
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-all-child-paper-support-aggregate",
            "independent_unit": "unique paper",
            "papers": 64,
            "paper_successes": paper_successes,
            "required_paper_successes": required_paper_successes,
            "total_precommitted_targets": sum(candidate_counts),
            "eligible_targets_nested_diagnostic": eligible_targets,
            "informative_targets_nested_diagnostic": informative_targets,
            "target_level_counts_inferential_n": False,
            "decision_unit_weighting": (
                "Each paper has outer weight one; its precommitted child "
                "targets split that weight equally."
            ),
            "passes": paper_successes >= required_paper_successes,
        }
    )
