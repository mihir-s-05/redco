"""Disposable production-semantic reconstruction for the v12 audit."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from types import SimpleNamespace
from typing import Any, Literal, cast

from redco.analysis.stage_d_exact_action import BehaviorAction
from redco.analysis.stage_d_source_contracts import (
    DecisionProvenance,
    RolloutDecision,
    SourceRollout,
)
from redco.analysis.stage_d_source_producer import (
    _normalize_child_weights,
    derive_source_trace,
    verify_source_trace_semantics,
)
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress
from redco.analysis.stage_d_v12_audit_common import (
    _ABSENT,
    _SEMANTIC_RECONSTRUCTION_NAMES,
    _canonical,
    _mapping,
    _mapping_list,
    _message_audit,
    sha256_bytes,
)
from redco.analysis.stage_d_v12_audit_trace import _address_key
from redco.integrations.verifiers_trace_v2 import extract_v2_rlm_provenance


class _ArchivedReceiptVerifier:
    """Read-only receipt lookup over the disposable archive extraction."""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self._receipts: dict[tuple[bytes, str], dict[str, Any]] = {}
        for record in records:
            if record.get("record_kind") != "receipt":
                continue
            receipt = _mapping(record.get("body", {}).get("receipt"), "receipt")
            kind = receipt.get("receipt_kind")
            if not isinstance(kind, str):
                continue
            raw = _canonical(receipt)
            key = (raw, kind)
            if key in self._receipts:
                raise ValueError("terminal archive contains a duplicate receipt")
            self._receipts[key] = receipt

    def __call__(self, receipt: bytes, *, receipt_kind: str) -> Mapping[str, Any]:
        value = self._receipts.get((receipt, receipt_kind))
        if value is None:
            raise ValueError("receipt is not present in the authenticated archive")
        return value


def _semantic_check(
    name: str,
    result: Literal["pass", "fail"],
    *,
    detail: str,
    values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "name": name,
        "status": "reconstructed_on_disposable_copy",
        "result": result,
        "detail": detail,
    }
    if values:
        item["values"] = dict(values)
    return item


def _semantic_reconstruction(
    trace: Mapping[str, Any],
    calls: list[dict[str, Any]],
    action_map: Mapping[tuple[Any, ...], tuple[dict[str, Any], dict[str, Any], BehaviorAction]],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the production derive/semantic suffix on a disposable repair copy."""

    def failed(error: BaseException) -> dict[str, Any]:
        reason = f"{type(error).__name__}: {error}"
        return {
            "status": "fail",
            "checks": [
                _semantic_check(name, "fail", detail=reason)
                for name in _SEMANTIC_RECONSTRUCTION_NAMES
            ],
            "derived_values": {},
        }

    try:
        provenance = extract_v2_rlm_provenance(dict(trace))
        if len(provenance) != len(calls):
            raise ValueError("production provenance count does not match trace calls")
        receipt_verifier = _ArchivedReceiptVerifier(records)
        decisions: list[RolloutDecision] = []
        for call, record in zip(calls, provenance, strict=True):
            entry = action_map[_address_key(_mapping(call.get("rlm"), "trace call rlm"))]
            receipt, _envelope, behavior_action = entry
            reservation_bytes = receipt.get("reservation_receipt")
            completion_bytes = receipt.get("completion_receipt")
            if not isinstance(reservation_bytes, bytes) or not isinstance(completion_bytes, bytes):
                raise ValueError("durable action entry lacks receipt bytes")
            verified_provenance = DecisionProvenance.from_receipts(
                reservation_bytes,
                completion_bytes,
                verifier=receipt_verifier,
            )
            if verified_provenance.event_address != record.scientific_address:
                raise ValueError("durable receipt address differs from trace provenance")
            decisions.append(
                RolloutDecision(
                    verified_provenance.decision_id,
                    verified_provenance.event_address,
                    behavior_action,
                    verified_provenance.node_kind,
                    verified_provenance.target_id,
                    verified_provenance.target_ordinal,
                    Fraction(1),
                    verified_provenance,
                )
            )

        child_records = [record for record in provenance if record.depth == 1]
        child_parent_event: Any = None
        child_parent_tool_call_slot = 0
        if child_records:
            first_child = child_records[0]
            if (
                first_child.parent_lineage is None
                or first_child.parent_call_ordinal is None
                or first_child.parent_turn is None
                or first_child.parent_tool_call_slot is None
            ):
                raise ValueError("child provenance lacks its durable parent address")
            child_parent_event = PolicyEventAddress(
                0,
                first_child.parent_lineage,
                first_child.parent_call_ordinal,
                first_child.parent_turn,
                "policy",
            )
            child_parent_tool_call_slot = first_child.parent_tool_call_slot

        semantic_trace = json.loads(_canonical(trace))
        semantic_calls = _mapping_list(semantic_trace.get("calls"), "semantic calls")
        semantic_nodes = _mapping_list(semantic_trace.get("nodes"), "semantic nodes")
        if len(semantic_calls) != len(decisions):
            raise ValueError("disposable semantic call count changed")
        for call_copy, decision in zip(semantic_calls, decisions, strict=True):
            sampler = decision.action.key.sampler
            call_copy["sampling"] = {
                "temperature": sampler.temperature,
                "top_p": sampler.top_p,
                "reasoning_effort": None,
                "max_tokens": sampler.max_tokens,
                "parallel_tool_calls": False,
                "seed": sampler.seed,
                "tool_choice": sampler.tool_choice,
            }
            call_copy["usage"] = {
                "prompt_tokens": decision.action.prompt_tokens,
                "completion_tokens": decision.action.completion_tokens,
                "cached_input_tokens": None,
                "reasoning_tokens": None,
                "cost": None,
            }
            call_copy["error"] = None
            node_index = call_copy.get("node")
            if type(node_index) is not int or not 0 <= node_index < len(semantic_nodes):
                raise ValueError("disposable semantic call names an absent node")
            message_result = _message_audit(
                decision.action.message,
                semantic_nodes[node_index].get("message", _ABSENT),
            )
            if message_result["canonical_equal_under_current_finalizer"]:
                continue
            differences = message_result["normalized_differences"]
            if differences != [
                {
                    "pointer": "/content",
                    "left": {
                        "presence": "present-null",
                        "type": "null",
                        "sha256": sha256_bytes(_canonical(None)),
                    },
                    "right": {"presence": "absent"},
                    "reason": "presence_or_value_difference",
                }
            ]:
                raise ValueError("disposable semantic copy found an unapproved message divergence")
            semantic_nodes[node_index]["message"] = decision.action.message

        raw_episode = _canonical(
            {
                "id": semantic_trace.get("id"),
                "env": "redco_evidence_selection_v2",
                "ok": True,
                "errors": [],
                "traces": [semantic_trace],
            }
        )
        root_count = sum(record.depth == 0 for record in provenance)
        derived = derive_source_trace(
            raw_episode,
            decisions=decisions,
            strict_two_slot=False,
            child_parent_event=child_parent_event,
            child_parent_tool_call_slot=child_parent_tool_call_slot,
            root_policy_turn_count=root_count,
            maximum_eligible_root_policy_turn_count=4,
        )
        strict_derived = derive_source_trace(
            raw_episode,
            decisions=decisions,
            strict_two_slot=True,
            child_parent_event=child_parent_event,
            child_parent_tool_call_slot=child_parent_tool_call_slot,
            root_policy_turn_count=root_count,
            maximum_eligible_root_policy_turn_count=4,
        )
        normalized_decisions = _normalize_child_weights(
            decisions,
            len({decision.target_id for decision in decisions if decision.node_kind == "child"}),
        )
        source_stub = cast(
            SourceRollout,
            SimpleNamespace(
                rollout_id=derived.trace_id,
                reward=derived.reward,
                stock_sequences=derived.stock_sequences,
                stock_sequence_decision_ids=derived.stock_sequence_decision_ids,
                child_target_roster=derived.child_target_roster,
                decisions=normalized_decisions,
            ),
        )
        verify_source_trace_semantics(
            source_stub,
            raw_episode=raw_episode,
            strict_two_slot=True,
            child_parent_event=child_parent_event,
            child_parent_tool_call_slot=child_parent_tool_call_slot,
            root_policy_turn_count=root_count,
            maximum_eligible_root_policy_turn_count=4,
        )
        derived_values = {
            "trace_id": derived.trace_id,
            "reward": derived.reward,
            "stock_sequence_count": len(derived.stock_sequences),
            "stock_sequence_lengths": [
                len(sequence.token_ids) for sequence in derived.stock_sequences
            ],
            "roster_lengths": [len(roster) for roster in derived.stock_sequence_decision_ids],
            "child_target_roster_count": len(strict_derived.child_target_roster),
            "child_target_roster_sha256": sha256_bytes(
                _canonical(strict_derived.child_target_roster)
            ),
            "normalized_decision_count": len(normalized_decisions),
        }
        checks = [
            _semantic_check(
                "episode_schema_and_trace_contract",
                "pass",
                detail=(
                    "production derive_source_trace parsed the disposable episode "
                    "and pinned trace schema"
                ),
            ),
            _semantic_check(
                "deployed_parent_links",
                "pass",
                detail="production derive_source_trace verified deployed parent links",
            ),
            _semantic_check(
                "strict_scaffold_eligibility",
                "pass",
                detail="production strict scaffold verification completed",
                values={"eligible": True},
            ),
            _semantic_check(
                "sampled_node_call_bijection",
                "pass",
                detail="production derive_source_trace routed every sampled node exactly once",
            ),
            _semantic_check(
                "sampled_node_mask_shape",
                "pass",
                detail=(
                    "production derive_source_trace validated boolean masks and "
                    "aligned token/logprob streams"
                ),
            ),
            _semantic_check(
                "leaf_path_sample_derivation",
                "pass",
                detail=(
                    "production derive_source_trace reconstructed leaf paths and training sequences"
                ),
            ),
            _semantic_check(
                "exactly_once_sampled_node_routing",
                "pass",
                detail=(
                    "production derive_source_trace enforced one training route per sampled node"
                ),
            ),
            _semantic_check(
                "finite_reward_summation",
                "pass",
                detail=(
                    "production derive_source_trace validated finite reward "
                    "components and summed them"
                ),
            ),
            _semantic_check(
                "child_target_roster",
                "pass",
                detail="production derive_source_trace reconstructed structural child targets",
            ),
            _semantic_check(
                "graph_to_source_mappings",
                "pass",
                detail=(
                    "derived sequences and decision rosters were produced by the production helper"
                ),
            ),
            _semantic_check(
                "child_weight_normalization",
                "pass",
                detail="production _normalize_child_weights completed on disposable decisions",
            ),
            _semantic_check(
                "source_semantic_equivalence",
                "pass",
                detail=(
                    "production verify_source_trace_semantics completed on a disposable source stub"
                ),
            ),
        ]
        return {
            "status": "pass",
            "checks": checks,
            "derived_values": derived_values,
        }
    except Exception as error:
        return failed(error)
