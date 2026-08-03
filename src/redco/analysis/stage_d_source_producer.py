"""Fail-closed Stage-D source production from intercepted calls and Verifiers traces."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.analysis.stage_d_receipt_ledger import (
    RecordedActionReservation,
    SourcePolicyCallReservation,
    StageDReceiptLedger,
)
from redco.analysis.stage_d_source_contracts import (
    DecisionProvenance,
    FrozenTrainingSequence,
    RolloutDecision,
    SourceRollout,
)
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress
from redco.contracts import canonical_json
from redco.integrations.verifiers_trace_v2 import (
    RecordedRLMProvenanceV2,
    extract_v2_rlm_provenance,
)

_EPISODE_FIELDS = {"id", "env", "ok", "errors", "traces"}
_TRACE_FIELDS = {
    "id",
    "task",
    "runtime",
    "version",
    "verifiers",
    "run",
    "agent",
    "nodes",
    "tools",
    "calls",
    "rewards",
    "metrics",
    "info",
    "extra_usage",
    "is_completed",
    "ok",
    "stop_condition",
    "errors",
    "timing",
}
_NODE_FIELDS = {
    "parent",
    "message",
    "sampled",
    "timestamp",
    "token_ids",
    "mask",
    "is_content",
    "logprobs",
}
_CALL_FIELDS = {
    "endpoint",
    "error",
    "finish_reason",
    "model",
    "node",
    "rlm",
    "sampling",
    "time",
    "usage",
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def structural_child_target_id(
    parent_event: PolicyEventAddress,
    *,
    rollout_id: str,
    parent_tool_call_slot: int,
    spawn_ordinal: int,
) -> str:
    """Identify a child slot without transport IDs or completion order."""
    if parent_event.call_kind != "policy":
        raise ValueError("child targets require a parent policy event")
    if not rollout_id:
        raise ValueError("child targets require a rollout ID")
    if parent_tool_call_slot < 0 or spawn_ordinal < 0:
        raise ValueError("child structural slots must be nonnegative")
    return (
        "child-slot-"
        + _sha256(
            canonical_json(
                {
                    "domain": "redco-stage-d-child-target-v2",
                    "rollout_id": rollout_id,
                    "parent_event": parent_event.as_payload(),
                    "parent_tool_call_slot": parent_tool_call_slot,
                    "spawn_ordinal": spawn_ordinal,
                }
            )
        )[:24]
    )


@dataclass(frozen=True, slots=True)
class PendingSourcePolicyCall:
    """A durably reserved request plus outcome-independent decision metadata."""

    reservation: SourcePolicyCallReservation
    event_address: PolicyEventAddress
    node_kind: Literal["root", "child"]
    target_id: str | None
    target_ordinal: int | None
    outer_weight: Fraction
    recorded_action_reservation: RecordedActionReservation | None
    raw_response_required: bool


@dataclass(frozen=True, slots=True)
class DerivedTraceSource:
    """The only source fields permitted to cross the Verifiers-to-Prime seam."""

    trace_id: str
    reward: float
    stock_sequences: tuple[FrozenTrainingSequence, ...]
    stock_sequence_decision_ids: tuple[tuple[str, ...], ...]
    child_target_roster: tuple[str, ...]


class SourceTopologyIneligible(ValueError):
    """A completed natural rollout that misses the branchable scaffold contract."""


class StageDSourceRolloutProducer:
    """Capture requests before forwarding, then derive all trainer bytes from the trace."""

    def __init__(
        self,
        *,
        ledger: StageDReceiptLedger,
        group_id: str,
        rollout_id: str,
        child_parent_event: PolicyEventAddress | None = None,
        child_parent_tool_call_slot: int = 0,
        root_policy_turn_count: int | None = None,
        child_target_roster: Sequence[str] | None = None,
        allow_test_fixture_roster: bool = False,
        maximum_eligible_root_policy_turn_count: int = 4,
        maximum_captured_session_call_count: int = 8,
        base_model_manifest_sha256: str,
    ) -> None:
        if not group_id or not rollout_id:
            raise ValueError("source producer identifiers must be nonempty")
        if allow_test_fixture_roster:
            if child_parent_event is not None or child_target_roster is None:
                raise ValueError("fixture rosters cannot mix with structural targeting")
            if root_policy_turn_count is not None:
                raise ValueError("fixture rosters cannot freeze a live root-turn count")
            roster = tuple(child_target_roster)
            strict_two_slot = False
        else:
            if child_parent_event is None or child_target_roster is not None:
                raise ValueError("live source production requires one structural parent")
            if type(root_policy_turn_count) is not int or root_policy_turn_count < 2:
                raise ValueError("live source production requires at least two root turns")
            roster = tuple(
                structural_child_target_id(
                    child_parent_event,
                    rollout_id=rollout_id,
                    parent_tool_call_slot=child_parent_tool_call_slot,
                    spawn_ordinal=ordinal,
                )
                for ordinal in (0, 1)
            )
            strict_two_slot = True
        if (
            type(maximum_eligible_root_policy_turn_count) is not int
            or maximum_eligible_root_policy_turn_count < 2
        ):
            raise ValueError("source producer root eligibility ceiling is invalid")
        if (
            type(maximum_captured_session_call_count) is not int
            or maximum_captured_session_call_count
            < maximum_eligible_root_policy_turn_count
        ):
            raise ValueError("source producer capture ceiling is invalid")
        if not roster or len(set(roster)) != len(roster) or any(not item for item in roster):
            raise ValueError("source producer child roster must be nonempty and unique")
        if len(base_model_manifest_sha256) != 64:
            raise ValueError("source producer base-model manifest must be SHA-256")
        self._ledger = ledger
        self._group_id = group_id
        self._rollout_id = rollout_id
        self._child_target_roster = roster
        self._child_target_ordinals = {
            target_id: ordinal for ordinal, target_id in enumerate(roster)
        }
        self._child_parent_event = child_parent_event
        self._child_parent_tool_call_slot = child_parent_tool_call_slot
        self._strict_two_slot = strict_two_slot
        self._root_policy_turn_count = root_policy_turn_count
        self._maximum_eligible_root_policy_turn_count = (
            maximum_eligible_root_policy_turn_count
        )
        self._maximum_captured_session_call_count = (
            maximum_captured_session_call_count
        )
        self._base_model_manifest_sha256 = base_model_manifest_sha256
        self._pending: dict[str, PendingSourcePolicyCall] = {}
        self._observed_responses: set[str] = set()
        self._completed: dict[str, RolloutDecision] = {}
        self._aborted = False
        self._rollout_completed = False

    def is_predeclared_child_target(self, target_id: str) -> bool:
        """Whether a child belongs to the outcome-independent two-slot target roster."""
        return target_id in self._child_target_roster

    def reserve_policy_call(
        self,
        *,
        event_address: PolicyEventAddress,
        action_key: ExactActionKey,
        node_kind: Literal["root", "child"],
        target_id: str | None,
        branch_selected: bool,
        raw_response_required: bool = False,
        recorded_action_reservation: RecordedActionReservation | None = None,
    ) -> PendingSourcePolicyCall:
        """Persist the exact request before the caller is allowed to send it."""
        target_ordinal, outer_weight = self._validate_policy_call(
            event_address=event_address,
            action_key=action_key,
            node_kind=node_kind,
            target_id=target_id,
        )
        decision_id = _decision_id(event_address)
        if decision_id in self._pending or decision_id in self._completed:
            raise ValueError("source policy address was already observed")
        request_sha256 = self._ledger.put_evidence(action_key.request)
        reservation = self._ledger.reserve_source_policy_call(
            group_id=self._group_id,
            rollout_id=self._rollout_id,
            decision_id=decision_id,
            node_kind=node_kind,
            target_id=target_id,
            target_ordinal=target_ordinal,
            target_address=event_address,
            recorded_action_key=action_key,
            request_sha256=request_sha256,
            branch_selected=branch_selected,
            raw_response_required=raw_response_required,
            recorded_action_reservation=recorded_action_reservation,
        )
        if recorded_action_reservation is not None:
            self._ledger.mark_recorded_action_model_call_started(
                recorded_action_reservation,
                request_sha256=request_sha256,
            )
        pending = PendingSourcePolicyCall(
            reservation,
            event_address,
            node_kind,
            target_id,
            target_ordinal,
            outer_weight,
            recorded_action_reservation,
            raw_response_required,
        )
        self._pending[decision_id] = pending
        return pending

    def _validate_policy_call(
        self,
        *,
        event_address: PolicyEventAddress,
        action_key: ExactActionKey,
        node_kind: Literal["root", "child"],
        target_id: str | None,
    ) -> tuple[int | None, Fraction]:
        """Validate every mutation-independent source-call field."""
        if type(event_address) is not PolicyEventAddress:
            raise ValueError("source policy address must be structural")
        if type(action_key) is not ExactActionKey:
            raise ValueError("source policy request requires an exact action key")
        if self._strict_two_slot and action_key.prepared_engine_request is None:
            raise ValueError("live source calls require the exact prepared engine request")
        if node_kind == "root":
            if target_id is not None:
                raise ValueError("root source decisions cannot name child targets")
            target_ordinal = None
            outer_weight = Fraction(1)
        elif node_kind == "child":
            if target_id is None:
                raise ValueError("child source decision lacks a structural target")
            if target_id in self._child_target_roster:
                target_ordinal = self._child_target_roster.index(target_id)
            else:
                target_ordinal = self._observed_child_ordinal(target_id)
            outer_weight = (
                Fraction(1, len(self._child_target_roster))
                if target_id in self._child_target_roster
                else Fraction(1)
            )
        else:
            raise ValueError("source policy node kind must be root or child")
        return target_ordinal, outer_weight

    def _observed_child_ordinal(self, target_id: str) -> int:
        if self._child_parent_event is None:
            raise ValueError("child source decision lacks a structural parent")
        for parent_ordinal in range(self._maximum_captured_session_call_count):
            parent = PolicyEventAddress(
                0,
                self._child_parent_event.lineage,
                parent_ordinal,
                parent_ordinal,
                "policy",
            )
            for spawn_ordinal in range(4):
                expected = structural_child_target_id(
                    parent,
                    rollout_id=self._rollout_id,
                    parent_tool_call_slot=self._child_parent_tool_call_slot,
                    spawn_ordinal=spawn_ordinal,
                )
                if target_id == expected:
                    return self._child_target_ordinals.setdefault(
                        target_id,
                        len(self._child_target_ordinals),
                    )
        raise ValueError("child source decision is outside the bounded observed roster")

    def reserve_selected_child_policy_call(
        self,
        *,
        event_address: PolicyEventAddress,
        target_id: str,
        action_key: ExactActionKey,
        pre_action_snapshot: bytes,
        branch_count: int,
        continuation_replicates: int,
        failure_reward: float,
        raw_response_required: bool = False,
    ) -> PendingSourcePolicyCall:
        """Commit and reserve one selected child before a single allowed POST."""
        self._validate_policy_call(
            event_address=event_address,
            action_key=action_key,
            node_kind="child",
            target_id=target_id,
        )
        decision_id = _decision_id(event_address)
        if decision_id in self._pending or decision_id in self._completed:
            raise ValueError("source policy address was already observed")
        recorded = self.commit_child_target(
            event_address=event_address,
            target_id=target_id,
            action_key=action_key,
            pre_action_snapshot=pre_action_snapshot,
            branch_count=branch_count,
            continuation_replicates=continuation_replicates,
            failure_reward=failure_reward,
        )
        try:
            return self.reserve_policy_call(
                event_address=event_address,
                action_key=action_key,
                node_kind="child",
                target_id=target_id,
                branch_selected=True,
                raw_response_required=raw_response_required,
                recorded_action_reservation=recorded,
            )
        except BaseException as error:
            self._aborted = True
            error_evidence = canonical_json(
                {
                    "domain": "redco-stage-d-source-child-pre-post-abort-v1",
                    "error_type": type(error).__qualname__,
                    "error_message": str(error),
                }
            )
            try:
                error_sha256 = self._ledger.put_evidence(error_evidence)
                self._ledger.abort_source_child_before_post(
                    recorded,
                    rollout_id=self._rollout_id,
                    error_sha256=error_sha256,
                )
            except BaseException:
                # Preserve the triggering exception. The ledger transaction wrapper
                # already poisons the writer when durable abort recording itself fails.
                pass
            raise

    def commit_child_target(
        self,
        *,
        event_address: PolicyEventAddress,
        target_id: str,
        action_key: ExactActionKey,
        pre_action_snapshot: bytes,
        branch_count: int,
        continuation_replicates: int,
        failure_reward: float,
    ) -> RecordedActionReservation:
        """Commit one structural child target before its source action is sampled."""
        if target_id not in self._child_target_roster:
            raise ValueError("child target commitment is outside the frozen roster")
        if type(pre_action_snapshot) is not bytes or not pre_action_snapshot:
            raise ValueError("pre-action snapshot must be nonempty immutable bytes")
        target_ordinal = self._child_target_roster.index(target_id)
        snapshot_sha256 = self._ledger.put_evidence(pre_action_snapshot)
        return self._ledger.commit_pre_action_and_reserve(
            group_id=self._group_id,
            rollout_id=self._rollout_id,
            target_roster=self._child_target_roster,
            target_ordinal=target_ordinal,
            target_id=target_id,
            target_address=event_address,
            pre_action_snapshot_sha256=snapshot_sha256,
            recorded_action_key=action_key,
            branch_count=branch_count,
            continuation_replicates=continuation_replicates,
            failure_reward=failure_reward,
        )

    def complete_policy_call(
        self,
        pending: PendingSourcePolicyCall,
        *,
        action: BehaviorAction,
    ) -> RolloutDecision:
        """Persist the actual response and bind it to its earlier reservation."""
        decision_id = pending.reservation.decision_id
        if self._pending.get(decision_id) != pending:
            raise ValueError("source policy completion is not pending in this producer")
        if pending.raw_response_required and decision_id not in self._observed_responses:
            raise ValueError("source policy completion lacks its raw response witness")
        response_sha256 = self._ledger.put_evidence(action.to_bytes())
        if pending.recorded_action_reservation is not None:
            self._ledger.complete_recorded_action(
                pending.recorded_action_reservation,
                action=action,
                response_sha256=response_sha256,
            )
        completion = self._ledger.complete_source_policy_call(
            pending.reservation,
            action=action,
            response_sha256=response_sha256,
        )
        provenance = DecisionProvenance.from_receipts(
            pending.reservation.receipt,
            completion,
            verifier=self._ledger,
        )
        decision = RolloutDecision(
            decision_id,
            pending.event_address,
            action,
            pending.node_kind,
            pending.target_id,
            pending.target_ordinal,
            pending.outer_weight,
            provenance,
        )
        del self._pending[decision_id]
        self._observed_responses.discard(decision_id)
        self._completed[decision_id] = decision
        return decision

    def mark_policy_response_observed(
        self,
        pending: PendingSourcePolicyCall,
        *,
        response_content: bytes,
    ) -> str:
        """Persist exact provider bytes before the renderer parses them."""
        decision_id = pending.reservation.decision_id
        if self._pending.get(decision_id) != pending:
            raise ValueError("source policy response is not pending in this producer")
        if decision_id in self._observed_responses:
            raise ValueError("source policy response was observed twice")
        if type(response_content) is not bytes or not response_content:
            raise ValueError("source policy raw response must be nonempty bytes")
        response_sha256 = self._ledger.put_evidence(response_content)
        self._ledger.mark_source_policy_response_observed(
            pending.reservation,
            response_sha256=response_sha256,
        )
        self._observed_responses.add(decision_id)
        return response_sha256

    def abort_policy_call(
        self,
        pending: PendingSourcePolicyCall,
        *,
        phase: Literal[
            "post_unknown",
            "response_received",
            "response_parsed",
            "typed_response",
        ],
        error: BaseException,
    ) -> bytes:
        """Record an ambiguous/failed observed call and permanently stop the rollout."""
        decision_id = pending.reservation.decision_id
        if self._pending.get(decision_id) != pending:
            raise ValueError("source policy abort is not pending in this producer")
        error_evidence = canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-source-policy-abort-v1",
                "error_type": type(error).__qualname__,
                "error_message": str(error),
                "phase": phase,
            }
        )
        error_sha256 = self._ledger.put_evidence(error_evidence)
        receipt = self._ledger.abort_source_policy_call(
            pending.reservation,
            phase=phase,
            error_sha256=error_sha256,
        )
        del self._pending[decision_id]
        self._observed_responses.discard(decision_id)
        self._aborted = True
        return receipt

    def intercept_policy_call(
        self,
        *,
        event_address: PolicyEventAddress,
        action_key: ExactActionKey,
        node_kind: Literal["root", "child"],
        target_id: str | None,
        branch_selected: bool,
        forward_once: Callable[[ExactActionKey], BehaviorAction],
        recorded_action_reservation: RecordedActionReservation | None = None,
    ) -> RolloutDecision:
        """Reserve, forward exactly once, and durably complete one real policy call."""
        pending = self.reserve_policy_call(
            event_address=event_address,
            action_key=action_key,
            node_kind=node_kind,
            target_id=target_id,
            branch_selected=branch_selected,
            recorded_action_reservation=recorded_action_reservation,
        )
        action = forward_once(action_key)
        if type(action) is not BehaviorAction or action.key != action_key:
            raise ValueError("forwarded source action differs from its reserved request")
        return self.complete_policy_call(pending, action=action)

    def finalize_episode(
        self,
        raw_episode: bytes,
        *,
        prepare_source_rollout: Callable[[bytes], None] | None = None,
    ) -> SourceRollout:
        """Derive reward, sample boundaries, masks, and rosters from one raw episode."""
        if self._aborted:
            raise ValueError("cannot finalize a source rollout after an observed call abort")
        if self._pending:
            raise ValueError("cannot finalize a source rollout with pending policy calls")
        captured_decisions = tuple(
            sorted(
                self._completed.values(),
                key=lambda item: item.provenance.request_sequence,
            )
        )
        derived = derive_source_trace(
            raw_episode,
            decisions=captured_decisions,
            strict_two_slot=False,
            child_parent_event=self._child_parent_event,
            child_parent_tool_call_slot=self._child_parent_tool_call_slot,
            root_policy_turn_count=self._root_policy_turn_count,
            maximum_eligible_root_policy_turn_count=(
                self._maximum_eligible_root_policy_turn_count
            ),
        )
        if derived.trace_id != self._rollout_id:
            raise ValueError("captured source rollout ID differs from the Verifiers trace")
        branch_eligible = True
        ineligibility_reason: str | None = None
        if self._strict_two_slot:
            try:
                derive_source_trace(
                    raw_episode,
                    decisions=captured_decisions,
                    strict_two_slot=True,
                    child_parent_event=self._child_parent_event,
                    child_parent_tool_call_slot=self._child_parent_tool_call_slot,
                    root_policy_turn_count=self._root_policy_turn_count,
                    maximum_eligible_root_policy_turn_count=(
                        self._maximum_eligible_root_policy_turn_count
                    ),
                )
            except SourceTopologyIneligible as error:
                branch_eligible = False
                ineligibility_reason = str(error)
        decisions = _normalize_child_weights(
            captured_decisions,
            len(
                {
                    decision.target_id
                    for decision in captured_decisions
                    if decision.node_kind == "child"
                }
            ),
        )
        reward_evidence = canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-reward-evidence-v1",
                "group_id": self._group_id,
                "rollout_id": self._rollout_id,
                "reward": derived.reward,
            }
        )
        stock_evidence = canonical_json(
            [sequence.to_payload() for sequence in derived.stock_sequences]
        )
        trace_sha256 = self._ledger.put_evidence(raw_episode)
        reward_sha256 = self._ledger.put_evidence(reward_evidence)
        stock_sha256 = self._ledger.put_evidence(stock_evidence)
        draft = SourceRollout._construct(
            self._group_id,
            self._rollout_id,
            derived.reward,
            derived.stock_sequences,
            derived.stock_sequence_decision_ids,
            decisions,
            derived.child_target_roster,
            branch_eligible,
            ineligibility_reason,
            trace_sha256,
            reward_sha256,
            stock_sha256,
            self._base_model_manifest_sha256,
            "live",
            canonical_json({}),
        )
        if prepare_source_rollout is not None:
            prepare_source_rollout(
                canonical_json(
                    {
                        "schema_version": 1,
                        "domain": "redco-stage-d-prepared-source-rollout-v1",
                        "source": draft.to_payload(),
                        "source_sha256": draft.source_sha256,
                    }
                )
            )
        completion = self._ledger.record_source_rollout_completed(
            group_id=self._group_id,
            rollout_id=self._rollout_id,
            source_sha256=draft.source_sha256,
            trace_sha256=trace_sha256,
            reward_evidence_sha256=reward_sha256,
            stock_sequences_evidence_sha256=stock_sha256,
            base_model_manifest_sha256=self._base_model_manifest_sha256,
            decision_ids=tuple(item.decision_id for item in decisions),
            decision_completion_receipt_sha256s=tuple(
                _sha256(item.provenance.completion_receipt) for item in decisions
            ),
        )
        self._rollout_completed = True
        source = SourceRollout._construct(
            self._group_id,
            self._rollout_id,
            derived.reward,
            derived.stock_sequences,
            derived.stock_sequence_decision_ids,
            decisions,
            derived.child_target_roster,
            branch_eligible,
            ineligibility_reason,
            trace_sha256,
            reward_sha256,
            stock_sha256,
            self._base_model_manifest_sha256,
            "live",
            completion.receipt,
        )
        verify_source_trace_semantics(
            source,
            raw_episode=raw_episode,
            strict_two_slot=branch_eligible and self._strict_two_slot,
            child_parent_event=self._child_parent_event,
            child_parent_tool_call_slot=self._child_parent_tool_call_slot,
            root_policy_turn_count=self._root_policy_turn_count,
            maximum_eligible_root_policy_turn_count=(
                self._maximum_eligible_root_policy_turn_count
            ),
        )
        return source

    def abort_finalization(self, error: BaseException) -> bytes | None:
        """Poison an observed rollout unless its source completion is already durable."""
        if self._rollout_completed or self._aborted or self._pending or not self._completed:
            return None
        self._aborted = True
        error_evidence = canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-source-finalization-abort-v1",
                "error_type": type(error).__qualname__,
                "error_message": str(error),
            }
        )
        error_sha256 = self._ledger.put_evidence(error_evidence)
        return self._ledger.abort_source_rollout_finalization(
            group_id=self._group_id,
            rollout_id=self._rollout_id,
            error_sha256=error_sha256,
        )


def derive_source_trace(
    raw_episode: bytes,
    *,
    decisions: Sequence[RolloutDecision],
    strict_two_slot: bool = False,
    child_parent_event: PolicyEventAddress | None = None,
    child_parent_tool_call_slot: int = 0,
    root_policy_turn_count: int | None = None,
    maximum_eligible_root_policy_turn_count: int = 4,
) -> DerivedTraceSource:
    """Independently reconstruct Prime's text-only trace-to-samples contract."""
    episode, trace = _parse_episode(raw_episode)
    del episode
    ordered_decisions = tuple(decisions)
    calls = _object_list(trace, "calls")
    nodes = _object_list(trace, "nodes")
    if any(not _node_fields_are_pinned(node) for node in nodes):
        raise ValueError("source trace node fields differ from pinned text output")
    if len(calls) != len(ordered_decisions):
        raise ValueError("captured policy-call count differs from the Verifiers trace")
    provenance = extract_v2_rlm_provenance(trace)
    if len(provenance) != len(calls):
        raise ValueError("RLM provenance does not biject source calls")
    if child_parent_event is not None:
        _verify_deployed_parent_links(
            provenance,
            nodes,
            root_lineage=child_parent_event.lineage,
            parent_tool_call_slot=child_parent_tool_call_slot,
        )
    addresses = tuple(record.scientific_address for record in provenance)
    if strict_two_slot:
        _verify_two_slot_scaffold(
            provenance,
            nodes,
            child_parent_event=child_parent_event,
            child_parent_tool_call_slot=child_parent_tool_call_slot,
            root_policy_turn_count=root_policy_turn_count,
            maximum_eligible_root_policy_turn_count=(
                maximum_eligible_root_policy_turn_count
            ),
        )
    decision_by_node: dict[int, RolloutDecision] = {}
    decisions_by_address = {
        canonical_json(decision.event_address.as_payload()): decision
        for decision in ordered_decisions
    }
    if len(decisions_by_address) != len(ordered_decisions):
        raise ValueError("captured decisions repeat a structural address")
    for call_index, (call, address, record) in enumerate(
        zip(calls, addresses, provenance, strict=True)
    ):
        decision = decisions_by_address.get(canonical_json(address.as_payload()))
        if decision is None:
            raise ValueError("Verifiers trace contains an unreserved policy event")
        _verify_trace_call(
            trace,
            nodes,
            call,
            call_index=call_index,
            address=address,
            record=record,
            decision=decision,
            rollout_id=str(trace["id"]),
        )
        node_index = _exact_int(call.get("node"), f"call {call_index} node")
        if node_index in decision_by_node:
            raise ValueError("two captured policy calls name one trace node")
        decision_by_node[node_index] = decision
    sampled_nodes = {index for index, node in enumerate(nodes) if node.get("sampled") is True}
    if set(decision_by_node) != sampled_nodes:
        raise ValueError("captured policy calls do not biject sampled trace nodes")
    temperature = _trace_temperature(trace)
    children = {
        _exact_int(node.get("parent"), f"node {index} parent")
        for index, node in enumerate(nodes)
        if node.get("parent") is not None
    }
    leaves = [index for index in range(len(nodes)) if index not in children]
    trained_nodes: set[int] = set()
    sequences: list[FrozenTrainingSequence] = []
    rosters: list[tuple[str, ...]] = []
    for leaf in leaves:
        path = _path_to_node(nodes, leaf)
        tokens: list[int] = []
        mask: list[bool] = []
        logprobs: list[float] = []
        roster: list[str] = []
        for node_index in path:
            node = nodes[node_index]
            node_tokens = _integer_list(node.get("token_ids"), f"node {node_index} token_ids")
            node_mask = _boolean_list(node.get("mask"), f"node {node_index} mask")
            if len(node_tokens) != len(node_mask):
                raise ValueError("trace node token IDs and mask differ")
            node_logprobs = _float_list(node.get("logprobs"), f"node {node_index} logprobs")
            if len(node_logprobs) != sum(node_mask):
                raise ValueError("trace node logprobs do not cover sampled tokens")
            trainable = node_mask
            if node.get("sampled") is True and any(node_mask):
                if node_index in trained_nodes:
                    trainable = [False] * len(node_mask)
                else:
                    trained_nodes.add(node_index)
                    roster.append(decision_by_node[node_index].decision_id)
            tokens.extend(node_tokens)
            mask.extend(trainable)
            logprobs.extend(
                _spread_logprobs(node_logprobs, node_mask)
                if trainable is node_mask
                else [0.0] * len(node_mask)
            )
        if not any(mask):
            continue
        sequences.append(
            FrozenTrainingSequence(
                tuple(tokens),
                tuple(mask),
                tuple(logprobs),
                (temperature,) * len(tokens),
                None,
                None,
            )
        )
        rosters.append(tuple(roster))
    if trained_nodes != sampled_nodes:
        raise ValueError("trace branch derivation did not route every sampled node once")
    rewards = trace.get("rewards")
    if not isinstance(rewards, dict) or not rewards:
        raise ValueError("successful source trace requires explicit reward components")
    values = tuple(_finite_float(value, f"reward {name}") for name, value in rewards.items())
    reward = float(sum(values))
    child_target_by_ordinal: dict[int, str] = {}
    if child_parent_event is not None:
        for ordinal in (0, 1):
            child_target_by_ordinal[ordinal] = structural_child_target_id(
                child_parent_event,
                rollout_id=str(trace["id"]),
                parent_tool_call_slot=child_parent_tool_call_slot,
                spawn_ordinal=ordinal,
            )
    for decision in ordered_decisions:
        if decision.node_kind != "child":
            continue
        ordinal = _exact_int(decision.target_ordinal, "child target ordinal")
        if decision.target_id is None:
            raise ValueError("captured child decision lacks its structural target")
        prior = child_target_by_ordinal.setdefault(ordinal, decision.target_id)
        if prior != decision.target_id:
            raise ValueError("captured child ordinal changes its structural target")
    child_targets = tuple(
        child_target_by_ordinal[ordinal]
        for ordinal in sorted(child_target_by_ordinal)
    )
    return DerivedTraceSource(
        str(trace["id"]),
        reward,
        tuple(sequences),
        tuple(rosters),
        child_targets,
    )


def verify_source_trace_semantics(
    source: SourceRollout,
    *,
    raw_episode: bytes,
    strict_two_slot: bool = False,
    child_parent_event: PolicyEventAddress | None = None,
    child_parent_tool_call_slot: int = 0,
    root_policy_turn_count: int | None = None,
    maximum_eligible_root_policy_turn_count: int = 4,
) -> None:
    """Reject a source whose serialized fields were not derived from its raw trace."""
    derived = derive_source_trace(
        raw_episode,
        decisions=source.decisions,
        strict_two_slot=strict_two_slot,
        child_parent_event=child_parent_event,
        child_parent_tool_call_slot=child_parent_tool_call_slot,
        root_policy_turn_count=root_policy_turn_count,
        maximum_eligible_root_policy_turn_count=(
            maximum_eligible_root_policy_turn_count
        ),
    )
    if (
        derived.trace_id != source.rollout_id
        or derived.reward != source.reward
        or derived.stock_sequences != source.stock_sequences
        or derived.stock_sequence_decision_ids != source.stock_sequence_decision_ids
        or derived.child_target_roster != source.child_target_roster
    ):
        raise ValueError("source rollout fields differ from semantic trace derivation")


def _parse_episode(raw_episode: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(raw_episode) is not bytes or not raw_episode:
        raise ValueError("source episode must be nonempty immutable bytes")
    try:
        episode = json.loads(raw_episode)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source episode must be one JSON object") from error
    if not isinstance(episode, dict) or set(episode) != _EPISODE_FIELDS:
        raise ValueError("source episode envelope differs from pinned Verifiers output")
    if episode.get("ok") is not True or episode.get("errors") != []:
        raise ValueError("source episode did not finish successfully")
    traces = episode.get("traces")
    if not isinstance(traces, list) or len(traces) != 1 or not isinstance(traces[0], dict):
        raise ValueError("source episode must contain exactly one trace")
    trace = traces[0]
    if set(trace) != _TRACE_FIELDS:
        raise ValueError("source trace fields differ from pinned Verifiers output")
    if (
        trace.get("ok") is not True
        or trace.get("is_completed") is not True
        or trace.get("errors") != []
    ):
        raise ValueError("source trace did not finish successfully")
    if not isinstance(trace.get("id"), str) or not trace["id"]:
        raise ValueError("source trace ID is absent")
    return episode, trace


def _verify_trace_call(
    trace: Mapping[str, Any],
    nodes: list[dict[str, Any]],
    call: Mapping[str, Any],
    *,
    call_index: int,
    address: PolicyEventAddress,
    record: RecordedRLMProvenanceV2,
    decision: RolloutDecision,
    rollout_id: str,
) -> None:
    if set(call) != _CALL_FIELDS:
        raise ValueError("source trace call fields differ from successful pinned output")
    if decision.event_address != address:
        raise ValueError("captured structural address differs from the Verifiers trace")
    expected_kind: Literal["root", "child"] = "root" if address.depth == 0 else "child"
    if decision.node_kind != expected_kind:
        raise ValueError("captured source node kind differs from recursive depth")
    if expected_kind == "child":
        assert record.parent_lineage is not None
        assert record.parent_call_ordinal is not None
        assert record.parent_tool_call_slot is not None
        assert record.spawn_ordinal is not None
        parent = PolicyEventAddress(
            record.depth - 1,
            record.parent_lineage,
            record.parent_call_ordinal,
            record.parent_turn or 0,
            "policy",
        )
        expected_target = structural_child_target_id(
            parent,
            rollout_id=rollout_id,
            parent_tool_call_slot=record.parent_tool_call_slot,
            spawn_ordinal=record.spawn_ordinal,
        )
        if decision.target_id != expected_target:
            raise ValueError("captured child target differs from its structural slot")
    node_index = _exact_int(call.get("node"), f"call {call_index} node")
    if node_index >= len(nodes):
        raise ValueError("source trace call names an absent node")
    node = nodes[node_index]
    if not _node_fields_are_pinned(node) or node.get("sampled") is not True:
        raise ValueError("source trace call does not name one sampled text node")
    path = _path_to_node(nodes, node_index)
    node_tokens = _integer_list(node.get("token_ids"), "sampled node token_ids")
    node_mask = _boolean_list(node.get("mask"), "sampled node mask")
    first_sampled = next((i for i, selected in enumerate(node_mask) if selected), len(node_mask))
    if (
        not node_mask
        or any(node_mask[:first_sampled])
        or any(not selected for selected in node_mask[first_sampled:])
    ):
        raise ValueError("sampled node mask is not one nonempty suffix")
    prompt_tokens = tuple(
        token
        for prior in path[:-1]
        for token in _integer_list(nodes[prior].get("token_ids"), "prompt node token_ids")
    ) + tuple(node_tokens[:first_sampled])
    action_tokens = tuple(node_tokens[first_sampled:])
    action_logprobs = tuple(_float_list(node.get("logprobs"), "sampled node logprobs"))
    action = decision.action
    if (
        prompt_tokens != action.key.prompt_token_ids
        or action_tokens != action.action_token_ids
        or action_logprobs != action.behavior_logprobs
    ):
        raise ValueError("captured prompt/action streams differ from the Verifiers trace")
    message = _normalize_openai_message(node.get("message"))
    if message != _normalize_openai_message(action.message):
        raise ValueError("captured transport message differs from the Verifiers trace")
    request = json.loads(action.key.request)
    if not isinstance(request, dict):
        raise ValueError("captured exact request is not an object")
    raw_request_messages = request.get("messages")
    if not isinstance(raw_request_messages, list):
        raise ValueError("captured exact request messages are not a list")
    request_messages = [
        _normalize_openai_message(item) for item in raw_request_messages
    ]
    graph_messages = [
        _normalize_openai_message(nodes[index].get("message"))
        for index in path[:-1]
    ]
    request_tools = _normalize_openai_tools(request.get("tools", []))
    trace_tools = _normalize_openai_tools(trace.get("tools"))
    if request_messages != graph_messages or request_tools != trace_tools:
        raise ValueError("captured request context differs from the Verifiers graph")
    if call.get("model") != action.key.checkpoint_id:
        raise ValueError("captured model identity differs from the Verifiers call")
    checkpoint_claims = _checkpoint_claims(trace)
    if checkpoint_claims and checkpoint_claims != {action.key.checkpoint_id}:
        raise ValueError("trace checkpoint claims differ from the exact action key")
    sampling = call.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("source trace call lacks sampling configuration")
    sampler = action.key.sampler
    expected_sampling = {
        "temperature": sampler.temperature,
        "top_p": sampler.top_p,
        "reasoning_effort": None,
        "max_tokens": sampler.max_tokens,
        "parallel_tool_calls": False,
        "seed": sampler.seed,
        "tool_choice": sampler.tool_choice,
    }
    if sampling != expected_sampling:
        raise ValueError("captured sampler differs from the Verifiers call")
    usage = call.get("usage")
    if not isinstance(usage, dict) or usage != {
        "prompt_tokens": action.prompt_tokens,
        "completion_tokens": action.completion_tokens,
        "cached_input_tokens": None,
        "reasoning_tokens": None,
        "cost": None,
    }:
        raise ValueError("captured usage differs from the Verifiers call")
    if call.get("error") is not None:
        raise ValueError("successful source trace call contains an error")
    if call.get("finish_reason") != action.finish_reason:
        raise ValueError("captured finish reason differs from the Verifiers call")
    if call.get("endpoint") != "/chat/completions":
        raise ValueError("source trace did not use the frozen chat-completions endpoint")


def _verify_two_slot_scaffold(
    records: Sequence[RecordedRLMProvenanceV2],
    nodes: Sequence[Mapping[str, Any]],
    *,
    child_parent_event: PolicyEventAddress | None,
    child_parent_tool_call_slot: int,
    root_policy_turn_count: int | None,
    maximum_eligible_root_policy_turn_count: int = 4,
) -> None:
    if child_parent_event is None:
        raise SourceTopologyIneligible(
            "strict source verification lacks its structural parent"
        )
    if (
        type(root_policy_turn_count) is not int
        or root_policy_turn_count < 2
        or type(maximum_eligible_root_policy_turn_count) is not int
        or maximum_eligible_root_policy_turn_count < root_policy_turn_count
    ):
        raise SourceTopologyIneligible(
            "strict source verification lacks its root-turn bounds"
        )
    roots = tuple(record for record in records if record.depth == 0)
    if (
        len(roots) < root_policy_turn_count
        or len(roots) > maximum_eligible_root_policy_turn_count
        or len(records) != len(roots) + 2
    ):
        raise SourceTopologyIneligible(
            "scientific scaffold has an unexpected policy-call count"
        )
    if any(
        record.lineage != child_parent_event.lineage
        or record.call_kind != "policy"
        or record.turn != ordinal
        or record.session_call_ordinal != ordinal
        for ordinal, record in enumerate(roots)
    ):
        raise SourceTopologyIneligible("scientific root policy turns are not contiguous")
    children = tuple(record for record in records if record.depth > 0)
    if len(children) != 2 or any(
        record.depth != 1 or record.call_kind != "policy" or record.session_call_ordinal != 0
        for record in children
    ):
        raise SourceTopologyIneligible(
            "scientific scaffold requires exactly two one-turn children"
        )
    if {record.spawn_ordinal for record in children} != {0, 1}:
        raise SourceTopologyIneligible(
            "scientific child roster must contain spawn slots zero and one"
        )
    if any(
        record.parent_lineage != child_parent_event.lineage
        or record.parent_call_ordinal != child_parent_event.session_call_ordinal
        or record.parent_tool_call_slot != child_parent_tool_call_slot
        for record in children
    ):
        raise SourceTopologyIneligible(
            "scientific children do not share the frozen parent scope"
        )
    parents = tuple(
        record
        for record in records
        if record.scientific_address.as_payload() == child_parent_event.as_payload()
    )
    if len(parents) != 1:
        raise SourceTopologyIneligible(
            "scientific child parent does not biject a committed policy event"
        )
    parent_node = _normalize_openai_message(nodes[parents[0].node_index].get("message"))
    tool_calls = parent_node.get("tool_calls")
    if not isinstance(tool_calls, list) or child_parent_tool_call_slot >= len(tool_calls):
        raise SourceTopologyIneligible(
            "scientific parent lacks its frozen tool-call slot"
        )
    parent_tool_call = tool_calls[child_parent_tool_call_slot]
    if not isinstance(parent_tool_call, dict) or not isinstance(parent_tool_call.get("id"), str):
        raise SourceTopologyIneligible(
            "scientific parent tool-call identity is absent"
        )
    if any(record.parent_tool_call_id != parent_tool_call["id"] for record in children):
        raise SourceTopologyIneligible(
            "child provenance does not link to the committed parent tool call"
        )


def _verify_deployed_parent_links(
    records: Sequence[RecordedRLMProvenanceV2],
    nodes: Sequence[Mapping[str, Any]],
    *,
    root_lineage: str,
    parent_tool_call_slot: int,
) -> None:
    """Validate causal child links before scientific eligibility is considered."""
    roots = {
        (record.lineage, record.session_call_ordinal): record
        for record in records
        if record.depth == 0
    }
    for child in (record for record in records if record.depth > 0):
        if (
            child.depth != 1
            or child.parent_lineage != root_lineage
            or child.parent_call_ordinal is None
            or child.parent_turn is None
            or child.parent_session_id is None
            or child.parent_tool_call_slot != parent_tool_call_slot
            or child.parent_tool_call_id is None
        ):
            raise ValueError("child provenance is outside the deployed parent contract")
        parent = roots.get((child.parent_lineage, child.parent_call_ordinal))
        if (
            parent is None
            or parent.call_kind != "policy"
            or parent.turn != child.parent_turn
            or parent.session_id != child.parent_session_id
        ):
            raise ValueError("child provenance does not bind its causal parent event")
        parent_node = _normalize_openai_message(nodes[parent.node_index].get("message"))
        tool_calls = parent_node.get("tool_calls")
        if not isinstance(tool_calls, list) or parent_tool_call_slot >= len(tool_calls):
            raise ValueError("child provenance parent lacks its deployed tool-call slot")
        tool_call = tool_calls[parent_tool_call_slot]
        if (
            not isinstance(tool_call, dict)
            or tool_call.get("id") != child.parent_tool_call_id
        ):
            raise ValueError("child provenance does not bind its parent tool-call ID")


def _normalize_child_weights(
    decisions: Sequence[RolloutDecision],
    child_count: int,
) -> tuple[RolloutDecision, ...]:
    """Apply the observed exact 1/m outer weight after natural topology is known."""
    if type(child_count) is not int or child_count < 0:
        raise ValueError("observed child count must be nonnegative")
    child_weight = Fraction(1, child_count) if child_count else None
    normalized: list[RolloutDecision] = []
    for decision in decisions:
        weight = Fraction(1) if decision.node_kind == "root" else child_weight
        if weight is None:
            raise ValueError("captured child decision is absent from the observed roster")
        normalized.append(
            RolloutDecision(
                decision.decision_id,
                decision.event_address,
                decision.action,
                decision.node_kind,
                decision.target_id,
                decision.target_ordinal,
                weight,
                decision.provenance,
            )
        )
    return tuple(normalized)


def _decision_id(address: PolicyEventAddress) -> str:
    return (
        "decision-"
        + _sha256(
            canonical_json({"domain": "redco-stage-d-source-decision-v1", **address.as_payload()})
        )[:24]
    )


def _path_to_node(nodes: Sequence[Mapping[str, Any]], node_index: int) -> list[int]:
    path: list[int] = []
    seen: set[int] = set()
    current: int | None = node_index
    while current is not None:
        if current < 0 or current >= len(nodes) or current in seen:
            raise ValueError("source trace graph contains an invalid parent chain")
        seen.add(current)
        path.append(current)
        parent = nodes[current].get("parent")
        current = None if parent is None else _exact_int(parent, f"node {current} parent")
    path.reverse()
    return path


def _spread_logprobs(values: Sequence[float], mask: Sequence[bool]) -> list[float]:
    iterator = iter(values)
    result = [next(iterator) if selected else 0.0 for selected in mask]
    try:
        next(iterator)
    except StopIteration:
        return result
    raise ValueError("trace logprobs exceed sampled mask")


def _trace_temperature(trace: Mapping[str, Any]) -> float:
    agent = trace.get("agent")
    if not isinstance(agent, dict):
        raise ValueError("source trace lacks agent identity")
    sampling = agent.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("source trace lacks agent sampling")
    return _finite_float(sampling.get("temperature"), "trace temperature", positive=True)


def _checkpoint_claims(trace: Mapping[str, Any]) -> set[str]:
    claims: set[str] = set()
    task = trace.get("task")
    if isinstance(task, dict) and isinstance(task.get("data"), dict):
        value = task["data"].get("policy_checkpoint_id")
        if isinstance(value, str) and value:
            claims.add(value)
    info = trace.get("info")
    if isinstance(info, dict):
        value = info.get("checkpoint_id")
        if isinstance(value, str) and value:
            claims.add(value)
    return claims


def _normalize_openai_message(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("trace message must be an object")
    message = deepcopy(value)
    if message.get("role") == "tool":
        message.pop("name", None)
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        return message
    if not isinstance(raw_calls, list):
        raise ValueError("trace tool_calls must be a list")
    normalized: list[dict[str, Any]] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            raise ValueError("trace tool call must be an object")
        if isinstance(raw.get("function"), dict):
            normalized.append(raw)
            continue
        call_id = _nonempty_string(raw.get("id"), "trace tool call id")
        name = _nonempty_string(raw.get("name"), "trace tool call name")
        arguments = raw.get("arguments")
        if not isinstance(arguments, str):
            raise ValueError("trace tool call arguments must be serialized JSON")
        normalized.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    message["tool_calls"] = normalized
    if message.get("role") == "assistant" and message.get("content") is None:
        message["content"] = ""
    return message


def _normalize_openai_tools(value: object) -> list[dict[str, Any]]:
    """Canonicalize pinned compact and OpenAI-wrapped function definitions."""
    if not isinstance(value, list):
        raise ValueError("trace tools must be a list")
    normalized: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("trace tool must be an object")
        required = {"name", "description", "parameters"}
        allowed = required | {"strict"}
        if required <= set(raw) <= allowed:
            function = raw
        elif set(raw) == {"type", "function"} and raw.get("type") == "function":
            wrapped_function = raw.get("function")
            if not isinstance(wrapped_function, dict):
                raise ValueError("trace function tool must contain an object")
            function = wrapped_function
        else:
            raise ValueError("trace tool differs from the pinned function schema")
        if not (required <= set(function) <= allowed):
            raise ValueError("trace function definition fields differ from the pinned schema")
        name = _nonempty_string(function.get("name"), "trace function name")
        description = function.get("description")
        parameters = function.get("parameters")
        if not isinstance(description, str):
            raise ValueError("trace function description must be a string")
        if not isinstance(parameters, dict):
            raise ValueError("trace function parameters must be an object")
        strict = function.get("strict")
        if strict is not None and type(strict) is not bool:
            raise ValueError("trace function strict must be boolean or null")
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": deepcopy(parameters),
                    "strict": strict,
                },
            }
        )
    return normalized


def _object_list(value: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    items = value.get(name)
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError(f"source trace {name} must be an object list")
    return items


def _node_fields_are_pinned(node: Mapping[str, Any]) -> bool:
    fields = set(node)
    return fields == _NODE_FIELDS or fields == _NODE_FIELDS - {"parent"}


def _integer_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{name} must be nonnegative integers")
    return value


def _boolean_list(value: object, name: str) -> list[bool]:
    if not isinstance(value, list) or any(type(item) is not bool for item in value):
        raise ValueError(f"{name} must be booleans")
    return value


def _float_list(value: object, name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be floats")
    return [_finite_float(item, name) for item in value]


def _exact_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be finite")
    assert isinstance(value, (int, float))
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty")
    return value
