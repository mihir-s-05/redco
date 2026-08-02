"""Immutable source-rollout contracts for Stage D."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

from redco.analysis.stage_d_exact_action import BehaviorAction
from redco.analysis.stage_d_scientific_branch_group import ReceiptVerifier
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress
from redco.contracts import canonical_json

SCHEMA_VERSION = 1


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_from_payload(value: object, name: str) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError(f"{name} must be an exact fraction")
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise ValueError(f"{name} must be an exact fraction")
    return Fraction(numerator, denominator)


@dataclass(frozen=True, slots=True)
class FrozenTrainingSequence:
    """One exact Prime-compatible sequence before advantage assignment."""

    token_ids: tuple[int, ...]
    mask: tuple[bool, ...]
    behavior_logprobs: tuple[float, ...]
    temperatures: tuple[float, ...]
    rl_weights: tuple[float, ...] | None
    rl_normalizer: Fraction | None

    def __post_init__(self) -> None:
        length = len(self.token_ids)
        streams: list[Sequence[object]] = [
            self.mask,
            self.behavior_logprobs,
            self.temperatures,
        ]
        if self.rl_weights is not None:
            streams.append(self.rl_weights)
        if length == 0 or any(len(stream) != length for stream in streams):
            raise ValueError("training sequence streams must be nonempty and aligned")
        if not any(self.mask):
            raise ValueError("training sequence must select at least one action token")
        if any(type(token) is not int or token < 0 for token in self.token_ids):
            raise ValueError("training token IDs must be nonnegative integers")
        weights = self.rl_weights or tuple(1.0 if selected else 0.0 for selected in self.mask)
        for selected, logprob, temperature, weight in zip(
            self.mask,
            self.behavior_logprobs,
            self.temperatures,
            weights,
            strict=True,
        ):
            if any(not math.isfinite(value) for value in (logprob, temperature, weight)):
                raise ValueError("training sequence numeric streams must be finite")
            if temperature <= 0.0:
                raise ValueError("training temperatures must be positive")
            if not selected and (logprob != 0.0 or weight != 0.0):
                raise ValueError("unselected positions must have zero logprob and weight")
            if selected and (logprob > 0.0 or weight <= 0.0):
                raise ValueError(
                    "selected positions require nonpositive logprobs and positive weights"
                )
        if self.rl_normalizer is not None and self.rl_normalizer <= 0:
            raise ValueError("training sequence normalizer must be positive")

    def to_payload(self) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "mask": list(self.mask),
            "behavior_logprobs": list(self.behavior_logprobs),
            "temperatures": list(self.temperatures),
            "rl_weights": None if self.rl_weights is None else list(self.rl_weights),
            "rl_normalizer": (
                None if self.rl_normalizer is None else _fraction_payload(self.rl_normalizer)
            ),
        }


@dataclass(frozen=True, slots=True, init=False)
class DecisionProvenance:
    """Two same-ledger receipts proving request-before-response chronology."""

    reservation_receipt: bytes
    completion_receipt: bytes
    ledger_id: str
    group_id: str
    rollout_id: str
    decision_id: str
    node_kind: Literal["root", "child"]
    target_id: str | None
    target_ordinal: int | None
    event_address: PolicyEventAddress
    branch_selected: bool
    target_commitment_receipt_sha256: str | None
    exact_action_key_digest: str
    action_digest: str
    request_sha256: str
    response_sha256: str
    request_sequence: int
    completion_sequence: int

    @classmethod
    def from_receipts(
        cls,
        reservation_receipt: bytes,
        completion_receipt: bytes,
        *,
        verifier: ReceiptVerifier,
    ) -> DecisionProvenance:
        reserved = _verified_source_receipt(
            reservation_receipt,
            "source_policy_call_reserved",
            verifier,
        )
        completed = _verified_source_receipt(
            completion_receipt,
            "source_policy_call_completed",
            verifier,
        )
        reservation_fields = {
            "schema_version",
            "receipt_kind",
            "ledger_id",
            "ledger_offset",
            "prior_chain_sha256",
            "group_id",
            "rollout_id",
            "decision_id",
            "node_kind",
            "target_id",
            "target_ordinal",
            "target_address",
            "exact_action_key_digest",
            "request_sha256",
            "branch_selected",
            "target_commitment_receipt_sha256",
            "recorded_action_reservation_id",
            "request_sequence",
        }
        completion_fields = {
            "schema_version",
            "receipt_kind",
            "ledger_id",
            "ledger_offset",
            "prior_chain_sha256",
            "group_id",
            "rollout_id",
            "decision_id",
            "request_receipt_sha256",
            "exact_action_key_digest",
            "action_digest",
            "response_sha256",
            "request_sequence",
            "completion_sequence",
        }
        if set(reserved) != reservation_fields or set(completed) != completion_fields:
            raise ValueError("source policy receipt fields differ")
        shared = ("ledger_id", "group_id", "rollout_id", "decision_id")
        if any(
            not isinstance(reserved[field], str)
            or not reserved[field]
            or reserved[field] != completed[field]
            for field in shared
        ):
            raise ValueError("source policy receipts cross ledgers or decisions")
        if completed["request_receipt_sha256"] != _sha256(reservation_receipt):
            raise ValueError("source policy completion names a different reservation")
        if (
            completed["exact_action_key_digest"] != reserved["exact_action_key_digest"]
            or completed["request_sequence"] != reserved["request_sequence"]
            or reserved["ledger_offset"] != reserved["request_sequence"]
            or completed["ledger_offset"] != completed["completion_sequence"]
            or reserved["request_sequence"] >= completed["completion_sequence"]
        ):
            raise ValueError("source policy receipt chronology or action key differs")
        address = _policy_event_address(reserved["target_address"])
        node_kind = reserved["node_kind"]
        target_id = reserved["target_id"]
        target_ordinal = reserved["target_ordinal"]
        branch_selected = reserved["branch_selected"]
        if node_kind not in {"root", "child"} or type(branch_selected) is not bool:
            raise ValueError("source policy receipt role is invalid")
        if (node_kind == "root" and (target_id is not None or target_ordinal is not None)) or (
            node_kind == "child"
            and (
                not isinstance(target_id, str)
                or not target_id
                or type(target_ordinal) is not int
                or target_ordinal < 0
            )
        ):
            raise ValueError("source policy receipt target is invalid")
        commitment_hash = reserved["target_commitment_receipt_sha256"]
        recorded_reservation_id = reserved["recorded_action_reservation_id"]
        if branch_selected:
            _require_sha256(commitment_hash, "target commitment receipt sha256")
            if not isinstance(recorded_reservation_id, str) or not recorded_reservation_id:
                raise ValueError("selected source receipt lacks recorded-action linkage")
        elif commitment_hash is not None:
            raise ValueError("unselected source policy receipt names a commitment")
        elif recorded_reservation_id is not None:
            raise ValueError("unselected source policy receipt names an action reservation")
        request_sequence = reserved["request_sequence"]
        completion_sequence = completed["completion_sequence"]
        if (
            type(request_sequence) is not int
            or type(completion_sequence) is not int
            or request_sequence < 0
            or request_sequence >= completion_sequence
        ):
            raise ValueError("source policy receipt sequences are invalid")
        values = {
            "reservation_receipt": reservation_receipt,
            "completion_receipt": completion_receipt,
            "ledger_id": reserved["ledger_id"],
            "group_id": reserved["group_id"],
            "rollout_id": reserved["rollout_id"],
            "decision_id": reserved["decision_id"],
            "node_kind": node_kind,
            "target_id": target_id,
            "target_ordinal": target_ordinal,
            "event_address": address,
            "branch_selected": branch_selected,
            "target_commitment_receipt_sha256": commitment_hash,
            "exact_action_key_digest": _require_sha256(
                reserved["exact_action_key_digest"],
                "action key digest",
            ),
            "action_digest": _require_sha256(
                completed["action_digest"],
                "action digest",
            ),
            "request_sha256": _require_sha256(
                reserved["request_sha256"],
                "request sha256",
            ),
            "response_sha256": _require_sha256(
                completed["response_sha256"],
                "response sha256",
            ),
            "request_sequence": request_sequence,
            "completion_sequence": completion_sequence,
        }
        self = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        return self

    def to_payload(self) -> dict[str, Any]:
        return {
            "reservation_receipt": json.loads(self.reservation_receipt),
            "completion_receipt": json.loads(self.completion_receipt),
        }


@dataclass(frozen=True, slots=True)
class RolloutDecision:
    """One trainable policy decision and its incumbent-comparability weight."""

    decision_id: str
    event_address: PolicyEventAddress
    action: BehaviorAction
    node_kind: Literal["root", "child"]
    target_id: str | None
    target_ordinal: int | None
    outer_weight: Fraction
    provenance: DecisionProvenance

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise ValueError("decision_id must be nonempty")
        if type(self.event_address) is not PolicyEventAddress:
            raise ValueError("decision event address must be structural")
        if type(self.action) is not BehaviorAction:
            raise ValueError("decision action must be a validated BehaviorAction")
        if self.node_kind not in {"root", "child"}:
            raise ValueError("decision node kind must be root or child")
        if (
            self.node_kind == "root"
            and (self.target_id is not None or self.target_ordinal is not None)
        ) or (self.node_kind == "child" and not self.target_id):
            raise ValueError("only child decisions require a target_id")
        if self.node_kind == "child" and (
            type(self.target_ordinal) is not int or self.target_ordinal < 0
        ):
            raise ValueError("child decisions require a nonnegative target ordinal")
        if self.outer_weight <= 0:
            raise ValueError("decision outer weight must be positive")
        if self.node_kind == "root" and self.outer_weight != 1:
            raise ValueError("root decisions must have outer weight one")
        if (
            self.provenance.decision_id != self.decision_id
            or self.provenance.node_kind != self.node_kind
            or self.provenance.target_id != self.target_id
            or self.provenance.target_ordinal != self.target_ordinal
            or self.provenance.event_address != self.event_address
            or self.provenance.exact_action_key_digest != self.action.key.digest
            or self.provenance.action_digest != self.action.digest
            or self.provenance.request_sha256 != self.action.key.request_sha256
            or self.provenance.response_sha256 != _sha256(self.action.to_bytes())
        ):
            raise ValueError("decision differs from its verified provenance")


@dataclass(frozen=True, slots=True, init=False)
class SourceRollout:
    """Full source rollout with exact stock bytes and explicit policy decisions."""

    group_id: str
    rollout_id: str
    reward: float
    stock_sequences: tuple[FrozenTrainingSequence, ...]
    stock_sequence_decision_ids: tuple[tuple[str, ...], ...]
    decisions: tuple[RolloutDecision, ...]
    child_target_roster: tuple[str, ...]
    branch_eligible: bool
    ineligibility_reason: str | None
    trace_sha256: str
    reward_evidence_sha256: str
    stock_sequences_evidence_sha256: str
    base_model_manifest_sha256: str
    evidence_class: Literal["live", "fixture-only"]
    producer_receipt: bytes | None
    source_sha256: str = field(init=False)

    def __new__(cls) -> SourceRollout:
        raise TypeError("SourceRollout requires a verified factory")

    @classmethod
    def fixture(
        cls,
        group_id: str,
        rollout_id: str,
        reward: float,
        stock_sequences: tuple[FrozenTrainingSequence, ...],
        stock_sequence_decision_ids: tuple[tuple[str, ...], ...],
        decisions: tuple[RolloutDecision, ...],
        child_target_roster: tuple[str, ...],
        trace_sha256: str,
        reward_evidence_sha256: str,
        stock_sequences_evidence_sha256: str,
        base_model_manifest_sha256: str,
        branch_eligible: bool = True,
        ineligibility_reason: str | None = None,
    ) -> SourceRollout:
        """Construct explicit non-live evidence for deterministic CPU tests only."""
        return cls._construct(
            group_id,
            rollout_id,
            reward,
            stock_sequences,
            stock_sequence_decision_ids,
            decisions,
            child_target_roster,
            branch_eligible,
            ineligibility_reason,
            trace_sha256,
            reward_evidence_sha256,
            stock_sequences_evidence_sha256,
            base_model_manifest_sha256,
            "fixture-only",
            None,
        )

    @classmethod
    def verify_bytes(
        cls,
        value: bytes,
        *,
        verifier: ReceiptVerifier,
        evidence_loader: Callable[[str], bytes],
        encode_action: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[int, ...]],
        render_prompt: Callable[[Mapping[str, Any]], tuple[int, ...]],
    ) -> SourceRollout:
        """Reconstruct one live source only from canonical, anchored evidence."""
        if type(value) is not bytes:
            raise ValueError("source rollout must be immutable bytes")
        try:
            envelope = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("source rollout must be canonical JSON") from error
        if not isinstance(envelope, dict) or canonical_json(envelope) != value:
            raise ValueError("source rollout must be canonical JSON")
        if set(envelope) != {
            "schema_version",
            "domain",
            "source",
            "source_sha256",
            "producer_receipt",
        }:
            raise ValueError("source rollout envelope fields differ")
        if (
            envelope["schema_version"] != SCHEMA_VERSION
            or envelope["domain"] != "redco-stage-d-source-rollout-v1"
        ):
            raise ValueError("unsupported source rollout")
        payload = envelope["source"]
        if not isinstance(payload, dict):
            raise ValueError("source rollout payload must be an object")
        producer = envelope["producer_receipt"]
        if not isinstance(producer, dict):
            raise ValueError("live source rollout lacks a producer receipt")
        source = _source_from_payload(
            payload,
            canonical_json(producer),
            verifier=verifier,
            encode_action=encode_action,
            render_prompt=render_prompt,
        )
        if source.evidence_class != "live":
            raise ValueError("verified source rollout must carry live evidence")
        if envelope["source_sha256"] != source.source_sha256:
            raise ValueError("source rollout digest mismatch")
        _verify_source_evidence(source, evidence_loader)
        _verify_source_producer_receipt(source, verifier)
        if source.to_bytes() != value:
            raise ValueError("source rollout derived fields disagree")
        return source

    @classmethod
    def _construct(
        cls,
        group_id: str,
        rollout_id: str,
        reward: float,
        stock_sequences: tuple[FrozenTrainingSequence, ...],
        stock_sequence_decision_ids: tuple[tuple[str, ...], ...],
        decisions: tuple[RolloutDecision, ...],
        child_target_roster: tuple[str, ...],
        branch_eligible: bool,
        ineligibility_reason: str | None,
        trace_sha256: str,
        reward_evidence_sha256: str,
        stock_sequences_evidence_sha256: str,
        base_model_manifest_sha256: str,
        evidence_class: Literal["live", "fixture-only"],
        producer_receipt: bytes | None,
    ) -> SourceRollout:
        self = object.__new__(cls)
        for name, item in (
            ("group_id", group_id),
            ("rollout_id", rollout_id),
            ("reward", reward),
            ("stock_sequences", stock_sequences),
            ("stock_sequence_decision_ids", stock_sequence_decision_ids),
            ("decisions", decisions),
            ("child_target_roster", child_target_roster),
            ("branch_eligible", branch_eligible),
            ("ineligibility_reason", ineligibility_reason),
            ("trace_sha256", trace_sha256),
            ("reward_evidence_sha256", reward_evidence_sha256),
            ("stock_sequences_evidence_sha256", stock_sequences_evidence_sha256),
            ("base_model_manifest_sha256", base_model_manifest_sha256),
            ("evidence_class", evidence_class),
            ("producer_receipt", producer_receipt),
        ):
            object.__setattr__(self, name, item)
        self.__post_init__()
        return self

    def __post_init__(self) -> None:
        if not self.group_id or not self.rollout_id:
            raise ValueError("source rollout identifiers must be nonempty")
        _finite(self.reward, "source reward")
        _require_sha256(self.trace_sha256, "source trace sha256")
        _require_sha256(self.reward_evidence_sha256, "reward evidence sha256")
        _require_sha256(
            self.stock_sequences_evidence_sha256,
            "stock sequences evidence sha256",
        )
        _require_sha256(
            self.base_model_manifest_sha256,
            "base model manifest sha256",
        )
        if not self.decisions:
            raise ValueError("source rollout must contain policy decisions")
        if type(self.branch_eligible) is not bool:
            raise ValueError("source branch eligibility must be an explicit bool")
        if self.branch_eligible:
            if self.ineligibility_reason is not None:
                raise ValueError("eligible source rollout cannot name an ineligibility reason")
        elif not isinstance(self.ineligibility_reason, str) or not self.ineligibility_reason:
            raise ValueError("ineligible source rollout requires a nonempty reason")
        if (self.evidence_class == "live") != (self.producer_receipt is not None):
            raise ValueError("only live source rollouts require a producer receipt")
        if self.producer_receipt is not None:
            try:
                parsed_producer_receipt = json.loads(self.producer_receipt)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    "source producer receipt must be canonical immutable bytes"
                ) from error
            if (
                type(self.producer_receipt) is not bytes
                or not self.producer_receipt
                or canonical_json(parsed_producer_receipt) != self.producer_receipt
            ):
                raise ValueError("source producer receipt must be canonical immutable bytes")
        if not self.stock_sequences or len(self.stock_sequences) != len(
            self.stock_sequence_decision_ids
        ):
            raise ValueError("stock sequence boundaries and decision rosters must align")
        if any(
            sequence.rl_weights is not None or sequence.rl_normalizer is not None
            for sequence in self.stock_sequences
        ):
            raise ValueError("stock sequences must preserve exact Prime token normalization")
        ids = [decision.decision_id for decision in self.decisions]
        if len(set(ids)) != len(ids):
            raise ValueError("source rollout decision IDs must be unique")
        addresses = [
            canonical_json(decision.event_address.as_payload()) for decision in self.decisions
        ]
        if len(set(addresses)) != len(addresses):
            raise ValueError("source rollout event addresses must be unique")
        ledger_ids = {decision.provenance.ledger_id for decision in self.decisions}
        if len(ledger_ids) != 1:
            raise ValueError("source rollout decisions cross receipt ledgers")
        if any(
            decision.provenance.group_id != self.group_id
            or decision.provenance.rollout_id != self.rollout_id
            for decision in self.decisions
        ):
            raise ValueError("source rollout decisions cross groups or rollouts")
        request_sequences = [decision.provenance.request_sequence for decision in self.decisions]
        if request_sequences != sorted(request_sequences) or len(set(request_sequences)) != len(
            request_sequences
        ):
            raise ValueError("source rollout decisions must follow unique request order")
        receipt_hashes = [
            _sha256(receipt)
            for decision in self.decisions
            for receipt in (
                decision.provenance.reservation_receipt,
                decision.provenance.completion_receipt,
            )
        ]
        if len(set(receipt_hashes)) != len(receipt_hashes):
            raise ValueError("source rollout receipt hashes must be unique")
        child_decisions = tuple(
            decision for decision in self.decisions if decision.node_kind == "child"
        )
        child_targets = tuple(str(decision.target_id) for decision in child_decisions)
        if len(set(self.child_target_roster)) != len(self.child_target_roster) or set(
            child_targets
        ) != set(self.child_target_roster):
            raise ValueError("predeclared child roster must biject child decisions")
        if any(
            decision.target_ordinal >= len(self.child_target_roster)
            or self.child_target_roster[decision.target_ordinal] != decision.target_id
            for decision in child_decisions
            if decision.target_ordinal is not None
        ):
            raise ValueError("child target ordinals differ from the predeclared roster")
        if child_decisions:
            expected_child_weight = Fraction(1, len(self.child_target_roster))
            if any(decision.outer_weight != expected_child_weight for decision in child_decisions):
                raise ValueError("child decisions must share one exact child-count weight")
        roster_ids = tuple(
            decision_id for roster in self.stock_sequence_decision_ids for decision_id in roster
        )
        if len(roster_ids) != len(ids) or set(roster_ids) != set(ids):
            raise ValueError("stock sequence rosters must biject every policy decision")
        by_id = {decision.decision_id: decision for decision in self.decisions}
        request_order = {decision_id: index for index, decision_id in enumerate(ids)}
        for sequence, roster in zip(
            self.stock_sequences,
            self.stock_sequence_decision_ids,
            strict=True,
        ):
            if not roster:
                raise ValueError("every stock sequence must own at least one decision")
            if tuple(request_order[decision_id] for decision_id in roster) != tuple(
                sorted(request_order[decision_id] for decision_id in roster)
            ):
                raise ValueError("each stock sequence roster must preserve request chronology")
            sequence_decisions = tuple(by_id[decision_id] for decision_id in roster)
            _validate_stock_sequence_decisions(sequence, sequence_decisions)
            if self.evidence_class == "fixture-only":
                if len(sequence_decisions) != 1:
                    raise ValueError("fixture stock sequences require one exact decision")
                decision = sequence_decisions[0]
                prompt = decision.action.key.prompt_token_ids
                action = decision.action.action_token_ids
                if sequence.token_ids != (*prompt, *action) or sequence.mask != (
                    (False,) * len(prompt) + (True,) * len(action)
                ):
                    raise ValueError("fixture stock sequence changed its exact request context")
        object.__setattr__(self, "source_sha256", _source_rollout_sha256(self))

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "group_id": self.group_id,
            "rollout_id": self.rollout_id,
            "reward": self.reward,
            "stock_sequences": [sequence.to_payload() for sequence in self.stock_sequences],
            "stock_sequence_decision_ids": [
                list(roster) for roster in self.stock_sequence_decision_ids
            ],
            "decisions": [_decision_payload(decision) for decision in self.decisions],
            "child_target_roster": list(self.child_target_roster),
            "trace_sha256": self.trace_sha256,
            "reward_evidence_sha256": self.reward_evidence_sha256,
            "stock_sequences_evidence_sha256": self.stock_sequences_evidence_sha256,
            "base_model_manifest_sha256": self.base_model_manifest_sha256,
            "evidence_class": self.evidence_class,
        }
        if not self.branch_eligible:
            payload["branch_eligible"] = False
            payload["ineligibility_reason"] = self.ineligibility_reason
        return payload

    def to_bytes(self) -> bytes:
        payload = self.to_payload()
        return canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": "redco-stage-d-source-rollout-v1",
                "source": payload,
                "source_sha256": self.source_sha256,
                "producer_receipt": (
                    None if self.producer_receipt is None else json.loads(self.producer_receipt)
                ),
            }
        )


def _decision_payload(decision: RolloutDecision) -> dict[str, Any]:
    return {
        "decision_id": decision.decision_id,
        "event_address": {
            **decision.event_address.as_payload(),
            "turn": decision.event_address.turn,
        },
        "action": json.loads(decision.action.to_bytes()),
        "node_kind": decision.node_kind,
        "target_id": decision.target_id,
        "target_ordinal": decision.target_ordinal,
        "outer_weight": _fraction_payload(decision.outer_weight),
        "provenance": decision.provenance.to_payload(),
    }


def _verified_source_receipt(
    receipt: bytes,
    receipt_kind: str,
    verifier: ReceiptVerifier,
) -> dict[str, Any]:
    if type(receipt) is not bytes:
        raise ValueError("source policy receipt must be immutable bytes")
    try:
        parsed = json.loads(receipt)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source policy receipt must be canonical JSON") from error
    if not isinstance(parsed, dict) or canonical_json(parsed) != receipt:
        raise ValueError("source policy receipt must be canonical JSON")
    verified = dict(verifier(receipt, receipt_kind=receipt_kind))
    if verified != parsed:
        raise ValueError("source policy receipt verifier returned different bytes")
    if parsed.get("schema_version") != 1 or parsed.get("receipt_kind") != receipt_kind:
        raise ValueError("source policy receipt envelope is invalid")
    return parsed


def _source_from_payload(
    payload: Mapping[str, Any],
    producer_receipt: bytes,
    *,
    verifier: ReceiptVerifier,
    encode_action: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[int, ...]],
    render_prompt: Callable[[Mapping[str, Any]], tuple[int, ...]],
) -> SourceRollout:
    expected = {
        "group_id",
        "rollout_id",
        "reward",
        "stock_sequences",
        "stock_sequence_decision_ids",
        "decisions",
        "child_target_roster",
        "trace_sha256",
        "reward_evidence_sha256",
        "stock_sequences_evidence_sha256",
        "base_model_manifest_sha256",
        "evidence_class",
    }
    ineligibility_fields = {"branch_eligible", "ineligibility_reason"}
    if set(payload) not in {frozenset(expected), frozenset(expected | ineligibility_fields)}:
        raise ValueError("source rollout payload fields differ")
    branch_eligible = payload.get("branch_eligible", True)
    ineligibility_reason = payload.get("ineligibility_reason")
    raw_sequences = payload["stock_sequences"]
    raw_rosters = payload["stock_sequence_decision_ids"]
    raw_decisions = payload["decisions"]
    raw_child_roster = payload["child_target_roster"]
    if (
        not isinstance(raw_sequences, list)
        or not isinstance(raw_rosters, list)
        or not isinstance(raw_decisions, list)
        or not isinstance(raw_child_roster, list)
    ):
        raise ValueError("source rollout collections must be lists")
    if any(
        not isinstance(roster, list)
        or any(not isinstance(item, str) or not item for item in roster)
        for roster in raw_rosters
    ) or any(not isinstance(item, str) or not item for item in raw_child_roster):
        raise ValueError("source rollout rosters are invalid")
    decisions = tuple(
        _rollout_decision_from_payload(
            value,
            verifier=verifier,
            encode_action=encode_action,
            render_prompt=render_prompt,
        )
        for value in raw_decisions
    )
    return SourceRollout._construct(
        payload["group_id"],
        payload["rollout_id"],
        payload["reward"],
        tuple(_training_sequence_from_payload(item) for item in raw_sequences),
        tuple(tuple(roster) for roster in raw_rosters),
        decisions,
        tuple(raw_child_roster),
        branch_eligible,
        ineligibility_reason,
        payload["trace_sha256"],
        payload["reward_evidence_sha256"],
        payload["stock_sequences_evidence_sha256"],
        payload["base_model_manifest_sha256"],
        payload["evidence_class"],
        producer_receipt,
    )


def _training_sequence_from_payload(value: object) -> FrozenTrainingSequence:
    expected = {
        "token_ids",
        "mask",
        "behavior_logprobs",
        "temperatures",
        "rl_weights",
        "rl_normalizer",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("source training sequence fields differ")
    token_ids = value["token_ids"]
    mask = value["mask"]
    logprobs = value["behavior_logprobs"]
    temperatures = value["temperatures"]
    weights = value["rl_weights"]
    if (
        not isinstance(token_ids, list)
        or any(type(item) is not int for item in token_ids)
        or not isinstance(mask, list)
        or any(type(item) is not bool for item in mask)
        or not isinstance(logprobs, list)
        or any(type(item) is not float for item in logprobs)
        or not isinstance(temperatures, list)
        or any(type(item) is not float for item in temperatures)
        or (
            weights is not None
            and (not isinstance(weights, list) or any(type(item) is not float for item in weights))
        )
    ):
        raise ValueError("source training sequence values are invalid")
    return FrozenTrainingSequence(
        tuple(token_ids),
        tuple(mask),
        tuple(logprobs),
        tuple(temperatures),
        None if weights is None else tuple(weights),
        (
            None
            if value["rl_normalizer"] is None
            else _fraction_from_payload(value["rl_normalizer"], "rl_normalizer")
        ),
    )


def _rollout_decision_from_payload(
    value: object,
    *,
    verifier: ReceiptVerifier,
    encode_action: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[int, ...]],
    render_prompt: Callable[[Mapping[str, Any]], tuple[int, ...]],
) -> RolloutDecision:
    expected = {
        "decision_id",
        "event_address",
        "action",
        "node_kind",
        "target_id",
        "target_ordinal",
        "outer_weight",
        "provenance",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("source rollout decision fields differ")
    provenance = value["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
        "reservation_receipt",
        "completion_receipt",
    }:
        raise ValueError("source rollout decision provenance fields differ")
    reservation = provenance["reservation_receipt"]
    completion = provenance["completion_receipt"]
    action = value["action"]
    if (
        not isinstance(reservation, dict)
        or not isinstance(completion, dict)
        or not isinstance(action, dict)
    ):
        raise ValueError("source rollout decision evidence must be objects")
    verified_provenance = DecisionProvenance.from_receipts(
        canonical_json(reservation),
        canonical_json(completion),
        verifier=verifier,
    )
    behavior_action = BehaviorAction.from_bytes(
        canonical_json(action),
        encode_action=encode_action,
        render_prompt=render_prompt,
    )
    return RolloutDecision(
        value["decision_id"],
        _policy_event_address(value["event_address"]),
        behavior_action,
        value["node_kind"],
        value["target_id"],
        value["target_ordinal"],
        _fraction_from_payload(value["outer_weight"], "outer_weight"),
        verified_provenance,
    )


def _verify_source_evidence(
    source: SourceRollout,
    evidence_loader: Callable[[str], bytes],
) -> None:
    evidence: dict[str, bytes] = {}
    for digest in (
        source.trace_sha256,
        source.reward_evidence_sha256,
        source.stock_sequences_evidence_sha256,
    ):
        value = evidence_loader(digest)
        if type(value) is not bytes or not value or _sha256(value) != digest:
            raise ValueError("source rollout evidence is absent or corrupted")
        evidence[digest] = value
    expected_reward = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-reward-evidence-v1",
            "group_id": source.group_id,
            "rollout_id": source.rollout_id,
            "reward": source.reward,
        }
    )
    if evidence[source.reward_evidence_sha256] != expected_reward:
        raise ValueError("source rollout reward differs from its raw evidence")
    expected_sequences = canonical_json(
        [sequence.to_payload() for sequence in source.stock_sequences]
    )
    if evidence[source.stock_sequences_evidence_sha256] != expected_sequences:
        raise ValueError("source rollout training sequences differ from raw evidence")
    from redco.analysis.stage_d_source_producer import verify_source_trace_semantics

    verify_source_trace_semantics(
        source,
        raw_episode=evidence[source.trace_sha256],
    )


def _verify_source_producer_receipt(
    source: SourceRollout,
    verifier: ReceiptVerifier,
) -> None:
    if source.producer_receipt is None:
        raise ValueError("live source rollout lacks a producer receipt")
    receipt = _verified_source_receipt(
        source.producer_receipt,
        "source_rollout_completed",
        verifier,
    )
    expected_fields = {
        "schema_version",
        "receipt_kind",
        "ledger_id",
        "ledger_offset",
        "prior_chain_sha256",
        "group_id",
        "rollout_id",
        "source_sha256",
        "trace_sha256",
        "reward_evidence_sha256",
        "stock_sequences_evidence_sha256",
        "base_model_manifest_sha256",
        "decision_ids",
        "decision_completion_receipt_sha256s",
        "completion_sequence",
    }
    if set(receipt) != expected_fields:
        raise ValueError("source producer receipt fields differ")
    decision_ids = [decision.decision_id for decision in source.decisions]
    completion_hashes = [
        _sha256(decision.provenance.completion_receipt) for decision in source.decisions
    ]
    ledger_ids = {decision.provenance.ledger_id for decision in source.decisions}
    if receipt != {
        "schema_version": 1,
        "receipt_kind": "source_rollout_completed",
        "ledger_id": next(iter(ledger_ids)),
        "ledger_offset": receipt["ledger_offset"],
        "prior_chain_sha256": receipt["prior_chain_sha256"],
        "group_id": source.group_id,
        "rollout_id": source.rollout_id,
        "source_sha256": source.source_sha256,
        "trace_sha256": source.trace_sha256,
        "reward_evidence_sha256": source.reward_evidence_sha256,
        "stock_sequences_evidence_sha256": source.stock_sequences_evidence_sha256,
        "base_model_manifest_sha256": source.base_model_manifest_sha256,
        "decision_ids": decision_ids,
        "decision_completion_receipt_sha256s": completion_hashes,
        "completion_sequence": receipt["completion_sequence"],
    }:
        raise ValueError("source producer receipt differs from reconstructed evidence")
    if (
        type(receipt["ledger_offset"]) is not int
        or type(receipt["completion_sequence"]) is not int
        or receipt["ledger_offset"] != receipt["completion_sequence"]
        or receipt["completion_sequence"]
        <= max(decision.provenance.completion_sequence for decision in source.decisions)
    ):
        raise ValueError("source producer receipt chronology is invalid")
    _require_sha256(receipt["prior_chain_sha256"], "producer prior chain sha256")


def _policy_event_address(value: object) -> PolicyEventAddress:
    if not isinstance(value, dict) or set(value) != {
        "depth",
        "lineage",
        "session_call_ordinal",
        "turn",
        "call_kind",
    }:
        raise ValueError("source policy address fields differ")
    if any(
        type(value[field]) is not int for field in ("depth", "session_call_ordinal", "turn")
    ) or any(
        not isinstance(value[field], str) or not value[field] for field in ("lineage", "call_kind")
    ):
        raise ValueError("source policy address values are invalid")
    return PolicyEventAddress(
        value["depth"],
        value["lineage"],
        value["session_call_ordinal"],
        value["turn"],
        value["call_kind"],
    )


def _validate_stock_sequence_decisions(
    sequence: FrozenTrainingSequence,
    decisions: tuple[RolloutDecision, ...],
) -> None:
    selected_tokens = tuple(
        token for token, selected in zip(sequence.token_ids, sequence.mask, strict=True) if selected
    )
    selected_logprobs = tuple(
        value
        for value, selected in zip(
            sequence.behavior_logprobs,
            sequence.mask,
            strict=True,
        )
        if selected
    )
    selected_temperatures = tuple(
        value
        for value, selected in zip(sequence.temperatures, sequence.mask, strict=True)
        if selected
    )
    decision_tokens = tuple(
        token for decision in decisions for token in decision.action.action_token_ids
    )
    decision_logprobs = tuple(
        value for decision in decisions for value in decision.action.behavior_logprobs
    )
    decision_temperatures = tuple(
        decision.action.key.sampler.temperature
        for decision in decisions
        for _ in decision.action.action_token_ids
    )
    if selected_tokens != decision_tokens or selected_logprobs != decision_logprobs:
        raise ValueError("stock sequence does not exactly cover its policy decisions")
    if selected_temperatures != decision_temperatures:
        raise ValueError("stock temperatures do not exactly cover its policy decisions")


def _source_rollout_sha256(source: SourceRollout) -> str:
    return _sha256(
        canonical_json(
            {
                "domain": "redco-stage-d-source-rollout-v1",
                "source": source.to_payload(),
            }
        )
    )
