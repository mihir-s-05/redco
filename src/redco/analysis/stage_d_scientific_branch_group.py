"""Pure receipt-verification contract for Stage D scientific branch groups."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from math import fsum
from typing import Any, Literal, Protocol

from redco.algo.branching import leave_one_out_advantages
from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.analysis.stage_d_spawn_provenance import (
    EventSeedScheduler,
    PolicyEventAddress,
    ScheduledSeed,
)
from redco.contracts import ActualEvaluationCost, LogicalDeploymentCost, canonical_json

SCHEMA_VERSION = 2
_DOMAIN = "redco-stage-d-scientific-branch-group-v2"


class ReceiptVerifier(Protocol):
    """Trusted external boundary backed by a durable append-only producer in D1."""

    def __call__(self, receipt: bytes, *, receipt_kind: str) -> Mapping[str, Any]: ...


class CandidateSampler(Protocol):
    def __call__(
        self,
        *,
        action_slot: int,
        action_seed: int,
        reference_key: ExactActionKey,
    ) -> CandidateSubmission: ...


class ArmExecutor(Protocol):
    def __call__(
        self,
        *,
        arm_id: str,
        action: BehaviorAction,
        continuation_replicate: int,
        seed_oracle: BranchSeedOracle,
    ) -> bytes: ...


class RepairableInfrastructureAbort(RuntimeError):
    """Trusted receipt proves zero scientific model calls occurred."""


class NonRepairableCampaignAbort(RuntimeError):
    """A raw failure cannot prove that no scientific information was observed."""


class ZeroCallInfrastructureFailure(Exception):
    """Producer-raised zero-call failure carrying its trusted receipt."""

    def __init__(self, receipt: bytes) -> None:
        super().__init__("zero-call infrastructure failure")
        self.receipt = receipt


class OutcomeKind(StrEnum):
    SUCCESS = "success"
    MALFORMED_ACTION = "malformed_action"
    RUNTIME_EXCEPTION = "runtime_exception"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    TERMINAL_WITHOUT_DOWNSTREAM = "terminal_without_downstream"


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


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _strict_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - observed)} "
            f"unknown={sorted(observed - expected)}"
        )


def _verified_receipt(
    receipt: bytes,
    *,
    receipt_kind: str,
    verifier: ReceiptVerifier,
) -> dict[str, Any]:
    if type(receipt) is not bytes:
        raise ValueError("receipt must be immutable bytes")
    try:
        parsed = json.loads(receipt)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("receipt must be canonical JSON") from error
    if not isinstance(parsed, dict) or canonical_json(parsed) != receipt:
        raise ValueError("receipt must be canonical JSON")
    verified = dict(verifier(receipt, receipt_kind=receipt_kind))
    if canonical_json(verified) != receipt:
        raise ValueError("trusted verifier returned different receipt bytes")
    if parsed.get("schema_version") != 1 or parsed.get("receipt_kind") != receipt_kind:
        raise ValueError(f"invalid {receipt_kind} receipt envelope")
    return parsed


def _address_from_payload(value: object, name: str) -> PolicyEventAddress:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an address object")
    _strict_keys(
        value,
        {"depth", "lineage", "session_call_ordinal", "turn", "call_kind"},
        name,
    )
    lineage = value["lineage"]
    call_kind = value["call_kind"]
    if not isinstance(lineage, str) or not isinstance(call_kind, str):
        raise ValueError(f"{name} text fields are invalid")
    return PolicyEventAddress(
        _exact_int(value["depth"], f"{name}.depth"),
        lineage,
        _exact_int(value["session_call_ordinal"], f"{name}.session_call_ordinal"),
        _exact_int(value["turn"], f"{name}.turn"),
        call_kind,
    )


def _address_payload(address: PolicyEventAddress) -> dict[str, str | int]:
    return {**address.as_payload(), "turn": address.turn}


def _address_key(address: PolicyEventAddress) -> bytes:
    return canonical_json(address.as_payload())


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def behavior_law_digest(key: ExactActionKey) -> str:
    """Identify one behavior law while excluding only action seed and cache salt."""
    if type(key) is not ExactActionKey:
        raise ValueError("behavior law requires a validated ExactActionKey")
    payload = key.to_payload()
    request = dict(payload["request"])
    sampler = dict(payload["sampler_config"])
    request.pop("seed")
    sampler.pop("seed")
    extra_body = dict(request.get("extra_body") or {})
    extra_body.pop("cache_salt", None)
    request["extra_body"] = extra_body
    payload.pop("request_sha256")
    payload.pop("sampler_config_sha256")
    payload["request"] = request
    payload["sampler_config"] = sampler
    prepared = payload.get("prepared_engine_request")
    if isinstance(prepared, dict):
        prepared = dict(prepared)
        prepared_sampling = prepared.get("sampling_params")
        if not isinstance(prepared_sampling, dict):
            raise ValueError("prepared behavior law lacks sampling parameters")
        prepared_sampling = dict(prepared_sampling)
        prepared_sampling.pop("seed")
        prepared.pop("cache_salt", None)
        prepared_sampling.pop("cache_salt", None)
        prepared["sampling_params"] = prepared_sampling
        payload["prepared_engine_request"] = prepared
        payload.pop("prepared_engine_request_sha256")
    return _sha256(canonical_json({"domain": "redco-behavior-law-v1", "key": payload}))


@dataclass(frozen=True, slots=True)
class PreActionTargetCommitment:
    receipt: bytes
    receipt_sha256: str
    ledger_id: str
    ledger_offset: int
    prior_chain_sha256: str
    group_id: str
    rollout_id: str
    target_roster: tuple[str, ...]
    target_ordinal: int
    target_id: str
    target_address: PolicyEventAddress
    pre_action_snapshot_sha256: str
    behavior_law_sha256: str
    recorded_action_seed: int
    branch_count: int
    continuation_replicates: int
    failure_reward: float
    master_seed_sha256: str
    commitment_sequence: int
    action_reservation_sequence: int

    @classmethod
    def from_receipt(
        cls,
        receipt: bytes,
        *,
        verifier: ReceiptVerifier,
    ) -> PreActionTargetCommitment:
        value = _verified_receipt(
            receipt,
            receipt_kind="pre_action_group_commitment",
            verifier=verifier,
        )
        expected = {
            "schema_version",
            "receipt_kind",
            "ledger_id",
            "ledger_offset",
            "prior_chain_sha256",
            "phase",
            "group_id",
            "rollout_id",
            "target_roster",
            "target_ordinal",
            "target_id",
            "target_address",
            "pre_action_snapshot_sha256",
            "behavior_law_sha256",
            "recorded_action_seed",
            "branch_count",
            "continuation_replicates",
            "failure_reward",
            "master_seed_sha256",
            "commitment_sequence",
            "action_reservation_sequence",
        }
        _strict_keys(value, expected, "commitment receipt")
        if value["phase"] != "pre_action":
            raise ValueError("commitment receipt is not pre-action")
        ledger_id = value["ledger_id"]
        group_id = value["group_id"]
        rollout_id = value["rollout_id"]
        target_id = value["target_id"]
        roster = value["target_roster"]
        if any(not isinstance(item, str) or not item for item in (ledger_id, group_id, rollout_id)):
            raise ValueError("commitment identifiers must be nonempty")
        if (
            not isinstance(roster, list)
            or not roster
            or any(not isinstance(item, str) or not item for item in roster)
            or len(set(roster)) != len(roster)
        ):
            raise ValueError("target roster must contain unique nonempty IDs")
        target_ordinal = _exact_int(value["target_ordinal"], "target_ordinal")
        if target_ordinal >= len(roster) or roster[target_ordinal] != target_id:
            raise ValueError("target ordinal does not identify target_id in the frozen roster")
        committed = _exact_int(value["commitment_sequence"], "commitment_sequence")
        reserved = _exact_int(
            value["action_reservation_sequence"],
            "action_reservation_sequence",
            minimum=1,
        )
        if committed >= reserved:
            raise ValueError("target commitment must precede action reservation")
        return cls(
            receipt,
            _sha256(receipt),
            ledger_id,
            _exact_int(value["ledger_offset"], "ledger_offset"),
            _require_sha256(value["prior_chain_sha256"], "prior_chain_sha256"),
            group_id,
            rollout_id,
            tuple(roster),
            target_ordinal,
            target_id,
            _address_from_payload(value["target_address"], "target_address"),
            _require_sha256(value["pre_action_snapshot_sha256"], "snapshot"),
            _require_sha256(value["behavior_law_sha256"], "behavior law"),
            _exact_int(value["recorded_action_seed"], "recorded_action_seed"),
            _exact_int(value["branch_count"], "branch_count", minimum=2),
            _exact_int(
                value["continuation_replicates"],
                "continuation_replicates",
                minimum=1,
            ),
            _finite_float(value["failure_reward"], "failure_reward"),
            _require_sha256(value["master_seed_sha256"], "master seed"),
            committed,
            reserved,
        )

    @property
    def outer_weight(self) -> Fraction:
        return Fraction(1, len(self.target_roster))

    def to_payload(self) -> dict[str, Any]:
        return {"receipt": json.loads(self.receipt), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class SeedCorrespondenceMap:
    receipt: bytes
    receipt_sha256: str
    group_id: str
    target_id: str
    pre_action_snapshot_sha256: str
    recorded_action_digest: str
    matched_addresses: tuple[PolicyEventAddress, ...]

    @classmethod
    def from_receipt(
        cls,
        receipt: bytes,
        *,
        verifier: ReceiptVerifier,
        commitment: PreActionTargetCommitment,
        recorded_action: BehaviorAction,
    ) -> SeedCorrespondenceMap:
        value = _verified_receipt(
            receipt,
            receipt_kind="seed_correspondence_map",
            verifier=verifier,
        )
        _strict_keys(
            value,
            {
                "schema_version",
                "receipt_kind",
                "group_id",
                "target_id",
                "pre_action_snapshot_sha256",
                "recorded_action_digest",
                "matched_addresses",
            },
            "correspondence receipt",
        )
        addresses = value["matched_addresses"]
        if not isinstance(addresses, list):
            raise ValueError("matched_addresses must be a list")
        parsed = tuple(
            _address_from_payload(address, f"matched_addresses[{index}]")
            for index, address in enumerate(addresses)
        )
        keys = tuple(_address_key(address) for address in parsed)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("matched address set must be sorted and collision-free")
        if (
            value["group_id"] != commitment.group_id
            or value["target_id"] != commitment.target_id
            or value["pre_action_snapshot_sha256"]
            != commitment.pre_action_snapshot_sha256
            or value["recorded_action_digest"] != recorded_action.digest
        ):
            raise ValueError("correspondence map is bound to a different group")
        return cls(
            receipt,
            _sha256(receipt),
            commitment.group_id,
            commitment.target_id,
            commitment.pre_action_snapshot_sha256,
            recorded_action.digest,
            parsed,
        )

    @property
    def matched_keys(self) -> frozenset[bytes]:
        return frozenset(_address_key(address) for address in self.matched_addresses)

    def to_payload(self) -> dict[str, Any]:
        return {"receipt": json.loads(self.receipt), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CandidateSubmission:
    action: BehaviorAction
    inference_receipt: bytes


@dataclass(frozen=True, slots=True)
class CandidateSample:
    action_slot: int
    action_seed: int
    action: BehaviorAction
    receipt: bytes
    receipt_sha256: str
    inference_call_id: str

    @classmethod
    def validate(
        cls,
        submission: CandidateSubmission,
        *,
        action_slot: int,
        action_seed: int,
        commitment: PreActionTargetCommitment,
        verifier: ReceiptVerifier,
    ) -> CandidateSample:
        if (
            type(submission) is not CandidateSubmission
            or type(submission.action) is not BehaviorAction
        ):
            raise ValueError("candidate sampler did not materialize a BehaviorAction submission")
        value = _verified_receipt(
            submission.inference_receipt,
            receipt_kind="candidate_action_inference",
            verifier=verifier,
        )
        _strict_keys(
            value,
            {
                "schema_version",
                "receipt_kind",
                "group_id",
                "target_id",
                "action_slot",
                "action_seed",
                "action_digest",
                "action_evidence_sha256",
                "behavior_law_sha256",
                "selection_policy",
                "sample_attempts",
                "rejected_attempts",
                "inference_call_id",
                "prompt_tokens",
                "completion_tokens",
                "response_sha256",
            },
            "candidate receipt",
        )
        call_id = value["inference_call_id"]
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("candidate inference_call_id must be nonempty")
        if (
            value["group_id"] != commitment.group_id
            or value["target_id"] != commitment.target_id
            or value["action_slot"] != action_slot
            or value["action_seed"] != action_seed
            or value["action_digest"] != submission.action.digest
            or value["action_evidence_sha256"] != _sha256(
                submission.action.to_bytes()
            )
            or value["behavior_law_sha256"] != commitment.behavior_law_sha256
            or value["selection_policy"] != "direct_single_sample"
            or value["sample_attempts"] != 1
            or value["rejected_attempts"] != 0
            or value["prompt_tokens"] != submission.action.prompt_tokens
            or value["completion_tokens"] != submission.action.completion_tokens
        ):
            raise ValueError("candidate inference receipt violates the frozen sample contract")
        _require_sha256(value["action_evidence_sha256"], "candidate action evidence")
        _require_sha256(value["response_sha256"], "candidate raw response evidence")
        if behavior_law_digest(submission.action.key) != commitment.behavior_law_sha256:
            raise ValueError("candidate changed the behavior law")
        if submission.action.key.sampler.seed != action_seed:
            raise ValueError("candidate used the wrong action seed")
        return cls(
            action_slot,
            action_seed,
            submission.action,
            submission.inference_receipt,
            _sha256(submission.inference_receipt),
            call_id,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "action": json.loads(self.action.to_bytes()),
            "inference_receipt": json.loads(self.receipt),
            "inference_receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class InferenceCallReceipt:
    call_id: str
    address: PolicyEventAddress
    scheduled_seed: ScheduledSeed
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class ReplayedInferenceCallReceipt:
    override_id: str
    address: PolicyEventAddress
    action_digest: str
    disposition: Literal["reuse", "inject"]
    prompt_tokens: int
    completion_tokens: int
    counts_toward_logical_cost: bool


class BranchSeedOracle:
    """Choose paired versus exogenous namespaces from a trusted correspondence map."""

    def __init__(
        self,
        scheduler: EventSeedScheduler,
        *,
        arm_id: str,
        correspondence: SeedCorrespondenceMap,
    ) -> None:
        self._scheduler = scheduler
        self._arm_id = arm_id
        self._matched = correspondence.matched_keys

    def seed_for(self, address: PolicyEventAddress) -> ScheduledSeed:
        if _address_key(address) in self._matched:
            return self._scheduler.paired_continuation_seed(
                address,
                committed_address=address,
            )
        return self._scheduler.exogenous_continuation_seed(
            address,
            action_arm=self._arm_id,
        )


@dataclass(frozen=True, slots=True)
class BranchOutcome:
    receipt: bytes
    receipt_sha256: str
    execution_id: str
    arm_id: str
    action_digest: str
    continuation_replicate: int
    kind: OutcomeKind
    reward: float
    calls: tuple[InferenceCallReceipt, ...]
    logical_cost: LogicalDeploymentCost
    actual_cost: ActualEvaluationCost
    replayed_calls: tuple[ReplayedInferenceCallReceipt, ...] = ()

    @classmethod
    def from_receipt(
        cls,
        receipt: bytes,
        *,
        verifier: ReceiptVerifier,
        commitment: PreActionTargetCommitment,
        correspondence: SeedCorrespondenceMap,
        action: BehaviorAction,
        arm_id: str,
        continuation_replicate: int,
        master_seed: str,
    ) -> BranchOutcome:
        value = _verified_receipt(
            receipt,
            receipt_kind="scientific_arm_execution",
            verifier=verifier,
        )
        receipt_fields = {
            "schema_version",
            "receipt_kind",
            "group_id",
            "target_id",
            "arm_id",
            "action_digest",
            "continuation_replicate",
            "execution_id",
            "outcome_kind",
            "reward",
            "calls",
            "logical_cost",
            "actual_non_token_cost",
        }
        if "replayed_calls" in value:
            receipt_fields.add("replayed_calls")
        _strict_keys(
            value,
            receipt_fields,
            "execution receipt",
        )
        if (
            value["group_id"] != commitment.group_id
            or value["target_id"] != commitment.target_id
            or value["arm_id"] != arm_id
            or value["action_digest"] != action.digest
            or value["continuation_replicate"] != continuation_replicate
        ):
            raise ValueError("execution receipt is bound to a different arm")
        execution_id = value["execution_id"]
        if not isinstance(execution_id, str) or not execution_id:
            raise ValueError("execution_id must be nonempty")
        try:
            kind = OutcomeKind(value["outcome_kind"])
        except (TypeError, ValueError) as error:
            raise ValueError("unknown scientific outcome kind") from error
        reward = _finite_float(value["reward"], "reward")
        if (
            kind
            in {
                OutcomeKind.MALFORMED_ACTION,
                OutcomeKind.RUNTIME_EXCEPTION,
                OutcomeKind.TIMEOUT,
                OutcomeKind.RESOURCE_LIMIT,
            }
            and reward != commitment.failure_reward
        ):
            raise ValueError("scientific failure did not retain frozen failure reward")
        if action.parse_status == "malformed" and kind is not OutcomeKind.MALFORMED_ACTION:
            raise ValueError("malformed action was not retained as a malformed outcome")
        raw_calls = value["calls"]
        if not isinstance(raw_calls, list):
            raise ValueError("execution calls must be a list")
        scheduler = EventSeedScheduler(
            master_seed,
            commitment.rollout_id,
            commitment.target_id,
            continuation_replicate,
        )
        oracle = BranchSeedOracle(
            scheduler,
            arm_id=arm_id,
            correspondence=correspondence,
        )
        calls: list[InferenceCallReceipt] = []
        local_ids: set[str] = set()
        local_addresses: set[bytes] = set()
        for index, raw in enumerate(raw_calls):
            if not isinstance(raw, dict):
                raise ValueError("execution call receipt must be an object")
            _strict_keys(
                raw,
                {
                    "call_id",
                    "address",
                    "seed",
                    "coupling_mode",
                    "prompt_tokens",
                    "completion_tokens",
                    "disposition",
                },
                f"calls[{index}]",
            )
            call_id = raw["call_id"]
            if not isinstance(call_id, str) or not call_id or call_id in local_ids:
                raise ValueError("execution call IDs must be unique and nonempty")
            local_ids.add(call_id)
            if raw["disposition"] != "generated":
                raise ValueError("scientific downstream calls may not be reused")
            address = _address_from_payload(raw["address"], f"calls[{index}].address")
            address_key = _address_key(address)
            if address_key in local_addresses:
                raise ValueError("one execution may not call one scientific address twice")
            local_addresses.add(address_key)
            expected_seed = oracle.seed_for(address)
            if (
                raw["seed"] != expected_seed.seed
                or raw["coupling_mode"] != expected_seed.coupling_mode.value
            ):
                raise ValueError("execution call violates the correspondence seed contract")
            calls.append(
                InferenceCallReceipt(
                    call_id,
                    address,
                    expected_seed,
                    _exact_int(raw["prompt_tokens"], "prompt_tokens"),
                    _exact_int(raw["completion_tokens"], "completion_tokens"),
                )
            )
        raw_replayed = value.get("replayed_calls", [])
        if not isinstance(raw_replayed, list):
            raise ValueError("execution replay calls must be a list")
        replayed: list[ReplayedInferenceCallReceipt] = []
        replay_ids: set[str] = set()
        for index, raw in enumerate(raw_replayed):
            if not isinstance(raw, dict):
                raise ValueError("execution replay receipt must be an object")
            _strict_keys(
                raw,
                {
                    "override_id",
                    "address",
                    "action_digest",
                    "disposition",
                    "prompt_tokens",
                    "completion_tokens",
                    "counts_toward_logical_cost",
                },
                f"replayed_calls[{index}]",
            )
            override_id = raw["override_id"]
            disposition = raw["disposition"]
            counts = raw["counts_toward_logical_cost"]
            action_digest = _require_sha256(
                raw["action_digest"],
                f"replayed_calls[{index}].action_digest",
            )
            if (
                not isinstance(override_id, str)
                or not override_id
                or override_id in replay_ids
                or disposition not in {"reuse", "inject"}
                or type(counts) is not bool
                or (disposition == "inject" and counts)
                or (disposition == "inject" and action_digest != action.digest)
            ):
                raise ValueError("execution replay call violates the exact override contract")
            replay_ids.add(override_id)
            address = _address_from_payload(
                raw["address"],
                f"replayed_calls[{index}].address",
            )
            address_key = _address_key(address)
            if address_key in local_addresses:
                raise ValueError("generated and replayed calls reuse a scientific address")
            local_addresses.add(address_key)
            replayed.append(
                ReplayedInferenceCallReceipt(
                    override_id,
                    address,
                    action_digest,
                    disposition,
                    _exact_int(raw["prompt_tokens"], "prompt_tokens"),
                    _exact_int(raw["completion_tokens"], "completion_tokens"),
                    counts,
                )
            )
        logical_replay_tokens = sum(
            replay.completion_tokens
            for replay in replayed
            if replay.counts_toward_logical_cost
        )
        if kind is OutcomeKind.TERMINAL_WITHOUT_DOWNSTREAM and (
            calls or logical_replay_tokens
        ):
            raise ValueError("terminal-without-downstream outcome contains model calls")
        if kind is OutcomeKind.SUCCESS and not calls and logical_replay_tokens == 0:
            raise ValueError("zero-call success must use terminal_without_downstream")
        logical = _logical_cost_from_payload(value["logical_cost"])
        generated_tokens = sum(call.completion_tokens for call in calls)
        as_if_fresh_tokens = (
            action.completion_tokens + generated_tokens + logical_replay_tokens
        )
        if logical.output_tokens != as_if_fresh_tokens:
            raise ValueError(
                "logical output tokens disagree with the as-if-fresh action and "
                "downstream calls"
            )
        actual = _actual_cost_from_non_token_payload(
            value["actual_non_token_cost"],
            generated_tokens=generated_tokens,
        )
        return cls(
            receipt,
            _sha256(receipt),
            execution_id,
            arm_id,
            action.digest,
            continuation_replicate,
            kind,
            reward,
            tuple(calls),
            logical,
            actual,
            tuple(replayed),
        )

    def to_payload(self) -> dict[str, Any]:
        return {"receipt": json.loads(self.receipt), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class ReconstructionQAResult:
    receipt: bytes
    receipt_sha256: str
    passed: bool
    actual_cost: ActualEvaluationCost

    @classmethod
    def from_receipt(
        cls,
        receipt: bytes,
        *,
        verifier: ReceiptVerifier,
        commitment: PreActionTargetCommitment,
        recorded_action: BehaviorAction,
    ) -> ReconstructionQAResult:
        value = _verified_receipt(
            receipt,
            receipt_kind="reconstruction_qa",
            verifier=verifier,
        )
        _strict_keys(
            value,
            {
                "schema_version",
                "receipt_kind",
                "group_id",
                "target_id",
                "pre_action_snapshot_sha256",
                "recorded_action_digest",
                "passed",
                "report_sha256",
                "actual_cost",
            },
            "QA receipt",
        )
        if (
            value["group_id"] != commitment.group_id
            or value["target_id"] != commitment.target_id
            or value["pre_action_snapshot_sha256"]
            != commitment.pre_action_snapshot_sha256
            or value["recorded_action_digest"] != recorded_action.digest
        ):
            raise ValueError("QA receipt is bound to a different group")
        if type(value["passed"]) is not bool:
            raise ValueError("QA passed must be bool")
        _require_sha256(value["report_sha256"], "QA report")
        return cls(
            receipt,
            _sha256(receipt),
            value["passed"],
            _actual_cost_from_payload(value["actual_cost"]),
        )

    def to_payload(self) -> dict[str, Any]:
        return {"receipt": json.loads(self.receipt), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class BranchGroupSpec:
    commitment: PreActionTargetCommitment
    recorded_action: BehaviorAction
    correspondence: SeedCorrespondenceMap
    master_seed: str

    def __post_init__(self) -> None:
        if type(self.recorded_action) is not BehaviorAction:
            raise ValueError("recorded_action must be a validated BehaviorAction")
        if (
            not self.master_seed
            or _sha256(self.master_seed.encode())
            != self.commitment.master_seed_sha256
        ):
            raise ValueError("master seed disagrees with durable pre-action commitment")
        if self.recorded_action.key.sampler.seed != self.commitment.recorded_action_seed:
            raise ValueError("recorded action seed disagrees with commitment")
        if behavior_law_digest(self.recorded_action.key) != self.commitment.behavior_law_sha256:
            raise ValueError("recorded action behavior law disagrees with commitment")
        if self.correspondence.recorded_action_digest != self.recorded_action.digest:
            raise ValueError("correspondence map disagrees with recorded action")


@dataclass(frozen=True, slots=True)
class ScientificArm:
    action_slot: int
    action_source: Literal["recorded", "sampled"]
    action: BehaviorAction
    candidate: CandidateSample | None
    outcomes: tuple[BranchOutcome, ...]
    q_value: float
    advantage: float
    record_weight: Fraction

    def to_payload(self) -> dict[str, Any]:
        return {
            "action_slot": self.action_slot,
            "action_source": self.action_source,
            "action": json.loads(self.action.to_bytes()),
            "candidate": self.candidate.to_payload() if self.candidate is not None else None,
            "outcomes": [outcome.to_payload() for outcome in self.outcomes],
            "q_value": self.q_value,
            "advantage": self.advantage,
            "record_weight": _fraction_payload(self.record_weight),
            "training_intent": {
                "scope": "target_action_tokens_only",
                "prompt_tokens_weight": 0,
                "continuation_tokens_weight": 0,
                "action_token_count": len(self.action.action_token_ids),
            },
        }


@dataclass(frozen=True, slots=True)
class GroupLedger:
    actual_action_generation_calls: int
    logical_action_generation_calls: int
    actual_downstream_policy_calls: int
    logical_downstream_policy_calls: int
    actual_generated_tokens: int
    logical_output_tokens: int

    def to_payload(self) -> dict[str, int]:
        return {
            "actual_action_generation_calls": self.actual_action_generation_calls,
            "logical_action_generation_calls": self.logical_action_generation_calls,
            "actual_downstream_policy_calls": self.actual_downstream_policy_calls,
            "logical_downstream_policy_calls": self.logical_downstream_policy_calls,
            "actual_generated_tokens": self.actual_generated_tokens,
            "logical_output_tokens": self.logical_output_tokens,
        }


@dataclass(frozen=True, slots=True)
class BranchGroupArtifact:
    commitment: PreActionTargetCommitment
    correspondence: SeedCorrespondenceMap
    reconstruction_qa: ReconstructionQAResult
    recorded_action: BehaviorAction
    arms: tuple[ScientificArm, ...]
    ledger: GroupLedger
    scientific_digest: str
    training_batch_identity: str

    def scientific_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "domain": _DOMAIN,
            "commitment": self.commitment.to_payload(),
            "correspondence": self.correspondence.to_payload(),
            "recorded_action": json.loads(self.recorded_action.to_bytes()),
            "arms": [arm.to_payload() for arm in self.arms],
            "ledger": self.ledger.to_payload(),
            "inferential_arm_count": len(self.arms),
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.scientific_payload(),
            "reconstruction_qa": self.reconstruction_qa.to_payload(),
            "scientific_digest": self.scientific_digest,
            "training_batch_identity": self.training_batch_identity,
            "single_use_enforced": False,
        }

    def to_bytes(self) -> bytes:
        payload = self.to_payload()
        return canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": _DOMAIN,
                "artifact": payload,
                "digest": _sha256(canonical_json(payload)),
            }
        )

    @classmethod
    def verify_bytes(
        cls,
        value: bytes,
        *,
        verifier: ReceiptVerifier,
        encode_action: Callable[[Mapping[str, Any], Mapping[str, Any]], Sequence[int]]
        | None = None,
        validate_action: Callable[
            [Mapping[str, Any], Mapping[str, Any], Sequence[int]], None
        ]
        | None = None,
        render_prompt: Callable[[Mapping[str, Any]], tuple[int, ...]],
        master_seed: str,
    ) -> BranchGroupArtifact:
        if type(value) is not bytes:
            raise ValueError("branch-group artifact must be immutable bytes")
        envelope = json.loads(value)
        if not isinstance(envelope, dict) or canonical_json(envelope) != value:
            raise ValueError("branch-group artifact must be canonical JSON")
        _strict_keys(envelope, {"schema_version", "domain", "artifact", "digest"}, "envelope")
        if envelope["schema_version"] != SCHEMA_VERSION or envelope["domain"] != _DOMAIN:
            raise ValueError("unsupported branch-group artifact")
        artifact = envelope["artifact"]
        if not isinstance(artifact, dict) or envelope["digest"] != _sha256(
            canonical_json(artifact)
        ):
            raise ValueError("artifact digest mismatch")
        _strict_keys(
            artifact,
            {
                "schema_version",
                "domain",
                "commitment",
                "correspondence",
                "reconstruction_qa",
                "recorded_action",
                "arms",
                "ledger",
                "inferential_arm_count",
                "scientific_digest",
                "training_batch_identity",
                "single_use_enforced",
            },
            "artifact",
        )
        if artifact["single_use_enforced"] is not False:
            raise ValueError("C1 may not claim atomic training-batch consumption")
        commitment = PreActionTargetCommitment.from_receipt(
            _receipt_bytes(artifact["commitment"], "commitment"),
            verifier=verifier,
        )
        recorded_action = BehaviorAction.from_bytes(
            canonical_json(artifact["recorded_action"]),
            encode_action=encode_action,
            validate_action=validate_action,
            render_prompt=render_prompt,
        )
        correspondence = SeedCorrespondenceMap.from_receipt(
            _receipt_bytes(artifact["correspondence"], "correspondence"),
            verifier=verifier,
            commitment=commitment,
            recorded_action=recorded_action,
        )
        spec = BranchGroupSpec(commitment, recorded_action, correspondence, master_seed)
        qa = ReconstructionQAResult.from_receipt(
            _receipt_bytes(artifact["reconstruction_qa"], "reconstruction_qa"),
            verifier=verifier,
            commitment=commitment,
            recorded_action=recorded_action,
        )
        raw_arms = artifact["arms"]
        if not isinstance(raw_arms, list) or len(raw_arms) != commitment.branch_count:
            raise ValueError("artifact arm count disagrees with commitment")
        candidates: list[CandidateSample] = []
        outcomes_by_arm: list[tuple[BranchOutcome, ...]] = []
        actions = [recorded_action]
        action_scheduler = EventSeedScheduler(
            master_seed,
            commitment.rollout_id,
            commitment.target_id,
            1,
        )
        for slot, raw_arm in enumerate(raw_arms):
            if not isinstance(raw_arm, dict) or raw_arm.get("action_slot") != slot:
                raise ValueError("artifact arm slots must be ordered")
            action = BehaviorAction.from_bytes(
                canonical_json(raw_arm.get("action")),
                encode_action=encode_action,
                validate_action=validate_action,
                render_prompt=render_prompt,
            )
            if slot == 0:
                if action != recorded_action or raw_arm.get("candidate") is not None:
                    raise ValueError("slot zero must be the exact recorded action")
            else:
                candidate_payload = raw_arm.get("candidate")
                if not isinstance(candidate_payload, dict):
                    raise ValueError("sampled arm is missing its candidate receipt")
                _strict_keys(
                    candidate_payload,
                    {"action", "inference_receipt", "inference_receipt_sha256"},
                    "candidate wrapper",
                )
                if canonical_json(candidate_payload["action"]) != action.to_bytes():
                    raise ValueError("candidate wrapper action disagrees with arm action")
                candidate_receipt = candidate_payload["inference_receipt"]
                if not isinstance(candidate_receipt, dict):
                    raise ValueError("candidate inference receipt must be an object")
                candidate_receipt_bytes = canonical_json(candidate_receipt)
                if candidate_payload["inference_receipt_sha256"] != _sha256(
                    candidate_receipt_bytes
                ):
                    raise ValueError("candidate inference receipt digest mismatch")
                candidate = CandidateSample.validate(
                    CandidateSubmission(
                        action,
                        candidate_receipt_bytes,
                    ),
                    action_slot=slot,
                    action_seed=action_scheduler.action_seed(action_slot=slot),
                    commitment=commitment,
                    verifier=verifier,
                )
                candidates.append(candidate)
                actions.append(action)
            raw_outcomes = raw_arm.get("outcomes")
            if (
                not isinstance(raw_outcomes, list)
                or len(raw_outcomes) != commitment.continuation_replicates
            ):
                raise ValueError("arm replicate count disagrees with commitment")
            outcomes_by_arm.append(
                tuple(
                    BranchOutcome.from_receipt(
                        _receipt_bytes(raw_outcome, "outcome"),
                        verifier=verifier,
                        commitment=commitment,
                        correspondence=correspondence,
                        action=action,
                        arm_id=f"arm-{slot}",
                        continuation_replicate=replicate,
                        master_seed=master_seed,
                    )
                    for replicate, raw_outcome in enumerate(raw_outcomes, start=1)
                )
            )
        derived = _assemble(spec, qa, tuple(candidates), tuple(actions), tuple(outcomes_by_arm))
        if canonical_json(derived.to_payload()) != canonical_json(artifact):
            raise ValueError("artifact derived fields disagree with validated primitive receipts")
        return derived


def run_scientific_branch_group(
    spec: BranchGroupSpec,
    *,
    verifier: ReceiptVerifier,
    sample_candidate: CandidateSampler,
    run_reconstruction_qa: Callable[[BranchGroupSpec], bytes] | None,
    execute_arm: ArmExecutor,
    prepare_artifact: Callable[[BranchGroupArtifact], None] | None = None,
    reconstruction_qa_receipt: bytes | None = None,
) -> BranchGroupArtifact:
    spec = _revalidated_spec(spec, verifier=verifier)
    if (run_reconstruction_qa is None) == (reconstruction_qa_receipt is None):
        raise ValueError("supply exactly one reconstruction QA source")
    if reconstruction_qa_receipt is None:
        assert run_reconstruction_qa is not None
        qa_receipt = run_reconstruction_qa(spec)
    else:
        qa_receipt = reconstruction_qa_receipt
    qa = ReconstructionQAResult.from_receipt(
        qa_receipt,
        verifier=verifier,
        commitment=spec.commitment,
        recorded_action=spec.recorded_action,
    )
    if not qa.passed:
        raise NonRepairableCampaignAbort("reconstruction QA failed")
    action_scheduler = EventSeedScheduler(
        spec.master_seed,
        spec.commitment.rollout_id,
        spec.commitment.target_id,
        1,
    )
    candidates: list[CandidateSample] = []
    actions = [spec.recorded_action]
    for slot in range(1, spec.commitment.branch_count):
        seed = action_scheduler.action_seed(action_slot=slot)
        try:
            submission = sample_candidate(
                action_slot=slot,
                action_seed=seed,
                reference_key=spec.recorded_action.key,
            )
        except ZeroCallInfrastructureFailure as error:
            try:
                successor_permitted = _validate_zero_call_failure(
                    error.receipt,
                    verifier=verifier,
                    commitment=spec.commitment,
                    action_slot=slot,
                    action_seed=seed,
                )
            except Exception as verification_error:
                raise NonRepairableCampaignAbort(
                    "candidate failure did not prove zero scientific model calls"
                ) from verification_error
            if not successor_permitted:
                raise NonRepairableCampaignAbort(
                    "the single zero-call successor was already consumed"
                ) from error
            raise RepairableInfrastructureAbort(
                f"candidate slot {slot} failed before a scientific model call"
            ) from error
        except BaseException as error:
            raise NonRepairableCampaignAbort(
                f"candidate slot {slot} raised without a zero-call receipt"
            ) from error
        try:
            candidate = CandidateSample.validate(
                submission,
                action_slot=slot,
                action_seed=seed,
                commitment=spec.commitment,
                verifier=verifier,
            )
        except Exception as error:
            raise NonRepairableCampaignAbort(
                f"candidate slot {slot} produced an invalid scientific receipt"
            ) from error
        candidates.append(candidate)
        actions.append(candidate.action)
    outcomes_by_arm: list[tuple[BranchOutcome, ...]] = []
    for slot, action in enumerate(actions):
        arm_id = f"arm-{slot}"
        arm_outcomes: list[BranchOutcome] = []
        for replicate in range(1, spec.commitment.continuation_replicates + 1):
            scheduler = EventSeedScheduler(
                spec.master_seed,
                spec.commitment.rollout_id,
                spec.commitment.target_id,
                replicate,
            )
            oracle = BranchSeedOracle(
                scheduler,
                arm_id=arm_id,
                correspondence=spec.correspondence,
            )
            try:
                receipt = execute_arm(
                    arm_id=arm_id,
                    action=action,
                    continuation_replicate=replicate,
                    seed_oracle=oracle,
                )
            except ZeroCallInfrastructureFailure as error:
                try:
                    successor_permitted = _validate_zero_call_execution_failure(
                        error.receipt,
                        verifier=verifier,
                        commitment=spec.commitment,
                        action=action,
                        arm_id=arm_id,
                        continuation_replicate=replicate,
                    )
                except Exception as verification_error:
                    raise NonRepairableCampaignAbort(
                        "execution failure did not prove zero scientific activity"
                    ) from verification_error
                if not successor_permitted:
                    raise NonRepairableCampaignAbort(
                        "the campaign-wide zero-call successor was already consumed"
                    ) from error
                raise RepairableInfrastructureAbort(
                    f"{arm_id} replicate {replicate} failed before scientific activity"
                ) from error
            except BaseException as error:
                raise NonRepairableCampaignAbort(
                    "executor exception cannot prove the scientific failure denominator"
                ) from error
            try:
                outcome = BranchOutcome.from_receipt(
                    receipt,
                    verifier=verifier,
                    commitment=spec.commitment,
                    correspondence=spec.correspondence,
                    action=action,
                    arm_id=arm_id,
                    continuation_replicate=replicate,
                    master_seed=spec.master_seed,
                )
            except Exception as error:
                raise NonRepairableCampaignAbort(
                    "scientific arm produced an invalid execution receipt"
                ) from error
            arm_outcomes.append(outcome)
        outcomes_by_arm.append(tuple(arm_outcomes))
    artifact = _assemble(
        spec,
        qa,
        tuple(candidates),
        tuple(actions),
        tuple(outcomes_by_arm),
    )
    if prepare_artifact is not None:
        try:
            prepare_artifact(artifact)
        except BaseException as error:
            raise RepairableInfrastructureAbort(
                "artifact publish failed; durable primitive receipts remain reusable"
            ) from error
    return artifact


def _assemble(
    spec: BranchGroupSpec,
    qa: ReconstructionQAResult,
    candidates: tuple[CandidateSample, ...],
    actions: tuple[BehaviorAction, ...],
    outcomes_by_arm: tuple[tuple[BranchOutcome, ...], ...],
) -> BranchGroupArtifact:
    commitment = spec.commitment
    if not qa.passed:
        raise ValueError("reconstruction QA must pass before estimator construction")
    if len(actions) != commitment.branch_count or len(candidates) != len(actions) - 1:
        raise ValueError("scientific group is incomplete")
    execution_ids: set[str] = set()
    inference_call_ids = {candidate.inference_call_id for candidate in candidates}
    if len(inference_call_ids) != len(candidates):
        raise ValueError("candidate inference calls must be fresh")
    for outcomes in outcomes_by_arm:
        for outcome in outcomes:
            if outcome.execution_id in execution_ids:
                raise ValueError("every arm replicate requires a fresh execution")
            execution_ids.add(outcome.execution_id)
            for call in outcome.calls:
                if call.call_id in inference_call_ids:
                    raise ValueError("downstream and candidate inference call IDs collided")
                inference_call_ids.add(call.call_id)
    q_values = tuple(
        fsum(outcome.reward for outcome in outcomes) / len(outcomes)
        for outcomes in outcomes_by_arm
    )
    advantages = leave_one_out_advantages(q_values)
    record_weight = commitment.outer_weight / commitment.branch_count
    by_slot = {candidate.action_slot: candidate for candidate in candidates}
    arms = tuple(
        ScientificArm(
            slot,
            "recorded" if slot == 0 else "sampled",
            action,
            by_slot.get(slot),
            outcomes_by_arm[slot],
            q_values[slot],
            advantages[slot],
            record_weight,
        )
        for slot, action in enumerate(actions)
    )
    all_outcomes = tuple(outcome for outcomes in outcomes_by_arm for outcome in outcomes)
    ledger = GroupLedger(
        actual_action_generation_calls=len(candidates),
        logical_action_generation_calls=(
            len(actions) * commitment.continuation_replicates
        ),
        actual_downstream_policy_calls=sum(len(outcome.calls) for outcome in all_outcomes),
        logical_downstream_policy_calls=sum(
            len(outcome.calls)
            + sum(
                replay.counts_toward_logical_cost
                for replay in outcome.replayed_calls
            )
            for outcome in all_outcomes
        ),
        actual_generated_tokens=(
            sum(candidate.action.completion_tokens for candidate in candidates)
            + sum(outcome.actual_cost.generated_tokens for outcome in all_outcomes)
        ),
        logical_output_tokens=sum(
            outcome.logical_cost.output_tokens for outcome in all_outcomes
        ),
    )
    provisional = BranchGroupArtifact(
        commitment,
        spec.correspondence,
        qa,
        spec.recorded_action,
        arms,
        ledger,
        "0" * 64,
        "0" * 64,
    )
    scientific_digest = _sha256(canonical_json(provisional.scientific_payload()))
    batch_identity = _sha256(
        canonical_json(
            {
                "domain": "redco-stage-d-training-batch-identity-v1",
                "scientific_digest": scientific_digest,
                "action_digests": [action.digest for action in actions],
            }
        )
    )
    return BranchGroupArtifact(
        commitment,
        spec.correspondence,
        qa,
        spec.recorded_action,
        arms,
        ledger,
        scientific_digest,
        batch_identity,
    )


def _revalidated_spec(
    spec: BranchGroupSpec,
    *,
    verifier: ReceiptVerifier,
) -> BranchGroupSpec:
    """Discard every caller-supplied primitive field in favor of trusted receipt bytes."""
    commitment = PreActionTargetCommitment.from_receipt(
        spec.commitment.receipt,
        verifier=verifier,
    )
    if commitment != spec.commitment:
        raise ValueError("commitment object differs from its trusted receipt")
    correspondence = SeedCorrespondenceMap.from_receipt(
        spec.correspondence.receipt,
        verifier=verifier,
        commitment=commitment,
        recorded_action=spec.recorded_action,
    )
    if correspondence != spec.correspondence:
        raise ValueError("correspondence object differs from its trusted receipt")
    return BranchGroupSpec(
        commitment,
        spec.recorded_action,
        correspondence,
        spec.master_seed,
    )


def _validate_zero_call_failure(
    receipt: bytes,
    *,
    verifier: ReceiptVerifier,
    commitment: PreActionTargetCommitment,
    action_slot: int,
    action_seed: int,
) -> bool:
    value = _verified_receipt(
        receipt,
        receipt_kind="zero_call_infrastructure_failure",
        verifier=verifier,
    )
    _strict_keys(
        value,
        {
            "schema_version",
            "receipt_kind",
            "ledger_id",
            "ledger_offset",
            "prior_chain_sha256",
            "group_id",
            "target_id",
            "action_slot",
            "action_seed",
            "attempt_ordinal",
            "attempt_id",
            "attempt_model_calls",
            "attempt_overrides",
            "prior_candidate_completions",
            "prior_execution_completions",
            "repair_sequence",
            "successor_permitted",
            "reason",
        },
        "zero-call failure receipt",
    )
    attempt_id = value["attempt_id"]
    repair_sequence = _exact_int(value["repair_sequence"], "repair_sequence")
    if (
        value["ledger_id"] != commitment.ledger_id
        or _exact_int(value["ledger_offset"], "zero-call ledger_offset")
        <= commitment.ledger_offset
        or value["group_id"] != commitment.group_id
        or value["target_id"] != commitment.target_id
        or value["action_slot"] != action_slot
        or value["action_seed"] != action_seed
        or _exact_int(value["attempt_ordinal"], "attempt_ordinal") not in {0, 1}
        or value["successor_permitted"] is not (repair_sequence == 0)
        or not isinstance(attempt_id, str)
        or not attempt_id
        or _exact_int(value["attempt_model_calls"], "attempt_model_calls") != 0
        or _exact_int(value["attempt_overrides"], "attempt_overrides") != 0
        or _exact_int(
            value["prior_candidate_completions"], "prior_candidate_completions"
        )
        < 0
        or _exact_int(
            value["prior_execution_completions"], "prior_execution_completions"
        )
        < 0
        or not isinstance(value["reason"], str)
        or not value["reason"]
    ):
        raise NonRepairableCampaignAbort("infrastructure receipt does not prove zero calls")
    _require_sha256(value["prior_chain_sha256"], "zero-call prior_chain_sha256")
    return bool(value["successor_permitted"])


def _validate_zero_call_execution_failure(
    receipt: bytes,
    *,
    verifier: ReceiptVerifier,
    commitment: PreActionTargetCommitment,
    action: BehaviorAction,
    arm_id: str,
    continuation_replicate: int,
) -> bool:
    value = _verified_receipt(
        receipt,
        receipt_kind="zero_call_execution_failure",
        verifier=verifier,
    )
    _strict_keys(
        value,
        {
            "schema_version",
            "receipt_kind",
            "ledger_id",
            "ledger_offset",
            "prior_chain_sha256",
            "group_id",
            "target_id",
            "arm_id",
            "action_digest",
            "continuation_replicate",
            "attempt_ordinal",
            "attempt_id",
            "attempt_model_calls",
            "attempt_overrides",
            "prior_candidate_completions",
            "prior_execution_completions",
            "repair_sequence",
            "successor_permitted",
            "reason",
        },
        "zero-call execution failure receipt",
    )
    repair_sequence = _exact_int(value["repair_sequence"], "repair_sequence")
    if (
        value["ledger_id"] != commitment.ledger_id
        or _exact_int(value["ledger_offset"], "zero-call ledger_offset")
        <= commitment.ledger_offset
        or value["group_id"] != commitment.group_id
        or value["target_id"] != commitment.target_id
        or value["arm_id"] != arm_id
        or value["action_digest"] != action.digest
        or value["continuation_replicate"] != continuation_replicate
        or _exact_int(value["attempt_ordinal"], "attempt_ordinal") not in {0, 1}
        or _exact_int(value["attempt_model_calls"], "attempt_model_calls") != 0
        or _exact_int(value["attempt_overrides"], "attempt_overrides") != 0
        or _exact_int(
            value["prior_candidate_completions"], "prior_candidate_completions"
        )
        < 0
        or _exact_int(
            value["prior_execution_completions"], "prior_execution_completions"
        )
        < 0
        or value["successor_permitted"] is not (repair_sequence == 0)
        or not isinstance(value["attempt_id"], str)
        or not value["attempt_id"]
        or not isinstance(value["reason"], str)
        or not value["reason"]
    ):
        raise NonRepairableCampaignAbort(
            "infrastructure receipt does not prove zero execution activity"
        )
    _require_sha256(value["prior_chain_sha256"], "zero-call prior_chain_sha256")
    return bool(value["successor_permitted"])


def _receipt_bytes(value: object, name: str) -> bytes:
    if not isinstance(value, dict):
        raise ValueError(f"{name} wrapper must be an object")
    _strict_keys(value, {"receipt", "receipt_sha256"}, f"{name} wrapper")
    receipt = value["receipt"]
    if not isinstance(receipt, dict):
        raise ValueError(f"{name} receipt must be an object")
    encoded = canonical_json(receipt)
    if value["receipt_sha256"] != _sha256(encoded):
        raise ValueError(f"{name} receipt digest mismatch")
    return encoded


def _logical_cost_from_payload(value: object) -> LogicalDeploymentCost:
    if not isinstance(value, dict):
        raise ValueError("logical cost must be an object")
    _strict_keys(value, {"output_tokens", "latency_seconds", "dollars"}, "logical cost")
    return LogicalDeploymentCost(
        _exact_int(value["output_tokens"], "logical output_tokens"),
        _finite_float(value["latency_seconds"], "logical latency_seconds"),
        _finite_float(value["dollars"], "logical dollars"),
    )


def _actual_cost_from_non_token_payload(
    value: object,
    *,
    generated_tokens: int,
) -> ActualEvaluationCost:
    if not isinstance(value, dict):
        raise ValueError("actual non-token cost must be an object")
    _strict_keys(
        value,
        {"judge_calls", "cpu_seconds", "gpu_seconds", "wall_seconds", "storage_bytes"},
        "actual non-token cost",
    )
    return ActualEvaluationCost(
        generated_tokens,
        _exact_int(value["judge_calls"], "judge_calls"),
        _finite_float(value["cpu_seconds"], "cpu_seconds"),
        _finite_float(value["gpu_seconds"], "gpu_seconds"),
        _finite_float(value["wall_seconds"], "wall_seconds"),
        _exact_int(value["storage_bytes"], "storage_bytes"),
    )


def _actual_cost_from_payload(value: object) -> ActualEvaluationCost:
    if not isinstance(value, dict):
        raise ValueError("actual cost must be an object")
    _strict_keys(
        value,
        {
            "generated_tokens",
            "judge_calls",
            "cpu_seconds",
            "gpu_seconds",
            "wall_seconds",
            "storage_bytes",
        },
        "actual cost",
    )
    return ActualEvaluationCost(
        _exact_int(value["generated_tokens"], "generated_tokens"),
        _exact_int(value["judge_calls"], "judge_calls"),
        _finite_float(value["cpu_seconds"], "cpu_seconds"),
        _finite_float(value["gpu_seconds"], "gpu_seconds"),
        _finite_float(value["wall_seconds"], "wall_seconds"),
        _exact_int(value["storage_bytes"], "storage_bytes"),
    )
