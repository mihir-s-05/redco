"""Pure read-only state-machine validation for Stage-D receipt ledgers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from redco.analysis.stage_d_ledger_contracts import LedgerPoisoned
from redco.analysis.stage_d_spawn_provenance import (
    CouplingMode,
    PolicyEventAddress,
    ScheduledSeed,
)
from redco.contracts import canonical_json


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _body(record: Mapping[str, Any]) -> dict[str, Any]:
    body = record.get("body")
    if not isinstance(body, dict):
        raise LedgerPoisoned("ledger record body must be an object")
    return body


def _event(body: Mapping[str, Any]) -> dict[str, Any]:
    event = body.get("event")
    if not isinstance(event, dict):
        raise LedgerPoisoned("event record lacks an event object")
    return event


def _group_key(group_id: str, target_id: str) -> tuple[str, str]:
    if not group_id or not target_id:
        raise ValueError("group_id and target_id must be nonempty")
    return group_id, target_id


def _address_payload(address: PolicyEventAddress) -> dict[str, str | int]:
    return {**address.as_payload(), "turn": address.turn}


def _scientific_address_key(address: PolicyEventAddress) -> bytes:
    return canonical_json(address.as_payload())


def _scanned_policy_address(value: object) -> PolicyEventAddress:
    if not isinstance(value, dict) or set(value) != {
        "depth",
        "lineage",
        "session_call_ordinal",
        "turn",
        "call_kind",
    }:
        raise LedgerPoisoned("ledger event contains an invalid policy address")
    try:
        address = PolicyEventAddress(
            cast(int, value["depth"]),
            cast(str, value["lineage"]),
            cast(int, value["session_call_ordinal"]),
            cast(int, value["turn"]),
            cast(str, value["call_kind"]),
        )
    except (TypeError, ValueError) as error:
        raise LedgerPoisoned("ledger event contains an invalid policy address") from error
    if _address_payload(address) != value:
        raise LedgerPoisoned("ledger event policy address changed types")
    return address


def _scanned_address_key(value: object) -> bytes:
    return _scientific_address_key(_scanned_policy_address(value))


def _scanned_scheduled_cache_salt(event: Mapping[str, Any]) -> str:
    seed = event.get("seed")
    coupling = event.get("coupling_mode")
    if type(seed) is not int or seed < 0 or coupling not in {"paired", "exogenous"}:
        raise LedgerPoisoned("execution model call has an invalid scheduled seed")
    address = _scanned_policy_address(event.get("address"))
    return ScheduledSeed(seed, CouplingMode(coupling), address).cache_salt


def validate_state_machine(
    records: Sequence[Mapping[str, Any]],
    *,
    allow_source_inflight: bool = False,
    allow_repairable_zero_call: bool = False,
) -> Mapping[str, Any] | None:
    if records[0]["record_kind"] != "genesis" or records[0]["offset"] != 0:
        raise LedgerPoisoned("first record must be genesis")
    genesis = _body(records[0])
    genesis_fields = {
        "preregistration_sha256",
        "source_sha256",
        "runtime_sha256",
        "config_sha256",
        "protocol_manifest_sha256",
        "master_seed_sha256",
        "support_rules_sha256",
    }
    if set(genesis) != genesis_fields or any(
        not _is_sha256(genesis[name]) for name in genesis_fields
    ):
        raise LedgerPoisoned("genesis binding fields or hashes are invalid")
    if any(record["record_kind"] == "genesis" for record in records[1:]):
        raise LedgerPoisoned("ledger contains multiple genesis records")
    if any(record["record_kind"] == "seal" for record in records[:-1]):
        raise LedgerPoisoned("seal must be the terminal record")
    commitments: dict[tuple[str, str], tuple[int, str, Mapping[str, Any]]] = {}
    reservations: dict[tuple[str, str], dict[str, Any]] = {}
    recorded_action_materialized: set[str] = set()
    correspondence: dict[tuple[str, str], str] = {}
    reconstruction_qa: dict[tuple[str, str], tuple[bool, str]] = {}
    reconstruction_qa_barrier_sha256: str | None = None
    candidate_attempts: dict[str, dict[str, Any]] = {}
    candidate_attempt_counts: dict[tuple[str, str, int], int] = {}
    candidate_zero_call_failures: dict[tuple[str, str, int], int] = {}
    zero_call_failure_count = 0
    execution_attempts: dict[str, dict[str, Any]] = {}
    execution_attempt_counts: dict[tuple[str, str, str, int], int] = {}
    execution_zero_call_failures: dict[tuple[str, str, str, int], int] = {}
    bound_execution_contexts: set[str] = set()
    dispatched_executions: set[str] = set()
    override_commits: dict[str, dict[str, Any]] = {}
    override_keys: set[tuple[str, bytes]] = set()
    execution_call_keys: set[tuple[str, bytes]] = set()
    delivered_overrides: set[str] = set()
    discarded_overrides: set[str] = set()
    starts: dict[str, dict[str, Any]] = {}
    completed: set[str] = set()
    completion_evidence_refs: dict[str, set[str]] = {}
    observed_responses: dict[str, str] = {}
    finished_candidates: set[str] = set()
    finished_executions: set[str] = set()
    candidate_slots: set[tuple[str, str, int]] = set()
    execution_slots: set[tuple[str, str, str, int]] = set()
    branch_artifacts: dict[tuple[str, str], str] = {}
    verified_support_report_sha256: str | None = None
    batch_claims: set[str] = set()
    stage_d_training_authorization_arms: set[str] = set()
    source_reservations: dict[tuple[str, str], tuple[int, str, Mapping[str, Any]]] = {}
    source_responses: dict[tuple[str, str], tuple[int, str, Mapping[str, Any]]] = {}
    source_completions: dict[tuple[str, str], str] = {}
    source_action_digests: dict[tuple[str, str], str] = {}
    source_aborts: dict[tuple[str, str], str] = {}
    source_pre_post_aborts: set[tuple[str, str]] = set()
    source_finalization_aborts: set[tuple[str, str]] = set()
    source_rollouts: dict[tuple[str, str], set[tuple[str, str]]] = {}
    source_rollout_sha256s: dict[tuple[str, str], str] = {}
    branch_target_roster_sha256: str | None = None
    branch_target_keys: set[tuple[str, str]] = set()
    for record in records[1:]:
        kind = record["record_kind"]
        body = _body(record)
        if kind == "receipt":
            receipt = body["receipt"]
            receipt_kind = receipt["receipt_kind"]
            if receipt_kind == "pre_action_group_commitment":
                key = _group_key(receipt["group_id"], receipt["target_id"])
                if key in commitments:
                    raise LedgerPoisoned("duplicate pre-action commitment")
                if (
                    receipt["commitment_sequence"] != record["offset"]
                    or receipt["action_reservation_sequence"] != record["offset"] + 1
                ):
                    raise LedgerPoisoned("commitment/reservation ordering proof is invalid")
                commitments[key] = (
                    record["offset"],
                    body["receipt_sha256"],
                    receipt,
                )
            elif receipt_kind == "source_policy_call_reserved":
                key = (receipt.get("rollout_id"), receipt.get("decision_id"))
                if (
                    not all(isinstance(value, str) and value for value in key)
                    or key in source_reservations
                    or receipt.get("request_sequence") != record["offset"]
                    or not _is_sha256(receipt.get("exact_action_key_digest"))
                    or not _is_sha256(receipt.get("request_sha256"))
                    or receipt.get("node_kind") not in {"root", "child"}
                    or type(receipt.get("branch_selected")) is not bool
                    or type(receipt.get("raw_response_required", False)) is not bool
                ):
                    raise LedgerPoisoned("source policy reservation is invalid")
                node_kind = receipt["node_kind"]
                target_id = receipt.get("target_id")
                target_ordinal = receipt.get("target_ordinal")
                if (
                    node_kind == "root" and (target_id is not None or target_ordinal is not None)
                ) or (
                    node_kind == "child"
                    and (
                        not isinstance(target_id, str)
                        or not target_id
                        or type(target_ordinal) is not int
                        or target_ordinal < 0
                    )
                ):
                    raise LedgerPoisoned("source policy target fields are invalid")
                commitment_hash = receipt.get("target_commitment_receipt_sha256")
                recorded_reservation_id = receipt.get("recorded_action_reservation_id")
                if receipt["branch_selected"]:
                    commitment = commitments.get((receipt.get("group_id"), target_id))
                    recorded_reservation = reservations.get((receipt.get("group_id"), target_id))
                    if (
                        commitment is None
                        or commitment[0] >= record["offset"]
                        or commitment[1] != commitment_hash
                        or commitment[2].get("rollout_id") != receipt["rollout_id"]
                        or commitment[2].get("target_ordinal") != target_ordinal
                        or commitment[2].get("target_address") != receipt.get("target_address")
                        or recorded_reservation is None
                        or recorded_reservation.get("reservation_id") != recorded_reservation_id
                        or recorded_reservation.get("exact_action_key_digest")
                        != receipt.get("exact_action_key_digest")
                        or recorded_reservation.get("request_sha256")
                        != receipt.get("request_sha256")
                    ):
                        raise LedgerPoisoned(
                            "selected source policy call lacks same-ledger pre-action proof"
                        )
                elif commitment_hash is not None or recorded_reservation_id is not None:
                    raise LedgerPoisoned(
                        "unselected source policy call names branch reservation evidence"
                    )
                source_reservations[key] = (
                    record["offset"],
                    body["receipt_sha256"],
                    receipt,
                )
            elif receipt_kind == "source_policy_response_observed":
                key = (receipt.get("rollout_id"), receipt.get("decision_id"))
                source_reservation = source_reservations.get(key)
                expected_fields = {
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
                    "raw_response_sha256",
                    "request_sequence",
                }
                if (
                    set(receipt) != expected_fields
                    or source_reservation is None
                    or key in source_responses
                    or key in source_completions
                    or key in source_aborts
                    or receipt.get("ledger_offset") != record["offset"]
                    or receipt.get("request_sequence") != source_reservation[0]
                    or source_reservation[0] >= record["offset"]
                    or receipt.get("request_receipt_sha256") != source_reservation[1]
                    or receipt.get("group_id") != source_reservation[2].get("group_id")
                    or receipt.get("exact_action_key_digest")
                    != source_reservation[2].get("exact_action_key_digest")
                    or not _is_sha256(receipt.get("raw_response_sha256"))
                    or set(body["evidence_refs"])
                    != {receipt.get("raw_response_sha256")}
                ):
                    raise LedgerPoisoned("source policy raw response witness is invalid")
                source_responses[key] = (
                    record["offset"],
                    body["receipt_sha256"],
                    receipt,
                )
            elif receipt_kind == "source_policy_call_completed":
                key = (receipt.get("rollout_id"), receipt.get("decision_id"))
                source_reservation = source_reservations.get(key)
                source_response = source_responses.get(key)
                raw_response_sha256 = receipt.get("raw_response_sha256")
                legacy_fields = {
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
                witnessed_fields = legacy_fields | {"raw_response_sha256"}
                if (
                    set(receipt) not in (legacy_fields, witnessed_fields)
                    or source_reservation is None
                    or key in source_completions
                    or key in source_aborts
                    or receipt.get("completion_sequence") != record["offset"]
                    or receipt.get("request_sequence") != source_reservation[0]
                    or receipt.get("request_receipt_sha256") != source_reservation[1]
                    or receipt.get("exact_action_key_digest")
                    != source_reservation[2].get("exact_action_key_digest")
                    or not _is_sha256(receipt.get("action_digest"))
                    or not _is_sha256(receipt.get("response_sha256"))
                    or (
                        source_reservation[2].get("raw_response_required", False)
                        and source_response is None
                    )
                    or (
                        source_response is not None
                        and (
                            source_response[0] >= record["offset"]
                            or raw_response_sha256
                            != source_response[2].get("raw_response_sha256")
                        )
                    )
                    or (source_response is None and raw_response_sha256 is not None)
                    or set(body["evidence_refs"])
                    != {
                        receipt.get("response_sha256"),
                        *(
                            [raw_response_sha256]
                            if raw_response_sha256 is not None
                            else []
                        ),
                    }
                ):
                    raise LedgerPoisoned("source policy completion is invalid")
                source_completions[key] = body["receipt_sha256"]
                source_action_digests[key] = receipt["action_digest"]
            elif receipt_kind == "source_policy_call_aborted":
                key = (receipt.get("rollout_id"), receipt.get("decision_id"))
                source_reservation = source_reservations.get(key)
                expected_fields = {
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
                    "error_sha256",
                    "phase",
                    "request_sequence",
                    "abort_sequence",
                }
                if (
                    set(receipt) != expected_fields
                    or source_reservation is None
                    or key in source_completions
                    or key in source_aborts
                    or receipt.get("abort_sequence") != record["offset"]
                    or receipt.get("request_sequence") != source_reservation[0]
                    or receipt.get("request_receipt_sha256") != source_reservation[1]
                    or receipt.get("exact_action_key_digest")
                    != source_reservation[2].get("exact_action_key_digest")
                    or not _is_sha256(receipt.get("error_sha256"))
                    or receipt.get("phase")
                    not in {
                        "post_unknown",
                        "response_received",
                        "response_parsed",
                        "typed_response",
                    }
                ):
                    raise LedgerPoisoned("source policy abort is invalid")
                source_aborts[key] = body["receipt_sha256"]
            elif receipt_kind == "source_child_pre_post_aborted":
                expected_fields = {
                    "schema_version",
                    "receipt_kind",
                    "ledger_id",
                    "ledger_offset",
                    "prior_chain_sha256",
                    "group_id",
                    "rollout_id",
                    "target_id",
                    "reservation_id",
                    "exact_action_key_digest",
                    "error_sha256",
                    "phase",
                    "abort_sequence",
                }
                key = _group_key(receipt.get("group_id"), receipt.get("target_id"))
                reservation = reservations.get(key)
                if (
                    set(receipt) != expected_fields
                    or reservation is None
                    or key in source_pre_post_aborts
                    or receipt.get("reservation_id") != reservation["reservation_id"]
                    or receipt.get("exact_action_key_digest")
                    != reservation["exact_action_key_digest"]
                    or not isinstance(receipt.get("rollout_id"), str)
                    or not receipt["rollout_id"]
                    or not _is_sha256(receipt.get("error_sha256"))
                    or receipt.get("phase") != "before_post"
                    or receipt.get("abort_sequence") != record["offset"]
                    or receipt.get("ledger_offset") != record["offset"]
                ):
                    raise LedgerPoisoned("source child pre-POST abort is invalid")
                source_pre_post_aborts.add(key)
            elif receipt_kind == "source_rollout_finalization_aborted":
                expected_fields = {
                    "schema_version",
                    "receipt_kind",
                    "ledger_id",
                    "ledger_offset",
                    "prior_chain_sha256",
                    "group_id",
                    "rollout_id",
                    "decision_ids",
                    "error_sha256",
                    "phase",
                    "abort_sequence",
                }
                group_id = receipt.get("group_id")
                rollout_id = receipt.get("rollout_id")
                decision_ids = receipt.get("decision_ids")
                key = (group_id, rollout_id)
                completed_for_rollout = {
                    decision_id
                    for completed_rollout, decision_id in source_completions
                    if completed_rollout == rollout_id
                }
                if (
                    set(receipt) != expected_fields
                    or not isinstance(group_id, str)
                    or not group_id
                    or not isinstance(rollout_id, str)
                    or not rollout_id
                    or key in source_finalization_aborts
                    or key in source_rollouts
                    or not isinstance(decision_ids, list)
                    or decision_ids != sorted(completed_for_rollout)
                    or any(
                        source_reservations[(rollout_id, decision_id)][2].get("group_id")
                        != group_id
                        for decision_id in completed_for_rollout
                    )
                    or not _is_sha256(receipt.get("error_sha256"))
                    or receipt.get("phase") != "source_finalization"
                    or receipt.get("abort_sequence") != record["offset"]
                ):
                    raise LedgerPoisoned("source rollout finalization abort is invalid")
                source_finalization_aborts.add(key)
            elif receipt_kind == "source_rollout_completed":
                group_id = receipt.get("group_id")
                rollout_id = receipt.get("rollout_id")
                decision_ids = receipt.get("decision_ids")
                completion_hashes = receipt.get("decision_completion_receipt_sha256s")
                key = (group_id, rollout_id)
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
                if (
                    set(receipt) != expected_fields
                    or not isinstance(group_id, str)
                    or not group_id
                    or not isinstance(rollout_id, str)
                    or not rollout_id
                    or key in source_rollouts
                    or receipt.get("completion_sequence") != record["offset"]
                    or not all(
                        _is_sha256(receipt.get(field))
                        for field in (
                            "source_sha256",
                            "trace_sha256",
                            "reward_evidence_sha256",
                            "stock_sequences_evidence_sha256",
                            "base_model_manifest_sha256",
                        )
                    )
                    or not isinstance(decision_ids, list)
                    or not decision_ids
                    or len(set(decision_ids)) != len(decision_ids)
                    or not all(isinstance(item, str) and item for item in decision_ids)
                    or not isinstance(completion_hashes, list)
                    or len(completion_hashes) != len(decision_ids)
                    or not all(_is_sha256(item) for item in completion_hashes)
                    or set(body["evidence_refs"])
                    != {
                        receipt.get("trace_sha256"),
                        receipt.get("reward_evidence_sha256"),
                        receipt.get("stock_sequences_evidence_sha256"),
                    }
                ):
                    raise LedgerPoisoned("source rollout completion is invalid")
                named = {
                    (rollout_id, decision_id): digest
                    for decision_id, digest in zip(
                        decision_ids,
                        completion_hashes,
                        strict=True,
                    )
                }
                completion_roster = {
                    completion_key: digest
                    for completion_key, digest in source_completions.items()
                    if completion_key[0] == rollout_id
                }
                if named != completion_roster:
                    raise LedgerPoisoned(
                        "source rollout completion roster differs from policy calls"
                    )
                request_order = [
                    source_reservations[(rollout_id, decision_id)][0]
                    for decision_id in decision_ids
                ]
                if request_order != sorted(request_order) or any(
                    source_reservations[(rollout_id, decision_id)][2].get("group_id") != group_id
                    for decision_id in decision_ids
                ):
                    raise LedgerPoisoned("source rollout completion changed group or request order")
                source_rollouts[key] = set(named)
                source_rollout_sha256s[key] = receipt["source_sha256"]
            elif receipt_kind == "branch_target_roster":
                expected_fields = {
                    "schema_version",
                    "receipt_kind",
                    "ledger_id",
                    "ledger_offset",
                    "prior_chain_sha256",
                    "roster_sha256",
                    "planned_source_count",
                    "completed_source_count",
                    "eligible_source_count",
                    "ineligible_source_count",
                    "minimum_eligible_sources",
                    "eligibility_passed",
                    "source_sha256s",
                    "targets",
                    "roster_sequence",
                }
                planned = receipt.get("planned_source_count")
                completed_count = receipt.get("completed_source_count")
                eligible = receipt.get("eligible_source_count")
                ineligible = receipt.get("ineligible_source_count")
                minimum = receipt.get("minimum_eligible_sources")
                passed = receipt.get("eligibility_passed")
                sources = receipt.get("source_sha256s")
                targets = receipt.get("targets")
                if (
                    set(receipt) != expected_fields
                    or branch_target_roster_sha256 is not None
                    or receipt.get("ledger_offset") != record["offset"]
                    or receipt.get("roster_sequence") != record["offset"]
                    or not _is_sha256(receipt.get("roster_sha256"))
                    or set(body["evidence_refs"]) != {receipt.get("roster_sha256")}
                    or type(planned) is not int
                    or planned < 1
                    or type(completed_count) is not int
                    or completed_count != planned
                    or completed_count != len(source_rollouts)
                    or type(eligible) is not int
                    or eligible < 0
                    or type(ineligible) is not int
                    or ineligible != completed_count - eligible
                    or type(minimum) is not int
                    or minimum < 1
                    or minimum > planned
                    or type(passed) is not bool
                    or passed is not (eligible >= minimum)
                    or not isinstance(sources, list)
                    or sources != sorted(set(sources))
                    or set(sources) != set(source_rollout_sha256s.values())
                    or not isinstance(targets, list)
                    or candidate_attempts
                    or execution_attempts
                ):
                    raise LedgerPoisoned("branch target roster is invalid or late")
                target_keys: set[tuple[str, str]] = set()
                target_source_hashes: set[str] = set()
                for target in targets:
                    expected_target_fields = {
                        "source_sha256",
                        "group_id",
                        "rollout_id",
                        "decision_id",
                        "target_id",
                        "target_ordinal",
                        "event_address",
                    }
                    if not isinstance(target, dict) or set(target) != expected_target_fields:
                        raise LedgerPoisoned("branch target roster entry fields differ")
                    group_id = target.get("group_id")
                    rollout_id = target.get("rollout_id")
                    decision_id = target.get("decision_id")
                    target_id = target.get("target_id")
                    if not all(
                        isinstance(item, str) and item
                        for item in (group_id, rollout_id, decision_id, target_id)
                    ):
                        raise LedgerPoisoned("branch target roster identifiers are invalid")
                    assert isinstance(group_id, str)
                    assert isinstance(rollout_id, str)
                    assert isinstance(decision_id, str)
                    assert isinstance(target_id, str)
                    key = (group_id, target_id)
                    source_key = (group_id, rollout_id)
                    reservation_key = (rollout_id, decision_id)
                    source_reservation = source_reservations.get(reservation_key)
                    commitment = commitments.get(key)
                    if (
                        key in target_keys
                        or not _is_sha256(target.get("source_sha256"))
                        or source_rollout_sha256s.get(source_key) != target["source_sha256"]
                        or source_reservation is None
                        or source_reservation[2].get("group_id") != target["group_id"]
                        or source_reservation[2].get("node_kind") != "child"
                        or source_reservation[2].get("branch_selected") is not True
                        or source_reservation[2].get("target_id") != target["target_id"]
                        or source_reservation[2].get("target_ordinal")
                        != target["target_ordinal"]
                        or source_reservation[2].get("target_address")
                        != target["event_address"]
                        or commitment is None
                        or commitment[2].get("rollout_id") != target["rollout_id"]
                        or commitment[2].get("target_ordinal") != target["target_ordinal"]
                        or commitment[2].get("target_address") != target["event_address"]
                    ):
                        raise LedgerPoisoned("branch target roster lacks exact source provenance")
                    target_keys.add(key)
                    target_source_hashes.add(target["source_sha256"])
                if target_keys != set(commitments) or len(target_source_hashes) != eligible:
                    raise LedgerPoisoned("branch target roster differs from committed denominator")
                branch_target_roster_sha256 = receipt["roster_sha256"]
                branch_target_keys = target_keys
            elif receipt_kind == "seed_correspondence_map":
                key = _group_key(receipt["group_id"], receipt["target_id"])
                if (
                    key not in reservations
                    or key in correspondence
                ):
                    raise LedgerPoisoned("correspondence is missing its unique reservation")
                if reservations[key]["reservation_id"] not in recorded_action_materialized:
                    raise LedgerPoisoned("correspondence predates the recorded action output")
                if any(
                    (attempt["group_id"], attempt["target_id"]) == key
                    for attempt in (*candidate_attempts.values(), *execution_attempts.values())
                ):
                    raise LedgerPoisoned("correspondence was frozen after scientific activity")
                correspondence[key] = receipt["recorded_action_digest"]
            elif receipt_kind == "reconstruction_qa":
                expected_fields = {
                    "schema_version",
                    "receipt_kind",
                    "group_id",
                    "target_id",
                    "pre_action_snapshot_sha256",
                    "recorded_action_digest",
                    "passed",
                    "report_sha256",
                    "actual_cost",
                }
                key = _group_key(receipt.get("group_id"), receipt.get("target_id"))
                actual_cost = receipt.get("actual_cost")
                expected_cost_fields = {
                    "generated_tokens",
                    "judge_calls",
                    "cpu_seconds",
                    "gpu_seconds",
                    "wall_seconds",
                    "storage_bytes",
                }
                if (
                    set(receipt) != expected_fields
                    or key in reconstruction_qa
                    or key not in correspondence
                    or (branch_target_roster_sha256 is not None and key not in branch_target_keys)
                    or receipt.get("recorded_action_digest") != correspondence.get(key)
                    or receipt.get("pre_action_snapshot_sha256")
                    != commitments.get(key, ({}, {}, {}))[2].get(
                        "pre_action_snapshot_sha256"
                    )
                    or type(receipt.get("passed")) is not bool
                    or not _is_sha256(receipt.get("report_sha256"))
                    or set(body["evidence_refs"]) != {receipt.get("report_sha256")}
                    or not isinstance(actual_cost, dict)
                    or set(actual_cost) != expected_cost_fields
                    or any(
                        type(actual_cost[field]) not in {int, float}
                        or actual_cost[field] < 0
                        for field in expected_cost_fields
                    )
                    or type(actual_cost["generated_tokens"]) is not int
                    or type(actual_cost["judge_calls"]) is not int
                    or type(actual_cost["storage_bytes"]) is not int
                    or (
                        branch_target_roster_sha256 is not None
                        and (
                            actual_cost["generated_tokens"] != 0
                            or actual_cost["judge_calls"] != 0
                        )
                    )
                    or candidate_attempts
                    or execution_attempts
                    or reconstruction_qa_barrier_sha256 is not None
                ):
                    raise LedgerPoisoned("reconstruction QA is invalid or late")
                reconstruction_qa[key] = (
                    receipt["passed"],
                    body["receipt_sha256"],
                )
            elif receipt_kind == "reconstruction_qa_barrier":
                expected_fields = {
                    "schema_version",
                    "receipt_kind",
                    "ledger_id",
                    "ledger_offset",
                    "prior_chain_sha256",
                    "branch_target_roster_sha256",
                    "qa_receipts",
                    "target_count",
                    "all_passed",
                    "scientific_model_calls_before_barrier",
                    "barrier_sequence",
                }
                qa_receipts = receipt.get("qa_receipts")
                expected_qa_receipts = [
                    {
                        "group_id": group_id,
                        "target_id": target_id,
                        "qa_receipt_sha256": reconstruction_qa[(group_id, target_id)][1],
                    }
                    for group_id, target_id in sorted(branch_target_keys)
                    if (group_id, target_id) in reconstruction_qa
                ]
                if (
                    set(receipt) != expected_fields
                    or reconstruction_qa_barrier_sha256 is not None
                    or branch_target_roster_sha256 is None
                    or receipt.get("branch_target_roster_sha256")
                    != branch_target_roster_sha256
                    or set(reconstruction_qa) != branch_target_keys
                    or not all(passed for passed, _ in reconstruction_qa.values())
                    or qa_receipts != expected_qa_receipts
                    or receipt.get("target_count") != len(branch_target_keys)
                    or receipt.get("all_passed") is not True
                    or receipt.get("scientific_model_calls_before_barrier") != 0
                    or receipt.get("ledger_offset") != record["offset"]
                    or receipt.get("barrier_sequence") != record["offset"]
                    or body["evidence_refs"]
                    or candidate_attempts
                    or execution_attempts
                ):
                    raise LedgerPoisoned("reconstruction QA barrier is invalid or late")
                reconstruction_qa_barrier_sha256 = body["receipt_sha256"]
            elif receipt_kind == "candidate_action_inference":
                expected_fields = {
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
                }
                slot = (receipt["group_id"], receipt["target_id"], receipt["action_slot"])
                matching = [
                    attempt_id
                    for attempt_id, attempt in candidate_attempts.items()
                    if (
                        attempt["group_id"],
                        attempt["target_id"],
                        attempt["action_slot"],
                    )
                    == slot
                ]
                if len(matching) != 1 or matching[0] in finished_candidates:
                    raise LedgerPoisoned("candidate receipt lacks one unique attempt")
                call_id = receipt["inference_call_id"]
                if (
                    set(receipt) != expected_fields
                    or not _is_sha256(receipt.get("action_evidence_sha256"))
                    or not _is_sha256(receipt.get("action_digest"))
                    or receipt.get("selection_policy") != "direct_single_sample"
                    or receipt.get("sample_attempts") != 1
                    or receipt.get("rejected_attempts") != 0
                    or call_id in completed
                    or call_id not in observed_responses
                    or receipt.get("response_sha256") != observed_responses[call_id]
                    or type(receipt.get("prompt_tokens")) is not int
                    or receipt.get("prompt_tokens", -1) < 0
                    or type(receipt.get("completion_tokens")) is not int
                    or receipt.get("completion_tokens", -1) < 0
                    or starts.get(call_id, {}).get("attempt_id") != matching[0]
                    or set(body["evidence_refs"])
                    != {
                        receipt.get("action_evidence_sha256"),
                        receipt.get("response_sha256"),
                    }
                ):
                    raise LedgerPoisoned("candidate receipt lacks one completed model call")
                completed.add(call_id)
                completion_evidence_refs[call_id] = {receipt["response_sha256"]}
                finished_candidates.add(matching[0])
                candidate_slots.add(slot)
            elif receipt_kind == "zero_call_infrastructure_failure":
                attempt_id = receipt["attempt_id"]
                attempt = candidate_attempts.get(attempt_id)
                slot = (receipt["group_id"], receipt["target_id"], receipt["action_slot"])
                ordinal = receipt.get("attempt_ordinal")
                if (
                    attempt is None
                    or attempt_id in finished_candidates
                    or ordinal != attempt.get("attempt_ordinal")
                    or type(ordinal) is not int
                    or ordinal not in {0, 1}
                    or receipt.get("attempt_model_calls") != 0
                    or receipt.get("attempt_overrides") != 0
                    or receipt.get("prior_candidate_completions")
                    != len(candidate_slots)
                    or receipt.get("prior_execution_completions")
                    != len(execution_slots)
                    or receipt.get("repair_sequence") != zero_call_failure_count
                    or receipt.get("successor_permitted")
                    is not (zero_call_failure_count == 0)
                ):
                    raise LedgerPoisoned("zero-call receipt lacks one candidate attempt")
                if any(start["attempt_id"] == attempt_id for start in starts.values()):
                    raise LedgerPoisoned("zero-call receipt follows model_call_started")
                finished_candidates.add(attempt_id)
                candidate_zero_call_failures[slot] = (
                    candidate_zero_call_failures.get(slot, 0) + 1
                )
                zero_call_failure_count += 1
            elif receipt_kind == "scientific_arm_execution":
                execution_key = (
                    receipt["group_id"],
                    receipt["target_id"],
                    receipt["arm_id"],
                    receipt["continuation_replicate"],
                )
                matching = [
                    attempt_id
                    for attempt_id, attempt in execution_attempts.items()
                    if (
                        attempt["group_id"],
                        attempt["target_id"],
                        attempt["arm_id"],
                        attempt["continuation_replicate"],
                    )
                    == execution_key
                    and attempt_id not in finished_executions
                ]
                if len(matching) != 1 or matching[0] in finished_executions:
                    raise LedgerPoisoned("execution receipt lacks one unique attempt")
                if matching[0] not in bound_execution_contexts:
                    raise LedgerPoisoned("execution receipt lacks a frozen context binding")
                if matching[0] not in dispatched_executions:
                    raise LedgerPoisoned("execution receipt predates action dispatch")
                expected_calls = {
                    call_id
                    for call_id, start in starts.items()
                    if start["attempt_id"] == matching[0]
                }
                receipt_calls = {call["call_id"] for call in receipt["calls"]}
                if expected_calls != receipt_calls or not expected_calls <= completed:
                    raise LedgerPoisoned("execution receipt does not cover completed calls exactly")
                execution_overrides = {
                    override_id
                    for override_id, override in override_commits.items()
                    if override["attempt_id"] == matching[0]
                }
                if not execution_overrides <= delivered_overrides:
                    raise LedgerPoisoned("execution receipt predates override delivery")
                replayed_calls = receipt.get("replayed_calls")
                if not isinstance(replayed_calls, list):
                    raise LedgerPoisoned("execution receipt lacks its replay ledger")
                replayed_by_id = {
                    replay.get("override_id"): replay
                    for replay in replayed_calls
                    if isinstance(replay, dict)
                }
                if set(replayed_by_id) != execution_overrides or len(replayed_by_id) != len(
                    replayed_calls
                ):
                    raise LedgerPoisoned("execution replay ledger changed its denominator")
                for receipt_override_id in execution_overrides:
                    receipt_committed = override_commits[receipt_override_id]
                    replayed = replayed_by_id[receipt_override_id]
                    expected = {
                        "override_id": receipt_override_id,
                        "address": receipt_committed["address"],
                        "action_digest": receipt_committed["action_digest"],
                        "disposition": receipt_committed["disposition"],
                        "prompt_tokens": receipt_committed["prompt_tokens"],
                        "completion_tokens": receipt_committed["completion_tokens"],
                        "counts_toward_logical_cost": receipt_committed[
                            "counts_toward_logical_cost"
                        ],
                    }
                    if replayed != expected:
                        raise LedgerPoisoned(
                            "execution replay ledger differs from its commit"
                        )
                if branch_target_roster_sha256 is not None:
                    commitment = commitments.get(
                        (receipt["group_id"], receipt["target_id"])
                    )
                    injected = [
                        override
                        for override in override_commits.values()
                        if override["attempt_id"] == matching[0]
                        and override["disposition"] == "inject"
                    ]
                    attempt = execution_attempts[matching[0]]
                    if (
                        commitment is None
                        or len(injected) != 1
                        or injected[0].get("address")
                        != commitment[2].get("target_address")
                        or injected[0].get("action_digest")
                        != attempt.get("action_digest")
                    ):
                        raise LedgerPoisoned(
                            "roster-backed execution lacks exactly one target injection"
                        )
                finished_executions.add(matching[0])
                execution_slots.add(execution_key)
            elif receipt_kind == "zero_call_execution_failure":
                attempt_id = receipt.get("attempt_id")
                attempt = (
                    execution_attempts.get(attempt_id)
                    if isinstance(attempt_id, str)
                    else None
                )
                execution_key = (
                    receipt.get("group_id"),
                    receipt.get("target_id"),
                    receipt.get("arm_id"),
                    receipt.get("continuation_replicate"),
                )
                ordinal = receipt.get("attempt_ordinal")
                attempt_override_ids = sorted(
                    override_id
                    for override_id, override in override_commits.items()
                    if override["attempt_id"] == attempt_id
                )
                if (
                    attempt is None
                    or attempt_id in finished_executions
                    or execution_key
                    != (
                        attempt.get("group_id"),
                        attempt.get("target_id"),
                        attempt.get("arm_id"),
                        attempt.get("continuation_replicate"),
                    )
                    or ordinal != attempt.get("attempt_ordinal")
                    or type(ordinal) is not int
                    or ordinal not in {0, 1}
                    or receipt.get("attempt_model_calls") != 0
                    or receipt.get("attempt_overrides") != len(attempt_override_ids)
                    or receipt.get("discarded_override_ids") != attempt_override_ids
                    or receipt.get("prior_candidate_completions")
                    != len(candidate_slots)
                    or receipt.get("prior_execution_completions")
                    != len(execution_slots)
                    or receipt.get("repair_sequence") != zero_call_failure_count
                    or receipt.get("successor_permitted")
                    is not (zero_call_failure_count == 0)
                    or attempt_id not in bound_execution_contexts
                    or attempt_id not in dispatched_executions
                    or any(start["attempt_id"] == attempt_id for start in starts.values())
                    or any(
                        override_id in delivered_overrides
                        for override_id in attempt_override_ids
                    )
                ):
                    raise LedgerPoisoned(
                        "zero-call execution receipt lacks one clean dispatched attempt"
                    )
                assert isinstance(attempt_id, str)
                discarded_overrides.update(attempt_override_ids)
                finished_executions.add(attempt_id)
                execution_zero_call_failures[execution_key] = (
                    execution_zero_call_failures.get(execution_key, 0) + 1
                )
                zero_call_failure_count += 1
            elif receipt_kind == "branch_group_artifact_completed":
                key = _group_key(receipt["group_id"], receipt["target_id"])
                commitment = commitments.get(key)
                if commitment is None or key in branch_artifacts:
                    raise LedgerPoisoned("branch artifact lacks one unique commitment")
                branch_count = commitment[2]["branch_count"]
                continuation_replicates = commitment[2]["continuation_replicates"]
                expected_candidates = {
                    (*key, slot) for slot in range(1, branch_count)
                }
                expected_executions = {
                    (*key, f"arm-{slot}", replicate)
                    for slot in range(branch_count)
                    for replicate in range(1, continuation_replicates + 1)
                }
                if (
                    receipt.get("branch_count") != branch_count
                    or receipt.get("continuation_replicates")
                    != continuation_replicates
                    or not _is_sha256(receipt.get("artifact_sha256"))
                    or not _is_sha256(receipt.get("training_batch_identity"))
                    or expected_candidates
                    != {slot for slot in candidate_slots if slot[:2] == key}
                    or expected_executions
                    != {item for item in execution_slots if item[:2] == key}
                    or set(body["evidence_refs"]) != {receipt.get("artifact_sha256")}
                ):
                    raise LedgerPoisoned("branch artifact completion changed its denominator")
                branch_artifacts[key] = receipt["artifact_sha256"]
            elif receipt_kind == "training_batch_consumption":
                identity = receipt["training_batch_identity"]
                if identity in batch_claims:
                    raise LedgerPoisoned("training batch was claimed twice")
                batch_claims.add(identity)
            elif receipt_kind == "stage_d_support_gate_pass":
                expected_fields = {
                    "schema_version",
                    "receipt_kind",
                    "ledger_id",
                    "support_rules_sha256",
                    "support_report_sha256",
                    "source_sha256s",
                    "branch_artifact_sha256s",
                }
                sources = receipt.get("source_sha256s")
                artifacts = receipt.get("branch_artifact_sha256s")
                if (
                    set(receipt) != expected_fields
                    or verified_support_report_sha256 is not None
                    or receipt.get("ledger_id") != records[0]["ledger_id"]
                    or receipt.get("support_rules_sha256")
                    != genesis["support_rules_sha256"]
                    or not _is_sha256(receipt.get("support_report_sha256"))
                    or sources != sorted(source_rollout_sha256s.values())
                    or artifacts != sorted(branch_artifacts.values())
                    or set(branch_artifacts) != branch_target_keys
                    or set(body["evidence_refs"])
                    != {receipt.get("support_report_sha256")}
                ):
                    raise LedgerPoisoned("Stage D support pass receipt is invalid")
                verified_support_report_sha256 = receipt["support_report_sha256"]
            elif receipt_kind == "stage_d_training_batch_authorization":
                expected_fields = {
                    "schema_version",
                    "receipt_kind",
                    "ledger_id",
                    "ledger_offset",
                    "prior_chain_sha256",
                    "arm",
                    "training_batch_identity",
                    "sealed_batch_sha256",
                    "objective_sha256",
                    "objective_authorization_sha256",
                    "collection_plan_sha256",
                    "collection_receipt_sha256",
                    "support_report_sha256",
                    "source_sha256s",
                    "branch_artifact_sha256s",
                    "consumer_id",
                    "claim_sequence",
                    "single_use",
                }
                identity = receipt.get("training_batch_identity")
                arm = receipt.get("arm")
                sources = receipt.get("source_sha256s")
                artifacts = receipt.get("branch_artifact_sha256s")
                if (
                    set(receipt) != expected_fields
                    or arm not in {"stock", "branch-global", "local"}
                    or arm in stage_d_training_authorization_arms
                    or not _is_sha256(identity)
                    or identity in batch_claims
                    or receipt.get("claim_sequence") != record["offset"]
                    or receipt.get("single_use") is not True
                    or receipt.get("support_report_sha256")
                    != verified_support_report_sha256
                    or not isinstance(receipt.get("consumer_id"), str)
                    or not receipt["consumer_id"]
                    or not all(
                        _is_sha256(receipt.get(field))
                        for field in (
                            "sealed_batch_sha256",
                            "objective_sha256",
                            "objective_authorization_sha256",
                            "collection_plan_sha256",
                            "collection_receipt_sha256",
                            "support_report_sha256",
                        )
                    )
                    or not isinstance(sources, list)
                    or not sources
                    or sources != sorted(set(sources))
                    or not all(_is_sha256(item) for item in sources)
                    or set(sources) != set(source_rollout_sha256s.values())
                    or not isinstance(artifacts, list)
                    or artifacts != sorted(set(artifacts))
                    or not all(_is_sha256(item) for item in artifacts)
                    or (arm == "stock") != (not artifacts)
                    or (
                        arm != "stock"
                        and (
                            set(branch_artifacts) != branch_target_keys
                            or set(artifacts) != set(branch_artifacts.values())
                        )
                    )
                    or set(body["evidence_refs"])
                    != {
                        receipt.get("sealed_batch_sha256"),
                        receipt.get("objective_authorization_sha256"),
                        receipt.get("collection_plan_sha256"),
                        receipt.get("collection_receipt_sha256"),
                        receipt.get("support_report_sha256"),
                        *artifacts,
                    }
                ):
                    raise LedgerPoisoned("Stage D training batch authorization is invalid")
                assert isinstance(identity, str)
                batch_claims.add(identity)
                assert isinstance(arm, str)
                stage_d_training_authorization_arms.add(arm)
        elif kind == "action_reservation":
            event = _event(body)
            key = _group_key(event["group_id"], event["target_id"])
            commitment = commitments.get(key)
            if (
                commitment is None
                or commitment[0] + 1 != record["offset"]
                or event["commitment_receipt_sha256"] != commitment[1]
                or not _is_sha256(event.get("exact_action_key_digest"))
                or not _is_sha256(event.get("request_sha256"))
                or not isinstance(event.get("reservation_id"), str)
                or not event["reservation_id"]
            ):
                raise LedgerPoisoned("reservation is not immediately after its commitment")
            if key in reservations:
                raise LedgerPoisoned("duplicate recorded action reservation")
            reservations[key] = event
        elif kind == "recorded_action_materialized":
            event = _event(body)
            key = _group_key(event["group_id"], event["target_id"])
            reservation = reservations.get(key)
            call_id = event.get("call_id")
            if (
                reservation is None
                or event.get("reservation_id") != reservation["reservation_id"]
                or event.get("exact_action_key_digest") != reservation["exact_action_key_digest"]
                or not _is_sha256(event.get("action_digest"))
                or call_id not in completed
                or starts.get(call_id, {}).get("attempt_id") != event["reservation_id"]
                or event["reservation_id"] in recorded_action_materialized
            ):
                raise LedgerPoisoned("recorded action materialization is not reservation-bound")
            recorded_action_materialized.add(event["reservation_id"])
        elif kind == "candidate_attempt":
            event = _event(body)
            attempt_id = event["attempt_id"]
            slot = (event["group_id"], event["target_id"], event["action_slot"])
            ordinal = event.get("attempt_ordinal")
            unfinished_same_slot = any(
                (
                    attempt["group_id"],
                    attempt["target_id"],
                    attempt["action_slot"],
                )
                == slot
                and existing_id not in finished_candidates
                for existing_id, attempt in candidate_attempts.items()
            )
            if (
                attempt_id in candidate_attempts
                or (
                    branch_target_roster_sha256 is not None
                    and reconstruction_qa_barrier_sha256 is None
                )
                or slot in candidate_slots
                or unfinished_same_slot
                or type(ordinal) is not int
                or ordinal not in {0, 1}
                or ordinal != candidate_attempt_counts.get(slot, 0)
                or ordinal != candidate_zero_call_failures.get(slot, 0)
            ):
                raise LedgerPoisoned("duplicate candidate attempt")
            candidate_attempts[attempt_id] = event
            candidate_attempt_counts[slot] = ordinal + 1
        elif kind == "execution_attempt":
            event = _event(body)
            attempt_id = event["attempt_id"]
            execution_key = (
                event["group_id"],
                event["target_id"],
                event["arm_id"],
                event["continuation_replicate"],
            )
            ordinal = event.get("attempt_ordinal")
            unfinished_same_slot = any(
                (
                    attempt["group_id"],
                    attempt["target_id"],
                    attempt["arm_id"],
                    attempt["continuation_replicate"],
                )
                == execution_key
                and existing_id not in finished_executions
                for existing_id, attempt in execution_attempts.items()
            )
            if (
                attempt_id in execution_attempts
                or execution_key in execution_slots
                or unfinished_same_slot
                or (
                    branch_target_roster_sha256 is not None
                    and reconstruction_qa_barrier_sha256 is None
                )
                or type(ordinal) is not int
                or ordinal not in {0, 1}
                or ordinal != execution_attempt_counts.get(execution_key, 0)
                or ordinal != execution_zero_call_failures.get(execution_key, 0)
            ):
                raise LedgerPoisoned("duplicate execution attempt")
            execution_attempts[attempt_id] = event
            execution_attempt_counts[execution_key] = ordinal + 1
        elif kind == "execution_context_bound":
            event = _event(body)
            attempt_id = event.get("attempt_id")
            attempt = execution_attempts.get(attempt_id) if isinstance(attempt_id, str) else None
            if (
                attempt is None
                or attempt_id in bound_execution_contexts
                or not _is_sha256(event.get("context_sha256"))
                or any(start["attempt_id"] == attempt_id for start in starts.values())
                or any(
                    event.get(name) != attempt.get(name)
                    for name in (
                        "group_id",
                        "target_id",
                        "arm_id",
                        "continuation_replicate",
                    )
                )
            ):
                raise LedgerPoisoned("execution context binding is invalid or late")
            assert isinstance(attempt_id, str)
            bound_execution_contexts.add(attempt_id)
        elif kind == "execution_dispatched":
            event = _event(body)
            attempt_id = event.get("attempt_id")
            attempt = execution_attempts.get(attempt_id) if isinstance(attempt_id, str) else None
            if (
                attempt is None
                or attempt_id not in bound_execution_contexts
                or attempt_id in dispatched_executions
                or any(start["attempt_id"] == attempt_id for start in starts.values())
                or any(
                    event.get(name) != attempt.get(name)
                    for name in (
                        "group_id",
                        "target_id",
                        "arm_id",
                        "continuation_replicate",
                    )
                )
            ):
                raise LedgerPoisoned("execution dispatch is invalid or out of order")
            assert isinstance(attempt_id, str)
            dispatched_executions.add(attempt_id)
        elif kind == "execution_override_committed":
            event = _event(body)
            attempt_id = event.get("attempt_id")
            override_id = event.get("override_id")
            attempt = execution_attempts.get(attempt_id) if isinstance(attempt_id, str) else None
            address_key = _scanned_address_key(event.get("address"))
            override_key = (attempt_id, address_key) if isinstance(attempt_id, str) else None
            commitment = None
            if attempt is not None:
                group_id = attempt.get("group_id")
                target_id = attempt.get("target_id")
                if isinstance(group_id, str) and isinstance(target_id, str):
                    commitment = commitments.get((group_id, target_id))
            source_reuse_digests = (
                [
                    source_action_digests[key]
                    for key, source_reservation in source_reservations.items()
                    if commitment is not None
                    and key[0] == commitment[2].get("rollout_id")
                    and _scanned_address_key(source_reservation[2].get("target_address"))
                    == address_key
                    and event.get("address") == source_reservation[2].get("target_address")
                    and key in source_action_digests
                ]
                if address_key is not None
                else []
            )
            if (
                attempt is None
                or attempt_id not in bound_execution_contexts
                or attempt_id not in dispatched_executions
                or not isinstance(override_id, str)
                or not override_id
                or override_id in override_commits
                or override_key in override_keys
                or not _is_sha256(event.get("action_digest"))
                or not _is_sha256(event.get("request_sha256"))
                or not _is_sha256(event.get("response_content_sha256"))
                or event.get("disposition") not in {"reuse", "inject"}
                or type(event.get("prompt_tokens")) is not int
                or event["prompt_tokens"] < 0
                or type(event.get("completion_tokens")) is not int
                or event["completion_tokens"] < 1
                or type(event.get("counts_toward_logical_cost")) is not bool
                or set(body["evidence_refs"])
                != {
                    event.get("request_sha256"),
                    event.get("response_content_sha256"),
                }
                or (
                    event.get("disposition") == "inject"
                    and event.get("counts_toward_logical_cost") is not False
                )
                or (
                    event.get("disposition") == "inject"
                    and event.get("action_digest") != attempt.get("action_digest")
                )
                or (
                    event.get("disposition") == "inject"
                    and (
                        commitment is None
                        or event.get("address") != commitment[2].get("target_address")
                    )
                )
                or (
                    branch_target_roster_sha256 is not None
                    and event.get("disposition") == "reuse"
                    and (
                        len(source_reuse_digests) != 1
                        or event.get("action_digest") != source_reuse_digests[0]
                    )
                )
                or any(
                    event.get(name) != attempt.get(name)
                    for name in (
                        "group_id",
                        "target_id",
                        "arm_id",
                        "continuation_replicate",
                    )
                )
                or any(
                    start.get("attempt_id") == attempt_id
                    and _scanned_address_key(start.get("address")) == address_key
                    for start in starts.values()
                )
            ):
                raise LedgerPoisoned("execution override commitment is invalid")
            assert isinstance(override_id, str) and override_key is not None
            override_commits[override_id] = event
            override_keys.add(override_key)
        elif kind == "execution_override_delivered":
            event = _event(body)
            override_id = event.get("override_id")
            committed = (
                override_commits.get(override_id)
                if isinstance(override_id, str)
                else None
            )
            if (
                committed is None
                or override_id in delivered_overrides
                or event.get("attempt_id") != committed.get("attempt_id")
                or not _is_sha256(event.get("typed_response_sha256"))
            ):
                raise LedgerPoisoned("execution override delivery is invalid")
            assert isinstance(override_id, str)
            delivered_overrides.add(override_id)
        elif kind == "model_call_started":
            event = _event(body)
            call_id = event["call_id"]
            if call_id in starts:
                raise LedgerPoisoned("duplicate model call ID")
            attempt_kind = event.get("attempt_kind")
            attempt_id = event.get("attempt_id")
            execution_attempt = (
                execution_attempts.get(attempt_id)
                if attempt_kind == "execution" and isinstance(attempt_id, str)
                else None
            )
            execution_address = (
                _scanned_address_key(event.get("address"))
                if execution_attempt is not None
                else None
            )
            execution_call_key = (
                (attempt_id, execution_address)
                if isinstance(attempt_id, str) and execution_address is not None
                else None
            )
            if (
                (attempt_kind == "candidate" and attempt_id not in candidate_attempts)
                or (
                    attempt_kind == "execution"
                    and (
                        event.get("cache_salt")
                        != _scanned_scheduled_cache_salt(event)
                        or
                        execution_attempt is None
                        or attempt_id not in bound_execution_contexts
                        or attempt_id not in dispatched_executions
                        or execution_call_key in execution_call_keys
                        or any(
                            event.get(name) != execution_attempt.get(name)
                            for name in (
                                "group_id",
                                "target_id",
                                "arm_id",
                                "continuation_replicate",
                            )
                        )
                    )
                )
                or (
                    attempt_kind == "recorded_action"
                    and attempt_id
                    not in {event["reservation_id"] for event in reservations.values()}
                )
                or attempt_kind not in {"candidate", "execution", "recorded_action"}
                or (
                    attempt_kind == "execution"
                    and (
                        attempt_id,
                        _scanned_address_key(event.get("address")),
                    )
                    in override_keys
                )
            ):
                raise LedgerPoisoned("model call start lacks its typed scientific attempt")
            starts[call_id] = event
            if execution_call_key is not None:
                execution_call_keys.add(execution_call_key)
        elif kind == "model_call_completed":
            event = _event(body)
            call_id = event["call_id"]
            if call_id not in starts or call_id in completed:
                raise LedgerPoisoned("completion lacks one in-flight model call")
            if starts[call_id]["attempt_id"] != event["attempt_id"]:
                raise LedgerPoisoned("model call completion changed attempt")
            if (
                starts[call_id].get("attempt_kind") == "execution"
                and call_id not in observed_responses
            ):
                raise LedgerPoisoned("execution completion lacks its raw response witness")
            completed.add(call_id)
            completion_evidence_refs[call_id] = set(body["evidence_refs"])
        elif kind == "model_call_response_observed":
            event = _event(body)
            expected_fields = {
                "attempt_kind",
                "attempt_id",
                "call_id",
                "response_sha256",
            }
            call_id = event.get("call_id")
            response_sha256 = event.get("response_sha256")
            if (
                set(event) != expected_fields
                or event.get("attempt_kind") not in {"candidate", "execution"}
                or call_id not in starts
                or call_id in completed
                or call_id in observed_responses
                or starts[call_id].get("attempt_kind") != event.get("attempt_kind")
                or starts[call_id].get("attempt_id") != event.get("attempt_id")
                or not _is_sha256(response_sha256)
                or set(body["evidence_refs"]) != {response_sha256}
            ):
                raise LedgerPoisoned("candidate raw response witness is invalid")
            assert isinstance(call_id, str) and isinstance(response_sha256, str)
            observed_responses[call_id] = response_sha256
    if set(commitments) != set(reservations):
        raise LedgerPoisoned("commitment is missing its promised action reservation")
    dangling_candidates = set(candidate_attempts) - finished_candidates
    dangling_executions = set(execution_attempts) - finished_executions
    repairable_attempt: dict[str, Any] | None = None
    if dangling_candidates or dangling_executions:
        safe_candidate = False
        safe_execution = False
        dangling_id: str | None = None
        if len(dangling_candidates) == 1 and not dangling_executions:
            dangling_id = next(iter(dangling_candidates))
            safe_candidate = not any(
                start.get("attempt_id") == dangling_id for start in starts.values()
            )
        elif len(dangling_executions) == 1 and not dangling_candidates:
            dangling_id = next(iter(dangling_executions))
            dangling_override_ids = {
                override_id
                for override_id, override in override_commits.items()
                if override["attempt_id"] == dangling_id
            }
            safe_execution = (
                dangling_id in bound_execution_contexts
                and dangling_id in dispatched_executions
                and not any(
                    start.get("attempt_id") == dangling_id for start in starts.values()
                )
                and not (dangling_override_ids & delivered_overrides)
            )
        if (
            not allow_repairable_zero_call
            or zero_call_failure_count != 0
            or not (safe_candidate or safe_execution)
            or dangling_id is None
        ):
            kind = "candidate" if dangling_candidates else "execution"
            raise LedgerPoisoned(f"ledger has a dangling {kind} attempt")
        source = (
            candidate_attempts[dangling_id]
            if safe_candidate
            else execution_attempts[dangling_id]
        )
        repairable_attempt = {
            **source,
            "attempt_kind": "candidate" if safe_candidate else "execution",
        }
    if set(execution_attempts) != bound_execution_contexts:
        raise LedgerPoisoned("execution attempt lacks one frozen context binding")
    if set(execution_attempts) != dispatched_executions:
        raise LedgerPoisoned("execution attempt lacks one irreversible dispatch marker")
    repairable_override_ids = {
        override_id
        for override_id, override in override_commits.items()
        if repairable_attempt is not None
        and repairable_attempt["attempt_kind"] == "execution"
        and override["attempt_id"] == repairable_attempt["attempt_id"]
    }
    if (
        delivered_overrides & discarded_overrides
        or delivered_overrides & repairable_override_ids
        or discarded_overrides & repairable_override_ids
        or set(override_commits)
        != (delivered_overrides | discarded_overrides | repairable_override_ids)
    ):
        raise LedgerPoisoned("ledger has a dangling replay override")
    unresolved_call_ids = set(starts) - completed
    pending_source_recorded_attempts = {
        reservation[2]["recorded_action_reservation_id"]
        for key, reservation in source_reservations.items()
        if key not in source_completions
        and key not in source_aborts
        and reservation[2].get("recorded_action_reservation_id") is not None
    }
    if completed - set(starts) or (
        unresolved_call_ids
        and (
            not allow_source_inflight
            or any(
                starts[call_id].get("attempt_kind") != "recorded_action"
                or starts[call_id].get("attempt_id") not in pending_source_recorded_attempts
                for call_id in unresolved_call_ids
            )
        )
    ):
        raise LedgerPoisoned("ledger has a dangling model_call_started record")
    terminal_source_calls = set(source_completions) | set(source_aborts)
    if set(source_completions) & set(source_aborts):
        raise LedgerPoisoned("source policy call has conflicting terminal receipts")
    if not allow_source_inflight and set(source_reservations) != terminal_source_calls:
        raise LedgerPoisoned("ledger has a dangling source policy call")
    if source_aborts:
        raise LedgerPoisoned("ledger records an aborted source policy call")
    if source_pre_post_aborts:
        raise LedgerPoisoned("ledger records an aborted source child before POST")
    if source_finalization_aborts:
        raise LedgerPoisoned("ledger records an aborted source rollout finalization")
    if records[-1]["record_kind"] == "seal":
        covered = set().union(*source_rollouts.values()) if source_rollouts else set()
        if covered != set(source_completions):
            raise LedgerPoisoned("sealed ledger has unbound source policy completions")
        if source_rollouts and commitments and branch_target_roster_sha256 is None:
            raise LedgerPoisoned("sealed source ledger lacks its branch target roster")
    started_recorded_actions = {
        start["attempt_id"]
        for start in starts.values()
        if start["attempt_kind"] == "recorded_action"
    }
    if (
        recorded_action_materialized - started_recorded_actions
        or (started_recorded_actions - recorded_action_materialized)
        - pending_source_recorded_attempts
    ):
        raise LedgerPoisoned("recorded action call lacks a materialized action output")
    return repairable_attempt



__all__ = ["validate_state_machine"]
