"""Authenticated scientific-contract and cumulative non-overlap projections."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from redco.analysis.stage_d_v13_draft import nonoverlap_digest, sha256_bytes, sha256_json

OBSERVED_EXAMPLE_ID = "qasper-69a7a6675c59a4c5fb70006523b9fe0f01ca415c"
OBSERVED_SEED = 1335879123
AUTHENTICATED_RESUME_RECEIPT_ORDINAL = 179


def _bounded_side(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    result: dict[str, object] = {"presence": value.get("presence", "unknown")}
    for key in ("type", "sha256", "length", "key_count", "keys_sha256"):
        if key in value:
            result[key] = value[key]
    return result


def observed_information(
    audit: dict[str, Any],
    terminal: dict[str, Any],
    evaluator_payload: dict[str, Any],
) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for call in audit["calls"]:
        message = call["message"]
        differences = [
            {
                "pointer": difference["pointer"],
                "reason": difference["reason"],
                "left": _bounded_side(difference["left"]),
                "right": _bounded_side(difference["right"]),
            }
            for difference in message["differences"]
        ]
        calls.append(
            {
                "call_index": call["call_index"],
                "decision_id": call["decision_id"],
                "lineage": call["lineage"],
                "depth": call["depth"],
                "node": call["node"],
                "request_sequence": call["request_sequence"],
                "completion_tokens": call["completion_tokens"],
                "finish_reason": call["finish_reason"],
                "termination_kind": call["termination_kind"],
                "action_evidence_sha256": call["evidence_sha256"],
                "raw_transport_message_sha256": call["evidence_identity"][
                    "action_raw_message_sha256"
                ],
                "message": {
                    "raw_equal": message["raw_equal"],
                    "canonical_equal_under_current_finalizer": message[
                        "canonical_equal_under_current_finalizer"
                    ],
                    "differences": differences,
                },
            }
        )
    if terminal["live_observation"]["source_rollouts_committed"] != 0:
        raise ValueError("v12 durable source-rollout count unexpectedly changed")
    return {
        "schema_version": 1,
        "draft_unfrozen": True,
        "launch_authorized": False,
        "status": "observed_engineering_information_not_admissible_scientific_outcome",
        "v12_not_zero_information": True,
        "durable_facts": {
            "terminal_status": terminal["status"],
            "trace_id": audit["terminal_trace"]["id"],
            "trace_completed": audit["terminal_trace"]["is_completed"],
            "trace_stop_condition": audit["terminal_trace"]["stop_condition"],
            "call_count": audit["terminal_trace"]["call_count"],
            "root_call_count": audit["terminal_trace"]["root_call_count"],
            "child_call_count": audit["terminal_trace"]["child_call_count"],
            "node_count": audit["terminal_trace"]["node_count"],
            "total_prompt_tokens": terminal["live_observation"]["prompt_tokens"],
            "total_completion_tokens": terminal["live_observation"]["completion_tokens"],
            "policy_outputs": 4,
            "child_outputs": 2,
            "evaluator_payload": evaluator_payload,
        },
        "calls": calls,
        "known_failure": audit["known_failure"],
        "durable_disposition": {
            "source_rollouts_committed": terminal["live_observation"]["source_rollouts_committed"],
            "support_gate_evaluations": terminal["live_observation"]["support_gate_evaluations"],
            "branch_target_rosters": terminal["live_observation"]["branch_target_rosters"],
            "branch_outcomes": terminal["live_observation"]["branch_outcomes"],
            "candidate_scores": terminal["live_observation"]["candidate_scores"],
            "support_measurements": terminal["live_observation"]["support_measurements"],
            "scientific_status": terminal["scientific_status"],
            "ledger_status": audit["ledger"]["status"],
            "ledger_records": audit["ledger"]["record_count"],
            "ledger_evidence": audit["ledger"]["evidence_count"],
            "committed_source_artifacts": audit["source_finalization"][
                "committed_source_artifacts"
            ],
            "pending_source_artifacts": audit["source_finalization"]["pending_source_artifacts"],
            "abort_receipt_count": audit["source_finalization"]["abort_receipt_count"],
        },
        "outcome_independence_certificate": {
            "observed_unit_permanently_excluded": True,
            "replacement_selection_uses_observed_score": False,
            "replacement_selection_uses_observed_prompt": False,
            "replacement_selection_uses_observed_sampling": False,
            "replacement_selection_uses_comparator_repair": False,
            "protocol_choices_changed_by_observation": False,
            "thresholds_changed_by_observation": False,
            "selection_inputs_are_predeclared_only": True,
        },
        "interpretation": (
            "The trace and evaluator payload are observed engineering information and are "
            "not admissible scientific outcomes. They did not form a committed SourceRollout, "
            "support evaluation, branch roster, candidate score, support measurement, or "
            "scientific conclusion."
        ),
    }


def _scientific_projection(
    prereg: dict[str, Any],
    protocol: dict[str, Any],
    source_config: dict[str, Any],
    collection: dict[str, Any],
    source_eval: dict[str, Any],
    successor_manifest: dict[str, Any],
    dependency: dict[str, Any],
    consumed_hashes: dict[str, str],
) -> dict[str, Any]:
    env = source_eval["env"]
    agent = env["agent"]
    support = prereg["support_rule"]
    projection = {
        "authenticated_input_hashes": {
            key: consumed_hashes[key]
            for key in sorted(consumed_hashes)
            if key
            in {
                "configs/stage-d/stage-d1-support-protocol-v12.json",
                "configs/stage-d/stage-d1-support-preregistration-v12.json",
                "configs/stage-d/stage-d1-support-collection-plan-v11.json",
                "configs/stage-d/stage-d1-support-source-v12.json",
                "configs/stage-d/stage-d1-support-source-eval-v12.toml",
                "configs/stage-d/stage-d1-dependency-stack-v12.json",
                "datasets/stage-d/qasper-support-successor-v6.jsonl",
                "datasets/stage-d/qasper-support-successor-manifest-v6.json",
                "reports/stage-d1-support-successor-address-audit-v6.json",
                "datasets/stage-d/qasper-support-successor-v5.jsonl",
                "datasets/stage-d/qasper-deterministic-v4.jsonl",
            }
        },
        "denominator_and_threshold": {
            "required_papers": support["required_papers"],
            "required_joint_successes": support["required_joint_successes"],
            "minimum_reward_f1_range": support["minimum_reward_f1_range"],
        },
        "support_definitions": {
            key: support[key] for key in ("N_scaffold", "N_eligible", "N_joint")
        },
        "estimator_and_reporting": {
            "reporting": support["reporting"],
            "confidence": 0.95,
            "interval": "Wilson",
            "ordinary_negatives_remain_in_denominator": support[
                "ordinary_negatives_remain_in_denominator"
            ],
        },
        "branch_protocol": prereg["branch_protocol"],
        "target_timing_and_selection": prereg["target_selection"],
        "protocol_and_evidence_rules": {
            "protocol": protocol,
            "source_action_contract": source_config["action_contract"],
            "source_measurement_semantics": source_config["measurement_semantics_changes"],
            "response_witness_required": source_config["response_witness_required"],
            "support_only": source_config["support_only"],
        },
        "model_checkpoint_tokenizer_renderer": {
            "model": source_eval["model"],
            "checkpoint_id": env["checkpoint_id"],
            "base_model_manifest_sha256": env["base_model_manifest_sha256"],
            "adapter_manifest_sha256": env["adapter_manifest_sha256"],
            "tokenizer_manifest_sha256": env["tokenizer_manifest_sha256"],
            "renderer_manifest_sha256": env["renderer_manifest_sha256"],
            "sampler_conformance_manifest_sha256": env["sampler_conformance_manifest_sha256"],
        },
        "prompt_and_corpus": {
            "dataset": successor_manifest["dataset"],
            "source_revision": successor_manifest["source_revision"],
            "converted_parquet_revision": successor_manifest["converted_parquet_revision"],
            "source_order": successor_manifest["selection"]["source_order"],
            "shuffle": source_eval["shuffle"],
            "scaffold_prompt_sha256": env["taskset"]["scaffold_prompt_sha256"],
            "collection_slots": collection["slots"],
        },
        "sampling_and_termination": {
            "sampling": source_eval["sampling"],
            "max_total_tokens": agent["max_total_tokens"],
            "max_turns": agent["max_turns"],
            "timeouts": agent["timeout"],
            "retries": agent["retries"],
            "maximum_captured_session_call_count": env["maximum_captured_session_call_count"],
            "maximum_observed_root_policy_turn_count": env[
                "maximum_observed_root_policy_turn_count"
            ],
            "no_resampling": prereg["branch_protocol"]["rejection_or_resampling"] == 0,
            "failure_reward": prereg["branch_protocol"]["failure_reward"],
        },
        "serial_execution_and_topology": {
            "max_concurrent": source_eval["max_concurrent"],
            "client_pool_size": source_eval["client"]["pool_size"],
            "branch_count_k": prereg["branch_protocol"]["branch_count"],
            "targets_per_rollout": prereg["target_selection"]["targets_per_rollout"],
            "continuation_replicates": prereg["branch_protocol"]["continuation_replicates"],
        },
        "scorer_evaluator_and_references": {
            "deterministic_scorer": prereg["branch_protocol"]["deterministic_scorer"],
            "evaluator_program_sha256s": dependency["program_sha256s"],
            "one_question_per_paper": successor_manifest["selection"]["one_question_per_paper"],
            "reference_exactness": successor_manifest["checks"]["every_reference_exact"],
            "source_runner_sha256": source_config["source_runner_sha256"],
            "branch_runner_sha256": source_config["branch_runner_sha256"],
        },
    }
    return deepcopy(projection)


def build_v12_scientific_contract(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Project the authenticated v12 pre-outcome inputs into a full contract."""

    return _scientific_projection(*args, **kwargs)


def build_v13_scientific_contract(
    prereg: dict[str, Any],
    protocol: dict[str, Any],
    source_config: dict[str, Any],
    collection: dict[str, Any],
    source_eval: dict[str, Any],
    successor_manifest: dict[str, Any],
    dependency: dict[str, Any],
    consumed_hashes: dict[str, str],
) -> dict[str, Any]:
    """Independently assemble the v13 inherited scientific projection."""

    result = {
        "authenticated_input_hashes": {
            key: consumed_hashes[key]
            for key in sorted(consumed_hashes)
            if key.startswith(("configs/stage-d/", "datasets/stage-d/", "reports/"))
            and key
            in {
                "configs/stage-d/stage-d1-support-protocol-v12.json",
                "configs/stage-d/stage-d1-support-preregistration-v12.json",
                "configs/stage-d/stage-d1-support-collection-plan-v11.json",
                "configs/stage-d/stage-d1-support-source-v12.json",
                "configs/stage-d/stage-d1-support-source-eval-v12.toml",
                "configs/stage-d/stage-d1-dependency-stack-v12.json",
                "datasets/stage-d/qasper-support-successor-v6.jsonl",
                "datasets/stage-d/qasper-support-successor-manifest-v6.json",
                "reports/stage-d1-support-successor-address-audit-v6.json",
                "datasets/stage-d/qasper-support-successor-v5.jsonl",
                "datasets/stage-d/qasper-deterministic-v4.jsonl",
            }
        },
        "denominator_and_threshold": dict(
            required_papers=prereg["support_rule"]["required_papers"],
            required_joint_successes=prereg["support_rule"]["required_joint_successes"],
            minimum_reward_f1_range=prereg["support_rule"]["minimum_reward_f1_range"],
        ),
        "support_definitions": {
            "N_scaffold": prereg["support_rule"]["N_scaffold"],
            "N_eligible": prereg["support_rule"]["N_eligible"],
            "N_joint": prereg["support_rule"]["N_joint"],
        },
        "estimator_and_reporting": {
            "reporting": list(prereg["support_rule"]["reporting"]),
            "confidence": 0.95,
            "interval": "Wilson",
            "ordinary_negatives_remain_in_denominator": prereg["support_rule"][
                "ordinary_negatives_remain_in_denominator"
            ],
        },
        "branch_protocol": dict(prereg["branch_protocol"]),
        "target_timing_and_selection": dict(prereg["target_selection"]),
        "protocol_and_evidence_rules": {
            "protocol": dict(protocol),
            "source_action_contract": str(source_config["action_contract"]),
            "source_measurement_semantics": list(source_config["measurement_semantics_changes"]),
            "response_witness_required": source_config["response_witness_required"],
            "support_only": source_config["support_only"],
        },
        "model_checkpoint_tokenizer_renderer": {
            "model": source_eval["model"],
            "checkpoint_id": source_eval["env"]["checkpoint_id"],
            "base_model_manifest_sha256": source_eval["env"]["base_model_manifest_sha256"],
            "adapter_manifest_sha256": source_eval["env"]["adapter_manifest_sha256"],
            "tokenizer_manifest_sha256": source_eval["env"]["tokenizer_manifest_sha256"],
            "renderer_manifest_sha256": source_eval["env"]["renderer_manifest_sha256"],
            "sampler_conformance_manifest_sha256": source_eval["env"][
                "sampler_conformance_manifest_sha256"
            ],
        },
        "prompt_and_corpus": {
            "dataset": successor_manifest["dataset"],
            "source_revision": successor_manifest["source_revision"],
            "converted_parquet_revision": successor_manifest["converted_parquet_revision"],
            "source_order": bool(successor_manifest["selection"]["source_order"]),
            "shuffle": bool(source_eval["shuffle"]),
            "scaffold_prompt_sha256": source_eval["env"]["taskset"]["scaffold_prompt_sha256"],
            "collection_slots": [dict(slot) for slot in collection["slots"]],
        },
        "sampling_and_termination": {
            "sampling": dict(source_eval["sampling"]),
            "max_total_tokens": source_eval["env"]["agent"]["max_total_tokens"],
            "max_turns": source_eval["env"]["agent"]["max_turns"],
            "timeouts": dict(source_eval["env"]["agent"]["timeout"]),
            "retries": dict(source_eval["env"]["agent"]["retries"]),
            "maximum_captured_session_call_count": source_eval["env"][
                "maximum_captured_session_call_count"
            ],
            "maximum_observed_root_policy_turn_count": source_eval["env"][
                "maximum_observed_root_policy_turn_count"
            ],
            "no_resampling": bool(prereg["branch_protocol"]["rejection_or_resampling"] == 0),
            "failure_reward": prereg["branch_protocol"]["failure_reward"],
        },
        "serial_execution_and_topology": {
            "max_concurrent": source_eval["max_concurrent"],
            "client_pool_size": source_eval["client"]["pool_size"],
            "branch_count_k": prereg["branch_protocol"]["branch_count"],
            "targets_per_rollout": prereg["target_selection"]["targets_per_rollout"],
            "continuation_replicates": prereg["branch_protocol"]["continuation_replicates"],
        },
        "scorer_evaluator_and_references": {
            "deterministic_scorer": str(prereg["branch_protocol"]["deterministic_scorer"]),
            "evaluator_program_sha256s": dict(dependency["program_sha256s"]),
            "one_question_per_paper": bool(
                successor_manifest["selection"]["one_question_per_paper"]
            ),
            "reference_exactness": bool(successor_manifest["checks"]["every_reference_exact"]),
            "source_runner_sha256": source_config["source_runner_sha256"],
            "branch_runner_sha256": source_config["branch_runner_sha256"],
        },
    }
    return deepcopy(result)


def compare_scientific_contracts(
    v12_contract: dict[str, Any], v13_contract: dict[str, Any]
) -> dict[str, Any]:
    if v12_contract is v13_contract:
        raise ValueError("scientific contract comparison requires independent objects")
    field_hashes = {
        key: {
            "v12_sha256": sha256_json(v12_contract[key]),
            "v13_sha256": sha256_json(v13_contract[key]),
            "equal": v12_contract[key] == v13_contract[key],
        }
        for key in sorted(set(v12_contract) | set(v13_contract))
        if key in v12_contract and key in v13_contract
    }
    missing = sorted(set(v12_contract) ^ set(v13_contract))
    if missing or not all(item["equal"] for item in field_hashes.values()):
        raise ValueError(f"scientific contract changed: missing={missing}")
    return {"exact_equal": True, "field_hashes": field_hashes}


def _row_values(rows: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    papers: list[str] = []
    examples: list[str] = []
    rendered: list[str] = []
    references: list[str] = []
    for row in rows:
        papers.append(str(row["paper_id"]))
        examples.append(str(row["example_id"]))
        rendered.append(sha256_bytes(str(row["paper"]).encode("utf-8")))
        references.extend(str(span) for span in row["reference_evidence"])
    return {
        "paper_ids": papers,
        "example_ids": examples,
        "rendered_paper_hashes": rendered,
        "reference_spans": references,
    }


def _recursive_identity_values(value: object, keys: set[str]) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys and isinstance(child, (str, int)):
                found.append(str(child))
            found.extend(_recursive_identity_values(child, keys))
    elif isinstance(value, list):
        for child in value:
            found.extend(_recursive_identity_values(child, keys))
    return found


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def build_nonoverlap(
    rows: list[dict[str, Any]],
    historical_rows: list[dict[str, Any]],
    collection_plan: dict[str, Any],
    identities: dict[str, str],
    audit: dict[str, Any],
    terminal: dict[str, Any],
    successor_manifest: dict[str, Any],
    source_records: Iterable[dict[str, Any]] = (),
    historical_identity_witness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if historical_identity_witness is None:
        raise ValueError("authenticated historical identity witness is required")
    row_values = {
        name: _unique(values) for name, values in _row_values([*historical_rows, *rows]).items()
    }
    row_values["paper_ids"] = _unique(
        [
            *row_values["paper_ids"],
            *(
                str(value)
                for value in successor_manifest["successor"]["historically_retired_paper_ids"]
            ),
        ]
    )
    slots = collection_plan["slots"]
    group_slot_values = {
        "scientific_group_ids": _unique(str(slot["scientific_group_id"]) for slot in slots),
        "slot_ids": _unique(str(slot["slot_id"]) for slot in slots),
        "cache_salts": _unique(str(slot["cache_salt"]) for slot in slots),
        "seeds": _unique(str(slot["seed"]) for slot in slots),
    }
    identity_keys = {
        "decision_id",
        "lineage",
        "trace_id",
        "session_id",
        "rollout_id",
        "source_rollout_id",
        "candidate_id",
        "request_id",
        "node_id",
    }
    raw_known_calls = _recursive_identity_values(
        {"audit": audit, "terminal": terminal, "successor": successor_manifest},
        identity_keys,
    )
    for record in source_records:
        raw_known_calls.extend(_recursive_identity_values(record, identity_keys))
    explicit_call_values = (
        {str(call["decision_id"]) for call in audit["calls"]}
        | {str(call["lineage"]) for call in audit["calls"]}
        | {str(call["request_sequence"]) for call in audit["calls"]}
    )
    known_calls = _unique(value for value in raw_known_calls if value not in explicit_call_values)
    prior_call_identifiers = {
        "decision_ids": _unique(str(call["decision_id"]) for call in audit["calls"]),
        "lineages": _unique(str(call["lineage"]) for call in audit["calls"]),
        "request_sequences": _unique(str(call["request_sequence"]) for call in audit["calls"]),
        "recursive_trace_and_session_ids": known_calls,
    }
    forbidden = {
        **row_values,
        **group_slot_values,
        **prior_call_identifiers,
        "fresh_campaign_identity_inputs": [],
    }
    witness_sets = historical_identity_witness.get("identity_sets")
    if not isinstance(witness_sets, dict) or not historical_identity_witness.get(
        "witness_sha256"
    ):
        raise ValueError("authenticated historical identity witness is incomplete")
    for name, values in witness_sets.items():
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"historical identity set is malformed: {name}")
        target = (
            name
            if name
            in {
                "paper_ids",
                "example_ids",
                "reference_spans",
                "seeds",
                "groups",
                "slots",
                "cache_salts",
            }
            else f"historical_{name}"
        )
        target = {"groups": "scientific_group_ids", "slots": "slot_ids"}.get(target, target)
        forbidden[target] = _unique([*forbidden.get(target, []), *values])
    fresh_ids = [
        identities["campaign_id"],
        identities["run_id"],
        identities["genesis_id"],
        identities["ledger_id"],
        identities["output_root_id"],
    ]
    forbidden["fresh_campaign_identity_inputs"] = fresh_ids
    checks = nonoverlap_digest(forbidden)
    forbidden_values = {
        item
        for name, values in forbidden.items()
        if name != "fresh_campaign_identity_inputs"
        for item in values
    }
    checks["checks"].update(
        {
            "fresh_administrative_ids_disjoint_from_forbidden": not (
                set(fresh_ids) & forbidden_values
            ),
            "fresh_campaign_seed_disjoint_from_forbidden": identities["campaign_seed"]
            not in forbidden_values,
            "retired_observed_unit_present_in_forbidden_universe": OBSERVED_EXAMPLE_ID
            in set(forbidden["example_ids"]),
        }
    )
    checks["all_known_nonoverlap_checks"] = all(checks["checks"].values())
    return {
        "schema_version": 1,
        "draft_unfrozen": True,
        "launch_authorized": False,
        "domain": "redco-stage-d1-support-v13-cumulative-nonoverlap-audit-v2",
        "status": "blocked_pending_authenticated_scan_after_receipt_179",
        "collision_rule": (
            "A collision is exact equality after UTF-8 canonicalization for any paper, example, "
            "rendered-paper hash, reference span, source/candidate/rollout/session/trace ID, "
            "group, slot, cache salt, seed, or prior call identifier."
        ),
        "known_values": {name: len(values) for name, values in forbidden.items()},
        "checks": checks,
        "candidate_dependent_checks": checks["candidate_checks"],
        "candidate": None,
        "historical_identity_witness": historical_identity_witness,
        "unresolved": (
            "Resume the authenticated source-order scan after receipt ordinal 179. Candidate "
            "ordinal, row, paper, reference, source/candidate/rollout/session identities, seed, "
            "and address remain null until that selector succeeds."
        ),
    }


__all__ = [
    "AUTHENTICATED_RESUME_RECEIPT_ORDINAL",
    "OBSERVED_EXAMPLE_ID",
    "OBSERVED_SEED",
    "build_nonoverlap",
    "build_v12_scientific_contract",
    "build_v13_scientific_contract",
    "compare_scientific_contracts",
    "observed_information",
]
