"""CPU-only v13 support protocol and the fixed ordinal-180 materializer.

This module consumes the already authenticated selection receipt; it does not
run the selector again.  The materializer authenticates Parquet metadata and
then asks PyArrow for one-row batches through ordinal 180, stopping before any
request for ordinal 181.  It emits compact candidate/cohort/protocol artifacts
and never grants provider, model, science, or launch authority.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

from redco.analysis import stage_d_v13_support_contract as support_contract
from redco.analysis.stage_d_collection import (
    SourceCollectionSlot,
    derive_scientific_group_id,
    derive_source_episode_seed_and_salt,
)
from redco.analysis.stage_d_dependency_stack import live_owner_dependency_payload
from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes, sha256_json
from redco.analysis.stage_d_v13_source_phase_a_decoder import source_row_sha256
from redco.analysis.stage_d_v13_source_phase_a_selector import exact_reference, render_paper
from redco.analysis.stage_d_v13_support_contract import (
    ADDRESS_AUDIT_RELATIVE,
    ADDRESS_AUDIT_SHA256,
    AUTHENTICATED_PREDECESSOR_HASHES,
    CANDIDATE_AUTHORITY,
    CANDIDATE_EXAMPLE_ID,
    CANDIDATE_PAPER_ID,
    CANDIDATE_QUESTION_INDEX,
    CANDIDATE_RELATIVE,
    CANDIDATE_ROW_SHA256,
    CANDIDATE_SELECTION_ADDRESS_SHA256,
    CANDIDATE_SOURCE_ORDINAL,
    COLLECTION_PLAN_RELATIVE,
    COLLECTION_PLAN_SHA256,
    COMPOSITION_AUTHORIZATION,
    COMPOSITION_RELATIVE,
    FROZEN_SUPPORT_RULES_RELATIVE,
    FROZEN_SUPPORT_RULES_SHA256,
    MASTER_SEED,
    PROTOCOL_AUDIT_RELATIVE,
    PROTOCOL_AUTHORIZATION,
    PROTOCOL_RELATIVE,
    RETAINED_SUPPORT_RELATIVE,
    RETAINED_SUPPORT_SHA256,
    REVIEWED_PROTOCOL_ARTIFACT_SHA256,
    SCIENTIFIC_NAMESPACE,
    SELECTION_CLAIM_RELATIVE,
    SELECTION_CLAIM_SHA256,
    SELECTION_MANIFEST_RELATIVE,
    SELECTION_MANIFEST_SHA256,
    SELECTION_ORIGINAL_CLAIM_RELATIVE,
    SELECTION_RECEIPT_RELATIVE,
    SELECTION_RECEIPT_SHA256,
    SOURCE_ARTIFACT_RELATIVE,
    SOURCE_BYTES,
    SOURCE_FIELDS,
    SOURCE_LOGICAL_URL,
    SOURCE_PATH,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    SOURCE_ROW_COUNT,
    SOURCE_SCHEMA_SHA256,
    SOURCE_SEMANTIC_COMMIT,
    SOURCE_SHA256,
    SUPPORT_COHORT,
    SUPPORT_RULES_SHA256,
    SUPPORTED_DATASETS,
    SUPPORTED_PYARROW,
    SUPPORTED_PYTHON,
    V6_MANIFEST_RELATIVE,
    V6_MANIFEST_SHA256,
    V12_ARCHIVE_RELATIVE,
    V12_ARCHIVE_SHA256,
    V12_EVIDENCE_MANIFEST_RELATIVE,
    V12_EVIDENCE_MANIFEST_SHA256,
    V12_FINALIZATION_AUDIT_RELATIVE,
    V12_FINALIZATION_AUDIT_SHA256,
    V12_PREREG_RELATIVE,
    V12_PREREG_SHA256,
    V12_PROTOCOL_RELATIVE,
    V12_PROTOCOL_SHA256,
    V12_SOURCE_EVAL_RELATIVE,
    V12_SOURCE_EVAL_SHA256,
    V12_TERMINAL_REPORT_RELATIVE,
    V12_TERMINAL_REPORT_SHA256,
    CandidateReadInstrumentation,
    authenticate_upstream_evidence,
    load_parquet,
    protocol_source_binding,
    read_authenticated,
    require_supported_runtime,
    runtime_payload,
    sampling_contract_binding,
    source_contract,
)
from redco.analysis.stage_d_v13_support_publication import (
    authenticate_protocol_artifact_bytes,
    check_protocol_artifacts,
)


def _candidate_record(row: Mapping[str, Any]) -> dict[str, Any]:
    paper = render_paper(row)
    qas = cast(Mapping[str, Any], row["qas"])
    questions = cast(list[str], qas["question"])
    answers = cast(list[Mapping[str, Any]], qas["answers"])
    question_ids = cast(list[str], qas["question_id"])
    if len(questions) <= CANDIDATE_QUESTION_INDEX or len(answers) <= CANDIDATE_QUESTION_INDEX:
        raise ValueError("authenticated candidate question index is unavailable")
    if str(row["id"]) != CANDIDATE_PAPER_ID:
        raise ValueError("authenticated candidate paper differs")
    example_id = f"qasper-{question_ids[CANDIDATE_QUESTION_INDEX]}"
    if example_id != CANDIDATE_EXAMPLE_ID:
        raise ValueError("authenticated candidate example differs")
    answer = answers[CANDIDATE_QUESTION_INDEX]
    reference = exact_reference(paper, answer)
    if reference is None:
        raise ValueError("authenticated candidate lacks reference evidence")
    evidence, answer_type = reference
    return {
        "answer_type": answer_type,
        "example_id": example_id,
        "paper": paper,
        "paper_id": CANDIDATE_PAPER_ID,
        "question": questions[CANDIDATE_QUESTION_INDEX],
        "reference_evidence": list(evidence),
        "split": "successor_support",
        "title": str(row["title"]),
    }


def _fresh_rollout(candidate: Mapping[str, Any]) -> dict[str, Any]:
    group_id = derive_scientific_group_id(
        namespace=SCIENTIFIC_NAMESPACE,
        example_id=str(candidate["example_id"]),
    )
    seed, cache_salt = derive_source_episode_seed_and_salt(
        master_seed=MASTER_SEED,
        scientific_group_id=group_id,
        rollout_slot=0,
    )
    slot = SourceCollectionSlot.build(
        {
            "scientific_group_id": group_id,
            "example_id": candidate["example_id"],
            "rollout_slot": 0,
        },
        master_seed=MASTER_SEED,
    )
    address_sha = sha256_json(
        {
            "domain": "redco-stage-d1-support-v13-fresh-rollout-address-v1",
            "slot_id": slot.slot_id,
            "group_id": group_id,
            "example_id": candidate["example_id"],
            "rollout_slot": 0,
        }
    )
    if address_sha == CANDIDATE_SELECTION_ADDRESS_SHA256:
        raise ValueError("support rollout address reused the selection address")
    return {
        "scientific_group_id": group_id,
        "rollout_slot": 0,
        "seed": seed,
        "cache_salt": cache_salt,
        "slot_id": slot.slot_id,
        "rollout_id": f"stage-d1-support-v13-rollout-{address_sha[:24]}",
        "address_sha256": address_sha,
        "selection_address_sha256": CANDIDATE_SELECTION_ADDRESS_SHA256,
    }


def materialize_candidate(
    root: Path,
    output_path: Path | None = None,
    *,
    instrumentation: CandidateReadInstrumentation | None = None,
) -> dict[str, Any]:
    """Materialize only the receipt-bound row at ordinal 180."""

    observer = instrumentation or CandidateReadInstrumentation()
    upstream = authenticate_upstream_evidence(root)
    parquet_path = root / SOURCE_ARTIFACT_RELATIVE
    source, parquet_file = source_contract(root, parquet_path)
    return _materialize_candidate_with_authenticated_inputs(
        output_path,
        observer,
        upstream,
        source,
        parquet_file,
    )


def _materialize_candidate_with_authenticated_inputs(
    output_path: Path | None,
    observer: CandidateReadInstrumentation,
    upstream: Mapping[str, Any],
    source: Mapping[str, Any],
    parquet_file: Any,
) -> dict[str, Any]:
    selected_row: dict[str, Any] | None = None
    for ordinal, batch in enumerate(
        parquet_file.iter_batches(batch_size=1, row_groups=[0], use_threads=False)
    ):
        observer.record_arrow_batch(ordinal, batch.num_rows)
        observer.record_request(ordinal)
        if ordinal > CANDIDATE_SOURCE_ORDINAL:
            raise RuntimeError("candidate materializer crossed ordinal wall")
        if ordinal == CANDIDATE_SOURCE_ORDINAL:
            values = batch.to_pylist()
            if len(values) != 1 or not isinstance(values[0], dict):
                raise ValueError("candidate batch cardinality differs")
            observer.record_materialized(ordinal)
            selected_row = cast(dict[str, Any], values[0])
            break
    if selected_row is None:
        raise ValueError("authenticated ordinal-180 row was not found")
    observer.record_canonicalized(CANDIDATE_SOURCE_ORDINAL)
    row_sha = source_row_sha256(selected_row)
    if row_sha != CANDIDATE_ROW_SHA256:
        raise ValueError("authenticated candidate row hash differs")
    candidate = _candidate_record(selected_row)
    observer.record_evaluated(CANDIDATE_SOURCE_ORDINAL)
    rollout = _fresh_rollout(candidate)
    payload = {
        "schema_version": 1,
        "domain": "redco-stage-d1-support-v13-candidate-ordinal-180-v1",
        "source": {
            "ordinal": CANDIDATE_SOURCE_ORDINAL,
            "paper_id": CANDIDATE_PAPER_ID,
            "example_id": CANDIDATE_EXAMPLE_ID,
            "question_index": CANDIDATE_QUESTION_INDEX,
            "row_sha256": row_sha,
            "rendered_paper_sha256": sha256_bytes(str(candidate["paper"]).encode("utf-8")),
            "reference_sha256": [
                sha256_bytes(str(value).encode("utf-8"))
                for value in cast(list[str], candidate["reference_evidence"])
            ],
            "reference_count": len(candidate["reference_evidence"]),
            "selection_receipt_sha256": SELECTION_RECEIPT_SHA256,
            "selection_evidence_manifest_sha256": upstream["selection_manifest_sha256"],
            "selection_claim_sha256": upstream["selection_claim_sha256"],
            "upstream_evidence_hashes": upstream["upstream_hashes"],
            "frozen_decision_rule_sha256": upstream["decision_rule_sha256"],
            "frozen_support_rules_sha256": upstream["support_rules_sha256"],
            "runtime": source["runtime"],
        },
        "candidate": candidate,
        "fresh_support_rollout": rollout,
        "instrumentation": observer.to_payload(),
        "authority": CANDIDATE_AUTHORITY.copy(),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(canonical_json_bytes(payload))
    return payload


def _binomial_tail(n: int, p: float, threshold: int) -> float:
    return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(threshold, n + 1))


def _power_report() -> dict[str, Any]:
    alpha = 0.05
    comparisons = 2
    adjusted_alpha = alpha / comparisons
    target_effect = 0.5
    n = 32
    z_critical = NormalDist().inv_cdf(1 - adjusted_alpha / 2)
    z_power = NormalDist().inv_cdf(0.8)
    mde = (z_critical + z_power) / (n**0.5)
    power = NormalDist().cdf(target_effect * (n**0.5) - z_critical)
    return {
        "paired_papers": n,
        "primary_contrasts": ["branch-global_vs_stock", "local_vs_branch-global"],
        "alpha": alpha,
        "multiplicity": "Holm; conservative Bonferroni planning alpha=0.025 per contrast",
        "target_standardized_effect": target_effect,
        "normal_approximation_mde_at_80_percent": mde,
        "normal_approximation_power_at_target_effect": power,
        "status": "exploratory_underpowered_for_80_percent_target",
        "adapt_after_support": False,
    }


def build_protocol_artifacts(root: Path, output_root: Path) -> dict[str, bytes]:
    """Build candidate, composition, protocol, and audit bytes in memory.

    The authenticated current tree is rebuilt from its reviewed local bytes;
    this keeps routine rebuild/check paths source-free.  The one-time
    materializer remains below for a repository that does not yet have the
    reviewed candidate set.  ``output_root`` is retained for API compatibility
    and is not written by this pure builder.
    """

    del output_root
    if all(
        (root / relative).is_file() and not (root / relative).is_symlink()
        for relative in support_contract.REVIEWED_PROTOCOL_ARTIFACT_SHA256
    ):
        return rebuild_protocol_artifacts_from_existing(root)
    upstream = authenticate_upstream_evidence(root)
    dependency_stack = live_owner_dependency_payload(root)
    sampling_contract = sampling_contract_binding(root)
    parquet_path = root / SOURCE_ARTIFACT_RELATIVE
    source, parquet_file = source_contract(root, parquet_path)
    candidate = _materialize_candidate_with_authenticated_inputs(
        None,
        CandidateReadInstrumentation(),
        upstream,
        source,
        parquet_file,
    )
    candidate["sampling_contract"] = sampling_contract
    authenticated_predecessors = cast(Mapping[str, bytes], upstream["authenticated_predecessors"])
    retained_raw = authenticated_predecessors[RETAINED_SUPPORT_RELATIVE]
    retained_rows = [json.loads(line) for line in retained_raw.splitlines()]
    if (
        len(retained_rows) != 111
        or sum(row["split"] == "successor_support" for row in retained_rows) != 63
    ):
        raise ValueError("retained support composition differs")
    plan_raw = authenticated_predecessors[COLLECTION_PLAN_RELATIVE]
    plan = json.loads(plan_raw)
    if not isinstance(plan, dict) or not isinstance(plan.get("slots"), list):
        raise ValueError("collection plan schema differs")
    address_raw = authenticated_predecessors[ADDRESS_AUDIT_RELATIVE]
    address_audit = json.loads(address_raw)
    if (
        not isinstance(address_audit, dict)
        or address_audit.get("schema_version") != 1
        or address_audit.get("domain") != "redco-stage-d-support-successor-address-audit-v1"
        or address_audit.get("checks")
        != {
            "preserved_63_addresses_exact": True,
            "prior_plan_has_64_unique_slots": True,
            "reserve_address_fresh": True,
            "retired_address_absent": True,
            "successor_plan_has_64_unique_slots": True,
        }
        or not isinstance(address_audit.get("preserved"), list)
        or len(address_audit["preserved"]) != 63
        or not isinstance(address_audit.get("retired"), dict)
        or not isinstance(address_audit.get("reserve"), dict)
    ):
        raise ValueError("authenticated predecessor address audit schema differs")
    candidate_raw = canonical_json_bytes(candidate)
    retained_examples = {str(row["example_id"]) for row in retained_rows}
    retained_papers = {str(row["paper_id"]) for row in retained_rows}
    retained_rendered = {sha256_bytes(str(row["paper"]).encode("utf-8")) for row in retained_rows}
    retained_references = {
        sha256_bytes(str(span).encode("utf-8"))
        for row in retained_rows
        for span in row["reference_evidence"]
    }
    candidate_source = cast(Mapping[str, Any], candidate["source"])
    candidate_rollout = cast(Mapping[str, Any], candidate["fresh_support_rollout"])
    plan_slots = cast(list[Mapping[str, Any]], plan["slots"])
    plan_identity_values = {
        value
        for slot in plan_slots
        for value in (
            str(slot["slot_id"]),
            str(slot["seed"]),
            str(slot["cache_salt"]),
        )
    }
    candidate_identity_values = {
        str(candidate_rollout["slot_id"]),
        str(candidate_rollout["seed"]),
        str(candidate_rollout["cache_salt"]),
    }
    historical_addresses = [
        *address_audit["preserved"],
        address_audit["retired"],
        address_audit["reserve"],
    ]
    historical_identity_values = {
        str(value)
        for address in historical_addresses
        for value in (
            address.get("paper_id"),
            address.get("example_id"),
            address.get("canonical_row_sha256", address.get("row_sha256")),
            address.get("scientific_group_id"),
            address.get("slot_id"),
            address.get("seed"),
            address.get("cache_salt"),
        )
        if value is not None
    }
    candidate_historical_values = {
        CANDIDATE_PAPER_ID,
        CANDIDATE_EXAMPLE_ID,
        CANDIDATE_ROW_SHA256,
        str(candidate_rollout["scientific_group_id"]),
        str(candidate_rollout["slot_id"]),
        str(candidate_rollout["seed"]),
        str(candidate_rollout["cache_salt"]),
    }
    historical_nonoverlap = not (candidate_historical_values & historical_identity_values)
    nonoverlap = {
        "example_id": CANDIDATE_EXAMPLE_ID not in retained_examples,
        "paper_id": CANDIDATE_PAPER_ID not in retained_papers,
        "rendered_paper": candidate_source["rendered_paper_sha256"] not in retained_rendered,
        "references": not (set(candidate_source["reference_sha256"]) & retained_references),
        "source_row": CANDIDATE_ROW_SHA256
        not in {str(row.get("source_row_sha256")) for row in retained_rows},
        "fresh_slot_seed_salt": not (candidate_identity_values & plan_identity_values),
        "historical_address_identities": historical_nonoverlap,
    }
    if not all(nonoverlap.values()):
        raise ValueError("candidate identity overlaps a retained support identity")
    selection_evidence = {
        "receipt_path": SELECTION_RECEIPT_RELATIVE,
        "receipt_sha256": SELECTION_RECEIPT_SHA256,
        "manifest_path": SELECTION_MANIFEST_RELATIVE,
        "manifest_sha256": upstream["selection_manifest_sha256"],
        "claim_sha256": upstream["selection_claim_sha256"],
        "upstream_v12_hashes": upstream["upstream_hashes"],
    }
    composition = {
        "schema_version": 1,
        "domain": "redco-stage-d-qasper-support-successor-v8-candidate-composition-v1",
        "status": "candidate_materialized_cpu_only_no_launch",
        "sampling_contract": sampling_contract,
        "support_cohort": SUPPORT_COHORT.copy(),
        "retained_base": {
            "path": RETAINED_SUPPORT_RELATIVE,
            "sha256": RETAINED_SUPPORT_SHA256,
            "bytes": len(retained_raw),
            "rows": len(retained_rows),
            "support_rows": 63,
            "byte_identity_preserved": True,
        },
        "candidate": {
            "path": CANDIDATE_RELATIVE,
            "sha256": sha256_bytes(candidate_raw),
            "bytes": len(candidate_raw),
            "ordinal": CANDIDATE_SOURCE_ORDINAL,
            "paper_id": CANDIDATE_PAPER_ID,
            "example_id": CANDIDATE_EXAMPLE_ID,
            "row_sha256": CANDIDATE_ROW_SHA256,
            "selection_address_sha256": CANDIDATE_SELECTION_ADDRESS_SHA256,
            "fresh_support_address_sha256": candidate["fresh_support_rollout"]["address_sha256"],
        },
        "predecessors": {
            "collection_plan": {"path": COLLECTION_PLAN_RELATIVE, "sha256": COLLECTION_PLAN_SHA256},
            "v6_manifest": {"path": V6_MANIFEST_RELATIVE, "sha256": V6_MANIFEST_SHA256},
            "address_audit": {"path": ADDRESS_AUDIT_RELATIVE, "sha256": ADDRESS_AUDIT_SHA256},
        },
        "selection_evidence": selection_evidence.copy(),
        "authenticated_address_audit": {
            "preserved_count": len(address_audit["preserved"]),
            "retired_count": 1,
            "reserve_count": 1,
            "checks": address_audit["checks"],
        },
        "nonoverlap": {
            **nonoverlap,
            "selection_address_not_reused_for_support": True,
        },
        "authorization": COMPOSITION_AUTHORIZATION.copy(),
    }
    probabilities = {
        "n": 64,
        "threshold": 58,
        "tail_probabilities": {
            "p_0.808": _binomial_tail(64, 0.808, 58),
            "p_0.90": _binomial_tail(64, 0.90, 58),
            "p_0.95": _binomial_tail(64, 0.95, 58),
        },
        "intended_reference_probability": "P(X >= 58 | p), exact binomial tail",
    }
    protocol = {
        "schema_version": 1,
        "domain": "redco-stage-d1-support-v13-frozen-support-protocol-v1",
        "status": "frozen_cpu_preregistration_no_live_authority",
        "dependency_stack": dependency_stack,
        "sampling_contract": sampling_contract,
        "candidate": {"ordinal": CANDIDATE_SOURCE_ORDINAL, "example_id": CANDIDATE_EXAMPLE_ID},
        "selection_evidence": {
            **selection_evidence,
            "frozen_decision_rule_sha256": upstream["decision_rule_sha256"],
            "frozen_support_rules_sha256": upstream["support_rules_sha256"],
        },
        "source": protocol_source_binding(),
        "support_sequence": [
            "zero_call_deployment_preflight",
            "one_outcome_bearing_64_paper_support_collection",
            "no_early_pass_retry_or_task_redesign",
            "recover_hash_evidence_terminate_resources",
            "review_support_report_before_science",
        ],
        "attempt_policy": {
            "maximum_live_support_attempts_global": 1,
            "outcome_bearing_cohorts": 1,
            "second_outcome_bearing_attempt": "forbidden_unconditionally",
            "environmental_redeployment": (
                "only_outcome_independent_zero_provider_call_failure_before_first_provider_post"
            ),
        },
        "environmental_repair_list": [
            "zero_provider_call_dependency_or_cache_reconstruction",
            "zero_provider_call_endpoint_readiness_repair",
            "zero_provider_call_resource_redeployment_before_first_provider_post",
        ],
        "inherited_frozen_scientific_inputs": {
            "v12_preregistration": {
                "path": V12_PREREG_RELATIVE,
                "sha256": V12_PREREG_SHA256,
            },
            "v12_protocol": {
                "path": V12_PROTOCOL_RELATIVE,
                "sha256": V12_PROTOCOL_SHA256,
            },
            "v12_source_eval": {
                "path": V12_SOURCE_EVAL_RELATIVE,
                "sha256": V12_SOURCE_EVAL_SHA256,
            },
            "support_rules_sha256": SUPPORT_RULES_SHA256,
            "sampling_termination_topology": "inherited byte-for-byte from v12",
            "scorer_reference_retry_evidence": "inherited byte-for-byte from v12",
        },
        "support_rule": {
            "denominator": 64,
            "required_joint_successes": 58,
            "minimum_f1_range": 0.05,
            "definitions": {
                "scaffold": "both first-turn children",
                "eligibility": "pre-action commitment, exact restoration/replay, frozen topology",
                "joint": (
                    "eligible and at least one target has full K=4 deterministic-score "
                    "range >= 0.05"
                ),
            },
            "probability_curve": probabilities,
            "no_early_pass": True,
            "no_retry": True,
        },
        "scientific_protocol": {
            "status": "conditional_design_frozen_before_support_outcome",
            "arms": ["stock", "branch-global", "local"],
            "training_papers": 16,
            "heldout_papers": 32,
            "initialization": "byte-identical",
            "estimands": {
                "branch-global_vs_stock": "matched-token/call economics",
                "local_vs_branch-global": "matched-data/update credit",
            },
            "ledgers": ["tokens", "calls", "wall_time", "gpu_hours", "billing", "per_update"],
            "scorer": "deterministic QASPER span scorer; no learned judge",
            "replay": "structural event replay",
            "power_mde": _power_report(),
            "support_outcome_adaptation": False,
            "checkpoint_retention_preflight": {
                "required_before_science": True,
                "save_reload_reproduce_outputs": True,
                "retain_full_merged_model_locally": False,
                "retain_optimizer_or_cuda_caches_locally": False,
            },
            "artifact_retention": "compact_canonical_evidence_only",
        },
        "budget_hardware": {
            "support_cap_usd": 12.0,
            "science_reserve_cap_usd": 16.0,
            "teardown_contingency_min_usd": 2.0,
            "wallet_min_before_support_usd": 30.0,
            "resource": "one non-spot 2x48GB L40/L40S/RTX 6000 Ada",
            "hourly_cap_usd": 2.0,
            "forbidden": ["A100", "H100", "spot", "persistent_storage"],
            "wallet_pricing_check": "read-only at launch; fail closed if full reserve cannot fit",
        },
        "stop_semantics": [
            "scaffold_or_branchability_support_failure",
            "action_outside_closure_after_provider_activity",
            "evidence_hash_ledger_ambiguity",
            "insufficient_wallet_power_or_hardware",
            "unplanned_scientific_or_protocol_change",
            "completed_science_primary_result",
        ],
        "user_stop_required_for": [
            "scaffold_or_branchability_support_failure",
            "action_outside_closure_after_provider_activity",
            "evidence_hash_ledger_ambiguity",
            "insufficient_wallet_power_or_hardware",
            "unplanned_scientific_or_protocol_change",
            "completed_science_primary_result",
        ],
        "support_pass_transition": (
            "user_checkpoint_required_before_any_support_spend_or_science_transition"
        ),
        "authorization": PROTOCOL_AUTHORIZATION.copy(),
    }
    audit = {
        "schema_version": 1,
        "domain": "redco-stage-d1-support-v13-protocol-audit-v1",
        "dependency_stack": dependency_stack,
        "sampling_contract": sampling_contract,
        "candidate_sha256": sha256_bytes(candidate_raw),
        "composition_sha256": sha256_json(composition),
        "protocol_sha256": sha256_json(protocol),
        "candidate_read_wall": candidate["instrumentation"],
        "runtime": candidate["source"]["runtime"],
        "selection_evidence": selection_evidence.copy(),
        "ready_for_live_support": False,
        "live_activity_performed": False,
        "reason": (
            "CPU materialization and preregistration only; live provider/model activity "
            "requires separate authorization and launch-time checks."
        ),
    }
    return {
        CANDIDATE_RELATIVE: candidate_raw,
        COMPOSITION_RELATIVE: canonical_json_bytes(composition),
        PROTOCOL_RELATIVE: canonical_json_bytes(protocol),
        PROTOCOL_AUDIT_RELATIVE: canonical_json_bytes(audit),
    }


def rebuild_protocol_artifacts_from_existing(root: Path) -> dict[str, bytes]:
    """Rebuild the reviewed set from authenticated local bytes only.

    This is the source-free verification/rebuild path used after the one-time
    candidate materialization.  It never opens the Parquet artifact or invokes
    the source decoder; the independent reviewed hash map is the authority.
    """

    artifacts = {
        relative: read_authenticated(root, relative, expected_sha256)
        for relative, expected_sha256 in (
            support_contract.REVIEWED_PROTOCOL_ARTIFACT_SHA256.items()
        )
    }
    authenticate_protocol_artifact_bytes(root, artifacts)
    return artifacts

__all__ = [
    "ADDRESS_AUDIT_RELATIVE",
    "ADDRESS_AUDIT_SHA256",
    "AUTHENTICATED_PREDECESSOR_HASHES",
    "CANDIDATE_EXAMPLE_ID",
    "CANDIDATE_PAPER_ID",
    "CANDIDATE_QUESTION_INDEX",
    "CANDIDATE_RELATIVE",
    "CANDIDATE_ROW_SHA256",
    "CANDIDATE_SELECTION_ADDRESS_SHA256",
    "CANDIDATE_SOURCE_ORDINAL",
    "COLLECTION_PLAN_RELATIVE",
    "COLLECTION_PLAN_SHA256",
    "COMPOSITION_RELATIVE",
    "FROZEN_SUPPORT_RULES_RELATIVE",
    "FROZEN_SUPPORT_RULES_SHA256",
    "MASTER_SEED",
    "PROTOCOL_AUDIT_RELATIVE",
    "PROTOCOL_RELATIVE",
    "RETAINED_SUPPORT_RELATIVE",
    "RETAINED_SUPPORT_SHA256",
    "REVIEWED_PROTOCOL_ARTIFACT_SHA256",
    "SCIENTIFIC_NAMESPACE",
    "SELECTION_CLAIM_RELATIVE",
    "SELECTION_CLAIM_SHA256",
    "SELECTION_MANIFEST_RELATIVE",
    "SELECTION_MANIFEST_SHA256",
    "SELECTION_ORIGINAL_CLAIM_RELATIVE",
    "SELECTION_RECEIPT_RELATIVE",
    "SELECTION_RECEIPT_SHA256",
    "SOURCE_ARTIFACT_RELATIVE",
    "SOURCE_BYTES",
    "SOURCE_FIELDS",
    "SOURCE_LOGICAL_URL",
    "SOURCE_PATH",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "SOURCE_ROW_COUNT",
    "SOURCE_SCHEMA_SHA256",
    "SOURCE_SEMANTIC_COMMIT",
    "SOURCE_SHA256",
    "SUPPORTED_DATASETS",
    "SUPPORTED_PYARROW",
    "SUPPORTED_PYTHON",
    "SUPPORT_RULES_SHA256",
    "V6_MANIFEST_RELATIVE",
    "V6_MANIFEST_SHA256",
    "V12_ARCHIVE_RELATIVE",
    "V12_ARCHIVE_SHA256",
    "V12_EVIDENCE_MANIFEST_RELATIVE",
    "V12_EVIDENCE_MANIFEST_SHA256",
    "V12_FINALIZATION_AUDIT_RELATIVE",
    "V12_FINALIZATION_AUDIT_SHA256",
    "V12_PREREG_RELATIVE",
    "V12_PREREG_SHA256",
    "V12_PROTOCOL_RELATIVE",
    "V12_PROTOCOL_SHA256",
    "V12_SOURCE_EVAL_RELATIVE",
    "V12_SOURCE_EVAL_SHA256",
    "V12_TERMINAL_REPORT_RELATIVE",
    "V12_TERMINAL_REPORT_SHA256",
    "CandidateReadInstrumentation",
    "authenticate_upstream_evidence",
    "build_protocol_artifacts",
    "canonical_json_bytes",
    "check_protocol_artifacts",
    "load_parquet",
    "materialize_candidate",
    "read_authenticated",
    "rebuild_protocol_artifacts_from_existing",
    "require_supported_runtime",
    "runtime_payload",
    "sha256_bytes",
    "sha256_json",
    "source_contract",
]
