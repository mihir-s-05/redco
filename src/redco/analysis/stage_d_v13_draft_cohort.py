"""Outcome-independent successor-cohort preparation for the v13 draft."""

from __future__ import annotations

import json
from typing import Any

from redco.analysis.stage_d_v13_draft import sha256_json
from redco.analysis.stage_d_v13_draft_contract import (
    AUTHENTICATED_RESUME_RECEIPT_ORDINAL,
    OBSERVED_EXAMPLE_ID,
    OBSERVED_SEED,
)


def _draft_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    return {"draft_unfrozen": True, "launch_authorized": False, **payload}


def prepare_successor(
    source_lines: list[bytes],
    collection_plan: dict[str, Any],
    successor_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Retire the observed row and leave the authenticated reserve unresolved.

    The v6 snapshot is an immutable exclusion/input witness.  It is not a
    source-order continuation, so this function intentionally never derives a
    replacement ordinal, row, seed, or address from it.
    """

    if len(source_lines) != 112:
        raise ValueError("frozen v6 successor does not contain 112 canonical JSONL rows")
    rows = [json.loads(line) for line in source_lines]
    observed_row = rows[0]
    if observed_row["example_id"] != OBSERVED_EXAMPLE_ID:
        raise ValueError("the observed v12 unit is not the first support row")
    if observed_row["split"] != "successor_support":
        raise ValueError("the observed v12 unit is not a support row")

    retained_lines = source_lines[1:]
    retained_rows = rows[1:]
    if len(retained_rows) != 111:
        raise ValueError("retained-only draft successor row count differs")

    slots = collection_plan["slots"]
    observed_slot = slots[0]
    if observed_slot["example_id"] != OBSERVED_EXAMPLE_ID:
        raise ValueError("observed v12 collection address differs")
    if observed_slot["seed"] != OBSERVED_SEED:
        raise ValueError("observed v12 collection seed differs")
    preserved_slots = slots[1:]
    if len(preserved_slots) != 63:
        raise ValueError("v12 collection plan did not yield 63 untouched slots")

    unresolved_candidate = {
        "source_ordinal": None,
        "example_id": None,
        "paper_id": None,
        "seed": None,
        "address": None,
        "row": None,
    }
    selection_rule = {
        "source_dataset": "allenai/qasper",
        "source_revision": successor_manifest["source_revision"],
        "converted_parquet_revision": successor_manifest["converted_parquet_revision"],
        "source_order": True,
        "resume_after_authenticated_receipt_ordinal": AUTHENTICATED_RESUME_RECEIPT_ORDINAL,
        "selection_status": "pending_authenticated_scan",
        "maximum_paper_characters": successor_manifest["selection"]["maximum_paper_characters"],
        "minimum_span_characters": successor_manifest["selection"]["minimum_span_characters"],
        "one_question_per_paper": successor_manifest["selection"]["one_question_per_paper"],
    }
    replacement = {
        "selection_status": "blocked_pending_authenticated_source_scan",
        "selection_rule": selection_rule,
        "candidate": None,
        "unresolved_candidate": unresolved_candidate,
        "address_derivation_domain": "redco-stage-d1-support-v13-fresh-reserve-address-v1",
        "selection_is_outcome_independent": True,
    }

    dataset_manifest = _draft_envelope(
        {
            "schema_version": 2,
            "domain": "redco-stage-d-qasper-support-successor-v7-draft",
            "status": "blocked_unmaterialized_retained_rows_only",
            "dataset": "allenai/qasper",
            "source_revision": successor_manifest["source_revision"],
            "converted_parquet_revision": successor_manifest["converted_parquet_revision"],
            "input_v6_path": "datasets/stage-d/qasper-support-successor-v6.jsonl",
            "selection": selection_rule,
            "output": {
                "path": "datasets/stage-d/qasper-support-successor-v7-draft-retained-only.jsonl",
                "bytes": len(b"".join(retained_lines)),
                "rows": len(retained_rows),
                "required_rows": 112,
            },
            "retirement": {
                "removed_first_row": OBSERVED_EXAMPLE_ID,
                "removed_row_bytes_preserved_in_v12": True,
                "preserved_untouched_rows": 63,
                "inherited_science_rows": 48,
            },
            "replacement": replacement,
            "checks": {
                "retained_rows_are_byte_identical_to_v6_after_observed_first_row": True,
                "observed_row_removed_exactly_once": sum(
                    row["example_id"] == OBSERVED_EXAMPLE_ID for row in retained_rows
                )
                == 0,
                "no_unverified_reserve_inserted": True,
                "candidate_materialized": False,
                "launch_eligibility": False,
            },
        }
    )

    collection_draft = _draft_envelope(
        {
            "schema_version": 2,
            "domain": "redco-stage-d-source-collection-plan-v13-draft",
            "status": "blocked_pending_fresh_reserve_address",
            "frozen_v12_plan_sha256": successor_manifest["collection_plan"]["sha256"],
            "required_slot_count": 64,
            "materialized_slot_count": len(preserved_slots),
            "preserved_slots": preserved_slots,
            "retired_observed_unit": {
                "example_id": OBSERVED_EXAMPLE_ID,
                "seed": OBSERVED_SEED,
                "slot": observed_slot,
            },
            "replacement": replacement,
            "integrity": {
                "preserved_slot_bytes_sha256": sha256_json({"slots": preserved_slots}),
                "retired_slot_not_reused": True,
                "scientific_group_namespace": collection_plan.get("scientific_group_namespace"),
            },
        }
    )

    reserve_receipt = _draft_envelope(
        {
            "schema_version": 2,
            "domain": "redco-stage-d1-support-v13-reserve-selection-receipt-v2",
            "status": "blocked_authenticated_source_scan_unavailable",
            "selection_rule": selection_rule,
            "last_authenticated_selection": successor_manifest["successor"]["selection_receipt"],
            "forbidden_history": {
                "old_snapshot_sha256": successor_manifest["old_snapshot"]["sha256"],
                "prior_successor_sha256": successor_manifest["prior_extension"]["sha256"],
                "historically_retired_paper_ids": successor_manifest["successor"][
                    "historically_retired_paper_ids"
                ],
                "exclusion_hashes": successor_manifest["successor"]["exclusion_hashes"],
            },
            "candidate": None,
            "unresolved_candidate": unresolved_candidate,
            "reason": (
                "Authenticated source-order continuation after receipt ordinal 179 is not "
                "available locally; the v6 snapshot is an exclusion input, not a source-order "
                "substitute, so no candidate was selected or guessed."
            ),
            "required_before_freeze": [
                "materialize the pinned QASPER source order offline",
                "resume the authenticated scan after receipt ordinal 179",
                "run the existing successor builder and address audit",
                "record candidate row/source/reference hashes",
                "derive a fresh reserve address",
                "rerun the cumulative non-overlap audit",
            ],
        }
    )

    return {
        "retained_lines": b"".join(retained_lines),
        "retained_rows": retained_rows,
        "preserved_slots": preserved_slots,
        "observed_slot": observed_slot,
        "dataset_manifest": dataset_manifest,
        "collection_draft": collection_draft,
        "reserve_receipt": reserve_receipt,
    }


__all__ = ["prepare_successor"]
