"""Run one Stage D CRN branch group over every precommitted child target."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import sign_payload
from redco.integrations.verifiers_trace import (
    audit_trace_file,
    load_trace_records,
)

from .empirical_branch_replay import (
    DeterministicReplayIneligibility,
    TokenInferenceClient,
    run_empirical_replay,
)
from .stage_d_all_child_support import verify_canonical_precommit


def run_group(
    *,
    trace_path: Path,
    precommit: dict[str, Any],
    client: TokenInferenceClient,
    master_seed: str,
    temperature: float,
    candidate_max_tokens: int,
    continuation_max_tokens: int,
) -> tuple[dict[str, Any], bool]:
    verify_canonical_precommit(trace_path, precommit)
    chain = {
        "source_trace_sha256": precommit["source_trace_sha256"],
        "precommit_signed_payload_sha256": precommit["signed_payload_sha256"],
        "candidate_set_sha256": precommit["candidate_set_sha256"],
        "target_node_ids": [
            candidate["structural_event_address"] for candidate in precommit["candidates"]
        ],
        "decision_unit_weights": [
            {
                "target_node_id": candidate["structural_event_address"],
                "weight": candidate["decision_unit_weight"],
            }
            for candidate in precommit["candidates"]
        ],
        "master_seed_sha256": hashlib.sha256(master_seed.encode("utf-8")).hexdigest(),
    }
    traces = load_trace_records(trace_path)
    audit = audit_trace_file(trace_path)
    if len(traces) != 1:
        raise ValueError("all-child branch input must contain exactly one trace")
    if not audit.calls:
        raise ValueError("all-child branch trace has no parseable policy calls")
    try:
        report = run_empirical_replay(
            trace_path=trace_path,
            client=client,
            alternatives_per_target=3,
            maximum_targets=None,
            master_seed=master_seed,
            temperature=temperature,
            candidate_max_tokens=candidate_max_tokens,
            continuation_max_tokens=continuation_max_tokens,
            minimum_distinct_candidate_fraction=1 / 3,
        )
    except DeterministicReplayIneligibility as error:
        return (
            sign_payload(
                {
                    "schema_version": 1,
                    "analysis": "stage-d-all-child-crn-branch-group",
                    "status": "deterministic_ineligible",
                    "trace_id": traces[0].get("id"),
                    **chain,
                    "reason_type": type(error).__name__,
                    "reason": str(error),
                }
            ),
            False,
        )
    payload = report.signed_dict()
    if payload.get("source_sha256") != chain["source_trace_sha256"]:
        raise ValueError("trace changed after precommit validation")
    observed_nodes = {
        str(row["target_node_id"]) for row in payload.get("regenerated_originals", [])
    }
    if observed_nodes != set(chain["target_node_ids"]):
        raise ValueError("branch runtime did not cover the committed target set")
    payload.update(
        {
            "analysis": "stage-d-all-child-crn-branch-group",
            **chain,
        }
    )
    return sign_payload(payload), True
