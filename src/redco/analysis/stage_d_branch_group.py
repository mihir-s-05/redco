"""Run one Stage D CRN branch group without aborting on valid negatives."""

from __future__ import annotations

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


def run_group(
    *,
    trace_path: Path,
    client: TokenInferenceClient,
    master_seed: str,
    temperature: float,
    candidate_max_tokens: int,
    continuation_max_tokens: int,
) -> tuple[dict[str, Any], bool]:
    # Corrupt or non-singleton inputs are infrastructure failures and remain
    # exceptions. They are not converted into scientific negatives.
    traces = load_trace_records(trace_path)
    audit = audit_trace_file(trace_path)
    if len(traces) != 1:
        raise ValueError("branch-group input must contain exactly one trace")
    if not audit.calls:
        raise ValueError("branch-group trace has no parseable policy calls")
    try:
        report = run_empirical_replay(
            trace_path=trace_path,
            client=client,
            alternatives_per_target=3,
            maximum_targets=1,
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
                    "analysis": "stage-d-crn-branch-group",
                    "status": "deterministic_ineligible",
                    "trace_id": traces[0].get("id"),
                    "reason_type": type(error).__name__,
                    "reason": str(error),
                }
            ),
            False,
        )
    return sign_payload(report.signed_dict()), True
