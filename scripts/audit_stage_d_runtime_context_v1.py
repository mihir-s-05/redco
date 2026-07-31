"""Assert recorded and one-target branch contexts fit the frozen 8192 limit."""

from __future__ import annotations

import argparse
from pathlib import Path

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload
from redco.integrations.verifiers_trace import audit_trace_file

MAX_MODEL_LEN = 8192
MAX_CANDIDATE_TOKENS = 512
MAX_CONTINUATION_TOKENS = 768
RENDER_SAFETY_TOKENS = 64


def audit(path: Path) -> dict[str, object]:
    trace = audit_trace_file(path)
    roots = [call for call in trace.calls if call.agent_depth == 0 and call.turn_index is not None]
    children = [call for call in trace.calls if call.agent_depth == 1]
    if not roots:
        raise ValueError("context audit requires a returning root")
    final_root = max(roots, key=lambda call: call.turn_index or 0)
    recorded_totals = [
        len(call.prompt_token_ids) + call.completion_tokens_reported for call in trace.calls
    ]
    target_branch_totals = [
        len(final_root.prompt_token_ids)
        + max(0, MAX_CANDIDATE_TOKENS - child.completion_tokens_reported)
        + MAX_CONTINUATION_TOKENS
        + RENDER_SAFETY_TOKENS
        for child in children
    ]
    maximum = max([*recorded_totals, *target_branch_totals])
    checks = {
        "two_to_four_children": 2 <= len(children) <= 4,
        "all_recorded_calls_fit": max(recorded_totals) <= MAX_MODEL_LEN,
        "all_one_target_branch_bounds_fit": (
            bool(target_branch_totals) and max(target_branch_totals) <= MAX_MODEL_LEN
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-runtime-context-v1",
            "maximum_model_length": MAX_MODEL_LEN,
            "maximum_observed_or_bounded_tokens": maximum,
            "recorded_call_totals": recorded_totals,
            "target_branch_totals": target_branch_totals,
            "checks": checks,
            "passes": all(checks.values()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.trace)
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(23)


if __name__ == "__main__":
    main()
