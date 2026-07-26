"""Audit native verifiers.v1 traces for exact ReDCO replay inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json
from redco.integrations.verifiers_trace import audit_trace_file, build_policy_cache


def evaluate_trace_gate(
    input_path: Path,
    *,
    require_recursive: bool,
) -> dict[str, Any]:
    report = audit_trace_file(input_path)
    if report.ready_for_exact_key_replay:
        build_policy_cache(report.calls)
    checks = {
        "all_episodes_successful": (
            report.episode_count > 0
            and report.successful_episode_count == report.episode_count
            and report.episode_error_count == 0
        ),
        "all_traces_successful": (
            report.trace_count > 0
            and report.successful_trace_count == report.trace_count
            and report.trace_error_count == 0
        ),
        "all_calls_linked": (
            report.model_call_count > 0
            and report.linked_call_count == report.model_call_count
        ),
        "no_failed_model_calls": report.failed_model_call_count == 0,
        "exact_prompt_action_coverage": (
            report.exact_prompt_action_coverage == 1.0
        ),
        "exact_key_replay_ready": report.ready_for_exact_key_replay,
        "usage_coverage": report.usage_coverage == 1.0,
        "native_task_audit_present": (
            report.trace_count > 0
            and report.native_audit_trace_count == report.trace_count
        ),
        "recursive_model_call_observed": (
            report.has_recursive_model_calls or not require_recursive
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "require_recursive": require_recursive,
        "checks": checks,
        "passed": all(checks.values()),
        "trace_audit": report.as_dict(),
    }
    payload["report_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    payload = evaluate_trace_gate(
        args.input,
        require_recursive=args.require_recursive,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(payload) + b"\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
