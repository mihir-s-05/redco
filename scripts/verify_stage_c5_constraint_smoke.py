"""Verify live Stage-C5 traces and packed trainer bytes under route constraints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import msgspec
from prime_rl.transport.types import TrainingBatch

from redco.analysis.stage_c3_power import ROUTES
from redco.analysis.stage_c5_constrained import (
    SEMANTICS,
    constrained_route_distribution,
)
from redco.analysis.stage_c5_smoke import verify_trace_file
from redco.integrations.signed_subprocess import atomic_write_json, sign_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--root-scores", type=Path, required=True)
    parser.add_argument("--expected-context-traces", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root_scores = json.loads(args.root_scores.read_text(encoding="utf-8"))
    trace_result = verify_trace_file(
        args.traces,
        root_scores,
        expected_context_traces=args.expected_context_traces,
    )
    batch = msgspec.msgpack.decode(args.batch.read_bytes(), type=TrainingBatch)
    details = root_scores["temperature_2"]["token_details"]
    rows = [details[route] for route in ROUTES]
    choice_index = next(
        index
        for index in range(min(map(len, rows)))
        if len({row[index]["token_id"] for row in rows}) > 1
    )
    token_to_route = {
        details[route][choice_index]["token_id"]: route for route in ROUTES
    }
    expected = constrained_route_distribution(root_scores)
    packed = []
    for example_index, example in enumerate(batch.examples):
        positions = [
            index
            for index, (token_id, trainable) in enumerate(
                zip(example.token_ids, example.mask, strict=True)
            )
            if trainable and token_id in token_to_route
        ]
        for position in positions:
            route = token_to_route[example.token_ids[position]]
            observed = float(example.logprobs[position])
            packed.append(
                {
                    "example_index": example_index,
                    "position": position,
                    "route": route,
                    "observed_logprob": observed,
                    "expected_logprob": math.log(expected[route]),
                    "absolute_error_nats": abs(observed - math.log(expected[route])),
                }
            )
    tolerance = trace_result["constraint_logprob_tolerance_nats"]
    batch_checks = {
        "packed_root_choice_count_matches_traces": (
            len(packed) == args.expected_context_traces
        ),
        "packed_choice_logprobs_are_finite": bool(packed)
        and all(math.isfinite(row["observed_logprob"]) for row in packed),
        "packed_choice_logprobs_match_exact_categorical": bool(packed)
        and all(row["absolute_error_nats"] <= tolerance for row in packed),
    }
    payload = sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-c5-constrained-interface-smoke",
            "status": (
                "passed"
                if trace_result["passed"] and all(batch_checks.values())
                else "failed"
            ),
            "constraint_semantics": SEMANTICS,
            "trace_verification": trace_result,
            "batch_checks": batch_checks,
            "packed_root_choices": packed,
        }
    )
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "signature": payload["signed_payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
