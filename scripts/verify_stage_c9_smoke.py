"""Verify the frozen two-epoch Stage-C9 integration smoke."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from redco.analysis.stage_c9_efficiency import _numeric_metrics
from redco.integrations.signed_subprocess import atomic_write_json, sign_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = _numeric_metrics(args.run_dir / "metrics.jsonl")
    required = (
        "optim/grad_norm",
        "redco_node_ratio/mean",
        "redco_node_clipped/mean",
        "redco_node_sampled_kl/mean",
        "redco_node_squared_log_ratio/mean",
    )
    for step in (1, 2):
        row = metrics.get(step, {})
        if not all(key in row and math.isfinite(row[key]) for key in required):
            raise ValueError(f"smoke step {step} lacks finite metrics")
    # The shared verifier expects six collections; the smoke has exactly one.
    first = (
        args.run_dir
        / "run_default"
        / "rollouts"
        / "step_1"
        / "train_rollouts.bin"
    )
    second = (
        args.run_dir
        / "run_default"
        / "rollouts"
        / "step_2"
        / "train_rollouts.bin"
    )
    from redco.analysis.stage_c9_efficiency import _examples_hash

    first_step, first_hash = _examples_hash(first)
    second_step, second_hash = _examples_hash(second)
    even_trace = second.parent / "train" / "all" / "traces.jsonl"
    passed = (
        first_step == 1
        and second_step == 2
        and first_hash == second_hash
        and not even_trace.exists()
    )
    if not passed:
        raise ValueError("two-epoch smoke reuse contract failed")
    atomic_write_json(
        args.output,
        sign_payload(
            {
                "schema_version": 1,
                "analysis": "stage-c9-two-epoch-integration-smoke",
                "status": "passed",
                "examples_sha256": first_hash,
                "transport_steps": [first_step, second_step],
                "no_rollout_between_updates": not even_trace.exists(),
                "metrics": {
                    str(step): {key: metrics[step][key] for key in required}
                    for step in (1, 2)
                },
            }
        ),
    )


if __name__ == "__main__":
    main()
