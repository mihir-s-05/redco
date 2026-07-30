from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prime_rl.orchestrator.algo.routing import stamp_advantages
from prime_rl.orchestrator.envs import ROLLOUT_TYPE
from prime_rl.orchestrator.trajectories import trace_to_samples

from redco.analysis.stage_d_trace_contract import audit_rlm_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.trace_jsonl.read_bytes()
    first = next(line for line in raw.decode("utf-8").splitlines() if line.strip())
    episode = json.loads(first)
    if len(episode["traces"]) != 1:
        raise ValueError("contract fixture must contain exactly one trace")
    trace_payload = episode["traces"][0]
    contract = audit_rlm_trace(trace_payload)

    rollout = ROLLOUT_TYPE.model_validate(trace_payload)
    samples = trace_to_samples(rollout, env_name="stage-d0-contract")
    rollout.samples = samples
    rollout.assign_advantages(0.5)
    stamp_advantages(rollout)

    trainable_counts = [sum(sample.mask) for sample in samples]
    advantage_counts = [
        sum(value != 0.0 for value in (sample.advantages or []))
        for sample in samples
    ]
    all_sampled_tokens_routed_once = (
        sum(trainable_counts) == rollout.num_output_tokens
        and trainable_counts == advantage_counts
    )
    result = {
        "schema_version": 1,
        "source": {
            "path": args.trace_jsonl.as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "structural_contract": contract.to_dict(),
        "prime_trace_to_samples": {
            "branches": len(rollout.branches),
            "samples": len(samples),
            "sample_token_lengths": [len(sample.token_ids) for sample in samples],
            "trainable_tokens_per_sample": trainable_counts,
            "advantaged_tokens_per_sample": advantage_counts,
            "trace_output_tokens": rollout.num_output_tokens,
            "all_sampled_tokens_routed_once": all_sampled_tokens_routed_once,
            "passes": (
                len(samples) >= 3
                and all(count > 0 for count in trainable_counts)
                and all_sampled_tokens_routed_once
            ),
        },
        "decision": {
            "cpu_trainable_trace_contract": (
                "pass"
                if contract.trace_contract_passes and all_sampled_tokens_routed_once
                else "fail"
            ),
            "stage_d_science_ready": contract.stage_d_science_ready,
            "remaining_blocker": (
                None
                if contract.checkpoint_stamped
                else "policy checkpoint/version must be demonstrated on a Prime training trace"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
