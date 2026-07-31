from __future__ import annotations

import argparse
from pathlib import Path

from redco.analysis.empirical_branch_replay import TokenInferenceClient
from redco.analysis.stage_d_branch_group import run_group
from redco.integrations.signed_subprocess import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--master-seed", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--candidate-max-tokens", type=int, default=512)
    parser.add_argument("--continuation-max-tokens", type=int, default=768)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    client = TokenInferenceClient(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )
    report, _ = run_group(
        trace_path=args.trace,
        client=client,
        master_seed=args.master_seed,
        temperature=args.temperature,
        candidate_max_tokens=args.candidate_max_tokens,
        continuation_max_tokens=args.continuation_max_tokens,
    )
    atomic_write_json(args.output, report)


if __name__ == "__main__":
    main()
