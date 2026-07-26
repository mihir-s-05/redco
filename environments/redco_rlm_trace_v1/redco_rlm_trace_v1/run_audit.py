"""Run the native RLM trace audit against a local prime-rl vLLM endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.utils.logging import setup_logging

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_RLM_VERSION = "56218f33796ecbe465445bc43948886354fde196"


def build_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        model=args.model,
        client={
            "type": "train",
            "base_url": args.base_url,
            "api_key_var": "VLLM_API_KEY",
            "renderer": {"name": "auto"},
            "renderer_model_name": args.model,
            "pool_size": 1,
        },
        sampling={
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.max_completion_tokens,
        },
        env={
            "taskset": {
                "id": "redco-rlm-trace-v1",
                "num_tasks": args.num_tasks,
            },
            "agent": {
                "max_total_tokens": args.max_total_tokens,
                "timeout": {
                    "setup": args.setup_timeout,
                    "rollout": args.harness_timeout,
                    "finalize": 60,
                    "scoring": 60,
                },
                "harness": {
                    "id": "rlm",
                    "version": args.rlm_version,
                    "max_depth": args.max_depth,
                    "runtime": {"type": "subprocess"},
                },
            },
        },
        num_tasks=args.num_tasks,
        num_rollouts=1,
        max_concurrent=1,
        rich=False,
        push=False,
        output_dir=args.output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--max-completion-tokens", type=int, default=768)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--rlm-version", default=DEFAULT_RLM_VERSION)
    parser.add_argument("--setup-timeout", type=float, default=900)
    parser.add_argument("--harness-timeout", type=float, default=600)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config(args)
    if args.dry_run:
        print(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", log_file=str(args.output_dir / "eval.log"))
    env = vf.load_environment(config.env)
    episodes = asyncio.run(run_eval(env, config))
    return 0 if episodes and all(episode.ok for episode in episodes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
