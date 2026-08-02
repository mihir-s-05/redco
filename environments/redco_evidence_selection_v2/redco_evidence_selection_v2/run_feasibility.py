"""Run grouped frozen-model Stage D0 feasibility rollouts on local inference."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from verifiers.v1.cli.output import append_episode, save_config
from verifiers.v1.clients import ModelContext, resolve_client
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.trace import EvalRunInfo
from verifiers.v1.utils.logging import setup_logging

from redco_evidence_selection_v2.seeding import derive_episode_seed

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_RLM_VERSION = "56218f33796ecbe465445bc43948886354fde196"
DEFAULT_DATASET_SHA256 = (
    "de84fda40c43fa7f977e063130f3f60fbcf05f625f947d941f3b6c0a80cbd347"
)


def build_config(args: argparse.Namespace) -> EvalConfig:
    harness: dict[str, Any] = {
        "id": "rlm",
        "version": args.rlm_version,
        "max_depth": 1,
        "runtime": {"type": "subprocess"},
        "forward_env": ["RLM_FORCE_TOOL_CHOICE_REQUIRED"],
    }
    bundle_names = (
        "rlm_archive",
        "rlm_archive_sha256",
        "rlm_uv_binary",
        "rlm_uv_binary_sha256",
        "rlm_uv_cache_archive",
        "rlm_uv_cache_archive_sha256",
        "rlm_uv_lock_sha256",
        "rlm_launcher",
        "rlm_launcher_sha256",
    )
    frozen_bundle = tuple(getattr(args, name, None) for name in bundle_names)
    if any(value is not None for value in frozen_bundle):
        if any(value is None for value in frozen_bundle):
            raise ValueError("the frozen RLM install bundle must be supplied together")
        harness.update(
            {
                "checkout_archive_path": str(frozen_bundle[0].resolve()),
                "checkout_archive_sha256": frozen_bundle[1],
                "checkout_uv_path": str(frozen_bundle[2].resolve()),
                "checkout_uv_sha256": frozen_bundle[3],
                "checkout_cache_archive_path": str(
                    frozen_bundle[4].resolve()
                ),
                "checkout_cache_archive_sha256": frozen_bundle[5],
                "checkout_uv_lock_sha256": frozen_bundle[6],
                "checkout_launcher_path": str(frozen_bundle[7].resolve()),
                "checkout_launcher_sha256": frozen_bundle[8],
            }
        )
    env = vf.SingleAgentEnvConfig.model_validate(
        {
            "id": "single-agent",
            "taskset": {
                "id": "redco-evidence-selection-v2",
                "dataset_path": args.dataset.resolve(),
                "dataset_sha256": args.dataset_sha256,
                "split": args.split,
                "prompt_profile": args.prompt_profile,
                "scaffold_prompt_path": (
                    args.scaffold_prompt.resolve()
                    if args.scaffold_prompt is not None
                    else None
                ),
                "scaffold_prompt_sha256": args.scaffold_prompt_sha256,
            },
            "agent": {
                "max_total_tokens": args.max_total_tokens,
                "timeout": {
                    "setup": args.setup_timeout,
                    "rollout": args.harness_timeout,
                    "finalize": 60,
                    "scoring": 60,
                },
                "harness": harness,
            },
        }
    )
    return EvalConfig(
        model=args.model,
        client={
            "type": "train",
            "base_url": args.base_url,
            "api_key_var": "VLLM_API_KEY",
            "renderer": {"name": "auto"},
            "renderer_model_name": args.renderer_model_name,
            "pool_size": 1,
        },
        # This placeholder never reaches a model call. Each RunSlot receives an
        # episode-addressed copy below; the exact mapping is persisted.
        sampling={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": 1,
            "max_tokens": args.max_completion_tokens,
        },
        env=env,
        num_tasks=args.num_tasks,
        num_rollouts=args.replicates,
        max_concurrent=1,
        rich=False,
        push=False,
        output_dir=args.output_dir,
    )


def _task_example_id(task: Any) -> str:
    value = getattr(task.data, "example_id", None)
    if not isinstance(value, str) or not value:
        raise ValueError("every feasibility task must expose example_id")
    return value


async def run_grouped(args: argparse.Namespace) -> int:
    config = build_config(args)
    env = vf.load_environment(config.env)
    tasks = env.taskset.select(config.num_tasks, config.shuffle)
    plan = [
        {
            "task_position": task_position,
            "task_index": task.data.idx,
            "example_id": _task_example_id(task),
            "replicate": replicate,
            "seed": derive_episode_seed(
                args.master_seed,
                _task_example_id(task),
                replicate,
            ),
        }
        for task_position, task in enumerate(tasks)
        for replicate in range(args.replicates)
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": config.model_dump(mode="json"),
                    "episode_seed_plan": plan,
                },
                sort_keys=True,
            )
        )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_config(config, args.output_dir)
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(1)
    client = resolve_client(config.client)
    records: list[dict[str, Any]] = []
    started = time.monotonic()

    async def persist(episode: Any) -> None:
        for trace in episode.traces:
            trace.stamp(EvalRunInfo(id=config.uuid))
        await append_episode(args.output_dir, episode, write_lock)

    try:
        async with env.serving():
            for task_position, task in enumerate(tasks):
                slots = env.slots(task, n=args.replicates)
                for replicate, slot in enumerate(slots):
                    seed = derive_episode_seed(
                        args.master_seed,
                        _task_example_id(task),
                        replicate,
                    )
                    sampling = config.sampling.model_copy(update={"seed": seed})
                    context = ModelContext(
                        client=client,
                        model=config.model,
                        sampling=sampling,
                    )
                    episode_started = time.monotonic()
                    episode = await env.run_slot(
                        slot,
                        context,
                        semaphore,
                        persist,
                    )
                    records.append(
                        {
                            "slot_id": (
                                f"{_task_example_id(task)}::"
                                f"replicate-{replicate}"
                            ),
                            "task_position": task_position,
                            "task_index": task.data.idx,
                            "example_id": _task_example_id(task),
                            "replicate": replicate,
                            "seed": seed,
                            "episode_id": episode.id,
                            "trace_ids": [trace.id for trace in episode.traces],
                            "ok": episode.ok,
                            "wall_seconds": time.monotonic() - episode_started,
                        }
                    )
    finally:
        await client.close()
        summary = {
            "schema_version": 1,
            "prompt_profile": args.prompt_profile,
            "master_seed": args.master_seed,
            "records": records,
            "total_wall_seconds": time.monotonic() - started,
        }
        (args.output_dir / "run-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--renderer-model-name",
        default=DEFAULT_MODEL,
        help=(
            "Canonical checkpoint ID used only to select the typed renderer; "
            "--model remains the exact served snapshot name."
        ),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-sha256", default=DEFAULT_DATASET_SHA256)
    parser.add_argument(
        "--split",
        choices=(
            "train",
            "validation",
            "fewshot_support",
            "power_audit",
            "science_train",
            "science_eval",
            "audit",
        ),
        default="train",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=(
            "natural",
            "forced_trace_fixture",
            "fewshot_scaffold_v2",
            "fewshot_fixture_v3",
            "fewshot_fixture_v4",
        ),
        default="natural",
    )
    parser.add_argument("--scaffold-prompt", type=Path)
    parser.add_argument("--scaffold-prompt-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument(
        "--master-seed",
        default="redco-stage-d0-qasper-feasibility-v1",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=768)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--rlm-version", default=DEFAULT_RLM_VERSION)
    parser.add_argument("--rlm-archive", type=Path)
    parser.add_argument("--rlm-archive-sha256")
    parser.add_argument("--rlm-uv-binary", type=Path)
    parser.add_argument("--rlm-uv-binary-sha256")
    parser.add_argument("--rlm-uv-cache-archive", type=Path)
    parser.add_argument("--rlm-uv-cache-archive-sha256")
    parser.add_argument("--rlm-uv-lock-sha256")
    parser.add_argument("--rlm-launcher", type=Path)
    parser.add_argument("--rlm-launcher-sha256")
    parser.add_argument("--setup-timeout", type=float, default=900)
    parser.add_argument("--harness-timeout", type=float, default=900)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        setup_logging("INFO", log_file=str(args.output_dir) + ".log")
    return asyncio.run(run_grouped(args))


if __name__ == "__main__":
    raise SystemExit(main())
