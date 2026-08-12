"""Run the matched ten-call QASPER credit-allocation sweep."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from run_qasper_evidence_pilot import _load_config, _train_arm

from redco.contracts import canonical_json
from redco.experiments.qasper_evidence import load_pilot_dataset
from redco.integrity import require_sha256_hex, sha256_bytes

SEEDS = (20260812, 20260813, 20260814, 20260815, 20260816)
ARMS = ("trajectory_loo", "branch_4_2", "branch_3_4", "branch_2_6")


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _load_sweep_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    expected = {
        "arms",
        "base_config",
        "base_config_sha256",
        "cost_limit_usd",
        "data",
        "data_sha256",
        "max_experiment_minutes",
        "max_hourly_rate_usd",
        "max_pod_lifetime_minutes",
        "pilot",
        "schema_version",
        "seeds",
    }
    if type(value) is not dict or set(value) != expected or value["schema_version"] != 1:
        raise ValueError("sweep config has the wrong schema")
    if type(value["seeds"]) is not list or tuple(value["seeds"]) != SEEDS:
        raise ValueError("sweep requires the exact reviewed five-seed block")
    if type(value["arms"]) is not list or tuple(value["arms"]) != ARMS:
        raise ValueError("sweep requires the exact reviewed allocation frontier")
    if value["pilot"] != "qasper-allocation-sweep-v1":
        raise ValueError("sweep config names the wrong pilot")
    if not 0 < float(value["cost_limit_usd"]) <= 3:
        raise ValueError("sweep cost limit is outside the reviewed bound")
    if not 0 < int(value["max_experiment_minutes"]) <= 60:
        raise ValueError("sweep deadline is outside the reviewed bound")
    hourly_rate = float(value["max_hourly_rate_usd"])
    lifetime_minutes = int(value["max_pod_lifetime_minutes"])
    if not 0 < hourly_rate <= 2 or not 0 < lifetime_minutes <= 75:
        raise ValueError("sweep pod bounds are outside the reviewed limits")
    if hourly_rate * lifetime_minutes / 60 >= float(value["cost_limit_usd"]):
        raise ValueError("sweep pod lifetime leaves no teardown cost reserve")
    return value


def _authenticated_path(raw_path: str, raw_sha: str) -> Path:
    path = Path(raw_path)
    expected = require_sha256_hex(raw_sha, f"{raw_path} sha256")
    if not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
        raise ValueError(f"{raw_path} is absent or has the wrong digest")
    return path


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes: dict[str, Any] = {}
    for arm in ARMS:
        arm_runs = [
            next(item for item in run["arms"] if item["arm"] == arm) for run in runs
        ]
        post_exact = [int(item["evaluation_after"]["exact_evidence"]) for item in arm_runs]
        post_paragraph = [int(item["evaluation_after"]["paragraph"]) for item in arm_runs]
        conditional = [
            float(item["evaluation_after"]["conditional_span_accuracy"])
            for item in arm_runs
        ]
        outcomes[arm] = {
            "mean_conditional_span_accuracy": statistics.fmean(conditional),
            "mean_post_exact": statistics.fmean(post_exact),
            "mean_post_paragraph": statistics.fmean(post_paragraph),
            "post_conditional_span_accuracy_by_seed": conditional,
            "post_exact_by_seed": post_exact,
            "post_paragraph_by_seed": post_paragraph,
        }
    return {
        "inferential_unit": "seed",
        "n": len(runs),
        "outcomes": outcomes,
        "primary_outcomes": [
            "paragraph_accuracy",
            "conditional_span_accuracy_given_correct_paragraph",
            "exact_evidence_accuracy",
        ],
        "statistical_claim": "descriptive allocation-frontier pilot",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qasper-allocation-sweep-v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/qasper-allocation-sweep-v1"),
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    sweep = _load_sweep_config(arguments.config)
    base_path = _authenticated_path(sweep["base_config"], sweep["base_config_sha256"])
    data_path = _authenticated_path(sweep["data"], sweep["data_sha256"])
    base = _load_config(base_path)
    tasks, budget = load_pilot_dataset(data_path)
    if budget.train_tasks != 24 or budget.eval_tasks != 96:
        raise ValueError("sweep requires a 24/96 train/evaluation split")
    if arguments.check:
        print(
            json.dumps(
                {
                    "arms": sweep["arms"],
                    "eval_tasks": budget.eval_tasks,
                    "seeds": sweep["seeds"],
                    "train_tasks": budget.train_tasks,
                },
                sort_keys=True,
            )
        )
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import torch
    import transformers

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the sweep requires exactly one visible CUDA GPU")
    started = time.monotonic()
    deadline = started + int(sweep["max_experiment_minutes"]) * 60
    arguments.output.mkdir(parents=True, exist_ok=False)
    runs: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(sweep["seeds"]):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"sweep deadline reached before seed {seed}")
        config = dict(base)
        config["seed"] = seed
        random.seed(seed)
        rotation = seed_index % len(ARMS)
        arm_order = (*ARMS[rotation:], *ARMS[:rotation])
        arms: list[dict[str, Any]] = []
        for arm_index, arm in enumerate(arm_order):
            reference = arms[0] if arm_index else None
            arms.append(
                _train_arm(
                    torch,
                    transformers,
                    config,
                    tasks,
                    arm,
                    None,
                    budget,
                    deadline,
                    None if reference is None else reference["initial_adapter_sha256"],
                    None if reference is None else reference["evaluation_before"],
                )
            )
        if len({item["initial_adapter_sha256"] for item in arms}) != 1:
            raise RuntimeError(f"seed {seed} arms have different initial LoRA tensors")
        if len({canonical_json(item["evaluation_before"]) for item in arms}) != 1:
            raise RuntimeError(f"seed {seed} arms have different baseline evaluations")
        runs.append({"arms": arms, "seed": seed})
        gc.collect()
        torch.cuda.empty_cache()
    elapsed = time.monotonic() - started
    if elapsed > int(sweep["max_experiment_minutes"]) * 60:
        raise RuntimeError("sweep exceeded its reviewed runtime bound")
    payload = {
        "base_config": base,
        "config_sha256": sha256_bytes(arguments.config.read_bytes()),
        "data_sha256": sha256_bytes(data_path.read_bytes()),
        "elapsed_seconds": elapsed,
        "environment": {
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "git_commit": _git_head(),
        "metric_interpretation": {
            "arm_specific_sampled_reward": "arm-specific rollout diagnostic only",
            "gradient_norm": "optimizer diagnostic only",
        },
        "runs": runs,
        "schema_version": 1,
        "summary": _summary(runs),
        "sweep": sweep,
    }
    report = {
        "payload": payload,
        "payload_sha256": sha256_bytes(canonical_json(payload)),
        "schema_version": 1,
    }
    (arguments.output / "report.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({"payload_sha256": report["payload_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
