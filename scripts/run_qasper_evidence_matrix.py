"""Run five matched QASPER trajectory-LOO versus ReDCO seeds."""

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


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _load_matrix_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    expected = {
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
        raise ValueError("matrix config has the wrong schema")
    seeds = value["seeds"]
    if type(seeds) is not list or tuple(seeds) != SEEDS:
        raise ValueError("matrix requires the exact reviewed five-seed block")
    if value["pilot"] != "qasper-evidence-matrix-v1":
        raise ValueError("matrix config names the wrong pilot")
    if not 0 < float(value["cost_limit_usd"]) <= 3:
        raise ValueError("matrix cost limit is outside the reviewed bound")
    if not 0 < int(value["max_experiment_minutes"]) <= 60:
        raise ValueError("matrix experiment deadline is outside the reviewed bound")
    hourly_rate = float(value["max_hourly_rate_usd"])
    lifetime_minutes = int(value["max_pod_lifetime_minutes"])
    if not 0 < hourly_rate <= 2 or not 0 < lifetime_minutes <= 75:
        raise ValueError("matrix pod bounds are outside the reviewed limits")
    if hourly_rate * lifetime_minutes / 60 >= float(value["cost_limit_usd"]):
        raise ValueError("matrix pod lifetime leaves no teardown cost reserve")
    return value


def _authenticated_path(raw_path: str, raw_sha: str) -> Path:
    path = Path(raw_path)
    expected = require_sha256_hex(raw_sha, f"{raw_path} sha256")
    if not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
        raise ValueError(f"{raw_path} is absent or has the wrong digest")
    return path


def _paired_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    post_differences: list[int] = []
    change_differences: list[int] = []
    raw_counts: list[dict[str, Any]] = []
    for run in runs:
        by_arm = {arm["arm"]: arm for arm in run["arms"]}
        trajectory = by_arm["trajectory_loo"]
        redco = by_arm["redco"]
        trajectory_pre = int(trajectory["evaluation_before"]["exact_evidence"])
        trajectory_post = int(trajectory["evaluation_after"]["exact_evidence"])
        redco_pre = int(redco["evaluation_before"]["exact_evidence"])
        redco_post = int(redco["evaluation_after"]["exact_evidence"])
        post_difference = redco_post - trajectory_post
        change_difference = (redco_post - redco_pre) - (
            trajectory_post - trajectory_pre
        )
        post_differences.append(post_difference)
        change_differences.append(change_difference)
        raw_counts.append(
            {
                "redco_post": redco_post,
                "redco_pre": redco_pre,
                "seed": run["seed"],
                "trajectory_post": trajectory_post,
                "trajectory_pre": trajectory_pre,
            }
        )
    return {
        "inferential_unit": "seed",
        "n": len(runs),
        "paired_change_from_baseline": {
            "differences": change_differences,
            "mean": statistics.fmean(change_differences),
            "median": statistics.median(change_differences),
        },
        "paired_post_exact_evidence": {
            "differences": post_differences,
            "mean": statistics.fmean(post_differences),
            "median": statistics.median(post_differences),
        },
        "raw_exact_evidence_counts": raw_counts,
        "statistical_claim": "descriptive repeatability pilot; no significance claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qasper-evidence-matrix-v1.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/qasper-evidence-matrix-v1"))
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    matrix = _load_matrix_config(arguments.config)
    base_path = _authenticated_path(matrix["base_config"], matrix["base_config_sha256"])
    data_path = _authenticated_path(matrix["data"], matrix["data_sha256"])
    base = _load_config(base_path)
    tasks, budget = load_pilot_dataset(data_path)
    if budget.train_tasks != 24 or budget.eval_tasks != 24:
        raise ValueError("matrix requires a 24/24 train/evaluation split")
    if arguments.check:
        print(
            json.dumps(
                {
                    "eval_tasks": budget.eval_tasks,
                    "seeds": matrix["seeds"],
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
        raise RuntimeError("the matrix requires exactly one visible CUDA GPU")
    started = time.monotonic()
    deadline = started + int(matrix["max_experiment_minutes"]) * 60
    arguments.output.mkdir(parents=True, exist_ok=False)
    runs: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(matrix["seeds"]):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"matrix deadline reached before seed {seed}")
        config = dict(base)
        config["seed"] = seed
        random.seed(seed)
        arm_order = (
            ("trajectory_loo", "redco")
            if seed_index % 2 == 0
            else ("redco", "trajectory_loo")
        )
        arms: list[dict[str, Any]] = []
        for arm_index, arm in enumerate(arm_order):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"matrix deadline reached before seed {seed} {arm}")
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
        by_arm = {arm["arm"]: arm for arm in arms}
        trajectory = by_arm["trajectory_loo"]
        redco = by_arm["redco"]
        if trajectory["evaluation_before"] != redco["evaluation_before"]:
            raise RuntimeError(f"seed {seed} arms have different baseline evaluations")
        if trajectory["initial_adapter_sha256"] != redco["initial_adapter_sha256"]:
            raise RuntimeError(f"seed {seed} arms have different initial LoRA tensors")
        runs.append({"arms": arms, "seed": seed})
        gc.collect()
        torch.cuda.empty_cache()
    elapsed = time.monotonic() - started
    if elapsed > int(matrix["max_experiment_minutes"]) * 60:
        raise RuntimeError("matrix exceeded its reviewed runtime bound")
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
        "matrix": matrix,
        "paired_summary": _paired_summary(runs),
        "metric_interpretation": {
            "arm_specific_sampled_reward": (
                "rollout diagnostic only; sampling differs between arms"
            ),
            "gradient_norm": "optimizer diagnostic only",
        },
        "runs": runs,
        "schema_version": 1,
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
