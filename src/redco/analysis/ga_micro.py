"""Generate, preregister, and evaluate the paired Stage-A GA-micro campaign."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json

PILOT_SEEDS = (2101, 2102)
CONFIRM_SEEDS = (3101, 3102, 3103, 3104)
MARGIN_MULTIPLIER = 3.0
PAIRED_90_PERCENT_T_CRITICAL_DF3 = 2.353
RESOLUTION_FLOORS = {
    "reward": 1e-12,
    "input_tokens": 1e-12,
    "output_tokens": 1e-12,
    "loss": 1e-12,
    "entropy": 1e-12,
    "mismatch_kl": 1e-12,
    "grad_norm": 1e-12,
}


@dataclass(frozen=True, slots=True)
class RunMetrics:
    reward: float
    input_tokens: float
    output_tokens: float
    loss: float
    entropy: float
    mismatch_kl: float
    grad_norm: float
    error_rate: float
    truncation_rate: float

    def primary(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in asdict(self).items()
            if key not in {"error_rate", "truncation_rate"}
        }


def generate_configs(root: Path) -> tuple[Path, ...]:
    specs = [
        *(
            (f"pilot-stock-s{seed}-{replicate}", seed, "grpo")
            for seed in PILOT_SEEDS
            for replicate in ("a", "b")
        ),
        *(
            (f"confirm-{arm}-s{seed}", seed, "grpo" if arm == "stock" else "redco_noop")
            for seed in CONFIRM_SEEDS
            for arm in ("stock", "redco")
        ),
    ]
    config_root = root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, seed, algorithm in specs:
        path = config_root / f"{name}.toml"
        path.write_text(_config_text(name, seed, algorithm), encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def preregister(root: Path, output: Path) -> dict[str, Any]:
    pairs = [
        (
            read_run_metrics(root / f"pilot-stock-s{seed}-a"),
            read_run_metrics(root / f"pilot-stock-s{seed}-b"),
        )
        for seed in PILOT_SEEDS
    ]
    pair_differences = {
        metric: [
            first.primary()[metric] - second.primary()[metric]
            for first, second in pairs
        ]
        for metric in RESOLUTION_FLOORS
    }
    margins = {
        metric: max(
            MARGIN_MULTIPLIER * max(abs(value) for value in differences),
            RESOLUTION_FLOORS[metric],
        )
        for metric, differences in pair_differences.items()
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "frozen_before_confirmatory_runs",
        "pilot_seeds": list(PILOT_SEEDS),
        "pilot_replicates_per_seed": 2,
        "confirmatory_seeds": list(CONFIRM_SEEDS),
        "confirmatory_runs_per_arm": len(CONFIRM_SEEDS),
        "paired": True,
        "confidence_interval": "two-sided 90% paired mean-difference interval",
        "critical_value": PAIRED_90_PERCENT_T_CRITICAL_DF3,
        "margin_rule": "max(3 * maximum absolute stock-stock paired difference, resolution floor)",
        "resolution_floors": RESOLUTION_FLOORS,
        "pilot_pair_differences": pair_differences,
        "equivalence_margins": margins,
        "mandatory_conditions": {
            "error_rate": 0.0,
            "truncation_rate": 0.0,
            "completed_optimizer_steps": 1,
        },
        "excluded_system_metrics": [
            "throughput",
            "step_time",
            "peak_memory",
        ],
        "seed_control": {
            "trainer_torch_seed": "REDCO_RUN_SEED",
            "orchestrator_task_seed": "REDCO_RUN_SEED",
            "inference_engine_seed": "inference.seed",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(payload) + b"\n")
    return payload


def evaluate(root: Path, preregistration: Path, output: Path) -> dict[str, Any]:
    registration = json.loads(preregistration.read_bytes())
    margins: dict[str, float] = registration["equivalence_margins"]
    stock = [
        read_run_metrics(root / f"confirm-stock-s{seed}") for seed in CONFIRM_SEEDS
    ]
    redco = [
        read_run_metrics(root / f"confirm-redco-s{seed}") for seed in CONFIRM_SEEDS
    ]
    metric_results: dict[str, dict[str, float | bool | list[float]]] = {}
    for metric, margin in margins.items():
        differences = [
            redco_run.primary()[metric] - stock_run.primary()[metric]
            for stock_run, redco_run in zip(stock, redco, strict=True)
        ]
        mean_difference = statistics.fmean(differences)
        standard_deviation = statistics.stdev(differences)
        half_width = (
            PAIRED_90_PERCENT_T_CRITICAL_DF3
            * standard_deviation
            / math.sqrt(len(differences))
        )
        bound = abs(mean_difference) + half_width
        metric_results[metric] = {
            "paired_differences": differences,
            "mean_difference": mean_difference,
            "standard_deviation": standard_deviation,
            "ci_half_width": half_width,
            "absolute_ci_bound": bound,
            "margin": margin,
            "equivalent": bound <= margin,
        }
    mandatory_passed = all(
        run.error_rate == 0.0 and run.truncation_rate == 0.0
        for run in (*stock, *redco)
    )
    passed = mandatory_passed and all(
        bool(result["equivalent"]) for result in metric_results.values()
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "preregistration": preregistration.as_posix(),
        "confirmatory_seeds": list(CONFIRM_SEEDS),
        "stock_runs": [asdict(run) for run in stock],
        "redco_noop_runs": [asdict(run) for run in redco],
        "metric_results": metric_results,
        "mandatory_conditions_passed": mandatory_passed,
        "passed_ga_micro": passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(payload) + b"\n")
    return payload


def read_run_metrics(run: Path) -> RunMetrics:
    trainer_rows = _read_jsonl(run / "metrics.jsonl")
    trainer: dict[str, Any] = {}
    for row in trainer_rows:
        trainer.update(row)
    orchestrator_rows = _read_jsonl(run / "run_default" / "metrics.jsonl")
    if not orchestrator_rows:
        raise ValueError(f"no orchestrator metrics in {run}")
    orchestrator = orchestrator_rows[-1]
    return RunMetrics(
        reward=float(orchestrator["train/agg/effective/reward/mean"]),
        input_tokens=float(orchestrator["progress/input_tokens"]),
        output_tokens=float(orchestrator["progress/output_tokens"]),
        loss=float(trainer["loss/mean"]),
        entropy=float(trainer["entropy/all/mean"]),
        mismatch_kl=float(trainer["mismatch_kl/all/mean"]),
        grad_norm=float(trainer["optim/grad_norm"]),
        error_rate=float(orchestrator["train/agg/all/has_error/mean"]),
        truncation_rate=float(orchestrator["train/agg/all/is_truncated/mean"]),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _config_text(name: str, seed: int, algorithm: str) -> str:
    algo_block = (
        ""
        if algorithm == "grpo"
        else '\n[orchestrator.algo]\ntype = "redco_noop"\n'
    )
    return f"""# Generated GA-micro one-step run. Do not edit after launch.
output_dir = "runs/stage-a/ga-micro/{name}"
clean_output_dir = true
max_steps = 1
seq_len = 1024

[env_vars]
REDCO_RUN_SEED = "{seed}"

[deployment]
type = "single_node"
gpus_per_node = 2
num_train_gpus = 1
num_infer_gpus = 1

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"

[file_monitor]
filename = "metrics.jsonl"

[trainer.model.ac]
freq = 1

[trainer.model.lora]
rank = 8
alpha = 16

[trainer.optim]
lr = 1e-5

[orchestrator]
batch_size = 4
group_size = 2
max_off_policy_steps = 0
{algo_block}
[orchestrator.train.sampling]
max_completion_tokens = 192

[[orchestrator.train.env]]
name = "alphabet-sort"
env.taskset = {{ id = "alphabet-sort-v1", min_turns = 2, max_turns = 2, \
min_names_per_turn = 1, max_names_per_turn = 2, \
task = {{ similarity_power = 4, power_per_turn = false }} }}
env.agent.harness = {{ id = "null", runtime = {{ type = "subprocess" }} }}

[inference]
enable_lora = true
gpu_memory_utilization = 0.75
seed = {seed}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--root", type=Path, required=True)
    register = subparsers.add_parser("preregister")
    register.add_argument("--root", type=Path, required=True)
    register.add_argument("--output", type=Path, required=True)
    confirm = subparsers.add_parser("evaluate")
    confirm.add_argument("--root", type=Path, required=True)
    confirm.add_argument("--preregistration", type=Path, required=True)
    confirm.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "generate":
        paths = generate_configs(args.root)
        print(json.dumps([path.as_posix() for path in paths], indent=2))
        return 0
    if args.command == "preregister":
        print(json.dumps(preregister(args.root, args.output), indent=2, sort_keys=True))
        return 0
    result = evaluate(args.root, args.preregistration, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed_ga_micro"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
