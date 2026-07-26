"""Prepare and evaluate frozen-trainer noise-transfer pairs.

The frozen batch is produced before this evaluator runs, so the trainer-only
comparison does not execute either orchestrator algorithm.  It can validate
the transfer of stock-derived numerical bounds, but not no-op integration.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from redco.analysis.frozen_rollout import (
    ADAPTER_RELATIVE_PATH,
    BATCH_RELATIVE_PATH,
    _assert_control_pair,
    _core_metrics,
    _sha256,
    _write_rebased_config,
)
from redco.analysis.stock_noise import _completed_exactly_one_step
from redco.contracts import canonical_json

CONFIRMATION_SEEDS = (5101, 5102, 5103, 5104)
ARMS = ("stock", "redco")


def pair_name(seed: int) -> str:
    return f"pair-s{seed}"


def prepare(
    source_run: Path,
    redco_control: Path,
    bounds_path: Path,
    root: Path,
) -> dict[str, Any]:
    source_batch = source_run / BATCH_RELATIVE_PATH
    trainer_template = source_run / "configs" / "trainer.toml"
    stock_control = source_run / "run_default" / "control" / "orch.toml"
    required = (source_batch, trainer_template, stock_control, redco_control)
    missing = [path.as_posix() for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing confirmation inputs: {missing}")

    bounds = json.loads(bounds_path.read_bytes())
    if bounds["status"] != "frozen_before_unseen_confirmation":
        raise ValueError("confirmation bounds are not frozen")
    source_hash = _sha256(source_batch)
    if source_hash != bounds["source"]["batch_sha256"]:
        raise ValueError("source batch does not match the frozen bounds")

    stock_text = stock_control.read_text(encoding="utf-8")
    redco_text = redco_control.read_text(encoding="utf-8")
    _assert_control_pair(stock_text, redco_text)
    trainer_text = trainer_template.read_text(encoding="utf-8")
    trainer_text, substitutions = re.subn(
        r'^matmul_precision = ".*"$',
        'matmul_precision = "highest"',
        trainer_text,
        count=1,
        flags=re.MULTILINE,
    )
    if substitutions != 1:
        raise ValueError("expected exactly one matmul_precision setting")

    pairs: dict[str, Any] = {}
    for seed in CONFIRMATION_SEEDS:
        name = pair_name(seed)
        pair_payload: dict[str, Any] = {"seed": seed, "arms": {}}
        for arm in ARMS:
            arm_root = root / name / arm
            control_text = redco_text if arm == "redco" else stock_text
            _write_rebased_config(
                trainer_text,
                arm_root,
                arm_root / "configs" / "trainer.toml",
            )
            _write_rebased_config(
                control_text,
                arm_root / "run_default",
                arm_root / "run_default" / "control" / "orch.toml",
            )
            batch_target = arm_root / BATCH_RELATIVE_PATH
            batch_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_batch, batch_target)
            pair_payload["arms"][arm] = {
                "algorithm": "redco_noop" if arm == "redco" else "grpo",
                "batch_sha256": _sha256(batch_target),
                "trainer_config_sha256": _sha256(
                    arm_root / "configs" / "trainer.toml"
                ),
                "control_config_sha256": _sha256(
                    arm_root / "run_default" / "control" / "orch.toml"
                ),
            }
        pairs[name] = pair_payload

    payload: dict[str, Any] = {
        "schema_version": 2,
        "status": "prepared_unseen",
        "source_run": source_run.as_posix(),
        "source_batch_sha256": source_hash,
        "source_batch_bytes": source_batch.stat().st_size,
        "bounds_path": bounds_path.as_posix(),
        "bounds_sha256": _sha256(bounds_path),
        "pairs": pairs,
    }
    manifest = root / "source-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(canonical_json(payload) + b"\n")
    return payload


def evaluate(root: Path, bounds_path: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((root / "source-manifest.json").read_bytes())
    bounds = json.loads(bounds_path.read_bytes())
    if _sha256(bounds_path) != manifest["bounds_sha256"]:
        raise ValueError("frozen bounds changed after confirmation preparation")

    exact_keys = list(bounds["exact_invariant_metrics"])
    margins = bounds["equivalence_margins"]
    adapter_payload = json.loads(
        (root / "adapter-pairwise.json").read_bytes()
    )
    adapter_by_pair = {
        str(comparison["pair"]): comparison
        for comparison in adapter_payload["comparisons"]
    }
    expected_names = {pair_name(seed) for seed in CONFIRMATION_SEEDS}
    if set(adapter_by_pair) != expected_names:
        raise ValueError("adapter comparisons do not match confirmation pairs")

    pair_results: dict[str, Any] = {}
    source_hash = str(manifest["source_batch_sha256"])
    for seed in CONFIRMATION_SEEDS:
        name = pair_name(seed)
        stock_root = root / name / "stock"
        redco_root = root / name / "redco"
        stock_metrics = _core_metrics(stock_root / "metrics.jsonl")
        redco_metrics = _core_metrics(redco_root / "metrics.jsonl")
        missing_exact = [
            key
            for key in exact_keys
            if key not in stock_metrics or key not in redco_metrics
        ]
        exact_differences = {
            key: redco_metrics[key] - stock_metrics[key]
            for key in exact_keys
            if key not in missing_exact
            and redco_metrics[key] != stock_metrics[key]
        }
        if (
            "optim/grad_norm" not in stock_metrics
            or "optim/grad_norm" not in redco_metrics
        ):
            raise ValueError(f"missing grad norm in {name}")
        grad_norm_difference = abs(
            redco_metrics["optim/grad_norm"]
            - stock_metrics["optim/grad_norm"]
        )
        adapter = adapter_by_pair[name]
        numerical = {
            "grad_norm_absolute_difference": {
                "observed": grad_norm_difference,
                "margin": float(margins["grad_norm_absolute_difference"]),
                "passed": (
                    grad_norm_difference
                    <= float(margins["grad_norm_absolute_difference"])
                ),
            },
            "adapter_l2": {
                "observed": float(adapter["l2"]),
                "margin": float(margins["adapter_l2"]),
                "passed": float(adapter["l2"]) <= float(margins["adapter_l2"]),
            },
            "adapter_max_abs": {
                "observed": float(adapter["max_abs"]),
                "margin": float(margins["adapter_max_abs"]),
                "passed": (
                    float(adapter["max_abs"])
                    <= float(margins["adapter_max_abs"])
                ),
            },
        }
        batch_hashes = {
            arm: _sha256(root / name / arm / BATCH_RELATIVE_PATH)
            for arm in ARMS
        }
        completed_one_step = {
            arm: _completed_exactly_one_step(
                root / name / arm / "metrics.jsonl"
            )
            for arm in ARMS
        }
        exact_passed = not missing_exact and not exact_differences
        batches_passed = all(
            batch_hash == source_hash for batch_hash in batch_hashes.values()
        )
        numerical_passed = all(
            bool(metric["passed"]) for metric in numerical.values()
        )
        pair_passed = (
            exact_passed
            and batches_passed
            and all(completed_one_step.values())
            and numerical_passed
        )
        pair_results[name] = {
            "seed": seed,
            "batch_sha256": batch_hashes,
            "batches_match_source": batches_passed,
            "completed_exactly_one_optimizer_step": completed_one_step,
            "missing_exact_metrics": missing_exact,
            "exact_metric_differences": exact_differences,
            "exact_metrics_passed": exact_passed,
            "numerical_checks": numerical,
            "passed": pair_passed,
            "artifact_sha256": {
                arm: {
                    "metrics": _sha256(root / name / arm / "metrics.jsonl"),
                    "adapter": _sha256(
                        root / name / arm / ADAPTER_RELATIVE_PATH
                    ),
                }
                for arm in ARMS
            },
        }

    passed = all(bool(pair["passed"]) for pair in pair_results.values())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "passed_frozen_trainer_noise_transfer_gate"
            if passed
            else "failed_frozen_trainer_noise_transfer_gate"
        ),
        "bounds_sha256": _sha256(bounds_path),
        "source_batch_sha256": source_hash,
        "pair_count": len(pair_results),
        "pairs": pair_results,
        "passed_frozen_trainer_noise_transfer_gate": passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(payload) + b"\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-run", type=Path, required=True)
    prepare_parser.add_argument("--redco-control", type=Path, required=True)
    prepare_parser.add_argument("--bounds", type=Path, required=True)
    prepare_parser.add_argument("--root", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--root", type=Path, required=True)
    evaluate_parser.add_argument("--bounds", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.source_run,
            args.redco_control,
            args.bounds,
            args.root,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    result = evaluate(args.root, args.bounds, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return (
        0
        if result["passed_frozen_trainer_noise_transfer_gate"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
