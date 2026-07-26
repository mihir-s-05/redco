"""Prepare and calibrate stock-only frozen-rollout trainer noise."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
from pathlib import Path
from typing import Any

from redco.analysis.frozen_rollout import (
    ADAPTER_RELATIVE_PATH,
    BATCH_RELATIVE_PATH,
    _core_metrics,
    _sha256,
    _write_rebased_config,
)
from redco.contracts import canonical_json

RUN_NAMES = tuple(f"stock-c{index:02d}" for index in range(1, 9))
MARGIN_MULTIPLIER = 2.0
RESOLUTION_FLOOR = 1e-8


def _completed_exactly_one_step(path: Path) -> bool:
    steps: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "step" in record:
            steps.add(int(record["step"]))
    return steps == {1}


def prepare(source_run: Path, root: Path) -> dict[str, Any]:
    source_batch = source_run / BATCH_RELATIVE_PATH
    trainer_template = source_run / "configs" / "trainer.toml"
    stock_control = source_run / "run_default" / "control" / "orch.toml"
    required = (source_batch, trainer_template, stock_control)
    missing = [path.as_posix() for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing stock-noise inputs: {missing}")

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
    control_text = stock_control.read_text(encoding="utf-8")

    for name in RUN_NAMES:
        run_root = root / name
        _write_rebased_config(
            trainer_text,
            run_root,
            run_root / "configs" / "trainer.toml",
        )
        _write_rebased_config(
            control_text,
            run_root / "run_default",
            run_root / "run_default" / "control" / "orch.toml",
        )
        batch_target = run_root / BATCH_RELATIVE_PATH
        batch_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_batch, batch_target)

    source_hash = _sha256(source_batch)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source_run": source_run.as_posix(),
        "source_batch_sha256": source_hash,
        "source_batch_bytes": source_batch.stat().st_size,
        "run_names": list(RUN_NAMES),
        "runs": {
            name: {
                "batch_sha256": _sha256(root / name / BATCH_RELATIVE_PATH),
                "trainer_config_sha256": _sha256(
                    root / name / "configs" / "trainer.toml"
                ),
                "control_config_sha256": _sha256(
                    root / name / "run_default" / "control" / "orch.toml"
                ),
            }
            for name in RUN_NAMES
        },
    }
    manifest = root / "source-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(canonical_json(payload) + b"\n")
    return payload


def calibrate(root: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((root / "source-manifest.json").read_bytes())
    source_hash = str(manifest["source_batch_sha256"])
    metrics = {
        name: _core_metrics(root / name / "metrics.jsonl") for name in RUN_NAMES
    }
    baseline_keys = set(metrics[RUN_NAMES[0]])
    invariant_keys = sorted(baseline_keys - {"optim/grad_norm"})
    exact_violations: dict[str, dict[str, float | None]] = {}
    baseline = metrics[RUN_NAMES[0]]
    for name in RUN_NAMES[1:]:
        for key in invariant_keys:
            candidate = metrics[name].get(key)
            if candidate != baseline.get(key):
                exact_violations[f"{name}:{key}"] = {
                    "reference": baseline.get(key),
                    "candidate": candidate,
                }
        if set(metrics[name]) != baseline_keys:
            exact_violations[f"{name}:metric_keys"] = {
                "reference": float(len(baseline_keys)),
                "candidate": float(len(metrics[name])),
            }

    grad_norm_pairwise: list[dict[str, Any]] = [
        {
            "first": first,
            "second": second,
            "absolute_difference": abs(
                metrics[first]["optim/grad_norm"]
                - metrics[second]["optim/grad_norm"]
            ),
        }
        for first, second in itertools.combinations(RUN_NAMES, 2)
    ]
    adapter_payload = json.loads((root / "adapter-pairwise.json").read_bytes())
    adapter_pairs: list[dict[str, Any]] = adapter_payload["comparisons"]
    expected_pairs = len(RUN_NAMES) * (len(RUN_NAMES) - 1) // 2
    if len(adapter_pairs) != expected_pairs:
        raise ValueError(
            f"expected {expected_pairs} adapter comparisons, got {len(adapter_pairs)}"
        )
    expected_pair_names = {
        tuple(sorted(pair)) for pair in itertools.combinations(RUN_NAMES, 2)
    }
    observed_pair_names = {
        tuple(sorted((str(pair["first"]), str(pair["second"]))))
        for pair in adapter_pairs
    }
    if observed_pair_names != expected_pair_names:
        raise ValueError(
            "adapter comparisons must cover every stock-stock pair exactly once"
        )

    batch_hashes = {
        name: _sha256(root / name / BATCH_RELATIVE_PATH) for name in RUN_NAMES
    }
    batches_exact = all(value == source_hash for value in batch_hashes.values())
    completed_one_step = {
        name: _completed_exactly_one_step(root / name / "metrics.jsonl")
        for name in RUN_NAMES
    }
    exact_passed = not exact_violations
    calibration_passed = (
        batches_exact and all(completed_one_step.values()) and exact_passed
    )
    observed_maxima: dict[str, float] = {
        "grad_norm_absolute_difference": max(
            float(pair["absolute_difference"]) for pair in grad_norm_pairwise
        ),
        "adapter_l2": max(float(pair["l2"]) for pair in adapter_pairs),
        "adapter_max_abs": max(float(pair["max_abs"]) for pair in adapter_pairs),
    }
    margins = {
        metric: max(MARGIN_MULTIPLIER * maximum, RESOLUTION_FLOOR)
        for metric, maximum in observed_maxima.items()
    }
    artifact_hashes: dict[str, dict[str, str]] = {
        name: {
            "batch": batch_hashes[name],
            "metrics": _sha256(root / name / "metrics.jsonl"),
            "adapter": _sha256(root / name / ADAPTER_RELATIVE_PATH),
        }
        for name in RUN_NAMES
    }
    artifact_hashes["calibration"] = {
        "adapter_pairwise": _sha256(root / "adapter-pairwise.json")
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "frozen_for_unseen_confirmation"
            if calibration_passed
            else "calibration_failed"
        ),
        "source_batch_sha256": source_hash,
        "stock_calibration_runs": list(RUN_NAMES),
        "pairwise_comparisons": expected_pairs,
        "exact_invariant_metrics": invariant_keys,
        "exact_invariant_violations": exact_violations,
        "exact_invariants_passed": exact_passed,
        "all_batches_match_source": batches_exact,
        "completed_exactly_one_optimizer_step": completed_one_step,
        "grad_norm_pairwise": grad_norm_pairwise,
        "adapter_pairwise": adapter_pairs,
        "observed_pairwise_maxima": observed_maxima,
        "margin_rule": "max(2 * observed stock-stock pairwise maximum, 1e-8)",
        "equivalence_margins": margins,
        "confirmation_design": {
            "runs_per_arm": 4,
            "arms": ["stock", "redco_noop"],
            "unseen": True,
            "same_frozen_batch": True,
            "exact_metrics_must_remain_exact": True,
            "each_pair_must_satisfy_all_numerical_margins": True,
        },
        "artifact_sha256": artifact_hashes,
        "calibration_passed": calibration_passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(payload) + b"\n")
    return payload

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-run", type=Path, required=True)
    prepare_parser.add_argument("--root", type=Path, required=True)
    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--root", type=Path, required=True)
    calibrate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source_run, args.root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    result = calibrate(args.root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["calibration_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
