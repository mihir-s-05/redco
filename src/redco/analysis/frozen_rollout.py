"""Prepare and evaluate a trainer-only frozen-rollout equivalence gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json

ARMS = ("stock-a", "stock-b", "redco")
STOCK_ARMS = ("stock-a", "stock-b")
CORE_PREFIXES = (
    "entropy/",
    "is_masked",
    "kl_ent_ratio/",
    "loss/",
    "masked_",
    "mismatch_kl/",
    "optim/grad_norm",
    "optim/zero_grad_ratio",
    "unmasked_",
)
BATCH_RELATIVE_PATH = Path("run_default/rollouts/step_1/train_rollouts.bin")
ADAPTER_RELATIVE_PATH = Path(
    "run_default/broadcasts/step_1/adapter_model.safetensors"
)


def prepare(
    source_run: Path,
    redco_control: Path,
    root: Path,
    matmul_precision: str | None = None,
) -> dict[str, Any]:
    """Create three trainer-only arms that consume one byte-identical batch."""
    source_batch = source_run / BATCH_RELATIVE_PATH
    trainer_template = source_run / "configs" / "trainer.toml"
    stock_control = source_run / "run_default" / "control" / "orch.toml"
    required = (source_batch, trainer_template, stock_control, redco_control)
    missing = [path.as_posix() for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen-replay inputs: {missing}")

    stock_text = stock_control.read_text(encoding="utf-8")
    redco_text = redco_control.read_text(encoding="utf-8")
    _assert_control_pair(stock_text, redco_text)
    trainer_text = trainer_template.read_text(encoding="utf-8")
    if matmul_precision is not None:
        trainer_text, substitutions = re.subn(
            r'^matmul_precision = ".*"$',
            f'matmul_precision = "{matmul_precision}"',
            trainer_text,
            count=1,
            flags=re.MULTILINE,
        )
        if substitutions != 1:
            raise ValueError("expected exactly one matmul_precision setting")

    for arm in ARMS:
        arm_root = root / arm
        control_template = redco_text if arm == "redco" else stock_text
        _write_rebased_config(
            trainer_text,
            arm_root,
            arm_root / "configs" / "trainer.toml",
        )
        _write_rebased_config(
            control_template,
            arm_root / "run_default",
            arm_root / "run_default" / "control" / "orch.toml",
        )
        batch_target = arm_root / BATCH_RELATIVE_PATH
        batch_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_batch, batch_target)

    source_hash = _sha256(source_batch)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source_run": source_run.as_posix(),
        "source_batch_sha256": source_hash,
        "source_batch_bytes": source_batch.stat().st_size,
        "matmul_precision": matmul_precision,
        "arms": {
            arm: {
                "algorithm": "redco_noop" if arm == "redco" else "grpo",
                "batch_sha256": _sha256(root / arm / BATCH_RELATIVE_PATH),
                "trainer_config_sha256": _sha256(
                    root / arm / "configs" / "trainer.toml"
                ),
                "control_config_sha256": _sha256(
                    root / arm / "run_default" / "control" / "orch.toml"
                ),
            }
            for arm in ARMS
        },
    }
    manifest = root / "source-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(canonical_json(payload) + b"\n")
    return payload


def evaluate(root: Path, output: Path) -> dict[str, Any]:
    """Require exact stock-repeat and stock-versus-no-op trainer equivalence."""
    manifest = json.loads((root / "source-manifest.json").read_bytes())
    source_hash = str(manifest["source_batch_sha256"])
    arms = {arm: _arm_result(root, arm, source_hash) for arm in ARMS}

    comparisons = {
        "stock_repeat": _comparison(arms["stock-a"], arms["stock-b"]),
        "stock_vs_redco_noop": _comparison(arms["stock-a"], arms["redco"]),
    }
    passed = all(
        bool(arm["batch_matches_source"]) for arm in arms.values()
    ) and all(
        bool(comparison["core_metrics_exact"])
        and bool(comparison["adapter_bytes_exact"])
        for comparison in comparisons.values()
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source_batch_sha256": source_hash,
        "arms": arms,
        "comparisons": comparisons,
        "passed_frozen_rollout_gate": passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(payload) + b"\n")
    return payload


def evaluate_stock_precondition(root: Path, output: Path) -> dict[str, Any]:
    """Evaluate stock-repeat determinism before running the ReDCO arm."""
    manifest = json.loads((root / "source-manifest.json").read_bytes())
    source_hash = str(manifest["source_batch_sha256"])
    arms = {arm: _arm_result(root, arm, source_hash) for arm in STOCK_ARMS}

    comparison = _comparison(arms["stock-a"], arms["stock-b"])
    passed = (
        all(bool(arm["batch_matches_source"]) for arm in arms.values())
        and bool(comparison["core_metrics_exact"])
        and bool(comparison["adapter_bytes_exact"])
    )
    redco_executed = (root / "redco" / "metrics.jsonl").is_file()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source_batch_sha256": source_hash,
        "deterministic_settings": {
            "torch_deterministic_algorithms": True,
            "warn_only": False,
            "cublas_workspace_config": ":4096:8",
            "tf32": False,
            "matmul_precision": "highest",
        },
        "arms": arms,
        "stock_repeat": comparison,
        "passed_stock_determinism_stage": passed,
        "redco_executed": redco_executed,
        "conditional_stop_honored": not passed and not redco_executed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(payload) + b"\n")
    return payload


def _arm_result(root: Path, arm: str, source_hash: str) -> dict[str, Any]:
    arm_root = root / arm
    batch = arm_root / BATCH_RELATIVE_PATH
    batch_hash = _sha256(batch)
    return {
        "batch_sha256": batch_hash,
        "batch_matches_source": batch_hash == source_hash,
        "adapter_sha256": _sha256(arm_root / ADAPTER_RELATIVE_PATH),
        "core_metrics": _core_metrics(arm_root / "metrics.jsonl"),
    }


def _comparison(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    reference_metrics: dict[str, float] = reference["core_metrics"]
    candidate_metrics: dict[str, float] = candidate["core_metrics"]
    keys = sorted(reference_metrics.keys() | candidate_metrics.keys())
    differences = {
        key: candidate_metrics.get(key, float("nan"))
        - reference_metrics.get(key, float("nan"))
        for key in keys
        if candidate_metrics.get(key) != reference_metrics.get(key)
    }
    return {
        "core_metrics_exact": not differences,
        "metric_differences": differences,
        "adapter_bytes_exact": (
            candidate["adapter_sha256"] == reference["adapter_sha256"]
        ),
    }


def _core_metrics(path: Path) -> dict[str, float]:
    combined: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            combined.update(json.loads(line))
    selected = {
        key: float(value)
        for key, value in combined.items()
        if any(key.startswith(prefix) for prefix in CORE_PREFIXES)
    }
    if not selected:
        raise ValueError(f"no core trainer metrics in {path}")
    return selected


def _assert_control_pair(stock: str, redco: str) -> None:
    normalized_stock = _normalize_control(stock)
    normalized_redco = _normalize_control(redco)
    if normalized_stock != normalized_redco:
        raise ValueError(
            "stock and ReDCO controls differ beyond output_dir and algorithm type"
        )
    if 'type = "grpo"' not in stock or 'type = "redco_noop"' not in redco:
        raise ValueError("control pair does not contain the expected algorithms")


def _normalize_control(text: str) -> str:
    text = re.sub(
        r'^output_dir = ".*"$',
        'output_dir = "<normalized>"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    return text.replace('type = "redco_noop"', 'type = "grpo"')


def _write_rebased_config(text: str, output_dir: Path, destination: Path) -> None:
    rebased, substitutions = re.subn(
        r'^output_dir = ".*"$',
        f'output_dir = "{output_dir.as_posix()}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if substitutions != 1:
        raise ValueError("expected exactly one top-level output_dir")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rebased, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-run", type=Path, required=True)
    prepare_parser.add_argument("--redco-control", type=Path, required=True)
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument(
        "--matmul-precision",
        choices=("high", "highest"),
    )
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--root", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        result = prepare(
            args.source_run,
            args.redco_control,
            args.root,
            args.matmul_precision,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    result = evaluate(args.root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed_frozen_rollout_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
