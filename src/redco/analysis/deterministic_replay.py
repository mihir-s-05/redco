"""Evaluate the stock-first deterministic frozen-rollout stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from redco.analysis.frozen_rollout import (
    ADAPTER_RELATIVE_PATH,
    BATCH_RELATIVE_PATH,
    _comparison,
    _core_metrics,
    _sha256,
)
from redco.contracts import canonical_json


def evaluate_stock_stage(root: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((root / "source-manifest.json").read_bytes())
    source_hash = str(manifest["source_batch_sha256"])
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("stock-a", "stock-b"):
        arm_root = root / arm
        batch = arm_root / BATCH_RELATIVE_PATH
        arms[arm] = {
            "batch_sha256": _sha256(batch),
            "batch_matches_source": _sha256(batch) == source_hash,
            "adapter_sha256": _sha256(arm_root / ADAPTER_RELATIVE_PATH),
            "core_metrics": _core_metrics(arm_root / "metrics.jsonl"),
        }

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_stock_stage(args.root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed_stock_determinism_stage"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
