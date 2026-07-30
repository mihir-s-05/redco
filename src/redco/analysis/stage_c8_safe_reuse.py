"""Verify the safe single-adapter two-update Stage-C8 microcheck."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload

REQUIRED_METRICS = (
    "optim/grad_norm",
    "redco_node_ratio/mean",
    "redco_node_clipped/mean",
    "redco_node_kl/mean",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(path: Path) -> dict[int, dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        step = int(row["step"])
        by_step.setdefault(step, {}).update(
            {
                str(key): float(value)
                for key, value in row.items()
                if key not in {"step", "time"} and isinstance(value, int | float)
            }
        )
    return by_step


def evaluate(
    run_root: Path,
    raw_summary_path: Path,
    batch_audit_path: Path,
) -> dict[str, Any]:
    raw_summary = json.loads(raw_summary_path.read_text(encoding="utf-8"))
    if raw_summary.get("status") != "completed":
        raise ValueError("raw grouped-MM probe did not reach a recorded disposition")
    batch_audit = json.loads(batch_audit_path.read_text(encoding="utf-8"))

    batch_hashes = {}
    for step in (1, 2):
        batch = (
            run_root
            / "run_default"
            / "rollouts"
            / f"step_{step}"
            / "train_rollouts.bin"
        )
        actual = _sha256(batch)
        expected = str(batch_audit["normalized_batches"][str(step)]["sha256"])
        if actual != expected:
            raise ValueError(f"step {step} normalized batch hash changed")
        batch_hashes[str(step)] = actual

    metrics = _metrics(run_root / "metrics.jsonl")
    if set(metrics) != {1, 2}:
        raise ValueError("safe reuse run must contain exactly optimizer steps 1 and 2")
    for step, row in metrics.items():
        if not all(
            key in row and math.isfinite(row[key])
            for key in REQUIRED_METRICS
        ):
            raise ValueError(f"step {step} is missing finite practical-loss metrics")

    adapters = {}
    for step in (1, 2):
        adapter = (
            run_root
            / "run_default"
            / "broadcasts"
            / f"step_{step}"
            / "adapter_model.safetensors"
        )
        if not adapter.is_file():
            raise ValueError(f"step {step} adapter is missing")
        adapters[str(step)] = _sha256(adapter)

    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-c8-safe-single-adapter-two-update-reuse",
            "status": "passed",
            "optimizer_updates": 2,
            "batch_examples_sha256": batch_audit["frozen_examples"][
                "msgpack_sha256"
            ],
            "normalized_batch_sha256": batch_hashes,
            "adapter_sha256": adapters,
            "steps": metrics,
            "raw_grouped_mm_interpretation": raw_summary["interpretation"],
            "interpretation": (
                "One trainer process completed two optimizer updates using the "
                "safe ordinary-matmul path for a single LoRA adapter. The examples "
                "were content-identical across steps and only transport step "
                "metadata changed. This validates reuse mechanics, not fresh-rollout "
                "learning or the cause of the earlier grouped-kernel crash."
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-summary", type=Path, required=True)
    parser.add_argument("--batch-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(
        args.output,
        evaluate(args.run_root, args.raw_summary, args.batch_audit),
    )


if __name__ == "__main__":
    main()
