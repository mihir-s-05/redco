from __future__ import annotations

import hashlib
import json
from pathlib import Path

from redco.analysis.stage_c8_safe_reuse import evaluate


def test_safe_reuse_requires_two_steps_and_normalized_batches(
    tmp_path: Path,
) -> None:
    batch_audit = {
        "frozen_examples": {"msgpack_sha256": "e" * 64},
        "normalized_batches": {},
    }
    for step in (1, 2):
        batch = (
            tmp_path
            / "run_default"
            / "rollouts"
            / f"step_{step}"
            / "train_rollouts.bin"
        )
        batch.parent.mkdir(parents=True)
        batch.write_bytes(f"batch-{step}".encode())
        batch_audit["normalized_batches"][str(step)] = {
            "sha256": hashlib.sha256(batch.read_bytes()).hexdigest()
        }
        adapter = (
            tmp_path
            / "run_default"
            / "broadcasts"
            / f"step_{step}"
            / "adapter_model.safetensors"
        )
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(f"adapter-{step}".encode())

    rows = []
    for step in (1, 2):
        rows.append(
            {
                "step": step,
                "optim/grad_norm": 1.0,
                "redco_node_ratio/mean": 1.0,
                "redco_node_clipped/mean": 0.0,
                "redco_node_kl/mean": 0.0,
            }
        )
    (tmp_path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    raw_summary = tmp_path / "raw-summary.json"
    raw_summary.write_text(
        json.dumps(
            {
                "status": "completed",
                "interpretation": "old_failure_unresolved_or_transient",
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "batch-audit.json"
    audit.write_text(json.dumps(batch_audit), encoding="utf-8")

    result = evaluate(tmp_path, raw_summary, audit)

    assert result["status"] == "passed"
    assert result["optimizer_updates"] == 2
