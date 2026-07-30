from __future__ import annotations

import json
from pathlib import Path

from redco.analysis.stage_c7_live_reuse import evaluate


def test_live_reuse_verifier_requires_exact_step_counts(tmp_path: Path) -> None:
    models = [
        {
            "name": name,
            "temperatures": {
                "2.0": [
                    {"action_probabilities": {"5": mass}},
                    {"action_probabilities": {"5": mass}},
                ]
            },
        }
        for name, mass in (
            ("warmstart", 0.1),
            ("reuse-1", 0.11),
            ("reuse-2", 0.12),
            ("reuse-3", 0.13),
        )
    ]
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"models": models}), encoding="utf-8")
    for updates in (1, 2, 3):
        arm = tmp_path / f"reuse-{updates}"
        arm.mkdir()
        rows = []
        for step in range(1, updates + 1):
            rows.append(
                {
                    "step": step,
                    "optim/grad_norm": 1.0,
                    "redco_node_ratio/mean": 1.0,
                    "redco_node_clipped/mean": 0.0,
                    "redco_node_kl/mean": 0.0,
                }
            )
        (arm / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        adapter = (
            arm
            / "run_default"
            / "broadcasts"
            / f"step_{updates}"
            / "adapter_model.safetensors"
        )
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(b"adapter")

    result = evaluate(tmp_path, scores)

    assert result["status"] == "passed"
    assert result["arms"]["reuse-3"]["optimizer_updates"] == 3
