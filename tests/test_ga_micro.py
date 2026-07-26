from __future__ import annotations

import json
from pathlib import Path

from redco.analysis.ga_micro import (
    CONFIRM_SEEDS,
    PILOT_SEEDS,
    evaluate,
    generate_configs,
    preregister,
)


def _write_run(root: Path, name: str, shift: float = 0.0) -> None:
    run = root / name
    (run / "run_default").mkdir(parents=True)
    trace_dir = run / "run_default" / "rollouts" / "step_1" / "train" / "effective"
    trace_dir.mkdir(parents=True)
    trainer = [
        {"optim/grad_norm": 0.9 + shift},
        {
            "loss/mean": 0.001 + shift,
            "entropy/all/mean": 0.02 + shift,
            "mismatch_kl/all/mean": 0.0003 + shift,
        },
    ]
    orchestrator = {
        "train/agg/effective/reward/mean": 0.5 + shift,
        "progress/input_tokens": 100.0 + shift,
        "progress/output_tokens": 50.0 + shift,
        "train/agg/all/has_error/mean": 0.0,
        "train/agg/all/is_truncated/mean": 0.0,
    }
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trainer),
        encoding="utf-8",
    )
    (run / "run_default" / "metrics.jsonl").write_text(
        json.dumps(orchestrator) + "\n",
        encoding="utf-8",
    )
    (trace_dir / "traces.jsonl").write_text(
        json.dumps({"name": name, "shift": shift}) + "\n",
        encoding="utf-8",
    )


def test_ga_micro_preregistration_is_frozen_before_confirmatory_evaluation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ga"
    paths = generate_configs(root)
    assert len(paths) == 12
    assert 'REDCO_RUN_SEED = "2101"' in paths[0].read_text(encoding="utf-8")

    for seed in PILOT_SEEDS:
        _write_run(root, f"pilot-stock-s{seed}-a")
        _write_run(root, f"pilot-stock-s{seed}-b", shift=1e-5)
    registration_path = root / "preregister.json"
    registration = preregister(root, registration_path)
    assert registration["status"] == "frozen_before_confirmatory_runs"
    assert registration["confirmatory_runs_per_arm"] == 4

    for seed in CONFIRM_SEEDS:
        _write_run(root, f"confirm-stock-s{seed}")
        _write_run(root, f"confirm-redco-s{seed}", shift=1e-6)
    result = evaluate(root, registration_path, root / "result.json")
    assert result["passed_ga_micro"]
