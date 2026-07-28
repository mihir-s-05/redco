from __future__ import annotations

import json
from pathlib import Path

from redco.analysis.stage_c3_live import RUNS, verify_campaign


def _rows(target_mass: float, shift: float = 0.0) -> list[dict[str, object]]:
    remaining = 1.0 - target_mass
    probabilities = {str(index): remaining / 7.0 for index in range(8)}
    probabilities["5"] = target_mass
    if shift:
        probabilities["0"] += shift
        probabilities["1"] -= shift
    return [
        {
            "case_id": f"case-{case}",
            "action_probabilities": probabilities,
        }
        for case in range(4)
    ]


def test_stage_c3_verifier_passes_directional_nuisance_gate(
    tmp_path: Path,
) -> None:
    models: list[dict[str, object]] = [
        {"name": "warmstart", "temperatures": {"2.0": _rows(0.1)}}
    ]
    run_root = tmp_path / "runs"
    for probe, seeds in RUNS.items():
        for seed in seeds:
            for arm in ("broadcast", "sliced"):
                shift = 0.04 if arm == "broadcast" else 0.01
                target = 0.2 if probe != "confusion_irrelevant" else 0.1
                models.append(
                    {
                        "name": f"{probe}--{arm}--s{seed}",
                        "temperatures": {
                            "2.0": _rows(target, shift=shift),
                        },
                    }
                )
                directory = run_root / probe / f"{arm}-s{seed}"
                directory.mkdir(parents=True)
                step = 6 if arm == "broadcast" else 1
                key = (
                    "eval/redco-credit-eval/all/metrics/"
                    "target_success/mean"
                )
                (directory / "orchestrator.jsonl").write_text(
                    json.dumps({"step": step, key: 0.6}) + "\n",
                    encoding="utf-8",
                )
    score_path = tmp_path / "scores.json"
    score_path.write_text(json.dumps({"models": models}), encoding="utf-8")

    result = verify_campaign(run_root, score_path)

    assert result["status"] == "passed"
    assert all(result["mandatory_checks"].values())
