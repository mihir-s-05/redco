"""Verify the frozen Stage-C2 SFT warm-start run and merge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from redco.analysis.stage_c_warmstart import select_warmstart_checkpoint


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model(raw: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [
        model
        for model in raw.get("models", [])
        if isinstance(model, dict) and model.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one scored model named {name}")
    return matches[0]


def _merge_equivalence(
    selected_raw: dict[str, Any],
    merged_raw: dict[str, Any],
    *,
    selected_name: str,
    tolerance: float,
) -> dict[str, Any]:
    selected = _model(selected_raw, selected_name)
    merged = _model(merged_raw, "merged")
    selected_cases = {row["case_id"]: row for row in selected["cases"]}
    merged_cases = {row["case_id"]: row for row in merged["cases"]}
    if set(selected_cases) != set(merged_cases):
        raise ValueError("selected-adapter and merged scoring cases differ")
    maximum = 0.0
    greedy_mismatches = 0
    compared = 0
    for case_id in sorted(selected_cases):
        left = selected_cases[case_id]
        right = merged_cases[case_id]
        greedy_mismatches += int(
            left["greedy_token_id"] != right["greedy_token_id"]
        )
        for field in (
            "full_vocab_action_probabilities_t1",
            "full_vocab_action_probabilities_t2",
        ):
            if set(left[field]) != set(right[field]):
                raise ValueError(f"action set differs in {case_id}")
            for action in left[field]:
                difference = abs(float(left[field][action]) - float(right[field][action]))
                maximum = max(maximum, difference)
                compared += 1
    return {
        "compared_probabilities": compared,
        "maximum_absolute_probability_difference": maximum,
        "greedy_token_mismatches": greedy_mismatches,
        "tolerance": tolerance,
        "pass": maximum <= tolerance and greedy_mismatches == 0,
    }


def verify_warmstart_gate(
    *,
    run_dir: Path,
    raw_scores: dict[str, Any],
    selected_scores: dict[str, Any],
    merged_scores: dict[str, Any],
    expected_steps: int = 12,
) -> dict[str, Any]:
    metrics = _read_jsonl(run_dir / "metrics.jsonl")
    metric_steps = {
        int(row["step"])
        for row in metrics
        if isinstance(row.get("step"), int | float)
    }
    if metric_steps != set(range(1, expected_steps + 1)):
        raise ValueError(f"expected metrics for SFT steps 1 through {expected_steps}")
    numeric_metrics = [
        float(value)
        for row in metrics
        for value in row.values()
        if isinstance(value, int | float)
    ]
    if not numeric_metrics or not all(math.isfinite(value) for value in numeric_metrics):
        raise ValueError("SFT metrics contain no finite numeric values")

    adapters: dict[int, dict[str, Any]] = {}
    for step in range(1, expected_steps + 1):
        step_dir = run_dir / "weights" / f"step_{step}"
        adapter_dir = step_dir / "lora_adapters"
        model_path = adapter_dir / "adapter_model.safetensors"
        config_path = adapter_dir / "adapter_config.json"
        if not (model_path.is_file() and config_path.is_file()):
            raise ValueError(f"step {step} is missing its adapter")
        if not (step_dir / "STABLE").is_file():
            raise ValueError(f"step {step} is not marked stable")
        forbidden = [
            path
            for path in step_dir.glob("*.safetensors")
            if path.name != "adapter_model.safetensors"
        ]
        if forbidden:
            raise ValueError(f"step {step} contains gathered model weights")
        adapters[step] = {
            "bytes": model_path.stat().st_size,
            "sha256": _sha256(model_path),
        }

    selection = select_warmstart_checkpoint(
        raw_scores,
        minimum_needle_mass_t2=0.15,
        maximum_needle_mass_t2=0.25,
        maximum_needle_greedy_rate=0.5,
        branch_count=6,
        groups_per_step=8,
        minimum_expected_informative_groups=4.75,
    )
    selected = selection["selected"]
    merge = (
        {
            "pass": False,
            "reason": "no checkpoint met the frozen selection rule",
        }
        if selected is None
        else _merge_equivalence(
            selected_scores,
            merged_scores,
            selected_name=selected["name"],
            tolerance=2e-5,
        )
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "gate": "stage-c2-warmstart",
        "status": (
            "pass"
            if selection["status"] == "pass" and merge["pass"]
            else "fail"
        ),
        "sft": {
            "steps": expected_steps,
            "metric_rows": len(metrics),
            "metric_steps": sorted(metric_steps),
            "adapters": {str(step): value for step, value in adapters.items()},
            "gathered_full_model_files": 0,
        },
        "selection": selection,
        "merge_equivalence": merge,
        "scope": (
            "Support is calibrated on exact gamma-route prefixes from training "
            "artifacts; held-out rewards are not used for checkpoint selection."
        ),
    }
    signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw-scores", type=Path, required=True)
    parser.add_argument("--selected-scores", type=Path, required=True)
    parser.add_argument("--merged-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_warmstart_gate(
        run_dir=args.run_dir,
        raw_scores=json.loads(args.raw_scores.read_text()),
        selected_scores=json.loads(args.selected_scores.read_text()),
        merged_scores=json.loads(args.merged_scores.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
