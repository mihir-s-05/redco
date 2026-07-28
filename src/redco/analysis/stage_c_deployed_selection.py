"""Select the earliest merged Stage-C warm start meeting frozen support bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from redco.analysis.stage_c_warmstart import select_warmstart_checkpoint


def select_deployed_warmstart(
    scored_candidates: list[tuple[int, dict[str, Any]]],
    *,
    start_step: int,
    minimum_needle_mass_t2: float,
    maximum_needle_mass_t2: float,
    maximum_needle_greedy_rate: float,
    branch_count: int,
    groups_per_step: int,
    minimum_expected_informative_groups: float,
) -> dict[str, Any]:
    """Apply the frozen earliest-passing rule to sequential merged models."""
    if not scored_candidates:
        raise ValueError("at least one merged candidate is required")
    ordered = sorted(scored_candidates, key=lambda item: item[0])
    steps = [step for step, _ in ordered]
    if len(set(steps)) != len(steps):
        raise ValueError("merged candidate steps must be unique")
    if steps != list(range(start_step, steps[-1] + 1)):
        raise ValueError("merged candidates must be contiguous from start_step")

    normalized_models: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for step, raw in ordered:
        models = raw.get("models")
        if not isinstance(models, list) or len(models) != 1:
            raise ValueError(f"step {step} must contain exactly one scored model")
        model = models[0]
        if not isinstance(model, dict):
            raise ValueError(f"step {step} scored model is malformed")
        normalized = dict(model)
        normalized["name"] = f"sft_step_{step}"
        normalized_models.append(normalized)
        source_records.append(
            {
                "step": step,
                "backend": raw.get("backend"),
                "source": raw.get("source"),
                "temperature_semantics": raw.get("temperature_semantics"),
                "original_model_name": model.get("name"),
            }
        )

    base = select_warmstart_checkpoint(
        {"models": normalized_models},
        minimum_needle_mass_t2=minimum_needle_mass_t2,
        maximum_needle_mass_t2=maximum_needle_mass_t2,
        maximum_needle_greedy_rate=maximum_needle_greedy_rate,
        branch_count=branch_count,
        groups_per_step=groups_per_step,
        minimum_expected_informative_groups=minimum_expected_informative_groups,
    )
    payload: dict[str, Any] = {
        key: value
        for key, value in base.items()
        if key != "signed_payload_sha256"
    }
    payload["analysis"] = "stage-c-deployed-warmstart-selection"
    payload["source_records"] = source_records
    payload["selection_rule"] = (
        "Starting at the first adapter-selected checkpoint, merge and score "
        "contiguous checkpoints in ascending order; select the earliest deployed "
        "merged model meeting every frozen support bound. No held-out reward is "
        "consulted."
    )
    signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        help="Merged candidate as STEP=JSON_PATH; repeat in contiguous order",
    )
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--minimum-needle-mass-t2", type=float, required=True)
    parser.add_argument("--maximum-needle-mass-t2", type=float, required=True)
    parser.add_argument("--maximum-needle-greedy-rate", type=float, required=True)
    parser.add_argument("--branch-count", type=int, required=True)
    parser.add_argument("--groups-per-step", type=int, required=True)
    parser.add_argument(
        "--minimum-expected-informative-groups",
        type=float,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scored: list[tuple[int, dict[str, Any]]] = []
    for value in args.score:
        raw_step, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError("--score must use STEP=JSON_PATH")
        scored.append((int(raw_step), json.loads(Path(raw_path).read_text())))
    report = select_deployed_warmstart(
        scored,
        start_step=args.start_step,
        minimum_needle_mass_t2=args.minimum_needle_mass_t2,
        maximum_needle_mass_t2=args.maximum_needle_mass_t2,
        maximum_needle_greedy_rate=args.maximum_needle_greedy_rate,
        branch_count=args.branch_count,
        groups_per_step=args.groups_per_step,
        minimum_expected_informative_groups=(
            args.minimum_expected_informative_groups
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
