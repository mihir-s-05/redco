"""Audit the frozen Stage-C2 three-arm campaign and apply its decision rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from redco.analysis.stage_c_postmortem import (
    ArmSpec,
    summarize_arm,
)

EXPECTED = {
    "full_suffix": {"steps": 12, "calls": 1152, "branching": True},
    "broadcast": {"steps": 72, "calls": 1152, "branching": False},
    "sliced": {"steps": 12, "calls": 1152, "branching": True},
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _step(path: Path) -> int:
    match = re.search(r"[\\/]step_(\d+)[\\/]", str(path))
    if match is None:
        raise ValueError(f"cannot infer step from {path}")
    return int(match.group(1))


def _endpoint(scores: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [model for model in scores["models"] if model["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one score model named {name}")
    rows = [
        row for row in matches[0]["temperatures"]["2.0"] if row["probe_name"] == "planted_needle"
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one planted_needle endpoint for {name}")
    row = rows[0]
    return {
        "action_5_mass_temperature_2": float(row["action_probabilities"]["5"]),
        "greedy_allowed_action": str(row["greedy_allowed_action"]),
        "greedy_token_id": int(row["greedy_token_id"]),
    }


def _structure(run_dir: Path, *, final_step: int) -> dict[str, Any]:
    paths = sorted(
        run_dir.glob("run_default/rollouts/step_*/train/effective/traces.jsonl"),
        key=_step,
    )
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    total_calls = 0
    for path in paths:
        step = _step(path)
        for trace in _jsonl(path):
            info = trace.get("info")
            if not isinstance(info, dict) or not isinstance(info.get("episode_id"), str):
                raise ValueError(f"missing episode_id in {path}")
            calls = trace.get("calls")
            if not isinstance(calls, list):
                raise ValueError(f"missing calls in {path}")
            total_calls += len(calls)
            groups[(step, info["episode_id"])].append(trace)

    errors: list[str] = []
    context_records = 0
    branch_records = 0
    for (step, episode), traces in groups.items():
        redco_records = []
        for trace in traces:
            info = trace["info"]
            record = info.get("redco")
            if not isinstance(record, dict):
                errors.append(f"step {step} episode {episode}: missing redco record")
                continue
            redco_records.append((trace, record))
        contexts = [
            (trace, record)
            for trace, record in redco_records
            if record.get("record_kind") == "context"
        ]
        branches = [
            (trace, record)
            for trace, record in redco_records
            if record.get("record_kind") == "branch"
        ]
        context_records += len(contexts)
        branch_records += len(branches)
        indices: list[int] = []
        for _, record in branches:
            index = record.get("branch_index")
            if isinstance(index, int):
                indices.append(index)
            else:
                errors.append(f"step {step} episode {episode}: invalid branch index {index}")
        indices.sort()
        if len(contexts) != 1:
            errors.append(f"step {step} episode {episode}: {len(contexts)} contexts")
        if indices != list(range(11)):
            errors.append(f"step {step} episode {episode}: branch indices {indices}")
        for trace, record in branches:
            sampling = trace.get("agent", {}).get("sampling", {})
            if sampling.get("temperature") != 2:
                errors.append(f"step {step} episode {episode}: non-T2 branch")
            if sampling.get("max_tokens") != 1:
                errors.append(f"step {step} episode {episode}: non-single-token branch")
            if not isinstance(sampling.get("seed"), int):
                errors.append(f"step {step} episode {episode}: missing branch seed")
            if record.get("selected_pre_action") is not True:
                errors.append(f"step {step} episode {episode}: post-action selection")
            if record.get("replay_equivalent") is not True:
                errors.append(f"step {step} episode {episode}: replay mismatch")

    expected_groups = final_step * 8
    return {
        "steps": len(paths),
        "policy_calls": total_calls,
        "episodes": len(groups),
        "expected_episodes": expected_groups,
        "context_records": context_records,
        "branch_records": branch_records,
        "expected_context_records": expected_groups,
        "expected_branch_records": expected_groups * 11,
        "errors": errors,
        "passed": (
            len(paths) == final_step
            and total_calls == final_step * 96
            and len(groups) == expected_groups
            and context_records == expected_groups
            and branch_records == expected_groups * 11
            and not errors
        ),
    }


def _valid_action_rate(summary: dict[str, Any], *, branching: bool) -> float:
    training = summary["training"]
    counts = training["actions_by_probe"].get("planted_needle", {})
    valid = sum(int(counts.get(str(action), 0)) for action in range(8))
    denominator = (
        int(training["branch_records"])
        if branching
        else sum(int(value) for value in counts.values())
    )
    return valid / denominator


def build_result(
    *,
    root: Path,
    scores_path: Path,
) -> dict[str, Any]:
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    summaries: dict[str, dict[str, Any]] = {}
    structures: dict[str, dict[str, Any] | None] = {}
    for name, expected in EXPECTED.items():
        run_dir = root / name.replace("_", "-")
        summary = summarize_arm(
            ArmSpec(
                name,
                run_dir,
                int(expected["steps"]),
                bool(expected["branching"]),
            )
        )
        summary["training"]["valid_action_rate"] = _valid_action_rate(
            summary,
            branching=bool(expected["branching"]),
        )
        branch_records = int(summary["training"]["branch_records"])
        nonzero = int(summary["training"]["nonzero_branch_advantage_records"])
        summary["training"]["zero_advantage_record_rate"] = (
            (branch_records - nonzero) / branch_records if branch_records else None
        )
        summaries[name] = summary
        structures[name] = (
            _structure(run_dir, final_step=int(expected["steps"]))
            if expected["branching"]
            else None
        )

    endpoints = {
        "initialization": _endpoint(scores, "warmstart"),
        **{name: _endpoint(scores, name) for name in ("full_suffix", "broadcast", "sliced")},
    }
    integration_checks: dict[str, bool] = {}
    for name, expected in EXPECTED.items():
        training = summaries[name]["training"]
        integration_checks[f"{name}_optimizer_steps"] = (
            training["optimizer_steps"] == expected["steps"]
        )
        integration_checks[f"{name}_policy_calls"] = training["policy_calls"] == expected["calls"]
        gradients = training["gradient_norm"]
        integration_checks[f"{name}_finite_gradients"] = all(
            math.isfinite(float(gradients[key])) for key in ("minimum", "median", "mean", "maximum")
        )
        if expected["branching"]:
            structure = structures[name]
            assert structure is not None
            integration_checks[f"{name}_branch_structure"] = bool(structure["passed"])

    broadcast_mass = endpoints["broadcast"]["action_5_mass_temperature_2"]
    credit_components = {}
    for name in ("full_suffix", "sliced"):
        training = summaries[name]["training"]
        gradients = training["gradient_norm"]
        credit_components[name] = {
            "mass_minus_broadcast": (
                endpoints[name]["action_5_mass_temperature_2"] - broadcast_mass
            ),
            "exceeds_broadcast_by_at_least_0_02": (
                endpoints[name]["action_5_mass_temperature_2"] - broadcast_mass >= 0.02
            ),
            "has_informative_group": (training["informative_branch_groups"] >= 1),
            "has_nonzero_finite_gradient_step": (
                math.isfinite(float(gradients["maximum"])) and float(gradients["maximum"]) > 0
            ),
        }

    replay_difference = abs(
        endpoints["full_suffix"]["action_5_mass_temperature_2"]
        - endpoints["sliced"]["action_5_mass_temperature_2"]
    )
    decisions = {
        "integration_passed": all(integration_checks.values()),
        "mechanistic_credit_positive": all(
            all(component.values()) for component in credit_components.values()
        ),
        "replay_equivalence_passed": (
            replay_difference <= 0.05
            and endpoints["full_suffix"]["greedy_allowed_action"]
            == endpoints["sliced"]["greedy_allowed_action"]
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "stage-c2-powered-warmstart-and-credit",
        "status": "frozen-campaign-result",
        "endpoints": endpoints,
        "arms": summaries,
        "structural_audit": structures,
        "integration_checks": integration_checks,
        "credit_signal_components": credit_components,
        "replay_equivalence": {
            "absolute_action_5_mass_difference": replay_difference,
            "maximum_allowed": 0.05,
            "greedy_actions_agree": (
                endpoints["full_suffix"]["greedy_allowed_action"]
                == endpoints["sliced"]["greedy_allowed_action"]
            ),
        },
        "decisions": decisions,
        "interpretation": (
            "The shared warm start removed the exploration bottleneck and all "
            "three arms learned the planted action. The frozen mechanistic "
            "credit-positive rule did not pass because neither ReDCO endpoint "
            "exceeded broadcast by 0.02. Sliced and full-suffix replay remained "
            "equivalent under the frozen endpoint rule."
        ),
        "statistical_scope": (
            "Exact descriptive differences only; repeated evaluation traces are "
            "not treated as independent statistical units."
        ),
        "input_sha256": {
            "final_policy_scores": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        },
    }
    signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_result(root=args.campaign_root, scores_path=args.scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
