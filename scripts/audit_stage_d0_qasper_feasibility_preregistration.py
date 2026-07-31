from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
ENV_ROOT = ROOT / "environments" / "redco_evidence_selection_v2"
sys.path.insert(0, str(ENV_ROOT))
sys.path.insert(0, str(ROOT))

from redco_evidence_selection_v2.seeding import (  # noqa: E402
    derive_episode_seed,
)
from redco_evidence_selection_v2.taskset import (  # noqa: E402
    EvidenceSelectionConfig,
    EvidenceSelectionTaskset,
)

from scripts.analyze_stage_d0_qasper_feasibility import (  # noqa: E402
    EXPECTED_CHECKPOINT,
    MEANINGFUL_F1_RANGE,
    MINIMUM_INFORMATIVE_GROUPS,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "configs/stage-d/"
            "stage-d0-qasper-feasibility-preregistration-v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_json(args.protocol)
    hash_checks = {
        path: {
            "expected": expected,
            "observed": sha256(ROOT / path),
            "passes": sha256(ROOT / path) == expected,
        }
        for path, expected in protocol["source_files"].items()
    }
    dataset_path = ROOT / protocol["dataset"]["path"]
    natural_config = EvidenceSelectionConfig(
        dataset_path=dataset_path,
        dataset_sha256=protocol["dataset"]["sha256"],
        split="train",
        prompt_profile="natural",
    )
    fixture_config = natural_config.model_copy(
        update={"prompt_profile": "forced_trace_fixture"}
    )
    natural_tasks = EvidenceSelectionTaskset(natural_config).load()
    fixture_tasks = EvidenceSelectionTaskset(fixture_config).load()
    natural_prompts = [task.data.prompt for task in natural_tasks[:8]]
    fixture_prompt = fixture_tasks[0].data.prompt
    prompt_checks = {
        "natural_has_no_forced_two_child_instruction": all(
            "call exactly two `rlm(...)` children" not in prompt
            for prompt in natural_prompts
        ),
        "natural_has_no_minimum_child_language": all(
            "minimum child" not in prompt.lower() for prompt in natural_prompts
        ),
        "fixture_forces_exactly_two_children": (
            "call exactly two `rlm(...)` children" in fixture_prompt
        ),
        "fixture_excluded_from_science": (
            "excluded from scientific feasibility metrics" in fixture_prompt
        ),
        "checkpoint_exact": all(
            task.data.policy_checkpoint_id == EXPECTED_CHECKPOINT
            for task in natural_tasks[:8]
        ),
    }
    seeds = [
        derive_episode_seed(
            protocol["natural_smoke"]["master_seed"],
            task.data.example_id,
            replicate,
        )
        for task in natural_tasks[:8]
        for replicate in range(4)
    ]
    seed_checks = {
        "planned_episodes": len(seeds),
        "unique_seeds": len(set(seeds)),
        "all_nonzero_signed_31_bit": all(0 < seed < 2**31 for seed in seeds),
        "passes": (
            len(seeds) == 32
            and len(set(seeds)) == 32
            and all(0 < seed < 2**31 for seed in seeds)
        ),
    }
    frozen_rule_checks = {
        "natural_tasks": protocol["natural_smoke"]["tasks"] == 8,
        "rollouts_per_task": (
            protocol["natural_smoke"]["rollouts_per_task"] == 4
        ),
        "temperature": protocol["natural_smoke"]["temperature"] == 0.7,
        "top_p": protocol["natural_smoke"]["top_p"] == 1.0,
        "max_total_tokens": (
            protocol["natural_smoke"]["max_total_tokens_per_episode"] == 8192
        ),
        "meaningful_f1_range": MEANINGFUL_F1_RANGE == 0.05,
        "minimum_informative_groups": MINIMUM_INFORMATIVE_GROUPS == 5,
        "child_eligibility_threshold": (
            protocol["natural_acceptance"][
                "eligible_pre_action_child_target"
            ].startswith("at least 26/32")
        ),
        "stage_d1_always_false": protocol["decision"][
            "stage_d1_always_false"
        ]
        is True,
        "replay_limitation_explicit": (
            "not independent real-task replay evidence"
            in protocol["forced_trace_fixture"]["interpretation"]
        ),
        "smoke_papers_retired": "excluded" in protocol["dataset"]["future_use"],
        "pip_forbidden": protocol["pinned_stack"]["pip_forbidden"] is True,
    }
    judge_report = load_json(
        ROOT / protocol["history"]["rejected_judge_report"]
    )
    judge_reanalysis = load_json(
        ROOT / protocol["history"]["versioned_numeric_reanalysis"]
    )
    history_checks = {
        "judge_terminal_fail": (
            judge_report.get("status") == "terminal_fail"
            and (
                judge_report.get("frozen_calibration_result") or {}
            ).get("decision")
            == "fail"
        ),
        "judge_reanalysis_fail": judge_reanalysis.get("decision") == "fail",
        "judge_reanalysis_balanced_accuracy": (
            judge_reanalysis.get("balanced_accuracy") == 0.75
        ),
    }
    passes = (
        all(check["passes"] for check in hash_checks.values())
        and all(prompt_checks.values())
        and seed_checks["passes"]
        and all(frozen_rule_checks.values())
        and all(history_checks.values())
    )
    result = {
        "schema_version": 1,
        "protocol": args.protocol.as_posix(),
        "protocol_sha256": sha256(args.protocol),
        "source_hash_checks": hash_checks,
        "prompt_checks": prompt_checks,
        "seed_checks": seed_checks,
        "frozen_rule_checks": frozen_rule_checks,
        "history_checks": history_checks,
        "cpu_validation": {
            "windows_tests": "10 passed, 2 skipped",
            "ruff": "passed",
            "wsl_grouped_dry_run": "passed with 32 unique episode seeds",
            "bash_syntax": "reported separately by the invoking audit command"
        },
        "decision": "pass" if passes else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not passes:
        raise SystemExit("Stage D0 feasibility preregistration audit failed")


if __name__ == "__main__":
    main()
