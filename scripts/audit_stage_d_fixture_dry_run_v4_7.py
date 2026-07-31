from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    sign_payload,
    verify_signed_payload,
)


def _resolved(value: str) -> str:
    return Path(value).resolve().as_posix()


def audit_dry_run(
    plan_path: Path,
    loader_report_path: Path,
    *,
    expected_model: str,
    expected_fixture: Path,
    expected_fixture_sha256: str,
    expected_scaffold: Path,
    expected_scaffold_sha256: str,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    loader = json.loads(loader_report_path.read_text(encoding="utf-8"))
    verify_signed_payload(loader)

    config = plan["config"]
    taskset = config["env"]["taskset"]
    episode_plan = plan["episode_seed_plan"]
    if config["model"] != expected_model:
        raise ValueError("dry-run model differs from frozen model")
    if _resolved(taskset["dataset_path"]) != expected_fixture.resolve().as_posix():
        raise ValueError("dry-run fixture path differs")
    if taskset["dataset_sha256"] != expected_fixture_sha256:
        raise ValueError("dry-run fixture SHA-256 differs")
    if taskset["split"] != "audit":
        raise ValueError("dry-run split differs")
    if taskset["prompt_profile"] != "fewshot_fixture_v3":
        raise ValueError("dry-run prompt profile differs")
    if (
        _resolved(taskset["scaffold_prompt_path"])
        != expected_scaffold.resolve().as_posix()
    ):
        raise ValueError("dry-run scaffold path differs")
    if taskset["scaffold_prompt_sha256"] != expected_scaffold_sha256:
        raise ValueError("dry-run scaffold SHA-256 differs")
    if len(episode_plan) != 1:
        raise ValueError("fixture dry-run must resolve exactly one slot")
    first_loader = loader["tasks"][0]
    first_plan = episode_plan[0]
    if (
        first_plan["task_position"] != 0
        or first_plan["task_index"] != first_loader["task_index"]
        or first_plan["example_id"] != first_loader["example_id"]
        or first_plan["replicate"] != 0
        or first_plan["seed"] != first_loader["episode_seed"]
    ):
        raise ValueError("production dry-run slot differs from loader audit")

    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-fixture-production-dry-run-v4-7",
            "plan": plan_path.as_posix(),
            "plan_sha256": hashlib.sha256(
                plan_path.read_bytes()
            ).hexdigest(),
            "loader_report": loader_report_path.as_posix(),
            "loader_report_sha256": hashlib.sha256(
                loader_report_path.read_bytes()
            ).hexdigest(),
            "slot": first_plan,
            "checks": {
                "vf_load_environment_path_resolved": True,
                "fixture_and_scaffold_exact": True,
                "prompt_profile_exact": True,
                "one_slot_exact": True,
                "episode_address_and_seed_exact": True,
            },
            "passes": True,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--loader-report", type=Path, required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-fixture", type=Path, required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--expected-scaffold", type=Path, required=True)
    parser.add_argument("--expected-scaffold-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_dry_run(
        args.plan,
        args.loader_report,
        expected_model=args.expected_model,
        expected_fixture=args.expected_fixture,
        expected_fixture_sha256=args.expected_fixture_sha256,
        expected_scaffold=args.expected_scaffold,
        expected_scaffold_sha256=args.expected_scaffold_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
