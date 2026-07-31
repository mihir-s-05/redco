from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from redco_evidence_selection_v2.seeding import derive_episode_seed
from redco_evidence_selection_v2.taskset import (
    EvidenceSelectionConfig,
    EvidenceSelectionTaskset,
)

from redco.integrations.signed_subprocess import (
    sign_payload,
    verify_signed_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--scaffold-sha256", required=True)
    parser.add_argument("--migration-report", type=Path, required=True)
    parser.add_argument("--master-seed", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    migration = json.loads(
        args.migration_report.read_text(encoding="utf-8")
    )
    verify_signed_payload(migration)
    if migration["master_seed"] != args.master_seed:
        raise ValueError("migration report master seed mismatch")
    expected_by_id = {
        row["example_id"]: row for row in migration["rows"]
    }

    config = EvidenceSelectionConfig(
        dataset_path=args.fixture,
        dataset_sha256=args.fixture_sha256,
        split="audit",
        prompt_profile="fewshot_fixture_v3",
        scaffold_prompt_path=args.scaffold,
        scaffold_prompt_sha256=args.scaffold_sha256,
    )
    tasks = list(EvidenceSelectionTaskset(config).load())
    if len(tasks) != 3:
        raise ValueError(f"expected three fixture tasks, got {len(tasks)}")

    rows = []
    for expected_index, task in enumerate(tasks):
        data = task.data
        if data.idx != expected_index:
            raise ValueError(
                f"unexpected task index: {data.idx} != {expected_index}"
            )
        if data.answer_type != "extractive":
            raise ValueError(
                f"{data.example_id} answer_type is {data.answer_type!r}"
            )
        if "call exactly two `rlm(...)` children" not in data.prompt:
            raise ValueError(
                f"{data.example_id} lacks the forced fixture instruction"
            )
        if "await asyncio.gather" not in data.prompt:
            raise ValueError(
                f"{data.example_id} lacks the shared scaffold demonstration"
            )
        prompt_sha256 = hashlib.sha256(
            data.prompt.encode("utf-8")
        ).hexdigest()
        episode_seed = derive_episode_seed(
            args.master_seed,
            data.example_id,
            0,
        )
        expected = expected_by_id[data.example_id]
        if prompt_sha256 != expected["prompt_sha256"]:
            raise ValueError(
                f"{data.example_id} rendered prompt differs from migration"
            )
        if episode_seed != expected["episode_seed"]:
            raise ValueError(
                f"{data.example_id} episode seed differs from migration"
            )
        rows.append(
            {
                "task_index": data.idx,
                "example_id": data.example_id,
                "answer_type": data.answer_type,
                "prompt_sha256": prompt_sha256,
                "episode_seed": episode_seed,
            }
        )

    payload = sign_payload(
        {
            "schema_version": 1,
            "fixture": args.fixture.as_posix(),
            "fixture_sha256": args.fixture_sha256,
            "scaffold": args.scaffold.as_posix(),
            "scaffold_sha256": args.scaffold_sha256,
            "migration_report": args.migration_report.as_posix(),
            "migration_report_sha256": hashlib.sha256(
                args.migration_report.read_bytes()
            ).hexdigest(),
            "master_seed": args.master_seed,
            "prompt_profile": "fewshot_fixture_v3",
            "tasks": rows,
            "passes": True,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
