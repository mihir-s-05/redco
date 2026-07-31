from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from redco.integrations.signed_subprocess import sign_payload


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _eval_string(node: ast.expr, values: dict[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif (
                isinstance(value, ast.FormattedValue)
                and isinstance(value.value, ast.Name)
                and value.value.id in values
            ):
                parts.append(values[value.value.id])
            else:
                raise ValueError("unsupported f-string in frozen taskset")
        return "".join(parts)
    raise ValueError("unsupported string expression in frozen taskset")


def _taskset_strings(path: Path) -> dict[str, str]:
    wanted = {
        "WORKDIR",
        "CONTEXT_PATH",
        "ALIGNED_SYSTEM_V2",
        "FORCED_TRACE_FIXTURE",
    }
    values: dict[str, str] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in wanted
        ):
            values[node.targets[0].id] = _eval_string(node.value, values)
    missing = wanted - set(values)
    if missing:
        raise ValueError(f"taskset string constants missing: {sorted(missing)}")
    return values


def _load_seeding_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "redco_stage_d_frozen_seeding",
        path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load seeding source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_migration(
    parent_path: Path,
    successor_path: Path,
    scaffold_path: Path,
    taskset_path: Path,
    seeding_path: Path,
    master_seed: str,
) -> dict[str, Any]:
    parent_rows = _load_rows(parent_path)
    successor_rows = _load_rows(successor_path)
    if len(parent_rows) != len(successor_rows):
        raise ValueError("v1 and v2 fixture row counts differ")

    strings = _taskset_strings(taskset_path)
    scaffold = scaffold_path.read_text(encoding="utf-8")
    system = (
        strings["ALIGNED_SYSTEM_V2"]
        + strings["FORCED_TRACE_FIXTURE"]
        + "\n\n"
        + scaffold
    )
    seeding = _load_seeding_module(seeding_path)

    row_reports: list[dict[str, Any]] = []
    for index, (parent, successor) in enumerate(
        zip(parent_rows, successor_rows, strict=True)
    ):
        if list(successor) != [*parent, "answer_type"]:
            raise ValueError(
                f"row {index} keys are not parent keys plus answer_type"
            )
        if successor["example_id"] != parent["example_id"]:
            raise ValueError(f"row {index} changes example order or ID")
        inherited = dict(successor)
        answer_type = inherited.pop("answer_type")
        if inherited != parent:
            raise ValueError(
                f"row {index} changes inherited fixture content"
            )
        if answer_type != "extractive":
            raise ValueError(
                f"row {index} answer_type must be 'extractive'"
            )
        prompt = f"{system}\n\nQuestion: {successor['question']}"
        row_reports.append(
            {
                "row_index": index,
                "example_id": successor["example_id"],
                "answer_type": answer_type,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "episode_seed": seeding.derive_episode_seed(
                    master_seed,
                    successor["example_id"],
                    0,
                ),
            }
        )

    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-fixture-v1-to-v2-migration-v4-7",
            "parent_fixture": parent_path.as_posix(),
            "parent_fixture_sha256": hashlib.sha256(
                parent_path.read_bytes()
            ).hexdigest(),
            "successor_fixture": successor_path.as_posix(),
            "successor_fixture_sha256": hashlib.sha256(
                successor_path.read_bytes()
            ).hexdigest(),
            "taskset_source": taskset_path.as_posix(),
            "taskset_source_sha256": hashlib.sha256(
                taskset_path.read_bytes()
            ).hexdigest(),
            "seeding_source": seeding_path.as_posix(),
            "seeding_source_sha256": hashlib.sha256(
                seeding_path.read_bytes()
            ).hexdigest(),
            "scaffold": scaffold_path.as_posix(),
            "scaffold_sha256": hashlib.sha256(
                scaffold_path.read_bytes()
            ).hexdigest(),
            "master_seed": master_seed,
            "rows": row_reports,
            "checks": {
                "row_count_order_and_ids_exact": True,
                "successor_keys_are_parent_keys_plus_answer_type": True,
                "all_inherited_fields_deep_equal": True,
                "all_answer_types_extractively_typed": True,
                "rendered_prompt_hashes_frozen": True,
                "episode_seeds_frozen": True,
            },
            "passes": True,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-fixture", type=Path, required=True)
    parser.add_argument("--successor-fixture", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--taskset-source", type=Path, required=True)
    parser.add_argument("--seeding-source", type=Path, required=True)
    parser.add_argument("--master-seed", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_migration(
        args.parent_fixture,
        args.successor_fixture,
        args.scaffold,
        args.taskset_source,
        args.seeding_source,
        args.master_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
