from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def check_hash_map(root: Path, expected: dict[str, str]) -> dict[str, Any]:
    observed = {relative: sha256(root / relative) for relative in expected}
    mismatches = {
        relative: {"expected": expected[relative], "observed": observed[relative]}
        for relative in expected
        if observed[relative] != expected[relative]
    }
    return {
        "files": len(expected),
        "passes": not mismatches,
        "mismatches": mismatches,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("configs/stage-d/stage-d0-reference-lock-v1.json"),
    )
    parser.add_argument(
        "--trace-contract",
        type=Path,
        default=Path("reports/stage-d0-rlm-trainable-contract-2026-07-30.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path.cwd()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    stack = lock["pinned_stack"]
    commit_checks = {
        "prime_rl": {
            "expected": stack["prime_rl_commit"],
            "observed": checked_commit(root / "external/prime-rl"),
        },
        "verifiers": {
            "expected": stack["verifiers_commit"],
            "observed": checked_commit(root / "external/prime-rl/deps/verifiers"),
        },
    }
    for item in commit_checks.values():
        item["passes"] = item["expected"] == item["observed"]

    hash_checks = {
        name: check_hash_map(root, lock[name])
        for name in ("local_stack_files", "structural_patches", "stage_d0_assets")
    }

    snapshot = lock["datasets"]["ga_d_lite_snapshot"]
    dataset_path = root / snapshot["path"]
    rows = load_jsonl(dataset_path)
    split_counts = Counter(row["split"] for row in rows)
    invalid_evidence = [
        row["example_id"]
        for row in rows
        if not row["reference_evidence"]
        or any(span not in row["paper"] for span in row["reference_evidence"])
    ]
    dataset_check = {
        "path": snapshot["path"],
        "expected_sha256": snapshot["sha256"],
        "observed_sha256": sha256(dataset_path),
        "expected_split_counts": {
            "train": snapshot["train_examples"],
            "validation": snapshot["validation_examples"],
        },
        "observed_split_counts": dict(split_counts),
        "invalid_evidence_examples": invalid_evidence,
    }
    dataset_check["passes"] = (
        dataset_check["expected_sha256"] == dataset_check["observed_sha256"]
        and dataset_check["expected_split_counts"] == dataset_check["observed_split_counts"]
        and not invalid_evidence
    )

    calibration_path = root / "datasets/stage-d/evidence-judge-calibration-v1.jsonl"
    calibration = load_jsonl(calibration_path)
    label_counts = Counter(bool(label) for row in calibration for label in row["expected"].values())
    calibration_check = {
        "cases": len(calibration),
        "criterion_labels": {"negative": label_counts[False], "positive": label_counts[True]},
        "unique_case_ids": len({row["case_id"] for row in calibration}),
    }
    calibration_check["passes"] = (
        calibration_check["cases"] == 12
        and calibration_check["unique_case_ids"] == 12
        and label_counts[False] > 0
        and label_counts[True] > 0
    )

    environment_root = root / "environments/redco_evidence_selection_v1"
    sys.path.insert(0, str(environment_root))
    from redco_evidence_selection_v1.taskset import (
        EvidenceSelectionConfig,
        EvidenceSelectionTaskset,
    )

    task_counts = {}
    taskset_passes = True
    for split, expected in (("train", 32), ("validation", 16)):
        config = EvidenceSelectionConfig(
            dataset_path=dataset_path,
            dataset_sha256=snapshot["sha256"],
            split=split,
        )
        tasks = list(EvidenceSelectionTaskset(config).load())
        task_counts[split] = len(tasks)
        taskset_passes = taskset_passes and len(tasks) == expected
    taskset_check = {"task_counts": task_counts, "passes": taskset_passes}

    trace_contract = json.loads(args.trace_contract.read_text(encoding="utf-8"))
    trace_check = {
        "path": args.trace_contract.as_posix(),
        "cpu_trainable_trace_contract": trace_contract["decision"]["cpu_trainable_trace_contract"],
        "all_sampled_tokens_routed_once": trace_contract["prime_trace_to_samples"][
            "all_sampled_tokens_routed_once"
        ],
        "stage_d_science_ready": trace_contract["decision"]["stage_d_science_ready"],
    }
    trace_check["passes"] = (
        trace_check["cpu_trainable_trace_contract"] == "pass"
        and trace_check["all_sampled_tokens_routed_once"]
    )

    cpu_passes = (
        all(item["passes"] for item in commit_checks.values())
        and all(item["passes"] for item in hash_checks.values())
        and dataset_check["passes"]
        and calibration_check["passes"]
        and taskset_check["passes"]
        and trace_check["passes"]
    )
    result = {
        "schema_version": 1,
        "reference_lock": {
            "path": args.lock.as_posix(),
            "sha256": sha256(args.lock),
        },
        "commit_checks": commit_checks,
        "hash_checks": hash_checks,
        "dataset_check": dataset_check,
        "calibration_check": calibration_check,
        "taskset_check": taskset_check,
        "trace_check": trace_check,
        "decision": {
            "cpu_prerequisites": "pass" if cpu_passes else "fail",
            "ready_for_live_judge_audit": cpu_passes,
            "ready_for_ga_d_lite": False,
            "ready_for_stage_d1": False,
            "remaining_gates": [
                "live frozen-judge calibration",
                "checkpoint-stamped Prime training trace",
                "GA-D-lite stock learning reproduction",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not cpu_passes:
        raise SystemExit("Stage D0 CPU prerequisites failed")


if __name__ == "__main__":
    main()
