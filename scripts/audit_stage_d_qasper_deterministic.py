from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/stage-d/qasper-deterministic-v2.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "datasets/stage-d/qasper-deterministic-manifest-v2.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = load_jsonl(args.dataset)
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation")
    }
    paper_sets = {
        split: {row["paper_id"] for row in split_rows}
        for split, split_rows in by_split.items()
    }
    reference_sets = {
        split: {
            span
            for row in split_rows
            for span in row["reference_evidence"]
        }
        for split, split_rows in by_split.items()
    }
    invalid_rows = [
        row["example_id"]
        for row in rows
        if (
            not row["reference_evidence"]
            or any(
                not span
                or len(span) < manifest["selection"]["minimum_span_characters"]
                or span not in row["paper"]
                for span in row["reference_evidence"]
            )
        )
    ]
    dataset_checks = {
        "expected_sha256": manifest["output"]["sha256"],
        "observed_sha256": sha256(args.dataset),
        "split_counts": {
            split: len(split_rows) for split, split_rows in by_split.items()
        },
        "unique_papers": len({row["paper_id"] for row in rows}),
        "unique_questions": len({row["example_id"] for row in rows}),
        "cross_split_papers": sorted(
            paper_sets["train"] & paper_sets["validation"]
        ),
        "cross_split_reference_spans": sorted(
            reference_sets["train"] & reference_sets["validation"]
        ),
        "invalid_rows": invalid_rows,
        "answer_types": {
            split: dict(
                Counter(row["answer_type"] for row in split_rows)
            )
            for split, split_rows in by_split.items()
        },
    }
    dataset_checks["passes"] = (
        dataset_checks["expected_sha256"] == dataset_checks["observed_sha256"]
        and dataset_checks["split_counts"] == {"train": 32, "validation": 32}
        and dataset_checks["unique_papers"] == 64
        and dataset_checks["unique_questions"] == 64
        and not dataset_checks["cross_split_papers"]
        and not dataset_checks["cross_split_reference_spans"]
        and not invalid_rows
    )

    environment_root = (
        Path.cwd() / "environments/redco_evidence_selection_v2"
    )
    sys.path.insert(0, str(environment_root))
    from redco_evidence_selection_v2.scoring import (
        score_evidence_reply,
        score_exact_spans,
    )
    from redco_evidence_selection_v2.taskset import (
        EvidenceSelectionConfig,
        EvidenceSelectionTaskset,
    )

    paper = (
        "Alpha result improved by 20 percent. "
        "Noise was unrelated. "
        "Alpha result improved by 20 percent."
    )
    reference = ("Alpha result improved by 20 percent.",)
    exploit_results = {
        "correct": score_evidence_reply(
            paper, repr(list(reference)), reference
        )["f1"],
        "correct_plus_hallucination": score_evidence_reply(
            paper,
            repr([reference[0], "not in the paper"]),
            reference,
        )["f1"],
        "correct_plus_empty": score_evidence_reply(
            paper, repr([reference[0], ""]), reference
        )["f1"],
        "whole_paper": score_exact_spans(paper, [paper], reference)["f1"],
        "single_word": score_exact_spans(paper, ["Alpha"], reference)["f1"],
        "verbatim_padding": score_exact_spans(
            paper, [reference[0], "Noise was unrelated."], reference
        )["f1"],
        "unparseable": score_evidence_reply(
            paper, "FINAL(nope)", reference
        )["f1"],
        "empty_prediction": score_evidence_reply(paper, "[]", reference)[
            "f1"
        ],
        "empty_reference": score_exact_spans(paper, [reference[0]], [])["f1"],
    }
    scorer_checks = {
        "results": exploit_results,
        "passes": (
            exploit_results["correct"] == 1.0
            and exploit_results["correct_plus_hallucination"] == 0.0
            and exploit_results["correct_plus_empty"] == 0.0
            and 0.0 < exploit_results["whole_paper"] < 1.0
            and 0.0 < exploit_results["single_word"] < 1.0
            and 0.0 < exploit_results["verbatim_padding"] < 1.0
            and exploit_results["unparseable"] == 0.0
            and exploit_results["empty_prediction"] == 0.0
            and exploit_results["empty_reference"] == 0.0
        ),
    }

    task_counts = {}
    for split in ("train", "validation"):
        config = EvidenceSelectionConfig(
            dataset_path=args.dataset,
            dataset_sha256=manifest["output"]["sha256"],
            split=split,
        )
        task_counts[split] = len(
            list(EvidenceSelectionTaskset(config).load())
        )
    taskset_checks = {
        "task_counts": task_counts,
        "passes": task_counts == {"train": 32, "validation": 32},
    }
    passes = (
        dataset_checks["passes"]
        and scorer_checks["passes"]
        and taskset_checks["passes"]
    )
    result = {
        "schema_version": 1,
        "task_name": "GA-QASPER-lite",
        "claim": "deterministic single-paper exact-evidence extraction",
        "forbidden_claims": [
            "GA-full",
            "alphaXiv incumbent reproduction",
            "SkyRL reward reproduction",
            "multi-paper result",
            "long-context generalization",
        ],
        "dataset_checks": dataset_checks,
        "scorer_checks": scorer_checks,
        "taskset_checks": taskset_checks,
        "decision": "pass" if passes else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not passes:
        raise SystemExit("deterministic QASPER audit failed")


if __name__ == "__main__":
    main()
