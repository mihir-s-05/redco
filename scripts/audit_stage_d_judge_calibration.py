from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def balanced_accuracy(expected: list[bool], predicted: list[bool]) -> float:
    true_positive = sum(e and p for e, p in zip(expected, predicted, strict=True))
    positive = sum(expected)
    true_negative = sum(
        not e and not p for e, p in zip(expected, predicted, strict=True)
    )
    negative = len(expected) - positive
    if positive == 0 or negative == 0:
        raise ValueError("calibration labels must contain both classes")
    return 0.5 * (true_positive / positive + true_negative / negative)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--strong-threshold", type=int, default=7)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.8)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.responses.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    parse_failures = sum(row["parsed"] is None for row in rows)
    repeat_disagreements = 0
    expected_labels: list[bool] = []
    predicted_labels: list[bool] = []
    case_results: list[dict[str, Any]] = []
    for case_id, repeats in sorted(by_case.items()):
        if len(repeats) != args.repeats:
            raise ValueError(
                f"{case_id} has {len(repeats)} repeats, expected {args.repeats}"
            )
        parsed = [row["parsed"] for row in repeats]
        if any(value != parsed[0] for value in parsed[1:]):
            repeat_disagreements += 1
        verdicts = {
            item["name"]: int(item["verdict"])
            for item in (parsed[0] or {}).get("verdicts", [])
        }
        outcome: dict[str, bool] = {}
        for criterion in ("precision", "recall"):
            predicted = verdicts.get(criterion, 0) >= args.strong_threshold
            expected = bool(repeats[0]["expected"][criterion])
            expected_labels.append(expected)
            predicted_labels.append(predicted)
            outcome[criterion] = predicted == expected
        case_results.append({"case_id": case_id, "correct": outcome})
    accuracy = balanced_accuracy(expected_labels, predicted_labels)
    passes = (
        parse_failures == 0
        and repeat_disagreements == 0
        and accuracy >= args.minimum_balanced_accuracy
    )
    result = {
        "schema_version": 1,
        "cases": len(by_case),
        "responses": len(rows),
        "parse_failures": parse_failures,
        "repeat_disagreements": repeat_disagreements,
        "balanced_accuracy": accuracy,
        "minimum_balanced_accuracy": args.minimum_balanced_accuracy,
        "case_results": case_results,
        "decision": "pass" if passes else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
