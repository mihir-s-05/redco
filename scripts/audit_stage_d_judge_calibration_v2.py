from __future__ import annotations

import argparse
import json
import tarfile
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


def verdict_signature(parsed: dict[str, Any] | None) -> tuple[tuple[str, int], ...]:
    if parsed is None:
        return ()
    return tuple(
        sorted(
            (item["name"], int(item["verdict"]))
            for item in parsed.get("verdicts", [])
        )
    )


def load_rows(
    responses: Path | None, archive: Path | None, member: str
) -> list[dict[str, Any]]:
    if responses is not None:
        text = responses.read_text(encoding="utf-8")
    elif archive is not None:
        with tarfile.open(archive, "r:gz") as handle:
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ValueError(f"archive has no member {member!r}")
            text = extracted.read().decode("utf-8")
    else:
        raise ValueError("set --responses or --archive")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--member",
        default="runs/stage-d0/judge-audit-v1/responses.jsonl",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--strong-threshold", type=int, default=7)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.responses is None) == (args.archive is None):
        raise ValueError("set exactly one of --responses or --archive")

    rows = load_rows(args.responses, args.archive, args.member)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    parse_failures = sum(row["parsed"] is None for row in rows)
    numeric_disagreements = 0
    reason_text_variations = 0
    expected_labels: list[bool] = []
    predicted_labels: list[bool] = []
    for case_id, repeats in sorted(by_case.items()):
        if len(repeats) != args.repeats:
            raise ValueError(
                f"{case_id} has {len(repeats)} repeats, expected {args.repeats}"
            )
        signatures = [verdict_signature(row["parsed"]) for row in repeats]
        numeric_disagreements += int(any(sig != signatures[0] for sig in signatures[1:]))
        parsed_objects = [row["parsed"] for row in repeats]
        reason_text_variations += int(
            any(obj != parsed_objects[0] for obj in parsed_objects[1:])
        )
        verdicts = dict(signatures[0])
        for criterion in ("precision", "recall"):
            expected_labels.append(bool(repeats[0]["expected"][criterion]))
            predicted_labels.append(
                verdicts.get(criterion, 0) >= args.strong_threshold
            )
    accuracy = balanced_accuracy(expected_labels, predicted_labels)
    passes = (
        parse_failures == 0
        and numeric_disagreements == 0
        and accuracy >= args.minimum_balanced_accuracy
    )
    result = {
        "schema_version": 2,
        "historical_result_changed": False,
        "cases": len(by_case),
        "responses": len(rows),
        "parse_failures": parse_failures,
        "numeric_verdict_repeat_disagreements": numeric_disagreements,
        "reason_text_variations": reason_text_variations,
        "balanced_accuracy": accuracy,
        "minimum_balanced_accuracy": args.minimum_balanced_accuracy,
        "decision": "pass" if passes else "fail",
        "note": (
            "The v1 terminal failure is preserved; v2 only separates "
            "decision-bearing verdict drift from free-text reason drift."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
