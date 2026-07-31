from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import sign_payload

REQUIRED_FIELDS = {
    "example_id",
    "paper_id",
    "title",
    "question",
    "split",
    "paper",
    "reference_evidence",
    "answer_type",
}
ALLOWED_ANSWER_TYPES = {"abstractive", "extractive", "yes_no"}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("fixture must contain at least one row")
    return rows


def audit_fixture(
    path: Path,
    expected_sha256: str,
    *,
    parent_path: Path | None = None,
) -> dict[str, Any]:
    raw = path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "fixture SHA-256 mismatch: "
            f"{observed_sha256} != {expected_sha256}"
        )

    rows = _load_rows(path)
    ids: set[str] = set()
    row_reports: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(f"row {index} missing fields: {missing}")
        example_id = row["example_id"]
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"row {index} has invalid example_id")
        if example_id in ids:
            raise ValueError(f"duplicate example_id: {example_id}")
        ids.add(example_id)
        if row["split"] != "audit":
            raise ValueError(f"{example_id} must use split='audit'")
        if row["answer_type"] not in ALLOWED_ANSWER_TYPES:
            raise ValueError(
                f"{example_id} has invalid answer_type: {row['answer_type']!r}"
            )
        if row["answer_type"] != "extractive":
            raise ValueError(
                f"{example_id} synthetic verbatim fixture must be extractive"
            )
        paper = row["paper"]
        evidence = row["reference_evidence"]
        if (
            not isinstance(paper, str)
            or not paper
            or not isinstance(evidence, list)
            or not evidence
            or any(
                not isinstance(span, str) or not span or span not in paper
                for span in evidence
            )
        ):
            raise ValueError(f"{example_id} has invalid reference evidence")
        row_reports.append(
            {
                "row_index": index,
                "example_id": example_id,
                "answer_type": row["answer_type"],
                "reference_spans": len(evidence),
                "passes": True,
            }
        )

    parent_check: dict[str, Any] | None = None
    if parent_path is not None:
        parent_rows = _load_rows(parent_path)
        if len(parent_rows) != len(rows):
            raise ValueError("fixture v2 changes the parent row count")
        for index, (parent, current) in enumerate(
            zip(parent_rows, rows, strict=True)
        ):
            current_without_type = dict(current)
            current_without_type.pop("answer_type")
            if current_without_type != parent:
                raise ValueError(
                    f"fixture v2 changes parent row {index} beyond answer_type"
                )
        parent_check = {
            "fixture": parent_path.as_posix(),
            "fixture_sha256": hashlib.sha256(
                parent_path.read_bytes()
            ).hexdigest(),
            "only_answer_type_added": True,
        }

    payload = {
        "schema_version": 1,
        "fixture": path.as_posix(),
        "fixture_sha256": observed_sha256,
        "rows": row_reports,
        "passes": True,
    }
    if parent_check is not None:
        payload["parent_check"] = parent_check
    return sign_payload(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--parent-fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_fixture(
        args.fixture,
        args.fixture_sha256,
        parent_path=args.parent_fixture,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
