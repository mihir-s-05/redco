"""Materialize every address observed before Stage D v4.5 terminated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)


def materialize(
    fewshot_path: Path,
    reload_scores_path: Path,
) -> dict[str, Any]:
    fewshot = json.loads(fewshot_path.read_text(encoding="utf-8"))
    reload_scores = json.loads(
        reload_scores_path.read_text(encoding="utf-8")
    )
    verify_signed_payload(fewshot)
    verify_signed_payload(reload_scores)
    rows = sorted(fewshot["rows"], key=lambda row: row["slot_id"])
    if len(rows) != 64 or len({row["slot_id"] for row in rows}) != 64:
        raise ValueError("few-shot support address set is not exactly 64")
    models = {
        str(model["name"]): model for model in reload_scores["models"]
    }
    scored_cases = sorted(
        {
            str(row["case_id"])
            for model_name in ("original", "reloaded")
            for temperature in ("1.0", "2.0")
            for row in models[model_name]["temperatures"][temperature]
        }
    )
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d0-v4-observed-addresses",
            "fewshot_support": [
                {
                    "slot_id": row["slot_id"],
                    "trace_id": row["trace_id"],
                    "expected_seed": row["expected_seed"],
                }
                for row in rows
            ],
            "sft_optimizer_steps": list(range(1, 9)),
            "co_resident_retention_score_cases": scored_cases,
            "selected_fixture_observed": False,
            "power_audit_addresses_observed": [],
            "scientific_arm_addresses_observed": [],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fewshot", type=Path, required=True)
    parser.add_argument("--reload-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(
        args.output,
        materialize(args.fewshot, args.reload_scores),
    )


if __name__ == "__main__":
    main()
