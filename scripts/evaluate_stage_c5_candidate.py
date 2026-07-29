"""Evaluate or aggregate constrained Stage-C5 warm-start candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from redco.analysis.stage_c5_constrained import (
    evaluate_constrained_candidate,
    select_constrained_candidates,
)
from redco.integrations.signed_subprocess import atomic_write_json


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--step", type=int, required=True)
    evaluate.add_argument("--action-scores", type=Path, required=True)
    evaluate.add_argument("--root-scores", type=Path, required=True)
    evaluate.add_argument("--dataset-manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--candidate-report", action="append", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "evaluate":
        payload = evaluate_constrained_candidate(
            step=args.step,
            action_scores=_load(args.action_scores),
            root_scores=_load(args.root_scores),
            dataset_manifest=_load(args.dataset_manifest),
        )
    else:
        payload = select_constrained_candidates(
            [_load(path) for path in args.candidate_report]
        )
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "signature": payload["signed_payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
