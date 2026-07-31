from __future__ import annotations

import argparse
import json
from pathlib import Path

from redco.analysis.stage_d_target_eligibility import (
    aggregate_support,
    evaluate_target,
)
from redco.integrations.signed_subprocess import atomic_write_json


def _single(args: argparse.Namespace) -> int:
    report = evaluate_target(
        trace_path=args.trace,
        replay_path=args.replay,
        scorer_path=args.scorer,
    )
    atomic_write_json(args.output, report)
    return 0 if report["eligible"] else 21


def _aggregate(args: argparse.Namespace) -> int:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.records_dir.glob("*.json"))
    ]
    report = aggregate_support(records)
    atomic_write_json(args.output, report)
    return 0 if report["passes"] else 22


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    single = subparsers.add_parser("single")
    single.add_argument("--trace", type=Path, required=True)
    single.add_argument("--replay", type=Path, required=True)
    single.add_argument("--scorer", type=Path, required=True)
    single.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--records-dir", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = _single(args) if args.command == "single" else _aggregate(args)
    raise SystemExit(result)


if __name__ == "__main__":
    main()
