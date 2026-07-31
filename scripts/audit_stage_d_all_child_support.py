"""Materialize and aggregate Stage D all-child support evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from redco.analysis.stage_d_all_child_support import (
    aggregate_paper_support,
    evaluate_all_precommitted_targets,
    precommit_all_depth_one_targets,
)
from redco.integrations.signed_subprocess import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    precommit = subparsers.add_parser("precommit")
    precommit.add_argument("--trace", type=Path, required=True)
    precommit.add_argument("--output", type=Path, required=True)
    single = subparsers.add_parser("single")
    single.add_argument("--trace", type=Path, required=True)
    single.add_argument("--replay", type=Path, required=True)
    single.add_argument("--scorer", type=Path, required=True)
    single.add_argument("--precommit", type=Path, required=True)
    single.add_argument("--master-seed", required=True)
    single.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--records-dir", type=Path, required=True)
    aggregate.add_argument("--required-paper-successes", type=int, default=58)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "precommit":
        report = precommit_all_depth_one_targets(args.trace)
        exit_code = 0
    elif args.command == "single":
        committed = json.loads(args.precommit.read_text(encoding="utf-8"))
        report = evaluate_all_precommitted_targets(
            trace_path=args.trace,
            replay_path=args.replay,
            scorer_path=args.scorer,
            precommit=committed,
            master_seed=args.master_seed,
        )
        exit_code = 0 if report["paper_joint_pass"] else 21
    else:
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(args.records_dir.glob("*.json"))
        ]
        report = aggregate_paper_support(
            records,
            required_paper_successes=args.required_paper_successes,
        )
        exit_code = 0 if report["passes"] else 22
    atomic_write_json(args.output, report)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
