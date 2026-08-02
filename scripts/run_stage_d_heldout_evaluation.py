#!/usr/bin/env python3
"""Run the bounded single-owner Stage-D held-out evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

from redco.analysis.stage_d_evaluation_supervisor_loop import (
    run_evaluation_supervisor,
)
from redco.analysis.stage_d_handoff_coordinator import StageDHandoffCoordinator
from redco.contracts import canonical_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    result = run_evaluation_supervisor(
        coordinator=StageDHandoffCoordinator(arguments.handoff_root),
        handoff_root=arguments.handoff_root,
        evaluation_root=arguments.evaluation_root,
    )
    print(
        canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-evaluation-supervisor-result-v1",
                "disposition": result.disposition,
                "ledger_head_sha256": result.ledger_head_sha256,
                "ledger_record_count": result.ledger_record_count,
                "block_code": result.block_code,
            }
        ).decode("utf-8")
    )


if __name__ == "__main__":
    main()
