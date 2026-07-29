"""Rescore the frozen Stage-C4 v4 curve under finite-choice route semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from redco.analysis.stage_c5_constrained import rescore_v4_bundle
from redco.integrations.signed_subprocess import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = rescore_v4_bundle(args.bundle_root)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "earliest_passing_step": result["earliest_passing_step"],
                "passing_steps": result["passing_steps"],
                "signature": result["signed_payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
