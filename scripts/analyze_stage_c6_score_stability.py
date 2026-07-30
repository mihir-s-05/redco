"""Analyze Stage-C5 versus Stage-C6 scores for byte-identical merged weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from redco.analysis.stage_c6_stability import analyze
from redco.integrations.signed_subprocess import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-c5-action", type=Path, required=True)
    parser.add_argument("--stage-c5-root", type=Path, required=True)
    parser.add_argument("--stage-c5-merge-manifest", type=Path, required=True)
    parser.add_argument("--stage-c6-action", type=Path, required=True)
    parser.add_argument("--stage-c6-root", type=Path, required=True)
    parser.add_argument("--stage-c6-merge-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        stage_c5_action=args.stage_c5_action,
        stage_c5_root=args.stage_c5_root,
        stage_c5_merge_manifest=args.stage_c5_merge_manifest,
        stage_c6_action=args.stage_c6_action,
        stage_c6_root=args.stage_c6_root,
        stage_c6_merge_manifest=args.stage_c6_merge_manifest,
        dataset_manifest=args.dataset_manifest,
    )
    atomic_write_json(args.output, result)
    print(json.dumps({"status": result["status"], "signature": result["signed_payload_sha256"]}))


if __name__ == "__main__":
    main()
