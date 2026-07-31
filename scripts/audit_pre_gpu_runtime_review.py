#!/usr/bin/env python3
"""Exit nonzero unless an exact signed pre-GPU runtime review says GO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from redco.analysis.pre_gpu_runtime_review import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    result = evaluate(
        policy_path=args.policy,
        review_path=args.review,
        expected_commit=args.expected_commit,
    )
    print(json.dumps(result, sort_keys=True))
    if not result["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

