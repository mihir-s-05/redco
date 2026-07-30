"""Verify the frozen Stage-C6 v3 credit-confusion scientific campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from redco.analysis.stage_c3_live import verify_campaign

RUNS = {
    "confusion_irrelevant": (9921, 9922),
    "confusion_redundant": (9923,),
    "confusion_lucky": (9924,),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_campaign(
        args.run_root,
        args.scores,
        run_seeds=RUNS,
        analysis="stage-c6-credit-confusion-live-v3",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
