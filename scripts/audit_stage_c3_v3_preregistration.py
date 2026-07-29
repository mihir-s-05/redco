"""Audit the Stage-C3 v3 preregistration before any live model call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from redco.analysis.stage_c3_v3_preregistration import audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1",
        type=Path,
        default=Path(
            "configs/stage-c3/credit-confusion-live-preregistration-v1.json"
        ),
    )
    parser.add_argument(
        "--v2",
        type=Path,
        default=Path(
            "configs/stage-c3/credit-confusion-live-preregistration-v2.json"
        ),
    )
    parser.add_argument(
        "--v3",
        type=Path,
        default=Path(
            "configs/stage-c3/credit-confusion-live-preregistration-v3.json"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.v1, args.v2, args.v3)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
