"""Render one frozen Stage-C3 arm config from a committed template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from redco.analysis.stage_c3_config import ARMS, PROBES, render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--probe", choices=sorted(PROBES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--run-root",
        default="runs/stage-c3/credit-confusion-live",
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = render(
        args.template,
        args.output,
        arm=args.arm,
        probe=args.probe,
        seed=args.seed,
        run_root=args.run_root,
    )
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
