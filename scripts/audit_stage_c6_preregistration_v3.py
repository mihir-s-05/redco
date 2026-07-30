"""Audit the exact-likelihood Stage-C6 v3 protocol before model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from redco.analysis.stage_c6_v3_preregistration import audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "configs/stage-c6/credit-confusion-live-preregistration-v3.json"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.protocol)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
