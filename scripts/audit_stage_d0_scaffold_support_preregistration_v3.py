from __future__ import annotations

import argparse
from pathlib import Path

from redco.analysis.stage_d_support_preregistration import evaluate
from redco.integrations.signed_subprocess import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.protocol)
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit("Stage D support preregistration audit failed")


if __name__ == "__main__":
    main()
