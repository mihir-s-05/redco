"""Audit the frozen Stage-C4 extended factorized selection v4 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from redco.analysis.stage_c4_v4_preregistration import audit
from redco.integrations.signed_subprocess import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--v4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.v3, args.v4)
    atomic_write_json(args.output, report)
    print(json.dumps({"passed": report["passed"]}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
