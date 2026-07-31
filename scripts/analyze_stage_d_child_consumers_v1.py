#!/usr/bin/env python3
"""Diagnose child-result consumers in one recorded Stage D trace."""

from __future__ import annotations

import argparse
from pathlib import Path

from redco.analysis.stage_d_child_consumers import analyze
from redco.integrations.signed_subprocess import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(args.output, analyze(args.trace))


if __name__ == "__main__":
    main()

