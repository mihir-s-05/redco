"""Versioned Stage D successor CLI with the two fresh split names."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

from redco_evidence_selection_v2 import run_feasibility as base

SUCCESSOR_SPLITS = ("successor_fixture", "successor_support")


def parse_args() -> argparse.Namespace:
    original: Callable[..., Any] = argparse.ArgumentParser.add_argument

    def add_argument(
        parser: argparse.ArgumentParser, *names: str, **kwargs: Any
    ) -> argparse.Action:
        if names == ("--split",):
            choices = tuple(kwargs.get("choices") or ())
            kwargs["choices"] = (*choices, *SUCCESSOR_SPLITS)
        return original(parser, *names, **kwargs)

    with patch.object(argparse.ArgumentParser, "add_argument", add_argument):
        return base.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        base.setup_logging("INFO", log_file=str(args.output_dir) + ".log")
    return asyncio.run(base.run_grouped(args))


if __name__ == "__main__":
    raise SystemExit(main())
