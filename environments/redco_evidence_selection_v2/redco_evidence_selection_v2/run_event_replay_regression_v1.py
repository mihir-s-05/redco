"""Run the event-replay regression through the exact Stage D train client."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

from redco_evidence_selection_v2 import run_feasibility as base

REGRESSION_SPLIT = "event_replay_regression"
_build_production_config = base.build_config


def build_config(args: argparse.Namespace) -> Any:
    """Preserve the production train-client path without scientific changes."""
    return _build_production_config(args)


def parse_args() -> argparse.Namespace:
    original: Callable[..., Any] = argparse.ArgumentParser.add_argument

    def add_argument(
        parser: argparse.ArgumentParser, *names: str, **kwargs: Any
    ) -> argparse.Action:
        if names == ("--split",):
            choices = tuple(kwargs.get("choices") or ())
            kwargs["choices"] = (*choices, REGRESSION_SPLIT)
        return original(parser, *names, **kwargs)

    with patch.object(argparse.ArgumentParser, "add_argument", add_argument):
        return base.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        base.setup_logging("INFO", log_file=str(args.output_dir) + ".log")
    with patch.object(base, "build_config", build_config):
        return asyncio.run(base.run_grouped(args))


if __name__ == "__main__":
    raise SystemExit(main())
