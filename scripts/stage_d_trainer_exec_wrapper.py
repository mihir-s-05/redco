#!/usr/bin/env python3
"""Write the durable same-PID Stage-D process receipt, then exec the trainer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import cast

from redco.analysis.stage_d_objective_binding import ArmName
from redco.analysis.stage_d_process_supervision import write_preexec_receipt


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--arm", choices=("stock", "branch-global", "local"), required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--environment-manifest-sha256", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    command = tuple(args.command)
    if command and command[0] == "--":
        command = command[1:]
    write_preexec_receipt(
        args.receipt,
        arm=cast(ArmName, args.arm),
        launch_id=args.launch_id,
        argv=command,
        environment_manifest_sha256=args.environment_manifest_sha256,
    )
    os.execvpe(command[0], command, os.environ.copy())


if __name__ == "__main__":
    main()
