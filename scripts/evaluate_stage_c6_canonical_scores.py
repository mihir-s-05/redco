"""Verify canonical replicates or combine canonical final-policy scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from redco.analysis.stage_c6_canonical import (
    combine_action_scores,
    verify_model_identity,
    verify_replicates,
    verify_runtime_support,
)
from redco.integrations.signed_subprocess import atomic_write_json


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-replicates")
    verify.add_argument("--action", action="append", type=Path, required=True)
    verify.add_argument("--root", action="append", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    combine = subparsers.add_parser("combine-actions")
    combine.add_argument("--input", action="append", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    identity = subparsers.add_parser("verify-model-identity")
    identity.add_argument("--reference", type=Path, required=True)
    identity.add_argument("--current", type=Path, required=True)
    identity.add_argument("--output", type=Path, required=True)
    runtime = subparsers.add_parser("verify-runtime-support")
    runtime.add_argument("--candidate", type=Path, required=True)
    runtime.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify-replicates":
        payload = verify_replicates(
            [_load(path) for path in args.action],
            [_load(path) for path in args.root],
        )
    elif args.command == "combine-actions":
        payload = combine_action_scores(
            [_load(path) for path in args.input]
        )
    elif args.command == "verify-model-identity":
        payload = verify_model_identity(
            _load(args.reference),
            _load(args.current),
        )
    else:
        payload = verify_runtime_support(_load(args.candidate))
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload.get("status", "combined"),
                "signature": payload["signed_payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
