"""Verify the frozen support launch bundle without authorizing science."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redco.analysis.stage_d_v13_support_launch import (  # noqa: E402
    execute_support_once,
    preflight_validate,
    summarize_bundle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute-once", action="store_true")
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight-snapshot", type=Path)
    parser.add_argument("--preflight-observation", type=Path)
    parser.add_argument("--pod-runtime-observation", type=Path)
    parser.add_argument("--capability", type=Path)
    parser.add_argument("--capability-signature", type=Path)
    parser.add_argument("--synthetic-preflight", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve()
    output_root = (args.output_root or repository).resolve()
    if args.execute_once:
        if args.preflight_observation is None or args.pod_runtime_observation is None:
            raise SystemExit(
                "execute-once requires raw Prime and pod runtime observations"
            )
        if args.synthetic_preflight or args.preflight_snapshot is not None:
            raise SystemExit("synthetic preflight cannot authorize execute-once")
        if args.capability is None or args.capability_signature is None:
            raise SystemExit("execute-once requires the fixed signed handoff and signature")
        hashes = execute_support_once(
            repository,
            preflight_observation=args.preflight_observation.resolve(),
            pod_runtime_observation=args.pod_runtime_observation.resolve(),
            capability=(
                args.capability.resolve()
            ),
            capability_signature=args.capability_signature.resolve(),
        )
        print(json.dumps({"mode": "execute-once", "hashes": hashes}, sort_keys=True))
        return 0
    if args.preflight_only:
        observation = args.preflight_observation or args.preflight_snapshot
        if observation is None:
            raise SystemExit(
                "preflight-only requires a raw Prime observation or explicit synthetic fixture"
            )
        if args.synthetic_preflight and args.preflight_observation is not None:
            raise SystemExit("choose raw or synthetic preflight, not both")
        if not args.synthetic_preflight and args.preflight_snapshot is not None:
            raise SystemExit("--preflight-snapshot requires --synthetic-preflight")
        hashes = preflight_validate(
            repository,
            observation.resolve(),
            require_post_commit=not args.synthetic_preflight,
            runtime_observation_path=(
                args.pod_runtime_observation.resolve()
                if args.pod_runtime_observation is not None
                else None
            ),
            synthetic=args.synthetic_preflight,
        )
        print(json.dumps({"mode": "preflight-only", "hashes": hashes}, sort_keys=True))
        return 0
    summary = summarize_bundle(repository, output_root)
    summary["mode"] = "verify-only"
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
