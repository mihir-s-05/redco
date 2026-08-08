"""Build or strictly check the candidate/materialized v13 support launch set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redco.analysis.stage_d_v13_draft_publication import atomic_publish_set  # noqa: E402
from redco.analysis.stage_d_v13_support_launch import (  # noqa: E402
    LAUNCH_AUDIT_RELATIVE,
    _bundle_immutable_paths,
    build_launch_artifacts,
    verify_launch_bundle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve()
    output_root = (args.output_root or repository).resolve()
    payloads = build_launch_artifacts(repository)
    if args.check_only:
        hashes = verify_launch_bundle(repository, output_root)
    else:
        hashes = atomic_publish_set(
            output_root,
            payloads,
            immutable_paths=_bundle_immutable_paths(repository),
            manifest_path=LAUNCH_AUDIT_RELATIVE,
            require_draft_envelope=False,
        )
        hashes = verify_launch_bundle(repository, output_root)
    print(json.dumps({"check_only": args.check_only, "hashes": hashes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
