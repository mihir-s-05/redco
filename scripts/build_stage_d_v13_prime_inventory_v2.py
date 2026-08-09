"""Build or strictly check non-authorizing Prime inventory evidence v2 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redco.analysis.stage_d_v13_draft_publication import atomic_publish_set  # noqa: E402
from redco.analysis.stage_d_v13_prime_inventory_v2 import (  # noqa: E402
    AUDIT_RELATIVE,
    build_prime_inventory_v2_artifacts,
    verify_prime_inventory_v2_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve()
    output_root = (args.output_root or repository).resolve()
    payloads = build_prime_inventory_v2_artifacts(repository)
    if not args.check_only:
        atomic_publish_set(
            output_root,
            payloads,
            immutable_paths={},
            manifest_path=AUDIT_RELATIVE,
            require_draft_envelope=False,
        )
    hashes = verify_prime_inventory_v2_artifacts(repository, output_root)
    print(json.dumps({"check_only": args.check_only, "hashes": hashes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
