"""Capture or check the raw read-only launch-time Prime observation.

This script is intentionally not run by the CPU implementation task.  The
orchestrator runs it only after the reviewed bundle is committed and the Prime
account/context are available.  No resource or wallet facts are CLI inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redco.analysis.stage_d_v13_draft import sha256_bytes  # noqa: E402
from redco.analysis.stage_d_v13_launch_observations import (  # noqa: E402
    PrimeObservationProducer,
    validate_prime_observation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    repository = args.repository.resolve()
    output = args.output.resolve()
    if args.capture:
        value = PrimeObservationProducer(repository).capture_to(output)
        print(json.dumps({"mode": "capture", "sha256": sha256_bytes(value)}, sort_keys=True))
    else:
        parsed = validate_prime_observation(repository, output)
        print(
            json.dumps(
                {"mode": "check-only", "commit": parsed["bundle"]["commit"]},
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
