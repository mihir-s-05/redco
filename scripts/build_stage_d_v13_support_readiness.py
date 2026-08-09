"""Build or strictly check the non-authorizing v13 support readiness set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redco.analysis.stage_d_v13_draft_publication import atomic_publish_set  # noqa: E402
from redco.analysis.stage_d_v13_support_readiness import (  # noqa: E402
    FROZEN_COHORT_RELATIVE,
    FROZEN_COHORT_SHA256,
    FROZEN_PLAN_RELATIVE,
    FROZEN_PLAN_SHA256,
    FROZEN_PROTOCOL_RELATIVE,
    FROZEN_PROTOCOL_SHA256,
    HISTORICAL_V1_AUTH_RELATIVE,
    HISTORICAL_V1_AUTH_SHA256,
    HISTORICAL_V12_DEPENDENCY_RELATIVE,
    HISTORICAL_V12_DEPENDENCY_SHA256,
    PHASE2_AUDIT_RELATIVE,
    PHASE2_AUDIT_SHA256,
    READINESS_AUDIT_RELATIVE,
    build_readiness_artifacts,
    validate_local_artifacts,
    verify_readiness_bundle,
)


def _immutable_inputs() -> dict[str, str]:
    return {
        FROZEN_COHORT_RELATIVE: FROZEN_COHORT_SHA256,
        FROZEN_PLAN_RELATIVE: FROZEN_PLAN_SHA256,
        FROZEN_PROTOCOL_RELATIVE: FROZEN_PROTOCOL_SHA256,
        HISTORICAL_V12_DEPENDENCY_RELATIVE: HISTORICAL_V12_DEPENDENCY_SHA256,
        HISTORICAL_V1_AUTH_RELATIVE: HISTORICAL_V1_AUTH_SHA256,
        PHASE2_AUDIT_RELATIVE: PHASE2_AUDIT_SHA256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-local-readiness", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve()
    output_root = (args.output_root or repository).resolve()
    payloads = build_readiness_artifacts(repository)
    if args.check_only:
        hashes = verify_readiness_bundle(repository, output_root)
    else:
        atomic_publish_set(
            output_root,
            payloads,
            immutable_paths=_immutable_inputs() if output_root == repository else {},
            manifest_path=READINESS_AUDIT_RELATIVE,
            require_draft_envelope=False,
        )
        hashes = verify_readiness_bundle(repository, output_root)
    local_readiness: dict[str, int] | str = "not_checked"
    if args.check_local_readiness:
        local_readiness = validate_local_artifacts()
    print(
        json.dumps(
            {
                "check_only": args.check_only,
                "hashes": hashes,
                "local_readiness": local_readiness,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
