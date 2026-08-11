"""Build/check the non-authorizing current-lineage Prime one-shot successor."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from redco.analysis import stage_d_v13_prime_test_one_shot_contract_v2 as v2
from redco.analysis.stage_d_v13_prime_test_one_shot_successor_v3 import (
    AUTHORIZATION_PATH,
    PARENT_COMMIT,
    ROOT,
    authenticate_successor,
    authorization_value,
    build_readiness_artifacts,
    canonical_json,
    current_head,
    publish_exclusive,
    sha256_bytes,
    verify_readiness_artifacts,
)


def _replace(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.urandom(16).hex()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(*, check_only: bool, prepare_authorization: bool) -> dict[str, str]:
    root = ROOT
    head = current_head(root)
    if prepare_authorization:
        if check_only or head == PARENT_COMMIT:
            raise ValueError("authorization preparation requires a committed v3 readiness child")
        lineage = v2.git_output(root, "rev-list", "--parents", "-n", "1", head).split()
        if lineage != [head, PARENT_COMMIT]:
            raise ValueError("authorization preparation requires a readiness predecessor")
        readiness_commit = head
        authenticate_successor(root, committed=True, head_override=readiness_commit)
        target = root / AUTHORIZATION_PATH
        raw = canonical_json(authorization_value(root, readiness_commit=readiness_commit))
        publish_exclusive(target, raw)
        return {AUTHORIZATION_PATH: sha256_bytes(raw)}

    committed = head != PARENT_COMMIT
    authenticate_successor(root, committed=committed)
    if check_only:
        return verify_readiness_artifacts(root, committed=committed)
    built = build_readiness_artifacts(root, committed=committed)
    for relative, raw in built.items():
        _replace(root / relative, raw)
    return verify_readiness_artifacts(root, committed=committed)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check-only", action="store_true")
    group.add_argument("--prepare-authorization", action="store_true")
    arguments = parser.parse_args()
    result = build(
        check_only=arguments.check_only,
        prepare_authorization=arguments.prepare_authorization,
    )
    print(canonical_json(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
