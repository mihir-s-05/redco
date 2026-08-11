"""Build/check the non-authorizing Prime test-only one-shot readiness checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from redco.analysis.stage_d_v13_prime_test_one_shot_contract_v2 import (
    AUTHORIZATION_PATH,
    IMPLEMENTATION_PARENT,
    ROOT,
    authenticate_readiness,
    authorization_value,
    build_readiness_artifacts,
    canonical_json,
    git_output,
    publish_once,
    verify_readiness_artifacts,
)


def _git_head(root: Path) -> str:
    return git_output(root, "rev-parse", "HEAD")


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
    head = _git_head(ROOT)
    if prepare_authorization:
        if check_only or head == IMPLEMENTATION_PARENT:
            raise ValueError("authorization preparation requires committed readiness")
        authenticate_readiness(ROOT, committed=True)
        target = ROOT / AUTHORIZATION_PATH
        if target.exists() or target.is_symlink():
            raise FileExistsError("Prime one-shot authorization already exists")
        raw = canonical_json(authorization_value(ROOT, current_is_authorization=False))
        publish_once(target, raw)
        return {AUTHORIZATION_PATH: __import__("hashlib").sha256(raw).hexdigest()}
    authenticate_readiness(ROOT, committed=head != IMPLEMENTATION_PARENT)
    if check_only:
        return verify_readiness_artifacts(ROOT)
    built = build_readiness_artifacts(ROOT)
    for relative, raw in built.items():
        _replace(ROOT / relative, raw)
    return verify_readiness_artifacts(ROOT)


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
    print(canonical_json(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
