"""Apply and verify the frozen Stage-D dependency stack on native storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from redco.analysis.stage_d_dependency_stack import (
    StageDDependencyStackManifest,
    canonical_tree_manifest_bytes,
)
from redco.analysis.stage_d_protocol_manifest import StageDProtocolManifest


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _run(*argv: str, cwd: Path) -> str:
    return subprocess.run(
        argv,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tree_sha256(root: Path, excluded_roots: tuple[str, ...] = ()) -> str:
    return _sha256(
        canonical_tree_manifest_bytes(root, excluded_roots=excluded_roots)
    )


def _gitlinks(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in _run("git", "ls-files", "--stage", cwd=root).splitlines():
        metadata, path = line.split("\t", 1)
        mode, commit, _stage = metadata.split()
        if mode == "160000":
            result[path] = commit
    return result


def _require_clean(root: Path, name: str) -> None:
    status = _run("git", "status", "--porcelain", "--untracked-files=all", cwd=root)
    ignored = _run("git", "clean", "-ndx", cwd=root)
    if status or ignored:
        raise RuntimeError(f"{name} dependency is not pristine")


def _apply_component(
    *,
    root: Path,
    repo: Path,
    component: dict[str, object],
    expected_tree: str,
    excluded_roots: tuple[str, ...] = (),
) -> str:
    name = str(component["name"])
    if _run("git", "rev-parse", "HEAD", cwd=root) != component["base_commit"]:
        raise RuntimeError(f"{name} base commit differs")
    actual = _tree_sha256(root, excluded_roots)
    if actual == expected_tree:
        return actual
    patch_paths: list[Path] = []
    for binding in component["patches"]:  # type: ignore[index]
        patch = repo / "patches" / binding["name"]
        if _sha256(patch.read_bytes()) != binding["sha256"]:
            raise RuntimeError(f"{name} patch bytes differ: {patch.name}")
        _run("git", "apply", "--check", str(patch), cwd=root)
        patch_paths.append(patch)
    for patch in patch_paths:
        _run("git", "apply", str(patch), cwd=root)
    _run("git", "diff", "--check", cwd=root)
    actual = _tree_sha256(root, excluded_roots)
    if actual != expected_tree:
        raise RuntimeError(f"{name} patched tree differs: {actual} != {expected_tree}")
    return actual


def verify(
    repo: Path,
    stack_path: Path,
    amendment_path: Path,
    *,
    expected_amendment_sha256: str,
    protocol_path: Path,
    expected_protocol_sha256: str,
) -> dict[str, object]:
    stack_bytes = stack_path.read_bytes()
    amendment_bytes = amendment_path.read_bytes()
    protocol_bytes = protocol_path.read_bytes()
    stack = StageDDependencyStackManifest.from_bytes(stack_bytes)
    protocol = StageDProtocolManifest.from_bytes(protocol_bytes)
    amendment = json.loads(amendment_bytes)
    amendment_sha256 = _sha256(amendment_bytes)
    protocol_sha256 = _sha256(protocol_bytes)
    stack_sha256 = _sha256(stack_bytes)
    if amendment_sha256 != expected_amendment_sha256:
        raise RuntimeError("deployment amendment external digest differs")
    if protocol_sha256 != expected_protocol_sha256:
        raise RuntimeError("protocol external digest differs")
    if protocol.dependency_stack_sha256 != stack_sha256:
        raise RuntimeError("protocol binds a different dependency stack")
    if amendment["dependency_stack_sha256"] != stack_sha256:
        raise RuntimeError("deployment amendment binds a different dependency stack")
    verifier = repo / amendment["verifier"]["path"]
    if _sha256(verifier.read_bytes()) != amendment["verifier"]["sha256"]:
        raise RuntimeError("deployment verifier bytes differ")

    prime = repo / "external/prime-rl"
    expected_gitlinks = amendment["prime_binding"]["excluded_gitlinks"]
    if _gitlinks(prime) != expected_gitlinks:
        raise RuntimeError("Prime gitlink map differs")
    roots = {
        "prime-rl": prime,
        "renderers": prime / "deps/renderers",
        "verifiers": prime / "deps/verifiers",
    }
    for path, commit in expected_gitlinks.items():
        child = prime / path
        if _run("git", "rev-parse", "HEAD", cwd=child) != commit:
            raise RuntimeError(f"submodule HEAD differs: {path}")
    for path in amendment["prime_binding"]["pristine_unpatched_gitlinks"]:
        _require_clean(prime / path, path)

    observed: dict[str, str] = {}
    for binding in stack.components:
        name = binding.name
        if name == "rlm":
            continue
        expected = binding.post_tree_sha256
        if (
            name == "prime-rl"
            and amendment["prime_binding"]["post_tree_sha256"] != expected
        ):
            raise RuntimeError("Prime amendment and dependency stack hashes differ")
        excluded = tuple(expected_gitlinks) if name == "prime-rl" else ()
        observed[name] = _apply_component(
            root=roots[name],
            repo=repo,
            component=binding.to_payload(),
            expected_tree=expected,
            excluded_roots=excluded,
        )
    return {
        "schema_version": 1,
        "domain": "redco-stage-d-dependency-deployment-verification-v1",
        "dependency_stack_sha256": stack_sha256,
        "amendment_sha256": amendment_sha256,
        "protocol_sha256": protocol_sha256,
        "component_tree_sha256s": observed,
        "gitlinks": expected_gitlinks,
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dependency-stack", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--expected-amendment-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(
        args.repo.resolve(),
        args.dependency_stack.resolve(),
        args.amendment.resolve(),
        expected_amendment_sha256=args.expected_amendment_sha256,
        protocol_path=args.protocol.resolve(),
        expected_protocol_sha256=args.expected_protocol_sha256,
    )
    encoded = _canonical(report) + b"\n"
    if args.output:
        args.output.write_bytes(encoded)
    print(encoded.decode(), end="")


if __name__ == "__main__":
    main()
