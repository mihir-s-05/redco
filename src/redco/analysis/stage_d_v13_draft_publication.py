"""Atomic draft publication and check-only validation helpers."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes

OUTPUT_RELATIVE_PATHS = frozenset(
    {
        "configs/stage-d/v13-draft/stage-d1-support-preregistration-v13-draft.json",
        "configs/stage-d/v13-draft/stage-d1-support-genesis-v13-draft.json",
        "configs/stage-d/v13-draft/stage-d1-support-config-v13-draft.json",
        "configs/stage-d/v13-draft/stage-d1-support-collection-plan-v13-draft.json",
        "configs/stage-d/v13-draft/stage-d1-support-state-machine-v1.json",
        "datasets/stage-d/qasper-support-successor-v7-draft-retained-only.jsonl",
        "datasets/stage-d/qasper-support-successor-manifest-v7-draft.json",
        "reports/stage-d1-support-v13-observed-information-disclosure-v1.json",
        "reports/stage-d1-support-v13-reserve-selection-receipt-v1.json",
        "reports/stage-d1-support-v13-nonoverlap-audit-v1.json",
        "reports/stage-d1-support-v13-delta-audit-v1.json",
        "reports/stage-d1-support-v13-affordability-ledger-v1.json",
        "reports/stage-d1-support-v13-draft-audit-v1.json",
        "reports/stage-d1-support-v13-draft-artifact-manifest-v1.json",
        "reports/stage-d1-support-v13-draft-cpu-manifest-v1.json",
    }
)


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & 0x400)


def _validate_parent_path(root: Path, parent: Path) -> None:
    root_resolved = root.resolve(strict=True)
    current = parent
    while True:
        if (current.exists() or current.is_symlink()) and _is_link_or_reparse(current):
            raise ValueError(f"draft output ancestor is a symlink or reparse point: {current}")
        resolved = current.resolve(strict=False)
        try:
            resolved.relative_to(root_resolved)
        except ValueError as error:
            raise ValueError(f"draft output ancestor escapes repository: {current}") from error
        if current == root:
            break
        if current.parent == current:
            raise ValueError(f"draft output ancestor does not reach repository root: {parent}")
        current = current.parent
    resolved_parent = parent.resolve(strict=False)
    try:
        resolved_parent.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"draft output parent escapes repository: {parent}") from error


def _validate_uncreated_root(root: Path) -> None:
    """Check a not-yet-created publication root without creating it."""

    if root.exists() or root.is_symlink():
        if _is_link_or_reparse(root):
            raise ValueError(f"draft output root is a symlink or reparse point: {root}")
        if not root.is_dir():
            raise ValueError(f"draft output root is not a directory: {root}")
        return
    current = root
    while not current.exists():
        if current.is_symlink() or _is_link_or_reparse(current):
            raise ValueError(f"draft output ancestor is a symlink or reparse point: {current}")
        if current.parent == current:
            raise ValueError(f"draft output root does not have a safe existing parent: {root}")
        current = current.parent
    if _is_link_or_reparse(current):
        raise ValueError(f"draft output ancestor is a symlink or reparse point: {current}")


def _validate_uncreated_parent(root: Path, parent: Path) -> None:
    """Validate an output parent chain before a fresh root is created."""

    root_resolved = root.resolve(strict=False)
    try:
        parent.resolve(strict=False).relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"draft output parent escapes repository: {parent}") from error
    current = parent
    while current != root:
        if (current.exists() or current.is_symlink()) and _is_link_or_reparse(current):
            raise ValueError(f"draft output ancestor is a symlink or reparse point: {current}")
        if current.parent == current:
            raise ValueError(f"draft output ancestor does not reach repository root: {parent}")
        current = current.parent


def validate_output_paths(
    root: Path,
    immutable_paths: Mapping[str, str],
    *,
    output_paths: Iterable[str] | None = None,
) -> None:
    """Reject symlink, hard-link, and cross-output aliases before publication."""

    approved_outputs = frozenset(output_paths or OUTPUT_RELATIVE_PATHS)
    destinations = [root / relative for relative in sorted(approved_outputs)]
    for destination in destinations:
        _validate_parent_path(root, destination.parent)
        if (destination.exists() or destination.is_symlink()) and _is_link_or_reparse(destination):
            raise ValueError(f"draft output is a symlink or reparse point: {destination}")
    resolved = [destination.resolve(strict=False) for destination in destinations]
    if len(resolved) != len(set(resolved)):
        raise ValueError("draft output paths resolve to aliases")
    for index, destination in enumerate(destinations):
        for other in destinations[index + 1 :]:
            if destination.exists() and other.exists() and os.path.samefile(destination, other):
                raise ValueError(f"draft outputs are hard-link aliases: {destination} and {other}")
    immutable = [
        Path(relative) if Path(relative).is_absolute() else root / relative
        for relative in immutable_paths
    ]
    for destination in destinations:
        resolved_destination = destination.resolve(strict=False)
        for source in immutable:
            if resolved_destination == source.resolve(strict=False):
                raise ValueError(f"draft output aliases immutable input: {destination}")
            if destination.exists() and source.exists() and os.path.samefile(destination, source):
                raise ValueError(f"draft output is a hard-link alias: {destination}")
        if destination.exists() and destination.is_dir():
            raise ValueError(f"draft output is a directory: {destination}")


def canonical_json_payload(value: dict[str, Any]) -> bytes:
    data = cast(bytes, canonical_json_bytes(value))
    if data.endswith(b"\n"):
        raise AssertionError("canonical JSON unexpectedly ended in a newline")
    return data


def atomic_write(
    root: Path,
    relative: str,
    data: bytes,
    *,
    output_paths: Iterable[str] | None = None,
) -> str:
    approved_outputs = frozenset(output_paths or OUTPUT_RELATIVE_PATHS)
    if relative not in approved_outputs:
        raise ValueError(f"refusing to write an unapproved draft path: {relative}")
    destination = root / relative
    _validate_parent_path(root, destination.parent)
    if (destination.exists() or destination.is_symlink()) and _is_link_or_reparse(destination):
        raise ValueError(f"draft output is a symlink or reparse point: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if destination.read_bytes() != data:
        raise AssertionError(f"short draft write: {relative}")
    return cast(str, sha256_bytes(data))


def publication_envelope(value: object, relative: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"draft JSON artifact is not an object: {relative}")
    if value.get("draft_unfrozen") is not True:
        raise ValueError(f"draft JSON artifact is not marked unfrozen: {relative}")
    if value.get("launch_authorized") is not False:
        raise ValueError(f"draft JSON artifact is launch-authorized: {relative}")


def validate_publication(
    actual: Mapping[str, bytes],
    expected: Mapping[str, bytes],
    *,
    output_paths: Iterable[str] | None = None,
    require_draft_envelope: bool = True,
) -> None:
    approved_outputs = frozenset(output_paths or OUTPUT_RELATIVE_PATHS)
    if set(actual) != set(expected) or set(actual) != set(approved_outputs):
        raise ValueError("draft publication artifact set differs")
    for relative in sorted(expected):
        if actual[relative] != expected[relative]:
            raise ValueError(f"draft publication bytes differ: {relative}")
        if relative.endswith(".json"):
            import json

            parsed = json.loads(actual[relative])
            if require_draft_envelope:
                publication_envelope(parsed, relative)
            if actual[relative] != canonical_json_payload(parsed):
                raise ValueError(f"non-canonical draft JSON: {relative}")
        elif relative.endswith(".jsonl"):
            lines = actual[relative].splitlines()
            if not lines or any(not line.strip() for line in lines):
                raise ValueError(f"draft JSONL is empty or contains a blank line: {relative}")
            import json

            for line in lines:
                parsed_line = json.loads(line)
                if not isinstance(parsed_line, dict):
                    raise ValueError(f"draft JSONL row is not an object: {relative}")


def validate_cross_artifact_references(
    values: Mapping[str, object], expected_hashes: Mapping[str, str]
) -> None:
    """Validate every path/hash pair that points to a published draft artifact."""

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(child, str):
                    walk(child)
                    continue
                if key == "path" and isinstance(value.get("sha256"), str):
                    path = child
                    if path in expected_hashes and value["sha256"] != expected_hashes[path]:
                        raise ValueError(f"cross-artifact hash differs for {path}")
                if key.endswith("_sha256"):
                    stem = key[: -len("_sha256")]
                    referenced_path = value.get(stem)
                    if (
                        isinstance(referenced_path, str)
                        and referenced_path in expected_hashes
                        and child != expected_hashes[referenced_path]
                    ):
                        raise ValueError(f"cross-artifact hash differs for {referenced_path}")
            return
        if isinstance(value, list):
            for child in value:
                walk(child)

    walk(values)


def atomic_publish_set(
    root: Path,
    payloads: Mapping[str, bytes],
    *,
    immutable_paths: Mapping[str, str] | None = None,
    manifest_path: str | None = None,
    check_only: bool = False,
    require_draft_envelope: bool = False,
) -> dict[str, str]:
    """Validate a complete set, then publish it with manifest-last recovery.

    ``check_only`` intentionally returns before directory creation, temporary
    staging, or any other write.  Build mode stages every byte on the same
    filesystem, validates the staged set, and replaces the manifest last.  If
    a replacement fails, prior bytes are restored and newly created outputs are
    removed, so a failed build cannot leave a partial publication.
    """

    root = root.resolve(strict=False)
    if not payloads:
        raise ValueError("publication set must not be empty")
    paths = tuple(payloads)
    if len(set(paths)) != len(paths):
        raise ValueError("publication set contains duplicate paths")
    if manifest_path is not None and manifest_path not in payloads:
        raise ValueError("manifest-last path is not in the publication set")
    immutable = immutable_paths or {}
    root_was_missing = not root.exists()
    if check_only or not root_was_missing:
        validate_output_paths(root, immutable, output_paths=paths)
    else:
        _validate_uncreated_root(root)
        for relative in paths:
            _validate_uncreated_parent(root, (root / relative).parent)
    for relative, data in payloads.items():
        if type(data) is not bytes or not data:
            raise ValueError(f"publication payload is not immutable bytes: {relative}")
    hashes = {relative: sha256_bytes(data) for relative, data in payloads.items()}
    if check_only:
        actual: dict[str, bytes] = {}
        for relative in paths:
            path = root / relative
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"published output is missing: {relative}")
            actual[relative] = path.read_bytes()
        validate_publication(
            actual,
            payloads,
            output_paths=paths,
            require_draft_envelope=require_draft_envelope,
        )
        return hashes

    root.mkdir(parents=True, exist_ok=True)
    validate_output_paths(root, immutable, output_paths=paths)
    staging_name = tempfile.mkdtemp(prefix=".stage-d-publication-", dir=root)
    staging = Path(staging_name)
    originals: dict[str, Path | None] = {}
    replaced: list[str] = []
    created_parents: list[Path] = []
    published = False
    try:
        staged_payloads: dict[str, bytes] = {}
        for relative, data in payloads.items():
            staged_path = staging / relative
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_bytes(data)
            staged_payloads[relative] = staged_path.read_bytes()
            destination = root / relative
            if destination.is_file() and not destination.is_symlink():
                backup = staging / ".rollback" / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(destination, backup)
                originals[relative] = backup
            else:
                originals[relative] = None
        validate_publication(
            staged_payloads,
            payloads,
            output_paths=paths,
            require_draft_envelope=require_draft_envelope,
        )
        ordered = [relative for relative in paths if relative != manifest_path]
        if manifest_path is not None:
            ordered.append(manifest_path)
        for relative in ordered:
            destination = root / relative
            parent = destination.parent
            missing: list[Path] = []
            current = parent
            while current != root and not current.exists():
                missing.append(current)
                current = current.parent
            parent.mkdir(parents=True, exist_ok=True)
            created_parents.extend(missing)
            os.replace(staging / relative, destination)
            replaced.append(relative)
        actual = {relative: (root / relative).read_bytes() for relative in paths}
        validate_publication(
            actual,
            payloads,
            output_paths=paths,
            require_draft_envelope=require_draft_envelope,
        )
        published = True
    except BaseException:
        for relative in reversed(replaced):
            destination = root / relative
            original = originals.get(relative)
            if original is None:
                if destination.exists() or destination.is_symlink():
                    destination.unlink()
            else:
                restored = False
                rollback_error: BaseException | None = None
                for _attempt in range(2):
                    try:
                        os.replace(original, destination)
                        restored = True
                        break
                    except BaseException as rollback_failure:
                        rollback_error = rollback_failure
                if not restored:
                    raise RuntimeError(
                        f"publication rollback failed for {relative}"
                    ) from rollback_error
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if not published:
            for parent in sorted(
                set(created_parents), key=lambda path: len(path.parts), reverse=True
            ):
                with contextlib.suppress(OSError):
                    parent.rmdir()
            if root_was_missing:
                with contextlib.suppress(OSError):
                    root.rmdir()
    return hashes


__all__ = [
    "OUTPUT_RELATIVE_PATHS",
    "atomic_publish_set",
    "atomic_write",
    "canonical_json_payload",
    "publication_envelope",
    "validate_cross_artifact_references",
    "validate_output_paths",
    "validate_publication",
]
