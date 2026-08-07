"""Canonical supply-chain trust root for one Stage-D deployment stack."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from redco.contracts import canonical_json

_DOMAIN = "redco-stage-d-dependency-stack-v1"
_COMPONENT_PATCHES = {
    "prime-rl": (
        "prime-rl-redco-stage-c9-practical-efficiency.patch",
        "prime-rl-stage-d-live-update-gate-v1.patch",
        "prime-rl-stage-d-objective-gate-v1.patch",
        "prime-rl-strict-tool-env-guard.patch",
    ),
    "renderers": (
        "renderers-stage-d-prepared-observer-v1.patch",
        "renderers-stage-d-replay-directives-v1.patch",
    ),
    "verifiers": (
        "verifiers-stage-d-provenance-baseline-v1.patch",
        "verifiers-stage-d-prepared-observer-v1.patch",
        "verifiers-stage-d-replay-directives-v1.patch",
        "verifiers-stage-d-sampling-director-v1.patch",
        "verifiers-stage-d-isolated-docker-v1.patch",
        "verifiers-stage-d-pre-generation-preflight-v1.patch",
        "verifiers-stage-d-patched-rlm-archive-v1.patch",
        "verifiers-stage-d-frozen-rlm-install-v1.patch",
        "verifiers-stage-d-observer-failfast-v1.patch",
    ),
    "rlm": (
        "rlm-event-replay-provenance.patch",
        "rlm-mcp-client-symbol-compat.patch",
        "rlm-root-initial-required-tool-choice.patch",
        "rlm-spawn-provenance-v2.patch",
    ),
}
_COMPONENT_ORDER = tuple(_COMPONENT_PATCHES)
_PROGRAMS = (
    "scientific_launcher",
    "trainer_entrypoint",
    "evaluator",
    "reporter",
)

# The prepared-observer patches remain the historical request/trace seam.  The
# await ownership hooks are deliberately applied last, after every downstream
# renderer/verifier patch that touches the same call sites.
_LIVE_OWNER_COMPONENTS = (
    (
        "renderers",
        "bdb96b0c84a307e2b71c6a366c9d718c3ac7fe78",
        (
            "renderers-stage-d-prepared-observer-v1.patch",
            "renderers-stage-d-replay-directives-v1.patch",
            "renderers-stage-d-watchdog-owner-v1.patch",
        ),
        "bd43d515c12dcaa1e1c0279941a1397d4ffba31a1557d6d7342a1322b195fcc4",
    ),
    (
        "verifiers",
        "b13ba60da63cea91389e7575766b7270d0d11fc5",
        (
            "verifiers-stage-d-provenance-baseline-v1.patch",
            "verifiers-stage-d-prepared-observer-v1.patch",
            "verifiers-stage-d-replay-directives-v1.patch",
            "verifiers-stage-d-sampling-director-v1.patch",
            "verifiers-stage-d-isolated-docker-v1.patch",
            "verifiers-stage-d-pre-generation-preflight-v1.patch",
            "verifiers-stage-d-patched-rlm-archive-v1.patch",
            "verifiers-stage-d-frozen-rlm-install-v1.patch",
            "verifiers-stage-d-observer-failfast-v1.patch",
            "verifiers-stage-d-watchdog-owner-v1.patch",
        ),
        "9dcf9e98dea73c2487d2165cd6cae35dc61fb66e00d377d85d5466886b3ea4e0",
    ),
)
_LIVE_OWNER_PATCH_SHA256S = {
    "renderers-stage-d-prepared-observer-v1.patch": (
        "1f10cff27b6097a70474926f916129948d1c176c1c8f12692c6e5498eb06163c"
    ),
    "renderers-stage-d-replay-directives-v1.patch": (
        "ce0c2fa4112a3fd8aafb7c3bf4aa18807e1f558d08a4b8a814272e2f100e73b6"
    ),
    "renderers-stage-d-watchdog-owner-v1.patch": (
        "3f3f3a30462a1cd81195a4dac60ffd5c845865b58e960028d32fa00ac59cea38"
    ),
    "verifiers-stage-d-provenance-baseline-v1.patch": (
        "2b9eeb758f7bdf3726763b9923eb6b3a83d89be9caa9804119a7c31465cb6c64"
    ),
    "verifiers-stage-d-prepared-observer-v1.patch": (
        "10292432e3324f8b086e352e56f516223e2cb733bcddafc4b43d385d33d0517f"
    ),
    "verifiers-stage-d-replay-directives-v1.patch": (
        "8b76b8638cb3d4435b240de67eccda8bce17c33a8f312b258623d5359985a714"
    ),
    "verifiers-stage-d-sampling-director-v1.patch": (
        "6a21db821d1f52647e284813ff71d4177be79ac5538b9ed3ce31d47b2af87ddb"
    ),
    "verifiers-stage-d-isolated-docker-v1.patch": (
        "c69961c89bd50b6597515ede4e28d68a5331574a477057328fa5c7455b231d83"
    ),
    "verifiers-stage-d-pre-generation-preflight-v1.patch": (
        "767708a04b360f4ec34377c195d029a3e4461e118898aa6e09487e80db931017"
    ),
    "verifiers-stage-d-patched-rlm-archive-v1.patch": (
        "c76f332c48e4a65a0524f15174c89825cfb636527e1f179be31fadc9e0997535"
    ),
    "verifiers-stage-d-frozen-rlm-install-v1.patch": (
        "136cc89de669acf50145cbe486520f0290da084516deca6a43c2dfbaeb679fa3"
    ),
    "verifiers-stage-d-observer-failfast-v1.patch": (
        "63c19112c1cdaf142339b95d8891fd384867cbc7dd7a23d9f7cd7c9055d4acb4"
    ),
    "verifiers-stage-d-watchdog-owner-v1.patch": (
        "3c7977d0d0f2781fc96bfec0154e4c86a40df64b163d2e63fbed5d4381ae1c19"
    ),
}

# These are the only Verifiers component bindings already frozen before the
# fail-fast patch existed. Binding the exception to their complete canonical
# payload prevents a newly authored manifest from silently selecting the old
# patch sequence.
_LEGACY_VERIFIERS_COMPONENT_SHA256S = frozenset(
    {
        "3cabac587d8c0f539bb0046304dae15f36e99665e39d3032469ab5a43701c3a8",
        "a92032d021336d9a99e6ed4f8c28b20fb4d6df4516cebf7579950b4d0a410d9c",
    }
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _git_run(repository: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _reconstruct_live_owner_tree_sha256(
    root: Path,
    *,
    name: str,
    base_commit: str,
    patch_names: tuple[str, ...],
) -> str:
    """Rebuild one clean pinned dependency stack and hash its canonical tree."""

    root = root.resolve()
    source = root / "external" / "prime-rl" / "deps" / name
    if not (source / ".git").exists():
        raise ValueError(f"pinned dependency checkout is absent: {source}")
    with tempfile.TemporaryDirectory(prefix=f"redco-{name}-stack-") as temporary:
        target = Path(temporary) / name
        subprocess.run(
            [
                "git",
                "clone",
                "--no-hardlinks",
                "--no-checkout",
                str(source),
                str(target),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _git_run(target, "checkout", "--detach", base_commit)
        for patch_name in patch_names:
            patch = root / "patches" / patch_name
            subprocess.run(
                ["git", "apply", "--check", str(patch)],
                cwd=target,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["git", "apply", str(patch)],
                cwd=target,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        return _sha256(canonical_tree_manifest_bytes(target))


def _git_tracked_modes(root: Path) -> dict[str, str]:
    """Read executable bits from Git so manifests do not depend on host modes."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--stage", "-z"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    modes: dict[str, str] = {}
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, relative_bytes = record.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0]
        relative = relative_bytes.decode("utf-8")
        if mode == b"100755":
            modes[relative] = "0755"
        elif mode == b"100644":
            modes[relative] = "0644"
    return modes


def live_owner_dependency_payload(root: Path) -> dict[str, Any]:
    """Authenticate the final composable renderer/verifier owner patches."""

    root = root.resolve()
    components: list[dict[str, Any]] = []
    for name, base_commit, patch_names, post_tree_sha256 in _LIVE_OWNER_COMPONENTS:
        patches: list[dict[str, str]] = []
        for patch_name in patch_names:
            path = root / "patches" / patch_name
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"missing live owner patch: {patch_name}")
            actual = _sha256(path.read_bytes())
            expected = _LIVE_OWNER_PATCH_SHA256S[patch_name]
            if actual != expected:
                raise ValueError(f"live owner patch changed: {patch_name}")
            patches.append({"name": patch_name, "sha256": actual})
        reconstructed = _reconstruct_live_owner_tree_sha256(
            root,
            name=name,
            base_commit=base_commit,
            patch_names=patch_names,
        )
        if reconstructed != post_tree_sha256:
            raise ValueError(f"live owner post-tree hash changed: {name}")
        components.append(
            {
                "base_commit": base_commit,
                "name": name,
                "patches": patches,
                "post_tree_sha256": post_tree_sha256,
            }
        )
    return {
        "schema_version": 1,
        "domain": "redco-stage-d-live-owner-stack-v1",
        "components": components,
    }


def _require_git_commit(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase Git commit")
    return value


def canonical_tree_manifest_bytes(
    root: Path,
    *,
    allow_relative_symlinks: bool = False,
    excluded_roots: tuple[str, ...] = (),
) -> bytes:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("dependency tree root must be a regular directory")
    normalized_exclusions: list[str] = []
    for exclusion in excluded_roots:
        excluded_path = PurePosixPath(exclusion)
        if (
            not exclusion
            or excluded_path.is_absolute()
            or "." in excluded_path.parts
            or ".." in excluded_path.parts
            or "\\" in exclusion
            or excluded_path.as_posix() != exclusion
        ):
            raise ValueError("dependency exclusion root is unsafe")
        normalized_exclusions.append(exclusion)
    if len(normalized_exclusions) != len(set(normalized_exclusions)):
        raise ValueError("dependency exclusion roots must be unique")
    tracked_modes = _git_tracked_modes(root)
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if ".git" in PurePosixPath(relative).parts or any(
            relative == excluded or relative.startswith(f"{excluded}/")
            for excluded in normalized_exclusions
        ):
            continue
        if path.is_symlink():
            if not allow_relative_symlinks:
                raise ValueError("dependency tree contains a symbolic member")
            target = os.readlink(path)
            if PurePosixPath(target).is_absolute():
                raise ValueError("dependency tree contains an absolute symbolic link")
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(root.resolve())
            except ValueError as error:
                raise ValueError("dependency symbolic link escapes its tree") from error
            entries.append(
                {
                    "path": relative,
                    "type": "symlink",
                    "mode": "0777",
                    "target": target.replace("\\", "/"),
                }
            )
            continue
        if not (path.is_file() or path.is_dir()):
            raise ValueError("dependency tree contains a special member")
        if path.is_dir():
            continue
        raw = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "type": "file",
                "mode": tracked_modes.get(
                    relative,
                    "0755" if path.stat().st_mode & 0o111 else "0644",
                ),
                "size": len(raw),
                "sha256": _sha256(raw),
            }
        )
    if not entries:
        raise ValueError("dependency tree contains no regular files")
    return cast(
        bytes,
        canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-canonical-tree-v1",
                "entries": entries,
            }
        ),
    )


def write_canonical_tree_tar(
    root: Path,
    output: Path,
    *,
    allow_relative_symlinks: bool = False,
) -> str:
    manifest = json.loads(
        canonical_tree_manifest_bytes(
            root,
            allow_relative_symlinks=allow_relative_symlinks,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(f".{output.name}.pending")
    with pending.open("wb") as raw:
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as archive:
            for entry in manifest["entries"]:
                info = tarfile.TarInfo(entry["path"])
                info.mode = int(entry["mode"], 8)
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                info.pax_headers = {}
                if entry["type"] == "symlink":
                    info.type = tarfile.SYMTYPE
                    info.linkname = entry["target"]
                    info.size = 0
                    archive.addfile(info)
                else:
                    value = (root / entry["path"]).read_bytes()
                    info.size = len(value)
                    archive.addfile(info, io.BytesIO(value))
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(pending, output)
    return _sha256(output.read_bytes())


def write_canonical_tree_tar_gzip(
    root: Path,
    output: Path,
    *,
    allow_relative_symlinks: bool = False,
) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_tar = output.with_name(f".{output.name}.raw-tar")
    pending = output.with_name(f".{output.name}.pending")
    write_canonical_tree_tar(
        root,
        raw_tar,
        allow_relative_symlinks=allow_relative_symlinks,
    )
    try:
        with raw_tar.open("rb") as source, pending.open("wb") as raw_output:
            with gzip.GzipFile(
                fileobj=raw_output,
                mode="wb",
                filename="",
                mtime=0,
                compresslevel=9,
            ) as compressed:
                shutil.copyfileobj(source, compressed, length=1024 * 1024)
            raw_output.flush()
            os.fsync(raw_output.fileno())
        os.replace(pending, output)
    finally:
        if raw_tar.exists():
            raw_tar.unlink()
    return _sha256(output.read_bytes())


@dataclass(frozen=True, slots=True)
class PatchBinding:
    name: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.name or PurePosixPath(self.name).name != self.name:
            raise ValueError("dependency patch name is unsafe")
        _require_sha256(self.sha256, "dependency patch sha256")

    def to_payload(self) -> dict[str, str]:
        return {"name": self.name, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ComponentBinding:
    name: str
    base_commit: str
    patches: tuple[PatchBinding, ...]
    post_tree_sha256: str

    def __post_init__(self) -> None:
        if self.name not in _COMPONENT_PATCHES:
            raise ValueError("dependency component is not frozen")
        _require_git_commit(self.base_commit, f"{self.name} base commit")
        patch_names = tuple(patch.name for patch in self.patches)
        expected = _COMPONENT_PATCHES[self.name]
        component_sha256 = _sha256(canonical_json(self.to_payload()))
        legacy_without_failfast = (
            self.name == "verifiers"
            and patch_names == expected[:-1]
            and component_sha256 in _LEGACY_VERIFIERS_COMPONENT_SHA256S
        )
        if patch_names != expected and not legacy_without_failfast:
            raise ValueError("dependency component patch order differs")
        _require_sha256(self.post_tree_sha256, f"{self.name} post-tree sha256")

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_commit": self.base_commit,
            "patches": [patch.to_payload() for patch in self.patches],
            "post_tree_sha256": self.post_tree_sha256,
        }

    @classmethod
    def from_payload(cls, value: object) -> ComponentBinding:
        expected = {"name", "base_commit", "patches", "post_tree_sha256"}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("dependency component fields differ")
        patches = value["patches"]
        if not isinstance(patches, list):
            raise ValueError("dependency component patches must be a list")
        parsed = []
        for patch in patches:
            if not isinstance(patch, dict) or set(patch) != {"name", "sha256"}:
                raise ValueError("dependency patch fields differ")
            parsed.append(PatchBinding(patch["name"], patch["sha256"]))
        return cls(value["name"], value["base_commit"], tuple(parsed), value["post_tree_sha256"])


@dataclass(frozen=True, slots=True)
class ImportedModuleBinding:
    name: str
    absolute_path: str
    sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.absolute_path)
        if not self.name or not self.name.isprintable() or not path.is_absolute():
            raise ValueError("imported module binding is invalid")
        _require_sha256(self.sha256, "imported module sha256")

    def to_payload(self) -> dict[str, str]:
        return {
            "name": self.name,
            "absolute_path": self.absolute_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class StageDDependencyStackManifest:
    redco_commit: str
    redco_tree_sha256: str
    components: tuple[ComponentBinding, ...]
    rlm_archive_sha256: str
    rlm_uv_binary_sha256: str
    rlm_uv_cache_archive_sha256: str
    rlm_executable_sha256: str
    uv_lock_sha256: str
    container_image: str
    runtime_manifest_sha256: str
    imported_modules: tuple[ImportedModuleBinding, ...]
    program_sha256s: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_git_commit(self.redco_commit, "Redco commit")
        for name in (
            "redco_tree_sha256",
            "rlm_archive_sha256",
            "rlm_uv_binary_sha256",
            "rlm_uv_cache_archive_sha256",
            "rlm_executable_sha256",
            "uv_lock_sha256",
            "runtime_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if tuple(component.name for component in self.components) != _COMPONENT_ORDER:
            raise ValueError("dependency components use the wrong order")
        if "@sha256:" not in self.container_image or not self.container_image.isprintable():
            raise ValueError("dependency container image must be digest pinned")
        module_keys = tuple(binding.name for binding in self.imported_modules)
        if module_keys != tuple(sorted(module_keys)) or len(module_keys) != len(set(module_keys)):
            raise ValueError("imported module bindings must be sorted and unique")
        if tuple(name for name, _ in self.program_sha256s) != _PROGRAMS:
            raise ValueError("dependency programs use the wrong order")
        for name, digest in self.program_sha256s:
            _require_sha256(digest, f"program {name}")

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self.to_bytes())

    def to_bytes(self) -> bytes:
        return cast(
            bytes,
            canonical_json(
                {
                    "schema_version": 1,
                    "domain": _DOMAIN,
                    "redco_commit": self.redco_commit,
                    "redco_tree_sha256": self.redco_tree_sha256,
                    "components": [component.to_payload() for component in self.components],
                    "rlm_archive_sha256": self.rlm_archive_sha256,
                    "rlm_uv_binary_sha256": self.rlm_uv_binary_sha256,
                    "rlm_executable_sha256": self.rlm_executable_sha256,
                    "uv_lock_sha256": self.uv_lock_sha256,
                    "container_image": self.container_image,
                    "runtime_manifest_sha256": self.runtime_manifest_sha256,
                    "imported_modules": [item.to_payload() for item in self.imported_modules],
                    "program_sha256s": dict(self.program_sha256s),
                }
            ),
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDDependencyStackManifest:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("dependency stack is not JSON") from error
        expected = {
            "schema_version",
            "domain",
            "redco_commit",
            "redco_tree_sha256",
            "components",
            "rlm_archive_sha256",
            "rlm_uv_binary_sha256",
            "rlm_uv_cache_archive_sha256",
            "rlm_executable_sha256",
            "uv_lock_sha256",
            "container_image",
            "runtime_manifest_sha256",
            "imported_modules",
            "program_sha256s",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload.get("schema_version") != 1
            or payload.get("domain") != _DOMAIN
            or canonical_json(payload) != value
            or not isinstance(payload.get("components"), list)
            or not isinstance(payload.get("imported_modules"), list)
            or not isinstance(payload.get("program_sha256s"), dict)
        ):
            raise ValueError("dependency stack is noncanonical or has different fields")
        modules = []
        for item in payload["imported_modules"]:
            if not isinstance(item, dict) or set(item) != {"name", "absolute_path", "sha256"}:
                raise ValueError("imported module binding fields differ")
            modules.append(ImportedModuleBinding(**item))
        programs = payload["program_sha256s"]
        return cls(
            redco_commit=payload["redco_commit"],
            redco_tree_sha256=payload["redco_tree_sha256"],
            components=tuple(
                ComponentBinding.from_payload(item) for item in payload["components"]
            ),
            rlm_archive_sha256=payload["rlm_archive_sha256"],
            rlm_uv_binary_sha256=payload["rlm_uv_binary_sha256"],
            rlm_uv_cache_archive_sha256=payload["rlm_uv_cache_archive_sha256"],
            rlm_executable_sha256=payload["rlm_executable_sha256"],
            uv_lock_sha256=payload["uv_lock_sha256"],
            container_image=payload["container_image"],
            runtime_manifest_sha256=payload["runtime_manifest_sha256"],
            imported_modules=tuple(modules),
            program_sha256s=tuple((name, programs[name]) for name in _PROGRAMS),
        )


__all__ = [
    "ComponentBinding",
    "ImportedModuleBinding",
    "PatchBinding",
    "StageDDependencyStackManifest",
    "canonical_tree_manifest_bytes",
    "live_owner_dependency_payload",
    "write_canonical_tree_tar",
    "write_canonical_tree_tar_gzip",
]
