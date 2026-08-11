"""Authenticated Git execution and repository-status owner for the Prime one-shot."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path
from typing import cast

GIT_EXECUTABLE: dict[str, str | int] = {
    "path": r"C:\Program Files\Git\mingw64\bin\git.exe",
    "bytes": 4_149_624,
    "sha256": "51c6331aab2426ae2df187975590587b5a10042e3423f4bc0fdcb54aeb3efab7",
}
GIT_LAUNCHER: dict[str, str | int] = {
    "path": r"C:\Program Files\Git\cmd\git.exe",
    "bytes": 46_968,
    "sha256": "f668c4ba88417ecdf29470b3af92d576a701cc0f76dd083b13d032f4b3f1f247",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authenticate_git_executable() -> Path:
    harmless = {"GIT_FLUSH", "GIT_OPTIONAL_LOCKS", "GIT_PAGER", "GIT_TERMINAL_PROMPT"}
    if any(key.startswith("GIT_") and key not in harmless for key in os.environ):
        raise ValueError("Prime one-shot Git environment redirects are forbidden")
    for label, binding in (("launcher", GIT_LAUNCHER), ("owner", GIT_EXECUTABLE)):
        path = Path(cast(str, binding["path"]))
        info = path.lstat()
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if (
            path.is_symlink()
            or getattr(info, "st_file_attributes", 0) & reparse
            or not stat.S_ISREG(info.st_mode)
            or info.st_size != binding["bytes"]
            or _sha256_file(path) != binding["sha256"]
        ):
            raise ValueError(f"Prime one-shot Git {label} differs")
    return Path(cast(str, GIT_EXECUTABLE["path"]))


def _git_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
    )
    return environment


def _authenticate_git_metadata(root: Path) -> None:
    marker = root / ".git"
    marker_info = marker.lstat()
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if marker.is_symlink() or getattr(marker_info, "st_file_attributes", 0) & reparse:
        raise ValueError("Prime one-shot Git metadata alias is forbidden")
    if stat.S_ISDIR(marker_info.st_mode):
        git_dir = marker
    elif stat.S_ISREG(marker_info.st_mode) and marker_info.st_nlink == 1:
        raw = marker.read_text(encoding="utf-8").strip()
        if not raw.startswith("gitdir: "):
            raise ValueError("Prime one-shot Git metadata file differs")
        git_dir = (root / raw.removeprefix("gitdir: ")).resolve()
        info = git_dir.lstat()
        if git_dir.is_symlink() or getattr(info, "st_file_attributes", 0) & reparse:
            raise ValueError("Prime one-shot Git metadata target is aliased")
    else:
        raise ValueError("Prime one-shot Git metadata differs")
    for relative in (
        "info/grafts",
        "shallow",
        "objects/info/alternates",
        "objects/info/http-alternates",
    ):
        candidate = git_dir / relative
        if candidate.exists() or candidate.is_symlink():
            raise ValueError("Prime one-shot Git object substitution is forbidden")


def _git_argv(root: Path, *arguments: str) -> tuple[str, ...]:
    executable = authenticate_git_executable()
    _authenticate_git_metadata(root)
    return (
        str(executable),
        "--no-replace-objects",
        "-c",
        f"safe.directory={root.resolve()}",
        "-C",
        str(root),
        *arguments,
    )


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        _git_argv(root, *arguments),
        check=True,
        capture_output=True,
        text=True,
        env=_git_environment(),
    ).stdout.strip()


def git_status(root: Path) -> set[tuple[str, str]]:
    raw = subprocess.run(
        _git_argv(root, "status", "--porcelain=v1", "-z", "--untracked-files=all"),
        check=True,
        capture_output=True,
        env=_git_environment(),
    ).stdout
    records: set[tuple[str, str]] = set()
    for item in raw.split(b"\0"):
        if not item:
            continue
        if len(item) < 4 or item[2:3] != b" ":
            raise ValueError("Prime one-shot Git status is malformed")
        records.add((item[:2].decode("ascii"), item[3:].decode("utf-8")))
    return records


__all__ = [
    "GIT_EXECUTABLE",
    "GIT_LAUNCHER",
    "authenticate_git_executable",
    "git_output",
    "git_status",
]
