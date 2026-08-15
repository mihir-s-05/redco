"""Minimal authenticated loader for the frozen MuSiQue capability snapshot."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

Depth = Literal[2, 4]
_TASK_ID = re.compile(r"^(2hop|4hop1)__([0-9]+(?:_[0-9]+)*)$")


def _finite(value: object, message: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(message)
    number = cast(int | float, value)
    if not math.isfinite(float(number)):
        raise ValueError(message)
    return float(number)


@dataclass(frozen=True, slots=True)
class Candidate:
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class MuSiQueTask:
    task_id: str
    hop_count: Depth
    question: str
    answer: str
    answer_aliases: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    support_path: tuple[str, ...]
    support_components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SnapshotConfig:
    candidate_count: int = 8
    task_counts: tuple[tuple[int, int], ...] = ((2, 24), (4, 21))


_DEFAULT_SNAPSHOT_CONFIG = SnapshotConfig()


def _candidate_digest(task_id: str, title: str) -> str:
    return hashlib.sha256(f"{task_id}\0{title}".encode()).hexdigest()


def _task(value: object, config: SnapshotConfig) -> MuSiQueTask:
    if type(value) is not dict:
        raise ValueError("snapshot task must be an object")
    raw = cast(dict[str, object], value)
    expected = {
        "answer",
        "answer_aliases",
        "candidates",
        "hop_count",
        "question",
        "support_components",
        "support_path",
        "task_id",
    }
    if set(raw) != expected:
        raise ValueError("snapshot task has the wrong schema")
    task_id = raw["task_id"]
    depth = raw["hop_count"]
    question = raw["question"]
    answer = raw["answer"]
    aliases = raw["answer_aliases"]
    candidates = raw["candidates"]
    path = raw["support_path"]
    components = raw["support_components"]
    if (
        type(task_id) is not str
        or type(depth) is not int
        or depth not in (2, 4)
        or type(question) is not str
        or not question
        or type(answer) is not str
        or not answer
        or type(aliases) is not list
        or not all(type(item) is str for item in aliases)
        or type(candidates) is not list
        or type(path) is not list
        or type(components) is not list
    ):
        raise ValueError("snapshot task has invalid primitive fields")
    prefix = "2hop__" if depth == 2 else "4hop1__"
    if not _TASK_ID.fullmatch(task_id) or not task_id.startswith(prefix):
        raise ValueError("snapshot task id has the wrong shape")
    if (
        len(path) != depth
        or len(components) != depth
        or not all(type(item) is str and item for item in (*path, *components))
        or len(set(path)) != depth
        or len(set(components)) != depth
        or task_id != prefix + "_".join(cast(str, item) for item in components)
    ):
        raise ValueError("snapshot support path or task id is invalid")
    parsed_candidates: list[Candidate] = []
    for candidate in candidates:
        if type(candidate) is not dict or set(candidate) != {"text", "title"}:
            raise ValueError("snapshot candidate has the wrong schema")
        title = candidate["title"]
        text = candidate["text"]
        if type(title) is not str or not title or type(text) is not str or not text:
            raise ValueError("snapshot candidate fields must be non-empty strings")
        parsed_candidates.append(Candidate(title, text))
    if len(parsed_candidates) != config.candidate_count:
        raise ValueError("snapshot task does not have eight candidates")
    titles = tuple(candidate.title for candidate in parsed_candidates)
    if len(set(titles)) != len(titles) or not set(path).issubset(titles):
        raise ValueError("snapshot candidate titles are invalid")
    expected_order = tuple(sorted(titles, key=lambda title: _candidate_digest(task_id, title)))
    if titles != expected_order:
        raise ValueError("snapshot candidate order is not authenticated")
    return MuSiQueTask(
        task_id,
        cast(Depth, depth),
        question,
        answer,
        tuple(cast(str, item) for item in aliases),
        tuple(parsed_candidates),
        tuple(cast(str, item) for item in path),
        tuple(cast(str, item) for item in components),
    )


def load_tasks(
    path: Path,
    *,
    expected_sha256: str,
    config: SnapshotConfig = _DEFAULT_SNAPSHOT_CONFIG,
) -> tuple[MuSiQueTask, ...]:
    """Load only the reviewed snapshot; no official raw-data path is accepted here."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("MuSiQue snapshot hash does not match the reviewed binding")
    value = json.loads(data)
    if type(value) is not dict or set(value) != {"cohort", "dataset", "schema_version", "tasks"}:
        raise ValueError("MuSiQue snapshot has the wrong schema")
    if (
        value["cohort"] != "short_document_linear_chain"
        or value["dataset"] != "MuSiQue-Ans"
        or value["schema_version"] != 1
        or type(value["tasks"]) is not list
    ):
        raise ValueError("MuSiQue snapshot is not the reviewed cohort")
    tasks = tuple(_task(item, config) for item in cast(list[object], value["tasks"]))
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("snapshot task ids must be unique")
    expected_order = tuple(sorted(tasks, key=lambda task: (task.hop_count, task.task_id)))
    if tasks != expected_order:
        raise ValueError("snapshot task order is not authenticated")
    counts = {depth: sum(task.hop_count == depth for task in tasks) for depth in (2, 4)}
    if counts != dict(config.task_counts):
        raise ValueError("snapshot depth counts are not the reviewed 24/21 cohort")
    titles: set[str] = set()
    components: set[str] = set()
    for task in tasks:
        if set(task.support_path) & titles or set(task.support_components) & components:
            raise ValueError("snapshot support titles/components are not globally disjoint")
        titles.update(task.support_path)
        components.update(task.support_components)
    return tasks


__all__ = ["Candidate", "MuSiQueTask", "SnapshotConfig", "load_tasks"]
