"""A small, source-safe MuSiQue-Ans depth-feasibility gate.

The snapshot retains full selected documents and scorer-only answer data.  The
policy seam receives only the question and document evidence; decomposition,
support flags, answers, and aliases never enter a choice prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import string
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

Depth = Literal[2, 4]
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_TASK_ID = re.compile(r"^(2hop|4hop1)__([0-9]+(?:_[0-9]+)*)$")


def _finite_float(value: object, message: str) -> float:
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

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Candidate:
        if set(value) != {"title", "text"}:
            raise ValueError("candidate has the wrong schema")
        title = value["title"]
        text = value["text"]
        if type(title) is not str or not title or type(text) is not str or not text:
            raise ValueError("candidate fields must be non-empty strings")
        return cls(title, text)


@dataclass(frozen=True, slots=True)
class Choice:
    """A policy's parsed numbered choice, kept separate from path validity."""

    title: str
    parse_valid: bool


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

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> MuSiQueTask:
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
        if set(value) != expected:
            raise ValueError("MuSiQue task has the wrong schema")
        task_id = value["task_id"]
        hop_count = value["hop_count"]
        question = value["question"]
        answer = value["answer"]
        aliases = value["answer_aliases"]
        raw_candidates = value["candidates"]
        raw_path = value["support_path"]
        raw_components = value["support_components"]
        if (
            type(task_id) is not str
            or not task_id
            or type(hop_count) is not int
            or hop_count not in (2, 4)
            or type(question) is not str
            or not question
            or type(answer) is not str
            or not answer
            or type(aliases) is not list
            or not all(type(item) is str for item in aliases)
            or type(raw_candidates) is not list
            or type(raw_path) is not list
            or type(raw_components) is not list
        ):
            raise ValueError("MuSiQue task has invalid primitive fields")
        if not all(type(item) is str and item for item in (*raw_path, *raw_components)):
            raise ValueError("support path and components must contain non-empty strings")
        if len(raw_path) != hop_count or len(raw_components) != hop_count:
            raise ValueError("support path and component counts must match hop_count")
        task_depth = cast(Depth, hop_count)
        return cls(
            task_id,
            task_depth,
            question,
            answer,
            tuple(cast(str, item) for item in aliases),
            tuple(
                Candidate.from_mapping(cast(Mapping[str, object], item)) for item in raw_candidates
            ),
            tuple(cast(str, item) for item in raw_path),
            tuple(cast(str, item) for item in raw_components),
        )


@dataclass(frozen=True, slots=True)
class GateConfig:
    candidate_count: int = 8
    tasks_per_depth: tuple[tuple[int, int], ...] = ((2, 24), (4, 21))
    planned_k: int = 4
    short_document_token_limit: int = 288
    base_seed: int = 20260815
    sampling_temperature: float = 1.0
    sampling_top_p: float = 0.95
    oracle_answer_f1_min: float = 0.50
    teacher_forced_margin: float = 0.10
    greedy_full_path_min: float = 0.10
    mixed_reward_min: float = 0.20

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GateConfig:
        expected = {
            "candidate_count",
            "generation",
            "planned_k",
            "short_document_token_limit",
            "tasks_per_depth",
            "thresholds",
        }
        if set(value) != expected:
            raise ValueError("capability config has the wrong schema")
        raw_tasks = value["tasks_per_depth"]
        generation = value["generation"]
        thresholds = value["thresholds"]
        if type(raw_tasks) is not dict or set(raw_tasks) != {"2", "4"}:
            raise ValueError("task cohort sizes have the wrong schema")
        if type(generation) is not dict or set(generation) != {
            "base_seed",
            "temperature",
            "top_p",
        }:
            raise ValueError("generation settings have the wrong schema")
        if type(thresholds) is not dict or set(thresholds) != {
            "oracle_answer_f1_min",
            "teacher_forced_margin",
            "greedy_full_path_min",
            "mixed_reward_min",
        }:
            raise ValueError("capability thresholds have the wrong schema")
        ints = (value["candidate_count"], value["planned_k"], value["short_document_token_limit"])
        counts = (raw_tasks["2"], raw_tasks["4"])
        seed = generation["base_seed"]
        if (
            any(type(item) is not int or item <= 0 for item in (*ints, *counts))
            or type(seed) is not int
            or seed < 0
            or type(value["planned_k"]) is not int
            or type(value["candidate_count"]) is not int
            or type(value["short_document_token_limit"]) is not int
        ):
            raise ValueError("capability integer bounds are invalid")
        candidate_count = value["candidate_count"]
        planned_k = value["planned_k"]
        if candidate_count != 8 or planned_k != 4:
            raise ValueError("the first gate requires eight candidates and K=4")
        temperature = _finite_float(generation["temperature"], "generation settings must be finite")
        top_p = _finite_float(generation["top_p"], "generation settings must be finite")
        if temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("generation settings must be bounded")
        raw_thresholds = cast(dict[str, object], thresholds)
        threshold_values = {
            key: _finite_float(raw_thresholds[key], "capability thresholds must be finite numbers")
            for key in raw_thresholds
        }
        return cls(
            candidate_count=candidate_count,
            tasks_per_depth=((2, counts[0]), (4, counts[1])),
            planned_k=planned_k,
            short_document_token_limit=value["short_document_token_limit"],
            base_seed=seed,
            sampling_temperature=temperature,
            sampling_top_p=top_p,
            oracle_answer_f1_min=threshold_values["oracle_answer_f1_min"],
            teacher_forced_margin=threshold_values["teacher_forced_margin"],
            greedy_full_path_min=threshold_values["greedy_full_path_min"],
            mixed_reward_min=threshold_values["mixed_reward_min"],
        )


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    model: str
    revision: str
    max_seconds: int
    choice_max_new_tokens: int
    answer_max_new_tokens: int
    max_input_tokens: int


def _load_manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict or set(value) != {
        "cohort",
        "dataset",
        "gate",
        "inference",
        "schema_version",
        "source",
        "subset",
    }:
        raise ValueError("capability manifest has the wrong schema")
    if (
        value["schema_version"] != 1
        or value["dataset"] != "MuSiQue-Ans"
        or value["subset"] != "linear_chain"
        or value["cohort"] != "short_document_linear_chain"
    ):
        raise ValueError("capability manifest is not the reviewed MuSiQue cohort")
    source = value["source"]
    if type(source) is not dict or set(source) != {
        "archive_sha256",
        "license",
        "official_snapshot_required",
        "raw_dev_bytes",
        "raw_dev_sha256",
        "repository",
        "snapshot_path",
        "snapshot_sha256",
    }:
        raise ValueError("capability source declaration has the wrong schema")
    if source["official_snapshot_required"] is not True:
        raise ValueError("the gate requires an authenticated official snapshot")
    return cast(dict[str, object], value)


def load_gate_config(path: Path) -> GateConfig:
    return GateConfig.from_mapping(cast(Mapping[str, object], _load_manifest(path)["gate"]))


def load_inference_config(path: Path) -> InferenceConfig:
    raw = _load_manifest(path)["inference"]
    if type(raw) is not dict or set(raw) != {
        "answer_max_new_tokens",
        "choice_max_new_tokens",
        "max_input_tokens",
        "max_seconds",
        "model",
        "revision",
    }:
        raise ValueError("inference configuration has the wrong schema")
    model = raw["model"]
    revision = raw["revision"]
    max_seconds = raw["max_seconds"]
    choice_max_new_tokens = raw["choice_max_new_tokens"]
    answer_max_new_tokens = raw["answer_max_new_tokens"]
    max_input_tokens = raw["max_input_tokens"]
    if (
        type(model) is not str
        or model != "Qwen/Qwen3-4B-Instruct-2507"
        or type(revision) is not str
        or revision != "cdbee75f17c01a7cc42f958dc650907174af0554"
        or type(max_seconds) is not int
        or not 1 <= max_seconds <= 3600
        or type(choice_max_new_tokens) is not int
        or choice_max_new_tokens != 4
        or type(answer_max_new_tokens) is not int
        or answer_max_new_tokens != 32
        or type(max_input_tokens) is not int
        or not 1 <= max_input_tokens <= 8192
    ):
        raise ValueError("inference configuration is outside the reviewed bound")
    return InferenceConfig(
        model,
        revision,
        max_seconds,
        choice_max_new_tokens,
        answer_max_new_tokens,
        max_input_tokens,
    )


def _candidate_key(task_id: str, title: str) -> str:
    return hashlib.sha256(f"{task_id}\0{title}".encode()).hexdigest()


_OFFICIAL_ROW_KEYS = {
    "answer",
    "answer_aliases",
    "answerable",
    "id",
    "paragraphs",
    "question",
    "question_decomposition",
}
_OFFICIAL_DECOMPOSITION_KEYS = {"answer", "id", "paragraph_support_idx", "question"}
_OFFICIAL_PARAGRAPH_KEYS = {"idx", "is_supporting", "paragraph_text", "title"}


def _hard_distractor_key(task_id: str, question: str, title: str, text: str) -> tuple[int, str]:
    question_words = set(_TOKEN.findall(question.lower()))
    text_words = set(_TOKEN.findall(f"{title} {text}".lower()))
    overlap = len(question_words & text_words)
    return (-overlap, hashlib.sha256(f"{task_id}\0{title}\0{text}".encode()).hexdigest())


def parse_official_row(value: Mapping[str, object], config: GateConfig) -> MuSiQueTask | None:
    """Normalize one official row, retaining aliases only for the scorer."""
    if set(value) != _OFFICIAL_ROW_KEYS:
        raise ValueError("official MuSiQue row has the wrong schema")
    task_id = value["id"]
    question = value["question"]
    answer = value["answer"]
    aliases = value["answer_aliases"]
    answerable = value["answerable"]
    raw_decomposition = value["question_decomposition"]
    raw_paragraphs = value["paragraphs"]
    if (
        type(task_id) is not str
        or not task_id
        or type(question) is not str
        or not question
        or type(answer) is not str
        or not answer
        or type(aliases) is not list
        or not all(type(item) is str for item in aliases)
        or type(answerable) is not bool
        or not answerable
        or type(raw_decomposition) is not list
        or type(raw_paragraphs) is not list
    ):
        raise ValueError("official MuSiQue row has invalid primitive fields")
    depth = len(raw_decomposition)
    if depth not in (2, 4):
        return None
    expected_prefix = "2hop__" if depth == 2 else "4hop1__"
    decomposition: list[tuple[int, int]] = []
    for item in raw_decomposition:
        if type(item) is not dict or set(item) != _OFFICIAL_DECOMPOSITION_KEYS:
            raise ValueError("official decomposition has the wrong schema")
        decomposition_value = cast(dict[str, object], item)
        component_id = decomposition_value["id"]
        support_idx = decomposition_value["paragraph_support_idx"]
        if (
            type(component_id) is not int
            or type(support_idx) is not int
            or type(decomposition_value["question"]) is not str
            or type(decomposition_value["answer"]) is not str
        ):
            raise ValueError("official decomposition has invalid fields")
        decomposition.append((component_id, support_idx))
    expected_id = expected_prefix + "_".join(str(component) for component, _ in decomposition)
    if task_id != expected_id:
        raise ValueError("task id does not encode ordered decomposition ids exactly")
    if len({component for component, _ in decomposition}) != depth:
        raise ValueError("official decomposition component ids must be unique")
    paragraphs: dict[int, tuple[str, str, bool]] = {}
    for item in raw_paragraphs:
        if type(item) is not dict or set(item) != _OFFICIAL_PARAGRAPH_KEYS:
            raise ValueError("official paragraph has the wrong schema")
        paragraph = cast(dict[str, object], item)
        idx = paragraph["idx"]
        title = paragraph["title"]
        text = paragraph["paragraph_text"]
        supporting = paragraph["is_supporting"]
        if (
            type(idx) is not int
            or type(title) is not str
            or not title
            or type(text) is not str
            or not text
            or type(supporting) is not bool
        ):
            raise ValueError("official paragraph has invalid fields")
        if idx in paragraphs:
            raise ValueError("official paragraph indices must be unique")
        paragraphs[idx] = (title, text, supporting)
    support_indices = tuple(support_idx for _, support_idx in decomposition)
    if len(set(support_indices)) != depth or not set(support_indices).issubset(paragraphs):
        raise ValueError("official support indices are invalid")
    marked_support = {idx for idx, (_, _, supporting) in paragraphs.items() if supporting}
    if marked_support != set(support_indices):
        raise ValueError("official support flags do not match the decomposition")
    support_titles = tuple(paragraphs[idx][0] for idx in support_indices)
    if len(set(support_titles)) != depth:
        return None
    distractors = [
        (idx, title, text)
        for idx, (title, text, _) in paragraphs.items()
        if idx not in support_indices
    ]
    distractors.sort(key=lambda item: _hard_distractor_key(task_id, question, item[1], item[2]))
    selected = [(paragraphs[idx][0], paragraphs[idx][1]) for idx in support_indices]
    used_titles = set(support_titles)
    for _, title, text in distractors:
        if title in used_titles:
            continue
        selected.append((title, text))
        used_titles.add(title)
        if len(selected) == config.candidate_count:
            break
    if len(selected) != config.candidate_count:
        return None
    if any(len(_TOKEN.findall(text)) > config.short_document_token_limit for _, text in selected):
        return None
    return MuSiQueTask(
        task_id,
        cast(Depth, depth),
        question,
        answer,
        tuple(cast(str, item) for item in aliases),
        tuple(Candidate(title, text) for title, text in selected),
        support_titles,
        tuple(str(component) for component, _ in decomposition),
    )


def select_official_tasks(
    rows: Iterable[Mapping[str, object]], config: GateConfig
) -> tuple[MuSiQueTask, ...]:
    """Select one global title/component-disjoint short-document cohort."""
    targets = dict(config.tasks_per_depth)
    parsed: dict[int, list[MuSiQueTask]] = {2: [], 4: []}
    for row in rows:
        task_id = row.get("id")
        if type(task_id) is not str:
            raise ValueError("official rows require string ids")
        if task_id.startswith("2hop__") or task_id.startswith("4hop1__"):
            task = parse_official_row(row, config)
            if task is not None:
                parsed[task.hop_count].append(task)
    selected: list[MuSiQueTask] = []
    used_titles: set[str] = set()
    used_components: set[str] = set()
    for depth in (2, 4):
        count = 0
        for task in sorted(parsed[depth], key=lambda item: _candidate_key("select", item.task_id)):
            if (
                set(task.support_path) & used_titles
                or set(task.support_components) & used_components
            ):
                continue
            selected.append(task)
            used_titles.update(task.support_path)
            used_components.update(task.support_components)
            count += 1
            if count == targets[depth]:
                break
        if count != targets[depth]:
            raise ValueError(f"official source lacks {targets[depth]} disjoint {depth}-hop rows")
    return normalize_tasks(tuple(selected), config)


def normalize_tasks(tasks: Sequence[MuSiQueTask], config: GateConfig) -> tuple[MuSiQueTask, ...]:
    """Validate the snapshot and impose deterministic candidate ordering."""
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("task identifiers must be unique")
    normalized: list[MuSiQueTask] = []
    all_titles: set[str] = set()
    all_components: set[str] = set()
    for task in tasks:
        prefix = "2hop__" if task.hop_count == 2 else "4hop1__"
        expected_id = prefix + "_".join(task.support_components)
        if task.task_id != expected_id:
            raise ValueError("task id and ordered support components do not match")
        if len(task.candidates) != config.candidate_count:
            raise ValueError("every task requires exactly eight candidates")
        if len(task.support_path) != task.hop_count:
            raise ValueError("support path length must equal hop_count")
        titles = tuple(candidate.title for candidate in task.candidates)
        if len(set(titles)) != len(titles):
            raise ValueError("candidate titles must be unique")
        if len(set(task.support_path)) != len(task.support_path):
            raise ValueError("support path may not repeat a document")
        if (
            len(task.support_components) != task.hop_count
            or len(set(task.support_components)) != task.hop_count
        ):
            raise ValueError("support components must be unique and match hop_count")
        if not set(task.support_path).issubset(titles):
            raise ValueError("support path must refer to candidate titles")
        if len(set(task.answer_aliases)) != len(task.answer_aliases):
            raise ValueError("answer aliases must be exact and unique")
        if any(
            len(_TOKEN.findall(candidate.text)) > config.short_document_token_limit
            for candidate in task.candidates
        ):
            raise ValueError("candidate exceeds the short-document selection rule")
        if set(task.support_path) & all_titles or set(task.support_components) & all_components:
            raise ValueError("support titles and components must be globally disjoint")
        all_titles.update(task.support_path)
        all_components.update(task.support_components)
        ordered = tuple(
            sorted(task.candidates, key=lambda item: _candidate_key(task.task_id, item.title))
        )
        normalized.append(
            MuSiQueTask(
                task.task_id,
                task.hop_count,
                task.question,
                task.answer,
                task.answer_aliases,
                ordered,
                task.support_path,
                task.support_components,
            )
        )
    counts = {depth: sum(task.hop_count == depth for task in normalized) for depth in (2, 4)}
    expected_counts = dict(config.tasks_per_depth)
    if counts != expected_counts:
        raise ValueError(f"snapshot depth counts {counts} do not equal {expected_counts}")
    return tuple(sorted(normalized, key=lambda task: (task.hop_count, task.task_id)))


def load_tasks(path: Path, config: GateConfig) -> tuple[MuSiQueTask, ...]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict or set(value) != {
        "cohort",
        "dataset",
        "schema_version",
        "tasks",
    }:
        raise ValueError("MuSiQue snapshot has the wrong schema")
    if (
        value["dataset"] != "MuSiQue-Ans"
        or value["schema_version"] != 1
        or value["cohort"] != "short_document_linear_chain"
    ):
        raise ValueError("snapshot is not the reviewed MuSiQue cohort")
    rows = value["tasks"]
    if type(rows) is not list:
        raise ValueError("MuSiQue snapshot tasks must be a list")
    return normalize_tasks(
        tuple(MuSiQueTask.from_mapping(cast(Mapping[str, object], row)) for row in rows), config
    )


class CapabilityPolicy(Protocol):
    def choose(
        self,
        task_id: str,
        step: int,
        question: str,
        evidence: tuple[Candidate, ...],
        candidates: tuple[Candidate, ...],
        *,
        sample: int | None,
        sampling_seed: int | None,
    ) -> Choice: ...

    def answer(
        self,
        task_id: str,
        question: str,
        evidence: tuple[Candidate, ...],
        *,
        mode: Literal["greedy", "oracle"],
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class _PathResult:
    positions: tuple[str | None, ...]
    selected: tuple[str, ...]
    parse_valid: bool
    path_valid: bool
    prefix_length: int
    conditional_reach: tuple[bool, ...]
    conditional_correct: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class SampledOutcome:
    positions: tuple[str | None, ...]
    parse_valid: bool
    path_valid: bool
    prefix_length: int
    normalized_prefix_reward: float
    terminal_reward: float
    conditional_reach: tuple[bool, ...]
    conditional_correct: tuple[bool, ...]
    outcome_support: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class TaskMetrics:
    task_id: str
    hop_count: Depth
    greedy_positions: tuple[str | None, ...]
    greedy_choice_parse_valid: bool
    greedy_path_valid: bool
    greedy_prefix_length: int
    greedy_full_path: bool
    greedy_answer_nonempty: bool
    greedy_answer_f1: float
    oracle_answer_nonempty: bool
    oracle_answer_f1: float
    teacher_choice_parse_valid: int
    teacher_correct: int
    teacher_total: int
    sampled_outcomes: tuple[SampledOutcome, ...]

    @property
    def mixed_reward(self) -> bool:
        return len({outcome.normalized_prefix_reward for outcome in self.sampled_outcomes}) > 1


def normalize_answer(text: str) -> str:
    punctuation = str.maketrans("", "", string.punctuation)
    words = text.lower().translate(punctuation).split()
    return " ".join(word for word in words if word not in {"a", "an", "the"})


def _tokens(text: str) -> Counter[str]:
    return Counter(normalize_answer(text).split())


def token_f1(prediction: str, reference: str) -> float:
    predicted = _tokens(prediction)
    expected = _tokens(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def answer_f1(prediction: str, references: Sequence[str]) -> float:
    return max((token_f1(prediction, reference) for reference in references), default=0.0)


def parse_numbered_choice(raw: object, candidates: Sequence[Candidate]) -> Choice:
    if type(raw) is not str or re.fullmatch(r"[1-9][0-9]*", raw.strip()) is None:
        return Choice("", False)
    index = int(raw.strip())
    if not 1 <= index <= len(candidates):
        return Choice("", False)
    return Choice(candidates[index - 1].title, True)


def deterministic_seed(base_seed: int, task_id: str, hop: int, sample: int) -> int:
    value = f"{base_seed}\0{task_id}\0{hop}\0{sample}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _docs(task: MuSiQueTask, titles: Sequence[str]) -> tuple[Candidate, ...]:
    by_title = {candidate.title: candidate for candidate in task.candidates}
    return tuple(by_title[title] for title in titles)


def _path(
    policy: CapabilityPolicy,
    task: MuSiQueTask,
    config: GateConfig,
    *,
    sample: int | None,
) -> _PathResult:
    positions: list[str | None] = []
    selected: list[str] = []
    parse_valid = True
    path_valid = True
    prefix_correct = True
    conditional_reach: list[bool] = []
    conditional_correct: list[bool] = []
    for step in range(task.hop_count):
        evidence = _docs(task, selected)
        candidates = tuple(
            candidate for candidate in task.candidates if candidate.title not in selected
        )
        seed = (
            None
            if sample is None
            else deterministic_seed(config.base_seed, task.task_id, step + 1, sample)
        )
        choice = policy.choose(
            task.task_id,
            step + 1,
            task.question,
            evidence,
            candidates,
            sample=sample,
            sampling_seed=seed,
        )
        choice_is_parsed = type(choice) is Choice and choice.parse_valid
        parse_valid = parse_valid and choice_is_parsed
        reach = prefix_correct
        valid_choice = choice_is_parsed and choice.title in {
            candidate.title for candidate in candidates
        }
        current_correct = valid_choice and choice.title == task.support_path[step]
        conditional_reach.append(reach)
        conditional_correct.append(reach and current_correct)
        if not valid_choice:
            positions.append(None)
            path_valid = False
        else:
            positions.append(choice.title)
            selected.append(choice.title)
        if not current_correct:
            prefix_correct = False
    prefix_length = sum(1 for correct in conditional_correct if correct)
    return _PathResult(
        positions=tuple(positions),
        selected=tuple(selected),
        parse_valid=parse_valid,
        path_valid=path_valid,
        prefix_length=prefix_length,
        conditional_reach=tuple(conditional_reach),
        conditional_correct=tuple(conditional_correct),
    )


def _sampled_outcome(task: MuSiQueTask, path: _PathResult) -> SampledOutcome:
    support = tuple(
        path.positions[step] == task.support_path[step] and path.prefix_length >= step + 1
        for step in range(task.hop_count)
    )
    return SampledOutcome(
        positions=path.positions,
        parse_valid=path.parse_valid,
        path_valid=path.path_valid,
        prefix_length=path.prefix_length,
        normalized_prefix_reward=path.prefix_length / task.hop_count,
        terminal_reward=float(path.positions == task.support_path),
        conditional_reach=path.conditional_reach,
        conditional_correct=path.conditional_correct,
        outcome_support=support,
    )


def evaluate_task(task: MuSiQueTask, policy: CapabilityPolicy, config: GateConfig) -> TaskMetrics:
    greedy = _path(policy, task, config, sample=None)
    greedy_answer = policy.answer(
        task.task_id, task.question, _docs(task, greedy.selected), mode="greedy"
    )
    if type(greedy_answer) is not str:
        greedy_answer = ""
    oracle_answer = policy.answer(
        task.task_id, task.question, _docs(task, task.support_path), mode="oracle"
    )
    if type(oracle_answer) is not str:
        oracle_answer = ""
    teacher_parse = 0
    teacher_correct = 0
    for step, expected in enumerate(task.support_path):
        evidence = _docs(task, task.support_path[:step])
        candidates = tuple(
            candidate
            for candidate in task.candidates
            if candidate.title not in task.support_path[:step]
        )
        choice = policy.choose(
            task.task_id,
            step + 1,
            task.question,
            evidence,
            candidates,
            sample=None,
            sampling_seed=None,
        )
        if type(choice) is Choice and choice.parse_valid:
            teacher_parse += 1
            teacher_correct += int(choice.title == expected)
    sampled = tuple(
        _sampled_outcome(task, _path(policy, task, config, sample=sample))
        for sample in range(config.planned_k)
    )
    return TaskMetrics(
        task_id=task.task_id,
        hop_count=task.hop_count,
        greedy_positions=greedy.positions,
        greedy_choice_parse_valid=greedy.parse_valid,
        greedy_path_valid=greedy.path_valid,
        greedy_prefix_length=greedy.prefix_length,
        greedy_full_path=greedy.positions == task.support_path,
        greedy_answer_nonempty=bool(greedy_answer.strip()),
        greedy_answer_f1=answer_f1(greedy_answer, (task.answer, *task.answer_aliases)),
        oracle_answer_nonempty=bool(oracle_answer.strip()),
        oracle_answer_f1=answer_f1(oracle_answer, (task.answer, *task.answer_aliases)),
        teacher_choice_parse_valid=teacher_parse,
        teacher_correct=teacher_correct,
        teacher_total=task.hop_count,
        sampled_outcomes=sampled,
    )


def _random_teacher_accuracy(depth: int, candidate_count: int) -> float:
    return sum(1.0 / (candidate_count - step) for step in range(depth)) / depth


def _metric_record(metric: TaskMetrics) -> dict[str, object]:
    return {
        "task_id": metric.task_id,
        "hop_count": metric.hop_count,
        "greedy": {
            "positions": metric.greedy_positions,
            "choice_parse_valid": metric.greedy_choice_parse_valid,
            "path_constructed": metric.greedy_path_valid,
            "prefix_length": metric.greedy_prefix_length,
            "full_path": metric.greedy_full_path,
            "answer_nonempty": metric.greedy_answer_nonempty,
            "answer_f1": metric.greedy_answer_f1,
        },
        "oracle": {
            "answer_nonempty": metric.oracle_answer_nonempty,
            "answer_f1": metric.oracle_answer_f1,
        },
        "teacher_forced": {
            "choice_parse_valid": metric.teacher_choice_parse_valid,
            "correct": metric.teacher_correct,
            "total": metric.teacher_total,
        },
        "sampled": [
            {
                "positions": outcome.positions,
                "parse_valid": outcome.parse_valid,
                "path_constructed": outcome.path_valid,
                "prefix_length": outcome.prefix_length,
                "normalized_prefix_reward": outcome.normalized_prefix_reward,
                "terminal_reward": outcome.terminal_reward,
                "conditional_reach": outcome.conditional_reach,
                "conditional_correct": outcome.conditional_correct,
                "outcome_support": outcome.outcome_support,
            }
            for outcome in metric.sampled_outcomes
        ],
    }


def _depth_summary(metrics: Sequence[TaskMetrics], config: GateConfig) -> dict[str, object]:
    if not metrics:
        raise ValueError("each depth requires at least one task")
    teacher_total = sum(metric.teacher_total for metric in metrics)
    sampled = [outcome for metric in metrics for outcome in metric.sampled_outcomes]
    per_hop: dict[str, dict[str, float | int]] = {}
    for step in range(max(metric.hop_count for metric in metrics)):
        reached = sum(
            outcome.conditional_reach[step]
            for outcome in sampled
            if step < len(outcome.conditional_reach)
        )
        correct = sum(
            outcome.conditional_correct[step]
            for outcome in sampled
            if step < len(outcome.conditional_correct)
        )
        support = sum(
            outcome.outcome_support[step]
            for outcome in sampled
            if step < len(outcome.outcome_support)
        )
        denominator = len(sampled)
        conditional_accuracy = correct / reached if reached else 0.0
        per_hop[str(step + 1)] = {
            "reached_count": reached,
            "correct_count": correct,
            "conditional_accuracy": conditional_accuracy,
            "both_outcomes_observed": 0 < correct < reached,
            "conditional_reach": reached,
            "conditional_correct": correct,
            "outcome_support": support,
            "reach_rate": reached / denominator,
            "correct_rate": correct / denominator,
            "outcome_support_rate": support / denominator,
        }
    informative = sum(metric.mixed_reward for metric in metrics)
    terminal_mixed = sum(
        len({outcome.terminal_reward for outcome in metric.sampled_outcomes}) > 1
        for metric in metrics
    )
    return {
        "tasks": len(metrics),
        "oracle_answer_f1": sum(metric.oracle_answer_f1 for metric in metrics) / len(metrics),
        "greedy_answer_f1": sum(metric.greedy_answer_f1 for metric in metrics) / len(metrics),
        "oracle_answer_nonempty_rate": sum(metric.oracle_answer_nonempty for metric in metrics)
        / len(metrics),
        "greedy_answer_nonempty_rate": sum(metric.greedy_answer_nonempty for metric in metrics)
        / len(metrics),
        "choice_parse_valid_rate": sum(metric.greedy_choice_parse_valid for metric in metrics)
        / len(metrics),
        "path_construction_rate": sum(metric.greedy_path_valid for metric in metrics)
        / len(metrics),
        "valid_prefix_mean": sum(metric.greedy_prefix_length for metric in metrics) / len(metrics),
        "full_support_path_rate": sum(metric.greedy_full_path for metric in metrics) / len(metrics),
        "teacher_forced_choice_parse_valid_rate": sum(
            metric.teacher_choice_parse_valid for metric in metrics
        )
        / teacher_total,
        "teacher_forced_hop_accuracy": sum(metric.teacher_correct for metric in metrics)
        / teacher_total,
        "random_teacher_forced_accuracy": _random_teacher_accuracy(
            metrics[0].hop_count, config.candidate_count
        ),
        "candidate_updates": len(metrics),
        "informative_candidate_updates": informative,
        "mixed_reward_rate": informative / len(metrics),
        "mixed_reward_definition": (
            "share of task updates whose K=4 normalized support-prefix rewards are nonconstant"
        ),
        "terminal_mixed_reward_rate_descriptive": terminal_mixed / len(metrics),
        "sampled_per_hop": per_hop,
    }


def evaluate_gate(
    tasks: Sequence[MuSiQueTask], policy: CapabilityPolicy, config: GateConfig | None = None
) -> dict[str, object]:
    if config is None:
        config = GateConfig()
    normalized = normalize_tasks(tasks, config)
    metrics = tuple(evaluate_task(task, policy, config) for task in normalized)
    by_depth = {
        depth: [metric for metric in metrics if metric.hop_count == depth] for depth in (2, 4)
    }
    summaries = {str(depth): _depth_summary(by_depth[depth], config) for depth in (2, 4)}
    blocked: list[str] = []
    for depth in (2, 4):
        summary = summaries[str(depth)]
        oracle = cast(float, summary["oracle_answer_f1"])
        teacher = cast(float, summary["teacher_forced_hop_accuracy"])
        random = cast(float, summary["random_teacher_forced_accuracy"])
        mixed = cast(float, summary["mixed_reward_rate"])
        greedy_path = cast(float, summary["full_support_path_rate"])
        if oracle < config.oracle_answer_f1_min:
            blocked.append(f"{depth}-hop oracle answer F1 below {config.oracle_answer_f1_min:.2f}")
        if teacher <= random + config.teacher_forced_margin:
            blocked.append(f"{depth}-hop teacher accuracy is not above random plus margin")
        if mixed < config.mixed_reward_min:
            blocked.append(f"{depth}-hop prefix reward support below {config.mixed_reward_min:.2f}")
        if greedy_path <= config.greedy_full_path_min:
            blocked.append(
                f"{depth}-hop greedy full-path rate is not above {config.greedy_full_path_min:.2f}"
            )
    choice_calls = sum(task.hop_count for task in normalized)
    return {
        "schema_version": 1,
        "dataset": "MuSiQue-Ans",
        "subset": "linear_chain",
        "cohort": "short_document_linear_chain",
        "candidate_count": config.candidate_count,
        "planned_k": config.planned_k,
        "passed": not blocked,
        "blocked_reasons": blocked,
        "by_hop": summaries,
        "records": [_metric_record(metric) for metric in metrics],
        "generation": {
            "choice_calls": {
                "greedy": choice_calls,
                "teacher_forced": choice_calls,
                "sampled": choice_calls * config.planned_k,
            },
            "answer_calls": len(normalized) * 2,
            "planned_total": choice_calls * (2 + config.planned_k) + len(normalized) * 2,
            "sampling": {
                "greedy": {"do_sample": False},
                "sampled": {
                    "do_sample": True,
                    "temperature": config.sampling_temperature,
                    "top_p": config.sampling_top_p,
                    "seed": "sha256(base_seed, task_id, hop, sample)",
                },
            },
            "gold_fields_in_policy": False,
        },
        "answer_normalization": (
            "lowercase; remove ASCII punctuation; remove a/an/the; normalize whitespace; "
            "max F1 over answer and aliases"
        ),
    }


__all__ = [
    "Candidate",
    "CapabilityPolicy",
    "Choice",
    "GateConfig",
    "InferenceConfig",
    "MuSiQueTask",
    "SampledOutcome",
    "answer_f1",
    "deterministic_seed",
    "evaluate_gate",
    "evaluate_task",
    "load_gate_config",
    "load_inference_config",
    "load_tasks",
    "normalize_answer",
    "normalize_tasks",
    "parse_numbered_choice",
    "parse_official_row",
    "select_official_tasks",
    "token_f1",
]
