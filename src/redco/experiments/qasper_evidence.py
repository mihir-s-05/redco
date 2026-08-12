"""Deterministic two-decision QASPER evidence-retrieval pilot."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from redco.contracts import canonical_json
from redco.integrity import require_sha256_hex, sha256_bytes

LABELS = ("A", "B", "C", "D")
_WORD = re.compile(r"[A-Za-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _words(text: str) -> set[str]:
    return {match.group().lower() for match in _WORD.finditer(text)}


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(
        part.strip()
        for part in _SENTENCE.split(text)
        if len(part.strip()) >= 25
    )


def _paragraphs(paper: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in paper.splitlines()
        if 100 <= len(line.strip()) <= 3_000
        and not line.lstrip().startswith(("#", "<"))
    )


def _rank_distractors(
    question: str,
    values: Iterable[str],
    *,
    salt: str,
) -> list[str]:
    question_words = _words(question)
    return sorted(
        values,
        key=lambda value: (
            -len(question_words & _words(value)),
            _digest(salt, value),
        ),
    )


def _shuffled(values: Sequence[str], *, salt: str) -> tuple[str, ...]:
    return tuple(sorted(values, key=lambda value: _digest(salt, value)))


@dataclass(frozen=True, slots=True)
class EvidenceTask:
    """One hierarchical retrieval problem with an exact evidence target."""

    task_id: str
    source_example_id: str
    source_paper_id: str
    split: Literal["train", "eval"]
    question: str
    paragraphs: tuple[str, str, str, str]
    gold_paragraph_index: int
    reference_evidence: str

    def __post_init__(self) -> None:
        if not self.task_id or not self.source_example_id or not self.source_paper_id:
            raise ValueError("task identifiers must be non-empty")
        if not self.question or not self.reference_evidence:
            raise ValueError("question and evidence must be non-empty")
        if len(self.paragraphs) != len(LABELS):
            raise ValueError("the pilot requires exactly four paragraphs")
        if len(set(self.paragraphs)) != len(self.paragraphs):
            raise ValueError("candidate paragraphs must be unique")
        if not 0 <= self.gold_paragraph_index < len(self.paragraphs):
            raise ValueError("gold paragraph index is out of range")
        if self.reference_evidence not in self.paragraphs[self.gold_paragraph_index]:
            raise ValueError("gold paragraph must contain the exact evidence")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> EvidenceTask:
        value = json.loads(raw)
        if type(value) is not dict:
            raise ValueError("task row must be a JSON object")
        paragraphs = value.get("paragraphs")
        if type(paragraphs) is not list or len(paragraphs) != 4:
            raise ValueError("task paragraphs must be a four-item JSON list")
        return cls(
            task_id=str(value["task_id"]),
            source_example_id=str(value["source_example_id"]),
            source_paper_id=str(value["source_paper_id"]),
            split=value["split"],
            question=str(value["question"]),
            paragraphs=(
                str(paragraphs[0]),
                str(paragraphs[1]),
                str(paragraphs[2]),
                str(paragraphs[3]),
            ),
            gold_paragraph_index=int(value["gold_paragraph_index"]),
            reference_evidence=str(value["reference_evidence"]),
        )


def _candidate_from_row(row: Mapping[str, Any]) -> EvidenceTask | None:
    try:
        source_id = row["example_id"]
        paper_id = row["paper_id"]
        question = row["question"]
        paper = row["paper"]
        evidence_values = row["reference_evidence"]
    except KeyError as error:
        raise ValueError(f"source row lacks {error.args[0]}") from error
    if not all(
        type(value) is str and value
        for value in (source_id, paper_id, question, paper)
    ):
        raise ValueError("source identifiers, question, and paper must be strings")
    if type(evidence_values) is not list or not all(
        type(value) is str and value for value in evidence_values
    ):
        raise ValueError("reference_evidence must be a non-empty string list")

    paragraphs = _paragraphs(paper)
    for evidence in sorted(evidence_values, key=lambda value: (len(value), value)):
        if len(evidence) > 1_800:
            continue
        hits = [paragraph for paragraph in paragraphs if evidence in paragraph]
        if len(hits) != 1:
            continue
        gold = hits[0]
        gold_sentences = [sentence for sentence in _sentences(gold) if sentence != evidence]
        if len(gold_sentences) < 3:
            continue
        eligible_distractors = list(
            dict.fromkeys(
                paragraph
                for paragraph in paragraphs
                if paragraph != gold and len(_sentences(paragraph)) >= 4
            )
        )
        if len(eligible_distractors) < 3:
            continue
        distractors = _rank_distractors(
            question,
            eligible_distractors,
            salt=f"{source_id}:paragraph-rank",
        )[:3]
        ordered = _shuffled(
            [gold, *distractors],
            salt=f"{source_id}:paragraph-order",
        )
        gold_index = ordered.index(gold)
        task_id = f"qasper-evidence-{_digest(source_id, evidence)[:16]}"
        return EvidenceTask(
            task_id=task_id,
            source_example_id=source_id,
            source_paper_id=paper_id,
            split="train",
            question=question,
            paragraphs=(ordered[0], ordered[1], ordered[2], ordered[3]),
            gold_paragraph_index=gold_index,
            reference_evidence=evidence,
        )
    return None


def build_pilot_tasks(
    source_rows: Iterable[Mapping[str, Any]],
    *,
    train_tasks: int = 24,
    eval_tasks: int = 8,
) -> tuple[EvidenceTask, ...]:
    """Compile a stable, paper-disjoint subset from historical QASPER rows."""
    if train_tasks < 0 or eval_tasks < 0 or train_tasks + eval_tasks == 0:
        raise ValueError("task counts must be non-negative with at least one task")
    candidates = [
        candidate
        for row in source_rows
        if (candidate := _candidate_from_row(row)) is not None
    ]
    if len({task.source_example_id for task in candidates}) != len(candidates):
        raise ValueError("source example identifiers must be unique")
    if len({task.source_paper_id for task in candidates}) != len(candidates):
        raise ValueError("pilot source must contain one task per paper")
    total = train_tasks + eval_tasks
    selected = sorted(candidates, key=lambda task: _digest("pilot-v1", task.task_id))[:total]
    if len(selected) != total:
        raise ValueError(f"source yields {len(selected)} tasks; {total} are required")
    tasks = [
        EvidenceTask(
            task_id=task.task_id,
            source_example_id=task.source_example_id,
            source_paper_id=task.source_paper_id,
            split="train" if index < train_tasks else "eval",
            question=task.question,
            paragraphs=task.paragraphs,
            gold_paragraph_index=task.gold_paragraph_index,
            reference_evidence=task.reference_evidence,
        )
        for index, task in enumerate(selected)
    ]
    return tuple(sorted(tasks, key=lambda task: (task.split, task.task_id)))


def build_span_options(
    task: EvidenceTask,
    paragraph_index: int,
) -> tuple[tuple[str, str, str, str], int | None]:
    """Build the second decision's exact-evidence choices for one paragraph."""
    if not 0 <= paragraph_index < len(task.paragraphs):
        raise ValueError("paragraph index is out of range")
    paragraph = task.paragraphs[paragraph_index]
    sentences = list(_sentences(paragraph))
    salt = f"{task.task_id}:span:{paragraph_index}"
    if paragraph_index == task.gold_paragraph_index:
        distractors = _rank_distractors(
            task.question,
            (sentence for sentence in sentences if sentence != task.reference_evidence),
            salt=salt,
        )[:3]
        if len(distractors) != 3:
            raise ValueError("gold paragraph does not yield three span distractors")
        ordered = _shuffled([task.reference_evidence, *distractors], salt=salt)
        return (
            (ordered[0], ordered[1], ordered[2], ordered[3]),
            ordered.index(task.reference_evidence),
        )
    distractors = _rank_distractors(task.question, sentences, salt=salt)[:4]
    if len(distractors) != 4:
        raise ValueError("distractor paragraph does not yield four span choices")
    ordered = _shuffled(distractors, salt=salt)
    return (ordered[0], ordered[1], ordered[2], ordered[3]), None


def _format_options(values: Sequence[str]) -> str:
    if len(values) != len(LABELS):
        raise ValueError("exactly four options are required")
    return "\n\n".join(f"{label}. {value}" for label, value in zip(LABELS, values, strict=True))


def stage_one_prompt(task: EvidenceTask) -> str:
    """Prompt the paragraph-selection decision."""
    return (
        "Select the paragraph most likely to contain exact evidence for the research "
        "question. Reply with only A, B, C, or D.\n\n"
        f"Question: {task.question}\n\n"
        f"{_format_options(task.paragraphs)}\n\nAnswer:"
    )


def stage_two_prompt(task: EvidenceTask, paragraph_index: int) -> str:
    """Prompt the exact-span decision after committing to a paragraph."""
    options, _ = build_span_options(task, paragraph_index)
    return (
        "Select the option that is the complete exact evidence for the research "
        "question. Reply with only A, B, C, or D.\n\n"
        f"Question: {task.question}\n\n"
        f"Chosen paragraph: {task.paragraphs[paragraph_index]}\n\n"
        f"{_format_options(options)}\n\nAnswer:"
    )


@dataclass(frozen=True, slots=True)
class PilotBudget:
    """Equal rollout budget for the trajectory and ReDCO arms."""

    train_tasks: int = 24
    eval_tasks: int = 8
    baseline_episodes_per_update: int = 5
    redco_primary_episodes_per_update: int = 2
    redco_additional_continuations: int = 6
    epochs: int = 1

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(type(value) is not int or value <= 0 for value in values.values()):
            raise ValueError("pilot budget values must be positive exact integers")
        if self.baseline_calls_per_update != self.redco_calls_per_update:
            raise ValueError("comparison arms must use equal rollout calls")

    @property
    def updates_per_arm(self) -> int:
        return self.train_tasks * self.epochs

    @property
    def baseline_calls_per_update(self) -> int:
        return self.baseline_episodes_per_update * 2

    @property
    def redco_calls_per_update(self) -> int:
        primary_calls = self.redco_primary_episodes_per_update * 2
        return primary_calls + self.redco_additional_continuations

    @property
    def rollout_calls_per_arm(self) -> int:
        return self.updates_per_arm * self.baseline_calls_per_update

    @property
    def evaluation_calls_per_checkpoint(self) -> int:
        return self.eval_tasks * 2


def load_pilot_dataset(path: Path) -> tuple[tuple[EvidenceTask, ...], PilotBudget]:
    """Authenticate and load tasks with their exact rollout budget."""
    value = json.loads(path.read_bytes())
    if type(value) is not dict or set(value) != {
        "payload",
        "payload_sha256",
        "schema_version",
    }:
        raise ValueError("pilot dataset envelope has the wrong schema")
    if value["schema_version"] != 1 or type(value["payload"]) is not dict:
        raise ValueError("pilot dataset envelope has an unsupported version")
    payload = value["payload"]
    expected = require_sha256_hex(value["payload_sha256"], "payload_sha256")
    if sha256_bytes(canonical_json(payload)) != expected:
        raise ValueError("pilot dataset payload hash mismatch")
    if set(payload) != {"budget", "source", "tasks"}:
        raise ValueError("pilot dataset payload has the wrong schema")
    rows = payload["tasks"]
    if type(rows) is not list:
        raise ValueError("pilot dataset tasks must be a list")
    tasks = tuple(
        EvidenceTask.from_json(json.dumps(row, ensure_ascii=False)) for row in rows
    )
    budget = payload["budget"]
    if type(budget) is not dict:
        raise ValueError("pilot dataset budget must be an object")
    if set(budget) != set(asdict(PilotBudget())):
        raise ValueError("pilot dataset budget has the wrong schema")
    try:
        loaded_budget = PilotBudget(**budget)
    except (TypeError, ValueError) as error:
        raise ValueError("pilot dataset budget is invalid") from error
    if len([task for task in tasks if task.split == "train"]) != loaded_budget.train_tasks:
        raise ValueError("pilot dataset has the wrong training task count")
    if len([task for task in tasks if task.split == "eval"]) != loaded_budget.eval_tasks:
        raise ValueError("pilot dataset has the wrong evaluation task count")
    return tasks, loaded_budget


def load_pilot_tasks(path: Path) -> tuple[EvidenceTask, ...]:
    """Authenticate and load tasks while preserving the original public helper."""
    return load_pilot_dataset(path)[0]


def assert_matrix_continuity(
    pilot_tasks: tuple[EvidenceTask, ...],
    matrix_tasks: tuple[EvidenceTask, ...],
) -> None:
    """Prove that the matrix only expands the pilot evaluation cohort."""
    pilot_train = tuple(task for task in pilot_tasks if task.split == "train")
    pilot_eval = tuple(task for task in pilot_tasks if task.split == "eval")
    matrix_train = tuple(task for task in matrix_tasks if task.split == "train")
    matrix_eval = tuple(task for task in matrix_tasks if task.split == "eval")
    if len(pilot_train) != 24 or len(pilot_eval) != 8:
        raise ValueError("pilot continuity requires the committed 24/8 cohort")
    if len(matrix_train) != 24 or len(matrix_eval) != 24:
        raise ValueError("matrix continuity requires a 24/24 cohort")
    if matrix_train != pilot_train:
        raise ValueError("matrix training tasks differ from the pilot")
    matrix_eval_by_id = {task.task_id: task for task in matrix_eval}
    if len(matrix_eval_by_id) != len(matrix_eval):
        raise ValueError("matrix evaluation task IDs are not unique")
    if any(matrix_eval_by_id.get(task.task_id) != task for task in pilot_eval):
        raise ValueError("pilot evaluation tasks are not preserved by the matrix")
    papers = [task.source_paper_id for task in matrix_tasks]
    if len(set(papers)) != 48:
        raise ValueError("matrix must contain 48 unique papers")
    train_papers = {task.source_paper_id for task in matrix_train}
    eval_papers = {task.source_paper_id for task in matrix_eval}
    if train_papers & eval_papers:
        raise ValueError("matrix training and evaluation papers overlap")
