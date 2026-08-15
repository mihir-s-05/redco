from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from redco.experiments.musique_capability import (
    Candidate,
    Choice,
    MuSiQueTask,
    _path,
    answer_f1,
    evaluate_gate,
    load_gate_config,
    load_tasks,
    normalize_answer,
    parse_numbered_choice,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/musique-ans-capability-gate-v1.json"
SNAPSHOT = ROOT / "data/musique-ans-capability-v1.json"


class DeterministicPolicy:
    def __init__(self, tasks: tuple[MuSiQueTask, ...]) -> None:
        self._support = {task.task_id: task.support_path for task in tasks}
        self._answers = {task.task_id: task.answer for task in tasks}
        self.choices: list[dict[str, Any]] = []
        self.answers: list[dict[str, Any]] = []

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
    ) -> Choice:
        del question
        self.choices.append(
            {
                "task_id": task_id,
                "step": step,
                "evidence": evidence,
                "candidates": candidates,
                "sample": sample,
                "sampling_seed": sampling_seed,
            }
        )
        support = self._support[task_id]
        if sample is not None and sample % 2 == 1 and not evidence:
            return Choice(
                next(candidate.title for candidate in candidates if candidate.title not in support),
                True,
            )
        return Choice(
            next(
                title
                for title in support
                if title not in {candidate.title for candidate in evidence}
            ),
            True,
        )

    def answer(
        self,
        task_id: str,
        question: str,
        evidence: tuple[Candidate, ...],
        *,
        mode: str,
    ) -> str:
        del question
        self.answers.append({"task_id": task_id, "evidence": evidence, "mode": mode})
        return (
            self._answers[task_id]
            if set(self._support[task_id]) <= {item.title for item in evidence}
            else "wrong"
        )


def _tasks() -> tuple[MuSiQueTask, ...]:
    return load_tasks(SNAPSHOT, load_gate_config(CONFIG))


def test_official_snapshot_is_global_short_document_cohort() -> None:
    config = load_gate_config(CONFIG)
    tasks = _tasks()
    assert len(tasks) == 45
    assert {task.hop_count for task in tasks} == {2, 4}
    assert {depth: sum(task.hop_count == depth for task in tasks) for depth in (2, 4)} == {
        2: 24,
        4: 21,
    }
    assert all(len(task.candidates) == config.candidate_count for task in tasks)
    titles = [title for task in tasks for title in task.support_path]
    components = [component for task in tasks for component in task.support_components]
    assert len(titles) == len(set(titles))
    assert len(components) == len(set(components))
    assert all(
        all(
            len(candidate.text.split()) <= config.short_document_token_limit
            for candidate in task.candidates
        )
        for task in tasks
    )
    assert all(task.task_id.endswith("_".join(task.support_components)) for task in tasks)


def test_gate_uses_all_tasks_and_records_exact_882_plan() -> None:
    tasks = _tasks()
    policy = DeterministicPolicy(tasks)
    result = evaluate_gate(tasks, policy, load_gate_config(CONFIG))
    assert result["passed"] is True
    assert result["blocked_reasons"] == []
    by_hop = cast(dict[str, dict[str, Any]], result["by_hop"])
    assert {key: value["tasks"] for key, value in by_hop.items()} == {"2": 24, "4": 21}
    generation = cast(dict[str, Any], result["generation"])
    assert generation["choice_calls"] == {"greedy": 132, "teacher_forced": 132, "sampled": 528}
    assert generation["answer_calls"] == 90
    assert generation["planned_total"] == 882
    assert len(policy.choices) == 792
    assert len(policy.answers) == 90
    assert all(call["task_id"] and call["step"] for call in policy.choices)
    assert all(
        not hasattr(candidate, "answer")
        for call in policy.choices
        for candidate in call["evidence"] + call["candidates"]
    )
    sampled = [call for call in policy.choices if call["sample"] is not None]
    assert len(sampled) == 528
    assert all(isinstance(call["sampling_seed"], int) for call in sampled)
    assert len({call["sampling_seed"] for call in sampled}) == 528


def test_official_answer_normalization_and_alias_f1() -> None:
    assert normalize_answer("The, A fox!") == "fox"
    assert normalize_answer("can't be the answer") == "cant be answer"
    assert answer_f1("The Blue Whale", ("red", "blue whale")) == 1.0
    raw = json.loads(SNAPSHOT.read_bytes())["tasks"][0]
    assert "answer_aliases" in raw
    raw["question_decomposition"] = []
    with pytest.raises(ValueError, match="wrong schema"):
        MuSiQueTask.from_mapping(raw)


def test_invalid_first_hop_keeps_later_choice_at_its_original_position() -> None:
    task = _tasks()[0]
    config = load_gate_config(CONFIG)

    class InvalidFirstPolicy:
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
        ) -> Choice:
            del task_id, question, evidence, candidates, sample, sampling_seed
            return Choice("", False) if step == 1 else Choice(task.support_path[step - 1], True)

        def answer(
            self,
            task_id: str,
            question: str,
            evidence: tuple[Candidate, ...],
            *,
            mode: str,
        ) -> str:
            del task_id, question, evidence, mode
            return ""

    path = _path(InvalidFirstPolicy(), task, config, sample=None)
    assert path.positions == (None, task.support_path[1])
    assert path.selected == (task.support_path[1],)
    assert path.prefix_length == 0
    assert path.conditional_reach == (True, False)
    assert path.conditional_correct == (False, False)


@pytest.mark.parametrize("raw", ["1 commentary", "1 2", "0", "9", "Title"])
def test_numbered_choice_is_strict(raw: str) -> None:
    candidates = (Candidate("Title", "text"), Candidate("Other", "text"))
    assert parse_numbered_choice(raw, candidates) == Choice("", False)
    assert parse_numbered_choice("2", candidates) == Choice("Other", True)
