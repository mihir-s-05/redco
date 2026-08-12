import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from redco.experiments.qasper_evidence import (
    EvidenceTask,
    PilotBudget,
    build_pilot_tasks,
    build_span_options,
    load_pilot_tasks,
    stage_one_prompt,
    stage_two_prompt,
)
from redco.experiments.qasper_runtime import redco_batch, trajectory_batch


def _row(index: int) -> dict[str, object]:
    evidence = (
        "The experiment uses four independent seeds. Accuracy improves by twelve points. "
        "The ablation removes the retrieval module. Runtime stays below one hour."
    )
    distractors = [
        (
            f"Distractor {number} for paper {index} reports a different setup. "
            "It includes enough sentences for deterministic candidate construction. "
            "The reported result concerns another research question entirely. "
            "A final sentence makes the paragraph structurally eligible."
        )
        for number in range(5)
    ]
    paper = "\n".join(["### PAPER: Fixture", evidence, *distractors])
    return {
        "example_id": f"example-{index}",
        "paper_id": f"paper-{index}",
        "question": "How many independent seeds does the experiment use?",
        "paper": paper,
        "reference_evidence": [evidence],
    }


def test_compiler_is_deterministic_and_round_trips() -> None:
    rows = [_row(index) for index in range(7)]
    first = build_pilot_tasks(rows, train_tasks=4, eval_tasks=2)
    second = build_pilot_tasks(reversed(rows), train_tasks=4, eval_tasks=2)

    assert first == second
    assert [task.split for task in first].count("train") == 4
    assert [task.split for task in first].count("eval") == 2
    assert len({task.source_example_id for task in first}) == 6
    assert EvidenceTask.from_json(first[0].to_json()) == first[0]


def test_hierarchical_prompts_and_rewards_have_exact_choices() -> None:
    (task,) = build_pilot_tasks([_row(0)], train_tasks=1, eval_tasks=0)
    stage_one = stage_one_prompt(task)
    assert stage_one.endswith("Answer:")
    assert all(f"{label}. " in stage_one for label in "ABCD")

    for paragraph_index in range(4):
        spans, gold = build_span_options(task, paragraph_index)
        assert len(spans) == len(set(spans)) == 4
        prompt = stage_two_prompt(task, paragraph_index)
        assert prompt.endswith("Answer:")
        if paragraph_index == task.gold_paragraph_index:
            assert gold is not None
            assert spans[gold] == task.reference_evidence
        else:
            assert gold is None


def test_equal_policy_call_budget() -> None:
    budget = PilotBudget()
    assert budget.baseline_calls_per_update == budget.redco_calls_per_update == 10
    assert budget.updates_per_arm == 24
    assert budget.rollout_calls_per_arm == 240
    assert budget.evaluation_calls_per_checkpoint == 16


def test_compiler_rejects_insufficient_source() -> None:
    with pytest.raises(ValueError, match="are required"):
        build_pilot_tasks([_row(0)], train_tasks=2, eval_tasks=1)


def test_task_json_rejects_wrong_paragraph_shape() -> None:
    raw = json.dumps(
        {
            "task_id": "t",
            "source_example_id": "s",
            "source_paper_id": "p",
            "split": "train",
            "question": "q",
            "paragraphs": ["one"],
            "gold_paragraph_index": 0,
            "reference_evidence": "one",
        }
    )
    with pytest.raises(ValueError, match="four-item"):
        EvidenceTask.from_json(raw)


def test_dataset_loader_authenticates_payload(tmp_path: Path) -> None:
    tasks = build_pilot_tasks(
        [_row(index) for index in range(40)],
        train_tasks=24,
        eval_tasks=8,
    )
    from redco.contracts import canonical_json
    from redco.integrity import sha256_bytes

    payload = {
        "budget": asdict(PilotBudget()),
        "source": {"fixture": True},
        "tasks": [json.loads(task.to_json()) for task in tasks],
    }
    envelope: dict[str, Any] = {
        "payload": payload,
        "payload_sha256": sha256_bytes(canonical_json(payload)),
        "schema_version": 1,
    }
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert load_pilot_tasks(path) == tasks

    envelope["payload"]["tasks"][0]["question"] = "tampered"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_pilot_tasks(path)


class _FakePolicy:
    def __init__(self) -> None:
        self.calls = 0

    def sample(self, prompts: Sequence[str]) -> tuple[tuple[int, float], ...]:
        self.calls += len(prompts)
        return tuple((index % 4, -1.0) for index, _ in enumerate(prompts))


def test_runtime_batches_have_equal_calls_and_decision_normalization() -> None:
    (task,) = build_pilot_tasks([_row(0)], train_tasks=1, eval_tasks=0)
    budget = PilotBudget()

    baseline_policy = _FakePolicy()
    baseline, _ = trajectory_batch(baseline_policy, task, budget)
    assert baseline_policy.calls == 10
    assert len(baseline) == 10
    assert sum(decision.outer_weight for decision in baseline) == 10
    assert sum(decision.decision_units for decision in baseline) == 10

    redco_policy = _FakePolicy()
    redco, _ = redco_batch(redco_policy, task)
    assert redco_policy.calls == 10
    assert len(redco) == 9
    assert sum(decision.outer_weight for decision in redco) == pytest.approx(2)
    assert sum(decision.decision_units for decision in redco) == pytest.approx(2)
