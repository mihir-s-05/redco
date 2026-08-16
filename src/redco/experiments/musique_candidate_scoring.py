"""Constrained eight-way MuSiQue document-choice scoring diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from redco.contracts import canonical_json
from redco.experiments.musique_capability import Candidate, MuSiQueTask, load_tasks

ChoiceLaw = Literal["conditional_log_likelihood"]
Mode = Literal["greedy", "teacher_forced"]
_PROMPT = (
    "Question:\n{question}\n\nAlready selected evidence:\n{evidence}\n\n"
    "Choose exactly one next document from the numbered candidates. Return only one integer "
    "from the displayed range.\n\nCandidates:\n{candidates}"
)


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    schema_version: int
    config_sha256: str
    snapshot_path: str
    snapshot_sha256: str
    gate_config_path: str
    gate_config_sha256: str
    task_counts: tuple[tuple[int, int], ...]
    model: str
    revision: str
    labels: tuple[str, ...]
    prompt_template: str
    max_input_tokens: int
    candidate_count: int
    expected_states: int
    teacher_margin: float
    four_hop_greedy_full_path_min: float
    four_hop_teacher_position_min_count: int
    four_hop_greedy_full_path_min_count: int
    four_hop_k10_mixed_sum_min: float
    max_seconds: int
    cuda_devices: int
    dtype: str
    use_cache: bool
    stronger_gate: bool = False


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    law: ChoiceLaw
    selected_index: int
    gold_index: int
    gold_rank: int
    selected_score: float
    gold_score: float
    selected_probability: float
    gold_probability: float
    entropy: float
    top1_top2_margin: float


@dataclass(frozen=True, slots=True)
class StateRecord:
    task_id: str
    hop_count: int
    mode: Mode
    step: int
    gold_index: int
    gold_candidate: str
    selected_index: int
    selected_candidate: str
    gold_rank: int
    selected_score: float
    gold_score: float
    selected_probability: float
    gold_probability: float
    entropy: float
    top1_top2_margin: float
    law: ChoiceLaw

    def as_mapping(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "hop_count": self.hop_count,
            "mode": self.mode,
            "step": self.step,
            "gold_index": self.gold_index,
            "gold_candidate": self.gold_candidate,
            "selected_index": self.selected_index,
            "selected_candidate": self.selected_candidate,
            "gold_rank": self.gold_rank,
            "selected_score": self.selected_score,
            "gold_score": self.gold_score,
            "selected_probability": self.selected_probability,
            "gold_probability": self.gold_probability,
            "entropy": self.entropy,
            "top1_top2_margin": self.top1_top2_margin,
            "law": self.law,
        }


class ChoiceScorer(Protocol):
    law: ChoiceLaw
    forward_calls: int

    def score(self, prompt: str, gold_index: int, candidate_count: int) -> ScoreSummary: ...


def _finite(value: object, message: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(message)
    number = cast(int | float, value)
    if not math.isfinite(float(number)):
        raise ValueError(message)
    return float(number)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> ScoringConfig:
    data = path.read_bytes()
    config_sha256 = hashlib.sha256(data).hexdigest()
    value = json.loads(data)
    if type(value) is not dict or set(value) != {
        "choice",
        "cohort",
        "dataset",
        "gate",
        "model",
        "runtime",
        "schema_version",
        "source",
    }:
        raise ValueError("candidate-scoring config has the wrong schema")
    schema_version = value["schema_version"]
    if type(schema_version) is not int or schema_version != 1 or value["dataset"] != "MuSiQue-Ans":
        raise ValueError("candidate-scoring config is not MuSiQue")
    if value["cohort"] != "short_document_linear_chain":
        raise ValueError("candidate-scoring config has the wrong cohort")
    source = value["source"]
    model = value["model"]
    choice = value["choice"]
    gate = value["gate"]
    runtime = value["runtime"]
    model_keys = {"name", "revision"}
    gate_keys = {
        "candidate_count",
        "expected_states",
        "four_hop_greedy_full_path_min",
        "teacher_margin",
    }
    runtime_keys = {"cuda_devices", "dtype", "max_seconds"}
    if (
        type(source) is not dict
        or set(source)
        != {
            "gate_config_path",
            "gate_config_sha256",
            "snapshot_path",
            "snapshot_sha256",
            "task_counts",
        }
        or type(model) is not dict
        or set(model) != model_keys
        or type(choice) is not dict
        or set(choice) != {"chat_template", "labels", "max_input_tokens", "prompt_template"}
        or type(gate) is not dict
        or set(gate) != gate_keys
        or type(runtime) is not dict
        or set(runtime) != runtime_keys
    ):
        raise ValueError("candidate-scoring config section has the wrong schema")
    source = cast(dict[str, object], source)
    model = cast(dict[str, object], model)
    choice = cast(dict[str, object], choice)
    gate = cast(dict[str, object], gate)
    runtime = cast(dict[str, object], runtime)
    counts = source["task_counts"]
    labels = choice["labels"]
    chat = choice["chat_template"]
    if (
        source["snapshot_path"] != "data/musique-ans-capability-v1.json"
        or source["gate_config_path"] != "configs/musique-ans-capability-gate-v1.json"
        or type(source["snapshot_sha256"]) is not str
        or len(source["snapshot_sha256"]) != 64
        or type(source["gate_config_sha256"]) is not str
        or len(source["gate_config_sha256"]) != 64
        or type(counts) is not dict
        or counts != {"2": 24, "4": 21}
        or (
            schema_version == 1
            and (
                model["name"] != "Qwen/Qwen3-4B-Instruct-2507"
                or model["revision"] != "cdbee75f17c01a7cc42f958dc650907174af0554"
            )
        )
        or labels != [str(index) for index in range(1, 9)]
        or choice["prompt_template"] != _PROMPT
        or chat != {"tokenize": False, "add_generation_prompt": True}
        or choice["max_input_tokens"] != 8192
        or gate["candidate_count"] != 8
        or gate["expected_states"] != 264
        or runtime["cuda_devices"] != 1
        or runtime["dtype"] != "bfloat16"
        or runtime["max_seconds"] != 900
    ):
        raise ValueError("candidate-scoring config is outside the frozen diagnostic")
    threshold = _finite(gate["teacher_margin"], "teacher margin must be finite")
    full_path_min = _finite(
        gate["four_hop_greedy_full_path_min"], "full-path threshold must be finite"
    )
    if threshold != 0.1 or full_path_min != 0.1:
        raise ValueError("candidate-scoring thresholds changed")
    return ScoringConfig(
        schema_version,
        config_sha256,
        source["snapshot_path"],
        source["snapshot_sha256"],
        source["gate_config_path"],
        source["gate_config_sha256"],
        ((2, 24), (4, 21)),
        cast(str, model["name"]),
        cast(str, model["revision"]),
        tuple(cast(list[str], labels)),
        choice["prompt_template"],
        choice["max_input_tokens"],
        gate["candidate_count"],
        gate["expected_states"],
        threshold,
        full_path_min,
        0,
        0,
        0.0,
        runtime["max_seconds"],
        runtime["cuda_devices"],
        runtime["dtype"],
        False,
        False,
    )


def load_inputs(
    config_path: Path,
    snapshot_path: Path,
    gate_config_path: Path,
) -> tuple[ScoringConfig, tuple[MuSiQueTask, ...]]:
    config = load_config(config_path)
    if _sha256(gate_config_path) != config.gate_config_sha256:
        raise ValueError("frozen capability config hash does not match")
    if _sha256(snapshot_path) != config.snapshot_sha256:
        raise ValueError("frozen MuSiQue snapshot hash does not match")
    if gate_config_path.resolve() != Path(config.gate_config_path).resolve():
        raise ValueError("gate config path is not the frozen path")
    if snapshot_path.resolve() != Path(config.snapshot_path).resolve():
        raise ValueError("snapshot path is not the frozen path")
    tasks = load_tasks(snapshot_path, expected_sha256=config.snapshot_sha256)
    return config, tasks


def prompt_for_state(
    config: ScoringConfig,
    task: MuSiQueTask,
    evidence: Sequence[Candidate],
    candidates: Sequence[Candidate],
) -> str:
    evidence_text = (
        "(none)" if not evidence else "\n\n".join(f"{item.title}\n{item.text}" for item in evidence)
    )
    candidate_text = "\n\n".join(
        f"[{index}] {item.title}\n{item.text}" for index, item in enumerate(candidates, 1)
    )
    return config.prompt_template.format(
        question=task.question,
        evidence=evidence_text,
        candidates=candidate_text,
    )


def score_values(scores: Sequence[float], gold_index: int, law: ChoiceLaw) -> ScoreSummary:
    if not scores or not 0 <= gold_index < len(scores):
        raise ValueError("choice scoring requires labels and one gold index")
    values = tuple(float(value) for value in scores)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("choice scores must be finite")
    top = max(values)
    weights = tuple(math.exp(value - top) for value in values)
    total = sum(weights)
    probabilities = tuple(weight / total for weight in weights)
    order = tuple(sorted(range(len(values)), key=lambda index: (-values[index], index)))
    selected = order[0]
    entropy = -sum(
        probability * math.log(probability) for probability in probabilities if probability > 0
    )
    return ScoreSummary(
        law,
        selected,
        gold_index,
        order.index(gold_index) + 1,
        values[selected],
        values[gold_index],
        probabilities[selected],
        probabilities[gold_index],
        entropy,
        probabilities[order[0]] - probabilities[order[1]],
    )


def _candidates(task: MuSiQueTask, selected: Sequence[str]) -> tuple[Candidate, ...]:
    excluded = set(selected)
    return tuple(candidate for candidate in task.candidates if candidate.title not in excluded)


def _state(
    config: ScoringConfig,
    task: MuSiQueTask,
    mode: Mode,
    step: int,
    evidence_titles: Sequence[str],
    candidates: tuple[Candidate, ...],
    scorer: ChoiceScorer,
) -> StateRecord:
    evidence = tuple(
        next(candidate for candidate in task.candidates if candidate.title == title)
        for title in evidence_titles
    )
    if mode == "greedy":
        gold_candidate = next(
            (title for title in task.support_path if title not in set(evidence_titles)), None
        )
    elif step < len(task.support_path):
        gold_candidate = task.support_path[step]
    else:
        gold_candidate = None
    if gold_candidate is None:
        raise ValueError("choice state has no remaining authenticated support target")
    try:
        gold_index = next(
            index for index, candidate in enumerate(candidates) if candidate.title == gold_candidate
        )
    except StopIteration as error:
        raise ValueError("support target is absent from the active candidate pool") from error
    summary = scorer.score(
        prompt_for_state(config, task, evidence, candidates), gold_index, len(candidates)
    )
    selected_candidate = candidates[summary.selected_index].title
    return StateRecord(
        task.task_id,
        task.hop_count,
        mode,
        step + 1,
        gold_index,
        gold_candidate,
        summary.selected_index,
        selected_candidate,
        summary.gold_rank,
        summary.selected_score,
        summary.gold_score,
        summary.selected_probability,
        summary.gold_probability,
        summary.entropy,
        summary.top1_top2_margin,
        summary.law,
    )


def evaluate_states(
    tasks: Sequence[MuSiQueTask], config: ScoringConfig, scorer: ChoiceScorer
) -> tuple[StateRecord, ...]:
    records: list[StateRecord] = []
    for task in tasks:
        selected: list[str] = []
        for step in range(task.hop_count):
            candidates = _candidates(task, selected)
            record = _state(config, task, "greedy", step, selected, candidates, scorer)
            records.append(record)
            selected.append(record.selected_candidate)
        for step in range(task.hop_count):
            selected = list(task.support_path[:step])
            candidates = _candidates(task, selected)
            records.append(
                _state(config, task, "teacher_forced", step, selected, candidates, scorer)
            )
    if len(records) != config.expected_states or scorer.forward_calls != config.expected_states:
        raise ValueError("candidate-scoring state or forward-call count is not exactly 264")
    return tuple(records)


def _mean(values: Iterable[float]) -> float:
    items = tuple(values)
    return sum(items) / len(items) if items else 0.0


def k10_mixed_sum(probabilities: Iterable[float]) -> float:
    return sum(1.0 - probability**10 - (1.0 - probability) ** 10 for probability in probabilities)


def summarize(
    tasks: Sequence[MuSiQueTask], config: ScoringConfig, records: Sequence[StateRecord]
) -> tuple[dict[str, object], list[str]]:
    blocked: list[str] = []
    by_hop: dict[str, object] = {}
    for depth, expected_tasks in config.task_counts:
        depth_tasks = [task for task in tasks if task.hop_count == depth]
        greedy = [
            record for record in records if record.hop_count == depth and record.mode == "greedy"
        ]
        teacher = [
            record
            for record in records
            if record.hop_count == depth and record.mode == "teacher_forced"
        ]
        teacher_by_task = {
            task.task_id: tuple(
                sorted(
                    (record for record in teacher if record.task_id == task.task_id),
                    key=lambda record: record.step,
                )
            )
            for task in depth_tasks
        }
        greedy_paths = {
            task.task_id: tuple(
                record.selected_candidate for record in greedy if record.task_id == task.task_id
            )
            for task in depth_tasks
        }
        full_paths = sum(
            path == task.support_path
            for task, path in ((task, greedy_paths[task.task_id]) for task in depth_tasks)
        )
        teacher_correct = sum(record.gold_rank == 1 for record in teacher)
        random_baseline = _mean(1.0 / (config.candidate_count - step) for step in range(depth))
        teacher_accuracy = teacher_correct / len(teacher)
        teacher_passed = teacher_accuracy > random_baseline + config.teacher_margin
        full_rate = full_paths / expected_tasks
        unordered_support_matches = sum(
            set(greedy_paths[task.task_id]) == set(task.support_path) for task in depth_tasks
        )
        position_correct = [
            sum(record.gold_rank == 1 for record in teacher if record.step == step)
            for step in range(1, depth + 1)
        ]
        position_gate = (
            all(count >= config.four_hop_teacher_position_min_count for count in position_correct)
            if depth == 4 and config.stronger_gate
            else None
        )
        k10_rows: list[dict[str, object]] = []
        k10_q_sum = 0.0
        k10_mixed_sum = 0.0
        k10_expected_tasks = 0.0
        k10_mixed_count = 0
        if depth == 4:
            for task in depth_tasks:
                q = math.prod(record.gold_probability for record in teacher_by_task[task.task_id])
                mixed = 1.0 - q**10 - (1.0 - q) ** 10
                k10_rows.append({"task_id": task.task_id, "q": q, "m": mixed})
                k10_q_sum += q
                k10_mixed_sum += mixed
                k10_expected_tasks += 1.0 - (1.0 - q) ** 10
                k10_mixed_count += mixed >= 0.5
        k10_gate = (
            k10_mixed_sum >= config.four_hop_k10_mixed_sum_min
            if depth == 4 and config.stronger_gate
            else None
        )
        full_path_count_gate = (
            full_paths >= config.four_hop_greedy_full_path_min_count
            if depth == 4 and config.stronger_gate
            else None
        )
        full_path_passed = full_rate > config.four_hop_greedy_full_path_min and (
            full_path_count_gate is not False
        )
        if not teacher_passed:
            blocked.append(f"{depth}-hop teacher top-1 accuracy does not exceed the frozen margin")
        if depth == 4 and not full_path_passed:
            blocked.append("4-hop greedy full-path rate does not exceed the frozen threshold")
        if depth == 4 and position_gate is False:
            blocked.append("4-hop teacher positions do not meet the stronger count gate")
        training_support_reasons = (
            ["4-hop K10 mixed expectation does not meet the frozen support threshold"]
            if depth == 4 and k10_gate is False
            else []
        )
        by_hop[str(depth)] = {
            "tasks": expected_tasks,
            "greedy_states": len(greedy),
            "teacher_states": len(teacher),
            "greedy_full_paths": full_paths,
            "greedy_full_path_rate": full_rate,
            "teacher_top1_correct": teacher_correct,
            "teacher_top1_accuracy": teacher_accuracy,
            "random_baseline": random_baseline,
            "teacher_margin_over_random": teacher_accuracy - random_baseline,
            "teacher_gate": teacher_passed,
            "greedy_full_path_gate": full_path_passed if depth == 4 else None,
            "greedy_full_path_count_gate": full_path_count_gate,
            "greedy_unordered_support_set_matches": unordered_support_matches,
            "teacher_position_top1_correct": position_correct,
            "teacher_position_top1_gate": position_gate,
            "capability_pass": teacher_passed
            and (full_path_passed if depth == 4 else True)
            and (position_gate is not False),
            "training_support_pass": not training_support_reasons,
            "training_support_reasons": training_support_reasons,
            "k10_mixed": (
                {
                    "q_values": k10_rows,
                    "sum_q": k10_q_sum,
                    "expected_tasks_with_at_least_one_exact": k10_expected_tasks,
                    "sum_m": k10_mixed_sum,
                    "count_m_at_least_half": k10_mixed_count,
                    "gate": k10_gate,
                }
                if depth == 4
                else None
            ),
        }
    return by_hop, blocked


def report_payload(
    config: ScoringConfig,
    snapshot_sha256: str,
    records: Sequence[StateRecord],
    by_hop: Mapping[str, object],
    blocked: Sequence[str],
    *,
    elapsed_seconds: float,
    forward_calls: int,
    law: ChoiceLaw,
) -> dict[str, object]:
    training_support_reasons = [
        str(reason)
        for value in by_hop.values()
        if isinstance(value, dict)
        for reason in cast(list[object], value.get("training_support_reasons", []))
    ]
    capability_reasons = list(blocked)
    capability_pass = not capability_reasons
    training_support_pass = not training_support_reasons
    return {
        "schema_version": config.schema_version,
        "dataset": "MuSiQue-Ans",
        "cohort": "short_document_linear_chain",
        "mode": "constrained_candidate_scoring",
        "config_sha256": config.config_sha256,
        "snapshot_sha256": snapshot_sha256,
        "model": config.model,
        "revision": config.revision,
        "cuda_device_count": config.cuda_devices,
        "dtype": config.dtype,
        "use_cache": config.use_cache,
        "choice_law": law,
        "gate": {
            "teacher_margin": config.teacher_margin,
            "four_hop_greedy_full_path_min": config.four_hop_greedy_full_path_min,
            "four_hop_teacher_position_min_count": config.four_hop_teacher_position_min_count,
            "four_hop_greedy_full_path_min_count": config.four_hop_greedy_full_path_min_count,
            "four_hop_k10_mixed_sum_min": config.four_hop_k10_mixed_sum_min,
        },
        "states": [record.as_mapping() for record in records],
        "generation": {
            "greedy_states": sum(record.mode == "greedy" for record in records),
            "teacher_forced_states": sum(record.mode == "teacher_forced" for record in records),
            "expected_forward_calls": config.expected_states,
            "actual_forward_calls": forward_calls,
            "answer_calls": 0,
            "sampled_calls": 0,
            "training_updates": 0,
        },
        "by_hop": dict(by_hop),
        "capability_pass": capability_pass,
        "capability_blocked_reasons": capability_reasons,
        "training_support_pass": training_support_pass,
        "training_support_blocked_reasons": training_support_reasons,
        "training_experiment_eligible": capability_pass and training_support_pass,
        "passed": capability_pass,
        "blocked_reasons": capability_reasons,
        "gold_fields_in_prompt": False,
        "deadline_cooperative": True,
        "elapsed_seconds": elapsed_seconds,
    }


def check_payload(
    config: ScoringConfig, snapshot_sha256: str, tasks: Sequence[MuSiQueTask]
) -> dict[str, object]:
    return {
        "mode": "check",
        "schema_version": config.schema_version,
        "dataset": "MuSiQue-Ans",
        "cohort": "short_document_linear_chain",
        "config_sha256": config.config_sha256,
        "snapshot_sha256": snapshot_sha256,
        "model": config.model,
        "revision": config.revision,
        "use_cache": config.use_cache,
        "choice_law": "conditional_log_likelihood",
        "tasks": len(tasks),
        "tasks_by_depth": {
            str(depth): sum(task.hop_count == depth for task in tasks) for depth in (2, 4)
        },
        "expected_states": config.expected_states,
        "gate": {
            "teacher_margin": config.teacher_margin,
            "four_hop_greedy_full_path_min": config.four_hop_greedy_full_path_min,
            "four_hop_teacher_position_min_count": config.four_hop_teacher_position_min_count,
            "four_hop_greedy_full_path_min_count": config.four_hop_greedy_full_path_min_count,
            "four_hop_k10_mixed_sum_min": config.four_hop_k10_mixed_sum_min,
        },
        "gold_fields_in_prompt": False,
        "model_calls": False,
        "answer_calls": 0,
        "sampled_calls": 0,
        "training_updates": 0,
    }


def canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return canonical_json(dict(payload))


def write_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_bytes(payload))


__all__ = [
    "ChoiceLaw",
    "ChoiceScorer",
    "ScoreSummary",
    "ScoringConfig",
    "StateRecord",
    "canonical_bytes",
    "check_payload",
    "evaluate_states",
    "k10_mixed_sum",
    "load_config",
    "load_inputs",
    "prompt_for_state",
    "report_payload",
    "score_values",
    "summarize",
    "write_exclusive",
]
