from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

from redco.experiments.musique_candidate_scoring import (
    ChoiceLaw,
    ScoreSummary,
    ScoringConfig,
    evaluate_states,
    load_inputs,
    prompt_for_state,
    score_values,
    summarize,
    write_exclusive,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/musique-ans-candidate-scoring-v1.json"
SNAPSHOT = ROOT / "data/musique-ans-capability-v1.json"
GATE_CONFIG = ROOT / "configs/musique-ans-capability-gate-v1.json"
sys.path.insert(0, str(ROOT / "scripts"))

from run_musique_candidate_scoring import (  # noqa: E402
    TorchChoiceScorer,
    _load_model,
    _prompts,
)


def _inputs() -> tuple[ScoringConfig, tuple[Any, ...]]:
    return load_inputs(CONFIG, SNAPSHOT, GATE_CONFIG)


class FixedScorer:
    law: ChoiceLaw = "conditional_log_likelihood"

    def __init__(self) -> None:
        self.forward_calls = 0

    def score(self, prompt: str, gold_index: int, candidate_count: int) -> ScoreSummary:
        del prompt
        self.forward_calls += 1
        scores = [0.0] * candidate_count
        scores[gold_index] = 1.0
        return score_values(scores, gold_index, self.law)


class OffOrderScorer:
    law: ChoiceLaw = "conditional_log_likelihood"

    def __init__(self, later_support: str) -> None:
        self.later_support = later_support
        self.forward_calls = 0

    def score(self, prompt: str, gold_index: int, candidate_count: int) -> ScoreSummary:
        self.forward_calls += 1
        selected_index = 0
        if self.forward_calls == 1:
            selected_index = next(
                index
                for index in range(candidate_count)
                if f"[{index + 1}] {self.later_support}\n" in prompt
            )
        scores = [0.0] * candidate_count
        scores[selected_index] = 1.0
        return score_values(scores, gold_index, self.law)


class _Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _Tensor:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows
        self.to_devices: list[str] = []

    def __getitem__(self, key: object) -> _Tensor | _Vector | _Scalar:
        if isinstance(key, tuple):
            row, column = key
            if isinstance(row, int) and isinstance(column, int):
                return _Scalar(0.0)
        if isinstance(key, int):
            return _Vector(self.rows[key])
        raise AssertionError(key)

    def to(self, device: str) -> _Tensor:
        self.to_devices.append(device)
        return self

    def tolist(self) -> list[list[int]]:
        return self.rows


class _Vector:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[int]:
        return self.values


class _FakeOutput:
    logits = _Tensor([[0]])


class _Mode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class _FakeTorch:
    inference_count: ClassVar[int] = 0

    @staticmethod
    def inference_mode() -> _Mode:
        _FakeTorch.inference_count += 1
        return _Mode()

    @staticmethod
    def log_softmax(logits: object, *, dim: int) -> object:
        del logits, dim
        return _FakeLogProb()


class _FakeLogProb:
    def __getitem__(self, key: tuple[int, int, int]) -> _Scalar:
        del key
        return _Scalar(0.0)


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _FakeOutput:
        self.calls += 1
        self.kwargs.append(kwargs)
        return _FakeOutput()


class _Tokenizer:
    def __init__(
        self, duplicate_suffix: bool = False, context_variant: str = ""
    ) -> None:
        self.duplicate_suffix = duplicate_suffix
        self.context_variant = context_variant
        self.prompts: list[str] = []

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        return "<chat>" + prompt

    def _ids(self, text: str) -> list[int]:
        ids = [ord(char) for char in text]
        if self.duplicate_suffix and text.endswith("3"):
            ids[-1] = ord("2")
        if self.context_variant and self.context_variant in text and text.endswith("7"):
            ids.append(777)
        return ids

    def __call__(self, value: str | list[str], **kwargs: Any) -> Mapping[str, _Tensor]:
        del kwargs
        texts = [value] if isinstance(value, str) else value
        sequences = [self._ids(text) for text in texts]
        width = max(len(row) for row in sequences)
        rows = [row + [0] * (width - len(row)) for row in sequences]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in sequences]
        return {"input_ids": _Tensor(rows), "attention_mask": _Tensor(masks)}


def test_config_snapshot_prompt_and_state_count_are_bound() -> None:
    config, tasks = _inputs()
    assert len(tasks) == 45
    assert {depth: sum(item.hop_count == depth for item in tasks) for depth in (2, 4)} == {
        2: 24,
        4: 21,
    }
    task = tasks[0]
    prompt = prompt_for_state(config, task, (), task.candidates)
    assert task.question in prompt
    assert task.candidates[0].title in prompt
    assert task.candidates[0].text in prompt
    assert all(field not in prompt for field in ("question_decomposition", "answer_aliases"))

    scorer = FixedScorer()
    records = evaluate_states(tasks, config, scorer)
    by_hop, blocked = summarize(tasks, config, records)
    assert len(records) == 264
    assert scorer.forward_calls == 264
    assert blocked == []
    assert cast(dict[str, object], by_hop["2"])["teacher_top1_accuracy"] == 1.0
    assert cast(dict[str, object], by_hop["4"])["greedy_full_path_rate"] == 1.0


def test_greedy_off_order_selection_keeps_later_targets_defined() -> None:
    config, tasks = _inputs()
    first_task = tasks[0]
    scorer = OffOrderScorer(first_task.support_path[1])
    records = evaluate_states(tasks, config, scorer)
    first_greedy = [
        record
        for record in records
        if record.task_id == first_task.task_id and record.mode == "greedy"
    ]
    assert len(first_greedy) == first_task.hop_count
    assert first_greedy[0].selected_candidate == first_task.support_path[1]
    assert tuple(record.selected_candidate for record in first_greedy) != first_task.support_path


def test_score_values_reports_rank_probability_entropy_and_margin() -> None:
    summary = score_values(
        [3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
        2,
        "conditional_log_likelihood",
    )
    assert summary.selected_index == 0
    assert summary.gold_index == 2
    assert summary.gold_rank == 3
    assert summary.selected_probability > summary.gold_probability > 0
    assert summary.entropy > 0
    assert summary.top1_top2_margin > 0
    with pytest.raises(ValueError):
        score_values([float("nan")] * 8, 0, "conditional_log_likelihood")


def test_cll_scorer_uses_active_pool_and_one_forward_per_state() -> None:
    config, _ = _inputs()
    tokenizer = _Tokenizer()
    model = _FakeModel()
    scorer = TorchChoiceScorer(tokenizer, model, _FakeTorch, config, 10**9)
    scorer.preflight([("Question", count) for count in (8, 7, 6, 5)])
    summaries = [scorer.score("Question", 1, count) for count in (8, 7, 6, 5)]
    assert scorer.law == "conditional_log_likelihood"
    assert scorer.forward_calls == 4
    assert model.calls == 4
    assert all(summary.law == "conditional_log_likelihood" for summary in summaries)
    assert all(value.to_devices == ["cuda"] for value in model.kwargs[0].values())
    assert [len(kwargs["input_ids"].rows) for kwargs in model.kwargs] == [8, 7, 6, 5]


def test_unpreflighted_actual_prefix_uses_fixed_cll_and_active_pool() -> None:
    config, tasks = _inputs()
    planned = _prompts(tasks, config)
    assert len(planned) == 132
    assert len(set(planned)) == 132
    tokenizer = _Tokenizer(context_variant="actual-prefix")
    model = _FakeModel()
    scorer = TorchChoiceScorer(tokenizer, model, _FakeTorch, config, 10**9)
    scorer.preflight([("gold-prefix", 8)])
    summary = scorer.score("actual-prefix", 0, 7)
    assert scorer.law == "conditional_log_likelihood"
    assert scorer.forward_calls == 1
    assert model.calls == 1
    assert summary.law == "conditional_log_likelihood"
    assert len(model.kwargs[0]["input_ids"].rows) == 7


def test_active_label_suffixes_must_be_pairwise_distinct() -> None:
    config, _ = _inputs()
    scorer = TorchChoiceScorer(
        _Tokenizer(duplicate_suffix=True), _FakeModel(), _FakeTorch, config, 10**9
    )
    with pytest.raises(ValueError, match="pairwise distinct"):
        scorer.preflight([("Question", 3)])


def test_model_loader_forwards_pinned_revision_without_local_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    config, _ = _inputs()
    tokenizer_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    model_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class _LoadedModel:
        def to(self, device: str) -> _LoadedModel:
            assert device == "cuda"
            return self

        def eval(self) -> None:
            return None

    class _TokenizerLoader:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> object:
            tokenizer_calls.append((args, kwargs))
            return object()

    class _ModelLoader:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> _LoadedModel:
            model_calls.append((args, kwargs))
            return _LoadedModel()

    fake_torch = types.SimpleNamespace(
        bfloat16=object(),
        cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
    )
    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=_ModelLoader,
        AutoTokenizer=_TokenizerLoader,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    _load_model(config)
    assert tokenizer_calls == [((config.model,), {"revision": config.revision})]
    assert model_calls == [
        ((config.model,), {"revision": config.revision, "torch_dtype": fake_torch.bfloat16})
    ]


def test_report_check_binds_the_actual_config_bytes() -> None:
    config, tasks = _inputs()
    payload = json.loads(
        subprocess.run(
            [
                sys.executable,
                "scripts/run_musique_candidate_scoring.py",
                "--check",
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "scripts"),
            },
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    assert payload["config_sha256"] == config.config_sha256
    assert payload["config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert payload["tasks"] == len(tasks)


def test_check_is_source_free_and_exclusive_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT / "scripts")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_musique_candidate_scoring.py",
            "--check",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["tasks"] == 45
    assert result["expected_states"] == 264
    assert result["model_calls"] is False
    assert not output.exists()
    write_exclusive(output, {"complete": True})
    with pytest.raises(FileExistsError):
        write_exclusive(output, {"complete": False})
