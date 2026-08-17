from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from redco.experiments.musique_candidate_scoring import (
    ChoiceLaw,
    ScoreSummary,
    ScoringConfig,
    evaluate_states,
    load_inputs,
    prompt_for_state,
    report_payload,
    score_values,
    summarize,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/musique-ans-candidate-scoring-v1.json"
SNAPSHOT = ROOT / "data/musique-ans-capability-v1.json"
GATE_CONFIG = ROOT / "configs/musique-ans-capability-gate-v1.json"
sys.path.insert(0, str(ROOT / "scripts"))

from run_musique_candidate_scoring import (  # noqa: E402
    TorchChoiceScorer,
    _load_model,
)


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
        selected = 0
        if self.forward_calls == 1:
            selected = next(
                index
                for index in range(candidate_count)
                if f"[{index + 1}] {self.later_support}\n" in prompt
            )
        scores = [0.0] * candidate_count
        scores[selected] = 1.0
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

    def __getitem__(self, key: object) -> Any:
        if isinstance(key, tuple):
            return _Scalar(0.0)
        if isinstance(key, int):
            return type("Vector", (), {"tolist": lambda _self: self.rows[key]})()
        raise AssertionError(key)

    def to(self, device: str) -> _Tensor:
        self.to_devices.append(device)
        return self

    def tolist(self) -> list[list[int]]:
        return self.rows


class _FakeLogits:
    def __init__(self, batch: int, sequence: int) -> None:
        self.shape = (batch, sequence, 4096)

    def __getitem__(self, key: tuple[int, int, int]) -> _Scalar:
        del key
        return _Scalar(0.0)


class _FakeOutput:
    def __init__(self, batch: int, sequence: int) -> None:
        self.logits = _FakeLogits(batch, sequence)
        self.past_key_values = None


class _Mode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        del args


class _FakeTorch:
    @staticmethod
    def inference_mode() -> _Mode:
        return _Mode()

    @staticmethod
    def log_softmax(logits: object, *, dim: int) -> object:
        del logits, dim
        return type("LogProb", (), {"__getitem__": lambda _self, _key: _Scalar(0.0)})()


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _FakeOutput:
        self.calls += 1
        self.kwargs.append(kwargs)
        rows = kwargs["input_ids"].rows
        return _FakeOutput(len(rows), len(rows[0]))


class _Tokenizer:
    def __init__(self, duplicate_suffix: bool = False) -> None:
        self.chat_template = "<frozen-test-chat-template>"
        self.duplicate_suffix = duplicate_suffix

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ) -> str:
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return "<chat>" + messages[0]["content"]

    def __call__(self, value: str | list[str], **kwargs: Any) -> Mapping[str, _Tensor]:
        del kwargs
        texts = [value] if isinstance(value, str) else value
        sequences = []
        for text in texts:
            ids = [ord(char) for char in text]
            if self.duplicate_suffix and text.endswith("3"):
                ids[-1] = ord("2")
            sequences.append(ids)
        width = max(len(row) for row in sequences)
        return {
            "input_ids": _Tensor([row + [0] * (width - len(row)) for row in sequences]),
            "attention_mask": _Tensor(
                [[1] * len(row) + [0] * (width - len(row)) for row in sequences]
            ),
        }


def _inputs() -> tuple[ScoringConfig, tuple[Any, ...]]:
    return load_inputs(CONFIG, SNAPSHOT, GATE_CONFIG)


def test_frozen_inputs_and_exact_264_states() -> None:
    config, tasks = _inputs()
    assert len(tasks) == 45
    prompt = prompt_for_state(config, tasks[0], (), tasks[0].candidates)
    assert tasks[0].question in prompt and tasks[0].candidates[0].text in prompt
    assert all(field not in prompt for field in ("question_decomposition", "answer_aliases"))
    scorer = FixedScorer()
    records = evaluate_states(tasks, config, scorer)
    by_hop, blocked = summarize(tasks, config, records)
    assert len(records) == scorer.forward_calls == 264
    assert blocked == []
    assert by_hop["2"]["teacher_top1_accuracy"] == 1.0  # type: ignore[index]


def test_greedy_off_order_selection_keeps_all_targets_defined() -> None:
    config, tasks = _inputs()
    task = tasks[0]
    records = evaluate_states(tasks, config, OffOrderScorer(task.support_path[1]))
    greedy = [
        record for record in records if record.task_id == task.task_id and record.mode == "greedy"
    ]
    assert len(greedy) == task.hop_count
    assert greedy[0].selected_candidate == task.support_path[1]
    assert tuple(record.selected_candidate for record in greedy) != task.support_path


def test_score_values_and_nonfinite_rejection() -> None:
    summary = score_values([3.0, 2.0, 1.0, 0.0], 2, "conditional_log_likelihood")
    assert summary.selected_index == 0 and summary.gold_rank == 3
    with pytest.raises(ValueError):
        score_values([float("nan")] * 4, 0, "conditional_log_likelihood")


def test_cll_uses_active_pool_and_one_forward_per_state() -> None:
    config, _ = _inputs()
    model = _FakeModel()
    scorer = TorchChoiceScorer(_Tokenizer(), model, _FakeTorch, config, 10**9)
    scorer.preflight([("Question", count) for count in (8, 7, 6, 5)])
    summaries = [scorer.score("Question", 1, count) for count in (8, 7, 6, 5)]
    assert scorer.forward_calls == model.calls == 4
    assert all(summary.law == "conditional_log_likelihood" for summary in summaries)
    assert [len(kwargs["input_ids"].rows) for kwargs in model.kwargs] == [8, 7, 6, 5]
    assert all(kwargs["use_cache"] is False for kwargs in model.kwargs)


def test_active_suffixes_must_be_distinct() -> None:
    config, _ = _inputs()
    scorer = TorchChoiceScorer(
        _Tokenizer(duplicate_suffix=True), _FakeModel(), _FakeTorch, config, 10**9
    )
    with pytest.raises(ValueError, match="pairwise distinct"):
        scorer.preflight([("Question", 3)])


def test_old_model_loader_forwards_revision_without_local_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    config, _ = _inputs()
    tokenizer_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    model_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class Loaded:
        def to(self, device: str) -> Loaded:
            assert device == "cuda"
            return self

        def eval(self) -> Loaded:
            return self

    class Tokenizer:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> object:
            tokenizer_calls.append((args, kwargs))
            return object()

    class Model:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Loaded:
            model_calls.append((args, kwargs))
            return Loaded()

    fake_torch = types.SimpleNamespace(
        bfloat16=object(),
        cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
    )
    fake_transformers = types.SimpleNamespace(AutoModelForCausalLM=Model, AutoTokenizer=Tokenizer)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    _load_model(config)
    assert "local_files_only" not in tokenizer_calls[0][1]
    assert "local_files_only" not in model_calls[0][1]


def test_check_is_source_free_and_binds_actual_config() -> None:
    config, tasks = _inputs()
    payload = json.loads(
        subprocess.run(
            [sys.executable, "scripts/run_musique_candidate_scoring.py", "--check"],
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
    assert (
        payload["config_sha256"]
        == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
        == config.config_sha256
    )
    assert payload["tasks"] == len(tasks)


def test_capability_and_support_gates_are_separate() -> None:
    config, _tasks = _inputs()
    payload = report_payload(
        config,
        config.snapshot_sha256,
        (),
        {"2": {"training_support_reasons": []}, "4": {"training_support_reasons": ["low support"]}},
        ["blocked"],
        elapsed_seconds=1.0,
        forward_calls=0,
        law="conditional_log_likelihood",
    )
    assert payload["capability_pass"] is False and payload["training_support_pass"] is False
