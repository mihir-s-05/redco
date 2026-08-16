from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

from redco.experiments.musique_candidate_scoring import (
    ChoiceLaw,
    ScoreSummary,
    ScoringConfig,
    canonical_bytes,
    evaluate_states,
    k10_mixed_sum,
    load_inputs,
    prompt_for_state,
    report_payload,
    score_values,
    summarize,
    write_exclusive,
)
from redco.experiments.musique_matrix import load_matrix_inputs

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/musique-ans-candidate-scoring-v1.json"
MATRIX_CONFIG = ROOT / "configs/musique-ans-candidate-scoring-matrix-v1.json"
SNAPSHOT = ROOT / "data/musique-ans-capability-v1.json"
GATE_CONFIG = ROOT / "configs/musique-ans-capability-gate-v1.json"
sys.path.insert(0, str(ROOT / "scripts"))

from run_musique_candidate_scoring import (  # noqa: E402
    TorchChoiceScorer,
    _authenticate_qwen35_repository_config,
    _load_model,
    _load_qwen35_model,
    _matrix_report,
    _prompts,
    _run_matrix,
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

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.rows[0]) if self.rows else 0


class _Vector:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[int]:
        return self.values


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
        rows = kwargs["input_ids"].rows
        return _FakeOutput(len(rows), len(rows[0]))


class _Tokenizer:
    def __init__(self, duplicate_suffix: bool = False, context_variant: str = "") -> None:
        self.duplicate_suffix = duplicate_suffix
        self.context_variant = context_variant
        self.prompts: list[str] = []

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
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
    assert model.kwargs[0]["use_cache"] is False
    assert all(
        value.to_devices == ["cuda"] for key, value in model.kwargs[0].items() if key != "use_cache"
    )
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
        (
            (config.model,),
            {
                "revision": config.revision,
                "torch_dtype": fake_torch.bfloat16,
                "use_cache": False,
            },
        )
    ]


def test_qwen35_loader_uses_text_causal_owner_and_frozen_chat_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    matrix, _ = load_matrix_inputs(MATRIX_CONFIG, SNAPSHOT, GATE_CONFIG)
    config_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    tokenizer_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    model_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class Qwen3_5ForCausalLM:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(
                model_type="qwen3_5_text",
                architectures=["Qwen3_5ForCausalLM"],
                num_hidden_layers=32,
                hidden_size=2560,
            )

        def to(self, device: str) -> Qwen3_5ForCausalLM:
            assert device == "cuda"
            return self

        def eval(self) -> None:
            return None

    class _TokenizerLoader:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> _Tokenizer:
            tokenizer_calls.append((args, kwargs))
            return _Tokenizer()

    class _ConfigLoader:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            config_calls.append((args, kwargs))
            return types.SimpleNamespace(
                model_type="qwen3_5",
                architectures=["Qwen3_5ForConditionalGeneration"],
                text_config=types.SimpleNamespace(
                    model_type="qwen3_5_text", num_hidden_layers=32, hidden_size=2560
                ),
            )

    class _ModelLoader:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Qwen3_5ForCausalLM:
            model_calls.append((args, kwargs))
            return Qwen3_5ForCausalLM()

    fake_torch = types.SimpleNamespace(
        bfloat16=object(),
        cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
    )
    fake_transformers = types.SimpleNamespace(
        __version__="5.6.2",
        AutoConfig=_ConfigLoader,
        AutoTokenizer=_TokenizerLoader,
        AutoModelForCausalLM=_ModelLoader,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    adapter, model, _ = _load_qwen35_model(matrix, matrix.models[0])
    assert type(model).__name__ == "Qwen3_5ForCausalLM"
    assert config_calls == [
        (
            ("Qwen/Qwen3.5-4B",),
            {"revision": matrix.models[0].revision, "trust_remote_code": False},
        )
    ]
    assert tokenizer_calls == [
        (
            ("Qwen/Qwen3.5-4B",),
            {"revision": matrix.models[0].revision, "trust_remote_code": False},
        )
    ]
    assert model_calls == [
        (
            ("Qwen/Qwen3.5-4B",),
            {
                "revision": matrix.models[0].revision,
                "torch_dtype": fake_torch.bfloat16,
                "use_cache": False,
                "trust_remote_code": False,
            },
        )
    ]
    rendered = adapter.apply_chat_template(
        [{"role": "user", "content": "Question"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered.startswith("<chat>")
    encoded = adapter("Question", add_special_tokens=False, return_tensors="pt", truncation=False)
    assert set(encoded) == {"input_ids", "attention_mask"}


def test_cached_transformers_auto_mapping_uses_native_qwen35_text_config() -> None:
    transformers = importlib.import_module("transformers")
    qwen_config = importlib.import_module("transformers.models.qwen3_5.configuration_qwen3_5")
    composite = qwen_config.Qwen3_5Config()
    assert composite.model_type == "qwen3_5"
    assert composite.text_config.model_type == "qwen3_5_text"
    resolved = transformers.AutoModelForCausalLM._model_mapping[type(composite.text_config)]
    assert resolved.__name__ == "Qwen3_5ForCausalLM"


def test_repository_qwen35_config_requires_architecture() -> None:
    import types

    matrix, _ = load_matrix_inputs(MATRIX_CONFIG, SNAPSHOT, GATE_CONFIG)
    repository_config = types.SimpleNamespace(
        model_type="qwen3_5",
        architectures=None,
        text_config=types.SimpleNamespace(
            model_type="qwen3_5_text", num_hidden_layers=32, hidden_size=2560
        ),
    )
    with pytest.raises(RuntimeError, match="repository config"):
        _authenticate_qwen35_repository_config(repository_config, matrix.models[0])


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


def test_qwen35_matrix_config_and_k10_gate_are_explicit() -> None:
    config, tasks = load_matrix_inputs(MATRIX_CONFIG, SNAPSHOT, GATE_CONFIG)
    assert config.models[0].name == "Qwen/Qwen3.5-4B"
    assert config.models[1].name == "Qwen/Qwen3.5-9B"
    assert all(model.architecture == "Qwen3_5ForCausalLM" for model in config.models)
    assert all(
        model.repository_architecture == "Qwen3_5ForConditionalGeneration"
        for model in config.models
    )
    assert config.transformers_version == "5.6.2"
    assert config.expected_states_per_model == 264
    assert config.expected_forward_calls == 528
    assert config.use_cache is False
    assert config.four_hop_teacher_position_min_count == 8
    assert config.four_hop_greedy_full_path_min_count == 3
    assert config.four_hop_k10_mixed_sum_min == 5.0
    assert len(tasks) == 45
    assert k10_mixed_sum((0.5, 0.5)) == pytest.approx(1.99609375)


def _worker_report(config: Any, index: int, prompt_digest: str = "0" * 64) -> dict[str, Any]:
    identity = {
        "task_id": "synthetic",
        "hop_count": 2,
        "mode": "greedy",
        "step": 1,
        "candidate_titles": ["candidate-1", "candidate-2"],
        "selected_evidence_titles": [],
    }
    return {
        "config_sha256": config.config_sha256,
        "snapshot_sha256": config.snapshot_sha256,
        "model_binding": config.models[index].as_mapping(),
        "choice_law": "conditional_log_likelihood",
        "unload_confirmation": {"model_deleted": True, "cuda_cache_emptied": True},
        "generation": {
            "expected_forward_calls": 264,
            "actual_forward_calls": 264,
            "answer_calls": 0,
            "sampled_calls": 0,
            "training_updates": 0,
        },
        "capability_pass": False,
        "capability_blocked_reasons": ["synthetic"],
        "training_support_pass": False,
        "training_support_blocked_reasons": ["synthetic"],
        "training_experiment_eligible": False,
        "state_identity": [identity] * 264,
        "rendered_prompt_digest_algorithm": "sha256_utf8",
        "rendered_prompt_sha256": [prompt_digest] * 264,
    }


def test_matrix_final_report_requires_exact_state_and_prompt_comparability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_musique_candidate_scoring as runner

    config, _ = load_matrix_inputs(MATRIX_CONFIG, SNAPSHOT, GATE_CONFIG)
    first = _worker_report(config, 0)
    second = _worker_report(config, 1)
    monkeypatch.setattr(runner, "_task_deltas", lambda reports: [])
    payload = _matrix_report(config, (first, second), 1.0)
    assert payload["matrix_complete"] is True
    comparability = payload["comparability"]
    assert isinstance(comparability, Mapping)
    assert comparability["rendered_prompt_digest_algorithm"] == "sha256_utf8"
    second["rendered_prompt_sha256"] = ["1" * 64] * 264
    with pytest.raises(ValueError, match="ordered state prompts"):
        _matrix_report(config, (first, second), 1.0)


def test_worker_two_failure_publishes_worker_one_and_stops_without_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, _ = load_matrix_inputs(MATRIX_CONFIG, SNAPSHOT, GATE_CONFIG)
    calls: list[int] = []

    def fake_run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        marker = command.index("--matrix-worker")
        index = int(command[marker + 1])
        calls.append(index)
        if index == 0:
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(canonical_bytes(_worker_report(config, 0)))
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 9, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    output = tmp_path / "matrix.json"
    with pytest.raises(RuntimeError, match="failed technically"):
        _run_matrix(MATRIX_CONFIG, SNAPSHOT, GATE_CONFIG, output)
    payload = json.loads(output.read_bytes())
    assert calls == [0, 1]
    assert payload["matrix_complete"] is False
    assert len(payload["completed_model_results"]) == 1
    assert payload["technical_failure"] == {
        "model_index": 1,
        "model": "Qwen/Qwen3.5-9B",
        "failure_class": "WorkerProcessError",
    }
    assert payload["generation"]["expected_forward_calls"] == 528
    assert payload["generation"]["known_completed_forward_calls"] == 264
    assert payload["generation"]["failed_worker_forward_calls_known"] is False
    assert payload["generation"]["failed_worker_forward_calls"] is None
    assert payload["generation"]["completed_model_count"] == 1
    assert payload["paired_task_deltas_9b_minus_4b"] == []
    assert payload["scientific_combined_inference"] is False


def test_worker_one_failure_publishes_zero_completed_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(command, 23, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    output = tmp_path / "matrix.json"
    with pytest.raises(RuntimeError, match="failed technically"):
        _run_matrix(MATRIX_CONFIG, SNAPSHOT, GATE_CONFIG, output)
    payload = json.loads(output.read_bytes())
    assert payload["matrix_complete"] is False
    assert payload["completed_model_results"] == []
    assert payload["technical_failure"] == {
        "model_index": 0,
        "model": "Qwen/Qwen3.5-4B",
        "failure_class": "WorkerProcessError",
    }
    assert payload["generation"]["expected_forward_calls"] == 528
    assert payload["generation"]["known_completed_forward_calls"] == 0
    assert payload["generation"]["failed_worker_forward_calls_known"] is False
    assert payload["generation"]["failed_worker_forward_calls"] is None


def test_remaining_window_failure_publishes_before_second_worker_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, _ = load_matrix_inputs(MATRIX_CONFIG, SNAPSHOT, GATE_CONFIG)
    calls: list[int] = []

    def fake_run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        index = int(command[command.index("--matrix-worker") + 1])
        calls.append(index)
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(canonical_bytes(_worker_report(config, index)))
        return subprocess.CompletedProcess(command, 0, "", "")

    clock = iter((0.0, 0.0, 1301.0, 1301.0))
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    output = tmp_path / "matrix.json"
    with pytest.raises(TimeoutError, match="window"):
        _run_matrix(MATRIX_CONFIG, SNAPSHOT, GATE_CONFIG, output)
    payload = json.loads(output.read_bytes())
    assert calls == [0]
    assert payload["matrix_complete"] is False
    assert payload["technical_failure"] == {
        "model_index": 1,
        "model": "Qwen/Qwen3.5-9B",
        "failure_class": "InsufficientRemainingWindow",
    }
    assert payload["generation"]["expected_forward_calls"] == 528
    assert payload["generation"]["known_completed_forward_calls"] == 264
    assert payload["generation"]["failed_worker_forward_calls_known"] is True
    assert payload["generation"]["failed_worker_forward_calls"] == 0
    assert "actual_forward_calls" not in payload["generation"]


def test_post_worker_aggregation_failure_publishes_both_completed_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, _ = load_matrix_inputs(MATRIX_CONFIG, SNAPSHOT, GATE_CONFIG)
    calls: list[int] = []

    def fake_run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        index = int(command[command.index("--matrix-worker") + 1])
        calls.append(index)
        output = Path(command[command.index("--output") + 1])
        digest = "0" * 64 if index == 0 else "1" * 64
        output.write_bytes(canonical_bytes(_worker_report(config, index, digest)))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    output = tmp_path / "matrix.json"
    with pytest.raises(RuntimeError, match="aggregation"):
        _run_matrix(MATRIX_CONFIG, SNAPSHOT, GATE_CONFIG, output)
    payload = json.loads(output.read_bytes())
    assert calls == [0, 1]
    assert payload["matrix_complete"] is False
    assert len(payload["completed_model_results"]) == 2
    assert payload["technical_failure"] == {
        "model_index": -1,
        "model": "matrix",
        "failure_class": "MatrixAggregationError",
    }
    assert payload["generation"]["expected_forward_calls"] == 528
    assert payload["generation"]["known_completed_forward_calls"] == 528
    assert payload["generation"]["failed_worker_forward_calls_known"] is False
    assert payload["generation"]["failed_worker_forward_calls"] is None
    assert payload["paired_task_deltas_9b_minus_4b"] == []
    assert payload["scientific_combined_inference"] is False


def test_compact_four_b_result_fails_the_stronger_gate() -> None:
    result = json.loads((ROOT / "results/musique-ans-candidate-scoring-v1.json").read_bytes())
    preview = result["stronger_gate_preview"]
    assert preview["passed"] is False
    assert preview["four_hop_teacher_position_top1_correct"] == [8, 4, 5, 10]
    assert preview["four_hop_greedy_full_paths"] == 0
    assert preview["four_hop_k10_mixed_sum"] < 5.0


def test_capability_and_training_support_gates_remain_separate() -> None:
    config, _ = _inputs()
    payload = report_payload(
        config,
        config.snapshot_sha256,
        (),
        {"2": {"training_support_reasons": []}, "4": {"training_support_reasons": ["low m"]}},
        ["capability failure"],
        elapsed_seconds=0.0,
        forward_calls=0,
        law="conditional_log_likelihood",
    )
    assert payload["capability_pass"] is False
    assert payload["training_support_pass"] is False
    assert payload["training_experiment_eligible"] is False
    assert payload["blocked_reasons"] == ["capability failure"]


def test_matrix_check_binds_models_and_all_thresholds(tmp_path: Path) -> None:
    output = tmp_path / "unused.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_musique_candidate_scoring.py",
            "--check",
            "--config",
            str(MATRIX_CONFIG),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "scripts"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["models"][0]["name"] == "Qwen/Qwen3.5-4B"
    assert payload["models"][1]["name"] == "Qwen/Qwen3.5-9B"
    assert payload["expected_states_per_model"] == 264
    assert payload["expected_forward_calls"] == 528
    assert payload["runtime"]["model_calls"] is False
    assert payload["gate"]["four_hop_teacher_position_min_count"] == 8
    assert payload["gate"]["four_hop_greedy_full_path_min_count"] == 3
    assert payload["gate"]["four_hop_k10_mixed_sum_min"] == 5.0
    assert not output.exists()


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
