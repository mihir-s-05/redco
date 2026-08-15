from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest

from redco.experiments.musique_capability import Candidate, parse_numbered_choice

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_musique_capability_gate import (  # noqa: E402
    TransformersPolicy,
    _prepare_cuda_model,
    _write_new,
)


class _Tensor:
    def __init__(self, length: int) -> None:
        self.shape = (1, length)
        self.moved: list[str] = []

    def to(self, device: str) -> _Tensor:
        self.moved.append(device)
        return self

    def __getitem__(self, key: Any) -> _Tensor:
        return self


class _Mode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class _Torch:
    bfloat16 = object()
    manual_seeds: ClassVar[list[int]] = []

    class cuda:
        seed_values: ClassVar[list[int]] = []

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def manual_seed_all(seed: int) -> None:
            _Torch.cuda.seed_values.append(seed)

    class random:
        @staticmethod
        def fork_rng(*, devices: list[int]) -> _Mode:
            assert devices == [0]
            return _Mode()

    @staticmethod
    def manual_seed(seed: int) -> None:
        _Torch.manual_seeds.append(seed)

    @staticmethod
    def inference_mode() -> _Mode:
        return _Mode()


class _Model:
    def __init__(self) -> None:
        self.kwargs: list[dict[str, Any]] = []
        self.devices: list[str] = []
        self.evaluated = False

    def to(self, device: str) -> _Model:
        self.devices.append(device)
        return self

    def eval(self) -> _Model:
        self.evaluated = True
        return self

    def generate(self, **kwargs: Any) -> _Tensor:
        self.kwargs.append(kwargs)
        return _Tensor(8)


class _Tokenizer:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.truncations: list[bool] = []

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

    def __call__(
        self, prompt: str, *, return_tensors: str, truncation: bool
    ) -> Mapping[str, _Tensor]:
        assert return_tensors == "pt"
        self.truncations.append(truncation)
        return {"input_ids": _Tensor(len(prompt)), "attention_mask": _Tensor(len(prompt))}

    def decode(self, tokens: Any, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "1"


def test_source_free_check_has_no_model_or_output_side_effect(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT / "scripts")
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_musique_capability_gate.py",
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
    assert result["tasks_by_depth"] == {"2": 24, "4": 21}
    assert result["planned_generations"] == 882
    assert result["gold_fields_in_policy"] is False
    assert result["model_calls"] is False
    assert not output.exists()


def test_cuda_policy_uses_chat_template_no_truncation_and_sampling_kwargs() -> None:
    _Torch.manual_seeds.clear()
    _Torch.cuda.seed_values.clear()
    tokenizer = _Tokenizer()
    model = _Model()
    policy = TransformersPolicy(
        tokenizer,
        model,
        _Torch,
        choice_max_new_tokens=4,
        answer_max_new_tokens=32,
        max_input_tokens=8192,
        deadline=10**9,
        temperature=1.0,
        top_p=0.95,
    )
    candidates = (Candidate("Alpha", "document"), Candidate("Beta", "document"))
    greedy = policy.choose("task", 1, "Question", (), candidates, sample=None, sampling_seed=None)
    sampled = policy.choose("task", 1, "Question", (), candidates, sample=0, sampling_seed=123)
    sampled_again = policy.choose(
        "task", 1, "Question", (), candidates, sample=1, sampling_seed=456
    )
    answer = policy.answer("task", "Question", candidates, mode="greedy")
    assert greedy.title == sampled.title == sampled_again.title == "Alpha"
    assert answer == "1"
    assert model.kwargs[0]["do_sample"] is False
    assert model.kwargs[1]["do_sample"] is True
    assert model.kwargs[1]["temperature"] == 1.0
    assert model.kwargs[1]["top_p"] == 0.95
    assert "generator" not in model.kwargs[1]
    assert model.kwargs[0]["max_new_tokens"] == 4
    assert model.kwargs[3]["max_new_tokens"] == 32
    assert policy.call_count == 4
    assert _Torch.manual_seeds == [123, 456]
    assert _Torch.cuda.seed_values == [123, 456]
    assert tokenizer.truncations == [False, False, False, False]
    assert all(
        "question_decomposition" not in prompt and "answer_aliases" not in prompt
        for prompt in tokenizer.prompts
    )


def test_cuda_model_preparation_requires_one_device_and_evaluates() -> None:
    model = _Model()
    prepared = _prepare_cuda_model(_Torch, model)
    assert prepared is model
    assert model.devices == ["cuda"]
    assert model.evaluated is True


def test_numbered_choice_does_not_accept_title_substrings() -> None:
    candidates = (Candidate("Alpha", "document"),)
    assert not parse_numbered_choice("Alpha", candidates).parse_valid
    assert parse_numbered_choice("1", candidates).title == "Alpha"


def test_report_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        _write_new(output, {"complete": True})
