from __future__ import annotations

import builtins
import hashlib
import json
import math
import subprocess
import sys
import time
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from redco.experiments.musique_support_warm_start import (
    MODEL_NAME,
    MODEL_REVISION,
    WarmCandidate,
    WarmTask,
    load_config,
    load_snapshot,
    normalize_identity,
    selection_key,
    state_order_digest,
    training_states,
    warm_prompt,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/musique-ans-support-warm-start-v1.json"
SNAPSHOT = ROOT / "data/musique-ans-support-warm-start-v1.json"
sys.path.insert(0, str(ROOT / "scripts"))

import run_musique_support_warm_start as warm_runner  # noqa: E402
from run_musique_support_warm_start import (  # noqa: E402
    _adapter_delta_l2,
    _adapter_metadata,
    _adapter_norm,
    _load_qwen35,
    _prompt_prefix_length,
    _publish_adapter,
    _remove_adapter,
    _write_report,
)


def test_exact_snapshot_has_only_train_cohort_and_mixed_state_order() -> None:
    config, config_sha = load_config(CONFIG)
    tasks = load_snapshot(SNAPSHOT, str(config["source"]["snapshot_sha256"]))  # type: ignore[index]
    assert config_sha == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert len(tasks) == 512
    assert sum(task.depth == 2 for task in tasks) == 256
    assert sum(task.depth == 4 for task in tasks) == 256
    rows = training_states(tasks)
    assert len(rows) == 3072
    assert state_order_digest(rows) == config["state_order"]["sha256"]  # type: ignore[index]
    assert {row[0].depth for row in rows} == {2, 4}
    assert {row[1] for row in rows} == {0, 1}
    assert any(rows[index][0].depth != rows[index + 1][0].depth for index in range(len(rows) - 1))
    snapshot = json.loads(SNAPSHOT.read_bytes())
    assert set(snapshot["counts"]) == {"train_tasks", "train_states"}
    assert "diagnostic_tasks" not in snapshot["counts"]
    assert all("split" not in task for task in snapshot["tasks"])


def test_full_entry_hash_ranking_is_independent_of_scan_order() -> None:
    config, _ = load_config(CONFIG)
    tasks = load_snapshot(SNAPSHOT, str(config["source"]["snapshot_sha256"]))  # type: ignore[index]
    for depth in (2, 4):
        current = [task for task in tasks if task.depth == depth]
        reordered = list(reversed(current))
        assert [task.selection_key for task in current] == [
            task.selection_key
            for task in sorted(
                reordered, key=lambda item: selection_key(item.depth, item.source_row_sha256)
            )
        ]
    assert config["selection"]["namespace"] == "redco-musique-support-warm-start-v1"  # type: ignore[index]
    assert "canonical_json" in config["selection"]["law"]  # type: ignore[index]


def test_model_prompt_has_only_question_evidence_and_candidates() -> None:
    config, _ = load_config(CONFIG)
    tasks = load_snapshot(SNAPSHOT, str(config["source"]["snapshot_sha256"]))  # type: ignore[index]
    prompt = warm_prompt(tasks[0], 1, 0)
    assert tasks[0].question in prompt
    assert tasks[0].support_path[0] in prompt
    assert all(
        field not in prompt
        for field in (
            "question_decomposition",
            "answer_aliases",
            "is_supporting",
            "paragraph_support_idx",
            "source_row_sha256",
            "selection_key",
        )
    )
    assert all(f"[{index}]" in prompt for index in range(1, 8))


def test_train_candidate_titles_questions_and_texts_are_eval_disjoint() -> None:
    config, _ = load_config(CONFIG)
    tasks = load_snapshot(SNAPSHOT, str(config["source"]["snapshot_sha256"]))  # type: ignore[index]
    from redco.experiments.musique_capability import load_tasks

    eval_tasks = load_tasks(
        ROOT / "data/musique-ans-capability-v1.json",
        expected_sha256="07f75ea217779b754a37136d204de19f45f26679bdb6b7e056089cb5e54c70ed",
    )
    eval_titles = {candidate.title for task in eval_tasks for candidate in task.candidates}
    eval_questions = {normalize_identity(task.question) for task in eval_tasks}
    eval_texts = {
        hashlib.sha256(candidate.text.encode()).hexdigest()
        for task in eval_tasks
        for candidate in task.candidates
    }
    for task in tasks:
        assert normalize_identity(task.question) not in eval_questions
        for candidate in task.candidates:
            assert candidate.title not in eval_titles
            assert hashlib.sha256(candidate.text.encode()).hexdigest() not in eval_texts


def test_train_permutations_keep_active_pool_and_recompute_target() -> None:
    candidates = tuple(WarmCandidate(f"doc-{index}", f"text-{index}") for index in range(8))
    task = WarmTask(
        "4hop-train-000",
        4,
        "question",
        candidates,
        ("doc-0", "doc-1", "doc-2", "doc-3"),
        "0" * 64,
        selection_key(4, "0" * 64),
        (tuple(range(8)), (1, 0, 3, 2, 5, 4, 7, 6)),
    )
    states = list(task.states())
    assert len(states) == 8
    assert [len(state[2]) for state in states] == [8, 7, 6, 5, 8, 7, 6, 5]
    assert states[0][3] == 1
    assert states[4][3] == 2


def test_prompt_mask_requires_exact_nonempty_prefix() -> None:
    assert _prompt_prefix_length([1, 2], [1, 2, 9]) == 2
    with pytest.raises(ValueError):
        _prompt_prefix_length([1, 2], [1, 7, 9])
    with pytest.raises(ValueError):
        _prompt_prefix_length([1, 2], [1, 2])


def test_adapter_delta_detects_equal_norm_rotation() -> None:
    class Vector:
        def __init__(self, values: tuple[float, ...]) -> None:
            self.values = values

        def detach(self) -> Vector:
            return self

        def float(self) -> Vector:
            return self

        def cpu(self) -> Vector:
            return self

        def clone(self) -> Vector:
            return Vector(self.values)

        def __sub__(self, other: Vector) -> Vector:
            return Vector(
                tuple(
                    left - right
                    for left, right in zip(self.values, other.values, strict=True)
                )
            )

        def pow(self, exponent: int) -> Vector:
            return Vector(tuple(value**exponent for value in self.values))

        def sum(self) -> Vector:
            return Vector((sum(self.values),))

        def item(self) -> builtins.float:
            return self.values[0]

    initial = Vector((1.0, 0.0))
    final = Vector((0.0, 1.0))
    assert _adapter_norm([("adapter", initial)]) == pytest.approx(1.0)
    assert _adapter_norm([("adapter", final)]) == pytest.approx(1.0)
    assert _adapter_delta_l2({"adapter": initial}, [("adapter", final)]) == pytest.approx(
        math.sqrt(2.0)
    )


def test_training_optimizer_and_clip_receive_parameters_not_named_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Parameter:
        def detach(self) -> Parameter:
            return self

        def float(self) -> Parameter:
            return self

        def cpu(self) -> Parameter:
            return self

        def clone(self) -> Parameter:
            return self

    first = Parameter()
    second = Parameter()
    optimizer_seen: list[object] = []
    clip_seen: list[object] = []

    class Optimizer:
        def __init__(self, parameters: Any, lr: float) -> None:
            optimizer_seen.extend(parameters)
            self.param_groups: list[dict[str, object]] = [{"lr": lr}]

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True

        def step(self) -> None:
            pass

    class Loss:
        def detach(self) -> Loss:
            return self

        def item(self) -> float:
            return 1.0

        def __truediv__(self, _value: int) -> Loss:
            return self

        def backward(self) -> None:
            pass

    class Model:
        def train(self) -> None:
            pass

        def __call__(self, **_kwargs: Any) -> object:
            return types.SimpleNamespace(loss=Loss(), past_key_values=None)

    def install(
        _torch: Any, _model: Any, _training: Mapping[str, object]
    ) -> tuple[list[tuple[str, Any]], dict[str, object]]:
        return [("first", first), ("second", second)], {
            "replacement_count": 224,
            "trainable_parameter_count": 2,
            "adapter_tensors": [],
        }

    def rows(_tasks: Any) -> list[tuple[Any, tuple[int, ...], int, int]]:
        return [(object(), (0,), 0, 1)]

    def batch(*_args: Any) -> dict[str, object]:
        return {"input_ids": object(), "attention_mask": object(), "labels": object()}

    def norm(_parameters: Any) -> float:
        return 1.0

    def delta(_initial: Any, _parameters: Any) -> float:
        return 0.0

    def clip(parameters: Any, _limit: float) -> float:
        clip_seen.extend(parameters)
        return 0.0

    fake_torch = types.SimpleNamespace(
        manual_seed=lambda _seed: None,
        cuda=types.SimpleNamespace(manual_seed_all=lambda _seed: None),
        optim=types.SimpleNamespace(AdamW=Optimizer),
        nn=types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=clip)),
    )
    monkeypatch.setattr(warm_runner, "_install_lora", install)
    monkeypatch.setattr(warm_runner, "training_states", rows)
    monkeypatch.setattr(warm_runner, "_training_batch", batch)
    monkeypatch.setattr(warm_runner, "_adapter_norm", norm)
    monkeypatch.setattr(warm_runner, "_adapter_delta_l2", delta)
    training = {
        "seed": 1,
        "gradient_accumulation": 1,
        "updates": 1,
        "warmup_fraction": 0.0,
        "learning_rate": 0.1,
        "gradient_clip": 1.0,
        "material_adapter_delta_l2_min": 1e-8,
    }
    warm_runner._train(
        cast(Any, None), Model(), fake_torch, [], {"training": training}, time.monotonic() + 10
    )
    assert optimizer_seen == [first, second]
    assert clip_seen == [first, second]
    assert all(not isinstance(value, tuple) for value in optimizer_seen + clip_seen)


def test_adapter_metadata_is_ordered_and_records_float32_path() -> None:
    class Parameter:
        requires_grad = True
        shape = (8, 4)
        dtype = "torch.float32"

        def numel(self) -> int:
            return 32

    metadata = _adapter_metadata(
        [("layer.a.weight", Parameter()), ("layer.b.weight", Parameter())], 224
    )
    assert metadata["replacement_count"] == 224
    assert metadata["trainable_parameter_count"] == 64
    assert metadata["adapter_tensors"] == [
        {"key": "layer.a.weight", "shape": [8, 4], "dtype": "torch.float32"},
        {"key": "layer.b.weight", "shape": [8, 4], "dtype": "torch.float32"},
    ]


def test_adapter_publication_is_exclusive_and_load_validated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Tensor:
        shape = (2,)
        dtype = "torch.float32"

        def detach(self) -> Tensor:
            return self

        def to(self, _device: str) -> Tensor:
            return self

        def contiguous(self) -> Tensor:
            return self

    saved: dict[str, Tensor] = {}

    def save_file(state: dict[str, Tensor], path: str) -> None:
        saved.update(state)
        Path(path).write_bytes(b"safe-tensor-fixture")

    fake_safetensors_torch = types.ModuleType("safetensors.torch")
    fake_safetensors_torch.save_file = save_file  # type: ignore[attr-defined]
    fake_safetensors_torch.load_file = lambda _path, device: saved  # type: ignore[attr-defined]
    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors.torch = fake_safetensors_torch  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)

    output = tmp_path / "adapter.safetensors"
    metadata = {"replacement_count": 224, "trainable_parameter_count": 2, "adapter_tensors": []}
    record = _publish_adapter([("layer.a.weight", Tensor())], metadata, output)
    assert output.read_bytes() == b"safe-tensor-fixture"
    assert record["bytes"] == len(b"safe-tensor-fixture")
    assert record["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        _publish_adapter([("layer.a.weight", Tensor())], metadata, output)


def test_failed_gate_adapter_cleanup_is_explicit(tmp_path: Path) -> None:
    output = tmp_path / "adapter.safetensors"
    output.write_bytes(b"stale")
    _remove_adapter(output)
    assert not output.exists()


def test_report_failure_removes_successfully_published_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"published")

    def fail_report(_path: Path, _payload: dict[str, object]) -> None:
        raise OSError("report publication failed")

    monkeypatch.setattr(warm_runner, "write_exclusive", fail_report)
    with pytest.raises(OSError, match="report publication failed"):
        _write_report(tmp_path / "report.json", {}, adapter, True)
    assert not adapter.exists()
    assert list(tmp_path.iterdir()) == []


def test_check_subprocess_is_source_free_and_has_exact_update_arithmetic() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_musique_support_warm_start.py", "--check"],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src") + ";" + str(ROOT / "scripts")},
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["train_tasks"] == 512
    assert payload["train_tasks_by_depth"] == {"2": 256, "4": 256}
    assert payload["train_states"] == 3072
    assert "diagnostic_states" not in payload
    assert payload["model_calls"] is False
    assert payload["training_updates"] == 0


def test_qwen35_loader_binds_native_text_shape_and_allows_future_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = load_config(CONFIG)
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class Tokenizer:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> object:
            calls.append(("tokenizer", args, kwargs))
            return object()

    class Model:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(
                model_type="qwen3_5_text", num_hidden_layers=32, hidden_size=2560, use_cache=True
            )

        def to(self, device: str) -> Model:
            assert device == "cuda"
            return self

        def eval(self) -> Model:
            return self

    class Qwen3_5ForCausalLM(Model):
        pass

    class ModelLoader:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Qwen3_5ForCausalLM:
            calls.append(("model", args, kwargs))
            return Qwen3_5ForCausalLM()

    class ConfigLoader:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> object:
            calls.append(("config", args, kwargs))
            return types.SimpleNamespace(
                model_type="qwen3_5",
                architectures=["Qwen3_5ForConditionalGeneration"],
                text_config=types.SimpleNamespace(
                    model_type="qwen3_5_text", num_hidden_layers=32, hidden_size=2560
                ),
            )

    fake_torch = types.SimpleNamespace(
        bfloat16=object(),
        cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
    )
    fake_transformers = types.SimpleNamespace(
        __version__="5.6.2",
        AutoConfig=ConfigLoader,
        AutoTokenizer=Tokenizer,
        AutoModelForCausalLM=ModelLoader,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    _load_qwen35(config)
    model_call = next(item for item in calls if item[0] == "model")
    assert model_call[1] == (MODEL_NAME,)
    assert model_call[2]["revision"] == MODEL_REVISION
    assert model_call[2]["trust_remote_code"] is False
    assert model_call[2]["torch_dtype"] is fake_torch.bfloat16
    assert "use_cache" not in model_call[2]


def test_qwen35_repository_architecture_none_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    config, _ = load_config(CONFIG)
    fake_torch = types.SimpleNamespace(
        bfloat16=object(),
        cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
    )
    bad_config = types.SimpleNamespace(
        model_type="qwen3_5",
        architectures=None,
        text_config=types.SimpleNamespace(
            model_type="qwen3_5_text", num_hidden_layers=32, hidden_size=2560
        ),
    )
    fake_transformers = types.SimpleNamespace(
        __version__="5.6.2",
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: bad_config),
        AutoTokenizer=object(),
        AutoModelForCausalLM=object(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    with pytest.raises(RuntimeError, match="repository config"):
        _load_qwen35(config)
