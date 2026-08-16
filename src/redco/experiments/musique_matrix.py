"""Versioned two-model MuSiQue matrix binding, separate from the CLL owner."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from redco.experiments.musique_candidate_scoring import ScoringConfig, write_exclusive
from redco.experiments.musique_capability import MuSiQueTask, load_tasks

_PROMPT = (
    "Question:\n{question}\n\nAlready selected evidence:\n{evidence}\n\n"
    "Choose exactly one next document from the numbered candidates. Return only one integer "
    "from the displayed range.\n\nCandidates:\n{candidates}"
)


@dataclass(frozen=True, slots=True)
class MatrixModel:
    name: str
    revision: str
    model_type: str
    architecture: str
    repository_architecture: str
    text_model_type: str
    layers: int
    hidden_size: int
    index_total_size: int
    index_shards: int

    def as_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "revision": self.revision,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "repository_architecture": self.repository_architecture,
            "text_model_type": self.text_model_type,
            "layers": self.layers,
            "hidden_size": self.hidden_size,
            "index_total_size": self.index_total_size,
            "index_shards": self.index_shards,
        }


@dataclass(frozen=True, slots=True)
class MatrixConfig:
    schema_version: int
    config_sha256: str
    snapshot_path: str
    snapshot_sha256: str
    gate_config_path: str
    gate_config_sha256: str
    task_counts: tuple[tuple[int, int], ...]
    models: tuple[MatrixModel, ...]
    labels: tuple[str, ...]
    prompt_template: str
    max_input_tokens: int
    candidate_count: int
    expected_states_per_model: int
    expected_forward_calls: int
    teacher_margin: float
    four_hop_greedy_full_path_min: float
    four_hop_teacher_position_min_count: int
    four_hop_greedy_full_path_min_count: int
    four_hop_k10_mixed_sum_min: float
    max_seconds: int
    minimum_seconds_per_model: int
    cuda_devices: int
    dtype: str
    transformers_version: str
    use_cache: bool
    unload_between_models: bool
    empty_cuda_cache: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_matrix_config(path: Path) -> MatrixConfig:
    data = path.read_bytes()
    config_sha256 = hashlib.sha256(data).hexdigest()
    value = json.loads(data)
    if type(value) is not dict or set(value) != {
        "choice",
        "cohort",
        "dataset",
        "gate",
        "mode",
        "models",
        "runtime",
        "schema_version",
        "source",
    }:
        raise ValueError("matrix config has the wrong schema")
    if (
        value["schema_version"] != 1
        or value["mode"] != "qwen3_5_matrix"
        or value["dataset"] != "MuSiQue-Ans"
        or value["cohort"] != "short_document_linear_chain"
    ):
        raise ValueError("matrix config is outside the frozen diagnostic")
    source, choice, gate, runtime, models = (
        value["source"],
        value["choice"],
        value["gate"],
        value["runtime"],
        value["models"],
    )
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
        or type(choice) is not dict
        or set(choice) != {"chat_template", "labels", "max_input_tokens", "prompt_template"}
        or type(gate) is not dict
        or set(gate)
        != {
            "candidate_count",
            "expected_forward_calls",
            "expected_states_per_model",
            "four_hop_greedy_full_path_min",
            "four_hop_greedy_full_path_min_count",
            "four_hop_k10_mixed_sum_min",
            "four_hop_teacher_position_min_count",
            "teacher_margin",
        }
        or type(runtime) is not dict
        or set(runtime)
        != {
            "cuda_devices",
            "dtype",
            "empty_cuda_cache",
            "max_seconds",
            "minimum_seconds_per_model",
            "models_run_in_separate_processes",
            "transformers_version",
            "unload_between_models",
            "use_cache",
        }
        or type(models) is not list
        or len(models) != 2
    ):
        raise ValueError("matrix config section has the wrong schema")
    source = cast(dict[str, object], source)
    choice = cast(dict[str, object], choice)
    gate = cast(dict[str, object], gate)
    runtime = cast(dict[str, object], runtime)
    if (
        source["snapshot_path"] != "data/musique-ans-capability-v1.json"
        or source["gate_config_path"] != "configs/musique-ans-capability-gate-v1.json"
        or source["snapshot_sha256"]
        != "07f75ea217779b754a37136d204de19f45f26679bdb6b7e056089cb5e54c70ed"
        or source["gate_config_sha256"]
        != "9978ac70b684026b15073786c960b33bfd3d4d9973ea41bb25ccb82a80eea646"
        or source["task_counts"] != {"2": 24, "4": 21}
        or choice["labels"] != [str(index) for index in range(1, 9)]
        or choice["prompt_template"] != _PROMPT
        or choice["chat_template"]
        != {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        or choice["max_input_tokens"] != 8192
        or gate["candidate_count"] != 8
        or gate["expected_states_per_model"] != 264
        or gate["expected_forward_calls"] != 528
        or gate["teacher_margin"] != 0.1
        or gate["four_hop_greedy_full_path_min"] != 0.1
        or gate["four_hop_teacher_position_min_count"] != 8
        or gate["four_hop_greedy_full_path_min_count"] != 3
        or gate["four_hop_k10_mixed_sum_min"] != 5.0
        or runtime["cuda_devices"] != 1
        or runtime["dtype"] != "bfloat16"
        or runtime["empty_cuda_cache"] is not True
        or runtime["models_run_in_separate_processes"] is not True
        or runtime["transformers_version"] != "5.6.2"
        or runtime["unload_between_models"] is not True
        or runtime["use_cache"] is not False
        or runtime["max_seconds"] != 1500
        or runtime["minimum_seconds_per_model"] != 300
    ):
        raise ValueError("matrix config changed a frozen binding")
    expected = (
        ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", 2560, 9319737856, 2),
        ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", 4096, 19306216416, 4),
    )
    parsed: list[MatrixModel] = []
    for item, (name, revision, hidden_size, total_size, shards) in zip(
        models, expected, strict=True
    ):
        if type(item) is not dict or set(item) != {
            "architecture",
            "hidden_size",
            "index_shards",
            "index_total_size",
            "layers",
            "model_type",
            "name",
            "repository_architecture",
            "revision",
            "text_model_type",
        }:
            raise ValueError("matrix model binding has the wrong schema")
        if (
            item["name"] != name
            or item["revision"] != revision
            or item["model_type"] != "qwen3_5"
            or item["architecture"] != "Qwen3_5ForCausalLM"
            or item["repository_architecture"] != "Qwen3_5ForConditionalGeneration"
            or item["text_model_type"] != "qwen3_5_text"
            or item["layers"] != 32
            or item["hidden_size"] != hidden_size
            or item["index_total_size"] != total_size
            or item["index_shards"] != shards
        ):
            raise ValueError("matrix model binding changed")
        parsed.append(
            MatrixModel(
                item["name"],
                item["revision"],
                item["model_type"],
                item["architecture"],
                item["repository_architecture"],
                item["text_model_type"],
                item["layers"],
                item["hidden_size"],
                item["index_total_size"],
                item["index_shards"],
            )
        )
    return MatrixConfig(
        1,
        config_sha256,
        source["snapshot_path"],
        source["snapshot_sha256"],
        source["gate_config_path"],
        source["gate_config_sha256"],
        ((2, 24), (4, 21)),
        tuple(parsed),
        tuple(choice["labels"]),
        choice["prompt_template"],
        choice["max_input_tokens"],
        gate["candidate_count"],
        gate["expected_states_per_model"],
        gate["expected_forward_calls"],
        gate["teacher_margin"],
        gate["four_hop_greedy_full_path_min"],
        gate["four_hop_teacher_position_min_count"],
        gate["four_hop_greedy_full_path_min_count"],
        gate["four_hop_k10_mixed_sum_min"],
        runtime["max_seconds"],
        runtime["minimum_seconds_per_model"],
        runtime["cuda_devices"],
        runtime["dtype"],
        runtime["transformers_version"],
        runtime["use_cache"],
        runtime["unload_between_models"],
        runtime["empty_cuda_cache"],
    )


def load_matrix_inputs(
    config_path: Path, snapshot_path: Path, gate_config_path: Path
) -> tuple[MatrixConfig, tuple[MuSiQueTask, ...]]:
    config = load_matrix_config(config_path)
    if (
        _sha256(gate_config_path) != config.gate_config_sha256
        or _sha256(snapshot_path) != config.snapshot_sha256
    ):
        raise ValueError("matrix input hash does not match the frozen binding")
    if (
        gate_config_path.resolve() != Path(config.gate_config_path).resolve()
        or snapshot_path.resolve() != Path(config.snapshot_path).resolve()
    ):
        raise ValueError("matrix input path is not the frozen path")
    return config, load_tasks(snapshot_path, expected_sha256=config.snapshot_sha256)


def matrix_scoring_config(config: MatrixConfig, model: MatrixModel) -> ScoringConfig:
    return ScoringConfig(
        schema_version=3,
        config_sha256=config.config_sha256,
        snapshot_path=config.snapshot_path,
        snapshot_sha256=config.snapshot_sha256,
        gate_config_path=config.gate_config_path,
        gate_config_sha256=config.gate_config_sha256,
        task_counts=config.task_counts,
        model=model.name,
        revision=model.revision,
        labels=config.labels,
        prompt_template=config.prompt_template,
        max_input_tokens=config.max_input_tokens,
        candidate_count=config.candidate_count,
        expected_states=config.expected_states_per_model,
        teacher_margin=config.teacher_margin,
        four_hop_greedy_full_path_min=config.four_hop_greedy_full_path_min,
        four_hop_teacher_position_min_count=config.four_hop_teacher_position_min_count,
        four_hop_greedy_full_path_min_count=config.four_hop_greedy_full_path_min_count,
        four_hop_k10_mixed_sum_min=config.four_hop_k10_mixed_sum_min,
        max_seconds=config.max_seconds,
        cuda_devices=config.cuda_devices,
        dtype=config.dtype,
        use_cache=config.use_cache,
        stronger_gate=True,
    )


def matrix_check_payload(
    config: MatrixConfig, snapshot_sha256: str, tasks: Sequence[MuSiQueTask]
) -> dict[str, object]:
    return {
        "mode": "check",
        "schema_version": config.schema_version,
        "dataset": "MuSiQue-Ans",
        "cohort": "short_document_linear_chain",
        "config_sha256": config.config_sha256,
        "snapshot_sha256": snapshot_sha256,
        "models": [model.as_mapping() for model in config.models],
        "choice_law": "conditional_log_likelihood",
        "tasks": len(tasks),
        "tasks_by_depth": {
            str(depth): sum(task.hop_count == depth for task in tasks) for depth in (2, 4)
        },
        "expected_states_per_model": config.expected_states_per_model,
        "expected_forward_calls": config.expected_forward_calls,
        "runtime": {
            "cuda_devices": config.cuda_devices,
            "dtype": config.dtype,
            "use_cache": config.use_cache,
            "models_run_in_separate_processes": True,
            "transformers_version": config.transformers_version,
            "model_calls": False,
        },
        "gate": {
            "teacher_margin": config.teacher_margin,
            "four_hop_teacher_position_min_count": config.four_hop_teacher_position_min_count,
            "four_hop_greedy_full_path_min_count": config.four_hop_greedy_full_path_min_count,
            "four_hop_k10_mixed_sum_min": config.four_hop_k10_mixed_sum_min,
        },
        "gold_fields_in_prompt": False,
        "sampling_calls": 0,
        "training_updates": 0,
    }


def incomplete_matrix_report(
    config: MatrixConfig,
    reports: Sequence[Mapping[str, Any]],
    failed_index: int,
    failure_class: str,
    elapsed_seconds: float,
    failed_worker_dispatched: bool | None,
) -> dict[str, object]:
    known_completed_calls = 0
    for report in reports:
        generation = report.get("generation")
        if isinstance(generation, Mapping) and type(generation.get("actual_forward_calls")) is int:
            known_completed_calls += generation["actual_forward_calls"]
    return {
        "schema_version": config.schema_version,
        "dataset": "MuSiQue-Ans",
        "cohort": "short_document_linear_chain",
        "mode": "qwen3_5_constrained_scoring_matrix",
        "config_sha256": config.config_sha256,
        "snapshot_sha256": config.snapshot_sha256,
        "models": [model.as_mapping() for model in config.models],
        "matrix_complete": False,
        "completed_model_results": [dict(report) for report in reports],
        "technical_failure": {
            "model_index": failed_index,
            "model": (
                config.models[failed_index].name
                if 0 <= failed_index < len(config.models)
                else "matrix"
            ),
            "failure_class": failure_class,
        },
        "generation": {
            "expected_forward_calls": config.expected_forward_calls,
            "known_completed_forward_calls": known_completed_calls,
            "failed_worker_forward_calls_known": failed_worker_dispatched is False,
            "failed_worker_forward_calls": (0 if failed_worker_dispatched is False else None),
            "expected_model_count": len(config.models),
            "completed_model_count": len(reports),
            "sampling_calls": 0,
            "training_updates": 0,
        },
        "paired_task_deltas_9b_minus_4b": [],
        "eligible_models": [],
        "scientific_combined_inference": False,
        "runtime": {
            "cuda_device_count": config.cuda_devices,
            "dtype": config.dtype,
            "use_cache": config.use_cache,
            "separate_processes": True,
            "unloaded_between_models": True,
            "elapsed_seconds": elapsed_seconds,
            "deadline_cooperative": True,
        },
        "gold_fields_in_prompt": False,
        "authority": {"sampling": False, "training": False},
    }


def publish_matrix_terminal_failure(
    output_path: Path,
    config: MatrixConfig,
    reports: Sequence[Mapping[str, Any]],
    failed_index: int,
    failure_class: str,
    elapsed_seconds: float,
    failed_worker_dispatched: bool | None,
) -> None:
    write_exclusive(
        output_path,
        incomplete_matrix_report(
            config,
            reports,
            failed_index,
            failure_class,
            elapsed_seconds,
            failed_worker_dispatched,
        ),
    )


__all__ = [
    "MatrixConfig",
    "MatrixModel",
    "incomplete_matrix_report",
    "load_matrix_config",
    "load_matrix_inputs",
    "matrix_check_payload",
    "matrix_scoring_config",
    "publish_matrix_terminal_failure",
]
