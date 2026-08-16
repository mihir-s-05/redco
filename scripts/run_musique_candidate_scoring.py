"""Run the constrained MuSiQue candidate-scoring diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from redco.experiments.musique_candidate_scoring import (
    ChoiceLaw,
    ScoreSummary,
    ScoringConfig,
    StateRecord,
    check_payload,
    evaluate_states,
    load_inputs,
    prompt_for_state,
    report_payload,
    score_values,
    summarize,
    write_exclusive,
)
from redco.experiments.musique_capability import MuSiQueTask
from redco.experiments.musique_matrix import (
    MatrixConfig,
    MatrixModel,
    load_matrix_inputs,
    matrix_check_payload,
    matrix_scoring_config,
    publish_matrix_terminal_failure,
)


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
    )
    values = encoded["input_ids"][0].tolist()
    if type(values) is not list or not all(type(item) is int for item in values):
        raise ValueError("tokenizer returned an invalid integer sequence")
    return tuple(values)


def _move_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"input_ids", "attention_mask", "token_type_ids"}
    if set(inputs) - allowed or "input_ids" not in inputs or "attention_mask" not in inputs:
        raise ValueError("choice model received non-text or incomplete tokenizer inputs")
    return {
        key: value.to("cuda") if hasattr(value, "to") else value for key, value in inputs.items()
    }


class TorchChoiceScorer:
    """Qwen choice-position scorer; no generation API is used."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        torch_module: Any,
        config: ScoringConfig,
        deadline: float,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._config = config
        self._deadline = deadline
        self.law: ChoiceLaw = "conditional_log_likelihood"
        self._law_authenticated = False
        self.forward_calls = 0
        self.suffix_lengths: list[int] = []
        self.rendered_prompt_sha256: list[str] = []

    def _render(self, prompt: str) -> str:
        rendered = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if type(rendered) is not str:
            raise ValueError("chat template must return text")
        return rendered

    def _prepare(
        self, prompt: str, candidate_count: int
    ) -> tuple[str, tuple[int, ...], tuple[tuple[int, ...], ...]]:
        if type(candidate_count) is not int or not 1 <= candidate_count <= len(self._config.labels):
            raise ValueError("choice state has an invalid candidate count")
        rendered = self._render(prompt)
        prompt_ids = _token_ids(self._tokenizer, rendered)
        if not prompt_ids or len(prompt_ids) > self._config.max_input_tokens:
            raise ValueError("choice prompt exceeds the frozen input-token bound")
        label_ids: list[tuple[int, ...]] = []
        for label in self._config.labels[:candidate_count]:
            full = _token_ids(self._tokenizer, rendered + label)
            suffix = full[len(prompt_ids) :] if full[: len(prompt_ids)] == prompt_ids else ()
            if not suffix:
                raise ValueError(
                    "choice label is not a continuation in the exact tokenizer context"
                )
            label_ids.append(suffix)
        if len(set(label_ids)) != len(label_ids):
            raise ValueError("active label continuations must be pairwise distinct")
        return rendered, prompt_ids, tuple(label_ids)

    def preflight(self, prompts: Sequence[tuple[str, int]]) -> None:
        if not prompts:
            raise ValueError("choice preflight requires at least one state")
        for prompt, candidate_count in prompts:
            self._prepare(prompt, candidate_count)
        self._law_authenticated = True

    def _sequence_scores(
        self,
        rendered: str,
        prompt_ids: tuple[int, ...],
        labels: tuple[tuple[int, ...], ...],
    ) -> tuple[float, ...]:
        active_labels = self._config.labels[: len(labels)]
        batch = self._tokenizer(
            [rendered + label for label in active_labels],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
            truncation=False,
        )
        rows = batch["input_ids"].tolist()
        masks = batch["attention_mask"].tolist()
        expected_count = len(labels)
        if (
            type(rows) is not list
            or type(masks) is not list
            or len(rows) != expected_count
            or len(masks) != expected_count
        ):
            raise ValueError("batched label encoding has the wrong shape")
        starts: list[int] = []
        for row, mask, suffix in zip(rows, masks, labels, strict=True):
            active_start = next((index for index, value in enumerate(mask) if value == 1), -1)
            if active_start < 0:
                raise ValueError("batched label encoding has no active tokens")
            active = tuple(row[active_start : active_start + len(prompt_ids) + len(suffix)])
            if active != prompt_ids + suffix:
                raise ValueError("batched label encoding differs from authenticated continuation")
            starts.append(active_start)
        with self._torch.inference_mode():
            self.forward_calls += 1
            output = self._model(**_move_inputs(batch), use_cache=False)
        if getattr(output, "past_key_values", None) is not None:
            raise ValueError("choice model returned a cache despite use_cache=False")
        shape = getattr(output.logits, "shape", None)
        expected_width = len(rows[0]) if rows else 0
        if (
            shape is None
            or len(shape) != 3
            or tuple(int(value) for value in shape[:2]) != (expected_count, expected_width)
            or int(shape[2]) <= 0
        ):
            raise ValueError("choice model returned logits with the wrong batch/sequence shape")
        if any(token < 0 or token >= int(shape[2]) for suffix in labels for token in suffix):
            raise ValueError("choice continuation token exceeds the model vocabulary")
        log_probs = self._torch.log_softmax(output.logits, dim=-1)
        scores: list[float] = []
        self.suffix_lengths.extend(len(suffix) for suffix in labels)
        for row, suffix in enumerate(labels):
            total = 0.0
            for offset, token in enumerate(suffix):
                position = starts[row] + len(prompt_ids) - 1 + offset
                total += float(log_probs[row, position, token].item())
            scores.append(total)
        return tuple(scores)

    def score(self, prompt: str, gold_index: int, candidate_count: int) -> ScoreSummary:
        if time.monotonic() >= self._deadline:
            raise TimeoutError("MuSiQue scoring time bound exceeded")
        if not self._law_authenticated:
            raise ValueError("choice scoring requires the complete preflight law")
        rendered, prompt_ids, labels = self._prepare(prompt, candidate_count)
        self.rendered_prompt_sha256.append(hashlib.sha256(rendered.encode("utf-8")).hexdigest())
        scores = self._sequence_scores(rendered, prompt_ids, labels)
        if time.monotonic() >= self._deadline:
            raise TimeoutError("MuSiQue scoring time bound exceeded")
        return score_values(scores, gold_index, self.law)


def _load_model(config: ScoringConfig) -> tuple[Any, Any, Any]:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != config.cuda_devices:
        raise RuntimeError("the candidate diagnostic requires exactly one CUDA device")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        revision=config.revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        revision=config.revision,
        torch_dtype=torch.bfloat16,
        use_cache=config.use_cache,
    )
    model = model.to("cuda")
    model.eval()
    return tokenizer, model, torch


class Qwen35TextTokenizer:
    """Text-only view of the pinned Qwen3.5 tokenizer/chat template."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ) -> str:
        if enable_thinking is not False:
            raise ValueError("Qwen3.5 choice scoring freezes enable_thinking=False")
        rendered = self._tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
        if type(rendered) is not str:
            raise ValueError("Qwen3.5 chat template must return text")
        return rendered

    def __call__(self, value: str | list[str], **kwargs: Any) -> Mapping[str, Any]:
        encoded = self._tokenizer(value, **kwargs)
        if not isinstance(encoded, Mapping):
            raise ValueError("Qwen3.5 tokenizer returned a non-mapping")
        allowed = {"input_ids", "attention_mask", "token_type_ids"}
        if set(encoded) - allowed or "input_ids" not in encoded or "attention_mask" not in encoded:
            raise ValueError("Qwen3.5 text-only scoring received multimodal inputs")
        return encoded

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _repository_architecture_matches(value: Any, expected: str) -> bool:
    return value in ([expected], (expected,))


def _loaded_architecture_matches(value: Any, expected: str) -> bool:
    return value is None or value in ([expected], (expected,))


def _authenticate_qwen35_repository_config(config: Any, spec: MatrixModel) -> None:
    text_config = _field(config, "text_config")
    architectures = _field(config, "architectures")
    if (
        _field(config, "model_type") != spec.model_type
        or not _repository_architecture_matches(architectures, spec.repository_architecture)
        or _field(text_config, "model_type") != spec.text_model_type
        or _field(text_config, "num_hidden_layers") != spec.layers
        or _field(text_config, "hidden_size") != spec.hidden_size
    ):
        raise RuntimeError("Qwen3.5 repository config does not match the frozen binding")


def _authenticate_qwen35_model(model: Any, spec: MatrixModel) -> None:
    if type(model).__name__ != spec.architecture:
        raise RuntimeError("Qwen3.5 resolved model class is not the frozen text-only class")
    model_config = getattr(model, "config", None)
    architectures = _field(model_config, "architectures")
    if (
        _field(model_config, "model_type") != spec.text_model_type
        or not _loaded_architecture_matches(architectures, spec.architecture)
        or _field(model_config, "num_hidden_layers") != spec.layers
        or _field(model_config, "hidden_size") != spec.hidden_size
    ):
        raise RuntimeError("Qwen3.5 loaded text config does not match the frozen binding")


def _load_qwen35_model(config: MatrixConfig, spec: MatrixModel) -> tuple[Any, Any, Any]:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    if getattr(transformers, "__version__", None) != config.transformers_version:
        raise RuntimeError("Qwen3.5 requires the frozen Transformers version")
    if not torch.cuda.is_available() or torch.cuda.device_count() != config.cuda_devices:
        raise RuntimeError("the Qwen3.5 matrix requires exactly one CUDA device")
    AutoTokenizer = getattr(transformers, "AutoTokenizer", None)
    AutoConfig = getattr(transformers, "AutoConfig", None)
    AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM", None)
    if AutoConfig is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("the pinned Transformers text-only owners are unavailable")
    repository_config = AutoConfig.from_pretrained(
        spec.name,
        revision=spec.revision,
        trust_remote_code=False,
    )
    _authenticate_qwen35_repository_config(repository_config, spec)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.name,
        revision=spec.revision,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec.name,
        revision=spec.revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    _authenticate_qwen35_model(model, spec)
    model.config.use_cache = False
    if model.config.use_cache is not False:
        raise RuntimeError("Qwen3.5 loaded model did not disable the cache")
    model = model.to("cuda")
    model.eval()
    return Qwen35TextTokenizer(tokenizer), model, torch


def _prompts(tasks: Sequence[MuSiQueTask], config: ScoringConfig) -> list[tuple[str, int]]:
    prompts: list[tuple[str, int]] = []
    for task in tasks:
        for step in range(task.hop_count):
            selected = list(task.support_path[:step])
            candidates = tuple(item for item in task.candidates if item.title not in selected)
            evidence = tuple(
                next(item for item in task.candidates if item.title == title) for title in selected
            )
            prompts.append((prompt_for_state(config, task, evidence, candidates), len(candidates)))
    return prompts


def _state_identity(
    tasks: Sequence[MuSiQueTask], records: Sequence[StateRecord]
) -> list[dict[str, object]]:
    task_by_id = {task.task_id: task for task in tasks}
    greedy_evidence: dict[str, list[str]] = {}
    identity: list[dict[str, object]] = []
    for record in records:
        task = task_by_id.get(record.task_id)
        if task is None:
            raise ValueError("matrix state is not bound to a frozen task")
        if record.mode == "greedy":
            evidence = tuple(greedy_evidence.get(record.task_id, ()))
        else:
            evidence = tuple(task.support_path[: record.step - 1])
        evidence_set = set(evidence)
        candidates = tuple(
            candidate.title for candidate in task.candidates if candidate.title not in evidence_set
        )
        identity.append(
            {
                "task_id": record.task_id,
                "hop_count": record.hop_count,
                "mode": record.mode,
                "step": record.step,
                "candidate_titles": list(candidates),
                "selected_evidence_titles": list(evidence),
            }
        )
        if record.mode == "greedy":
            greedy_evidence.setdefault(record.task_id, []).append(record.selected_candidate)
    return identity


def _adapter_metadata(
    tokenizer: Qwen35TextTokenizer, scorer: TorchChoiceScorer
) -> dict[str, object]:
    base = tokenizer.tokenizer
    chat_template = getattr(base, "chat_template", None)
    if type(chat_template) is not str:
        raise RuntimeError("Qwen3.5 tokenizer has no exact text chat template")
    return {
        "processor_class": None,
        "tokenizer_class": type(base).__name__,
        "chat_template": {"enable_thinking": False, "add_generation_prompt": True},
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
        "padding_side": getattr(base, "padding_side", None),
        "pad_token_id": getattr(base, "pad_token_id", None),
        "active_suffix_length_distribution": {
            str(length): count for length, count in sorted(Counter(scorer.suffix_lengths).items())
        },
    }


def _reset_peak_memory(torch_module: Any) -> None:
    cuda = torch_module.cuda
    reset = getattr(cuda, "reset_peak_memory_stats", None)
    if not callable(reset):
        raise RuntimeError("CUDA peak-memory owners are required for the matrix report")
    reset()


def _peak_memory(torch_module: Any) -> int:
    peak = getattr(torch_module.cuda, "max_memory_allocated", None)
    if not callable(peak):
        raise RuntimeError("CUDA peak-memory owners are required for the matrix report")
    value = peak()
    if type(value) is not int:
        raise RuntimeError("CUDA peak-memory owner returned a non-integer")
    return value


def _run_matrix_model(
    config: MatrixConfig,
    tasks: Sequence[MuSiQueTask],
    spec: MatrixModel,
    output_path: Path,
    deadline: float,
    worker_index: int,
) -> None:
    started = time.monotonic()
    tokenizer, model, torch_module = _load_qwen35_model(config, spec)
    scorer = TorchChoiceScorer(
        tokenizer,
        model,
        torch_module,
        matrix_scoring_config(config, spec),
        deadline,
    )
    try:
        _reset_peak_memory(torch_module)
        scorer.preflight(_prompts(tasks, matrix_scoring_config(config, spec)))
        records = evaluate_states(tasks, matrix_scoring_config(config, spec), scorer)
        by_hop, blocked = summarize(tasks, matrix_scoring_config(config, spec), records)
        peak_after = _peak_memory(torch_module)
        payload = report_payload(
            matrix_scoring_config(config, spec),
            config.snapshot_sha256,
            records,
            by_hop,
            blocked,
            elapsed_seconds=time.monotonic() - started,
            forward_calls=scorer.forward_calls,
            law=scorer.law,
        )
        payload["model_binding"] = spec.as_mapping()
        payload["runtime"] = {
            "worker_index": worker_index,
            "peak_memory_bytes": peak_after,
            "separate_process": True,
            "unload_between_models": True,
        }
        payload["adapter"] = _adapter_metadata(tokenizer, scorer)
        payload["state_identity"] = _state_identity(tasks, records)
        payload["rendered_prompt_digest_algorithm"] = "sha256_utf8"
        payload["rendered_prompt_sha256"] = list(scorer.rendered_prompt_sha256)
    finally:
        del scorer
        del model
        del tokenizer
        if config.empty_cuda_cache:
            empty_cache = getattr(torch_module.cuda, "empty_cache", None)
            if not callable(empty_cache):
                raise RuntimeError("CUDA cache owner is required between matrix workers")
            empty_cache()
    payload["unload_confirmation"] = {"model_deleted": True, "cuda_cache_emptied": True}
    write_exclusive(output_path, payload)
    print(json.dumps({"model": spec.name, "output": str(output_path)}, sort_keys=True))


def _validate_state_identity(payload: Mapping[str, Any], expected_states: int) -> None:
    identities = payload.get("state_identity")
    prompt_digests = payload.get("rendered_prompt_sha256")
    if (
        payload.get("rendered_prompt_digest_algorithm") != "sha256_utf8"
        or not isinstance(identities, list)
        or len(identities) != expected_states
        or not isinstance(prompt_digests, list)
        or len(prompt_digests) != expected_states
    ):
        raise ValueError("matrix worker report is missing the exact state/prompt topology")
    for identity in identities:
        if not isinstance(identity, Mapping) or set(identity) != {
            "candidate_titles",
            "hop_count",
            "mode",
            "selected_evidence_titles",
            "step",
            "task_id",
        }:
            raise ValueError("matrix state identity has the wrong schema")
        if (
            type(identity["task_id"]) is not str
            or identity["mode"] not in ("greedy", "teacher_forced")
            or type(identity["hop_count"]) is not int
            or type(identity["step"]) is not int
            or identity["hop_count"] not in (2, 4)
            or identity["step"] < 1
            or not isinstance(identity["candidate_titles"], list)
            or not isinstance(identity["selected_evidence_titles"], list)
            or not all(type(item) is str for item in identity["candidate_titles"])
            or not all(type(item) is str for item in identity["selected_evidence_titles"])
        ):
            raise ValueError("matrix state identity has invalid fields")
    for digest in prompt_digests:
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("matrix rendered prompt digest is invalid")


def _validate_matrix_model_report(
    payload: Mapping[str, Any], config: MatrixConfig, spec: MatrixModel
) -> None:
    if (
        payload.get("config_sha256") != config.config_sha256
        or payload.get("snapshot_sha256") != config.snapshot_sha256
        or payload.get("model_binding") != spec.as_mapping()
        or payload.get("choice_law") != "conditional_log_likelihood"
        or payload.get("unload_confirmation") != {"model_deleted": True, "cuda_cache_emptied": True}
    ):
        raise ValueError("matrix worker report is not bound to the frozen model")
    generation = payload.get("generation")
    if (
        not isinstance(generation, Mapping)
        or generation.get("expected_forward_calls") != config.expected_states_per_model
        or generation.get("actual_forward_calls") != config.expected_states_per_model
        or generation.get("answer_calls") != 0
        or generation.get("sampled_calls") != 0
        or generation.get("training_updates") != 0
    ):
        raise ValueError("matrix worker report has the wrong call topology")
    if (
        type(payload.get("capability_pass")) is not bool
        or type(payload.get("training_support_pass")) is not bool
    ):
        raise ValueError("matrix worker report is missing independent gate results")
    adapter = payload.get("adapter")
    if not isinstance(adapter, Mapping):
        raise ValueError("matrix worker report is missing adapter metadata")
    template_digest = adapter.get("chat_template_sha256")
    if (
        type(template_digest) is not str
        or len(template_digest) != 64
        or any(character not in "0123456789abcdef" for character in template_digest)
    ):
        raise ValueError("matrix worker report has an invalid chat-template digest")
    _validate_state_identity(payload, config.expected_states_per_model)


def _comparability_projection(
    report: Mapping[str, Any], expected_states: int
) -> tuple[list[tuple[object, ...]], list[Mapping[str, Any]], list[str]]:
    identities = report["state_identity"]
    prompt_digests = report["rendered_prompt_sha256"]
    if not isinstance(identities, list) or not isinstance(prompt_digests, list):
        raise ValueError("matrix worker report has no state/prompt topology")
    structural: list[tuple[object, ...]] = []
    teacher_identities: list[Mapping[str, Any]] = []
    teacher_prompts: list[str] = []
    for identity, prompt_digest in zip(identities, prompt_digests, strict=True):
        if not isinstance(identity, Mapping) or type(prompt_digest) is not str:
            raise ValueError("matrix worker report has invalid state/prompt topology")
        candidate_titles = identity["candidate_titles"]
        selected_titles = identity["selected_evidence_titles"]
        if not isinstance(candidate_titles, list) or not isinstance(selected_titles, list):
            raise ValueError("matrix worker report has invalid state topology")
        structural.append(
            (
                identity["task_id"],
                identity["hop_count"],
                identity["mode"],
                identity["step"],
                len(candidate_titles),
                len(selected_titles),
            )
        )
        if identity["mode"] == "teacher_forced":
            teacher_identities.append(identity)
            teacher_prompts.append(prompt_digest)
    if len(structural) != expected_states or len(teacher_identities) * 2 != expected_states:
        raise ValueError("matrix worker report has the wrong fixed/dynamic state split")
    return structural, teacher_identities, teacher_prompts


def _task_deltas(reports: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    if len(reports) != 2:
        raise ValueError("paired matrix deltas require both model reports")
    states_by_model: list[dict[str, list[Mapping[str, Any]]]] = []
    for report in reports:
        states = report.get("states")
        if not isinstance(states, list):
            raise ValueError("matrix report has no state records")
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for state in states:
            if not isinstance(state, Mapping):
                raise ValueError("matrix state is not an object")
            task_id = state.get("task_id")
            if type(task_id) is not str:
                raise ValueError("matrix state has no task ID")
            grouped.setdefault(task_id, []).append(state)
        states_by_model.append(grouped)
    first_ids = set(states_by_model[0])
    if first_ids != set(states_by_model[1]):
        raise ValueError("matrix models scored different task IDs")
    rows: list[dict[str, object]] = []
    for task_id in sorted(first_ids):
        metrics: list[tuple[float, float, bool, int]] = []
        for grouped in states_by_model:
            states = grouped[task_id]
            teacher = [state for state in states if state.get("mode") == "teacher_forced"]
            greedy = [state for state in states if state.get("mode") == "greedy"]
            if not teacher or not greedy:
                raise ValueError("matrix task is missing teacher or greedy states")
            teacher_probability = sum(float(state["gold_probability"]) for state in teacher) / len(
                teacher
            )
            greedy_probability = math.prod(float(state["selected_probability"]) for state in greedy)
            ordered = sorted(greedy, key=lambda item: int(item["step"]))
            path = tuple(str(state["selected_candidate"]) for state in ordered)
            gold = tuple(str(state["gold_candidate"]) for state in ordered)
            metrics.append((teacher_probability, greedy_probability, path == gold, len(greedy)))
        rows.append(
            {
                "task_id": task_id,
                "hop_count": metrics[0][3],
                "teacher_gold_probability_mean_delta_9b_minus_4b": (metrics[1][0] - metrics[0][0]),
                "greedy_selected_probability_product_delta_9b_minus_4b": (
                    metrics[1][1] - metrics[0][1]
                ),
                "greedy_ordered_full_path_4b": metrics[0][2],
                "greedy_ordered_full_path_9b": metrics[1][2],
            }
        )
    return rows


def _matrix_report(
    config: MatrixConfig, reports: Sequence[Mapping[str, Any]], elapsed_seconds: float
) -> dict[str, object]:
    if len(reports) != len(config.models):
        raise ValueError("matrix requires both model reports")
    for report, spec in zip(reports, config.models, strict=True):
        _validate_matrix_model_report(report, config, spec)
    topology, teacher_identity, teacher_prompts = _comparability_projection(
        reports[0], config.expected_states_per_model
    )
    first_adapter = reports[0]["adapter"]
    if not isinstance(first_adapter, Mapping):
        raise ValueError("matrix worker report is missing adapter metadata")
    template_digest = first_adapter["chat_template_sha256"]
    for report in reports[1:]:
        other_topology, other_teacher_identity, other_teacher_prompts = (
            _comparability_projection(report, config.expected_states_per_model)
        )
        adapter = report["adapter"]
        if (
            not isinstance(adapter, Mapping)
            or adapter.get("chat_template_sha256") != template_digest
            or other_topology != topology
            or other_teacher_identity != teacher_identity
            or other_teacher_prompts != teacher_prompts
        ):
            raise ValueError("matrix workers differ on fixed prompt comparability")
    eligible: list[str] = []
    for report in reports:
        if report["training_experiment_eligible"] is True:
            binding = report["model_binding"]
            if not isinstance(binding, Mapping) or type(binding.get("name")) is not str:
                raise ValueError("matrix report has an invalid model binding")
            eligible.append(binding["name"])
    actual_calls = 0
    for report in reports:
        generation = report["generation"]
        if (
            not isinstance(generation, Mapping)
            or type(generation.get("actual_forward_calls")) is not int
        ):
            raise ValueError("matrix report has an invalid generation summary")
        actual_calls += generation["actual_forward_calls"]
    if actual_calls != config.expected_forward_calls:
        raise ValueError("matrix forward-call total is not exactly 528")
    return {
        "schema_version": config.schema_version,
        "dataset": "MuSiQue-Ans",
        "cohort": "short_document_linear_chain",
        "mode": "qwen3_5_constrained_scoring_matrix",
        "config_sha256": config.config_sha256,
        "snapshot_sha256": config.snapshot_sha256,
        "models": [report["model_binding"] for report in reports],
        "matrix_complete": True,
        "model_results": [dict(report) for report in reports],
        "per_model_gate": [
            {
                "model": report["model_binding"]["name"],
                "capability_pass": report["capability_pass"],
                "capability_reasons": report["capability_blocked_reasons"],
                "training_support_pass": report["training_support_pass"],
                "training_support_reasons": report["training_support_blocked_reasons"],
                "training_experiment_eligible": report["training_experiment_eligible"],
            }
            for report in reports
        ],
        "eligible_models": eligible,
        "paired_task_deltas_9b_minus_4b": _task_deltas(reports),
        "comparability": {
            "structural_state_topology": [list(item) for item in topology],
            "teacher_forced_state_identity": teacher_identity,
            "rendered_prompt_digest_algorithm": "sha256_utf8",
            "teacher_forced_rendered_prompt_sha256": teacher_prompts,
            "chat_template_sha256": template_digest,
            "greedy_prompt_relationship": "model_dependent_trajectory",
        },
        "generation": {
            "expected_forward_calls": config.expected_forward_calls,
            "actual_forward_calls": actual_calls,
            "model_count": 2,
            "sampling_calls": 0,
            "training_updates": 0,
        },
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


def _run(
    config_path: Path,
    snapshot_path: Path,
    gate_config_path: Path,
    output_path: Path,
) -> None:
    started = time.monotonic()
    config, tasks = load_inputs(config_path, snapshot_path, gate_config_path)
    tokenizer, model, torch_module = _load_model(config)
    scorer = TorchChoiceScorer(
        tokenizer,
        model,
        torch_module,
        config,
        started + config.max_seconds,
    )
    scorer.preflight(_prompts(tasks, config))
    records = evaluate_states(tasks, config, scorer)
    by_hop, blocked = summarize(tasks, config, records)
    payload = report_payload(
        config,
        config.snapshot_sha256,
        records,
        by_hop,
        blocked,
        elapsed_seconds=time.monotonic() - started,
        forward_calls=scorer.forward_calls,
        law=scorer.law,
    )
    write_exclusive(output_path, payload)
    print(json.dumps({"output": str(output_path), "passed": payload["passed"]}, sort_keys=True))


def _run_matrix(
    config_path: Path,
    snapshot_path: Path,
    gate_config_path: Path,
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    config, _tasks = load_matrix_inputs(config_path, snapshot_path, gate_config_path)
    started = time.monotonic()
    minimum = config.minimum_seconds_per_model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if config.max_seconds < minimum * len(config.models):
        publish_matrix_terminal_failure(
            output_path,
            config,
            (),
            -1,
            "InsufficientInitialWindow",
            time.monotonic() - started,
            False,
        )
        raise RuntimeError("matrix window is not credible for both model processes")
    reports: list[Mapping[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="musique-matrix-", dir=output_path.parent) as temp_dir:
        for index, _spec in enumerate(config.models):
            remaining_models = len(config.models) - index
            remaining = config.max_seconds - (time.monotonic() - started)
            if remaining < minimum * remaining_models:
                publish_matrix_terminal_failure(
                    output_path,
                    config,
                    reports,
                    index,
                    "InsufficientRemainingWindow",
                    time.monotonic() - started,
                    False,
                )
                raise TimeoutError("matrix window cannot complete both model processes")
            worker_budget = remaining - minimum * (remaining_models - 1)
            worker_output = Path(temp_dir) / f"model-{index}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run",
                "--matrix-worker",
                str(index),
                "--worker-budget",
                str(worker_budget),
                "--config",
                str(config_path),
                "--snapshot",
                str(snapshot_path),
                "--gate-config",
                str(gate_config_path),
                "--output",
                str(worker_output),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    timeout=max(1.0, worker_budget),
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                publish_matrix_terminal_failure(
                    output_path,
                    config,
                    reports,
                    index,
                    "WorkerTimeout",
                    time.monotonic() - started,
                    True,
                )
                raise RuntimeError("Qwen3.5 matrix worker timed out") from error
            except OSError as error:
                publish_matrix_terminal_failure(
                    output_path,
                    config,
                    reports,
                    index,
                    "WorkerLaunchError",
                    time.monotonic() - started,
                    True,
                )
                raise RuntimeError("Qwen3.5 matrix worker could not start") from error
            if completed.returncode != 0:
                publish_matrix_terminal_failure(
                    output_path,
                    config,
                    reports,
                    index,
                    "WorkerProcessError",
                    time.monotonic() - started,
                    True,
                )
                raise RuntimeError("Qwen3.5 matrix worker failed technically")
            if not worker_output.is_file():
                publish_matrix_terminal_failure(
                    output_path,
                    config,
                    reports,
                    index,
                    "MissingWorkerReport",
                    time.monotonic() - started,
                    True,
                )
                raise RuntimeError("Qwen3.5 matrix worker produced no report")
            try:
                report = json.loads(worker_output.read_bytes())
                if not isinstance(report, Mapping):
                    raise ValueError("worker report is not an object")
                _validate_matrix_model_report(report, config, config.models[index])
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                publish_matrix_terminal_failure(
                    output_path,
                    config,
                    reports,
                    index,
                    "InvalidWorkerReport",
                    time.monotonic() - started,
                    True,
                )
                raise RuntimeError("Qwen3.5 matrix worker report failed validation") from error
            reports.append(report)
    try:
        payload = _matrix_report(config, reports, time.monotonic() - started)
    except Exception as error:
        publish_matrix_terminal_failure(
            output_path,
            config,
            reports,
            -1,
            "MatrixAggregationError",
            time.monotonic() - started,
            None,
        )
        raise RuntimeError("Qwen3.5 matrix aggregation failed") from error
    write_exclusive(output_path, payload)
    print(json.dumps({"output": str(output_path), "matrix_complete": True}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/musique-ans-candidate-scoring-v1.json")
    )
    parser.add_argument(
        "--snapshot", type=Path, default=Path("data/musique-ans-capability-v1.json")
    )
    parser.add_argument(
        "--gate-config", type=Path, default=Path("configs/musique-ans-capability-gate-v1.json")
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--matrix-worker", type=int, choices=(0, 1), default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-budget", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    raw_config = json.loads(args.config.read_bytes())
    is_matrix = isinstance(raw_config, dict) and raw_config.get("mode") == "qwen3_5_matrix"
    output = args.output or Path(
        "runs/musique-ans-candidate-scoring-matrix-v1/report.json"
        if is_matrix
        else "runs/musique-ans-candidate-scoring-v1/report.json"
    )
    if args.matrix_worker is not None:
        if not is_matrix or args.worker_budget is None:
            raise ValueError("matrix worker requires the matrix config and a bounded budget")
        matrix_config, matrix_tasks = load_matrix_inputs(
            args.config, args.snapshot, args.gate_config
        )
        _run_matrix_model(
            matrix_config,
            matrix_tasks,
            matrix_config.models[args.matrix_worker],
            output,
            time.monotonic() + args.worker_budget,
            args.matrix_worker,
        )
        return
    if args.check:
        if is_matrix:
            matrix_config, matrix_tasks = load_matrix_inputs(
                args.config, args.snapshot, args.gate_config
            )
            print(
                json.dumps(
                    matrix_check_payload(
                        matrix_config, matrix_config.snapshot_sha256, matrix_tasks
                    ),
                    sort_keys=True,
                )
            )
        else:
            config, tasks = load_inputs(args.config, args.snapshot, args.gate_config)
            print(json.dumps(check_payload(config, config.snapshot_sha256, tasks), sort_keys=True))
        return
    if is_matrix:
        _run_matrix(args.config, args.snapshot, args.gate_config, output)
    else:
        config, tasks = load_inputs(args.config, args.snapshot, args.gate_config)
        _run(args.config, args.snapshot, args.gate_config, output)


if __name__ == "__main__":
    main()
