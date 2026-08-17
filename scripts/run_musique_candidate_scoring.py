"""Run the frozen MuSiQue candidate-scoring diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from redco.experiments.musique_candidate_scoring import (
    ChoiceLaw,
    ScoreSummary,
    ScoringConfig,
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


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt", truncation=False)
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
    """Text-only conditional likelihood scorer; no generation API is used."""

    def __init__(
        self, tokenizer: Any, model: Any, torch_module: Any, config: ScoringConfig, deadline: float
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
        suffixes: list[tuple[int, ...]] = []
        for label in self._config.labels[:candidate_count]:
            full = _token_ids(self._tokenizer, rendered + label)
            suffix = full[len(prompt_ids) :] if full[: len(prompt_ids)] == prompt_ids else ()
            if not suffix:
                raise ValueError("choice label is not an exact tokenizer continuation")
            suffixes.append(suffix)
        if len(set(suffixes)) != len(suffixes):
            raise ValueError("active label continuations must be pairwise distinct")
        return rendered, prompt_ids, tuple(suffixes)

    def preflight(self, prompts: Sequence[tuple[str, int]]) -> None:
        if not prompts:
            raise ValueError("choice preflight requires at least one state")
        for prompt, count in prompts:
            self._prepare(prompt, count)
        self._law_authenticated = True

    def _sequence_scores(
        self, rendered: str, prompt_ids: tuple[int, ...], suffixes: tuple[tuple[int, ...], ...]
    ) -> tuple[float, ...]:
        labels = self._config.labels[: len(suffixes)]
        batch = self._tokenizer(
            [rendered + label for label in labels],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
            truncation=False,
        )
        rows = batch["input_ids"].tolist()
        masks = batch["attention_mask"].tolist()
        if len(rows) != len(suffixes) or len(masks) != len(suffixes):
            raise ValueError("batched label encoding has the wrong shape")
        starts: list[int] = []
        for row, mask, suffix in zip(rows, masks, suffixes, strict=True):
            start = next((index for index, value in enumerate(mask) if value == 1), -1)
            if start < 0:
                raise ValueError("batched label encoding has no active tokens")
            active = tuple(row[start : start + len(prompt_ids) + len(suffix)])
            if active != prompt_ids + suffix:
                raise ValueError(
                    "batched label encoding differs from the authenticated continuation"
                )
            starts.append(start)
        with self._torch.inference_mode():
            self.forward_calls += 1
            output = self._model(**_move_inputs(batch), use_cache=False)
        if getattr(output, "past_key_values", None) is not None:
            raise ValueError("choice model returned a cache despite use_cache=False")
        shape = getattr(output.logits, "shape", None)
        width = len(rows[0]) if rows else 0
        if (
            shape is None
            or len(shape) != 3
            or tuple(int(value) for value in shape[:2]) != (len(rows), width)
            or int(shape[2]) <= 0
        ):
            raise ValueError("choice model returned logits with the wrong shape")
        if any(token < 0 or token >= int(shape[2]) for suffix in suffixes for token in suffix):
            raise ValueError("choice continuation token exceeds the model vocabulary")
        log_probs = self._torch.log_softmax(output.logits, dim=-1)
        scores: list[float] = []
        self.suffix_lengths.extend(len(suffix) for suffix in suffixes)
        for row, suffix in enumerate(suffixes):
            total = 0.0
            for offset, token in enumerate(suffix):
                position = starts[row] + len(prompt_ids) - 1 + offset
                total += float(log_probs[row, position, token].item())
            scores.append(total)
        return tuple(scores)

    def score(self, prompt: str, gold_index: int, candidate_count: int) -> ScoreSummary:
        if time.monotonic() >= self._deadline or not self._law_authenticated:
            raise TimeoutError("choice scoring is outside its authenticated window")
        rendered, prompt_ids, suffixes = self._prepare(prompt, candidate_count)
        self.rendered_prompt_sha256.append(hashlib.sha256(rendered.encode("utf-8")).hexdigest())
        scores = self._sequence_scores(rendered, prompt_ids, suffixes)
        return score_values(scores, gold_index, self.law)


class Qwen35TextTokenizer:
    """Small text-only adapter for Qwen3.5 chat-template calls."""

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
            raise ValueError("Qwen3.5 scoring freezes enable_thinking=False")
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
        if (
            not isinstance(encoded, Mapping)
            or set(encoded) - {"input_ids", "attention_mask", "token_type_ids"}
            or "input_ids" not in encoded
            or "attention_mask" not in encoded
        ):
            raise ValueError("Qwen3.5 scoring received non-text tokenizer inputs")
        return encoded

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer


def _load_model(config: ScoringConfig) -> tuple[Any, Any, Any]:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    if not torch.cuda.is_available() or torch.cuda.device_count() != config.cuda_devices:
        raise RuntimeError("candidate scoring requires exactly one CUDA device")
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, revision=config.revision)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model,
        revision=config.revision,
        torch_dtype=torch.bfloat16,
        use_cache=config.use_cache,
    )
    return tokenizer, model.to("cuda").eval(), torch


def _prompts(tasks: Sequence[MuSiQueTask], config: ScoringConfig) -> list[tuple[str, int]]:
    prompts: list[tuple[str, int]] = []
    for task in tasks:
        for step in range(task.hop_count):
            selected = tuple(task.support_path[:step])
            evidence = tuple(
                next(item for item in task.candidates if item.title == title) for title in selected
            )
            candidates = tuple(item for item in task.candidates if item.title not in selected)
            prompts.append((prompt_for_state(config, task, evidence, candidates), len(candidates)))
    return prompts


def _run(config_path: Path, snapshot_path: Path, gate_path: Path, output_path: Path) -> None:
    started = time.monotonic()
    config, tasks = load_inputs(config_path, snapshot_path, gate_path)
    tokenizer, model, torch = _load_model(config)
    scorer = TorchChoiceScorer(tokenizer, model, torch, config, started + config.max_seconds)
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
    parser.add_argument(
        "--output", type=Path, default=Path("runs/musique-ans-candidate-scoring-v1/report.json")
    )
    args = parser.parse_args()
    if args.check:
        config, tasks = load_inputs(args.config, args.snapshot, args.gate_config)
        print(json.dumps(check_payload(config, config.snapshot_sha256, tasks), sort_keys=True))
        return
    _run(args.config, args.snapshot, args.gate_config, args.output)


if __name__ == "__main__":
    main()
