"""Run the constrained MuSiQue candidate-scoring diagnostic."""

from __future__ import annotations

import argparse
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

    def _render(self, prompt: str) -> str:
        rendered = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
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
            output = self._model(**_move_inputs(batch))
        log_probs = self._torch.log_softmax(output.logits, dim=-1)
        scores: list[float] = []
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
    config, tasks = load_inputs(args.config, args.snapshot, args.gate_config)
    if args.check:
        print(json.dumps(check_payload(config, config.snapshot_sha256, tasks), sort_keys=True))
        return
    _run(args.config, args.snapshot, args.gate_config, args.output)


if __name__ == "__main__":
    main()
