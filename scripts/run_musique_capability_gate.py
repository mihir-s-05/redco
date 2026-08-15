"""Run or source-check the MuSiQue capability gate.

The run path is intentionally CUDA-only and loads the pinned Qwen revision;
``--check`` exercises the authenticated snapshot without importing either
Transformers or Torch.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from redco.contracts import canonical_json
from redco.experiments.musique_capability import (
    Candidate,
    CapabilityPolicy,
    Choice,
    InferenceConfig,
    MuSiQueTask,
    evaluate_gate,
    load_gate_config,
    load_inference_config,
    load_tasks,
    parse_numbered_choice,
)
from redco.integrity import sha256_bytes


class _Tokenizer(Protocol):
    def apply_chat_template(
        self, messages: Sequence[Mapping[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str: ...

    def __call__(
        self, prompt: str, *, return_tensors: str, truncation: bool
    ) -> Mapping[str, Any]: ...

    def decode(self, tokens: Any, *, skip_special_tokens: bool) -> str: ...


class _Model(Protocol):
    def to(self, device: str) -> _Model: ...

    def eval(self) -> _Model: ...

    def generate(self, **kwargs: Any) -> Any: ...


def _prepare_cuda_model(torch_module: Any, model: _Model) -> _Model:
    cuda = torch_module.cuda
    if not bool(cuda.is_available()) or int(cuda.device_count()) != 1:
        raise RuntimeError("MuSiQue capability gate requires exactly one CUDA device")
    model.to("cuda")
    model.eval()
    return model


def _load_model(model_name: str, revision: str) -> tuple[_Tokenizer, _Model, Any]:
    transformers = cast(Any, importlib.import_module("transformers"))
    torch_module = cast(Any, importlib.import_module("torch"))
    tokenizer = cast(
        _Tokenizer,
        transformers.AutoTokenizer.from_pretrained(model_name, revision=revision),
    )
    model = cast(
        _Model,
        transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch_module.bfloat16,
        ),
    )
    return tokenizer, _prepare_cuda_model(torch_module, model), torch_module


def _evidence_text(evidence: tuple[Candidate, ...]) -> str:
    if not evidence:
        return "(none)"
    return "\n\n".join(f"{candidate.title}\n{candidate.text}" for candidate in evidence)


class TransformersPolicy(CapabilityPolicy):
    """A direct Qwen callback with strict numbered-choice parsing."""

    def __init__(
        self,
        tokenizer: _Tokenizer,
        model: _Model,
        torch_module: Any,
        *,
        choice_max_new_tokens: int,
        answer_max_new_tokens: int,
        max_input_tokens: int,
        deadline: float,
        temperature: float,
        top_p: float,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._choice_max_new_tokens = choice_max_new_tokens
        self._answer_max_new_tokens = answer_max_new_tokens
        self._max_input_tokens = max_input_tokens
        self._deadline = deadline
        self._temperature = temperature
        self._top_p = top_p
        self.call_count = 0

    def _render(self, prompt: str) -> str:
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _encode(self, prompt: str) -> Mapping[str, Any]:
        encoded = self._tokenizer(self._render(prompt), return_tensors="pt", truncation=False)
        input_ids = encoded["input_ids"]
        if int(input_ids.shape[-1]) > self._max_input_tokens:
            raise ValueError("MuSiQue prompt exceeds the frozen input-token bound")
        return encoded

    def preflight(self, tasks: Sequence[MuSiQueTask]) -> None:
        """Tokenize worst-case choice and answer prompts without generation."""
        for task in tasks:
            candidate_text = "\n\n".join(
                f"[{index}] {candidate.title}\n{candidate.text}"
                for index, candidate in enumerate(task.candidates, 1)
            )
            choice_prompt = (
                "Question:\n"
                f"{task.question}\n\n"
                "Already selected evidence:\n"
                f"{_evidence_text(task.candidates[:-1])}\n\n"
                "Choose exactly one next document from the numbered candidates. "
                "Return only one integer.\n\n"
                f"Candidates:\n{candidate_text}"
            )
            answer_prompt = (
                "Question:\n"
                f"{task.question}\n\n"
                "Answer using only the selected evidence below. Return only the short answer.\n\n"
                f"Selected evidence:\n{_evidence_text(task.candidates)}"
            )
            self._encode(choice_prompt)
            self._encode(answer_prompt)

    def _complete(
        self,
        prompt: str,
        *,
        sample: int | None,
        sampling_seed: int | None,
        max_new_tokens: int,
    ) -> str:
        if time.monotonic() >= self._deadline:
            raise TimeoutError("MuSiQue inference time bound exceeded")
        encoded = self._encode(prompt)
        if time.monotonic() >= self._deadline:
            raise TimeoutError("MuSiQue inference time bound exceeded")
        remaining = self._deadline - time.monotonic()
        moved = {
            key: value.to("cuda") if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        kwargs: dict[str, Any] = {
            **moved,
            "do_sample": sample is not None,
            "max_new_tokens": max_new_tokens,
            "max_time": remaining,
        }
        with self._torch.inference_mode():
            if sample is None:
                self.call_count += 1
                generated = self._model.generate(**kwargs)
            else:
                if sampling_seed is None:
                    raise ValueError("sampled generation requires a deterministic seed")
                kwargs["temperature"] = self._temperature
                kwargs["top_p"] = self._top_p
                device = int(self._torch.cuda.current_device())
                with self._torch.random.fork_rng(devices=[device]):
                    self._torch.manual_seed(sampling_seed)
                    self._torch.cuda.manual_seed_all(sampling_seed)
                    self.call_count += 1
                    generated = self._model.generate(**kwargs)
        if time.monotonic() >= self._deadline:
            raise TimeoutError("MuSiQue inference time bound exceeded")
        prompt_length = int(moved["input_ids"].shape[-1])
        return self._tokenizer.decode(
            generated[0][prompt_length:], skip_special_tokens=True
        ).strip()

    def choose(
        self,
        task_id: str,
        step: int,
        question: str,
        evidence: tuple[Candidate, ...],
        candidates: tuple[Candidate, ...],
        *,
        sample: int | None,
        sampling_seed: int | None,
    ) -> Choice:
        del task_id, step
        candidate_text = "\n\n".join(
            f"[{index}] {candidate.title}\n{candidate.text}"
            for index, candidate in enumerate(candidates, 1)
        )
        prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Already selected evidence:\n"
            f"{_evidence_text(evidence)}\n\n"
            "Choose exactly one next document from the numbered candidates. "
            "Return only one integer from the displayed range.\n\n"
            f"Candidates:\n{candidate_text}"
        )
        return parse_numbered_choice(
            self._complete(
                prompt,
                sample=sample,
                sampling_seed=sampling_seed,
                max_new_tokens=self._choice_max_new_tokens,
            ),
            candidates,
        )

    def answer(
        self,
        task_id: str,
        question: str,
        evidence: tuple[Candidate, ...],
        *,
        mode: Literal["greedy", "oracle"],
    ) -> str:
        del task_id, mode
        prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Answer using only the selected evidence below. Return only the short answer.\n\n"
            f"Selected evidence:\n{_evidence_text(evidence)}"
        )
        return self._complete(
            prompt,
            sample=None,
            sampling_seed=None,
            max_new_tokens=self._answer_max_new_tokens,
        )


def _inputs(
    manifest_path: Path, snapshot_path: Path
) -> tuple[dict[str, Any], Any, InferenceConfig, tuple[MuSiQueTask, ...]]:
    manifest = json.loads(manifest_path.read_bytes())
    if type(manifest) is not dict:
        raise ValueError("capability manifest must be an object")
    gate = load_gate_config(manifest_path)
    inference = load_inference_config(manifest_path)
    tasks = load_tasks(snapshot_path, gate)
    source = manifest["source"]
    if type(source) is not dict or source["snapshot_sha256"] != sha256_bytes(
        snapshot_path.read_bytes()
    ):
        raise ValueError("capability snapshot does not match the manifest")
    return cast(dict[str, Any], manifest), gate, inference, tasks


def _check(manifest_path: Path, snapshot_path: Path) -> dict[str, object]:
    _manifest, gate, inference, tasks = _inputs(manifest_path, snapshot_path)
    config_hash = sha256_bytes(manifest_path.read_bytes())
    snapshot_hash = sha256_bytes(snapshot_path.read_bytes())
    counts = {str(depth): sum(task.hop_count == depth for task in tasks) for depth in (2, 4)}
    planned_choices = sum(task.hop_count for task in tasks)
    return {
        "schema_version": 1,
        "mode": "check",
        "dataset": "MuSiQue-Ans",
        "cohort": "short_document_linear_chain",
        "config_sha256": config_hash,
        "snapshot_sha256": snapshot_hash,
        "model": inference.model,
        "revision": inference.revision,
        "tasks": len(tasks),
        "tasks_by_depth": counts,
        "candidate_count": gate.candidate_count,
        "planned_generations": planned_choices * (2 + gate.planned_k) + len(tasks) * 2,
        "gold_fields_in_policy": False,
        "model_calls": False,
        "cuda_required": True,
    }


def _timeout_result(
    gate: Any, inference: InferenceConfig, error: TimeoutError, actual_call_count: int
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": "MuSiQue-Ans",
        "cohort": "short_document_linear_chain",
        "candidate_count": gate.candidate_count,
        "planned_k": gate.planned_k,
        "passed": False,
        "partial_success": False,
        "timeout": True,
        "actual_call_count": actual_call_count,
        "blocked_reasons": [str(error)],
        "model": inference.model,
        "revision": inference.revision,
    }


def _write_new(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json(dict(payload))
    with path.open("xb") as handle:
        handle.write(data)


def _run(manifest_path: Path, snapshot_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError("capability report output already exists")
    started = time.monotonic()
    _, gate, inference, tasks = _inputs(manifest_path, snapshot_path)
    deadline = started + inference.max_seconds
    policy: TransformersPolicy | None = None
    try:
        tokenizer, model, torch_module = _load_model(inference.model, inference.revision)
        policy = TransformersPolicy(
            tokenizer,
            model,
            torch_module,
            choice_max_new_tokens=inference.choice_max_new_tokens,
            answer_max_new_tokens=inference.answer_max_new_tokens,
            max_input_tokens=inference.max_input_tokens,
            deadline=deadline,
            temperature=gate.sampling_temperature,
            top_p=gate.sampling_top_p,
        )
        policy.preflight(tasks)
        result = evaluate_gate(tasks, policy, gate)
    except TimeoutError as error:
        result = _timeout_result(gate, inference, error, policy.call_count if policy else 0)
    actual_calls = policy.call_count if policy is not None else 0
    result = {
        **result,
        "config_sha256": sha256_bytes(manifest_path.read_bytes()),
        "snapshot_sha256": sha256_bytes(snapshot_path.read_bytes()),
        "cuda_device_count": 1,
        "dtype": "bfloat16",
        "actual_call_count": actual_calls,
        "elapsed_seconds": time.monotonic() - started,
        "deadline_cooperative": True,
        "model": inference.model,
        "revision": inference.revision,
    }
    _write_new(output_path, cast(Mapping[str, object], result))
    print(json.dumps({"output": str(output_path), "passed": result["passed"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/musique-ans-capability-gate-v1.json")
    )
    parser.add_argument(
        "--snapshot", type=Path, default=Path("data/musique-ans-capability-v1.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/musique-ans-capability-gate-v1/report.json")
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if arguments.check == arguments.run:
        parser.error("choose exactly one of --check or --run")
    if arguments.check:
        print(json.dumps(_check(arguments.manifest, arguments.snapshot), sort_keys=True))
    else:
        _run(arguments.manifest, arguments.snapshot, arguments.output)


if __name__ == "__main__":
    main()
