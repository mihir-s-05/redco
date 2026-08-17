"""Build, check, or run the compact MuSiQue support-path warm-start."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from run_musique_candidate_scoring import (
    Qwen35TextTokenizer,
    TorchChoiceScorer,
    _prompts,
    _token_ids,
)

from redco.experiments.musique_candidate_scoring import ScoringConfig, evaluate_states, summarize
from redco.experiments.musique_capability import load_tasks
from redco.experiments.musique_support_warm_start import (
    EVAL_GATE_PATH,
    EVAL_GATE_SHA256,
    EVAL_SNAPSHOT_PATH,
    EVAL_SNAPSHOT_SHA256,
    MODEL_NAME,
    MODEL_REVISION,
    WarmTask,
    build_snapshot,
    check_payload,
    config_payload,
    load_config,
    load_snapshot,
    state_order_digest,
    training_states,
    warm_prompt,
    write_exclusive,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/musique-ans-support-warm-start-v1.json"
DEFAULT_SNAPSHOT = ROOT / "data/musique-ans-support-warm-start-v1.json"
DEFAULT_OUTPUT = ROOT / "runs/musique-ans-support-warm-start-v1/report.json"
DEFAULT_ADAPTER_OUTPUT = ROOT / "runs/musique-ans-support-warm-start-v1/adapter.safetensors"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_map(config: Mapping[str, object]) -> Mapping[str, object]:
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("warm-start config has no source binding")
    return source


def _int(value: object, message: str) -> int:
    if type(value) is not int:
        raise ValueError(message)
    return value


def _number(value: object, message: str) -> float:
    if type(value) is int:
        number = float(value)
    elif type(value) is float:
        number = value
    else:
        raise ValueError(message)
    if not math.isfinite(number):
        raise ValueError(message)
    return number


def _scoring_config(config: Mapping[str, object], config_sha256: str) -> ScoringConfig:
    source = _source_map(config)
    model = cast(Mapping[str, object], config["model"])
    choice = cast(Mapping[str, object], config["choice"])
    post_gate = cast(Mapping[str, object], cast(Mapping[str, object], config["gate"])["post"])
    return ScoringConfig(
        1,
        config_sha256,
        str(source["eval_snapshot_path"]),
        str(source["eval_snapshot_sha256"]),
        str(source["eval_gate_path"]),
        str(source["eval_gate_sha256"]),
        ((2, 24), (4, 21)),
        str(model["name"]),
        str(model["revision"]),
        tuple(str(value) for value in cast(list[object], choice["labels"])),
        str(choice["prompt_template"]),
        _int(choice["max_input_tokens"], "choice input bound is not an integer"),
        _int(choice["candidate_count"], "candidate count is not an integer"),
        264,
        0.1,
        _number(post_gate["4_path_rate_strict_gt"], "path threshold is not numeric"),
        _int(post_gate["4_position_min"], "position threshold is not an integer"),
        _int(post_gate["4_paths_min"], "path count threshold is not an integer"),
        _number(post_gate["k10_mixed_sum_strict_gt"], "support threshold is not numeric"),
        _int(
            cast(Mapping[str, object], config["runtime"])["hard_process_seconds"],
            "runtime bound is not an integer",
        ),
        1,
        "bfloat16",
        False,
        True,
    )


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _load_qwen35(config: Mapping[str, object]) -> tuple[Qwen35TextTokenizer, Any, Any]:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    model_spec = cast(Mapping[str, object], config["model"])
    if getattr(transformers, "__version__", None) != str(model_spec["transformers"]):
        raise RuntimeError("warm-start requires the pinned Transformers version")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("warm-start requires exactly one CUDA device")
    auto_config = getattr(transformers, "AutoConfig", None)
    auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
    auto_model = getattr(transformers, "AutoModelForCausalLM", None)
    if auto_config is None or auto_tokenizer is None or auto_model is None:
        raise RuntimeError("pinned Qwen3.5 text owners are unavailable")
    name = str(model_spec["name"])
    revision = str(model_spec["revision"])
    repository = auto_config.from_pretrained(name, revision=revision, trust_remote_code=False)
    text_config = _field(repository, "text_config")
    if (
        _field(repository, "model_type") != "qwen3_5"
        or _field(repository, "architectures")
        not in (["Qwen3_5ForConditionalGeneration"], ("Qwen3_5ForConditionalGeneration",))
        or _field(text_config, "model_type") != "qwen3_5_text"
        or _field(text_config, "num_hidden_layers") != 32
        or _field(text_config, "hidden_size") != 2560
    ):
        raise RuntimeError("Qwen3.5 repository config does not match the pinned model")
    tokenizer = auto_tokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
    model = auto_model.from_pretrained(
        name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    if type(model).__name__ != "Qwen3_5ForCausalLM":
        raise RuntimeError("Qwen3.5 did not resolve to the native text causal model")
    loaded = getattr(model, "config", None)
    if loaded is None:
        raise RuntimeError("loaded Qwen3.5 model has no config")
    if (
        _field(loaded, "model_type") != "qwen3_5_text"
        or _field(loaded, "num_hidden_layers") != 32
        or _field(loaded, "hidden_size") != 2560
    ):
        raise RuntimeError("loaded Qwen3.5 text config is not the pinned shape")
    loaded.use_cache = False
    if loaded.use_cache is not False:
        raise RuntimeError("Qwen3.5 cache could not be disabled")
    model = model.to("cuda")
    model.eval()
    return Qwen35TextTokenizer(tokenizer), model, torch


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("warm-start cooperative deadline exceeded")


def _evaluate(
    tasks: Sequence[Any],
    scoring: ScoringConfig,
    tokenizer: Any,
    model: Any,
    torch: Any,
    deadline: float,
) -> dict[str, object]:
    scorer = TorchChoiceScorer(tokenizer, model, torch, scoring, deadline)
    scorer.preflight(_prompts(tasks, scoring))
    records = evaluate_states(tasks, scoring, scorer)
    by_hop, blocked = summarize(tasks, scoring, records)
    if scorer.forward_calls != 264:
        raise RuntimeError("base/post evaluation did not use exactly 264 choice forwards")
    return {"by_hop": by_hop, "blocked_reasons": blocked, "forward_calls": scorer.forward_calls}


def _assert_base(result: Mapping[str, object]) -> None:
    by_hop = cast(Mapping[str, object], result["by_hop"])
    two = cast(Mapping[str, object], by_hop["2"])
    four = cast(Mapping[str, object], by_hop["4"])
    if (
        two["teacher_top1_correct"] != 31
        or two["greedy_full_paths"] != 9
        or four["teacher_top1_correct"] != 30
        or four["teacher_position_top1_correct"] != [7, 4, 2, 17]
        or four["greedy_full_paths"] != 0
    ):
        raise RuntimeError("Qwen3.5 base evaluation does not match the authenticated baseline")


def _adapter_parameters(model: Any) -> list[tuple[str, Any]]:
    return [
        (str(name), parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def _adapter_metadata(
    parameters: Sequence[tuple[str, Any]], replacements: int
) -> dict[str, object]:
    tensors = [
        {
            "key": name,
            "shape": [int(value) for value in parameter.shape],
            "dtype": str(parameter.dtype),
        }
        for name, parameter in parameters
    ]
    return {
        "replacement_count": replacements,
        "trainable_parameter_count": sum(int(parameter.numel()) for _, parameter in parameters),
        "adapter_tensors": tensors,
    }


def _install_lora(
    torch: Any, model: Any, training: Mapping[str, object]
) -> tuple[list[tuple[str, Any]], dict[str, object]]:
    nn = torch.nn
    spec = cast(Mapping[str, object], training["lora"])
    rank = _int(spec["rank"], "LoRA rank is not an integer")
    alpha = _number(spec["alpha"], "LoRA alpha is not numeric")
    dropout = _number(spec["dropout"], "LoRA dropout is not numeric")
    expected = _int(spec["expected_replacements"], "LoRA replacement count is not an integer")
    targets = frozenset(str(value) for value in cast(list[object], spec["target_modules"]))

    class LoRALinear(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self, base: Any) -> None:
            super().__init__()
            self.base = base
            self.base.requires_grad_(False)
            adapter_dtype = getattr(torch, "float32", base.weight.dtype)
            self.a = nn.Linear(
                base.in_features,
                rank,
                bias=False,
                device=base.weight.device,
                dtype=adapter_dtype,
            )
            self.b = nn.Linear(
                rank,
                base.out_features,
                bias=False,
                device=base.weight.device,
                dtype=adapter_dtype,
            )
            self.dropout = nn.Dropout(dropout)
            self.scale = alpha / rank
            nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
            nn.init.zeros_(self.b.weight)

        def forward(self, inputs: Any) -> Any:
            base_value = self.base(inputs)
            delta = self.b(self.a(self.dropout(inputs.to(self.a.weight.dtype)))) * self.scale
            return base_value + delta.to(base_value.dtype)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replaced = 0
    for module in tuple(model.modules()):
        for name, child in tuple(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child))
                replaced += 1
    if replaced != expected or replaced != 224:
        raise RuntimeError(f"Qwen3.5 LoRA replaced {replaced} projections, expected 224")
    trainable = _adapter_parameters(model)
    if not trainable:
        raise RuntimeError("LoRA produced no trainable parameters")
    return trainable, _adapter_metadata(trainable, replaced)


def _prompt_prefix_length(prompt_ids: Sequence[int], full_ids: Sequence[int]) -> int:
    prefix = list(full_ids[: len(prompt_ids)])
    if prefix != list(prompt_ids) or len(full_ids) <= len(prompt_ids):
        raise ValueError("assistant sequence does not preserve the exact prompt prefix")
    return len(prompt_ids)


def _training_batch(
    tokenizer: Qwen35TextTokenizer,
    task: WarmTask,
    permutation: int,
    step: int,
    target: int,
    torch: Any,
) -> Mapping[str, Any]:
    prompt = warm_prompt(task, step, permutation)
    label = str(target)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": label}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_ids = _token_ids(tokenizer.tokenizer, prompt_text)
    full_ids = _token_ids(tokenizer.tokenizer, full_text)
    prompt_length = _prompt_prefix_length(prompt_ids, full_ids)
    if len(full_ids) > 8192:
        raise ValueError("warm-start training prompt exceeds the exact tokenizer bound")
    encoded = tokenizer(full_text, add_special_tokens=False, return_tensors="pt", truncation=False)
    input_ids = encoded["input_ids"]
    attention = encoded["attention_mask"]
    if getattr(input_ids, "shape", None) != (1, len(full_ids)):
        raise ValueError("warm-start training tokenizer shape is invalid")
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    if int((labels != -100).sum().item()) == 0:
        raise ValueError("warm-start assistant response has no loss tokens")
    return {
        "input_ids": input_ids.to("cuda"),
        "attention_mask": attention.to("cuda"),
        "labels": labels.to("cuda"),
        "torch": torch,
    }


def _adapter_norm(parameters: Sequence[tuple[str, Any]]) -> float:
    squared = 0.0
    for _, parameter in parameters:
        squared += float(parameter.detach().float().pow(2).sum().item())
    return math.sqrt(squared)


def _adapter_delta_l2(
    initial: Mapping[str, Any], parameters: Sequence[tuple[str, Any]]
) -> float:
    squared = 0.0
    for name, parameter in parameters:
        final = parameter.detach().float().cpu()
        squared += float((final - initial[name]).pow(2).sum().item())
    return math.sqrt(squared)


def _train(
    tokenizer: Qwen35TextTokenizer,
    model: Any,
    torch: Any,
    tasks: Sequence[WarmTask],
    config: Mapping[str, object],
    deadline: float,
) -> dict[str, object]:
    training = cast(Mapping[str, object], config["training"])
    torch.manual_seed(_int(training["seed"], "training seed is not an integer"))
    torch.cuda.manual_seed_all(_int(training["seed"], "training seed is not an integer"))
    trainable, metadata = _install_lora(torch, model, training)
    initial_adapter = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in trainable
    }
    optimizer_parameters = [parameter for _, parameter in trainable]
    initial_norm = _adapter_norm(trainable)
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=_number(training["learning_rate"], "learning rate is not numeric"),
    )
    rows = training_states(tasks)
    accumulation = _int(
        training["gradient_accumulation"], "gradient accumulation is not an integer"
    )
    expected_updates = _int(training["updates"], "update count is not an integer")
    if len(rows) != expected_updates * accumulation:
        raise ValueError("warm-start update arithmetic is not exact")
    warmup = max(
        1,
        math.ceil(
            expected_updates
            * _number(training["warmup_fraction"], "warmup fraction is not numeric")
        ),
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    updates = 0
    micro_losses: list[float] = []
    update_losses: list[float] = []
    gradient_norms: list[float] = []
    zero_gradient_updates = 0
    for index, (task, permutation, step, target) in enumerate(rows, 1):
        _check_deadline(deadline)
        batch = _training_batch(tokenizer, task, permutation, step, target, torch)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        if getattr(output, "past_key_values", None) is not None:
            raise RuntimeError("warm-start training returned a cache")
        loss = getattr(output, "loss", None)
        if loss is None or not math.isfinite(float(loss.detach().item())):
            raise RuntimeError("warm-start loss is not finite")
        micro_losses.append(float(loss.detach().item()))
        (loss / accumulation).backward()
        if index % accumulation == 0:
            preclip = torch.nn.utils.clip_grad_norm_(
                optimizer_parameters,
                _number(training["gradient_clip"], "gradient clip is not numeric"),
            )
            preclip_value = float(
                preclip.detach().item() if hasattr(preclip, "detach") else preclip
            )
            if not math.isfinite(preclip_value):
                raise RuntimeError("warm-start preclip gradient norm is not finite")
            gradient_norms.append(preclip_value)
            if preclip_value == 0.0:
                zero_gradient_updates += 1
            update_losses.append(sum(micro_losses[-accumulation:]) / accumulation)
            updates += 1
            if updates <= warmup:
                rate = (
                    _number(training["learning_rate"], "learning rate is not numeric")
                    * updates
                    / warmup
                )
            else:
                rate = _number(training["learning_rate"], "learning rate is not numeric") * max(
                    0.0, (expected_updates - updates) / max(1, expected_updates - warmup)
                )
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    if updates != expected_updates:
        raise RuntimeError("warm-start optimizer update count is not exact")
    final_norm = _adapter_norm(trainable)
    try:
        adapter_delta_l2 = _adapter_delta_l2(initial_adapter, trainable)
        material_min = _number(
            training["material_adapter_delta_l2_min"],
            "material adapter delta bound is not numeric",
        )
        health = {
            "mean_update_loss_first16": sum(update_losses[:16]) / min(16, len(update_losses)),
            "mean_update_loss_final16": sum(update_losses[-16:]) / min(16, len(update_losses)),
            "microbatch_loss_min": min(micro_losses),
            "microbatch_loss_max": max(micro_losses),
            "mean_preclip_gradient_norm": sum(gradient_norms) / len(gradient_norms),
            "max_preclip_gradient_norm": max(gradient_norms),
            "zero_gradient_update_count": zero_gradient_updates,
            "initial_adapter_norm": initial_norm,
            "final_adapter_norm": final_norm,
            "adapter_delta_l2": adapter_delta_l2,
            "material_change": adapter_delta_l2 > material_min,
            "optimization_failure": (
                adapter_delta_l2 <= material_min
                or zero_gradient_updates == expected_updates
            ),
        }
    finally:
        initial_adapter.clear()
    return {
        "updates": updates,
        "microbatches": len(rows),
        **metadata,
        "health": health,
        "trainable": trainable,
    }


def _publish_adapter(
    parameters: Sequence[tuple[str, Any]], metadata: Mapping[str, object], path: Path
) -> dict[str, object]:
    if path.exists():
        raise FileExistsError("warm-start adapter output already exists")
    safetensors = importlib.import_module("safetensors.torch")
    state = {name: parameter.detach().to("cpu").contiguous() for name, parameter in parameters}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".warm-adapter-", dir=path.parent)
    os.close(fd)
    try:
        safetensors.save_file(state, temporary)
        if path.exists():
            raise FileExistsError("warm-start adapter output appeared during publication")
        os.link(temporary, path)
        loaded = safetensors.load_file(str(path), device="cpu")
        if set(loaded) != set(state):
            raise RuntimeError("published adapter keys are not exact")
        for name, tensor in state.items():
            if tuple(loaded[name].shape) != tuple(tensor.shape) or str(loaded[name].dtype) != str(
                tensor.dtype
            ):
                raise RuntimeError("published adapter tensor metadata is not exact")
        return {
            "path": "runs/musique-ans-support-warm-start-v1/adapter.safetensors",
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "replacement_count": metadata["replacement_count"],
            "trainable_parameter_count": metadata["trainable_parameter_count"],
            "adapter_tensors": metadata["adapter_tensors"],
        }
    finally:
        Path(temporary).unlink(missing_ok=True)


def _remove_adapter(path: Path) -> None:
    if path.exists():
        path.unlink()


def _write_report(
    output_path: Path,
    payload: Mapping[str, object],
    adapter_output: Path,
    adapter_published: bool,
) -> None:
    try:
        write_exclusive(output_path, payload)
    except BaseException:
        if adapter_published:
            _remove_adapter(adapter_output)
        raise


def _gate_summary(
    post: Mapping[str, object], config: Mapping[str, object], training: Mapping[str, object]
) -> dict[str, object]:
    by_hop = cast(Mapping[str, object], post["by_hop"])
    two = cast(Mapping[str, object], by_hop["2"])
    four = cast(Mapping[str, object], by_hop["4"])
    thresholds = cast(Mapping[str, object], cast(Mapping[str, object], config["gate"])["post"])
    reasons: list[str] = []
    if _int(two["teacher_top1_correct"], "2-hop teacher count is not an integer") < _int(
        thresholds["2_teacher_min"], "2-hop teacher threshold is not an integer"
    ):
        reasons.append("2-hop teacher gate failed")
    if _int(two["greedy_full_paths"], "2-hop path count is not an integer") < _int(
        thresholds["2_paths_min"], "2-hop path threshold is not an integer"
    ):
        reasons.append("2-hop ordered path gate failed")
    positions = cast(list[int], four["teacher_position_top1_correct"])
    if _int(four["teacher_top1_correct"], "4-hop teacher count is not an integer") < _int(
        thresholds["4_teacher_min"], "4-hop teacher threshold is not an integer"
    ):
        reasons.append("4-hop teacher gate failed")
    if any(
        value < _int(thresholds["4_position_min"], "4-hop position threshold is not an integer")
        for value in positions
    ):
        reasons.append("4-hop position gate failed")
    if _int(four["greedy_full_paths"], "4-hop path count is not an integer") < _int(
        thresholds["4_paths_min"], "4-hop path threshold is not an integer"
    ) or not _number(four["greedy_full_path_rate"], "4-hop path rate is not numeric") > _number(
        thresholds["4_path_rate_strict_gt"], "4-hop path rate threshold is not numeric"
    ):
        reasons.append("4-hop ordered path gate failed")
    k10 = cast(Mapping[str, object], four["k10_mixed"])
    if not _number(k10["sum_m"], "K10 support sum is not numeric") > _number(
        thresholds["k10_mixed_sum_strict_gt"], "support threshold is not numeric"
    ):
        reasons.append("4-hop K10 support gate failed")
    health = cast(Mapping[str, object], training["health"])
    optimization_failure = bool(health["optimization_failure"])
    return {
        "passed": not reasons and not optimization_failure,
        "capability_pass": not reasons,
        "capability_blocked_reasons": reasons,
        "training_optimization_pass": not optimization_failure,
        "optimization_failure": optimization_failure,
        "training_experiment_eligible": not reasons and not optimization_failure,
        "capability_only": True,
    }


def _run(config_path: Path, snapshot_path: Path, output_path: Path, adapter_output: Path) -> None:
    config, config_sha = load_config(config_path)
    source = _source_map(config)
    adapter_spec = cast(Mapping[str, object], config["adapter"])
    if adapter_output.resolve() != (ROOT / str(adapter_spec["path"])).resolve():
        raise ValueError("warm-start adapter path is not the fixed path")
    if snapshot_path.resolve() != (ROOT / str(source["snapshot_path"])).resolve():
        raise ValueError("warm-start snapshot path is not the frozen path")
    if adapter_output.exists():
        raise FileExistsError("warm-start adapter output already exists")
    tasks = load_snapshot(snapshot_path, str(source["snapshot_sha256"]))
    if (
        state_order_digest(training_states(tasks))
        != cast(Mapping[str, object], config["state_order"])["sha256"]
    ):
        raise ValueError("warm-start config and snapshot state order disagree")
    eval_snapshot = ROOT / EVAL_SNAPSHOT_PATH
    eval_gate = ROOT / EVAL_GATE_PATH
    if _sha256(eval_snapshot) != EVAL_SNAPSHOT_SHA256 or _sha256(eval_gate) != EVAL_GATE_SHA256:
        raise ValueError("frozen eval inputs changed")
    eval_tasks = load_tasks(eval_snapshot, expected_sha256=EVAL_SNAPSHOT_SHA256)
    scoring = _scoring_config(config, config_sha)
    started = time.monotonic()
    runtime = cast(Mapping[str, object], config["runtime"])
    deadline = started + _int(runtime["hard_process_seconds"], "runtime bound is not an integer")
    tokenizer, model, torch = _load_qwen35(config)
    adapter_published = False
    try:
        base = _evaluate(eval_tasks, scoring, tokenizer, model, torch, deadline)
        _assert_base(base)
        training = _train(tokenizer, model, torch, tasks, config, deadline)
        model.eval()
        post = _evaluate(eval_tasks, scoring, tokenizer, model, torch, deadline)
        gate_summary = _gate_summary(post, config, training)
        adapter_record: dict[str, object] | None = None
        if bool(gate_summary["passed"]):
            adapter_record = _publish_adapter(
                cast(Sequence[tuple[str, Any]], training["trainable"]), training, adapter_output
            )
            adapter_published = True
        else:
            _remove_adapter(adapter_output)
        training_report = {key: value for key, value in training.items() if key != "trainable"}
        payload = {
            "schema_version": 1,
            "experiment": config["experiment"],
            "status": "complete",
            "config_sha256": config_sha,
            "snapshot_sha256": source["snapshot_sha256"],
            "model": {
                "name": MODEL_NAME,
                "revision": MODEL_REVISION,
                "dtype": "bfloat16",
                "cuda_devices": 1,
            },
            "base": base,
            "training": training_report,
            "adapter": adapter_record,
            "post": post,
            "gate": gate_summary,
            "runtime": {
                "elapsed_seconds": time.monotonic() - started,
                "choice_forward_calls": _int(
                    base["forward_calls"], "base call count is not an integer"
                )
                + _int(post["forward_calls"], "post call count is not an integer"),
                "training_microbatches": len(training_states(tasks)),
                "deadline_cooperative": True,
            },
            "authority": {
                "prime": False,
                "source": False,
                "parquet": False,
                "model_calls": True,
                "training": True,
                "science": False,
            },
            "gold_decomposition_in_prompt": False,
        }
    finally:
        if not adapter_published:
            _remove_adapter(adapter_output)
        del model, tokenizer
        gc.collect()
        empty_cache = getattr(torch.cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    _write_report(output_path, payload, adapter_output, adapter_published)
    print(
        json.dumps({"output": str(output_path), "passed": gate_summary["passed"]}, sort_keys=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--build", action="store_true")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter-output", type=Path, default=DEFAULT_ADAPTER_OUTPUT)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--eval-snapshot", type=Path, default=ROOT / EVAL_SNAPSHOT_PATH)
    args = parser.parse_args()
    if args.build:
        if args.archive is None:
            raise ValueError("--build requires the authenticated official archive")
        built = build_snapshot(args.archive, args.eval_snapshot, args.snapshot)
        source = cast(Mapping[str, object], built["source"])
        tasks = load_snapshot(args.snapshot, _sha256(args.snapshot))
        config = config_payload(
            _sha256(args.snapshot),
            str(source["train_entry_sha256"]),
            state_order_digest(training_states(tasks)),
        )
        with args.config.open("xb") as handle:
            import redco.contracts

            handle.write(redco.contracts.canonical_json(config))
        print(
            json.dumps(
                {
                    "config": str(args.config),
                    "snapshot": str(args.snapshot),
                    "snapshot_sha256": _sha256(args.snapshot),
                    "state_order_sha256": cast(Mapping[str, object], config["state_order"])[
                        "sha256"
                    ],
                },
                sort_keys=True,
            )
        )
        return
    config, config_sha = load_config(args.config)
    tasks = load_snapshot(args.snapshot, str(_source_map(config)["snapshot_sha256"]))
    if args.check:
        print(json.dumps(check_payload(config, config_sha, tasks), sort_keys=True))
        return
    _run(args.config, args.snapshot, args.output, args.adapter_output)


if __name__ == "__main__":
    main()
