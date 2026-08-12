"""Run the bounded QASPER trajectory-LOO versus ReDCO model pilot."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json
from redco.experiments.qasper_evidence import (
    EvidenceTask,
    PilotBudget,
    build_span_options,
    load_pilot_tasks,
    stage_one_prompt,
    stage_two_prompt,
)
from redco.experiments.qasper_runtime import Decision, redco_batch, trajectory_batch
from redco.integrity import sha256_bytes

LABELS = (" A", " B", " C", " D")


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    expected = {
        "behavior_drift_weight",
        "cost_limit_usd",
        "gradient_clip",
        "learning_rate",
        "lora",
        "max_runtime_minutes",
        "model",
        "model_revision",
        "optimizer",
        "pilot",
        "runtime",
        "schema_version",
        "seed",
        "temperature",
        "weight_decay",
    }
    if type(value) is not dict or set(value) != expected or value["schema_version"] != 1:
        raise ValueError("experiment config has the wrong schema")
    if value["optimizer"] != "adamw" or value["pilot"] != "qasper-evidence-v1":
        raise ValueError("experiment config does not describe the reviewed pilot")
    if not 0 < float(value["cost_limit_usd"]) <= 6:
        raise ValueError("experiment cost limit is outside the reviewed bound")
    if not 0 < int(value["max_runtime_minutes"]) <= 180:
        raise ValueError("experiment runtime is outside the reviewed bound")
    return value


def _install_lora(torch: Any, model: Any, config: dict[str, Any]) -> list[Any]:
    nn = torch.nn
    lora = config["lora"]
    rank = int(lora["rank"])
    alpha = float(lora["alpha"])
    dropout = float(lora["dropout"])
    targets = frozenset(str(value) for value in lora["target_modules"])

    class _LoRALinear(nn.Module):
        def __init__(self, base: Any) -> None:
            super().__init__()
            self.base = base
            self.base.requires_grad_(False)
            self.a = nn.Linear(
                base.in_features,
                rank,
                bias=False,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            self.b = nn.Linear(
                rank,
                base.out_features,
                bias=False,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            self.dropout = nn.Dropout(dropout)
            self.scale = alpha / rank
            nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
            nn.init.zeros_(self.b.weight)

        def forward(self, inputs: Any) -> Any:
            return self.base(inputs) + self.b(self.a(self.dropout(inputs))) * self.scale

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replaced = 0
    for module in tuple(model.modules()):
        for name, child in tuple(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, _LoRALinear(child))
                replaced += 1
    if replaced == 0:
        raise ValueError("LoRA target modules did not match the model")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA installation produced no trainable parameters")
    return trainable


class Policy:
    def __init__(self, torch: Any, transformers: Any, config: dict[str, Any]) -> None:
        self.torch = torch
        self.temperature = float(config["temperature"])
        torch.manual_seed(int(config["seed"]))
        torch.cuda.manual_seed_all(int(config["seed"]))
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            config["model"],
            revision=config["model_revision"],
            trust_remote_code=False,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            config["model"],
            revision=config["model_revision"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        ).cuda()
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        self.trainable = _install_lora(torch, self.model, config)
        self.label_ids = tuple(self._one_token(label) for label in LABELS)
        if len(set(self.label_ids)) != 4:
            raise ValueError("choice labels do not map to four unique tokens")

    def _one_token(self, label: str) -> int:
        values = self.tokenizer.encode(label, add_special_tokens=False)
        if len(values) != 1:
            raise ValueError(f"choice {label!r} is not one tokenizer token: {values}")
        return int(values[0])

    def _choice_logprobs(self, prompts: Sequence[str]) -> Any:
        encoded = self.tokenizer(
            list(prompts),
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        output = self.model(**encoded, logits_to_keep=1)
        label_ids = self.torch.tensor(self.label_ids, device="cuda")
        logits = output.logits[:, -1, :].index_select(1, label_ids)
        return self.torch.log_softmax(logits / self.temperature, dim=-1)

    def sample(self, prompts: Sequence[str]) -> tuple[tuple[int, float], ...]:
        with self.torch.no_grad():
            logprobs = self._choice_logprobs(prompts)
            actions = self.torch.multinomial(logprobs.exp(), num_samples=1).squeeze(1)
            chosen = logprobs.gather(1, actions[:, None]).squeeze(1)
        return tuple(
            (int(action), float(logprob))
            for action, logprob in zip(actions.cpu(), chosen.cpu(), strict=True)
        )

    def greedy(self, prompts: Sequence[str]) -> tuple[int, ...]:
        with self.torch.no_grad():
            return tuple(int(value) for value in self._choice_logprobs(prompts).argmax(1).cpu())

    def loss(self, decisions: Sequence[Decision], drift_weight: float) -> Any:
        if not decisions:
            raise ValueError("training update needs decisions")
        logprobs = self._choice_logprobs([decision.prompt for decision in decisions])
        indices = self.torch.tensor(
            [decision.action for decision in decisions],
            device="cuda",
        )
        current = logprobs.gather(1, indices[:, None]).squeeze(1)
        behavior = self.torch.tensor(
            [decision.behavior_logprob for decision in decisions],
            dtype=current.dtype,
            device="cuda",
        )
        advantages = self.torch.tensor(
            [decision.advantage for decision in decisions],
            dtype=current.dtype,
            device="cuda",
        )
        weights = self.torch.tensor(
            [decision.outer_weight for decision in decisions],
            dtype=current.dtype,
            device="cuda",
        )
        normalizer = sum(decision.decision_units for decision in decisions)
        policy_gradient = (-advantages * current * weights).sum() / normalizer
        behavior_drift = ((current - behavior).square() * weights).sum() / normalizer
        return policy_gradient + drift_weight * behavior_drift

    def adapter_state(self) -> dict[str, Any]:
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.model.state_dict().items()
            if name.endswith((".a.weight", ".b.weight"))
        }


def _evaluate(policy: Policy, tasks: Sequence[EvidenceTask]) -> dict[str, float | int]:
    paragraph_actions = policy.greedy(tuple(stage_one_prompt(task) for task in tasks))
    span_prompts = tuple(
        stage_two_prompt(task, action)
        for task, action in zip(tasks, paragraph_actions, strict=True)
    )
    span_actions = policy.greedy(span_prompts)
    paragraphs = 0
    exact = 0
    for task, paragraph, span in zip(tasks, paragraph_actions, span_actions, strict=True):
        paragraph_correct = paragraph == task.gold_paragraph_index
        paragraphs += int(paragraph_correct)
        _, gold_span = build_span_options(task, paragraph)
        exact += int(paragraph_correct and span == gold_span)
    return {
        "exact_evidence": exact,
        "exact_evidence_rate": exact / len(tasks),
        "paragraph": paragraphs,
        "paragraph_rate": paragraphs / len(tasks),
        "tasks": len(tasks),
    }


def _train_arm(
    torch: Any,
    transformers: Any,
    config: dict[str, Any],
    tasks: tuple[EvidenceTask, ...],
    arm: str,
    output_dir: Path,
) -> dict[str, Any]:
    policy = Policy(torch, transformers, config)
    optimizer = torch.optim.AdamW(
        policy.trainable,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    budget = PilotBudget()
    train_tasks = tuple(task for task in tasks if task.split == "train")
    eval_tasks = tuple(task for task in tasks if task.split == "eval")
    initial = _evaluate(policy, eval_tasks)
    updates: list[dict[str, float | int]] = []
    for step, task in enumerate(train_tasks, 1):
        decisions, mean_reward = (
            trajectory_batch(policy, task, budget)
            if arm == "trajectory_loo"
            else redco_batch(policy, task)
        )
        optimizer.zero_grad(set_to_none=True)
        loss = policy.loss(decisions, float(config["behavior_drift_weight"]))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.trainable,
            float(config["gradient_clip"]),
        )
        optimizer.step()
        updates.append(
            {
                "decision_units": sum(item.decision_units for item in decisions),
                "gradient_norm": float(grad_norm),
                "loss": float(loss.detach()),
                "mean_reward": mean_reward,
                "policy_calls": budget.baseline_calls_per_update,
                "step": step,
            }
        )
    final = _evaluate(policy, eval_tasks)
    adapter_path = output_dir / f"{arm}-adapter.pt"
    torch.save(policy.adapter_state(), adapter_path)
    result = {
        "adapter": {
            "bytes": adapter_path.stat().st_size,
            "path": adapter_path.name,
            "sha256": sha256_bytes(adapter_path.read_bytes()),
        },
        "arm": arm,
        "evaluation_after": final,
        "evaluation_before": initial,
        "rollout_calls": budget.rollout_calls_per_arm,
        "updates": updates,
    }
    del optimizer, policy
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qasper-evidence-pilot-v1.json"),
    )
    parser.add_argument("--data", type=Path, default=Path("data/qasper-evidence-pilot-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/qasper-evidence-pilot-v1"))
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    config = _load_config(arguments.config)
    tasks = load_pilot_tasks(arguments.data)
    if arguments.check:
        print(
            json.dumps(
                {
                    "config_sha256": sha256_bytes(arguments.config.read_bytes()),
                    "data_sha256": sha256_bytes(arguments.data.read_bytes()),
                    "eval_tasks": len([task for task in tasks if task.split == "eval"]),
                    "train_tasks": len([task for task in tasks if task.split == "train"]),
                },
                sort_keys=True,
            )
        )
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import torch
    import transformers

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the pilot requires exactly one visible CUDA GPU")
    random.seed(int(config["seed"]))
    started = time.monotonic()
    arguments.output.mkdir(parents=True, exist_ok=False)
    arms = [
        _train_arm(torch, transformers, config, tasks, arm, arguments.output)
        for arm in ("trajectory_loo", "redco")
    ]
    elapsed = time.monotonic() - started
    if elapsed > int(config["max_runtime_minutes"]) * 60:
        raise RuntimeError("pilot exceeded its reviewed runtime bound")
    payload = {
        "arms": arms,
        "config": config,
        "config_sha256": sha256_bytes(arguments.config.read_bytes()),
        "data_sha256": sha256_bytes(arguments.data.read_bytes()),
        "elapsed_seconds": elapsed,
        "environment": {
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "git_commit": _git_head(),
        "schema_version": 1,
    }
    report = {
        "payload": payload,
        "payload_sha256": sha256_bytes(canonical_json(payload)),
        "schema_version": 1,
    }
    (arguments.output / "report.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({"payload_sha256": report["payload_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
