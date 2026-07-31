"""Exercise the frozen Stage D SFT corpus through Prime-RL's real data path."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

from prime_rl.configs.sft import SFTConfig
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.trainer.sft.data import (
    SFTDataset,
    load_sft_dataset,
)
from renderers.base import create_renderer

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
)


def _prefix_example(
    example: dict[str, Any], *, through_role: str
) -> dict[str, Any]:
    messages = list(example["messages"])
    end = next(
        index + 1
        for index, message in enumerate(messages)
        if message["role"] == through_role
    )
    return {**example, "messages": messages[:end]}


def audit(config_path: Path) -> dict[str, Any]:
    raw_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config = SFTConfig.model_validate(raw_config)
    if config.data.type != "sft":
        raise ValueError("Stage D renderer audit requires SFT data")
    if config.deployment.num_gpus != 1:
        raise ValueError("Stage D v4 fallback must use one GPU")

    tokenizer = setup_tokenizer(config.tokenizer)
    renderer = create_renderer(tokenizer, config.renderer)
    raw_dataset = load_sft_dataset(config.data)
    processor = SFTDataset(
        raw_dataset,
        renderer,
        shuffle=False,
        seed=config.data.seed,
        seq_len=config.data.seq_len,
        loss_mask_config=config.data.loss_mask,
        non_dp_size=1,
        multimodal=False,
    )

    rows = []
    for index in range(len(raw_dataset)):
        example = dict(raw_dataset[index])
        full = processor._process(example)
        tool_call = processor._process(
            _prefix_example(example, through_role="assistant")
        )
        through_tool = processor._process(
            _prefix_example(example, through_role="tool")
        )
        if full is None or tool_call is None or through_tool is None:
            raise ValueError(f"SFT row {index} was dropped by Prime-RL")
        full_tokens = len(full["input_ids"])
        full_trainable = sum(full["loss_mask"])
        tool_call_trainable = sum(tool_call["loss_mask"])
        through_tool_trainable = sum(through_tool["loss_mask"])
        final_trainable = full_trainable - through_tool_trainable
        checks = {
            "within_2048_without_truncation": (
                full_tokens <= config.data.seq_len
            ),
            "tool_call_has_trainable_tokens": tool_call_trainable > 0,
            "final_answer_adds_trainable_tokens": final_trainable > 0,
            "tool_call_remains_trainable_in_real_sequence": (
                through_tool_trainable > 0
            ),
        }
        rows.append(
            {
                "index": index,
                "synthetic_id": example.get("synthetic_id"),
                "tokens": full_tokens,
                "trainable_tokens": full_trainable,
                "tool_call_trainable_tokens": tool_call_trainable,
                "final_answer_trainable_tokens": final_trainable,
                "checks": checks,
                "passes": all(checks.values()),
            }
        )
    checks = {
        "exactly_32_rows": len(rows) == 32,
        "all_rows_pass": all(row["passes"] for row in rows),
        "maximum_tokens_at_most_2048": (
            max(row["tokens"] for row in rows) <= config.data.seq_len
        ),
        "minimum_tool_call_trainable_positive": (
            min(row["tool_call_trainable_tokens"] for row in rows) > 0
        ),
        "minimum_final_answer_trainable_positive": (
            min(row["final_answer_trainable_tokens"] for row in rows) > 0
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-sft-prime-renderer-preflight-v4",
            "config": config_path.as_posix(),
            "renderer": config.renderer.name,
            "dataset": str(config.data.data_files),
            "rows": rows,
            "checks": checks,
            "passes": all(checks.values()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.config)
    atomic_write_json(args.output, report)
    if not report["passes"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
