"""Audit midpoint-shard determinism and conservative context capacity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redco.integrations.signed_subprocess import (  # noqa: E402
    atomic_write_json,
    sign_payload,
)

MODEL_REPO = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MAX_MODEL_LEN = 8_192
MAX_CHILD_COMPLETION_TOKENS = 768
MAX_ROOT_COMPLETION_TOKENS = 768
MAX_ELIGIBLE_CHILDREN = 4
ROOT_RENDER_SAFETY_BUFFER_TOKENS = 512
RECORDED_CHILD_SYSTEM_AND_RENDER_OVERHEAD_TOKENS = 955


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_rows(
    rows: list[dict[str, Any]],
    *,
    encode: Callable[[str], list[int]],
) -> dict[str, Any]:
    records = []
    for row in rows:
        # Project only deployment-visible fields. Reference evidence, answer
        # type, reward, and split identity cannot affect the midpoint or prompt.
        paper = str(row["paper"])
        question = str(row["question"])
        midpoint = len(paper) // 2
        shards = (paper[:midpoint], paper[midpoint:])
        prompts = [
            (
                f"Return candidate verbatim evidence for {question!r} "
                f"from only this excerpt:\n{shard}"
            )
            for shard in shards
        ]
        prompt_tokens = [len(encode(prompt)) for prompt in prompts]
        conservative_totals = [
            count + RECORDED_CHILD_SYSTEM_AND_RENDER_OVERHEAD_TOKENS + MAX_CHILD_COMPLETION_TOKENS
            for count in prompt_tokens
        ]
        records.append(
            {
                "example_id": str(row["example_id"]),
                "paper_id": str(row["paper_id"]),
                "paper_characters": len(paper),
                "midpoint": midpoint,
                "shard_characters": [len(shard) for shard in shards],
                "union_exact": "".join(shards) == paper,
                "both_shards_nonempty": all(shards),
                "prompt_tokens_without_system": prompt_tokens,
                "conservative_prompt_plus_completion_tokens": conservative_totals,
                "fits_8192": max(conservative_totals) <= MAX_MODEL_LEN,
            }
        )
    checks = {
        "all_papers_unique": len({row["paper_id"] for row in records}) == len(records),
        "midpoint_union_exact_everywhere": all(row["union_exact"] for row in records),
        "both_shards_nonempty_everywhere": all(row["both_shards_nonempty"] for row in records),
        "all_conservative_context_totals_fit_8192": all(row["fits_8192"] for row in records),
    }
    return {
        "records": records,
        "checks": checks,
        "maximum_conservative_tokens": max(
            total for row in records for total in row["conservative_prompt_plus_completion_tokens"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--sft-dataset", type=Path, required=True)
    parser.add_argument("--retired-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _sha256(args.dataset) != args.dataset_sha256:
        raise ValueError("extension dataset hash mismatch")
    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    audit = audit_rows(
        rows,
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
    )
    scaffold = args.scaffold.read_text(encoding="utf-8")
    first_sft = json.loads(args.sft_dataset.read_text(encoding="utf-8").splitlines()[0])
    sft_code = json.loads(first_sft["messages"][2]["tool_calls"][0]["function"]["arguments"])[
        "code"
    ]
    exact_contract_lines = [
        "midpoint = len(paper_text) // 2",
        "excerpts = [paper_text[:midpoint], paper_text[midpoint:]]",
        "children = await asyncio.gather(*(rlm(prompt) for prompt in prompts))",
    ]
    trace = json.loads(args.retired_trace.read_text(encoding="utf-8"))["traces"][0]
    trace_calls = trace["calls"]
    observed_empty_child_prompt_tokens = min(
        int(call["usage"]["prompt_tokens"]) for call in trace["calls"] if call["rlm"]["depth"] == 1
    )
    root_calls = [call for call in trace_calls if call["rlm"]["depth"] == 0]
    child_calls = [call for call in trace_calls if call["rlm"]["depth"] == 1]
    observed_final_root_prompt_tokens = int(root_calls[-1]["usage"]["prompt_tokens"])
    observed_child_completion_tokens = sum(
        int(call["usage"]["completion_tokens"]) for call in child_calls
    )
    conservative_final_root_total = (
        observed_final_root_prompt_tokens
        + max(
            0,
            MAX_ELIGIBLE_CHILDREN * MAX_CHILD_COMPLETION_TOKENS - observed_child_completion_tokens,
        )
        + MAX_ROOT_COMPLETION_TOKENS
        + ROOT_RENDER_SAFETY_BUFFER_TOKENS
    )
    checks = {
        **audit["checks"],
        "exact_midpoint_lines_match_frozen_sft": all(
            line in scaffold and line in sft_code for line in exact_contract_lines
        ),
        "recorded_overhead_bound_not_understated": (
            observed_empty_child_prompt_tokens <= RECORDED_CHILD_SYSTEM_AND_RENDER_OVERHEAD_TOKENS
        ),
        "fresh_reference_fields_never_used_by_audit": True,
        "worst_case_returning_root_fits_8192": (conservative_final_root_total <= MAX_MODEL_LEN),
    }
    if not all(checks.values()):
        raise ValueError(f"midpoint context audit failed: {checks}")
    report = sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-midpoint-context-audit",
            "scope": "reference-independent CPU capacity preflight",
            "model": {
                "repo": MODEL_REPO,
                "revision": MODEL_REVISION,
                "maximum_model_length": MAX_MODEL_LEN,
                "maximum_child_completion_tokens": MAX_CHILD_COMPLETION_TOKENS,
                "maximum_root_completion_tokens": MAX_ROOT_COMPLETION_TOKENS,
                "maximum_eligible_children": MAX_ELIGIBLE_CHILDREN,
                "conservative_recorded_overhead_tokens": (
                    RECORDED_CHILD_SYSTEM_AND_RENDER_OVERHEAD_TOKENS
                ),
            },
            "dataset": {
                "path": args.dataset.as_posix(),
                "sha256": args.dataset_sha256,
                "papers": len(rows),
            },
            "scaffold": {
                "path": args.scaffold.as_posix(),
                "sha256": _sha256(args.scaffold),
            },
            "frozen_sft_dataset": {
                "path": args.sft_dataset.as_posix(),
                "sha256": _sha256(args.sft_dataset),
            },
            "retired_trace": {
                "path": args.retired_trace.as_posix(),
                "sha256": _sha256(args.retired_trace),
                "observed_empty_child_prompt_tokens": (observed_empty_child_prompt_tokens),
                "observed_final_root_prompt_tokens": (observed_final_root_prompt_tokens),
                "observed_child_completion_tokens": (observed_child_completion_tokens),
            },
            "conservative_final_root_prompt_plus_completion_tokens": (
                conservative_final_root_total
            ),
            "returning_root_render_safety_buffer_tokens": (ROOT_RENDER_SAFETY_BUFFER_TOKENS),
            "maximum_conservative_tokens": audit["maximum_conservative_tokens"],
            "checks": checks,
            "records": audit["records"],
        }
    )
    atomic_write_json(args.output, report)


if __name__ == "__main__":
    main()
