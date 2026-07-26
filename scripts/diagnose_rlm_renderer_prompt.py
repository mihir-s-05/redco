"""Compare CPU chat-template variants with a recorded returning-root prompt."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from redco.analysis.empirical_branch_replay import (
    _chat_tools,
    _openai_messages,
)
from redco.integrations.verifiers_trace import audit_trace_file, path_to_node


def _first_mismatch(expected: tuple[int, ...], actual: tuple[int, ...]) -> int | None:
    for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
        if left != right:
            return index
    return min(len(expected), len(actual)) if expected != actual else None


def _first_text_mismatch(expected: str, actual: str) -> int | None:
    for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
        if left != right:
            return index
    return min(len(expected), len(actual)) if expected != actual else None


def _render(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> tuple[int, ...]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(rendered, list) and hasattr(rendered, "get"):
        rendered = rendered.get("input_ids")
    if not isinstance(rendered, list):
        raise TypeError(
            f"tokenizer returned {type(rendered).__name__}, expected a list"
        )
    try:
        return tuple(int(token_id) for token_id in rendered)
    except (TypeError, ValueError) as exc:
        raise TypeError("tokenizer returned invalid token IDs") from exc


def _template_messages(
    messages: list[dict[str, Any]],
    *,
    retain_tool_name: bool,
) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        if message.get("role") == "tool" and not retain_tool_name:
            message.pop("name", None)
        for call in message.get("tool_calls", []):
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                function["arguments"] = json.loads(arguments)
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()

    envelope = json.loads(args.input.read_bytes())
    raw_trace = envelope["traces"][0]
    audit = audit_trace_file(args.input)
    final_call = max(
        (
            call
            for call in audit.calls
            if call.agent_depth == 0 and call.turn_index is not None
        ),
        key=lambda call: call.turn_index or 0,
    )
    nodes = raw_trace["nodes"]
    message_path = path_to_node(nodes, final_call.node_index)[:-1]
    messages = _openai_messages(
        [copy.deepcopy(nodes[index]["message"]) for index in message_path]
    )
    tools = _chat_tools(raw_trace["tools"])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    expected = final_call.prompt_token_ids
    recorded_static_prefix = tuple(
        token_id
        for index in message_path[:-1]
        for token_id in nodes[index]["token_ids"]
    )
    variants = {}
    for retain_tool_name in (True, False):
        label = "tool_name_retained" if retain_tool_name else "tool_name_removed"
        actual = _render(
            tokenizer,
            _template_messages(
                messages,
                retain_tool_name=retain_tool_name,
            ),
            tools,
            add_generation_prompt=True,
        )
        canonical_prefix = _render(
            tokenizer,
            _template_messages(
                messages[:-1],
                retain_tool_name=retain_tool_name,
            ),
            tools,
            add_generation_prompt=False,
        )
        if actual[: len(canonical_prefix)] != canonical_prefix:
            raise RuntimeError("canonical history is not a prefix of full render")
        hybrid = recorded_static_prefix + actual[len(canonical_prefix) :]
        recorded_prefix_text = tokenizer.decode(recorded_static_prefix)
        recorded_suffix = expected[len(recorded_static_prefix) :]
        recorded_suffix_text = tokenizer.decode(recorded_suffix)
        boundary_candidates = []
        for split in range(
            max(0, len(canonical_prefix) - 8),
            min(len(actual), len(canonical_prefix) + 9),
        ):
            candidate = recorded_static_prefix + actual[split:]
            boundary_candidates.append(
                {
                    "split": split,
                    "canonical_prefix_text_exact": (
                        tokenizer.decode(actual[:split])
                        == recorded_prefix_text
                    ),
                    "canonical_suffix_text_exact": (
                        tokenizer.decode(actual[split:])
                        == recorded_suffix_text
                    ),
                    "hybrid_exact": candidate == expected,
                }
            )
        mismatch = _first_mismatch(expected, actual)
        expected_text = tokenizer.decode(expected)
        actual_text = tokenizer.decode(actual)
        text_mismatch = _first_text_mismatch(expected_text, actual_text)
        variants[label] = {
            "exact": actual == expected,
            "decoded_text_exact": actual_text == expected_text,
            "expected_tokens": len(expected),
            "actual_tokens": len(actual),
            "first_mismatch": mismatch,
            "first_text_mismatch": text_mismatch,
            "expected_window": (
                list(expected[max(0, mismatch - 5) : mismatch + 6])
                if mismatch is not None
                else []
            ),
            "actual_window": (
                list(actual[max(0, mismatch - 5) : mismatch + 6])
                if mismatch is not None
                else []
            ),
            "expected_decoded_window": (
                tokenizer.decode(expected[max(0, mismatch - 12) : mismatch + 13])
                if mismatch is not None
                else ""
            ),
            "actual_decoded_window": (
                tokenizer.decode(actual[max(0, mismatch - 12) : mismatch + 13])
                if mismatch is not None
                else ""
            ),
            "canonical_prefix_tokens": len(canonical_prefix),
            "recorded_static_prefix_tokens": len(recorded_static_prefix),
            "hybrid_tokens": len(hybrid),
            "hybrid_exact": hybrid == expected,
            "hybrid_decoded_text_exact": (
                tokenizer.decode(hybrid) == expected_text
            ),
            "boundary_candidates": boundary_candidates,
        }
    print(json.dumps(variants, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
