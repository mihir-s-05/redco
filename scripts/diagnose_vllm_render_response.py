"""Persist exact boundary diagnostics from vLLM's GPU-less render server."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from redco.analysis.empirical_branch_replay import (
    TokenInferenceClient,
    _chat_tools,
    _integer_tuple,
    _message,
    _openai_messages,
    derive_lossless_render_boundary,
)
from redco.contracts import canonical_json
from redco.integrations.verifiers_trace import (
    audit_trace_file,
    load_trace_records,
    path_to_node,
)


def _common_suffix_length(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> int:
    common = 0
    while (
        common < len(left)
        and common < len(right)
        and left[-(common + 1)] == right[-(common + 1)]
    ):
        common += 1
    return common


def _common_text_prefix_length(left: str, right: str) -> int:
    common = 0
    while (
        common < len(left)
        and common < len(right)
        and left[common] == right[common]
    ):
        common += 1
    return common


def _common_text_suffix_length(left: str, right: str) -> int:
    common = 0
    while (
        common < len(left)
        and common < len(right)
        and left[-(common + 1)] == right[-(common + 1)]
    ):
        common += 1
    return common


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()

    source_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
    audit = audit_trace_file(args.input)
    traces = load_trace_records(args.input)
    if len(traces) != 1:
        raise ValueError("diagnostic requires exactly one trace")
    raw_trace = traces[0]
    calls = tuple(sorted(audit.calls, key=lambda call: call.call_index))
    final_call = max(
        (
            call
            for call in calls
            if call.agent_depth == 0 and call.turn_index is not None
        ),
        key=lambda call: call.turn_index or 0,
    )
    nodes = raw_trace["nodes"]
    message_path = path_to_node(nodes, final_call.node_index)[:-1]
    messages = _openai_messages(
        [copy.deepcopy(_message(nodes[index])) for index in message_path]
    )
    tools = _chat_tools(raw_trace.get("tools"))
    recorded_static_prefix = tuple(
        token_id
        for node_index in message_path[:-1]
        for token_id in _integer_tuple(
            nodes[node_index].get("token_ids"),
            f"trace.nodes[{node_index}].token_ids",
        )
    )
    if final_call.event_seed is None:
        raise ValueError("final call has no event seed")
    client = TokenInferenceClient(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=300.0,
    )
    canonical = client.render_chat(
        messages,
        tools,
        seed=final_call.event_seed,
        temperature=0.7,
        max_tokens=768,
    )
    recorded = final_call.prompt_token_ids
    common_suffix = _common_suffix_length(recorded, canonical)
    recorded_suffix_start = len(recorded) - common_suffix
    canonical_suffix_start = len(canonical) - common_suffix
    boundary = derive_lossless_render_boundary(
        recorded_prompt=recorded,
        recorded_static_prefix=recorded_static_prefix,
        canonical_render=canonical,
    )
    hybrid = (
        recorded_static_prefix
        + canonical[boundary.canonical_suffix_start_tokens :]
    )
    recorded_text = client.detokenize(recorded)
    canonical_text = client.detokenize(canonical)
    common_text_prefix = _common_text_prefix_length(recorded_text, canonical_text)
    common_text_suffix = _common_text_suffix_length(recorded_text, canonical_text)
    context_start = max(0, common_text_prefix - 160)
    recorded_context_end = min(len(recorded_text), common_text_prefix + 320)
    canonical_context_end = min(len(canonical_text), common_text_prefix + 320)
    payload: dict[str, object] = {
        "schema_version": 1,
        "source_sha256": source_sha256,
        "trace_id": final_call.trace_id,
        "model": args.model,
        "recorded_prompt_tokens": len(recorded),
        "canonical_render_tokens": len(canonical),
        "canonical_renderer_prompt_exact": canonical == recorded,
        "decoded_text_exact": canonical_text == recorded_text,
        "recorded_decoded_characters": len(recorded_text),
        "canonical_decoded_characters": len(canonical_text),
        "recorded_decoded_sha256": hashlib.sha256(
            recorded_text.encode("utf-8")
        ).hexdigest(),
        "canonical_decoded_sha256": hashlib.sha256(
            canonical_text.encode("utf-8")
        ).hexdigest(),
        "common_decoded_prefix_characters": common_text_prefix,
        "common_decoded_suffix_characters": common_text_suffix,
        "recorded_first_difference_context": recorded_text[
            context_start:recorded_context_end
        ],
        "canonical_first_difference_context": canonical_text[
            context_start:canonical_context_end
        ],
        "recorded_static_prefix_tokens": len(recorded_static_prefix),
        "recorded_suffix_start_tokens": recorded_suffix_start,
        "canonical_suffix_start_tokens": canonical_suffix_start,
        "exact_common_suffix_tokens": common_suffix,
        "static_boundary_exact": (
            recorded_suffix_start == len(recorded_static_prefix)
        ),
        "selected_canonical_suffix_start_tokens": (
            boundary.canonical_suffix_start_tokens
        ),
        "selected_exact_suffix_tokens": boundary.exact_common_suffix_tokens,
        "lossless_hybrid_exact": hybrid == recorded,
        "canonical_prompt_token_ids": list(canonical),
    }
    unsigned = canonical_json(payload)
    payload["report_sha256"] = hashlib.sha256(unsigned).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        key: value
        for key, value in payload.items()
        if key != "canonical_prompt_token_ids"
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
