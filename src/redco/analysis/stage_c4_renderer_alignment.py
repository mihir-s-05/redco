"""Audit Stage-C4 SFT supervision against the exact live policy prefixes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

PRIME_QWEN3_RENDERER = "prime-qwen3"
LEGACY_QWEN3_THINK_PREFIX = "<think>\n\n</think>\n\n"


def _signed(payload: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        **payload,
        "signed_payload_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def prime_qwen3_pair(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """Return the exact prompt/completion text for a two-turn Prime-Qwen3 sample."""
    if len(messages) != 2:
        raise ValueError("alignment audit requires exactly two messages")
    user, assistant = messages
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError("alignment audit requires one user then one assistant message")
    user_content = user.get("content")
    assistant_content = assistant.get("content")
    if not isinstance(user_content, str) or not isinstance(assistant_content, str):
        raise ValueError("alignment audit requires string message content")
    prefix = (
        "<|im_start|>user\n"
        + user_content
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    completion = assistant_content + "<|im_end|>"
    return prefix, completion


def audit_renderer_alignment(
    *,
    renderer_name: str,
    dataset: list[dict[str, Any]],
    root_cases: dict[str, Any],
    action_cases: dict[str, Any],
    encode: Callable[[str], list[int]],
) -> dict[str, Any]:
    """Compare every supervised target with its frozen live scoring context."""
    root_by_route = {str(case["route"]): case for case in root_cases["cases"]}
    action_by_route = {
        str(case["context_route"]): case for case in action_cases["cases"]
    }
    row_reports: list[dict[str, Any]] = []
    root_trainable_tokens = 0
    target_trainable_tokens = 0
    exact_prime_rows = 0
    legacy_qwen3_rows_matching_live = 0

    for index, row in enumerate(dataset):
        prefix_text, completion_text = prime_qwen3_pair(row["messages"])
        prefix_ids = encode(prefix_text)
        completion_ids = encode(completion_text)
        route = str(row["route_label"])
        kind = str(row["example_kind"])
        if kind == "root_format":
            case = root_by_route[route]
            prefix_exact = prefix_ids == [int(value) for value in case["prefix_token_ids"]]
            target_exact = completion_ids == [
                int(value) for value in case["completion_token_ids"]
            ]
            root_trainable_tokens += len(completion_ids)
        elif kind == "target_format":
            case = action_by_route[route]
            action = str(row["digit_label"])
            expected_action = int(case["action_token_ids"][action])
            prefix_exact = prefix_ids == [int(value) for value in case["prefix_token_ids"]]
            target_exact = bool(completion_ids) and completion_ids[0] == expected_action
            target_trainable_tokens += len(completion_ids)
        else:
            raise ValueError(f"unsupported example_kind at row {index}: {kind}")

        exact_prime_rows += int(prefix_exact and target_exact)
        legacy_completion_ids = encode(LEGACY_QWEN3_THINK_PREFIX + completion_text)
        legacy_matches = legacy_completion_ids == completion_ids
        legacy_qwen3_rows_matching_live += int(legacy_matches)
        row_reports.append(
            {
                "index": index,
                "kind": kind,
                "route": route,
                "prefix_exact": prefix_exact,
                "target_exact": target_exact,
                "prime_trainable_tokens": len(completion_ids),
                "legacy_qwen3_trainable_tokens": len(legacy_completion_ids),
                "legacy_qwen3_matches_live_target": legacy_matches,
            }
        )

    checks = {
        "sft_renderer_is_prime_qwen3": renderer_name == PRIME_QWEN3_RENDERER,
        "all_40_rows_audited": len(row_reports) == 40,
        "every_prime_qwen3_prefix_and_target_matches_live": (
            exact_prime_rows == len(row_reports)
        ),
        "legacy_qwen3_mismatch_reproduced_on_every_row": (
            legacy_qwen3_rows_matching_live == 0
        ),
        "root_and_target_token_weight_are_balanced": (
            root_trainable_tokens == target_trainable_tokens
        ),
    }
    legacy_root_tokens = sum(
        row["legacy_qwen3_trainable_tokens"]
        for row in row_reports
        if row["kind"] == "root_format"
    )
    legacy_target_tokens = sum(
        row["legacy_qwen3_trainable_tokens"]
        for row in row_reports
        if row["kind"] == "target_format"
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c4-renderer-alignment",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "renderer_name": renderer_name,
        "rows": row_reports,
        "measurements": {
            "rows": len(row_reports),
            "exact_prime_rows": exact_prime_rows,
            "legacy_qwen3_rows_matching_live": legacy_qwen3_rows_matching_live,
            "prime_root_trainable_tokens": root_trainable_tokens,
            "prime_target_trainable_tokens": target_trainable_tokens,
            "legacy_qwen3_root_trainable_tokens": legacy_root_tokens,
            "legacy_qwen3_target_trainable_tokens": legacy_target_tokens,
        },
        "finding": (
            "Prime-Qwen3 supervises the exact frozen live completion. The prior "
            "Qwen3 renderer prepended a trainable empty thinking block, changing "
            "both the conditional prefix and root/target token weighting."
        ),
    }
    return _signed(payload)
