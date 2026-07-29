from __future__ import annotations

from redco.analysis.stage_c4_renderer_alignment import (
    audit_renderer_alignment,
    prime_qwen3_pair,
)


def _encode(text: str) -> list[int]:
    pieces = {
        "<|im_start|>": [1000],
        "<|im_end|>": [1001],
        "<think>\n\n</think>\n\n": [1002, 1003, 1004, 1003],
        "<route>": [1005, 1006, 1007],
        "</route>": [1008, 1006, 1007],
        "alpha": [1010],
        "beta": [1011],
        "gamma": [1012],
        "delta": [1013],
    }
    result: list[int] = []
    offset = 0
    ordered = sorted(pieces, key=len, reverse=True)
    while offset < len(text):
        match = next(
            (piece for piece in ordered if text.startswith(piece, offset)),
            None,
        )
        if match is None:
            result.append(ord(text[offset]))
            offset += 1
        else:
            result.extend(pieces[match])
            offset += len(match)
    return result


def _fixture():
    dataset = []
    root_cases = []
    action_cases = []
    for route in ("alpha", "beta", "gamma", "delta"):
        user = f"root {route}"
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": f"<route>{route}</route>"},
        ]
        prefix, completion = prime_qwen3_pair(messages)
        root_cases.append(
            {
                "route": route,
                "prefix_token_ids": _encode(prefix),
                "completion_token_ids": _encode(completion),
            }
        )
        for _ in range(2):
            dataset.append(
                {
                    "messages": messages,
                    "example_kind": "root_format",
                    "route_label": route,
                    "digit_label": None,
                }
            )

        target_user = f"target {route}"
        target_prefix, _ = prime_qwen3_pair(
            [
                {"role": "user", "content": target_user},
                {"role": "assistant", "content": "0"},
            ]
        )
        action_cases.append(
            {
                "context_route": route,
                "prefix_token_ids": _encode(target_prefix),
                "action_token_ids": {str(value): ord(str(value)) for value in range(8)},
            }
        )
        for digit in range(8):
            dataset.append(
                {
                    "messages": [
                        {"role": "user", "content": target_user},
                        {"role": "assistant", "content": str(digit)},
                    ],
                    "example_kind": "target_format",
                    "route_label": route,
                    "digit_label": str(digit),
                }
            )
    return dataset, {"cases": root_cases}, {"cases": action_cases}


def test_prime_qwen3_alignment_matches_every_live_case():
    dataset, root_cases, action_cases = _fixture()
    result = audit_renderer_alignment(
        renderer_name="prime-qwen3",
        dataset=dataset,
        root_cases=root_cases,
        action_cases=action_cases,
        encode=_encode,
    )
    assert result["status"] == "passed"
    assert result["measurements"]["exact_prime_rows"] == 40
    assert result["measurements"]["legacy_qwen3_rows_matching_live"] == 0


def test_generic_qwen3_renderer_is_rejected_even_with_same_dataset():
    dataset, root_cases, action_cases = _fixture()
    result = audit_renderer_alignment(
        renderer_name="qwen3",
        dataset=dataset,
        root_cases=root_cases,
        action_cases=action_cases,
        encode=_encode,
    )
    assert result["status"] == "failed"
    assert result["checks"]["sft_renderer_is_prime_qwen3"] is False
