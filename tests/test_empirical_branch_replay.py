from __future__ import annotations

import pytest

from redco.analysis.empirical_branch_replay import (
    _chat_tools,
    _distinct_candidate_fractions,
    _openai_messages,
    _request_json,
    build_replay_indices,
    derive_branch_group_seeds,
    derive_lossless_render_boundary,
    execute_cached_arm,
    replace_unique,
    splice_lossless_rendered_suffix,
)
from redco.env.policy_cache import (
    CachedPolicyAction,
    PolicyActionCache,
    PolicyCallKey,
)
from redco.env.replay import ReplayMode
from redco.integrations.verifiers_trace import RecordedPolicyCall


def _call(
    index: int,
    *,
    prompt: tuple[int, ...],
    action: tuple[int, ...],
    seed: int,
) -> RecordedPolicyCall:
    return RecordedPolicyCall(
        trace_id="trace",
        call_index=index,
        node_index=index,
        component_root_node=0,
        prompt_token_ids=prompt,
        action_token_ids=action,
        checkpoint_id="model",
        decoding_config_hash="decode",
        event_seed=seed,
        prompt_tokens_reported=len(prompt),
        completion_tokens_reported=len(action),
        cost_reported=None,
        wall_seconds=0.0,
        agent_depth=0,
        session_id="root",
        turn_index=index,
        call_kind="policy",
        parent_session_id=None,
        parent_turn_index=None,
    )


def test_request_json_preserves_nested_schema_order() -> None:
    payload: dict[str, object] = {
        "tools": [
            {
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                }
            }
        ]
    }

    assert _request_json(payload) == (
        b'{"tools":[{"parameters":{"type":"object","properties":'
        b'{"code":{"type":"string"}},"required":["code"]}}]}'
    )


def test_distinct_candidate_fractions_are_per_target() -> None:
    assert _distinct_candidate_fractions(
        {
            "target-b": {"one"},
            "target-a": {"one", "two"},
        },
        alternatives_per_target=4,
    ) == {
        "target-a": 0.5,
        "target-b": 0.25,
    }

    with pytest.raises(ValueError, match="must be positive"):
        _distinct_candidate_fractions(
            {"target": set()},
            alternatives_per_target=0,
        )


def test_replay_indices_isolate_descendants() -> None:
    full, sliced = build_replay_indices(
        target_call_index=1,
        target_node_id="policy-1",
        policy_node_ids_by_call={
            0: "policy-0",
            1: "policy-1",
            2: "policy-2",
            3: "policy-3",
            4: "policy-4",
        },
        descendants=frozenset({"message-return", "policy-4"}),
    )

    assert full == (2, 3, 4)
    assert sliced == (4,)


def test_branch_group_uses_one_crn_seed_and_unique_action_seeds() -> None:
    continuation, actions = derive_branch_group_seeds(
        master_seed="master",
        rollout_id="rollout",
        target_node_id="target",
        final_turn_index=1,
        alternatives=3,
    )
    assert continuation > 0
    assert len(actions) == len(set(actions)) == 3
    assert derive_branch_group_seeds(
        master_seed="master",
        rollout_id="rollout",
        target_node_id="target",
        final_turn_index=1,
        alternatives=3,
    ) == (continuation, actions)


def test_cached_arms_reach_the_same_terminal_action() -> None:
    calls = {
        2: _call(2, prompt=(2,), action=(20,), seed=2),
        3: _call(3, prompt=(3,), action=(30,), seed=3),
    }
    branch_prompt = (3, 9)
    branch_action = (31, 32)
    branch_seed = 99
    cache = PolicyActionCache()
    for call in calls.values():
        assert call.event_seed is not None
        cache.record(
            CachedPolicyAction(
                PolicyCallKey.from_call(
                    call.prompt_token_ids,
                    checkpoint_id=call.checkpoint_id,
                    decoding_config_hash=call.decoding_config_hash,
                    event_seed=call.event_seed,
                ),
                call.action_token_ids,
            )
        )
    cache.record(
        CachedPolicyAction(
            PolicyCallKey.from_call(
                branch_prompt,
                checkpoint_id="model",
                decoding_config_hash="decode",
                event_seed=branch_seed,
            ),
            branch_action,
        )
    )

    full = execute_cached_arm(
        mode=ReplayMode.FULL_SUFFIX,
        calls_by_index=calls,
        visited_call_indices=(2, 3),
        final_call_index=3,
        branch_final_prompt=branch_prompt,
        branch_final_seed=branch_seed,
        branch_final_decoding_config_hash="decode",
        branch_final_action=branch_action,
        cache=cache.fork(),
        reward=1.0,
    )
    sliced = execute_cached_arm(
        mode=ReplayMode.SLICED,
        calls_by_index=calls,
        visited_call_indices=(3,),
        final_call_index=3,
        branch_final_prompt=branch_prompt,
        branch_final_seed=branch_seed,
        branch_final_decoding_config_hash="decode",
        branch_final_action=branch_action,
        cache=cache.fork(),
        reward=1.0,
    )

    assert full.terminal_action_sha256 == sliced.terminal_action_sha256
    assert full.reward == sliced.reward == 1.0
    assert full.exact_key_reused_call_indices == (2, 3)
    assert sliced.exact_key_reused_call_indices == (3,)
    assert full.actual_cost.generated_tokens == 0
    assert sliced.actual_cost.generated_tokens == 0


def test_unique_replacement_rejects_ambiguous_input() -> None:
    assert replace_unique("before old after", "old", "new") == "before new after"
    with pytest.raises(ValueError, match="exactly once"):
        replace_unique("old old", "old", "new")
    with pytest.raises(ValueError, match="non-empty"):
        replace_unique("text", "", "new")


def test_openai_messages_normalizes_verifiers_tool_calls_without_mutation() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_0",
                    "name": "ipython",
                    "arguments": '{"code":"print(1)"}',
                }
            ],
        }
    ]

    normalized = _openai_messages(messages)

    assert normalized == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "ipython",
                        "arguments": '{"code":"print(1)"}',
                    },
                }
            ],
        }
    ]
    assert messages == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_0",
                    "name": "ipython",
                    "arguments": '{"code":"print(1)"}',
                }
            ],
        }
    ]


def test_chat_tools_wraps_each_raw_function_definition() -> None:
    raw_tools = [
        {
            "name": "ipython",
            "description": "Execute code",
            "parameters": {"type": "object"},
        }
    ]

    normalized = _chat_tools(raw_tools)

    assert normalized == [
        {
            "type": "function",
            "function": {
                "name": "ipython",
                "description": "Execute code",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert raw_tools[0]["name"] == "ipython"


def test_chat_tools_preserves_already_wrapped_openai_tools() -> None:
    wrapped = {
        "type": "function",
        "function": {
            "name": "ipython",
            "parameters": {"type": "object"},
        },
    }

    assert _chat_tools([wrapped]) == [wrapped]


def test_lossless_render_boundary_reconstructs_noncanonical_history() -> None:
    recorded_static_prefix = (1, 2)
    recorded_prompt = (1, 2, 8, 9)
    canonical_original = (1, 3, 4, 8, 9)

    boundary = derive_lossless_render_boundary(
        recorded_prompt=recorded_prompt,
        recorded_static_prefix=recorded_static_prefix,
        canonical_render=canonical_original,
    )
    branch = splice_lossless_rendered_suffix(
        recorded_static_prefix=recorded_static_prefix,
        canonical_original=canonical_original,
        canonical_branch=(1, 3, 4, 7, 6),
        boundary=boundary,
    )

    assert boundary.recorded_static_prefix_tokens == 2
    assert boundary.canonical_suffix_start_tokens == 3
    assert boundary.exact_common_suffix_tokens == 2
    assert branch == (1, 2, 7, 6)


def test_lossless_boundary_does_not_overextend_matching_suffix() -> None:
    boundary = derive_lossless_render_boundary(
        recorded_prompt=(1, 2, 3, 4, 8, 9),
        recorded_static_prefix=(1, 2, 3),
        canonical_render=(7, 2, 3, 4, 8, 9),
    )

    assert boundary.canonical_suffix_start_tokens == 3
    assert boundary.exact_common_suffix_tokens == 3


def test_lossless_render_splice_rejects_changed_history() -> None:
    boundary = derive_lossless_render_boundary(
        recorded_prompt=(1, 2, 8, 9),
        recorded_static_prefix=(1, 2),
        canonical_render=(1, 3, 4, 8, 9),
    )

    with pytest.raises(ValueError, match="before the affected suffix"):
        splice_lossless_rendered_suffix(
            recorded_static_prefix=(1, 2),
            canonical_original=(1, 3, 4, 8, 9),
            canonical_branch=(1, 7, 4, 8, 9),
            boundary=boundary,
        )
