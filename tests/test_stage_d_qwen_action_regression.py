import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

QWEN_FIXTURE = Path(__file__).parent / "fixtures" / "stage_d_qwen_tool_call_310.json"
QWEN_FIXTURE_SHA256 = "16e1ce9493befe768539adc9057c765843c619bba1b16481ca5f5cbd080a7f54"


def _fixture() -> dict:
    return json.loads(QWEN_FIXTURE.read_bytes())


def test_qwen_fixture_has_reviewed_bytes_and_token_counts() -> None:
    fixture_bytes = QWEN_FIXTURE.read_bytes()
    fixture = json.loads(fixture_bytes)

    assert hashlib.sha256(fixture_bytes).hexdigest() == QWEN_FIXTURE_SHA256
    assert len(fixture["action_token_ids"]) == 310
    assert fixture["typed_rerender_sampled_token_count"] == 314


def test_qwen_tool_action_validates_from_310_sampled_tokens() -> None:
    pytest.importorskip("verifiers.v1")
    from verifiers.v1.clients.train import TrainClient, tool_to_wire
    from verifiers.v1.dialects import parse_tools

    fixture = _fixture()
    client = TrainClient(
        SimpleNamespace(),
        pool_size=1,
        renderer_model_name=fixture["model"],
    )
    if not hasattr(client, "validate_assistant_action"):
        pytest.skip("requires the pinned prepared-observer Verifiers patch")

    action_ids = tuple(fixture["action_token_ids"])
    request = {"messages": fixture["messages"], "tools": fixture["tools"]}
    renderer = client._renderer_pool(fixture["model"])
    client.validate_assistant_action(
        request,
        fixture["typed_message"],
        model=fixture["model"],
        action_token_ids=action_ids,
    )

    tools = parse_tools(fixture["tools"])
    wire_tools = [tool_to_wire(tool) for tool in tools]
    prompt_ids = renderer.render_ids(
        fixture["messages"],
        tools=wire_tools,
        add_generation_prompt=True,
    )
    rerendered = renderer.render(
        [*fixture["messages"], fixture["typed_message"]],
        tools=wire_tools,
        add_generation_prompt=False,
    )
    rerendered_count = sum(rerendered.sampled_mask[len(prompt_ids) :])
    assert len(action_ids) == 310
    assert rerendered_count == fixture["typed_rerender_sampled_token_count"] == 314


def test_qwen_tool_action_replays_through_parser_without_post() -> None:
    pytest.importorskip("verifiers.v1")
    from renderers.client import PreparedGenerateReturn, generate
    from verifiers.v1.clients.train import (
        TrainClient,
        message_to_wire,
        response_from_generate,
        tool_to_wire,
    )
    from verifiers.v1.dialects import parse_tools

    fixture = _fixture()
    train_client = TrainClient(
        SimpleNamespace(),
        pool_size=1,
        renderer_model_name=fixture["model"],
    )
    renderer = train_client._renderer_pool(fixture["model"])
    wire_tools = [tool_to_wire(tool) for tool in parse_tools(fixture["tools"])]
    prompt_ids = renderer.render_ids(
        fixture["messages"],
        tools=wire_tools,
        add_generation_prompt=True,
    )
    response_content = json.dumps(
        {
            "request_id": "frozen-qwen-tool-call",
            "choices": [
                {
                    "token_ids": fixture["action_token_ids"],
                    "logprobs": {
                        "content": [
                            {"logprob": -0.1} for _ in fixture["action_token_ids"]
                        ]
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        separators=(",", ":"),
    ).encode()

    class ReturningObserver:
        async def before_forward(self, _prepared):
            return PreparedGenerateReturn("replay-ticket", response_content)

        async def after_raw_response(self, _ticket, _response_content):
            return None

        async def after_response(self, _ticket, _response):
            return None

        async def on_abort(self, _ticket, _phase, _error):
            return None

    post = AsyncMock()
    engine_client = SimpleNamespace(base_url="http://engine/v1", post=post)
    result = asyncio.run(
        generate(
            client=engine_client,
            renderer=renderer,
            messages=fixture["messages"],
            model=fixture["model"],
            tools=wire_tools,
            sampling_params={"temperature": 0.7, "max_tokens": 512},
            prompt_ids=prompt_ids,
            max_prompt_len=4096,
            effective_request={
                "model": fixture["model"],
                "messages": fixture["messages"],
                "tools": fixture["tools"],
            },
            observer_context={"trace_id": "frozen-replay"},
            observer=ReturningObserver(),
        )
    )
    result.pop("_prepared_observation_ticket")
    typed = response_from_generate(result, fixture["model"])

    post.assert_not_awaited()
    assert message_to_wire(typed.message) == fixture["typed_message"]
    train_client.validate_assistant_action(
        {"messages": fixture["messages"], "tools": fixture["tools"]},
        fixture["typed_message"],
        model=fixture["model"],
        action_token_ids=fixture["action_token_ids"],
    )


def test_parsed_tool_call_ids_keep_original_attempt_indices() -> None:
    pytest.importorskip("verifiers.v1")
    from verifiers.v1.clients.train import _assistant_message_from_parsed

    attempts = [
        SimpleNamespace(id=None, name=None, arguments={}),
        SimpleNamespace(id=None, name="ipython", arguments={"code": "1 + 1"}),
    ]

    message = _assistant_message_from_parsed(None, None, attempts)

    assert message.tool_calls is not None
    assert [call.id for call in message.tool_calls] == ["call_1"]
