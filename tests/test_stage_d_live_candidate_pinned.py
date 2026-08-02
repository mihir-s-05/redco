from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("verifiers.v1")

ENV_ROOT = Path(__file__).parents[1] / "environments" / "redco_evidence_selection_v2"
sys.path.insert(0, str(ENV_ROOT))

from redco_evidence_selection_v2.live_candidate import (  # noqa: E402
    LiveVLLMCandidateEngine,
)
from test_stage_d_exact_action import _prepared_key  # noqa: E402
from verifiers.v1.clients.train import TrainClient  # noqa: E402

from redco.analysis.stage_d_exact_action import ExactActionKey  # noqa: E402


@pytest.mark.parametrize("nested_cache_salt", [False, True])
def test_live_candidate_marks_exact_post_and_returns_typed_action(
    nested_cache_salt: bool,
) -> None:
    reference = _prepared_key()
    if nested_cache_salt:
        request = json.loads(reference.request)
        engine = json.loads(reference.prepared_engine_request or b"null")
        engine["sampling_params"]["cache_salt"] = engine.pop("cache_salt")
        reference = ExactActionKey.build_prepared(
            checkpoint_id=reference.checkpoint_id,
            base_model_manifest=b"base manifest",
            adapter_manifest=b"adapter manifest",
            tokenizer_manifest=b"tokenizer manifest",
            renderer_manifest=b"renderer manifest",
            sampler_conformance_manifest=reference.sampler_conformance_manifest,
            action_selection_policy="direct_single_sample",
            transport_retry_policy="fail_before_action_no_resample",
            request=request,
            prompt_token_ids=reference.prompt_token_ids,
            prepared_engine_request=engine,
        )
    provider_content = (
        b'{"choices":[{"finish_reason":"stop","logprobs":{"content":['
        b'{"logprob":-0.2},{"logprob":-0.1}]},"token_ids":[20,2]}],'
        b'"request_id":"candidate-request"}'
    )
    openai = SimpleNamespace(
        base_url="http://engine/v1",
        max_retries=0,
        post=AsyncMock(return_value=SimpleNamespace(content=provider_content)),
        get=AsyncMock(
            return_value={"data": [{"id": "model@commit", "max_model_len": 4096}]}
        ),
    )
    client = TrainClient(openai)
    renderer = SimpleNamespace(
        get_stop_token_ids=lambda: [2],
        parse_response=lambda _ids, **_kwargs: SimpleNamespace(
            content="ok",
            reasoning_content=None,
            tool_calls=[],
        ),
    )
    client._pool = renderer
    client.encode_assistant_action = (  # type: ignore[method-assign]
        lambda _request, _message, **_kwargs: (20, 2)
    )
    posted: list[bytes] = []
    responses: list[bytes] = []

    result = asyncio.run(
        LiveVLLMCandidateEngine(client=client, eos_token_id=2)(
            reference_key=reference,
            action_seed=9917,
            before_post=posted.append,
            after_post=responses.append,
        )
    )

    assert posted == [result.action.key.prepared_engine_request]
    assert responses == [provider_content]
    assert result.response_evidence == provider_content
    assert result.action.request_id == "candidate-request"
    assert result.action.action_token_ids == (20, 2)
    assert result.action.behavior_logprobs == (-0.2, -0.1)
    assert result.action.key.sampler.seed == 9917
    assert openai.post.await_count == 1


def test_live_candidate_rejects_body_drift_before_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _prepared_key()
    openai = SimpleNamespace(
        base_url="http://engine/v1",
        max_retries=0,
        post=AsyncMock(),
        get=AsyncMock(),
    )
    client = TrainClient(openai)
    client._pool = object()
    observed: list[bytes] = []

    async def drifting_generate(*, client, **kwargs):
        del kwargs
        await client.post(
            "/inference/v1/generate",
            body={"model": "drifted"},
        )

    import renderers.client as renderer_client

    monkeypatch.setattr(renderer_client, "generate", drifting_generate)
    with pytest.raises(ValueError, match="exact prepared request"):
        asyncio.run(
            LiveVLLMCandidateEngine(client=client, eos_token_id=2)(
                reference_key=reference,
                action_seed=9917,
                before_post=observed.append,
                after_post=lambda _response: None,
            )
        )

    assert observed == []
    assert openai.post.await_count == 0


def test_live_candidate_rejects_transport_retries_before_post() -> None:
    openai = SimpleNamespace(
        base_url="http://engine/v1",
        max_retries=1,
        post=AsyncMock(),
        get=AsyncMock(),
    )
    client = TrainClient(openai)

    with pytest.raises(ValueError, match="forbids transport retries"):
        LiveVLLMCandidateEngine(client=client, eos_token_id=2)

    assert openai.post.await_count == 0
