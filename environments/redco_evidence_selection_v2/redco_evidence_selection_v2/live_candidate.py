"""One-call candidate action sampler for the live Stage-D campaign."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from typing import Any, Literal

from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.contracts import canonical_json
from redco_evidence_selection_v2.scientific_campaign_driver import CandidateEngineResult


class _PostWitnessClient:
    def __init__(
        self,
        delegate: Any,
        *,
        expected_request: bytes,
        before_post: Callable[[bytes], None],
        after_post: Callable[[bytes], None],
    ) -> None:
        self._delegate = delegate
        self.base_url = delegate.base_url
        self._expected_request = expected_request
        self._before_post = before_post
        self._after_post = after_post
        self.response_content: bytes | None = None
        self.post_count = 0

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        return await self._delegate.get(*args, **kwargs)

    async def post(self, *args: Any, **kwargs: Any) -> Any:
        if self.post_count != 0:
            raise RuntimeError("candidate sampler attempted more than one POST")
        body = kwargs.get("body")
        if not isinstance(body, dict):
            raise ValueError("candidate POST lacks its prepared engine body")
        request = canonical_json(body)
        if request != self._expected_request:
            raise ValueError("candidate POST differs from the exact prepared request")
        self._before_post(request)
        self.post_count = 1
        response = await self._delegate.post(*args, **kwargs)
        content = getattr(response, "content", None)
        if type(content) is not bytes or not content:
            raise ValueError("candidate engine response lacks immutable bytes")
        self._after_post(content)
        self.response_content = content
        return response


class LiveVLLMCandidateEngine:
    """Resample one frozen target prompt through the pinned Renderer/vLLM stack."""

    def __init__(
        self,
        *,
        client: Any,
        eos_token_id: int,
    ) -> None:
        from verifiers.v1.clients.train import TrainClient

        if not isinstance(client, TrainClient):
            raise TypeError("live candidate engine requires the pinned TrainClient")
        if getattr(client.openai, "max_retries", None) != 0:
            raise ValueError("live candidate engine forbids transport retries")
        if type(eos_token_id) is not int or eos_token_id < 0:
            raise ValueError("live candidate engine requires a nonnegative EOS token")
        self._client = client
        self._eos_token_id = eos_token_id

    async def __call__(
        self,
        *,
        reference_key: ExactActionKey,
        action_seed: int,
        before_post: Callable[[bytes], None],
        after_post: Callable[[bytes], None],
    ) -> CandidateEngineResult:
        from renderers.client import generate
        from verifiers.v1.clients.train import response_from_generate, serialize_completion

        salt = "redco-stage-d-candidate-" + hashlib.sha256(
            canonical_json(
                {
                    "domain": "redco-stage-d-candidate-cache-salt-v1",
                    "reference_key": reference_key.digest,
                    "seed": action_seed,
                }
            )
        ).hexdigest()[:32]
        key = ExactActionKey.resample_prepared(
            reference_key,
            seed=action_seed,
            cache_salt=salt,
        )
        if key.prepared_engine_request is None:
            raise ValueError("candidate key lacks its prepared engine request")
        engine = json.loads(key.prepared_engine_request)
        if not isinstance(engine, dict) or not isinstance(engine.get("sampling_params"), dict):
            raise ValueError("candidate engine request is malformed")
        renderer = self._client._renderer_pool(key.checkpoint_id)
        witness = _PostWitnessClient(
            self._client.openai,
            expected_request=key.prepared_engine_request,
            before_post=before_post,
            after_post=after_post,
        )
        generate_cache_salt = salt if "cache_salt" in engine else None
        result = await generate(
            client=witness,  # type: ignore[arg-type]
            renderer=renderer,
            messages=[],
            model=key.checkpoint_id,
            prompt_ids=list(key.prompt_token_ids),
            sampling_params=dict(engine["sampling_params"]),
            cache_salt=generate_cache_salt,
        )
        if witness.post_count != 1 or witness.response_content is None:
            raise RuntimeError("candidate sampler did not complete exactly one POST")
        response = response_from_generate(result, key.checkpoint_id)
        response.raw = serialize_completion(response, key.checkpoint_id)
        raw = response.raw
        choices = raw.get("choices") if isinstance(raw, dict) else None
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        if not isinstance(message, Mapping):
            raise ValueError("candidate response lacks its typed assistant message")
        finish_reason = response.finish_reason
        if finish_reason == "tool_calls":
            termination_kind: Literal["eos", "max_tokens", "tool_calls"] = (
                "tool_calls"
            )
            eos_token_id = None
        elif finish_reason == "length":
            termination_kind = "max_tokens"
            eos_token_id = None
        elif finish_reason == "stop":
            termination_kind = "eos"
            eos_token_id = self._eos_token_id
        else:
            raise ValueError("candidate response has an unsupported finish reason")
        request_id = result.get("request_id")
        if (
            not isinstance(request_id, str)
            or not request_id
            or len(request_id) > 512
            or not request_id.isprintable()
        ):
            raise ValueError("candidate response lacks a valid request_id")
        raw_logprobs = result.get("completion_logprobs")
        if (
            not isinstance(raw_logprobs, list)
            or any(type(value) is not float or not math.isfinite(value) for value in raw_logprobs)
        ):
            raise ValueError("candidate completion logprobs must be finite JSON floats")
        action = BehaviorAction.build(
            key=key,
            action_token_ids=tuple(result["completion_ids"]),
            behavior_logprobs=tuple(raw_logprobs),
            raw_transport_message=dict(message),
            finish_reason=finish_reason,
            prompt_tokens=len(result["prompt_ids"]),
            completion_tokens=len(result["completion_ids"]),
            termination_kind=termination_kind,
            eos_token_id=eos_token_id,
            validate_action=lambda request, assistant, action_ids: (
                self._client.validate_assistant_action(
                    request,
                    assistant,
                    model=key.checkpoint_id,
                    action_token_ids=action_ids,
                )
            ),
            request_id=request_id,
        )
        return CandidateEngineResult(action, witness.response_content)


__all__ = ["LiveVLLMCandidateEngine"]
