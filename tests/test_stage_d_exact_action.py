from __future__ import annotations

import json
from dataclasses import fields

import pytest

from redco.analysis.stage_d_exact_action import (
    BehaviorAction,
    ExactActionKey,
    ExactActionMismatch,
    ResolvedSamplerConfig,
    categorical_probabilities,
    require_exact_reuse,
)
from redco.contracts import canonical_json
from redco.integrations.signed_subprocess import sign_payload


def _conformance(stack_hash: str = "a" * 64) -> bytes:
    return canonical_json(
        sign_payload(
            {
                "schema_version": 1,
                "analysis": "served-stack-categorical-logprob-conformance-v1",
                "passes": True,
                "logprob_semantics": "served_chosen_token_post_transform",
                "categorical_case_count": 3,
                "served_stack_sha256": stack_hash,
                "tool_call_termination_includes_all_generated_tokens": True,
                "eos_is_included_in_action_tokens_and_logprobs": True,
            }
        )
    )


def _request(**changes: object) -> dict[str, object]:
    request: dict[str, object] = {
        "model": "model@commit",
        "messages": [{"role": "user", "content": "q"}],
        "tools": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": None,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "logit_bias": {},
        "seed": 17,
        "max_tokens": 2,
        "stop": None,
        "n": 1,
        "best_of": None,
        "use_beam_search": False,
        "logprobs": True,
        "top_logprobs": 0,
        "ignore_eos": False,
        "min_tokens": 0,
        "extra_body": {"cache_salt": "exact"},
    }
    request.update(changes)
    return request


def _key(**request_changes: object) -> ExactActionKey:
    return ExactActionKey.build(
        checkpoint_id="model@commit",
        base_model_manifest=b"base manifest",
        adapter_manifest=b"adapter manifest",
        tokenizer_manifest=b"tokenizer manifest",
        renderer_manifest=b"renderer manifest",
        sampler_conformance_manifest=_conformance(),
        action_selection_policy="direct_single_sample",
        transport_retry_policy="fail_before_action_no_resample",
        request=_request(**request_changes),
        prompt_token_ids=(10, 11),
        render_prompt=lambda _: (10, 11),
    )


def _action(
    *,
    key: ExactActionKey | None = None,
    action_token_ids: tuple[int, ...] = (20, 2),
    behavior_logprobs: tuple[float, ...] = (-0.2, -0.1),
    message: dict[str, object] | None = None,
    finish_reason: str = "stop",
    termination_kind: str = "eos",
    eos_token_id: int | None = 2,
) -> BehaviorAction:
    resolved_message = message or {"role": "assistant", "content": "ok"}
    return BehaviorAction.build(
        key=key or _key(),
        action_token_ids=action_token_ids,
        behavior_logprobs=behavior_logprobs,
        raw_transport_message=resolved_message,
        finish_reason=finish_reason,
        prompt_tokens=2,
        completion_tokens=len(action_token_ids),
        termination_kind=termination_kind,  # type: ignore[arg-type]
        eos_token_id=eos_token_id,
        encode_action=lambda _request, _message: action_token_ids,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_id", "other"),
        ("base_model_manifest", b"other base"),
        ("adapter_manifest", None),
        ("tokenizer_manifest", b"other tokenizer"),
        ("renderer_manifest", b"other renderer"),
        ("sampler_conformance_manifest", _conformance("b" * 64)),
    ],
)
def test_exact_reuse_rejects_every_policy_identity_field(field: str, value: object) -> None:
    action = _action()
    kwargs = {
        "checkpoint_id": "model@commit",
        "base_model_manifest": b"base manifest",
        "adapter_manifest": b"adapter manifest",
        "tokenizer_manifest": b"tokenizer manifest",
        "renderer_manifest": b"renderer manifest",
        "sampler_conformance_manifest": _conformance(),
        "action_selection_policy": "direct_single_sample",
        "transport_retry_policy": "fail_before_action_no_resample",
        "request": _request(),
        "prompt_token_ids": (10, 11),
        "render_prompt": lambda _: (10, 11),
    }
    kwargs[field] = value
    if field == "checkpoint_id":
        kwargs["request"] = _request(model=value)
    observed = ExactActionKey.build(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(ExactActionMismatch):
        require_exact_reuse(action, observed)


def test_request_and_tool_schema_are_bound_and_unknown_fields_rejected() -> None:
    action = _action()
    changed = _key(messages=[{"role": "user", "content": "changed"}])
    with pytest.raises(ExactActionMismatch):
        require_exact_reuse(action, changed)
    with pytest.raises(ValueError, match="unknown"):
        _key(future_behavior_field="not allowlisted")
    with pytest.raises(ExactActionMismatch):
        require_exact_reuse(action, _key(tools=[{"type": "function"}]))


def test_request_must_render_to_prompt_tokens_on_build_and_reload() -> None:
    with pytest.raises(ExactActionMismatch, match="prompt tokens"):
        ExactActionKey.build(
            checkpoint_id="model@commit",
            base_model_manifest=b"base manifest",
            adapter_manifest=b"adapter manifest",
            tokenizer_manifest=b"tokenizer manifest",
            renderer_manifest=b"renderer manifest",
            sampler_conformance_manifest=_conformance(),
            action_selection_policy="direct_single_sample",
            transport_retry_policy="fail_before_action_no_resample",
            request=_request(),
            prompt_token_ids=(10, 11),
            render_prompt=lambda _: (10, 12),
        )
    with pytest.raises(ExactActionMismatch, match="prompt tokens"):
        BehaviorAction.from_bytes(
            _action().to_bytes(),
            encode_action=lambda _request, _message: (20, 2),
            render_prompt=lambda _: (10, 12),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"top_p": 0.9},
        {"top_k": 20},
        {"min_p": 0.1},
        {"repetition_penalty": 1.1},
        {"frequency_penalty": 0.1},
        {"presence_penalty": 0.1},
        {"logit_bias": {"1": 2.0}},
        {"n": 2},
        {"best_of": 2},
        {"use_beam_search": True},
        {"logprobs": False},
        {"top_logprobs": 2},
        {"tool_choice": "required"},
        {"ignore_eos": True},
        {"min_tokens": 1},
        {"stop": ["END"]},
        {"extra_body": {"guided_json": {}}},
    ],
)
def test_every_nontrivial_sampling_transform_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _key(**changes)


def test_missing_sampler_defaults_and_request_config_conflicts_are_rejected() -> None:
    request = _request()
    del request["top_k"]
    with pytest.raises(ValueError, match="missing"):
        ResolvedSamplerConfig.from_request(request)
    with pytest.raises(ValueError, match="model"):
        _key(model="other")
    with pytest.raises(ValueError, match="parallel"):
        _key(parallel_tool_calls=True)


def test_selection_retry_and_conformance_evidence_are_mandatory() -> None:
    base = {
        "checkpoint_id": "model@commit",
        "base_model_manifest": b"base manifest",
        "adapter_manifest": b"adapter manifest",
        "tokenizer_manifest": b"tokenizer manifest",
        "renderer_manifest": b"renderer manifest",
        "sampler_conformance_manifest": _conformance(),
        "action_selection_policy": "direct_single_sample",
        "transport_retry_policy": "fail_before_action_no_resample",
        "request": _request(),
        "prompt_token_ids": (10, 11),
        "render_prompt": lambda _: (10, 11),
    }
    for name, value in (
        ("action_selection_policy", "best_of"),
        ("transport_retry_policy", "retry_and_resample"),
        ("sampler_conformance_manifest", b""),
    ):
        with pytest.raises(ValueError):
            ExactActionKey.build(**{**base, name: value})  # type: ignore[arg-type]


def test_factory_normalizes_inputs_to_deeply_immutable_values() -> None:
    prompt = [10, 11]
    key = ExactActionKey.build(
        checkpoint_id="model@commit",
        base_model_manifest=b"base manifest",
        adapter_manifest=None,
        tokenizer_manifest=b"tokenizer manifest",
        renderer_manifest=b"renderer manifest",
        sampler_conformance_manifest=_conformance(),
        action_selection_policy="direct_single_sample",
        transport_retry_policy="fail_before_action_no_resample",
        request=_request(),
        prompt_token_ids=prompt,
        render_prompt=lambda _: (10, 11),
    )
    prompt[0] = 99
    action_tokens = [20, 2]
    action = BehaviorAction.build(
        key=key,
        action_token_ids=action_tokens,
        behavior_logprobs=[-0.2, -0.1],
        raw_transport_message={"role": "assistant", "content": "ok"},
        finish_reason="stop",
        prompt_tokens=2,
        completion_tokens=2,
        termination_kind="eos",
        eos_token_id=2,
        encode_action=lambda _request, _message: (20, 2),
    )
    action_tokens[0] = 99
    assert key.prompt_token_ids == (10, 11)
    assert action.action_token_ids == (20, 2)
    assert fields(ExactActionKey)[0].name == "schema_version"


@pytest.mark.parametrize("bad", [(0.1, -0.1), (True, -0.1), (-0.1,), (-0.1, float("nan"))])
def test_behavior_logprobs_must_be_token_aligned_finite_nonpositive_floats(
    bad: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match="logprobs"):
        _action(behavior_logprobs=bad)  # type: ignore[arg-type]


def test_action_roundtrip_is_mandatory_and_malformed_tool_action_is_retained() -> None:
    key = _key()
    with pytest.raises(ExactActionMismatch, match="round-trip"):
        BehaviorAction.build(
            key=key,
            action_token_ids=(20, 2),
            behavior_logprobs=(-0.2, -0.1),
            raw_transport_message={"role": "assistant", "content": "ok"},
            finish_reason="stop",
            prompt_tokens=2,
            completion_tokens=2,
            termination_kind="eos",
            eos_token_id=2,
            encode_action=lambda _request, _message: (99,),
        )
    malformed = _action(
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call", "type": "function"}],
        },
        finish_reason="tool_calls",
        termination_kind="tool_calls",
        eos_token_id=None,
    )
    assert malformed.parse_status == "malformed"
    assert malformed.parse_error
    assert (
        BehaviorAction.from_bytes(
            malformed.to_bytes(),
            encode_action=lambda _request, _message: (20, 2),
            render_prompt=lambda _: (10, 11),
        )
        == malformed
    )


def test_only_eos_included_max_tokens_and_tool_calls_are_authorized() -> None:
    assert _action().termination_kind == "eos"
    maxed = _action(finish_reason="length", termination_kind="max_tokens", eos_token_id=None)
    assert maxed.termination_kind == "max_tokens"
    tool_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "ipython", "arguments": '{"code":"1+1"}'},
            }
        ],
    }
    assert (
        _action(
            message=tool_message,
            finish_reason="tool_calls",
            termination_kind="tool_calls",
            eos_token_id=None,
        ).termination_kind
        == "tool_calls"
    )
    with pytest.raises(ValueError, match="not authorized"):
        _action(termination_kind="stop_sequence", eos_token_id=None)
    with pytest.raises(ValueError, match="max-token"):
        _action(
            key=_key(max_tokens=3),
            finish_reason="length",
            termination_kind="max_tokens",
            eos_token_id=None,
        )


def test_hashes_and_versioned_canonical_digest_bind_every_action_payload() -> None:
    action = _action()
    payload = action.to_payload()
    assert payload["schema_version"] == 1
    assert len(action.key.prompt_token_ids_sha256) == 64
    assert len(action.action_token_ids_sha256) == 64
    assert len(action.behavior_logprobs_sha256) == 64
    assert len(action.raw_transport_message_sha256) == 64
    assert len(action.digest) == 64
    assert _action(behavior_logprobs=(-0.3, -0.1)).digest != action.digest


def test_versioned_serialization_round_trips_and_rejects_unknown_or_corrupt_fields() -> None:
    action = _action()
    assert (
        BehaviorAction.from_bytes(
            action.to_bytes(),
            encode_action=lambda _request, _message: (20, 2),
            render_prompt=lambda _: (10, 11),
        )
        == action
    )
    payload = action.to_payload()
    payload["unknown"] = "rejected"
    envelope = {
        "schema_version": 1,
        "domain": "redco-stage-d-behavior-action-v1",
        "action": payload,
        "digest": action.digest,
    }
    with pytest.raises(ValueError, match="unknown"):
        BehaviorAction.from_bytes(
            canonical_json(envelope),
            encode_action=lambda _request, _message: (20, 2),
            render_prompt=lambda _: (10, 11),
        )
    corrupt = json.loads(action.to_bytes())
    corrupt["action"]["behavior_logprobs_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="disagree"):
        BehaviorAction.from_bytes(
            canonical_json(corrupt),
            encode_action=lambda _request, _message: (20, 2),
            render_prompt=lambda _: (10, 11),
        )


def test_usage_must_match_exact_prompt_and_action_tokens() -> None:
    with pytest.raises(ValueError, match="usage"):
        BehaviorAction.build(
            key=_key(),
            action_token_ids=(20, 2),
            behavior_logprobs=(-0.2, -0.1),
            raw_transport_message={"role": "assistant", "content": "ok"},
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=2,
            termination_kind="eos",
            eos_token_id=2,
            encode_action=lambda _request, _message: (20, 2),
        )


def test_categorical_math_matches_declared_temperature_distribution() -> None:
    probabilities = categorical_probabilities((0.0, 1.0, -0.5), temperature=0.7)
    assert probabilities == pytest.approx((0.17660744, 0.73693586, 0.08645670))
