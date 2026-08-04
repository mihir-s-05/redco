from __future__ import annotations

import hashlib
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
    resolve_behavior_termination,
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


@pytest.mark.parametrize("fixture_marker", [True, False, 1, "true"])
def test_fixture_only_conformance_cannot_create_live_behavior_action(
    fixture_marker: object,
) -> None:
    fixture = canonical_json(
        sign_payload(
            {
                "schema_version": 1,
                "analysis": "served-stack-categorical-logprob-conformance-v1",
                "passes": True,
                "logprob_semantics": "served_chosen_token_post_transform",
                "categorical_case_count": 1,
                "served_stack_sha256": "a" * 64,
                "tool_call_termination_includes_all_generated_tokens": True,
                "eos_is_included_in_action_tokens_and_logprobs": True,
                "fixture_only": fixture_marker,
            }
        )
    )
    with pytest.raises(ValueError, match="did not pass"):
        ExactActionKey.build(
            checkpoint_id="model@commit",
            base_model_manifest=b"base",
            adapter_manifest=None,
            tokenizer_manifest=b"tokenizer",
            renderer_manifest=b"renderer",
            sampler_conformance_manifest=fixture,
            action_selection_policy="direct_single_sample",
            transport_retry_policy="fail_before_action_no_resample",
            request=_request(),
            prompt_token_ids=(10, 11),
            render_prompt=lambda _: (10, 11),
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


def _prepared_key(**engine_changes: object) -> ExactActionKey:
    request_extra_body = engine_changes.pop(
        "request_extra_body", {"cache_salt": "exact"}
    )
    engine: dict[str, object] = {
        "model": "model@commit",
        "token_ids": [10, 11],
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 1.0,
            "seed": 17,
            "max_tokens": 2,
            "stop_token_ids": [2],
            "logprobs": 1,
            "skip_special_tokens": False,
            "parallel_tool_calls": False,
        },
        "cache_salt": "exact",
    }
    engine.update(engine_changes)
    return ExactActionKey.build_prepared(
        checkpoint_id="model@commit",
        base_model_manifest=b"base manifest",
        adapter_manifest=b"adapter manifest",
        tokenizer_manifest=b"tokenizer manifest",
        renderer_manifest=b"renderer manifest",
        sampler_conformance_manifest=_conformance(),
        action_selection_policy="direct_single_sample",
        transport_retry_policy="fail_before_action_no_resample",
        request=_request(extra_body=request_extra_body),
        prompt_token_ids=(10, 11),
        prepared_engine_request=engine,
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


def test_prepared_engine_request_is_bound_and_round_trips() -> None:
    key = _prepared_key()
    assert key.schema_version == 2
    assert key.prepared_engine_request is not None
    assert key.prepared_engine_request_sha256 is not None
    payload = key.to_payload()
    loaded = ExactActionKey.from_payload(payload, render_prompt=lambda _: (99,))
    assert loaded == key
    payload["prepared_engine_request"]["token_ids"] = [99]  # type: ignore[index]
    with pytest.raises(ExactActionMismatch, match="prompt differs"):
        ExactActionKey.from_payload(payload, render_prompt=lambda _: (10, 11))


def test_prepared_action_requires_token_semantic_validation() -> None:
    key = _prepared_key()
    observed: list[tuple[int, ...]] = []
    action = BehaviorAction.build(
        key=key,
        action_token_ids=(20, 2),
        behavior_logprobs=(-0.2, -0.1),
        raw_transport_message={"role": "assistant", "content": "ok"},
        finish_reason="stop",
        prompt_tokens=2,
        completion_tokens=2,
        termination_kind="eos",
        eos_token_id=2,
        validate_action=lambda _request, _message, tokens: observed.append(tuple(tokens)),
    )
    assert observed == [(20, 2)]
    assert (
        BehaviorAction.from_bytes(
            action.to_bytes(),
            validate_action=lambda _request, _message, _tokens: None,
            render_prompt=lambda _: (99,),
        )
        == action
    )
    with pytest.raises(ValueError, match="token-semantic validation"):
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
            encode_action=lambda _request, _message: (20, 2),
        )


def test_prepared_resampling_changes_only_seed_salt_and_derived_hashes() -> None:
    reference = _prepared_key(priority=7)
    candidate = ExactActionKey.resample_prepared(
        reference,
        seed=9917,
        cache_salt="candidate-9917",
    )
    request = json.loads(candidate.request)
    sampler = json.loads(candidate.sampler_config)
    engine = json.loads(candidate.prepared_engine_request or b"null")

    assert request["seed"] == 9917
    assert request["extra_body"] == {"cache_salt": "candidate-9917"}
    assert sampler["seed"] == 9917
    assert engine["sampling_params"]["seed"] == 9917
    assert engine["cache_salt"] == "candidate-9917"
    assert engine["priority"] == 7
    assert candidate.prompt_token_ids == reference.prompt_token_ids
    assert candidate.checkpoint_id == reference.checkpoint_id
    assert candidate.tool_schema_sha256 == reference.tool_schema_sha256
    assert candidate.sampler_config_sha256 != reference.sampler_config_sha256
    assert candidate.request_sha256 != reference.request_sha256
    assert candidate.prepared_engine_request_sha256 != (
        reference.prepared_engine_request_sha256
    )


def test_prepared_resampling_rejects_legacy_keys_and_missing_salt() -> None:
    with pytest.raises(ValueError, match="prepared"):
        ExactActionKey.resample_prepared(_key(), seed=1, cache_salt="candidate")
    with pytest.raises(ValueError, match="nonempty"):
        ExactActionKey.resample_prepared(_prepared_key(), seed=1, cache_salt="")


def test_prepared_engine_allows_only_disabled_parallel_tool_calls() -> None:
    assert _prepared_key().prepared_engine_request is not None
    sampling = {
        "temperature": 0.7,
        "top_p": 1.0,
        "seed": 17,
        "max_tokens": 2,
        "stop_token_ids": [2],
        "logprobs": 1,
        "skip_special_tokens": False,
        "parallel_tool_calls": True,
    }
    with pytest.raises(ValueError, match="unauthorized transform"):
        _prepared_key(sampling_params=sampling)


def test_prepared_engine_binds_optional_routed_expert_prompt_boundary() -> None:
    sampling = {
        "temperature": 0.7,
        "top_p": 1.0,
        "seed": 17,
        "max_tokens": 2,
        "stop_token_ids": [2],
        "logprobs": 1,
        "skip_special_tokens": False,
        "routed_experts_prompt_start": 1,
    }
    assert _prepared_key(sampling_params=sampling).prepared_engine_request is not None
    sampling["routed_experts_prompt_start"] = 2
    with pytest.raises(ValueError, match="routed-expert boundary"):
        _prepared_key(sampling_params=sampling)


def test_legacy_exact_action_payload_stays_byte_identical_without_prepared_fields() -> None:
    key = _key()
    payload = key.to_payload()
    assert payload["schema_version"] == 1
    assert "prepared_engine_request" not in payload
    assert "prepared_engine_request_sha256" not in payload
    assert ExactActionKey.from_payload(payload, render_prompt=lambda _: (10, 11)) == key


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "other"},
        {"token_ids": [10, 12]},
        {"sampling_params": []},
        {
            "sampling_params": {
                "temperature": 0.9,
                "top_p": 1.0,
                "seed": 17,
                "max_tokens": 2,
                "stop_token_ids": [2],
                "logprobs": 1,
                "skip_special_tokens": False,
            }
        },
        {"features": {}},
    ],
)
def test_prepared_engine_request_rejects_unbound_or_unsupported_transport(
    changes: dict[str, object],
) -> None:
    with pytest.raises((ValueError, ExactActionMismatch)):
        _prepared_key(**changes)


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
    with pytest.raises(ValueError, match="unsupported termination"):
        _action(
            key=_key(max_tokens=3),
            finish_reason="length",
            termination_kind="max_tokens",
            eos_token_id=None,
        )


def test_pinned_action_termination_matrix_is_closed_over_valid_behaviors() -> None:
    accepted = (
        ("stop", (17, 2), 4, ("eos", 2)),
        ("length", (17, 18), 2, ("max_tokens", None)),
        ("tool_calls", (17,), 4, ("tool_calls", None)),
    )
    for finish_reason, tokens, cap, expected in accepted:
        assert resolve_behavior_termination(
            finish_reason=finish_reason,
            action_token_ids=tokens,
            eos_token_id=2,
            max_tokens=cap,
        ) == expected

    rejected = (
        ("stop", (17,), 2),
        ("length", (17,), 2),
        ("content_filter", (17, 2), 2),
        ("refusal", (17, 2), 2),
        (None, (17, 2), 2),
        ("stop", (), 2),
        ("tool_calls", (17, 18, 19), 2),
    )
    for finish_reason, tokens, cap in rejected:
        with pytest.raises(ValueError):
            resolve_behavior_termination(
                finish_reason=finish_reason,
                action_token_ids=tokens,
                eos_token_id=2,
                max_tokens=cap,
            )

    for cap in range(1, 9):
        exact = tuple(range(10, 10 + cap))
        assert resolve_behavior_termination(
            finish_reason="length",
            action_token_ids=exact,
            eos_token_id=2,
            max_tokens=cap,
        ) == ("max_tokens", None)
        assert resolve_behavior_termination(
            finish_reason="stop",
            action_token_ids=(*exact[:-1], 2),
            eos_token_id=2,
            max_tokens=cap,
        ) == ("eos", 2)
        if cap > 1:
            with pytest.raises(ValueError):
                resolve_behavior_termination(
                    finish_reason="length",
                    action_token_ids=exact[:-1],
                    eos_token_id=2,
                    max_tokens=cap,
                )


def test_textual_refusal_is_an_ordinary_exact_action() -> None:
    action = _action(message={"role": "assistant", "content": "I cannot help with that."})
    assert action.parse_status == "valid"
    assert action.message["content"] == "I cannot help with that."


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": "ordinary text"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": "ordinary text", "tool_calls": None},
        {"role": "assistant", "content": "ordinary text", "tool_calls": []},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "ipython", "arguments": '{"code":"1+1"}'},
                }
            ],
        },
    ],
)
def test_pinned_typed_message_optional_field_matrix_is_closed(
    message: dict[str, object],
) -> None:
    finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
    termination_kind = "tool_calls" if finish_reason == "tool_calls" else "eos"
    action = _action(
        message=message,
        finish_reason=finish_reason,
        termination_kind=termination_kind,
        eos_token_id=None if termination_kind == "tool_calls" else 2,
    )
    assert action.parse_status == "valid"
    assert action.message == message


def test_malformed_max_token_action_strictly_reloads_without_semantic_roundtrip() -> None:
    def reject_truncated(
        _request: object,
        _message: object,
        _action_token_ids: object,
    ) -> None:
        raise ValueError("truncated action does not round-trip")

    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "ipython", "arguments": "{"},
            }
        ],
    }
    action = BehaviorAction.build(
        key=_prepared_key(),
        action_token_ids=(20, 21),
        behavior_logprobs=(-0.2, -0.1),
        raw_transport_message=message,
        finish_reason="length",
        prompt_tokens=2,
        completion_tokens=2,
        termination_kind="max_tokens",
        eos_token_id=None,
        validate_action=reject_truncated,
    )
    assert action.parse_status == "malformed"
    restored = BehaviorAction.from_bytes(
        action.to_bytes(),
        validate_action=reject_truncated,
        render_prompt=lambda _request: (10, 11),
    )
    assert restored.to_bytes() == action.to_bytes()


def test_hashes_and_versioned_canonical_digest_bind_every_action_payload() -> None:
    action = _action()
    payload = action.to_payload()
    assert payload["schema_version"] == 2
    assert json.loads(action.to_bytes())["domain"] == "redco-stage-d-behavior-action-v2"
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
    envelope = json.loads(action.to_bytes())
    envelope["action"]["unknown"] = "rejected"
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


def test_legacy_v1_action_remains_byte_readable_without_claiming_live_request_id() -> None:
    action = _action()
    legacy_payload = action.to_payload()
    legacy_payload["schema_version"] = 1
    legacy_payload.pop("request_id")
    legacy_digest = hashlib.sha256(
        canonical_json(
            {
                "domain": "redco-stage-d-behavior-action-v1",
                "action": legacy_payload,
            }
        )
    ).hexdigest()
    legacy = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-behavior-action-v1",
            "action": legacy_payload,
            "digest": legacy_digest,
        }
    )

    restored = BehaviorAction.from_bytes(
        legacy,
        encode_action=lambda _request, _message: (20, 2),
        render_prompt=lambda _: (10, 11),
    )

    assert restored.schema_version == 1
    assert restored.request_id.startswith("redco-fixture-")
    assert restored.to_bytes() == legacy


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
