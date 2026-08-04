"""Fail-closed Stage D action and behavior-probability contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from redco.contracts import canonical_json
from redco.integrations.signed_subprocess import verify_signed_payload

SCHEMA_VERSION = 1
EXACT_ACTION_KEY_PREPARED_SCHEMA_VERSION = 2
BEHAVIOR_ACTION_SCHEMA_VERSION = 2
_LEGACY_BEHAVIOR_DOMAIN = "redco-stage-d-behavior-action-v1"
_BEHAVIOR_DOMAIN = "redco-stage-d-behavior-action-v2"
_KEY_DOMAIN = _LEGACY_BEHAVIOR_DOMAIN
_TRANSPORT_FIELDS = {
    "model",
    "messages",
    "tools",
    "parallel_tool_calls",
    "tool_choice",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "seed",
    "max_tokens",
    "stop",
    "n",
    "best_of",
    "use_beam_search",
    "logprobs",
    "top_logprobs",
    "ignore_eos",
    "min_tokens",
    "extra_body",
}
_SAMPLER_FIELDS = _TRANSPORT_FIELDS - {
    "model",
    "messages",
    "tools",
    "parallel_tool_calls",
    "extra_body",
}


class ExactActionMismatch(ValueError):
    """A recorded action does not belong to the observed policy state."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _exact_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _token_tuple(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, str | bytes | bytearray):
        raise ValueError(f"{name} must be an integer sequence")
    result = tuple(values)
    if not result or any(type(value) is not int or value < 0 for value in result):
        raise ValueError(f"{name} must be nonempty nonnegative integers")
    return result


def canonical_mapping(value: Mapping[str, Any]) -> bytes:
    """Freeze an exact request-like mapping as canonical JSON bytes."""
    return canonical_json(dict(value))


def _strict_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - observed)} "
            f"unknown={sorted(observed - expected)}"
        )


def _verify_prepared_engine_request(
    engine: Mapping[str, Any],
    request: Mapping[str, Any],
    prompt: tuple[int, ...],
) -> None:
    allowed_engine = {"model", "token_ids", "sampling_params", "cache_salt", "priority"}
    if set(engine) - allowed_engine:
        raise ValueError("prepared engine request contains unsupported fields")
    if not {"model", "token_ids", "sampling_params"} <= set(engine):
        raise ValueError("prepared engine request lacks required fields")
    if engine.get("model") != request.get("model"):
        raise ValueError("prepared engine model differs from the application request")
    if tuple(engine.get("token_ids", ())) != prompt:
        raise ExactActionMismatch("prepared engine prompt differs from rendered tokens")
    sampling = engine.get("sampling_params")
    if not isinstance(sampling, dict):
        raise ValueError("prepared engine sampling params must be an object")
    required = {
        "temperature": request.get("temperature"),
        "top_p": request.get("top_p"),
        "seed": request.get("seed"),
        "max_tokens": request.get("max_tokens"),
        "logprobs": 1,
        "skip_special_tokens": False,
    }
    if any(
        type(sampling.get(name)) is not type(expected)
        or sampling.get(name) != expected
        for name, expected in required.items()
    ):
        raise ValueError("prepared engine sampling law differs from the application request")
    neutral_if_present: dict[str, set[tuple[type[Any], Any]]] = {
        "top_k": {(type(None), None), (int, -1)},
        "min_p": {(float, 0.0)},
        "repetition_penalty": {(float, 1.0)},
        "frequency_penalty": {(float, 0.0)},
        "presence_penalty": {(float, 0.0)},
        "n": {(int, 1)},
        "best_of": {(type(None), None), (int, 1)},
        "use_beam_search": {(bool, False)},
        "ignore_eos": {(bool, False)},
        "min_tokens": {(int, 0)},
        "parallel_tool_calls": {(bool, False)},
    }
    allowed_sampling = set(required) | set(neutral_if_present) | {
        "stop_token_ids",
        "cache_salt",
        "routed_experts_prompt_start",
    }
    if set(sampling) - allowed_sampling:
        raise ValueError("prepared engine sampling params contain unsupported fields")
    if any(
        name in sampling
        and (type(sampling[name]), sampling[name]) not in allowed
        for name, allowed in neutral_if_present.items()
    ):
        raise ValueError("prepared engine request enables an unauthorized transform")
    routing_start = sampling.get("routed_experts_prompt_start")
    if routing_start is not None and (
        type(routing_start) is not int
        or routing_start < 0
        or routing_start >= len(prompt)
    ):
        raise ValueError("prepared engine request has an invalid routed-expert boundary")
    stop_ids = sampling.get("stop_token_ids")
    if (
        not isinstance(stop_ids, list)
        or not stop_ids
        or any(type(token) is not int or token < 0 for token in stop_ids)
    ):
        raise ValueError("prepared engine request lacks pinned stop token IDs")
    extra_body = request.get("extra_body")
    assert isinstance(extra_body, dict)
    expected_salt = extra_body.get("cache_salt")
    engine_salts = [
        value
        for value in (engine.get("cache_salt"), sampling.get("cache_salt"))
        if value is not None
    ]
    if engine_salts != [expected_salt]:
        raise ValueError("prepared engine cache salt differs from the application request")


def _verify_sampler_conformance(payload: bytes) -> None:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("sampler conformance manifest must be JSON") from error
    if not isinstance(value, dict) or canonical_json(value) != payload:
        raise ValueError("sampler conformance manifest must be canonical JSON")
    verify_signed_payload(value)
    if (
        value.get("schema_version") != 1
        or value.get("analysis") != "served-stack-categorical-logprob-conformance-v1"
        or value.get("passes") is not True
        or value.get("logprob_semantics") != "served_chosen_token_post_transform"
        or value.get("tool_call_termination_includes_all_generated_tokens") is not True
        or value.get("eos_is_included_in_action_tokens_and_logprobs") is not True
        or type(value.get("categorical_case_count")) is not int
        or value["categorical_case_count"] < 1
        or "fixture_only" in value
    ):
        raise ValueError("sampler conformance manifest did not pass its frozen contract")
    _require_sha256(value.get("served_stack_sha256"), "served_stack_sha256")


@dataclass(frozen=True, slots=True)
class ResolvedSamplerConfig:
    """Fully resolved, non-truncated single-sample behavior law."""

    temperature: float
    top_p: float
    top_k: None
    min_p: float
    repetition_penalty: float
    frequency_penalty: float
    presence_penalty: float
    logit_bias: tuple[()]
    seed: int
    max_tokens: int
    stop: None
    n: int
    best_of: None
    use_beam_search: bool
    logprobs: bool
    top_logprobs: int
    ignore_eos: bool
    min_tokens: int
    tool_choice: Literal["auto"]
    logprob_semantics: Literal["served_chosen_token_post_transform"] = (
        "served_chosen_token_post_transform"
    )

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_request(cls, request: Mapping[str, Any]) -> ResolvedSamplerConfig:
        unknown = set(request) - _TRANSPORT_FIELDS
        missing = _SAMPLER_FIELDS - set(request)
        if unknown or missing:
            raise ValueError(
                f"transport sampler is not fully allowlisted: "
                f"missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        if not isinstance(request.get("model"), str) or not request["model"]:
            raise ValueError("request model must be nonempty")
        if not isinstance(request.get("messages"), list):
            raise ValueError("request messages must be a list")
        if type(request.get("parallel_tool_calls")) is not bool:
            raise ValueError("parallel_tool_calls must be explicit bool")
        if request.get("tool_choice") != "auto":
            raise ValueError("tool_choice must be explicitly resolved to auto")
        extra_body = request.get("extra_body")
        if not isinstance(extra_body, dict) or set(extra_body) - {"cache_salt"}:
            raise ValueError("extra_body may contain only non-behavioral cache_salt")
        if "cache_salt" in extra_body and (
            not isinstance(extra_body["cache_salt"], str) or not extra_body["cache_salt"]
        ):
            raise ValueError("cache_salt must be a nonempty string")
        if request["top_k"] is not None or request["stop"] is not None:
            raise ValueError("top_k and stop must be explicitly disabled")
        if request["best_of"] is not None:
            raise ValueError("best_of must be explicitly disabled")
        logit_bias = request["logit_bias"]
        if type(logit_bias) is not dict or logit_bias:
            raise ValueError("logit_bias must be an exact empty object")
        values = cls(
            temperature=_exact_float(request["temperature"], "temperature"),
            top_p=_exact_float(request["top_p"], "top_p"),
            top_k=None,
            min_p=_exact_float(request["min_p"], "min_p"),
            repetition_penalty=_exact_float(request["repetition_penalty"], "repetition_penalty"),
            frequency_penalty=_exact_float(request["frequency_penalty"], "frequency_penalty"),
            presence_penalty=_exact_float(request["presence_penalty"], "presence_penalty"),
            logit_bias=(),
            seed=_exact_int(request["seed"], "seed"),
            max_tokens=_exact_int(request["max_tokens"], "max_tokens", minimum=1),
            stop=None,
            n=_exact_int(request["n"], "n", minimum=1),
            best_of=None,
            use_beam_search=cast(bool, request["use_beam_search"]),
            logprobs=cast(bool, request["logprobs"]),
            top_logprobs=_exact_int(request["top_logprobs"], "top_logprobs"),
            ignore_eos=cast(bool, request["ignore_eos"]),
            min_tokens=_exact_int(request["min_tokens"], "min_tokens"),
            tool_choice=cast(Literal["auto"], request["tool_choice"]),
        )
        return values

    def _validate(self) -> None:
        for name in (
            "temperature",
            "top_p",
            "min_p",
            "repetition_penalty",
            "frequency_penalty",
            "presence_penalty",
        ):
            _exact_float(getattr(self, name), name)
        _exact_int(self.seed, "seed")
        _exact_int(self.max_tokens, "max_tokens", minimum=1)
        _exact_int(self.n, "n", minimum=1)
        _exact_int(self.top_logprobs, "top_logprobs")
        _exact_int(self.min_tokens, "min_tokens")
        if self.top_k is not None or self.stop is not None or self.best_of is not None:
            raise ValueError("disabled sampler transforms must remain None")
        if type(self.logit_bias) is not tuple or self.logit_bias:
            raise ValueError("logit_bias must remain an immutable empty tuple")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_p != 1.0 or self.min_p != 0.0:
            raise ValueError("top-p and min-p truncation are not authorized")
        if (
            self.repetition_penalty != 1.0
            or self.frequency_penalty != 0.0
            or self.presence_penalty != 0.0
        ):
            raise ValueError("sampling penalties are not authorized")
        if type(self.use_beam_search) is not bool or self.use_beam_search:
            raise ValueError("beam search is not authorized")
        if self.n != 1:
            raise ValueError("exact behavior action requires n=1")
        if type(self.logprobs) is not bool or not self.logprobs:
            raise ValueError("served token logprobs are required")
        if self.top_logprobs != 0:
            raise ValueError("only chosen-token logprobs are authorized")
        if type(self.ignore_eos) is not bool or self.ignore_eos:
            raise ValueError("ignore_eos must be explicitly disabled")
        if self.min_tokens != 0:
            raise ValueError("minimum-token forcing is not authorized")
        if self.tool_choice != "auto":
            raise ValueError("tool_choice must remain auto")
        if self.logprob_semantics != "served_chosen_token_post_transform":
            raise ValueError("logprob semantics are not authorized")

    def to_payload(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "logit_bias": {},
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "n": self.n,
            "best_of": self.best_of,
            "use_beam_search": self.use_beam_search,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "ignore_eos": self.ignore_eos,
            "min_tokens": self.min_tokens,
            "tool_choice": self.tool_choice,
            "logprob_semantics": self.logprob_semantics,
        }


@dataclass(frozen=True, slots=True, init=False)
class ExactActionKey:
    """Everything that identifies one exact behavior-policy action state."""

    schema_version: int
    checkpoint_id: str
    base_model_manifest_sha256: str
    adapter_manifest_sha256: str | None
    tokenizer_manifest_sha256: str
    renderer_manifest_sha256: str
    sampler_conformance_manifest: bytes
    sampler_conformance_manifest_sha256: str
    tool_schema_sha256: str
    action_selection_policy: Literal["direct_single_sample"]
    transport_retry_policy: Literal["fail_before_action_no_resample"]
    sampler_config: bytes
    sampler_config_sha256: str
    request: bytes
    request_sha256: str
    prepared_engine_request: bytes | None
    prepared_engine_request_sha256: str | None
    prompt_token_ids: tuple[int, ...]
    prompt_token_ids_sha256: str

    @classmethod
    def build(
        cls,
        *,
        checkpoint_id: str,
        base_model_manifest: bytes,
        adapter_manifest: bytes | None,
        tokenizer_manifest: bytes,
        renderer_manifest: bytes,
        sampler_conformance_manifest: bytes,
        action_selection_policy: Literal["direct_single_sample"],
        transport_retry_policy: Literal["fail_before_action_no_resample"],
        request: Mapping[str, Any],
        prompt_token_ids: Sequence[int],
        render_prompt: Callable[[Mapping[str, Any]], Sequence[int]],
    ) -> ExactActionKey:
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError("checkpoint_id must be nonempty")
        for value, name in (
            (base_model_manifest, "base_model_manifest"),
            (tokenizer_manifest, "tokenizer_manifest"),
            (renderer_manifest, "renderer_manifest"),
            (sampler_conformance_manifest, "sampler_conformance_manifest"),
        ):
            if type(value) is not bytes or not value:
                raise ValueError(f"{name} must be nonempty immutable bytes")
        if adapter_manifest is not None and (
            type(adapter_manifest) is not bytes or not adapter_manifest
        ):
            raise ValueError("adapter_manifest must be nonempty immutable bytes or None")
        _verify_sampler_conformance(sampler_conformance_manifest)
        if action_selection_policy != "direct_single_sample":
            raise ValueError("action selection must use one direct sample")
        if transport_retry_policy != "fail_before_action_no_resample":
            raise ValueError("transport retries may not resample an action")
        sampler = ResolvedSamplerConfig.from_request(request)
        if request["model"] != checkpoint_id or request["seed"] != sampler.seed:
            raise ValueError("request model or seed disagrees with exact action key")
        tools = request.get("tools", [])
        if not isinstance(tools, list):
            raise ValueError("request tools must be a list")
        if request["parallel_tool_calls"]:
            raise ValueError("parallel tool-call selection is not authorized")
        sampler_bytes = canonical_json(sampler.to_payload())
        request_bytes = canonical_mapping(request)
        prompt = _token_tuple(prompt_token_ids, "prompt_token_ids")
        rendered_prompt = _token_tuple(render_prompt(request), "rendered_prompt_token_ids")
        if rendered_prompt != prompt:
            raise ExactActionMismatch("request does not round-trip to prompt tokens")
        prompt_bytes = canonical_json(prompt)
        self = object.__new__(cls)
        for name, field_value in {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "base_model_manifest_sha256": _sha256(base_model_manifest),
            "adapter_manifest_sha256": (
                _sha256(adapter_manifest) if adapter_manifest is not None else None
            ),
            "tokenizer_manifest_sha256": _sha256(tokenizer_manifest),
            "renderer_manifest_sha256": _sha256(renderer_manifest),
            "sampler_conformance_manifest": sampler_conformance_manifest,
            "sampler_conformance_manifest_sha256": _sha256(sampler_conformance_manifest),
            "tool_schema_sha256": _sha256(canonical_json(tools)),
            "action_selection_policy": action_selection_policy,
            "transport_retry_policy": transport_retry_policy,
            "sampler_config": sampler_bytes,
            "sampler_config_sha256": _sha256(sampler_bytes),
            "request": request_bytes,
            "request_sha256": _sha256(request_bytes),
            "prepared_engine_request": None,
            "prepared_engine_request_sha256": None,
            "prompt_token_ids": prompt,
            "prompt_token_ids_sha256": _sha256(prompt_bytes),
        }.items():
            object.__setattr__(self, name, field_value)
        return self

    @classmethod
    def build_prepared(
        cls,
        *,
        prepared_engine_request: Mapping[str, Any],
        **kwargs: Any,
    ) -> ExactActionKey:
        """Build from the renderer's already-prepared request without re-rendering."""
        prompt_token_ids = kwargs.get("prompt_token_ids")
        if not isinstance(prompt_token_ids, Sequence) or isinstance(
            prompt_token_ids, str | bytes | bytearray
        ):
            raise ValueError("prepared prompt token IDs must be a sequence")
        prompt = _token_tuple(prompt_token_ids, "prompt_token_ids")
        engine = dict(prepared_engine_request)
        request = kwargs.get("request")
        if not isinstance(request, Mapping):
            raise ValueError("prepared application request must be an object")
        if kwargs.get("checkpoint_id") != request.get("model"):
            raise ValueError("prepared checkpoint differs from the application request")
        _verify_prepared_engine_request(engine, request, prompt)
        self = cls.build(
            **kwargs,
            render_prompt=lambda _: prompt,
        )
        engine_bytes = canonical_mapping(engine)
        object.__setattr__(self, "schema_version", EXACT_ACTION_KEY_PREPARED_SCHEMA_VERSION)
        object.__setattr__(self, "prepared_engine_request", engine_bytes)
        object.__setattr__(
            self,
            "prepared_engine_request_sha256",
            _sha256(engine_bytes),
        )
        return self

    @classmethod
    def resample_prepared(
        cls,
        reference: ExactActionKey,
        *,
        seed: int,
        cache_salt: str,
    ) -> ExactActionKey:
        """Change only the randomized draw address of a frozen prepared policy state."""
        if type(reference) is not ExactActionKey or reference.schema_version != (
            EXACT_ACTION_KEY_PREPARED_SCHEMA_VERSION
        ):
            raise ValueError("candidate resampling requires a prepared exact action key")
        _exact_int(seed, "seed")
        if not isinstance(cache_salt, str) or not cache_salt:
            raise ValueError("candidate cache salt must be nonempty")
        payload = reference.to_payload()
        request = cast(dict[str, Any], payload["request"])
        sampler = cast(dict[str, Any], payload["sampler_config"])
        engine = cast(dict[str, Any], payload["prepared_engine_request"])
        engine_sampling = engine.get("sampling_params")
        if not isinstance(engine_sampling, dict):
            raise ValueError("prepared candidate engine request lacks sampling params")
        extra_body = request.get("extra_body")
        if not isinstance(extra_body, dict):
            raise ValueError("prepared application request lacks extra_body")
        request["seed"] = seed
        extra_body["cache_salt"] = cache_salt
        sampler["seed"] = seed
        engine_sampling["seed"] = seed
        if "cache_salt" in engine:
            engine["cache_salt"] = cache_salt
        elif "cache_salt" in engine_sampling:
            engine_sampling["cache_salt"] = cache_salt
        else:
            raise ValueError("prepared candidate engine request lacks cache salt")
        payload["sampler_config_sha256"] = _sha256(canonical_json(sampler))
        payload["request_sha256"] = _sha256(canonical_json(request))
        payload["prepared_engine_request_sha256"] = _sha256(canonical_json(engine))
        candidate = cls.from_payload(
            payload,
            render_prompt=lambda _: reference.prompt_token_ids,
        )
        before = reference.to_payload()
        after = candidate.to_payload()
        for value in (before, after):
            cast(dict[str, Any], value["request"])["seed"] = 0
            request_extra = cast(dict[str, Any], value["request"])["extra_body"]
            if not isinstance(request_extra, dict):
                raise ValueError("prepared application request lacks extra_body")
            request_extra["cache_salt"] = "*"
            cast(dict[str, Any], value["sampler_config"])["seed"] = 0
            prepared = cast(dict[str, Any], value["prepared_engine_request"])
            cast(dict[str, Any], prepared["sampling_params"])["seed"] = 0
            if "cache_salt" in prepared:
                prepared["cache_salt"] = "*"
            else:
                cast(dict[str, Any], prepared["sampling_params"])["cache_salt"] = "*"
            value["sampler_config_sha256"] = "*"
            value["request_sha256"] = "*"
            value["prepared_engine_request_sha256"] = "*"
        if before != after:
            raise ValueError("candidate resampling changed fields beyond seed and cache salt")
        return candidate

    @property
    def sampler(self) -> ResolvedSamplerConfig:
        request = json.loads(self.request)
        if not isinstance(request, dict):
            raise ValueError("stored request is not an object")
        return ResolvedSamplerConfig.from_request(request)

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "base_model_manifest_sha256": self.base_model_manifest_sha256,
            "adapter_manifest_sha256": self.adapter_manifest_sha256,
            "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            "renderer_manifest_sha256": self.renderer_manifest_sha256,
            "sampler_conformance_manifest": json.loads(self.sampler_conformance_manifest),
            "sampler_conformance_manifest_sha256": (self.sampler_conformance_manifest_sha256),
            "tool_schema_sha256": self.tool_schema_sha256,
            "action_selection_policy": self.action_selection_policy,
            "transport_retry_policy": self.transport_retry_policy,
            "sampler_config": json.loads(self.sampler_config),
            "sampler_config_sha256": self.sampler_config_sha256,
            "request": json.loads(self.request),
            "request_sha256": self.request_sha256,
            "prompt_token_ids": list(self.prompt_token_ids),
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
        }
        if self.schema_version == EXACT_ACTION_KEY_PREPARED_SCHEMA_VERSION:
            if (
                self.prepared_engine_request is None
                or self.prepared_engine_request_sha256 is None
            ):
                raise ValueError("prepared exact action key lacks engine evidence")
            payload["prepared_engine_request"] = json.loads(
                self.prepared_engine_request
            )
            payload["prepared_engine_request_sha256"] = (
                self.prepared_engine_request_sha256
            )
        elif self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported exact action key schema")
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        render_prompt: Callable[[Mapping[str, Any]], Sequence[int]],
    ) -> ExactActionKey:
        legacy_expected = {
            "schema_version",
            "checkpoint_id",
            "base_model_manifest_sha256",
            "adapter_manifest_sha256",
            "tokenizer_manifest_sha256",
            "renderer_manifest_sha256",
            "sampler_conformance_manifest",
            "sampler_conformance_manifest_sha256",
            "tool_schema_sha256",
            "action_selection_policy",
            "transport_retry_policy",
            "sampler_config",
            "sampler_config_sha256",
            "request",
            "request_sha256",
            "prompt_token_ids",
            "prompt_token_ids_sha256",
        }
        schema_version = payload.get("schema_version")
        expected = (
            legacy_expected
            if schema_version == SCHEMA_VERSION
            else legacy_expected
            | {"prepared_engine_request", "prepared_engine_request_sha256"}
        )
        _strict_keys(payload, expected, "exact action key")
        if (
            type(schema_version) is not int
            or schema_version
            not in {SCHEMA_VERSION, EXACT_ACTION_KEY_PREPARED_SCHEMA_VERSION}
        ):
            raise ValueError("unsupported exact action key schema")
        checkpoint_id = payload["checkpoint_id"]
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError("checkpoint_id must be nonempty")
        adapter_hash = payload["adapter_manifest_sha256"]
        if adapter_hash is not None:
            adapter_hash = _require_sha256(adapter_hash, "adapter_manifest_sha256")
        request = payload["request"]
        sampler_payload = payload["sampler_config"]
        conformance_payload = payload["sampler_conformance_manifest"]
        if (
            not isinstance(request, dict)
            or not isinstance(sampler_payload, dict)
            or not isinstance(conformance_payload, dict)
        ):
            raise ValueError("request, sampler config, and conformance must be objects")
        conformance_bytes = canonical_json(conformance_payload)
        _verify_sampler_conformance(conformance_bytes)
        sampler = ResolvedSamplerConfig.from_request(request)
        if canonical_json(sampler.to_payload()) != canonical_json(sampler_payload):
            raise ValueError("stored sampler config disagrees with transport request")
        request_bytes = canonical_json(request)
        raw_engine = payload.get("prepared_engine_request")
        raw_engine_hash = payload.get("prepared_engine_request_sha256")
        if schema_version == EXACT_ACTION_KEY_PREPARED_SCHEMA_VERSION:
            if not isinstance(raw_engine, dict):
                raise ValueError("prepared engine request must be an object")
            engine_bytes: bytes | None = canonical_json(raw_engine)
            engine_hash: str | None = _require_sha256(
                raw_engine_hash,
                "prepared_engine_request_sha256",
            )
            if raw_engine.get("model") != checkpoint_id:
                raise ValueError("prepared engine request model differs")
        else:
            engine_bytes = None
            engine_hash = None
        sampler_bytes = canonical_json(sampler_payload)
        tools = request.get("tools", [])
        if not isinstance(tools, list):
            raise ValueError("request tools must be a list")
        if request["model"] != checkpoint_id or request["parallel_tool_calls"]:
            raise ValueError("stored request disagrees with model or tool selection policy")
        prompt = _token_tuple(payload["prompt_token_ids"], "prompt_token_ids")
        if engine_bytes is not None:
            assert isinstance(raw_engine, dict)
            if tuple(raw_engine.get("token_ids", ())) != prompt:
                raise ExactActionMismatch("prepared engine request prompt differs")
            _verify_prepared_engine_request(raw_engine, request, prompt)
        if schema_version == SCHEMA_VERSION:
            rendered_prompt = _token_tuple(
                render_prompt(request),
                "rendered_prompt_token_ids",
            )
            if rendered_prompt != prompt:
                raise ExactActionMismatch("stored request does not render to prompt tokens")
        validated = {
            "schema_version": schema_version,
            "checkpoint_id": checkpoint_id,
            "base_model_manifest_sha256": _require_sha256(
                payload["base_model_manifest_sha256"], "base_model_manifest_sha256"
            ),
            "adapter_manifest_sha256": adapter_hash,
            "tokenizer_manifest_sha256": _require_sha256(
                payload["tokenizer_manifest_sha256"], "tokenizer_manifest_sha256"
            ),
            "renderer_manifest_sha256": _require_sha256(
                payload["renderer_manifest_sha256"], "renderer_manifest_sha256"
            ),
            "sampler_conformance_manifest": conformance_bytes,
            "sampler_conformance_manifest_sha256": _require_sha256(
                payload["sampler_conformance_manifest_sha256"],
                "sampler_conformance_manifest_sha256",
            ),
            "tool_schema_sha256": _require_sha256(
                payload["tool_schema_sha256"], "tool_schema_sha256"
            ),
            "action_selection_policy": payload["action_selection_policy"],
            "transport_retry_policy": payload["transport_retry_policy"],
            "sampler_config": sampler_bytes,
            "sampler_config_sha256": _require_sha256(
                payload["sampler_config_sha256"], "sampler_config_sha256"
            ),
            "request": request_bytes,
            "request_sha256": _require_sha256(payload["request_sha256"], "request_sha256"),
            "prepared_engine_request": engine_bytes,
            "prepared_engine_request_sha256": engine_hash,
            "prompt_token_ids": prompt,
            "prompt_token_ids_sha256": _require_sha256(
                payload["prompt_token_ids_sha256"], "prompt_token_ids_sha256"
            ),
        }
        if validated["action_selection_policy"] != "direct_single_sample":
            raise ValueError("stored action selection policy is not authorized")
        if validated["transport_retry_policy"] != "fail_before_action_no_resample":
            raise ValueError("stored retry policy is not authorized")
        expected_hashes = {
            "tool_schema_sha256": _sha256(canonical_json(tools)),
            "sampler_conformance_manifest_sha256": _sha256(conformance_bytes),
            "sampler_config_sha256": _sha256(sampler_bytes),
            "request_sha256": _sha256(request_bytes),
            "prompt_token_ids_sha256": _sha256(canonical_json(prompt)),
        }
        if engine_bytes is not None:
            expected_hashes["prepared_engine_request_sha256"] = _sha256(engine_bytes)
        if any(validated[name] != value for name, value in expected_hashes.items()):
            raise ValueError("stored exact action key hash mismatch")
        self = object.__new__(cls)
        for name, field_value in validated.items():
            object.__setattr__(self, name, field_value)
        return self

    @property
    def digest(self) -> str:
        return _sha256(canonical_json({"domain": _KEY_DOMAIN, "key": self.to_payload()}))


TerminationKind = Literal["eos", "max_tokens", "tool_calls"]


def resolve_behavior_termination(
    *,
    finish_reason: object,
    action_token_ids: Sequence[int],
    eos_token_id: int | None,
    max_tokens: int,
) -> tuple[TerminationKind, int | None]:
    """Resolve every terminal behavior admitted by the pinned Stage-D interface."""
    action = _token_tuple(action_token_ids, "action_token_ids")
    cap = _exact_int(max_tokens, "max_tokens")
    if cap < 1:
        raise ValueError("max_tokens must be positive")
    if len(action) > cap:
        raise ValueError("action exceeds the resolved max_tokens")
    if finish_reason == "tool_calls":
        return "tool_calls", None
    if finish_reason == "length" and len(action) == cap:
        return "max_tokens", None
    if finish_reason == "stop":
        eos = _exact_int(eos_token_id, "eos_token_id")
        if action[-1] == eos:
            return "eos", eos
    raise ValueError("sampled action has an unsupported termination")


@dataclass(frozen=True, slots=True, init=False)
class BehaviorAction:
    """One exact sampled action with token-aligned served-policy logprobs."""

    schema_version: int
    key: ExactActionKey
    action_token_ids: tuple[int, ...]
    action_token_ids_sha256: str
    behavior_logprobs: tuple[float, ...]
    behavior_logprobs_sha256: str
    raw_transport_message: bytes
    raw_transport_message_sha256: str
    parse_status: Literal["valid", "malformed"]
    parse_error: str | None
    request_id: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    termination_kind: TerminationKind
    eos_token_id: int | None

    @classmethod
    def build(
        cls,
        *,
        key: ExactActionKey,
        action_token_ids: Sequence[int],
        behavior_logprobs: Sequence[float],
        raw_transport_message: Mapping[str, Any],
        finish_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
        termination_kind: TerminationKind,
        eos_token_id: int | None,
        encode_action: Callable[[Mapping[str, Any], Mapping[str, Any]], Sequence[int]]
        | None = None,
        validate_action: Callable[
            [Mapping[str, Any], Mapping[str, Any], Sequence[int]], None
        ]
        | None = None,
        request_id: str | None = None,
    ) -> BehaviorAction:
        if type(key) is not ExactActionKey:
            raise ValueError("key must be a validated ExactActionKey")
        action = _token_tuple(action_token_ids, "action_token_ids")
        logprobs = tuple(behavior_logprobs)
        if len(action) != len(logprobs) or any(
            type(value) is not float or not math.isfinite(value) or value > 0.0
            for value in logprobs
        ):
            raise ValueError("behavior logprobs must be finite non-positive floats per token")
        message = dict(raw_transport_message)
        message_bytes = canonical_json(message)
        parse_error: str | None
        try:
            _strict_typed_message(message)
        except (ValueError, json.JSONDecodeError) as error:
            parse_status: Literal["valid", "malformed"] = "malformed"
            parse_error = f"{type(error).__name__}: {error}"
        else:
            parse_status = "valid"
            parse_error = None
        request = json.loads(key.request)
        if not isinstance(request, dict):
            raise ValueError("exact key request is not an object")
        prepared = key.schema_version == EXACT_ACTION_KEY_PREPARED_SCHEMA_VERSION
        if prepared and (validate_action is None or encode_action is not None):
            raise ValueError("prepared actions require token-semantic validation")
        if not prepared and (encode_action is None or validate_action is not None):
            raise ValueError("legacy actions require typed-message re-encoding")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise ValueError("finish_reason must be nonempty")
        if request_id is None:
            resolved_request_id = "redco-fixture-" + _sha256(
                canonical_json(
                    {
                        "key": key.digest,
                        "action_token_ids": action,
                        "behavior_logprobs": logprobs,
                    }
                )
            )[:24]
        elif (
            not isinstance(request_id, str)
            or not request_id
            or len(request_id) > 512
            or not request_id.isprintable()
        ):
            raise ValueError("request_id must be a nonempty printable string")
        else:
            resolved_request_id = request_id
        prompt_usage = _exact_int(prompt_tokens, "prompt_tokens")
        completion_usage = _exact_int(completion_tokens, "completion_tokens")
        if prompt_usage != len(key.prompt_token_ids) or completion_usage != len(action):
            raise ValueError("usage must exactly match prompt and action token arrays")
        sampler = key.sampler
        if termination_kind not in {"eos", "max_tokens", "tool_calls"}:
            raise ValueError("stop-sequence and unknown termination are not authorized")
        expected_termination, expected_eos = resolve_behavior_termination(
            finish_reason=finish_reason,
            action_token_ids=action,
            eos_token_id=eos_token_id,
            max_tokens=sampler.max_tokens,
        )
        if (termination_kind, eos_token_id) != (expected_termination, expected_eos):
            raise ValueError("sampled action termination contract is inconsistent")
        if expected_termination != "max_tokens":
            if prepared:
                assert validate_action is not None
                validate_action(request, message, action)
            else:
                assert encode_action is not None
                rendered = _token_tuple(
                    encode_action(request, message),
                    "rendered_action_token_ids",
                )
                if rendered != action:
                    raise ExactActionMismatch(
                        "typed message does not round-trip to action tokens"
                    )
        eos = expected_eos
        self = object.__new__(cls)
        action_bytes = canonical_json(action)
        logprob_bytes = canonical_json(logprobs)
        for name, value in {
            "schema_version": BEHAVIOR_ACTION_SCHEMA_VERSION,
            "key": key,
            "action_token_ids": action,
            "action_token_ids_sha256": _sha256(action_bytes),
            "behavior_logprobs": logprobs,
            "behavior_logprobs_sha256": _sha256(logprob_bytes),
            "raw_transport_message": message_bytes,
            "raw_transport_message_sha256": _sha256(message_bytes),
            "parse_status": parse_status,
            "parse_error": parse_error,
            "request_id": resolved_request_id,
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_usage,
            "completion_tokens": completion_usage,
            "termination_kind": termination_kind,
            "eos_token_id": eos,
        }.items():
            object.__setattr__(self, name, value)
        return self

    @property
    def message(self) -> dict[str, Any]:
        value = json.loads(self.raw_transport_message)
        assert isinstance(value, dict)
        return value

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "key": self.key.to_payload(),
            "action_token_ids": list(self.action_token_ids),
            "action_token_ids_sha256": self.action_token_ids_sha256,
            "behavior_logprobs": list(self.behavior_logprobs),
            "behavior_logprobs_sha256": self.behavior_logprobs_sha256,
            "raw_transport_message": self.message,
            "raw_transport_message_sha256": self.raw_transport_message_sha256,
            "parse_status": self.parse_status,
            "parse_error": self.parse_error,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "termination_kind": self.termination_kind,
            "eos_token_id": self.eos_token_id,
        }
        if self.schema_version == BEHAVIOR_ACTION_SCHEMA_VERSION:
            payload["request_id"] = self.request_id
        elif self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported behavior action schema")
        return payload

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": self.schema_version,
                "domain": (
                    _BEHAVIOR_DOMAIN
                    if self.schema_version == BEHAVIOR_ACTION_SCHEMA_VERSION
                    else _LEGACY_BEHAVIOR_DOMAIN
                ),
                "action": self.to_payload(),
                "digest": self.digest,
            }
        )

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        encode_action: Callable[[Mapping[str, Any], Mapping[str, Any]], Sequence[int]]
        | None = None,
        validate_action: Callable[
            [Mapping[str, Any], Mapping[str, Any], Sequence[int]], None
        ]
        | None = None,
        render_prompt: Callable[[Mapping[str, Any]], Sequence[int]],
    ) -> BehaviorAction:
        if type(payload) is not bytes:
            raise ValueError("serialized action must be immutable bytes")
        envelope = json.loads(payload)
        if not isinstance(envelope, dict) or canonical_json(envelope) != payload:
            raise ValueError("serialized action must be canonical JSON")
        _strict_keys(envelope, {"schema_version", "domain", "action", "digest"}, "action")
        schema_version = envelope["schema_version"]
        expected_domain = {
            SCHEMA_VERSION: _LEGACY_BEHAVIOR_DOMAIN,
            BEHAVIOR_ACTION_SCHEMA_VERSION: _BEHAVIOR_DOMAIN,
        }.get(schema_version)
        if type(schema_version) is not int or envelope["domain"] != expected_domain:
            raise ValueError("unsupported behavior action envelope")
        action_payload = envelope["action"]
        if not isinstance(action_payload, dict):
            raise ValueError("behavior action payload must be an object")
        expected = {
            "schema_version",
            "key",
            "action_token_ids",
            "action_token_ids_sha256",
            "behavior_logprobs",
            "behavior_logprobs_sha256",
            "raw_transport_message",
            "raw_transport_message_sha256",
            "parse_status",
            "parse_error",
            "finish_reason",
            "prompt_tokens",
            "completion_tokens",
            "termination_kind",
            "eos_token_id",
        }
        if schema_version == BEHAVIOR_ACTION_SCHEMA_VERSION:
            expected.add("request_id")
        _strict_keys(action_payload, expected, "behavior action")
        key_payload = action_payload["key"]
        message = action_payload["raw_transport_message"]
        if not isinstance(key_payload, dict) or not isinstance(message, dict):
            raise ValueError("stored key and typed message must be objects")
        action = cls.build(
            key=ExactActionKey.from_payload(key_payload, render_prompt=render_prompt),
            action_token_ids=action_payload["action_token_ids"],
            behavior_logprobs=action_payload["behavior_logprobs"],
            raw_transport_message=message,
            finish_reason=action_payload["finish_reason"],
            prompt_tokens=action_payload["prompt_tokens"],
            completion_tokens=action_payload["completion_tokens"],
            termination_kind=action_payload["termination_kind"],
            eos_token_id=action_payload["eos_token_id"],
            encode_action=encode_action,
            validate_action=validate_action,
            request_id=action_payload.get("request_id"),
        )
        if schema_version == SCHEMA_VERSION:
            object.__setattr__(action, "schema_version", SCHEMA_VERSION)
        if canonical_json(action.to_payload()) != canonical_json(action_payload):
            raise ValueError("stored behavior action fields or hashes disagree")
        if envelope["digest"] != action.digest:
            raise ValueError("stored behavior action digest mismatch")
        return action

    @property
    def digest(self) -> str:
        domain = (
            _BEHAVIOR_DOMAIN
            if self.schema_version == BEHAVIOR_ACTION_SCHEMA_VERSION
            else _LEGACY_BEHAVIOR_DOMAIN
        )
        payload = self.to_payload()
        if self.schema_version == BEHAVIOR_ACTION_SCHEMA_VERSION:
            # The request ID remains authenticated transport evidence in to_bytes(),
            # but is not part of the sampled scientific behavior identity.
            payload.pop("request_id")
        return _sha256(canonical_json({"domain": domain, "action": payload}))


def _strict_typed_message(value: Mapping[str, Any]) -> dict[str, Any]:
    message = dict(value)
    if set(message) - {"role", "content", "tool_calls"}:
        raise ValueError("typed message contains unknown fields")
    if message.get("role") != "assistant":
        raise ValueError("typed message role must be assistant")
    if not isinstance(message.get("content"), str | type(None)):
        raise ValueError("typed message content must be string or null")
    calls = message.get("tool_calls")
    if calls is None:
        return message
    if not isinstance(calls, list):
        raise ValueError("tool_calls must be a list")
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"id", "type", "function"}:
            raise ValueError("tool call schema is not exact")
        if not isinstance(call["id"], str) or not call["id"] or call["type"] != "function":
            raise ValueError("tool call identity or type is invalid")
        function = call["function"]
        if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
            raise ValueError("tool function schema is not exact")
        if not isinstance(function["name"], str) or not function["name"]:
            raise ValueError("tool function name is invalid")
        if not isinstance(function["arguments"], str):
            raise ValueError("tool function arguments must be serialized JSON")
        arguments = json.loads(function["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("tool function arguments must encode an object")
    return message


def require_exact_reuse(recorded: BehaviorAction, observed: ExactActionKey) -> None:
    """Allow recorded action reuse only under byte-exact state identity."""
    if type(observed) is not ExactActionKey or recorded.key != observed:
        observed_digest = observed.digest if type(observed) is ExactActionKey else "invalid"
        raise ExactActionMismatch(
            f"exact action key mismatch: recorded={recorded.key.digest} observed={observed_digest}"
        )


def categorical_probabilities(logits: Sequence[float], *, temperature: float) -> tuple[float, ...]:
    """Exact finite categorical behavior distribution after temperature."""
    if not logits or any(type(value) is not float or not math.isfinite(value) for value in logits):
        raise ValueError("logits must be nonempty finite floats")
    temperature = _exact_float(temperature, "temperature")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [value / temperature for value in logits]
    offset = max(scaled)
    weights = [math.exp(value - offset) for value in scaled]
    normalizer = sum(weights)
    return tuple(value / normalizer for value in weights)
