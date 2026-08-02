"""Strict parser for witnessed OpenAI-compatible evaluation responses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from redco.contracts import canonical_json


@dataclass(frozen=True, slots=True)
class ParsedOpenAIResponse:
    message: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    finish_kind: str

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-parsed-openai-response-v1",
                "message": self.message,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "finish_kind": self.finish_kind,
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> ParsedOpenAIResponse:
        payload = _canonical_object(value, "parsed OpenAI response")
        if set(payload) != {
            "schema_version",
            "domain",
            "message",
            "prompt_tokens",
            "completion_tokens",
            "finish_kind",
        } or (
            payload["schema_version"],
            payload["domain"],
        ) != (1, "redco-stage-d-parsed-openai-response-v1"):
            raise ValueError("parsed OpenAI response fields differ")
        return cls._validated(
            payload["message"],
            payload["prompt_tokens"],
            payload["completion_tokens"],
            payload["finish_kind"],
        )

    @classmethod
    def _validated(
        cls,
        message: object,
        prompt_tokens: object,
        completion_tokens: object,
        finish_kind: object,
    ) -> ParsedOpenAIResponse:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError("OpenAI response message is invalid")
        for name, value in (
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"OpenAI response {name} is invalid")
        if not isinstance(finish_kind, str) or not finish_kind.isprintable():
            raise ValueError("OpenAI response finish kind is invalid")
        return cls(
            message,
            cast(int, prompt_tokens),
            cast(int, completion_tokens),
            finish_kind,
        )


def parse_openai_response(raw: bytes, *, status_code: int) -> ParsedOpenAIResponse:
    if status_code != 200:
        raise ValueError("evaluation model response status is not successful")
    payload = _json_object(raw, "OpenAI response")
    choices = payload.get("choices")
    usage = payload.get("usage")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(usage, dict):
        raise ValueError("OpenAI response choices or usage differs")
    choice = choices[0]
    if (
        not isinstance(choice, dict)
        or choice.get("index") != 0
        or not isinstance(choice.get("message"), dict)
    ):
        raise ValueError("OpenAI response choice is invalid")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if (
        type(total_tokens) is not int
        or type(prompt_tokens) is not int
        or type(completion_tokens) is not int
        or total_tokens != prompt_tokens + completion_tokens
    ):
        raise ValueError("OpenAI response usage is inconsistent")
    finish = choice.get("finish_reason")
    finish_kind = "none" if finish is None else finish
    return ParsedOpenAIResponse._validated(
        choice["message"],
        prompt_tokens,
        completion_tokens,
        finish_kind,
    )


def _json_object(value: bytes, name: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"{name} contains duplicate keys")
            result[key] = item
        return result

    try:
        payload = json.loads(
            value,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} is not an object")
    return payload


def _canonical_object(value: bytes, name: str) -> dict[str, Any]:
    payload = _json_object(value, name)
    if canonical_json(payload) != value:
        raise ValueError(f"{name} is not canonical JSON")
    return payload


__all__ = ["ParsedOpenAIResponse", "parse_openai_response"]
