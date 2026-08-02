"""Canonical no-retry local transport contracts for Stage-D evaluation."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

from redco.analysis.stage_d_evaluation_codec import canonical_object
from redco.contracts import canonical_json


def build_local_post_transport(
    *,
    endpoint: str,
    request_body: bytes,
    timeout_seconds: float,
) -> tuple[bytes, tuple[tuple[str, str], ...]]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("evaluation HTTP timeout must be positive and finite")
    port = endpoint.rsplit(":", 1)[-1]
    if not port.isdigit():
        raise ValueError("evaluation HTTP endpoint lacks a numeric port")
    headers = (
        ("content-length", str(len(request_body))),
        ("content-type", "application/json"),
        ("host", f"127.0.0.1:{port}"),
    )
    value = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-evaluation-transport-v1",
            "method": "POST",
            "url": f"{endpoint}/v1/chat/completions",
            "headers": dict(headers),
            "body_sha256": hashlib.sha256(request_body).hexdigest(),
            "timeout_seconds": timeout_seconds,
            "transport_retries": 0,
        }
    )
    verify_transport_request(
        value,
        expected_endpoint=endpoint,
        expected_body_sha256=hashlib.sha256(request_body).hexdigest(),
    )
    return value, headers


def canonical_response_headers(
    headers: list[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for name, value in headers:
        grouped[name.lower()].append(value.strip())
    result = tuple((name, ",".join(grouped[name])) for name in sorted(grouped))
    verify_nonsecret_headers(result)
    return result


def verify_transport_request(
    value: bytes,
    *,
    expected_endpoint: str,
    expected_body_sha256: str,
) -> None:
    payload = canonical_object(value, "evaluation transport request")
    if (
        set(payload)
        != {
            "schema_version",
            "domain",
            "method",
            "url",
            "headers",
            "body_sha256",
            "timeout_seconds",
            "transport_retries",
        }
        or payload.get("schema_version") != 1
        or payload.get("domain") != "redco-stage-d-evaluation-transport-v1"
        or payload.get("method") != "POST"
        or payload.get("url") != f"{expected_endpoint}/v1/chat/completions"
        or payload.get("body_sha256") != expected_body_sha256
        or payload.get("transport_retries") != 0
        or not isinstance(payload.get("headers"), dict)
        or not isinstance(payload.get("timeout_seconds"), (int, float))
        or not math.isfinite(payload["timeout_seconds"])
        or payload["timeout_seconds"] <= 0
    ):
        raise ValueError("evaluation transport request differs from the frozen local POST")
    verify_nonsecret_headers(tuple(sorted(payload["headers"].items())))


def verify_nonsecret_headers(headers: tuple[tuple[str, str], ...]) -> None:
    names = tuple(name for name, _ in headers)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError("evaluation transport headers are not sorted and unique")
    forbidden = ("authorization", "cookie", "token", "api-key", "secret")
    for name, value in headers:
        if (
            not name
            or any(item in name.lower() for item in forbidden)
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise ValueError("evaluation transport contains a secret or invalid header")


__all__ = [
    "build_local_post_transport",
    "canonical_response_headers",
    "verify_nonsecret_headers",
    "verify_transport_request",
]
