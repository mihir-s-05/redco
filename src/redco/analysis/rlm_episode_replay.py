"""Event-addressed scripted model service for whole-episode RLM replay."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json
from redco.integrations.verifiers_trace import audit_trace_file, load_trace_records


class EpisodeReplayIneligibility(ValueError):
    """The recorded episode lacks an exact event replay contract."""


class ReplayResamplingRequired(RuntimeError):
    """A policy action cannot be reused because its recorded request changed."""


def _optional_header(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        value = headers.get(name.lower())
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _required_header(headers: Mapping[str, str], name: str) -> str:
    value = _optional_header(headers, name)
    if value is None:
        raise EpisodeReplayIneligibility(f"missing {name}")
    return value


def _child_lineage(
    *,
    parent_lineage: str,
    depth: int,
    parent_turn: int,
    parent_tool_call_id: str,
    invocation_id: str,
) -> str:
    """Derive a stable lineage path without runtime-generated session IDs."""
    segment = hashlib.sha256(
        canonical_json(
            {
                "depth": depth,
                "parent_turn": parent_turn,
                "parent_tool_call_id": parent_tool_call_id,
                "invocation_id": invocation_id,
            }
        )
    ).hexdigest()[:16]
    return f"{parent_lineage}/{segment}"


@dataclass(frozen=True, slots=True)
class RLMEventAddress:
    depth: int
    turn: int
    call_kind: str
    parent_lineage: str | None = None
    parent_turn: int | None = None
    parent_tool_call_id: str | None = None
    invocation_id: str | None = None

    def __post_init__(self) -> None:
        if self.depth < 0 or self.turn < 0:
            raise ValueError("depth and turn must be nonnegative")
        if self.call_kind not in {"policy", "compaction"}:
            raise ValueError("call_kind must be policy or compaction")
        provenance_fields = (
            self.parent_turn,
            self.parent_tool_call_id,
            self.invocation_id,
        )
        if self.depth == 0 and (
            self.parent_lineage is not None or any(value is not None for value in provenance_fields)
        ):
            raise ValueError("root addresses cannot contain child provenance")
        if self.depth > 0 and any(value is None for value in provenance_fields):
            raise EpisodeReplayIneligibility(
                "recursive calls require parent turn, parent tool call, and invocation ID"
            )

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> RLMEventAddress:
        def text(name: str) -> str | None:
            value = headers.get(name)
            if value is None:
                value = headers.get(name.lower())
            if value is None:
                return None
            normalized = value.strip()
            return normalized or None

        def integer(name: str, *, required: bool) -> int | None:
            value = text(name)
            if value is None:
                if required:
                    raise EpisodeReplayIneligibility(f"missing {name}")
                return None
            try:
                parsed = int(value)
            except ValueError as error:
                raise EpisodeReplayIneligibility(f"invalid {name}") from error
            if parsed < 0:
                raise EpisodeReplayIneligibility(f"negative {name}")
            return parsed

        depth = integer("X-RLM-Depth", required=True)
        turn = integer("X-RLM-Turn", required=True)
        assert depth is not None and turn is not None
        call_kind = text("X-RLM-Call-Kind")
        if call_kind is None:
            raise EpisodeReplayIneligibility("missing X-RLM-Call-Kind")
        return cls(
            depth=depth,
            turn=turn,
            call_kind=call_kind,
            parent_lineage=None,
            parent_turn=integer("X-RLM-Parent-Turn", required=depth > 0),
            parent_tool_call_id=text("X-RLM-Parent-Tool-Call-ID"),
            invocation_id=text("X-RLM-Invocation-ID"),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RLMEventAddress:
        return cls(
            depth=_integer(payload, "depth"),
            turn=_integer(payload, "turn"),
            call_kind=_string(payload, "call_kind"),
            parent_lineage=_optional_string(payload.get("parent_lineage")),
            parent_turn=_optional_integer(payload.get("parent_turn")),
            parent_tool_call_id=_optional_string(payload.get("parent_tool_call_id")),
            invocation_id=_optional_string(payload.get("invocation_id")),
        )

    def key(self) -> str:
        if self.depth == 0:
            return f"depth:0:root:{self.call_kind}:{self.turn}"
        if self.parent_lineage is None:
            raise EpisodeReplayIneligibility(
                "recursive transport address lacks its resolved parent lineage"
            )
        return (
            f"depth:{self.depth}:parent:{self.parent_lineage}:"
            f"child:{self.parent_turn}:{self.parent_tool_call_id}:"
            f"{self.invocation_id}:{self.call_kind}:{self.turn}"
        )


@dataclass(frozen=True, slots=True)
class ScriptedEvent:
    address: RLMEventAddress
    message: dict[str, Any]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    delay_seconds: float = 0.0
    response_mode: str = "literal"
    expected_request_sha256: str | None = None
    checkpoint_id: str | None = None
    decoding_config_hash: str | None = None
    event_seed: int | None = None
    prompt_token_ids_sha256: str | None = None
    action_token_ids_sha256: str | None = None
    engineering_transport_path_normalization: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScriptedEvent:
        message = payload.get("message")
        if not isinstance(message, dict):
            raise TypeError("scripted event message must be an object")
        delay = payload.get("delay_seconds", 0.0)
        if not isinstance(delay, int | float) or delay < 0:
            raise ValueError("delay_seconds must be nonnegative")
        return cls(
            address=RLMEventAddress.from_dict(_mapping(payload, "address")),
            message=dict(message),
            finish_reason=_string(payload, "finish_reason"),
            prompt_tokens=_integer(payload, "prompt_tokens"),
            completion_tokens=_integer(payload, "completion_tokens"),
            delay_seconds=float(delay),
            response_mode=str(payload.get("response_mode", "literal")),
            expected_request_sha256=_optional_sha256(payload.get("expected_request_sha256")),
            checkpoint_id=_optional_string(payload.get("checkpoint_id")),
            decoding_config_hash=_optional_sha256(payload.get("decoding_config_hash")),
            event_seed=_optional_integer(payload.get("event_seed")),
            prompt_token_ids_sha256=_optional_sha256(payload.get("prompt_token_ids_sha256")),
            action_token_ids_sha256=_optional_sha256(payload.get("action_token_ids_sha256")),
            engineering_transport_path_normalization=_boolean(
                payload.get("engineering_transport_path_normalization", False),
                "engineering_transport_path_normalization",
            ),
        )


class ScriptedCompletionRouter:
    """Serve one immutable response per structural RLM event address."""

    def __init__(self, events: tuple[ScriptedEvent, ...]) -> None:
        self._events = {event.address.key(): event for event in events}
        if len(self._events) != len(events):
            raise ValueError("scripted event addresses must be unique")
        self._requests: dict[str, bytes] = {}
        self._request_counts: dict[str, int] = {}
        self._records: list[dict[str, Any]] = []
        self._completion_order: list[str] = []
        self._root_session_id: str | None = None
        self._session_lineages: dict[str, tuple[int, str]] = {}
        self._lock = threading.Lock()

    def respond(
        self,
        *,
        headers: Mapping[str, str],
        request: dict[str, Any],
    ) -> dict[str, Any]:
        address = self._validate_episode_scope(RLMEventAddress.from_headers(headers), headers)
        key = address.key()
        event = self._events.get(key)
        if event is None:
            raise EpisodeReplayIneligibility(f"unrecorded event address: {key}")
        request_sha256 = _request_projection_sha256(
            request,
            normalize_transport_paths=(event.engineering_transport_path_normalization),
        )
        if (
            event.expected_request_sha256 is not None
            and request_sha256 != event.expected_request_sha256
        ):
            with self._lock:
                self._records.append(
                    {
                        "address": key,
                        "mode": "recorded_request_mismatch",
                        "expected_request_projection_sha256": (event.expected_request_sha256),
                        "request_projection_sha256": request_sha256,
                        "request_projection": _request_projection(
                            request,
                            normalize_transport_paths=(
                                event.engineering_transport_path_normalization
                            ),
                        ),
                    }
                )
            raise ReplayResamplingRequired(
                "recorded action is ineligible for changed request: "
                f"{key} expected={event.expected_request_sha256} "
                f"observed={request_sha256}"
            )
        digest = hashlib.sha256(canonical_json(request)).digest()
        with self._lock:
            previous = self._requests.get(key)
            if previous is not None and previous != digest:
                raise EpisodeReplayIneligibility(
                    f"event address received a different retry body: {key}"
                )
            count = self._request_counts.get(key, 0) + 1
            if count != 1:
                raise EpisodeReplayIneligibility(f"event address appeared more than once: {key}")
            self._requests[key] = digest
            self._request_counts[key] = count
            self._records.append(
                {
                    "address": key,
                    "request_sha256": digest.hex(),
                    "request_projection_sha256": request_sha256,
                    "normalized_transport_request_match": (
                        event.expected_request_sha256 is None
                        or request_sha256 == event.expected_request_sha256
                    ),
                    "checkpoint_id": event.checkpoint_id,
                    "decoding_config_hash": event.decoding_config_hash,
                    "event_seed": event.event_seed,
                    "prompt_token_ids_sha256": event.prompt_token_ids_sha256,
                    "action_token_ids_sha256": event.action_token_ids_sha256,
                    "session_id_present": bool(
                        headers.get("X-RLM-Session-ID") or headers.get("x-rlm-session-id")
                    ),
                }
            )
        if event.delay_seconds:
            time.sleep(event.delay_seconds)
        message = dict(event.message)
        if event.response_mode == "echo_last_tool":
            messages = request.get("messages")
            if not isinstance(messages, list):
                raise EpisodeReplayIneligibility("request messages are unavailable")
            tool_contents = [
                row.get("content")
                for row in messages
                if isinstance(row, dict) and row.get("role") == "tool"
            ]
            if not tool_contents or not isinstance(tool_contents[-1], str):
                raise EpisodeReplayIneligibility("no downstream tool result to echo")
            message["content"] = tool_contents[-1].strip()
        elif event.response_mode != "literal":
            raise EpisodeReplayIneligibility(
                f"unknown scripted response mode: {event.response_mode}"
            )
        with self._lock:
            self._completion_order.append(key)
        model = request.get("model")
        return {
            "id": f"redco-replay-{hashlib.sha256(key.encode()).hexdigest()[:16]}",
            "object": "chat.completion",
            "created": 0,
            "model": model if isinstance(model, str) else "scripted-replay",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": event.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": event.prompt_tokens,
                "completion_tokens": event.completion_tokens,
                "total_tokens": event.prompt_tokens + event.completion_tokens,
            },
        }

    def audit(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
            seen = set(self._requests)
            counts = dict(self._request_counts)
        expected = set(self._events)
        return {
            "expected_addresses": sorted(expected),
            "seen_addresses": sorted(seen),
            "missing_addresses": sorted(expected - seen),
            "unexpected_addresses": sorted(seen - expected),
            "requests": records,
            "request_counts": counts,
            "completion_order": list(self._completion_order),
            "root_session_id_present": self._root_session_id is not None,
            "complete": seen == expected and all(count == 1 for count in counts.values()),
        }

    def _validate_episode_scope(
        self,
        address: RLMEventAddress,
        headers: Mapping[str, str],
    ) -> RLMEventAddress:
        session_id = _required_header(headers, "X-RLM-Session-ID")
        with self._lock:
            if address.depth == 0:
                if self._root_session_id is None:
                    self._root_session_id = session_id
                elif session_id != self._root_session_id:
                    raise EpisodeReplayIneligibility(
                        "one scripted router cannot serve multiple root sessions"
                    )
                prior_root = self._session_lineages.setdefault(session_id, (0, "root"))
                if prior_root != (0, "root"):
                    raise EpisodeReplayIneligibility(
                        "root session ID is already bound to a recursive lineage"
                    )
                return address
            parent_session_id = _required_header(headers, "X-RLM-Parent-Session-ID")
            if self._root_session_id is None:
                raise EpisodeReplayIneligibility(
                    "a recursive request arrived before its root session"
                )
            parent_scope = self._session_lineages.get(parent_session_id)
            if parent_scope is None:
                raise EpisodeReplayIneligibility(
                    "recursive request is linked to an unknown parent session"
                )
            parent_depth, parent_lineage = parent_scope
            if address.depth != parent_depth + 1:
                raise EpisodeReplayIneligibility(
                    "recursive request depth is inconsistent with its parent session"
                )
            assert address.invocation_id is not None
            assert address.parent_turn is not None
            assert address.parent_tool_call_id is not None
            lineage = _child_lineage(
                parent_lineage=parent_lineage,
                depth=address.depth,
                parent_turn=address.parent_turn,
                parent_tool_call_id=address.parent_tool_call_id,
                invocation_id=address.invocation_id,
            )
            prior = self._session_lineages.setdefault(session_id, (address.depth, lineage))
            if prior != (address.depth, lineage):
                raise EpisodeReplayIneligibility(
                    "one session ID cannot identify multiple recursive lineages"
                )
            return replace(address, parent_lineage=parent_lineage)


class CounterfactualCompletionRouter:
    """Reuse exact recorded actions and resample every changed downstream call."""

    def __init__(
        self,
        events: tuple[ScriptedEvent, ...],
        *,
        target: RLMEventAddress,
        candidate_message: Mapping[str, Any],
        candidate_finish_reason: str,
        candidate_prompt_tokens: int,
        candidate_completion_tokens: int,
        master_seed: str,
        trace_id: str,
        target_id: str,
        generator: Callable[[dict[str, Any], RLMEventAddress, int], dict[str, Any]],
    ) -> None:
        if target.depth == 0:
            raise ValueError("counterfactual target must be recursive")
        if not master_seed:
            raise ValueError("master_seed must be nonempty")
        if not trace_id or not target_id:
            raise ValueError("trace_id and target_id must be nonempty")
        if not candidate_finish_reason:
            raise ValueError("candidate finish reason must be nonempty")
        if candidate_prompt_tokens < 0 or candidate_completion_tokens < 0:
            raise ValueError("candidate token counts must be nonnegative")
        self._events = {event.address.key(): event for event in events}
        if len(self._events) != len(events):
            raise ValueError("recorded event addresses must be unique")
        if target.key() not in self._events:
            raise EpisodeReplayIneligibility("target is absent from recorded events")
        self._target = target
        self._candidate_message = _openai_message(candidate_message)
        self._candidate_finish_reason = candidate_finish_reason
        self._candidate_prompt_tokens = candidate_prompt_tokens
        self._candidate_completion_tokens = candidate_completion_tokens
        self._master_seed = master_seed
        self._trace_id = trace_id
        self._target_id = target_id
        self._generator = generator
        self._scope = ScriptedCompletionRouter(events)
        self._records: list[dict[str, Any]] = []
        self._counts: dict[str, int] = {}
        self._seed_slots: dict[tuple[int, int, str], str] = {}
        self._tainted_sessions: set[str] = set()
        self._intervention_served = False
        self._lock = threading.Lock()

    def respond(
        self,
        *,
        headers: Mapping[str, str],
        request: dict[str, Any],
    ) -> dict[str, Any]:
        address = self._scope._validate_episode_scope(
            RLMEventAddress.from_headers(headers), headers
        )
        key = address.key()
        source = self._events.get(key)
        request_sha256 = _request_projection_sha256(
            request,
            normalize_transport_paths=(
                source.engineering_transport_path_normalization if source is not None else False
            ),
        )
        with self._lock:
            count = self._counts.get(key, 0) + 1
            self._counts[key] = count
        if count != 1:
            raise EpisodeReplayIneligibility(f"counterfactual event appeared more than once: {key}")

        if address == self._target:
            if source is None or source.expected_request_sha256 != request_sha256:
                raise EpisodeReplayIneligibility(
                    "target prompt differs from the committed recorded target"
                )
            target_session_id = _required_header(headers, "X-RLM-Session-ID")
            with self._lock:
                self._intervention_served = True
                self._tainted_sessions.add(target_session_id)
            response = _completion_payload(
                key=key,
                request=request,
                message=self._candidate_message,
                finish_reason=self._candidate_finish_reason,
                prompt_tokens=self._candidate_prompt_tokens,
                completion_tokens=self._candidate_completion_tokens,
            )
            mode = "committed_intervention"
            seed = None
        elif (
            source is not None
            and source.expected_request_sha256 == request_sha256
            and not self._is_causally_tainted(address, headers)
        ):
            response = _completion_payload(
                key=key,
                request=request,
                message=source.message,
                finish_reason=source.finish_reason,
                prompt_tokens=source.prompt_tokens,
                completion_tokens=source.completion_tokens,
            )
            mode = "recorded_action_transport_reuse"
            seed = None
        else:
            if not self._intervention_served:
                raise EpisodeReplayIneligibility(
                    f"unrecorded or changed event occurred before intervention: {key}"
                )
            slot = (address.depth, address.turn, address.call_kind)
            with self._lock:
                prior_key = self._seed_slots.setdefault(slot, key)
            if prior_key != key:
                raise EpisodeReplayIneligibility(
                    f"counterfactual topology has multiple calls in one stable seed slot: {slot}"
                )
            seed = _counterfactual_seed(
                self._master_seed,
                trace_id=self._trace_id,
                target_id=self._target_id,
                address=address,
            )
            forwarded = deepcopy(request)
            forwarded["seed"] = seed
            response = self._generator(forwarded, address, seed)
            _validate_completion_response(response)
            mode = "counterfactual_resample"

        with self._lock:
            self._records.append(
                {
                    "address": key,
                    "mode": mode,
                    "request_projection_sha256": request_sha256,
                    "recorded_address": source is not None,
                    "normalized_transport_request_match": (
                        source is not None and source.expected_request_sha256 == request_sha256
                    ),
                    "recorded_checkpoint_id": (
                        source.checkpoint_id if source is not None else None
                    ),
                    "recorded_decoding_config_hash": (
                        source.decoding_config_hash if source is not None else None
                    ),
                    "recorded_event_seed": (source.event_seed if source is not None else None),
                    "recorded_prompt_token_ids_sha256": (
                        source.prompt_token_ids_sha256 if source is not None else None
                    ),
                    "recorded_action_token_ids_sha256": (
                        source.action_token_ids_sha256 if source is not None else None
                    ),
                    "seed": seed,
                }
            )
        return response

    def _is_causally_tainted(
        self,
        address: RLMEventAddress,
        headers: Mapping[str, str],
    ) -> bool:
        """Track intervention influence through root turns and recursive sessions."""
        with self._lock:
            if not self._intervention_served:
                return False
            session_id = _required_header(headers, "X-RLM-Session-ID")
            parent_session_id = _optional_header(headers, "X-RLM-Parent-Session-ID")
            inherited = (
                parent_session_id is not None and parent_session_id in self._tainted_sessions
            )
            tainted = (
                session_id in self._tainted_sessions
                or inherited
                or _is_causally_downstream(address, self._target)
            )
            if tainted:
                self._tainted_sessions.add(session_id)
            return tainted

    def audit(self) -> dict[str, Any]:
        with self._lock:
            records = deepcopy(self._records)
            seen = set(self._counts)
        expected = set(self._events)
        resampled = [row for row in records if row["mode"] == "counterfactual_resample"]
        return {
            "target": self._target.key(),
            "intervention_served": self._intervention_served,
            "records": records,
            "seen_addresses": sorted(seen),
            "missing_recorded_addresses": sorted(expected - seen),
            "dynamic_addresses": sorted(seen - expected),
            "resampled_policy_calls": len(resampled),
            "topology_diverged": bool((expected - seen) or (seen - expected)),
            "tainted_session_count": len(self._tainted_sessions),
            "valid_counterfactual": self._intervention_served,
        }


class HTTPCompletionGenerator:
    """Forward changed RLM requests to an OpenAI-compatible frozen policy."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        temperature: float,
        max_tokens: int,
    ) -> None:
        if timeout_seconds <= 0 or max_tokens <= 0:
            raise ValueError("timeout and max_tokens must be positive")
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/v1"):
            self.url += "/v1"
        self.url += "/chat/completions"
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(
        self,
        request: dict[str, Any],
        address: RLMEventAddress,
        seed: int,
    ) -> dict[str, Any]:
        del address
        body = deepcopy(request)
        body.update(
            {
                "seed": seed,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
        )
        wire = urllib.request.Request(
            self.url,
            data=json.dumps(body, separators=(",", ":")).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(wire, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"counterfactual generation failed with HTTP {error.code}: {detail}"
            ) from error
        if not isinstance(payload, dict):
            raise TypeError("counterfactual completion must be an object")
        return payload


class _ReplayHTTPServer(ThreadingHTTPServer):
    router: Any


class _ReplayHandler(BaseHTTPRequestHandler):
    server: _ReplayHTTPServer

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            request = json.loads(body)
            if not isinstance(request, dict):
                raise TypeError("request must be an object")
            response = self.server.router.respond(
                headers=dict(self.headers.items()),
                request=request,
            )
        except (
            EpisodeReplayIneligibility,
            ReplayResamplingRequired,
            TypeError,
            ValueError,
        ) as error:
            encoded = json.dumps({"error": {"message": str(error)}}).encode()
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        encoded = json.dumps(response, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _: str, *args: object) -> None:
        del args


class ScriptedModelServer:
    def __init__(self, router: Any) -> None:
        self._server = _ReplayHTTPServer(("127.0.0.1", 0), _ReplayHandler)
        self._server.router = router
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="redco-scripted-model",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        address = self._server.server_address
        raw_host, raw_port = address[0], address[1]
        host = raw_host.decode() if isinstance(raw_host, bytes) else raw_host
        port = int(raw_port)
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> ScriptedModelServer:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)
        self._server.server_close()


def load_scripted_events(path: str) -> tuple[ScriptedEvent, ...]:
    with open(path, encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise TypeError("cassette must be an object")
    events = payload.get("events")
    if not isinstance(events, list):
        raise TypeError("cassette events must be a list")
    return tuple(ScriptedEvent.from_dict(_as_mapping(event)) for event in events)


def recorded_event_addresses(trace_path: Path) -> tuple[RLMEventAddress, ...]:
    """Require exact invocation provenance for every recorded recursive call."""
    audit = audit_trace_file(trace_path)
    addresses: list[RLMEventAddress] = []
    session_lineages: dict[str, tuple[int, str]] = {}
    root_session_id: str | None = None
    for call in audit.calls:
        if (
            call.agent_depth is None
            or call.turn_index is None
            or call.call_kind is None
            or call.session_id is None
        ):
            raise EpisodeReplayIneligibility("model call lacks RLM depth or turn")
        address = RLMEventAddress(
            depth=call.agent_depth,
            turn=call.turn_index,
            call_kind=call.call_kind,
            parent_turn=call.parent_turn_index,
            parent_tool_call_id=call.parent_tool_call_id,
            invocation_id=call.invocation_id,
        )
        if address.depth == 0:
            if root_session_id is None:
                root_session_id = call.session_id
            elif call.session_id != root_session_id:
                raise EpisodeReplayIneligibility("recorded trace contains multiple root sessions")
            prior = session_lineages.setdefault(call.session_id, (0, "root"))
            if prior != (0, "root"):
                raise EpisodeReplayIneligibility("recorded root session has conflicting lineage")
        else:
            if call.parent_session_id is None:
                raise EpisodeReplayIneligibility("recorded recursive call lacks parent session")
            parent_scope = session_lineages.get(call.parent_session_id)
            if parent_scope is None:
                raise EpisodeReplayIneligibility(
                    "recorded recursive call precedes its parent session"
                )
            parent_depth, parent_lineage = parent_scope
            if address.depth != parent_depth + 1:
                raise EpisodeReplayIneligibility(
                    "recorded recursive depth is inconsistent with parent session"
                )
            assert address.parent_turn is not None
            assert address.parent_tool_call_id is not None
            assert address.invocation_id is not None
            lineage = _child_lineage(
                parent_lineage=parent_lineage,
                depth=address.depth,
                parent_turn=address.parent_turn,
                parent_tool_call_id=address.parent_tool_call_id,
                invocation_id=address.invocation_id,
            )
            prior = session_lineages.setdefault(call.session_id, (address.depth, lineage))
            if prior != (address.depth, lineage):
                raise EpisodeReplayIneligibility("recorded session has conflicting lineage")
            address = replace(address, parent_lineage=parent_lineage)
        addresses.append(address)
    return tuple(addresses)


def trace_to_scripted_events(
    trace_path: Path,
    *,
    expected_sha256: str,
    signed_precommit: Mapping[str, Any],
    engineering_transport_path_normalization: bool = False,
) -> tuple[ScriptedEvent, ...]:
    """Build an exact structural cassette from one immutable native trace."""
    observed_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    if observed_sha256 != expected_sha256:
        raise EpisodeReplayIneligibility(
            "source trace SHA-256 does not match the frozen replay input"
        )
    from redco.integrations.signed_subprocess import verify_signed_payload

    precommit = dict(signed_precommit)
    verify_signed_payload(precommit)
    if precommit.get("source_trace_sha256") != observed_sha256:
        raise EpisodeReplayIneligibility("signed precommit is bound to a different source trace")
    traces = load_trace_records(trace_path)
    if len(traces) != 1:
        raise EpisodeReplayIneligibility("event replay requires exactly one recorded trace")
    trace = traces[0]
    raw_calls = trace.get("calls")
    nodes = trace.get("nodes")
    if not isinstance(raw_calls, list) or not isinstance(nodes, list):
        raise TypeError("trace calls and nodes must be lists")
    audit = audit_trace_file(trace_path)
    if len(audit.calls) != len(raw_calls):
        raise EpisodeReplayIneligibility(
            "every model call must be linked to an exact response node"
        )
    addresses = recorded_event_addresses(trace_path)
    events: list[ScriptedEvent] = []
    for address, extracted, raw_call in zip(addresses, audit.calls, raw_calls, strict=True):
        if not isinstance(raw_call, dict):
            raise TypeError("trace call must be an object")
        if extracted.call_index >= len(raw_calls):
            raise EpisodeReplayIneligibility("call order is inconsistent")
        if (
            extracted.agent_depth is None
            or extracted.turn_index is None
            or extracted.call_kind is None
        ):
            raise EpisodeReplayIneligibility("model call lacks an exact RLM event address")
        if extracted.node_index < 0 or extracted.node_index >= len(nodes):
            raise EpisodeReplayIneligibility("model call response node is unavailable")
        node = nodes[extracted.node_index]
        if not isinstance(node, dict) or not isinstance(node.get("message"), dict):
            raise EpisodeReplayIneligibility("model call response node lacks a typed message")
        if raw_call.get("error") is not None:
            raise EpisodeReplayIneligibility(
                "failed model calls are not eligible for exact event replay"
            )
        finish_reason = raw_call.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise EpisodeReplayIneligibility("model call lacks finish_reason")
        events.append(
            ScriptedEvent(
                address=address,
                message=_openai_message(node["message"]),
                finish_reason=finish_reason,
                prompt_tokens=extracted.prompt_tokens_reported or 0,
                completion_tokens=extracted.completion_tokens_reported or 0,
                expected_request_sha256=_recorded_request_sha256(
                    trace,
                    node_index=extracted.node_index,
                    model=extracted.checkpoint_id,
                    normalize_transport_paths=(engineering_transport_path_normalization),
                ),
                checkpoint_id=extracted.checkpoint_id,
                decoding_config_hash=extracted.decoding_config_hash,
                event_seed=extracted.event_seed,
                prompt_token_ids_sha256=hashlib.sha256(
                    canonical_json(extracted.prompt_token_ids)
                ).hexdigest(),
                action_token_ids_sha256=hashlib.sha256(
                    canonical_json(extracted.action_token_ids)
                ).hexdigest(),
                engineering_transport_path_normalization=(engineering_transport_path_normalization),
            )
        )
    if not events:
        raise EpisodeReplayIneligibility("trace contains no replayable model calls")
    keys = [event.address.key() for event in events]
    if len(set(keys)) != len(keys):
        raise EpisodeReplayIneligibility("trace contains duplicate structural event addresses")
    committed = _committed_child_addresses(precommit)
    recorded_children = {
        event.address.key()
        for event in events
        if event.address.depth == 1 and event.address.call_kind == "policy"
    }
    if committed != recorded_children:
        raise EpisodeReplayIneligibility(
            "signed precommit does not cover the exact recorded child event set"
        )
    return tuple(events)


def inject_committed_child_answer(
    events: tuple[ScriptedEvent, ...],
    *,
    signed_precommit: Mapping[str, Any],
    candidate_rank: int,
    answer: str,
) -> tuple[tuple[ScriptedEvent, ...], RLMEventAddress, dict[str, Any]]:
    """Inject only a target selected from the verified canonical precommit."""
    from redco.integrations.signed_subprocess import verify_signed_payload

    precommit = dict(signed_precommit)
    verify_signed_payload(precommit)
    candidates = precommit.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise EpisodeReplayIneligibility("precommit candidates are unavailable")
    if candidate_rank < 0 or candidate_rank >= len(candidates):
        raise EpisodeReplayIneligibility("candidate rank is outside the precommit")
    candidate = candidates[candidate_rank]
    if not isinstance(candidate, dict):
        raise TypeError("precommit candidate must be an object")
    target = _candidate_address(candidate)
    if target.key() not in _committed_child_addresses(precommit):
        raise EpisodeReplayIneligibility("target is not a committed child event")
    updated = inject_child_answer(events, target=target, answer=answer)
    return updated, target, deepcopy(candidate)


def inject_child_answer(
    events: tuple[ScriptedEvent, ...],
    *,
    target: RLMEventAddress,
    answer: str,
) -> tuple[ScriptedEvent, ...]:
    """Replace exactly one child completion by its structural event address."""
    if target.depth == 0:
        raise ValueError("the intervention target must be recursive")
    if not isinstance(answer, str):
        raise TypeError("candidate child answer must be a string")
    matches = [index for index, event in enumerate(events) if event.address == target]
    if len(matches) != 1:
        raise EpisodeReplayIneligibility(f"target address matched {len(matches)} scripted events")
    index = matches[0]
    message = dict(events[index].message)
    if message.get("role") != "assistant" or message.get("tool_calls"):
        raise EpisodeReplayIneligibility(
            "typed child-answer injection requires a terminal assistant event"
        )
    message["content"] = answer
    updated = list(events)
    updated[index] = replace(events[index], message=message)
    return tuple(updated)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("expected an object")
    return value


def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _as_mapping(payload.get(name))


def _string(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def _integer(payload: Mapping[str, Any], name: str) -> int:
    value = payload.get(name)
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be a nonnegative integer")
    return value


def _optional_integer(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise TypeError("optional integer must be nonnegative")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError("optional string must be nonempty")
    return value


def _optional_sha256(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise TypeError("optional SHA-256 must be a 64-character string")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise TypeError("optional SHA-256 must be hexadecimal") from error
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _request_projection(
    request: Mapping[str, Any], *, normalize_transport_paths: bool = False
) -> dict[str, Any]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise EpisodeReplayIneligibility("request messages are unavailable")
    projection = deepcopy(dict(request))
    projection["messages"] = [
        _openai_message(message, normalize_transport_paths=normalize_transport_paths)
        for message in messages
    ]
    tools = request.get("tools")
    if tools is not None:
        projection["tools"] = _openai_tools(tools)
    return projection


def _request_projection_sha256(
    request: Mapping[str, Any], *, normalize_transport_paths: bool = False
) -> str:
    return hashlib.sha256(
        canonical_json(
            _request_projection(request, normalize_transport_paths=normalize_transport_paths)
        )
    ).hexdigest()


def _recorded_request_sha256(
    trace: Mapping[str, Any],
    *,
    node_index: int,
    model: str,
    normalize_transport_paths: bool,
) -> str:
    nodes = trace.get("nodes")
    if not isinstance(nodes, list):
        raise TypeError("trace nodes must be a list")
    from redco.integrations.verifiers_trace import path_to_node

    path = path_to_node(nodes, node_index)
    messages = []
    for index in path[:-1]:
        node = nodes[index]
        if not isinstance(node, dict) or not isinstance(node.get("message"), dict):
            raise EpisodeReplayIneligibility("recorded prompt path contains an untyped message")
        messages.append(
            _openai_message(
                node["message"],
                normalize_transport_paths=normalize_transport_paths,
            )
        )
    request: dict[str, Any] = {"model": model, "messages": messages}
    tools = trace.get("tools")
    if tools:
        request["tools"] = _openai_tools(tools)
        request["parallel_tool_calls"] = False
    return _request_projection_sha256(request, normalize_transport_paths=normalize_transport_paths)


def _openai_tools(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("tools must be a list")
    normalized: list[dict[str, Any]] = []
    for tool in value:
        if not isinstance(tool, dict):
            raise TypeError("tool must be an object")
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            normalized.append(deepcopy(tool))
        else:
            normalized.append({"type": "function", "function": deepcopy(tool)})
    return normalized


def _candidate_address(candidate: Mapping[str, Any]) -> RLMEventAddress:
    depth = _integer(candidate, "agent_depth")
    parent_lineage = _optional_string(candidate.get("parent_lineage"))
    if depth == 1 and parent_lineage is None:
        parent_lineage = "root"
    return RLMEventAddress(
        depth=depth,
        turn=_integer(candidate, "turn_index"),
        call_kind=_string(candidate, "call_kind"),
        parent_lineage=parent_lineage,
        parent_turn=_optional_integer(candidate.get("parent_turn_index")),
        parent_tool_call_id=_optional_string(candidate.get("parent_tool_call_id")),
        invocation_id=_optional_string(candidate.get("invocation_id")),
    )


def _committed_child_addresses(precommit: Mapping[str, Any]) -> set[str]:
    candidates = precommit.get("candidates")
    if not isinstance(candidates, list):
        raise EpisodeReplayIneligibility("precommit candidates must be a list")
    addresses = {
        _candidate_address(candidate).key()
        for candidate in candidates
        if isinstance(candidate, dict)
    }
    if len(addresses) != len(candidates):
        raise EpisodeReplayIneligibility(
            "precommit contains duplicate or malformed child addresses"
        )
    return addresses


def _completion_payload(
    *,
    key: str,
    request: Mapping[str, Any],
    message: Mapping[str, Any],
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    model = request.get("model")
    return {
        "id": f"redco-replay-{hashlib.sha256(key.encode()).hexdigest()[:16]}",
        "object": "chat.completion",
        "created": 0,
        "model": model if isinstance(model, str) else "redco-replay",
        "choices": [
            {
                "index": 0,
                "message": deepcopy(dict(message)),
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _validate_completion_response(response: Mapping[str, Any]) -> None:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise EpisodeReplayIneligibility("counterfactual response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise EpisodeReplayIneligibility("counterfactual response lacks a typed assistant message")


def _counterfactual_seed(
    master_seed: str,
    *,
    trace_id: str,
    target_id: str,
    address: RLMEventAddress,
) -> int:
    # Do not condition continuation randomness on model-generated tool-call or
    # invocation IDs. The stable slot is the typed depth/turn role.
    namespace = canonical_json(
        {
            "trace_id": trace_id,
            "target_id": target_id,
            "continuation_role": address.call_kind,
            "depth": address.depth,
            "ordinal_call_slot": address.turn,
        }
    )
    digest = hmac.new(master_seed.encode(), namespace, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _is_causally_downstream(address: RLMEventAddress, target: RLMEventAddress) -> bool:
    """Classify later root turns; recursive causality is session-lineage based."""
    if target.parent_turn is None:
        raise EpisodeReplayIneligibility("target lacks its committed parent turn")
    return address.depth == 0 and address.turn > target.parent_turn


def _openai_message(value: Any, *, normalize_transport_paths: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("trace message must be an object")
    message = deepcopy(value)
    if (
        normalize_transport_paths
        and message.get("role") == "system"
        and isinstance(message.get("content"), str)
    ):
        message["content"] = _normalize_volatile_system_paths(message["content"])
    # Verifiers keeps the executing tool name as trace metadata, but the pinned
    # OpenAI client sends tool results using the tool_call_id alone.
    if message.get("role") == "tool":
        message.pop("name", None)
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        return message
    if not isinstance(raw_calls, list):
        raise TypeError("trace tool_calls must be a list")
    normalized: list[dict[str, Any]] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            raise TypeError("trace tool call must be an object")
        if isinstance(raw.get("function"), dict):
            normalized.append(raw)
            continue
        call_id = raw.get("id")
        name = raw.get("name")
        arguments = raw.get("arguments")
        if not all(isinstance(item, str) and item for item in (call_id, name)):
            raise EpisodeReplayIneligibility("trace tool call lacks an exact ID or function name")
        if not isinstance(arguments, str):
            raise EpisodeReplayIneligibility("trace tool call arguments are not serialized JSON")
        normalized.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    message["tool_calls"] = normalized
    # The pinned RLM client serializes an assistant tool-call message with an
    # empty content string, while the native Verifiers trace stores the same
    # wire-semantic field as null.  Normalize only this typed tool-call case.
    if message.get("role") == "assistant" and message.get("content") is None:
        message["content"] = ""
    return message


def _normalize_volatile_system_paths(content: str) -> str:
    """Normalize only RLM's documented per-sandbox path fields."""
    normalized: list[str] = []
    for line in content.splitlines(keepends=True):
        ending = "\n" if line.endswith("\n") else ""
        body = line.removesuffix("\n")
        if body.startswith("Working directory: "):
            body = "Working directory: <RLM_WORKDIR>"
        elif body.startswith("Conversation log: "):
            body = "Conversation log: <RLM_CONVERSATION_LOG>"
        normalized.append(body + ending)
    return "".join(normalized)
