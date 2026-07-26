"""Dependency-free reader for pinned verifiers.v1 ``traces.jsonl`` records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json
from redco.env.policy_cache import (
    CachedPolicyAction,
    PolicyActionCache,
    PolicyCallKey,
)


@dataclass(frozen=True, slots=True)
class RecordedPolicyCall:
    trace_id: str
    call_index: int
    node_index: int
    component_root_node: int
    prompt_token_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]
    checkpoint_id: str
    decoding_config_hash: str
    event_seed: int | None
    prompt_tokens_reported: int | None
    completion_tokens_reported: int | None
    cost_reported: float | None
    wall_seconds: float
    agent_depth: int | None
    session_id: str | None
    turn_index: int | None
    call_kind: str | None
    parent_session_id: str | None
    parent_turn_index: int | None

    @property
    def exact_key_complete(self) -> bool:
        return (
            bool(self.prompt_token_ids)
            and bool(self.action_token_ids)
            and self.checkpoint_id != "unknown"
            and self.event_seed is not None
        )

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.prompt_token_ids)).hexdigest()


@dataclass(frozen=True, slots=True)
class TraceAuditReport:
    schema_version: int
    source: str
    episode_count: int
    successful_episode_count: int
    episode_error_count: int
    trace_count: int
    successful_trace_count: int
    trace_error_count: int
    model_call_count: int
    linked_call_count: int
    failed_model_call_count: int
    exact_prompt_action_count: int
    exact_key_complete_count: int
    seed_coverage: float
    usage_coverage: float
    connected_components: int
    message_graph_leaves: int
    recursive_trace_count: int
    native_audit_trace_count: int
    calls: tuple[RecordedPolicyCall, ...]

    @property
    def exact_prompt_action_coverage(self) -> float:
        if not self.model_call_count:
            return 0.0
        return self.exact_prompt_action_count / self.model_call_count

    @property
    def ready_for_exact_key_replay(self) -> bool:
        return (
            self.model_call_count > 0
            and self.exact_key_complete_count == self.model_call_count
        )

    @property
    def has_recursive_model_calls(self) -> bool:
        return self.recursive_trace_count > 0

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["exact_prompt_action_coverage"] = (
            self.exact_prompt_action_coverage
        )
        payload["ready_for_exact_key_replay"] = self.ready_for_exact_key_replay
        payload["has_recursive_model_calls"] = self.has_recursive_model_calls
        return payload


def load_trace_records(path: Path) -> list[dict[str, Any]]:
    """Read bare traces or episode-wrapped traces without importing verifiers."""
    return _load_trace_file(path).traces


@dataclass(frozen=True, slots=True)
class _TraceFileRecords:
    traces: list[dict[str, Any]]
    episode_count: int
    successful_episode_count: int
    episode_error_count: int


def _load_trace_file(path: Path) -> _TraceFileRecords:
    traces: list[dict[str, Any]] = []
    episode_count = 0
    successful_episodes = 0
    episode_errors = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"trace row {line_number} must be an object")
            nested = row.get("traces")
            if nested is None:
                traces.append(row)
                episode_count += 1
                successful_episodes += row.get("ok") is True
                errors = row.get("errors", [])
                if not isinstance(errors, list):
                    raise TypeError(f"trace row {line_number} errors must be a list")
                episode_errors += len(errors)
                continue
            if not isinstance(nested, list):
                raise TypeError(
                    f"episode row {line_number} traces must be a list"
                )
            episode_count += 1
            successful_episodes += row.get("ok") is True
            errors = row.get("errors", [])
            if not isinstance(errors, list):
                raise TypeError(
                    f"episode row {line_number} errors must be a list"
                )
            episode_errors += len(errors)
            for trace in nested:
                if not isinstance(trace, dict):
                    raise TypeError(
                        f"episode row {line_number} contains a non-object trace"
                    )
                traces.append(trace)
    return _TraceFileRecords(
        traces=traces,
        episode_count=episode_count,
        successful_episode_count=successful_episodes,
        episode_error_count=episode_errors,
    )


def extract_policy_calls(trace: dict[str, Any]) -> tuple[RecordedPolicyCall, ...]:
    """Recover the exact prompt/action token split for each linked model call."""
    nodes = _require_object_list(trace, "nodes")
    calls = _require_object_list(trace, "calls")
    trace_id = str(trace.get("id") or "unknown")
    agent = trace.get("agent")
    agent_model = (
        str(agent.get("model"))
        if isinstance(agent, dict) and agent.get("model")
        else "unknown"
    )
    extracted: list[RecordedPolicyCall] = []
    for call_index, call in enumerate(calls):
        node_value = call.get("node")
        if type(node_value) is not int:
            continue
        node_index = node_value
        path = path_to_node(nodes, node_index)
        current = nodes[node_index]
        current_tokens = _integer_list(
            current.get("token_ids"),
            f"node {node_index} token_ids",
        )
        mask = _boolean_list(current.get("mask"), f"node {node_index} mask")
        if len(current_tokens) != len(mask):
            raise ValueError(
                f"node {node_index} token_ids and mask lengths differ"
            )
        first_sampled = next(
            (index for index, sampled in enumerate(mask) if sampled),
            len(mask),
        )
        if any(not sampled for sampled in mask[first_sampled:]):
            raise ValueError(
                f"node {node_index} has non-suffix sampled-token mask"
            )
        prompt_tokens: list[int] = []
        for prior_index in path[:-1]:
            prompt_tokens.extend(
                _integer_list(
                    nodes[prior_index].get("token_ids"),
                    f"node {prior_index} token_ids",
                )
            )
        prompt_tokens.extend(current_tokens[:first_sampled])
        action_tokens = current_tokens[first_sampled:]

        sampling = call.get("sampling")
        sampling_payload = dict(sampling) if isinstance(sampling, dict) else {}
        seed_value = sampling_payload.pop("seed", None)
        event_seed = (
            seed_value
            if type(seed_value) is int and seed_value >= 0
            else None
        )
        decoding_hash = hashlib.sha256(
            canonical_json(sampling_payload)
        ).hexdigest()
        usage = call.get("usage")
        usage_payload = usage if isinstance(usage, dict) else {}
        timing = call.get("time")
        timing_payload = timing if isinstance(timing, dict) else {}
        start = _number_or_none(timing_payload.get("start")) or 0.0
        end = _number_or_none(timing_payload.get("end")) or 0.0
        model = str(call.get("model") or agent_model)
        rlm = call.get("rlm")
        rlm_payload = rlm if isinstance(rlm, dict) else {}
        extracted.append(
            RecordedPolicyCall(
                trace_id=trace_id,
                call_index=call_index,
                node_index=node_index,
                component_root_node=path[0],
                prompt_token_ids=tuple(prompt_tokens),
                action_token_ids=tuple(action_tokens),
                checkpoint_id=model,
                decoding_config_hash=decoding_hash,
                event_seed=event_seed,
                prompt_tokens_reported=_integer_or_none(
                    usage_payload.get("prompt_tokens")
                ),
                completion_tokens_reported=_integer_or_none(
                    usage_payload.get("completion_tokens")
                ),
                cost_reported=_number_or_none(usage_payload.get("cost")),
                wall_seconds=max(0.0, end - start) if end else 0.0,
                agent_depth=_nonnegative_integer_or_none(
                    rlm_payload.get("depth")
                ),
                session_id=_nonempty_string_or_none(
                    rlm_payload.get("session_id")
                ),
                turn_index=_nonnegative_integer_or_none(
                    rlm_payload.get("turn")
                ),
                call_kind=_nonempty_string_or_none(
                    rlm_payload.get("call_kind")
                ),
                parent_session_id=_nonempty_string_or_none(
                    rlm_payload.get("parent_session_id")
                ),
                parent_turn_index=_nonnegative_integer_or_none(
                    rlm_payload.get("parent_turn")
                ),
            )
        )
    return tuple(extracted)


def audit_trace_file(path: Path) -> TraceAuditReport:
    records = _load_trace_file(path)
    traces = records.traces
    recorded_calls: list[RecordedPolicyCall] = []
    model_calls = 0
    linked_calls = 0
    usage_calls = 0
    successful_traces = 0
    trace_errors = 0
    failed_calls = 0
    connected_components = 0
    leaves = 0
    recursive_traces = 0
    native_audit_traces = 0
    for trace in traces:
        nodes = _require_object_list(trace, "nodes")
        calls = _require_object_list(trace, "calls")
        successful_traces += trace.get("ok") is True
        errors = trace.get("errors", [])
        if not isinstance(errors, list):
            raise TypeError("errors must be a list")
        trace_errors += len(errors)
        model_calls += len(calls)
        linked_calls += sum(type(call.get("node")) is int for call in calls)
        usage_calls += sum(isinstance(call.get("usage"), dict) for call in calls)
        failed_calls += sum(isinstance(call.get("error"), dict) for call in calls)
        components = sum(node.get("parent") is None for node in nodes)
        connected_components += components
        recursive_traces += components > 1
        parent_ids = {
            parent
            for node in nodes
            if type(parent := node.get("parent")) is int
        }
        leaves += sum(index not in parent_ids for index in range(len(nodes)))
        info = trace.get("info")
        if (
            isinstance(info, dict)
            and isinstance(info.get("redco_trace_audit"), dict)
        ):
            native_audit_traces += 1
        recorded_calls.extend(extract_policy_calls(trace))

    exact_prompt_actions = sum(
        bool(call.prompt_token_ids) and bool(call.action_token_ids)
        for call in recorded_calls
    )
    exact_keys = sum(call.exact_key_complete for call in recorded_calls)
    seeded = sum(call.event_seed is not None for call in recorded_calls)
    return TraceAuditReport(
        schema_version=1,
        source=path.as_posix(),
        episode_count=records.episode_count,
        successful_episode_count=records.successful_episode_count,
        episode_error_count=records.episode_error_count,
        trace_count=len(traces),
        successful_trace_count=successful_traces,
        trace_error_count=trace_errors,
        model_call_count=model_calls,
        linked_call_count=linked_calls,
        failed_model_call_count=failed_calls,
        exact_prompt_action_count=exact_prompt_actions,
        exact_key_complete_count=exact_keys,
        seed_coverage=(seeded / model_calls) if model_calls else 0.0,
        usage_coverage=(usage_calls / model_calls) if model_calls else 0.0,
        connected_components=connected_components,
        message_graph_leaves=leaves,
        recursive_trace_count=recursive_traces,
        native_audit_trace_count=native_audit_traces,
        calls=tuple(recorded_calls),
    )


def build_policy_cache(
    calls: tuple[RecordedPolicyCall, ...],
) -> PolicyActionCache:
    """Build the frozen exact-key action table, failing on incomplete calls."""
    cache = PolicyActionCache()
    for call in calls:
        if not call.exact_key_complete or call.event_seed is None:
            raise ValueError(
                f"call {call.trace_id}:{call.call_index} lacks an exact replay key"
            )
        key = PolicyCallKey.from_call(
            call.prompt_token_ids,
            checkpoint_id=call.checkpoint_id,
            decoding_config_hash=call.decoding_config_hash,
            event_seed=call.event_seed,
        )
        cache.record(CachedPolicyAction(key, call.action_token_ids))
    return cache


def path_to_node(nodes: list[dict[str, Any]], node_index: int) -> list[int]:
    """Return a root-to-node path, rejecting missing parents and cycles."""
    if node_index < 0 or node_index >= len(nodes):
        raise ValueError(f"call links to unavailable node {node_index}")
    reversed_path: list[int] = []
    seen: set[int] = set()
    current: int | None = node_index
    while current is not None:
        if current in seen:
            raise ValueError(f"message graph cycle at node {current}")
        if current < 0 or current >= len(nodes):
            raise ValueError(f"node parent is unavailable: {current}")
        seen.add(current)
        reversed_path.append(current)
        parent = nodes[current].get("parent")
        if parent is not None and type(parent) is not int:
            raise TypeError(f"node {current} parent must be int or null")
        current = parent
    return list(reversed(reversed_path))


def _require_object_list(
    payload: dict[str, Any],
    key: str,
) -> list[dict[str, Any]]:
    value = payload.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"{key} must be a list of objects")
    return value


def _integer_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise TypeError(f"{label} must be a list of integers")
    return value


def _boolean_list(value: Any, label: str) -> list[bool]:
    if not isinstance(value, list) or any(type(item) is not bool for item in value):
        raise TypeError(f"{label} must be a list of booleans")
    return value


def _integer_or_none(value: Any) -> int | None:
    return value if type(value) is int else None


def _nonnegative_integer_or_none(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _nonempty_string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
