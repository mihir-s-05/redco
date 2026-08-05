"""Frozen oracle and production-boundary fixture for the future repair.

The directional rule remains test-only until the comparator repair.  The binding
fixture below nevertheless invokes the real production trace-to-source boundary,
so the matrix cannot become disconnected from the sampled-response path.
"""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace
from typing import Any, cast

from test_stage_d_source_producer import _episode, _prepared_action

from redco.analysis.stage_d_source_contracts import RolloutDecision
from redco.analysis.stage_d_source_producer import _verify_trace_call, derive_source_trace
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress
from redco.contracts import canonical_json
from redco.integrations.verifiers_trace_v2 import extract_v2_rlm_provenance

PRODUCTION_BOUNDARY_CONTRACT = {
    "version": "v1",
    "message_hook": ("tests/stage_d_source_comparison_oracle.py:production_boundary_observation"),
    "message_cases_bound": True,
    "record_hook": (
        "tests/stage_d_source_comparison_oracle.py:record_exactness_binding_observation"
    ),
    "record_cases_bound": False,
    "record_cases_policy": (
        "frozen exactness vectors only; production record binding is required before "
        "a repaired comparator is eligible"
    ),
    "raw_bytes_passthrough": True,
}

FUTURE_PRODUCTION_BINDING = {
    "version": "v1",
    "callable": "redco.analysis.stage_d_source_producer:_verify_trace_call",
    "fixture": "tests/stage_d_source_comparison_oracle.py:production_boundary_observation",
    "required_before_repair_green": True,
    "currently_bound": False,
    "binding_site": (
        "_verify_trace_call sampled-response transport-versus-persisted-trace comparison"
    ),
    "raw_forms": "canonical JSON bytes are retained before production parsing",
    "record_binding_hook": (
        "tests/stage_d_source_comparison_oracle.py:record_exactness_binding_observation"
    ),
    "record_cases_bound": False,
    "record_cases_policy": "frozen exactness vectors until a real production hook is supplied",
}


def production_boundary_observation(
    transport: dict[str, Any], trace: dict[str, Any]
) -> dict[str, Any]:
    """Invoke production source derivation with untouched transport/trace forms.

    This is a versioned test seam, not a comparator repair.  It records the raw
    canonical message bytes, invokes the real ``_verify_trace_call`` comparison
    boundary with a durable provenance record and ``BehaviorAction``, and then
    traverses ``derive_source_trace``.  The current production comparator
    therefore remains strict for the permitted null/absent case.
    """
    transport_bytes = canonical_json(transport)
    trace_bytes = canonical_json(trace)
    episode = json.loads(_episode())
    source_trace = episode["traces"][0]
    production_record = extract_v2_rlm_provenance(source_trace)[0]
    production_address = production_record.scientific_address
    source_trace["nodes"] = source_trace["nodes"][:2]
    source_trace["calls"] = source_trace["calls"][:1]
    source_trace["nodes"][1]["message"] = json.loads(trace_bytes)
    raw_episode = canonical_json(episode)
    action = _prepared_action(71, message=json.loads(transport_bytes))
    action_bytes = action.to_bytes()
    action_payload = json.loads(action_bytes)["action"]
    if action_payload["raw_transport_message"] != json.loads(transport_bytes):
        raise AssertionError("production fixture changed raw transport message bytes")
    if source_trace["nodes"][1]["message"] != json.loads(trace_bytes):
        raise AssertionError("production fixture changed raw trace message bytes")
    decision = SimpleNamespace(
        decision_id="oracle-production-root",
        event_address=PolicyEventAddress(0, "root", 0, 0),
        action=action,
        node_kind="root",
        target_id=None,
        target_ordinal=None,
    )
    try:
        _verify_trace_call(
            source_trace,
            source_trace["nodes"],
            source_trace["calls"][0],
            call_index=0,
            address=production_address,
            record=production_record,
            decision=cast(Any, decision),
            rollout_id=source_trace["id"],
        )
        derive_source_trace(
            raw_episode,
            decisions=(cast(RolloutDecision, decision),),
        )
    except ValueError:
        accepted = False
    else:
        accepted = True
    return {
        "accepted": accepted,
        "hook_version": "v1",
        "boundary_kind": "message",
        "transport_bytes": transport_bytes,
        "trace_bytes": trace_bytes,
        "transport_sha256": sha256(transport_bytes).hexdigest(),
        "trace_sha256": sha256(trace_bytes).hexdigest(),
        "production_callable": "redco.analysis.stage_d_source_producer:_verify_trace_call",
    }


def record_exactness_binding_observation(
    transport: dict[str, Any], trace: dict[str, Any]
) -> dict[str, Any]:
    """Freeze record vectors without claiming a production record binding.

    The current production comparator has no isolated record boundary.  This
    explicit fail-closed hook therefore records unchanged raw vectors and
    returns ``not_bound_pre_repair``.  A future repair must replace this
    binding with a real production record/action route; the frozen vectors and
    their expected exactness results must remain unchanged.
    """
    transport_bytes = canonical_json(transport)
    trace_bytes = canonical_json(trace)
    return {
        "hook_version": "v1",
        "boundary_kind": "record_exactness_only",
        "bound": False,
        "status": "not_bound_pre_repair",
        "transport_bytes": transport_bytes,
        "trace_bytes": trace_bytes,
        "transport_sha256": sha256(transport_bytes).hexdigest(),
        "trace_sha256": sha256(trace_bytes).hexdigest(),
        "reason": (
            "record-field vectors are frozen exactness cases; no current production "
            "sampled-response record seam exists"
        ),
    }


def directional_message_match(transport: dict[str, Any], trace: dict[str, Any]) -> bool:
    """Implement only the versioned future directional rule in the oracle."""
    if bool(canonical_json(transport) == canonical_json(trace)):
        return True
    if transport.get("role") != "assistant" or trace.get("role") != "assistant":
        return False
    if "tool_calls" in transport or "tool_calls" in trace:
        return False
    if set(transport) - {"role", "content"} != set(trace) - {"role"}:
        return False
    if transport.get("content") is not None or "content" not in transport:
        return False
    if "content" in trace:
        return False
    left = dict(transport)
    left.pop("content")
    return bool(canonical_json(left) == canonical_json(trace))


def exact_record_match(transport: dict[str, Any], trace: dict[str, Any]) -> bool:
    """Require every non-message/canonicalization field to remain byte-exact."""
    return bool(canonical_json(transport) == canonical_json(trace))


def _message(role: str = "assistant", content: object = None) -> dict[str, Any]:
    return {"role": role, "content": content}


def _base_record() -> dict[str, Any]:
    return {
        "request": {
            "messages": [{"role": "user", "content": "q"}],
            "tools": [{"type": "function", "function": {"name": "ipython"}}],
        },
        "request_context": {
            "parent_messages": [{"role": "user", "content": "q"}],
            "parent_tools": [{"type": "function", "function": {"name": "ipython"}}],
            "parent_address": {
                "depth": 0,
                "lineage": "root",
                "session_call_ordinal": 1,
                "turn": 1,
            },
        },
        "model": "model@commit",
        "checkpoint": "model@commit",
        "sampler": {
            "temperature": 0.7,
            "top_p": 1.0,
            "seed": 81,
            "max_tokens": 768,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
        },
        "usage": {
            "prompt_tokens": 42,
            "completion_tokens": 768,
            "cached_input_tokens": None,
            "reasoning_tokens": None,
            "cost": None,
        },
        "finish_reason": "length",
        "token_cap": 768,
        "address": {
            "depth": 1,
            "lineage": "root/child",
            "session_call_ordinal": 0,
            "turn": 0,
        },
        "raw_action_sha256": "a" * 64,
        "raw_trace_sha256": "b" * 64,
        "later_invariants": {
            "endpoint": "pass",
            "ledger": "pass",
            "source_artifacts": "pass",
        },
    }


def _mutation_cases() -> dict[str, tuple[dict[str, Any], dict[str, Any], bool]]:
    cases: dict[str, tuple[dict[str, Any], dict[str, Any], bool]] = {
        "transport-null-trace-absent-no-tool": (_message(), {"role": "assistant"}, True),
        "empty-transport-to-absent": (_message(content=""), {"role": "assistant"}, False),
        "nonempty-transport-to-absent": (_message(content="answer"), {"role": "assistant"}, False),
        "transport-missing-trace-null": ({"role": "assistant"}, _message(), False),
        "transport-missing-trace-absent": ({"role": "assistant"}, {"role": "assistant"}, True),
        "role-change": (_message(), _message(role="user"), False),
        "unknown-field": ({**_message(), "x-unknown": 1}, {"role": "assistant"}, False),
        "unknown-null-field-to-absent": (
            {**_message(), "x-unknown": None},
            {"role": "assistant"},
            False,
        ),
        "tool-call-added": (
            {**_message(), "tool_calls": []},
            _message(),
            False,
        ),
        "tool-call-removed": (
            _message(),
            {**_message(), "tool_calls": []},
            False,
        ),
        "tool-call-reordered": (
            {
                "role": "assistant",
                "tool_calls": [{"id": "a"}, {"id": "b"}],
            },
            {
                "role": "assistant",
                "tool_calls": [{"id": "b"}, {"id": "a"}],
            },
            False,
        ),
        "tool-call-id-changed": (
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "assistant", "tool_calls": [{"id": "b"}]},
            False,
        ),
        "tool-call-name-changed": (
            {
                "role": "assistant",
                "tool_calls": [{"id": "a", "function": {"name": "x"}}],
            },
            {
                "role": "assistant",
                "tool_calls": [{"id": "a", "function": {"name": "y"}}],
            },
            False,
        ),
        "tool-calls-null-to-absent": (
            {"role": "assistant", "tool_calls": None},
            {"role": "assistant"},
            False,
        ),
        "tool-argument-bytes-changed": (
            {
                "role": "assistant",
                "tool_calls": [{"id": "a", "function": {"arguments": '{"a":1}'}}],
            },
            {
                "role": "assistant",
                "tool_calls": [{"id": "a", "function": {"arguments": '{"a":01}'}}],
            },
            False,
        ),
    }
    base = _base_record()
    for field in (
        "request",
        "request_context",
        "model",
        "checkpoint",
        "sampler",
        "usage",
        "address",
        "raw_action_sha256",
        "raw_trace_sha256",
        "later_invariants",
    ):
        mutated = deepcopy(base)
        value = mutated[field]
        if isinstance(value, dict):
            first = next(iter(value))
            value[first] = "mutated" if isinstance(value[first], str) else 0
        elif isinstance(value, int):
            mutated[field] = value - 1
        else:
            mutated[field] = "mutated"
        cases[f"record-{field.replace('_', '-')}-changed"] = (base, mutated, False)
    for field in ("messages", "tools"):
        mutated = deepcopy(base)
        mutated["request"][field] = "mutated"
        cases[f"record-request-{field}-changed"] = (base, mutated, False)
    for field in ("parent_messages", "parent_tools"):
        mutated = deepcopy(base)
        mutated["request_context"][field] = "mutated"
        cases[f"record-request-context-{field.replace('_', '-')}-changed"] = (
            base,
            mutated,
            False,
        )
    mutated = deepcopy(base)
    mutated["request_context"]["parent_address"]["lineage"] = "root/other"
    cases["record-request-context-lineage-changed"] = (base, mutated, False)
    for field, value in (
        ("temperature", 0.8),
        ("top_p", 0.9),
        ("seed", 82),
        ("max_tokens", 767),
        ("parallel_tool_calls", True),
        ("tool_choice", "none"),
    ):
        mutated = deepcopy(base)
        mutated["sampler"][field] = value
        cases[f"record-sampler-{field.replace('_', '-')}-changed"] = (
            base,
            mutated,
            False,
        )
    for field, value in (
        ("prompt_tokens", 43),
        ("completion_tokens", 767),
        ("cached_input_tokens", 1),
        ("reasoning_tokens", 1),
        ("cost", 0.1),
    ):
        mutated = deepcopy(base)
        mutated["usage"][field] = value
        cases[f"record-usage-{field.replace('_', '-')}-changed"] = (
            base,
            mutated,
            False,
        )
    for field, value in (
        ("depth", 0),
        ("lineage", "root/other"),
        ("session_call_ordinal", 1),
        ("turn", 1),
    ):
        mutated = deepcopy(base)
        mutated["address"][field] = value
        cases[f"record-address-{field}-changed"] = (base, mutated, False)
    for field, value in (
        ("endpoint", "other"),
        ("ledger", "fail"),
        ("source_artifacts", "mutated"),
    ):
        mutated = deepcopy(base)
        mutated["later_invariants"][field] = value
        cases[f"record-later-invariants-{field.replace('_', '-')}-changed"] = (
            base,
            mutated,
            False,
        )
    for field, case_id in (
        ("finish_reason", "finish-reason-disagreement"),
        ("token_cap", "token-cap-disagreement"),
    ):
        mutated = deepcopy(base)
        mutated[field] = "stop" if field == "finish_reason" else 767
        cases[case_id] = (base, mutated, False)
    return cases


MUTATION_CASES = _mutation_cases()
