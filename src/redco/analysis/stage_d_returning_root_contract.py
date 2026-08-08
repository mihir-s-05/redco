"""Canonical, candidate-independent contract data for live correspondence."""

from __future__ import annotations

import hashlib
from typing import Final

from redco.analysis import stage_d_source_producer as _source_producer
from redco.contracts import canonical_json

type JSON = dict[str, object]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


RETURNING_ROOT_CONTRACT_VERSION: Final = (
    "redco-stage-d-live-returning-root-correspondence-v1"
)
LIVE_CORRESPONDENCE_RECEIPT_VERSION: Final = (
    "redco-stage-d-live-returning-root-correspondence-receipt-v1"
)
PROVENANCE_FIELDS: Final = (
    "provenance_version", "depth", "session_id", "turn", "call_kind",
    "parent_session_id", "parent_turn", "parent_tool_call_id", "invocation_id",
    "lineage", "parent_lineage", "session_call_ordinal", "parent_call_ordinal",
    "parent_tool_call_slot", "spawn_ordinal", "episode_spawn_ordinal",
    "completed_predecessor_spawn_ordinals", "completed_episode_spawn_ordinals",
)
TERMINATION_FIELDS: Final = (
    "schema_version", "state", "recurrence_observed", "source_completed",
    "terminal_call_ordinal", "terminal_policy_turn", "terminal_policy_lineage",
)
TERMINAL_BINDING_FIELDS: Final = (
    "schema_version", "domain", "state", "recurrence_observed", "source_completed",
    "source_id", "source_sha256", "trace_id",
    "trace_sha256", "group_id", "rollout_id", "terminal_owner_identity",
    "terminal_disposition", "terminal_reason", "terminal_call_ordinal",
    "terminal_policy_turn", "terminal_policy_lineage", "target_id", "target_ordinal",
    "target_address", "ledger_id", "target_commitment_receipt_kind",
    "target_commitment_receipt_b64", "target_commitment_receipt_sha256",
    "source_completion_receipt_kind", "source_completion_receipt_b64",
    "source_completion_receipt_sha256",
    "terminal_trace_evidence_sha256",
)
TERMINAL_OWNER_IDENTITY: Final = (
    "redco.analysis.stage_d_receipt_ledger.StageDReceiptLedger"
)


def _obj(required: tuple[str, ...], properties: JSON) -> JSON:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": properties,
    }


def _sha_schema() -> JSON:
    return {"type": "string", "pattern": "^[0-9a-f]{64}$"}


_ADDRESS_SCHEMA: Final = _obj(
    ("depth", "lineage", "session_call_ordinal", "turn", "call_kind"),
    {
        "depth": {"type": "integer", "minimum": 0},
        "lineage": {"type": "string", "minLength": 1},
        "session_call_ordinal": {"type": "integer", "minimum": 0},
        "turn": {"type": "integer", "minimum": 0},
        "call_kind": {"const": "policy"},
    },
)
_NONNEGATIVE: Final = {"type": ["integer", "null"], "minimum": 0}
_RECORD_SCHEMA: Final = _obj(
    PROVENANCE_FIELDS,
    {
        "provenance_version": {"const": 2},
        "depth": {"type": "integer", "minimum": 0},
        "session_id": {"type": "string", "minLength": 1},
        "turn": {"type": "integer", "minimum": 0},
        "call_kind": {"enum": ["policy", "compaction"]},
        "parent_session_id": {"type": ["string", "null"]},
        "parent_turn": _NONNEGATIVE,
        "parent_tool_call_id": {"type": ["string", "null"]},
        "invocation_id": {"type": ["string", "null"]},
        "lineage": {"type": "string", "minLength": 1},
        "parent_lineage": {"type": ["string", "null"]},
        "session_call_ordinal": {"type": "integer", "minimum": 0},
        "parent_call_ordinal": _NONNEGATIVE,
        "parent_tool_call_slot": _NONNEGATIVE,
        "spawn_ordinal": _NONNEGATIVE,
        "episode_spawn_ordinal": _NONNEGATIVE,
        "completed_predecessor_spawn_ordinals": {
            "type": ["array", "null"],
            "items": {"type": "integer", "minimum": 0},
        },
        "completed_episode_spawn_ordinals": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        },
    },
)
_TERMINATION_SCHEMA: Final = _obj(
    TERMINATION_FIELDS,
    {
        "schema_version": {"const": 1},
        "state": {"const": "clean_terminated_before_recurrence"},
        "recurrence_observed": {"const": False},
        "source_completed": {"const": False},
        "terminal_call_ordinal": {"type": "integer", "minimum": 0},
        "terminal_policy_turn": {"type": "integer", "minimum": 0},
        "terminal_policy_lineage": {"type": "string", "minLength": 1},
    },
)
_B64_SCHEMA: Final = {"type": "string", "pattern": "^[A-Za-z0-9+/]*={0,2}$"}
TERMINAL_BINDING_SCHEMA: Final = _obj(
    TERMINAL_BINDING_FIELDS,
    {
        "schema_version": {"const": 1},
        "domain": {"const": "redco-stage-d-live-returning-root-source-terminal-binding-v1"},
        "state": {"const": "clean_terminated_before_recurrence"},
        "recurrence_observed": {"const": False},
        "source_completed": {"const": False},
        "source_id": {"type": "string", "minLength": 1},
        "source_sha256": _sha_schema(),
        "trace_id": {"type": "string", "minLength": 1},
        "trace_sha256": _sha_schema(),
        "group_id": {"type": "string", "minLength": 1},
        "rollout_id": {"type": "string", "minLength": 1},
        "terminal_owner_identity": {"const": TERMINAL_OWNER_IDENTITY},
        "terminal_disposition": {"const": "terminal_without_downstream"},
        "terminal_reason": {"const": "no_downstream_model_call"},
        "terminal_call_ordinal": {"type": "integer", "minimum": 0},
        "terminal_policy_turn": {"type": "integer", "minimum": 0},
        "terminal_policy_lineage": {"type": "string", "minLength": 1},
        "target_id": {"type": "string", "minLength": 1},
        "target_ordinal": {"type": "integer", "minimum": 0},
        "target_address": _ADDRESS_SCHEMA,
        "ledger_id": {"type": "string", "minLength": 1},
        "target_commitment_receipt_kind": {
            "const": "pre_action_group_commitment"
        },
        "target_commitment_receipt_b64": _B64_SCHEMA,
        "target_commitment_receipt_sha256": _sha_schema(),
        "source_completion_receipt_kind": {"const": "source_rollout_completed"},
        "source_completion_receipt_b64": _B64_SCHEMA,
        "source_completion_receipt_sha256": _sha_schema(),
        "terminal_trace_evidence_sha256": _sha_schema(),
    },
)

_RAW_RLM_PROPERTIES: Final = {
    "provenance_version": {"const": 2},
    "depth": {"type": "integer", "minimum": 0},
    "session_id": {"type": "string", "minLength": 1},
    "turn": {"type": "integer", "minimum": 0},
    "call_kind": {"enum": ["policy", "compaction"]},
    "parent_session_id": {"type": ["string", "null"]},
    "parent_turn": _NONNEGATIVE,
    "parent_tool_call_id": {"type": ["string", "null"]},
    "invocation_id": {"type": ["string", "null"]},
    "lineage": {"type": "string", "minLength": 1},
    "parent_lineage": {"type": ["string", "null"]},
    "session_call_ordinal": {"type": "integer", "minimum": 0},
    "parent_call_ordinal": _NONNEGATIVE,
    "parent_tool_call_slot": _NONNEGATIVE,
    "spawn_ordinal": _NONNEGATIVE,
    "episode_spawn_ordinal": _NONNEGATIVE,
    "completed_predecessor_spawn_ordinals": {
        "type": ["array", "null"],
        "items": {"type": "integer", "minimum": 0},
    },
    "completed_episode_spawn_ordinals": {
        "type": "array",
        "items": {"type": "integer", "minimum": 0},
    },
}
_RAW_RLM_SCHEMA: Final = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "provenance_version", "depth", "session_id", "turn", "call_kind",
        "lineage", "session_call_ordinal",
    ],
    "properties": _RAW_RLM_PROPERTIES,
    "allOf": [
        {
            "if": {"properties": {"depth": {"const": 0}}},
            "then": {
                "properties": {
                    "parent_session_id": {"const": None},
                    "parent_turn": {"const": None},
                    "parent_tool_call_id": {"const": None},
                    "parent_lineage": {"const": None},
                    "parent_call_ordinal": {"const": None},
                    "parent_tool_call_slot": {"const": None},
                    "spawn_ordinal": {"const": None},
                    "episode_spawn_ordinal": {"const": None},
                    "completed_predecessor_spawn_ordinals": {"const": None},
                }
            },
        },
        {
            "if": {"properties": {"depth": {"const": 1}}},
            "then": {
                "required": [
                    "parent_session_id", "parent_turn", "parent_tool_call_id",
                    "invocation_id", "parent_lineage", "parent_call_ordinal",
                    "parent_tool_call_slot", "spawn_ordinal", "episode_spawn_ordinal",
                    "completed_predecessor_spawn_ordinals",
                ]
            },
        },
    ],
}
_SOURCE_CALL_FIELDS: Final = frozenset(_source_producer._CALL_FIELDS)
_SOURCE_NODE_FIELDS: Final = frozenset(_source_producer._NODE_FIELDS)
_SOURCE_TRACE_FIELDS: Final = frozenset(_source_producer._TRACE_FIELDS)
_SOURCE_EPISODE_FIELDS: Final = frozenset(_source_producer._EPISODE_FIELDS)
_SOURCE_SAMPLING_FIELDS: Final = tuple(_source_producer._SAMPLING_CONTRACT_FIELDS)
RAW_TRACE_FIELDS: Final = tuple(sorted(_SOURCE_TRACE_FIELDS))
RAW_EPISODE_FIELDS: Final = tuple(sorted(_SOURCE_EPISODE_FIELDS))

_SOURCE_TIME_SCHEMA: Final = _obj(
    ("start", "end"),
    {"start": {"type": "number"}, "end": {"type": "number"}},
)
_SOURCE_USAGE_SCHEMA: Final = _obj(
    (
        "prompt_tokens", "completion_tokens", "cached_input_tokens",
        "reasoning_tokens", "cost",
    ),
    {
        "prompt_tokens": {"type": "integer", "minimum": 0},
        "completion_tokens": {"type": "integer", "minimum": 0},
        "cached_input_tokens": {"type": ["integer", "null"], "minimum": 0},
        "reasoning_tokens": {"type": ["integer", "null"], "minimum": 0},
        "cost": {"type": ["number", "null"]},
    },
)
_SOURCE_SAMPLING_SCHEMA: Final = _obj(
    _SOURCE_SAMPLING_FIELDS,
    {
        "temperature": {"type": "number"},
        "top_p": {"type": "number"},
        "reasoning_effort": {"const": None},
        "min_p": {"type": "number"},
        "repetition_penalty": {"type": "number"},
        "frequency_penalty": {"type": "number"},
        "presence_penalty": {"type": "number"},
        "seed": {"type": "integer"},
        "max_tokens": {"type": "integer"},
        "n": {"type": "integer"},
        "tool_choice": {"type": "string"},
        "parallel_tool_calls": {"type": "boolean"},
    },
)
_SOURCE_CALL_SCHEMA: Final = _obj(
    tuple(sorted(_SOURCE_CALL_FIELDS)),
    {
        "endpoint": {"const": "/chat/completions"},
        "error": {"const": None},
        "finish_reason": {"type": "string"},
        "model": {"type": "string", "minLength": 1},
        "node": {"type": "integer", "minimum": 0},
        "rlm": _RAW_RLM_SCHEMA,
        "sampling": _SOURCE_SAMPLING_SCHEMA,
        "time": _SOURCE_TIME_SCHEMA,
        "usage": _SOURCE_USAGE_SCHEMA,
    },
)
_SOURCE_NODE_SCHEMA: Final = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "parent": {"type": ["integer", "null"], "minimum": 0},
        "message": {"type": "object"},
        "sampled": {"type": "boolean"},
        "timestamp": {"type": "number"},
        "token_ids": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "mask": {"type": "array", "items": {"type": "boolean"}},
        "is_content": {"type": "array", "items": {"type": "boolean"}},
        "logprobs": {"type": "array", "items": {"type": "number"}},
    },
    "required": sorted(_SOURCE_NODE_FIELDS - {"parent"}),
}
_SOURCE_TRACE_SCHEMA: Final = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "task": {"type": "object"},
        "runtime": {"type": ["object", "null"]},
        "version": {"type": "integer"},
        "verifiers": {"type": "object"},
        "run": {"type": "object"},
        "agent": {"type": "object"},
        "nodes": {"type": "array", "items": _SOURCE_NODE_SCHEMA},
        "tools": {"type": "array", "items": {"type": "object"}},
        "calls": {"type": "array", "items": _SOURCE_CALL_SCHEMA},
        "rewards": {"type": "object"},
        "metrics": {"type": "object"},
        "info": {"type": "object"},
        "extra_usage": {"type": "array"},
        "is_completed": {"type": "boolean"},
        "ok": {"type": "boolean"},
        "stop_condition": {"type": ["string", "null"]},
        "errors": {"type": "array"},
        "timing": {"type": "object"},
    },
    "required": list(RAW_TRACE_FIELDS),
}
RAW_TRACE_SCHEMA: Final = _SOURCE_TRACE_SCHEMA
RAW_EPISODE_SCHEMA: Final = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "env": {"type": "string"},
        "ok": {"type": "boolean"},
        "errors": {"type": "array"},
        "traces": {
            "type": "array",
            "minItems": 1,
            "maxItems": 1,
            "items": _SOURCE_TRACE_SCHEMA,
        },
    },
    "required": list(RAW_EPISODE_FIELDS),
}
_INPUT_FIELDS: Final = (
    "source_id", "source_sha256", "trace_id", "trace_sha256", "trace",
    "group_id", "rollout_id", "causal_graph_schema_sha256", "target_ordinal",
    "target_id", "target_address", "spawn_ordinal", "parent_lineage",
    "commitment_receipt_sha256", "recorded_action_digest", "termination_evidence",
)
_INPUT_PROPERTIES: Final = {
    "source_id": {"type": "string", "minLength": 1},
    "source_sha256": _sha_schema(),
    "trace_id": {"type": "string", "minLength": 1},
    "trace_sha256": _sha_schema(),
    "trace": {"$ref": "#/$defs/raw_trace"},
    "group_id": {"type": "string", "minLength": 1},
    "rollout_id": {"type": "string", "minLength": 1},
    "causal_graph_schema_sha256": _sha_schema(),
    "target_ordinal": {"type": "integer", "minimum": 0},
    "target_id": {"type": "string", "minLength": 1},
    "target_address": _ADDRESS_SCHEMA,
    "spawn_ordinal": {"type": "integer", "minimum": 0},
    "parent_lineage": {"type": "string", "minLength": 1},
    "commitment_receipt_sha256": _sha_schema(),
    "recorded_action_digest": _sha_schema(),
    "termination_evidence": {"anyOf": [TERMINAL_BINDING_SCHEMA, {"type": "null"}]},
}
_EVALUATOR_FIELDS: Final = tuple(
    field for field in _INPUT_FIELDS if field != "recorded_action_digest"
)
_EVALUATOR_PROPERTIES: Final = {
    field: properties
    for field, properties in _INPUT_PROPERTIES.items()
    if field != "recorded_action_digest"
}
INPUT_SCHEMA: Final = {
    **_obj(_INPUT_FIELDS, _INPUT_PROPERTIES),
    "$defs": {"raw_trace": RAW_TRACE_SCHEMA},
}
RESULT_SCHEMA: Final = _obj(
    ("disposition", "target_ordinal", "target_id", "matched_address"),
    {
        "disposition": {
            "enum": [
                "eligible_match", "terminal_without_downstream", "missing_target",
                "no_valid_return", "ambiguous_conflicting_minima",
                "provenance_divergence", "topology_divergence",
            ]
        },
        "target_ordinal": {"type": "integer", "minimum": 0},
        "target_id": {"type": "string", "minLength": 1},
        "matched_address": {"anyOf": [_ADDRESS_SCHEMA, {"type": "null"}]},
    },
)
EVALUATOR_SCHEMA: Final = _obj(
    _EVALUATOR_FIELDS,
    {
        **_EVALUATOR_PROPERTIES,
        "trace": _obj(
            ("id", "provenance_records"),
            {
                "id": {"type": "string", "minLength": 1},
                "provenance_records": {"type": "array", "items": _RECORD_SCHEMA},
            },
        ),
    },
)
REPLAY_SCHEMA: Final = _obj(
    ("source_input", "replay_input"),
    {
        "source_input": {"$ref": "#/$defs/input"},
        "replay_input": {"$ref": "#/$defs/input"},
    },
)
REPLAY_SCHEMA["$defs"] = {
    "input": EVALUATOR_SCHEMA,
    "result": RESULT_SCHEMA,
}
RECEIPT_SCHEMA: Final = {
    **_obj(
        (
            "schema_version", "domain", "contract_version", "contract_sha256",
            "causal_graph_schema_sha256", "evaluator_input",
            "evaluator_input_sha256", "evaluator_result", "evaluator_result_sha256",
        ),
        {
            "schema_version": {"const": 1},
            "domain": {"const": LIVE_CORRESPONDENCE_RECEIPT_VERSION},
            "contract_version": {"const": RETURNING_ROOT_CONTRACT_VERSION},
            "contract_sha256": _sha_schema(),
            "causal_graph_schema_sha256": _sha_schema(),
            "evaluator_input": {"$ref": "#/$defs/input"},
            "evaluator_input_sha256": _sha_schema(),
            "evaluator_result": {"$ref": "#/$defs/result"},
            "evaluator_result_sha256": _sha_schema(),
        },
    ),
    "$defs": {"input": EVALUATOR_SCHEMA, "result": RESULT_SCHEMA},
}

CAUSAL_PROVENANCE_SCHEMA: Final[JSON] = {
    "schema_version": 2,
    "domain": "redco-stage-d-rlm-provenance-v2",
    "strict_owner": "redco.integrations.verifiers_trace_v2.extract_v2_rlm_provenance",
    "fields": list(PROVENANCE_FIELDS),
    "completion_snapshot": "completed_episode_spawn_ordinals",
    "diagnostic_fields": ["turn", "parent_turn", "parent_tool_call_id"],
    "additional_properties": False,
}
CAUSAL_PROVENANCE_SCHEMA_BYTES: Final = canonical_json(CAUSAL_PROVENANCE_SCHEMA)
CAUSAL_PROVENANCE_SCHEMA_SHA256: Final = _sha256(CAUSAL_PROVENANCE_SCHEMA_BYTES)

CONTRACT_PAYLOAD: Final[JSON] = {
    "schema_version": 2,
    "domain": RETURNING_ROOT_CONTRACT_VERSION,
    "selection_order": ["target_ordinal", "target_id"],
    "strict_provenance_owner": (
        "redco.integrations.verifiers_trace_v2.extract_v2_rlm_provenance"
    ),
    "returning_root_law": {
        "required_target_depth": 1,
        "required_target_call_kind": "policy",
        "required_root_depth": 0,
        "required_root_call_kind": "policy",
        "same_parent_lineage_and_session": True,
        "strictly_downstream": True,
        "completion_snapshot_contains_target_spawn": True,
        "minimum": ["session_call_ordinal", "turn"],
        "turn_recurrence": (
            "each_authenticated_lineage_starts_at_zero_and_increments_by_one"
        ),
        "arrival_order_ignored": True,
    },
    "excluded_policy": {
        "upstream_parent": "never_match",
        "target_itself": "never_match",
        "later_descendant": "exogenous",
        "dynamic_or_structurally_divergent": "exogenous",
    },
    "replay": {
        "structural_seed_key": "full_policy_address",
        "compare_projection": list(PROVENANCE_FIELDS),
        "diagnostics": ["turn", "parent_turn", "parent_tool_call_id"],
        "invalid_provenance": "fail_closed",
    },
    "content_independent": ["action_text", "reward", "alternatives", "node_index", "arrival_order"],
    "termination": {
        "source": "StageDReceiptLedger source-rollout completion owner",
        "clean_state": "clean_terminated_before_recurrence",
        "caller_boolean": "forbidden",
        "owner_identity": TERMINAL_OWNER_IDENTITY,
        "target_commitment_receipt_kind": "pre_action_group_commitment",
        "source_receipt_kind": "source_rollout_completed",
        "receipt_authentication": (
            "active-clean ledger chain, source completion, target commitment, and trace evidence"
        ),
        "phase_1_available_before_correspondence": True,
        "scientific_execution_receipt_forbidden": True,
        "completed_trace_is_generic_not_terminal": True,
        "returning_root_evaluation_precedes_terminal_disposition": True,
    },
    "recorded_action_digest": {
        "phase": "phase-2-pending",
        "authenticated_in_phase_1": False,
        "used_by_returning_root_selection": False,
        "evaluator_projection_includes_field": False,
    },
    "dispositions": [
        "eligible_match", "terminal_without_downstream", "missing_target", "no_valid_return",
        "ambiguous_conflicting_minima", "provenance_divergence", "topology_divergence",
    ],
    "schemas": {
        "input": INPUT_SCHEMA,
        "authenticated_termination_evidence": TERMINAL_BINDING_SCHEMA,
        "terminal_binding": TERMINAL_BINDING_SCHEMA,
        "result_output": RESULT_SCHEMA,
        "replay_validation": REPLAY_SCHEMA,
        "live_receipt": RECEIPT_SCHEMA,
        "evaluator_input_projection": EVALUATOR_SCHEMA,
    },
    "live_activation": "phase-2-only",
}
_SCHEMAS: Final = {
    "input": INPUT_SCHEMA,
    "authenticated_termination_evidence": TERMINAL_BINDING_SCHEMA,
    "terminal_binding": TERMINAL_BINDING_SCHEMA,
    "result_output": RESULT_SCHEMA,
    "replay_validation": REPLAY_SCHEMA,
    "live_receipt": RECEIPT_SCHEMA,
    "evaluator_input_projection": EVALUATOR_SCHEMA,
}
SCHEMA_SHA256S: Final = {
    name: _sha256(canonical_json(schema)) for name, schema in _SCHEMAS.items()
}
CONTRACT_PAYLOAD["schemas"] = _SCHEMAS
CONTRACT_PAYLOAD["schema_sha256s"] = SCHEMA_SHA256S
RETURNING_ROOT_CONTRACT_BYTES: Final = canonical_json(CONTRACT_PAYLOAD)
RETURNING_ROOT_CONTRACT_SHA256: Final = _sha256(RETURNING_ROOT_CONTRACT_BYTES)

E2_SOURCE_SHA256: Final = "add76383d43ebcbe03a6285bd01464553848738fa96de8e7efdfa6f7fa3fdb55"
E2_TRACE_SHA256: Final = "dabe6869c58aaa988eecbf97bf5bac8e183496be0541e95589ab576981648351"
E2_COMMITMENT_SHA256: Final = "167156d7990d9dd355524f845168fe4a61b3629b72dd7e9559d7f76614217149"
E2_ACTION_SHA256: Final = "bc46a0a13ec318190631782e35e89a3b93a3ee97cc356f304aaa533fdc0694c7"


__all__ = [
    "CAUSAL_PROVENANCE_SCHEMA",
    "CAUSAL_PROVENANCE_SCHEMA_BYTES",
    "CAUSAL_PROVENANCE_SCHEMA_SHA256",
    "CONTRACT_PAYLOAD",
    "E2_ACTION_SHA256",
    "E2_COMMITMENT_SHA256",
    "E2_SOURCE_SHA256",
    "E2_TRACE_SHA256",
    "EVALUATOR_SCHEMA",
    "INPUT_SCHEMA",
    "LIVE_CORRESPONDENCE_RECEIPT_VERSION",
    "PROVENANCE_FIELDS",
    "RAW_EPISODE_FIELDS",
    "RAW_EPISODE_SCHEMA",
    "RAW_TRACE_FIELDS",
    "RAW_TRACE_SCHEMA",
    "RECEIPT_SCHEMA",
    "REPLAY_SCHEMA",
    "RESULT_SCHEMA",
    "RETURNING_ROOT_CONTRACT_BYTES",
    "RETURNING_ROOT_CONTRACT_SHA256",
    "RETURNING_ROOT_CONTRACT_VERSION",
    "SCHEMA_SHA256S",
    "TERMINAL_BINDING_FIELDS",
    "TERMINAL_BINDING_SCHEMA",
    "TERMINAL_OWNER_IDENTITY",
    "TERMINATION_FIELDS",
]
