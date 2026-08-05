from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from stage_d_source_comparison_oracle import (
    FUTURE_PRODUCTION_BINDING,
    MUTATION_CASES,
    PRODUCTION_BOUNDARY_CONTRACT,
    directional_message_match,
    exact_record_match,
    production_boundary_observation,
    record_exactness_binding_observation,
)

from redco.analysis.stage_d_source_producer import (
    _normalize_openai_message,
    _normalize_openai_tools,
)
from redco.contracts import canonical_json

CONTRACT_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "stage-d"
    / "stage-d1-source-comparison-contract-v1.json"
)
SELECTED_MANIFEST_PATH = (
    Path(__file__).parents[1] / "reports" / "stage-d1-source-comparison-selected-tests-v1.json"
)
GREEN_MANIFEST_PATH = (
    Path(__file__).parents[1] / "reports" / "stage-d1-source-comparison-green-suite-v1.json"
)


def _contract() -> dict[str, object]:
    value = json.loads(CONTRACT_PATH.read_bytes())
    assert isinstance(value, dict)
    return value


def test_directional_contract_is_versioned_and_raw_canonical() -> None:
    raw = CONTRACT_PATH.read_bytes()
    assert not raw.endswith(b"\n")
    assert raw == canonical_json(json.loads(raw))
    contract = _contract()
    assert contract["schema_version"] == 1
    assert contract["domain"] == "redco-stage-d-source-comparison-contract-v1"
    assert contract["pointer_syntax"] == (
        "RFC 6901 JSON Pointer; root is empty string and content is /content"
    )
    divergence = contract["single_permitted_message_divergence"]
    assert isinstance(divergence, dict)
    assert divergence == {
        "message_role": "assistant",
        "pointer": "/content",
        "scope": "no-tool assistant message only",
        "tool_calls": "absent",
        "trace_presence": "absent",
        "trace_value": "absent",
        "transport_presence": "present-null",
        "transport_value": None,
    }
    binding = contract["future_oracle_binding"]
    assert isinstance(binding, dict)
    assert binding["callable"] == (
        "tests/stage_d_source_comparison_oracle.py:directional_message_match"
    )
    assert binding["production_callable"] == (
        "redco.analysis.stage_d_source_producer:_verify_trace_call"
    )
    assert binding["production_fixture"] == (
        "tests/stage_d_source_comparison_oracle.py:production_boundary_observation"
    )
    assert binding["raw_bytes_passthrough"] is True
    assert binding["record_oracle"] == (
        "tests/stage_d_source_comparison_oracle.py:exact_record_match"
    )
    assert binding["version"] == "v1"
    assert binding["message_cases_bound"] is True
    assert binding["record_cases_bound"] is False
    assert binding["record_binding_hook"] == PRODUCTION_BOUNDARY_CONTRACT["record_hook"]
    assert binding["record_cases_policy"] == PRODUCTION_BOUNDARY_CONTRACT["record_cases_policy"]
    future_callable = FUTURE_PRODUCTION_BINDING["callable"]
    assert future_callable == "redco.analysis.stage_d_source_producer:_verify_trace_call"
    fixture_callable = FUTURE_PRODUCTION_BINDING["fixture"]
    assert fixture_callable == (
        "tests/stage_d_source_comparison_oracle.py:production_boundary_observation"
    )
    assert callable(production_boundary_observation)
    assert callable(record_exactness_binding_observation)
    assert FUTURE_PRODUCTION_BINDING["version"] == PRODUCTION_BOUNDARY_CONTRACT["version"]
    assert FUTURE_PRODUCTION_BINDING["record_cases_bound"] is False
    assert binding["currently_bound"] is False
    assert binding["required_before_repair_green"] is True
    scope = contract["scope"]
    assert isinstance(scope, dict)
    assert scope["raw_transport_preserved"] is True
    assert scope["raw_trace_preserved"] is True
    production_diagnostic = contract["production_diagnostic"]
    assert isinstance(production_diagnostic, dict)
    assert production_diagnostic["still_fails_closed"] is True
    forbidden = contract["forbidden_changes"]
    assert isinstance(forbidden, list)
    assert {
        "recursive null dropping",
        "empty/null equivalence",
        "global message normalization",
        "changes to canonical_json",
        "changes to action evidence",
        "changes to trace persistence",
        "changes to source serialization",
        "request-context normalization",
    } <= set(forbidden)
    proof = contract["post_repair_proof"]
    assert isinstance(proof, dict)
    assert all(proof.values())


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "case_id", sorted(MUTATION_CASES)
)
def test_frozen_oracle_matrix_is_fail_closed(case_id: str) -> None:
    transport, trace, expected = MUTATION_CASES[case_id]
    if case_id.startswith("record-") or case_id in {
        "finish-reason-disagreement",
        "token-cap-disagreement",
    }:
        observed = exact_record_match(transport, trace)
        boundary = record_exactness_binding_observation(transport, trace)
        assert boundary["hook_version"] == PRODUCTION_BOUNDARY_CONTRACT["version"]
        assert boundary["boundary_kind"] == "record_exactness_only"
        assert boundary["bound"] is False
        assert boundary["status"] == "not_bound_pre_repair"
        assert boundary["transport_bytes"] == canonical_json(transport)
        assert boundary["trace_bytes"] == canonical_json(trace)
        assert boundary["transport_sha256"] == hashlib.sha256(canonical_json(transport)).hexdigest()
        assert boundary["trace_sha256"] == hashlib.sha256(canonical_json(trace)).hexdigest()
    else:
        boundary = production_boundary_observation(transport, trace)
        assert boundary["hook_version"] == PRODUCTION_BOUNDARY_CONTRACT["version"]
        assert boundary["boundary_kind"] == "message"
        assert boundary["transport_bytes"] == canonical_json(transport)
        assert boundary["trace_bytes"] == canonical_json(trace)
        assert boundary["transport_sha256"] == hashlib.sha256(canonical_json(transport)).hexdigest()
        assert boundary["trace_sha256"] == hashlib.sha256(canonical_json(trace)).hexdigest()
        assert boundary["production_callable"] == FUTURE_PRODUCTION_BINDING["callable"]
        current_production_expectation = canonical_json(transport) == canonical_json(trace)
        if case_id == "transport-null-trace-absent-no-tool":
            assert boundary["accepted"] in {False, True}
            if boundary["accepted"]:
                assert directional_message_match(transport, trace) is True
        else:
            assert boundary["accepted"] is current_production_expectation
        observed = directional_message_match(transport, trace)
    assert observed is expected, case_id


def test_recursive_transport_null_stripper_fails_adversarial_frozen_cases() -> None:
    """Ensure prohibited recursive null dropping cannot satisfy the frozen matrix."""

    def strip_transport_nulls(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: strip_transport_nulls(item)
                for key, item in value.items()
                if item is not None
            }
        if isinstance(value, list):
            return [strip_transport_nulls(item) for item in value]
        return value

    for case_id in ("unknown-null-field-to-absent", "tool-calls-null-to-absent"):
        transport, trace, expected = MUTATION_CASES[case_id]
        assert (
            canonical_json(strip_transport_nulls(transport))
            == canonical_json(strip_transport_nulls(trace))
        ), case_id
        assert expected is False, case_id
        boundary = production_boundary_observation(transport, trace)
        assert boundary["accepted"] is False, case_id


def test_contract_matrix_is_complete_for_the_frozen_fields() -> None:
    contract = _contract()
    matrix = contract["regression_matrix"]
    assert isinstance(matrix, list)
    ids = {entry["id"] for entry in matrix if isinstance(entry, dict)}
    assert ids == set(MUTATION_CASES)
    exactness = contract["exactness_scope"]
    assert isinstance(exactness, dict)
    assert {
        "action_evidence",
        "addresses",
        "checkpoint",
        "later_invariants",
        "model",
        "request",
        "sampler",
        "trace",
        "transport",
        "usage",
    } == set(exactness)


def test_existing_compact_wrapped_tool_canonicalization_is_the_only_tool_normalization() -> None:
    function = {
        "name": "ipython",
        "description": "Execute code.",
        "parameters": {"type": "object"},
        "strict": None,
    }
    compact = [function]
    wrapped = [{"type": "function", "function": function}]
    assert _normalize_openai_tools(compact) == _normalize_openai_tools(wrapped)
    compact_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "a", "name": "ipython", "arguments": "{}"}],
    }
    wrapped_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "a",
                "type": "function",
                "function": {"name": "ipython", "arguments": "{}"},
            }
        ],
    }
    assert _normalize_openai_message(compact_message) == _normalize_openai_message(wrapped_message)


def test_cpu_manifests_are_canonical_reproducible_and_path_free() -> None:
    for path in (SELECTED_MANIFEST_PATH, GREEN_MANIFEST_PATH):
        raw = path.read_bytes()
        assert not raw.endswith(b"\n")
        assert raw == canonical_json(json.loads(raw))
        text = raw.decode("utf-8")
        assert "/home/" not in text
        assert "C:\\" not in text
        assert "dependency_copy" not in text
        assert "duration" not in text
    selected = json.loads(SELECTED_MANIFEST_PATH.read_bytes())
    green = json.loads(GREEN_MANIFEST_PATH.read_bytes())
    for key in ("collected", "passed", "failed", "skipped", "xfailed", "deselected"):
        assert selected["actual"][key] == selected["expectation"][key]
    assert selected["bound_project_hashes"] == {
        "pyproject.toml": selected["source_sha256"]["pyproject.toml"],
        "uv.lock": selected["source_sha256"]["uv.lock"],
    }
    assert all(command["skipped"] == 0 for command in green["commands"])
    assert green["aggregate"]["skipped"] == 0
