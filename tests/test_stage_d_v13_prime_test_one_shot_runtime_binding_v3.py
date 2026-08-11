"""Closed runtime-binding and no-context execution regressions."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from redco.analysis import stage_d_v13_prime_inventory_v5 as v5
from redco.analysis import stage_d_v13_prime_test_one_shot_contract_v2 as contract
from redco.analysis import stage_d_v13_prime_test_one_shot_lifecycle_v2 as lifecycle
from redco.analysis import stage_d_v13_prime_test_one_shot_prime_v2 as prime
from redco.analysis import stage_d_v13_prime_test_one_shot_runtime_binding_v2 as binding
from redco.analysis import stage_d_v13_prime_test_one_shot_runtime_v3 as runtime_v3
from redco.analysis import stage_d_v13_prime_test_one_shot_successor_v3 as successor
from redco.analysis.stage_d_v13_prime_test_one_shot_runtime_v3 import (
V3_RUNTIME_BINDING,
)


def _replace_binding_field(field: str, value: object) -> binding.RuntimeBinding:
    if field == "authorization_authenticator_name":
        return replace(
            binding.V2_RUNTIME_BINDING,
            authorization_authenticator_name=cast(str, value),
        )
    if field == "authorization_path":
        return replace(binding.V2_RUNTIME_BINDING, authorization_path=cast(str, value))
    if field == "evidence_root":
        return replace(binding.V2_RUNTIME_BINDING, evidence_root=cast(str, value))
    if field == "assessment_ttl_seconds":
        return replace(
            binding.V2_RUNTIME_BINDING,
            assessment_ttl_seconds=cast(int, value),
        )
    if field == "schema_versions":
        return replace(
            binding.V2_RUNTIME_BINDING,
            schema_versions=cast(tuple[tuple[str, int], ...], value),
        )
    raise AssertionError(f"unexpected binding field: {field}")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authorization_authenticator_name", "caller_authenticator"),
        ("authorization_path", "C:/caller/auth.json"),
        ("evidence_root", "runs/../caller"),
        ("assessment_ttl_seconds", True),
        ("schema_versions", (("claim", 3),)),
    ],
)
def test_runtime_binding_rejects_unsupported_semantics(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _replace_binding_field(field, value)


def test_runtime_binding_rejects_duplicate_authority_and_artifact_entries() -> None:
    authority = binding.V2_RUNTIME_BINDING.claim_authority
    with pytest.raises(ValueError):
        replace(binding.V2_RUNTIME_BINDING, claim_authority=(*authority, authority[0]))
    filenames = binding.V2_RUNTIME_BINDING.artifact_filenames
    with pytest.raises(ValueError):
        replace(binding.V2_RUNTIME_BINDING, artifact_filenames=(*filenames, filenames[0]))


def test_forged_binding_fails_before_owner_or_root(monkeypatch: pytest.MonkeyPatch) -> None:
    owner_calls = 0
    root_calls = 0

    def forbidden_owner() -> object:
        nonlocal owner_calls
        owner_calls += 1
        raise AssertionError("installed owner was reached")

    def forbidden_root(*_args: object, **_kwargs: object) -> Path:
        nonlocal root_calls
        root_calls += 1
        raise AssertionError("evidence root was reached")

    monkeypatch.setattr(
        v5, "authenticate_installed_capture_owners", forbidden_owner
    )
    monkeypatch.setattr(lifecycle, "exclusive_runtime_root", forbidden_root)
    forged = replace(V3_RUNTIME_BINDING)
    with pytest.raises(ValueError, match="canonical singleton"):
        binding._build_production_context(forged)
    with pytest.raises(ValueError, match="canonical singleton"):
        lifecycle._run_one_shot(cast(binding.RuntimeBinding, object()))
    assert owner_calls == 0
    assert root_calls == 0


def test_no_context_executor_or_registration_capability_is_reachable() -> None:
    assert not hasattr(binding, "_FACTORY_ATTESTATION")
    assert not hasattr(binding, "_RuntimeAttestation")
    assert not hasattr(binding, "_identity_registry")
    assert not hasattr(binding, "_is_factory_object")
    assert inspect.signature(lifecycle._run_one_shot).parameters.keys() == {"binding"}
    assert inspect.signature(binding._build_production_context).parameters.keys() == {
        "binding"
    }
    for function in (
        lifecycle.run_prime_test_one_shot_v2,
        runtime_v3.run_prime_test_one_shot_v3,
        lifecycle._run_one_shot,
        binding._build_production_context,
    ):
        assert function.__closure__ is None
        assert function.__defaults__ is None
        assert function.__kwdefaults__ is None
        assert not {
            name.lower()
            for name in function.__code__.co_names
            if any(word in name.lower() for word in ("registry", "register", "attestation"))
        }


def test_canonical_public_routes_have_no_injection_parameters() -> None:
    assert not inspect.signature(lifecycle.run_prime_test_one_shot_v2).parameters
    assert not inspect.signature(runtime_v3.run_prime_test_one_shot_v3).parameters
    assert not inspect.signature(binding._production_context_v2).parameters
    assert not inspect.signature(binding._production_context_v3).parameters
    assert "RuntimeContext" not in prime.__all__
    assert "run_one_shot" not in prime.__all__
    assert "_run_one_shot" not in lifecycle.__all__


def test_public_route_globals_do_not_expose_context_registration() -> None:
    reachable_names = set(runtime_v3.run_prime_test_one_shot_v3.__code__.co_names)
    reachable_names.update(lifecycle.run_prime_test_one_shot_v2.__code__.co_names)
    assert not any(
        any(word in name.lower() for word in ("registry", "register", "attestation"))
        for name in reachable_names
    )
    assert "_RuntimeContext" not in reachable_names
    assert "_RuntimeAttestation" not in reachable_names


def test_context_shaped_forgery_is_rejected_without_reading_operational_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root_calls = 0

    def forbidden_root(*_args: object, **_kwargs: object) -> Path:
        nonlocal root_calls
        root_calls += 1
        raise AssertionError("evidence root was reached")

    class ForgedContext:
        def __init__(self) -> None:
            self.repository = tmp_path
            self.authorization = {"commit": "forged"}
            self.client = object()
            self.run = object()
            self.signing_key = tmp_path / "forged-key"

    monkeypatch.setattr(lifecycle, "exclusive_runtime_root", forbidden_root)
    with pytest.raises(ValueError, match="canonical singleton"):
        lifecycle._run_one_shot(cast(binding.RuntimeBinding, ForgedContext()))
    assert root_calls == 0


def test_v3_authority_remains_closed_and_test_only() -> None:
    assert len(dict(V3_RUNTIME_BINDING.claim_authority)) == 10
    assert dict(V3_RUNTIME_BINDING.claim_authority)["parquet_access_authorized"] is False
    assert not any(dict(V3_RUNTIME_BINDING.result_authority).values())
    assert "use uv only; never pip" in successor.SUCCESSOR_AUTHORIZATION_TEXT.lower()
    assert V3_RUNTIME_BINDING.handoff_namespace == contract.HANDOFF_NAMESPACE
