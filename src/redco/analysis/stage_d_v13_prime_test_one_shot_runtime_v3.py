"""Committed v3 runtime consumer for the test-only Prime one-shot."""

from __future__ import annotations

from redco.analysis import stage_d_v13_prime_test_one_shot_contract_v2 as contract
from redco.analysis import stage_d_v13_prime_test_one_shot_lifecycle_v2 as lifecycle
from redco.analysis import stage_d_v13_prime_test_one_shot_successor_v3 as successor
from redco.analysis.stage_d_v13_prime_test_one_shot_runtime_binding_v2 import RuntimeBinding

ROOT = successor.ROOT
V3_RUNTIME_BINDING = RuntimeBinding(
    authorization_path=successor.AUTHORIZATION_PATH,
    authorization_authenticator_name="authenticate_authorization_v3",
    evidence_root=successor.EVIDENCE_ROOT,
    claim_domain=successor.CLAIM_DOMAIN_V3,
    assessment_domain=contract.ASSESSMENT_DOMAIN,
    assessment_schema_version=2,
    assessment_namespace=contract.ASSESSMENT_NAMESPACE,
    signed_envelope_domain=contract.SIGNED_ENVELOPE_DOMAIN,
    handoff_namespace=contract.HANDOFF_NAMESPACE,
    terminal_domain=successor.TERMINAL_DOMAIN_V3,
    terminal_namespace=successor.TERMINAL_NAMESPACE_V3,
    terminal_purpose=successor.TERMINAL_PURPOSE_V3,
    claim_authority=tuple(successor.RUNTIME_AUTHORITY.items()),
    assessment_authority=tuple(successor.RUNTIME_AUTHORITY.items()),
    create_authority=tuple(successor.RUNTIME_AUTHORITY.items()),
    terminal_authority=tuple(successor.READINESS_AUTHORITY.items()),
    result_authority=tuple(successor.READINESS_AUTHORITY.items()),
    handoff_authority=tuple(successor.READINESS_AUTHORITY.items()),
    artifact_filenames=tuple(contract.ARTIFACT_FILENAMES.items()),
    assessment_ttl_seconds=contract.ASSESSMENT_TTL_SECONDS,
    schema_versions=tuple((key, 2) for key in sorted(
        {"claim", "assessment", "assessment-envelope", "handoff", "handoff-envelope",
         "terminal", "terminal-envelope", "result"}
    )),
)
def run_prime_test_one_shot_v3() -> lifecycle.OneShotResult:
    # The lifecycle constructs one authenticated context and verifies the terminal
    # with that same identity; hostile in-process Python is process compromise.
    return lifecycle._run_one_shot(V3_RUNTIME_BINDING)


__all__ = ["ROOT", "V3_RUNTIME_BINDING", "run_prime_test_one_shot_v3"]
