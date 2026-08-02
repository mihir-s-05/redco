from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from redco.analysis.stage_d_handoff_coordinator import StageDHandoffCoordinator
from redco.analysis.stage_d_provider_billing import StageDProviderBilling
from redco.analysis.stage_d_terminalization import (
    StageDCleanupEvidence,
    StageDDecisionOutcome,
    StageDDecisionVector,
    StageDTerminalSeal,
)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _blocked_inputs() -> tuple[
    StageDDecisionVector,
    StageDProviderBilling,
    dict[str, bytes],
    StageDCleanupEvidence,
]:
    rule = _digest(b"decision rule")
    decisions = StageDDecisionVector(
        StageDDecisionOutcome("not-evaluated", rule, None, "no pod was provisioned"),
        StageDDecisionOutcome("not-evaluated", rule, None, "no pod was provisioned"),
    )
    wallet_before = b"wallet before"
    wallet_after = b"wallet after"
    receipts = {
        _digest(wallet_before): wallet_before,
        _digest(wallet_after): wallet_after,
    }
    billing = StageDProviderBilling(
        "USD",
        (),
        0,
        0,
        40_000_000,
        40_000_000,
        0,
        _digest(wallet_before),
        _digest(wallet_after),
    )
    cleanup = StageDCleanupEvidence("not-provisioned", "not-created", "not-started", ())
    return decisions, billing, receipts, cleanup


def _coordinator(tmp_path: Path) -> StageDHandoffCoordinator:
    return StageDHandoffCoordinator.create(
        tmp_path / "handoff",
        preregistration_sha256=_digest(b"preregistration"),
        protocol_manifest_sha256=_digest(b"protocol"),
        handoff_policy_sha256=_digest(b"handoff policy"),
    )


def test_pre_provision_block_is_single_atomic_typed_terminal_seal(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    decisions, billing, billing_receipts, cleanup = _blocked_inputs()
    seal_bytes = coordinator.finalize_terminal(
        terminal_status="blocked",
        terminal_phase="pre-provision",
        termination_code="capacity-unavailable",
        decisions=decisions,
        decision_evidence={},
        billing=billing,
        billing_receipts=billing_receipts,
        cleanup=cleanup,
        cleanup_receipts={},
    )
    seal = StageDTerminalSeal.from_bytes(seal_bytes)
    assert seal.terminal_status == "blocked"
    assert coordinator.inspect().terminal_seal_sha256 == _digest(seal_bytes)
    assert coordinator.inspect().sealed
    assert not tuple(coordinator.records.glob("*report*"))

    assert (
        coordinator.finalize_terminal(
            terminal_status="blocked",
            terminal_phase="pre-provision",
            termination_code="capacity-unavailable",
            decisions=decisions,
            decision_evidence={},
            billing=billing,
            billing_receipts=billing_receipts,
            cleanup=cleanup,
            cleanup_receipts={},
        )
        == seal_bytes
    )


def test_terminal_retry_with_different_typed_input_is_rejected(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    decisions, billing, billing_receipts, cleanup = _blocked_inputs()
    coordinator.finalize_terminal(
        terminal_status="blocked",
        terminal_phase="pre-provision",
        termination_code="capacity-unavailable",
        decisions=decisions,
        decision_evidence={},
        billing=billing,
        billing_receipts=billing_receipts,
        cleanup=cleanup,
        cleanup_receipts={},
    )
    changed = replace(
        decisions,
        credit=replace(decisions.credit, reason="capacity disappeared before provisioning"),
    )
    with pytest.raises(FileExistsError, match="terminal seal differs"):
        coordinator.finalize_terminal(
            terminal_status="blocked",
            terminal_phase="pre-provision",
            termination_code="capacity-unavailable",
            decisions=changed,
            decision_evidence={},
            billing=billing,
            billing_receipts=billing_receipts,
            cleanup=cleanup,
            cleanup_receipts={},
        )


def test_completed_terminal_requires_evaluation_and_verified_cleanup(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    decisions, billing, billing_receipts, cleanup = _blocked_inputs()
    evaluated = StageDDecisionVector(
        replace(
            decisions.economic,
            status="indeterminate",
            metrics_evidence_sha256=_digest(b"economic metrics"),
            reason=None,
        ),
        replace(
            decisions.credit,
            status="negative",
            metrics_evidence_sha256=_digest(b"credit metrics"),
            reason=None,
        ),
    )
    with pytest.raises(ValueError, match="completed evaluation"):
        coordinator.finalize_terminal(
            terminal_status="completed",
            terminal_phase="evaluation",
            termination_code="success",
            decisions=evaluated,
            decision_evidence={
                _digest(b"economic metrics"): b"economic metrics",
                _digest(b"credit metrics"): b"credit metrics",
            },
            billing=billing,
            billing_receipts=billing_receipts,
            cleanup=cleanup,
            cleanup_receipts={},
        )


def test_legacy_arbitrary_report_path_is_disabled(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    with pytest.raises(RuntimeError, match="arbitrary report commits are disabled"):
        coordinator.commit_report(
            report_bytes=b"report",
            decision_evidence_bytes=b"decision",
            billing_evidence_bytes=b"billing",
        )
