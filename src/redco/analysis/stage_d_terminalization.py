"""Typed, single-publication terminal closure for a Stage-D handoff."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from redco.analysis.stage_d_evaluation_barrier import StageDSealedEvaluationCompletion
from redco.analysis.stage_d_evaluation_codec import (
    EvaluationEvidenceStore,
    FaultHook,
    atomic_publish,
    exclusive_lock,
    sha256,
)
from redco.analysis.stage_d_evaluation_ledger import StageDEvaluationLedger
from redco.analysis.stage_d_provider_billing import StageDProviderBilling
from redco.contracts import canonical_json

TerminalStatus = Literal["completed", "blocked", "failed"]
TerminalPhase = Literal[
    "pre-provision",
    "setup",
    "source",
    "scientific-replay",
    "training",
    "evaluation",
    "cleanup",
]
TerminationCode = Literal[
    "success",
    "capacity-unavailable",
    "ssh-failed",
    "preflight-failed",
    "source-failed",
    "science-failed",
    "training-failed",
    "evaluation-failed",
    "cleanup-failed",
    "billing-unavailable",
    "internal-error",
]
DecisionStatus = Literal["positive", "negative", "indeterminate", "not-evaluated"]
EvaluationState = Literal["not-started", "partial", "completed"]

_TERMINAL_STATUSES = {"completed", "blocked", "failed"}
_TERMINAL_PHASES = {
    "pre-provision",
    "setup",
    "source",
    "scientific-replay",
    "training",
    "evaluation",
    "cleanup",
}
_TERMINATION_CODES = {
    "success",
    "capacity-unavailable",
    "ssh-failed",
    "preflight-failed",
    "source-failed",
    "science-failed",
    "training-failed",
    "evaluation-failed",
    "cleanup-failed",
    "billing-unavailable",
    "internal-error",
}
_DECISION_STATUSES = {"positive", "negative", "indeterminate", "not-evaluated"}


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.isprintable():
        raise ValueError(f"{name} must be nonempty printable text")
    return value


@dataclass(frozen=True, slots=True)
class StageDDecisionOutcome:
    status: DecisionStatus
    decision_rule_sha256: str
    metrics_evidence_sha256: str | None
    reason: str | None

    def __post_init__(self) -> None:
        if self.status not in _DECISION_STATUSES:
            raise ValueError("Stage-D decision status is invalid")
        _require_sha256(self.decision_rule_sha256, "decision rule sha256")
        if self.status == "not-evaluated":
            if self.metrics_evidence_sha256 is not None:
                raise ValueError("unevaluated decision has metrics evidence")
            _require_text(self.reason, "unevaluated decision reason")
        else:
            _require_sha256(self.metrics_evidence_sha256, "decision metrics evidence")
            if self.reason is not None:
                _require_text(self.reason, "decision reason")

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "decision_rule_sha256": self.decision_rule_sha256,
            "metrics_evidence_sha256": self.metrics_evidence_sha256,
            "reason": self.reason,
        }

    @classmethod
    def from_payload(cls, value: object) -> StageDDecisionOutcome:
        fields = {
            "status",
            "decision_rule_sha256",
            "metrics_evidence_sha256",
            "reason",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("Stage-D decision outcome fields differ")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class StageDDecisionVector:
    """The economic and credit questions are deliberately not composited."""

    economic: StageDDecisionOutcome
    credit: StageDDecisionOutcome

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-decision-vector-v1",
                "no_composite_verdict": True,
                "economic": self.economic.to_payload(),
                "credit": self.credit.to_payload(),
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDDecisionVector:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Stage-D decision vector is not JSON") from error
        fields = {
            "schema_version",
            "domain",
            "no_composite_verdict",
            "economic",
            "credit",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or payload.get("schema_version") != 1
            or payload.get("domain") != "redco-stage-d-decision-vector-v1"
            or payload.get("no_composite_verdict") is not True
            or canonical_json(payload) != value
        ):
            raise ValueError("Stage-D decision vector fields differ")
        return cls(
            StageDDecisionOutcome.from_payload(payload["economic"]),
            StageDDecisionOutcome.from_payload(payload["credit"]),
        )


@dataclass(frozen=True, slots=True)
class StageDCleanupEvidence:
    compute_state: Literal["not-provisioned", "terminated", "termination-unverified"]
    persistent_storage_state: Literal[
        "not-created", "zero-confirmed", "nonzero", "unverified"
    ]
    in_pod_process_state: Literal[
        "not-started", "contained-empty", "contained-orphan", "unverified"
    ]
    receipt_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.compute_state not in {
            "not-provisioned",
            "terminated",
            "termination-unverified",
        }:
            raise ValueError("Stage-D cleanup compute state is invalid")
        if self.persistent_storage_state not in {
            "not-created",
            "zero-confirmed",
            "nonzero",
            "unverified",
        }:
            raise ValueError("Stage-D cleanup storage state is invalid")
        if self.in_pod_process_state not in {
            "not-started",
            "contained-empty",
            "contained-orphan",
            "unverified",
        }:
            raise ValueError("Stage-D cleanup process state is invalid")
        if self.receipt_sha256s != tuple(sorted(set(self.receipt_sha256s))):
            raise ValueError("Stage-D cleanup receipt roster is not sorted and unique")
        for digest in self.receipt_sha256s:
            _require_sha256(digest, "cleanup receipt")

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-cleanup-evidence-v1",
                "compute_state": self.compute_state,
                "persistent_storage_state": self.persistent_storage_state,
                "in_pod_process_state": self.in_pod_process_state,
                "receipt_sha256s": list(self.receipt_sha256s),
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDCleanupEvidence:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Stage-D cleanup evidence is not JSON") from error
        fields = {
            "schema_version",
            "domain",
            "compute_state",
            "persistent_storage_state",
            "in_pod_process_state",
            "receipt_sha256s",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or payload.get("schema_version") != 1
            or payload.get("domain") != "redco-stage-d-cleanup-evidence-v1"
            or not isinstance(payload.get("receipt_sha256s"), list)
            or canonical_json(payload) != value
        ):
            raise ValueError("Stage-D cleanup evidence fields differ")
        return cls(
            payload["compute_state"],
            payload["persistent_storage_state"],
            payload["in_pod_process_state"],
            tuple(payload["receipt_sha256s"]),
        )

    def verify_receipts(self, receipts: Mapping[str, bytes]) -> None:
        if set(receipts) != set(self.receipt_sha256s):
            raise ValueError("Stage-D cleanup receipt roster differs")
        _verify_digest_bytes(receipts, "cleanup receipt")


@dataclass(frozen=True, slots=True)
class StageDEvaluationTerminal:
    state: EvaluationState
    evaluation_ledger_id: str | None
    terminal_status: str | None
    ledger_head_sha256: str | None
    record_count: int
    ledger_bundle_sha256: str | None
    sealed_completion_sha256: str | None

    def __post_init__(self) -> None:
        if self.state not in {"not-started", "partial", "completed"}:
            raise ValueError("Stage-D evaluation terminal state is invalid")
        if type(self.record_count) is not int or self.record_count < 0:
            raise ValueError("Stage-D evaluation terminal record count is invalid")
        if self.state == "not-started":
            if any(
                value is not None
                for value in (
                    self.evaluation_ledger_id,
                    self.terminal_status,
                    self.ledger_head_sha256,
                    self.ledger_bundle_sha256,
                    self.sealed_completion_sha256,
                )
            ) or self.record_count != 0:
                raise ValueError("not-started evaluation has ledger state")
            return
        _require_text(self.evaluation_ledger_id, "evaluation ledger id")
        _require_text(self.terminal_status, "evaluation terminal status")
        _require_sha256(self.ledger_head_sha256, "evaluation ledger head")
        _require_sha256(self.ledger_bundle_sha256, "evaluation ledger bundle")
        if self.record_count < 1:
            raise ValueError("started evaluation has no records")
        if self.state == "completed":
            _require_sha256(self.sealed_completion_sha256, "evaluation completion")
            if self.terminal_status != "sealed":
                raise ValueError("completed evaluation is not sealed")
        elif self.sealed_completion_sha256 is not None:
            raise ValueError("partial evaluation has a sealed completion")

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-evaluation-terminal-v1",
                "state": self.state,
                "evaluation_ledger_id": self.evaluation_ledger_id,
                "terminal_status": self.terminal_status,
                "ledger_head_sha256": self.ledger_head_sha256,
                "record_count": self.record_count,
                "ledger_bundle_sha256": self.ledger_bundle_sha256,
                "sealed_completion_sha256": self.sealed_completion_sha256,
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDEvaluationTerminal:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Stage-D evaluation terminal is not JSON") from error
        fields = {
            "schema_version",
            "domain",
            "state",
            "evaluation_ledger_id",
            "terminal_status",
            "ledger_head_sha256",
            "record_count",
            "ledger_bundle_sha256",
            "sealed_completion_sha256",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or payload.get("schema_version") != 1
            or payload.get("domain") != "redco-stage-d-evaluation-terminal-v1"
            or canonical_json(payload) != value
        ):
            raise ValueError("Stage-D evaluation terminal fields differ")
        return cls(
            payload["state"],
            payload["evaluation_ledger_id"],
            payload["terminal_status"],
            payload["ledger_head_sha256"],
            payload["record_count"],
            payload["ledger_bundle_sha256"],
            payload["sealed_completion_sha256"],
        )


@dataclass(frozen=True, slots=True)
class StageDTerminalSeal:
    terminal_status: TerminalStatus
    terminal_phase: TerminalPhase
    termination_code: TerminationCode
    preregistration_sha256: str
    protocol_manifest_sha256: str
    handoff_policy_sha256: str
    handoff_head_sha256: str
    handoff_record_count: int
    evaluation_terminal_sha256: str
    decision_vector_sha256: str
    provider_billing_sha256: str
    cleanup_evidence_sha256: str
    report_sha256: str
    evidence_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.terminal_status not in _TERMINAL_STATUSES:
            raise ValueError("Stage-D terminal status is invalid")
        if self.terminal_phase not in _TERMINAL_PHASES:
            raise ValueError("Stage-D terminal phase is invalid")
        if self.termination_code not in _TERMINATION_CODES:
            raise ValueError("Stage-D termination code is invalid")
        if (self.terminal_status == "completed") != (self.termination_code == "success"):
            raise ValueError("Stage-D terminal status and code disagree")
        for name in (
            "preregistration_sha256",
            "protocol_manifest_sha256",
            "handoff_policy_sha256",
            "handoff_head_sha256",
            "evaluation_terminal_sha256",
            "decision_vector_sha256",
            "provider_billing_sha256",
            "cleanup_evidence_sha256",
            "report_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.handoff_record_count) is not int or self.handoff_record_count < 1:
            raise ValueError("Stage-D handoff record count is invalid")
        if self.evidence_sha256s != tuple(sorted(set(self.evidence_sha256s))):
            raise ValueError("Stage-D terminal evidence roster is not sorted and unique")
        for digest in self.evidence_sha256s:
            _require_sha256(digest, "terminal evidence")
        required = {
            self.evaluation_terminal_sha256,
            self.decision_vector_sha256,
            self.provider_billing_sha256,
            self.cleanup_evidence_sha256,
            self.report_sha256,
        }
        if not required.issubset(self.evidence_sha256s):
            raise ValueError("Stage-D terminal evidence roster lacks a typed record")

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-terminal-seal-v1",
                "terminal_status": self.terminal_status,
                "terminal_phase": self.terminal_phase,
                "termination_code": self.termination_code,
                "preregistration_sha256": self.preregistration_sha256,
                "protocol_manifest_sha256": self.protocol_manifest_sha256,
                "handoff_policy_sha256": self.handoff_policy_sha256,
                "handoff_head_sha256": self.handoff_head_sha256,
                "handoff_record_count": self.handoff_record_count,
                "evaluation_terminal_sha256": self.evaluation_terminal_sha256,
                "decision_vector_sha256": self.decision_vector_sha256,
                "provider_billing_sha256": self.provider_billing_sha256,
                "cleanup_evidence_sha256": self.cleanup_evidence_sha256,
                "report_sha256": self.report_sha256,
                "evidence_sha256s": list(self.evidence_sha256s),
            }
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDTerminalSeal:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Stage-D terminal seal is not JSON") from error
        fields = {
            "schema_version",
            "domain",
            "terminal_status",
            "terminal_phase",
            "termination_code",
            "preregistration_sha256",
            "protocol_manifest_sha256",
            "handoff_policy_sha256",
            "handoff_head_sha256",
            "handoff_record_count",
            "evaluation_terminal_sha256",
            "decision_vector_sha256",
            "provider_billing_sha256",
            "cleanup_evidence_sha256",
            "report_sha256",
            "evidence_sha256s",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or payload.get("schema_version") != 1
            or payload.get("domain") != "redco-stage-d-terminal-seal-v1"
            or not isinstance(payload.get("evidence_sha256s"), list)
            or canonical_json(payload) != value
        ):
            raise ValueError("Stage-D terminal seal fields differ")
        return cls(
            payload["terminal_status"],
            payload["terminal_phase"],
            payload["termination_code"],
            payload["preregistration_sha256"],
            payload["protocol_manifest_sha256"],
            payload["handoff_policy_sha256"],
            payload["handoff_head_sha256"],
            payload["handoff_record_count"],
            payload["evaluation_terminal_sha256"],
            payload["decision_vector_sha256"],
            payload["provider_billing_sha256"],
            payload["cleanup_evidence_sha256"],
            payload["report_sha256"],
            tuple(payload["evidence_sha256s"]),
        )


class HandoffSnapshot(Protocol):
    preregistration_sha256: str
    protocol_manifest_sha256: str
    handoff_policy_sha256: str
    evaluation_ledger_id: str | None
    head_sha256: str
    record_count: int
    sealed: bool


class HandoffCoordinator(Protocol):
    root: Path
    lock_path: Path
    evidence: EvaluationEvidenceStore
    _fault_hook: FaultHook | None

    def inspect(self) -> HandoffSnapshot: ...


def finalize_stage_d(
    coordinator: HandoffCoordinator,
    *,
    terminal_status: TerminalStatus,
    terminal_phase: TerminalPhase,
    termination_code: TerminationCode,
    decisions: StageDDecisionVector,
    decision_evidence: Mapping[str, bytes],
    billing: StageDProviderBilling,
    billing_receipts: Mapping[str, bytes],
    cleanup: StageDCleanupEvidence,
    cleanup_receipts: Mapping[str, bytes],
    evaluation_ledger: StageDEvaluationLedger | None = None,
    evaluation_completion_bytes: bytes | None = None,
) -> bytes:
    """Publish exactly one canonical terminal seal after installing its evidence closure."""
    with exclusive_lock(coordinator.lock_path):
        snapshot = coordinator.inspect()
        existing_path = coordinator.root / "terminal_seal.json"
        if snapshot.sealed and not existing_path.is_file():
            raise RuntimeError("legacy-sealed handoff cannot receive a terminal seal")

        billing.verify_receipts(billing_receipts)
        cleanup.verify_receipts(cleanup_receipts)
        _verify_digest_bytes(decision_evidence, "decision evidence")
        expected_decisions = {
            digest
            for digest in (
                decisions.economic.metrics_evidence_sha256,
                decisions.credit.metrics_evidence_sha256,
            )
            if digest is not None
        }
        if set(decision_evidence) != expected_decisions:
            raise ValueError("Stage-D decision evidence roster differs")

        installed = set()
        for values in (decision_evidence, billing_receipts, cleanup_receipts):
            for digest, value in values.items():
                installed.add(_put_expected(coordinator.evidence, digest, value))

        evaluation = _adopt_evaluation_terminal(
            coordinator,
            snapshot,
            evaluation_ledger=evaluation_ledger,
            completion_bytes=evaluation_completion_bytes,
        )
        evaluation_bytes = evaluation.to_bytes()
        evaluation_sha256 = coordinator.evidence.put(
            evaluation_bytes, fault_hook=coordinator._fault_hook
        )
        decision_bytes = decisions.to_bytes()
        decision_sha256 = coordinator.evidence.put(
            decision_bytes, fault_hook=coordinator._fault_hook
        )
        billing_bytes = billing.to_bytes()
        billing_sha256 = coordinator.evidence.put(
            billing_bytes, fault_hook=coordinator._fault_hook
        )
        cleanup_bytes = cleanup.to_bytes()
        cleanup_sha256 = coordinator.evidence.put(
            cleanup_bytes, fault_hook=coordinator._fault_hook
        )

        if terminal_status == "completed":
            if terminal_phase != "evaluation" or evaluation.state != "completed":
                raise ValueError("completed Stage-D terminal lacks completed evaluation")
            if "not-evaluated" in {decisions.economic.status, decisions.credit.status}:
                raise ValueError("completed Stage-D terminal has an unevaluated decision")
            if (
                cleanup.compute_state != "terminated"
                or cleanup.persistent_storage_state != "zero-confirmed"
                or cleanup.in_pod_process_state != "contained-empty"
            ):
                raise ValueError("completed Stage-D terminal lacks verified cleanup")

        typed_digests = {
            evaluation_sha256,
            decision_sha256,
            billing_sha256,
            cleanup_sha256,
        }
        installed.update(typed_digests)
        if evaluation.ledger_bundle_sha256 is not None:
            installed.update(_evaluation_bundle_evidence(coordinator, evaluation))

        report = canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-terminal-report-v1",
                "terminal_status": terminal_status,
                "terminal_phase": terminal_phase,
                "termination_code": termination_code,
                "handoff_head_sha256": snapshot.head_sha256,
                "handoff_record_count": snapshot.record_count,
                "evaluation": json.loads(evaluation_bytes),
                "decisions": json.loads(decision_bytes),
                "billing": json.loads(billing_bytes),
                "cleanup": json.loads(cleanup_bytes),
            }
        )
        report_sha256 = coordinator.evidence.put(report, fault_hook=coordinator._fault_hook)
        installed.add(report_sha256)
        seal = StageDTerminalSeal(
            terminal_status,
            terminal_phase,
            termination_code,
            snapshot.preregistration_sha256,
            snapshot.protocol_manifest_sha256,
            snapshot.handoff_policy_sha256,
            snapshot.head_sha256,
            snapshot.record_count,
            evaluation_sha256,
            decision_sha256,
            billing_sha256,
            cleanup_sha256,
            report_sha256,
            tuple(sorted(installed)),
        )
        seal_bytes = seal.to_bytes()
        try:
            atomic_publish(existing_path, seal_bytes, fault_hook=coordinator._fault_hook)
        except FileExistsError as error:
            raise FileExistsError(
                "Stage-D terminal seal differs from durable state"
            ) from error
        if existing_path.read_bytes() != seal_bytes:
            raise FileExistsError("Stage-D terminal seal differs from durable state")
        return seal_bytes


def verify_stage_d_terminal_seal(
    coordinator: HandoffCoordinator,
    value: bytes,
    *,
    preregistration_sha256: str,
    protocol_manifest_sha256: str,
    handoff_policy_sha256: str,
    handoff_head_sha256: str,
    handoff_record_count: int,
) -> StageDTerminalSeal:
    """Verify a terminal seal and the exact reachable evidence closure."""
    seal = StageDTerminalSeal.from_bytes(value)
    if (
        seal.preregistration_sha256 != preregistration_sha256
        or seal.protocol_manifest_sha256 != protocol_manifest_sha256
        or seal.handoff_policy_sha256 != handoff_policy_sha256
        or seal.handoff_head_sha256 != handoff_head_sha256
        or seal.handoff_record_count != handoff_record_count
    ):
        raise ValueError("Stage-D terminal seal differs from its handoff")
    evidence = {digest: coordinator.evidence.get(digest) for digest in seal.evidence_sha256s}
    decisions = StageDDecisionVector.from_bytes(evidence[seal.decision_vector_sha256])
    billing = StageDProviderBilling.from_bytes(evidence[seal.provider_billing_sha256])
    cleanup = StageDCleanupEvidence.from_bytes(evidence[seal.cleanup_evidence_sha256])
    evaluation = StageDEvaluationTerminal.from_bytes(
        evidence[seal.evaluation_terminal_sha256]
    )
    decision_digests = {
        digest
        for digest in (
            decisions.economic.metrics_evidence_sha256,
            decisions.credit.metrics_evidence_sha256,
        )
        if digest is not None
    }
    billing_digests = {
        billing.wallet_before_receipt_sha256,
        billing.wallet_after_receipt_sha256,
        *(item.provider_receipt_sha256 for item in billing.deployments),
    }
    billing.verify_receipts({digest: evidence[digest] for digest in billing_digests})
    cleanup.verify_receipts(
        {digest: evidence[digest] for digest in cleanup.receipt_sha256s}
    )
    _verify_digest_bytes(
        {digest: evidence[digest] for digest in decision_digests},
        "decision evidence",
    )
    expected = {
        seal.evaluation_terminal_sha256,
        seal.decision_vector_sha256,
        seal.provider_billing_sha256,
        seal.cleanup_evidence_sha256,
        seal.report_sha256,
        *decision_digests,
        *billing_digests,
        *cleanup.receipt_sha256s,
    }
    if evaluation.ledger_bundle_sha256 is not None:
        expected.update(_evaluation_bundle_evidence(coordinator, evaluation))
    if expected != set(seal.evidence_sha256s):
        raise ValueError("Stage-D terminal evidence closure differs")
    report = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-terminal-report-v1",
            "terminal_status": seal.terminal_status,
            "terminal_phase": seal.terminal_phase,
            "termination_code": seal.termination_code,
            "handoff_head_sha256": seal.handoff_head_sha256,
            "handoff_record_count": seal.handoff_record_count,
            "evaluation": json.loads(evidence[seal.evaluation_terminal_sha256]),
            "decisions": json.loads(evidence[seal.decision_vector_sha256]),
            "billing": json.loads(evidence[seal.provider_billing_sha256]),
            "cleanup": json.loads(evidence[seal.cleanup_evidence_sha256]),
        }
    )
    if sha256(report) != seal.report_sha256 or evidence[seal.report_sha256] != report:
        raise ValueError("Stage-D terminal report differs from its typed evidence")
    return seal


def _adopt_evaluation_terminal(
    coordinator: HandoffCoordinator,
    snapshot: HandoffSnapshot,
    *,
    evaluation_ledger: StageDEvaluationLedger | None,
    completion_bytes: bytes | None,
) -> StageDEvaluationTerminal:
    if evaluation_ledger is None:
        if completion_bytes is not None:
            raise ValueError("evaluation completion was supplied without a ledger")
        return StageDEvaluationTerminal("not-started", None, None, None, 0, None, None)
    if (
        snapshot.evaluation_ledger_id is None
        or evaluation_ledger.root.name != snapshot.evaluation_ledger_id
    ):
        raise ValueError("terminal evaluation ledger differs from handoff authorization")
    allowed = {"inputs", "records", "responses", "evidence", "writer.lock"}
    if any(item.name not in allowed for item in evaluation_ledger.root.iterdir()):
        raise ValueError("terminal evaluation ledger has unknown top-level state")
    with exclusive_lock(evaluation_ledger.lock_path):
        observed = evaluation_ledger.inspect()
        entries: list[dict[str, object]] = []
        for directory_name in ("inputs", "records", "responses", "evidence"):
            directory = evaluation_ledger.root / directory_name
            for path in sorted(directory.iterdir()):
                if path.is_symlink() or not path.is_file():
                    raise ValueError("terminal evaluation bundle has non-regular state")
                value = path.read_bytes()
                digest = coordinator.evidence.put(value, fault_hook=coordinator._fault_hook)
                entries.append(
                    {
                        "path": f"{directory_name}/{path.name}",
                        "sha256": digest,
                        "size_bytes": len(value),
                    }
                )
        completion_sha256 = None
        state: EvaluationState = "partial"
        if completion_bytes is not None:
            completion = StageDSealedEvaluationCompletion.from_bytes(completion_bytes)
            completion.verify_ledger(evaluation_ledger)
            completion_sha256 = coordinator.evidence.put(
                completion_bytes, fault_hook=coordinator._fault_hook
            )
            entries.append(
                {
                    "path": "completion.json",
                    "sha256": completion_sha256,
                    "size_bytes": len(completion_bytes),
                }
            )
            state = "completed"
        elif observed.sealed:
            raise ValueError("sealed evaluation ledger lacks its completion")
        entries.sort(key=lambda item: str(item["path"]))
        manifest = canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-terminal-evaluation-bundle-v1",
                "state": state,
                "evaluation_ledger_id": evaluation_ledger.root.name,
                "terminal_status": observed.terminal_status,
                "ledger_head_sha256": observed.head_sha256,
                "record_count": observed.record_count,
                "entries": entries,
            }
        )
        bundle_sha256 = coordinator.evidence.put(
            manifest, fault_hook=coordinator._fault_hook
        )
    return StageDEvaluationTerminal(
        state,
        evaluation_ledger.root.name,
        observed.terminal_status,
        observed.head_sha256,
        observed.record_count,
        bundle_sha256,
        completion_sha256,
    )


def _evaluation_bundle_evidence(
    coordinator: HandoffCoordinator,
    evaluation: StageDEvaluationTerminal,
) -> set[str]:
    bundle_sha256 = _require_sha256(evaluation.ledger_bundle_sha256, "evaluation bundle")
    manifest_bytes = coordinator.evidence.get(bundle_sha256)
    payload = json.loads(manifest_bytes)
    if (
        not isinstance(payload, dict)
        or payload.get("domain") != "redco-stage-d-terminal-evaluation-bundle-v1"
        or canonical_json(payload) != manifest_bytes
        or not isinstance(payload.get("entries"), list)
    ):
        raise ValueError("terminal evaluation bundle manifest differs")
    digests = {bundle_sha256}
    for item in payload["entries"]:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}:
            raise ValueError("terminal evaluation bundle entry fields differ")
        value = coordinator.evidence.get(item["sha256"])
        if len(value) != item["size_bytes"]:
            raise ValueError("terminal evaluation bundle entry size differs")
        digests.add(item["sha256"])
    return digests


def _put_expected(store: EvaluationEvidenceStore, digest: str, value: bytes) -> str:
    _require_sha256(digest, "evidence digest")
    if sha256(value) != digest:
        raise ValueError("Stage-D supplied evidence bytes differ from their digest")
    return store.put(value)


def _verify_digest_bytes(values: Mapping[str, bytes], name: str) -> None:
    for digest, value in values.items():
        _require_sha256(digest, name)
        if sha256(value) != digest:
            raise ValueError(f"Stage-D {name} bytes differ")


__all__ = [
    "DecisionStatus",
    "StageDCleanupEvidence",
    "StageDDecisionOutcome",
    "StageDDecisionVector",
    "StageDEvaluationTerminal",
    "StageDTerminalSeal",
    "TerminalPhase",
    "TerminalStatus",
    "TerminationCode",
    "finalize_stage_d",
    "verify_stage_d_terminal_seal",
]
