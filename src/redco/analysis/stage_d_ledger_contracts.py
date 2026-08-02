"""Small shared types for the Stage-D receipt ledger and its verifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress, ScheduledSeed


class LedgerError(RuntimeError):
    """The ledger cannot safely continue or verify."""


class LedgerPoisoned(LedgerError):
    """A torn, ambiguous, or scientifically dangling ledger was observed."""


class BatchAlreadyClaimed(LedgerError):
    """A training batch already has a durable single-use claim."""


@dataclass(frozen=True, slots=True)
class CandidateAttempt:
    ledger_id: str
    group_id: str
    target_id: str
    action_slot: int
    action_seed: int
    attempt_ordinal: int
    attempt_id: str


@dataclass(frozen=True, slots=True)
class ExecutionAttempt:
    ledger_id: str
    group_id: str
    target_id: str
    arm_id: str
    action_digest: str
    continuation_replicate: int
    attempt_ordinal: int
    attempt_id: str


@dataclass(frozen=True, slots=True)
class ModelCallAttempt:
    execution_attempt_id: str
    call_id: str
    address: PolicyEventAddress
    scheduled_seed: ScheduledSeed


@dataclass(frozen=True, slots=True)
class ReplayOverrideTicket:
    execution_attempt_id: str
    override_id: str
    address: PolicyEventAddress
    action_digest: str
    disposition: Literal["reuse", "inject"]
    response_content_sha256: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    counts_toward_logical_cost: bool = False


__all__ = [
    "BatchAlreadyClaimed",
    "CandidateAttempt",
    "ExecutionAttempt",
    "LedgerError",
    "LedgerPoisoned",
    "ModelCallAttempt",
    "ReplayOverrideTicket",
]
