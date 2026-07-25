"""Normative, dependency-free contracts shared by the Tier-0 implementation."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


def canonical_json(value: Any) -> bytes:
    """Serialize JSON-compatible data deterministically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class PolicyNodeKind(StrEnum):
    ROOT_TURN = "root_turn"
    SUBCALL_OUTPUT = "subcall_output"


class SnapshotPhase(StrEnum):
    READY = "ready"
    COLLECTING = "collecting"
    COLLECTED = "collected"
    UPDATED = "updated"


@dataclass(frozen=True, slots=True)
class EventAddress:
    """Structural address for a stochastic event; never content-derived."""

    parent_node_id: str
    turn_index: int
    call_slot_index: int
    occurrence_index: int = 0

    def __post_init__(self) -> None:
        if not self.parent_node_id:
            raise ValueError("parent_node_id must be non-empty")
        for field_name in ("turn_index", "call_slot_index", "occurrence_index"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")

    def as_payload(self) -> dict[str, str | int]:
        return {
            "parent_node_id": self.parent_node_id,
            "turn_index": self.turn_index,
            "call_slot_index": self.call_slot_index,
            "occurrence_index": self.occurrence_index,
        }


@dataclass(frozen=True, slots=True)
class SeedNamespace:
    """Counter-based PRF namespace for continuation randomness."""

    master_seed: str
    rollout_id: str
    target_id: str
    replicate: int

    def __post_init__(self) -> None:
        if not self.master_seed or not self.rollout_id or not self.target_id:
            raise ValueError("seed namespace identifiers must be non-empty")
        if self.replicate < 1:
            raise ValueError("replicate is one-indexed")

    def derive(self, address: EventAddress, *, purpose: str = "continuation") -> int:
        """Return a stable non-negative 63-bit seed for one event."""
        if not purpose:
            raise ValueError("purpose must be non-empty")
        payload = canonical_json(
            {
                "rollout_id": self.rollout_id,
                "target_id": self.target_id,
                "replicate": self.replicate,
                "purpose": purpose,
                "event_address": address.as_payload(),
            }
        )
        digest = hmac.new(self.master_seed.encode("utf-8"), payload, hashlib.sha256).digest()
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

    def action_seed(self, action_slot: int) -> int:
        """Derive candidate-action randomness separately from continuation randomness."""
        if action_slot < 1:
            raise ValueError("action_slot is one-indexed")
        address = EventAddress(
            parent_node_id=self.target_id,
            turn_index=0,
            call_slot_index=action_slot,
        )
        return self.derive(address, purpose="candidate_action")


@dataclass(frozen=True, slots=True)
class PrefixFeatures:
    """Information permitted at online target-selection time."""

    node_kind: PolicyNodeKind
    depth: int
    turn_index: int
    task_metadata: tuple[tuple[str, str], ...] = ()
    predicted_replay_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.depth < 0 or self.turn_index < 0:
            raise ValueError("depth and turn_index must be non-negative")
        if self.predicted_replay_cost < 0:
            raise ValueError("predicted_replay_cost must be non-negative")
        if tuple(sorted(self.task_metadata)) != self.task_metadata:
            raise ValueError("task_metadata must be sorted for stable logging")


@dataclass(frozen=True, slots=True)
class LogicalDeploymentCost:
    """As-if-fresh workflow cost used by the reward function."""

    output_tokens: int = 0
    latency_seconds: float = 0.0
    dollars: float = 0.0

    def __post_init__(self) -> None:
        if self.output_tokens < 0 or self.latency_seconds < 0 or self.dollars < 0:
            raise ValueError("logical costs must be non-negative")


@dataclass(frozen=True, slots=True)
class ActualEvaluationCost:
    """Compute actually spent evaluating a branch; never part of reward."""

    generated_tokens: int = 0
    judge_calls: int = 0
    cpu_seconds: float = 0.0
    gpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    storage_bytes: int = 0

    def __post_init__(self) -> None:
        values = (
            self.generated_tokens,
            self.judge_calls,
            self.cpu_seconds,
            self.gpu_seconds,
            self.wall_seconds,
            self.storage_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("actual evaluation costs must be non-negative")


@dataclass(frozen=True, slots=True)
class DecisionUnitWeight:
    """Explicit macro-action weighting used by the clean loss."""

    outer_weight: float
    branch_count: int = 1

    def __post_init__(self) -> None:
        if self.outer_weight <= 0:
            raise ValueError("outer_weight must be positive")
        if self.branch_count < 1:
            raise ValueError("branch_count must be positive")

    @property
    def record_weight(self) -> float:
        return self.outer_weight / self.branch_count


@dataclass(slots=True)
class SnapshotLifecycle:
    """Enforce collect-once/update-once behavior for the clean Stage-C arm."""

    checkpoint_id: str
    phase: SnapshotPhase = SnapshotPhase.READY
    optimizer_steps: int = 0

    def begin_collection(self, *, rollout_checkpoint: str, branch_checkpoint: str) -> None:
        if self.phase is not SnapshotPhase.READY:
            raise RuntimeError(f"cannot begin collection from {self.phase}")
        if rollout_checkpoint != self.checkpoint_id or branch_checkpoint != self.checkpoint_id:
            raise ValueError("rollout and branch roles must serve the exact snapshot")
        self.phase = SnapshotPhase.COLLECTING

    def finish_collection(self) -> None:
        if self.phase is not SnapshotPhase.COLLECTING:
            raise RuntimeError(f"cannot finish collection from {self.phase}")
        self.phase = SnapshotPhase.COLLECTED

    def record_optimizer_step(self) -> None:
        if self.phase is not SnapshotPhase.COLLECTED:
            raise RuntimeError(f"cannot update from {self.phase}")
        if self.optimizer_steps != 0:
            raise RuntimeError("exactly one optimizer step is allowed per snapshot")
        self.optimizer_steps = 1
        self.phase = SnapshotPhase.UPDATED

