from __future__ import annotations

import pytest

from redco.contracts import (
    DecisionUnitWeight,
    EventAddress,
    SeedNamespace,
    SnapshotLifecycle,
    SnapshotPhase,
)


def test_structural_seed_is_stable_and_addressed() -> None:
    namespace = SeedNamespace("secret", "rollout-1", "target-2", 1)
    address = EventAddress("parent", 3, 4, 0)

    assert namespace.derive(address) == namespace.derive(address)
    assert namespace.derive(address) != namespace.derive(
        EventAddress("parent", 3, 4, 1)
    )
    assert namespace.action_seed(1) != namespace.action_seed(2)


def test_decision_unit_weight_uses_branch_mean() -> None:
    assert DecisionUnitWeight(outer_weight=0.5, branch_count=4).record_weight == 0.125


def test_snapshot_lifecycle_allows_exactly_one_update() -> None:
    lifecycle = SnapshotLifecycle("theta-0")
    lifecycle.begin_collection(rollout_checkpoint="theta-0", branch_checkpoint="theta-0")
    lifecycle.finish_collection()
    lifecycle.record_optimizer_step()

    assert lifecycle.phase is SnapshotPhase.UPDATED
    assert lifecycle.optimizer_steps == 1
    with pytest.raises(RuntimeError):
        lifecycle.record_optimizer_step()


def test_snapshot_rejects_checkpoint_drift() -> None:
    lifecycle = SnapshotLifecycle("theta-0")
    with pytest.raises(ValueError, match="exact snapshot"):
        lifecycle.begin_collection(
            rollout_checkpoint="theta-0",
            branch_checkpoint="theta-1",
        )

