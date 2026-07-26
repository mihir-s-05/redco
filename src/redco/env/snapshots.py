"""Content-addressed state snapshots for the deterministic replay oracle."""

from __future__ import annotations

import json
from dataclasses import dataclass

from redco.env.artifacts import ArtifactRef, ArtifactStore
from redco.env.commands import JsonValue


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    checkpoint_id: str
    state_ref: ArtifactRef

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must be non-empty")


class SnapshotStore:
    """Persist and restore immutable JSON-domain workflow state."""

    def __init__(self, artifacts: ArtifactStore) -> None:
        self.artifacts = artifacts

    def capture(
        self,
        *,
        checkpoint_id: str,
        state: dict[str, JsonValue],
    ) -> StateSnapshot:
        return StateSnapshot(checkpoint_id, self.artifacts.put_json(state))

    def restore(self, snapshot: StateSnapshot) -> dict[str, JsonValue]:
        value = json.loads(self.artifacts.get_bytes(snapshot.state_ref))
        if not isinstance(value, dict):
            raise RuntimeError("snapshot root must be an object")
        return value
