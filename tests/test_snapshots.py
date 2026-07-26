from __future__ import annotations

from pathlib import Path

from redco.env.artifacts import ArtifactStore
from redco.env.commands import JsonValue
from redco.env.snapshots import SnapshotStore


def test_snapshot_roundtrip_uses_verified_content_addressed_state(tmp_path: Path) -> None:
    snapshots = SnapshotStore(ArtifactStore(tmp_path / "cas"))
    state: dict[str, JsonValue] = {
        "answer": 42,
        "items": ["a", "b"],
        "valid": True,
    }

    snapshot = snapshots.capture(checkpoint_id="theta-0", state=state)

    assert snapshots.restore(snapshot) == state
