from __future__ import annotations

from pathlib import Path

from redco.env.artifacts import ArtifactLedger, ArtifactStore


def test_artifacts_are_content_addressed_and_versioned(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = ArtifactLedger(store)

    first = ledger.write_json("answer", {"value": 1})
    second = ledger.write_json("answer", {"value": 2})
    duplicate = store.put_json({"value": 1})

    assert first.version == 0
    assert second.version == 1
    assert first.ref == duplicate
    assert store.get_bytes(first.ref) == b'{"value":1}'
    assert ledger.latest("answer") == second
