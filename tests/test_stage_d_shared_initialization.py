from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_stage_d_protocol_manifest import _manifest

from redco.analysis.stage_d_shared_initialization import (
    StageDSharedInitializationManifest,
)
from redco.contracts import canonical_json


def _shared() -> StageDSharedInitializationManifest:
    protocol = _manifest()
    return StageDSharedInitializationManifest(
        initialization_id="shared-scaffold-v1",
        checkpoint_id=protocol.policy_identity.checkpoint_id,
        base_model_manifest_sha256=(
            protocol.policy_identity.base_model_manifest_sha256
        ),
        adapter_manifest_sha256=protocol.policy_identity.adapter_manifest_sha256,
        expected_pre_model_sha256="f" * 64,
    )


def test_shared_initialization_is_canonical_and_protocol_bound() -> None:
    shared = _shared()
    parsed = StageDSharedInitializationManifest.from_bytes(shared.to_bytes())
    assert parsed == shared
    protocol = _manifest()
    payload = json.loads(protocol.to_bytes())
    payload["shared_initialization_sha256"] = shared.manifest_sha256
    bound_protocol = type(protocol).from_bytes(canonical_json(payload))
    shared.verify_protocol(bound_protocol)


def test_shared_initialization_rejects_policy_swap() -> None:
    shared = _shared()
    protocol = _manifest()
    payload = json.loads(protocol.to_bytes())
    payload["shared_initialization_sha256"] = shared.manifest_sha256
    payload["policy_identity"]["checkpoint_id"] = "other"
    with pytest.raises(ValueError, match="differs"):
        shared.verify_protocol(type(protocol).from_bytes(canonical_json(payload)))


def test_shared_initialization_is_derived_from_retained_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "base.json"
    adapter_manifest = tmp_path / "adapter.json"
    adapter = tmp_path / "adapter.safetensors"
    base.write_bytes(b"base manifest")
    adapter_manifest.write_bytes(b"adapter manifest")
    adapter.write_bytes(b"retained adapter")
    import redco.analysis.stage_d_live_update as live_update

    observed: dict[str, object] = {}

    def state_hash(path: Path, *, base_snapshot_manifest_sha256: str) -> str:
        observed.update(path=path, base=base_snapshot_manifest_sha256)
        return "e" * 64

    monkeypatch.setattr(live_update, "adapter_file_state_sha256", state_hash)
    manifest = StageDSharedInitializationManifest.from_retained_adapter(
        initialization_id="retained-v1",
        checkpoint_id="model@commit",
        base_model_manifest_path=base,
        adapter_manifest_path=adapter_manifest,
        adapter_path=adapter,
    )
    assert manifest.expected_pre_model_sha256 == "e" * 64
    assert observed == {"path": adapter, "base": manifest.base_model_manifest_sha256}
