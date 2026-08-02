from __future__ import annotations

from pathlib import Path

import pytest

from redco.analysis.stage_d_handoff_coordinator import StageDHandoffCoordinator


def _sha(character: str) -> str:
    return character * 64


def test_inspection_rejects_adoption_manifest_of_the_wrong_kind(tmp_path: Path) -> None:
    coordinator = StageDHandoffCoordinator.create(
        tmp_path / "handoff",
        preregistration_sha256=_sha("1"),
        protocol_manifest_sha256=_sha("2"),
        handoff_policy_sha256=_sha("3"),
    )
    adoption = coordinator._install_adoption("training", [("manifest.json", b"payload")])
    coordinator._append_unlocked(
        "campaign_adopted",
        {
            "campaign_bundle_manifest_sha256": _sha("4"),
            "adoption_manifest_sha256": adoption,
        },
    )
    with pytest.raises(ValueError, match="adoption kind differs"):
        coordinator.inspect()
