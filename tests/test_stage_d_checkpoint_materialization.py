from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from redco.analysis.stage_d_checkpoint_evidence import (
    CheckpointMember,
    StageDCheckpointManifest,
)
from redco.analysis.stage_d_checkpoint_materialization import (
    materialize_adopted_checkpoint,
)
from redco.analysis.stage_d_training_completion import (
    StageDTrainingCompletion,
    TrainingArmCompletion,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture() -> tuple[dict[str, bytes], StageDCheckpointManifest]:
    values = {
        "STABLE": b"",
        "adapter_config.json": b"{}",
        "adapter_model.safetensors": b"adapter",
    }
    manifest = StageDCheckpointManifest(
        arm="stock",
        trainer_step=1,
        base_model_manifest_sha256="e" * 64,
        post_model_sha256="a" * 64,
        members=tuple(
            CheckpointMember(name, len(value), _sha(value))
            for name, value in sorted(values.items())
        ),
    )

    def arm_completion(arm: str) -> TrainingArmCompletion:
        if arm == "stock":
            checkpoint_sha256 = manifest.manifest_sha256
            member_sha256s = tuple(member.sha256 for member in manifest.members)
        else:
            checkpoint_sha256 = _sha(f"checkpoint-{arm}".encode())
            member_sha256s = (_sha(f"member-{arm}".encode()),)
        return TrainingArmCompletion(
            arm=arm,  # type: ignore[arg-type]
            post_model_sha256=manifest.post_model_sha256,
            checkpoint_manifest_sha256=checkpoint_sha256,
            checkpoint_member_sha256s=member_sha256s,
            metrics_sha256=_sha(f"metrics-{arm}".encode()),
            reload_evidence_sha256=_sha(f"reload-{arm}".encode()),
            reload_output_sha256=_sha(f"output-{arm}".encode()),
            reload_process_result_sha256s=(
                _sha(f"process-1-{arm}".encode()),
                _sha(f"process-2-{arm}".encode()),
            ),
        )

    completion = StageDTrainingCompletion(
        campaign_manifest_sha256="1" * 64,
        protocol_manifest_sha256="2" * 64,
        trainer_ledger_head_sha256="3" * 64,
        trainer_record_count=1,
        record_sha256s=("3" * 64,),
        evidence_sha256s=tuple(
            sorted(
                {
                    manifest.manifest_sha256,
                    *(member.sha256 for member in manifest.members),
                }
            )
        ),
        arms=tuple(arm_completion(arm) for arm in ("stock", "branch-global", "local")),  # type: ignore[arg-type]
    )
    entries = {
        "completion.json": completion.to_bytes(),
        f"evidence/{manifest.manifest_sha256}": manifest.to_bytes(),
        **{f"evidence/{_sha(value)}": value for value in values.values()},
    }
    return entries, manifest


def test_materialization_is_exact_read_only_and_restartable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries, expected = _fixture()
    monkeypatch.setattr(
        "redco.analysis.stage_d_checkpoint_evidence.adapter_file_state_sha256",
        lambda *_args, **_kwargs: expected.post_model_sha256,
    )
    destination = tmp_path / "adopted" / "stock"
    assert (
        materialize_adopted_checkpoint(
            training_entries=entries,
            arm="stock",
            destination=destination,
        )
        == expected
    )
    assert (
        materialize_adopted_checkpoint(
            training_entries=entries,
            arm="stock",
            destination=destination,
        )
        == expected
    )
    assert {path.name for path in destination.iterdir()} == {
        "STABLE",
        "adapter_config.json",
        "adapter_model.safetensors",
    }


def test_materialization_rejects_missing_or_mutated_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries, expected = _fixture()
    monkeypatch.setattr(
        "redco.analysis.stage_d_checkpoint_evidence.adapter_file_state_sha256",
        lambda *_args, **_kwargs: expected.post_model_sha256,
    )
    missing = dict(entries)
    missing.pop(f"evidence/{expected.members[0].sha256}")
    with pytest.raises(ValueError, match="lacks checkpoint member"):
        materialize_adopted_checkpoint(
            training_entries=missing,
            arm="stock",
            destination=tmp_path / "missing",
        )
    mutated = dict(entries)
    mutated[f"evidence/{expected.members[0].sha256}"] = b"changed"
    with pytest.raises(ValueError, match="member bytes differ"):
        materialize_adopted_checkpoint(
            training_entries=mutated,
            arm="stock",
            destination=tmp_path / "mutated",
        )
