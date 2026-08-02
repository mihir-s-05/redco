from __future__ import annotations

import hashlib
import json

import pytest

from redco.analysis.stage_d_protocol_manifest import (
    StageDPolicyIdentity,
    StageDProtocolManifest,
)
from redco.contracts import canonical_json


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest() -> StageDProtocolManifest:
    identity = StageDPolicyIdentity(
        checkpoint_id="model@commit",
        base_model_manifest_sha256="1" * 64,
        adapter_manifest_sha256="2" * 64,
        tokenizer_manifest_sha256="3" * 64,
        renderer_manifest_sha256="4" * 64,
        sampler_conformance_manifest_sha256="5" * 64,
        resolved_agent_sampling_law_sha256="6" * 64,
        resolved_train_client_sha256="7" * 64,
    )
    return StageDProtocolManifest(
        preregistration_sha256="8" * 64,
        dependency_stack_sha256="0" * 64,
        genesis_config_sha256="9" * 64,
        master_seed_sha256="a" * 64,
        source_sha256="b" * 64,
        runtime_sha256="c" * 64,
        source_eval_config_sha256="d" * 64,
        scientific_eval_config_sha256="e" * 64,
        heldout_eval_config_sha256="f" * 64,
        collection_plan_sha256="0" * 64,
        evaluation_plan_sha256="0" * 64,
        decision_rule_sha256="1" * 64,
        reload_probe_sha256="2" * 64,
        shared_initialization_sha256="a" * 64,
        objective_authorization_sha256="3" * 64,
        objective_binding_sha256s=(
            ("stock", "4" * 64),
            ("branch-global", "5" * 64),
            ("local", "6" * 64),
        ),
        trainer_config_sha256s=(
            ("stock", "7" * 64),
            ("branch-global", "8" * 64),
            ("local", "9" * 64),
        ),
        policy_identity=identity,
        arm_order=("stock", "branch-global", "local"),
        branch_global_scope="within-source-group-all-target-branches-v1",
        trainer_step=1,
        seq_len=4096,
    )


def test_protocol_manifest_is_canonical_and_hash_bound(tmp_path) -> None:
    manifest = _manifest()
    encoded = manifest.to_bytes()
    assert StageDProtocolManifest.from_bytes(encoded) == manifest
    assert manifest.manifest_sha256 == _sha(encoded)
    path = tmp_path / "protocol.json"
    path.write_bytes(encoded)
    assert StageDProtocolManifest.verify_file(path, _sha(encoded)) == manifest


@pytest.mark.parametrize("checkpoint_id", [1, "", "bad\ncheckpoint", "x" * 513])
def test_protocol_rejects_malformed_checkpoint_identity(checkpoint_id: object) -> None:
    payload = json.loads(_manifest().to_bytes())
    payload["policy_identity"]["checkpoint_id"] = checkpoint_id
    with pytest.raises(ValueError, match="checkpoint_id"):
        StageDProtocolManifest.from_bytes(canonical_json(payload))


def test_protocol_rejects_noncanonical_or_changed_arm_order() -> None:
    encoded = _manifest().to_bytes()
    with pytest.raises(ValueError, match="noncanonical"):
        StageDProtocolManifest.from_bytes(encoded + b"\n")
    payload = json.loads(encoded)
    payload["arm_order"] = ["local", "branch-global", "stock"]
    with pytest.raises(ValueError, match="arm_order"):
        StageDProtocolManifest.from_bytes(canonical_json(payload))


def test_protocol_requires_positive_trainer_step() -> None:
    payload = json.loads(_manifest().to_bytes())
    payload["trainer_step"] = 0
    with pytest.raises(ValueError, match="positive"):
        StageDProtocolManifest.from_bytes(canonical_json(payload))
