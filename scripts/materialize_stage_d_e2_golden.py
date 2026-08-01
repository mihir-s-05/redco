"""Materialize the frozen synthetic C1 golden batch for the Stage-D E2 smoke."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from redco.analysis.stage_d_bridge_golden import (
    build_synthetic_golden,
    encode_synthetic_action,
    render_synthetic_prompt,
    shared_token_credit,
)
from redco.analysis.stage_d_scientific_branch_group import BranchGroupArtifact
from redco.analysis.stage_d_training_bridge import (
    ArtifactVerificationContext,
    TrainingBridgeBinding,
    compile_training_batch,
)
from redco.contracts import canonical_json

_MASTER_SEED = "stage-d-e2-synthetic-golden"
_PRIME_COMMIT = "3b22dd951cad1036d1fe8dd0a0bfc40807a9b360"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trainer-config", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--c9-patch", type=Path, required=True)
    parser.add_argument("--live-gate-patch", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("golden output directory must not already exist")

    golden = build_synthetic_golden()
    artifacts = tuple(
        BranchGroupArtifact.verify_bytes(
            value,
            verifier=golden.store,
            encode_action=encode_synthetic_action,
            render_prompt=render_synthetic_prompt,
            master_seed=_MASTER_SEED,
        )
        for value in golden.artifact_bytes
    )
    artifact_sha256s = [_sha256(value) for value in golden.artifact_bytes]
    generator_path = Path("src/redco/analysis/stage_d_bridge_golden.py")
    bridge_path = Path("src/redco/analysis/stage_d_training_bridge.py")
    runtime_manifest = {
        "prime_commit": _PRIME_COMMIT,
        "control_config_sha256": _file_sha256(args.control_config),
        "sources": {
            path: _file_sha256(Path(path))
            for path in (
                "src/redco/analysis/stage_d_live_update.py",
                "src/redco/analysis/stage_d_e2_live.py",
                "scripts/stage_d_e2_control.py",
                "scripts/run_stage_d_e2_4b_live_v1.sh",
            )
        },
        "patches": {
            str(args.c9_patch).replace("\\", "/"): _file_sha256(args.c9_patch),
            str(args.live_gate_patch).replace("\\", "/"): _file_sha256(
                args.live_gate_patch
            ),
        },
    }
    prime_runtime_sha256 = _sha256(canonical_json(runtime_manifest))
    producer_seal = _sha256(
        canonical_json(
            {
                "domain": "redco-stage-d-e2-synthetic-producer-seal-v1",
                "artifact_sha256s": artifact_sha256s,
                "generator_sha256": _file_sha256(generator_path),
                "synthetic_test_double_receipts": True,
            }
        )
    )
    batch = compile_training_batch(
        golden.artifact_bytes,
        verification_context=ArtifactVerificationContext(
            golden.store,
            encode_synthetic_action,
            render_synthetic_prompt,
            _MASTER_SEED,
        ),
        binding=TrainingBridgeBinding(
            producer_seal,
            _file_sha256(bridge_path),
            prime_runtime_sha256,
            _file_sha256(args.trainer_config),
            golden.policy_sha256,
        ),
        trainer_step=1,
        seq_len=8,
    )
    credit = shared_token_credit(artifacts)
    if not any(abs(value) > 0 for value in credit.values()):
        raise ValueError("synthetic golden structurally cancels under shared token parameters")

    args.output_dir.mkdir(parents=True)
    for index, value in enumerate(golden.artifact_bytes):
        (args.output_dir / f"artifact-{index}.json").write_bytes(value)
    batch_path = args.output_dir / "sealed-training-batch.json"
    batch_path.write_bytes(batch.to_bytes())
    manifest = {
        "schema_version": 1,
        "analysis": "stage-d-e2-synthetic-golden-v1",
        "scientific_scope": "engineering_only_no_on_policy_claim",
        "synthetic_test_double_receipts": True,
        "master_seed": _MASTER_SEED,
        "producer_seal_sha256": producer_seal,
        "artifact_sha256s": artifact_sha256s,
        "distinct_nonflat_action_count": len(
            {arm.action.action_token_ids for arm in artifacts[0].arms}
        ),
        "shared_token_credit": {str(token): value for token, value in sorted(credit.items())},
        "shared_token_gradient_structurally_nonzero": True,
        "record_count": len(batch.records),
        "training_batch_identity": batch.training_batch_identity,
        "bridge_payload_sha256": batch.payload_sha256,
        "sealed_batch_sha256": _file_sha256(batch_path),
        "expected_policy_sha256": golden.policy_sha256,
        "prime_runtime": runtime_manifest,
        "prime_runtime_sha256": prime_runtime_sha256,
        "trainer_config_sha256": _file_sha256(args.trainer_config),
        "control_config_sha256": _file_sha256(args.control_config),
        "bridge_source_sha256": _file_sha256(bridge_path),
        "generator_source_sha256": _file_sha256(generator_path),
    }
    (args.output_dir / "golden-manifest.json").write_bytes(canonical_json(manifest))


if __name__ == "__main__":
    main()
