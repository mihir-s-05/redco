"""One-shot Stage-D compile, authorize, seal, and read-only revalidation transaction."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_arm_contracts import ThreeArmCompilation
from redco.analysis.stage_d_collection import (
    StageDCollectionPlan,
    verify_collection_receipt,
)
from redco.analysis.stage_d_objective_binding import (
    ArmName,
    ObjectiveAuthorization,
    ObjectiveBinding,
)
from redco.analysis.stage_d_receipt_ledger import (
    LedgerSeal,
    SealedReceiptVerifier,
    StageDReceiptLedger,
)
from redco.analysis.stage_d_scientific_branch_group import BranchGroupArtifact
from redco.analysis.stage_d_source_contracts import SourceRollout
from redco.analysis.stage_d_three_arm_bridge import (
    _compile_three_arm_batches,
    compile_verified_three_arm_batches,
)
from redco.analysis.stage_d_three_arm_prime import (
    _verify_stage_d_batch_authorization,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class SealedStageDCampaign:
    """All immutable inputs needed by the three mandatory trainer entrypoints."""

    compilation: ThreeArmCompilation
    objective_authorization: bytes
    objective_authorization_sha256: str
    batch_authorization_receipts: tuple[tuple[ArmName, bytes], ...]
    ledger_seal: LedgerSeal
    ledger_seal_bytes: bytes
    ledger_seal_sha256: str


def compile_authorize_seal_campaign(
    *,
    ledger: StageDReceiptLedger,
    ledger_root: Path,
    source_rollout_bytes: Sequence[bytes],
    collection_plan: StageDCollectionPlan,
    collection_receipt_bytes: bytes,
    preregistered_collection_plan_sha256: str,
    branch_artifact_bytes: Sequence[bytes],
    encode_action: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[int, ...]],
    render_prompt: Callable[[Mapping[str, Any]], tuple[int, ...]],
    master_seed: str,
    objective_binding_bytes: Mapping[ArmName, bytes],
    objective_authorization_bytes: bytes,
    preregistered_objective_authorization_sha256: str,
    trainer_step: int,
    seq_len: int,
    allow_test_fixture_collection: bool = False,
) -> SealedStageDCampaign:
    """Seal once, then prove the pre-seal compilation survives read-only reconstruction."""
    if _sha256(objective_authorization_bytes) != (preregistered_objective_authorization_sha256):
        raise ValueError("objective authorization differs from preregistration")
    if (
        collection_plan.plan_sha256 != preregistered_collection_plan_sha256
        or _sha256(collection_plan.to_bytes()) != preregistered_collection_plan_sha256
    ):
        raise ValueError("source collection plan differs from preregistration")
    authorization = ObjectiveAuthorization.from_bytes(objective_authorization_bytes)
    bindings = {
        arm: ObjectiveBinding.from_bytes(value) for arm, value in objective_binding_bytes.items()
    }
    if set(bindings) != {"stock", "branch-global", "local"}:
        raise ValueError("Stage D objective binding bytes do not cover all arms")
    authorization.authorize(tuple(bindings.values()))

    sources = tuple(
        SourceRollout.verify_bytes(
            value,
            verifier=ledger,
            evidence_loader=lambda digest: (ledger_root / "evidence" / digest).read_bytes(),
            encode_action=encode_action,
            render_prompt=render_prompt,
        )
        for value in source_rollout_bytes
    )
    supplied_source_sha256s = tuple(sorted(source.source_sha256 for source in sources))
    if supplied_source_sha256s != ledger.completed_source_sha256s:
        raise ValueError("campaign source roster differs from every completed ledger source")
    collection_receipt_sha256 = verify_collection_receipt(
        collection_plan,
        sources,
        collection_receipt_bytes,
        evidence_loader=lambda digest: (ledger_root / "evidence" / digest).read_bytes(),
        allow_fixture_only=allow_test_fixture_collection,
    )
    collection_plan_sha256 = ledger.put_evidence(collection_plan.to_bytes())
    if collection_plan_sha256 != preregistered_collection_plan_sha256:
        raise ValueError("ledger stored different source collection plan bytes")
    if ledger.put_evidence(collection_receipt_bytes) != collection_receipt_sha256:
        raise ValueError("ledger stored different source collection receipt bytes")
    artifacts = tuple(
        (
            _sha256(value),
            BranchGroupArtifact.verify_bytes(
                value,
                verifier=ledger,
                encode_action=encode_action,
                render_prompt=render_prompt,
                master_seed=master_seed,
            ),
        )
        for value in branch_artifact_bytes
    )
    supplied_artifact_sha256s = tuple(sorted(digest for digest, _ in artifacts))
    if supplied_artifact_sha256s != ledger.completed_branch_artifact_sha256s:
        raise ValueError(
            "campaign branch roster differs from every completed ledger artifact"
        )
    compilation = _compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=trainer_step,
        seq_len=seq_len,
        evidence_class="live",
        objective_bindings=bindings,
    )
    objective_authorization_sha256 = ledger.put_evidence(objective_authorization_bytes)
    if objective_authorization_sha256 != preregistered_objective_authorization_sha256:
        raise ValueError("ledger stored different objective authorization bytes")
    artifact_sha256s = tuple(sorted(ledger.put_evidence(value) for value in branch_artifact_bytes))
    if artifact_sha256s != compilation.local.branch_artifact_sha256s:
        raise ValueError("compiled branch artifact roster differs from ledger evidence")
    receipts: dict[ArmName, bytes] = {}
    for batch in (
        compilation.stock,
        compilation.branch_global,
        compilation.local,
    ):
        batch_bytes = batch.to_bytes()
        batch_sha256 = ledger.put_evidence(batch_bytes)
        receipt = ledger.authorize_stage_d_training_batch(
            arm=batch.arm,
            training_batch_identity=batch.batch_identity,
            sealed_batch_sha256=batch_sha256,
            objective_sha256=batch.objective_binding.objective_sha256,
            objective_authorization_sha256=objective_authorization_sha256,
            collection_plan_sha256=collection_plan_sha256,
            collection_receipt_sha256=collection_receipt_sha256,
            source_sha256s=batch.source_sha256s,
            branch_artifact_sha256s=batch.branch_artifact_sha256s,
            consumer_id=f"stage-d-prime:{batch.arm}:step:{batch.trainer_step}",
        )
        receipts[batch.arm] = receipt.receipt

    seal = ledger.seal()
    seal_bytes = seal.to_bytes()
    seal_sha256 = _sha256(seal_bytes)
    verifier = SealedReceiptVerifier(ledger_root, seal)
    reconstructed = compile_verified_three_arm_batches(
        source_rollout_bytes,
        branch_artifact_bytes,
        verifier=verifier,
        evidence_loader=lambda digest: (ledger_root / "evidence" / digest).read_bytes(),
        encode_action=encode_action,
        render_prompt=render_prompt,
        master_seed=master_seed,
        objective_bindings=bindings,
        objective_authorization_bytes=objective_authorization_bytes,
        preregistered_objective_authorization_sha256=(preregistered_objective_authorization_sha256),
        trainer_step=trainer_step,
        seq_len=seq_len,
    )
    for original, rebuilt in zip(
        (
            compilation.stock,
            compilation.branch_global,
            compilation.local,
        ),
        (reconstructed.stock, reconstructed.branch_global, reconstructed.local),
        strict=True,
    ):
        if original.to_bytes() != rebuilt.to_bytes():
            raise ValueError("post-seal Stage D compilation changed trainer bytes")
        _verify_stage_d_batch_authorization(
            receipts[original.arm],
            verifier=verifier,
            batch=rebuilt,
            sealed_batch_bytes=rebuilt.to_bytes(),
            objective_authorization_sha256=objective_authorization_sha256,
        )
    return SealedStageDCampaign(
        reconstructed,
        objective_authorization_bytes,
        objective_authorization_sha256,
        tuple(sorted(receipts.items())),
        seal,
        seal_bytes,
        seal_sha256,
    )
