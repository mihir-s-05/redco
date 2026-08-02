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
from redco.analysis.stage_d_protocol_manifest import StageDProtocolManifest
from redco.analysis.stage_d_receipt_ledger import (
    LedgerSeal,
    SealedReceiptVerifier,
    StageDReceiptLedger,
    inspect_ledger,
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
from redco.contracts import canonical_json


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class SealedStageDCampaign:
    """All immutable inputs needed by the three mandatory trainer entrypoints."""

    compilation: ThreeArmCompilation
    protocol_manifest: bytes
    protocol_manifest_sha256: str
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
    protocol_manifest_bytes: bytes,
    preregistered_protocol_manifest_sha256: str,
    source_rollout_bytes: Sequence[bytes],
    collection_plan: StageDCollectionPlan,
    collection_receipt_bytes: bytes,
    preregistered_collection_plan_sha256: str,
    branch_artifact_bytes: Sequence[bytes],
    encode_action: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[int, ...]],
    render_prompt: Callable[[Mapping[str, Any]], tuple[int, ...]],
    master_seed: str,
    objective_binding_bytes: Mapping[ArmName, bytes],
    trainer_toml_bytes: Mapping[ArmName, bytes],
    objective_authorization_bytes: bytes,
    preregistered_objective_authorization_sha256: str,
    trainer_step: int,
    seq_len: int,
    allow_test_fixture_collection: bool = False,
    after_arm_authorized: Callable[[ArmName], None] | None = None,
) -> SealedStageDCampaign:
    """Seal once, then prove the pre-seal compilation survives read-only reconstruction."""
    if _sha256(protocol_manifest_bytes) != preregistered_protocol_manifest_sha256:
        raise ValueError("protocol manifest differs from its frozen SHA-256")
    protocol = StageDProtocolManifest.from_bytes(protocol_manifest_bytes)
    if protocol.manifest_sha256 != preregistered_protocol_manifest_sha256:
        raise ValueError("protocol manifest identity is inconsistent")
    genesis = ledger.genesis_binding
    if (
        genesis.protocol_manifest_sha256 != protocol.manifest_sha256
        or genesis.preregistration_sha256 != protocol.preregistration_sha256
        or genesis.config_sha256 != protocol.genesis_config_sha256
        or genesis.source_sha256 != protocol.source_sha256
        or genesis.runtime_sha256 != protocol.runtime_sha256
        or genesis.master_seed_sha256 != protocol.master_seed_sha256
    ):
        raise ValueError("ledger genesis differs from the protocol manifest")
    if (
        collection_plan.plan_sha256 != protocol.collection_plan_sha256
        or preregistered_collection_plan_sha256 != protocol.collection_plan_sha256
        or preregistered_objective_authorization_sha256
        != protocol.objective_authorization_sha256
        or trainer_step != protocol.trainer_step
        or seq_len != protocol.seq_len
    ):
        raise ValueError("compile inputs differ from the protocol manifest")
    if set(objective_binding_bytes) != {"stock", "branch-global", "local"}:
        raise ValueError("Stage D objective binding bytes do not cover all arms")
    if set(trainer_toml_bytes) != {"stock", "branch-global", "local"}:
        raise ValueError("Stage D trainer TOMLs do not cover all arms")
    for arm in ("stock", "branch-global", "local"):
        if _sha256(objective_binding_bytes[arm]) != protocol.arm_hash(
            "objective_binding", arm
        ):
            raise ValueError(f"{arm} objective binding differs from protocol")
        if _sha256(trainer_toml_bytes[arm]) != protocol.arm_hash("trainer_config", arm):
            raise ValueError(f"{arm} trainer config differs from protocol")
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
        if after_arm_authorized is not None:
            after_arm_authorized(batch.arm)

    seal = ledger.seal_scientific_campaign()
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
        protocol_manifest_bytes,
        preregistered_protocol_manifest_sha256,
        objective_authorization_bytes,
        objective_authorization_sha256,
        tuple(sorted(receipts.items())),
        seal,
        seal_bytes,
        seal_sha256,
    )


def recover_sealed_campaign(
    *,
    ledger_root: Path,
    expected_ledger_seal_bytes: bytes,
    protocol_manifest_bytes: bytes,
    preregistered_protocol_manifest_sha256: str,
    source_rollout_bytes: Sequence[bytes],
    collection_plan: StageDCollectionPlan,
    collection_receipt_bytes: bytes,
    branch_artifact_bytes: Sequence[bytes],
    encode_action: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[int, ...]],
    render_prompt: Callable[[Mapping[str, Any]], tuple[int, ...]],
    master_seed: str,
    objective_binding_bytes: Mapping[ArmName, bytes],
    trainer_toml_bytes: Mapping[ArmName, bytes],
    objective_authorization_bytes: bytes,
    allow_test_fixture_collection: bool = False,
) -> SealedStageDCampaign:
    """Reconstruct packaging state read-only after the one scientific seal exists."""
    if _sha256(protocol_manifest_bytes) != preregistered_protocol_manifest_sha256:
        raise ValueError("protocol manifest differs from its frozen SHA-256")
    protocol = StageDProtocolManifest.from_bytes(protocol_manifest_bytes)
    seal = LedgerSeal.from_bytes(expected_ledger_seal_bytes)
    scan = inspect_ledger(ledger_root)
    if scan.status != "sealed-valid" or scan.seal != seal:
        raise ValueError("ledger differs from the expected terminal seal")
    genesis = scan.records[0]["body"]
    if (
        genesis.get("protocol_manifest_sha256") != protocol.manifest_sha256
        or genesis.get("preregistration_sha256") != protocol.preregistration_sha256
        or genesis.get("config_sha256") != protocol.genesis_config_sha256
        or genesis.get("source_sha256") != protocol.source_sha256
        or genesis.get("runtime_sha256") != protocol.runtime_sha256
        or genesis.get("master_seed_sha256") != protocol.master_seed_sha256
        or _sha256(master_seed.encode("utf-8")) != protocol.master_seed_sha256
    ):
        raise ValueError("sealed ledger genesis differs from protocol")
    if (
        collection_plan.plan_sha256 != protocol.collection_plan_sha256
        or _sha256(objective_authorization_bytes)
        != protocol.objective_authorization_sha256
        or set(objective_binding_bytes) != {"stock", "branch-global", "local"}
        or set(trainer_toml_bytes) != {"stock", "branch-global", "local"}
    ):
        raise ValueError("sealed recovery inputs differ from protocol")
    bindings = {
        arm: ObjectiveBinding.from_bytes(value)
        for arm, value in objective_binding_bytes.items()
    }
    authorization = ObjectiveAuthorization.from_bytes(objective_authorization_bytes)
    authorization.authorize(tuple(bindings.values()))
    for arm in ("stock", "branch-global", "local"):
        if _sha256(objective_binding_bytes[arm]) != protocol.arm_hash(
            "objective_binding", arm
        ) or _sha256(trainer_toml_bytes[arm]) != protocol.arm_hash(
            "trainer_config", arm
        ):
            raise ValueError(f"sealed recovery {arm} inputs differ from protocol")
    verifier = SealedReceiptVerifier(ledger_root, seal)
    def evidence_loader(digest: str) -> bytes:
        return (ledger_root / "evidence" / digest).read_bytes()
    sources = tuple(
        SourceRollout.verify_bytes(
            value,
            verifier=verifier,
            evidence_loader=evidence_loader,
            encode_action=encode_action,
            render_prompt=render_prompt,
        )
        for value in source_rollout_bytes
    )
    ledger_source_roster = tuple(
        sorted(
            receipt["source_sha256"]
            for (kind, _), receipt in scan.receipts.items()
            if kind == "source_rollout_completed"
        )
    )
    if tuple(sorted(source.source_sha256 for source in sources)) != ledger_source_roster:
        raise ValueError("sealed recovery source roster differs from ledger")
    ledger_branch_roster = tuple(
        sorted(
            receipt["artifact_sha256"]
            for (kind, _), receipt in scan.receipts.items()
            if kind == "branch_group_artifact_completed"
        )
    )
    if tuple(sorted(_sha256(value) for value in branch_artifact_bytes)) != (
        ledger_branch_roster
    ):
        raise ValueError("sealed recovery branch roster differs from ledger")
    verify_collection_receipt(
        collection_plan,
        sources,
        collection_receipt_bytes,
        evidence_loader=evidence_loader,
        allow_fixture_only=allow_test_fixture_collection,
    )
    compilation = compile_verified_three_arm_batches(
        source_rollout_bytes,
        branch_artifact_bytes,
        verifier=verifier,
        evidence_loader=evidence_loader,
        encode_action=encode_action,
        render_prompt=render_prompt,
        master_seed=master_seed,
        objective_bindings=bindings,
        objective_authorization_bytes=objective_authorization_bytes,
        preregistered_objective_authorization_sha256=(
            protocol.objective_authorization_sha256
        ),
        trainer_step=protocol.trainer_step,
        seq_len=protocol.seq_len,
    )
    receipts = {
        receipt["arm"]: canonical_json(receipt)
        for (kind, _), receipt in scan.receipts.items()
        if kind == "stage_d_training_batch_authorization"
    }
    if set(receipts) != {"stock", "branch-global", "local"}:
        raise ValueError("sealed ledger lacks the exact three training authorizations")
    for batch in (
        compilation.stock,
        compilation.branch_global,
        compilation.local,
    ):
        _verify_stage_d_batch_authorization(
            receipts[batch.arm],
            verifier=verifier,
            batch=batch,
            sealed_batch_bytes=batch.to_bytes(),
            objective_authorization_sha256=protocol.objective_authorization_sha256,
        )
    return SealedStageDCampaign(
        compilation,
        protocol_manifest_bytes,
        protocol.manifest_sha256,
        objective_authorization_bytes,
        protocol.objective_authorization_sha256,
        tuple(sorted(receipts.items())),
        seal,
        expected_ledger_seal_bytes,
        _sha256(expected_ledger_seal_bytes),
    )
