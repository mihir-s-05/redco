"""Versioned compiler facade for the three Stage-D scientific arms."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from redco.algo.branching import (
    inclusive_group_mean_advantages,
    leave_one_out_advantages,
    trajectory_rloo,
)
from redco.analysis.stage_d_arm_contracts import (
    ArmTrainerRecord,
    RecordKind,
    SealedArmBatch,
    ThreeArmCompilation,
    _batch_identity,
    _common_branch_layout_digest,
    _require_sha256,
    _sha256,
)
from redco.analysis.stage_d_objective_binding import (
    ArmName,
    ObjectiveAuthorization,
    ObjectiveBinding,
    fixture_objective_binding,
)
from redco.analysis.stage_d_receipt_ledger import SealedReceiptVerifier
from redco.analysis.stage_d_scientific_branch_group import (
    BranchGroupArtifact,
    ReceiptVerifier,
)
from redco.analysis.stage_d_source_contracts import (
    DecisionProvenance,
    FrozenTrainingSequence,
    RolloutDecision,
    SourceRollout,
)
from redco.analysis.stage_d_training_bridge import policy_identity_sha256

__all__ = [
    "ArmName",
    "ArmTrainerRecord",
    "DecisionProvenance",
    "FrozenTrainingSequence",
    "RecordKind",
    "RolloutDecision",
    "SealedArmBatch",
    "SourceRollout",
    "ThreeArmCompilation",
    "compile_three_arm_batches",
    "compile_verified_three_arm_batches",
]


def compile_three_arm_batches(
    source_rollouts: Sequence[SourceRollout],
    branch_artifacts: Sequence[tuple[str, BranchGroupArtifact]],
    *,
    trainer_step: int,
    seq_len: int,
    allow_fixture_only: bool = False,
) -> ThreeArmCompilation:
    """Compile deterministic fixture inputs; live inputs require raw verification."""
    if not allow_fixture_only:
        raise ValueError("in-memory fixture-only compilation must be explicit")
    if any(source.evidence_class != "fixture-only" for source in source_rollouts):
        raise ValueError("live evidence must enter through canonical byte verification")
    bindings: dict[ArmName, ObjectiveBinding] = {
        arm: fixture_objective_binding(arm) for arm in ("stock", "branch-global", "local")
    }
    ObjectiveAuthorization(
        "fixture-only",
        tuple(sorted((arm, binding.objective_sha256) for arm, binding in bindings.items())),
    ).authorize(tuple(bindings.values()))
    return _compile_three_arm_batches(
        source_rollouts,
        branch_artifacts,
        trainer_step=trainer_step,
        seq_len=seq_len,
        evidence_class="fixture-only",
        objective_bindings=bindings,
    )


def compile_verified_three_arm_batches(
    source_rollout_bytes: Sequence[bytes],
    branch_artifact_bytes: Sequence[bytes],
    *,
    verifier: ReceiptVerifier,
    evidence_loader: Callable[[str], bytes],
    encode_action: Callable[[Mapping[str, Any], Mapping[str, Any]], Sequence[int]]
    | None = None,
    validate_action: Callable[
        [Mapping[str, Any], Mapping[str, Any], Sequence[int]], None
    ]
    | None = None,
    render_prompt: Callable[[Mapping[str, Any]], tuple[int, ...]],
    master_seed: str,
    objective_bindings: Mapping[ArmName, ObjectiveBinding],
    objective_authorization_bytes: bytes,
    preregistered_objective_authorization_sha256: str,
    trainer_step: int,
    seq_len: int,
) -> ThreeArmCompilation:
    """Compile live arms only after re-verifying every canonical producer artifact."""
    if type(verifier) is not SealedReceiptVerifier:
        raise ValueError("live compilation requires an out-of-band sealed ledger verifier")
    sources = tuple(
        SourceRollout.verify_bytes(
            value,
            verifier=verifier,
            evidence_loader=evidence_loader,
            encode_action=encode_action,
            validate_action=validate_action,
            render_prompt=render_prompt,
        )
        for value in source_rollout_bytes
    )
    artifacts = tuple(
        (
            _sha256(value),
            BranchGroupArtifact.verify_bytes(
                value,
                verifier=verifier,
                encode_action=encode_action,
                validate_action=validate_action,
                render_prompt=render_prompt,
                master_seed=master_seed,
            ),
        )
        for value in branch_artifact_bytes
    )
    bindings = dict(objective_bindings)
    if set(bindings) != {"stock", "branch-global", "local"}:
        raise ValueError("live objective binding roster differs")
    if _require_sha256(
        preregistered_objective_authorization_sha256,
        "preregistered objective authorization sha256",
    ) != _sha256(objective_authorization_bytes):
        raise ValueError("objective authorization differs from preregistered bytes")
    objective_authorization = ObjectiveAuthorization.from_bytes(objective_authorization_bytes)
    objective_authorization.authorize(tuple(bindings.values()))
    return _compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=trainer_step,
        seq_len=seq_len,
        evidence_class="live",
        objective_bindings=bindings,
    )


def _compile_three_arm_batches(
    source_rollouts: Sequence[SourceRollout],
    branch_artifacts: Sequence[tuple[str, BranchGroupArtifact]],
    *,
    trainer_step: int,
    seq_len: int,
    evidence_class: Literal["live", "fixture-only"],
    objective_bindings: Mapping[ArmName, ObjectiveBinding],
) -> ThreeArmCompilation:
    """Compile stock plus branch-layout-matched global-LOO and local-LOO batches."""
    sources = tuple(
        sorted(source_rollouts, key=lambda source: (source.group_id, source.rollout_id))
    )
    artifacts = tuple(
        sorted(
            branch_artifacts,
            key=lambda item: (
                item[1].commitment.rollout_id,
                item[1].commitment.target_ordinal,
                item[0],
            ),
        )
    )
    if len(sources) < 2:
        raise ValueError("trajectory LOO requires at least two source rollouts")
    if type(trainer_step) is not int or trainer_step < 1:
        raise ValueError("trainer_step must be positive")
    if type(seq_len) is not int or seq_len < 1:
        raise ValueError("seq_len must be positive")
    if len({source.rollout_id for source in sources}) != len(sources):
        raise ValueError("source rollout IDs must be unique")
    sources_by_group: dict[str, list[SourceRollout]] = {}
    for source in sources:
        sources_by_group.setdefault(source.group_id, []).append(source)
    if any(len(members) < 2 for members in sources_by_group.values()):
        raise ValueError("every trajectory group requires at least two source rollouts")
    source_ledger_ids = {
        decision.provenance.ledger_id for source in sources for decision in source.decisions
    }
    if len(source_ledger_ids) != 1:
        raise ValueError("one compilation must use one source receipt ledger")
    evidence_classes = {source.evidence_class for source in sources}
    if evidence_classes != {evidence_class}:
        raise ValueError("one compilation cannot mix live and fixture evidence")
    deployed_manifests = {source.base_model_manifest_sha256 for source in sources}
    if len(deployed_manifests) != 1:
        raise ValueError("source rollouts mix base model manifests")
    source_by_rollout = {source.rollout_id: source for source in sources}
    _validate_complete_target_rosters(artifacts)

    policy_hashes = {
        policy_identity_sha256(decision.action.key)
        for source in sources
        for decision in source.decisions
    }
    policy_hashes.update(
        policy_identity_sha256(arm.action.key) for _, artifact in artifacts for arm in artifact.arms
    )
    if len(policy_hashes) != 1:
        raise ValueError("three-arm compiler inputs mix behavior policies")
    policy_sha256 = policy_hashes.pop()
    if any(
        decision.action.key.base_model_manifest_sha256 != source.base_model_manifest_sha256
        for source in sources
        for decision in source.decisions
    ):
        raise ValueError("source base model manifest differs from its behavior actions")

    artifacts_by_rollout: dict[str, list[tuple[str, BranchGroupArtifact, RolloutDecision]]] = {}
    targeted_decisions: set[tuple[str, str]] = set()
    for artifact_sha256, artifact in artifacts:
        _require_sha256(artifact_sha256, "branch artifact sha256")
        if artifact_sha256 != _sha256(artifact.to_bytes()):
            raise ValueError("branch artifact sha256 differs from immutable bytes")
        if artifact.commitment.branch_count != 4 or len(artifact.arms) != 4:
            raise ValueError("Stage D requires exactly K=4 complete branch records")
        artifact_source = source_by_rollout.get(artifact.commitment.rollout_id)
        if artifact_source is None:
            raise ValueError("branch artifact references an unknown source rollout")
        if not artifact_source.branch_eligible:
            raise ValueError("branch artifact references a topology-ineligible source rollout")
        if artifact.commitment.group_id != artifact_source.group_id:
            raise ValueError("branch artifact group differs from source trajectory group")
        matches = [
            decision
            for decision in artifact_source.decisions
            if decision.action.digest == artifact.recorded_action.digest
        ]
        if len(matches) != 1:
            raise ValueError("branch artifact must match exactly one source decision")
        decision = matches[0]
        if (
            artifact.commitment.target_roster != artifact_source.child_target_roster
            or artifact.commitment.target_id != decision.target_id
        ):
            raise ValueError("branch commitment differs from the full child-decision roster")
        if decision.event_address != artifact.commitment.target_address:
            raise ValueError("branch artifact target address differs from source decision")
        if (
            artifact.commitment.ledger_id != decision.provenance.ledger_id
            or decision.provenance.target_commitment_receipt_sha256
            != artifact.commitment.receipt_sha256
        ):
            raise ValueError("branch artifact and source decision cross receipt ledgers")
        if not decision.provenance.branch_selected:
            raise ValueError("branch artifact names an unselected source decision")
        if artifact.commitment.commitment_sequence >= decision.provenance.request_sequence:
            raise ValueError("branch target was not committed before its source action")
        key = (artifact_source.rollout_id, decision.decision_id)
        if key in targeted_decisions:
            raise ValueError("one source decision cannot be targeted twice")
        targeted_decisions.add(key)
        expected_weight = decision.outer_weight / len(artifact.arms)
        if any(arm.record_weight != expected_weight for arm in artifact.arms):
            raise ValueError("branch artifact weight differs from source decision weight")
        artifacts_by_rollout.setdefault(artifact_source.rollout_id, []).append(
            (artifact_sha256, artifact, decision)
        )
    if not artifacts_by_rollout:
        raise ValueError("scientific compilation requires at least one selected target")
    selected_decisions = {
        (source.rollout_id, decision.decision_id)
        for source in sources
        if source.branch_eligible
        for decision in source.decisions
        if decision.provenance.branch_selected
    }
    if selected_decisions != targeted_decisions:
        raise ValueError("selected source decisions and branch artifacts differ")

    source_advantages: dict[str, tuple[float, float]] = {}
    for group_id in sorted(sources_by_group):
        members = sources_by_group[group_id]
        rewards = tuple(source.reward for source in members)
        stock_advantages = inclusive_group_mean_advantages(rewards)
        trajectory_advantages = trajectory_rloo(rewards)
        for source, stock_advantage, trajectory_advantage in zip(
            members,
            stock_advantages,
            trajectory_advantages,
            strict=True,
        ):
            source_advantages[source.rollout_id] = (
                stock_advantage,
                trajectory_advantage,
            )

    artifacts_by_group: dict[str, list[tuple[str, BranchGroupArtifact]]] = {}
    for artifact_sha256, artifact in artifacts:
        source = source_by_rollout[artifact.commitment.rollout_id]
        artifacts_by_group.setdefault(source.group_id, []).append(
            (artifact_sha256, artifact)
        )
    global_by_key: dict[tuple[str, str, int], float] = {}
    for group_id in sorted(artifacts_by_group):
        group_artifacts = artifacts_by_group[group_id]
        flat_arms = [arm for _, artifact in group_artifacts for arm in artifact.arms]
        global_advantages = leave_one_out_advantages(
            tuple(arm.q_value for arm in flat_arms)
        )
        offset = 0
        for _, artifact in group_artifacts:
            for arm in artifact.arms:
                global_by_key[
                    (
                        artifact.commitment.rollout_id,
                        artifact.commitment.target_id,
                        arm.action_slot,
                    )
                ] = global_advantages[offset]
                offset += 1

    stock_records: list[ArmTrainerRecord] = []
    local_records: list[ArmTrainerRecord] = []
    global_records: list[ArmTrainerRecord] = []
    for source in sources:
        stock_advantage, trajectory_advantage = source_advantages[source.rollout_id]
        stock_records.extend(_stock_records(source, stock_advantage))
        for decision in source.decisions:
            if (
                source.branch_eligible
                and (source.rollout_id, decision.decision_id) in targeted_decisions
            ):
                continue
            local_records.append(_decision_record("local", source, decision, trajectory_advantage))
            global_records.append(
                _decision_record("branch-global", source, decision, trajectory_advantage)
            )
        for _, artifact, decision in artifacts_by_rollout.get(source.rollout_id, []):
            for arm in artifact.arms:
                local_records.append(
                    _branch_record(
                        "local", source, decision, artifact, arm.action_slot, arm.advantage
                    )
                )
                global_records.append(
                    _branch_record(
                        "branch-global",
                        source,
                        decision,
                        artifact,
                        arm.action_slot,
                        global_by_key[
                            (source.rollout_id, artifact.commitment.target_id, arm.action_slot)
                        ],
                    )
                )

    source_hashes = tuple(sorted(source.source_sha256 for source in sources))
    artifact_hashes = tuple(sorted(digest for digest, _ in artifacts))
    stock_batch = _seal(
        "stock",
        stock_records,
        source_hashes,
        (),
        evidence_class,
        objective_bindings["stock"],
        policy_sha256,
        trainer_step,
        seq_len,
    )
    global_batch = _seal(
        "branch-global",
        global_records,
        source_hashes,
        artifact_hashes,
        evidence_class,
        objective_bindings["branch-global"],
        policy_sha256,
        trainer_step,
        seq_len,
    )
    local_batch = _seal(
        "local",
        local_records,
        source_hashes,
        artifact_hashes,
        evidence_class,
        objective_bindings["local"],
        policy_sha256,
        trainer_step,
        seq_len,
    )
    layout_digest = _common_branch_layout_digest(
        global_batch.records,
        local_batch.records,
    )
    return ThreeArmCompilation(
        stock_batch,
        global_batch,
        local_batch,
        layout_digest,
    )


def _stock_records(
    source: SourceRollout,
    advantage: float,
) -> tuple[ArmTrainerRecord, ...]:
    return tuple(
        ArmTrainerRecord(
            "stock",
            "stock-trajectory",
            source.source_sha256,
            source.group_id,
            source.rollout_id,
            None,
            None,
            None,
            sequence.token_ids,
            sequence.mask,
            sequence.behavior_logprobs,
            sequence.temperatures,
            tuple(advantage if selected else 0.0 for selected in sequence.mask),
            None,
            None,
        )
        for sequence in source.stock_sequences
    )


def _decision_record(
    arm: Literal["branch-global", "local"],
    source: SourceRollout,
    decision: RolloutDecision,
    advantage: float,
) -> ArmTrainerRecord:
    prompt = decision.action.key.prompt_token_ids
    action = decision.action.action_token_ids
    prompt_count = len(prompt)
    return ArmTrainerRecord(
        arm,
        "untargeted-decision",
        source.source_sha256,
        source.group_id,
        source.rollout_id,
        decision.decision_id,
        None,
        None,
        (*prompt, *action),
        (False,) * prompt_count + (True,) * len(action),
        (0.0,) * prompt_count + decision.action.behavior_logprobs,
        (decision.action.key.sampler.temperature,) * (prompt_count + len(action)),
        (0.0,) * prompt_count + (advantage,) * len(action),
        (0.0,) * prompt_count + (float(decision.outer_weight),) * len(action),
        decision.outer_weight,
    )


def _branch_record(
    arm_name: Literal["branch-global", "local"],
    source: SourceRollout,
    decision: RolloutDecision,
    artifact: BranchGroupArtifact,
    action_slot: int,
    advantage: float,
) -> ArmTrainerRecord:
    arm = artifact.arms[action_slot]
    prompt = arm.action.key.prompt_token_ids
    action = arm.action.action_token_ids
    prompt_count = len(prompt)
    weight = decision.outer_weight / len(artifact.arms)
    return ArmTrainerRecord(
        arm_name,
        "target-branch",
        source.source_sha256,
        source.group_id,
        source.rollout_id,
        decision.decision_id,
        artifact.commitment.target_id,
        action_slot,
        (*prompt, *action),
        (False,) * prompt_count + (True,) * len(action),
        (0.0,) * prompt_count + arm.action.behavior_logprobs,
        (arm.action.key.sampler.temperature,) * (prompt_count + len(action)),
        (0.0,) * prompt_count + (advantage,) * len(action),
        (0.0,) * prompt_count + (float(weight),) * len(action),
        weight,
    )


def _seal(
    arm: ArmName,
    records: Sequence[ArmTrainerRecord],
    source_hashes: tuple[str, ...],
    artifact_hashes: tuple[str, ...],
    evidence_class: Literal["live", "fixture-only"],
    objective_binding: ObjectiveBinding,
    policy_sha256: str,
    trainer_step: int,
    seq_len: int,
) -> SealedArmBatch:
    record_tuple = tuple(records)
    identity = _batch_identity(
        arm,
        record_tuple,
        source_hashes,
        artifact_hashes,
        evidence_class,
        objective_binding,
        policy_sha256,
        trainer_step,
        seq_len,
    )
    return SealedArmBatch(
        arm,
        record_tuple,
        source_hashes,
        artifact_hashes,
        evidence_class,
        objective_binding,
        policy_sha256,
        trainer_step,
        seq_len,
        identity,
    )


def _validate_complete_target_rosters(
    artifacts: Sequence[tuple[str, BranchGroupArtifact]],
) -> None:
    by_rollout: dict[str, list[BranchGroupArtifact]] = {}
    for _, artifact in artifacts:
        by_rollout.setdefault(artifact.commitment.rollout_id, []).append(artifact)
    for rollout_id, members in by_rollout.items():
        roster = members[0].commitment.target_roster
        if any(member.commitment.target_roster != roster for member in members):
            raise ValueError(f"rollout {rollout_id} mixes target rosters")
        ordinals = [member.commitment.target_ordinal for member in members]
        if len(set(ordinals)) != len(ordinals):
            raise ValueError(f"rollout {rollout_id} repeats a selected target")
