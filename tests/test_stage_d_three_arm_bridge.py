from __future__ import annotations

import hashlib
import importlib
import json
import struct
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest
from test_stage_d_scientific_branch_group import (
    TrustedReceiptStore,
    _action,
    _fixture,
    _run,
)

from redco.analysis.stage_d_scientific_branch_group import BranchGroupArtifact
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress
from redco.analysis.stage_d_three_arm_bridge import (
    ArmTrainerRecord,
    DecisionProvenance,
    FrozenTrainingSequence,
    RolloutDecision,
    SealedArmBatch,
    SourceRollout,
    compile_three_arm_batches,
)
from redco.analysis.stage_d_three_arm_prime import (
    StageDPrimeRuntimeGate,
    audit_three_arm_prime_batch,
    materialize_prime_rollout_bytes,
    validate_prime_packed_sequences,
)
from redco.contracts import canonical_json


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decision(
    decision_id: str,
    action,
    *,
    node_kind: str,
    outer_weight: Fraction,
    sequence: int,
    event_address: PolicyEventAddress | None = None,
    target_id: str | None = None,
    verifier: TrustedReceiptStore,
    rollout_id: str,
    branch_selected: bool = False,
    target_commitment_receipt_sha256: str | None = None,
    group_id: str = "trajectory-group",
) -> RolloutDecision:
    resolved_address = event_address or PolicyEventAddress(
        0 if node_kind == "root" else 1,
        decision_id,
        sequence,
        sequence,
    )
    target_ordinal = None if target_id is None else int(target_id.removeprefix("target-"))
    reservation = verifier.issue(
        "source_policy_call_reserved",
        {
            "ledger_id": "durable-ledger",
            "ledger_offset": sequence,
            "prior_chain_sha256": f"{sequence + 10:064x}",
            "group_id": group_id,
            "rollout_id": rollout_id,
            "decision_id": decision_id,
            "node_kind": node_kind,
            "target_id": target_id,
            "target_ordinal": target_ordinal,
            "target_address": {**resolved_address.as_payload(), "turn": resolved_address.turn},
            "exact_action_key_digest": action.key.digest,
            "request_sha256": action.key.request_sha256,
            "branch_selected": branch_selected,
            "target_commitment_receipt_sha256": target_commitment_receipt_sha256,
            "recorded_action_reservation_id": (
                "fixture-recorded-action" if branch_selected else None
            ),
            "request_sequence": sequence,
        },
    )
    completion = verifier.issue(
        "source_policy_call_completed",
        {
            "ledger_id": "durable-ledger",
            "ledger_offset": sequence + 1,
            "prior_chain_sha256": f"{sequence + 11:064x}",
            "group_id": group_id,
            "rollout_id": rollout_id,
            "decision_id": decision_id,
            "request_receipt_sha256": _sha256(reservation),
            "exact_action_key_digest": action.key.digest,
            "action_digest": action.digest,
            "response_sha256": _sha256(action.to_bytes()),
            "request_sequence": sequence,
            "completion_sequence": sequence + 1,
        },
    )
    provenance = DecisionProvenance.from_receipts(
        reservation,
        completion,
        verifier=verifier,
    )
    return RolloutDecision(
        decision_id,
        resolved_address,
        action,
        node_kind,
        target_id,
        target_ordinal,
        outer_weight,
        provenance,
    )


def _sequence(decisions: tuple[RolloutDecision, ...]) -> FrozenTrainingSequence:
    token_ids: list[int] = []
    mask: list[bool] = []
    logprobs: list[float] = []
    temperatures: list[float] = []
    for decision in decisions:
        prompt = decision.action.key.prompt_token_ids
        action = decision.action.action_token_ids
        token_ids.extend((*prompt, *action))
        mask.extend((False,) * len(prompt) + (True,) * len(action))
        logprobs.extend((0.0,) * len(prompt) + decision.action.behavior_logprobs)
        temperatures.extend(
            (decision.action.key.sampler.temperature,) * (len(prompt) + len(action))
        )
    return FrozenTrainingSequence(
        tuple(token_ids),
        tuple(mask),
        tuple(logprobs),
        tuple(temperatures),
        None,
        None,
    )


def _source(
    rollout_id: str,
    reward: float,
    decisions: tuple[RolloutDecision, ...],
    source_digit: str,
    *,
    group_id: str = "trajectory-group",
) -> SourceRollout:
    sequences = tuple(_sequence((decision,)) for decision in decisions)
    return SourceRollout.fixture(
        group_id,
        rollout_id,
        reward,
        sequences,
        tuple((decision.decision_id,) for decision in decisions),
        decisions,
        tuple(str(decision.target_id) for decision in decisions if decision.node_kind == "child"),
        _sha256(f"trace-{source_digit}".encode()),
        _sha256(f"reward-{source_digit}".encode()),
        _sha256(canonical_json([sequence.to_payload() for sequence in sequences])),
        decisions[0].action.key.base_model_manifest_sha256,
    )


def _replace_source(source: SourceRollout, **changes) -> SourceRollout:
    values = {
        "group_id": source.group_id,
        "rollout_id": source.rollout_id,
        "reward": source.reward,
        "stock_sequences": source.stock_sequences,
        "stock_sequence_decision_ids": source.stock_sequence_decision_ids,
        "decisions": source.decisions,
        "child_target_roster": source.child_target_roster,
        "branch_eligible": source.branch_eligible,
        "ineligibility_reason": source.ineligibility_reason,
        "trace_sha256": source.trace_sha256,
        "reward_evidence_sha256": source.reward_evidence_sha256,
        "stock_sequences_evidence_sha256": source.stock_sequences_evidence_sha256,
        "base_model_manifest_sha256": source.base_model_manifest_sha256,
    }
    values.update(changes)
    return SourceRollout.fixture(**values)


def test_topology_ineligible_source_is_kept_in_all_arms_without_branches() -> None:
    sources, artifacts = _inputs()
    ineligible = _replace_source(
        sources[0],
        branch_eligible=False,
        ineligibility_reason="scientific scaffold has an unexpected policy-call count",
    )
    kept_artifacts = tuple(
        item for item in artifacts if item[1].commitment.rollout_id != ineligible.rollout_id
    )

    compiled = compile_three_arm_batches(
        (ineligible, *sources[1:]),
        kept_artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    )

    assert any(record.rollout_id == ineligible.rollout_id for record in compiled.stock.records)
    for batch in (compiled.branch_global, compiled.local):
        ineligible_records = tuple(
            record for record in batch.records if record.rollout_id == ineligible.rollout_id
        )
        assert ineligible_records
        assert all(record.record_kind == "untargeted-decision" for record in ineligible_records)


def _live_source_bytes() -> tuple[bytes, TrustedReceiptStore, dict[str, bytes]]:
    store = TrustedReceiptStore()
    action = _action(71, prompt_content="live root")
    decision = _decision(
        "root-live",
        action,
        node_kind="root",
        outer_weight=Fraction(1),
        sequence=0,
        verifier=store,
        rollout_id="rollout-live",
    )
    fixture = _source("rollout-live", 0.25, (decision,), "live")
    trace = b"immutable raw live trace"
    reward = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-reward-evidence-v1",
            "group_id": fixture.group_id,
            "rollout_id": fixture.rollout_id,
            "reward": fixture.reward,
        }
    )
    stock = canonical_json([sequence.to_payload() for sequence in fixture.stock_sequences])
    evidence = {_sha256(item): item for item in (trace, reward, stock)}
    payload = fixture.to_payload()
    payload.update(
        {
            "trace_sha256": _sha256(trace),
            "reward_evidence_sha256": _sha256(reward),
            "stock_sequences_evidence_sha256": _sha256(stock),
            "evidence_class": "live",
        }
    )
    source_sha256 = _sha256(
        canonical_json({"domain": "redco-stage-d-source-rollout-v1", "source": payload})
    )
    producer = store.issue(
        "source_rollout_completed",
        {
            "ledger_id": "durable-ledger",
            "ledger_offset": store.offset,
            "prior_chain_sha256": store.chain,
            "group_id": fixture.group_id,
            "rollout_id": fixture.rollout_id,
            "source_sha256": source_sha256,
            "trace_sha256": _sha256(trace),
            "reward_evidence_sha256": _sha256(reward),
            "stock_sequences_evidence_sha256": _sha256(stock),
            "base_model_manifest_sha256": fixture.base_model_manifest_sha256,
            "decision_ids": [decision.decision_id],
            "decision_completion_receipt_sha256s": [
                _sha256(decision.provenance.completion_receipt)
            ],
            "completion_sequence": store.offset,
        },
    )
    return (
        canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-source-rollout-v1",
                "source": payload,
                "source_sha256": source_sha256,
                "producer_receipt": json.loads(producer),
            }
        ),
        store,
        evidence,
    )


def _inputs(
    *,
    branch_count: int = 4,
    selected_rollouts: tuple[str, ...] = ("rollout-1", "rollout-2"),
):
    first_rewards = (1.0, 0.0, -1.0, 0.5)[:branch_count]
    second_rewards = (0.0, 1.0, 1.0, -1.0)[:branch_count]
    artifact0, _, _, fixture0 = _run(
        _fixture(
            target_ordinal=0,
            prompt_content="target zero",
            branch_count=branch_count,
            target_address=PolicyEventAddress(1, "root/midpoint-shard-0", 0, 0),
            group_id="trajectory-group",
        ),
        rewards=first_rewards,
    )
    artifact1, _, _, fixture1 = _run(
        _fixture(
            target_ordinal=1,
            prompt_content="target one",
            branch_count=branch_count,
            target_address=PolicyEventAddress(1, "root/midpoint-shard-1", 0, 0),
            rollout_id="rollout-2",
            group_id="trajectory-group",
        ),
        rewards=second_rewards,
    )
    _artifact2, _, _, fixture2 = _run(
        _fixture(
            target_ordinal=0,
            prompt_content="target two",
            branch_count=branch_count,
            target_address=PolicyEventAddress(1, "root/midpoint-shard-0", 0, 0),
            rollout_id="rollout-3",
            group_id="trajectory-group",
        ),
        rewards=(-0.5, 0.5, 0.0, 1.0)[:branch_count],
    )
    root0 = _action(41, prompt_content="root zero")
    root1 = _action(42, prompt_content="root one")
    root2 = _action(43, prompt_content="root two")
    untargeted0 = _action(45, prompt_content="untargeted zero")
    untargeted1 = _action(46, prompt_content="untargeted one")
    untargeted2 = _action(47, prompt_content="untargeted two")
    source0 = _source(
        "rollout-1",
        1.0,
        (
            _decision(
                "root-0",
                root0,
                node_kind="root",
                outer_weight=Fraction(1),
                sequence=1,
                verifier=fixture0.store,
                rollout_id="rollout-1",
            ),
            _decision(
                "midpoint-shard-0",
                fixture0.spec.recorded_action,
                node_kind="child",
                outer_weight=Fraction(1, 2),
                sequence=7,
                event_address=fixture0.spec.commitment.target_address,
                target_id="target-0",
                verifier=fixture0.store,
                rollout_id="rollout-1",
                branch_selected="rollout-1" in selected_rollouts,
                target_commitment_receipt_sha256=(
                    fixture0.spec.commitment.receipt_sha256
                    if "rollout-1" in selected_rollouts
                    else None
                ),
            ),
            _decision(
                "midpoint-shard-1",
                untargeted0,
                node_kind="child",
                outer_weight=Fraction(1, 2),
                sequence=11,
                event_address=PolicyEventAddress(1, "root/midpoint-shard-1", 0, 0),
                target_id="target-1",
                verifier=fixture0.store,
                rollout_id="rollout-1",
            ),
        ),
        "b",
    )
    source1 = _source(
        "rollout-2",
        0.0,
        (
            _decision(
                "root-1",
                root1,
                node_kind="root",
                outer_weight=Fraction(1),
                sequence=1,
                verifier=fixture1.store,
                rollout_id="rollout-2",
            ),
            _decision(
                "midpoint-shard-0",
                untargeted1,
                node_kind="child",
                outer_weight=Fraction(1, 2),
                sequence=7,
                event_address=PolicyEventAddress(1, "root/midpoint-shard-0", 0, 0),
                target_id="target-0",
                verifier=fixture1.store,
                rollout_id="rollout-2",
            ),
            _decision(
                "midpoint-shard-1",
                fixture1.spec.recorded_action,
                node_kind="child",
                outer_weight=Fraction(1, 2),
                sequence=11,
                event_address=fixture1.spec.commitment.target_address,
                target_id="target-1",
                verifier=fixture1.store,
                rollout_id="rollout-2",
                branch_selected="rollout-2" in selected_rollouts,
                target_commitment_receipt_sha256=(
                    fixture1.spec.commitment.receipt_sha256
                    if "rollout-2" in selected_rollouts
                    else None
                ),
            ),
        ),
        "d",
    )
    source2 = _source(
        "rollout-3",
        -1.0,
        (
            _decision(
                "root-2",
                root2,
                node_kind="root",
                outer_weight=Fraction(1),
                sequence=1,
                verifier=fixture2.store,
                rollout_id="rollout-3",
            ),
            _decision(
                "midpoint-shard-0",
                fixture2.spec.recorded_action,
                node_kind="child",
                outer_weight=Fraction(1, 2),
                sequence=7,
                event_address=fixture2.spec.commitment.target_address,
                target_id="target-0",
                verifier=fixture2.store,
                rollout_id="rollout-3",
            ),
            _decision(
                "midpoint-shard-1",
                untargeted2,
                node_kind="child",
                outer_weight=Fraction(1, 2),
                sequence=11,
                event_address=PolicyEventAddress(1, "root/midpoint-shard-1", 0, 0),
                target_id="target-1",
                verifier=fixture2.store,
                rollout_id="rollout-3",
            ),
        ),
        "f",
    )
    artifacts_by_rollout = {
        "rollout-1": (_sha256(artifact0.to_bytes()), artifact0),
        "rollout-2": (_sha256(artifact1.to_bytes()), artifact1),
    }
    artifacts = tuple(
        artifacts_by_rollout[rollout_id]
        for rollout_id in selected_rollouts
        if rollout_id in artifacts_by_rollout
    )
    return (source0, source1, source2), artifacts


def _minimal_group(
    group_id: str,
    prefix: str,
    rewards: tuple[float, float],
) -> tuple[tuple[SourceRollout, ...], tuple[tuple[str, BranchGroupArtifact], ...]]:
    sources: list[SourceRollout] = []
    artifacts: list[tuple[str, BranchGroupArtifact]] = []
    for index, reward in enumerate(rewards):
        rollout_id = f"{prefix}-rollout-{index}"
        artifact, _, _, fixture = _run(
            _fixture(
                target_ordinal=0,
                prompt_content=f"{prefix} target {index}",
                branch_count=4,
                target_address=PolicyEventAddress(
                    1,
                    f"root/{prefix}-child-{index}",
                    0,
                    0,
                ),
                rollout_id=rollout_id,
                group_id=group_id,
            ),
            rewards=(1.0, 0.0, -1.0, 0.5),
        )
        root = _decision(
            f"{prefix}-root-{index}",
            _action(100 + index, prompt_content=f"{prefix} root {index}"),
            node_kind="root",
            outer_weight=Fraction(1),
            sequence=1,
            verifier=fixture.store,
            rollout_id=rollout_id,
            group_id=group_id,
        )
        child = _decision(
            f"{prefix}-child-{index}",
            fixture.spec.recorded_action,
            node_kind="child",
            outer_weight=Fraction(1, 2),
            sequence=7,
            event_address=fixture.spec.commitment.target_address,
            target_id="target-0",
            verifier=fixture.store,
            rollout_id=rollout_id,
            branch_selected=True,
            target_commitment_receipt_sha256=fixture.spec.commitment.receipt_sha256,
            group_id=group_id,
        )
        untargeted = _decision(
            f"{prefix}-untargeted-{index}",
            _action(200 + index, prompt_content=f"{prefix} untargeted {index}"),
            node_kind="child",
            outer_weight=Fraction(1, 2),
            sequence=9,
            event_address=PolicyEventAddress(
                1,
                f"root/{prefix}-untargeted-{index}",
                0,
                0,
            ),
            target_id="target-1",
            verifier=fixture.store,
            rollout_id=rollout_id,
            group_id=group_id,
        )
        sources.append(
            _source(
                rollout_id,
                reward,
                (root, child, untargeted),
                f"{prefix}-{index}",
                group_id=group_id,
            )
        )
        artifacts.append((_sha256(artifact.to_bytes()), artifact))
    return tuple(sources), tuple(artifacts)


def test_campaign_compiler_combines_groups_with_within_group_baselines() -> None:
    first_sources, first_artifacts = _minimal_group("group-a", "a", (1.0, 0.0))
    second_sources, second_artifacts = _minimal_group("group-b", "b", (10.0, 8.0))
    sources = (*first_sources, *second_sources)
    artifacts = (*first_artifacts, *second_artifacts)

    compiled = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    )

    stock_by_group: dict[str, set[float]] = {}
    for record in compiled.stock.records:
        advantage = next(
            value
            for value, selected in zip(record.advantages, record.mask, strict=True)
            if selected
        )
        stock_by_group.setdefault(record.group_id, set()).add(advantage)
    assert stock_by_group == {"group-a": {-0.5, 0.5}, "group-b": {-1.0, 1.0}}
    assert compiled.stock.source_sha256s == tuple(
        sorted(source.source_sha256 for source in sources)
    )
    assert {record.group_id for record in compiled.local.records} == {
        "group-a",
        "group-b",
    }

    permuted = compile_three_arm_batches(
        tuple(reversed(sources)),
        tuple(reversed(artifacts)),
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    )
    assert permuted.stock.to_bytes() == compiled.stock.to_bytes()
    assert permuted.branch_global.to_bytes() == compiled.branch_global.to_bytes()
    assert permuted.local.to_bytes() == compiled.local.to_bytes()


def test_hand_computed_b3_k4_three_arm_arithmetic_and_layout() -> None:
    sources, artifacts = _inputs()
    compiled = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    )

    assert len(compiled.stock.records) == 9
    stock_advantages = [
        next(
            value
            for value, selected in zip(record.advantages, record.mask, strict=True)
            if selected
        )
        for record in compiled.stock.records
    ]
    assert stock_advantages == pytest.approx([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0])

    local_untargeted = [
        record for record in compiled.local.records if record.record_kind == "untargeted-decision"
    ]
    assert [
        next(
            value
            for value, selected in zip(record.advantages, record.mask, strict=True)
            if selected
        )
        for record in local_untargeted
    ] == pytest.approx([1.5, 1.5, 0.0, 0.0, -1.5, -1.5, -1.5])

    local_targets = [
        record for record in compiled.local.records if record.record_kind == "target-branch"
    ]
    global_targets = [
        record for record in compiled.branch_global.records if record.record_kind == "target-branch"
    ]
    assert len(local_targets) == len(global_targets) == 8
    local_scalars = [
        next(
            value
            for value, selected in zip(record.advantages, record.mask, strict=True)
            if selected
        )
        for record in local_targets
    ]
    global_scalars = [
        next(
            value
            for value, selected in zip(record.advantages, record.mask, strict=True)
            if selected
        )
        for record in global_targets
    ]
    assert local_scalars == pytest.approx(
        [
            7 / 6,
            -1 / 6,
            -3 / 2,
            1 / 2,
            -1 / 3,
            1,
            1,
            -5 / 3,
        ]
    )
    assert global_scalars == pytest.approx(
        [
            13 / 14,
            -3 / 14,
            -19 / 14,
            5 / 14,
            -3 / 14,
            13 / 14,
            13 / 14,
            -19 / 14,
        ]
    )
    assert all(record.rl_normalizer == Fraction(1, 8) for record in local_targets)
    assert (
        sum(
            record.rl_normalizer
            for record in compiled.local.records
            if record.rollout_id == "rollout-1"
        )
        == 2
    )
    assert compiled.common_branch_layout_sha256
    assert SealedArmBatch.verify_bytes(compiled.stock.to_bytes()) == compiled.stock
    assert SealedArmBatch.verify_bytes(compiled.branch_global.to_bytes()) == compiled.branch_global
    assert SealedArmBatch.verify_bytes(compiled.local.to_bytes()) == compiled.local

    permuted = compile_three_arm_batches(
        tuple(reversed(sources)),
        tuple(reversed(artifacts)),
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    )
    assert permuted.stock.to_bytes() == compiled.stock.to_bytes()
    assert permuted.branch_global.to_bytes() == compiled.branch_global.to_bytes()
    assert permuted.local.to_bytes() == compiled.local.to_bytes()


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("token_ids", True, "token IDs"),
        ("token_ids", -1, "token IDs"),
        ("mask", 1, "exact booleans"),
        ("behavior_logprobs", 0, "finite floats"),
        ("temperatures", 1, "finite floats"),
        ("advantages", 0, "finite floats"),
    ],
)
def test_sealed_batch_rejects_noncanonical_record_scalar_types(
    field: str,
    bad_value: object,
    message: str,
) -> None:
    sources, artifacts = _inputs()
    batch = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    ).stock
    envelope = json.loads(batch.to_bytes())
    payload = envelope["payload"]
    payload["records"][0][field][0] = bad_value
    envelope["payload_sha256"] = _sha256(canonical_json(payload))

    with pytest.raises(ValueError, match=message):
        SealedArmBatch.verify_bytes(canonical_json(envelope))


def test_contract_refactor_golden_bytes_and_hashes_are_frozen() -> None:
    sources, artifacts = _inputs()
    compiled = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    )
    assert [(_sha256(source.to_bytes()), source.source_sha256) for source in sources] == [
        (
            "24bda6d0da03c6054bb41b068549382c278adb9331a4ada5937c5939f85d12b0",
            "b8d90fb02cab38cc007f5d33bcdd78f6568f24dc091461b725be9e68a6fbc530",
        ),
        (
            "69b2c65556af2d087a804af4fe9991d1155e0b8fcf23d6386b8b7ce55fe01d23",
            "ec83007b4e29f9868012d3654241cba41fc3ba1d6852f234f949c54ad6784779",
        ),
        (
            "8b3b295f745edc862043a2b966b2527422d31733e7996dd471c55dcc711a6cf2",
            "5c01f6d441ce75222a9c66472015a5279a4955fd30d1b7810356318b1dd57e14",
        ),
    ]
    assert {
        arm: (_sha256(batch.to_bytes()), batch.batch_identity)
        for arm, batch in (
            ("stock", compiled.stock),
            ("branch-global", compiled.branch_global),
            ("local", compiled.local),
        )
    } == {
        "stock": (
            "e6f85eea4c7884f32ffd3f0a2a4cc37e1a8223ebaab1a13113d09da736c1eaa2",
            "0ad51a410f2a7747a3a2fb374834966cfd5b2c7693bb7f37ea89b8ee476e38cf",
        ),
        "branch-global": (
            "652b0e3277237018f9bcf6d66515280c3dca1c6882f7a53fcf06de2b551ab04d",
            "5224424e2ae0a37997d7f176293f33c181ea4eb962b7fe3093462774287ccc4b",
        ),
        "local": (
            "46e2163df815d30c1bdd42112966e55745a250ef89f20fa7a4c5e0c9ec9f73a1",
            "aa27ab403ac9a6ed26fd204c513e82fe7ccc93c0663ed811ea8bc06f08f445f2",
        ),
    }
    assert (
        compiled.common_branch_layout_sha256
        == "e193ac9102013141435ae96d0b2a85d6e8ee827bea0af019fa15334578a35034"
    )


def test_fixture_evidence_is_rejected_by_live_default() -> None:
    sources, artifacts = _inputs()
    with pytest.raises(ValueError, match="fixture-only"):
        compile_three_arm_batches(sources, artifacts, trainer_step=1, seq_len=64)


def test_live_source_rejects_opaque_content_addressed_trace_evidence() -> None:
    encoded, store, evidence = _live_source_bytes()
    with pytest.raises(ValueError, match="one JSON object"):
        SourceRollout.verify_bytes(
            encoded,
            verifier=store,
            evidence_loader=evidence.__getitem__,
            encode_action=lambda _request, _message: (20, 2),
            render_prompt=lambda _request: (10, 11),
        )
    with pytest.raises(TypeError):
        SourceRollout()  # type: ignore[call-arg]


def test_compiler_rejects_singleton_incomplete_roster_and_mixed_manifest() -> None:
    sources, artifacts = _inputs()
    with pytest.raises(ValueError, match="at least two"):
        compile_three_arm_batches(
            sources[:1],
            artifacts,
            trainer_step=1,
            seq_len=64,
            allow_fixture_only=True,
        )

    with pytest.raises(ValueError, match="cross groups or rollouts"):
        _replace_source(sources[1], rollout_id="rollout-1")
    with pytest.raises(ValueError, match="cross groups or rollouts"):
        _replace_source(sources[0], rollout_id="unknown-rollout")


def test_compiler_rejects_non_k4_and_mixed_policy() -> None:
    sources, artifacts = _inputs(branch_count=3)
    with pytest.raises(ValueError, match="K=4"):
        compile_three_arm_batches(
            sources,
            artifacts,
            trainer_step=1,
            seq_len=64,
            allow_fixture_only=True,
        )

    sources, artifacts = _inputs()
    off_policy = _action(44, prompt_content="root two", temperature=0.8)
    off_policy_store = TrustedReceiptStore()
    off_policy_decision = _decision(
        "root-2",
        off_policy,
        node_kind="root",
        outer_weight=Fraction(1),
        sequence=9,
        verifier=off_policy_store,
        rollout_id="rollout-3",
    )
    mixed = (*sources[:2], _source("rollout-3", -1.0, (off_policy_decision,), "f"))
    with pytest.raises(ValueError, match="mix behavior policies"):
        compile_three_arm_batches(
            mixed,
            artifacts,
            trainer_step=1,
            seq_len=64,
            allow_fixture_only=True,
        )
    one_sources, one_artifact = _inputs(selected_rollouts=("rollout-1",))
    one_selected = compile_three_arm_batches(
        one_sources,
        one_artifact,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    )
    assert sum(record.record_kind == "target-branch" for record in one_selected.local.records) == 4
    mixed = (
        sources[0],
        _replace_source(sources[1], base_model_manifest_sha256="c" * 64),
        sources[2],
    )
    with pytest.raises(ValueError, match="mix base model"):
        compile_three_arm_batches(
            mixed,
            artifacts,
            trainer_step=1,
            seq_len=64,
            allow_fixture_only=True,
        )


def test_compiler_rejects_duplicate_decision_and_stock_coverage_drift() -> None:
    sources, _ = _inputs()
    first = sources[0]
    with pytest.raises(ValueError, match="verified provenance"):
        _replace_source(
            first,
            decisions=(first.decisions[0], replace(first.decisions[1], decision_id="root-0")),
        )
    sequence = first.stock_sequences[0]
    selected = sequence.mask.index(True)
    tokens = list(sequence.token_ids)
    tokens[selected] += 1
    with pytest.raises(ValueError, match="exactly cover"):
        _replace_source(
            first,
            stock_sequences=(
                replace(sequence, token_ids=tuple(tokens)),
                *first.stock_sequences[1:],
            ),
        )
    with pytest.raises(ValueError, match="exact Prime token normalization"):
        _replace_source(
            first,
            stock_sequences=(
                replace(
                    sequence,
                    rl_weights=tuple(1.0 if selected else 0.0 for selected in sequence.mask),
                    rl_normalizer=Fraction(1),
                ),
                *first.stock_sequences[1:],
            ),
        )


def test_branch_records_bind_numerator_to_decision_denominator() -> None:
    sources, artifacts = _inputs()
    record = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    ).local.records[0]
    assert record.rl_weights is not None
    changed_weights = tuple(
        (weight * 2.0) if selected else weight
        for selected, weight in zip(record.mask, record.rl_weights, strict=True)
    )
    with pytest.raises(ValueError, match="numerator weight"):
        replace(record, rl_weights=changed_weights)


def test_sealed_batches_and_compilation_reject_cross_binding() -> None:
    sources, artifacts = _inputs()
    compiled = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    )
    with pytest.raises(ValueError, match="source roster"):
        replace(compiled.stock, source_sha256s=("0" * 64,))

    later = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=2,
        seq_len=64,
        allow_fixture_only=True,
    )
    with pytest.raises(ValueError, match="trainer steps"):
        replace(compiled, stock=later.stock)

    changed_sources = (_replace_source(sources[0], reward=0.75), *sources[1:])
    different = compile_three_arm_batches(
        changed_sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    )
    with pytest.raises(ValueError, match="source rollout sets"):
        replace(compiled, stock=different.stock)


def test_source_and_target_addresses_are_cryptographically_bound() -> None:
    sources, _artifacts = _inputs()
    source = sources[0]
    assert source.source_sha256 == _replace_source(source).source_sha256
    changed = _replace_source(source, reward=0.75)
    assert changed.source_sha256 != source.source_sha256

    wrong_address = PolicyEventAddress(1, "root/wrong", 0, 0)
    decisions = list(source.decisions)
    with pytest.raises(ValueError, match="verified provenance"):
        decisions[1] = replace(decisions[1], event_address=wrong_address)


def test_record_schema_rejects_stock_normalizer_even_without_source() -> None:
    sources, artifacts = _inputs()
    stock = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    ).stock.records[0]
    with pytest.raises(ValueError, match="exact Prime token normalization"):
        ArmTrainerRecord(
            stock.arm,
            stock.record_kind,
            stock.source_sha256,
            stock.group_id,
            stock.rollout_id,
            stock.decision_id,
            stock.target_id,
            stock.action_slot,
            stock.token_ids,
            stock.mask,
            stock.behavior_logprobs,
            stock.temperatures,
            stock.advantages,
            tuple(1.0 if selected else 0.0 for selected in stock.mask),
            Fraction(1),
        )


def test_sealed_batch_rejects_nested_tampering() -> None:
    sources, artifacts = _inputs()
    batch = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    ).local
    payload = json.loads(batch.to_bytes())
    payload["payload"]["records"][0]["advantages"][-1] = 999.0
    tampered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="digest mismatch"):
        SealedArmBatch.verify_bytes(tampered)


def test_sealed_batch_identity_rejects_non_advantage_layout_drift() -> None:
    sources, artifacts = _inputs()
    compiled = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    )
    first = compiled.branch_global.records[0]
    tokens = list(first.token_ids)
    tokens[0] += 1
    changed = replace(first, token_ids=tuple(tokens))
    with pytest.raises(ValueError, match="identity mismatch"):
        replace(
            compiled,
            branch_global=replace(
                compiled.branch_global,
                records=(changed, *compiled.branch_global.records[1:]),
            ),
        )


def test_actual_prime_packer_losses_and_gradients_match_independent_objectives() -> None:
    pytest.importorskip("prime_rl")
    sources, artifacts = _inputs()
    compiled = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    )
    audits = (
        audit_three_arm_prime_batch(compiled.stock),
        audit_three_arm_prime_batch(compiled.branch_global),
        audit_three_arm_prime_batch(compiled.local),
    )
    assert [audit.normalization_mode for audit in audits] == [
        "token",
        "decision",
        "decision",
    ]
    assert all(audit.max_gradient_error <= 1e-12 for audit in audits)


def test_actual_prime_rollout_payload_is_byte_stable_and_arm_specific() -> None:
    pytest.importorskip("prime_rl")
    sources, artifacts = _inputs()
    compiled = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    )
    payloads = tuple(
        materialize_prime_rollout_bytes(batch)
        for batch in (compiled.stock, compiled.branch_global, compiled.local)
    )
    assert all(payload for payload in payloads)
    assert len(set(payloads)) == 3
    assert payloads == tuple(
        materialize_prime_rollout_bytes(batch)
        for batch in (compiled.stock, compiled.branch_global, compiled.local)
    )


def _install_fake_distributed(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gathered_shards: list[list[tuple]] | None = None,
) -> object:
    process_group = object()
    original_import = importlib.import_module

    def all_gather_object(output, local, *, group=None):
        assert group is process_group
        values = (
            [{"error": None, "sequences": shard} for shard in gathered_shards]
            if gathered_shards is not None
            else [local]
        )
        assert len(output) == len(values)
        output[:] = values

    fake_dist = SimpleNamespace(
        is_initialized=lambda: True,
        get_world_size=lambda *, group=None: (
            len(gathered_shards) if gathered_shards is not None else 1
        ),
        all_gather_object=all_gather_object,
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_dist if name == "torch.distributed" else original_import(name),
    )
    return process_group


class _FakeTensor:
    def __init__(self, values) -> None:
        self._values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, _shape: int):
        return self

    def tolist(self):
        return self._values


def _tensor_microbatch(records) -> dict[str, object]:
    def f32(value: float) -> float:
        return float(struct.unpack("!f", struct.pack("!f", value))[0])

    weights = [f32(value) for record in records for value in (record.rl_weights or ())]
    return {
        "sequence_lengths": [len(record.token_ids) for record in records],
        "rl_normalizers": [float(record.rl_normalizer) for record in records],
        "input_ids": _FakeTensor(value for record in records for value in record.token_ids),
        "loss_mask": _FakeTensor(value for record in records for value in record.mask),
        "inference_logprobs": _FakeTensor(
            f32(value) for record in records for value in record.behavior_logprobs
        ),
        "temperatures": _FakeTensor(
            f32(value) for record in records for value in record.temperatures
        ),
        "advantages": _FakeTensor(f32(value) for record in records for value in record.advantages),
        "rl_weights": None if not weights else _FakeTensor(weights),
        "env_names": ["redco-stage-d-local" for record in records for _ in record.token_ids],
    }


def _dummy_tensor_microbatch(record, *, nonzero_advantage: bool = False):
    length = len(record.token_ids)
    return {
        "sequence_lengths": [length],
        "rl_normalizers": [0.0],
        "input_ids": _FakeTensor(record.token_ids),
        "loss_mask": _FakeTensor([False] * length),
        "inference_logprobs": _FakeTensor(record.behavior_logprobs),
        "temperatures": _FakeTensor(record.temperatures),
        "advantages": _FakeTensor([1.0 if nonzero_advantage else 0.0] + [0.0] * (length - 1)),
        "rl_weights": None,
        "env_names": ["redco-stage-d-local"] * length,
    }


def test_runtime_gate_gathers_two_dp_shards_and_skips_valid_dummy_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, artifacts = _inputs()
    batch = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    ).local
    midpoint = len(batch.records) // 2
    local_microbatches = [
        _tensor_microbatch(batch.records[:midpoint]),
        _dummy_tensor_microbatch(batch.records[0]),
    ]
    remote_sequences = [
        (
            record.token_ids,
            record.mask,
            tuple(
                float(struct.unpack("!f", struct.pack("!f", value))[0])
                for value in record.behavior_logprobs
            ),
            tuple(
                float(struct.unpack("!f", struct.pack("!f", value))[0])
                for value in record.temperatures
            ),
            tuple(
                float(struct.unpack("!f", struct.pack("!f", value))[0])
                for value in record.advantages
            ),
            (
                None
                if record.rl_weights is None
                else tuple(
                    float(struct.unpack("!f", struct.pack("!f", value))[0])
                    for value in record.rl_weights
                )
            ),
            float(record.rl_normalizer),
        )
        for record in batch.records[midpoint:]
    ]
    local_sequences = [
        (
            record.token_ids,
            record.mask,
            tuple(
                float(struct.unpack("!f", struct.pack("!f", value))[0])
                for value in record.behavior_logprobs
            ),
            tuple(
                float(struct.unpack("!f", struct.pack("!f", value))[0])
                for value in record.temperatures
            ),
            tuple(
                float(struct.unpack("!f", struct.pack("!f", value))[0])
                for value in record.advantages
            ),
            (
                None
                if record.rl_weights is None
                else tuple(
                    float(struct.unpack("!f", struct.pack("!f", value))[0])
                    for value in record.rl_weights
                )
            ),
            float(record.rl_normalizer),
        )
        for record in batch.records[:midpoint]
    ]
    process_group = _install_fake_distributed(
        monkeypatch,
        gathered_shards=[local_sequences, remote_sequences],
    )
    gate = StageDPrimeRuntimeGate(
        batch.objective_binding,
        batch,
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    gate.verify_consumed_micro_batches(
        local_microbatches,
        trainer_step=1,
        process_group=process_group,
    )


def test_runtime_gate_rejects_malformed_dummy_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, artifacts = _inputs()
    batch = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    ).local
    process_group = _install_fake_distributed(monkeypatch)
    gate = StageDPrimeRuntimeGate(
        batch.objective_binding,
        batch,
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    with pytest.raises(ValueError, match="dummy padding has nonzero advantages"):
        gate.verify_consumed_micro_batches(
            [_dummy_tensor_microbatch(batch.records[0], nonzero_advantage=True)],
            trainer_step=1,
            process_group=process_group,
        )


@pytest.mark.parametrize("delta", (-1, 1))
def test_runtime_gate_rejects_trailing_or_overrun_tensor_streams(
    monkeypatch: pytest.MonkeyPatch,
    delta: int,
) -> None:
    sources, artifacts = _inputs()
    batch = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    ).local
    microbatch = _tensor_microbatch(batch.records)
    lengths = list(microbatch["sequence_lengths"])
    lengths[-1] += delta
    microbatch["sequence_lengths"] = lengths
    process_group = _install_fake_distributed(monkeypatch)
    gate = StageDPrimeRuntimeGate(
        batch.objective_binding,
        batch,
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    with pytest.raises(ValueError, match="misaligned or trailing"):
        gate.verify_consumed_micro_batches(
            [microbatch],
            trainer_step=1,
            process_group=process_group,
        )


def test_actual_prime_tensor_conversion_passes_single_use_runtime_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("prime_rl")
    from prime_rl import transport
    from prime_rl.trainer import batch as trainer_batch
    from prime_rl.trainer import utils as trainer_utils
    from prime_rl.trainer.rl.data import DataLoader

    sources, artifacts = _inputs()
    batch = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    ).local
    examples = [
        transport.TrainingSample(
            token_ids=list(record.token_ids),
            mask=list(record.mask),
            logprobs=list(record.behavior_logprobs),
            temperatures=list(record.temperatures),
            env_name="redco-stage-d-local",
            rl_weights=list(record.rl_weights or ()),
            advantages=list(record.advantages),
            rl_normalizer=float(record.rl_normalizer),
        )
        for record in batch.records
    ]
    prepared = trainer_batch.prepare_batch(
        rollouts=examples,
        seq_len=batch.seq_len,
        num_train_workers=1,
        idxs=[0] * len(examples),
        num_loras=1,
        bin_cost=trainer_utils.build_bin_cost(None),
    )
    packed = [micro_batch for worker in prepared for micro_batch in worker]
    loader = object.__new__(DataLoader)
    tensor_micro_batches = [
        DataLoader._micro_batch_to_tensor(loader, micro_batch) for micro_batch in packed
    ]
    gate = StageDPrimeRuntimeGate(
        batch.objective_binding,
        batch,
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    process_group = _install_fake_distributed(monkeypatch)
    gate.verify_consumed_micro_batches(
        tensor_micro_batches,
        trainer_step=1,
        process_group=process_group,
    )
    with pytest.raises(ValueError, match="second training batch"):
        gate.verify_consumed_micro_batches(
            tensor_micro_batches,
            trainer_step=1,
            process_group=process_group,
        )


def test_cpu_packer_audit_rejects_every_stream_tamper() -> None:
    sources, artifacts = _inputs()
    batch = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    ).local
    sequences = [
        (
            record.token_ids,
            record.mask,
            record.behavior_logprobs,
            record.temperatures,
            record.advantages,
            record.rl_weights,
            float(record.rl_normalizer) if record.rl_normalizer is not None else None,
        )
        for record in batch.records
    ]
    assert validate_prime_packed_sequences(batch, sequences) > 0.0

    for field_index in (0, 2, 3, 4, 5, 6):
        tampered = list(sequences)
        fields = list(tampered[0])
        if field_index == 0:
            fields[0] = (fields[0][0] + 1, *fields[0][1:])
        elif field_index == 6:
            fields[6] = float(fields[6]) + 1.0
        else:
            stream = list(fields[field_index])
            stream[-1] += 0.125
            fields[field_index] = tuple(stream)
        tampered[0] = tuple(fields)
        with pytest.raises(ValueError, match="changed sealed"):
            validate_prime_packed_sequences(batch, tampered)


def test_consumed_tensor_gate_uses_exact_float32_canonicalization() -> None:
    sources, artifacts = _inputs()
    batch = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    ).local

    def f32(value: float) -> float:
        return float(struct.unpack("!f", struct.pack("!f", value))[0])

    tensor_sequences = [
        (
            record.token_ids,
            record.mask,
            tuple(f32(value) for value in record.behavior_logprobs),
            tuple(f32(value) for value in record.temperatures),
            tuple(f32(value) for value in record.advantages),
            (
                None
                if record.rl_weights is None
                else tuple(f32(value) for value in record.rl_weights)
            ),
            float(record.rl_normalizer) if record.rl_normalizer is not None else None,
        )
        for record in batch.records
    ]
    assert validate_prime_packed_sequences(batch, tensor_sequences) > 0.0


def test_runtime_gate_checks_the_actual_tensor_microbatch_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, artifacts = _inputs()
    batch = compile_three_arm_batches(
        sources,
        artifacts,
        trainer_step=1,
        seq_len=64,
        allow_fixture_only=True,
    ).local

    class FakeTensor:
        def __init__(self, values) -> None:
            self._values = list(values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def reshape(self, _shape: int):
            return self

        def tolist(self):
            return self._values

    def f32(value: float) -> float:
        return float(struct.unpack("!f", struct.pack("!f", value))[0])

    microbatch = {
        "sequence_lengths": [len(record.token_ids) for record in batch.records],
        "rl_normalizers": [float(record.rl_normalizer) for record in batch.records],
        "input_ids": FakeTensor(value for record in batch.records for value in record.token_ids),
        "loss_mask": FakeTensor(value for record in batch.records for value in record.mask),
        "inference_logprobs": FakeTensor(
            f32(value) for record in batch.records for value in record.behavior_logprobs
        ),
        "temperatures": FakeTensor(
            f32(value) for record in batch.records for value in record.temperatures
        ),
        "advantages": FakeTensor(
            f32(value) for record in batch.records for value in record.advantages
        ),
        "rl_weights": FakeTensor(
            f32(value) for record in batch.records for value in (record.rl_weights or ())
        ),
        "env_names": ["redco-stage-d-local" for record in batch.records for _ in record.token_ids],
    }
    gate = StageDPrimeRuntimeGate(
        batch.objective_binding,
        batch,
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    process_group = _install_fake_distributed(monkeypatch)
    gate.verify_consumed_micro_batches(
        [microbatch],
        trainer_step=1,
        process_group=process_group,
    )
    with pytest.raises(ValueError, match="second training batch"):
        gate.verify_consumed_micro_batches(
            [microbatch],
            trainer_step=1,
            process_group=process_group,
        )
