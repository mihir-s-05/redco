from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.analysis.stage_d_scientific_branch_group import (
    ArmExecutor,
    BranchGroupArtifact,
    BranchGroupSpec,
    BranchSeedOracle,
    CandidateSampler,
    CandidateSubmission,
    NonRepairableCampaignAbort,
    OutcomeKind,
    PreActionTargetCommitment,
    RepairableInfrastructureAbort,
    SeedCorrespondenceMap,
    ZeroCallInfrastructureFailure,
    behavior_law_digest,
    run_scientific_branch_group,
)
from redco.analysis.stage_d_spawn_provenance import EventSeedScheduler, PolicyEventAddress
from redco.contracts import canonical_json


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _address(address: PolicyEventAddress) -> dict[str, str | int]:
    return {**address.as_payload(), "turn": address.turn}


@dataclass
class TrustedReceiptStore:
    """Test double for the durable receipt producer/verifier owned by D1."""

    allowed: dict[str, set[str]] = field(default_factory=dict)
    chain: str = "0" * 64
    offset: int = 0

    def issue(self, receipt_kind: str, payload: dict[str, Any]) -> bytes:
        receipt = canonical_json(
            {
                "schema_version": 1,
                "receipt_kind": receipt_kind,
                **payload,
            }
        )
        digest = _sha256(receipt)
        self.allowed.setdefault(receipt_kind, set()).add(digest)
        self.chain = _sha256(bytes.fromhex(self.chain) + receipt)
        self.offset += 1
        return receipt

    def __call__(self, receipt: bytes, *, receipt_kind: str) -> Mapping[str, Any]:
        if _sha256(receipt) not in self.allowed.get(receipt_kind, set()):
            raise ValueError("receipt is not anchored in the trusted store")
        value = json.loads(receipt)
        assert isinstance(value, dict)
        return value


def _conformance() -> bytes:
    payload = {
        "schema_version": 1,
        "analysis": "served-stack-categorical-logprob-conformance-v1",
        "passes": True,
        "logprob_semantics": "served_chosen_token_post_transform",
        "categorical_case_count": 3,
        "served_stack_sha256": "a" * 64,
        "tool_call_termination_includes_all_generated_tokens": True,
        "eos_is_included_in_action_tokens_and_logprobs": True,
    }
    payload["signed_payload_sha256"] = _sha256(canonical_json(payload))
    return canonical_json(payload)


def _request(seed: int, *, temperature: float = 0.7) -> dict[str, object]:
    return {
        "model": "model@commit",
        "messages": [{"role": "user", "content": "q"}],
        "tools": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "temperature": temperature,
        "top_p": 1.0,
        "top_k": None,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "logit_bias": {},
        "seed": seed,
        "max_tokens": 2,
        "stop": None,
        "n": 1,
        "best_of": None,
        "use_beam_search": False,
        "logprobs": True,
        "top_logprobs": 0,
        "ignore_eos": False,
        "min_tokens": 0,
        "extra_body": {"cache_salt": f"seed-{seed}"},
    }


def _key(seed: int, *, temperature: float = 0.7) -> ExactActionKey:
    return ExactActionKey.build(
        checkpoint_id="model@commit",
        base_model_manifest=b"base",
        adapter_manifest=b"adapter",
        tokenizer_manifest=b"tokenizer",
        renderer_manifest=b"renderer",
        sampler_conformance_manifest=_conformance(),
        action_selection_policy="direct_single_sample",
        transport_retry_policy="fail_before_action_no_resample",
        request=_request(seed, temperature=temperature),
        prompt_token_ids=(10, 11),
        render_prompt=lambda _: (10, 11),
    )


def _action(seed: int, *, temperature: float = 0.7) -> BehaviorAction:
    return BehaviorAction.build(
        key=_key(seed, temperature=temperature),
        action_token_ids=(20, 2),
        behavior_logprobs=(-0.2, -0.1),
        raw_transport_message={"role": "assistant", "content": "duplicate"},
        finish_reason="stop",
        prompt_tokens=2,
        completion_tokens=2,
        termination_kind="eos",
        eos_token_id=2,
        encode_action=lambda _request, _message: (20, 2),
    )


@dataclass
class Fixture:
    store: TrustedReceiptStore
    spec: BranchGroupSpec
    matched: PolicyEventAddress
    dynamic: PolicyEventAddress


def _fixture(
    *,
    target_ordinal: int = 0,
    branch_count: int = 4,
    continuation_replicates: int = 1,
) -> Fixture:
    store = TrustedReceiptStore()
    recorded = _action(17)
    target = PolicyEventAddress(1, "root/child", 0, 0)
    master_seed = "master"
    prior_chain = store.chain
    receipt = store.issue(
        "pre_action_group_commitment",
        {
            "ledger_id": "durable-ledger",
            "ledger_offset": store.offset,
            "prior_chain_sha256": prior_chain,
            "phase": "pre_action",
            "group_id": f"group-{target_ordinal}",
            "rollout_id": "rollout-1",
            "target_roster": ["target-0", "target-1"],
            "target_ordinal": target_ordinal,
            "target_id": f"target-{target_ordinal}",
            "target_address": _address(target),
            "pre_action_snapshot_sha256": "b" * 64,
            "behavior_law_sha256": behavior_law_digest(recorded.key),
            "recorded_action_seed": 17,
            "branch_count": branch_count,
            "continuation_replicates": continuation_replicates,
            "failure_reward": -1.0,
            "master_seed_sha256": _sha256(master_seed.encode()),
            "commitment_sequence": 4,
            "action_reservation_sequence": 5,
        },
    )
    commitment = PreActionTargetCommitment.from_receipt(receipt, verifier=store)
    matched = PolicyEventAddress(0, "root", 2, 2)
    dynamic = PolicyEventAddress(1, "root/dynamic", 0, 0)
    correspondence_receipt = store.issue(
        "seed_correspondence_map",
        {
            "group_id": commitment.group_id,
            "target_id": commitment.target_id,
            "pre_action_snapshot_sha256": commitment.pre_action_snapshot_sha256,
            "recorded_action_digest": recorded.digest,
            "matched_addresses": [_address(matched)],
        },
    )
    correspondence = SeedCorrespondenceMap.from_receipt(
        correspondence_receipt,
        verifier=store,
        commitment=commitment,
        recorded_action=recorded,
    )
    return Fixture(
        store,
        BranchGroupSpec(commitment, recorded, correspondence, master_seed),
        matched,
        dynamic,
    )


def _qa_receipt(fixture: Fixture, *, report: str = "c" * 64) -> bytes:
    return fixture.store.issue(
        "reconstruction_qa",
        {
            "group_id": fixture.spec.commitment.group_id,
            "target_id": fixture.spec.commitment.target_id,
            "pre_action_snapshot_sha256": (
                fixture.spec.commitment.pre_action_snapshot_sha256
            ),
            "recorded_action_digest": fixture.spec.recorded_action.digest,
            "passed": True,
            "report_sha256": report,
            "actual_cost": {
                "generated_tokens": 3,
                "judge_calls": 0,
                "cpu_seconds": 0.1,
                "gpu_seconds": 0.0,
                "wall_seconds": 0.1,
                "storage_bytes": 0,
            },
        },
    )


def _sampler(fixture: Fixture) -> tuple[list[tuple[int, int]], CandidateSampler]:
    calls: list[tuple[int, int]] = []

    def sample(
        *,
        action_slot: int,
        action_seed: int,
        reference_key: ExactActionKey,
    ) -> CandidateSubmission:
        assert reference_key == fixture.spec.recorded_action.key
        calls.append((action_slot, action_seed))
        action = _action(action_seed)
        receipt = fixture.store.issue(
            "candidate_action_inference",
            {
                "group_id": fixture.spec.commitment.group_id,
                "target_id": fixture.spec.commitment.target_id,
                "action_slot": action_slot,
                "action_seed": action_seed,
                "action_digest": action.digest,
                "behavior_law_sha256": behavior_law_digest(action.key),
                "selection_policy": "direct_single_sample",
                "sample_attempts": 1,
                "rejected_attempts": 0,
                "inference_call_id": f"candidate-{action_slot}",
            },
        )
        return CandidateSubmission(action, receipt)

    return calls, sample


def _executor(
    fixture: Fixture,
    rewards: tuple[float, ...],
) -> tuple[list[str], ArmExecutor]:
    calls: list[str] = []

    def execute(
        *,
        arm_id: str,
        action: BehaviorAction,
        continuation_replicate: int,
        seed_oracle: BranchSeedOracle,
    ) -> bytes:
        calls.append(arm_id)
        slot = int(arm_id.removeprefix("arm-"))
        events = []
        for label, address in (("matched", fixture.matched), ("dynamic", fixture.dynamic)):
            scheduled = seed_oracle.seed_for(address)
            events.append(
                {
                    "call_id": f"{arm_id}-{continuation_replicate}-{label}",
                    "address": _address(address),
                    "seed": scheduled.seed,
                    "coupling_mode": scheduled.coupling_mode.value,
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "disposition": "generated",
                }
            )
        return fixture.store.issue(
            "scientific_arm_execution",
            {
                "group_id": fixture.spec.commitment.group_id,
                "target_id": fixture.spec.commitment.target_id,
                "arm_id": arm_id,
                "action_digest": action.digest,
                "continuation_replicate": continuation_replicate,
                "execution_id": f"execution-{arm_id}-{continuation_replicate}",
                "outcome_kind": OutcomeKind.SUCCESS.value,
                "reward": float(rewards[slot]),
                "calls": events,
                "logical_cost": {
                    "output_tokens": action.completion_tokens + 4,
                    "latency_seconds": 0.1,
                    "dollars": 0.0,
                },
                "actual_non_token_cost": {
                    "judge_calls": 0,
                    "cpu_seconds": 0.1,
                    "gpu_seconds": 0.0,
                    "wall_seconds": 0.1,
                    "storage_bytes": 0,
                },
            },
        )

    return calls, execute


def _run(
    fixture: Fixture | None = None,
    *,
    rewards: tuple[float, ...] = (1.0, 0.0, 0.5, -1.0),
    report: str = "c" * 64,
) -> tuple[BranchGroupArtifact, list[tuple[int, int]], list[str], Fixture]:
    resolved = fixture or _fixture()
    sampler_calls, sampler = _sampler(resolved)
    executor_calls, executor = _executor(resolved, rewards)
    artifact = run_scientific_branch_group(
        resolved.spec,
        verifier=resolved.store,
        sample_candidate=sampler,
        run_reconstruction_qa=lambda _: _qa_receipt(resolved, report=report),
        execute_arm=executor,
    )
    return artifact, sampler_calls, executor_calls, resolved


def test_k4_group_retains_duplicates_and_freshly_executes_every_arm() -> None:
    artifact, sampler_calls, executor_calls, _ = _run()

    assert [slot for slot, _ in sampler_calls] == [1, 2, 3]
    assert [arm.action_source for arm in artifact.arms] == [
        "recorded",
        "sampled",
        "sampled",
        "sampled",
    ]
    assert len({arm.action.action_token_ids for arm in artifact.arms}) == 1
    assert executor_calls == ["arm-0", "arm-1", "arm-2", "arm-3"]
    assert len({outcome.execution_id for arm in artifact.arms for outcome in arm.outcomes}) == 4
    assert [arm.q_value for arm in artifact.arms] == [1.0, 0.0, 0.5, -1.0]
    assert [arm.advantage for arm in artifact.arms] == pytest.approx(
        [7 / 6, -1 / 6, 0.5, -1.5]
    )
    assert all(arm.record_weight == artifact.commitment.outer_weight / 4 for arm in artifact.arms)


def test_target_roster_proves_group_weights_sum_to_one() -> None:
    first, *_ = _run(_fixture(target_ordinal=0))
    second, *_ = _run(_fixture(target_ordinal=1))

    assert sum(arm.record_weight for arm in first.arms) == first.commitment.outer_weight
    assert sum(arm.record_weight for arm in second.arms) == second.commitment.outer_weight
    assert first.commitment.outer_weight + second.commitment.outer_weight == 1


def test_flat_group_is_retained() -> None:
    artifact, *_ = _run(rewards=(0.25, 0.25, 0.25, 0.25))
    assert len(artifact.arms) == 4
    assert [arm.advantage for arm in artifact.arms] == [0.0] * 4


def test_qa_is_outside_scientific_digest_ledger_and_batch_identity() -> None:
    first, *_ = _run(report="c" * 64)
    second, *_ = _run(report="d" * 64)

    assert first.reconstruction_qa.receipt_sha256 != second.reconstruction_qa.receipt_sha256
    assert first.scientific_digest == second.scientific_digest
    assert first.training_batch_identity == second.training_batch_identity
    assert "qa" not in json.dumps(first.ledger.to_payload()).lower()


def test_untrusted_or_posthoc_commitment_is_rejected() -> None:
    fixture = _fixture()
    receipt = json.loads(fixture.spec.commitment.receipt)
    receipt["branch_count"] = 8
    tampered = canonical_json(receipt)

    with pytest.raises(ValueError, match="not anchored"):
        PreActionTargetCommitment.from_receipt(tampered, verifier=fixture.store)

    forged = replace(
        fixture.spec,
        commitment=replace(fixture.spec.commitment, branch_count=8),
    )
    with pytest.raises(ValueError, match="differs from its trusted receipt"):
        run_scientific_branch_group(
            forged,
            verifier=fixture.store,
            sample_candidate=_sampler(fixture)[1],
            run_reconstruction_qa=lambda _: _qa_receipt(fixture),
            execute_arm=_executor(fixture, (1.0,) * 8)[1],
        )


def test_candidate_receipt_prevents_resampling_and_behavior_drift() -> None:
    fixture = _fixture()
    _, executor = _executor(fixture, (1.0, 0.0, 0.5, -1.0))

    def bad_sampler(**kwargs: Any) -> CandidateSubmission:
        action = _action(kwargs["action_seed"], temperature=0.8)
        receipt = fixture.store.issue(
            "candidate_action_inference",
            {
                "group_id": fixture.spec.commitment.group_id,
                "target_id": fixture.spec.commitment.target_id,
                "action_slot": kwargs["action_slot"],
                "action_seed": kwargs["action_seed"],
                "action_digest": action.digest,
                "behavior_law_sha256": behavior_law_digest(action.key),
                "selection_policy": "best_of_two",
                "sample_attempts": 2,
                "rejected_attempts": 1,
                "inference_call_id": "bad",
            },
        )
        return CandidateSubmission(action, receipt)

    with pytest.raises(NonRepairableCampaignAbort, match="candidate slot 1"):
        run_scientific_branch_group(
            fixture.spec,
            verifier=fixture.store,
            sample_candidate=bad_sampler,
            run_reconstruction_qa=lambda _: _qa_receipt(fixture),
            execute_arm=executor,
        )


def test_correspondence_map_alone_authorizes_pairing() -> None:
    artifact, _, _, fixture = _run()
    matched = [arm.outcomes[0].calls[0].scheduled_seed.seed for arm in artifact.arms]
    dynamic = [arm.outcomes[0].calls[1].scheduled_seed.seed for arm in artifact.arms]

    assert len(set(matched)) == 1
    assert len(set(dynamic)) == 4
    assert all(
        arm.outcomes[0].calls[0].scheduled_seed.coupling_mode.value == "paired"
        for arm in artifact.arms
    )
    assert fixture.matched != fixture.dynamic


def test_executor_cannot_self_declare_dynamic_event_as_paired() -> None:
    fixture = _fixture()
    _, sampler = _sampler(fixture)

    def executor(
        *,
        arm_id: str,
        action: BehaviorAction,
        continuation_replicate: int,
        seed_oracle: BranchSeedOracle,
    ) -> bytes:
        scheduled = seed_oracle.seed_for(fixture.matched)
        return fixture.store.issue(
            "scientific_arm_execution",
            {
                "group_id": fixture.spec.commitment.group_id,
                "target_id": fixture.spec.commitment.target_id,
                "arm_id": arm_id,
                "action_digest": action.digest,
                "continuation_replicate": continuation_replicate,
                "execution_id": f"execution-{arm_id}",
                "outcome_kind": "success",
                "reward": 1.0,
                "calls": [
                    {
                        "call_id": f"call-{arm_id}",
                        "address": _address(fixture.dynamic),
                        "seed": scheduled.seed,
                        "coupling_mode": "paired",
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "disposition": "generated",
                    }
                ],
                "logical_cost": {
                    "output_tokens": action.completion_tokens + 1,
                    "latency_seconds": 0.0,
                    "dollars": 0.0,
                },
                "actual_non_token_cost": {
                    "judge_calls": 0,
                    "cpu_seconds": 0.0,
                    "gpu_seconds": 0.0,
                    "wall_seconds": 0.0,
                    "storage_bytes": 0,
                },
            },
        )

    with pytest.raises(NonRepairableCampaignAbort, match="invalid execution receipt"):
        run_scientific_branch_group(
            fixture.spec,
            verifier=fixture.store,
            sample_candidate=sampler,
            run_reconstruction_qa=lambda _: _qa_receipt(fixture),
            execute_arm=executor,
        )


def test_raw_executor_exception_is_nonrepairable() -> None:
    fixture = _fixture()
    _, sampler = _sampler(fixture)

    with pytest.raises(NonRepairableCampaignAbort, match="failure denominator"):
        run_scientific_branch_group(
            fixture.spec,
            verifier=fixture.store,
            sample_candidate=sampler,
            run_reconstruction_qa=lambda _: _qa_receipt(fixture),
            execute_arm=lambda **_: (_ for _ in ()).throw(TimeoutError("unknown phase")),
        )


def test_trusted_zero_call_candidate_failure_is_separately_repairable() -> None:
    fixture = _fixture()
    _, executor = _executor(fixture, (1.0, 0.0, 0.5, -1.0))
    receipt = fixture.store.issue(
        "zero_call_infrastructure_failure",
        {
            "ledger_id": fixture.spec.commitment.ledger_id,
            "ledger_offset": fixture.spec.commitment.ledger_offset + 1,
            "prior_chain_sha256": fixture.store.chain,
            "group_id": fixture.spec.commitment.group_id,
            "target_id": fixture.spec.commitment.target_id,
            "action_slot": 1,
            "action_seed": EventSeedScheduler(
                "master", "rollout-1", "target-0", 1
            ).action_seed(action_slot=1),
            "attempt_id": "attempt-1",
            "scientific_model_calls": 0,
            "reason": "capacity vanished",
        },
    )

    with pytest.raises(RepairableInfrastructureAbort, match="before a scientific"):
        run_scientific_branch_group(
            fixture.spec,
            verifier=fixture.store,
            sample_candidate=lambda **_: (_ for _ in ()).throw(
                ZeroCallInfrastructureFailure(receipt)
            ),
            run_reconstruction_qa=lambda _: _qa_receipt(fixture),
            execute_arm=executor,
        )


def test_materialized_failure_receipt_remains_in_k() -> None:
    fixture = _fixture()
    _, sampler = _sampler(fixture)
    _, normal = _executor(fixture, (1.0, 0.0, 0.5, -1.0))

    def executor(**kwargs: Any) -> bytes:
        if kwargs["arm_id"] != "arm-2":
            return normal(**kwargs)
        return fixture.store.issue(
            "scientific_arm_execution",
            {
                "group_id": fixture.spec.commitment.group_id,
                "target_id": fixture.spec.commitment.target_id,
                "arm_id": "arm-2",
                "action_digest": kwargs["action"].digest,
                "continuation_replicate": 1,
                "execution_id": "execution-arm-2",
                "outcome_kind": "timeout",
                "reward": -1.0,
                "calls": [],
                "logical_cost": {
                    "output_tokens": kwargs["action"].completion_tokens,
                    "latency_seconds": 0.0,
                    "dollars": 0.0,
                },
                "actual_non_token_cost": {
                    "judge_calls": 0,
                    "cpu_seconds": 0.0,
                    "gpu_seconds": 0.0,
                    "wall_seconds": 0.0,
                    "storage_bytes": 0,
                },
            },
        )

    artifact = run_scientific_branch_group(
        fixture.spec,
        verifier=fixture.store,
        sample_candidate=sampler,
        run_reconstruction_qa=lambda _: _qa_receipt(fixture),
        execute_arm=executor,
    )
    assert len(artifact.arms) == 4
    assert artifact.arms[2].outcomes[0].kind is OutcomeKind.TIMEOUT
    assert artifact.arms[2].q_value == -1.0


def test_ledgers_are_recomputed_from_trusted_call_receipts() -> None:
    artifact, *_ = _run()
    assert artifact.ledger.actual_action_generation_calls == 3
    assert artifact.ledger.logical_action_generation_calls == 4
    assert artifact.ledger.actual_downstream_policy_calls == 8
    assert artifact.ledger.logical_downstream_policy_calls == 8
    assert artifact.ledger.actual_generated_tokens == 22
    assert artifact.ledger.logical_output_tokens == 24


def test_logical_ledger_bills_each_replicate_as_a_fresh_workflow() -> None:
    fixture = _fixture(continuation_replicates=2)
    artifact, *_ = _run(fixture)

    assert artifact.ledger.actual_action_generation_calls == 3
    assert artifact.ledger.logical_action_generation_calls == 8
    assert artifact.ledger.actual_downstream_policy_calls == 16
    assert artifact.ledger.logical_downstream_policy_calls == 16
    assert artifact.ledger.actual_generated_tokens == 38
    assert artifact.ledger.logical_output_tokens == 48
    assert all(
        outcome.logical_cost.output_tokens
        == arm.action.completion_tokens
        + sum(call.completion_tokens for call in outcome.calls)
        for arm in artifact.arms
        for outcome in arm.outcomes
    )


def test_full_reload_recomputation_rejects_derived_tampering() -> None:
    artifact, _, _, fixture = _run()
    encoded = artifact.to_bytes()
    loaded = BranchGroupArtifact.verify_bytes(
        encoded,
        verifier=fixture.store,
        encode_action=lambda _request, _message: (20, 2),
        render_prompt=lambda _: (10, 11),
        master_seed="master",
    )
    assert loaded == artifact

    envelope = json.loads(encoded)
    payload = envelope["artifact"]
    payload["arms"][0]["advantage"] = 999.0
    scientific = {
        key: payload[key]
        for key in (
            "schema_version",
            "domain",
            "commitment",
            "correspondence",
            "recorded_action",
            "arms",
            "ledger",
            "inferential_arm_count",
        )
    }
    payload["scientific_digest"] = _sha256(canonical_json(scientific))
    payload["training_batch_identity"] = _sha256(
        canonical_json(
            {
                "domain": "redco-stage-d-training-batch-identity-v1",
                "scientific_digest": payload["scientific_digest"],
                "action_digests": [arm.action.digest for arm in artifact.arms],
            }
        )
    )
    envelope["digest"] = _sha256(canonical_json(payload))
    with pytest.raises(ValueError, match="derived fields"):
        BranchGroupArtifact.verify_bytes(
            canonical_json(envelope),
            verifier=fixture.store,
            encode_action=lambda _request, _message: (20, 2),
            render_prompt=lambda _: (10, 11),
            master_seed="master",
        )


def test_reload_rejects_a_trusted_but_failed_qa_receipt() -> None:
    artifact, _, _, fixture = _run()
    failed = fixture.store.issue(
        "reconstruction_qa",
        {
            "group_id": fixture.spec.commitment.group_id,
            "target_id": fixture.spec.commitment.target_id,
            "pre_action_snapshot_sha256": fixture.spec.commitment.pre_action_snapshot_sha256,
            "recorded_action_digest": fixture.spec.recorded_action.digest,
            "passed": False,
            "report_sha256": "f" * 64,
            "actual_cost": {
                "generated_tokens": 0,
                "judge_calls": 0,
                "cpu_seconds": 0.0,
                "gpu_seconds": 0.0,
                "wall_seconds": 0.0,
                "storage_bytes": 0,
            },
        },
    )
    envelope = json.loads(artifact.to_bytes())
    envelope["artifact"]["reconstruction_qa"] = {
        "receipt": json.loads(failed),
        "receipt_sha256": _sha256(failed),
    }
    envelope["digest"] = _sha256(canonical_json(envelope["artifact"]))

    with pytest.raises(ValueError, match="QA must pass"):
        BranchGroupArtifact.verify_bytes(
            canonical_json(envelope),
            verifier=fixture.store,
            encode_action=lambda _request, _message: (20, 2),
            render_prompt=lambda _: (10, 11),
            master_seed="master",
        )


def test_nonfinite_qa_cost_is_rejected() -> None:
    fixture = _fixture()
    with pytest.raises(ValueError, match="not JSON compliant"):
        fixture.store.issue(
            "reconstruction_qa",
            {
                "group_id": fixture.spec.commitment.group_id,
                "target_id": fixture.spec.commitment.target_id,
                "pre_action_snapshot_sha256": (
                    fixture.spec.commitment.pre_action_snapshot_sha256
                ),
                "recorded_action_digest": fixture.spec.recorded_action.digest,
                "passed": True,
                "report_sha256": "c" * 64,
                "actual_cost": {
                    "generated_tokens": 0,
                    "judge_calls": 0,
                    "cpu_seconds": float("nan"),
                    "gpu_seconds": 0.0,
                    "wall_seconds": 0.0,
                    "storage_bytes": 0,
                },
            },
        )


def test_training_mask_and_single_use_claim_are_explicit() -> None:
    artifact, *_ = _run()
    payload = artifact.to_payload()
    assert payload["single_use_enforced"] is False
    for arm in payload["arms"]:
        assert arm["training_intent"] == {
            "scope": "target_action_tokens_only",
            "prompt_tokens_weight": 0,
            "continuation_tokens_weight": 0,
            "action_token_count": 2,
        }


def test_c1_cannot_reach_legacy_replay_or_live_interfaces() -> None:
    import redco.analysis.stage_d_scientific_branch_group as module

    source = inspect.getsource(module)
    assert "run_empirical_replay" not in source
    assert "rlm_episode_replay" not in source
    assert "urllib" not in source
    assert "subprocess" not in source
