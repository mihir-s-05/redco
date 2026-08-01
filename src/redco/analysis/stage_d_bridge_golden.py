"""Deterministic synthetic C1 artifacts for trainer-bridge integration checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.analysis.stage_d_scientific_branch_group import (
    BranchGroupArtifact,
    BranchGroupSpec,
    BranchSeedOracle,
    CandidateSubmission,
    OutcomeKind,
    PreActionTargetCommitment,
    SeedCorrespondenceMap,
    behavior_law_digest,
    run_scientific_branch_group,
)
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress
from redco.contracts import canonical_json

_MASTER_SEED = "stage-d-e2-synthetic-golden"
_ACTION_TOKENS = ((20, 2), (21, 2), (22, 2), (23, 2))
_ACTION_TEXT = {tokens: f"synthetic-action-{index}" for index, tokens in enumerate(_ACTION_TOKENS)}
_TEXT_ACTION = {text: tokens for tokens, text in _ACTION_TEXT.items()}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _address(value: PolicyEventAddress) -> dict[str, str | int]:
    return {**value.as_payload(), "turn": value.turn}


@dataclass(slots=True)
class SyntheticReceiptStore:
    """Explicitly synthetic receipt anchor; never evidence of a live producer path."""

    allowed: dict[str, set[str]] = field(default_factory=dict)
    chain: str = "0" * 64
    offset: int = 0

    def issue(self, receipt_kind: str, payload: dict[str, Any]) -> bytes:
        receipt = canonical_json(
            {"schema_version": 1, "receipt_kind": receipt_kind, **payload}
        )
        self.allowed.setdefault(receipt_kind, set()).add(_sha256(receipt))
        self.chain = _sha256(bytes.fromhex(self.chain) + receipt)
        self.offset += 1
        return receipt

    def __call__(self, receipt: bytes, *, receipt_kind: str) -> Mapping[str, Any]:
        if _sha256(receipt) not in self.allowed.get(receipt_kind, set()):
            raise ValueError("receipt is not anchored in the synthetic golden store")
        value = json.loads(receipt)
        if not isinstance(value, dict):
            raise ValueError("synthetic receipt must be an object")
        return value


@dataclass(frozen=True, slots=True)
class SyntheticGolden:
    artifact_bytes: tuple[bytes, ...]
    store: SyntheticReceiptStore
    policy_sha256: str


@dataclass(frozen=True, slots=True)
class _Fixture:
    store: SyntheticReceiptStore
    spec: BranchGroupSpec
    matched: PolicyEventAddress
    dynamic: PolicyEventAddress
    prompt_content: str
    prompt_tokens: tuple[int, ...]


def build_synthetic_golden() -> SyntheticGolden:
    """Build two complete target groups with distinct actions and one flat group."""
    store = SyntheticReceiptStore()
    artifacts = (
        _run_group(
            _fixture(
                store,
                target_ordinal=0,
                prompt_content="golden target zero",
                prompt_tokens=(10, 11),
            ),
            rewards=(1.0, 0.0, 0.5, -1.0),
        ),
        _run_group(
            _fixture(
                store,
                target_ordinal=1,
                prompt_content="golden target one",
                prompt_tokens=(12, 13),
            ),
            rewards=(0.5, 0.5, 0.5, 0.5),
        ),
    )
    policy_hashes = {artifact.recorded_action.key.checkpoint_id for artifact in artifacts}
    if policy_hashes != {"synthetic-model@fixture"}:
        raise RuntimeError("synthetic golden policy identity drifted")
    from redco.analysis.stage_d_training_bridge import policy_identity_sha256

    return SyntheticGolden(
        tuple(artifact.to_bytes() for artifact in artifacts),
        store,
        policy_identity_sha256(artifacts[0].recorded_action.key),
    )


def encode_synthetic_action(
    _request: Mapping[str, Any],
    message: Mapping[str, Any],
) -> tuple[int, ...]:
    content = message.get("content")
    if not isinstance(content, str) or content not in _TEXT_ACTION:
        raise ValueError("unknown synthetic golden action")
    return _TEXT_ACTION[content]


def render_synthetic_prompt(request: Mapping[str, Any]) -> tuple[int, ...]:
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("synthetic golden request must have one message")
    content = messages[0].get("content")
    if content == "golden target zero":
        return (10, 11)
    if content == "golden target one":
        return (12, 13)
    raise ValueError("unknown synthetic golden prompt")


def shared_token_credit(artifacts: tuple[BranchGroupArtifact, ...]) -> dict[int, float]:
    """Analytic coefficients for a shared token-indexed parameterization."""
    credit: dict[int, float] = {}
    for artifact in artifacts:
        for arm in artifact.arms:
            for token in arm.action.action_token_ids:
                credit[token] = credit.get(token, 0.0) + float(
                    arm.advantage * arm.record_weight
                )
    return credit


def _fixture(
    store: SyntheticReceiptStore,
    *,
    target_ordinal: int,
    prompt_content: str,
    prompt_tokens: tuple[int, ...],
) -> _Fixture:
    recorded = _action(17, slot=0, prompt_content=prompt_content, prompt_tokens=prompt_tokens)
    target = PolicyEventAddress(1, "root/child", 0, 0)
    prior_chain = store.chain
    commitment_receipt = store.issue(
        "pre_action_group_commitment",
        {
            "ledger_id": "synthetic-golden-ledger",
            "ledger_offset": store.offset,
            "prior_chain_sha256": prior_chain,
            "phase": "pre_action",
            "group_id": f"golden-group-{target_ordinal}",
            "rollout_id": "golden-rollout-1",
            "target_roster": ["golden-target-0", "golden-target-1"],
            "target_ordinal": target_ordinal,
            "target_id": f"golden-target-{target_ordinal}",
            "target_address": _address(target),
            "pre_action_snapshot_sha256": "b" * 64,
            "behavior_law_sha256": behavior_law_digest(recorded.key),
            "recorded_action_seed": 17,
            "branch_count": 4,
            "continuation_replicates": 1,
            "failure_reward": -1.0,
            "master_seed_sha256": _sha256(_MASTER_SEED.encode()),
            "commitment_sequence": 4,
            "action_reservation_sequence": 5,
        },
    )
    commitment = PreActionTargetCommitment.from_receipt(
        commitment_receipt,
        verifier=store,
    )
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
    return _Fixture(
        store,
        BranchGroupSpec(commitment, recorded, correspondence, _MASTER_SEED),
        matched,
        dynamic,
        prompt_content,
        prompt_tokens,
    )


def _run_group(fixture: _Fixture, *, rewards: tuple[float, ...]) -> BranchGroupArtifact:
    def sample(
        *,
        action_slot: int,
        action_seed: int,
        reference_key: ExactActionKey,
    ) -> CandidateSubmission:
        if reference_key != fixture.spec.recorded_action.key:
            raise ValueError("synthetic sampler received a different behavior law")
        action = _action(
            action_seed,
            slot=action_slot,
            prompt_content=fixture.prompt_content,
            prompt_tokens=fixture.prompt_tokens,
        )
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
                "inference_call_id": f"synthetic-candidate-{action_slot}",
            },
        )
        return CandidateSubmission(action, receipt)

    def execute(
        *,
        arm_id: str,
        action: BehaviorAction,
        continuation_replicate: int,
        seed_oracle: BranchSeedOracle,
    ) -> bytes:
        slot = int(arm_id.removeprefix("arm-"))
        calls = []
        for label, address in (("matched", fixture.matched), ("dynamic", fixture.dynamic)):
            scheduled = seed_oracle.seed_for(address)
            calls.append(
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
                "execution_id": f"synthetic-execution-{arm_id}-{continuation_replicate}",
                "outcome_kind": OutcomeKind.SUCCESS.value,
                "reward": rewards[slot],
                "calls": calls,
                "logical_cost": {"output_tokens": 6, "latency_seconds": 0.1, "dollars": 0.0},
                "actual_non_token_cost": {
                    "judge_calls": 0,
                    "cpu_seconds": 0.1,
                    "gpu_seconds": 0.0,
                    "wall_seconds": 0.1,
                    "storage_bytes": 0,
                },
            },
        )

    def qa(_: object) -> bytes:
        return fixture.store.issue(
            "reconstruction_qa",
            {
                "group_id": fixture.spec.commitment.group_id,
                "target_id": fixture.spec.commitment.target_id,
                "pre_action_snapshot_sha256": fixture.spec.commitment.pre_action_snapshot_sha256,
                "recorded_action_digest": fixture.spec.recorded_action.digest,
                "passed": True,
                "report_sha256": "c" * 64,
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

    return run_scientific_branch_group(
        fixture.spec,
        verifier=fixture.store,
        sample_candidate=sample,
        run_reconstruction_qa=qa,
        execute_arm=execute,
    )


def _action(
    seed: int,
    *,
    slot: int,
    prompt_content: str,
    prompt_tokens: tuple[int, ...],
) -> BehaviorAction:
    tokens = _ACTION_TOKENS[slot]
    return BehaviorAction.build(
        key=_key(seed, prompt_content=prompt_content, prompt_tokens=prompt_tokens),
        action_token_ids=tokens,
        behavior_logprobs=(-0.2 - 0.05 * slot, -0.1),
        raw_transport_message={"role": "assistant", "content": _ACTION_TEXT[tokens]},
        finish_reason="stop",
        prompt_tokens=len(prompt_tokens),
        completion_tokens=len(tokens),
        termination_kind="eos",
        eos_token_id=2,
        encode_action=encode_synthetic_action,
    )


def _key(seed: int, *, prompt_content: str, prompt_tokens: tuple[int, ...]) -> ExactActionKey:
    request = {
        "model": "synthetic-model@fixture",
        "messages": [{"role": "user", "content": prompt_content}],
        "tools": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "temperature": 0.7,
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
        "extra_body": {"cache_salt": f"synthetic-seed-{seed}"},
    }
    conformance = {
        "schema_version": 1,
        "analysis": "served-stack-categorical-logprob-conformance-v1",
        "passes": True,
        "logprob_semantics": "served_chosen_token_post_transform",
        "categorical_case_count": 3,
        "served_stack_sha256": "a" * 64,
        "tool_call_termination_includes_all_generated_tokens": True,
        "eos_is_included_in_action_tokens_and_logprobs": True,
    }
    conformance["signed_payload_sha256"] = _sha256(canonical_json(conformance))
    return ExactActionKey.build(
        checkpoint_id="synthetic-model@fixture",
        base_model_manifest=b"synthetic-base",
        adapter_manifest=b"synthetic-adapter",
        tokenizer_manifest=b"synthetic-tokenizer",
        renderer_manifest=b"synthetic-renderer",
        sampler_conformance_manifest=canonical_json(conformance),
        action_selection_policy="direct_single_sample",
        transport_retry_policy="fail_before_action_no_resample",
        request=request,
        prompt_token_ids=prompt_tokens,
        render_prompt=render_synthetic_prompt,
    )
