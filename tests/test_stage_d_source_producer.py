from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_stage_d_scientific_branch_group import _action, _key

import redco.analysis.stage_d_campaign_store as campaign_store_module
from redco.analysis.stage_d_branch_artifacts import (
    StageDBranchArtifactStore,
    StageDBranchTargetRoster,
)
from redco.analysis.stage_d_campaign_controller import (
    compile_authorize_seal_campaign,
)
from redco.analysis.stage_d_campaign_store import (
    StageDCampaignStore,
    verify_campaign_bundle,
)
from redco.analysis.stage_d_collection import (
    StageDCollectionPlan,
    verify_collection_outcomes,
)
from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.analysis.stage_d_objective_binding import (
    ObjectiveAuthorization,
    ObjectiveBinding,
    fixture_objective_binding,
)
from redco.analysis.stage_d_receipt_ledger import (
    BatchAlreadyClaimed,
    GenesisBinding,
    SealedReceiptVerifier,
    StageDReceiptLedger,
    inspect_ledger,
)
from redco.analysis.stage_d_scientific_branch_group import (
    BranchGroupArtifact,
    BranchGroupSpec,
    BranchSeedOracle,
    CandidateSubmission,
    OutcomeKind,
    PreActionTargetCommitment,
    SeedCorrespondenceMap,
    run_scientific_branch_group,
)
from redco.analysis.stage_d_source_artifacts import (
    SourceArtifactError,
    StageDSourceArtifactStore,
)
from redco.analysis.stage_d_source_producer import (
    StageDSourceRolloutProducer,
    structural_child_target_id,
    verify_source_trace_semantics,
)
from redco.analysis.stage_d_spawn_provenance import (
    PolicyEventAddress,
    SpawnScope,
    derive_child_lineage,
)
from redco.analysis.stage_d_three_arm_bridge import (
    ArmName,
    ArmTrainerRecord,
    SealedArmBatch,
    SourceRollout,
    _batch_identity,
)
from redco.analysis.stage_d_three_arm_prime import (
    _verify_stage_d_batch_authorization,
)
from redco.analysis.stage_d_training_bridge import policy_identity_sha256
from redco.contracts import ActualEvaluationCost, canonical_json

MASTER_SEED = "stage-d-source-producer-test"


class _CollectionTaskData:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return dict(self._payload)


def _collection_evidence(sources):
    data = [
        {
            "scientific_group_id": source.group_id,
            "example_id": "campaign-example",
            "rollout_slot": index,
        }
        for index, source in enumerate(sources)
    ]
    plan = StageDCollectionPlan.build(data, master_seed=MASTER_SEED)
    episodes = [
        SimpleNamespace(
            ok=True,
            traces=[
                SimpleNamespace(
                    id=source.rollout_id,
                    task=SimpleNamespace(data=_CollectionTaskData(payload)),
                )
            ],
        )
        for source, payload in zip(sources, data, strict=True)
    ]
    return plan, verify_collection_outcomes(plan, episodes, sources)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _binding() -> GenesisBinding:
    return GenesisBinding(
        preregistration_sha256="1" * 64,
        source_sha256="2" * 64,
        runtime_sha256="3" * 64,
        config_sha256="4" * 64,
        master_seed_sha256=_sha256(MASTER_SEED.encode()),
    )


def _child_address() -> PolicyEventAddress:
    lineage = derive_child_lineage(
        SpawnScope(1, "root", 0, 0, 0),
        spawn_ordinal=0,
    )
    return PolicyEventAddress(1, lineage, 0, 0)


def _target_id(rollout_id: str = "rollout-live") -> str:
    return structural_child_target_id(
        PolicyEventAddress(0, "root", 0, 0),
        rollout_id=rollout_id,
        parent_tool_call_slot=0,
        spawn_ordinal=0,
    )


def test_structural_child_target_is_stable_within_and_unique_across_rollouts() -> None:
    parent = PolicyEventAddress(0, "root", 0, 0)
    first = structural_child_target_id(
        parent,
        rollout_id="rollout-a",
        parent_tool_call_slot=0,
        spawn_ordinal=0,
    )
    assert first == structural_child_target_id(
        parent,
        rollout_id="rollout-a",
        parent_tool_call_slot=0,
        spawn_ordinal=0,
    )
    assert first != structural_child_target_id(
        parent,
        rollout_id="rollout-b",
        parent_tool_call_slot=0,
        spawn_ordinal=0,
    )


def _call(
    node: int,
    *,
    seed: int,
    rlm: dict[str, object],
    finish_reason: str = "stop",
) -> dict[str, object]:
    return {
        "endpoint": "/chat/completions",
        "error": None,
        "finish_reason": finish_reason,
        "model": "model@commit",
        "node": node,
        "rlm": rlm,
        "sampling": {
            "temperature": 0.7,
            "top_p": 1.0,
            "reasoning_effort": None,
            "max_tokens": 2,
            "parallel_tool_calls": False,
            "seed": seed,
        },
        "time": {"start": 1.0, "end": 2.0},
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "cached_input_tokens": None,
            "reasoning_tokens": None,
            "cost": None,
        },
    }


def _node(
    message: dict[str, object],
    token_ids: list[int],
    mask: list[bool],
    logprobs: list[float],
    *,
    parent: int | None = None,
    sampled: bool = False,
) -> dict[str, object]:
    value: dict[str, object] = {
        "message": message,
        "sampled": sampled,
        "timestamp": 1.0,
        "token_ids": token_ids,
        "mask": mask,
        "is_content": [],
        "logprobs": logprobs,
    }
    if parent is not None:
        value["parent"] = parent
    return value


def _episode(
    *,
    trace_id: str = "rollout-live",
    root_seed: int = 71,
    child_seed: int = 72,
    reward: float = 0.75,
) -> bytes:
    trace = {
        "id": trace_id,
        "task": {
            "type": "EvidenceSelectionTask",
            "data": {"policy_checkpoint_id": "model@commit"},
        },
        "runtime": None,
        "version": 1,
        "verifiers": {"version": "pinned"},
        "run": {"type": "train", "id": "run-1"},
        "agent": {
            "model": "model@commit",
            "sampling": {"temperature": 0.7},
            "name": "agent",
            "trainable": True,
        },
        "nodes": [
            _node({"role": "user", "content": "q"}, [10, 11], [False, False], []),
            _node(
                {"role": "assistant", "content": "duplicate"},
                [20, 2],
                [True, True],
                [-0.2, -0.1],
                parent=0,
                sampled=True,
            ),
            _node({"role": "user", "content": "q"}, [10, 11], [False, False], []),
            _node(
                {"role": "assistant", "content": "duplicate"},
                [20, 2],
                [True, True],
                [-0.2, -0.1],
                parent=2,
                sampled=True,
            ),
        ],
        "tools": [],
        "calls": [
            _call(
                1,
                seed=root_seed,
                rlm={
                    "provenance_version": 2,
                    "depth": 0,
                    "session_id": "root-session",
                    "turn": 0,
                    "call_kind": "policy",
                    "lineage": "root",
                    "session_call_ordinal": 0,
                    "completed_episode_spawn_ordinals": [],
                },
            ),
            _call(
                3,
                seed=child_seed,
                rlm={
                    "provenance_version": 2,
                    "depth": 1,
                    "session_id": "child-session",
                    "turn": 0,
                    "call_kind": "policy",
                    "lineage": _child_address().lineage,
                    "session_call_ordinal": 0,
                    "parent_session_id": "root-session",
                    "parent_turn": 0,
                    "parent_tool_call_id": "call_0",
                    "invocation_id": "midpoint-shard-0",
                    "parent_lineage": "root",
                    "parent_call_ordinal": 0,
                    "parent_tool_call_slot": 0,
                    "spawn_ordinal": 0,
                    "episode_spawn_ordinal": 0,
                    "completed_predecessor_spawn_ordinals": [],
                    "completed_episode_spawn_ordinals": [],
                },
            ),
        ],
        "rewards": {"exact_span_f1": reward},
        "metrics": {},
        "info": {"checkpoint_id": "model@commit"},
        "extra_usage": [],
        "is_completed": True,
        "ok": True,
        "stop_condition": "final_answer",
        "errors": [],
        "timing": {},
    }
    return canonical_json(
        {
            "id": "episode-1",
            "env": "redco-evidence-selection-v2",
            "ok": True,
            "errors": [],
            "traces": [trace],
        }
    )


def _tool_action(seed: int) -> BehaviorAction:
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "ipython", "arguments": "{}"},
            }
        ],
    }
    return _prepared_action(seed, message=message, tool_call=True)


def _prepared_action(
    seed: int,
    *,
    message: dict[str, object] | None = None,
    tool_call: bool = False,
) -> BehaviorAction:
    legacy = _key(seed)
    request = json.loads(legacy.request)
    key = ExactActionKey.build_prepared(
        checkpoint_id="model@commit",
        base_model_manifest=b"base",
        adapter_manifest=b"adapter",
        tokenizer_manifest=b"tokenizer",
        renderer_manifest=b"renderer",
        sampler_conformance_manifest=legacy.sampler_conformance_manifest,
        action_selection_policy="direct_single_sample",
        transport_retry_policy="fail_before_action_no_resample",
        request=request,
        prompt_token_ids=(10, 11),
        prepared_engine_request={
            "model": "model@commit",
            "token_ids": [10, 11],
            "sampling_params": {
                "temperature": request["temperature"],
                "top_p": request["top_p"],
                "seed": seed,
                "max_tokens": 2,
                "logprobs": 1,
                "skip_special_tokens": False,
                "stop_token_ids": [2],
                "cache_salt": request["extra_body"]["cache_salt"],
            },
        },
    )
    raw_message = message or {"role": "assistant", "content": "duplicate"}
    return BehaviorAction.build(
        key=key,
        action_token_ids=(20, 2),
        behavior_logprobs=(-0.2, -0.1),
        raw_transport_message=raw_message,
        finish_reason="tool_calls" if tool_call else "stop",
        prompt_tokens=2,
        completion_tokens=2,
        termination_kind="tool_calls" if tool_call else "eos",
        eos_token_id=None if tool_call else 2,
        encode_action=lambda _request, _message: (20, 2),
    )


def _strict_episode() -> bytes:
    parent = PolicyEventAddress(0, "root", 0, 0)

    def child_rlm(spawn_ordinal: int, episode_ordinal: int) -> dict[str, object]:
        lineage = derive_child_lineage(
            SpawnScope(1, "root", 0, 0, 0),
            spawn_ordinal=spawn_ordinal,
        )
        return {
            "provenance_version": 2,
            "depth": 1,
            "session_id": f"child-{spawn_ordinal}",
            "turn": 0,
            "call_kind": "policy",
            "lineage": lineage,
            "session_call_ordinal": 0,
            "parent_session_id": "root-session",
            "parent_turn": 0,
            "parent_tool_call_id": "call_0",
            "invocation_id": f"misleading-label-{1 - spawn_ordinal}",
            "parent_lineage": "root",
            "parent_call_ordinal": 0,
            "parent_tool_call_slot": 0,
            "spawn_ordinal": spawn_ordinal,
            "episode_spawn_ordinal": episode_ordinal,
            "completed_predecessor_spawn_ordinals": [],
            "completed_episode_spawn_ordinals": [],
        }

    root_message = _tool_action(81).message
    trace = json.loads(_episode(trace_id="rollout-strict", root_seed=81, child_seed=82))["traces"][
        0
    ]
    trace["nodes"] = [
        _node({"role": "user", "content": "q"}, [10, 11], [False, False], []),
        _node(root_message, [20, 2], [True, True], [-0.2, -0.1], parent=0, sampled=True),
        _node({"role": "user", "content": "q"}, [10, 11], [False, False], []),
        _node(
            {"role": "assistant", "content": "duplicate"},
            [20, 2],
            [True, True],
            [-0.2, -0.1],
            parent=2,
            sampled=True,
        ),
        _node({"role": "user", "content": "q"}, [10, 11], [False, False], []),
        _node(
            {"role": "assistant", "content": "duplicate"},
            [20, 2],
            [True, True],
            [-0.2, -0.1],
            parent=4,
            sampled=True,
        ),
        _node({"role": "user", "content": "q"}, [10, 11], [False, False], []),
        _node(
            {"role": "assistant", "content": "duplicate"},
            [20, 2],
            [True, True],
            [-0.2, -0.1],
            parent=6,
            sampled=True,
        ),
    ]
    trace["calls"] = [
        _call(
            1,
            seed=81,
            finish_reason="tool_calls",
            rlm={
                "provenance_version": 2,
                "depth": 0,
                "session_id": "root-session",
                "turn": 0,
                "call_kind": "policy",
                "lineage": "root",
                "session_call_ordinal": 0,
                "completed_episode_spawn_ordinals": [],
            },
        ),
        _call(3, seed=83, rlm=child_rlm(1, 1)),
        _call(5, seed=82, rlm=child_rlm(0, 0)),
        _call(
            7,
            seed=84,
            rlm={
                "provenance_version": 2,
                "depth": 0,
                "session_id": "root-session",
                "turn": 1,
                "call_kind": "policy",
                "lineage": "root",
                "session_call_ordinal": 1,
                "completed_episode_spawn_ordinals": [0, 1],
            },
        ),
    ]
    assert structural_child_target_id(
        parent,
        rollout_id="rollout-strict",
        parent_tool_call_slot=0,
        spawn_ordinal=0,
    ) != structural_child_target_id(
        parent,
        rollout_id="rollout-strict",
        parent_tool_call_slot=0,
        spawn_ordinal=1,
    )
    return canonical_json(
        {
            "id": "episode-strict",
            "env": "redco-evidence-selection-v2",
            "ok": True,
            "errors": [],
            "traces": [trace],
        }
    )


def _one_child_episode() -> bytes:
    episode = json.loads(_strict_episode())
    trace = episode["traces"][0]
    nodes = trace["nodes"]
    calls = trace["calls"]
    kept_nodes = [nodes[index] for index in (0, 1, 4, 5, 6, 7)]
    kept_nodes[3]["parent"] = 2
    kept_nodes[5]["parent"] = 4
    kept_calls = [calls[index] for index in (0, 2, 3)]
    for call, node_index in zip(kept_calls, (1, 3, 5), strict=True):
        call["node"] = node_index
    kept_calls[-1]["rlm"]["completed_episode_spawn_ordinals"] = [0]
    trace["nodes"] = kept_nodes
    trace["calls"] = kept_calls
    return canonical_json(episode)


def _produce(root: Path) -> tuple[SourceRollout, StageDReceiptLedger]:
    writer = StageDReceiptLedger.create(
        root,
        binding=_binding(),
        master_seed=MASTER_SEED,
    )
    producer = StageDSourceRolloutProducer(
        ledger=writer,
        group_id="group-1",
        rollout_id="rollout-live",
        child_target_roster=(_target_id(),),
        allow_test_fixture_roster=True,
        base_model_manifest_sha256=_sha256(b"base"),
    )
    root_action = _action(71)
    producer.intercept_policy_call(
        event_address=PolicyEventAddress(0, "root", 0, 0),
        action_key=root_action.key,
        node_kind="root",
        target_id=None,
        branch_selected=False,
        forward_once=lambda _key: root_action,
    )
    child_action = _action(72)
    producer.intercept_policy_call(
        event_address=_child_address(),
        action_key=child_action.key,
        node_kind="child",
        target_id=_target_id(),
        branch_selected=False,
        forward_once=lambda _key: child_action,
    )
    return producer.finalize_episode(_episode()), writer


def _freeze_target_roster(
    writer: StageDReceiptLedger,
    sources: tuple[SourceRollout, ...],
    *,
    minimum_eligible_sources: int = 1,
) -> StageDBranchTargetRoster:
    roster = StageDBranchTargetRoster.from_sources(
        sources,
        planned_source_count=len(sources),
        minimum_eligible_sources=minimum_eligible_sources,
    )
    writer.record_branch_target_roster(roster.to_bytes())
    return roster


def test_source_finalization_failure_is_durably_terminal_after_restart(
    tmp_path: Path,
) -> None:
    root = tmp_path / "finalization-abort"
    writer = StageDReceiptLedger.create(
        root,
        binding=_binding(),
        master_seed=MASTER_SEED,
    )
    producer = StageDSourceRolloutProducer(
        ledger=writer,
        group_id="group-1",
        rollout_id="rollout-live",
        child_target_roster=(_target_id(),),
        allow_test_fixture_roster=True,
        base_model_manifest_sha256=_sha256(b"base"),
    )
    action = _action(71)
    producer.intercept_policy_call(
        event_address=PolicyEventAddress(0, "root", 0, 0),
        action_key=action.key,
        node_kind="root",
        target_id=None,
        branch_selected=False,
        forward_once=lambda _key: action,
    )

    receipt = producer.abort_finalization(RuntimeError("injected finalization failure"))
    assert receipt is not None
    writer.close()

    scan = inspect_ledger(root)
    assert scan.status == "poisoned"
    assert scan.reason == "ledger records an aborted source rollout finalization"


def _produce_into(
    writer: StageDReceiptLedger,
    *,
    rollout_id: str,
    root_seed: int,
    child_seed: int,
    reward: float,
    selected: bool,
) -> tuple[SourceRollout, BranchGroupSpec | None]:
    producer = StageDSourceRolloutProducer(
        ledger=writer,
        group_id="group-1",
        rollout_id=rollout_id,
        child_target_roster=(_target_id(rollout_id),),
        allow_test_fixture_roster=True,
        base_model_manifest_sha256=_sha256(b"base"),
    )
    root_action = _action(root_seed)
    producer.intercept_policy_call(
        event_address=PolicyEventAddress(0, "root", 0, 0),
        action_key=root_action.key,
        node_kind="root",
        target_id=None,
        branch_selected=False,
        forward_once=lambda _key: root_action,
    )
    child_action = _action(child_seed)
    recorded = None
    if selected:
        snapshot = writer.put_evidence(f"snapshot:{rollout_id}".encode())
        recorded = writer.commit_pre_action_and_reserve(
            group_id="group-1",
            rollout_id=rollout_id,
            target_roster=(_target_id(rollout_id),),
            target_ordinal=0,
            target_id=_target_id(rollout_id),
            target_address=_child_address(),
            pre_action_snapshot_sha256=snapshot,
            recorded_action_key=child_action.key,
            branch_count=4,
            continuation_replicates=1,
            failure_reward=-1.0,
        )
    producer.intercept_policy_call(
        event_address=_child_address(),
        action_key=child_action.key,
        node_kind="child",
        target_id=_target_id(rollout_id),
        branch_selected=selected,
        recorded_action_reservation=recorded,
        forward_once=lambda _key: child_action,
    )
    source = producer.finalize_episode(
        _episode(
            trace_id=rollout_id,
            root_seed=root_seed,
            child_seed=child_seed,
            reward=reward,
        )
    )
    if recorded is None:
        return source, None
    matched = PolicyEventAddress(0, "root", 2, 2)
    correspondence_evidence = writer.put_evidence(f"correspondence:{rollout_id}".encode())
    correspondence_receipt = writer.freeze_correspondence(
        group_id="group-1",
        target_id=_target_id(rollout_id),
        recorded_action=child_action,
        matched_addresses=(matched,),
        evidence_sha256=correspondence_evidence,
    )
    commitment = PreActionTargetCommitment.from_receipt(
        recorded.commitment_receipt,
        verifier=writer,
    )
    correspondence = SeedCorrespondenceMap.from_receipt(
        correspondence_receipt,
        verifier=writer,
        commitment=commitment,
        recorded_action=child_action,
    )
    return source, BranchGroupSpec(
        commitment,
        child_action,
        correspondence,
        MASTER_SEED,
    )


def test_two_concurrent_source_rollouts_share_one_ledger_without_id_collision(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    writer = StageDReceiptLedger.create(
        root,
        binding=_binding(),
        master_seed=MASTER_SEED,
    )

    def produce(arguments: tuple[str, int, int, float]) -> SourceRollout:
        rollout_id, root_seed, child_seed, reward = arguments
        source, artifact = _produce_into(
            writer,
            rollout_id=rollout_id,
            root_seed=root_seed,
            child_seed=child_seed,
            reward=reward,
            selected=False,
        )
        assert artifact is None
        return source

    with ThreadPoolExecutor(max_workers=2) as pool:
        sources = tuple(
            pool.map(
                produce,
                (
                    ("rollout-concurrent-a", 101, 102, 0.25),
                    ("rollout-concurrent-b", 103, 104, 0.75),
                ),
            )
        )

    assert {source.rollout_id for source in sources} == {
        "rollout-concurrent-a",
        "rollout-concurrent-b",
    }
    assert sources[0].child_target_roster != sources[1].child_target_roster
    writer.close()


def _run_live_artifact(
    writer: StageDReceiptLedger,
    spec: BranchGroupSpec,
) -> bytes:
    matched = spec.correspondence.matched_addresses[0]

    def qa(_: BranchGroupSpec) -> bytes:
        report = writer.put_evidence(b"model-free reconstruction QA")
        return writer.record_reconstruction_qa(
            group_id=spec.commitment.group_id,
            target_id=spec.commitment.target_id,
            recorded_action=spec.recorded_action,
            passed=True,
            report_sha256=report,
            actual_cost=ActualEvaluationCost(cpu_seconds=0.01, wall_seconds=0.01),
        )

    def sample(
        *,
        action_slot: int,
        action_seed: int,
        reference_key: ExactActionKey,
    ) -> CandidateSubmission:
        assert reference_key == spec.recorded_action.key
        attempt = writer.begin_candidate_attempt(
            group_id=spec.commitment.group_id,
            target_id=spec.commitment.target_id,
            action_slot=action_slot,
        )
        request = writer.put_evidence(f"candidate-request:{action_slot}".encode())
        writer.mark_candidate_model_call_started(attempt, request_sha256=request)
        action = _action(action_seed)
        response = writer.put_evidence(action.to_bytes())
        receipt = writer.complete_candidate_call(
            attempt,
            action=action,
            response_sha256=response,
        )
        return CandidateSubmission(action, receipt)

    def execute(
        *,
        arm_id: str,
        action: BehaviorAction,
        continuation_replicate: int,
        seed_oracle: BranchSeedOracle,
    ) -> bytes:
        attempt = writer.begin_execution(
            group_id=spec.commitment.group_id,
            target_id=spec.commitment.target_id,
            arm_id=arm_id,
            action=action,
            continuation_replicate=continuation_replicate,
        )
        context = writer.put_evidence(f"context:{arm_id}".encode())
        writer.bind_execution_context(attempt, context_sha256=context)
        writer.mark_execution_dispatched(attempt)
        request = writer.put_evidence(f"execution-request:{arm_id}".encode())
        call = writer.mark_execution_model_call_started(
            attempt,
            address=matched,
            scheduled_seed=seed_oracle.seed_for(matched),
            request_sha256=request,
        )
        response = writer.put_evidence(f"execution-response:{arm_id}".encode())
        writer.complete_execution_model_call(
            attempt,
            call,
            prompt_tokens=3,
            completion_tokens=2,
            response_sha256=response,
        )
        score = writer.put_evidence(f"score:{arm_id}".encode())
        return writer.finish_execution(
            attempt,
            outcome_kind=OutcomeKind.SUCCESS,
            scored_reward=1.0 if arm_id == "arm-0" else 0.0,
            scorer_evidence_sha256=score,
            latency_seconds=0.01,
            dollars=0.0,
            judge_calls=0,
            cpu_seconds=0.01,
            gpu_seconds=0.0,
            wall_seconds=0.01,
            storage_bytes=0,
        )

    artifact = run_scientific_branch_group(
        spec,
        verifier=writer,
        sample_candidate=sample,
        run_reconstruction_qa=qa,
        execute_arm=execute,
    )
    value = artifact.to_bytes()
    digest = writer.put_evidence(value)
    writer.record_branch_group_artifact_completed(
        group_id=spec.commitment.group_id,
        target_id=spec.commitment.target_id,
        artifact_sha256=digest,
        training_batch_identity=artifact.training_batch_identity,
    )
    return value


def test_live_source_is_derived_from_intercepted_calls_and_trace(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    source, writer = _produce(root)
    assert source.reward == 0.75
    assert [sequence.token_ids for sequence in source.stock_sequences] == [
        (10, 11, 20, 2),
        (10, 11, 20, 2),
    ]
    assert source.stock_sequence_decision_ids == (
        (source.decisions[0].decision_id,),
        (source.decisions[1].decision_id,),
    )
    _freeze_target_roster(writer, (source,))
    seal = writer.seal()
    verifier = SealedReceiptVerifier(root, seal)
    restored = SourceRollout.verify_bytes(
        source.to_bytes(),
        verifier=verifier,
        evidence_loader=lambda digest: (root / "evidence" / digest).read_bytes(),
        encode_action=lambda _request, _message: (20, 2),
        render_prompt=lambda _request: (10, 11),
    )
    assert restored == source


@pytest.mark.parametrize("mutation", ["reward", "prompt", "mask", "call-node"])
def test_trace_semantics_reject_scientific_field_substitution(
    tmp_path: Path,
    mutation: str,
) -> None:
    source, writer = _produce(tmp_path / "ledger")
    episode = json.loads(_episode())
    trace = episode["traces"][0]
    if mutation == "reward":
        trace["rewards"]["exact_span_f1"] = 0.5
    elif mutation == "prompt":
        trace["nodes"][0]["token_ids"][0] = 99
    elif mutation == "mask":
        trace["nodes"][1]["mask"] = [False, True]
        trace["nodes"][1]["logprobs"] = [-0.1]
    else:
        trace["calls"][0]["node"] = trace["calls"][1]["node"]
    with pytest.raises(ValueError):
        verify_source_trace_semantics(source, raw_episode=canonical_json(episode))
    writer.close()


def test_strict_source_uses_two_structural_slots_despite_reverse_completion(
    tmp_path: Path,
) -> None:
    ledger_root = tmp_path / "strict-ledger"
    writer = StageDReceiptLedger.create(
        ledger_root,
        binding=_binding(),
        master_seed=MASTER_SEED,
    )
    parent = PolicyEventAddress(0, "root", 0, 0)
    producer = StageDSourceRolloutProducer(
        ledger=writer,
        group_id="group-1",
        rollout_id="rollout-strict",
        child_parent_event=parent,
        child_parent_tool_call_slot=0,
        root_policy_turn_count=2,
        base_model_manifest_sha256=_sha256(b"base"),
    )
    root = _tool_action(81)
    producer.intercept_policy_call(
        event_address=parent,
        action_key=root.key,
        node_kind="root",
        target_id=None,
        branch_selected=False,
        forward_once=lambda _key: root,
    )
    pending = []
    for ordinal, seed in ((0, 82), (1, 83)):
        address = PolicyEventAddress(
            1,
            derive_child_lineage(
                SpawnScope(1, "root", 0, 0, 0),
                spawn_ordinal=ordinal,
            ),
            0,
            0,
        )
        action = _prepared_action(seed)
        ticket = producer.reserve_policy_call(
            event_address=address,
            action_key=action.key,
            node_kind="child",
            target_id=structural_child_target_id(
                parent,
                rollout_id="rollout-strict",
                parent_tool_call_slot=0,
                spawn_ordinal=ordinal,
            ),
            branch_selected=False,
        )
        pending.append((ticket, action))
    producer.complete_policy_call(pending[1][0], action=pending[1][1])
    producer.complete_policy_call(pending[0][0], action=pending[0][1])
    returning_root = _prepared_action(84)
    producer.intercept_policy_call(
        event_address=PolicyEventAddress(0, "root", 1, 1),
        action_key=returning_root.key,
        node_kind="root",
        target_id=None,
        branch_selected=False,
        forward_once=lambda _key: returning_root,
    )

    prepared_sources: list[bytes] = []
    store = StageDSourceArtifactStore(tmp_path / "source-artifacts")

    def prepare(value: bytes) -> None:
        prepared_sources.append(value)
        store.prepare(value)

    source = producer.finalize_episode(
        _strict_episode(),
        prepare_source_rollout=prepare,
    )

    assert source.child_target_roster == tuple(
        structural_child_target_id(
            parent,
            rollout_id="rollout-strict",
            parent_tool_call_slot=0,
            spawn_ordinal=ordinal,
        )
        for ordinal in (0, 1)
    )
    assert [
        decision.target_ordinal for decision in source.decisions if decision.node_kind == "child"
    ] == [0, 1]
    assert len(prepared_sources) == 1
    prepared = json.loads(prepared_sources[0])
    assert prepared["domain"] == "redco-stage-d-prepared-source-rollout-v1"
    assert prepared["source_sha256"] == source.source_sha256
    assert prepared["source"] == source.to_payload()
    (recovered,) = store.recover_completed(ledger_root)
    assert recovered.read_bytes() == source.to_bytes()
    store.assert_no_pending()

    tampered = json.loads(prepared_sources[0])
    tampered["source"]["reward"] = 0.125
    with pytest.raises(ValueError, match="digest mismatch"):
        store.prepare(canonical_json(tampered))

    orphan_ledger_root = tmp_path / "orphan-ledger"
    orphan_writer = StageDReceiptLedger.create(
        orphan_ledger_root,
        binding=_binding(),
        master_seed=MASTER_SEED,
    )
    orphan_store = StageDSourceArtifactStore(tmp_path / "orphan-artifacts")
    orphan_store.prepare(prepared_sources[0])
    with pytest.raises(SourceArtifactError, match="no durable completion"):
        orphan_store.recover_completed(orphan_ledger_root)
    orphan_writer.close()
    writer.close()


def test_completed_one_child_topology_is_durable_ineligible_evidence(
    tmp_path: Path,
) -> None:
    writer = StageDReceiptLedger.create(
        tmp_path / "ledger",
        binding=_binding(),
        master_seed=MASTER_SEED,
    )
    parent = PolicyEventAddress(0, "root", 0, 0)
    producer = StageDSourceRolloutProducer(
        ledger=writer,
        group_id="group-1",
        rollout_id="rollout-strict",
        child_parent_event=parent,
        child_parent_tool_call_slot=0,
        root_policy_turn_count=2,
        base_model_manifest_sha256=_sha256(b"base"),
    )
    root = _tool_action(81)
    producer.intercept_policy_call(
        event_address=parent,
        action_key=root.key,
        node_kind="root",
        target_id=None,
        branch_selected=False,
        forward_once=lambda _key: root,
    )
    child_address = PolicyEventAddress(
        1,
        derive_child_lineage(SpawnScope(1, "root", 0, 0, 0), spawn_ordinal=0),
        0,
        0,
    )
    child_target = structural_child_target_id(
        parent,
        rollout_id="rollout-strict",
        parent_tool_call_slot=0,
        spawn_ordinal=0,
    )
    child = _prepared_action(82)
    pending = producer.reserve_selected_child_policy_call(
        event_address=child_address,
        target_id=child_target,
        action_key=child.key,
        pre_action_snapshot=b"one-child-pre-action-snapshot",
        branch_count=4,
        continuation_replicates=1,
        failure_reward=-1.0,
    )
    producer.complete_policy_call(pending, action=child)
    returning_root = _prepared_action(84)
    producer.intercept_policy_call(
        event_address=PolicyEventAddress(0, "root", 1, 1),
        action_key=returning_root.key,
        node_kind="root",
        target_id=None,
        branch_selected=False,
        forward_once=lambda _key: returning_root,
    )

    source = producer.finalize_episode(_one_child_episode())

    assert source.branch_eligible is False
    assert source.ineligibility_reason == (
        "scientific scaffold has an unexpected policy-call count"
    )
    assert source.child_target_roster == (child_target,)
    child_decision = next(
        decision for decision in source.decisions if decision.node_kind == "child"
    )
    assert child_decision.outer_weight == 1
    assert inspect_ledger(tmp_path / "ledger").status == "active-clean"
    writer.close()


def test_producer_refuses_finalization_while_request_is_in_flight(tmp_path: Path) -> None:
    writer = StageDReceiptLedger.create(
        tmp_path / "ledger",
        binding=_binding(),
        master_seed=MASTER_SEED,
    )
    producer = StageDSourceRolloutProducer(
        ledger=writer,
        group_id="group-1",
        rollout_id="rollout-live",
        child_target_roster=(_target_id(),),
        allow_test_fixture_roster=True,
        base_model_manifest_sha256=_sha256(b"base"),
    )
    action = _action(71)
    producer.reserve_policy_call(
        event_address=PolicyEventAddress(0, "root", 0, 0),
        action_key=action.key,
        node_kind="root",
        target_id=None,
        branch_selected=False,
    )
    with pytest.raises(ValueError, match="pending policy calls"):
        producer.finalize_episode(_episode())
    writer.close()


def test_exact_live_batch_authorization_is_single_use_and_sealed(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    source, writer = _produce(root)
    _freeze_target_roster(writer, (source,))
    sealed_batch_sha256 = writer.put_evidence(b"exact sealed arm batch")
    objective_authorization_sha256 = writer.put_evidence(b"preregistered objective authorization")
    collection_plan_sha256 = writer.put_evidence(b"collection plan")
    collection_receipt_sha256 = writer.put_evidence(b"collection receipt")
    authorization = writer.authorize_stage_d_training_batch(
        arm="stock",
        training_batch_identity="8" * 64,
        sealed_batch_sha256=sealed_batch_sha256,
        objective_sha256="9" * 64,
        objective_authorization_sha256=objective_authorization_sha256,
        collection_plan_sha256=collection_plan_sha256,
        collection_receipt_sha256=collection_receipt_sha256,
        source_sha256s=(source.source_sha256,),
        branch_artifact_sha256s=(),
        consumer_id="stage-d-prime:stock:step:1",
    )
    with pytest.raises(BatchAlreadyClaimed):
        writer.authorize_stage_d_training_batch(
            arm="stock",
            training_batch_identity="8" * 64,
            sealed_batch_sha256=sealed_batch_sha256,
            objective_sha256="9" * 64,
            objective_authorization_sha256=objective_authorization_sha256,
            collection_plan_sha256=collection_plan_sha256,
            collection_receipt_sha256=collection_receipt_sha256,
            source_sha256s=(source.source_sha256,),
            branch_artifact_sha256s=(),
            consumer_id="stage-d-prime:stock:step:1",
        )
    seal = writer.seal()
    assert (
        SealedReceiptVerifier(root, seal)(
            authorization.receipt,
            receipt_kind="stage_d_training_batch_authorization",
        )["sealed_batch_sha256"]
        == sealed_batch_sha256
    )
    assert seal == type(seal).from_bytes(seal.to_bytes())


def test_selected_intercept_is_the_same_recorded_action_call(tmp_path: Path) -> None:
    writer = StageDReceiptLedger.create(
        tmp_path / "ledger",
        binding=_binding(),
        master_seed=MASTER_SEED,
    )
    producer = StageDSourceRolloutProducer(
        ledger=writer,
        group_id="group-1",
        rollout_id="rollout-live",
        child_target_roster=(_target_id(),),
        allow_test_fixture_roster=True,
        base_model_manifest_sha256=_sha256(b"base"),
    )
    action = _action(72)
    snapshot = writer.put_evidence(b"pre-action child snapshot")
    recorded = writer.commit_pre_action_and_reserve(
        group_id="group-1",
        rollout_id="rollout-live",
        target_roster=(_target_id(),),
        target_ordinal=0,
        target_id=_target_id(),
        target_address=_child_address(),
        pre_action_snapshot_sha256=snapshot,
        recorded_action_key=action.key,
        branch_count=4,
        continuation_replicates=1,
        failure_reward=-1.0,
    )
    decision = producer.intercept_policy_call(
        event_address=_child_address(),
        action_key=action.key,
        node_kind="child",
        target_id=_target_id(),
        branch_selected=True,
        recorded_action_reservation=recorded,
        forward_once=lambda _key: action,
    )
    receipt = json.loads(decision.provenance.reservation_receipt)
    assert receipt["recorded_action_reservation_id"] == recorded.reservation_id
    assert receipt["target_commitment_receipt_sha256"] == _sha256(recorded.commitment_receipt)
    writer.close()


def test_trainer_gate_accepts_only_the_exact_ledger_authorized_batch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    source, writer = _produce(root)
    _freeze_target_roster(writer, (source,))
    fixture_payload = fixture_objective_binding("stock").to_payload()
    fixture_payload["evidence_class"] = "live"
    binding = ObjectiveBinding.from_bytes(canonical_json(fixture_payload))
    sequence = source.stock_sequences[0]
    record = ArmTrainerRecord(
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
        tuple(0.25 if selected else 0.0 for selected in sequence.mask),
        None,
        None,
    )
    policy_sha256 = policy_identity_sha256(source.decisions[0].action.key)
    identity = _batch_identity(
        "stock",
        (record,),
        (source.source_sha256,),
        (),
        "live",
        binding,
        policy_sha256,
        1,
        64,
    )
    batch = SealedArmBatch(
        "stock",
        (record,),
        (source.source_sha256,),
        (),
        "live",
        binding,
        policy_sha256,
        1,
        64,
        identity,
    )
    objective_authorization = ObjectiveAuthorization(
        "live",
        (
            ("branch-global", "7" * 64),
            ("local", "8" * 64),
            ("stock", binding.objective_sha256),
        ),
    ).to_bytes()
    objective_authorization_sha256 = writer.put_evidence(objective_authorization)
    collection_plan_sha256 = writer.put_evidence(b"collection plan")
    collection_receipt_sha256 = writer.put_evidence(b"collection receipt")
    batch_sha256 = writer.put_evidence(batch.to_bytes())
    authorization = writer.authorize_stage_d_training_batch(
        arm="stock",
        training_batch_identity=batch.batch_identity,
        sealed_batch_sha256=batch_sha256,
        objective_sha256=binding.objective_sha256,
        objective_authorization_sha256=objective_authorization_sha256,
        collection_plan_sha256=collection_plan_sha256,
        collection_receipt_sha256=collection_receipt_sha256,
        source_sha256s=batch.source_sha256s,
        branch_artifact_sha256s=(),
        consumer_id="stage-d-prime:stock:step:1",
    )
    seal = writer.seal()
    verifier = SealedReceiptVerifier(root, seal)
    assert _verify_stage_d_batch_authorization(
        authorization.receipt,
        verifier=verifier,
        batch=batch,
        sealed_batch_bytes=batch.to_bytes(),
        objective_authorization_sha256=objective_authorization_sha256,
    ) == _sha256(authorization.receipt)
    with pytest.raises(ValueError, match="differs from its single-use"):
        _verify_stage_d_batch_authorization(
            authorization.receipt,
            verifier=verifier,
            batch=batch,
            sealed_batch_bytes=batch.to_bytes() + b"tamper",
            objective_authorization_sha256=objective_authorization_sha256,
        )


def test_campaign_transaction_reconstructs_after_single_terminal_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "ledger"
    writer = StageDReceiptLedger.create(
        root,
        binding=_binding(),
        master_seed=MASTER_SEED,
    )
    selected, spec = _produce_into(
        writer,
        rollout_id="rollout-live-1",
        root_seed=71,
        child_seed=72,
        reward=0.75,
        selected=True,
    )
    control, no_spec = _produce_into(
        writer,
        rollout_id="rollout-live-2",
        root_seed=73,
        child_seed=74,
        reward=0.25,
        selected=False,
    )
    assert spec is not None
    assert no_spec is None
    target_roster = StageDBranchTargetRoster.from_sources(
        (selected, control),
        planned_source_count=2,
        minimum_eligible_sources=1,
    )
    assert target_roster.eligibility_passed is True
    assert len(target_roster.targets) == 1
    branch_store = StageDBranchArtifactStore(tmp_path / "branch-artifacts")
    branch_store.assert_pristine()
    assert branch_store.persist_target_roster(target_roster).read_bytes() == (
        target_roster.to_bytes()
    )
    writer.record_branch_target_roster(target_roster.to_bytes())
    artifact_bytes = _run_live_artifact(writer, spec)
    artifact = BranchGroupArtifact.verify_bytes(
        artifact_bytes,
        verifier=writer,
        encode_action=lambda _request, _message: (20, 2),
        render_prompt=lambda _request: (10, 11),
        master_seed=MASTER_SEED,
    )
    branch_store.prepare(artifact)
    scan = inspect_ledger(root)
    artifact_receipt = next(
        canonical_json(receipt)
        for (kind, _), receipt in scan.receipts.items()
        if kind == "branch_group_artifact_completed"
    )
    branch_store.commit(artifact, artifact_receipt)
    assert len(branch_store.completed_paths()) == 1

    bindings: dict[ArmName, bytes] = {}
    objective_hashes: list[tuple[ArmName, str]] = []
    arms: tuple[ArmName, ...] = ("stock", "branch-global", "local")
    for arm in arms:
        payload = fixture_objective_binding(arm).to_payload()
        payload["evidence_class"] = "live"
        binding = ObjectiveBinding.from_bytes(canonical_json(payload))
        bindings[arm] = canonical_json(binding.to_payload())
        objective_hashes.append((arm, binding.objective_sha256))
    authorization = ObjectiveAuthorization(
        "live",
        tuple(sorted(objective_hashes)),
    ).to_bytes()
    collection_plan, collection_receipt = _collection_evidence((selected, control))
    campaign = compile_authorize_seal_campaign(
        ledger=writer,
        ledger_root=root,
        source_rollout_bytes=(selected.to_bytes(), control.to_bytes()),
        collection_plan=collection_plan,
        collection_receipt_bytes=collection_receipt,
        preregistered_collection_plan_sha256=collection_plan.plan_sha256,
        branch_artifact_bytes=(artifact_bytes,),
        encode_action=lambda _request, _message: (20, 2),
        render_prompt=lambda _request: (10, 11),
        master_seed=MASTER_SEED,
        objective_binding_bytes=bindings,
        objective_authorization_bytes=authorization,
        preregistered_objective_authorization_sha256=_sha256(authorization),
        trainer_step=1,
        seq_len=64,
        allow_test_fixture_collection=True,
    )
    assert tuple(arm for arm, _ in campaign.batch_authorization_receipts) == (
        "branch-global",
        "local",
        "stock",
    )
    assert campaign.compilation.stock.source_sha256s == tuple(
        sorted((selected.source_sha256, control.source_sha256))
    )
    assert campaign.compilation.local.branch_artifact_sha256s == (_sha256(artifact_bytes),)
    assert campaign.ledger_seal == type(campaign.ledger_seal).from_bytes(campaign.ledger_seal_bytes)
    monkeypatch.setattr(
        campaign_store_module,
        "materialize_prime_rollout_bytes",
        lambda batch: canonical_json(
            {"arm": batch.arm, "batch_identity": batch.batch_identity}
        ),
    )
    bundle = StageDCampaignStore(tmp_path / "bundle").persist(
        campaign=campaign,
        ledger_root=root,
        collection_plan=collection_plan,
        collection_receipt_bytes=collection_receipt,
        source_rollout_bytes=(selected.to_bytes(), control.to_bytes()),
        branch_artifact_bytes=(artifact_bytes,),
        objective_binding_bytes=bindings,
        trainer_toml_bytes={arm: f"# {arm}\n".encode() for arm in arms},
        frozen_inputs={"preregistration.json": b"{}"},
    )
    assert verify_campaign_bundle(bundle.root) == bundle
    assert tuple(arm for arm, _ in bundle.prime_rollout_paths) == arms
    with pytest.raises(FileExistsError, match="already exists"):
        StageDCampaignStore(bundle.root)


def test_campaign_transaction_rejects_omitted_completed_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "omitted-source-ledger"
    writer = StageDReceiptLedger.create(
        root,
        binding=_binding(),
        master_seed=MASTER_SEED,
    )
    selected, spec = _produce_into(
        writer,
        rollout_id="rollout-live-1",
        root_seed=71,
        child_seed=72,
        reward=0.75,
        selected=True,
    )
    control, _ = _produce_into(
        writer,
        rollout_id="rollout-live-2",
        root_seed=73,
        child_seed=74,
        reward=0.25,
        selected=False,
    )
    omitted, _ = _produce_into(
        writer,
        rollout_id="rollout-live-3",
        root_seed=75,
        child_seed=76,
        reward=0.5,
        selected=False,
    )
    assert spec is not None
    _freeze_target_roster(writer, (selected, control, omitted))
    artifact_bytes = _run_live_artifact(writer, spec)
    bindings: dict[ArmName, bytes] = {}
    objective_hashes: list[tuple[ArmName, str]] = []
    for arm in ("stock", "branch-global", "local"):
        payload = fixture_objective_binding(arm).to_payload()
        payload["evidence_class"] = "live"
        binding = ObjectiveBinding.from_bytes(canonical_json(payload))
        bindings[arm] = canonical_json(binding.to_payload())
        objective_hashes.append((arm, binding.objective_sha256))
    authorization = ObjectiveAuthorization(
        "live",
        tuple(sorted(objective_hashes)),
    ).to_bytes()
    collection_plan, collection_receipt = _collection_evidence((selected, control))

    with pytest.raises(ValueError, match="every completed ledger source"):
        compile_authorize_seal_campaign(
            ledger=writer,
            ledger_root=root,
            source_rollout_bytes=(selected.to_bytes(), control.to_bytes()),
            collection_plan=collection_plan,
            collection_receipt_bytes=collection_receipt,
            preregistered_collection_plan_sha256=collection_plan.plan_sha256,
            branch_artifact_bytes=(artifact_bytes,),
            encode_action=lambda _request, _message: (20, 2),
            render_prompt=lambda _request: (10, 11),
            master_seed=MASTER_SEED,
            objective_binding_bytes=bindings,
            objective_authorization_bytes=authorization,
            preregistered_objective_authorization_sha256=_sha256(authorization),
            trainer_step=1,
            seq_len=64,
            allow_test_fixture_collection=True,
        )
    writer.close()
