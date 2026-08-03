from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_stage_d_evaluation_ledger import _finish_task as _finish_evaluation_task
from test_stage_d_evaluation_ledger import _start_arm as _start_evaluation_arm
from test_stage_d_scientific_branch_group import _action, _key
from test_stage_d_trainer_supervisor import _checkpoint_evidence, _mark_process_started

import redco.analysis.stage_d_campaign_store as campaign_store_module
from redco.analysis.stage_d_branch_artifacts import (
    StageDBranchArtifactStore,
    StageDBranchTargetRoster,
)
from redco.analysis.stage_d_campaign_controller import (
    compile_authorize_seal_campaign,
    recover_sealed_campaign,
)
from redco.analysis.stage_d_campaign_store import (
    FrozenProtocolInputs,
    StageDCampaignStore,
    verify_campaign_bundle,
)
from redco.analysis.stage_d_collection import (
    StageDCollectionPlan,
    verify_collection_outcomes,
)
from redco.analysis.stage_d_evaluation_actuation import ActuatedProcessReceipt
from redco.analysis.stage_d_evaluation_barrier import (
    StageDEvaluationPlan,
    StageDEvaluationTask,
    commit_sealed_heldout_evaluation,
)
from redco.analysis.stage_d_evaluation_contracts import (
    EvaluationProgramBinding,
    EvaluationRuntimeEntrypoint,
    EvaluationScheduleUnit,
    EvaluationSupervisorLimits,
    StageDEvaluationExecutionManifest,
)
from redco.analysis.stage_d_evaluation_ledger import StageDEvaluationLedger
from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.analysis.stage_d_handoff_coordinator import StageDHandoffCoordinator
from redco.analysis.stage_d_objective_binding import (
    ObjectiveAuthorization,
    ObjectiveBinding,
    fixture_objective_binding,
)
from redco.analysis.stage_d_process_supervision import TrainerProcessStartReceipt
from redco.analysis.stage_d_protocol_manifest import (
    StageDPolicyIdentity,
    StageDProtocolManifest,
)
from redco.analysis.stage_d_provider_billing import (
    ProviderDeploymentBilling,
    StageDProviderBilling,
    rate_duration_estimate_micro_usd,
)
from redco.analysis.stage_d_receipt_ledger import (
    GenesisBinding,
    LedgerError,
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
from redco.analysis.stage_d_shared_initialization import (
    StageDSharedInitializationManifest,
)
from redco.analysis.stage_d_source_artifacts import (
    SourceArtifactError,
    StageDSourceArtifactStore,
)
from redco.analysis.stage_d_source_producer import (
    StageDSourceRolloutProducer,
    _verify_two_slot_scaffold,
    derive_source_trace,
    structural_child_target_id,
    verify_source_trace_semantics,
)
from redco.analysis.stage_d_spawn_provenance import (
    PolicyEventAddress,
    SpawnScope,
    derive_child_lineage,
)
from redco.analysis.stage_d_support_gate import StageDSupportRules, evaluate_support_gate
from redco.analysis.stage_d_terminalization import (
    StageDCleanupEvidence,
    StageDDecisionOutcome,
    StageDDecisionVector,
    StageDTerminalSeal,
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
from redco.analysis.stage_d_trainer_supervisor import StageDTrainerRunLedger
from redco.analysis.stage_d_training_bridge import policy_identity_sha256
from redco.analysis.stage_d_training_completion import StageDTrainingCompletion
from redco.contracts import ActualEvaluationCost, canonical_json

MASTER_SEED = "stage-d-source-producer-test"
EVALUATION_PLAN_BYTES = StageDEvaluationPlan(
    tasks=(StageDEvaluationTask("heldout-1", 9101),),
    reward_min=0.0,
    reward_max=1.0,
    success_reward_threshold=0.5,
).to_bytes()
PREREGISTRATION_BYTES = b"preregistration"
GENESIS_CONFIG_BYTES = b"genesis config"
SUPPORT_RULES_BYTES = canonical_json(
    {
        "schema_version": 1,
        "domain": "redco-stage-d-support-rules-v1",
        "required_papers": 2,
        "required_successes": 1,
        "minimum_targets": 1,
        "maximum_targets": 1,
        "minimum_reward_range": 1e-12,
    }
)


def test_two_slot_scaffold_allows_extra_contiguous_returning_root_turns() -> None:
    root_addresses = tuple(PolicyEventAddress(0, "root", turn, turn) for turn in range(5))
    roots = tuple(
        SimpleNamespace(
            depth=0,
            lineage="root",
            call_kind="policy",
            turn=turn,
            session_call_ordinal=turn,
            scientific_address=address,
            node_index=turn,
        )
        for turn, address in enumerate(root_addresses)
    )
    children = tuple(
        SimpleNamespace(
            depth=1,
            lineage=f"root/child-{slot}",
            call_kind="policy",
            turn=0,
            session_call_ordinal=0,
            spawn_ordinal=slot,
            parent_lineage="root",
            parent_call_ordinal=0,
            parent_tool_call_slot=0,
            parent_tool_call_id="call-0",
            scientific_address=PolicyEventAddress(1, f"root/child-{slot}", 0, 0),
            node_index=3 + slot,
        )
        for slot in range(2)
    )
    nodes = (
        {
            "message": {
                "tool_calls": [
                    {
                        "id": "call-0",
                        "type": "function",
                        "function": {"name": "ipython", "arguments": "{}"},
                    }
                ]
            }
        },
        {"message": {}},
        {"message": {}},
        {"message": {}},
        {"message": {}},
    )
    _verify_two_slot_scaffold(
        (*roots[:3], *children),
        nodes,
        child_parent_event=root_addresses[0],
        child_parent_tool_call_slot=0,
        root_policy_turn_count=2,
    )
    with pytest.raises(ValueError, match="unexpected policy-call count"):
        _verify_two_slot_scaffold(
            (*roots, *children),
            nodes,
            child_parent_event=root_addresses[0],
            child_parent_tool_call_slot=0,
            root_policy_turn_count=2,
            maximum_eligible_root_policy_turn_count=4,
        )
SOURCE_BYTES = b"source manifest"
RUNTIME_BYTES = b"runtime manifest"
SOURCE_EVAL_BYTES = b"source eval"
SCIENTIFIC_EVAL_BYTES = b"scientific eval"
HELDOUT_EVAL_BYTES = b"heldout eval"
BASE_MODEL_BYTES = b"base"
ADAPTER_BYTES = b"adapter"
TOKENIZER_BYTES = b"tokenizer"
RENDERER_BYTES = b"renderer"
SAMPLER_CONFORMANCE_BYTES = b"sampler conformance"
RESOLVED_AGENT_BYTES = b"resolved agent sampling law"
RESOLVED_CLIENT_BYTES = b"resolved train client"
_INITIALIZATION_KEY = _key(71)
SHARED_INITIALIZATION_BYTES = StageDSharedInitializationManifest(
    initialization_id="fixture-shared-initialization",
    checkpoint_id=_INITIALIZATION_KEY.checkpoint_id,
    base_model_manifest_sha256=_INITIALIZATION_KEY.base_model_manifest_sha256,
    adapter_manifest_sha256=_INITIALIZATION_KEY.adapter_manifest_sha256,
    expected_pre_model_sha256="f" * 64,
).to_bytes()


class _CollectionTaskData:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return dict(self._payload)


def _collection_plan(count: int) -> StageDCollectionPlan:
    data = [
        {
            "scientific_group_id": "group-1",
            "example_id": "campaign-example",
            "rollout_slot": index,
        }
        for index in range(count)
    ]
    return StageDCollectionPlan.build(data, master_seed=MASTER_SEED)


def _collection_evidence(sources):
    plan = _collection_plan(len(sources))
    data = [slot.to_payload() for slot in plan.slots]
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


def _zip_bytes(entries: tuple[tuple[str, bytes], ...]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries:
            archive.writestr(name, value)
    return output.getvalue()


def _binding(
    *,
    protocol_manifest_sha256: str = "5" * 64,
    preregistration_sha256: str = "1" * 64,
    source_sha256: str = "2" * 64,
    runtime_sha256: str = "3" * 64,
    config_sha256: str = "4" * 64,
) -> GenesisBinding:
    return GenesisBinding(
        preregistration_sha256=preregistration_sha256,
        source_sha256=source_sha256,
        runtime_sha256=runtime_sha256,
        config_sha256=config_sha256,
        protocol_manifest_sha256=protocol_manifest_sha256,
        master_seed_sha256=_sha256(MASTER_SEED.encode()),
        support_rules_sha256=_sha256(SUPPORT_RULES_BYTES),
    )


def _campaign_protocol(
    *,
    plan: StageDCollectionPlan,
    bindings: dict[ArmName, bytes],
    trainer_tomls: dict[ArmName, bytes],
    authorization: bytes,
) -> StageDProtocolManifest:
    key = _key(71)
    return StageDProtocolManifest(
        preregistration_sha256=_sha256(PREREGISTRATION_BYTES),
        dependency_stack_sha256=_sha256(b"dependency stack"),
        genesis_config_sha256=_sha256(GENESIS_CONFIG_BYTES),
        master_seed_sha256=_sha256(MASTER_SEED.encode()),
        source_sha256=_sha256(SOURCE_BYTES),
        runtime_sha256=_sha256(RUNTIME_BYTES),
        source_eval_config_sha256=_sha256(SOURCE_EVAL_BYTES),
        scientific_eval_config_sha256=_sha256(SCIENTIFIC_EVAL_BYTES),
        heldout_eval_config_sha256=_sha256(HELDOUT_EVAL_BYTES),
        collection_plan_sha256=plan.plan_sha256,
        evaluation_plan_sha256=_sha256(EVALUATION_PLAN_BYTES),
        decision_rule_sha256=_sha256(b"decision rule"),
        support_rules_sha256=_sha256(SUPPORT_RULES_BYTES),
        reload_probe_sha256=_sha256(b"reload probe"),
        shared_initialization_sha256=_sha256(SHARED_INITIALIZATION_BYTES),
        objective_authorization_sha256=_sha256(authorization),
        objective_binding_sha256s=tuple(
            (arm, _sha256(bindings[arm])) for arm in ("stock", "branch-global", "local")
        ),
        trainer_config_sha256s=tuple(
            (arm, _sha256(trainer_tomls[arm])) for arm in ("stock", "branch-global", "local")
        ),
        policy_identity=StageDPolicyIdentity(
            checkpoint_id=key.checkpoint_id,
            base_model_manifest_sha256=key.base_model_manifest_sha256,
            adapter_manifest_sha256=key.adapter_manifest_sha256,
            tokenizer_manifest_sha256=key.tokenizer_manifest_sha256,
            renderer_manifest_sha256=key.renderer_manifest_sha256,
            sampler_conformance_manifest_sha256=(key.sampler_conformance_manifest_sha256),
            resolved_agent_sampling_law_sha256=_sha256(RESOLVED_AGENT_BYTES),
            resolved_train_client_sha256=_sha256(RESOLVED_CLIENT_BYTES),
        ),
        arm_order=("stock", "branch-global", "local"),
        branch_global_scope="within-source-group-all-target-branches-v1",
        trainer_step=1,
        seq_len=64,
    )


def _frozen_protocol_inputs() -> FrozenProtocolInputs:
    key = _key(71)
    return FrozenProtocolInputs(
        preregistration=PREREGISTRATION_BYTES,
        dependency_stack_manifest=b"dependency stack",
        genesis_config=GENESIS_CONFIG_BYTES,
        source=SOURCE_BYTES,
        runtime=RUNTIME_BYTES,
        source_eval_config=SOURCE_EVAL_BYTES,
        scientific_eval_config=SCIENTIFIC_EVAL_BYTES,
        heldout_eval_config=HELDOUT_EVAL_BYTES,
        support_rules=SUPPORT_RULES_BYTES,
        shared_initialization_manifest=SHARED_INITIALIZATION_BYTES,
        base_model_manifest=BASE_MODEL_BYTES,
        adapter_manifest=ADAPTER_BYTES,
        tokenizer_manifest=TOKENIZER_BYTES,
        renderer_manifest=RENDERER_BYTES,
        sampler_conformance_manifest=key.sampler_conformance_manifest,
        resolved_agent_sampling_law=RESOLVED_AGENT_BYTES,
        resolved_train_client=RESOLVED_CLIENT_BYTES,
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
    prompt_tokens: int = 2,
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
            "prompt_tokens": prompt_tokens,
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
            "data": {
                "policy_checkpoint_id": "model@commit",
                "paper_id": f"paper-{trace_id}",
            },
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


def _validate_prepared_action(
    _request: Mapping[str, object],
    _message: Mapping[str, object],
    action_ids: Sequence[int],
) -> None:
    if tuple(action_ids) != (20, 2):
        raise ValueError("prepared action IDs changed")


def _unexpected_prompt_render(_request: Mapping[str, object]) -> tuple[int, ...]:
    raise AssertionError("prepared source reload must use stored engine prompt IDs")


def _prepared_action(
    seed: int,
    *,
    message: dict[str, object] | None = None,
    tool_call: bool = False,
    messages: list[dict[str, object]] | None = None,
    prompt_token_ids: tuple[int, ...] = (10, 11),
    routed_experts_prompt_start: int | None = None,
) -> BehaviorAction:
    legacy = _key(seed)
    request = json.loads(legacy.request)
    if messages is not None:
        request["messages"] = messages
    sampling_params = {
        "temperature": request["temperature"],
        "top_p": request["top_p"],
        "seed": seed,
        "max_tokens": 2,
        "logprobs": 1,
        "skip_special_tokens": False,
        "stop_token_ids": [2],
        "cache_salt": request["extra_body"]["cache_salt"],
    }
    if routed_experts_prompt_start is not None:
        sampling_params["routed_experts_prompt_start"] = routed_experts_prompt_start
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
        prompt_token_ids=prompt_token_ids,
        prepared_engine_request={
            "model": "model@commit",
            "token_ids": list(prompt_token_ids),
            "sampling_params": sampling_params,
        },
    )
    raw_message = message or {"role": "assistant", "content": "duplicate"}
    return BehaviorAction.build(
        key=key,
        action_token_ids=(20, 2),
        behavior_logprobs=(-0.2, -0.1),
        raw_transport_message=raw_message,
        finish_reason="tool_calls" if tool_call else "stop",
        prompt_tokens=len(prompt_token_ids),
        completion_tokens=2,
        termination_kind="tool_calls" if tool_call else "eos",
        eos_token_id=None if tool_call else 2,
        validate_action=_validate_prepared_action,
    )


def _strict_episode() -> bytes:
    parent = PolicyEventAddress(0, "root", 0, 0)

    def child_rlm(
        spawn_ordinal: int,
        episode_ordinal: int,
        *,
        turn: int = 0,
    ) -> dict[str, object]:
        lineage = derive_child_lineage(
            SpawnScope(1, "root", 0, 0, 0),
            spawn_ordinal=spawn_ordinal,
        )
        return {
            "provenance_version": 2,
            "depth": 1,
            "session_id": f"child-{spawn_ordinal}",
            "turn": turn,
            "call_kind": "policy",
            "lineage": lineage,
            "session_call_ordinal": turn,
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


def _two_turn_child_episode() -> bytes:
    episode = json.loads(_strict_episode())
    trace = episode["traces"][0]
    child_call = trace["calls"][2]
    child_node = trace["nodes"][child_call["node"]]
    child_message = _tool_action(82).message
    child_node["message"] = child_message
    child_call["finish_reason"] = "tool_calls"
    tool_node_index = len(trace["nodes"])
    trace["nodes"].append(
        _node(
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": "computed",
            },
            [30],
            [False],
            [],
            parent=child_call["node"],
        )
    )
    continuation_node_index = len(trace["nodes"])
    trace["nodes"].append(
        _node(
            {"role": "assistant", "content": "duplicate"},
            [20, 2],
            [True, True],
            [-0.2, -0.1],
            parent=tool_node_index,
            sampled=True,
        )
    )
    continuation_rlm = dict(child_call["rlm"])
    continuation_rlm["turn"] = 1
    continuation_rlm["session_call_ordinal"] = 1
    trace["calls"].insert(
        3,
        _call(
            continuation_node_index,
            seed=85,
            rlm=continuation_rlm,
            prompt_tokens=5,
        ),
    )
    returning_root_call = trace["calls"][-1]
    returning_root_node = trace["nodes"][returning_root_call["node"]]
    returning_root_node["message"] = _tool_action(84).message
    returning_root_call["finish_reason"] = "tool_calls"
    later_prompt_index = len(trace["nodes"])
    trace["nodes"].append(
        _node({"role": "user", "content": "q"}, [10, 11], [False, False], [])
    )
    later_node_index = len(trace["nodes"])
    trace["nodes"].append(
        _node(
            {"role": "assistant", "content": "duplicate"},
            [20, 2],
            [True, True],
            [-0.2, -0.1],
            parent=later_prompt_index,
            sampled=True,
        )
    )
    later_lineage = derive_child_lineage(
        SpawnScope(1, "root", 1, 0, 1),
        spawn_ordinal=0,
    )
    trace["calls"].append(
        _call(
            later_node_index,
            seed=86,
            rlm={
                "provenance_version": 2,
                "depth": 1,
                "session_id": "later-child-0",
                "turn": 0,
                "call_kind": "policy",
                "lineage": later_lineage,
                "session_call_ordinal": 0,
                "parent_session_id": "root-session",
                "parent_turn": 1,
                "parent_tool_call_id": "call_0",
                "invocation_id": "later-shard",
                "parent_lineage": "root",
                "parent_call_ordinal": 1,
                "parent_tool_call_slot": 0,
                "spawn_ordinal": 0,
                "episode_spawn_ordinal": 2,
                "completed_predecessor_spawn_ordinals": [],
                "completed_episode_spawn_ordinals": [0, 1],
            },
        )
    )
    return canonical_json(episode)


def _later_child_only_episode() -> bytes:
    episode = json.loads(_two_turn_child_episode())
    trace = episode["traces"][0]
    nodes = trace["nodes"]
    calls = trace["calls"]
    kept_nodes = [nodes[index] for index in (0, 1, 6, 7, 10, 11)]
    kept_nodes[3]["parent"] = 2
    kept_nodes[5]["parent"] = 4
    kept_calls = [calls[index] for index in (0, 4, 5)]
    for call, node_index in zip(kept_calls, (1, 3, 5), strict=True):
        call["node"] = node_index
    kept_calls[1]["rlm"]["completed_episode_spawn_ordinals"] = []
    later_rlm = kept_calls[2]["rlm"]
    later_rlm["episode_spawn_ordinal"] = 0
    later_rlm["completed_episode_spawn_ordinals"] = []
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
        receipt = writer.record_reconstruction_qa(
            group_id=spec.commitment.group_id,
            target_id=spec.commitment.target_id,
            recorded_action=spec.recorded_action,
            passed=True,
            report_sha256=report,
            actual_cost=ActualEvaluationCost(cpu_seconds=0.01, wall_seconds=0.01),
        )
        if writer.branch_target_roster_sha256 is not None:
            writer.seal_reconstruction_qa_barrier()
        return receipt

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
        writer.mark_candidate_response_observed(attempt, response_sha256=response)
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
        target_request = writer.put_evidence(f"target-request:{arm_id}".encode())
        target_response = writer.put_evidence(action.to_bytes())
        target_ticket = writer.commit_execution_override(
            attempt,
            address=spec.commitment.target_address,
            action_digest=action.digest,
            disposition="inject",
            request_sha256=target_request,
            response_content_sha256=target_response,
            prompt_tokens=action.prompt_tokens,
            completion_tokens=action.completion_tokens,
            counts_toward_logical_cost=False,
        )
        writer.mark_execution_override_delivered(
            attempt,
            target_ticket,
            typed_response_sha256=target_response,
        )
        request = writer.put_evidence(f"execution-request:{arm_id}".encode())
        call = writer.mark_execution_model_call_started(
            attempt,
            address=matched,
            scheduled_seed=seed_oracle.seed_for(matched),
            request_sha256=request,
        )
        response = writer.put_evidence(f"execution-response:{arm_id}".encode())
        writer.mark_execution_response_observed(
            attempt,
            call,
            response_sha256=response,
        )
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


def test_global_reconstruction_qa_barrier_blocks_all_candidates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    writer = StageDReceiptLedger.create(
        root,
        binding=_binding(),
        master_seed=MASTER_SEED,
    )
    first, first_spec = _produce_into(
        writer,
        rollout_id="rollout-qa-a",
        root_seed=81,
        child_seed=82,
        reward=0.75,
        selected=True,
    )
    second, second_spec = _produce_into(
        writer,
        rollout_id="rollout-qa-b",
        root_seed=83,
        child_seed=84,
        reward=0.25,
        selected=True,
    )
    assert first_spec is not None and second_spec is not None
    _freeze_target_roster(writer, (first, second))

    for index, spec in enumerate((first_spec, second_spec)):
        report = writer.put_evidence(f"qa:{index}".encode())
        writer.record_reconstruction_qa(
            group_id=spec.commitment.group_id,
            target_id=spec.commitment.target_id,
            recorded_action=spec.recorded_action,
            passed=True,
            report_sha256=report,
            actual_cost=ActualEvaluationCost(cpu_seconds=0.01, wall_seconds=0.01),
        )
        with pytest.raises(LedgerError, match="whole-roster"):
            writer.begin_candidate_attempt(
                group_id=first_spec.commitment.group_id,
                target_id=first_spec.commitment.target_id,
                action_slot=1,
            )

    barrier = writer.seal_reconstruction_qa_barrier()
    assert writer.reconstruction_qa_barrier_sha256 == _sha256(barrier)
    attempt = writer.begin_candidate_attempt(
        group_id=first_spec.commitment.group_id,
        target_id=first_spec.commitment.target_id,
        action_slot=1,
    )
    supervisor = writer.put_evidence(b"intentional zero-call cleanup")
    writer.record_zero_call_candidate_failure(
        attempt,
        reason="test cleanup",
        supervisor_evidence_sha256=supervisor,
    )
    writer.close()
    assert inspect_ledger(root).status == "active-clean"


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


@pytest.mark.parametrize(
    "mutation",
    ["reward", "prompt", "mask", "call-node", "node-transport"],
)
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
    elif mutation == "call-node":
        trace["calls"][0]["node"] = trace["calls"][1]["node"]
    else:
        trace["nodes"][0]["routed_experts"] = {"unexpected": "transport-data"}
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
    restored = SourceRollout.verify_bytes(
        source.to_bytes(),
        verifier=writer,
        evidence_loader=lambda digest: (ledger_root / "evidence" / digest).read_bytes(),
        render_prompt=_unexpected_prompt_render,
        validate_action=_validate_prepared_action,
    )
    assert restored == source
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
    assert source.child_target_roster == tuple(
        structural_child_target_id(
            parent,
            rollout_id="rollout-strict",
            parent_tool_call_slot=0,
            spawn_ordinal=ordinal,
        )
        for ordinal in (0, 1)
    )
    child_decision = next(
        decision for decision in source.decisions if decision.node_kind == "child"
    )
    assert child_decision.outer_weight == 1
    roster = StageDBranchTargetRoster.from_sources(
        (source,),
        planned_source_count=1,
        minimum_eligible_sources=1,
    )
    assert roster.targets == ()
    assert len(roster.excluded_targets) == 1
    assert roster.excluded_targets[0].target.target_id == child_target
    assert roster.excluded_targets[0].reason == source.ineligibility_reason
    writer.record_branch_target_roster(roster.to_bytes())
    with pytest.raises(LedgerError, match="absent from the frozen branch target roster"):
        writer.begin_candidate_attempt(
            group_id="group-1",
            target_id=child_target,
            action_slot=1,
        )
    assert inspect_ledger(tmp_path / "ledger").status == "active-clean"
    writer.close()


def test_two_turn_child_finalizes_and_freezes_excluded_commitments(
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
        maximum_eligible_root_policy_turn_count=4,
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
    child_targets: dict[int, str] = {}
    child_addresses: dict[int, PolicyEventAddress] = {}
    for spawn, seed in ((1, 83), (0, 82)):
        address = PolicyEventAddress(
            1,
            derive_child_lineage(
                SpawnScope(1, "root", 0, 0, 0),
                spawn_ordinal=spawn,
            ),
            0,
            0,
        )
        target = structural_child_target_id(
            parent,
            rollout_id="rollout-strict",
            parent_tool_call_slot=0,
            spawn_ordinal=spawn,
        )
        child_targets[spawn] = target
        child_addresses[spawn] = address
        action = _tool_action(seed) if spawn == 0 else _prepared_action(seed)
        pending = producer.reserve_selected_child_policy_call(
            event_address=address,
            target_id=target,
            action_key=action.key,
            pre_action_snapshot=f"child-{spawn}-snapshot".encode(),
            branch_count=4,
            continuation_replicates=1,
            failure_reward=-1.0,
        )
        producer.complete_policy_call(pending, action=action)
    continuation_tool_message = dict(_tool_action(82).message)
    continuation_tool_message["content"] = ""
    continuation_messages = [
        {"role": "user", "content": "q"},
        continuation_tool_message,
        {"role": "tool", "tool_call_id": "call_0", "content": "computed"},
    ]
    continuation = _prepared_action(
        85,
        messages=continuation_messages,
        prompt_token_ids=(10, 11, 20, 2, 30),
        routed_experts_prompt_start=3,
    )
    continuation_address = PolicyEventAddress(
        1,
        child_addresses[0].lineage,
        1,
        1,
    )
    producer.intercept_policy_call(
        event_address=continuation_address,
        action_key=continuation.key,
        node_kind="child",
        target_id=child_targets[0],
        branch_selected=False,
        forward_once=lambda _key: continuation,
    )
    returning_root = _tool_action(84)
    producer.intercept_policy_call(
        event_address=PolicyEventAddress(0, "root", 1, 1),
        action_key=returning_root.key,
        node_kind="root",
        target_id=None,
        branch_selected=False,
        forward_once=lambda _key: returning_root,
    )
    later_parent = PolicyEventAddress(0, "root", 1, 1)
    later_target = structural_child_target_id(
        later_parent,
        rollout_id="rollout-strict",
        parent_tool_call_slot=0,
        spawn_ordinal=0,
    )
    later_address = PolicyEventAddress(
        1,
        derive_child_lineage(
            SpawnScope(1, "root", 1, 0, 1),
            spawn_ordinal=0,
        ),
        0,
        0,
    )
    later_action = _prepared_action(86)
    producer.intercept_policy_call(
        event_address=later_address,
        action_key=later_action.key,
        node_kind="child",
        target_id=later_target,
        branch_selected=False,
        forward_once=lambda _key: later_action,
    )

    source = producer.finalize_episode(_two_turn_child_episode())

    assert source.branch_eligible is False
    assert source.ineligibility_reason == (
        "scientific scaffold has an unexpected policy-call count"
    )
    assert len(source.child_target_roster) == 3
    assert len(set(source.child_target_roster)) == 3
    continuation_decision = next(
        decision
        for decision in source.decisions
        if decision.event_address == continuation_address
    )
    assert continuation_decision.target_id == child_targets[0]
    assert continuation_decision.target_ordinal == 0
    assert continuation_decision.provenance.branch_selected is False
    later_decision = next(
        decision for decision in source.decisions if decision.event_address == later_address
    )
    assert later_decision.target_id == later_target
    assert later_decision.target_ordinal == 2
    assert later_decision.provenance.branch_selected is False
    malformed = json.loads(_two_turn_child_episode())
    for call_index in (2, 3):
        malformed["traces"][0]["calls"][call_index]["rlm"][
            "parent_session_id"
        ] = "bogus-session"
    with pytest.raises(ValueError, match="bind its causal parent event"):
        derive_source_trace(
            canonical_json(malformed),
            decisions=source.decisions,
            child_parent_event=parent,
            child_parent_tool_call_slot=0,
            root_policy_turn_count=2,
            maximum_eligible_root_policy_turn_count=4,
        )
    roster = StageDBranchTargetRoster.from_sources(
        (source,),
        planned_source_count=1,
        minimum_eligible_sources=1,
    )
    assert roster.targets == ()
    assert {item.target.target_id for item in roster.excluded_targets} == set(
        child_targets.values()
    )
    writer.record_branch_target_roster(roster.to_bytes())
    assert inspect_ledger(tmp_path / "ledger").status == "active-clean"
    for target_id in child_targets.values():
        with pytest.raises(LedgerError, match="absent from the frozen branch target roster"):
            writer.begin_candidate_attempt(
                group_id="group-1",
                target_id=target_id,
                action_slot=1,
            )
    writer.close()


def test_later_parent_child_without_frozen_children_finalizes_ineligible(
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
        maximum_eligible_root_policy_turn_count=4,
        base_model_manifest_sha256=_sha256(b"base"),
    )
    for turn, action in ((0, _tool_action(81)), (1, _tool_action(84))):
        producer.intercept_policy_call(
            event_address=PolicyEventAddress(0, "root", turn, turn),
            action_key=action.key,
            node_kind="root",
            target_id=None,
            branch_selected=False,
            forward_once=lambda _key, value=action: value,
        )
    later_parent = PolicyEventAddress(0, "root", 1, 1)
    later_target = structural_child_target_id(
        later_parent,
        rollout_id="rollout-strict",
        parent_tool_call_slot=0,
        spawn_ordinal=0,
    )
    later_address = PolicyEventAddress(
        1,
        derive_child_lineage(
            SpawnScope(1, "root", 1, 0, 1),
            spawn_ordinal=0,
        ),
        0,
        0,
    )
    later_action = _prepared_action(86)
    producer.intercept_policy_call(
        event_address=later_address,
        action_key=later_action.key,
        node_kind="child",
        target_id=later_target,
        branch_selected=False,
        forward_once=lambda _key: later_action,
    )

    source = producer.finalize_episode(_later_child_only_episode())

    assert source.branch_eligible is False
    assert source.ineligibility_reason == (
        "scientific scaffold has an unexpected policy-call count"
    )
    assert len(source.child_target_roster) == 3
    later_decision = next(
        decision for decision in source.decisions if decision.node_kind == "child"
    )
    assert later_decision.target_id == later_target
    assert later_decision.target_ordinal == 2
    assert later_decision.outer_weight == 1
    roster = StageDBranchTargetRoster.from_sources(
        (source,),
        planned_source_count=1,
        minimum_eligible_sources=1,
    )
    assert roster.targets == ()
    assert roster.excluded_targets == ()
    writer.record_branch_target_roster(roster.to_bytes())
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


def test_exact_live_batch_authorization_is_idempotent_and_sealed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    source, writer = _produce(root)
    _freeze_target_roster(writer, (source,))
    sealed_batch_sha256 = writer.put_evidence(b"exact sealed arm batch")
    objective_authorization_sha256 = writer.put_evidence(b"preregistered objective authorization")
    collection_plan_sha256 = writer.put_evidence(b"collection plan")
    collection_receipt_sha256 = writer.put_evidence(b"collection receipt")
    support_report_sha256 = writer.put_evidence(b"support report")
    writer._record_verified_support_gate(support_report_sha256)
    authorization = writer.authorize_stage_d_training_batch(
        arm="stock",
        training_batch_identity="8" * 64,
        sealed_batch_sha256=sealed_batch_sha256,
        objective_sha256="9" * 64,
        objective_authorization_sha256=objective_authorization_sha256,
        collection_plan_sha256=collection_plan_sha256,
        collection_receipt_sha256=collection_receipt_sha256,
        support_report_sha256=support_report_sha256,
        source_sha256s=(source.source_sha256,),
        branch_artifact_sha256s=(),
        consumer_id="stage-d-prime:stock:step:1",
    )
    duplicate = writer.authorize_stage_d_training_batch(
        arm="stock",
        training_batch_identity="8" * 64,
        sealed_batch_sha256=sealed_batch_sha256,
        objective_sha256="9" * 64,
        objective_authorization_sha256=objective_authorization_sha256,
        collection_plan_sha256=collection_plan_sha256,
        collection_receipt_sha256=collection_receipt_sha256,
        support_report_sha256=support_report_sha256,
        source_sha256s=(source.source_sha256,),
        branch_artifact_sha256s=(),
        consumer_id="stage-d-prime:stock:step:1",
    )
    assert duplicate.receipt == authorization.receipt
    writer.close()
    writer = StageDReceiptLedger(root, master_seed=MASTER_SEED)
    assert (
        writer.authorize_stage_d_training_batch(
            arm="stock",
            training_batch_identity="8" * 64,
            sealed_batch_sha256=sealed_batch_sha256,
            objective_sha256="9" * 64,
            objective_authorization_sha256=objective_authorization_sha256,
            collection_plan_sha256=collection_plan_sha256,
            collection_receipt_sha256=collection_receipt_sha256,
            support_report_sha256=support_report_sha256,
            source_sha256s=(source.source_sha256,),
            branch_artifact_sha256s=(),
            consumer_id="stage-d-prime:stock:step:1",
        ).receipt
        == authorization.receipt
    )
    with pytest.raises(LedgerError, match="different authorization"):
        writer.authorize_stage_d_training_batch(
            arm="stock",
            training_batch_identity="8" * 64,
            sealed_batch_sha256=sealed_batch_sha256,
            objective_sha256="9" * 64,
            objective_authorization_sha256=objective_authorization_sha256,
            collection_plan_sha256=collection_plan_sha256,
            collection_receipt_sha256=collection_receipt_sha256,
            support_report_sha256=support_report_sha256,
            source_sha256s=(source.source_sha256,),
            branch_artifact_sha256s=(),
            consumer_id="different-consumer",
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
    support_report_sha256 = writer.put_evidence(b"support report")
    writer._record_verified_support_gate(support_report_sha256)
    batch_sha256 = writer.put_evidence(batch.to_bytes())
    authorization = writer.authorize_stage_d_training_batch(
        arm="stock",
        training_batch_identity=batch.batch_identity,
        sealed_batch_sha256=batch_sha256,
        objective_sha256=binding.objective_sha256,
        objective_authorization_sha256=objective_authorization_sha256,
        collection_plan_sha256=collection_plan_sha256,
        collection_receipt_sha256=collection_receipt_sha256,
        support_report_sha256=support_report_sha256,
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


@pytest.mark.parametrize("crash_after", [None, "stock", "branch-global", "local"])
def test_campaign_transaction_reconstructs_after_single_terminal_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_after: str | None,
) -> None:
    arms: tuple[ArmName, ...] = ("stock", "branch-global", "local")
    trainer_tomls = {arm: f"# {arm}\n".encode() for arm in arms}
    bindings: dict[ArmName, bytes] = {}
    objective_hashes: list[tuple[ArmName, str]] = []
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
    collection_plan = _collection_plan(2)
    protocol = _campaign_protocol(
        plan=collection_plan,
        bindings=bindings,
        trainer_tomls=trainer_tomls,
        authorization=authorization,
    )
    root = tmp_path / "ledger"
    writer = StageDReceiptLedger.create(
        root,
        binding=_binding(
            protocol_manifest_sha256=protocol.manifest_sha256,
            preregistration_sha256=protocol.preregistration_sha256,
            source_sha256=protocol.source_sha256,
            runtime_sha256=protocol.runtime_sha256,
            config_sha256=protocol.genesis_config_sha256,
        ),
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
    support_report = evaluate_support_gate(
        (selected, control),
        (artifact,),
        target_roster,
        paper_ids={
            selected.source_sha256: "paper-rollout-live-1",
            control.source_sha256: "paper-rollout-live-2",
        },
        rules=StageDSupportRules.from_bytes(SUPPORT_RULES_BYTES),
    )

    observed_plan, collection_receipt = _collection_evidence((selected, control))
    assert observed_plan == collection_plan
    compile_kwargs = dict(
        ledger=writer,
        ledger_root=root,
        protocol_manifest_bytes=protocol.to_bytes(),
        preregistered_protocol_manifest_sha256=protocol.manifest_sha256,
        source_rollout_bytes=(selected.to_bytes(), control.to_bytes()),
        collection_plan=collection_plan,
        collection_receipt_bytes=collection_receipt,
        preregistered_collection_plan_sha256=collection_plan.plan_sha256,
        branch_artifact_bytes=(artifact_bytes,),
        support_report_bytes=support_report,
        support_rules_bytes=SUPPORT_RULES_BYTES,
        encode_action=lambda _request, _message: (20, 2),
        render_prompt=lambda _request: (10, 11),
        master_seed=MASTER_SEED,
        objective_binding_bytes=bindings,
        trainer_toml_bytes=trainer_tomls,
        objective_authorization_bytes=authorization,
        preregistered_objective_authorization_sha256=_sha256(authorization),
        trainer_step=1,
        seq_len=64,
        allow_test_fixture_collection=True,
    )
    triggered = False

    def crash_hook(arm: str) -> None:
        nonlocal triggered
        if arm == crash_after and not triggered:
            triggered = True
            raise RuntimeError(f"injected crash after {arm}")

    try:
        campaign = compile_authorize_seal_campaign(
            **compile_kwargs,
            after_arm_authorized=crash_hook if crash_after is not None else None,
        )
    except RuntimeError as error:
        assert str(error) == f"injected crash after {crash_after}"
        writer.close()
        writer = StageDReceiptLedger(root, master_seed=MASTER_SEED)
        compile_kwargs["ledger"] = writer
        campaign = compile_authorize_seal_campaign(**compile_kwargs)
    assert triggered is (crash_after is not None)
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
    recovered_campaign = recover_sealed_campaign(
        ledger_root=root,
        expected_ledger_seal_bytes=campaign.ledger_seal_bytes,
        protocol_manifest_bytes=protocol.to_bytes(),
        preregistered_protocol_manifest_sha256=protocol.manifest_sha256,
        source_rollout_bytes=(selected.to_bytes(), control.to_bytes()),
        collection_plan=collection_plan,
        collection_receipt_bytes=collection_receipt,
        branch_artifact_bytes=(artifact_bytes,),
        encode_action=lambda _request, _message: (20, 2),
        render_prompt=lambda _request: (10, 11),
        master_seed=MASTER_SEED,
        objective_binding_bytes=bindings,
        trainer_toml_bytes=trainer_tomls,
        objective_authorization_bytes=authorization,
        allow_test_fixture_collection=True,
    )
    assert recovered_campaign == campaign
    monkeypatch.setattr(
        campaign_store_module,
        "materialize_prime_rollout_bytes",
        lambda batch: canonical_json({"arm": batch.arm, "batch_identity": batch.batch_identity}),
    )
    bundle = StageDCampaignStore(tmp_path / "bundle").persist(
        campaign=campaign,
        ledger_root=root,
        collection_plan=collection_plan,
        collection_receipt_bytes=collection_receipt,
        source_rollout_bytes=(selected.to_bytes(), control.to_bytes()),
        branch_artifact_bytes=(artifact_bytes,),
        objective_binding_bytes=bindings,
        trainer_toml_bytes=trainer_tomls,
        evaluation_plan_bytes=EVALUATION_PLAN_BYTES,
        decision_rule_bytes=b"decision rule",
        reload_probe_bytes=b"reload probe",
        frozen_inputs=_frozen_protocol_inputs(),
    )
    assert verify_campaign_bundle(bundle.root) == bundle
    if crash_after is None:
        crash_root = tmp_path / "handoff-crash"
        StageDHandoffCoordinator.create(
            crash_root,
            preregistration_sha256=protocol.preregistration_sha256,
            protocol_manifest_sha256=protocol.manifest_sha256,
            handoff_policy_sha256=_sha256(b"bounded handoff policy"),
        )
        crashed = [False]

        def crash_handoff(stage: str, _path: Path) -> None:
            if stage == "after-evaluation-ledger-pending-fsync" and not crashed[0]:
                crashed[0] = True
                raise RuntimeError("injected handoff adoption crash")

        with pytest.raises(RuntimeError, match="handoff adoption crash"):
            StageDHandoffCoordinator(
                crash_root,
                fault_hook=crash_handoff,
            ).adopt_campaign(bundle.root)
        assert crashed[0]
        StageDHandoffCoordinator(crash_root).adopt_campaign(bundle.root)
        linked_root = tmp_path / "handoff-linked-crash"
        StageDHandoffCoordinator.create(
            linked_root,
            preregistration_sha256=protocol.preregistration_sha256,
            protocol_manifest_sha256=protocol.manifest_sha256,
            handoff_policy_sha256=_sha256(b"bounded handoff policy"),
        )
        linked = [False]

        def crash_after_record_link(stage: str, path: Path) -> None:
            if (
                stage == "after-evaluation-ledger-link"
                and path.parent.name == "records"
                and path.name == "00000001.json"
                and not linked[0]
            ):
                linked[0] = True
                raise RuntimeError("injected linked handoff record crash")

        with pytest.raises(RuntimeError, match="linked handoff record crash"):
            StageDHandoffCoordinator(
                linked_root,
                fault_hook=crash_after_record_link,
            ).adopt_campaign(bundle.root)
        assert linked[0]
        linked_handoff = StageDHandoffCoordinator(linked_root)
        linked_record = linked_handoff.adopt_campaign(bundle.root)
        linked_event = linked_handoff.inspect().event("campaign_adopted")
        assert linked_event is not None
        assert linked_record == linked_event[1]
    handoff = StageDHandoffCoordinator.create(
        tmp_path / "handoff",
        preregistration_sha256=protocol.preregistration_sha256,
        protocol_manifest_sha256=protocol.manifest_sha256,
        handoff_policy_sha256=_sha256(b"bounded handoff policy"),
    )
    with pytest.raises(RuntimeError, match="arbitrary report commits are disabled"):
        handoff.commit_report(
            report_bytes=b"premature report",
            decision_evidence_bytes=b"premature decision",
            billing_evidence_bytes=b"premature billing",
        )
    if crash_after is None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            campaign_adoptions = tuple(
                pool.map(lambda _: handoff.adopt_campaign(bundle.root), range(2))
            )
        assert len(set(campaign_adoptions)) == 1
        campaign_adoption = campaign_adoptions[0]
        assert handoff.inspect().record_count == 2
    else:
        campaign_adoption = handoff.adopt_campaign(bundle.root)
    assert handoff.adopt_campaign(bundle.root) == campaign_adoption
    assert handoff.inspect().campaign_bundle_manifest_sha256 == bundle.manifest_sha256
    trainer_ledger = StageDTrainerRunLedger.create(
        tmp_path / "trainer-ledger",
        campaign_manifest_sha256=bundle.manifest_sha256,
        protocol_manifest_sha256=protocol.manifest_sha256,
        shared_initialization_manifest_sha256=_sha256(b"shared initialization"),
        expected_pre_model_sha256="d" * 64,
        expected_base_model_manifest_sha256="e" * 64,
        reload_probe_sha256="f" * 64,
        trainer_step=1,
        batch_identities={
            "stock": "1" * 64,
            "branch-global": "2" * 64,
            "local": "3" * 64,
        },
        trainer_config_sha256s={
            "stock": "4" * 64,
            "branch-global": "5" * 64,
            "local": "6" * 64,
        },
        process_command_sha256s={
            "stock": "a" * 64,
            "branch-global": "b" * 64,
            "local": "c" * 64,
        },
        process_environment_sha256s={
            "stock": "7" * 64,
            "branch-global": "8" * 64,
            "local": "9" * 64,
        },
    )
    post_by_arm = {"stock": "7" * 64, "branch-global": "8" * 64, "local": "9" * 64}
    monkeypatch.setattr(
        "redco.analysis.stage_d_checkpoint_evidence.adapter_file_state_sha256",
        lambda path, **_kwargs: post_by_arm[path.parent.name.removeprefix("checkpoint-")],
    )
    for arm in arms:
        launch_id = f"{arm}-handoff"
        trainer_ledger.claim_launch(arm=arm, launch_id=launch_id)
        _mark_process_started(trainer_ledger, arm=arm, launch_id=launch_id)
        trainer_ledger.mark_initialization_verified(
            arm=arm,
            launch_id=launch_id,
            observed_pre_model_sha256="d" * 64,
        )
        trainer_ledger.mark_batch_verified(
            arm=arm,
            launch_id=launch_id,
            batch_identity={"stock": "1" * 64, "branch-global": "2" * 64, "local": "3" * 64}[arm],
        )
        trainer_ledger.mark_optimizer_started(
            arm=arm,
            launch_id=launch_id,
            trainer_step=1,
        )
        post_model = post_by_arm[arm]
        trainer_ledger.mark_optimizer_completed(
            arm=arm,
            launch_id=launch_id,
            trainer_step=1,
            post_model_sha256=post_model,
        )
        checkpoint = _checkpoint_evidence(
            arm,
            tmp_path,
            post_model,
            launch_id=launch_id,
        )
        trainer_ledger.commit_checkpoint(
            arm=arm,
            launch_id=launch_id,
            checkpoint_root=checkpoint[0],
            checkpoint_manifest_bytes=checkpoint[2],
            metrics_bytes=checkpoint[3],
            reload_evidence_bytes=checkpoint[4],
            reload_output_bytes=checkpoint[5],
            reload_process_result_bytes=checkpoint[6],
            trainer_step=1,
        )
    training_adoption = handoff.adopt_training(trainer_ledger)
    assert handoff.adopt_training(trainer_ledger) == training_adoption
    training_completion = StageDTrainingCompletion.build(trainer_ledger)
    checkpoint_by_arm = {item.arm: item for item in training_completion.arms}
    runtime_bundle = _zip_bytes(
        (
            ("client.py", b"client source"),
            ("scorer.py", b"scorer source"),
            ("serializer.py", b"serializer source"),
            ("server.py", b"server source"),
            ("task_runtime.py", b"task runtime source"),
        )
    )
    programs = tuple(
        EvaluationProgramBinding(
            arm=arm,
            role=role,
            absolute_executable="/opt/redco/.venv/bin/python",
            executable_sha256="a" * 64,
            argv=(
                "/opt/redco/.venv/bin/python",
                f"{role}.py",
                arm,
                *([f"/opt/redco/checkpoints/{arm}"] if role == "server" else []),
            ),
            working_directory="/opt/redco",
            checkpoint_root=f"/opt/redco/checkpoints/{arm}",
            environment=(("PYTHONHASHSEED", "0"),),
            source_sha256s=((f"{role}.py", _sha256(f"{role} source".encode())),),
            checkpoint_manifest_sha256=checkpoint_by_arm[arm].checkpoint_manifest_sha256,
            post_model_sha256=checkpoint_by_arm[arm].post_model_sha256,
            reload_evidence_sha256=checkpoint_by_arm[arm].reload_evidence_sha256,
            endpoint=f"http://127.0.0.1:{8100 + arm_index}",
            gpu_assignment=(arm_index,),
            cache_namespace=f"heldout-{arm}",
        )
        for arm_index, arm in enumerate(arms)
        for role in ("server", "client")
    )
    execution_manifest = StageDEvaluationExecutionManifest(
        evaluation_ledger_id="0" * 64,
        protocol_manifest_sha256=protocol.manifest_sha256,
        trainer_ledger_head_sha256=training_completion.trainer_ledger_head_sha256,
        trainer_record_count=training_completion.trainer_record_count,
        heldout_eval_config_sha256=protocol.heldout_eval_config_sha256,
        evaluation_plan_sha256=protocol.evaluation_plan_sha256,
        decision_rule_sha256=protocol.decision_rule_sha256,
        runtime_entrypoints=(
            EvaluationRuntimeEntrypoint(
                "task_runner",
                "task_runtime.py",
                "task_runtime",
                "run_task",
                "redco-stage-d-worker-ipc-v1",
                _sha256(b"task runtime source"),
            ),
            EvaluationRuntimeEntrypoint(
                "scorer",
                "scorer.py",
                "scorer",
                "score",
                "redco-stage-d-scorer-v1",
                _sha256(b"scorer source"),
            ),
            EvaluationRuntimeEntrypoint(
                "request_serializer",
                "serializer.py",
                "serializer",
                "serialize",
                "redco-stage-d-request-serializer-v1",
                _sha256(b"serializer source"),
            ),
        ),
        runtime_worker_image="python@sha256:" + "f" * 64,
        runtime_bundle_path=str((handoff.evidence.root / _sha256(runtime_bundle)).resolve()),
        runtime_bundle_sha256=_sha256(runtime_bundle),
        container_runtime_executable="/usr/bin/docker",
        container_runtime_executable_sha256=_sha256(b"docker executable"),
        supervisor_limits=EvaluationSupervisorLimits(
            "/opt/redco/evaluation-control",
            "/opt/redco/evaluation-logs",
            "/sys/fs/cgroup",
            7200,
            30,
            30,
            5,
            50,
            1048576,
        ),
        max_server_launches_per_arm=2,
        max_client_launches_per_arm=2,
        server_replacement_policy="before-first-dispatch-only-v1",
        programs=programs,
        schedule=tuple(
            EvaluationScheduleUnit(index, arm, 0, "heldout-1", 9101)
            for index, arm in enumerate(arms)
        ),
    )
    authorized_evaluation = handoff.authorize_evaluation(
        execution_manifest_bytes=execution_manifest.to_bytes(),
        runtime_bundle_bytes=runtime_bundle,
    )
    evaluation_authorization = authorized_evaluation.authorization
    authorization_record = authorized_evaluation.authorization_record_sha256
    assert (
        handoff.authorize_evaluation(
            execution_manifest_bytes=execution_manifest.to_bytes(),
            runtime_bundle_bytes=runtime_bundle,
        ).authorization_record_sha256
        == authorization_record
    )
    evaluation_ledger = handoff.materialize_evaluation_ledger(tmp_path / "evaluations")
    monkeypatch.setattr(
        StageDEvaluationLedger,
        "_verify_process_receipt",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        ActuatedProcessReceipt,
        "is_same_live_process",
        lambda _self: True,
    )
    monkeypatch.setattr(
        ActuatedProcessReceipt,
        "is_same_live_tree",
        lambda _self: True,
    )
    monkeypatch.setattr(
        TrainerProcessStartReceipt,
        "is_same_live_process",
        lambda _self: True,
    )
    for arm in arms:
        task, session = _start_evaluation_arm(evaluation_ledger, arm)
        _finish_evaluation_task(evaluation_ledger, task, session)
        evaluation_ledger.complete_arm(arm)
    evaluation_ledger.seal()
    authorization_path = tmp_path / "evaluation-authorization.json"
    authorization_path.write_bytes(evaluation_authorization.to_bytes())
    heldout_config_path = tmp_path / "heldout-eval.toml"
    heldout_config_path.write_bytes(HELDOUT_EVAL_BYTES)
    evaluation_plan_path = tmp_path / "evaluation-plan.json"
    evaluation_plan_path.write_bytes(EVALUATION_PLAN_BYTES)
    completion_path = tmp_path / "evaluation-completion.json"
    evaluation_completion = commit_sealed_heldout_evaluation(
        authorization_path=authorization_path,
        trainer_ledger=trainer_ledger,
        evaluation_ledger=evaluation_ledger,
        heldout_eval_config_path=heldout_config_path,
        evaluation_plan_path=evaluation_plan_path,
        retained_evidence_root=tmp_path / "retained-evaluation",
        destination=completion_path,
    )
    evaluation_adoption = handoff.adopt_evaluation(
        evaluation_ledger,
        evaluation_completion.to_bytes(),
    )
    assert (
        handoff.adopt_evaluation(
            evaluation_ledger,
            evaluation_completion.to_bytes(),
        )
        == evaluation_adoption
    )
    economic_metrics = b"frozen Stage D economic metrics"
    credit_metrics = b"frozen Stage D credit metrics"
    decisions = StageDDecisionVector(
        StageDDecisionOutcome(
            "positive",
            protocol.decision_rule_sha256,
            _sha256(economic_metrics),
            None,
        ),
        StageDDecisionOutcome(
            "indeterminate",
            protocol.decision_rule_sha256,
            _sha256(credit_metrics),
            "paired interval crossed the frozen boundary",
        ),
    )
    wallet_before = b"Prime wallet before"
    wallet_after = b"Prime wallet after"
    provider_receipt = b"Prime deployment receipt"
    duration = 3_600_000
    provider_entry = ProviderDeploymentBilling(
        attempt_id="stage-d-production-001",
        phase="evaluation",
        provider="prime-intellect",
        resource_id="resource-001",
        location="us-west",
        gpu_type="L40S",
        gpu_count=2,
        pricing_type="on-demand",
        started_unix_milliseconds=1_000_000,
        ended_unix_milliseconds=4_600_000,
        billed_duration_milliseconds=duration,
        rate_micro_usd_per_hour=2_000_000,
        rate_duration_estimate_micro_usd=rate_duration_estimate_micro_usd(
            rate_micro_usd_per_hour=2_000_000,
            billed_duration_milliseconds=duration,
        ),
        provider_charge_status="reported",
        provider_charge_micro_usd=2_000_000,
        provider_charge_unavailable_reason=None,
        provider_receipt_sha256=_sha256(provider_receipt),
    )
    billing = StageDProviderBilling(
        currency="USD",
        deployments=(provider_entry,),
        total_provider_charge_micro_usd=2_000_000,
        total_rate_duration_estimate_micro_usd=2_000_000,
        wallet_before_micro_usd=40_000_000,
        wallet_after_micro_usd=38_000_000,
        wallet_delta_micro_usd=2_000_000,
        wallet_before_receipt_sha256=_sha256(wallet_before),
        wallet_after_receipt_sha256=_sha256(wallet_after),
    )
    cleanup_receipts = {
        _sha256(value): value
        for value in (b"pod terminated", b"zero persistent disks", b"cgroup empty")
    }
    cleanup = StageDCleanupEvidence(
        "terminated",
        "zero-confirmed",
        "contained-empty",
        tuple(sorted(cleanup_receipts)),
    )
    seal_bytes = handoff.finalize_terminal(
        terminal_status="completed",
        terminal_phase="evaluation",
        termination_code="success",
        decisions=decisions,
        decision_evidence={
            _sha256(economic_metrics): economic_metrics,
            _sha256(credit_metrics): credit_metrics,
        },
        billing=billing,
        billing_receipts={
            _sha256(wallet_before): wallet_before,
            _sha256(wallet_after): wallet_after,
            _sha256(provider_receipt): provider_receipt,
        },
        cleanup=cleanup,
        cleanup_receipts=cleanup_receipts,
        evaluation_ledger=evaluation_ledger,
        evaluation_completion_bytes=evaluation_completion.to_bytes(),
    )
    assert StageDTerminalSeal.from_bytes(seal_bytes).terminal_status == "completed"
    assert handoff.inspect().sealed
    assert (
        handoff.finalize_terminal(
            terminal_status="completed",
            terminal_phase="evaluation",
            termination_code="success",
            decisions=decisions,
            decision_evidence={
                _sha256(economic_metrics): economic_metrics,
                _sha256(credit_metrics): credit_metrics,
            },
            billing=billing,
            billing_receipts={
                _sha256(wallet_before): wallet_before,
                _sha256(wallet_after): wallet_after,
                _sha256(provider_receipt): provider_receipt,
            },
            cleanup=cleanup,
            cleanup_receipts=cleanup_receipts,
            evaluation_ledger=evaluation_ledger,
            evaluation_completion_bytes=evaluation_completion.to_bytes(),
        )
        == seal_bytes
    )
    assert tuple(arm for arm, _ in bundle.prime_rollout_paths) == arms
    assert (
        StageDCampaignStore(bundle.root).persist(
            campaign=campaign,
            ledger_root=root,
            collection_plan=collection_plan,
            collection_receipt_bytes=collection_receipt,
            source_rollout_bytes=(selected.to_bytes(), control.to_bytes()),
            branch_artifact_bytes=(artifact_bytes,),
            objective_binding_bytes=bindings,
            trainer_toml_bytes=trainer_tomls,
            evaluation_plan_bytes=EVALUATION_PLAN_BYTES,
            decision_rule_bytes=b"decision rule",
            reload_probe_bytes=b"reload probe",
            frozen_inputs=_frozen_protocol_inputs(),
        )
        == bundle
    )

    def remanifest(root: Path, changed_path: Path) -> None:
        manifest_path = root / "manifest.json"
        payload = json.loads(manifest_path.read_bytes())
        relative = changed_path.relative_to(root).as_posix()
        for entry in payload["entries"]:
            if entry["path"] == relative:
                changed = changed_path.read_bytes()
                entry["sha256"] = _sha256(changed)
                entry["size_bytes"] = len(changed)
                break
        manifest_path.write_bytes(canonical_json(payload))

    tampered_payload_root = tmp_path / "tampered-payload-bundle"
    shutil.copytree(bundle.root, tampered_payload_root)
    payload_path = next((tampered_payload_root / "sources").glob("*.json"))
    payload = json.loads(payload_path.read_bytes())
    payload["source"]["reward"] = 0.125
    payload_path.write_bytes(canonical_json(payload))
    remanifest(tampered_payload_root, payload_path)
    with pytest.raises(ValueError, match="semantic digest"):
        verify_campaign_bundle(tampered_payload_root)

    tampered_receipt_root = tmp_path / "tampered-receipt-bundle"
    shutil.copytree(bundle.root, tampered_receipt_root)
    receipt_path = next((tampered_receipt_root / "sources").glob("*.json"))
    receipt_payload = json.loads(receipt_path.read_bytes())
    receipt_payload["producer_receipt"]["completion_sequence"] += 1
    receipt_path.write_bytes(canonical_json(receipt_payload))
    remanifest(tampered_receipt_root, receipt_path)
    with pytest.raises(ValueError, match="anchored"):
        verify_campaign_bundle(tampered_receipt_root)


def test_campaign_transaction_rejects_omitted_completed_source(
    tmp_path: Path,
) -> None:
    arms: tuple[ArmName, ...] = ("stock", "branch-global", "local")
    trainer_tomls = {arm: f"# {arm}\n".encode() for arm in arms}
    bindings: dict[ArmName, bytes] = {}
    objective_hashes: list[tuple[ArmName, str]] = []
    for arm in arms:
        payload = fixture_objective_binding(arm).to_payload()
        payload["evidence_class"] = "live"
        binding = ObjectiveBinding.from_bytes(canonical_json(payload))
        bindings[arm] = canonical_json(binding.to_payload())
        objective_hashes.append((arm, binding.objective_sha256))
    authorization = ObjectiveAuthorization("live", tuple(sorted(objective_hashes))).to_bytes()
    collection_plan = _collection_plan(2)
    protocol = _campaign_protocol(
        plan=collection_plan,
        bindings=bindings,
        trainer_tomls=trainer_tomls,
        authorization=authorization,
    )
    root = tmp_path / "omitted-source-ledger"
    writer = StageDReceiptLedger.create(
        root,
        binding=_binding(
            protocol_manifest_sha256=protocol.manifest_sha256,
            preregistration_sha256=protocol.preregistration_sha256,
            source_sha256=protocol.source_sha256,
            runtime_sha256=protocol.runtime_sha256,
            config_sha256=protocol.genesis_config_sha256,
        ),
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
    observed_plan, collection_receipt = _collection_evidence((selected, control))
    assert observed_plan == collection_plan

    with pytest.raises(ValueError, match="every completed ledger source"):
        compile_authorize_seal_campaign(
            ledger=writer,
            ledger_root=root,
            protocol_manifest_bytes=protocol.to_bytes(),
            preregistered_protocol_manifest_sha256=protocol.manifest_sha256,
            source_rollout_bytes=(selected.to_bytes(), control.to_bytes()),
            collection_plan=collection_plan,
            collection_receipt_bytes=collection_receipt,
            preregistered_collection_plan_sha256=collection_plan.plan_sha256,
            branch_artifact_bytes=(artifact_bytes,),
            support_report_bytes=b"not-a-passing-report",
            support_rules_bytes=SUPPORT_RULES_BYTES,
            encode_action=lambda _request, _message: (20, 2),
            render_prompt=lambda _request: (10, 11),
            master_seed=MASTER_SEED,
            objective_binding_bytes=bindings,
            trainer_toml_bytes=trainer_tomls,
            objective_authorization_bytes=authorization,
            preregistered_objective_authorization_sha256=_sha256(authorization),
            trainer_step=1,
            seq_len=64,
            allow_test_fixture_collection=True,
        )
    writer.close()
