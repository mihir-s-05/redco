from __future__ import annotations

import asyncio
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_stage_d_scientific_branch_group import (
    TrustedReceiptStore,
    _conformance,
    _request,
)
from test_stage_d_three_arm_bridge import (
    _decision,
    _inputs,
    _replace_source,
    _source,
)

from redco.analysis.stage_d_collection import (
    StageDCollectionPlan,
    derive_scientific_group_id,
    derive_source_episode_seed_and_salt,
    run_exact_source_collection,
    verify_collection_outcomes,
    verify_collection_receipt,
    verify_direct_collection_config,
)
from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.contracts import canonical_json


class _Data:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return dict(self.payload)


def test_scientific_group_identity_matches_frozen_stage_d_plan() -> None:
    assert derive_scientific_group_id(
        namespace="redco-stage-d1-support-v1",
        example_id="qasper-71f2b368228a748fd348f1abf540236568a61b07",
    ) == "stage-d-group-4346a8ce81f1b4968e5be9a3"


def _plan_and_episodes():
    sources, _ = _inputs()
    task_data = [
        {
            "scientific_group_id": source.group_id,
            "example_id": "shared-example",
            "rollout_slot": index,
        }
        for index, source in enumerate(sources)
    ]
    plan = StageDCollectionPlan.build(task_data, master_seed="master")
    episodes = [
        SimpleNamespace(
            ok=True,
            traces=[
                SimpleNamespace(
                    id=source.rollout_id,
                    task=SimpleNamespace(data=_Data(data)),
                )
            ],
        )
        for source, data in zip(sources, task_data, strict=True)
    ]
    return plan, episodes, sources


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_sources_and_evidence(*, tamper: str | None = None):
    task_data = [
        {
            "scientific_group_id": "shared-group",
            "example_id": "shared-example",
            "rollout_slot": slot,
        }
        for slot in range(2)
    ]
    plan = StageDCollectionPlan.build(task_data, master_seed="strict-master")
    store = TrustedReceiptStore()
    sources = []
    evidence: dict[str, bytes] = {}
    episodes = []
    for index, (slot, data) in enumerate(zip(plan.slots, task_data, strict=True)):
        request_seed = slot.seed + 1 if tamper == "seed" and index == 0 else slot.seed
        request_salt = (
            f"{slot.cache_salt}-tampered"
            if tamper == "cache_salt" and index == 0
            else slot.cache_salt
        )
        request = _request(request_seed)
        request["extra_body"] = {"cache_salt": request_salt}
        key = ExactActionKey.build(
            checkpoint_id="model@commit",
            base_model_manifest=b"base",
            adapter_manifest=b"adapter",
            tokenizer_manifest=b"tokenizer",
            renderer_manifest=b"renderer",
            sampler_conformance_manifest=_conformance(),
            action_selection_policy="direct_single_sample",
            transport_retry_policy="fail_before_action_no_resample",
            request=request,
            prompt_token_ids=(10, 11),
            render_prompt=lambda _request: (10, 11),
        )
        action = BehaviorAction.build(
            key=key,
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
        rollout_id = f"strict-rollout-{index}"
        decision = _decision(
            f"root-{index}",
            action,
            node_kind="root",
            outer_weight=Fraction(1),
            sequence=index * 2,
            verifier=store,
            rollout_id=rollout_id,
            group_id="shared-group",
        )
        source = _source(
            rollout_id,
            0.0,
            (decision,),
            str(index),
            group_id="shared-group",
        )
        trace = canonical_json({"traces": [{"task": {"data": data}}]})
        source = _replace_source(source, trace_sha256=_sha256(trace))
        sources.append(source)
        evidence[source.trace_sha256] = trace
        episodes.append(
            SimpleNamespace(
                ok=True,
                traces=[
                    SimpleNamespace(
                        id=rollout_id,
                        task=SimpleNamespace(data=_Data(data)),
                    )
                ],
            )
        )
    return plan, episodes, tuple(sources), evidence


def test_episode_seed_is_unique_per_slot_and_exact_on_replay() -> None:
    first = derive_source_episode_seed_and_salt(
        master_seed="master",
        scientific_group_id="group",
        rollout_slot=0,
    )
    replay = derive_source_episode_seed_and_salt(
        master_seed="master",
        scientific_group_id="group",
        rollout_slot=0,
    )
    second = derive_source_episode_seed_and_salt(
        master_seed="master",
        scientific_group_id="group",
        rollout_slot=1,
    )
    assert first == replay
    assert first != second


def test_collection_receipt_covers_every_planned_source() -> None:
    plan, episodes, sources = _plan_and_episodes()
    receipt = verify_collection_outcomes(
        plan,
        episodes,
        sources,
    )
    expected = str(len(plan.slots)).encode()
    assert b'"planned_slot_count":' + expected in receipt
    assert b'"terminal_slot_count":' + expected in receipt


def test_collection_receipt_rejects_premature_stop() -> None:
    plan, episodes, sources = _plan_and_episodes()
    with pytest.raises(ValueError, match="complete planned roster"):
        verify_collection_outcomes(
            plan,
            episodes[:1],
            sources[:1],
        )


def test_strict_receipt_rejects_same_group_slot_swap() -> None:
    plan, episodes, sources, evidence = _strict_sources_and_evidence()
    receipt = verify_collection_outcomes(plan, episodes, sources)
    evidence_loader = evidence.__getitem__
    verify_collection_receipt(
        plan,
        sources,
        receipt,
        evidence_loader=evidence_loader,
    )
    payload = json.loads(receipt)
    first = payload["dispositions"][0]
    second = payload["dispositions"][1]
    for field in ("rollout_id", "source_sha256", "branch_eligible", "ineligibility_reason"):
        first[field], second[field] = second[field], first[field]
    swapped = canonical_json(payload)
    with pytest.raises(ValueError, match="differs from its exact source"):
        verify_collection_receipt(
            plan,
            sources,
            swapped,
            evidence_loader=evidence_loader,
        )


@pytest.mark.parametrize("tamper", ["seed", "cache_salt"])
def test_strict_receipt_rejects_source_sampling_tamper(tamper: str) -> None:
    plan, episodes, sources, evidence = _strict_sources_and_evidence(tamper=tamper)
    receipt = verify_collection_outcomes(plan, episodes, sources)
    with pytest.raises(ValueError, match="differs from its exact source"):
        verify_collection_receipt(
            plan,
            sources,
            receipt,
            evidence_loader=evidence.__getitem__,
        )


def test_direct_collection_config_is_one_shot_and_serial(tmp_path: Path) -> None:
    retries = SimpleNamespace(max_retries=0)
    config = SimpleNamespace(
        server=False,
        num_rollouts=1,
        num_tasks=2,
        shuffle=False,
        resume=None,
        push=False,
        rich=False,
        max_concurrent=1,
        output_dir=tmp_path / "run",
        env=SimpleNamespace(
            max_concurrent=1,
            retries=retries,
            agent=SimpleNamespace(retries=retries),
        ),
    )
    verify_direct_collection_config(config, planned_slot_count=2)
    config.resume = "latest"
    with pytest.raises(ValueError, match="not exact"):
        verify_direct_collection_config(config, planned_slot_count=2)


def test_exact_collection_materializes_and_persists_roster_before_calls(
    tmp_path: Path,
) -> None:
    plan, episodes, sources = _plan_and_episodes()
    tasks = [episode.traces[0].task for episode in episodes]
    retries = SimpleNamespace(max_retries=0)
    env = SimpleNamespace(
        master_seed="master",
        max_concurrent=1,
        retries=retries,
        agent=SimpleNamespace(retries=retries),
        taskset=SimpleNamespace(select=lambda count, shuffle: tasks[:count]),
    )
    config = SimpleNamespace(
        server=False,
        num_rollouts=1,
        num_tasks=len(tasks),
        shuffle=False,
        resume=None,
        push=False,
        rich=False,
        max_concurrent=1,
        output_dir=tmp_path / "run",
        env=env,
    )
    events: list[str] = []
    persisted: list[bytes] = []

    async def run_eval(loaded_env, loaded_config):
        assert loaded_env is env
        assert loaded_config is config
        assert persisted == [plan.to_bytes()]
        events.append("called")
        return episodes

    actual_plan, actual_episodes, receipt = asyncio.run(
        run_exact_source_collection(
            config,
            preregistered_plan_sha256=plan.plan_sha256,
            run_eval=run_eval,
            load_environment=lambda _config: env,
            load_verified_sources=lambda: sources,
            persist_plan=persisted.append,
        )
    )
    assert actual_plan == plan
    assert actual_episodes == tuple(episodes)
    assert events == ["called"]
    assert (
        b'"terminal_slot_count":' + str(len(plan.slots)).encode("ascii")
        in receipt
    )


def test_exact_collection_rejects_plan_mismatch_before_calls(tmp_path: Path) -> None:
    _plan, episodes, sources = _plan_and_episodes()
    tasks = [episode.traces[0].task for episode in episodes]
    retries = SimpleNamespace(max_retries=0)
    env = SimpleNamespace(
        master_seed="master",
        max_concurrent=1,
        retries=retries,
        agent=SimpleNamespace(retries=retries),
        taskset=SimpleNamespace(select=lambda count, shuffle: tasks[:count]),
    )
    config = SimpleNamespace(
        server=False,
        num_rollouts=1,
        num_tasks=len(tasks),
        shuffle=False,
        resume=None,
        push=False,
        rich=False,
        max_concurrent=1,
        output_dir=tmp_path / "run",
        env=env,
    )
    calls: list[str] = []

    async def run_eval(_env, _config):
        calls.append("called")
        return episodes

    with pytest.raises(ValueError, match="differs from preregistration"):
        asyncio.run(
            run_exact_source_collection(
                config,
                preregistered_plan_sha256="0" * 64,
                run_eval=run_eval,
                load_environment=lambda _config: env,
                load_verified_sources=lambda: sources,
                persist_plan=lambda _value: None,
            )
        )
    assert calls == []
