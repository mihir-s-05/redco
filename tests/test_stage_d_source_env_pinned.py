from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_stage_d_source_producer import (
    _action,
    _child_address,
    _episode,
    _target_id,
)
from test_stage_d_three_arm_bridge import _inputs as _bridge_inputs

pytest.importorskip("verifiers.v1")

ENV_ROOT = Path(__file__).parents[1] / "environments" / "redco_evidence_selection_v2"
sys.path.insert(0, str(ENV_ROOT))

import verifiers.v1 as vf  # noqa: E402
from redco_evidence_selection_v2 import __all__ as plugin_exports  # noqa: E402
from redco_evidence_selection_v2.source_env import (  # noqa: E402
    StageDSourceEnv,
    StageDSourceEnvConfig,
    StageDSourceTaskset,
    StageDSourceTasksetConfig,
    _episode_sampling,
    _episode_seed_and_salt,
    _resolved_agent_sampling_law_sha256,
    _resolved_train_client_sha256,
)
from redco_evidence_selection_v2.source_env import __all__ as source_exports  # noqa: E402
from verifiers.v1.task import task_data_cls  # noqa: E402

from redco.analysis.stage_d_receipt_ledger import inspect_ledger  # noqa: E402
from redco.analysis.stage_d_source_producer import (  # noqa: E402
    _CALL_FIELDS,
    StageDSourceRolloutProducer,
)
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress  # noqa: E402


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _inputs(tmp_path: Path) -> dict[str, object]:
    row = {
        "example_id": "source-env-1",
        "paper_id": "paper-1",
        "title": "Fixture",
        "question": "What is the evidence?",
        "paper": "The exact evidence is here.",
        "reference_evidence": ["exact evidence"],
        "answer_type": "extractive",
        "split": "train",
    }
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifests = tmp_path / "manifests"
    manifests.mkdir(exist_ok=True)
    values = {
        "base_model": b'{"model":"fixture"}',
        "tokenizer": b'{"eos_token_id":2}',
        "renderer": b'{"renderer":"fixture"}',
        "sampler": b'{"passes":true}',
    }
    paths: dict[str, Path] = {}
    for name, value in values.items():
        path = manifests / f"{name}.json"
        path.write_bytes(value)
        paths[name] = path
    return {
        "dataset": dataset,
        "dataset_sha256": _sha256(dataset.read_bytes()),
        "paths": paths,
        "values": values,
    }


def _config_payload(tmp_path: Path) -> dict[str, object]:
    inputs = _inputs(tmp_path)
    paths = inputs["paths"]
    values = inputs["values"]
    assert isinstance(paths, dict) and isinstance(values, dict)
    return {
        "id": "redco-evidence-selection-v2",
        "taskset": {
            "id": "redco-evidence-selection-v2",
            "dataset_path": str(inputs["dataset"]),
            "dataset_sha256": inputs["dataset_sha256"],
            "split": "train",
            "scientific_group_namespace": "pinned-source-env-test",
            "rollouts_per_task": 2,
        },
        "agent": {"model": "fixture-model", "retries": {"max_retries": 0}},
        "retries": {"max_retries": 0},
        "max_concurrent": 1,
        "ledger_path": str(tmp_path / "ledger"),
        "artifact_path": str(tmp_path / "artifacts"),
        "master_seed": "fixture-master",
        "preregistration_sha256": "1" * 64,
        "source_sha256": "2" * 64,
        "runtime_sha256": "3" * 64,
        "config_sha256": "4" * 64,
        "protocol_manifest_sha256": "7" * 64,
        "support_rules_sha256": "8" * 64,
        "checkpoint_id": "fixture-model",
        "base_model_manifest_path": str(paths["base_model"]),
        "base_model_manifest_sha256": _sha256(values["base_model"]),
        "tokenizer_manifest_path": str(paths["tokenizer"]),
        "tokenizer_manifest_sha256": _sha256(values["tokenizer"]),
        "renderer_manifest_path": str(paths["renderer"]),
        "renderer_manifest_sha256": _sha256(values["renderer"]),
        "sampler_conformance_manifest_path": str(paths["sampler"]),
        "sampler_conformance_manifest_sha256": _sha256(values["sampler"]),
        "resolved_agent_sampling_law_sha256": "5" * 64,
        "resolved_train_client_sha256": "6" * 64,
        "branch_count": 4,
        "continuation_replicates": 1,
        "failure_reward": -1.0,
    }


def test_plugin_exports_exactly_one_taskset_and_one_env() -> None:
    assert plugin_exports == ["EvidenceSelectionTaskset", "StageDSourceEnv"]
    assert source_exports == ["StageDSourceEnv", "StageDSourceTaskset"]
    assert task_data_cls(StageDSourceTaskset.task_type()).__name__ == "StageDSourceData"
    parameters = tuple(inspect.signature(vf.Env.prepared_call_observer).parameters)
    if parameters != ("self", "task", "trace", "agent_config", "client"):
        pytest.skip("requires the pinned prepared-observer Verifiers patch")


def test_pinned_loader_resolves_exact_source_env_profile(tmp_path: Path) -> None:
    from verifiers.v1.loaders import (
        env_config_type,
        load_environment,
        resolve_env_config,
    )

    payload = _config_payload(tmp_path)
    assert (
        env_config_type(
            "redco-evidence-selection-v2",
            "redco-evidence-selection-v2",
        )
        is StageDSourceEnvConfig
    )
    config = resolve_env_config(payload)
    assert type(config) is StageDSourceEnvConfig
    assert type(load_environment(config)) is StageDSourceEnv


def test_isolated_source_profile_binds_closed_docker_and_task_bytes(
    tmp_path: Path,
) -> None:
    image = "python@sha256:" + "a" * 64
    manifest = tmp_path / "workspace-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "path": "/workspace/evidence_context.txt",
                        "mode": "0444",
                    }
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    payload = _config_payload(tmp_path)
    taskset = payload["taskset"]
    agent = payload["agent"]
    assert isinstance(taskset, dict) and isinstance(agent, dict)
    taskset["isolated_runtime_image"] = image
    agent["harness"] = {
        "id": "rlm",
        "version": "fixture",
        "runtime": {
            "type": "docker",
            "image": image,
            "workdir": "/workspace",
            "allow": [],
            "block": [],
            "gpu": None,
            "execution_user": "65534:65534",
            "execution_home": "/tmp/redco-agent",
        },
    }
    payload["frozen_workspace_manifest_path"] = str(manifest)
    payload["frozen_workspace_manifest_sha256"] = _sha256(manifest.read_bytes())
    config = StageDSourceEnvConfig.model_validate(payload)
    env = StageDSourceEnv(config)
    task = env.taskset.load()[0]
    assert task.data.image == image
    assert task.data.network_allow == []
    snapshot = json.loads(env._runtime_snapshot(task, config.agent))
    assert snapshot["domain"] == "redco-stage-d-pre-action-runtime-snapshot-v1"
    assert snapshot["network"]["agent_egress"] is False
    assert snapshot["paper"]["sha256"] == _sha256(task.data.paper.encode())


def test_pinned_successful_model_call_dump_matches_source_schema() -> None:
    call = vf.ModelCall(
        node=1,
        model="fixture-model",
        sampling=vf.Sampling(
            temperature=0.7,
            top_p=1.0,
            reasoning_effort=None,
            max_tokens=2,
            parallel_tool_calls=False,
            seed=71,
        ),
        endpoint="/chat/completions",
        finish_reason="stop",
        usage=vf.Usage(prompt_tokens=2, completion_tokens=2),
    ).model_dump(mode="json")

    assert set(call) == _CALL_FIELDS
    assert call["error"] is None
    assert call["sampling"] == {
        "temperature": 0.7,
        "top_p": 1.0,
        "reasoning_effort": None,
        "max_tokens": 2,
        "parallel_tool_calls": False,
        "seed": 71,
    }
    assert call["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "cached_input_tokens": None,
        "reasoning_tokens": None,
        "cost": None,
    }


def test_scientific_group_id_is_stable_and_namespace_bound(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    def load(namespace: str) -> str:
        task = StageDSourceTaskset(
            StageDSourceTasksetConfig(
                dataset_path=inputs["dataset"],
                dataset_sha256=inputs["dataset_sha256"],
                split="train",
                scientific_group_namespace=namespace,
                rollouts_per_task=2,
            )
        ).load()[0]
        return task.data.scientific_group_id

    assert load("campaign-a") == load("campaign-a")
    assert load("campaign-a") != load("campaign-b")


def test_support_profile_can_use_one_unique_rollout_per_paper(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    tasks = StageDSourceTaskset(
        StageDSourceTasksetConfig(
            dataset_path=inputs["dataset"],
            dataset_sha256=inputs["dataset_sha256"],
            split="train",
            scientific_group_namespace="support-v1",
            rollouts_per_task=1,
        )
    ).load()
    assert len(tasks) == 1
    assert tasks[0].data.rollout_slot == 0


def test_source_config_rejects_retry_or_parallel_collection(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path)
    StageDSourceEnvConfig.model_validate(payload)
    for mutation in (
        {"retries": {"max_retries": 1}},
        {"agent": {"model": "fixture-model", "retries": {"max_retries": 1}}},
        {"max_concurrent": 2},
    ):
        candidate = {**payload, **mutation}
        with pytest.raises(ValueError):
            StageDSourceEnvConfig.model_validate(candidate)


def test_second_source_worker_cannot_open_the_campaign_ledger(tmp_path: Path) -> None:
    first = StageDSourceEnv(StageDSourceEnvConfig.model_validate(_config_payload(tmp_path)))
    second = StageDSourceEnv(StageDSourceEnvConfig.model_validate(_config_payload(tmp_path)))

    async def scenario() -> None:
        await first.start()
        try:
            with pytest.raises(RuntimeError, match="one static worker"):
                await second.start()
        finally:
            await first.stop()

    asyncio.run(scenario())


def test_startup_manifest_failure_releases_ledger_and_worker_lease(
    tmp_path: Path,
) -> None:
    bad_payload = _config_payload(tmp_path)
    bad_payload["base_model_manifest_sha256"] = "0" * 64
    first = StageDSourceEnv(StageDSourceEnvConfig.model_validate(bad_payload))
    second = StageDSourceEnv(StageDSourceEnvConfig.model_validate(_config_payload(tmp_path)))

    async def scenario() -> None:
        with pytest.raises(ValueError, match="base model manifest"):
            await first.start()
        await second.start()
        await second.stop()

    asyncio.run(scenario())


def test_source_collection_restart_fails_before_calls_after_any_prior_attempt(
    tmp_path: Path,
) -> None:
    payload = _config_payload(tmp_path)
    first = StageDSourceEnv(StageDSourceEnvConfig.model_validate(payload))

    async def scenario() -> None:
        await first.start()
        assert first._ledger is not None
        producer = StageDSourceRolloutProducer(
            ledger=first._ledger,
            group_id="prior-group",
            rollout_id="prior-rollout",
            child_target_roster=(_target_id("prior-rollout"),),
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
            target_id=_target_id("prior-rollout"),
            branch_selected=False,
            forward_once=lambda _key: child_action,
        )
        producer.finalize_episode(
            _episode(trace_id="prior-rollout"),
        )
        await first.stop()
        second = StageDSourceEnv(StageDSourceEnvConfig.model_validate(payload))
        with pytest.raises(RuntimeError, match="cannot restart"):
            await second.start()

    asyncio.run(scenario())


@pytest.mark.parametrize("subdirectory", ["pending", "sources"])
def test_source_collection_rejects_stale_artifacts_before_calls(
    tmp_path: Path,
    subdirectory: str,
) -> None:
    payload = _config_payload(tmp_path)
    artifact_path = Path(payload["artifact_path"])
    stale_directory = artifact_path / subdirectory
    stale_directory.mkdir(parents=True)
    (stale_directory / "stale.json").write_text("{}", encoding="utf-8")
    env = StageDSourceEnv(StageDSourceEnvConfig.model_validate(payload))

    async def scenario() -> None:
        with pytest.raises(
            RuntimeError,
            match=r"stale (pending payloads|completed sources)",
        ):
            await env.start()
        assert env._ledger is None
        assert env._identity is None

    asyncio.run(scenario())


def test_verified_completed_sources_requires_exact_terminal_registry(
    tmp_path: Path,
) -> None:
    env = StageDSourceEnv(StageDSourceEnvConfig.model_validate(_config_payload(tmp_path)))
    with pytest.raises(RuntimeError, match="not terminally complete"):
        env.verified_completed_sources()
    sources, _artifacts = _bridge_inputs()
    source = sources[0]
    env._producers[source.rollout_id] = SimpleNamespace()
    env._terminal_traces.add(source.rollout_id)
    env._completed_sources[source.rollout_id] = source
    assert env.verified_completed_sources() == (source,)


def test_cancellation_after_observation_invokes_terminal_finalization_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = StageDSourceEnv(StageDSourceEnvConfig.model_validate(_config_payload(tmp_path)))
    entered = asyncio.Event()
    observed: list[BaseException] = []

    class Producer:
        def abort_finalization(self, error: BaseException) -> None:
            observed.append(error)

    async def cancelled_super(self, task, ctx, **kwargs):
        del task, ctx, kwargs
        self._producers["cancelled-trace"] = Producer()
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(vf.Env, "run_episode", cancelled_super)

    async def scenario() -> None:
        source_task = env.taskset.load()[0]
        ctx = SimpleNamespace(model="fixture", client=SimpleNamespace(), sampling=vf.Sampling())
        task = asyncio.create_task(env.run_episode(source_task, ctx))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert len(observed) == 1
    assert isinstance(observed[0], asyncio.CancelledError)


def test_fresh_process_env_server_starts_and_closes_one_unsealed_ledger(
    tmp_path: Path,
) -> None:
    parameters = tuple(inspect.signature(vf.Env.prepared_call_observer).parameters)
    if parameters != ("self", "task", "trace", "agent_config", "client"):
        pytest.skip("requires the pinned prepared-observer Verifiers patch")
    payload_path = tmp_path / "config.json"
    payload_path.write_text(json.dumps(_config_payload(tmp_path)), encoding="utf-8")
    code = """
import asyncio,json,sys
from redco_evidence_selection_v2.source_env import StageDSourceEnvConfig
from verifiers.v1.serve.server import EnvServer

async def main():
    with open(sys.argv[1], encoding='utf-8') as handle:
        config = StageDSourceEnvConfig.model_validate(json.load(handle))
    server = EnvServer(config, address='tcp://127.0.0.1:0')
    task = asyncio.create_task(server.run())
    await asyncio.sleep(0.2)
    task.cancel()
    await task

asyncio.run(main())
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(payload_path)],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (str(ENV_ROOT), os.environ.get("PYTHONPATH", ""))
            ),
        },
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert inspect_ledger(tmp_path / "ledger").status == "active-clean"


def test_source_observer_binds_actual_resolved_train_client_once(
    tmp_path: Path,
) -> None:
    from verifiers.v1.clients.train import TrainClient

    parameters = tuple(inspect.signature(vf.Env.prepared_call_observer).parameters)
    if parameters != ("self", "task", "trace", "agent_config", "client"):
        pytest.skip("requires the pinned prepared-observer Verifiers patch")
    resolved_agent = vf.AgentConfig(
        model="fixture-model",
        retries={"max_retries": 0},
        sampling=vf.Sampling(),
    )
    client = TrainClient(
        SimpleNamespace(max_retries=0, base_url="http://127.0.0.1:8000/v1"),
        default_headers={"X-Stage-D-Route": "route-a"},
        api_key_var="STAGE_D_API_KEY",
    )
    payload = _config_payload(tmp_path)
    episode_sampling = _episode_sampling(
        resolved_agent.sampling,
        master_seed=str(payload["master_seed"]),
        scientific_group_id=env_group_id(payload),
        rollout_slot=0,
    )
    resolved_agent = resolved_agent.model_copy(update={"sampling": episode_sampling})
    payload["resolved_agent_sampling_law_sha256"] = _resolved_agent_sampling_law_sha256(
        episode_sampling
    )
    payload["resolved_train_client_sha256"] = _resolved_train_client_sha256(client)
    env = StageDSourceEnv(StageDSourceEnvConfig.model_validate(payload))

    async def scenario() -> None:
        await env.start()
        task = env.taskset.load()[0]
        trace = vf.Trace(task=vf.TraceTask(type=type(task).__name__, data=task.data))
        observer = env.prepared_call_observer(task, trace, resolved_agent, client)
        assert observer is not None
        with pytest.raises(ValueError, match="more than once"):
            env.prepared_call_observer(task, trace, resolved_agent, client)
        await env.stop()

    import asyncio

    asyncio.run(scenario())


def env_group_id(payload: dict[str, object]) -> str:
    config = StageDSourceEnvConfig.model_validate(payload)
    return StageDSourceTaskset(config.taskset).load()[0].data.scientific_group_id


def test_episode_addressed_sampling_is_distinct_and_reproducible() -> None:
    base = vf.Sampling(
        temperature=0.7,
        top_p=1.0,
        max_tokens=64,
        top_k=None,
        min_p=0.0,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logit_bias={},
        n=1,
        best_of=None,
        use_beam_search=False,
        logprobs=True,
        top_logprobs=0,
        ignore_eos=False,
        min_tokens=0,
        tool_choice="auto",
        parallel_tool_calls=False,
    )
    first = _episode_sampling(
        base,
        master_seed="master",
        scientific_group_id="same-task",
        rollout_slot=0,
    )
    replay = _episode_sampling(
        base,
        master_seed="master",
        scientific_group_id="same-task",
        rollout_slot=0,
    )
    second = _episode_sampling(
        base,
        master_seed="master",
        scientific_group_id="same-task",
        rollout_slot=1,
    )

    assert first == replay
    assert first.seed != second.seed
    assert first.extra_body != second.extra_body
    assert _resolved_agent_sampling_law_sha256(first) == (
        _resolved_agent_sampling_law_sha256(second)
    )
    assert (first.seed, first.extra_body["cache_salt"]) == _episode_seed_and_salt(
        master_seed="master",
        scientific_group_id="same-task",
        rollout_slot=0,
    )


def test_resolved_train_client_hash_binds_nonsecret_routing_identity() -> None:
    from verifiers.v1.clients.train import TrainClient

    def identity(route: str, api_key_var: str) -> str:
        return _resolved_train_client_sha256(
            TrainClient(
                SimpleNamespace(
                    max_retries=0,
                    base_url="http://127.0.0.1:8000/v1",
                ),
                default_headers={"X-Stage-D-Route": route},
                api_key_var=api_key_var,
            )
        )

    baseline = identity("route-a", "STAGE_D_API_KEY")
    assert identity("route-b", "STAGE_D_API_KEY") != baseline
    assert identity("route-a", "OTHER_STAGE_D_API_KEY") != baseline
