from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from test_stage_d_receipt_ledger import _commit as _ledger_commit
from test_stage_d_receipt_ledger import _create as _ledger_create
from test_stage_d_scientific_branch_group import _action
from test_stage_d_source_producer import (
    _child_address,
    _episode,
    _target_id,
)
from test_stage_d_three_arm_bridge import _inputs as _bridge_inputs

pytest.importorskip("verifiers.v1")

import redco.analysis.stage_d_receipt_ledger as receipt_ledger_module

ENV_ROOT = Path(__file__).parents[1] / "environments" / "redco_evidence_selection_v2"
sys.path.insert(0, str(ENV_ROOT))

import verifiers.v1 as vf  # noqa: E402
from redco_evidence_selection_v2 import __all__ as plugin_exports  # noqa: E402
from redco_evidence_selection_v2.source_env import (  # noqa: E402
    StageDSourceEnv,
    StageDSourceEnvConfig,
    StageDSourceTaskset,
    StageDSourceTasksetConfig,
    _canonical_source_episode,
    _episode_sampling,
    _episode_seed_and_salt,
    _resolved_agent_sampling_law_sha256,
    _resolved_train_client_sha256,
)
from redco_evidence_selection_v2.source_env import __all__ as source_exports  # noqa: E402
from verifiers.v1.task import task_data_cls  # noqa: E402

from redco.analysis.stage_d_action_closure import (  # noqa: E402
    ActionClosureWatchdog,
    WatchdogDeadlines,
)
from redco.analysis.stage_d_exact_action import (  # noqa: E402
    BehaviorAction,
    ExactActionKey,
)
from redco.analysis.stage_d_receipt_ledger import (  # noqa: E402
    LedgerError,
    LedgerPoisoned,
    StageDReceiptLedger,
    inspect_ledger,
)
from redco.analysis.stage_d_source_producer import (  # noqa: E402
    _CALL_FIELDS,
    _NODE_FIELDS,
    StageDSourceRolloutProducer,
)
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress  # noqa: E402
from redco.contracts import ActualEvaluationCost, canonical_json  # noqa: E402


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tree_digest(root: Path) -> str:
    if not root.exists():
        return _sha256(b"missing")
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries.append(
                {"kind": "symlink", "path": relative, "target": os.readlink(path)}
            )
        elif path.is_dir():
            entries.append({"kind": "directory", "path": relative})
        elif path.is_file():
            content = path.read_bytes()
            entries.append(
                {
                    "kind": "file",
                    "path": relative,
                    "sha256": _sha256(content),
                    "size": len(content),
                }
            )
        else:
            raise AssertionError(f"unsupported checkout output entry: {path}")
    return _sha256(canonical_json(entries))


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
        "agent": {
            "model": "fixture-model",
            "max_turns": 8,
            "retries": {"max_retries": 0},
        },
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
            min_p=0.0,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=71,
            max_tokens=2,
            n=1,
            tool_choice="auto",
            parallel_tool_calls=False,
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
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "seed": 71,
        "max_tokens": 2,
        "n": 1,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    }
    assert call["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "cached_input_tokens": None,
        "reasoning_tokens": None,
        "cost": None,
    }


def test_typed_episode_serializes_only_node_nulls_like_persisted_trace() -> None:
    episode = vf.WireEpisode.model_validate(json.loads(_episode()))
    full_payload = episode.model_dump(mode="json")
    full_node = full_payload["traces"][0]["nodes"][0]
    assert full_node["multi_modal_data"] is None
    assert full_node["routed_experts"] is None
    assert full_node["kept_tokens"] is None

    payload = json.loads(_canonical_source_episode(episode))
    trace = payload["traces"][0]
    assert all(
        set(node) == _NODE_FIELDS or set(node) == _NODE_FIELDS - {"parent"}
        for node in trace["nodes"]
    )
    assert set(trace["calls"][0]) == _CALL_FIELDS
    assert trace["calls"][0]["error"] is None
    assert set(trace["calls"][0]["sampling"]) == {
        "temperature",
        "top_p",
        "reasoning_effort",
        "min_p",
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "max_tokens",
        "n",
        "tool_choice",
        "parallel_tool_calls",
    }
    assert trace["calls"][0]["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "cached_input_tokens": None,
        "reasoning_tokens": None,
        "cost": None,
    }


def test_typed_episode_runs_through_real_source_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = StageDSourceEnv(StageDSourceEnvConfig.model_validate(_config_payload(tmp_path)))

    async def completed_super(
        self: Any,
        task: Any,
        ctx: Any,
        **kwargs: Any,
    ) -> vf.Episode:
        del task, ctx, kwargs
        episode = vf.WireEpisode.model_validate(json.loads(_episode()))
        trace = episode.traces[0]
        assert self._ledger is not None
        producer = StageDSourceRolloutProducer(
            ledger=self._ledger,
            group_id="group-1",
            rollout_id=trace.id,
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
        self._producers[trace.id] = producer
        return episode

    monkeypatch.setattr(vf.Env, "run_episode", completed_super)

    async def scenario() -> None:
        await env.start()
        try:
            task = env.taskset.load()[0]
            ctx = vf.ModelContext(
                model="fixture-model",
                client=SimpleNamespace(),
                sampling=vf.Sampling(),
            )
            episode = await env.run_episode(task, ctx)
            assert episode.ok
            (source,) = env.verified_completed_sources()
            raw_episode = (
                Path(env.config.ledger_path) / "evidence" / source.trace_sha256
            ).read_bytes()
            payload = json.loads(raw_episode)
            trace = payload["traces"][0]
            assert all(
                set(node) == _NODE_FIELDS or set(node) == _NODE_FIELDS - {"parent"}
                for node in trace["nodes"]
            )
            assert trace["calls"][0]["error"] is None
            assert trace["calls"][0]["sampling"]["reasoning_effort"] is None
            assert trace["calls"][0]["usage"]["cost"] is None
        finally:
            await env.stop()

    asyncio.run(scenario())


def test_bound_scientific_qa_missing_roster_fails_before_run_eval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public QA owner rejects an unfrozen roster before invoking Verifiers."""
    import test_stage_d_receipt_ledger as ledger_tests
    from redco_evidence_selection_v2.scientific_env import run_bound_scientific_episode
    from verifiers.v1.cli.eval import runner
    from verifiers.v1.clients.config import TrainClientConfig
    from verifiers.v1.configs.eval import EvalConfig

    from redco.analysis.stage_d_receipt_ledger import LedgerError

    config_root = tmp_path / "config"
    config_root.mkdir()
    env_config = StageDSourceEnvConfig.model_validate(_config_payload(config_root))
    writer = ledger_tests._create(tmp_path / "missing-roster")
    recorded, _, _, _ = ledger_tests._commit(writer, freeze_roster=False)
    binding = SimpleNamespace(
        mode="qa",
        source=SimpleNamespace(group_id="group-1"),
        target_id="target-0",
        recorded_action=recorded,
        ledger=writer,
    )
    eval_config = EvalConfig(
        env=env_config,
        model="fixture-model",
        client=TrainClientConfig(base_url="http://unused.invalid/v1"),
        sampling=vf.Sampling(),
        num_tasks=1,
        num_rollouts=1,
        max_concurrent=1,
        output_dir=tmp_path / "missing-roster-output",
        push=False,
        rich=False,
    )

    async def fail_run_eval(*_args: object, **_kwargs: object) -> list[object]:
        raise AssertionError("run_eval was reached before roster readiness")

    monkeypatch.setattr(runner, "run_eval", fail_run_eval)
    before = writer.record_count
    with pytest.raises(LedgerError, match="frozen target roster"):
        asyncio.run(
            run_bound_scientific_episode(
                binding=binding,
                env_config=env_config,
                eval_config=eval_config,
                watchdog=ActionClosureWatchdog(),
            )
        )
    assert writer.record_count == before
    writer.close()


def test_scientific_finalizer_timeout_is_bounded_and_side_effect_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout during the real fsync path rolls back the isolated commit."""

    writer = _ledger_create(tmp_path / "ledger")
    recorded, _, _, _ = _ledger_commit(writer)
    before_record_count = writer.record_count
    child_processes: list[asyncio.subprocess.Process] = []
    real_create_subprocess_exec = asyncio.create_subprocess_exec

    async def capture_transaction_child(
        *args: Any, **kwargs: Any
    ) -> asyncio.subprocess.Process:
        process = await real_create_subprocess_exec(*args, **kwargs)
        if "--finalize-transaction-stdin" in args:
            child_processes.append(process)
        return process

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        capture_transaction_child,
    )

    async def scenario() -> None:
        terminal_events: list[tuple[str, str]] = []

        def record_terminal(phase: str, disposition: str) -> None:
            terminal_events.append((phase, disposition))

        watchdog = ActionClosureWatchdog(
            deadlines=WatchdogDeadlines(
                provider_call=1.0,
                episode=1.0,
                concurrent_children=1.0,
                finalizer=3.0,
                campaign=1.0,
                pod_lifetime=1.0,
            ),
            terminal_callback=record_terminal,
        )

        async def suspend_after_commit_manifest() -> int:
            started = time.monotonic()
            while not child_processes:
                await asyncio.sleep(0.01)
                assert time.monotonic() - started < 5.0
            process = child_processes[0]
            manifest_paths = list((tmp_path / "ledger").glob(".*.finalization.json"))
            while process.returncode is None and not manifest_paths:
                await asyncio.sleep(0.01)
                assert time.monotonic() - started < 5.0
                manifest_paths = list((tmp_path / "ledger").glob(".*.finalization.json"))
            if process.returncode is not None:
                raise AssertionError("transaction child exited before commit interruption")
            child_pid = process.pid
            assert child_pid is not None
            if os.name == "nt":
                process.terminate()
            else:
                os.kill(child_pid, signal.SIGSTOP)
            return child_pid

        suspender = asyncio.create_task(
            suspend_after_commit_manifest(),
            name="stage-d-commit-interrupter",
        )
        finalizer_task = asyncio.create_task(
            watchdog.run_finalizer(
                writer.record_reconstruction_qa_transaction(
                    group_id="group-1",
                    target_id="target-0",
                    recorded_action=recorded,
                    passed=True,
                    report=b"commit-path-report",
                    actual_cost=ActualEvaluationCost(),
                )
            ),
            name="stage-d-finalizer",
        )
        child_pid = await suspender
        timeout_started = time.monotonic()
        if os.name == "nt":
            with pytest.raises(LedgerError):
                await finalizer_task
        else:
            with pytest.raises(TimeoutError):
                await finalizer_task
        elapsed = time.monotonic() - timeout_started
        assert elapsed < 4.5
        assert finalizer_task.done()
        assert watchdog.closed
        assert watchdog.terminal
        assert terminal_events == [("finalizer", "timeout")]
        assert inspect_ledger(tmp_path / "ledger").status == "active-clean"
        assert writer.record_count == before_record_count
        assert writer.reconstruction_qa_receipt("group-1", "target-0") is None
        assert not (tmp_path / "ledger" / "evidence" / _sha256(b"commit-path-report")).exists()
        assert not list((tmp_path / "ledger" / "records").glob(".*.tmp"))
        assert not list((tmp_path / "ledger" / "evidence").glob(".*.tmp"))
        assert not list(tmp_path.glob(".*.stage-d-finalization-copy"))
        if os.name == "nt":
            process_listing = subprocess.run(
                ["tasklist", "/FI", f"PID eq {child_pid}"],
                capture_output=True,
                text=True,
                check=False,
            )
            assert str(child_pid) not in process_listing.stdout
        else:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError("isolated finalization child survived timeout")
        assert not any(
            task.get_name() == "stage-d-finalizer"
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
        )

        success_watchdog = ActionClosureWatchdog(
            deadlines=WatchdogDeadlines(
                provider_call=1.0,
                episode=1.0,
                concurrent_children=1.0,
                finalizer=1.0,
                campaign=1.0,
                pod_lifetime=1.0,
            )
        )
        assert (
            await success_watchdog.run_finalizer(asyncio.sleep(0, result=b"receipt"))
            == b"receipt"
        )

        async def failing_finalizer() -> bytes:
            raise ValueError("primary-finalizer-error")

        error_watchdog = ActionClosureWatchdog(
            deadlines=WatchdogDeadlines(
                provider_call=1.0,
                episode=1.0,
                concurrent_children=1.0,
                finalizer=1.0,
                campaign=1.0,
                pod_lifetime=1.0,
            )
        )
        with pytest.raises(ValueError, match="primary-finalizer-error"):
            await error_watchdog.run_finalizer(failing_finalizer())
        assert error_watchdog.closed
        assert error_watchdog.terminal

    try:
        asyncio.run(scenario())
    finally:
        writer.close()


def test_finalization_manifest_is_strict_and_baseline_bound(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    writer = _ledger_create(tmp_path / "ledger")
    recorded, _, _, _ = _ledger_commit(writer)
    with writer._state_lock:
        spec = writer._new_finalization_spec(
            operation="qa",
            group_id="group-1",
            target_id="target-0",
            evidence=b"manifest-report",
            recorded_action=recorded,
            passed=True,
            actual_cost=ActualEvaluationCost(),
            execution_attempt=None,
            outcome_kind=None,
        )
    try:
        receipt_ledger_module._run_finalization_transaction_worker(spec)
        capsys.readouterr()
        manifest_path = receipt_ledger_module._finalization_manifest_path(spec)
        original = json.loads(manifest_path.read_bytes())

        def rejected(mutator: Any) -> None:
            manifest_path.write_bytes(canonical_json(mutator(json.loads(manifest_path.read_bytes()))))
            try:
                with pytest.raises(LedgerPoisoned):
                    receipt_ledger_module._resolve_stale_finalization_manifests(spec.root)
            finally:
                manifest_path.write_bytes(canonical_json(original))

        rejected(lambda value: {**value, "unknown": None})

        def traversal(value: dict[str, Any]) -> dict[str, Any]:
            result = dict(value["result"])
            patches = [dict(item) for item in result["record_patches"]]
            patches[0]["relative_path"] = "records/../victim"
            result["record_patches"] = patches
            result.pop("result_sha256", None)
            result["result_sha256"] = _sha256(canonical_json(result))
            return {
                **value,
                "result": result,
                "result_sha256": result["result_sha256"],
            }

        rejected(traversal)

        def changed_baseline(value: dict[str, Any]) -> dict[str, Any]:
            baseline = [dict(item) for item in value["baseline_paths"]]
            baseline[0]["sha256"] = "0" * 64
            return {
                **value,
                "baseline_paths": baseline,
                "baseline_sha256": _sha256(canonical_json(baseline)),
            }

        rejected(changed_baseline)

        def swapped_root(value: dict[str, Any]) -> dict[str, Any]:
            return {**value, "root_path": str(tmp_path / "other")}

        rejected(swapped_root)

        wrong_name = spec.root / ".00000000000000000000000000000000.finalization.json"
        wrong_name.write_bytes(canonical_json(original))
        try:
            with pytest.raises(LedgerPoisoned):
                receipt_ledger_module._resolve_stale_finalization_manifests(spec.root)
        finally:
            wrong_name.unlink()

        record_patch = original["result"]["record_patches"][0]
        preexisting_temp = spec.root / record_patch["temporary_relative_path"]
        preexisting_temp.write_bytes(b"pre-existing-victim")
        committing = {**original, "state": "committing"}
        committing_bytes = canonical_json(committing)
        manifest_path.write_bytes(committing_bytes)
        try:
            with pytest.raises(LedgerPoisoned):
                receipt_ledger_module._resolve_stale_finalization_manifests(spec.root)
            assert preexisting_temp.read_bytes() == b"pre-existing-victim"
            assert manifest_path.read_bytes() == committing_bytes
        finally:
            preexisting_temp.unlink()
            manifest_path.write_bytes(canonical_json(original))

        baseline_source = spec.root / "records" / receipt_ledger_module._record_name(0)
        aliased_temp = spec.root / record_patch["temporary_relative_path"]
        os.link(baseline_source, aliased_temp)
        manifest_path.write_bytes(committing_bytes)
        try:
            with pytest.raises(LedgerPoisoned):
                receipt_ledger_module._resolve_stale_finalization_manifests(spec.root)
            assert baseline_source.is_file()
            assert aliased_temp.is_file()
        finally:
            aliased_temp.unlink()
            manifest_path.write_bytes(canonical_json(original))

        def forged_request(value: dict[str, Any]) -> dict[str, Any]:
            request = json.loads(bytes.fromhex(value["request_hex"]))
            request["master_seed"] = "not-the-genesis-seed"
            request["master_seed_sha256"] = _sha256(request["master_seed"].encode())
            request_bytes = canonical_json(request)
            result = dict(value["result"])
            result["request_sha256"] = _sha256(request_bytes)
            result.pop("result_sha256", None)
            result["result_sha256"] = _sha256(canonical_json(result))
            return {
                **value,
                "request_hex": request_bytes.hex(),
                "request_sha256": _sha256(request_bytes),
                "result": result,
                "result_sha256": result["result_sha256"],
            }

        def forged_receipt(value: dict[str, Any]) -> dict[str, Any]:
            result = dict(value["result"])
            result["receipt_hex"] = canonical_json(
                {
                    "schema_version": 1,
                    "receipt_kind": "reconstruction_qa",
                    "forged": True,
                }
            ).hex()
            result.pop("result_sha256", None)
            result["result_sha256"] = _sha256(canonical_json(result))
            return {
                **value,
                "result": result,
                "result_sha256": result["result_sha256"],
            }

        rejected(forged_receipt)
        rejected(forged_request)
    finally:
        writer._finalization_active = False
        writer.close()


def test_finalization_refuses_preexisting_transaction_temporary(tmp_path: Path) -> None:
    writer = _ledger_create(tmp_path / "ledger")
    recorded, _, _, _ = _ledger_commit(writer)
    with writer._state_lock:
        spec = writer._new_finalization_spec(
            operation="qa",
            group_id="group-1",
            target_id="target-0",
            evidence=b"preexisting-temp",
            recorded_action=recorded,
            passed=True,
            actual_cost=ActualEvaluationCost(),
            execution_attempt=None,
            outcome_kind=None,
        )
    victim = (
        spec.root
        / "records"
        / f".{spec.transaction_id}.{receipt_ledger_module._record_name(spec.base_record_count)}.tmp"
    )
    victim.write_bytes(b"keep-me")
    try:
        with pytest.raises(LedgerPoisoned):
            receipt_ledger_module._run_finalization_transaction_worker(spec)
        assert victim.read_bytes() == b"keep-me"
        assert not receipt_ledger_module._finalization_manifest_path(spec).exists()
    finally:
        writer._finalization_active = False
        writer.close()


def test_finalization_ack_failure_poison_closes_until_authenticated_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _ledger_create(tmp_path / "ledger")
    recorded, _, _, _ = _ledger_commit(writer)
    with writer._state_lock:
        spec = writer._new_finalization_spec(
            operation="qa",
            group_id="group-1",
            target_id="target-0",
            evidence=b"ack-failure-report",
            recorded_action=recorded,
            passed=True,
            actual_cost=ActualEvaluationCost(),
            execution_attempt=None,
            outcome_kind=None,
        )
    real_cleanup = receipt_ledger_module._run_finalization_cleanup_owner

    async def fail_ack(
        cleanup_spec: Any,
        *,
        action: Literal["resolve", "ack"],
    ) -> dict[str, Any]:
        if action == "ack":
            raise LedgerPoisoned("forced post-commit acknowledgement failure")
        return await real_cleanup(cleanup_spec, action=action)

    monkeypatch.setattr(receipt_ledger_module, "_run_finalization_cleanup_owner", fail_ack)
    with pytest.raises(LedgerPoisoned, match="acknowledgement"):
        asyncio.run(writer._run_finalization_transaction(spec))
    assert writer._closed is True
    assert writer._poisoned is True
    assert writer._finalization_active is False
    assert writer._lock_descriptor is None
    with pytest.raises((LedgerError, LedgerPoisoned)):
        writer.put_evidence(b"after-unresolved-commit")
    assert list((tmp_path / "ledger").glob(".*.finalization.json"))

    monkeypatch.undo()
    recovered = StageDReceiptLedger(tmp_path / "ledger", master_seed="durable-master")
    assert recovered.put_evidence(b"after-authenticated-recovery") == _sha256(
        b"after-authenticated-recovery"
    )
    recovered.close()


def test_finalization_envelopes_have_separate_cleanup_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _ledger_create(tmp_path / "ledger")
    recorded, _, _, _ = _ledger_commit(writer)
    with writer._state_lock:
        spec = writer._new_finalization_spec(
            operation="qa",
            group_id="group-1",
            target_id="target-0",
            evidence=b"cleanup-bound",
            recorded_action=recorded,
            passed=True,
            actual_cost=ActualEvaluationCost(),
            execution_attempt=None,
            outcome_kind=None,
        )
    request = receipt_ledger_module._finalization_request_bytes(spec)
    cleanup = receipt_ledger_module._cleanup_request_bytes(spec, "resolve")
    assert len(request) <= receipt_ledger_module._MAX_FINALIZATION_REQUEST_BYTES
    assert len(cleanup) <= receipt_ledger_module._MAX_FINALIZATION_CLEANUP_BYTES
    monkeypatch.setattr(
        receipt_ledger_module,
        "_MAX_FINALIZATION_CLEANUP_BYTES",
        len(cleanup) - 1,
    )
    with pytest.raises(LedgerPoisoned, match="cleanup request exceeds"):
        receipt_ledger_module._cleanup_request_bytes(spec, "resolve")
    writer._finalization_active = False
    writer.close()


def test_finalization_rejects_root_ancestor_and_forged_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_parent = tmp_path / "root-parent"
    root_parent.mkdir()
    root = root_parent / "ledger"
    real_link_check = receipt_ledger_module._is_link_or_reparse
    monkeypatch.setattr(
        receipt_ledger_module,
        "_is_link_or_reparse",
        lambda path: path == root_parent or real_link_check(path),
    )
    with pytest.raises(LedgerPoisoned, match="ancestor"):
        StageDReceiptLedger(root, master_seed="durable-master")
    monkeypatch.undo()

    action = _action(17)
    encoded = action.to_bytes()
    assert receipt_ledger_module._action_from_bytes(encoded).digest == action.digest
    envelope = json.loads(encoded)
    for mutation in (
        {**envelope, "domain": "forged-domain"},
        {**envelope, "digest": "0" * 64},
        {"schema_version": 2, "domain": envelope["domain"], "action": {}, "digest": "0" * 64},
    ):
        with pytest.raises(LedgerPoisoned):
            receipt_ledger_module._action_from_bytes(canonical_json(mutation))


def test_real_bound_scientific_qa_runs_with_zero_provider_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the real Verifiers QA owner against a local, deterministic HTTP fixture.

    The fixture is only the external model-response boundary: ``run_eval``,
    ``StageDScientificReplayEnv``, the prepared observer/controller, and the
    ledger receipt writer remain the production implementations.
    """
    checkout_outputs = Path(__file__).parents[1] / "outputs"
    checkout_outputs_before = _tree_digest(checkout_outputs)
    import httpx
    import test_stage_d_source_producer as producer_tests
    from redco_evidence_selection_v2.scientific_campaign_driver import (
        _terminal_reply,
        source_task_from_trace,
    )
    from redco_evidence_selection_v2.scientific_env import (
        StageDScientificEpisodeBinding,
        run_bound_scientific_episode,
    )
    from renderers.base import ParsedResponse, RenderedTokens
    from verifiers.v1.clients.config import TrainClientConfig
    from verifiers.v1.clients.train import TrainClient, split_engine_sampling
    from verifiers.v1.configs.eval import EvalConfig
    from verifiers.v1.harness import Harness, HarnessConfig
    from verifiers.v1.runtimes.base import ProgramResult
    from verifiers.v1.runtimes.docker import DockerConfig

    from redco.analysis.stage_d_branch_artifacts import StageDBranchTargetRoster
    from redco.analysis.stage_d_dynamic_taint import build_source_causal_graph
    from redco.analysis.stage_d_receipt_ledger import StageDReceiptLedger
    from redco.integrations.verifiers_trace_v2 import extract_v2_rlm_provenance

    observer = getattr(vf.Env, "prepared_call_observer", None)
    if observer is None or tuple(inspect.signature(observer).parameters) != (
        "self",
        "task",
        "trace",
        "agent_config",
        "client",
    ):
        pytest.skip("requires the pinned prepared-observer Verifiers overlay")

    source_trace_holder: dict[str, object] = {}
    action_holder: list[BehaviorAction] = []
    base_key = producer_tests._key(71)

    payload = _config_payload(tmp_path)
    # _config_payload returns the manifest paths only through its private helper;
    # use the same fixture bytes as the authenticated action key.
    manifests = tmp_path / "manifests"
    manifest_values = {
        "base_model": b"base",
        "adapter": b"adapter",
        "tokenizer": b"tokenizer",
        "renderer": b"renderer",
        "sampler": base_key.sampler_conformance_manifest,
    }
    manifest_paths: dict[str, Path] = {}
    for name, value in manifest_values.items():
        path = manifests / f"qa-{name}.bin"
        path.write_bytes(value)
        manifest_paths[name] = path
    payload["checkpoint_id"] = base_key.checkpoint_id
    payload["base_model_manifest_path"] = str(manifest_paths["base_model"])
    payload["base_model_manifest_sha256"] = _sha256(manifest_values["base_model"])
    payload["adapter_manifest_path"] = str(manifest_paths["adapter"])
    payload["adapter_manifest_sha256"] = _sha256(manifest_values["adapter"])
    payload["tokenizer_manifest_path"] = str(manifest_paths["tokenizer"])
    payload["tokenizer_manifest_sha256"] = _sha256(manifest_values["tokenizer"])
    payload["renderer_manifest_path"] = str(manifest_paths["renderer"])
    payload["renderer_manifest_sha256"] = _sha256(manifest_values["renderer"])
    payload["sampler_conformance_manifest_path"] = str(manifest_paths["sampler"])
    payload["sampler_conformance_manifest_sha256"] = _sha256(
        manifest_values["sampler"]
    )

    request_root = json.loads(base_key.request)
    sampling_fields = {
        key: value
        for key, value in request_root.items()
        if key
        in {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "reasoning_effort",
            "seed",
            "stop",
            "n",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "tool_choice",
            "parallel_tool_calls",
            "extra_body",
        }
    }
    base_sampling = vf.Sampling.model_validate(sampling_fields)
    payload["resolved_agent_sampling_law_sha256"] = _resolved_agent_sampling_law_sha256(
        base_sampling
    )

    class FixtureRenderer:
        supports_tools = False

        def render(
            self,
            _messages: list[dict[str, object]],
            *,
            tools: object = None,
            add_generation_prompt: bool = False,
        ) -> RenderedTokens:
            del tools
            if not add_generation_prompt:
                raise AssertionError("QA fixture renderer requires generation prompting")
            return RenderedTokens(
                token_ids=[10, 11],
                message_indices=[0, 0],
                sampled_mask=[False, False],
                is_content=[False, False],
                message_roles=["user", "user"],
            )

        def bridge_to_next_turn(
            self,
            _prompt_ids: list[int],
            _completion_ids: list[int],
            _messages: list[dict[str, object]],
            *,
            tools: object = None,
        ) -> RenderedTokens:
            del tools
            return self.render([], add_generation_prompt=True)

        def get_stop_token_ids(self) -> list[int]:
            return [2]

        def parse_response(
            self,
            completion_ids: list[int],
            *,
            tools: object = None,
        ) -> ParsedResponse:
            del tools
            if tuple(completion_ids) != (20, 2):
                raise AssertionError("QA fixture received unexpected frozen token IDs")
            return ParsedResponse(content="['exact evidence']")

    class FixtureHarness(Harness[HarnessConfig]):
        async def launch(
            self,
            _ctx: object,
            _trace: object,
            _runtime: object,
            endpoint: str,
            secret: str,
            _mcp_urls: dict[str, str],
        ) -> ProgramResult:
            trace_payload = cast(dict[str, object], source_trace_holder["trace"])
            calls = cast(list[dict[str, object]], trace_payload["calls"])
            actions = tuple(action_holder)
            async with httpx.AsyncClient(timeout=20.0) as http:
                for action, call in zip(actions, calls, strict=True):
                    rlm = call.get("rlm")
                    if not isinstance(rlm, dict):
                        raise AssertionError("fixture source call lacks RLM provenance")
                    names = {
                        "provenance_version": "X-RLM-Provenance-Version",
                        "depth": "X-RLM-Depth",
                        "session_id": "X-RLM-Session-ID",
                        "turn": "X-RLM-Turn",
                        "call_kind": "X-RLM-Call-Kind",
                        "lineage": "X-RLM-Lineage",
                        "session_call_ordinal": "X-RLM-Session-Call-Ordinal",
                        "parent_session_id": "X-RLM-Parent-Session-ID",
                        "parent_turn": "X-RLM-Parent-Turn",
                        "parent_tool_call_id": "X-RLM-Parent-Tool-Call-ID",
                        "invocation_id": "X-RLM-Invocation-ID",
                        "parent_lineage": "X-RLM-Parent-Lineage",
                        "parent_call_ordinal": "X-RLM-Parent-Call-Ordinal",
                        "parent_tool_call_slot": "X-RLM-Parent-Tool-Call-Slot",
                        "spawn_ordinal": "X-RLM-Spawn-Ordinal",
                        "episode_spawn_ordinal": "X-RLM-Episode-Spawn-Ordinal",
                        "completed_predecessor_spawn_ordinals": (
                            "X-RLM-Completed-Predecessor-Spawn-Ordinals"
                        ),
                        "completed_episode_spawn_ordinals": (
                            "X-RLM-Completed-Episode-Spawn-Ordinals"
                        ),
                    }
                    headers = {
                        "Authorization": f"Bearer {secret}",
                        "Content-Type": "application/json",
                        "X-Session-ID": str(rlm["session_id"]),
                    }
                    for key, value in rlm.items():
                        if key in names:
                            headers[names[key]] = (
                                ",".join(str(item) for item in value)
                                if isinstance(value, list)
                                else str(value)
                            )
                    response = await http.post(
                        endpoint.rstrip("/") + "/chat/completions",
                        content=action.key.request,
                        headers=headers,
                    )
                    if response.status_code != 200:
                        raise AssertionError(
                            f"fixture interception returned {response.status_code}: "
                            f"{response.text}"
                        )
            return ProgramResult(0, "", "")

    from verifiers.v1.runtimes.base import BaseRuntimeInfo

    class FixtureRuntime:
        is_local = True

        def __init__(self) -> None:
            self.name = "stage-d-qa-runtime"
            self.stopped = False
            self.info = BaseRuntimeInfo(id=self.name)

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            self.stopped = True

        async def run(
            self,
            argv: list[str],
            _env: dict[str, str],
        ) -> ProgramResult:
            if argv[:2] == ["sh", "-c"] and "uid=65534" in argv[2]:
                return ProgramResult(
                    0,
                    "uid=65534\n"
                    "gid=65534\n"
                    "home=/tmp/redco-agent\n"
                    "network=direct-blocked\n",
                    "",
                )
            return ProgramResult(0, "", "")

        async def write(self, _path: str, _data: bytes) -> None:
            return None

        async def read(self, _path: str) -> bytes:
            return b""

        async def prepare_setup(self) -> None:
            return None

        async def prepare_execution(self, _routes: list[str]) -> None:
            return None

        def host_url(self, url: str) -> str:
            return url

    class FixtureDockerConfig(DockerConfig):
        execution_user: str = "65534:65534"
        execution_home: str = "/tmp/redco-agent"

    import verifiers.v1.loaders as vf_loaders

    original_harness_class = vf_loaders.harness_class
    original_load_harness = vf_loaders.load_harness

    def fixture_harness_class(identifier: str) -> type[Harness[HarnessConfig]]:
        if identifier == "stage-d-qa-http-fixture":
            return FixtureHarness
        return original_harness_class(identifier)

    monkeypatch.setattr(vf_loaders, "harness_class", fixture_harness_class)
    monkeypatch.setattr(
        vf_loaders,
        "load_harness",
        lambda config: FixtureHarness(config)
        if config.id == "stage-d-qa-http-fixture"
        else original_load_harness(config),
    )
    import verifiers.v1.agent as vf_agent

    monkeypatch.setattr(
        vf_agent,
        "make_runtime",
        lambda _config, name=None: FixtureRuntime(),
    )
    import verifiers.v1.rollout as vf_rollout

    monkeypatch.setattr(
        vf_rollout,
        "make_runtime",
        lambda _config, name=None: FixtureRuntime(),
    )
    image = "python@sha256:" + "b" * 64
    harness_config = HarnessConfig(
        id="stage-d-qa-http-fixture",
        runtime={
            "type": "docker",
            "image": image,
            "workdir": "/workspace",
            "allow": [],
            "block": [],
        },
    )
    agent = cast(dict[str, object], payload["agent"])
    agent["model"] = base_key.checkpoint_id
    agent["harness"] = harness_config.model_dump(mode="json")
    taskset = cast(dict[str, object], payload["taskset"])
    workspace_manifest = tmp_path / "qa-workspace-manifest.json"
    workspace_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [{"path": "/workspace/evidence_context.txt", "mode": "0444"}],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    taskset["isolated_runtime_image"] = image
    taskset["policy_checkpoint_id"] = base_key.checkpoint_id
    payload["frozen_workspace_manifest_path"] = str(workspace_manifest)
    payload["frozen_workspace_manifest_sha256"] = _sha256(workspace_manifest.read_bytes())
    async def scenario() -> None:
        fake_openai = SimpleNamespace(
            base_url="http://qa-provider.invalid/v1",
            max_retries=0,
            post=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("provider POST was reached during reconstruction QA")
            ),
            close=lambda: asyncio.sleep(0),
        )
        client = TrainClient(fake_openai)
        client._pool = FixtureRenderer()
        client.default_headers = {}
        client.api_key_var = "STAGE_D_QA_FIXTURE_KEY"
        payload["resolved_train_client_sha256"] = _resolved_train_client_sha256(client)
        nonlocal_config = StageDSourceEnvConfig.model_validate(payload)
        fixture_runtime_config = FixtureDockerConfig(
            image=image,
            workdir="/workspace",
            allow=[],
            block=[],
        )
        object.__setattr__(
            nonlocal_config.agent.harness, "runtime", fixture_runtime_config
        )

        writer = StageDReceiptLedger.create(
            Path(cast(str, payload["ledger_path"])),
            binding=producer_tests._binding(),
            master_seed=producer_tests.MASTER_SEED,
        )
        original_episode = producer_tests._episode
        task = StageDSourceTaskset(nonlocal_config.taskset).load()[0]
        task_messages = [{"role": "user", "content": task.data.prompt}]
        renderer = client._pool
        if not isinstance(renderer, FixtureRenderer):
            raise AssertionError("QA fixture did not install its exact renderer")
        rendered = renderer.render(task_messages, add_generation_prompt=True)
        prompt_token_ids = tuple(rendered.token_ids)

        def exact_task_action(seed: int) -> BehaviorAction:
            action = producer_tests._prepared_action(
                seed,
                message={"role": "assistant", "content": "['exact evidence']"},
                messages=task_messages,
                prompt_token_ids=prompt_token_ids,
            )
            request = json.loads(action.key.request)
            if request.get("messages") != task_messages:
                raise AssertionError("frozen action request did not bind the replay task")
            if action.key.prompt_token_ids != prompt_token_ids:
                raise AssertionError("frozen action prompt did not bind the replay renderer")
            directed_sampling = base_sampling.model_copy(
                update={
                    "seed": seed,
                    "extra_body": {"cache_salt": f"seed-{seed}"},
                },
                deep=True,
            )
            engine_sampling, _template_kwargs, cache_salt, priority = (
                split_engine_sampling(directed_sampling)
            )
            engine_sampling["stop_token_ids"] = renderer.get_stop_token_ids()
            engine_sampling["logprobs"] = 1
            engine_sampling.setdefault("skip_special_tokens", False)
            prepared_engine: dict[str, object] = {
                "model": action.key.checkpoint_id,
                "token_ids": list(prompt_token_ids),
                "sampling_params": engine_sampling,
            }
            if cache_salt is not None:
                prepared_engine["cache_salt"] = cache_salt
            if priority is not None:
                prepared_engine["priority"] = priority
            exact_key = ExactActionKey.build_prepared(
                checkpoint_id=action.key.checkpoint_id,
                base_model_manifest=manifest_values["base_model"],
                adapter_manifest=manifest_values["adapter"],
                tokenizer_manifest=manifest_values["tokenizer"],
                renderer_manifest=manifest_values["renderer"],
                sampler_conformance_manifest=action.key.sampler_conformance_manifest,
                action_selection_policy=action.key.action_selection_policy,
                transport_retry_policy=action.key.transport_retry_policy,
                request=json.loads(action.key.request),
                prompt_token_ids=action.key.prompt_token_ids,
                prepared_engine_request=prepared_engine,
            )
            return BehaviorAction.build(
                key=exact_key,
                action_token_ids=action.action_token_ids,
                behavior_logprobs=action.behavior_logprobs,
                raw_transport_message=json.loads(action.raw_transport_message),
                finish_reason=action.finish_reason,
                prompt_tokens=action.prompt_tokens,
                completion_tokens=action.completion_tokens,
                termination_kind=action.termination_kind,
                eos_token_id=action.eos_token_id,
                validate_action=producer_tests._validate_prepared_action,
                request_id=action.request_id,
            )

        action_root = exact_task_action(71)
        action_child = exact_task_action(72)
        action_holder[:] = [action_root, action_child]
        if json.loads(action_root.key.request) == json.loads(base_key.request):
            raise AssertionError("QA fixture did not bind actions to the replay task")
        monkeypatch.setattr(producer_tests, "_action", exact_task_action)

        def full_task_episode(**kwargs: object) -> bytes:
            value = json.loads(original_episode(**kwargs))
            trace = cast(dict[str, object], value["traces"][0])
            trace["task"] = {
                "type": type(task).__name__,
                "data": task.data.model_dump(mode="json", exclude_none=False),
            }
            nodes = cast(list[object], trace["nodes"])
            for node_value in nodes:
                node = cast(dict[str, object], node_value)
                message = node.get("message")
                if isinstance(message, dict) and message.get("role") == "user":
                    node["message"] = dict(task_messages[0])
                elif isinstance(message, dict) and message.get("role") == "assistant":
                    node["message"] = {
                        "role": "assistant",
                        "content": "['exact evidence']",
                    }
            return cast(bytes, canonical_json(value))

        monkeypatch.setattr(producer_tests, "_episode", full_task_episode)
        source, spec = producer_tests._produce_into(
            writer,
            rollout_id="qa-real-owner",
            root_seed=71,
            child_seed=72,
            reward=1.0,
            selected=True,
        )
        if spec is None:
            raise AssertionError("fixture source did not produce its selected target")
        roster = StageDBranchTargetRoster.from_sources(
            (source,), planned_source_count=1, minimum_eligible_sources=1
        )
        writer.record_branch_target_roster(roster.to_bytes())
        raw_source = json.loads(
            (
                Path(cast(str, payload["ledger_path"]))
                / "evidence"
                / source.trace_sha256
            ).read_bytes()
        )
        trace = cast(dict[str, object], cast(dict[str, object], raw_source)["traces"][0])
        source_trace_holder["trace"] = trace
        records = extract_v2_rlm_provenance(trace)
        graph = build_source_causal_graph(records)
        task_from_trace = source_task_from_trace(trace, nonlocal_config.taskset.task)
        base_sampling_for_eval = vf.Sampling.model_validate(
            {key: value for key, value in sampling_fields.items()}
        )
        eval_config = EvalConfig(
            env=nonlocal_config,
            model=action_root.key.checkpoint_id,
            client=TrainClientConfig(base_url="http://qa-provider.invalid/v1"),
            sampling=base_sampling_for_eval,
            num_tasks=1,
            num_rollouts=1,
            max_concurrent=1,
            output_dir=tmp_path / "eval-output",
            push=False,
            rich=False,
        )
        import verifiers.v1.cli.eval.runner as runner
        from redco_evidence_selection_v2 import scientific_env as scientific_module

        monkeypatch.setattr(runner, "resolve_client", lambda _config: client)
        binding = StageDScientificEpisodeBinding(
            mode="qa",
            task=task_from_trace,
            source=source,
            source_records=tuple(records),
            source_graph=graph,
            target=spec.commitment.target_address,
            expected_runtime_snapshot=b"placeholder-runtime-snapshot",
            expected_terminal_reply=_terminal_reply(trace),
            ledger=writer,
        )
        watchdog = ActionClosureWatchdog()
        probe_env = scientific_module.StageDScientificReplayEnv(
            nonlocal_config,
            binding=binding,
            watchdog=watchdog,
        )
        resolved_agent = vf.AgentConfig.model_validate(agent).model_copy(
            update={"model": action_root.key.checkpoint_id, "sampling": base_sampling_for_eval},
            deep=True,
        )
        object.__setattr__(resolved_agent.harness, "runtime", fixture_runtime_config)
        expected_snapshot = probe_env._runtime_snapshot(
            probe_env.taskset.load()[0], resolved_agent
        )
        binding = replace(binding, expected_runtime_snapshot=expected_snapshot)
        # The actual production entry point performs the readiness check, creates
        # the scientific replay environment, and invokes the pinned run_eval.
        receipt = await run_bound_scientific_episode(
            binding=binding,
            env_config=nonlocal_config,
            eval_config=eval_config,
            watchdog=watchdog,
        )
        assert receipt == writer.reconstruction_qa_receipt("group-1", spec.commitment.target_id)
        assert receipt is not None
        assert writer.reconstruction_qa_barrier_receipt() is None
        watchdog.complete()
        with pytest.raises(LedgerError, match="reconstruction QA is already recorded"):
            await run_bound_scientific_episode(
                binding=binding,
                env_config=nonlocal_config,
                eval_config=eval_config,
                watchdog=watchdog,
            )
        recovered = writer.reconstruction_qa_receipt("group-1", spec.commitment.target_id)
        assert recovered == receipt
        writer.close()
        assert _tree_digest(checkout_outputs) == checkout_outputs_before

    asyncio.run(scenario())


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
        return cast(str, task.data.scientific_group_id)

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
        {
            "agent": {
                "model": "fixture-model",
                "max_turns": 8,
                "retries": {"max_retries": 1},
            }
        },
        {"agent": {"model": "fixture-model", "max_turns": None}},
        {"maximum_captured_session_call_count": 7},
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


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "subdirectory", ["pending", "sources"]
)
def test_source_collection_rejects_stale_artifacts_before_calls(
    tmp_path: Path,
    subdirectory: str,
) -> None:
    payload = _config_payload(tmp_path)
    artifact_path = Path(cast(str, payload["artifact_path"]))
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

    async def cancelled_super(
        self: Any,
        task: Any,
        ctx: Any,
        **kwargs: Any,
    ) -> None:
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
    return cast(str, StageDSourceTaskset(config.taskset).load()[0].data.scientific_group_id)


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
        return cast(
            str,
            _resolved_train_client_sha256(
                TrainClient(
                    SimpleNamespace(
                        max_retries=0,
                        base_url="http://127.0.0.1:8000/v1",
                    ),
                    default_headers={"X-Stage-D-Route": route},
                    api_key_var=api_key_var,
                )
            ),
        )

    baseline = identity("route-a", "STAGE_D_API_KEY")
    assert identity("route-b", "STAGE_D_API_KEY") != baseline
    assert identity("route-a", "OTHER_STAGE_D_API_KEY") != baseline
