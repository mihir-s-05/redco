from __future__ import annotations

import asyncio
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomli_w
from test_stage_d_collection import _plan_and_episodes
from test_stage_d_source_env_pinned import _config_payload

pytest.importorskip("verifiers.v1")

_RUNNER_PATH = Path(__file__).parents[1] / "scripts" / "run_stage_d_source_collection.py"
_SPEC = importlib.util.spec_from_file_location("redco_stage_d_collection_runner", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _protocol(config_sha256: str, env: dict[str, object]) -> SimpleNamespace:
    identity = SimpleNamespace(
        checkpoint_id="fixture-model",
        base_model_manifest_sha256=env["base_model_manifest_sha256"],
        adapter_manifest_sha256=None,
        tokenizer_manifest_sha256=env["tokenizer_manifest_sha256"],
        renderer_manifest_sha256=env["renderer_manifest_sha256"],
        sampler_conformance_manifest_sha256=env[
            "sampler_conformance_manifest_sha256"
        ],
        resolved_agent_sampling_law_sha256=env[
            "resolved_agent_sampling_law_sha256"
        ],
        resolved_train_client_sha256=env["resolved_train_client_sha256"],
    )
    return SimpleNamespace(
        source_eval_config_sha256=config_sha256,
        genesis_config_sha256="4" * 64,
        preregistration_sha256="1" * 64,
        source_sha256="2" * 64,
        runtime_sha256="3" * 64,
        support_rules_sha256="8" * 64,
        master_seed_sha256=_sha256(b"fixture-master"),
        collection_plan_sha256="9" * 64,
        manifest_sha256="a" * 64,
        policy_identity=identity,
    )


def test_real_eval_config_materializes_without_hash_fixed_point(tmp_path: Path) -> None:
    env = _config_payload(tmp_path)
    payload = {
        "model": "fixture-model",
        "client": {
            "type": "train",
            "base_url": "http://127.0.0.1:8000/v1",
            "renderer": {"name": "auto"},
            "pool_size": 1,
        },
        "sampling": {
            "temperature": 0.7,
            "top_p": 1.0,
            "seed": 1,
            "max_tokens": 2,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "extra_body": {"cache_salt": "placeholder-only-before-episode-addressing"},
        },
        "env": env,
        "num_tasks": 2,
        "num_rollouts": 1,
        "shuffle": False,
        "max_concurrent": 1,
        "rich": False,
        "push": False,
        "server": False,
        "output_dir": str(tmp_path / "run"),
    }
    config_path = tmp_path / "real-source.toml"
    config_bytes = tomli_w.dumps(payload).encode("utf-8")
    config_path.write_bytes(config_bytes)
    args = SimpleNamespace(
        config=config_path,
        config_sha256=_sha256(config_bytes),
        genesis_config_sha256="4" * 64,
        preregistration_sha256="1" * 64,
        source_sha256="2" * 64,
        runtime_sha256="3" * 64,
        plan_sha256="9" * 64,
    )
    protocol = _protocol(args.config_sha256, env)

    config = runner._authenticated_config(args, protocol)

    assert config.env.config_sha256 == "4" * 64
    assert args.config_sha256 != config.env.config_sha256
    config_path.write_bytes(config_bytes + b"# changed\n")
    with pytest.raises(ValueError, match="externally frozen hash"):
        runner._authenticated_config(args, protocol)


def test_source_runner_rejects_forced_root_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        sampling=SimpleNamespace(tool_choice="auto"),
        env=SimpleNamespace(
            agent=SimpleNamespace(harness=SimpleNamespace(forward_env=[]))
        ),
    )
    monkeypatch.delenv("RLM_FORCE_TOOL_CHOICE_REQUIRED", raising=False)
    runner._verify_unforced_root_tool_choice(config)
    monkeypatch.setenv("RLM_FORCE_TOOL_CHOICE_REQUIRED", "1")
    with pytest.raises(ValueError, match="forced root tool choice"):
        runner._verify_unforced_root_tool_choice(config)


def test_rlm_preflight_executes_the_installed_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class Runtime:
        async def start(self) -> None:
            events.append("start")

        async def prepare_setup(self) -> None:
            events.append("prepare")

        async def run(self, command, env):
            events.append((command, env))
            return SimpleNamespace(exit_code=0, stderr="", stdout="usage")

        async def stop(self) -> None:
            events.append("stop")

    class Harness:
        config = SimpleNamespace(runtime="runtime-config", resolved_env={"PINNED": "1"})

        async def setup(self, runtime) -> None:
            events.append(("setup", runtime))

    runtime = Runtime()
    harness = Harness()
    monkeypatch.setattr(runner.vf, "load_harness", lambda _config: harness)
    monkeypatch.setattr(runner, "make_runtime", lambda _config: runtime)
    config = SimpleNamespace(env=SimpleNamespace(agent=SimpleNamespace(harness="config")))

    asyncio.run(runner._preflight_rlm_install(config))

    assert events == [
        "start",
        "prepare",
        ("setup", runtime),
        (["/tmp/vf-rlm/bin/rlm", "--help"], {"PINNED": "1"}),
        "stop",
    ]


def test_receipt_recovery_is_deterministic_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, episodes, sources = _plan_and_episodes()
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    plan_path.write_bytes(plan.to_bytes())
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    config = SimpleNamespace(output_dir=output_dir)
    args = SimpleNamespace(plan_output=plan_path, receipt_output=receipt_path)

    import verifiers.v1.cli.output as output

    monkeypatch.setattr(output, "read_episodes", lambda _path, _type: episodes)

    async def recovered_sources(_config):
        return sources

    monkeypatch.setattr(runner, "_recover_verified_sources", recovered_sources)
    count, first, first_sources = asyncio.run(
        runner._recover_receipt(args, config, plan)
    )
    assert count == len(episodes)
    assert first_sources == sources
    assert receipt_path.read_bytes() == first

    count, second, second_sources = asyncio.run(
        runner._recover_receipt(args, config, plan)
    )
    assert count == len(episodes)
    assert second_sources == sources
    assert second == first
