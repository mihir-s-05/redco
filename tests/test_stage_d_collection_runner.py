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


def _args(config: Path, config_sha256: str) -> SimpleNamespace:
    return SimpleNamespace(
        config=config,
        config_sha256=config_sha256,
        genesis_config_sha256="1" * 64,
        preregistration_sha256="2" * 64,
        source_sha256="3" * 64,
        runtime_sha256="4" * 64,
    )


def test_config_bytes_and_independent_genesis_binding_are_authenticated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "source.toml"
    config_bytes = b'title = "fixture"\n'
    config_path.write_bytes(config_bytes)
    resolved = SimpleNamespace(
        env=SimpleNamespace(
            config_sha256="1" * 64,
            preregistration_sha256="2" * 64,
            source_sha256="3" * 64,
            runtime_sha256="4" * 64,
        )
    )
    monkeypatch.setattr(
        runner,
        "EvalConfig",
        SimpleNamespace(model_validate=lambda _raw: resolved),
    )
    args = _args(config_path, _sha256(config_bytes))

    assert runner._authenticated_config(args) is resolved
    assert args.config_sha256 != args.genesis_config_sha256

    config_path.write_bytes(config_bytes + b"# changed\n")
    with pytest.raises(ValueError, match="externally frozen hash"):
        runner._authenticated_config(args)


def test_config_authentication_rejects_embedded_trust_root_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "source.toml"
    config_bytes = b'title = "fixture"\n'
    config_path.write_bytes(config_bytes)
    resolved = SimpleNamespace(
        env=SimpleNamespace(
            config_sha256="9" * 64,
            preregistration_sha256="2" * 64,
            source_sha256="3" * 64,
            runtime_sha256="4" * 64,
        )
    )
    monkeypatch.setattr(
        runner,
        "EvalConfig",
        SimpleNamespace(model_validate=lambda _raw: resolved),
    )
    with pytest.raises(ValueError, match="trust roots"):
        runner._authenticated_config(_args(config_path, _sha256(config_bytes)))


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
    )

    config = runner._authenticated_config(args)

    assert config.env.config_sha256 == "4" * 64
    assert args.config_sha256 != config.env.config_sha256


def test_evidence_write_is_exact_and_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "evidence" / "receipt.json"
    runner._exclusive_write(path, b"first")
    assert path.read_bytes() == b"first"
    assert tuple(path.parent.glob(".*.tmp")) == ()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        runner._exclusive_write(path, b"second")
    assert path.read_bytes() == b"first"


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
