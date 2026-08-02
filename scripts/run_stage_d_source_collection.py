"""Run the exact one-shot Stage-D source collection lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import secrets
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.configs.eval import EvalConfig

from redco.analysis.stage_d_collection import (
    StageDCollectionPlan,
    run_exact_source_collection,
    verify_collection_outcomes,
    verify_direct_collection_config,
)
from redco.analysis.stage_d_protocol_manifest import StageDProtocolManifest
from redco.analysis.stage_d_receipt_ledger import StageDReceiptLedger
from redco.analysis.stage_d_source_artifacts import StageDSourceArtifactStore
from redco.analysis.stage_d_source_contracts import SourceRollout


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exclusive_write(path: Path, value: bytes) -> None:
    if not value or path.exists():
        raise RuntimeError(f"refusing to overwrite collection evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(f"refusing to overwrite collection evidence: {path}") from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--protocol-manifest-sha256", required=True)
    parser.add_argument("--genesis-config-sha256", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--recover", action="store_true")
    return parser.parse_args()


def _require_sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _authenticated_config(
    args: argparse.Namespace,
    protocol: StageDProtocolManifest,
) -> EvalConfig:
    config_bytes = args.config.read_bytes()
    if _sha256(config_bytes) != _require_sha256(args.config_sha256, "config SHA-256"):
        raise ValueError("config bytes differ from the externally frozen hash")
    if args.config_sha256 != protocol.source_eval_config_sha256:
        raise ValueError("source config differs from the protocol manifest")
    try:
        raw = tomllib.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("frozen config is not valid UTF-8 TOML") from error
    config = EvalConfig.model_validate(raw)
    expected = {
        "config_sha256": _require_sha256(
            args.genesis_config_sha256,
            "genesis config SHA-256",
        ),
        "preregistration_sha256": _require_sha256(
            args.preregistration_sha256,
            "preregistration SHA-256",
        ),
        "source_sha256": _require_sha256(args.source_sha256, "source SHA-256"),
        "runtime_sha256": _require_sha256(args.runtime_sha256, "runtime SHA-256"),
    }
    mismatches = {
        name: (getattr(config.env, name, None), value)
        for name, value in expected.items()
        if getattr(config.env, name, None) != value
    }
    if mismatches:
        raise ValueError(f"environment trust roots differ from external hashes: {mismatches}")
    if (
        _sha256(str(config.env.master_seed).encode("utf-8"))
        != protocol.master_seed_sha256
        or args.genesis_config_sha256 != protocol.genesis_config_sha256
        or args.plan_sha256 != protocol.collection_plan_sha256
        or args.preregistration_sha256 != protocol.preregistration_sha256
        or args.source_sha256 != protocol.source_sha256
        or args.runtime_sha256 != protocol.runtime_sha256
    ):
        raise ValueError("source runtime inputs differ from the protocol manifest")
    identity = protocol.policy_identity
    if config.model != identity.checkpoint_id:
        raise ValueError("source model and environment checkpoint must both match protocol")
    config_identity = (
        config.env.checkpoint_id,
        config.env.base_model_manifest_sha256,
        config.env.adapter_manifest_sha256,
        config.env.tokenizer_manifest_sha256,
        config.env.renderer_manifest_sha256,
        config.env.sampler_conformance_manifest_sha256,
        config.env.resolved_agent_sampling_law_sha256,
        config.env.resolved_train_client_sha256,
    )
    protocol_identity = (
        identity.checkpoint_id,
        identity.base_model_manifest_sha256,
        identity.adapter_manifest_sha256,
        identity.tokenizer_manifest_sha256,
        identity.renderer_manifest_sha256,
        identity.sampler_conformance_manifest_sha256,
        identity.resolved_agent_sampling_law_sha256,
        identity.resolved_train_client_sha256,
    )
    if config_identity != protocol_identity:
        raise ValueError("source policy identity differs from the protocol manifest")
    return config.model_copy(
        update={
            "env": config.env.model_copy(
                update={"protocol_manifest_sha256": protocol.manifest_sha256}
            )
        },
        deep=True,
    )


def _materialize_authenticated_plan(
    config: EvalConfig,
    *,
    plan_sha256: str,
    require_new_output: bool,
) -> tuple[Any, StageDCollectionPlan]:
    env = vf.load_environment(config.env)
    tasks = env.taskset.select(config.num_tasks, config.shuffle)
    plan = StageDCollectionPlan.build(
        [task.data.model_dump(mode="json") for task in tasks],
        master_seed=config.env.master_seed,
    )
    verify_direct_collection_config(
        config,
        planned_slot_count=len(plan.slots),
        require_new_output=require_new_output,
    )
    if plan.plan_sha256 != plan_sha256:
        raise ValueError("materialized source collection plan differs from preregistration")
    return env, plan


def _verify_unforced_root_tool_choice(config: EvalConfig) -> None:
    if getattr(config.sampling, "tool_choice", None) != "auto":
        raise ValueError("source collection requires tool_choice=auto")
    if os.environ.get("RLM_FORCE_TOOL_CHOICE_REQUIRED") not in {None, "0"}:
        raise ValueError("source collection forbids forced root tool choice")
    harness = getattr(config.env.agent, "harness", None)
    forwarded = getattr(harness, "forward_env", ()) if harness is not None else ()
    if "RLM_FORCE_TOOL_CHOICE_REQUIRED" in forwarded:
        raise ValueError("source harness must not forward forced root tool choice")


async def _recover_verified_sources(
    config: EvalConfig,
) -> tuple[SourceRollout, ...]:
    from verifiers.v1.clients import resolve_client
    from verifiers.v1.clients.train import TrainClient, tool_to_wire
    from verifiers.v1.dialects import parse_tools

    ledger = StageDReceiptLedger(
        config.env.ledger_path,
        master_seed=config.env.master_seed,
    )
    client = resolve_client(config.client)
    try:
        if not isinstance(client, TrainClient):
            raise TypeError("receipt recovery requires the frozen TrainClient")
        renderer = client._renderer_pool(config.model)

        def render_prompt(request: Mapping[str, Any]) -> tuple[int, ...]:
            messages = request.get("messages")
            if not isinstance(messages, list):
                raise ValueError("recovery request lacks messages")
            tools = parse_tools(request.get("tools"))
            wire_tools = [tool_to_wire(tool) for tool in tools] if tools else None
            rendered = tuple(
                renderer.render_ids(
                    messages,
                    tools=wire_tools,
                    add_generation_prompt=True,
                )
            )
            if not rendered or any(type(token) is not int or token < 0 for token in rendered):
                raise ValueError("recovery renderer returned invalid prompt tokens")
            return rendered

        def encode_action(
            request: Mapping[str, Any],
            message: Mapping[str, Any],
        ) -> tuple[int, ...]:
            action = tuple(
                client.encode_assistant_action(
                    request,
                    message,
                    model=config.model,
                    prompt_token_ids=render_prompt(request),
                )
            )
            if not action or any(type(token) is not int or token < 0 for token in action):
                raise ValueError("recovery renderer returned invalid action tokens")
            return action

        evidence_root = Path(config.env.ledger_path) / "evidence"

        def evidence_loader(digest: str) -> bytes:
            value = (evidence_root / digest).read_bytes()
            if _sha256(value) != digest:
                raise ValueError("ledger evidence bytes differ from their digest")
            return value

        store = StageDSourceArtifactStore(Path(config.env.artifact_path))
        store.assert_no_pending()
        sources = tuple(
            SourceRollout.verify_bytes(
                path.read_bytes(),
                verifier=ledger,
                evidence_loader=evidence_loader,
                encode_action=encode_action,
                render_prompt=render_prompt,
            )
            for path in store.source_paths()
        )
        if tuple(sorted(source.source_sha256 for source in sources)) != (
            ledger.completed_source_sha256s
        ):
            raise ValueError("recovered sources differ from the durable ledger roster")
        return sources
    finally:
        await client.close()
        ledger.close()


async def _recover_receipt(
    args: argparse.Namespace,
    config: EvalConfig,
    plan: StageDCollectionPlan,
) -> tuple[int, bytes]:
    from verifiers.v1.cli.output import read_episodes
    from verifiers.v1.task import WireTaskData
    from verifiers.v1.trace import Trace

    if args.plan_output.read_bytes() != plan.to_bytes():
        raise ValueError("persisted collection plan differs from the authenticated plan")
    episodes = tuple(read_episodes(config.output_dir, Trace[WireTaskData]))
    sources = await _recover_verified_sources(config)
    receipt = verify_collection_outcomes(plan, episodes, sources)
    if args.receipt_output.exists():
        if args.receipt_output.read_bytes() != receipt:
            raise ValueError("existing collection receipt differs from recovered evidence")
    else:
        _exclusive_write(args.receipt_output, receipt)
    return len(episodes), receipt


async def _run(args: argparse.Namespace) -> None:
    _require_sha256(args.plan_sha256, "plan SHA-256")
    if args.plan_output.resolve() == args.receipt_output.resolve():
        raise ValueError("collection plan and receipt paths must differ")
    protocol = StageDProtocolManifest.verify_file(
        args.protocol_manifest,
        _require_sha256(
            args.protocol_manifest_sha256,
            "protocol manifest SHA-256",
        ),
    )
    config = _authenticated_config(args, protocol)
    _verify_unforced_root_tool_choice(config)
    _env, authenticated_plan = _materialize_authenticated_plan(
        config,
        plan_sha256=args.plan_sha256,
        require_new_output=not args.recover,
    )
    if args.recover:
        if not args.plan_output.is_file():
            raise RuntimeError("receipt recovery requires the durable collection plan")
        episode_count, receipt = await _recover_receipt(
            args,
            config,
            authenticated_plan,
        )
        print(
            f"recovered slots={len(authenticated_plan.slots)} episodes={episode_count} "
            f"plan_sha256={authenticated_plan.plan_sha256} "
            f"receipt_sha256={_sha256(receipt)}"
        )
        return
    if args.plan_output.exists() or args.receipt_output.exists():
        raise RuntimeError("collection evidence path already exists")
    loaded_env: Any = None

    def load_environment(env_config: Any) -> Any:
        nonlocal loaded_env
        if loaded_env is not None:
            raise RuntimeError("source environment was loaded more than once")
        loaded_env = vf.load_environment(env_config)
        return loaded_env

    def load_verified_sources() -> tuple[SourceRollout, ...]:
        if loaded_env is None or not hasattr(loaded_env, "verified_completed_sources"):
            raise RuntimeError("source environment lacks verified completed sources")
        sources = loaded_env.verified_completed_sources()
        if not isinstance(sources, tuple) or any(
            type(source) is not SourceRollout for source in sources
        ):
            raise TypeError("source environment returned unverified source contracts")
        return sources

    plan, episodes, receipt = await run_exact_source_collection(
        config,
        preregistered_plan_sha256=args.plan_sha256,
        run_eval=run_eval,
        load_environment=load_environment,
        load_verified_sources=load_verified_sources,
        persist_plan=lambda value: _exclusive_write(args.plan_output, value),
    )
    _exclusive_write(args.receipt_output, receipt)
    print(
        f"slots={len(plan.slots)} episodes={len(episodes)} "
        f"plan_sha256={plan.plan_sha256} receipt_sha256={_sha256(receipt)}"
    )


def main() -> None:
    asyncio.run(_run(_arguments()))


if __name__ == "__main__":
    main()
