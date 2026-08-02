#!/usr/bin/env python3
"""Run the frozen live Stage-D QA and branch campaign through pinned Verifiers."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from uuid import UUID

from redco.analysis.stage_d_branch_artifacts import StageDBranchTargetRoster
from redco.analysis.stage_d_protocol_manifest import StageDProtocolManifest
from redco.analysis.stage_d_receipt_ledger import (
    GenesisBinding,
    StageDReceiptLedger,
    inspect_ledger,
)
from redco.analysis.stage_d_rlm_runtime import (
    load_stage_d_rlm_runtime,
    verify_stage_d_env_rlm_harnesses,
)
from redco.analysis.stage_d_scientific_branch_group import (
    BranchGroupSpec,
    PreActionTargetCommitment,
    SeedCorrespondenceMap,
)
from redco.analysis.stage_d_scientific_campaign import (
    runtime_snapshot_from_pre_action_evidence,
)
from redco.analysis.stage_d_source_contracts import SourceRollout
from redco.analysis.stage_d_support_gate import (
    evaluate_support_gate,
    load_support_rules,
    verify_support_pass,
)
from redco.analysis.stage_d_zero_call_recovery import (
    recover_or_open_scientific_ledger,
)
from redco.contracts import canonical_json
from redco.integrations.write_once import write_once


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--protocol-manifest-sha256", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--source-artifacts", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--episode-output", type=Path, required=True)
    parser.add_argument("--master-seed", required=True)
    parser.add_argument("--dependency-stack", type=Path, required=True)
    parser.add_argument("--rlm-archive", type=Path, required=True)
    parser.add_argument("--uv-binary", type=Path, required=True)
    parser.add_argument("--uv-cache-archive", type=Path, required=True)
    parser.add_argument("--rlm-launcher", type=Path, required=True)
    parser.add_argument("--recover-zero-call", action="store_true")
    parser.add_argument("--supervisor-evidence", type=Path)
    parser.add_argument("--repair-archive", type=Path)
    parser.add_argument("--support-report", type=Path, required=True)
    parser.add_argument("--support-rules", type=Path, required=True)
    parser.add_argument("--support-rules-sha256", required=True)
    return parser.parse_args()


def _evidence_loader(root: Path, digest: str) -> bytes:
    value = (root / "evidence" / digest).read_bytes()
    if _sha256(value) != digest:
        raise ValueError("ledger evidence bytes differ from their digest")
    return value


def _preflight_scientific_artifact_root(
    root: Path,
    *,
    ledger: StageDReceiptLedger,
) -> None:
    expected_existing: set[Path] = set()
    expected_all: set[Path] = set()
    for group_id, target_id in ledger.branch_target_keys:
        path = (root / f"{group_id}--{target_id}.json").resolve()
        if path.parent != root.resolve() or path in expected_all:
            raise ValueError("scientific artifact path is unsafe or duplicated")
        expected_all.add(path)
        completed = ledger.completed_branch_artifact_sha256(
            group_id=group_id,
            target_id=target_id,
        )
        if completed is None:
            if path.exists():
                raise RuntimeError("uncommitted scientific artifact exists before calls")
        elif not path.is_file() or _sha256(path.read_bytes()) != completed:
            raise RuntimeError("completed scientific artifact is absent or differs")
        else:
            expected_existing.add(path)
    if root.exists():
        actual = {path.resolve() for path in root.iterdir()}
        if actual != expected_existing:
            raise RuntimeError("scientific artifact output has stale or unknown members")


def _run(args: argparse.Namespace) -> None:
    protocol = StageDProtocolManifest.verify_file(
        args.protocol_manifest,
        args.protocol_manifest_sha256,
    )
    if args.support_rules_sha256 != protocol.support_rules_sha256:
        raise ValueError("support rules hash differs from the protocol manifest")
    support_rules = load_support_rules(args.support_rules, protocol.support_rules_sha256)
    config_bytes = args.config.read_bytes()
    if _sha256(config_bytes) != args.config_sha256:
        raise ValueError("scientific config differs from its frozen SHA-256")
    if args.config_sha256 != protocol.scientific_eval_config_sha256:
        raise ValueError("scientific config differs from the protocol manifest")
    if _sha256(args.master_seed.encode("utf-8")) != protocol.master_seed_sha256:
        raise ValueError("scientific master seed differs from the protocol manifest")
    raw_config = tomllib.loads(config_bytes.decode("utf-8"))
    raw_env = raw_config.get("env")
    identity = protocol.policy_identity
    expected_raw_identity = {
        "checkpoint_id": identity.checkpoint_id,
        "base_model_manifest_sha256": identity.base_model_manifest_sha256,
        "adapter_manifest_sha256": identity.adapter_manifest_sha256,
        "tokenizer_manifest_sha256": identity.tokenizer_manifest_sha256,
        "renderer_manifest_sha256": identity.renderer_manifest_sha256,
        "sampler_conformance_manifest_sha256": (
            identity.sampler_conformance_manifest_sha256
        ),
        "resolved_agent_sampling_law_sha256": (
            identity.resolved_agent_sampling_law_sha256
        ),
        "resolved_train_client_sha256": identity.resolved_train_client_sha256,
        "support_rules_sha256": protocol.support_rules_sha256,
    }
    if (
        raw_config.get("model") != identity.checkpoint_id
        or not isinstance(raw_env, dict)
        or any(raw_env.get(name) != value for name, value in expected_raw_identity.items())
    ):
        raise ValueError("raw scientific policy identity differs from protocol manifest")
    pre_repair_scan = inspect_ledger(args.ledger, allow_repairable_zero_call=True)
    genesis = pre_repair_scan.records[0]["body"]
    expected_genesis = GenesisBinding(
        preregistration_sha256=protocol.preregistration_sha256,
        source_sha256=protocol.source_sha256,
        runtime_sha256=protocol.runtime_sha256,
        config_sha256=protocol.genesis_config_sha256,
        protocol_manifest_sha256=protocol.manifest_sha256,
        master_seed_sha256=protocol.master_seed_sha256,
        support_rules_sha256=protocol.support_rules_sha256,
    )
    if any(
        genesis.get(name) != value
        for name, value in expected_genesis.to_payload().items()
    ):
        raise ValueError("scientific ledger differs from the protocol manifest")
    ledger = recover_or_open_scientific_ledger(
        ledger_root=args.ledger,
        master_seed=args.master_seed,
        recover_requested=args.recover_zero_call,
        supervisor_evidence_path=args.supervisor_evidence,
        repair_archive=args.repair_archive,
        episode_output=args.episode_output,
    )
    if ledger.genesis_binding != expected_genesis:
        raise ValueError("recovered scientific ledger differs from the protocol manifest")
    _preflight_scientific_artifact_root(args.artifact_output, ledger=ledger)
    from redco_evidence_selection_v2.live_candidate import LiveVLLMCandidateEngine
    from redco_evidence_selection_v2.scientific_campaign_driver import (
        LiveScientificGroup,
        run_live_scientific_campaign,
        source_task_from_trace,
    )
    from redco_evidence_selection_v2.scientific_env import (
        StageDScientificEpisodeBinding,
        run_bound_scientific_episode,
    )
    from redco_evidence_selection_v2.source_env import _resolved_train_client_sha256
    from verifiers.v1.clients import resolve_client
    from verifiers.v1.clients.train import TrainClient, tool_to_wire
    from verifiers.v1.configs.eval import EvalConfig
    from verifiers.v1.dialects import parse_tools

    config = EvalConfig.model_validate(raw_config)
    dependency_stack, rlm_bundle = load_stage_d_rlm_runtime(
        protocol=protocol,
        dependency_stack_path=args.dependency_stack,
        archive_path=args.rlm_archive,
        uv_path=args.uv_binary,
        cache_archive_path=args.uv_cache_archive,
        launcher_path=args.rlm_launcher,
    )
    verify_stage_d_env_rlm_harnesses(
        config.env,
        manifest=dependency_stack,
        bundle=rlm_bundle,
    )
    if config.num_tasks != 1 or config.num_rollouts != 1 or config.max_concurrent != 1:
        raise ValueError("scientific runner base config must be one-by-one")
    if config.resume is not None or config.server or config.push:
        raise ValueError("scientific runner forbids resume, server pools, and remote push")
    client = None
    event_loop: asyncio.AbstractEventLoop | None = None
    try:
        if ledger.genesis_binding.protocol_manifest_sha256 != protocol.manifest_sha256:
            raise ValueError("scientific ledger differs from the protocol manifest")
        identity = protocol.policy_identity
        if config.env.checkpoint_id != identity.checkpoint_id:
            raise ValueError(
                "scientific model and environment checkpoint must both match protocol"
            )
        if (
            config.model,
            config.env.base_model_manifest_sha256,
            config.env.adapter_manifest_sha256,
            config.env.tokenizer_manifest_sha256,
            config.env.renderer_manifest_sha256,
            config.env.sampler_conformance_manifest_sha256,
            config.env.resolved_agent_sampling_law_sha256,
            config.env.resolved_train_client_sha256,
        ) != (
            identity.checkpoint_id,
            identity.base_model_manifest_sha256,
            identity.adapter_manifest_sha256,
            identity.tokenizer_manifest_sha256,
            identity.renderer_manifest_sha256,
            identity.sampler_conformance_manifest_sha256,
            identity.resolved_agent_sampling_law_sha256,
            identity.resolved_train_client_sha256,
        ):
            raise ValueError("scientific policy identity differs from protocol manifest")
        client = resolve_client(config.client)
        event_loop = asyncio.new_event_loop()
        if not isinstance(client, TrainClient):
            raise TypeError("scientific runner requires the pinned TrainClient")
        if _resolved_train_client_sha256(client) != (
            protocol.policy_identity.resolved_train_client_sha256
        ):
            raise ValueError("resolved scientific client differs from protocol manifest")
        renderer = client._renderer_pool(config.model)

        def render_prompt(request: Mapping[str, object]) -> tuple[int, ...]:
            messages = request.get("messages")
            if not isinstance(messages, list):
                raise ValueError("source request lacks messages")
            tools = parse_tools(request.get("tools"))
            wire_tools = [tool_to_wire(tool) for tool in tools] if tools else None
            return tuple(
                renderer.render_ids(
                    messages,
                    tools=wire_tools,
                    add_generation_prompt=True,
                )
            )

        def encode_action(
            request: Mapping[str, object],
            message: Mapping[str, object],
        ) -> tuple[int, ...]:
            return tuple(
                client.encode_assistant_action(
                    request,
                    message,
                    model=config.model,
                    prompt_token_ids=render_prompt(request),
                )
            )

        def decode_action(value: bytes):
            from redco.analysis.stage_d_exact_action import BehaviorAction

            return BehaviorAction.from_bytes(
                value,
                encode_action=encode_action,
                render_prompt=render_prompt,
            )

        sources = tuple(
            SourceRollout.verify_bytes(
                path.read_bytes(),
                verifier=ledger,
                evidence_loader=lambda digest: _evidence_loader(args.ledger, digest),
                encode_action=encode_action,
                render_prompt=render_prompt,
            )
            for path in sorted((args.source_artifacts / "sources").glob("*.json"))
        )
        if tuple(sorted(source.source_sha256 for source in sources)) != (
            ledger.completed_source_sha256s
        ):
            raise ValueError("source artifact roster differs from the live ledger")
        scan = inspect_ledger(args.ledger)
        if scan.status != "active-clean":
            raise RuntimeError("scientific runner requires an active-clean ledger")
        commitments = {
            (receipt["group_id"], receipt["target_id"]): canonical_json(receipt)
            for (kind, _), receipt in scan.receipts.items()
            if kind == "pre_action_group_commitment"
        }
        correspondences = {
            (receipt["group_id"], receipt["target_id"]): canonical_json(receipt)
            for (kind, _), receipt in scan.receipts.items()
            if kind == "seed_correspondence_map"
        }
        source_by_rollout = {source.rollout_id: source for source in sources}
        if len(source_by_rollout) != len(sources):
            raise ValueError("scientific source roster contains duplicate rollout IDs")
        trace_by_rollout: dict[str, dict[str, object]] = {}
        paper_ids: dict[str, str] = {}
        for source in sources:
            episode = json.loads(_evidence_loader(args.ledger, source.trace_sha256))
            traces = episode.get("traces") if isinstance(episode, dict) else None
            if not isinstance(traces, list) or len(traces) != 1:
                raise ValueError("source episode evidence is malformed")
            trace = traces[0]
            if not isinstance(trace, dict):
                raise ValueError("source trace evidence is malformed")
            task = trace.get("task")
            data = task.get("data") if isinstance(task, dict) else None
            paper_id = data.get("paper_id") if isinstance(data, dict) else None
            if not isinstance(paper_id, str) or not paper_id:
                raise ValueError("source trace lacks an authenticated paper identity")
            trace_by_rollout[source.rollout_id] = trace
            paper_ids[source.source_sha256] = paper_id
        if len(set(paper_ids.values())) != len(sources):
            raise ValueError("scientific source roster repeats an authenticated paper")

        async def run_episode(binding: StageDScientificEpisodeBinding) -> bytes:
            identity = binding.episode_identity
            output = args.episode_output / f"episode-{identity}"
            episode_config = config.model_copy(
                update={
                    "uuid": str(UUID(hex=identity[:32])),
                    "output_dir": output,
                    "rich": False,
                    "push": False,
                    "server": False,
                },
                deep=True,
            )
            return await run_bound_scientific_episode(
                binding=binding,
                env_config=config.env,
                eval_config=episode_config,
            )

        eos_token_id = json.loads(config.env.tokenizer_manifest_path.read_bytes()).get(
            "eos_token_id"
        )
        candidate_engine = LiveVLLMCandidateEngine(
            client=client,
            eos_token_id=eos_token_id,
        )
        groups = []
        for group_id, target_id in ledger.branch_target_keys:
            commitment_bytes = commitments[(group_id, target_id)]
            commitment = PreActionTargetCommitment.from_receipt(
                commitment_bytes,
                verifier=ledger,
            )
            source = source_by_rollout[commitment.rollout_id]
            recorded = next(
                decision.action
                for decision in source.decisions
                if decision.event_address == commitment.target_address
            )
            correspondence = SeedCorrespondenceMap.from_receipt(
                correspondences[(group_id, target_id)],
                verifier=ledger,
                commitment=commitment,
                recorded_action=recorded,
            )
            spec = BranchGroupSpec(commitment, recorded, correspondence, args.master_seed)
            trace = trace_by_rollout[source.rollout_id]
            task = source_task_from_trace(trace, config.env.taskset.task)
            runtime_snapshot = runtime_snapshot_from_pre_action_evidence(
                _evidence_loader(
                    args.ledger,
                    commitment.pre_action_snapshot_sha256,
                ),
                commitment=commitment,
                recorded_action=recorded,
            )
            groups.append(
                LiveScientificGroup(
                    spec=spec,
                    source=source,
                    task=task,
                    source_trace=trace,
                    expected_runtime_snapshot=runtime_snapshot,
                    candidate_engine=candidate_engine,
                    decode_action=decode_action,
                    run_episode=run_episode,
                    artifact_path=args.artifact_output / f"{group_id}--{target_id}.json",
                )
            )
        result = run_live_scientific_campaign(
            groups,
            ledger=ledger,
            event_loop=event_loop,
        )
        if len(result.artifacts) != len(ledger.branch_target_keys):
            raise RuntimeError("scientific campaign returned an incomplete artifact roster")
        scan = inspect_ledger(args.ledger)
        if scan.status != "active-clean":
            raise RuntimeError("scientific campaign did not leave an active-clean ledger")
        roster = StageDBranchTargetRoster.from_sources(
            sources,
            planned_source_count=support_rules.required_papers,
            minimum_eligible_sources=support_rules.required_successes,
        )
        if roster.roster_sha256 != ledger.branch_target_roster_sha256:
            raise ValueError("support target roster differs from the durable ledger")
        report = evaluate_support_gate(
            sources,
            result.artifacts,
            roster,
            paper_ids=paper_ids,
            rules=support_rules,
        )
        if args.support_report.exists():
            if args.support_report.read_bytes() != report:
                raise RuntimeError("existing support report differs")
        else:
            write_once(args.support_report, report)
        verify_support_pass(
            report,
            expected_rules_sha256=support_rules.rules_sha256,
            source_sha256s=tuple(source.source_sha256 for source in sources),
            artifact_sha256s=tuple(
                _sha256(artifact.to_bytes()) for artifact in result.artifacts
            ),
        )
        print(
            canonical_json(
                {
                    "status": "support-pass",
                    "artifact_count": len(result.artifacts),
                    "ledger_head_sha256": scan.record_sha256s[-1],
                    "record_count": len(scan.records),
                }
            ).decode("utf-8")
        )
    finally:
        try:
            if client is not None and event_loop is not None:
                event_loop.run_until_complete(client.close())
        finally:
            if event_loop is not None:
                event_loop.close()
            ledger.close()


def main() -> None:
    _run(_arguments())


if __name__ == "__main__":
    main()
