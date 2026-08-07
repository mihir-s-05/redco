#!/usr/bin/env python3
"""Build the compact, evidence-addressed Stage-D closure-corpus v2 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_action_closure import (
    RUNTIME_TERM_TIMEOUT_SECONDS,
    WatchdogDeadlines,
    action_closure_case_manifest,
    audit_raw_response_fixtures,
    sha256_bytes,
)
from redco.analysis.stage_d_dependency_stack import live_owner_dependency_payload
from redco.analysis.stage_d_source_producer import (
    SAMPLING_CONTRACT_SHA256,
    SAMPLING_CONTRACT_VERSION,
)
from redco.analysis.stage_d_v13_draft_publication import atomic_publish_set
from redco.contracts import canonical_json

_FIXTURE_DATA: tuple[tuple[int, str, str, int, str], ...] = (
    (
        4,
        "00000000000000000002.json",
        "d28d827628213ee4c3f2bb4cc5d44edc4505605d769f45c1b3e7bfef1a95162d",
        43008,
        "raw_response_authenticated_no_action_envelope",
    ),
    (
        7,
        "00000000000000000002.json",
        "e9eb38142b90511dd40c8d292518421798984f98fccddb5cf6adde2ff8fbe866",
        35321,
        "raw_and_action_path",
    ),
    (
        7,
        "00000000000000000012.json",
        "e58bde34b81d3391cb4bdd97778afa0ef7976edf078743eeeef545ff2b34370d",
        7471,
        "raw_and_action_path",
    ),
    (
        7,
        "00000000000000000016.json",
        "42610e9abf93fb27ec866564909307646c8109fa15468ba7a6670028796a0139",
        11936,
        "raw_and_action_path",
    ),
    (
        8,
        "00000000000000000002.json",
        "c73d4ce6f9b6a6d97dacbbf959995fcf5d8ff3792626b496ac72cf6fb5ff3ec0",
        50739,
        "raw_and_action_path",
    ),
    (
        8,
        "00000000000000000012.json",
        "7327fb1b6ac35e713aa303213d4049c999abb1c264c5f279bcbc922f93ed4097",
        49927,
        "raw_and_action_path",
    ),
    (
        8,
        "00000000000000000016.json",
        "4a8d974cd55443384dd8f19a4b3e4078c8771d2af2aeb798970a285419bb796c",
        64089,
        "raw_and_action_path",
    ),
    (
        8,
        "00000000000000000021.json",
        "bf5038d2dad5b47a8453549ad7991d48145654b26bd6a130354212dd69f397f5",
        3422,
        "raw_and_action_path",
    ),
    (
        9,
        "00000000000000000002.json",
        "0d48ea4fbe3956863c851547f1e936a6de9d206927efb63abdf6191e57176f05",
        37084,
        "raw_and_action_path",
    ),
    (
        9,
        "00000000000000000012.json",
        "3693e91f3d6922d21a85e5ab2403fe92f3fcff080c1b9ec83d740dd418ccc57c",
        6123,
        "raw_and_action_path",
    ),
    (
        9,
        "00000000000000000016.json",
        "9cb7983d538689ad540eeffb59ad105693b26293414e49bb2edbe2f6830cabc4",
        7144,
        "raw_and_action_path",
    ),
    (
        10,
        "00000000000000000002.json",
        "89c0c4fc229c9f6b6262f0babaac6b805a5c0476fa7f618477734245a137b0d3",
        43123,
        "raw_and_action_path",
    ),
    (
        10,
        "00000000000000000012.json",
        "6627164f35c32b4afa67c71523a1b994be1f5c4fe021fd4d6dacdfca0d603026",
        23615,
        "raw_and_action_path",
    ),
    (
        10,
        "00000000000000000016.json",
        "aebb4586561ab1490b888233f967b23b3fa5302b816d621b2dfaa46e0ab710e4",
        104873,
        "raw_and_action_path",
    ),
)

_FIXTURES = tuple(
    {
        "version": version,
        "record": record,
        "path": f"runs/stage-d/stage-d1-support-v{version}/ledger/evidence/{digest}",
        "sha256": digest,
        "bytes": size,
        "replay": replay,
    }
    for version, record, digest, size, replay in _FIXTURE_DATA
)

_FROZEN_V1_AUDIT_PATH = "reports/stage-d1-action-closure-corpus-audit-v1.json"
_FROZEN_V1_AUDIT_SHA256 = "25c777aa9c121b818f3315bed5b13fe98336fe14aba31fe9c46f6e53808e6b6c"
_HISTORICAL_REPLAY_PATH = "reports/stage-d1-historical-semantic-replay-v1.json"
_HISTORICAL_REPLAY_SHA256 = "beb6de2ad14118299542682232035729569937c7c803a7823d5ac6de2049d552"


def build(repository: Path) -> tuple[bytes, bytes]:
    fixture_audit = audit_raw_response_fixtures(repository, _FIXTURES)
    frozen_v1_audit = repository / _FROZEN_V1_AUDIT_PATH
    if not frozen_v1_audit.is_file():
        raise ValueError(f"missing frozen v1 audit: {_FROZEN_V1_AUDIT_PATH}")
    frozen_v1_audit_sha256 = sha256_bytes(frozen_v1_audit.read_bytes())
    if frozen_v1_audit_sha256 != _FROZEN_V1_AUDIT_SHA256:
        raise ValueError("frozen v1 action-closure audit hash changed")
    historical_replay = repository / _HISTORICAL_REPLAY_PATH
    if not historical_replay.is_file():
        raise ValueError(f"missing historical semantic replay: {_HISTORICAL_REPLAY_PATH}")
    historical_replay_sha256 = sha256_bytes(historical_replay.read_bytes())
    if historical_replay_sha256 != _HISTORICAL_REPLAY_SHA256:
        raise ValueError("historical semantic replay hash changed")
    replay_payload = json.loads(historical_replay.read_bytes())
    if (
        not isinstance(replay_payload, dict)
        or replay_payload.get("domain") != "redco-stage-d-historical-semantic-replay-v1"
        or replay_payload.get("historical_versions_semantically_replayed")
        != [4, 7, 8, 9, 10]
        or replay_payload.get("semantic_renderer_observer_replay_count") != 14
        or replay_payload.get("completed_action_replay_count") != 12
        or replay_payload.get("unavailable_versions_were_not_reconstructed")
        != [1, 2, 3, 5, 6]
        or replay_payload.get("live_support_run_authorized") is not False
        or replay_payload.get("scientific_training_authorized") is not False
    ):
        raise ValueError(
            "historical semantic replay evidence is not the authenticated v4/v7-v10 path"
        )
    cases = action_closure_case_manifest()
    dependency_stack = live_owner_dependency_payload(repository)
    producer_source_path = repository / "src/redco/analysis/stage_d_source_producer.py"
    if not producer_source_path.is_file() or producer_source_path.is_symlink():
        raise ValueError("source producer implementation is missing")
    producer_source_sha256 = sha256_bytes(producer_source_path.read_bytes())
    sampling_contract = {
        "version": SAMPLING_CONTRACT_VERSION,
        "sha256": SAMPLING_CONTRACT_SHA256,
        "producer_source_path": "src/redco/analysis/stage_d_source_producer.py",
        "producer_source_sha256": producer_source_sha256,
    }
    config: dict[str, Any] = {
        "schema_version": 2,
        "domain": "redco-stage-d-action-closure-corpus-v2",
        "historical_versions": [
            {"version": version, "raw_response_replayable": version in {4, 7, 8, 9, 10}}
            for version in range(1, 11)
        ],
        "metadata_only_versions": [1, 2, 3, 5, 6],
        "raw_response_fixtures": list(_FIXTURES),
        "raw_response_fixture_count": 14,
        "completed_action_reload_count": 12,
        "action_case_manifest": cases,
        "retry": False,
        "sampled_action_retries": 0,
        "no_second_provider_request": True,
        "scratch_replay_policy": "historical ledgers and evidence are read-only",
        "watchdog_deadlines_seconds": WatchdogDeadlines().to_payload(),
        "watchdog_runtime_term_timeout_seconds": RUNTIME_TERM_TIMEOUT_SECONDS,
        "dependency_stack": dependency_stack,
        "sampling_contract": sampling_contract,
        "watchdog_integration": (
            "provider_call -> concurrent_children -> episode -> finalizer -> campaign; "
            "pod_lifetime owns launcher teardown; scripts/run_stage_d_source_collection.py "
            "owns the outer campaign"
        ),
        "owner_execution": {
            "accepted_vectors": {
                "conversion": "pinned TrainClient response conversion when available",
                "interception": "StageDPreparedCallObserver before/after response boundary",
                "action": "BehaviorAction.build/from_bytes exact evidence owner",
                "finalizer": "StageDSourceRolloutProducer.finalize_episode",
                "ledger": "StageDReceiptLedger terminal/source ownership",
            },
            "abort_vectors": {
                "owner": (
                    "StageDPreparedCallObserver.abort -> "
                    "StageDSourceRolloutProducer.abort_policy_call"
                ),
                "terminal_record": "exactly_once_canonical_abort_or_completion",
                "second_provider_post": False,
            },
            "mutation_failure_origins": {
                "finish_reason": "BehaviorAction.build",
                "usage": "BehaviorAction.build",
                "tool_argument_bytes": "StageDSourceRolloutProducer._verify_trace_call",
                "address": "StageDSourceRolloutProducer.finalize_episode",
            },
        },
    }
    config_raw = canonical_json(config)
    report = {
        "schema_version": 2,
        "domain": "redco-stage-d-action-closure-corpus-audit-v2",
        "passes": True,
        "corpus_sha256": sha256_bytes(config_raw),
        "raw_response_fixture_count": fixture_audit["fixture_count"],
        "raw_response_total_bytes": fixture_audit["total_bytes"],
        "raw_response_fixture_manifest_sha256": fixture_audit["raw_fixture_manifest_sha256"],
        "frozen_v1_audit_path": _FROZEN_V1_AUDIT_PATH,
        "frozen_v1_audit_sha256": frozen_v1_audit_sha256,
        "historical_semantic_replay_path": _HISTORICAL_REPLAY_PATH,
        "historical_semantic_replay_sha256": historical_replay_sha256,
        "historical_semantic_replayed_versions": [4, 7, 8, 9, 10],
        "semantic_renderer_observer_replay_count": 14,
        "completed_action_reload_count": 12,
        "metadata_only_versions": [1, 2, 3, 5, 6],
        "semantic_renderer_replay_performed": False,
        "sampled_action_retries": 0,
        "watchdog_deadlines_seconds": WatchdogDeadlines().to_payload(),
        "watchdog_runtime_term_timeout_seconds": RUNTIME_TERM_TIMEOUT_SECONDS,
        "dependency_stack": dependency_stack,
        "sampling_contract": sampling_contract,
        "watchdog_integration": (
            "production collection entry point with one canonical terminal callback; "
            "launcher pod lifetime is separately bounded"
        ),
        "action_case_manifest_sha256": cases["manifest_sha256"],
        "density": {
            "sampling_unit": "paper_episode",
            "observed_episodes": 4,
            "observed_scaffold_proxy": (
                "3/4 descriptive only: v7, v8, and v9 retain both first-turn child ordinals; "
                "v10 retains only child ordinal 1; no eligibility or informativeness inference"
            ),
            "eligibility": "unknown",
            "informativeness": "unknown",
            "joint_density": "not_identifiable",
            "reward_ranges": "missing_or_censored_unknown",
            "selected_ordinal_180_excluded": True,
        },
    }
    return config_raw, canonical_json(report)


def build_artifacts(repository: Path) -> dict[str, bytes]:
    """Build the closure bytes without writing any output."""

    config_raw, report_raw = build(repository)
    return {
        "configs/stage-d/stage-d1-action-closure-corpus-v2.json": config_raw,
        "reports/stage-d1-action-closure-corpus-audit-v2.json": report_raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.repository
    output_root = (args.output_root or root).resolve()
    artifacts = build_artifacts(root)
    immutable_inputs = {
        str((root / _FROZEN_V1_AUDIT_PATH).resolve()): _FROZEN_V1_AUDIT_SHA256,
        str((root / _HISTORICAL_REPLAY_PATH).resolve()): _HISTORICAL_REPLAY_SHA256,
        **{
            str((root / str(fixture["path"])).resolve()): str(fixture["sha256"])
            for fixture in _FIXTURES
        },
    }
    hashes = atomic_publish_set(
        output_root,
        artifacts,
        immutable_paths=immutable_inputs,
        manifest_path="reports/stage-d1-action-closure-corpus-audit-v2.json",
        check_only=args.check_only,
    )
    print(
        canonical_json(
            {
                "check_only": args.check_only,
                "artifacts": [
                    {"path": path, "sha256": hashes[path], "bytes": len(artifacts[path])}
                    for path in sorted(artifacts)
                ],
            }
        ).decode()
    )


if __name__ == "__main__":
    main()
