#!/usr/bin/env python3
"""Replay retained Stage-D responses through the pinned renderer and observer path."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from audit_stage_d_action_closure import audit as audit_action_closure

from redco.analysis.stage_d_dependency_stack import canonical_tree_manifest_bytes
from redco.analysis.stage_d_exact_action import BehaviorAction
from redco.analysis.stage_d_live_observer import (
    StageDForwardDirectiveObserver,
    StageDObserverIdentity,
    StageDObserverProtocol,
    StageDPreparedCallObserver,
)
from redco.analysis.stage_d_receipt_ledger import GenesisBinding, StageDReceiptLedger
from redco.analysis.stage_d_source_producer import StageDSourceRolloutProducer
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress
from redco.contracts import canonical_json
from redco.integrations.signed_subprocess import sign_payload

MASTER_SEED = "stage-d-historical-semantic-replay-v1"
CHECKPOINT_ID = "/workspace/models/stage-d1-merged"
FROZEN_INPUT_SHA256S = {
    "action_closure_corpus": "37db1d80f993fa90ee9c88f18347322f54a47332ee354edaa3fa16647b1604a9",
    "action_closure_corpus_audit": (
        "25c777aa9c121b818f3315bed5b13fe98336fe14aba31fe9c46f6e53808e6b6c"
    ),
    "dependency_auth_amendment_v11_2": (
        "3712b6719283504a725e728e67423a298cda7b3e02efb6130b7d17e7aa890f7d"
    ),
    "dependency_stack_v11_2": "681791c039804924f0bd3ccaca42653128088442bf5f803436fb11b2d53cbe47",
    "preregistration_v11_2": "0c8fc2794260cdf74a3a742a938790913799b88216ce67da633b47d91052e2ee",
    "protocol_v11_2": "7cc7528b736365690b62580827bcc165f9972f5341932027533ca725d8e20575",
    "redco_cpu_amendment": "f3a5f1b6a68fdde022b77752b5841bdaf5e1e35dbbb079555b858555feb10370",
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _require_sha256(path: Path, expected: str) -> bytes:
    value = path.read_bytes()
    if _sha256(value) != expected:
        raise ValueError(f"{path} differs from its frozen SHA-256")
    return value


def _conformance() -> bytes:
    return canonical_json(
        sign_payload(
            {
                "schema_version": 1,
                "analysis": "served-stack-categorical-logprob-conformance-v1",
                "passes": True,
                "logprob_semantics": "served_chosen_token_post_transform",
                "categorical_case_count": 3,
                "served_stack_sha256": "a" * 64,
                "tool_call_termination_includes_all_generated_tokens": True,
                "eos_is_included_in_action_tokens_and_logprobs": True,
            }
        )
    )


def _sampling(vf: Any, request: Mapping[str, Any]) -> Any:
    return vf.Sampling(
        temperature=float(request["temperature"]),
        top_p=float(request["top_p"]),
        top_k=request.get("top_k"),
        min_p=float(request.get("min_p", 0.0)),
        repetition_penalty=float(request.get("repetition_penalty", 1.0)),
        frequency_penalty=float(request.get("frequency_penalty", 0.0)),
        presence_penalty=float(request.get("presence_penalty", 0.0)),
        logit_bias=dict(request.get("logit_bias") or {}),
        seed=int(request["seed"]),
        max_tokens=int(request["max_tokens"]),
        stop=request.get("stop"),
        n=int(request.get("n", 1)),
        best_of=request.get("best_of"),
        use_beam_search=bool(request.get("use_beam_search", False)),
        logprobs=bool(request.get("logprobs", True)),
        top_logprobs=int(request.get("top_logprobs", 0)),
        ignore_eos=bool(request.get("ignore_eos", False)),
        min_tokens=int(request.get("min_tokens", 0)),
        tool_choice=request.get("tool_choice", "auto"),
        parallel_tool_calls=bool(request.get("parallel_tool_calls", False)),
        extra_body=dict(request.get("extra_body") or {}),
    )


def _ledger(root: Path) -> StageDReceiptLedger:
    return StageDReceiptLedger.create(
        root,
        binding=GenesisBinding(
            preregistration_sha256="1" * 64,
            source_sha256="2" * 64,
            runtime_sha256="3" * 64,
            config_sha256="4" * 64,
            protocol_manifest_sha256="5" * 64,
            master_seed_sha256=_sha256(MASTER_SEED.encode()),
            support_rules_sha256="6" * 64,
        ),
        master_seed=MASTER_SEED,
    )


def _observer(
    root: Path,
    *,
    trace_id: str,
    eos_token_id: int,
    validate_action: Any,
    manifests: Mapping[str, bytes],
) -> tuple[StageDPreparedCallObserver, StageDReceiptLedger, StageDSourceRolloutProducer]:
    ledger = _ledger(root)
    parent = PolicyEventAddress(0, "root", 0, 0)
    producer = StageDSourceRolloutProducer(
        ledger=ledger,
        group_id=f"historical-{trace_id}",
        rollout_id=trace_id,
        child_parent_event=parent,
        child_parent_tool_call_slot=0,
        root_policy_turn_count=2,
        base_model_manifest_sha256=_sha256(manifests["base_model"]),
    )
    observer = StageDPreparedCallObserver(
        producer=producer,
        trace_id=trace_id,
        identity=StageDObserverIdentity(
            checkpoint_id=CHECKPOINT_ID,
            base_model_manifest=manifests["base_model"],
            adapter_manifest=manifests["adapter"],
            tokenizer_manifest=manifests["tokenizer"],
            renderer_manifest=manifests["renderer"],
            sampler_conformance_manifest=manifests["sampler"],
            eos_token_id=eos_token_id,
        ),
        protocol=StageDObserverProtocol(
            branch_count=4,
            continuation_replicates=1,
            failure_reward=0.0,
            root_policy_turn_count=2,
            maximum_captured_session_call_count=16,
        ),
        runtime_snapshot=canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-historical-replay-runtime-v1",
            }
        ),
        validate_action=validate_action,
    )
    return observer, ledger, producer


def _ledger_snapshot(repository: Path, version: int) -> dict[str, Any]:
    """Read one authenticated historical ledger once and pair its evidence."""
    root = repository / f"runs/stage-d/stage-d1-support-v{version}/ledger"
    if not root.is_dir():
        raise ValueError(f"v{version} retained ledger is missing")
    reserved: dict[str, dict[str, Any]] = {}
    observed: dict[str, dict[str, Any]] = {}
    completed: dict[str, dict[str, Any]] = {}
    commitments: list[dict[str, Any]] = []
    receipt_kinds: dict[str, int] = {}
    for path in sorted((root / "records").glob("*.json")):
        record = _read_object(path)
        if record.get("record_kind") != "receipt":
            continue
        receipt = record["body"]["receipt"]
        kind = str(receipt.get("receipt_kind"))
        receipt_kinds[kind] = receipt_kinds.get(kind, 0) + 1
        if kind == "pre_action_group_commitment":
            commitments.append(receipt)
        decision_id = receipt.get("decision_id")
        if not isinstance(decision_id, str):
            continue
        destination = {
            "source_policy_call_reserved": reserved,
            "source_policy_response_observed": observed,
            "source_policy_call_completed": completed,
        }.get(kind)
        if destination is not None:
            if decision_id in destination:
                raise ValueError(f"v{version} duplicates {kind} for {decision_id}")
            destination[decision_id] = receipt
    rows: list[dict[str, Any]] = []
    for decision_id, response in sorted(
        observed.items(), key=lambda item: int(item[1]["request_sequence"])
    ):
        reservation = reserved.get(decision_id)
        if reservation is None:
            raise ValueError(f"v{version} observed response lacks its reservation")
        row: dict[str, Any] = {
            "decision_id": decision_id,
            "request": str(reservation["request_sha256"]),
            "raw": str(response["raw_response_sha256"]),
            "reservation": reservation,
        }
        completion = completed.get(decision_id)
        if completion is not None:
            if (
                completion["raw_response_sha256"] != row["raw"]
                or completion["request_sequence"] != response["request_sequence"]
                or completion["exact_action_key_digest"]
                != reservation["exact_action_key_digest"]
            ):
                raise ValueError(f"v{version} completed evidence pairing differs")
            row["action"] = str(completion["response_sha256"])
        rows.append(row)
    if set(completed) - set(observed):
        raise ValueError(f"v{version} completed action lacks its raw-response witness")
    completed_ids = set(completed)
    completed_target_ordinals = sorted(
        int(row["reservation"]["target_ordinal"])
        for row in rows
        if row["decision_id"] in completed_ids
        and (receipt := row["reservation"])
        and receipt.get("node_kind") == "child"
        and receipt.get("target_ordinal") in {0, 1}
        and receipt.get("target_address", {}).get("depth") == 1
        and receipt.get("target_address", {}).get("turn") == 0
        and receipt.get("target_address", {}).get("session_call_ordinal") == 0
    )
    eligible_commitments = sorted(
        (
            str(receipt["group_id"]),
            str(receipt["rollout_id"]),
            str(receipt["target_id"]),
            int(receipt["target_ordinal"]),
        )
        for receipt in commitments
        if receipt.get("target_ordinal") in {0, 1}
        and receipt.get("target_address", {}).get("depth") == 1
        and receipt.get("target_address", {}).get("turn") == 0
        and receipt.get("target_address", {}).get("session_call_ordinal") == 0
    )
    completed_commitments = sorted(
        (
            str(receipt["group_id"]),
            str(receipt["rollout_id"]),
            str(receipt["target_id"]),
            int(receipt["target_ordinal"]),
        )
        for row in rows
        if row["decision_id"] in completed_ids
        and (receipt := row["reservation"])
        and receipt.get("node_kind") == "child"
        and receipt.get("target_ordinal") in {0, 1}
        and receipt.get("target_address", {}).get("depth") == 1
        and receipt.get("target_address", {}).get("turn") == 0
        and receipt.get("target_address", {}).get("session_call_ordinal") == 0
    )
    if any(item not in eligible_commitments for item in completed_commitments):
        raise ValueError(f"v{version} completed child action lacks its commitment")
    proxy = {
        "version": version,
        "completed_policy_actions": len(completed_ids),
        "pre_action_commitments": receipt_kinds.get("pre_action_group_commitment", 0),
        "completed_frozen_first_turn_child_ordinals": completed_target_ordinals,
        "descriptive_scaffold_proxy": completed_target_ordinals == [0, 1],
        "source_rollout_finalized": receipt_kinds.get("source_rollout_finalized", 0) > 0,
        "branch_outcomes_observed": sum(
            count for kind, count in receipt_kinds.items() if kind.startswith("branch_")
        ),
    }
    return {"rows": rows, "support_proxy": proxy, "receipt_counts": receipt_kinds}


def _verify_frozen_inputs(
    repository: Path,
    tokenizer_path: Path,
    renderers_root: Path,
    verifiers_root: Path,
) -> dict[str, Any]:
    paths = {
        "action_closure_corpus": repository
        / "configs/stage-d/stage-d1-action-closure-corpus-v1.json",
        "action_closure_corpus_audit": repository
        / "reports/stage-d1-action-closure-corpus-audit-v1.json",
        "dependency_auth_amendment_v11_2": repository
        / "configs/stage-d/stage-d1-support-dependency-auth-amendment-v11-2.json",
        "dependency_stack_v11_2": repository
        / "configs/stage-d/stage-d1-dependency-stack-v11-2.json",
        "preregistration_v11_2": repository
        / "configs/stage-d/stage-d1-support-preregistration-v11-2.json",
        "protocol_v11_2": repository
        / "configs/stage-d/stage-d1-support-protocol-v11-2.json",
        "redco_cpu_amendment": repository
        / "configs/stage-d/stage-d1-historical-replay-redco-amendment-v1.json",
    }
    frozen = {
        name: _require_sha256(path, FROZEN_INPUT_SHA256S[name])
        for name, path in paths.items()
    }
    regenerated_corpus_audit = (
        canonical_json(audit_action_closure(repository, paths["action_closure_corpus"]))
        + b"\n"
    )
    if regenerated_corpus_audit != frozen["action_closure_corpus_audit"]:
        raise ValueError("historical ledger/corpus audit no longer reproduces byte-exactly")

    preregistration = json.loads(frozen["preregistration_v11_2"])
    protocol = json.loads(frozen["protocol_v11_2"])
    dependency_amendment = json.loads(frozen["dependency_auth_amendment_v11_2"])
    redco_amendment = json.loads(frozen["redco_cpu_amendment"])
    if (
        preregistration["deployment_authentication"]["dependency_stack_sha256"]
        != FROZEN_INPUT_SHA256S["dependency_stack_v11_2"]
        or preregistration["deployment_authentication"]["amendment_sha256"]
        != FROZEN_INPUT_SHA256S["dependency_auth_amendment_v11_2"]
        or protocol["dependency_stack_sha256"]
        != FROZEN_INPUT_SHA256S["dependency_stack_v11_2"]
        or dependency_amendment["dependency_stack_sha256"]
        != FROZEN_INPUT_SHA256S["dependency_stack_v11_2"]
    ):
        raise ValueError("v11.2 dependency-authentication chain differs")

    manifest_paths = {
        "base_model_manifest_sha256": repository
        / "configs/stage-d/stage-d1-base-model-manifest.json",
        "adapter_manifest_sha256": repository
        / "reports/stage-d0-scaffold-step8-adapter-manifest-v1.json",
        "tokenizer_manifest_sha256": repository
        / "configs/stage-d/stage-d1-tokenizer-manifest.json",
        "renderer_manifest_sha256": repository
        / "configs/stage-d/stage-d1-renderer-manifest.json",
        "sampler_conformance_manifest_sha256": repository
        / "configs/stage-d/stage-d1-sampler-conformance-manifest.json",
    }
    observed_policy = {
        "checkpoint_id": CHECKPOINT_ID,
        **{field: _sha256(path.read_bytes()) for field, path in manifest_paths.items()},
    }
    prereg_policy = preregistration["policy"]
    protocol_policy = protocol["policy_identity"]
    for field, observed in observed_policy.items():
        if prereg_policy.get(field) != observed or protocol_policy.get(field) != observed:
            raise ValueError(f"v11.2 policy identity differs for {field}")

    tokenizer_manifest = _read_object(manifest_paths["tokenizer_manifest_sha256"])
    tokenizer_files = {
        "tokenizer_config_sha256": tokenizer_path / "tokenizer_config.json",
        "tokenizer_json_sha256": tokenizer_path / "tokenizer.json",
    }
    for field, path in tokenizer_files.items():
        _require_sha256(path, str(tokenizer_manifest[field]))

    stack = json.loads(frozen["dependency_stack_v11_2"])
    components = {item["name"]: item for item in stack["components"]}
    roots = {"renderers": renderers_root.resolve(), "verifiers": verifiers_root.resolve()}
    observed_trees = {
        name: _sha256(canonical_tree_manifest_bytes(root))
        for name, root in roots.items()
    }
    for name, digest in observed_trees.items():
        if digest != components[name]["post_tree_sha256"]:
            raise ValueError(f"active {name} tree differs from the frozen patched tree")

    redco_modules = (
        "src/redco/analysis/stage_d_exact_action.py",
        "src/redco/analysis/stage_d_live_observer.py",
        "src/redco/analysis/stage_d_receipt_ledger.py",
        "src/redco/analysis/stage_d_source_producer.py",
    )
    active_redco_modules = {
        path: _sha256((repository / path).read_bytes()) for path in redco_modules
    }
    if redco_amendment["redco_module_sha256s"] != active_redco_modules or (
        redco_amendment["parent"]
        != {
            "dependency_stack_v11_2_sha256": FROZEN_INPUT_SHA256S[
                "dependency_stack_v11_2"
            ],
            "preregistration_v11_2_sha256": FROZEN_INPUT_SHA256S[
                "preregistration_v11_2"
            ],
            "protocol_v11_2_sha256": FROZEN_INPUT_SHA256S["protocol_v11_2"],
        }
    ):
        raise ValueError("active Redco modules differ from the CPU replay amendment")
    stack_redco_paths: set[str] = set()
    for imported in stack["imported_modules"]:
        if not str(imported["name"]).startswith("redco."):
            continue
        absolute = str(imported["absolute_path"])
        prefix = "/workspace/redco/"
        if not absolute.startswith(prefix):
            raise ValueError("frozen Redco import path is outside the repository")
        relative = absolute.removeprefix(prefix)
        stack_redco_paths.add(relative)
        expected = redco_amendment["redco_module_sha256s"].get(
            relative, imported["sha256"]
        )
        if _sha256((repository / relative).read_bytes()) != expected:
            raise ValueError(f"active Redco import differs for {imported['name']}")
    if not set(redco_amendment["redco_module_sha256s"]).issubset(stack_redco_paths):
        raise ValueError("CPU replay amendment names an unbound Redco module")
    return {
        "frozen_input_sha256s": FROZEN_INPUT_SHA256S,
        "audit_script_sha256": _sha256(Path(__file__).read_bytes()),
        "active_redco_module_sha256s": active_redco_modules,
        "tokenizer_checkpoint": tokenizer_manifest["checkpoint"],
        "tokenizer_file_sha256s": {
            field: str(tokenizer_manifest[field]) for field in tokenizer_files
        },
        "policy_identity": observed_policy,
        "active_dependency_tree_sha256s": observed_trees,
        "active_dependency_roots": {
            name: str(root) for name, root in roots.items()
        },
    }


async def _replay_one(
    *,
    scratch: Path,
    renderer: Any,
    manifests: Mapping[str, bytes],
    version: int,
    ordinal: int,
    request_bytes: bytes,
    raw_response: bytes,
    expected_action: BehaviorAction | None,
) -> dict[str, Any]:
    import multidict
    import verifiers.v1 as vf
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.clients.train import TrainClient
    from verifiers.v1.dialects.chat import ChatDialect
    from verifiers.v1.interception.server import InterceptionServer
    from verifiers.v1.session import RolloutSession

    request_payload = json.loads(request_bytes)
    if not isinstance(request_payload, dict):
        raise ValueError("historical application request is not an object")
    trace_id = f"historical-v{version}-{ordinal}"
    openai = SimpleNamespace(
        base_url="http://engine/v1",
        max_retries=0,
        post=AsyncMock(return_value=SimpleNamespace(content=raw_response)),
        close=AsyncMock(),
    )
    client = TrainClient(openai)
    client._pool = renderer

    def validate_action(
        request: Mapping[str, Any],
        message: Mapping[str, Any],
        action_token_ids: list[int] | tuple[int, ...],
    ) -> None:
        parsed = renderer.parse_response(
            list(action_token_ids), tools=request.get("tools") or None
        )
        if parsed.content != (message.get("content") or ""):
            raise ValueError("historical renderer content differs")
        parsed_calls = list(parsed.tool_calls)
        message_calls = list(message.get("tool_calls") or [])
        if len(parsed_calls) != len(message_calls):
            raise ValueError("historical renderer tool-call count differs")
        for parsed_call, message_call in zip(parsed_calls, message_calls, strict=True):
            function = message_call.get("function")
            if not isinstance(function, dict):
                raise ValueError("historical message tool wrapper differs")
            if parsed_call.name != function.get("name"):
                raise ValueError("historical renderer tool name differs")
            if parsed_call.arguments != json.loads(str(function.get("arguments"))):
                raise ValueError("historical renderer tool arguments differ")

    observer, ledger, producer = _observer(
        scratch / trace_id,
        trace_id=trace_id,
        eos_token_id=renderer.get_stop_token_ids()[0],
        validate_action=validate_action,
        manifests=manifests,
    )
    sampling = _sampling(vf, request_payload)
    session = RolloutSession(
        ModelContext(CHECKPOINT_ID, client, sampling),
        vf.Trace(
            id=trace_id,
            task=vf.TraceTask(type="HistoricalReplay", data=vf.TaskData(prompt="replay")),
        ),
        observer=StageDForwardDirectiveObserver(observer),
    )
    server = InterceptionServer()
    server.sessions["secret"] = session
    headers = multidict.CIMultiDict(
        {
            "Authorization": "Bearer secret",
            "X-RLM-Provenance-Version": "2",
            "X-RLM-Depth": "0",
            "X-RLM-Session-ID": "root-session",
            "X-RLM-Lineage": "root",
            "X-RLM-Session-Call-Ordinal": "0",
            "X-RLM-Turn": "0",
            "X-RLM-Call-Kind": "policy",
            "X-RLM-Completed-Episode-Spawn-Ordinals": "",
        }
    )
    request = SimpleNamespace(
        headers=headers,
        path="/v1/chat/completions",
        read=AsyncMock(return_value=request_bytes),
        _read_bytes=request_bytes,
    )
    try:
        response = await server.handle_request(request, ChatDialect())
        openai.post.assert_awaited_once()
        if response.status != 200:
            raise ValueError(
                f"historical v{version} response failed current path: "
                f"{response.status} {response.body.decode(errors='replace')}"
            )
        if len(producer._completed) != 1 or producer._pending:
            raise ValueError("historical replay did not close exactly one observer lifecycle")
        (observed,) = producer._completed.values()
    finally:
        await client.close()
        ledger.close()
    result: dict[str, Any] = {
        "version": version,
        "ordinal": ordinal,
        "raw_response_sha256": _sha256(raw_response),
        "current_action_digest": observed.action.digest,
        "current_key_digest": observed.action.key.digest,
        "finish_reason": observed.action.finish_reason,
        "termination_kind": observed.action.termination_kind,
        "parse_status": observed.action.parse_status,
        "completion_tokens": observed.action.completion_tokens,
        "request_max_tokens": observed.action.key.sampler.max_tokens,
        "semantic_renderer_observer_replay": True,
        "historical_completed_action_available": expected_action is not None,
        "current_observer_accepted": True,
    }
    if expected_action is not None:
        current_payload = observed.action.to_payload()
        historical_payload = expected_action.to_payload()
        current_payload.pop("key")
        historical_payload.pop("key")
        compared_fields = tuple(sorted(set(current_payload) | set(historical_payload)))
        changed_behavior_fields = [
            field
            for field in compared_fields
            if canonical_json(current_payload.get(field))
            != canonical_json(historical_payload.get(field))
        ]
        context_derived_fields = {"prompt_tokens"}
        changed_response_fields = sorted(
            set(changed_behavior_fields) - context_derived_fields
        )
        if changed_response_fields:
            differences = ", ".join(
                f"{field}={historical_payload.get(field)!r} -> "
                f"{current_payload.get(field)!r}"
                for field in changed_response_fields
            )
            raise ValueError(
                f"historical v{version}/{ordinal} completed action changed "
                f"response-derived behavior fields: {differences}"
            )
        result.update(
            {
                "historical_action_digest": expected_action.digest,
                "historical_key_digest": expected_action.key.digest,
                "all_non_key_behavior_fields_compared": list(compared_fields),
                "response_derived_non_key_fields_exact": True,
                "context_derived_non_key_fields_changed": changed_behavior_fields,
                "historical_prompt_tokens": expected_action.prompt_tokens,
                "fresh_context_prompt_tokens": observed.action.prompt_tokens,
                "fresh_context_action_digest_exact": (
                    observed.action.digest == expected_action.digest
                ),
                "fresh_context_key_digest_exact": (
                    observed.action.key.digest == expected_action.key.digest
                ),
                "fresh_context_key_fields_changed": sorted(
                    key
                    for key in set(observed.action.key.to_payload())
                    | set(expected_action.key.to_payload())
                    if canonical_json(observed.action.key.to_payload().get(key))
                    != canonical_json(expected_action.key.to_payload().get(key))
                ),
            }
        )
    return result


async def audit(
    repository: Path,
    tokenizer_path: Path,
    renderers_root: Path,
    verifiers_root: Path,
) -> dict[str, Any]:
    authentication = _verify_frozen_inputs(
        repository,
        tokenizer_path,
        renderers_root,
        verifiers_root,
    )
    import renderers
    import verifiers.v1 as vf
    from renderers.qwen3 import Qwen3Renderer
    from transformers import AutoTokenizer

    module_paths = {
        "renderers": Path(str(renderers.__file__)).resolve(),
        "verifiers.v1": Path(str(vf.__file__)).resolve(),
    }
    roots = {
        "renderers": renderers_root.resolve(),
        "verifiers.v1": verifiers_root.resolve(),
    }
    stack = _read_object(
        repository / "configs/stage-d/stage-d1-dependency-stack-v11-2.json"
    )
    imported_bindings = {
        item["name"]: item["sha256"] for item in stack["imported_modules"]
    }
    for name, path in module_paths.items():
        if not path.is_relative_to(roots[name]) or _sha256(path.read_bytes()) != (
            imported_bindings[name]
        ):
            raise ValueError(f"active {name} import differs from its frozen binding")
    authentication["active_imported_modules"] = {
        name: {"path": str(path), "sha256": _sha256(path.read_bytes())}
        for name, path in module_paths.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    renderer = Qwen3Renderer(tokenizer)
    manifests = {
        "base_model": (
            repository / "configs/stage-d/stage-d1-base-model-manifest.json"
        ).read_bytes(),
        "adapter": (
            repository / "reports/stage-d0-scaffold-step8-adapter-manifest-v1.json"
        ).read_bytes(),
        "tokenizer": (
            repository / "configs/stage-d/stage-d1-tokenizer-manifest.json"
        ).read_bytes(),
        "renderer": (
            repository / "configs/stage-d/stage-d1-renderer-manifest.json"
        ).read_bytes(),
        "sampler": (
            repository / "configs/stage-d/stage-d1-sampler-conformance-manifest.json"
        ).read_bytes(),
    }
    availability = {
        1: "no_model_response",
        2: "pre_forward_no_response",
        3: "raw_response_not_retained",
        4: "request_and_raw_response_no_completed_action",
        5: "no_response_campaign",
        6: "no_response_campaign",
        7: "completed_actions_and_raw_responses",
        8: "completed_actions_and_raw_responses",
        9: "completed_actions_and_raw_responses",
        10: "partial_episode_with_completed_actions_and_raw_responses",
    }
    snapshots = {
        version: _ledger_snapshot(repository, version) for version in (4, 7, 8, 9, 10)
    }
    replays: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="redco-stage-d-history-") as temporary:
        scratch = Path(temporary)
        v4_rows = snapshots[4]["rows"]
        if len(v4_rows) != 1 or "action" in v4_rows[0]:
            raise ValueError("v4 retained evidence lacks its request or raw response")
        v4 = v4_rows[0]
        v4_root = repository / "runs/stage-d/stage-d1-support-v4/ledger/evidence"
        replays.append(
            await _replay_one(
                scratch=scratch,
                renderer=renderer,
                manifests=manifests,
                version=4,
                ordinal=0,
                request_bytes=(v4_root / v4["request"]).read_bytes(),
                raw_response=(v4_root / v4["raw"]).read_bytes(),
                expected_action=None,
            )
        )
        for version in range(7, 11):
            root = repository / f"runs/stage-d/stage-d1-support-v{version}/ledger/evidence"
            for ordinal, evidence in enumerate(snapshots[version]["rows"]):
                action: BehaviorAction | None = None
                if "action" in evidence:
                    action_bytes = (root / evidence["action"]).read_bytes()
                    envelope = json.loads(action_bytes)
                    prompt = tuple(envelope["action"]["key"]["prompt_token_ids"])
                    action = BehaviorAction.from_bytes(
                        action_bytes,
                        validate_action=lambda _request, _message, _tokens: None,
                        render_prompt=lambda _request, prompt=prompt: prompt,
                    )
                    request_bytes = action.key.request
                else:
                    request_bytes = (root / evidence["request"]).read_bytes()
                replays.append(
                    await _replay_one(
                        scratch=scratch,
                        renderer=renderer,
                        manifests=manifests,
                        version=version,
                        ordinal=ordinal,
                        request_bytes=request_bytes,
                        raw_response=(root / evidence["raw"]).read_bytes(),
                        expected_action=action,
                    )
                )
    v10_length_replays = [
        item
        for item in replays
        if item["version"] == 10
        and not item["historical_completed_action_available"]
    ]
    if len(v10_length_replays) != 1 or (
        v10_length_replays[0]["finish_reason"],
        v10_length_replays[0]["termination_kind"],
    ) != ("length", "max_tokens"):
        raise ValueError("v10 retained length-capped response did not replay exactly")
    v10_length = v10_length_replays[0]
    if (
        not v10_length["current_observer_accepted"]
        or v10_length["completion_tokens"] != v10_length["request_max_tokens"]
        or not isinstance(v10_length.get("current_action_digest"), str)
    ):
        raise ValueError("v10 length-capped action did not complete durably")
    support_proxies = [
        snapshots[version]["support_proxy"] for version in range(7, 11)
    ]
    return {
        "schema_version": 1,
        "domain": "redco-stage-d-historical-semantic-replay-v1",
        "passes": True,
        "live_support_run_authorized": False,
        "scientific_training_authorized": False,
        "authorization_boundary": (
            "A CPU audit pass does not authorize provider or model calls. A newly "
            "hash-bound preregistration and independent review are required first."
        ),
        "scope": (
            "Authenticated response-ingestion, pinned-renderer, and observer compatibility "
            "under a fresh scratch root context."
        ),
        "historical_topology_replay_performed": False,
        "authentication": authentication,
        "availability": [
            {"version": version, "evidence": evidence}
            for version, evidence in availability.items()
        ],
        "historical_versions_semantically_replayed": sorted(
            {item["version"] for item in replays}
        ),
        "semantic_renderer_observer_replay_count": len(replays),
        "completed_action_replay_count": sum(
            "historical_action_digest" in item for item in replays
        ),
        "digest_interpretation": (
            "Every non-key BehaviorAction field is compared. Response-derived fields must "
            "match exactly. Prompt-token usage is disclosed as context-derived because the "
            "current replay intentionally creates a fresh prepared request; fresh-context "
            "action/key digests are therefore descriptive rather than historical identity."
        ),
        "unavailable_versions_were_not_reconstructed": [1, 2, 3, 5, 6],
        "support_density": {
            "confirmatory_probability_identifiable": False,
            "reason": (
                "The four retained post-SFT rollouts are partial, adaptively selected, "
                "and censored by different observer failures. None finalized a natural "
                "source rollout or observed a K=4 branch reward range."
            ),
            "descriptive_scaffold_proxy_successes": sum(
                item["descriptive_scaffold_proxy"] for item in support_proxies
            ),
            "descriptive_scaffold_proxy_rollouts": len(support_proxies),
            "N_eligible": None,
            "N_joint": None,
            "binomial_projection_authorized": False,
            "pre_sft_fewshot_result_is_not_applicable": "8/64",
            "unchanged_fresh_support_rule": {
                "rollouts": 64,
                "required_joint_successes": 58,
                "branch_count": 4,
                "minimum_reward_f1_range": 0.05,
            },
            "descriptive_rollouts": support_proxies,
        },
        "replays": replays,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--renderers-root", type=Path, required=True)
    parser.add_argument("--verifiers-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = canonical_json(
        asyncio.run(
            audit(
                args.repository,
                args.tokenizer_path,
                args.renderers_root,
                args.verifiers_root,
            )
        )
    )
    if args.output is not None:
        args.output.write_bytes(payload)
    print(payload.decode())


if __name__ == "__main__":
    main()
