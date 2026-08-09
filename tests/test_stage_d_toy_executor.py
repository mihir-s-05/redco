from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

import redco.analysis.stage_d_toy_executor as toy_executor
from redco.analysis.stage_d_exact_action import BehaviorAction, ExactActionKey
from redco.analysis.stage_d_receipt_ledger import (
    GenesisBinding,
    LedgerError,
    SealedReceiptVerifier,
    StageDReceiptLedger,
)
from redco.analysis.stage_d_scientific_branch_group import (
    BranchGroupArtifact,
    BranchGroupSpec,
    OutcomeKind,
    PreActionTargetCommitment,
    SeedCorrespondenceMap,
    run_scientific_branch_group,
)
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress, ScheduledSeed
from redco.analysis.stage_d_toy_executor import (
    ContentAddressedStore,
    DeterministicScorer,
    DurableCandidateSampler,
    GatewayActionResponse,
    GatewayContinuationResponse,
    ProvenanceResolver,
    ReplayResourceLimits,
    ScoredResult,
    ToyExecutionContext,
    ToySubprocessArmExecutor,
    WorkspaceFile,
    WorkspaceManifest,
    toy_worker_runtime_manifest,
    verify_workspace,
    workspace_manifest,
)
from redco.contracts import ActualEvaluationCost, canonical_json

MASTER_SEED = "toy-executor-master"
WORKER_PYTHON = str(getattr(sys, "_base_executable", sys.executable))
MATCHED = PolicyEventAddress(0, "root", 2, 2)
DYNAMIC = PolicyEventAddress(1, "root/dynamic", 0, 0)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_expired_callback_deadline_never_dispatches() -> None:
    called = False

    def callback() -> None:
        nonlocal called
        called = True

    with pytest.raises(RuntimeError, match="wall-time budget exhausted"):
        toy_executor._call_with_deadline(callback, time.monotonic() - 1.0)
    assert not called


def _conformance() -> bytes:
    payload = {
        "schema_version": 1,
        "analysis": "served-stack-categorical-logprob-conformance-v1",
        "passes": True,
        "logprob_semantics": "served_chosen_token_post_transform",
        "categorical_case_count": 3,
        "served_stack_sha256": "a" * 64,
        "tool_call_termination_includes_all_generated_tokens": True,
        "eos_is_included_in_action_tokens_and_logprobs": True,
    }
    payload["signed_payload_sha256"] = _sha256(canonical_json(payload))
    return canonical_json(payload)


def _request(seed: int) -> dict[str, object]:
    return {
        "model": "model@commit",
        "messages": [{"role": "user", "content": "q"}],
        "tools": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": None,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "logit_bias": {},
        "seed": seed,
        "max_tokens": 2,
        "stop": None,
        "n": 1,
        "best_of": None,
        "use_beam_search": False,
        "logprobs": True,
        "top_logprobs": 0,
        "ignore_eos": False,
        "min_tokens": 0,
        "extra_body": {"cache_salt": f"seed-{seed}"},
    }


def _key(seed: int, *, cache_salt_prefix: str = "seed") -> ExactActionKey:
    return ExactActionKey.build(
        checkpoint_id="model@commit",
        base_model_manifest=b"base",
        adapter_manifest=b"adapter",
        tokenizer_manifest=b"tokenizer",
        renderer_manifest=b"renderer",
        sampler_conformance_manifest=_conformance(),
        action_selection_policy="direct_single_sample",
        transport_retry_policy="fail_before_action_no_resample",
        request={
            **_request(seed),
            "extra_body": {"cache_salt": f"{cache_salt_prefix}-{seed}"},
        },
        prompt_token_ids=(10, 11),
        render_prompt=lambda _: (10, 11),
    )


def _action(seed: int, *, cache_salt_prefix: str = "seed") -> BehaviorAction:
    return BehaviorAction.build(
        key=_key(seed, cache_salt_prefix=cache_salt_prefix),
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


def _binding(*, runtime_sha256: str, config_sha256: str) -> GenesisBinding:
    return GenesisBinding(
        preregistration_sha256="1" * 64,
        source_sha256="2" * 64,
        runtime_sha256=runtime_sha256,
        config_sha256=config_sha256,
        protocol_manifest_sha256="5" * 64,
        master_seed_sha256=_sha256(MASTER_SEED.encode()),
        support_rules_sha256="6" * 64,
    )


@dataclass
class FakeGateway:
    runtime_sha256: str
    config_sha256: str
    candidate_calls: list[int] = field(default_factory=list)
    dispatch_ids: list[str] = field(default_factory=list)
    continuation_calls: list[tuple[dict[str, Any], ScheduledSeed]] = field(default_factory=list)
    fail_continuation: bool = False
    wrong_candidate_request: bool = False
    wrong_continuation_binding: bool = False
    candidate_delay: float = 0.0
    continuation_delay: float = 0.0

    def sample_action(
        self,
        *,
        reference_key: ExactActionKey,
        action_seed: int,
        exact_request: Mapping[str, Any],
        dispatch_id: str,
    ) -> GatewayActionResponse:
        if self.candidate_delay:
            time.sleep(self.candidate_delay)
        assert reference_key == _key(17)
        assert exact_request == {
            **_request(action_seed),
            "extra_body": {"cache_salt": f"candidate-{action_seed}"},
        }
        self.candidate_calls.append(action_seed)
        self.dispatch_ids.append(dispatch_id)
        action = _action(
            action_seed,
            cache_salt_prefix=("seed" if self.wrong_candidate_request else "candidate"),
        )
        return GatewayActionResponse(
            action,
            canonical_json({"action_sha256": action.digest, "dispatch_id": dispatch_id}),
        )

    def continue_policy(
        self,
        semantic_request: Mapping[str, Any],
        *,
        scheduled_seed: ScheduledSeed,
        dispatch_id: str,
    ) -> GatewayContinuationResponse:
        self.dispatch_ids.append(dispatch_id)
        self.continuation_calls.append((dict(semantic_request), scheduled_seed))
        if self.continuation_delay:
            time.sleep(self.continuation_delay)
        if self.fail_continuation:
            raise RuntimeError("gateway crashed after write-ahead dispatch")
        return GatewayContinuationResponse(
            request_sha256=_sha256(
                canonical_json(
                    {
                        "semantic_request": semantic_request,
                        "scheduled_seed": {
                            "seed": scheduled_seed.seed,
                            "coupling_mode": scheduled_seed.coupling_mode.value,
                            "address": {
                                **scheduled_seed.address.as_payload(),
                                "turn": scheduled_seed.address.turn,
                            },
                        },
                    }
                )
            ),
            scheduled_seed=scheduled_seed,
            dispatch_id=("wrong-dispatch" if self.wrong_continuation_binding else dispatch_id),
            prompt_tokens=3,
            completion_tokens=2,
            raw_response=canonical_json(
                {
                    "dispatch_id": dispatch_id,
                    "request": semantic_request,
                    "seed": scheduled_seed.seed,
                }
            ),
        )


def _resolve(provenance: Mapping[str, Any]) -> PolicyEventAddress:
    if provenance == {"kind": "matched"}:
        return MATCHED
    if provenance == {"kind": "dynamic"}:
        return DYNAMIC
    raise ValueError("untrusted worker provenance did not resolve")


@dataclass
class FakeScorer:
    artifact_sha256: str
    delay_seconds: float = 0.0
    fail: bool = False
    answers: list[str] = field(default_factory=list)

    def __call__(
        self,
        *,
        action: BehaviorAction,
        continuations: Sequence[GatewayContinuationResponse],
        worker_result: Mapping[str, Any],
    ) -> ScoredResult:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.fail:
            raise RuntimeError("trusted scorer failure")
        self.answers.append(str(worker_result["answer"]))
        reward = float(len(continuations) + action.key.sampler.seed % 2)
        evidence = canonical_json(
            {
                "action": action.digest,
                "continuations": len(continuations),
                "worker_result_sha256": _sha256(canonical_json(worker_result)),
                "reward": reward,
            }
        )
        return ScoredResult(reward, evidence)


class CrashResolver:
    def __call__(self, provenance: Mapping[str, Any]) -> PolicyEventAddress:
        del provenance
        raise RuntimeError("trusted resolver failure")


@dataclass
class Fixture:
    writer: StageDReceiptLedger
    spec: BranchGroupSpec
    context: ToyExecutionContext
    gateway: FakeGateway
    scorer: FakeScorer
    cas: ContentAddressedStore
    worker: Path
    scratch: Path


def _worker_source() -> str:
    return """
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

mode, request_name, output_name = sys.argv[1:]
request = pathlib.Path(request_name).read_bytes()
digest = hashlib.sha256(request).hexdigest()
if mode == "timeout":
    time.sleep(5)
if mode == "nonzero":
    raise SystemExit(7)
events = [{"provenance": {"kind": "matched"}, "request": {"prompt": "go"}}]
terminal = False
answer = "ok"
if mode == "two":
    events = [
        {"provenance": {"kind": "dynamic"}, "request": {"prompt": "dynamic"}},
        {"provenance": {"kind": "matched"}, "request": {"prompt": "matched"}},
    ]
elif mode == "terminal":
    events = []
    terminal = True
elif mode == "empty":
    events = []
elif mode == "duplicate":
    events = [events[0], events[0]]
elif mode == "forged":
    events = [{"provenance": {"kind": "matched"}, "request": {}, "seed": 123}]
elif mode == "transport":
    events = [{"provenance": {"kind": "matched"}, "request": {"model": "evil"}}]
elif mode == "forged_result":
    pass
elif mode == "mutate":
    pathlib.Path("input.txt").write_text("mutated", encoding="utf-8")
elif mode == "directory":
    pathlib.Path(output_name).mkdir()
    raise SystemExit(0)
elif mode == "huge_workspace":
    with pathlib.Path("huge.bin").open("wb") as handle:
        handle.truncate(10 * 1024 * 1024)
elif mode == "many_entries":
    for index in range(100):
        pathlib.Path(f"empty-{index}.txt").touch()
elif mode == "many_calls":
    events = events * 100
elif mode == "descendant":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    answer = f"{os.getpid()}:{child.pid}"
elif mode == "orphaned_descendant":
    pid_file = pathlib.Path(output_name).with_suffix(".pid")
    intermediate_source = (
        "import pathlib,subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))"
    )
    intermediate = subprocess.Popen([
        sys.executable,
        "-c",
        intermediate_source,
        str(pid_file),
    ])
    intermediate.wait()
    answer = f"{os.getpid()}:{pid_file.read_text()}"
payload = {
    "schema_version": 1,
    "request_sha256": digest,
    "downstream_events": events,
    "terminal_without_downstream": terminal,
    "worker_result": {"answer": answer},
}
if mode == "forged_result":
    payload["worker_result"] = {"reward": 999}
pathlib.Path(output_name).write_bytes(
    json.dumps(
        payload if mode != "oversized" else {**payload, "worker_result": {"blob": "x" * 10000}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
)
""".strip()


def _make_fixture(
    tmp_path: Path,
    *,
    limits: ReplayResourceLimits | None = None,
) -> Fixture:
    tmp_path.mkdir(parents=True, exist_ok=True)
    worker = tmp_path / "worker.py"
    worker.write_text(_worker_source(), encoding="utf-8")
    cas = ContentAddressedStore(tmp_path / "cas")
    runtime_sha256 = cas.put(toy_worker_runtime_manifest(Path(WORKER_PYTHON), worker, cas=cas))
    config_sha256 = cas.put(b"frozen-config")
    scorer_sha256 = cas.put(b"frozen-scorer")
    writer = StageDReceiptLedger.create(
        tmp_path / "ledger",
        binding=_binding(
            runtime_sha256=runtime_sha256,
            config_sha256=config_sha256,
        ),
        master_seed=MASTER_SEED,
    )
    recorded = _action(17)
    snapshot_sha = writer.put_evidence(b"pre-action-snapshot")
    reservation = writer.commit_pre_action_and_reserve(
        group_id="group-1",
        rollout_id="rollout-1",
        target_roster=("target-0",),
        target_ordinal=0,
        target_id="target-0",
        target_address=PolicyEventAddress(1, "root/child", 0, 0),
        pre_action_snapshot_sha256=snapshot_sha,
        recorded_action_key=recorded.key,
        branch_count=2,
        continuation_replicates=1,
        failure_reward=-1.0,
    )
    request_sha = writer.put_evidence(recorded.key.request)
    source_reservation = writer.reserve_source_policy_call(
        group_id="group-1",
        rollout_id="rollout-1",
        decision_id="child-decision-0",
        node_kind="child",
        target_id="target-0",
        target_ordinal=0,
        target_address=PolicyEventAddress(1, "root/child", 0, 0),
        recorded_action_key=recorded.key,
        request_sha256=request_sha,
        branch_selected=True,
        recorded_action_reservation=reservation,
    )
    writer.mark_recorded_action_model_call_started(
        reservation,
        request_sha256=request_sha,
    )
    response_sha = writer.put_evidence(recorded.to_bytes())
    writer.complete_recorded_action(
        reservation,
        action=recorded,
        response_sha256=response_sha,
    )
    source_completion = writer.complete_source_policy_call(
        source_reservation,
        action=recorded,
        response_sha256=response_sha,
    )
    correspondence_evidence = writer.put_evidence(b"frozen-correspondence")
    correspondence_receipt = writer.freeze_correspondence(
        group_id="group-1",
        target_id="target-0",
        recorded_action=recorded,
        matched_addresses=(MATCHED,),
        evidence_sha256=correspondence_evidence,
    )
    trace = writer.put_evidence(b"raw-trace")
    reward = writer.put_evidence(b"reward-evidence")
    stock = writer.put_evidence(b"stock-sequences")
    writer.record_source_rollout_completed(
        group_id="group-1",
        rollout_id="rollout-1",
        source_sha256="5" * 64,
        trace_sha256=trace,
        reward_evidence_sha256=reward,
        stock_sequences_evidence_sha256=stock,
        base_model_manifest_sha256="6" * 64,
        decision_ids=("child-decision-0",),
        decision_completion_receipt_sha256s=(_sha256(source_completion),),
    )
    writer.record_branch_target_roster(
        canonical_json(
            {
                "schema_version": 2,
                "domain": "redco-stage-d-branch-target-roster-v2",
                "planned_source_count": 1,
                "completed_source_count": 1,
                "eligible_source_count": 1,
                "ineligible_source_count": 0,
                "minimum_eligible_sources": 1,
                "eligibility_passed": True,
                "source_sha256s": ["5" * 64],
                "targets": [
                    {
                        "source_sha256": "5" * 64,
                        "group_id": "group-1",
                        "rollout_id": "rollout-1",
                        "decision_id": "child-decision-0",
                        "target_id": "target-0",
                        "target_ordinal": 0,
                        "event_address": {
                            **PolicyEventAddress(1, "root/child", 0, 0).as_payload(),
                            "turn": 0,
                        },
                    }
                ],
                "excluded_targets": [],
            }
        )
    )
    commitment = PreActionTargetCommitment.from_receipt(
        reservation.commitment_receipt,
        verifier=writer,
    )
    correspondence = SeedCorrespondenceMap.from_receipt(
        correspondence_receipt,
        verifier=writer,
        commitment=commitment,
        recorded_action=recorded,
    )
    spec = BranchGroupSpec(commitment, recorded, correspondence, MASTER_SEED)
    source_workspace = tmp_path / "source-workspace"
    source_workspace.mkdir()
    (source_workspace / "input.txt").write_text("frozen", encoding="utf-8")
    manifest = workspace_manifest(source_workspace, cas)
    context = ToyExecutionContext(
        spec=spec,
        runtime_sha256=runtime_sha256,
        config_sha256=config_sha256,
        scorer_sha256=scorer_sha256,
        workspace=manifest,
        limits=limits
        or ReplayResourceLimits(
            wall_seconds=1.0,
            policy_calls=4,
            prompt_tokens=32,
            completion_tokens=32,
            worker_output_bytes=4096,
            workspace_snapshot_bytes=4096,
            workspace_entries=64,
        ),
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    gateway = FakeGateway(runtime_sha256, config_sha256)
    scorer = FakeScorer(scorer_sha256)
    return Fixture(writer, spec, context, gateway, scorer, cas, worker, scratch)


def _qa(fixture: Fixture) -> bytes:
    report = fixture.writer.put_evidence(b"reconstruction-qa")
    receipt = fixture.writer.record_reconstruction_qa(
        group_id="group-1",
        target_id="target-0",
        recorded_action=fixture.spec.recorded_action,
        passed=True,
        report_sha256=report,
        actual_cost=ActualEvaluationCost(cpu_seconds=0.01, wall_seconds=0.01),
    )
    fixture.writer.seal_reconstruction_qa_barrier()
    return receipt


def _executor(
    fixture: Fixture,
    mode: str,
    *,
    resolver: ProvenanceResolver = _resolve,
    scorer: DeterministicScorer | None = None,
) -> ToySubprocessArmExecutor:
    return ToySubprocessArmExecutor(
        writer=fixture.writer,
        gateway=fixture.gateway,
        provenance_resolver=resolver,
        scorer=scorer or fixture.scorer,
        cas=fixture.cas,
        context=fixture.context,
        worker_command=(WORKER_PYTHON, str(fixture.worker), mode),
        scratch_root=fixture.scratch,
    )


def _run(
    fixture: Fixture,
    mode: str,
    *,
    resolver: ProvenanceResolver = _resolve,
    scorer: DeterministicScorer | None = None,
    executor: ToySubprocessArmExecutor | None = None,
) -> BranchGroupArtifact:
    return run_scientific_branch_group(
        fixture.spec,
        verifier=fixture.writer,
        sample_candidate=DurableCandidateSampler(
            fixture.writer,
            fixture.gateway,
            group_id="group-1",
            target_id="target-0",
            wall_seconds=fixture.context.limits.wall_seconds,
        ),
        run_reconstruction_qa=lambda _: _qa(fixture),
        execute_arm=executor or _executor(fixture, mode, resolver=resolver, scorer=scorer),
    )


def test_full_c1_group_runs_through_toy_subprocess_and_sealed_ledger(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    artifact = _run(fixture, "two")

    assert len(fixture.gateway.candidate_calls) == 1
    assert len(fixture.gateway.continuation_calls) == 4
    assert len(fixture.gateway.dispatch_ids) == len(set(fixture.gateway.dispatch_ids)) == 5
    dynamic_seeds = [fixture.gateway.continuation_calls[index][1].seed for index in (0, 2)]
    matched_seeds = [fixture.gateway.continuation_calls[index][1].seed for index in (1, 3)]
    assert dynamic_seeds[0] != dynamic_seeds[1]
    assert matched_seeds[0] == matched_seeds[1]
    assert {
        fixture.gateway.continuation_calls[index][1].coupling_mode.value for index in (0, 2)
    } == {"exogenous"}
    assert {
        fixture.gateway.continuation_calls[index][1].coupling_mode.value for index in (1, 3)
    } == {"paired"}
    assert artifact.ledger.actual_action_generation_calls == 1
    assert artifact.ledger.actual_downstream_policy_calls == 4
    assert not any(fixture.scratch.iterdir())

    encoded = artifact.to_bytes()
    artifact_sha = fixture.writer.put_evidence(encoded)
    fixture.writer.claim_training_batch(
        training_batch_identity=artifact.training_batch_identity,
        artifact_sha256s=(artifact_sha,),
        consumer_id="toy-trainer",
    )
    seal = fixture.writer.seal()
    loaded = BranchGroupArtifact.verify_bytes(
        encoded,
        verifier=SealedReceiptVerifier(tmp_path / "ledger", seal),
        encode_action=lambda _request, _message: (20, 2),
        render_prompt=lambda _: (10, 11),
        master_seed=MASTER_SEED,
    )
    assert loaded == artifact


def test_terminal_without_downstream_is_a_valid_scientific_outcome(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    artifact = _run(fixture, "terminal")

    assert fixture.gateway.continuation_calls == []
    assert all(
        outcome.kind is OutcomeKind.TERMINAL_WITHOUT_DOWNSTREAM
        for arm in artifact.arms
        for outcome in arm.outcomes
    )


@pytest.mark.parametrize(
    ("mode", "kind"),
    [
        ("timeout", OutcomeKind.TIMEOUT),
        ("nonzero", OutcomeKind.RUNTIME_EXCEPTION),
        ("empty", OutcomeKind.RUNTIME_EXCEPTION),
        ("forged", OutcomeKind.RUNTIME_EXCEPTION),
        ("transport", OutcomeKind.RUNTIME_EXCEPTION),
        ("forged_result", OutcomeKind.RUNTIME_EXCEPTION),
        ("mutate", OutcomeKind.RUNTIME_EXCEPTION),
        ("directory", OutcomeKind.RUNTIME_EXCEPTION),
        ("oversized", OutcomeKind.RESOURCE_LIMIT),
        ("huge_workspace", OutcomeKind.RESOURCE_LIMIT),
        ("many_entries", OutcomeKind.RESOURCE_LIMIT),
    ],
)
def test_pre_dispatch_worker_failures_are_retained_in_denominator(
    tmp_path: Path,
    mode: str,
    kind: OutcomeKind,
) -> None:
    fixture = _make_fixture(tmp_path)
    artifact = _run(fixture, mode)

    assert all(outcome.kind is kind for arm in artifact.arms for outcome in arm.outcomes)
    assert all(outcome.reward == -1.0 for arm in artifact.arms for outcome in arm.outcomes)


def test_duplicate_event_address_poisoning_is_not_recast_as_a_clean_failure(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)

    with pytest.raises(Exception, match="executor exception"):
        _run(fixture, "duplicate")
    with pytest.raises(LedgerError, match="dangling"):
        fixture.writer.seal()


def test_gateway_crash_after_write_ahead_dispatch_leaves_nonrepairable_attempt(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    fixture.gateway.fail_continuation = True

    with pytest.raises(Exception, match="executor exception"):
        _run(fixture, "success")
    with pytest.raises(LedgerError, match="dangling"):
        fixture.writer.seal()


def test_candidate_gateway_must_execute_the_exact_ledgered_request(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    fixture.gateway.wrong_candidate_request = True

    with pytest.raises(Exception, match="candidate slot"):
        _run(fixture, "success")
    with pytest.raises(LedgerError, match="dangling"):
        fixture.writer.seal()


def test_candidate_gateway_must_match_ledger_genesis(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    mismatched = FakeGateway("f" * 64, fixture.context.config_sha256)

    with pytest.raises(ValueError, match="ledger genesis binding"):
        DurableCandidateSampler(
            fixture.writer,
            mismatched,
            group_id="group-1",
            target_id="target-0",
            wall_seconds=1.0,
        )


def test_candidate_gateway_binding_is_rechecked_at_dispatch(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    sampler = DurableCandidateSampler(
        fixture.writer,
        fixture.gateway,
        group_id="group-1",
        target_id="target-0",
        wall_seconds=1.0,
    )
    fixture.gateway.runtime_sha256 = "f" * 64

    with pytest.raises(ValueError, match="ledger genesis binding"):
        sampler(
            action_slot=1,
            action_seed=19,
            reference_key=fixture.spec.recorded_action.key,
        )


def test_hung_candidate_gateway_returns_control_and_leaves_dangling_call(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    fixture.gateway.candidate_delay = 5.0

    started = time.monotonic()
    with pytest.raises(Exception, match="candidate slot"):
        _run(fixture, "success")
    assert time.monotonic() - started < 3.0
    with pytest.raises(LedgerError, match="closed"):
        fixture.writer.seal()


def test_call_count_limit_precedes_any_provenance_callback(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    artifact = _run(fixture, "many_calls", resolver=CrashResolver())

    assert fixture.gateway.continuation_calls == []
    assert all(
        outcome.kind is OutcomeKind.RESOURCE_LIMIT
        for arm in artifact.arms
        for outcome in arm.outcomes
    )


def test_executor_callable_binding_is_rechecked_before_each_arm(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    executor = _executor(fixture, "success")
    fixture.scorer.artifact_sha256 = "f" * 64

    with pytest.raises(Exception, match="executor exception"):
        _run(fixture, "success", executor=executor)
    assert fixture.gateway.continuation_calls == []


def test_continuation_must_echo_the_ledgered_schedule(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    fixture.gateway.wrong_continuation_binding = True

    with pytest.raises(Exception, match="executor exception"):
        _run(fixture, "success")
    with pytest.raises(LedgerError, match="dangling"):
        fixture.writer.seal()


def test_resolver_and_scorer_runtime_errors_are_retained_when_no_call_is_dangling(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path / "resolver")

    resolver_artifact = _run(fixture, "success", resolver=CrashResolver())
    assert all(
        outcome.kind is OutcomeKind.RUNTIME_EXCEPTION
        for arm in resolver_artifact.arms
        for outcome in arm.outcomes
    )

    fixture = _make_fixture(tmp_path / "scorer")

    scorer_artifact = _run(
        fixture,
        "success",
        scorer=FakeScorer(fixture.context.scorer_sha256, fail=True),
    )
    assert all(
        outcome.kind is OutcomeKind.RUNTIME_EXCEPTION
        for arm in scorer_artifact.arms
        for outcome in arm.outcomes
    )
    assert scorer_artifact.ledger.actual_downstream_policy_calls == 2


def test_supervisor_callback_timeout_poison_stops_the_group(tmp_path: Path) -> None:
    fixture = _make_fixture(
        tmp_path,
        limits=ReplayResourceLimits(
            wall_seconds=0.5,
            policy_calls=4,
            prompt_tokens=32,
            completion_tokens=32,
            worker_output_bytes=4096,
            workspace_snapshot_bytes=4096,
            workspace_entries=64,
        ),
    )

    started = time.monotonic()
    with pytest.raises(Exception, match="executor exception"):
        _run(
            fixture,
            "success",
            scorer=FakeScorer(fixture.context.scorer_sha256, delay_seconds=5.0),
        )
    assert time.monotonic() - started < 3.0
    with pytest.raises(LedgerError, match="closed"):
        fixture.writer.seal()


def test_hung_gateway_returns_control_but_keeps_dispatch_dangling(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(
        tmp_path,
        limits=ReplayResourceLimits(
            wall_seconds=1.0,
            policy_calls=4,
            prompt_tokens=32,
            completion_tokens=32,
            worker_output_bytes=4096,
            workspace_snapshot_bytes=4096,
            workspace_entries=64,
        ),
    )
    fixture.gateway.continuation_delay = 5.0
    started = time.monotonic()
    with pytest.raises(Exception, match="executor exception"):
        _run(fixture, "success")
    assert time.monotonic() - started < 3.0
    with pytest.raises(LedgerError, match="closed"):
        fixture.writer.seal()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process-tree contract")
def test_normal_worker_exit_terminates_descendants_before_next_arm(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    artifact = _run(fixture, "descendant")

    kinds = [outcome.kind for arm in artifact.arms for outcome in arm.outcomes]
    assert kinds == [OutcomeKind.SUCCESS, OutcomeKind.SUCCESS], (
        fixture.scorer.answers,
        list(fixture.scratch.iterdir()),
    )
    roots_and_pids = [tuple(map(int, value.split(":"))) for value in fixture.scorer.answers]
    pids = [pid for _, pid in roots_and_pids]
    for pid in pids:
        probe = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10.0,
        )
        assert f',"{pid}",' not in probe.stdout
    assert not any(fixture.scratch.iterdir())


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process-tree contract")
def test_orphaned_grandchild_is_killed_by_worker_job(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    artifact = _run(fixture, "orphaned_descendant")

    assert all(
        outcome.kind is OutcomeKind.SUCCESS for arm in artifact.arms for outcome in arm.outcomes
    )
    pids = [int(value.split(":")[1]) for value in fixture.scorer.answers]
    for pid in pids:
        probe = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10.0,
        )
        assert f',"{pid}",' not in probe.stdout
    assert not any(fixture.scratch.iterdir())


def test_cas_and_workspace_validation_fail_closed(tmp_path: Path) -> None:
    cas = ContentAddressedStore(tmp_path / "cas")
    digest = cas.put(b"trusted")
    (cas.root / digest).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        cas.read_verified(digest)

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "data.txt").write_text("data", encoding="utf-8")
    manifest = workspace_manifest(root, cas=ContentAddressedStore(tmp_path / "cas-2"))
    (root / "data.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        verify_workspace(root, manifest)


def test_workspace_contract_rejects_paths_symlinks_and_wrong_ledger(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="beneath"):
        WorkspaceFile("../escape", "1" * 64, 0o644)
    with pytest.raises(ValueError, match="unique nonempty"):
        WorkspaceManifest.build(())

    if hasattr(Path, "symlink_to"):
        root = tmp_path / "links"
        root.mkdir()
        target = root / "target"
        target.write_text("x", encoding="utf-8")
        link = root / "link"
        try:
            link.symlink_to(target)
        except OSError:
            pass
        else:
            with pytest.raises(ValueError, match="symlinks"):
                workspace_manifest(root, ContentAddressedStore(tmp_path / "link-cas"))

    if sys.platform == "win32":
        junction_root = tmp_path / "junctions"
        junction_root.mkdir()
        junction_target = tmp_path / "junction-target"
        junction_target.mkdir()
        junction = junction_root / "junction"
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction), str(junction_target)],
            capture_output=True,
            check=False,
            text=True,
            timeout=10.0,
        )
        if created.returncode == 0:
            with pytest.raises(ValueError, match="reparse points"):
                workspace_manifest(
                    junction_root,
                    ContentAddressedStore(tmp_path / "junction-cas"),
                )

    first = _make_fixture(tmp_path / "first")
    second_writer = StageDReceiptLedger.create(
        tmp_path / "second-ledger",
        binding=first.writer.genesis_binding,
        master_seed=MASTER_SEED,
    )
    with pytest.raises(ValueError, match="different durable ledger"):
        ToySubprocessArmExecutor(
            writer=second_writer,
            gateway=first.gateway,
            provenance_resolver=_resolve,
            scorer=first.scorer,
            cas=first.cas,
            context=first.context,
            worker_command=(WORKER_PYTHON, str(first.worker), "success"),
            scratch_root=first.scratch,
        )
    first.writer.close()
    second_writer.close()


def test_runtime_config_and_scorer_bytes_are_cas_and_genesis_bound(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    wrong = replace(fixture.context, runtime_sha256=fixture.context.scorer_sha256)
    with pytest.raises(ValueError, match="genesis binding"):
        ToySubprocessArmExecutor(
            writer=fixture.writer,
            gateway=fixture.gateway,
            provenance_resolver=_resolve,
            scorer=fixture.scorer,
            cas=fixture.cas,
            context=wrong,
            worker_command=(WORKER_PYTHON, str(fixture.worker), "success"),
            scratch_root=fixture.scratch,
        )

    fixture.worker.write_text("raise SystemExit(0)", encoding="utf-8")
    executor = ToySubprocessArmExecutor(
        writer=fixture.writer,
        gateway=fixture.gateway,
        provenance_resolver=_resolve,
        scorer=fixture.scorer,
        cas=fixture.cas,
        context=fixture.context,
        worker_command=(WORKER_PYTHON, str(fixture.worker), "success"),
        scratch_root=fixture.scratch,
    )
    artifact = _run(fixture, "success", executor=executor)
    assert all(
        outcome.kind is OutcomeKind.SUCCESS for arm in artifact.arms for outcome in arm.outcomes
    )

    fixture = _make_fixture(tmp_path / "scorer")
    scorer_path = fixture.cas.root / fixture.context.scorer_sha256
    scorer_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        ToySubprocessArmExecutor(
            writer=fixture.writer,
            gateway=fixture.gateway,
            provenance_resolver=_resolve,
            scorer=fixture.scorer,
            cas=fixture.cas,
            context=fixture.context,
            worker_command=(WORKER_PYTHON, str(fixture.worker), "success"),
            scratch_root=fixture.scratch,
        )
    fixture.writer.close()


def test_cas_is_reverified_immediately_before_each_arm_launch(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    executor = _executor(fixture, "success")
    config_path = fixture.cas.root / fixture.context.config_sha256
    config_path.write_bytes(b"changed-after-construction")

    with pytest.raises(Exception, match="executor exception"):
        _run(fixture, "success", executor=executor)
    assert fixture.gateway.continuation_calls == []
