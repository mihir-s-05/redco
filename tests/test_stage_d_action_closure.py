from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from redco.analysis import stage_d_collection as collection
from redco.analysis.stage_d_action_closure import (
    ABORT_DISPOSITIONS,
    ACCEPTED_TERMINATION_KINDS,
    WATCHDOG_PHASES,
    ActionClosureWatchdog,
    WatchdogDeadlines,
    action_closure_case_manifest,
    atomic_terminal_record_writer,
    audit_raw_response_fixtures,
    build_action_closure_cases,
    mutate_one_field,
    stop_owned_runtime,
    terminate_process_then_kill,
)
from redco.analysis.stage_d_exact_action import BehaviorAction
from redco.analysis.stage_d_source_contracts import SourceRollout
from redco.analysis.stage_d_v13_support_contract import sampling_contract_binding
from redco.contracts import canonical_json


def test_bounded_closure_manifest_is_stable_and_complete() -> None:
    first = action_closure_case_manifest()
    second = action_closure_case_manifest()
    assert first == second
    assert len(build_action_closure_cases()) == 17
    assert first["manifest_sha256"]
    assert set(first["accepted_termination_kinds"]) == set(ACCEPTED_TERMINATION_KINDS)
    assert set(first["abort_dispositions"]) == set(ABORT_DISPOSITIONS)
    assert all(case["disposition"] in {"accept", "abort", "reject"} for case in first["cases"])


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "field", ["finish_reason", "usage", "tool_argument_bytes", "address"]
)
def test_one_field_mutations_are_not_silent(field: str) -> None:
    original = {
        "finish_reason": "stop",
        "completion_tokens": 2,
        "tool_arguments": "{}",
        "address": "a",
    }
    mutated = mutate_one_field(original, field)
    assert mutated != original
    assert original == {
        "finish_reason": "stop",
        "completion_tokens": 2,
        "tool_arguments": "{}",
        "address": "a",
    }


def _run_real_source_owner_case(
    root: Path,
    case_id: str,
    *,
    mutate_trace: Callable[[dict[str, object]], None] | None = None,
) -> SourceRollout:
    """Use the existing producer/ledger path for one closure vector."""

    from test_stage_d_source_producer import (
        _binding,
        _child_address,
        _episode,
        _prepared_action,
        _target_id,
        _tool_action,
    )

    from redco.analysis.stage_d_exact_action import BehaviorAction
    from redco.analysis.stage_d_receipt_ledger import StageDReceiptLedger
    from redco.analysis.stage_d_source_producer import StageDSourceRolloutProducer
    from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress

    factories = {
        "eos": lambda seed: _prepared_action(seed),
        "exact-cap": lambda seed: _prepared_action(seed, max_tokens=True),
        "tool-calls": lambda seed: _tool_action(seed),
        "empty-string-content": lambda seed: _prepared_action(
            seed, message={"role": "assistant", "content": ""}
        ),
        "textual-refusal": lambda seed: _prepared_action(
            seed, message={"role": "assistant", "content": "I cannot comply."}
        ),
        "multi-turn-child": lambda seed: _prepared_action(seed),
        "concurrent-child-order-a": lambda seed: _prepared_action(seed),
        "concurrent-child-order-b": lambda seed: _prepared_action(seed),
    }
    action_factory = cast(Callable[[int], BehaviorAction], factories[case_id])
    actions = (action_factory(71), action_factory(72))
    for action in actions:
        reloaded = BehaviorAction.from_bytes(
            action.to_bytes(),
            validate_action=lambda _request, _message, _tokens: None,
            render_prompt=lambda _request: (10, 11),
        )
        assert reloaded.to_bytes() == action.to_bytes()
    episode = json.loads(_episode())
    trace = episode["traces"][0]
    for node_index, call_index, action in zip((1, 3), (0, 1), actions, strict=True):
        node = trace["nodes"][node_index]
        node["message"] = action.message
        node["token_ids"] = list(action.action_token_ids)
        node["mask"] = [True] * len(action.action_token_ids)
        node["logprobs"] = list(action.behavior_logprobs)
        trace["calls"][call_index]["finish_reason"] = action.finish_reason
        trace["calls"][call_index]["usage"]["completion_tokens"] = action.completion_tokens
    if mutate_trace is not None:
        mutate_trace(trace)
    writer = StageDReceiptLedger.create(
        root,
        binding=_binding(),
        master_seed="stage-d-source-producer-test",
    )
    producer = StageDSourceRolloutProducer(
        ledger=writer,
        group_id="group-1",
        rollout_id="rollout-live",
        child_target_roster=(_target_id(),),
        allow_test_fixture_roster=True,
        base_model_manifest_sha256="a" * 64,
    )
    for event_address, target_id, node_kind, action in (
        (PolicyEventAddress(0, "root", 0, 0), None, "root", actions[0]),
        (_child_address(), _target_id(), "child", actions[1]),
    ):
        pending = producer.reserve_policy_call(
            event_address=event_address,
            action_key=action.key,
            node_kind=node_kind,
            target_id=target_id,
            branch_selected=False,
            raw_response_required=True,
        )
        producer.mark_policy_response_observed(pending, response_content=b"response-bytes")
        producer.complete_policy_call(pending, action=action)
    return cast(SourceRollout, producer.finalize_episode(canonical_json(episode)))


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "case_id",
    [
        "eos",
        "exact-cap",
        "tool-calls",
        "empty-string-content",
        "textual-refusal",
        "multi-turn-child",
        "concurrent-child-order-a",
        "concurrent-child-order-b",
    ],
)
def test_every_accepted_vector_reaches_real_action_finalizer_and_ledger(
    case_id: str, tmp_path: Path
) -> None:
    source = _run_real_source_owner_case(tmp_path / case_id, case_id)
    assert source.rollout_id == "rollout-live"
    assert source.decisions


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "content_state", ["absent", "null", "empty"]
)
def test_message_content_presence_states_roundtrip_exactly(
    content_state: str,
) -> None:
    """Keep absent, null, and empty assistant content observably distinct."""

    from test_stage_d_source_producer import _prepared_action

    message: dict[str, object] = {"role": "assistant"}
    if content_state == "null":
        message["content"] = None
    elif content_state == "empty":
        message["content"] = ""
    action = _prepared_action(71, message=message)
    raw = action.to_bytes()
    reloaded = BehaviorAction.from_bytes(
        raw,
        validate_action=lambda _request, _message, _tokens: None,
        render_prompt=lambda _request: (10, 11),
    )
    assert reloaded.message == message
    assert reloaded.to_bytes() == raw


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("field", "expected_error"),
    [
        ("finish_reason", "sampled action termination contract is inconsistent"),
        ("usage", "usage must exactly match prompt and action token arrays"),
        ("tool_argument_bytes", "captured transport message differs from the Verifiers trace"),
        ("address", "captured structural address differs from the Verifiers trace"),
    ],
)
def test_invalid_vectors_reach_the_named_production_owner(
    field: str, expected_error: str, tmp_path: Path
) -> None:
    """Reject each mutation at its owning action or source-finalizer check."""

    from test_stage_d_source_producer import _prepared_action

    if field in {"finish_reason", "usage"}:
        action = _prepared_action(71)
        with pytest.raises(ValueError, match=expected_error):
            BehaviorAction.build(
                key=action.key,
                action_token_ids=action.action_token_ids,
                behavior_logprobs=action.behavior_logprobs,
                raw_transport_message=action.message,
                finish_reason="length" if field == "finish_reason" else action.finish_reason,
                prompt_tokens=action.prompt_tokens,
                completion_tokens=(
                    action.completion_tokens + 1
                    if field == "usage"
                    else action.completion_tokens
                ),
                termination_kind=action.termination_kind,
                eos_token_id=action.eos_token_id,
                validate_action=lambda _request, _message, _tokens: None,
            )
        return

    def mutate_trace(trace: dict[str, object]) -> None:
        if field == "tool_argument_bytes":
            node = cast(list[object], trace["nodes"])[1]
            message = cast(dict[str, object], cast(dict[str, object], node)["message"])
            tool_call = cast(list[object], message["tool_calls"])[0]
            function = cast(dict[str, object], cast(dict[str, object], tool_call)["function"])
            function["arguments"] = '{"changed":true}'
        else:
            call = cast(list[object], trace["calls"])[0]
            rlm = cast(dict[str, object], cast(dict[str, object], call)["rlm"])
            rlm["turn"] = 1

    with pytest.raises(ValueError, match=expected_error):
        _run_real_source_owner_case(
            tmp_path / field,
            "tool-calls" if field == "tool_argument_bytes" else "eos",
            mutate_trace=mutate_trace,
        )


def _observer_response(action: BehaviorAction) -> SimpleNamespace:
    return SimpleNamespace(
        tokens=SimpleNamespace(
            prompt_ids=[10, 11],
            completion_ids=list(action.action_token_ids),
            completion_logprobs=list(action.behavior_logprobs),
        ),
        raw={
            "id": "closure-provider-response",
            "choices": [{"message": action.message}],
        },
        usage=SimpleNamespace(
            input_tokens=2,
            completion_tokens=action.completion_tokens,
        ),
        finish_reason=action.finish_reason,
    )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "case_id",
    [
        "eos",
        "exact-cap",
        "tool-calls",
        "empty-string-content",
        "textual-refusal",
        "multi-turn-child",
        "concurrent-child-order-a",
        "concurrent-child-order-b",
    ],
)
def test_every_accepted_vector_reaches_interception_response_owner(
    case_id: str, tmp_path: Path
) -> None:
    """Bind every accepted vector to the real observer conversion boundary."""

    from test_stage_d_live_observer import (
        _child_rlm,
        _observer,
        _prepared,
        _root_rlm,
    )
    from test_stage_d_source_producer import _prepared_action, _tool_action

    def make_action(seed: int, *, tool: bool = False) -> BehaviorAction:
        if tool:
            return _tool_action(seed)
        return _prepared_action(
            seed,
            message=(
                {"role": "assistant", "content": ""}
                if case_id == "empty-string-content"
                else (
                    {"role": "assistant", "content": "I cannot comply."}
                    if case_id == "textual-refusal"
                    else None
                )
            ),
            max_tokens=case_id == "exact-cap",
        )
    root_action = make_action(
        71,
        tool=case_id
        in {
            "tool-calls",
            "multi-turn-child",
            "concurrent-child-order-a",
            "concurrent-child-order-b",
        },
    )
    child_actions = (
        make_action(72),
        make_action(73),
    )
    observer, ledger, producer = _observer(
        tmp_path / case_id,
        rollout_id="rollout-live",
    )

    async def scenario() -> None:
        root_ticket = await observer.before_forward(
            _prepared(71, "q", _root_rlm(), trace_id="rollout-live")
        )
        await observer.after_raw_response(root_ticket, b"closure-response-root")
        await observer.after_response(root_ticket, _observer_response(root_action))
        if case_id == "multi-turn-child":
            child_ticket = await observer.before_forward(
                _prepared(72, "child", _child_rlm(0, "closure-child"), trace_id="rollout-live")
            )
            await observer.after_raw_response(child_ticket, b"closure-response-child")
            await observer.after_response(child_ticket, _observer_response(child_actions[0]))
            returning = await observer.before_forward(
                _prepared(73, "return", _root_rlm(1), trace_id="rollout-live")
            )
            await observer.after_raw_response(returning, b"closure-response-return")
            await observer.after_response(returning, _observer_response(_prepared_action(74)))
        elif case_id.startswith("concurrent-child-order"):
            tickets = [
                await observer.before_forward(
                    _prepared(
                        72 + index,
                        f"child-{index}",
                        _child_rlm(index, f"closure-child-{index}"),
                        trace_id="rollout-live",
                    )
                )
                for index in (0, 1)
            ]
            ready = 0
            release = asyncio.Event()

            async def deliver(index: int) -> None:
                nonlocal ready
                ready += 1
                if ready == 2:
                    release.set()
                await release.wait()
                await observer.after_raw_response(tickets[index], b"closure-response-child")
                await observer.after_response(
                    tickets[index], _observer_response(child_actions[index])
                )

            await observer.run_concurrent_children(
                asyncio.gather(deliver(1), deliver(0))
            )
        else:
            child_ticket = await observer.before_forward(
                _prepared(72, "child", _child_rlm(0, "closure-child"), trace_id="rollout-live")
            )
            await observer.after_raw_response(child_ticket, b"closure-response-child")
            await observer.after_response(child_ticket, _observer_response(child_actions[0]))

    asyncio.run(scenario())
    expected_completed = 3 if (
        case_id == "multi-turn-child" or case_id.startswith("concurrent-child-order")
    ) else 2
    assert len(producer._completed) == expected_completed
    assert not producer._pending
    ledger.close()


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "disposition",
    [
        "cancellation",
        "transport_error",
        "malformed_provider_refusal",
        "schema_failure",
        "zero_token_response",
    ],
)
def test_abort_vectors_use_observer_abort_owner_once(
    disposition: str, tmp_path: Path
) -> None:
    """Every terminal transport/schema disposition poisons the real owner once."""

    from test_stage_d_live_observer import _observer, _prepared, _root_rlm

    observer, ledger, producer = _observer(
        tmp_path / disposition,
        rollout_id="abort-rollout",
    )

    async def scenario() -> object:
        ticket = await observer.before_forward(
            _prepared(71, "q", _root_rlm(), trace_id="abort-rollout")
        )
        boundary_error: BaseException
        if disposition == "cancellation":
            async def cancelled_provider() -> None:
                raise asyncio.CancelledError()

            with pytest.raises(asyncio.CancelledError) as caught:
                await observer.run_provider_call(cancelled_provider())
            boundary_error = caught.value
        elif disposition == "transport_error":
            async def failed_provider() -> None:
                raise ConnectionError("provider transport failed")

            with pytest.raises(ConnectionError) as caught:
                await observer.run_provider_call(failed_provider())
            boundary_error = caught.value
        else:
            from test_stage_d_source_producer import _prepared_action

            action = _prepared_action(71)
            response = _observer_response(action)
            if disposition == "malformed_provider_refusal":
                response.raw = {"id": "refusal", "choices": []}
            elif disposition == "schema_failure":
                response.tokens.completion_ids = "not-a-token-list"
            else:
                response.tokens.completion_ids = []
                response.tokens.completion_logprobs = []
            with pytest.raises(ValueError) as caught:
                await observer.after_response(ticket, response)
            boundary_error = caught.value
        await observer.abort(ticket, "response_received", boundary_error)
        with pytest.raises(ValueError, match="not pending"):
            await observer.abort(ticket, "response_received", RuntimeError("second"))
        with pytest.raises(
            ValueError,
            match="cannot reserve a source policy call after an observed call abort",
        ):
            await observer.before_forward(
                _prepared(
                    72,
                    "retry",
                    _root_rlm(1, call_ordinal=1),
                    trace_id="abort-rollout",
                )
            )
        return ticket

    asyncio.run(scenario())
    assert producer._aborted is True
    assert producer._pending == {}
    with pytest.raises(ValueError, match="observed call abort"):
        producer.finalize_episode(b"not-a-source-episode")
    ledger.close()


def test_concurrent_children_owner_proves_overlap_and_awaits_both() -> None:
    active = 0
    maximum_active = 0

    async def child() -> str:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1
        return "child"

    async def scenario() -> list[str]:
        watchdog = ActionClosureWatchdog()
        result = await watchdog.run_concurrent_children(asyncio.gather(child(), child()))
        return list(result)

    assert asyncio.run(scenario()) == ["child", "child"]
    assert maximum_active == 2


def test_retained_raw_fixture_manifest_authenticates_exactly_fourteen() -> None:
    root = Path(__file__).parents[1]
    config = json.loads(
        (root / "configs/stage-d/stage-d1-action-closure-corpus-v2.json").read_bytes()
    )
    report = json.loads(
        (root / "reports/stage-d1-action-closure-corpus-audit-v2.json").read_bytes()
    )
    audited = audit_raw_response_fixtures(root, config["raw_response_fixtures"])
    assert audited["fixture_count"] == 14
    assert audited["total_bytes"] > 0
    assert config["metadata_only_versions"] == [1, 2, 3, 5, 6]
    assert config["completed_action_reload_count"] == 12
    assert [fixture["version"] for fixture in config["raw_response_fixtures"]].count(4) == 1
    assert [fixture["version"] for fixture in config["raw_response_fixtures"]].count(7) == 3
    assert [fixture["version"] for fixture in config["raw_response_fixtures"]].count(8) == 4
    assert [fixture["version"] for fixture in config["raw_response_fixtures"]].count(9) == 3
    assert [fixture["version"] for fixture in config["raw_response_fixtures"]].count(10) == 3
    assert report["historical_semantic_replayed_versions"] == [4, 7, 8, 9, 10]
    assert report["semantic_renderer_observer_replay_count"] == 14
    assert report["completed_action_reload_count"] == 12
    assert config["sampling_contract"] == sampling_contract_binding(root)
    assert report["sampling_contract"] == sampling_contract_binding(root)
    assert report["frozen_v1_audit_sha256"] == (
        "25c777aa9c121b818f3315bed5b13fe98336fe14aba31fe9c46f6e53808e6b6c"
    )


def test_collection_path_owns_watchdog_and_notifies_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Data:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"scientific_group_id": "group", "example_id": "example", "rollout_slot": 0}

    task = SimpleNamespace(data=Data())
    retries = SimpleNamespace(max_retries=0)
    env = SimpleNamespace(
        master_seed="master",
        max_concurrent=1,
        retries=retries,
        agent=SimpleNamespace(retries=retries),
        taskset=SimpleNamespace(select=lambda _count, _shuffle: [task]),
    )
    config = SimpleNamespace(
        server=False,
        num_rollouts=1,
        num_tasks=1,
        shuffle=False,
        resume=None,
        push=False,
        rich=False,
        max_concurrent=1,
        output_dir=tmp_path / "fresh",
        env=env,
    )
    events: list[str] = []
    watchdog_events: list[tuple[str, str]] = []
    persisted: list[bytes] = []
    terminal_records: list[bytes] = []

    async def run_eval(_env: object, _config: object) -> list[object]:
        events.append("eval")
        return [SimpleNamespace()]

    monkeypatch.setattr(collection, "verify_collection_outcomes", lambda *_args: b"receipt")
    watchdog = ActionClosureWatchdog(
        deadlines=WatchdogDeadlines(episode=1.0),
        terminal_callback=lambda phase, disposition: watchdog_events.append(
            (phase, disposition)
        ),
        terminal_record_callback=terminal_records.append,
    )
    plan, episodes, receipt = asyncio.run(
        collection.run_exact_source_collection(
            config,
            preregistered_plan_sha256=collection.StageDCollectionPlan.build(
                [task.data.model_dump(mode="json")], master_seed="master"
            ).plan_sha256,
            run_eval=run_eval,
            load_environment=lambda _config: env,
            load_verified_sources=lambda: (),
            persist_plan=persisted.append,
            watchdog=watchdog,
        )
    )
    assert plan.slots[0].example_id == "example"
    assert episodes == (SimpleNamespace(),) or len(episodes) == 1
    assert receipt == b"receipt"
    assert events == ["eval"]
    assert watchdog_events == []
    assert len(persisted) == 1
    assert terminal_records == []
    watchdog.complete()
    assert watchdog_events == [("campaign", "completed")]
    assert len(terminal_records) == 1
    assert json.loads(terminal_records[0])["disposition"] == "completed"


def test_collection_episode_timeout_cancels_and_records_once(tmp_path: Path) -> None:
    class Data:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"scientific_group_id": "group", "example_id": "example", "rollout_slot": 0}

    task = SimpleNamespace(data=Data())
    retries = SimpleNamespace(max_retries=0)
    env = SimpleNamespace(
        master_seed="master",
        max_concurrent=1,
        retries=retries,
        agent=SimpleNamespace(retries=retries),
        taskset=SimpleNamespace(select=lambda _count, _shuffle: [task]),
    )
    config = SimpleNamespace(
        server=False,
        num_rollouts=1,
        num_tasks=1,
        shuffle=False,
        resume=None,
        push=False,
        rich=False,
        max_concurrent=1,
        output_dir=tmp_path / "fresh",
        env=env,
    )
    persisted: list[bytes] = []
    terminal_records: list[bytes] = []

    async def hanging_eval(_env: object, _config: object) -> list[object]:
        try:
            await asyncio.sleep(1.0)
        finally:
            persisted.append(b"cancelled")
        return []

    plan = collection.StageDCollectionPlan.build(
        [task.data.model_dump(mode="json")], master_seed="master"
    )
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(
            collection.run_exact_source_collection(
                config,
                preregistered_plan_sha256=plan.plan_sha256,
                run_eval=hanging_eval,
                load_environment=lambda _config: env,
                load_verified_sources=lambda: (),
                persist_plan=lambda _value: None,
                watchdog=ActionClosureWatchdog(
                    deadlines=WatchdogDeadlines(episode=0.001),
                    terminal_record_callback=terminal_records.append,
                ),
            )
        )
    assert len(terminal_records) == 1
    assert json.loads(terminal_records[0]) == {
        "domain": "redco-stage-d-action-closure-terminal-v1",
        "disposition": "timeout",
        "phase": "episode",
        "phase_count": 2,
        "schema_version": 1,
    }
    assert persisted == [b"cancelled"]


def test_watchdog_timeout_is_terminal_and_cannot_retry() -> None:
    events: list[tuple[str, str]] = []
    cancelled = False

    async def slow() -> None:
        nonlocal cancelled
        try:
            await asyncio.sleep(0.05)
        finally:
            cancelled = True

    watchdog = ActionClosureWatchdog(
        deadlines=WatchdogDeadlines(provider_call=0.001),
        terminal_callback=lambda phase, disposition: events.append((phase, disposition)),
    )
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(watchdog.run(slow(), phase="provider_call"))
    assert events == [("provider_call", "timeout")]
    assert cancelled is True

    async def second_attempt() -> None:
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)
        await watchdog.run(future, phase="provider_call")

    with pytest.raises(RuntimeError, match="terminal disposition"):
        asyncio.run(second_attempt())


@pytest.mark.parametrize("phase", WATCHDOG_PHASES)  # type: ignore[untyped-decorator]
def test_watchdog_freezes_every_live_phase(phase: str) -> None:
    events: list[tuple[str, str]] = []

    async def completed() -> str:
        return phase

    watchdog = ActionClosureWatchdog(
        terminal_callback=lambda observed_phase, disposition: events.append(
            (observed_phase, disposition)
        )
    )
    assert asyncio.run(watchdog.run(completed(), phase=phase)) == phase
    watchdog.complete()
    assert events == [("campaign", "completed")]


def test_watchdog_supervises_nested_phases_and_records_one_terminal() -> None:
    events: list[tuple[str, str]] = []
    records: list[bytes] = []

    async def completed(value: str) -> str:
        await asyncio.sleep(0)
        return value

    watchdog = ActionClosureWatchdog(
        terminal_callback=lambda phase, disposition: events.append((phase, disposition)),
        terminal_record_callback=records.append,
    )

    async def run() -> None:
        assert await watchdog.run_provider_call(completed("provider")) == "provider"
        assert await watchdog.run_concurrent_children(completed("children")) == "children"
        assert await watchdog.run_episode(completed("episode")) == "episode"
        assert await watchdog.run_finalizer(completed("finalizer")) == "finalizer"
        await watchdog.run_campaign(completed("campaign"))
        assert await watchdog.run_pod_lifetime(completed("pod")) == "pod"
        watchdog.complete()

    asyncio.run(run())
    assert watchdog.phase_count == 6
    assert events == [("campaign", "completed")]
    assert len(records) == 1
    assert json.loads(records[0]) == {
        "domain": "redco-stage-d-action-closure-terminal-v1",
        "disposition": "completed",
        "phase": "campaign",
        "phase_count": 6,
        "schema_version": 1,
    }


def test_terminal_publication_failure_can_be_replaced_by_one_failure_record() -> None:
    attempts = 0
    records: list[bytes] = []

    def durable_writer(value: bytes) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("receipt publication failed")
        records.append(value)

    watchdog = ActionClosureWatchdog(terminal_record_callback=durable_writer)
    with pytest.raises(OSError, match="receipt publication failed"):
        watchdog.complete()
    assert watchdog.terminal is False

    watchdog.fail("campaign_publication", "error")
    assert watchdog.terminal is True
    assert len(records) == 1
    assert json.loads(records[0]) == {
        "domain": "redco-stage-d-action-closure-terminal-v1",
        "disposition": "error",
        "phase": "campaign_publication",
        "phase_count": 0,
        "schema_version": 1,
    }
    with pytest.raises(RuntimeError, match="second terminal disposition"):
        watchdog.fail("campaign_publication", "error")


def test_watchdog_timeout_stops_the_bound_launcher_runtime_once() -> None:
    class Process:
        returncode = None

        def __init__(self) -> None:
            self.calls: list[str] = []
            self.waits = 0

        def terminate(self) -> None:
            self.calls.append("term")

        async def wait(self) -> None:
            self.waits += 1
            if self.waits == 1:
                await asyncio.sleep(0.01)

        def kill(self) -> None:
            self.calls.append("kill")

    class Runtime:
        def __init__(self, process: Process) -> None:
            self._background = [process]
            self.stop_count = 0

        async def stop(self) -> None:
            self.stop_count += 1

    process = Process()
    runtime = Runtime(process)
    watchdog = ActionClosureWatchdog(
        deadlines=WatchdogDeadlines(provider_call=0.001),
        runtime_term_timeout=0.001,
    )
    watchdog.bind_runtime(runtime)

    async def slow_provider() -> None:
        await asyncio.sleep(0.05)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(watchdog.run_provider_call(slow_provider()))
    assert watchdog.terminal is True
    assert process.calls == ["term", "kill"]
    assert runtime.stop_count == 1


def test_timeout_teardown_survives_first_and_replacement_terminal_failures() -> None:
    class Process:
        returncode = None

        def __init__(self) -> None:
            self.calls: list[str] = []
            self.waits = 0

        def terminate(self) -> None:
            self.calls.append("term")

        async def wait(self) -> None:
            self.waits += 1
            if self.waits == 1:
                await asyncio.sleep(0.01)

        def kill(self) -> None:
            self.calls.append("kill")

    class Runtime:
        def __init__(self, process: Process) -> None:
            self._background = [process]
            self.stop_count = 0

        async def stop(self) -> None:
            self.stop_count += 1

    attempts = 0
    provider_posts = 0

    def terminal_writer(_value: bytes) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError(
            "first durable failure" if attempts == 1 else "replacement durable failure"
        )

    async def provider() -> None:
        nonlocal provider_posts
        provider_posts += 1
        await asyncio.sleep(0.05)

    process = Process()
    runtime = Runtime(process)
    watchdog = ActionClosureWatchdog(
        deadlines=WatchdogDeadlines(provider_call=0.001),
        runtime_term_timeout=0.001,
        terminal_record_callback=terminal_writer,
    )
    watchdog.bind_runtime(runtime)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(watchdog.run_provider_call(provider()))

    assert provider_posts == 1
    assert process.calls == ["term", "kill"]
    assert runtime.stop_count == 1
    assert watchdog.closed is True
    assert watchdog.terminal is False

    async def second_provider() -> None:
        nonlocal provider_posts
        provider_posts += 1

    with pytest.raises(RuntimeError, match="after closure"):
        asyncio.run(watchdog.run_provider_call(second_provider()))
    assert provider_posts == 1
    with pytest.raises(OSError, match="replacement durable failure"):
        watchdog.fail("campaign_publication", "error")
    assert attempts == 2


def test_process_teardown_uses_term_then_kill() -> None:
    class Process:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.waits = 0

        def terminate(self) -> None:
            self.calls.append("term")

        async def wait(self) -> None:
            self.waits += 1
            if self.waits == 1:
                await asyncio.sleep(0.01)

        def kill(self) -> None:
            self.calls.append("kill")

    process = Process()
    assert asyncio.run(terminate_process_then_kill(process, term_timeout=0.001)) == "kill"
    assert process.calls == ["term", "kill"]


def test_launcher_owns_process_teardown_and_terminal_record_once(tmp_path: Path) -> None:
    class Process:
        returncode = None

        def __init__(self) -> None:
            self.calls: list[str] = []
            self.waits = 0

        def terminate(self) -> None:
            self.calls.append("term")

        async def wait(self) -> None:
            self.waits += 1
            if self.waits == 1:
                await asyncio.sleep(0.01)

        def kill(self) -> None:
            self.calls.append("kill")

    class Runtime:
        def __init__(self, process: Process) -> None:
            self._background = [process]
            self.stopped = False

        async def stop(self) -> None:
            self.stopped = True

    process = Process()
    runtime = Runtime(process)
    asyncio.run(stop_owned_runtime(runtime, term_timeout=0.001))
    assert process.calls == ["term", "kill"]
    assert runtime.stopped is True

    record_path = tmp_path / "terminal.json"
    write_record = atomic_terminal_record_writer(record_path)
    write_record(canonical_json({"disposition": "completed"}))
    with pytest.raises(RuntimeError, match="write-once collision"):
        write_record(canonical_json({"disposition": "completed"}))
