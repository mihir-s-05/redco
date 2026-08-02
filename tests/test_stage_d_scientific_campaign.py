from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import pytest
from test_stage_d_scientific_branch_group import (
    Fixture,
    _executor,
    _fixture,
    _qa_receipt,
    _sampler,
)

from redco.analysis.stage_d_exact_action import ExactActionKey
from redco.analysis.stage_d_receipt_ledger import StageDReceiptLedger
from redco.analysis.stage_d_scientific_branch_group import (
    BranchGroupArtifact,
    BranchGroupSpec,
    CandidateSubmission,
    RepairableInfrastructureAbort,
    ZeroCallInfrastructureFailure,
)
from redco.analysis.stage_d_scientific_campaign import (
    ScientificGroupRun,
    run_scientific_campaign,
)
from redco.analysis.stage_d_spawn_provenance import EventSeedScheduler


@dataclass
class BarrierLedger:
    events: list[str]
    qa_count: int
    keys: tuple[tuple[str, str], ...] = ()

    @property
    def branch_target_keys(self) -> tuple[tuple[str, str], ...]:
        return self.keys

    def reconstruction_qa_receipt(self, group_id: str, target_id: str) -> None:
        del group_id, target_id
        return None

    def reconstruction_qa_barrier_receipt(self) -> None:
        return None

    def seal_reconstruction_qa_barrier(self) -> bytes:
        if self.qa_count != 2:
            raise RuntimeError("incomplete QA roster")
        self.events.append("barrier")
        return b"barrier"


def _run(fixture: Fixture, events: list[str], qa_count: list[int]) -> ScientificGroupRun:
    _, sampler = _sampler(fixture)
    _, executor = _executor(fixture, (0.0, 1.0))

    def qa(spec: BranchGroupSpec) -> bytes:
        del spec
        events.append(f"qa:{fixture.spec.commitment.target_id}")
        qa_count[0] += 1
        return _qa_receipt(fixture)

    def sample(
        *,
        action_slot: int,
        action_seed: int,
        reference_key: ExactActionKey,
    ) -> CandidateSubmission:
        events.append(f"sample:{fixture.spec.commitment.target_id}")
        return sampler(
            action_slot=action_slot,
            action_seed=action_seed,
            reference_key=reference_key,
        )

    def prepare(_artifact: BranchGroupArtifact) -> None:
        events.append(f"prepare:{fixture.spec.commitment.target_id}")

    return ScientificGroupRun(
        fixture.spec,
        qa,
        sample,
        executor,
        prepare,
    )


def test_whole_roster_qa_precedes_every_candidate_and_artifact() -> None:
    first = _fixture(target_ordinal=0, branch_count=2, group_id="group-0")
    second = _fixture(target_ordinal=1, branch_count=2, group_id="group-1")
    events: list[str] = []
    qa_count = [0]
    first_run = _run(first, events, qa_count)
    second_run = _run(second, events, qa_count)

    def verify(receipt: bytes, *, receipt_kind: str) -> Mapping[str, Any]:
        for store in (first.store, second.store):
            try:
                return store(receipt, receipt_kind=receipt_kind)
            except ValueError:
                continue
        raise ValueError("receipt is absent from both trusted stores")

    class BoundLedger(BarrierLedger):
        def seal_reconstruction_qa_barrier(self) -> bytes:
            self.qa_count = qa_count[0]
            return super().seal_reconstruction_qa_barrier()

    result = run_scientific_campaign(
        (first_run, second_run),
        ledger=cast(
            StageDReceiptLedger,
            BoundLedger(events, 0, (("group-0", "target-0"), ("group-1", "target-1"))),
        ),
        verifier=verify,
    )

    assert result.reconstruction_qa_barrier_receipt == b"barrier"
    assert events[:3] == ["qa:target-0", "qa:target-1", "barrier"]
    assert all(
        events.index(candidate) > events.index("barrier")
        for candidate in ("sample:target-0", "sample:target-1")
    )


def test_duplicate_campaign_target_fails_before_qa() -> None:
    fixture = _fixture(branch_count=2)
    events: list[str] = []
    run = _run(fixture, events, [0])

    with pytest.raises(ValueError, match="unique"):
        run_scientific_campaign(
            (run, run),
            ledger=cast(
                StageDReceiptLedger,
                BarrierLedger(events, 0, (("group", "target-0"),)),
            ),
            verifier=fixture.store,
        )

    assert events == []


def test_failed_second_qa_prevents_barrier_and_science() -> None:
    first = _fixture(target_ordinal=0, branch_count=2, group_id="group-0")
    second = _fixture(target_ordinal=1, branch_count=2, group_id="group-1")
    events: list[str] = []
    first_run = _run(first, events, [0])
    second_run = _run(second, events, [0])

    def fail_qa(spec: BranchGroupSpec) -> bytes:
        del spec
        events.append("qa-failure")
        raise RuntimeError("broken snapshot")

    second_run = ScientificGroupRun(
        second_run.spec,
        fail_qa,
        second_run.sample_candidate,
        second_run.execute_arm,
        second_run.prepare_artifact,
    )
    with pytest.raises(RepairableInfrastructureAbort, match="valid receipt"):
        run_scientific_campaign(
            (first_run, second_run),
            ledger=cast(
                StageDReceiptLedger,
                BarrierLedger(
                    events,
                    0,
                    (("group-0", "target-0"), ("group-1", "target-1")),
                ),
            ),
            verifier=first.store,
        )

    assert not any(event.startswith("sample:") for event in events)
    assert "barrier" not in events


def test_later_group_zero_call_failure_is_repairable_with_prior_artifacts() -> None:
    first = _fixture(target_ordinal=0, branch_count=2, group_id="group-0")
    second = _fixture(target_ordinal=1, branch_count=2, group_id="group-1")
    events: list[str] = []
    first_run = _run(first, events, [0])
    second_run = _run(second, events, [0])
    failure = second.store.issue(
        "zero_call_infrastructure_failure",
        {
            "ledger_id": second.spec.commitment.ledger_id,
            "ledger_offset": second.spec.commitment.ledger_offset + 1,
            "prior_chain_sha256": second.store.chain,
            "group_id": second.spec.commitment.group_id,
            "target_id": second.spec.commitment.target_id,
            "action_slot": 1,
            "action_seed": EventSeedScheduler(
                "master",
                second.spec.commitment.rollout_id,
                second.spec.commitment.target_id,
                1,
            ).action_seed(action_slot=1),
            "attempt_ordinal": 0,
            "attempt_id": "later-zero-call",
            "attempt_model_calls": 0,
            "attempt_overrides": 0,
            "prior_candidate_completions": 1,
            "prior_execution_completions": 2,
            "repair_sequence": 0,
            "successor_permitted": True,
            "reason": "capacity vanished",
        },
    )
    second_run = ScientificGroupRun(
        second_run.spec,
        second_run.run_reconstruction_qa,
        lambda **_: (_ for _ in ()).throw(ZeroCallInfrastructureFailure(failure)),
        second_run.execute_arm,
        second_run.prepare_artifact,
    )

    def verify(receipt: bytes, *, receipt_kind: str) -> Mapping[str, Any]:
        for store in (first.store, second.store):
            try:
                return store(receipt, receipt_kind=receipt_kind)
            except ValueError:
                continue
        raise ValueError("unknown receipt")

    ledger = BarrierLedger(
        events,
        2,
        (("group-0", "target-0"), ("group-1", "target-1")),
    )
    with pytest.raises(RepairableInfrastructureAbort, match="candidate slot 1"):
        run_scientific_campaign(
            (first_run, second_run),
            ledger=cast(StageDReceiptLedger, ledger),
            verifier=verify,
        )
