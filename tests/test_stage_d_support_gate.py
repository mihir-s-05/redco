from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from types import SimpleNamespace

import pytest

from redco.analysis.stage_d_branch_artifacts import (
    StageDBranchArtifactStore,
    StageDBranchTarget,
    StageDBranchTargetRoster,
)
from redco.analysis.stage_d_collection import StageDCollectionPlan
from redco.analysis.stage_d_source_artifacts import (
    SourceArtifactError,
    StageDSourceArtifactStore,
)
from redco.analysis.stage_d_support_gate import (
    StageDSupportRules,
    evaluate_support_gate,
    load_support_rules,
    verify_support_pass,
    verify_support_report,
)
from redco.contracts import canonical_json
from redco.integrations.write_once import write_once
from scripts.run_stage_d_scientific_campaign import _run_branch_groups


def _rules(*, papers: int = 1, successes: int = 1) -> StageDSupportRules:
    return StageDSupportRules.from_bytes(
        canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-support-rules-v1",
                "required_papers": papers,
                "required_successes": successes,
                "minimum_targets": 1,
                "maximum_targets": 1,
                "minimum_reward_range": 0.05,
            }
        )
    )


def _fixture(*, informative: bool = True, index: int = 1):
    address = {"depth": 1, "lineage": "root/child", "session_call_ordinal": 0, "turn": 0}
    source_sha256 = hashlib.sha256(f"source-{index}".encode()).hexdigest()
    source = SimpleNamespace(
        source_sha256=source_sha256,
        group_id=f"paper-{index}",
        rollout_id=f"rollout-{index}",
        branch_eligible=True,
        child_target_roster=("target-1",),
        decisions=(SimpleNamespace(node_kind="child", target_id="target-1"),),
    )
    target = StageDBranchTarget(
        source.source_sha256,
        source.group_id,
        source.rollout_id,
        "decision-1",
        "target-1",
        0,
        address,
    )
    roster = StageDBranchTargetRoster(
        1,
        1,
        1,
        0,
        1,
        True,
        (source.source_sha256,),
        (target,),
    )
    commitment = SimpleNamespace(
        group_id=source.group_id,
        rollout_id=source.rollout_id,
        target_id=target.target_id,
        target_ordinal=target.target_ordinal,
        target_address=SimpleNamespace(
            turn=0,
            as_payload=lambda: {key: value for key, value in address.items() if key != "turn"},
        ),
        outer_weight=Fraction(1),
    )
    q_values = (0.0, 0.2) if informative else (0.1, 0.1)
    artifact = SimpleNamespace(
        commitment=commitment,
        reconstruction_qa=SimpleNamespace(passed=True),
        arms=tuple(SimpleNamespace(q_value=value) for value in q_values),
        to_bytes=lambda: canonical_json({"artifact": index}),
    )
    return source, roster, artifact


def test_support_gate_uses_paper_denominator_and_reward_range() -> None:
    source, roster, artifact = _fixture()
    report_bytes = evaluate_support_gate(
        (source,),  # type: ignore[arg-type]
        (artifact,),  # type: ignore[arg-type]
        roster,
        paper_ids={source.source_sha256: "paper-1"},
        rules=_rules(),
    )
    report = json.loads(report_bytes)
    assert report["decision"] == "pass"
    assert report["paper_successes"] == 1
    assert report["nested_support"] == {
        "N_scaffold": 1,
        "N_eligible": 1,
        "N_joint": 1,
        "first_failed_gate": None,
        "first_failed_gate_is_descriptive_not_causal": True,
        "rates_95pct_wilson": {
            "scaffold_over_all": {
                "successes": 1,
                "total": 1,
                "lower": pytest.approx(0.20654931437723745),
                "upper": 1.0,
            },
            "eligible_given_scaffold": {
                "successes": 1,
                "total": 1,
                "lower": pytest.approx(0.20654931437723745),
                "upper": 1.0,
            },
            "joint_given_eligible": {
                "successes": 1,
                "total": 1,
                "lower": pytest.approx(0.20654931437723745),
                "upper": 1.0,
            },
        },
    }
    assert report["papers"][0]["outer_weight_sum"] == {
        "numerator": 1,
        "denominator": 1,
    }
    assert verify_support_pass(
        report_bytes,
        expected_rules_sha256=_rules().rules_sha256,
        source_sha256s=(source.source_sha256,),
        artifact_sha256s=(hashlib.sha256(artifact.to_bytes()).hexdigest(),),
    ) == hashlib.sha256(report_bytes).hexdigest()


def test_support_gate_retains_flat_groups_as_failures() -> None:
    source, roster, artifact = _fixture(informative=False)
    report = json.loads(
        evaluate_support_gate(
            (source,),  # type: ignore[arg-type]
            (artifact,),  # type: ignore[arg-type]
            roster,
            paper_ids={source.source_sha256: "paper-1"},
            rules=_rules(),
        )
    )
    assert report["decision"] == "fail"
    assert report["papers"][0]["has_informative_target"] is False
    assert report["nested_support"]["first_failed_gate"] == "informativeness"


def test_support_taxonomy_reports_first_gate_without_causal_overclaim() -> None:
    source, roster, artifact = _fixture()
    source.child_target_roster = ()
    source.decisions = ()
    report = json.loads(
        evaluate_support_gate(
            (source,),  # type: ignore[arg-type]
            (artifact,),  # type: ignore[arg-type]
            roster,
            paper_ids={source.source_sha256: "paper-1"},
            rules=_rules(),
        )
    )
    taxonomy = report["nested_support"]
    assert taxonomy["N_scaffold"] == taxonomy["N_eligible"] == taxonomy["N_joint"] == 0
    assert taxonomy["first_failed_gate"] == "scaffold"
    assert taxonomy["first_failed_gate_is_descriptive_not_causal"] is True
    assert taxonomy["rates_95pct_wilson"]["eligible_given_scaffold"] == {
        "successes": 0,
        "total": 0,
        "lower": None,
        "upper": None,
    }


def test_zero_eligible_sources_produce_a_verified_failure_report() -> None:
    source, _roster, _artifact = _fixture()
    source.branch_eligible = False
    source.child_target_roster = ()
    source.decisions = ()
    roster = StageDBranchTargetRoster(
        1,
        0,
        1,
        1,
        1,
        False,
        (source.source_sha256,),
        (),
    )
    report_bytes = evaluate_support_gate(
        (source,),  # type: ignore[arg-type]
        (),
        roster,
        paper_ids={source.source_sha256: "paper-1"},
        rules=_rules(),
    )

    report = json.loads(report_bytes)
    report_sha256, decision = verify_support_report(
        report_bytes,
        expected_rules_sha256=_rules().rules_sha256,
        source_sha256s=(source.source_sha256,),
        artifact_sha256s=(),
    )
    assert decision == "fail"
    assert report_sha256 == hashlib.sha256(report_bytes).hexdigest()
    assert report["nested_support"]["N_scaffold"] == 0
    assert report["nested_support"]["N_eligible"] == 0
    assert report["nested_support"]["N_joint"] == 0
    assert report["nested_support"]["rates_95pct_wilson"]["joint_given_eligible"] == {
        "successes": 0,
        "total": 0,
        "lower": None,
        "upper": None,
    }


def test_zero_branch_groups_make_no_campaign_calls() -> None:
    def forbidden_campaign_runner(*_args, **_kwargs):
        raise AssertionError("zero eligible sources must not call a model")

    assert _run_branch_groups(
        (),
        campaign_runner=forbidden_campaign_runner,
        ledger=SimpleNamespace(),  # type: ignore[arg-type]
        event_loop=SimpleNamespace(),  # type: ignore[arg-type]
    ) == ()


def test_support_gate_rejects_an_incomplete_artifact_roster() -> None:
    source, roster, _artifact = _fixture()
    with pytest.raises(ValueError, match="complete committed target set"):
        evaluate_support_gate(
            (source,),  # type: ignore[arg-type]
            (),
            roster,
            paper_ids={source.source_sha256: "paper-1"},
            rules=_rules(),
        )


def test_support_gate_freezes_the_58_of_64_boundary() -> None:
    fixtures = [
        _fixture(informative=index < 57, index=index + 1) for index in range(64)
    ]
    sources = tuple(row[0] for row in fixtures)
    targets = tuple(row[1].targets[0] for row in fixtures)
    artifacts = tuple(row[2] for row in fixtures)
    roster = StageDBranchTargetRoster(
        64,
        64,
        64,
        0,
        58,
        True,
        tuple(sorted(source.source_sha256 for source in sources)),
        targets,
    )
    paper_ids = {
        source.source_sha256: f"paper-{index}"
        for index, source in enumerate(sources, start=1)
    }
    failed = json.loads(
        evaluate_support_gate(
            sources,  # type: ignore[arg-type]
            artifacts,  # type: ignore[arg-type]
            roster,
            paper_ids=paper_ids,
            rules=_rules(papers=64, successes=58),
        )
    )
    assert failed["paper_successes"] == 57
    assert failed["decision"] == "fail"

    sources, rosters, artifacts = zip(
        *(_fixture(informative=index < 58, index=index + 1) for index in range(64)),
        strict=True,
    )
    roster = StageDBranchTargetRoster(
        64,
        64,
        64,
        0,
        58,
        True,
        tuple(sorted(source.source_sha256 for source in sources)),
        tuple(item.targets[0] for item in rosters),
    )
    paper_ids = {
        source.source_sha256: f"paper-{index}"
        for index, source in enumerate(sources, start=1)
    }
    passed = json.loads(
        evaluate_support_gate(
            sources,  # type: ignore[arg-type]
            artifacts,  # type: ignore[arg-type]
            roster,
            paper_ids=paper_ids,
            rules=_rules(papers=64, successes=58),
        )
    )
    assert passed["paper_successes"] == 58
    assert passed["decision"] == "pass"


def test_support_collection_plan_allows_one_slot_per_unique_paper() -> None:
    plan = StageDCollectionPlan.build(
        [
            {
                "scientific_group_id": f"paper-{index}",
                "example_id": f"example-{index}",
                "rollout_slot": 0,
            }
            for index in range(64)
        ],
        master_seed="support-master",
    )
    assert len(plan.slots) == 64


def test_write_once_is_idempotent_but_never_replaces(tmp_path) -> None:
    path = tmp_path / "evidence" / "artifact.json"
    write_once(path, b"frozen")
    write_once(path, b"frozen")
    with pytest.raises(FileExistsError, match="write-once collision"):
        write_once(path, b"changed")
    assert path.read_bytes() == b"frozen"


def test_write_once_rejects_regular_and_dangling_symlinks(tmp_path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"frozen")
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable")
    with pytest.raises(FileExistsError, match="write-once collision"):
        write_once(linked, b"frozen")
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="write-once collision"):
        write_once(dangling, b"new")


def test_support_rules_are_hash_bound_and_fail_closed(tmp_path) -> None:
    rules = _rules()
    path = tmp_path / "rules.json"
    value = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-support-rules-v1",
            "required_papers": 1,
            "required_successes": 1,
            "minimum_targets": 1,
            "maximum_targets": 1,
            "minimum_reward_range": 0.05,
        }
    )
    path.write_bytes(value)
    assert load_support_rules(path, rules.rules_sha256) == rules
    with pytest.raises(ValueError, match="protocol manifest"):
        load_support_rules(path, "0" * 64)
    with pytest.raises(RuntimeError, match="failed authentication"):
        verify_support_pass(
            canonical_json({"decision": "fail"}),
            expected_rules_sha256=rules.rules_sha256,
            source_sha256s=(),
            artifact_sha256s=(),
        )


def test_artifact_stores_reject_symlinked_roots(tmp_path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are unavailable")
    with pytest.raises(RuntimeError, match="symbolic link"):
        StageDBranchArtifactStore(linked)
    with pytest.raises(SourceArtifactError, match="symbolic link"):
        StageDSourceArtifactStore(linked)
