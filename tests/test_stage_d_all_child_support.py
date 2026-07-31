from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from redco.analysis import (
    stage_d_all_child_branch_group,
    stage_d_all_child_support,
)
from redco.analysis.stage_d_all_child_support import aggregate_paper_support
from redco.env.tracer import EventNodeKind
from redco.integrations.signed_subprocess import (
    sign_payload,
    verify_signed_payload,
)
from redco.integrations.verifiers_trace import RecordedPolicyCall

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_stage_d_midpoint_context_v1 import audit_rows  # noqa: E402
from build_stage_d_complementary_fixtures_v1 import build  # noqa: E402
from build_stage_d_qasper_extension_v1 import (  # noqa: E402
    materialize_extension,
)


def _call(index: int) -> RecordedPolicyCall:
    return RecordedPolicyCall(
        trace_id="trace-1",
        call_index=index,
        node_index=index,
        component_root_node=0,
        prompt_token_ids=(10 + index,),
        action_token_ids=(90 + index,),
        checkpoint_id="checkpoint",
        decoding_config_hash="decode",
        event_seed=100 + index,
        prompt_tokens_reported=1,
        completion_tokens_reported=1,
        cost_reported=None,
        wall_seconds=0.0,
        agent_depth=1,
        session_id=f"child-{index}",
        turn_index=0,
        call_kind="policy",
        parent_session_id="root",
        parent_turn_index=1,
    )


def test_all_child_precommit_excludes_outcome_fields_and_normalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("trace bytes", encoding="utf-8")
    calls = [_call(3), _call(2)]
    nodes = {
        f"node-{index}": SimpleNamespace(
            kind=EventNodeKind.POLICY,
            node_id=f"policy:{index}",
            metadata={"call_index": index},
        )
        for index in (2, 3)
    }
    monkeypatch.setattr(
        stage_d_all_child_support,
        "load_trace_records",
        lambda _: [
            {
                "id": "trace-1",
                "task": {"data": {"paper_id": "paper-1"}},
            }
        ],
    )
    monkeypatch.setattr(
        stage_d_all_child_support,
        "audit_trace_file",
        lambda _: SimpleNamespace(calls=calls),
    )
    monkeypatch.setattr(
        stage_d_all_child_support,
        "import_trace_file",
        lambda _: SimpleNamespace(traces=[SimpleNamespace(graph=SimpleNamespace(nodes=nodes))]),
    )

    report = stage_d_all_child_support.precommit_all_depth_one_targets(trace_path)
    verify_signed_payload(report)
    assert report["candidate_count"] == 2
    assert len(report["source_trace_sha256"]) == 64
    assert len(report["candidate_set_sha256"]) == 64
    assert report["native_call_order_used_for_selection"] is False
    for candidate in report["candidates"]:
        assert candidate["decision_unit_weight"] == {
            "numerator": 1,
            "denominator": 2,
        }
        assert "action_token_ids" not in candidate
        assert "reward" not in candidate
        assert "behavior_logprobs" not in candidate
    monkeypatch.setattr(
        stage_d_all_child_support,
        "audit_trace_file",
        lambda _: SimpleNamespace(calls=list(reversed(calls))),
    )
    reversed_report = stage_d_all_child_support.precommit_all_depth_one_targets(trace_path)
    assert reversed_report["candidates"] == report["candidates"]
    assert reversed_report["candidate_set_sha256"] == report["candidate_set_sha256"]


def test_all_child_branch_group_disables_target_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("trace bytes", encoding="utf-8")
    monkeypatch.setattr(
        stage_d_all_child_branch_group,
        "load_trace_records",
        lambda _: [{"id": "trace-1"}],
    )
    monkeypatch.setattr(
        stage_d_all_child_branch_group,
        "audit_trace_file",
        lambda _: SimpleNamespace(calls=[object()]),
    )
    observed: dict[str, object] = {}
    precommit = sign_payload(
        {
            "source_trace_sha256": "a" * 64,
            "candidate_set_sha256": "b" * 64,
            "candidates": [
                {
                    "structural_event_address": "node-a",
                    "decision_unit_weight": {"numerator": 1, "denominator": 2},
                },
                {
                    "structural_event_address": "node-b",
                    "decision_unit_weight": {"numerator": 1, "denominator": 2},
                },
            ],
        }
    )
    monkeypatch.setattr(
        stage_d_all_child_branch_group,
        "verify_canonical_precommit",
        lambda *_: precommit,
    )

    def run(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        return SimpleNamespace(
            signed_dict=lambda: {
                "ok": True,
                "source_sha256": "a" * 64,
                "regenerated_originals": [
                    {"target_node_id": "node-a"},
                    {"target_node_id": "node-b"},
                ],
            }
        )

    monkeypatch.setattr(stage_d_all_child_branch_group, "run_empirical_replay", run)
    report, replayable = stage_d_all_child_branch_group.run_group(
        trace_path=trace_path,
        precommit=precommit,
        client=object(),  # type: ignore[arg-type]
        master_seed="master",
        temperature=0.7,
        candidate_max_tokens=512,
        continuation_max_tokens=768,
    )
    assert replayable is True
    assert observed["maximum_targets"] is None
    verify_signed_payload(report)


def test_paper_aggregate_keeps_nested_targets_out_of_inferential_n() -> None:
    records = [
        sign_payload(
            {
                "paper_id": f"paper-{index}",
                "candidate_count": 2,
                "eligible_target_count": 2,
                "informative_target_count": 1,
                "all_committed_targets_eligible": True,
                "exact_decision_unit_weight_contract": True,
                "outer_decision_unit_weight_sum": {
                    "numerator": 2,
                    "denominator": 2,
                },
                "paper_joint_pass": index < 58,
            }
        )
        for index in range(64)
    ]
    report = aggregate_paper_support(records)
    verify_signed_payload(report)
    assert report["paper_successes"] == 58
    assert report["total_precommitted_targets"] == 128
    assert report["target_level_counts_inferential_n"] is False
    assert report["passes"] is True


def test_noncanonical_subset_precommit_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("trace bytes", encoding="utf-8")
    expected = sign_payload(
        {
            "candidate_count": 2,
            "candidates": [{"id": "a"}, {"id": "b"}],
        }
    )
    subset = sign_payload({"candidate_count": 2, "candidates": [{"id": "a"}]})
    monkeypatch.setattr(
        stage_d_all_child_support,
        "precommit_all_depth_one_targets",
        lambda _: expected,
    )
    with pytest.raises(ValueError, match="canonical complete"):
        stage_d_all_child_support.verify_canonical_precommit(trace, subset)


def test_branch_rejects_bad_precommit_before_model_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def fail_precommit(*_: object) -> None:
        raise ValueError("bad precommit")

    def forbidden_run(**_: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        stage_d_all_child_branch_group,
        "verify_canonical_precommit",
        fail_precommit,
    )
    monkeypatch.setattr(
        stage_d_all_child_branch_group,
        "run_empirical_replay",
        forbidden_run,
    )
    with pytest.raises(ValueError, match="bad precommit"):
        stage_d_all_child_branch_group.run_group(
            trace_path=tmp_path / "trace.jsonl",
            precommit={},
            client=object(),  # type: ignore[arg-type]
            master_seed="master",
            temperature=0.7,
            candidate_max_tokens=512,
            continuation_max_tokens=768,
        )
    assert called is False


def _valid_chain(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("trace bytes", encoding="utf-8")
    candidates = [
        {
            "native_call_index_diagnostic_only": index,
            "structural_event_address": f"node-{index}",
            "decision_unit_weight": {"numerator": 1, "denominator": 2},
        }
        for index in (2, 3)
    ]
    committed = sign_payload(
        {
            "candidate_count": 2,
            "candidate_set_sha256": "c" * 64,
            "candidates": candidates,
        }
    )
    pairs = [
        {
            "target_call_index": index,
            "target_node_id": f"node-{index}",
            "alternative_index": alternative,
            "action_seed": alternative,
            "continuation_seed": 900,
        }
        for index in (2, 3)
        for alternative in (1, 2, 3)
    ]
    replay = sign_payload(
        {
            "source_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "precommit_signed_payload_sha256": committed["signed_payload_sha256"],
            "candidate_set_sha256": "c" * 64,
            "master_seed_sha256": hashlib.sha256(b"master").hexdigest(),
            "target_node_ids": ["node-2", "node-3"],
            "decision_unit_weights": [
                {
                    "target_node_id": f"node-{index}",
                    "weight": {"numerator": 1, "denominator": 2},
                }
                for index in (2, 3)
            ],
            "target_count": 2,
            "alternatives_per_target": 3,
            "regenerated_originals": [
                {
                    "target_call_index": index,
                    "target_node_id": f"node-{index}",
                    "continuation_seed": 900,
                }
                for index in (2, 3)
            ],
            "pairs": pairs,
        }
    )
    return trace, committed, replay


def test_replay_chain_rejects_duplicate_alternative_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace, committed, replay = _valid_chain(tmp_path)
    monkeypatch.setattr(
        stage_d_all_child_support,
        "verify_canonical_precommit",
        lambda *_: committed,
    )
    monkeypatch.setattr(
        stage_d_all_child_support,
        "load_trace_records",
        lambda _: [{"id": "trace-1"}],
    )
    monkeypatch.setattr(
        stage_d_all_child_support,
        "audit_trace_file",
        lambda _: SimpleNamespace(calls=[SimpleNamespace(agent_depth=0, turn_index=2)]),
    )
    monkeypatch.setattr(
        stage_d_all_child_support,
        "derive_branch_group_seeds",
        lambda **_: (900, (1, 2, 3)),
    )
    stage_d_all_child_support.verify_replay_chain(
        trace_path=trace,
        committed=committed,
        replay=replay,
        master_seed="master",
    )
    unsigned = {key: value for key, value in replay.items() if key != "signed_payload_sha256"}
    unsigned["pairs"][-1]["alternative_index"] = 2  # type: ignore[index]
    duplicate = sign_payload(unsigned)
    with pytest.raises(ValueError, match="alternative indices"):
        stage_d_all_child_support.verify_replay_chain(
            trace_path=trace,
            committed=committed,
            replay=duplicate,
            master_seed="master",
        )


def test_one_ineligible_committed_target_fails_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = [
        {
            "native_call_index_diagnostic_only": index,
            "decision_unit_weight": {"numerator": 1, "denominator": 3},
        }
        for index in (2, 3, 4)
    ]
    committed = sign_payload(
        {
            "trace_id": "trace-1",
            "paper_id": "paper-1",
            "source_trace_sha256": "a" * 64,
            "candidate_set_sha256": "b" * 64,
            "candidate_count": 3,
            "candidates": candidates,
        }
    )
    replay = sign_payload({"kind": "replay"})
    scorer = sign_payload({"kind": "scorer"})
    replay_path = tmp_path / "replay.json"
    scorer_path = tmp_path / "scorer.json"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    scorer_path.write_text(json.dumps(scorer), encoding="utf-8")
    monkeypatch.setattr(
        stage_d_all_child_support,
        "verify_canonical_precommit",
        lambda *_: committed,
    )
    monkeypatch.setattr(stage_d_all_child_support, "verify_replay_chain", lambda **_: None)
    monkeypatch.setattr(stage_d_all_child_support, "verify_scorer_chain", lambda **_: None)

    def target(**kwargs: object) -> dict[str, object]:
        index = int(kwargs["target_call_index"])
        eligible = index != 4
        return sign_payload(
            {
                "eligible": eligible,
                "informative": index == 2,
                "joint_eligible_and_informative": index == 2,
                "decision_unit_weight": kwargs["decision_unit_weight"],
            }
        )

    monkeypatch.setattr(
        stage_d_all_child_support,
        "_evaluate_precommitted_target",
        target,
    )
    report = stage_d_all_child_support.evaluate_all_precommitted_targets(
        trace_path=tmp_path / "trace.jsonl",
        replay_path=replay_path,
        scorer_path=scorer_path,
        precommit=committed,
        master_seed="master",
    )
    assert report["eligible_target_count"] == 2
    assert report["all_committed_targets_eligible"] is False
    assert report["paper_joint_pass"] is False


def test_complementary_fixture_generator_is_answer_blind_and_balanced(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "fixtures.jsonl"
    manifest = tmp_path / "manifest.json"
    build(dataset, manifest)
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    audit = json.loads(manifest.read_text())
    assert len(rows) == 8
    assert all(len(row["paper"]) == 4000 for row in rows)
    assert all(audit["checks"].values())
    assert audit["partition_contract"]["ordering_input"] == ("public fixture_id only")
    assert audit["partition_contract"]["model_visible_reference_fields"] is False


def _source_row(index: int, split: str) -> dict[str, object]:
    evidence = f"Unique {split} evidence sentence number {index:04d} is present here."
    return {
        "id": f"paper-{split}-{index:04d}",
        "title": f"Paper {index:04d}",
        "abstract": "A neutral abstract.",
        "full_text": {
            "section_name": ["Results"],
            "paragraphs": [[evidence]],
        },
        "qas": {
            "question": [f"What happened in paper {index:04d}?"],
            "question_id": [f"question-{split}-{index:04d}"],
            "answers": [
                {
                    "answer": [
                        {
                            "unanswerable": False,
                            "yes_no": None,
                            "extractive_spans": [evidence],
                            "free_form_answer": "",
                            "evidence": [evidence],
                        }
                    ]
                }
            ],
        },
    }


def test_qasper_extension_seeds_all_old_ids_and_references() -> None:
    old_reference = "Old reference must remain quarantined."
    old = [
        {
            "paper_id": "paper-train-0000",
            "reference_evidence": [old_reference],
        }
    ]
    train = [_source_row(index, "train") for index in range(81)]
    validation = [_source_row(index, "validation") for index in range(32)]
    rows = materialize_extension(
        old_rows=old,
        train_rows=train,
        validation_rows=validation,
        maximum_paper_characters=60_000,
        minimum_span_characters=20,
    )
    assert len(rows) == 112
    assert all(row["paper_id"] != "paper-train-0000" for row in rows)
    assert [row["split"] for row in rows[:64]] == ["successor_support"] * 64
    assert [row["split"] for row in rows[64:80]] == ["successor_science_train"] * 16
    assert [row["split"] for row in rows[80:]] == ["successor_science_eval"] * 32


def test_midpoint_scaffold_matches_frozen_sft_code_contract() -> None:
    scaffold = (ROOT / "configs/stage-d/stage-d0-scaffold-fewshot-v3.txt").read_text(
        encoding="utf-8"
    )
    sft_row = json.loads(
        (ROOT / "datasets/stage-d/scaffold-sft-v2.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    code = json.loads(sft_row["messages"][2]["tool_calls"][0]["function"]["arguments"])["code"]
    for exact_line in (
        "midpoint = len(paper_text) // 2",
        "excerpts = [paper_text[:midpoint], paper_text[midpoint:]]",
        "children = await asyncio.gather(*(rlm(prompt) for prompt in prompts))",
    ):
        assert exact_line in scaffold
        assert exact_line in code


def test_midpoint_context_audit_uses_exact_gapless_partition() -> None:
    report = audit_rows(
        [
            {
                "example_id": "example-1",
                "paper_id": "paper-1",
                "paper": "abcdefghij",
                "question": "What changed?",
                "reference_evidence": ["must not be read"],
            }
        ],
        encode=lambda text: list(range(len(text) // 2)),
    )
    row = report["records"][0]
    assert row["midpoint"] == 5
    assert row["shard_characters"] == [5, 5]
    assert row["union_exact"] is True
    assert all(report["checks"].values())
