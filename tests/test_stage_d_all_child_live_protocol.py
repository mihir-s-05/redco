from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from redco.integrations.signed_subprocess import sign_payload, verify_signed_payload

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_stage_d_runtime_context_v1 as context_audit  # noqa: E402
from audit_stage_d_all_child_live_plan_v1 import (  # noqa: E402
    audit_dry_run,
    audit_progress,
    audit_summary,
)
from audit_stage_d_fixture_integration_v1 import audit as audit_fixture  # noqa: E402
from generate_stage_d_all_child_live_runner_v1 import generate  # noqa: E402
from generate_stage_d_all_child_live_runner_v1_1 import generate as generate_repair  # noqa: E402

SLOTS = ROOT / "configs/stage-d/stage-d0-all-child-live-slots-v1.json"


def _manifest() -> dict[str, object]:
    value = json.loads(SLOTS.read_text(encoding="utf-8"))
    verify_signed_payload(value)
    return value


def _support_slots() -> list[dict[str, object]]:
    return [row for row in _manifest()["slots"] if row["kind"] == "support"]  # type: ignore[index]


def _paper_record(paper_id: str, passes: bool) -> dict[str, object]:
    return sign_payload(
        {
            "paper_id": paper_id,
            "candidate_count": 2,
            "eligible_target_count": 2 if passes else 1,
            "informative_target_count": 1 if passes else 0,
            "all_committed_targets_eligible": passes,
            "exact_decision_unit_weight_contract": True,
            "outer_decision_unit_weight_sum": {"numerator": 2, "denominator": 2},
            "paper_joint_pass": passes,
        }
    )


def _write_records(tmp_path: Path, outcomes: list[bool]) -> Path:
    records = tmp_path / "records"
    records.mkdir(parents=True)
    slots = _support_slots()[: len(outcomes)]
    for index, (slot, passes) in enumerate(zip(slots, outcomes, strict=True)):
        (records / f"{index:03d}.json").write_text(
            json.dumps(_paper_record(str(slot["paper_id"]), passes)),
            encoding="utf-8",
        )
    return records


def test_frozen_slots_cover_exact_fixture_orders_and_unique_support() -> None:
    manifest = _manifest()
    fixture = [row for row in manifest["slots"] if row["kind"] == "fixture"]  # type: ignore[index]
    support = _support_slots()
    assert manifest["fixture_shard_orders"] == [[0, 1], [1, 0]]
    assert len(fixture) == 2
    assert len(support) == 64
    assert len({row["paper_id"] for row in support}) == 64
    assert len({row["episode_seed"] for row in support}) == 64


def test_dry_run_and_one_slot_summary_require_exact_frozen_address(
    tmp_path: Path,
) -> None:
    slots = _support_slots()
    dry_run = tmp_path / "dry-run.json"
    dry_run.write_text(
        json.dumps(
            {
                "episode_seed_plan": [
                    {
                        "task_position": index,
                        "example_id": slot["example_id"],
                        "replicate": 0,
                        "seed": slot["episode_seed"],
                    }
                    for index, slot in enumerate(slots)
                ]
            }
        ),
        encoding="utf-8",
    )
    assert audit_dry_run(SLOTS, "support", dry_run)["exact"] is True
    tampered = json.loads(dry_run.read_text(encoding="utf-8"))
    tampered["episode_seed_plan"][0]["seed"] += 1
    dry_run.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from frozen slots"):
        audit_dry_run(SLOTS, "support", dry_run)

    slot = slots[0]
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "master_seed": slot["episode_master_seed"],
                "records": [
                    {
                        "slot_id": slot["slot_id"],
                        "example_id": slot["example_id"],
                        "replicate": 0,
                        "seed": slot["episode_seed"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert audit_summary(SLOTS, "support", 0, summary)["exact"] is True


def test_progress_forbids_early_success_and_stops_at_seventh_failure(
    tmp_path: Path,
) -> None:
    early_success = audit_progress(SLOTS, _write_records(tmp_path / "a", [True] * 58))
    assert early_success["decision"] == "continue"
    assert early_success["early_success_forbidden"] is True

    six_failures = audit_progress(
        SLOTS,
        _write_records(tmp_path / "b", [False] * 6 + [True] * 57),
    )
    assert six_failures["decision"] == "continue"
    seventh_failure = audit_progress(
        SLOTS,
        _write_records(tmp_path / "c", [False] * 7),
    )
    assert seventh_failure["decision"] == "terminal_fail"
    assert seventh_failure["observed_prefix"] == 7


def test_complete_progress_uses_all_64_papers_and_rejects_reordering(
    tmp_path: Path,
) -> None:
    complete = _write_records(tmp_path / "complete", [True] * 58 + [False] * 6)
    report = audit_progress(SLOTS, complete)
    assert report["decision"] == "pass"
    assert report["observed_prefix"] == 64
    assert report["aggregate"]["passes"] is True

    first = complete / "000.json"
    payload = json.loads(first.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in payload.items() if key != "signed_payload_sha256"}
    unsigned["paper_id"] = str(_support_slots()[1]["paper_id"])
    first.write_text(json.dumps(sign_payload(unsigned)), encoding="utf-8")
    with pytest.raises(ValueError, match="prefix differs"):
        audit_progress(SLOTS, complete)


def _target_audit(*, semantic_passes: bool, structural_passes: bool) -> dict[str, object]:
    reports = []
    for _ in range(2):
        reports.append(
            sign_payload(
                {
                    "exact_field_checks": {
                        "source_trace_exact": structural_passes,
                        "seed_coverage_exact": structural_passes,
                        "recorded_trace_output_valid": semantic_passes,
                        "regenerated_original_output_valid": semantic_passes,
                        "all_alternative_outputs_valid": semantic_passes,
                    }
                }
            )
        )
    return sign_payload(
        {
            "candidate_count": 2,
            "target_reports": reports,
            "exact_decision_unit_weight_contract": True,
            "outer_decision_unit_weight_sum": {"numerator": 2, "denominator": 2},
            "source_trace_sha256": "a" * 64,
            "precommit_signed_payload_sha256": "b" * 64,
            "candidate_set_sha256": "c" * 64,
            "replay_signed_payload_sha256": "d" * 64,
            "scorer_signed_payload_sha256": "e" * 64,
        }
    )


def test_fixture_semantics_are_diagnostic_but_structure_vetoes(tmp_path: Path) -> None:
    path = tmp_path / "target.json"
    path.write_text(
        json.dumps(_target_audit(semantic_passes=False, structural_passes=True)),
        encoding="utf-8",
    )
    report = audit_fixture(path)
    assert report["passes"] is True
    assert report["diagnostics_not_gate_inputs"]["informativeness"] is True

    path.write_text(
        json.dumps(_target_audit(semantic_passes=True, structural_passes=False)),
        encoding="utf-8",
    )
    assert audit_fixture(path)["passes"] is False


def test_runtime_context_checks_recorded_and_branch_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = [
        SimpleNamespace(
            agent_depth=0,
            turn_index=0,
            prompt_token_ids=tuple(range(200)),
            completion_tokens_reported=100,
        ),
        SimpleNamespace(
            agent_depth=1,
            turn_index=0,
            prompt_token_ids=tuple(range(100)),
            completion_tokens_reported=100,
        ),
        SimpleNamespace(
            agent_depth=1,
            turn_index=0,
            prompt_token_ids=tuple(range(100)),
            completion_tokens_reported=100,
        ),
        SimpleNamespace(
            agent_depth=0,
            turn_index=2,
            prompt_token_ids=tuple(range(6000)),
            completion_tokens_reported=200,
        ),
    ]
    monkeypatch.setattr(
        context_audit,
        "audit_trace_file",
        lambda _: SimpleNamespace(calls=calls),
    )
    assert context_audit.audit(tmp_path / "unused.jsonl")["passes"] is True
    calls[-1].prompt_token_ids = tuple(range(7000))
    assert context_audit.audit(tmp_path / "unused.jsonl")["passes"] is False


def test_generated_runner_preserves_atomic_order_and_separates_fixture_semantics() -> None:
    parent = (ROOT / "scripts/run_stage_d0_scaffold_support_v4_6.sh").read_text(encoding="utf-8")
    runner = generate(parent)
    assert runner.index('>"$work/ADDRESS_STARTED"') < runner.index('run_eval "$selected_model"')
    assert runner.index('touch "$work/RECORDED_COMPLETION_OBSERVED"') < runner.index(
        'mv "$work" "$run_root/completed/$kind-$slot"'
    )
    assert "scripts/audit_stage_d_fixture_integration_v1.py" in runner
    assert 'done <"$materialized/fixture.tsv"' in runner
    assert 'done <"$materialized/support.tsv"' in runner
    assert 'if test "$decision" = terminal_fail' in runner
    assert 'if test "$decision" = pass' not in runner.split('done <"$materialized/support.tsv"')[0]


def test_live_parser_accepts_both_frozen_successor_splits() -> None:
    source = (
        ROOT
        / "environments/redco_evidence_selection_v2/redco_evidence_selection_v2"
        / "run_feasibility_successor_v1.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SUCCESSOR_SPLITS"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Tuple)
    values = {
        element.value for element in assignments[0].value.elts if isinstance(element, ast.Constant)
    }
    assert {"successor_fixture", "successor_support"} <= values
    runner = generate_repair(
        (ROOT / "scripts/run_stage_d0_scaffold_support_v4_6.sh").read_text(encoding="utf-8")
    )
    assert runner.count("redco_evidence_selection_v2.run_feasibility_successor_v1") == 2
