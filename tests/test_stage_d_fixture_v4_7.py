from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "datasets/stage-d/evidence-selection-fixture-v2.jsonl"
PARENT_FIXTURE = (
    ROOT / "datasets/stage-d/evidence-selection-fixture-v1.jsonl"
)
sys.path.insert(0, str(ROOT / "scripts"))

from audit_stage_d_fixture_migration_v4_7 import (  # noqa: E402
    audit_migration,
)
from audit_stage_d_fixture_schema_v4_7 import audit_fixture  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_fixture_schema_is_complete_and_extractively_typed() -> None:
    report = audit_fixture(
        FIXTURE,
        _sha256(FIXTURE),
        parent_path=PARENT_FIXTURE,
    )
    assert report["passes"]
    assert len(report["rows"]) == 3
    assert report["parent_check"]["only_answer_type_added"]
    assert {
        row["answer_type"] for row in report["rows"]
    } == {"extractive"}


def test_v2_fixture_rejects_missing_answer_type(tmp_path: Path) -> None:
    row = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    row.pop("answer_type")
    broken = tmp_path / "broken.jsonl"
    broken.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="answer_type"):
        audit_fixture(broken, _sha256(broken))


def test_v1_to_v2_migration_changes_only_answer_type() -> None:
    report = audit_migration(
        PARENT_FIXTURE,
        FIXTURE,
        ROOT / "configs/stage-d/stage-d0-scaffold-fewshot-v2.txt",
        ROOT
        / "environments/redco_evidence_selection_v2"
        / "redco_evidence_selection_v2/taskset.py",
        ROOT
        / "environments/redco_evidence_selection_v2"
        / "redco_evidence_selection_v2/seeding.py",
        "redco-stage-d0-selected-fixture-v4",
    )
    assert report["passes"]
    assert all(report["checks"].values())
    assert len({row["episode_seed"] for row in report["rows"]}) == 3
    assert len({row["prompt_sha256"] for row in report["rows"]}) == 3


def test_v2_fixture_loads_through_real_taskset() -> None:
    pytest.importorskip("verifiers.v1")
    from redco_evidence_selection_v2.taskset import (
        EvidenceSelectionConfig,
        EvidenceSelectionTaskset,
    )

    scaffold = ROOT / "configs/stage-d/stage-d0-scaffold-fewshot-v2.txt"
    tasks = EvidenceSelectionTaskset(
        EvidenceSelectionConfig(
            dataset_path=FIXTURE,
            dataset_sha256=_sha256(FIXTURE),
            split="audit",
            prompt_profile="fewshot_fixture_v3",
            scaffold_prompt_path=scaffold,
            scaffold_prompt_sha256=_sha256(scaffold),
        )
    ).load()
    assert len(tasks) == 3
    assert all(task.data.answer_type == "extractive" for task in tasks)
    assert all(
        "call exactly two `rlm(...)` children" in task.data.prompt
        for task in tasks
    )
