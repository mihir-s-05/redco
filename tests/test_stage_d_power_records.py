from __future__ import annotations

import json
from pathlib import Path

from redco.analysis.stage_d_power_records import materialize
from redco.analysis.stage_d_scaffold_support import _derive_episode_seed
from redco.integrations.signed_subprocess import verify_signed_payload


def test_materialize_preserves_all_missing_slots_as_negatives(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    summary = tmp_path / "run-summary.json"
    traces = tmp_path / "traces"
    targets = tmp_path / "targets"
    output = tmp_path / "power-records"
    traces.mkdir()
    targets.mkdir()
    rows = [
        {
            "split": "power_audit",
            "example_id": f"example-{index}",
            "paper_id": f"paper-{index}",
            "answer_type": (
                "abstractive"
                if index % 3 == 0
                else "extractive"
                if index % 3 == 1
                else "yes_no"
            ),
        }
        for index in range(64)
    ]
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    master_seed = "stage-d-test"
    slots = [
        {
            "slot_id": f"{row['example_id']}::replicate-0",
            "example_id": row["example_id"],
            "replicate": 0,
            "seed": _derive_episode_seed(
                master_seed, row["example_id"], 0
            ),
            "trace_ids": [],
        }
        for row in rows
    ]
    summary.write_text(
        json.dumps({"master_seed": master_seed, "records": slots}) + "\n",
        encoding="utf-8",
    )

    records = materialize(
        summary_path=summary,
        traces_dir=traces,
        target_records_dir=targets,
        dataset_path=dataset,
        selected_initialization_sha256="a" * 64,
        output_dir=output,
    )

    assert len(records) == 64
    assert len(list(output.glob("*.json"))) == 64
    assert all(record["eligible"] is False for record in records)
    assert all(record["seed_contract"] is True for record in records)
    assert all(
        record["reason"] == "missing_trace_or_target_report"
        for record in records
    )
    for record in records:
        verify_signed_payload(record)
