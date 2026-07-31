"""Materialize all 64 Stage D power slots, including denominator negatives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)
from redco.integrations.verifiers_trace import (
    extract_policy_calls,
    load_trace_records,
)

from .stage_d_scaffold_support import _derive_episode_seed


def _signature_valid(value: dict[str, Any]) -> bool:
    try:
        verify_signed_payload(value)
    except ValueError:
        return False
    return True


def materialize(
    *,
    summary_path: Path,
    traces_dir: Path,
    target_records_dir: Path,
    dataset_path: Path,
    selected_initialization_sha256: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    if len(selected_initialization_sha256) != 64:
        raise ValueError("selected initialization hash must be SHA-256")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    master_seed = str(summary["master_seed"])
    tasks = {
        row["example_id"]: row
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row["split"] == "power_audit"
    }
    traces: dict[str, dict[str, Any]] = {}
    for path in sorted(traces_dir.glob("*.jsonl")):
        values = load_trace_records(path)
        if len(values) != 1:
            raise ValueError(f"{path} is not a single trace")
        trace_id = str(values[0].get("id"))
        if trace_id in traces:
            raise ValueError(f"duplicate trace ID {trace_id}")
        traces[trace_id] = values[0]
    target_reports: dict[str, dict[str, Any]] = {}
    for path in sorted(target_records_dir.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        trace_id = str(report.get("trace_id"))
        if trace_id in target_reports:
            raise ValueError(f"duplicate target report {trace_id}")
        if not _signature_valid(report):
            raise ValueError(f"invalid target report signature {path}")
        target_reports[trace_id] = report

    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    referenced_traces: set[str] = set()
    for slot_index, slot in enumerate(summary.get("records") or []):
        example_id = str(slot["example_id"])
        replicate = int(slot["replicate"])
        task = tasks[example_id]
        expected_seed = _derive_episode_seed(
            master_seed, example_id, replicate
        )
        trace_ids = [str(value) for value in slot.get("trace_ids") or []]
        trace = (
            traces.get(trace_ids[0]) if len(trace_ids) == 1 else None
        )
        target_report = (
            target_reports.get(trace_ids[0])
            if len(trace_ids) == 1
            else None
        )
        observed_root_seed = None
        if trace is not None:
            referenced_traces.add(trace_ids[0])
            roots = [
                call
                for call in extract_policy_calls(trace)
                if call.agent_depth == 0
            ]
            observed_root_seed = roots[0].event_seed if roots else None
        seed_contract = (
            expected_seed == slot["seed"]
            and (
                observed_root_seed is None
                or observed_root_seed == slot["seed"]
            )
        )
        report_signature = (
            target_report.get("signed_payload_sha256")
            if target_report is not None
            else None
        )
        eligible = (
            target_report is not None
            and target_report.get("eligible") is True
            and seed_contract
        )
        informative = (
            eligible and target_report.get("informative") is True
        )
        wrapper = sign_payload(
            {
                "schema_version": 1,
                "analysis": "stage-d-power-slot",
                "slot_index": slot_index,
                "slot_id": slot["slot_id"],
                "trace_id": (
                    trace_ids[0]
                    if len(trace_ids) == 1
                    else f"missing::{slot['slot_id']}"
                ),
                "example_id": example_id,
                "paper_id": task["paper_id"],
                "answer_type": task["answer_type"],
                "replicate": replicate,
                "expected_seed": expected_seed,
                "recorded_plan_seed": slot["seed"],
                "observed_root_seed": observed_root_seed,
                "seed_contract": seed_contract,
                "selected_initialization_sha256": (
                    selected_initialization_sha256
                ),
                "target_report_signed_payload_sha256": report_signature,
                "root_calls": (
                    int(target_report.get("root_calls", 0))
                    if target_report is not None
                    else 0
                ),
                "child_calls": (
                    int(target_report.get("child_calls", 0))
                    if target_report is not None
                    else 0
                ),
                "eligible": eligible,
                "informative": informative,
                "joint_eligible_and_informative": (
                    eligible and informative
                ),
                "reason": (
                    target_report.get("reason")
                    if target_report is not None
                    else "missing_trace_or_target_report"
                ),
            }
        )
        output = output_dir / f"{slot_index:03d}.json"
        atomic_write_json(output, wrapper)
        records.append(wrapper)
    if referenced_traces != set(traces):
        raise ValueError("one or more traces were not assigned to a slot")
    return records
