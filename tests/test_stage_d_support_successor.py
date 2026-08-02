from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

import pytest

from redco.analysis.stage_d_collection import StageDCollectionPlan

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stage_d_qasper_builder",
    ROOT / "scripts/build_stage_d_qasper_extension_v1.py",
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
EXTENSION_SPLITS = BUILDER.EXTENSION_SPLITS
_decode_rows = BUILDER._decode_rows
_encode_rows = BUILDER._encode_rows
_write_fresh_bundle = BUILDER._write_fresh_bundle
materialize_support_successor = BUILDER.materialize_support_successor


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_frozen_support_successor_preserves_rows_and_addresses() -> None:
    prior_bytes = (
        ROOT / "datasets/stage-d/qasper-successor-extension-v1.jsonl"
    ).read_bytes()
    successor_bytes = (
        ROOT / "datasets/stage-d/qasper-support-successor-v1.jsonl"
    ).read_bytes()
    manifest = json.loads(
        (ROOT / "datasets/stage-d/qasper-support-successor-manifest-v1.json").read_bytes()
    )
    prior_rows = _decode_rows(prior_bytes)
    successor_rows = _decode_rows(successor_bytes)
    assert _encode_rows(prior_rows) == prior_bytes
    assert _encode_rows(successor_rows) == successor_bytes
    prior_lines = {
        row["example_id"]: line
        for row, line in zip(prior_rows, prior_bytes.splitlines(keepends=True), strict=True)
    }
    successor_lines = {
        row["example_id"]: line
        for row, line in zip(
            successor_rows, successor_bytes.splitlines(keepends=True), strict=True
        )
    }
    shared = set(prior_lines) & set(successor_lines)
    assert len(shared) == 111
    assert all(prior_lines[key] == successor_lines[key] for key in shared)
    assert set(prior_lines) - set(successor_lines) == {
        "qasper-71f2b368228a748fd348f1abf540236568a61b07"
    }
    assert set(successor_lines) - set(prior_lines) == {
        manifest["successor"]["reserve"]["example_id"]
    }
    assert Counter(row["split"] for row in successor_rows) == Counter(EXTENSION_SPLITS)
    assert _sha256(successor_bytes) == manifest["output"]["sha256"]

    old_plan = StageDCollectionPlan.from_bytes(
        (ROOT / "configs/stage-d/stage-d1-support-collection-plan-v1.json").read_bytes()
    )
    new_plan = StageDCollectionPlan.from_bytes(
        (ROOT / "configs/stage-d/stage-d1-support-collection-plan-v4.json").read_bytes()
    )
    old_slots = {slot.example_id: slot.to_payload() for slot in old_plan.slots}
    new_slots = {slot.example_id: slot.to_payload() for slot in new_plan.slots}
    common = set(old_slots) & set(new_slots)
    assert len(common) == 63
    assert all(old_slots[key] == new_slots[key] for key in common)
    assert new_plan.plan_sha256 == manifest["collection_plan"]["sha256"]


def test_successor_builder_rejects_partial_or_reordered_input() -> None:
    prior_rows = _decode_rows(
        (ROOT / "datasets/stage-d/qasper-successor-extension-v1.jsonl").read_bytes()
    )
    with pytest.raises(ValueError, match="112 rows"):
        materialize_support_successor(
            old_rows=[],
            prior_rows=prior_rows[:-1],
            train_rows=[],
            retired_example_id=prior_rows[0]["example_id"],
            maximum_paper_characters=60_000,
            minimum_span_characters=20,
        )
    reordered = list(prior_rows)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="first support row"):
        materialize_support_successor(
            old_rows=[],
            prior_rows=reordered,
            train_rows=[],
            retired_example_id=prior_rows[0]["example_id"],
            maximum_paper_characters=60_000,
            minimum_span_characters=20,
        )


def test_fresh_bundle_never_promotes_manifest_after_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "dataset.jsonl"
    plan = tmp_path / "plan.json"
    manifest = tmp_path / "manifest.json"
    real_replace = os.replace
    calls = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fixture promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="fixture promotion failure"):
        _write_fresh_bundle(
            [(data, b"data"), (plan, b"plan"), (manifest, b"manifest")]
        )
    assert data.read_bytes() == b"data"
    assert not plan.exists()
    assert not manifest.exists()
    assert not list(tmp_path.glob(".*.tmp"))
