from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tomllib
from collections import Counter
from pathlib import Path

import pytest

from redco.analysis.stage_d_collection import StageDCollectionPlan
from redco.analysis.stage_d_dependency_stack import StageDDependencyStackManifest
from redco.analysis.stage_d_protocol_manifest import StageDProtocolManifest

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


def test_successor_protocol_freezes_only_authorized_changes() -> None:
    paths = {
        "address_audit": ROOT / "reports/stage-d1-support-successor-address-audit-v1.json",
        "amendment": ROOT / "configs/stage-d/stage-d1-support-repair-amendment-v4.json",
        "collection_plan": ROOT / "configs/stage-d/stage-d1-support-collection-plan-v4.json",
        "dependency_stack": ROOT / "configs/stage-d/stage-d1-dependency-stack-v3.json",
        "genesis": ROOT / "configs/stage-d/stage-d1-support-genesis-v4.json",
        "preregistration": ROOT / "configs/stage-d/stage-d1-support-preregistration-v4.json",
        "protocol": ROOT / "configs/stage-d/stage-d1-support-protocol-v4.json",
        "replay_config": ROOT / "configs/stage-d/stage-d1-support-replay-eval-v4.toml",
        "source": ROOT / "configs/stage-d/stage-d1-support-source-v4.json",
        "source_config": ROOT / "configs/stage-d/stage-d1-support-source-eval-v4.toml",
    }
    audit = json.loads(
        (
            ROOT
            / "reports/stage-d1-support-successor-preregistration-audit-v4.json"
        ).read_bytes()
    )
    assert {
        name: _sha256(path.read_bytes()) for name, path in paths.items()
    } == audit["hashes"]
    dependency = StageDDependencyStackManifest.from_bytes(
        paths["dependency_stack"].read_bytes()
    )
    protocol = StageDProtocolManifest.from_bytes(paths["protocol"].read_bytes())
    assert dependency.manifest_sha256 == audit["hashes"]["dependency_stack"]
    assert protocol.manifest_sha256 == audit["hashes"]["protocol"]
    for binding in dependency.imported_modules:
        relative = binding.absolute_path.removeprefix("/workspace/redco/")
        assert relative != binding.absolute_path
        assert _sha256((ROOT / relative).read_bytes()) == binding.sha256

    prior_protocol = json.loads(
        (ROOT / "configs/stage-d/stage-d1-support-protocol-v3.json").read_bytes()
    )
    successor_protocol = json.loads(paths["protocol"].read_bytes())
    changed = {
        key
        for key in successor_protocol
        if successor_protocol[key] != prior_protocol[key]
    }
    assert changed == {
        "collection_plan_sha256",
        "dependency_stack_sha256",
        "genesis_config_sha256",
        "preregistration_sha256",
        "scientific_eval_config_sha256",
        "source_eval_config_sha256",
        "source_sha256",
    }
    preregistration = json.loads(paths["preregistration"].read_bytes())
    assert preregistration["replacement"]["preserved_addresses"] == 63
    assert preregistration["parent_terminal"]["typed_responses"] == 0
    assert preregistration["parent_terminal"]["scientific_outputs"] == 0
    assert preregistration["preflight"]["require_zero_skips"] is True
    assert any(
        "test_actual_interception_train_renderer_path" in node
        for node in preregistration["preflight"]["mandatory_prime_tests"]
    )
    source = json.loads(paths["source"].read_bytes())
    assert source["response_witness_required"] is True
    assert source["collection_plan_sha256"] == audit["hashes"]["collection_plan"]
    assert _sha256((ROOT / source["source_runner"]).read_bytes()) == source[
        "source_runner_sha256"
    ]
    genesis = json.loads(paths["genesis"].read_bytes())
    assert genesis["preregistration_sha256"] == audit["hashes"]["preregistration"]
    assert genesis["dependency_stack_sha256"] == audit["hashes"]["dependency_stack"]
    amendment = json.loads(paths["amendment"].read_bytes())
    assert amendment["successor_protocol_sha256"] == audit["hashes"]["protocol"]
    assert amendment["changes"]["source_config_sha256"] == audit["hashes"][
        "source_config"
    ]
    assert amendment["changes"]["replay_config_sha256"] == audit["hashes"][
        "replay_config"
    ]

    for config_name in ("source_config", "replay_config"):
        config = tomllib.loads(paths[config_name].read_text(encoding="utf-8"))
        assert config["env"]["taskset"]["dataset_sha256"] == (
            "d118db801f660d2163fa3bdd676e842da436d69362b754be7d01afff58eabeab"
        )
        assert config["env"]["preregistration_sha256"] == audit["hashes"][
            "preregistration"
        ]
        assert config["env"]["config_sha256"] == audit["hashes"]["genesis"]
        assert "stage-d1-support-v4" in config["output_dir"]
