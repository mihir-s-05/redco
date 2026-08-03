from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
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


def _verify_dependency_bindings(dependency: StageDDependencyStackManifest) -> None:
    for binding in dependency.imported_modules:
        relative = binding.absolute_path.removeprefix("/workspace/redco/")
        assert relative != binding.absolute_path
        if relative.startswith(("src/", "environments/", "scripts/")):
            value = subprocess.run(
                ["git", "show", f"{dependency.redco_commit}:{relative}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
        else:
            value = (ROOT / relative).read_bytes()
        assert _sha256(value) == binding.sha256


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


@pytest.mark.parametrize(
    (
        "prior_version",
        "successor_version",
        "plan_version",
        "audit_version",
        "retired_history",
    ),
    [
        (1, 2, 7, 2, ["1911.03894"]),
        (2, 3, 8, 3, ["1911.03894", "2001.09899"]),
        (3, 4, 9, 4, ["1710.01492", "1911.03894", "2001.09899"]),
        (
            4,
            5,
            10,
            5,
            ["1710.01492", "1911.03894", "1912.01673", "2001.09899"],
        ),
        (
            5,
            6,
            11,
            6,
            [
                "1710.01492",
                "1909.12231",
                "1911.03894",
                "1912.01673",
                "2001.09899",
            ],
        ),
    ],
)
def test_later_support_successors_preserve_history_and_addresses(
    prior_version: int,
    successor_version: int,
    plan_version: int,
    audit_version: int,
    retired_history: list[str],
) -> None:
    prior_bytes = (
        ROOT / f"datasets/stage-d/qasper-support-successor-v{prior_version}.jsonl"
    ).read_bytes()
    successor_path = (
        ROOT / f"datasets/stage-d/qasper-support-successor-v{successor_version}.jsonl"
    )
    successor_bytes = successor_path.read_bytes()
    manifest = json.loads(
        (
            ROOT
            / f"datasets/stage-d/qasper-support-successor-manifest-v{successor_version}.json"
        ).read_bytes()
    )
    audit = json.loads(
        (
            ROOT
            / f"reports/stage-d1-support-successor-address-audit-v{audit_version}.json"
        ).read_bytes()
    )
    prior_rows = _decode_rows(prior_bytes)
    successor_rows = _decode_rows(successor_bytes)
    prior_ids = {row["example_id"] for row in prior_rows}
    successor_ids = {row["example_id"] for row in successor_rows}

    assert _encode_rows(successor_rows) == successor_bytes
    assert len(prior_ids & successor_ids) == 111
    assert prior_ids - successor_ids == {audit["retired"]["example_id"]}
    assert successor_ids - prior_ids == {audit["reserve"]["example_id"]}
    assert "qasper-71f2b368228a748fd348f1abf540236568a61b07" not in successor_ids
    assert manifest["successor"]["historically_retired_paper_ids"] == retired_history
    assert _sha256(successor_bytes) == manifest["output"]["sha256"]
    assert _sha256(
        (
            ROOT
            / f"configs/stage-d/stage-d1-support-collection-plan-v{plan_version}.json"
        ).read_bytes()
    ) == manifest["collection_plan"]["sha256"]
    assert _sha256(
        (
            ROOT
            / f"reports/stage-d1-support-successor-address-audit-v{audit_version}.json"
        ).read_bytes()
    ) == manifest["address_audit"]["sha256"]
    assert len(audit["preserved"]) == 63
    assert all(audit["checks"].values())


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


def test_successor_builder_never_recycles_retired_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_rows = _decode_rows(
        (ROOT / "datasets/stage-d/qasper-support-successor-v1.jsonl").read_bytes()
    )
    retired = prior_rows[0]
    fresh = {
        **retired,
        "example_id": "fresh-reserve",
        "paper_id": "fresh-paper",
        "paper": "fresh reserve evidence unique to this regression",
        "reference_evidence": ["fresh reserve evidence unique"],
    }

    def select_one(_rows, *, forbidden_paper_ids, selection_receipts, **_kwargs):
        assert "retired-paper" in forbidden_paper_ids
        selection_receipts.append({"selected_example_id": "fresh-reserve"})
        return [fresh]

    monkeypatch.setattr(BUILDER, "select_extension_examples", select_one)

    rows, details = materialize_support_successor(
        old_rows=[],
        prior_rows=prior_rows,
        train_rows=[],
        retired_example_id=retired["example_id"],
        maximum_paper_characters=60_000,
        minimum_span_characters=20,
        historically_retired_paper_ids={"retired-paper"},
    )

    assert rows[63]["example_id"] == "fresh-reserve"
    assert details["historically_retired_paper_ids"] == ["retired-paper"]
    assert details["checks"]["reserve_excluded_from_retired_history"] is True


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
    _verify_dependency_bindings(dependency)

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
    frozen_runner = subprocess.run(
        ["git", "show", f"{dependency.redco_commit}:{source['source_runner']}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert _sha256(frozen_runner) == source["source_runner_sha256"]
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


@pytest.mark.parametrize(
    (
        "version",
        "audit_version",
        "parent_protocol_version",
        "dataset_sha256",
        "retirement_count",
    ),
    [
        (
            7,
            2,
            6,
            "f5b762a5380c976995517a556400f12c44afb4e77d73b1291991762519508408",
            2,
        ),
        (
            8,
            3,
            7,
            "ffd7c6e658ed8cad8278c29a01b97bbeb742e1c552eb63aca2079a6d4ef3c070",
            3,
        ),
        (
            9,
            4,
            8,
            "b7cdd5a0998dcfde739fe5a542b2e8b4dc6e8ef6c18ed7100df81860be3a1735",
            4,
        ),
        (
            10,
            5,
            9,
            "bb576082ba15535d7b0a996ea5c14dd008ebde634a0d8c5c7258f81d5ac9577d",
            5,
        ),
    ],
)
def test_later_successor_protocols_freeze_only_authorized_changes(
    version: int,
    audit_version: int,
    parent_protocol_version: int,
    dataset_sha256: str,
    retirement_count: int,
) -> None:
    paths = {
        "address_audit": ROOT
        / f"reports/stage-d1-support-successor-address-audit-v{audit_version}.json",
        "amendment": ROOT
        / f"configs/stage-d/stage-d1-support-repair-amendment-v{version}.json",
        "collection_plan": ROOT
        / f"configs/stage-d/stage-d1-support-collection-plan-v{version}.json",
        "dependency_stack": ROOT
        / f"configs/stage-d/stage-d1-dependency-stack-v{version}.json",
        "genesis": ROOT
        / f"configs/stage-d/stage-d1-support-genesis-v{version}.json",
        "preregistration": ROOT
        / f"configs/stage-d/stage-d1-support-preregistration-v{version}.json",
        "protocol": ROOT
        / f"configs/stage-d/stage-d1-support-protocol-v{version}.json",
        "replay_config": ROOT
        / f"configs/stage-d/stage-d1-support-replay-eval-v{version}.toml",
        "source": ROOT / f"configs/stage-d/stage-d1-support-source-v{version}.json",
        "source_config": ROOT
        / f"configs/stage-d/stage-d1-support-source-eval-v{version}.toml",
    }
    if version in (8, 9, 10):
        paths["inference_amendment"] = (
            ROOT
            / f"configs/stage-d/stage-d1-support-inference-amendment-v{version}-1.json"
        )
    audit = json.loads(
        (
            ROOT
            / f"reports/stage-d1-support-successor-preregistration-audit-v{version}.json"
        ).read_bytes()
    )
    assert {name: _sha256(path.read_bytes()) for name, path in paths.items()} == audit[
        "hashes"
    ]
    dependency = StageDDependencyStackManifest.from_bytes(
        paths["dependency_stack"].read_bytes()
    )
    protocol = StageDProtocolManifest.from_bytes(paths["protocol"].read_bytes())
    assert dependency.manifest_sha256 == audit["hashes"]["dependency_stack"]
    assert protocol.manifest_sha256 == audit["hashes"]["protocol"]
    assert dependency.redco_commit == audit["redco_commit"]
    _verify_dependency_bindings(dependency)

    old_protocol = json.loads(
        (
            ROOT
            / f"configs/stage-d/stage-d1-support-protocol-v{parent_protocol_version}.json"
        ).read_bytes()
    )
    new_protocol = json.loads(paths["protocol"].read_bytes())
    assert {
        key for key in new_protocol if new_protocol[key] != old_protocol[key]
    } == {
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
    assert (
        len(preregistration["replacement"]["cumulative_retired_example_ids"])
        == retirement_count
    )
    assert (
        preregistration["parent_terminal"].get(
            "scientific_outputs",
            preregistration["parent_terminal"].get("scientific_arm_outcomes"),
        )
        == 0
    )
    assert preregistration["preflight"]["require_zero_skips"] is True
    assert "tests/test_stage_d_qwen_action_regression.py" in preregistration["preflight"][
        "mandatory_prime_tests"
    ]
    assert preregistration["preflight"]["runner"] == {
        "argv_prefix": [
            "uv",
            "run",
            "--active",
            "--no-sync",
            "--with",
            "pytest-asyncio==1.3.0",
            "python",
            "-m",
            "pytest",
            "--asyncio-mode=auto",
        ],
        "apply_pinned_dependency_patches_before_import": True,
        "single_fresh_process": True,
    }
    source = json.loads(paths["source"].read_bytes())
    assert source["action_contract"] == (
        "canonical-engine-token-ids-with-exact-parser-validation-v1"
    )
    frozen_runner = subprocess.run(
        ["git", "show", f"{dependency.redco_commit}:{source['source_runner']}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert _sha256(frozen_runner) == source["source_runner_sha256"]
    assert any(
        binding.name == "redco.analysis.stage_d_exact_action"
        for binding in dependency.imported_modules
    )
    for name in ("source_config", "replay_config"):
        config = tomllib.loads(paths[name].read_text(encoding="utf-8"))
        assert config["env"]["taskset"]["dataset_sha256"] == dataset_sha256
        assert f"stage-d1-support-v{version}" in config["output_dir"]

    if version == 8:
        assert preregistration["parent_terminal"]["negative_support_trace_observed"]
        assert "final bounded successor" in preregistration["repair_rule"]
        amendment = json.loads(paths["amendment"].read_bytes())
        assert amendment["changes"]["episode_contract"] == (
            "persisted-text-node-null-elision-v1"
        )
    if version == 9:
        assert preregistration["parent_terminal"]["negative_support_trace_observed"]
        assert preregistration["user_authorization"]["date"] == "2026-08-02"
        assert preregistration["execution"]["maximum_captured_session_call_count"] == 16
        assert preregistration["execution"]["maximum_harness_policy_turn_count"] == 8
        assert set(preregistration["failure_dispositions"]) == {
            "bounded_redeployment",
            "ordinary_negative_support",
            "terminal_invariant",
            "future_successors",
        }
        assert set(preregistration["forward_plan"]) == {
            "capacity_unavailable",
            "pre_response_infrastructure_failure",
            "support_fail",
            "support_pass",
            "terminal_invariant",
        }
        assert (
            "tests/test_stage_d_live_observer.py::"
            "test_actual_two_turn_child_finalizes_as_excluded_without_replay"
            in preregistration["preflight"]["mandatory_prime_tests"]
        )
        amendment = json.loads(paths["amendment"].read_bytes())
        assert amendment["changes"]["capture_contract"] == (
            "well-formed-root-and-depth1-sessions-through-sixteen-calls-v1"
        )
        assert amendment["changes"]["roster_contract"] == (
            "active-and-excluded-target-partition-v2"
        )
        for name in ("source_config", "replay_config"):
            config = tomllib.loads(paths[name].read_text(encoding="utf-8"))
            assert config["env"]["maximum_captured_session_call_count"] == 16
    if version == 10:
        assert preregistration["parent_terminal"] == {
            "branch_replays": 0,
            "downstream_model_generations": 3,
            "evidence_archive_sha256": (
                "09df217c974aeec9c1eae9485c78877697edc5691a78058f3533e2d9d5ae514f"
            ),
            "negative_support_trace_observed": False,
            "offline_finalizer_audit_sha256": (
                "8b383973b0aaa63c6dd6636580e7108b8244112549e4da9d4c85bcde0c9a78e9"
            ),
            "report_sha256": (
                "f7203c119d89bdcc11178fcc0233f8e3f0e9e14d06600b934e25bbfb3c7cbb7b"
            ),
            "request_contract_failure": True,
            "scientific_arm_outcomes": 0,
            "source_rollouts_committed": 0,
            "support_gate_evaluations": 0,
        }
        assert preregistration["user_authorization"]["date"] == "2026-08-02"
        assert "no automatic successor" in preregistration[
            "failure_dispositions"
        ]["future_successors"].lower()
        assert set(preregistration["failure_dispositions"]) == {
            "pre_response_infrastructure",
            "post_response_infrastructure",
            "ordinary_negative_support",
            "terminal_invariant",
            "future_successors",
        }
        assert set(preregistration["forward_plan"]) == {
            "capacity_unavailable",
            "pre_response_infrastructure_failure",
            "post_response_infrastructure_failure",
            "support_fail",
            "support_pass",
            "terminal_invariant",
        }
        assert (
            "tests/test_stage_d_live_observer.py::"
            "test_actual_two_turn_child_finalizes_as_excluded_without_replay"
            in preregistration["preflight"]["mandatory_prime_tests"]
        )
        amendment = json.loads(paths["amendment"].read_bytes())
        assert amendment["changes"]["request_contract"] == (
            "canonical-compact-or-openai-function-tools-plus-exact-tool-choice-v1"
        )
        for name in ("source_config", "replay_config"):
            config = tomllib.loads(paths[name].read_text(encoding="utf-8"))
            assert config["env"]["maximum_captured_session_call_count"] == 16
