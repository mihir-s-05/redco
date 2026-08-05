from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from redco.analysis.stage_d_v12_finalization_audit import (
    ARCHIVE_SHA256,
    EVIDENCE_MANIFEST_SHA256,
    FROZEN_ARCHIVE_ROOT_RELATIVE,
    FROZEN_REPO_FILE_SHA256,
    _reject_output_alias,
    _require_sha256,
    _source_hashes,
    _validate_output_path,
    _validate_terminal_report_schema,
    audit_archive,
)
from redco.contracts import canonical_json

ROOT = Path(__file__).parents[1]
ARCHIVE = ROOT / "runs" / "stage-d" / "stage-d1-support-v12-terminal.tar.gz"
MANIFEST = ROOT / "runs" / "stage-d" / "stage-d1-support-v12-evidence-sha256.txt"
TERMINAL_REPORT = ROOT / "reports" / "stage-d1-support-v12-terminal.json"


def _audit() -> dict[str, Any]:
    return audit_archive(
        ARCHIVE,
        MANIFEST,
        repo_root=ROOT,
        terminal_report=TERMINAL_REPORT,
    )


def test_terminal_v12_audit_is_total_and_immutable() -> None:
    report = _audit()
    assert report["status"] == "engineering_audit_only"
    assert report["inputs_untouched"] is True
    assert report["hashes"]["archive_sha256"] == ARCHIVE_SHA256
    assert report["hashes"]["evidence_manifest_sha256"] == EVIDENCE_MANIFEST_SHA256
    assert report["archive_manifest"]["all_match"] is True
    assert report["archive_manifest"]["listed_count"] == 38
    assert report["terminal_trace"] == {
        "id": "c7834d223cdb4f568d5835314e04d5ba",
        "stop_condition": "max_total_tokens",
        "is_completed": True,
        "ok": True,
        "call_count": 4,
        "node_count": 11,
        "root_call_count": 2,
        "child_call_count": 2,
        "evidence_file_count": 38,
        "model_call_count": 4,
    }
    assert len(report["calls"]) == 4
    assert report["downstream_audit"] == {
        "status": "pass",
        "call_count_audited": 4,
        "post_call_invariants_executed": True,
        "semantic_reconstruction_executed": True,
    }
    for call in report["calls"]:
        failures = [item["name"] for item in call["invariants"] if item["status"] == "fail"]
        assert failures == (["message_comparison"] if call["node"] == 10 else [])
        assert {
            item["name"]
            for item in call["invariants"]
            if item["status"] == "not_observable_from_persisted_schema"
        } == {"call_fields_exact", "sampler", "usage", "successful_call_error"}
    capped = next(item for item in report["calls"] if item["node"] == 10)
    assert capped["decision_id"] == "decision-9eb4cf0ed5732ad818483e5f"
    assert capped["lineage"] == "root/cbb0fb0fe8ca42bac12b7225"
    assert capped["completion_tokens"] == 768
    assert capped["termination_kind"] == "max_tokens"
    assert capped["message"]["canonical_equal_under_current_finalizer"] is False
    assert capped["message"]["normalized_differences"][0]["pointer"] == "/content"
    assert capped["message"]["normalized_differences"][0]["left"]["presence"] == ("present-null")
    assert capped["message"]["normalized_differences"][0]["right"]["presence"] == ("absent")
    assert report["ledger"] == {
        "status": "poisoned",
        "reason": "ledger records an aborted source rollout finalization",
        "record_count": 14,
        "evidence_count": 13,
        "chain_and_poison_invariant": {
            "name": "ledger_chain_and_poison",
            "status": "pass",
            "method": "directly_verified_from_archive",
        },
    }
    assert len(report["action_evidence_hashes"]) == 4
    assert all(item["status"] == "pass" for item in report["action_evidence_hashes"])
    assert all(item["status"] == "pass" for item in report["post_call_invariants"])
    assert {item["name"] for item in report["semantic_reconstruction"]["checks"]} == {
        "episode_schema_and_trace_contract",
        "deployed_parent_links",
        "strict_scaffold_eligibility",
        "sampled_node_call_bijection",
        "sampled_node_mask_shape",
        "leaf_path_sample_derivation",
        "exactly_once_sampled_node_routing",
        "finite_reward_summation",
        "child_target_roster",
        "graph_to_source_mappings",
        "child_weight_normalization",
        "source_semantic_equivalence",
    }
    assert all(
        item["status"] == "reconstructed_on_disposable_copy" and item["result"] == "pass"
        for item in report["semantic_reconstruction"]["checks"]
    )
    assert report["source_finalization"]["production_status"] == "aborted"
    assert report["source_finalization"]["abort_receipt_count"] == 1
    assert report["source_finalization"]["completion_receipt_count"] == 0
    assert report["source_finalization"]["committed_source_artifacts"] == 0
    assert report["source_finalization"]["pending_source_artifacts"] == 0
    assert report["source_finalization"]["abort_receipt"]["error_sha256"] == (
        "4ef73345e848461568b2be8c93d80d8700715ff608c196d12eab53b099249dac"
    )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("name", "call"),
    [
        (
            "wrong-report",
            lambda root: {
                "terminal_report": root
                / "configs/stage-d/stage-d1-source-comparison-contract-v1.json"
            },
        ),
        ("wrong-root", lambda root: {"repo_root": root / "does-not-authenticate"}),
    ],
)
def test_immutable_input_aliases_are_rejected(
    name: str,
    call: Callable[[Path], dict[str, Path]],
    tmp_path: Path,
) -> None:
    del name
    kwargs = call(ROOT)
    with pytest.raises(ValueError):
        audit_archive(
            ARCHIVE,
            MANIFEST,
            repo_root=kwargs.get("repo_root", ROOT),
            terminal_report=kwargs.get("terminal_report", TERMINAL_REPORT),
        )
    assert not list(tmp_path.iterdir())


def test_wrong_archive_hash_is_rejected_before_extraction(tmp_path: Path) -> None:
    wrong = tmp_path / "archive.tar.gz"
    wrong.write_bytes(b"not-the-frozen-archive")
    with pytest.raises(ValueError, match="hash"):
        _require_sha256(wrong, ARCHIVE_SHA256, "terminal archive")


def test_missing_expected_repository_file_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        _source_hashes(Path("C:/definitely-missing-redco-root"))


def test_wrong_terminal_report_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="schema version"):
        _validate_terminal_report_schema({"schema_version": 2})


def test_output_aliases_and_frozen_descendants_are_rejected() -> None:
    representative_immutable_inputs = (
        *tuple(
            ROOT / relative
            for relative in (
                "configs/stage-d/stage-d1-support-protocol-v12.json",
                "src/redco/analysis/stage_d_source_producer.py",
                "pyproject.toml",
                "uv.lock",
            )
        ),
        ARCHIVE,
        MANIFEST,
        TERMINAL_REPORT,
        ROOT / FROZEN_ARCHIVE_ROOT_RELATIVE / "audit.json",
    )
    for output in representative_immutable_inputs:
        with pytest.raises(ValueError, match="immutable"):
            _validate_output_path(output, ROOT)
    safe = _validate_output_path(
        ROOT / "reports" / "stage-d1-support-v12-finalization-audit-v2.json",
        ROOT,
    )
    assert safe.name.endswith(".json")


def test_output_hard_link_alias_is_rejected_without_touching_frozen_inputs(
    tmp_path: Path,
) -> None:
    frozen_copy = tmp_path / "protocol-copy.json"
    frozen_copy.write_bytes(
        (ROOT / "configs/stage-d/stage-d1-support-protocol-v12.json").read_bytes()
    )
    hard_link = tmp_path / "hard-link.json"
    try:
        os.link(frozen_copy, hard_link)
    except OSError as error:
        pytest.skip(f"hard links are unavailable in this filesystem: {error}")
    assert hard_link.stat().st_ino == frozen_copy.stat().st_ino
    with pytest.raises(ValueError, match="hard-link"):
        _reject_output_alias(hard_link.resolve(), (frozen_copy.resolve(),))
    assert (
        frozen_copy.read_bytes()
        == (ROOT / "configs/stage-d/stage-d1-support-protocol-v12.json").read_bytes()
    )


def test_every_authenticated_repository_file_is_an_immutable_output_input() -> None:
    for relative in FROZEN_REPO_FILE_SHA256:
        with pytest.raises(ValueError, match="immutable"):
            _validate_output_path(ROOT / relative, ROOT)


def test_generated_audit_report_is_canonical_and_reproducible() -> None:
    path = ROOT / "reports" / "stage-d1-support-v12-finalization-audit-v1.json"
    payload = json.loads(path.read_bytes())
    assert canonical_json(payload) == path.read_bytes()
    assert not path.read_bytes().endswith(b"\n")
    assert payload["schema_version"] == 2
    assert payload["scientific_interpretation"].startswith(
        "This output is engineering audit evidence"
    )
    first = _audit()
    second = _audit()
    assert canonical_json(first) == canonical_json(second)
    assert b"redco-v12-audit-" not in canonical_json(first)
