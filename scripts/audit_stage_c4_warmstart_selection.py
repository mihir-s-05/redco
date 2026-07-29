"""Audit the frozen Stage-C4 factorized warm-start selection protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from redco.analysis.stage_c4_warmstart import (
    SELECTION_THRESHOLDS,
    audit_factorized_dataset,
)

EARLIEST_PASSING_RULE = (
    "Select the earliest ascending optimizer checkpoint satisfying every frozen "
    "check; later checkpoints are not consulted."
)
UNCHANGED_GATE_POLICY = (
    "Every v3 exact-power check remains mandatory and byte-identical; selection "
    "adds stricter margins but changes no campaign threshold."
)
SECOND_FREEZE_CONDITION = (
    "A selected adapter hash, merged-model manifest, complete support report, and "
    "selection terminal record are local before any v4 RL model call."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit(protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dataset_path = Path(protocol["factorized_dataset"]["path"])
    dataset_manifest_path = Path(protocol["factorized_dataset"]["manifest_path"])
    dataset_rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recorded_dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    recomputed_dataset_manifest = audit_factorized_dataset(dataset_rows)

    v3_protocol = json.loads(
        Path(protocol["unchanged_v3_power_gate"]["source_protocol"]).read_text(encoding="utf-8")
    )
    v3_gate = v3_protocol["execution"]["exact_power_gate"]
    expected_v3_gate_sha = protocol["unchanged_v3_power_gate"]["canonical_sha256"]

    source_checks = {
        path: _sha256(Path(path)) == expected
        for path, expected in protocol["source"]["sha256"].items()
    }
    checks = {
        "status_is_frozen_before_candidate_calls": (
            protocol["status"]
            == "frozen_before_any_stage_c4_candidate_model_load_or_optimizer_step"
        ),
        "candidate_steps_are_exactly_1_through_16": (
            protocol["candidate_selection"]["candidate_steps"] == list(range(1, 17))
        ),
        "selection_rule_is_earliest_passing": (
            protocol["candidate_selection"]["rule"] == EARLIEST_PASSING_RULE
        ),
        "selection_thresholds_match_code_exactly": (
            protocol["candidate_selection"]["buffered_thresholds"] == SELECTION_THRESHOLDS
        ),
        "dataset_audit_recomputes_exactly": (
            recorded_dataset_manifest == recomputed_dataset_manifest
        ),
        "dataset_factorization_is_exact": (
            recomputed_dataset_manifest["status"] == "passed"
            and recomputed_dataset_manifest["factorization"]["empirical_total_variation"] == 0.0
            and recomputed_dataset_manifest["factorization"]["empirical_mutual_information_nats"]
            == 0.0
        ),
        "no_joint_or_reward_supervision": (
            recomputed_dataset_manifest["supervision"]["root_and_target_labels_in_same_example"]
            == 0
            and recomputed_dataset_manifest["supervision"]["reward_or_causality_fields"] == 0
        ),
        "v3_gate_hash_matches_frozen_source": (_canonical_sha256(v3_gate) == expected_v3_gate_sha),
        "v3_gate_is_not_relaxed": (
            protocol["unchanged_v3_power_gate"]["policy"] == UNCHANGED_GATE_POLICY
        ),
        "deployed_merged_scoring_is_authoritative": (
            protocol["candidate_selection"]["authoritative_scoring"]["model_form"]
            == "BF16 merged model, never runtime-LoRA scores"
        ),
        "selection_uses_exact_campaign_prefixes": (
            protocol["candidate_selection"]["authoritative_scoring"]["action_cases"]
            == "configs/stage-c4/selection-action-cases.json"
            and protocol["candidate_selection"]["authoritative_scoring"]["root_cases"]
            == "configs/stage-c4/selection-root-cases.json"
        ),
        "selection_has_no_scientific_reward_evaluation": (
            protocol["separation"]["scientific_reward_calls"] == 0
            and protocol["separation"]["rl_optimizer_steps"] == 0
        ),
        "campaign_requires_second_freeze": (
            protocol["separation"]["campaign_freeze_condition"] == SECOND_FREEZE_CONDITION
        ),
        "hardware_is_nonspot_non_a100_non_h100": (
            protocol["hardware"]["spot"] is False
            and "A100" in protocol["hardware"]["forbidden"]
            and "H100" in protocol["hardware"]["forbidden"]
        ),
        "no_persistent_storage": (protocol["hardware"]["persistent_storage"] is False),
        "all_source_hashes_match": all(source_checks.values()),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c4-warmstart-selection-preregistration-audit",
        "passed": all(checks.values()),
        "checks": checks,
        "source_checks": source_checks,
        "dataset_manifest_signed_payload_sha256": (
            recomputed_dataset_manifest["signed_payload_sha256"]
        ),
        "unchanged_v3_power_gate_canonical_sha256": _canonical_sha256(v3_gate),
    }
    payload["signed_payload_sha256"] = _canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
