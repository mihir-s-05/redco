"""Build/check the unfrozen v13 Phase A source-authentication artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes
from redco.analysis.stage_d_v13_source_phase_a import (
    PHASE_A_OUTPUTS,
    phase_a_immutable_paths,
    phase_a_payloads,
    validate_foundation_envelope,
    write_phase_a_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
PHASE_A_TEST_NODE_IDS = (
    "test_pinned_source_artifact_metadata",
    "test_historical_receipts_174_through_179",
    "test_six_retired_rows_and_observed_unit",
    "test_forbidden_witness_is_complete_and_hashed",
    "test_phase_a_wall_stops_before_ordinal_180",
    "test_phase_a_candidate_and_launch_flags_remain_unresolved",
    "test_collision_disposition_is_predeclared",
    "test_scientific_binding_reuses_v1_law",
    "test_phase_a_artifacts_are_canonical_and_unfrozen",
    "test_phase_a_cpu_manifest_matches_collection",
    "test_two_fresh_output_roots_are_byte_identical",
    "test_phase_a_check_only_rejects_tampering",
    "test_legacy_datasets_decoder_batch_policy_is_rejected",
    "test_real_pinned_decoder_emits_one_bounded_batch",
    "test_decoder_instrumentation_rejects_oversized_object",
    "test_terminal_selector_collisions_fail_closed_without_later_candidate[example]",
    "test_terminal_selector_collisions_fail_closed_without_later_candidate[rendered]",
    "test_terminal_selector_collisions_fail_closed_without_later_candidate[row]",
    "test_terminal_selector_collisions_fail_closed_without_later_candidate[address]",
    "test_terminal_collision_dominates_continuable_collision[example-paper]",
    "test_terminal_collision_dominates_continuable_collision[rendered-paper]",
    "test_terminal_collision_dominates_continuable_collision[row-paper]",
    "test_terminal_collision_dominates_continuable_collision[address-paper]",
    "test_terminal_collision_dominates_continuable_collision[example-reference]",
    "test_terminal_collision_dominates_continuable_collision[rendered-reference]",
    "test_terminal_collision_dominates_continuable_collision[row-reference]",
    "test_terminal_collision_dominates_continuable_collision[address-reference]",
    "test_multiple_terminal_collisions_have_complete_set_and_primary_reason",
    "test_paper_and_reference_collisions_are_the_only_continuable_set",
    "test_paper_collision_continues_to_next_candidate",
    "test_selector_continues_after_raw_reference_collision",
    "test_selector_universe_contains_all_authenticated_classes",
    "test_forbidden_witness_rebuild_rejects_each_mutation[retired_units]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[retired_papers]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[examples]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[rows]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[rendered]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[references]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[source_addresses]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[historical_addresses]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[old_snapshot_papers]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[predecessor_examples]",
    "test_forbidden_witness_rebuild_rejects_each_mutation[exclusion_hash]",
    "test_forbidden_witness_rejects_recomputed_self_hash",
    "test_phase_a_publication_rejects_authenticated_input_hardlink",
    "test_phase_a_publication_rejects_cross_output_hardlink",
    "test_phase_a_publication_rejects_symlink_parent_before_write",
    "test_phase_a_status_capture_is_independent_and_exact",
    "test_phase_a_missing_source_fails_closed",
    "test_immutable_v1_audit_rejects_repaired_tree",
    "test_phase_a_approval_anchor_authenticates_registry_policy",
    "test_phase_a_approval_anchor_mutations_fail_before_publication[selector]",
    "test_phase_a_approval_anchor_mutations_fail_before_publication[selector_and_registry]",
    "test_phase_a_approval_anchor_mutations_fail_before_publication[registry]",
    "test_phase_a_approval_anchor_mutations_fail_before_publication[derivation]",
    "test_phase_a_approval_anchor_mutations_fail_before_publication[status]",
    "test_phase_a_approval_anchor_mutations_fail_before_publication[resume]",
    "test_phase_a_approval_anchor_mutations_fail_before_publication[anchor]",
    "test_phase_a_resume_invocation_count_is_zero",
    "test_dormant_resume_decoder_starts_at_ordinal_180",
    "test_dormant_resume_decoder_rejects_wrong_binding[source]",
    "test_dormant_resume_decoder_rejects_wrong_binding[schema]",
    "test_dormant_resume_decoder_rejects_wrong_binding[version]",
    "test_dormant_resume_decoder_rejects_wrong_binding[config]",
    "test_dormant_resume_decoder_requires_reviewed_checkpoint",
    "test_foundation_resume_entrypoint_rejects_all_caller_authority",
    "test_future_phase_b_authorization_is_unusable_without_committed_c[missing]",
    "test_future_phase_b_authorization_is_unusable_without_committed_c[wrong_ancestry]",
    "test_future_phase_b_authorization_is_unusable_without_committed_c[working_tree]",
    "test_future_phase_b_authorization_is_unusable_without_committed_c[wrong_source]",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Phase A artifact is not an object: {path}")
    return value


def _validate_payloads(
    root: Path,
    payloads: dict[str, bytes],
    *,
    compare_existing: bool = True,
) -> None:
    if set(payloads) != set(PHASE_A_OUTPUTS):
        raise ValueError("Phase A artifact set differs")
    expected_config = sha256_bytes(payloads[PHASE_A_OUTPUTS[0]])
    expected_hashes = {relative: sha256_bytes(payload) for relative, payload in payloads.items()}
    for relative, payload in payloads.items():
        if not relative.endswith(".json"):
            raise ValueError(f"unexpected Phase A artifact suffix: {relative}")
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError(f"Phase A artifact is not an object: {relative}")
        validate_foundation_envelope(parsed)
        if payload != canonical_json_bytes(parsed):
            raise ValueError(f"Phase A artifact is not canonical: {relative}")
    audit = json.loads(payloads[PHASE_A_OUTPUTS[1]])
    if audit["config"]["sha256"] != expected_config:
        raise ValueError("Phase A audit/config hash reference differs")
    manifest = json.loads(payloads[PHASE_A_OUTPUTS[3]])
    if manifest["phase_a_artifacts"] != {
        relative: expected_hashes[relative]
        for relative in (*PHASE_A_OUTPUTS[:3], PHASE_A_OUTPUTS[4])
    }:
        raise ValueError("Phase A artifact manifest references differ")
    if compare_existing:
        for relative in expected_hashes:
            path = root / relative
            if path.exists() and path.read_bytes() != payloads[relative]:
                raise ValueError(f"Phase A existing bytes differ: {relative}")


def build(*, output_root: Path = ROOT, auth_root: Path = ROOT) -> dict[str, str]:
    payloads = phase_a_payloads(auth_root, test_node_ids=PHASE_A_TEST_NODE_IDS)
    _validate_payloads(output_root, payloads, compare_existing=False)
    return cast(
        dict[str, str],
        write_phase_a_outputs(
            output_root,
            payloads,
            immutable_paths=phase_a_immutable_paths(auth_root),
        ),
    )


def check_only(root: Path = ROOT) -> None:
    payloads = phase_a_payloads(ROOT, test_node_ids=PHASE_A_TEST_NODE_IDS)
    _validate_payloads(root, payloads)
    if any(not (root / relative).is_file() for relative in PHASE_A_OUTPUTS):
        raise FileNotFoundError("Phase A artifact set is incomplete")
    actual = {relative: (root / relative).read_bytes() for relative in PHASE_A_OUTPUTS}
    if actual != payloads:
        raise ValueError("Phase A check-only bytes differ from authenticated rebuild")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.check_only:
        check_only(args.output_root)
        return
    print(json.dumps(build(output_root=args.output_root), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
