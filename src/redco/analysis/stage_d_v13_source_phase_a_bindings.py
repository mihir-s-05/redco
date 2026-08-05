"""Versioned CPU Phase-A behavior/dependency bindings.

The registry is authenticated by the independent Phase-A approval-anchor
file before any of these values are trusted.  The registry is intentionally
not self-authorizing: changing it changes its raw digest and therefore fails
the external anchor check.
"""

from __future__ import annotations

BEHAVIOR_BINDING_FILES = (
    "src/redco/analysis/stage_d_v13_source_phase_a.py",
    "src/redco/analysis/stage_d_v13_source_phase_a_decoder.py",
    "src/redco/analysis/stage_d_v13_source_phase_a_selector.py",
    "src/redco/analysis/stage_d_v13_source_phase_a_witness.py",
    "src/redco/analysis/stage_d_v13_source_phase_a_publication.py",
    "scripts/build_stage_d_v13_source_phase_a.py",
    "tests/test_stage_d_v13_source_phase_a.py",
    "scripts/build_stage_d_qasper_extension_v1.py",
    "src/redco/analysis/stage_d_collection.py",
    "src/redco/analysis/stage_d_v13_draft.py",
    "src/redco/analysis/stage_d_v13_draft_inputs.py",
    "src/redco/analysis/stage_d_v13_draft_publication.py",
    "pyproject.toml",
    "uv.lock",
)

# Filled with the approved post-review bytes once the source/test tree is stable.
APPROVED_BEHAVIOR_HASHES: dict[str, str] = {
    "src/redco/analysis/stage_d_v13_source_phase_a.py": (
        "2da37bc2eca3f1708a7b4bc0aef5f3d4d859f3d9506f2375a38ac24bbb2ca0ec"
    ),
    "src/redco/analysis/stage_d_v13_source_phase_a_decoder.py": (
        "c20cc2fa1d8d3092d76ebcedb205e80e7a4e540dece8e833d6ec5b51c39985eb"
    ),
    "src/redco/analysis/stage_d_v13_source_phase_a_selector.py": (
        "f8ba3ef87879d36343aa517e3177add03de22d26755f360f6e8eb9e93903db59"
    ),
    "src/redco/analysis/stage_d_v13_source_phase_a_witness.py": (
        "3dc86cf0bc7570f887e9802fe5577b1043bf96570df654762a94a48c2d22851b"
    ),
    "src/redco/analysis/stage_d_v13_source_phase_a_publication.py": (
        "d463986f3c290f6ba5a4d9fe418dbebc853ad693bd548110de1abda156edea7d"
    ),
    "scripts/build_stage_d_v13_source_phase_a.py": (
        "fed5d4c0f8ba282c384db453d41c7b5dac57997f8a64319abcbc4eeb57732df5"
    ),
    "tests/test_stage_d_v13_source_phase_a.py": (
        "f69b0fcf025ff725e83e303b81299c73937b50fa14051f3967b179da6d3408f1"
    ),
    "scripts/build_stage_d_qasper_extension_v1.py": (
        "03a0f9be25134eb94673eba4b3bb442f673470613eac7ddbd39039dc4c01545f"
    ),
    "src/redco/analysis/stage_d_collection.py": (
        "acf3ad753d5ffd190666e883b27251ae81b669d0ac6e162a4f34d24c8210bb93"
    ),
    "src/redco/analysis/stage_d_v13_draft.py": (
        "d4acac0fd3932e91d845737ae3a42073ed46cbf14765d03db6629cd95639c43b"
    ),
    "src/redco/analysis/stage_d_v13_draft_inputs.py": (
        "9a889ceed6d0ebc12db175f3aaccd04bb019fc2b96c017eddb39228daf7977bd"
    ),
    "src/redco/analysis/stage_d_v13_draft_publication.py": (
        "3b02427b4a84e3123b9b7568c49b281bbea91ee712c8b14c904f36978bc82755"
    ),
    "pyproject.toml": "94c85ca6ffd627b07cfee14ce8ba80b3cb19fb279e7b98792fddf12695e0699b",
    "uv.lock": "60e9fe7396d45d8e8edd13d2de708fa4895452410b43e1ad860f720047634d31",
}
APPROVED_DERIVATION_VECTOR: dict[str, object] = {
    "namespace": "redco-stage-d1-support-v1",
    "master_seed": "redco-stage-d1-support-v1-20260802-78b65e4cc16ac31f",
    "example_id": "qasper-f33236ebd6f5a9ccb9b9dbf05ac17c3724f93f91",
    "group_id": "stage-d-group-ecddeefc288bf06b1626e644",
    "rollout_slot": 0,
    "seed": 39582467,
    "cache_salt": "stage-d-source-3b2a5dab825bfb03182108d423a5dfd161acd914e91caef84a91862634d8cc0d",
    "slot_id": "source-slot-c21e31d65aa7a219cf64af0a",
    "group_domain": "redco-stage-d-scientific-group-v1",
    "seed_domain": "redco-stage-d-source-episode-seed-v1",
    "slot_domain": "redco-stage-d-source-slot-v1",
    "hmac": {
        "algorithm": "HMAC-SHA256",
        "key": "master_seed",
        "seed_bytes": "first_8_big_endian_mod_2^31",
        "cache_salt_prefix": "stage-d-source-",
    },
    "canonical_json": {
        "sort_keys": True,
        "ensure_ascii": False,
        "allow_nan": False,
        "trailing_newline": False,
    },
}
PHASE_A_STATUS_SIGNATURE = "db7ea9bea40fc2194391881f15124c9b1ddecb30409189b623eea95460a99fb2"

PHASE_B_BINDING_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-v13-phase-b-binding-b-v1.json"
)
PHASE_B_BINDING_DOMAIN = "redco-stage-d1-support-v13-phase-b-binding-b-v1"
PHASE_B_AUTHORIZATION_DOMAIN = (
    "redco-stage-d1-support-v13-phase-b-authorization-c-v2"
)

FOUNDATION_STATUS_ENVELOPE: dict[str, object] = {
    "draft_unfrozen": True,
    "launch_authorized": False,
    "provider_calls_authorized": False,
    "phase_b_authorized": False,
    "foundation_only": True,
    "non_authorizing": True,
    "candidate": {
        "source_ordinal": None,
        "paper_id": None,
        "example_id": None,
        "row": None,
        "seed": None,
        "address": None,
    },
    "seed": None,
    "address": None,
    "status_signature": PHASE_A_STATUS_SIGNATURE,
}

APPROVED_DECODER_RULES: dict[str, object] = {
    "phase_a_cutoff": 179,
    "phase_a_batch_size": 180,
    "phase_b_start_ordinal": 180,
    "phase_b_batch_size": 180,
    "row_groups": [0],
    "use_threads": False,
    "logical_readahead": False,
    "metadata_only_for_authentication": True,
}

APPROVED_COLLISION_DISPOSITIONS: dict[str, str] = {
    "paper_id_collision": "continue_source_order_scan",
    "reference_span_collision": "continue_source_order_scan",
    "example_id_collision": "terminal_fail_closed",
    "rendered_paper_collision": "terminal_fail_closed",
    "source_row_collision": "terminal_fail_closed",
    "source_address_collision": "terminal_fail_closed",
}

PHASE_B_RESUME_CONTRACT: dict[str, object] = {
    "schema_version": 1,
    "version": "stage-d-v13-phase-b-resume-v2",
    "start_ordinal": 180,
    "batch_size": 180,
    "row_groups": [0],
    "use_threads": False,
    "logical_readahead": False,
    "source_order": "physical_ordinal",
    "authorization_state": "reviewed_preselection_checkpoint",
    "binding_artifact": (
        "configs/stage-d/v13-draft/"
        "stage-d1-support-v13-phase-b-binding-b-v1.json"
    ),
    "authorization_artifact": (
        "configs/stage-d/v13-draft/"
        "stage-d1-support-v13-phase-b-authorization-c-v1.json"
    ),
    "checkpoint_identities": {
        "foundation_f": "binding_b.foundation_commit_and_direct_parent",
        "binding_b": "authorization_c.binding_commit_and_direct_parent",
        "authorization_c": "exact_current_head_commit",
        "pre_f_parent": (
            "c41fd18446cecf1c7c98e5aa3a962d1568072c1b"
            "(F-parent proof only; never F identity)"
        ),
    },
    "binding_b_schema": {
        "domain": "redco-stage-d1-support-v13-phase-b-binding-b-v1",
        "non_authorizing": True,
        "self_commit_or_blob_hash_forbidden": True,
    },
    "git_authentication": {
        "fixed_artifact_path_from_exact_head_tree": True,
        "worktree_bytes_must_match_committed_blob": True,
        "c_identity_derived_from_exact_current_head": True,
        "external_approval_is_containing_exact_git_commit": True,
        "checkpoint_chain": "pre_f_parent_to_f_to_direct_b_to_direct_c",
        "binding_b_is_direct_parent_of_c": True,
        "foundation_f_is_direct_parent_of_b": True,
        "foundation_f_identity_is_derived_from_b": True,
        "pre_f_parent_is_not_f_identity": True,
        "binding_b_diff_allowlist": [
            "configs/stage-d/v13-draft/"
            "stage-d1-support-v13-phase-b-binding-b-v1.json"
        ],
        "c_diff_allowlist": [
            "configs/stage-d/v13-draft/"
            "stage-d1-support-v13-phase-b-authorization-c-v1.json"
        ],
        "artifact_self_blob_or_commit_hash_forbidden": True,
        "caller_supplied_path_bytes_commits_forbidden": True,
        "replace_refs_and_git_path_environment_forbidden": True,
    },
        "foundation_f_parent_for_reviewed_tree_only": (
        "c41fd18446cecf1c7c98e5aa3a962d1568072c1b"
    ),
    "foundation_f_state": "C_absent_and_unusable",
    "production_source_path": "qasper/train/0000.parquet",
    "production_source_revision": "06806e4608976fc2fac0a090ac425d5b2b29caf4",
    "production_source_sha256": (
        "9af08092ee26c4f700202c1f90d1592b662926f23f3a308a10ff0a53345e37fe"
    ),
    "production_source_schema_sha256": (
        "85c0addf53d5cfbcb709744f75a3a5f47272b854db453173f6cde96666cf965b"
    ),
    "production_source_row_count": 888,
    "phase_a_reachability": "unreachable_without_reviewed_preselection_checkpoint",
    "phase_a_resume_invocations_required": 0,
    "selector_and_derivation": "phase_a_canonical_selector_and_v1_scientific_law",
    "activation_scope": {
        "source_selection_authorized": True,
        "launch_authorized": False,
        "provider_calls_authorized": False,
        "model_calls_authorized": False,
        "prime_gpu_scientific_launch_authorized": False,
        "caller_authority": False,
        "phase_b_source_selection_only": True,
    },
}


__all__ = [
    "APPROVED_BEHAVIOR_HASHES",
    "APPROVED_COLLISION_DISPOSITIONS",
    "APPROVED_DECODER_RULES",
    "APPROVED_DERIVATION_VECTOR",
    "BEHAVIOR_BINDING_FILES",
    "FOUNDATION_STATUS_ENVELOPE",
    "PHASE_A_STATUS_SIGNATURE",
    "PHASE_B_AUTHORIZATION_DOMAIN",
    "PHASE_B_BINDING_DOMAIN",
    "PHASE_B_BINDING_RELATIVE",
    "PHASE_B_RESUME_CONTRACT",
]
