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
PHASE_C3_AUTHORIZATION_RELATIVE = (
    "configs/stage-d/v13-draft/"
    "stage-d1-support-v13-phase-b-authorization-c3-v1.json"
)
PHASE_B_RESUME_CONTRACT_V2_SHA256 = (
    "cade25b90061b817423307b5e63fb6c76756ac3f5b365671572a6d16eb2e8e08"
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

PHASE_B_RESUME_CONTRACT_V2: dict[str, object] = {
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

# This alias preserves the v2 object and its historical digest for the
# already-committed Binding B.  Repaired runtime validation must use the
# separately versioned v3 contract below.
PHASE_B_RESUME_CONTRACT = PHASE_B_RESUME_CONTRACT_V2

PHASE_B_RESUME_CONTRACT_V3: dict[str, object] = {
    "schema_version": 1,
    "version": "stage-d-v13-phase-b-resume-v3",
    "start_ordinal": 180,
    "batch_size": 180,
    "row_groups": [0],
    "use_threads": False,
    "logical_readahead": False,
    "source_order": "physical_ordinal",
    "authorization_state": "reviewed_preselection_checkpoint",
    "binding_artifact": PHASE_B_BINDING_RELATIVE,
    "authorization_artifact": PHASE_C3_AUTHORIZATION_RELATIVE,
    "legacy_v2_contract_sha256": PHASE_B_RESUME_CONTRACT_V2_SHA256,
    "preselection_checkpoint_sha256": (
        "93915d220f1bcb6357f0910633e6d8f2b5fa7d5727f71ae665f34d5bf36c1e8e"
    ),
    "runtime_versions": {
        "python": "3.12.3",
        "datasets": "5.0.0",
        "pyarrow": "25.0.0",
    },
    "source_contract": {
        "path": "qasper/train/0000.parquet",
        "revision": "06806e4608976fc2fac0a090ac425d5b2b29caf4",
        "sha256": (
            "9af08092ee26c4f700202c1f90d1592b662926f23f3a308a10ff0a53345e37fe"
        ),
        "schema_sha256": (
            "85c0addf53d5cfbcb709744f75a3a5f47272b854db453173f6cde96666cf965b"
        ),
        "row_count": 888,
    },
    "checkpoint_chain": {
        "pre_f_parent": "exact_parent_of_foundation_f",
        "foundation_f": "exact_parent_of_binding_b",
        "binding_b": "exact_parent_of_repair_r",
        "repair_r": "exact_parent_of_authorization_c3",
        "authorization_c3": "exact_current_head_commit",
    },
    "git_authentication": {
        "chain": "pre_f_parent_to_f_to_direct_b_to_direct_r_to_direct_c3",
        "f_to_b_diff": "exact_binding_b_artifact_only",
        "b_to_r_diff": "exact_three_file_repair_allowlist",
        "r_to_c3_diff": "exact_c3_artifact_only",
        "legacy_c_path_forbidden": True,
        "caller_supplied_authority_forbidden": True,
        "self_commit_or_blob_hash_forbidden": True,
        "replace_refs_and_git_path_environment_forbidden": True,
    },
    "activation_scope": {
        "source_selection_authorized": True,
        "launch_authorized": False,
        "provider_calls_authorized": False,
        "model_calls_authorized": False,
        "prime_gpu_scientific_launch_authorized": False,
        "science_authorized": False,
        "caller_authority": False,
        "phase_b_source_selection_only": True,
    },
}

# Selection Gate G is a new, non-authorizing checkpoint after the committed
# Repair R.  The v2/v3 objects and their historical meanings remain frozen.
REPAIR_R_COMMIT = "60f3b0565efa27ec18680a8b8089f53626ef39f6"
PHASE_C3_V2_AUTHORIZATION_RELATIVE = (
    "configs/stage-d/v13-draft/"
    "stage-d1-support-v13-phase-b-authorization-c3-v2.json"
)
PHASE_C3_V2_AUTHORIZATION_DOMAIN = (
    "redco-stage-d1-support-v13-phase-b-authorization-c3-v2"
)
PHASE_A_AUDIT_RELATIVE = "reports/stage-d1-support-v13-source-phase-a-audit-v1.json"
PHASE_A_AUDIT_SHA256 = (
    "e92c6ddb63c86c6fbc99490f0575625f3be12b1c852da284d6f08d92760362bc"
)
PHASE_A_WITNESS_SHA256 = (
    "907e7e7a23cd9aaeb4b4be6e38521f8f522927a8a3eb9149c8088ff2b6b34b8b"
)
SUCCESSOR_EXTENSION_RELATIVE = "datasets/stage-d/qasper-successor-extension-v1.jsonl"
SUCCESSOR_EXTENSION_SHA256 = (
    "8a1b55482d3c5f151741f42ba645b1cad2a0d20e2a445f69432c4e31c0c744b8"
)
SUCCESSOR_EXTENSION_GIT_BLOB_SHA1 = "6f4ebb4d5c44287f0664620f0145784cf5f1bfba"
SUCCESSOR_EXTENSION_MANIFEST_RELATIVE = (
    "datasets/stage-d/qasper-successor-extension-manifest-v1.json"
)
SUCCESSOR_EXTENSION_MANIFEST_SHA256 = (
    "6960fcd92e4c7806cefa70da00dd1d3b65ba69fda1d0c0d1938a5205b4dfda69"
)
SUCCESSOR_EXTENSION_MANIFEST_GIT_BLOB_SHA1 = "5b2d865372270ad15deed71c8b18104cbc0b0765"
SUCCESSOR_MANIFEST_RELATIVE = "datasets/stage-d/qasper-support-successor-manifest-v1.json"
SUCCESSOR_MANIFEST_SHA256 = (
    "f090d4cd382fce120ab3bd3ae15a2123102941e53fbc344f3a2098880d2daba6"
)
SUCCESSOR_MANIFEST_GIT_BLOB_SHA1 = "6bf5ba929d0254dc4ce65fb9eb8717ef8fc76f4c"
SUCCESSOR_ADDRESS_AUDIT_V1_RELATIVE = (
    "reports/stage-d1-support-successor-address-audit-v1.json"
)
SUCCESSOR_ADDRESS_AUDIT_V1_SHA256 = (
    "ee5fc07b6bf76d470d9bcf26e1085d3055a283fb3df3212359353d5b2586d6df"
)
SUCCESSOR_ADDRESS_AUDIT_V1_GIT_BLOB_SHA1 = "c67f1066726c3bca4563b69b524441ff8da41bfe"
SUCCESSOR_EXTENSION_INTRODUCED_COMMIT = "6d1d8543e4056f851b0f0dd6dff9b73701c93fae"
RECOVERED_REFERENCE_SOURCE_ORDINAL = 89
RECOVERED_REFERENCE_PAPER_ID = "1911.03894"
RECOVERED_REFERENCE_EXAMPLE_ID = "qasper-71f2b368228a748fd348f1abf540236568a61b07"
RECOVERED_REFERENCE_QUESTION_INDEX = 0
RECOVERED_REFERENCE_CANONICAL_ROW_SHA256 = (
    "4453db56b2a8f6055ddad911274e9a672d3ae61788d945e6c670e5ec0f7e059a"
)
RECOVERED_REFERENCE_RENDERED_PAPER_SHA256 = (
    "8e77d988ff4e6f2b79232d7f991a00d3092dc10d9560414fd9295a363243583f"
)
RECOVERED_REFERENCE_SHA256 = (
    "f51c54401b49f414690423a810f6bbb204a7e804a41a2d53f665c9e96baa9df0"
)
RECOVERED_REFERENCE_CARDINALITY = 1
RECOVERED_REFERENCE_EXPECTED_DIGEST_COUNT = 414
SELECTION_CLAIM_RELATIVE = "runs/stage-d/stage-d1-support-v13-source-selection-claim-v1.json"
SELECTION_RECEIPT_RELATIVE = (
    "reports/stage-d1-support-v13-source-selection-receipt-v2.json"
)
SELECTION_GATE_APPROVAL_THREAD_ID = "019f9ab9-ec45-7ac3-82b1-09757b92a7c3"
SELECTION_GATE_APPROVAL_TEXT = (
    "I authorize Redco C3-v2 under Selection Gate G 42813cfc64c454fa10579df68f7e9fd8449d6c49 "
    "and scan contract aeb47f23445a0aeddef5a4d66dbec2788b58401367478ed81b3d935a0a16f09f "
    "to commit one candidate-null authorization artifact and perform exactly one local "
    "CPU-only QASPER source-order selection attempt from ordinal 180, stopping at the "
    "first frozen-selector eligible, terminal-collision, or exhausted result. No retry, "
    "provider/model/GPU/Prime/scientific/launch activity is authorized; stop after the "
    "canonical receipt."
)
SELECTION_GATE_APPROVAL_TEXT_SHA256 = (
    "791360ec6dbef7e533729b023f4bb005b898c466b9a176f30f298a467391a768"
)

PHASE_B_SOURCE_SELECTION_CONTRACT_V4: dict[str, object] = {
    "schema_version": 1,
    "version": "stage-d-v13-phase-b-source-selection-v4",
    "authorization_artifact": PHASE_C3_V2_AUTHORIZATION_RELATIVE,
    "authorization_domain": PHASE_C3_V2_AUTHORIZATION_DOMAIN,
    "predecessor_repair_r": REPAIR_R_COMMIT,
    "start_ordinal": 180,
    "final_possible_ordinal": 887,
    "attempt_limit": 1,
    "retry": False,
    "stop_rule": (
        "first_eligible_candidate_or_terminal_identity_collision_or_exhaustion"
    ),
    "source_order": "physical_ordinal",
    "decoder": {
        "batch_size": 180,
        "row_groups": [0],
        "use_threads": False,
        "logical_readahead": False,
        "metadata_only_for_authentication": True,
    },
    "selector": {
        "maximum_paper_characters": 60_000,
        "minimum_exact_span_characters": 20,
        "first_eligible_question_per_paper": True,
        "collision_dispositions": APPROVED_COLLISION_DISPOSITIONS,
    },
    "forbidden_universe": {
        "artifact_path": PHASE_A_AUDIT_RELATIVE,
        "artifact_sha256": PHASE_A_AUDIT_SHA256,
        "witness_sha256": PHASE_A_WITNESS_SHA256,
        "required_sets": [
            "paper_ids",
            "example_ids",
            "rendered_paper_sha256",
            "reference_spans",
            "row_sha256",
            "addresses",
        ],
        "raw_reference_spans_required_in_c3_v2": False,
        "raw_reference_spans_source": "authenticated_committed_historical_artifacts",
        "raw_reference_source_hashes": {
            "datasets/stage-d/qasper-support-successor-v1.jsonl": (
                "d118db801f660d2163fa3bdd676e842da436d69362b754be7d01afff58eabeab"
            ),
            "datasets/stage-d/qasper-support-successor-v2.jsonl": (
                "f5b762a5380c976995517a556400f12c44afb4e77d73b1291991762519508408"
            ),
            "datasets/stage-d/qasper-support-successor-v3.jsonl": (
                "ffd7c6e658ed8cad8278c29a01b97bbeb742e1c552eb63aca2079a6d4ef3c070"
            ),
            "datasets/stage-d/qasper-support-successor-v4.jsonl": (
                "b7cdd5a0998dcfde739fe5a542b2e8b4dc6e8ef6c18ed7100df81860be3a1735"
            ),
            "datasets/stage-d/qasper-support-successor-v5.jsonl": (
                "bb576082ba15535d7b0a996ea5c14dd008ebde634a0d8c5c7258f81d5ac9577d"
            ),
            "datasets/stage-d/qasper-support-successor-v6.jsonl": (
                "153c25a1697737d4df58883adedf55e056d6cd58f08f86e2489391b40b5183ac"
            ),
            "datasets/stage-d/qasper-deterministic-v4.jsonl": (
                "88fa2c114d2f251b8ce0400023980fe652e4733d14b0357f5517f517d5775d71"
            ),
        },
        "recovery_projection": {
            "extension_path": SUCCESSOR_EXTENSION_RELATIVE,
            "extension_sha256": SUCCESSOR_EXTENSION_SHA256,
            "extension_git_blob_sha1": SUCCESSOR_EXTENSION_GIT_BLOB_SHA1,
            "extension_manifest_path": SUCCESSOR_EXTENSION_MANIFEST_RELATIVE,
            "extension_manifest_sha256": SUCCESSOR_EXTENSION_MANIFEST_SHA256,
            "extension_manifest_git_blob_sha1": SUCCESSOR_EXTENSION_MANIFEST_GIT_BLOB_SHA1,
            "successor_manifest_path": SUCCESSOR_MANIFEST_RELATIVE,
            "successor_manifest_sha256": SUCCESSOR_MANIFEST_SHA256,
            "successor_manifest_git_blob_sha1": SUCCESSOR_MANIFEST_GIT_BLOB_SHA1,
            "address_audit_path": SUCCESSOR_ADDRESS_AUDIT_V1_RELATIVE,
            "address_audit_sha256": SUCCESSOR_ADDRESS_AUDIT_V1_SHA256,
            "address_audit_git_blob_sha1": SUCCESSOR_ADDRESS_AUDIT_V1_GIT_BLOB_SHA1,
            "introduced_commit": SUCCESSOR_EXTENSION_INTRODUCED_COMMIT,
            "source_ordinal": RECOVERED_REFERENCE_SOURCE_ORDINAL,
            "paper_id": RECOVERED_REFERENCE_PAPER_ID,
            "example_id": RECOVERED_REFERENCE_EXAMPLE_ID,
            "question_index": RECOVERED_REFERENCE_QUESTION_INDEX,
            "canonical_row_sha256": RECOVERED_REFERENCE_CANONICAL_ROW_SHA256,
            "rendered_paper_sha256": RECOVERED_REFERENCE_RENDERED_PAPER_SHA256,
            "reference_cardinality": RECOVERED_REFERENCE_CARDINALITY,
            "reference_sha256": RECOVERED_REFERENCE_SHA256,
            "expected_reference_digest_count": RECOVERED_REFERENCE_EXPECTED_DIGEST_COUNT,
        },
    },
    "paths": {
        "claim": SELECTION_CLAIM_RELATIVE,
        "receipt": SELECTION_RECEIPT_RELATIVE,
    },
    "derivation": {
        "law": "stage_d_collection.py frozen v1 scientific group and source seed law",
        "outcome_independent_scan_id": True,
    },
    "approval": {
        "thread_id": SELECTION_GATE_APPROVAL_THREAD_ID,
        "text_sha256": SELECTION_GATE_APPROVAL_TEXT_SHA256,
    },
    "authorization": {
        "phase_b_authorized": False,
        "phase_b_source_selection_authorized": True,
        "source_selection_authorized": False,
        "launch_authorized": False,
        "provider_calls_authorized": False,
        "model_calls_authorized": False,
        "prime_gpu_scientific_launch_authorized": False,
        "science_authorized": False,
    },
}
# This literal is the reviewed digest of the canonical object above.  It is
# intentionally not derived from a future C3-v2 artifact.
PHASE_B_SOURCE_SELECTION_CONTRACT_V4_SHA256 = (
    "7cdf21a40c7cc5aa442d92983cb9b5d5dc6e30ea5740efe1f9284d9594662766"
)


__all__ = [
    "APPROVED_BEHAVIOR_HASHES",
    "APPROVED_COLLISION_DISPOSITIONS",
    "APPROVED_DECODER_RULES",
    "APPROVED_DERIVATION_VECTOR",
    "BEHAVIOR_BINDING_FILES",
    "FOUNDATION_STATUS_ENVELOPE",
    "PHASE_A_AUDIT_RELATIVE",
    "PHASE_A_AUDIT_SHA256",
    "PHASE_A_STATUS_SIGNATURE",
    "PHASE_A_WITNESS_SHA256",
    "PHASE_B_BINDING_DOMAIN",
    "PHASE_B_BINDING_RELATIVE",
    "PHASE_B_RESUME_CONTRACT",
    "PHASE_B_RESUME_CONTRACT_V2",
    "PHASE_B_RESUME_CONTRACT_V2_SHA256",
    "PHASE_B_RESUME_CONTRACT_V3",
    "PHASE_B_SOURCE_SELECTION_CONTRACT_V4",
    "PHASE_B_SOURCE_SELECTION_CONTRACT_V4_SHA256",
    "PHASE_C3_AUTHORIZATION_RELATIVE",
    "PHASE_C3_V2_AUTHORIZATION_DOMAIN",
    "PHASE_C3_V2_AUTHORIZATION_RELATIVE",
    "RECOVERED_REFERENCE_CANONICAL_ROW_SHA256",
    "RECOVERED_REFERENCE_CARDINALITY",
    "RECOVERED_REFERENCE_EXAMPLE_ID",
    "RECOVERED_REFERENCE_EXPECTED_DIGEST_COUNT",
    "RECOVERED_REFERENCE_PAPER_ID",
    "RECOVERED_REFERENCE_QUESTION_INDEX",
    "RECOVERED_REFERENCE_RENDERED_PAPER_SHA256",
    "RECOVERED_REFERENCE_SHA256",
    "RECOVERED_REFERENCE_SOURCE_ORDINAL",
    "REPAIR_R_COMMIT",
    "SELECTION_CLAIM_RELATIVE",
    "SELECTION_GATE_APPROVAL_TEXT",
    "SELECTION_GATE_APPROVAL_TEXT_SHA256",
    "SELECTION_GATE_APPROVAL_THREAD_ID",
    "SELECTION_RECEIPT_RELATIVE",
    "SUCCESSOR_ADDRESS_AUDIT_V1_GIT_BLOB_SHA1",
    "SUCCESSOR_ADDRESS_AUDIT_V1_RELATIVE",
    "SUCCESSOR_ADDRESS_AUDIT_V1_SHA256",
    "SUCCESSOR_EXTENSION_GIT_BLOB_SHA1",
    "SUCCESSOR_EXTENSION_INTRODUCED_COMMIT",
    "SUCCESSOR_EXTENSION_MANIFEST_GIT_BLOB_SHA1",
    "SUCCESSOR_EXTENSION_MANIFEST_RELATIVE",
    "SUCCESSOR_EXTENSION_MANIFEST_SHA256",
    "SUCCESSOR_EXTENSION_RELATIVE",
    "SUCCESSOR_EXTENSION_SHA256",
    "SUCCESSOR_MANIFEST_GIT_BLOB_SHA1",
    "SUCCESSOR_MANIFEST_RELATIVE",
    "SUCCESSOR_MANIFEST_SHA256",
]
