"""Non-circular integrity-witness checking for the Phase-A registry.

The canonical anchor is a reviewed input, not a value generated from the
registry.  It pins the registry's raw bytes and an independently materialized
policy projection.  The verifier's expected anchor digest is the second
binding.  Foundation F does not externally authorize this witness; a later
reviewed Git commit or canonical approval record is the trust root required
before freeze or launch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes
from redco.analysis.stage_d_v13_draft_inputs import sha256_file

APPROVAL_ANCHOR_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-source-phase-a-approval-anchor-v1.json"
)
BINDINGS_RELATIVE = "src/redco/analysis/stage_d_v13_source_phase_a_bindings.py"
TRUST_MODULE_RELATIVE = "src/redco/analysis/stage_d_v13_source_phase_a_trust.py"

# This literal is deliberately outside the registry and is updated only after
# the canonical anchor bytes are independently reviewed.  It is not computed
# from the module at runtime, so registry edits cannot bless themselves.
APPROVED_ANCHOR_SHA256 = "df5eb58e2f5e0493ec6b39864e9087bebcb2335346a2dc312a7a8e0789f7b21d"


def _policy_projection() -> dict[str, Any]:
    from redco.analysis.stage_d_v13_source_phase_a_bindings import (
        APPROVED_BEHAVIOR_HASHES,
        APPROVED_COLLISION_DISPOSITIONS,
        APPROVED_DECODER_RULES,
        APPROVED_DERIVATION_VECTOR,
        BEHAVIOR_BINDING_FILES,
        FOUNDATION_STATUS_ENVELOPE,
        PHASE_A_STATUS_SIGNATURE,
        PHASE_B_RESUME_CONTRACT,
    )

    return {
        "behavior_file_order": list(BEHAVIOR_BINDING_FILES),
        "behavior_file_map": dict(APPROVED_BEHAVIOR_HASHES),
        "derivation_vector": APPROVED_DERIVATION_VECTOR,
        "status_signature": PHASE_A_STATUS_SIGNATURE,
        "status_envelope": dict(FOUNDATION_STATUS_ENVELOPE),
        "decoder_rules": APPROVED_DECODER_RULES,
        "selector_collision_dispositions": APPROVED_COLLISION_DISPOSITIONS,
        "phase_b_resume_contract": PHASE_B_RESUME_CONTRACT,
    }


def authenticate_external_anchor(root: Path) -> dict[str, Any]:
    """Authenticate the anchor and exact registry before registry use."""

    anchor_path = root / APPROVAL_ANCHOR_RELATIVE
    if not anchor_path.is_file():
        raise FileNotFoundError(f"Phase-A approval anchor is missing: {APPROVAL_ANCHOR_RELATIVE}")
    anchor_bytes = anchor_path.read_bytes()
    anchor_sha256 = sha256_bytes(anchor_bytes)
    if anchor_sha256 != APPROVED_ANCHOR_SHA256:
        raise ValueError("Phase-A approval anchor hash differs from the reviewed binding")
    parsed = json.loads(anchor_bytes)
    if not isinstance(parsed, dict) or anchor_bytes != canonical_json_bytes(parsed):
        raise ValueError("Phase-A approval anchor is not canonical JSON")
    if (
        parsed.get("schema_version") != 1
        or parsed.get("domain") != "redco-stage-d1-support-v13-source-phase-a-approval-v1"
        or parsed.get("draft_unfrozen") is not True
        or parsed.get("launch_authorized") is not False
        or parsed.get("provider_calls_authorized") is not False
        or parsed.get("phase_b_authorized") is not False
        or parsed.get("foundation_only") is not True
        or parsed.get("non_authorizing") is not True
        or parsed.get("candidate") != {
            "source_ordinal": None,
            "paper_id": None,
            "example_id": None,
            "row": None,
            "seed": None,
            "address": None,
        }
        or parsed.get("seed") is not None
        or parsed.get("address") is not None
    ):
        raise ValueError("Phase-A approval anchor envelope differs")
    registry = cast(dict[str, Any], parsed.get("registry"))
    if registry.get("path") != BINDINGS_RELATIVE:
        raise ValueError("Phase-A approval anchor registry path differs")
    registry_path = root / BINDINGS_RELATIVE
    if not registry_path.is_file():
        raise FileNotFoundError(f"Phase-A bindings registry is missing: {BINDINGS_RELATIVE}")
    registry_sha256 = sha256_file(registry_path)
    if registry_sha256 != registry.get("sha256"):
        raise ValueError("Phase-A bindings registry hash differs from the anchor")
    expected_policy = _policy_projection()
    policy = parsed.get("policy")
    if not isinstance(policy, dict) or policy != expected_policy:
        raise ValueError("Phase-A approval policy differs from the external anchor")
    if policy.get("status_envelope") != expected_policy["status_envelope"]:
        raise ValueError("Phase-A non-authorizing status envelope differs")
    trust_root = parsed.get("trust_root")
    if (
        not isinstance(trust_root, dict)
        or trust_root.get("kind") != "future_reviewed_git_commit"
        or trust_root.get("status") != "uncommitted_integrity_witness_only"
        or trust_root.get("externally_authorized") is not False
    ):
        raise ValueError("Phase-A approval anchor is not an F-only integrity witness")
    if trust_root.get("review_required_before_freeze") is not True:
        raise ValueError("Phase-A approval anchor is not review-gated")
    return {
        "anchor_path": APPROVAL_ANCHOR_RELATIVE,
        "anchor_sha256": anchor_sha256,
        "registry_path": BINDINGS_RELATIVE,
        "registry_sha256": registry_sha256,
        "policy_sha256": sha256_bytes(canonical_json_bytes(expected_policy)),
        "trust_root": trust_root,
        "integrity_witness_only": True,
        "externally_authorized": False,
    }


__all__ = [
    "APPROVAL_ANCHOR_RELATIVE",
    "APPROVED_ANCHOR_SHA256",
    "BINDINGS_RELATIVE",
    "TRUST_MODULE_RELATIVE",
    "authenticate_external_anchor",
]
