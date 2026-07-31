"""Fail-closed validation for independent pre-GPU runtime review artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import verify_signed_payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(
    *,
    policy_path: Path,
    review_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    verify_signed_payload(review)
    required = policy.get("required_gates")
    if not isinstance(required, list) or not all(
        isinstance(gate, str) and gate for gate in required
    ):
        raise ValueError("policy required_gates must be nonempty strings")
    gates = review.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("review gates must be an object")
    missing = [gate for gate in required if gate not in gates]
    failed = [gate for gate in required if gates.get(gate) is not True]
    extra = sorted(set(gates) - set(required))
    assumptions = review.get("unresolved_assumptions")
    if not isinstance(assumptions, list):
        raise ValueError("unresolved_assumptions must be a list")
    checks = {
        "policy_hash_exact": review.get("policy_sha256") == _sha256(policy_path),
        "reviewed_commit_exact": review.get("reviewed_commit") == expected_commit,
        "reviewer_model_exact": review.get("reviewer", {}).get("model_family")
        == policy["reviewer"]["model_family"],
        "reviewer_effort_sufficient": review.get("reviewer", {}).get(
            "reasoning_effort"
        )
        in {"xhigh", "max", "ultra"},
        "all_required_gates_present": not missing,
        "all_required_gates_pass": not failed,
        "no_extra_gates": not extra,
        "no_unresolved_assumptions": not assumptions,
        "decision_go": review.get("decision") == "GO",
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "missing_gates": missing,
        "failed_gates": failed,
        "extra_gates": extra,
        "unresolved_assumptions": assumptions,
        "review_signature": review["signed_payload_sha256"],
    }

