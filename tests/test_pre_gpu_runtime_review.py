from __future__ import annotations

import hashlib
import json
from pathlib import Path

from redco.analysis.pre_gpu_runtime_review import evaluate
from redco.integrations.signed_subprocess import sign_payload

GATES = ["cli", "golden_trace", "linux"]


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    policy = tmp_path / "policy.json"
    _write(
        policy,
        {
            "required_gates": GATES,
            "reviewer": {"model_family": "gpt-5.6-sol"},
        },
    )
    policy_hash = hashlib.sha256(policy.read_bytes()).hexdigest()
    review = tmp_path / "review.json"
    _write(
        review,
        sign_payload(
            {
                "policy_sha256": policy_hash,
                "reviewed_commit": "abc123",
                "reviewer": {
                    "model_family": "gpt-5.6-sol",
                    "reasoning_effort": "xhigh",
                },
                "gates": {gate: True for gate in GATES},
                "unresolved_assumptions": [],
                "decision": "GO",
            }
        ),
    )
    return policy, review


def test_exact_signed_go_passes(tmp_path: Path) -> None:
    policy, review = _fixture(tmp_path)
    assert evaluate(
        policy_path=policy,
        review_path=review,
        expected_commit="abc123",
    )["passes"]


def test_any_failed_gate_or_commit_mismatch_fails(tmp_path: Path) -> None:
    policy, review = _fixture(tmp_path)
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload.pop("signed_payload_sha256")
    payload["gates"]["golden_trace"] = False
    _write(review, sign_payload(payload))
    result = evaluate(
        policy_path=policy,
        review_path=review,
        expected_commit="different",
    )
    assert not result["passes"]
    assert result["failed_gates"] == ["golden_trace"]
    assert not result["checks"]["reviewed_commit_exact"]
