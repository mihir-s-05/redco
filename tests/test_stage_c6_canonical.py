from copy import deepcopy

import pytest

from redco.analysis.stage_c6_canonical import (
    combine_action_scores,
    verify_model_identity,
    verify_replicates,
    verify_runtime_support,
)


def _payload(name: str = "warmstart") -> dict:
    return {
        "backend": "transformers-eager-cuda",
        "canonical_settings": {"batch_size": 1},
        "temperature_semantics": "canonical",
        "source": {
            "model": "model",
            "cases_sha256": "cases",
            "adapters": [],
        },
        "models": [{"name": name, "temperatures": {"2.0": []}}],
        "signed_payload_sha256": f"signature-{name}",
    }


def test_replicates_require_three_identical_signatures() -> None:
    action = _payload()
    root = {**_payload(), "analysis": "root"}
    result = verify_replicates(
        [deepcopy(action) for _ in range(3)],
        [deepcopy(root) for _ in range(3)],
    )
    assert result["status"] == "passed"
    changed = deepcopy(action)
    changed["signed_payload_sha256"] = "different"
    result = verify_replicates(
        [action, deepcopy(action), changed],
        [deepcopy(root) for _ in range(3)],
    )
    assert result["status"] == "failed"


def test_combiner_requires_unique_models_and_warmstart() -> None:
    warmstart = _payload()
    arm = _payload("arm")
    combined = combine_action_scores([warmstart, arm])
    assert [model["name"] for model in combined["models"]] == [
        "warmstart",
        "arm",
    ]
    with pytest.raises(ValueError, match="duplicate"):
        combine_action_scores([warmstart, deepcopy(warmstart)])
    with pytest.raises(ValueError, match="warmstart"):
        combine_action_scores([arm])


def test_model_identity_requires_every_file_hash() -> None:
    reference = {
        "adapter_model_sha256": "adapter",
        "files": {"model": {"sha256": "weights"}},
    }
    assert verify_model_identity(reference, deepcopy(reference))["status"] == "passed"
    changed = deepcopy(reference)
    changed["files"]["model"]["sha256"] = "changed"
    assert verify_model_identity(reference, changed)["status"] == "failed"


def test_runtime_support_delegates_factorization_to_canonical_scorer() -> None:
    checks = {
        "route_digit_joint_tv_at_most_0_05": False,
        "route_digit_mutual_information_at_most_0_01_nats": True,
        "exploration": True,
    }
    candidate = {
        "candidate": {
            "checks": checks,
            "campaign_power": {"status": "passed"},
        },
        "signed_payload_sha256": "candidate",
    }
    result = verify_runtime_support(candidate)
    assert result["status"] == "passed"
    assert result[
        "factorization_checks_reported_but_decided_by_canonical_scorer"
    ]["route_digit_joint_tv_at_most_0_05"] is False
