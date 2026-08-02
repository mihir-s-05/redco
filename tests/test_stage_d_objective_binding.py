from __future__ import annotations

import json

import pytest

from redco.analysis.stage_d_objective_binding import (
    ObjectiveAuthorization,
    ObjectiveBinding,
    fixture_objective_binding,
)
from redco.contracts import canonical_json


def test_fixture_bindings_are_arm_specific_and_independently_authorized() -> None:
    bindings = tuple(
        fixture_objective_binding(arm)
        for arm in ("stock", "branch-global", "local")
    )
    assert len({binding.objective_sha256 for binding in bindings}) == 3
    authorization = ObjectiveAuthorization(
        "fixture-only",
        tuple(sorted((binding.arm, binding.objective_sha256) for binding in bindings)),
    )
    assert ObjectiveAuthorization.from_bytes(authorization.to_bytes()) == authorization
    authorization.authorize(bindings)
    with pytest.raises(ValueError, match="independently authorized"):
        authorization.authorize((*bindings[:2], fixture_objective_binding("stock")))


def test_binding_rejects_arm_loss_cli_and_categorical_drift() -> None:
    binding = fixture_objective_binding("local")
    payload = binding.to_payload()

    wrong_loss = json.loads(canonical_json(payload))
    wrong_loss["loss_config"]["import_path"] = "evil.loss"
    with pytest.raises(ValueError, match="clean decision loss"):
        ObjectiveBinding.from_bytes(canonical_json(wrong_loss))

    override = json.loads(canonical_json(payload))
    override["effective_argv"].extend(["--loss.kwargs.kl_tau", "1.0"])
    with pytest.raises(ValueError, match="one TOML"):
        ObjectiveBinding.from_bytes(canonical_json(override))

    invalid_fused = json.loads(canonical_json(payload))
    invalid_fused["fused_lm_head_token_chunk_size"] = 0
    with pytest.raises(ValueError, match="fused LM-head"):
        ObjectiveBinding.from_bytes(canonical_json(invalid_fused))

    constrained = json.loads(canonical_json(payload))
    constrained["exact_categorical"] = {"token_groups": [[1, 2, 3, 4]]}
    with pytest.raises(ValueError, match="forbids token-group"):
        ObjectiveBinding.from_bytes(canonical_json(constrained))

    with pytest.raises(TypeError):
        ObjectiveBinding()  # type: ignore[call-arg]


def test_authorization_rejects_fixture_live_confusion() -> None:
    binding = fixture_objective_binding("stock")
    authorization = ObjectiveAuthorization(
        "live",
        (
            ("branch-global", "1" * 64),
            ("local", "2" * 64),
            ("stock", binding.objective_sha256),
        ),
    )
    with pytest.raises(ValueError, match="evidence classes"):
        authorization.authorize(
            (
                fixture_objective_binding("branch-global"),
                fixture_objective_binding("local"),
                binding,
            )
        )
