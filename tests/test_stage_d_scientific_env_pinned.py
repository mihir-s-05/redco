from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers.v1")

ENV_ROOT = Path(__file__).parents[1] / "environments" / "redco_evidence_selection_v2"
sys.path.insert(0, str(ENV_ROOT))

from redco_evidence_selection_v2.scientific_env import (  # noqa: E402
    StageDScientificReplayEnv,
    _classify_execution_outcome,
)
from test_stage_d_source_producer import _prepared_action  # noqa: E402

from redco.analysis.stage_d_scientific_branch_group import OutcomeKind  # noqa: E402


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_scientific_env_rejects_source_policy_or_manifest_drift(tmp_path: Path) -> None:
    action = _prepared_action(72)
    values = {
        "base": b"base",
        "adapter": b"adapter",
        "tokenizer": b"tokenizer",
        "renderer": b"renderer",
        "sampler": action.key.sampler_conformance_manifest,
    }
    paths = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        path.write_bytes(value)
        paths[name] = path
    config = SimpleNamespace(
        checkpoint_id=action.key.checkpoint_id,
        base_model_manifest_path=paths["base"],
        base_model_manifest_sha256=_sha256(values["base"]),
        adapter_manifest_path=paths["adapter"],
        adapter_manifest_sha256=_sha256(values["adapter"]),
        tokenizer_manifest_path=paths["tokenizer"],
        tokenizer_manifest_sha256=_sha256(values["tokenizer"]),
        renderer_manifest_path=paths["renderer"],
        renderer_manifest_sha256=_sha256(values["renderer"]),
        sampler_conformance_manifest_path=paths["sampler"],
        sampler_conformance_manifest_sha256=_sha256(values["sampler"]),
    )
    binding = SimpleNamespace(
        task=SimpleNamespace(
            data=SimpleNamespace(policy_checkpoint_id=action.key.checkpoint_id)
        ),
        source_actions={"target": action},
    )

    StageDScientificReplayEnv._validate_policy_identity(config, binding)

    paths["renderer"].write_bytes(b"drift")
    with pytest.raises(ValueError, match="renderer manifest bytes changed"):
        StageDScientificReplayEnv._validate_policy_identity(config, binding)
    paths["renderer"].write_bytes(values["renderer"])
    config.checkpoint_id = "different-checkpoint"
    with pytest.raises(ValueError, match="task checkpoint"):
        StageDScientificReplayEnv._validate_policy_identity(config, binding)


@pytest.mark.parametrize(
    ("stop", "finish_reason", "downstream", "expected"),
    [
        ("harness_timeout", "stop", True, OutcomeKind.TIMEOUT),
        ("max_turns", "stop", True, OutcomeKind.RESOURCE_LIMIT),
        ("context_length", "stop", True, OutcomeKind.RESOURCE_LIMIT),
        ("agent_completed", "length", True, OutcomeKind.RESOURCE_LIMIT),
        ("agent_completed", "stop", True, OutcomeKind.SUCCESS),
        (
            "agent_completed",
            "stop",
            False,
            OutcomeKind.TERMINAL_WITHOUT_DOWNSTREAM,
        ),
    ],
)
def test_scientific_outcome_classifier_accepts_only_exact_clean_stops(
    stop: str,
    finish_reason: str,
    downstream: bool,
    expected: OutcomeKind,
) -> None:
    trace = SimpleNamespace(
        ok=True,
        errors=[],
        stop_condition=stop,
        calls=[SimpleNamespace(error=None, finish_reason=finish_reason)],
    )
    assert (
        _classify_execution_outcome(
            episode=SimpleNamespace(ok=True, errors=[]),
            trace=trace,
            action=SimpleNamespace(parse_status="valid"),
            controller=SimpleNamespace(
                target_injection_delivered=True,
                logical_downstream_observed=downstream,
            ),
        )
        is expected
    )


def test_scientific_outcome_classifier_requires_delivery_and_retains_errors() -> None:
    trace = SimpleNamespace(
        ok=False,
        errors=[SimpleNamespace(type="HarnessError")],
        stop_condition="error",
        calls=[],
    )
    with pytest.raises(RuntimeError, match="not durably delivered"):
        _classify_execution_outcome(
            episode=SimpleNamespace(ok=False, errors=[]),
            trace=trace,
            action=SimpleNamespace(parse_status="valid"),
            controller=SimpleNamespace(
                target_injection_delivered=False,
                logical_downstream_observed=False,
            ),
        )
    assert (
        _classify_execution_outcome(
            episode=SimpleNamespace(ok=False, errors=[]),
            trace=trace,
            action=SimpleNamespace(parse_status="valid"),
            controller=SimpleNamespace(
                target_injection_delivered=True,
                logical_downstream_observed=False,
            ),
        )
        is OutcomeKind.RUNTIME_EXCEPTION
    )


def test_scientific_outcome_classifier_records_malformed_only_after_delivery() -> None:
    outcome = _classify_execution_outcome(
        episode=SimpleNamespace(ok=True, errors=[]),
        trace=SimpleNamespace(
            ok=True,
            errors=[],
            stop_condition="agent_completed",
            calls=[],
        ),
        action=SimpleNamespace(parse_status="malformed"),
        controller=SimpleNamespace(
            target_injection_delivered=True,
            logical_downstream_observed=False,
        ),
    )
    assert outcome is OutcomeKind.MALFORMED_ACTION
