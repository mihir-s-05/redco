from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from redco.integrations.signed_subprocess import sign_payload


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_6_runner_never_reruns_support_or_sft() -> None:
    root = Path(__file__).parents[1]
    runner = (
        root / "scripts" / "run_stage_d0_scaffold_support_v4_6.sh"
    ).read_text(encoding="utf-8")
    assert "redco-stage-d0-fewshot-support-v4" not in runner
    assert "stage-d0-scaffold-sft-v4.toml" not in runner
    assert "sft @" not in runner
    assert "selected-adapter.tar.gz" in runner
    assert runner.index("run_stage_d_retention_canonical_v4_6.py") < runner.index(
        '"$run_root/selected-fixture"'
    )
    assert runner.index('"$run_root/selected-fixture"') < runner.index(
        '"$run_root/power-audit" 64 1'
    )


def test_canonical_scorer_applies_adapter_to_action_and_root() -> None:
    root = Path(__file__).parents[1]
    scorer = (
        root
        / "scripts"
        / "score_stage_d_retained_adapter_canonical_v4_6.py"
    ).read_text(encoding="utf-8")
    context = scorer[
        scorer.index("with adapter_hooks(model, args.adapter):") :
        scorer.index("action = _action_payload(")
    ]
    assert "_action_model(" in context
    assert "_root_payload(" in context


def _root_payload(
    *,
    model: str,
    cases: str = (
        "0274d78b630201b5363fcc1a6348eef3aa1a52c33153e0a25627d4cfb2dfcff9"
    ),
) -> dict[str, object]:
    routes = ("alpha", "beta", "gamma", "delta")
    probabilities = {route: 0.2 for route in routes}
    return sign_payload(
        {
            "source": {"model": model, "cases_sha256": cases},
            "temperature_2": {
                "route_sequence_logprobabilities": {
                    route: -1.0 for route in routes
                },
                "route_sequence_probabilities": probabilities,
                "valid_route_sequence_mass": 0.8,
                "token_details": {
                    route: [
                        {
                            "token_id": index,
                            "temperature_2_logprob": -0.1,
                        }
                    ]
                    for index, route in enumerate(routes)
                },
            },
        }
    )


def _write_payload(tmp_path: Path, name: str, payload: object) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _health_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    canonical = sign_payload(
        {
            "models": [
                {
                    "name": "retained",
                    "temperatures": {
                        "1.0": [
                            {
                                "case_id": "case-1",
                                "greedy_token_id": 7,
                                "action_probabilities": {"5": 0.9},
                            }
                        ]
                    },
                }
            ]
        }
    )
    runtime = sign_payload(
        {
            "models": [
                {
                    "name": "selected",
                    "temperatures": {
                        "1.0": [
                            {
                                "case_id": "case-1",
                                "greedy_token_id": 7,
                                "action_probabilities": {"5": 0.1},
                            }
                        ]
                    },
                }
            ]
        }
    )
    return (
        _write_payload(tmp_path, "canonical-action.json", canonical),
        _write_payload(tmp_path, "runtime-action.json", runtime),
        _write_payload(
            tmp_path,
            "canonical-root.json",
            _root_payload(model="/workspace/base"),
        ),
        _write_payload(
            tmp_path,
            "runtime-root.json",
            _root_payload(model="/tmp/selected"),
        ),
    )


def test_vllm_health_is_finite_and_greedy_not_probability_equality(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    module = _load(
        "stage_d_v4_6_vllm_health",
        root / "scripts" / "audit_stage_d_vllm_health_v4_6.py",
    )
    report = module.audit(*_health_inputs(tmp_path), "/tmp/selected")
    assert report["passes"]
    assert report["checks"]["canonical_runtime_greedy_tokens_agree"]


@pytest.mark.parametrize(
    "mutation",
    ("missing_scores", "wrong_cases", "nan_logprob", "wrong_routes"),
)
def test_vllm_health_rejects_malformed_runtime_root(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = Path(__file__).parents[1]
    module = _load(
        f"stage_d_v4_6_vllm_health_{mutation}",
        root / "scripts" / "audit_stage_d_vllm_health_v4_6.py",
    )
    canonical_action, runtime_action, canonical_root, runtime_root = (
        _health_inputs(tmp_path)
    )
    payload = json.loads(runtime_root.read_text(encoding="utf-8"))
    payload.pop("signed_payload_sha256")
    if mutation == "missing_scores":
        payload.pop("temperature_2")
    elif mutation == "wrong_cases":
        payload["source"]["cases_sha256"] = "wrong"
    elif mutation == "nan_logprob":
        payload["temperature_2"]["route_sequence_logprobabilities"][
            "alpha"
        ] = float("nan")
    else:
        payload["temperature_2"]["route_sequence_probabilities"].pop("alpha")
    runtime_root.write_text(
        json.dumps(sign_payload(payload), sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        module.audit(
            canonical_action,
            runtime_action,
            canonical_root,
            runtime_root,
            "/tmp/selected",
        )
