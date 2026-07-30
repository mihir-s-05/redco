"""Validate and combine canonical Stage-C6 score payloads."""

from __future__ import annotations

import copy
from typing import Any

from redco.integrations.signed_subprocess import sign_payload


def verify_model_identity(
    reference: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    reference_files = {
        name: value["sha256"] for name, value in reference["files"].items()
    }
    current_files = {
        name: value["sha256"] for name, value in current["files"].items()
    }
    file_checks = {
        name: (
            name in current_files
            and current_files[name] == expected
        )
        for name, expected in reference_files.items()
    }
    checks = {
        "file_sets_identical": set(reference_files) == set(current_files),
        "every_file_sha256_identical": all(file_checks.values()),
        "adapter_sha256_identical": (
            reference["adapter_model_sha256"]
            == current["adapter_model_sha256"]
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-c6-merged-model-identity",
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "file_checks": file_checks,
            "reference_adapter_sha256": reference["adapter_model_sha256"],
            "current_adapter_sha256": current["adapter_model_sha256"],
        }
    )


def verify_runtime_support(candidate: dict[str, Any]) -> dict[str, Any]:
    checks = dict(candidate["candidate"]["checks"])
    factorization = {
        name: checks.pop(name)
        for name in (
            "route_digit_joint_tv_at_most_0_05",
            "route_digit_mutual_information_at_most_0_01_nats",
        )
    }
    required = {
        **checks,
        "campaign_power_passed": (
            candidate["candidate"]["campaign_power"]["status"] == "passed"
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-c6-runtime-support",
            "status": "passed" if all(required.values()) else "failed",
            "required_checks": required,
            "factorization_checks_reported_but_decided_by_canonical_scorer": (
                factorization
            ),
            "candidate_signed_payload_sha256": candidate[
                "signed_payload_sha256"
            ],
        }
    )


def verify_replicates(
    action_payloads: list[dict[str, Any]],
    root_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(action_payloads) != 3 or len(root_payloads) != 3:
        raise ValueError("exactly three action and root replicates are required")
    action_signatures = [
        str(payload["signed_payload_sha256"]) for payload in action_payloads
    ]
    root_signatures = [
        str(payload["signed_payload_sha256"]) for payload in root_payloads
    ]
    action_identical = len(set(action_signatures)) == 1
    root_identical = len(set(root_signatures)) == 1
    settings = [
        payload.get("canonical_settings")
        for payload in [*action_payloads, *root_payloads]
    ]
    checks = {
        "three_action_payloads_byte_identical": action_identical,
        "three_root_payloads_byte_identical": root_identical,
        "canonical_settings_identical": all(
            setting == settings[0] for setting in settings
        ),
        "backend_is_transformers_eager_cuda": all(
            payload.get("backend") == "transformers-eager-cuda"
            for payload in [*action_payloads, *root_payloads]
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-c6-canonical-score-reproducibility",
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "action_signatures": action_signatures,
            "root_signatures": root_signatures,
        }
    )


def combine_action_scores(
    payloads: list[dict[str, Any]],
    *,
    warmstart_name: str = "warmstart",
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("at least one canonical action payload is required")
    models = []
    names = set()
    for payload in payloads:
        if payload.get("backend") != "transformers-eager-cuda":
            raise ValueError("all action scores must use the canonical backend")
        for model in payload["models"]:
            name = str(model["name"])
            if name in names:
                raise ValueError(f"duplicate canonical model name: {name}")
            names.add(name)
            models.append(copy.deepcopy(model))
    if warmstart_name not in names:
        raise ValueError("combined scores are missing the warmstart")
    first = payloads[0]
    if any(
        payload["source"]["cases_sha256"] != first["source"]["cases_sha256"]
        or payload["canonical_settings"] != first["canonical_settings"]
        for payload in payloads[1:]
    ):
        raise ValueError("canonical action payload contracts differ")
    return sign_payload(
        {
            "schema_version": 1,
            "backend": "transformers-eager-cuda",
            "canonical_settings": first["canonical_settings"],
            "temperature_semantics": first["temperature_semantics"],
            "source": {
                "model": first["source"]["model"],
                "cases_sha256": first["source"]["cases_sha256"],
                "input_signatures": [
                    payload["signed_payload_sha256"] for payload in payloads
                ],
            },
            "models": models,
        }
    )
