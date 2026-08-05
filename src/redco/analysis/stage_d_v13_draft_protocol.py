"""Draft-only protocol envelopes and zero-call deployment checklist."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from redco.analysis.stage_d_v13_draft import (
    AttemptState,
    DraftEvent,
    state_machine_contract,
    transition,
)


def state_ledger() -> dict[str, Any]:
    def run(events: Iterable[DraftEvent]) -> dict[str, object]:
        state = AttemptState()
        for event in events:
            state = transition(state, event)
        return state.to_payload()

    pre_post = [
        DraftEvent.PROVISION_ATTEMPT,
        DraftEvent.PROVISION_FAILED,
        DraftEvent.REDEPLOY,
        DraftEvent.PROVISION_READY,
    ]
    post_dispatch = [
        DraftEvent.PROVISION_ATTEMPT,
        DraftEvent.PROVISION_READY,
        DraftEvent.PROVIDER_POST,
        DraftEvent.ABORT,
        DraftEvent.RECOVER_ARTIFACTS,
    ]
    return {
        "schema_version": 1,
        "draft_unfrozen": True,
        "launch_authorized": False,
        "domain": "redco-stage-d1-support-v13-attempt-abort-ledger-v2",
        "status": "draft_unlaunched_no_actual_attempts",
        "actual_events": [],
        "campaign_attempts_consumed": 0,
        "state_machine": state_machine_contract(),
        "examples": [
            {
                "name": "one_pre_post_redeployment",
                "events": [event.value for event in pre_post],
                "expected": run(pre_post),
                "admissible": True,
            },
            {
                "name": "lost_response_consumes_attempt_and_retires_unit",
                "events": [event.value for event in post_dispatch],
                "expected": run(post_dispatch),
                "admissible": True,
                "redeployment_after_abort": "forbidden",
            },
        ],
        "retirement_rule": (
            "Every unit with a dispatched provider request is retired, including a lost response."
        ),
        "successor_data_rule": (
            "Committed support data cannot be carried into a successor; a future successor "
            "requires a new statistical design."
        ),
    }


def deployment_checklist(
    *,
    source_eval: dict[str, Any],
    dependency_stack_sha256: str,
    post_repair_regression_sha256: str,
) -> dict[str, Any]:
    return {
        "status": "required_before_any_provider_post",
        "provider_posts_in_this_draft": 0,
        "prefer_zero_call": True,
        "checks": [
            {
                "name": "model_cache_reference_and_dependency_authentication",
                "status": "inherited_hash_bindings_only",
                "hashes": {
                    "base_model_manifest": source_eval["env"]["base_model_manifest_sha256"],
                    "tokenizer_manifest": source_eval["env"]["tokenizer_manifest_sha256"],
                    "renderer_manifest": source_eval["env"]["renderer_manifest_sha256"],
                    "dependency_stack": dependency_stack_sha256,
                },
            },
            {
                "name": "tokenizer_renderer_golden_checks",
                "status": "required_zero_call_preflight",
                "no_model_call": True,
            },
            {
                "name": "offline_rlm_restoration",
                "status": "inherited_v12_preflight_pass",
                "evidence": "reports/stage-d1-support-v12-terminal.json",
            },
            {
                "name": "pristine_ledger_and_output_root",
                "status": "required_before_launch_not_run_in_draft",
                "no_model_call": True,
            },
            {
                "name": "vllm_health_and_endpoint_readiness",
                "status": "inherited_v12_health_pass_not_a_model_call",
                "evidence": "reports/stage-d1-support-v12-terminal.json",
            },
            {
                "name": "scripted_provider_to_finalizer_regression",
                "status": "authenticated_existing_cpu_regression",
                "evidence": "reports/stage-d1-source-comparison-post-repair-v1.json",
                "evidence_sha256": post_repair_regression_sha256,
            },
        ],
        "real_call_smoke": {
            "required": False,
            "if_authorized": (
                "separate sacrificial preregistration and identity; permanently retire it"
            ),
        },
    }


__all__ = ["deployment_checklist", "state_ledger"]
