"""Pure CPU-only policy helpers for the unfrozen Stage-D v13 draft.

This module deliberately has no provider, filesystem, subprocess, or campaign
side effects.  The draft builder owns read-only input authentication and writes
only new ``v13-draft`` artifacts.  Keeping the state machine here makes the
redeployment rule executable without adding a second campaign controller.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Final


class DraftPolicyError(ValueError):
    """Raised when a draft-only policy transition is not admissible."""


class DraftState(StrEnum):
    """States relevant to the pre-provider redeployment boundary."""

    UNLAUNCHED = "draft_unlaunched"
    PROVISIONING = "provisioning"
    READY_PRE_POST = "ready_before_provider_post"
    REDEPLOYABLE_PRE_POST = "redeployable_before_provider_post"
    PROVIDER_DISPATCHED = "provider_dispatched"
    RESPONSE_WITNESSED = "response_witnessed"
    ARTIFACT_OBSERVED = "source_or_score_observed"
    TERMINAL_INCOMPLETE = "terminal_incomplete"


class DraftEvent(StrEnum):
    """Events accepted by :func:`transition` in deterministic tests."""

    PROVISION_ATTEMPT = "provision_attempt"
    PROVISION_READY = "provision_ready"
    PROVISION_FAILED = "provision_failed"
    REDEPLOY = "redeploy"
    PROVIDER_POST = "provider_post"
    RESPONSE_BYTES = "response_bytes"
    SOURCE_OUTPUT = "source_output"
    SCORE_OUTPUT = "score_output"
    ABORT = "abort"
    RECOVER_ARTIFACTS = "recover_artifacts"


@dataclass(frozen=True, slots=True)
class AttemptState:
    """Minimal state needed to enforce the bounded redeployment rule."""

    state: DraftState = DraftState.UNLAUNCHED
    provisioning_attempts: int = 0
    campaign_attempt_consumed: bool = False
    provider_post_seen: bool = False
    response_witness_seen: bool = False
    source_or_score_seen: bool = False
    unit_retired: bool = False

    def to_payload(self) -> dict[str, object]:
        return {
            "campaign_attempt_consumed": self.campaign_attempt_consumed,
            "provisioning_attempts": self.provisioning_attempts,
            "provider_post_seen": self.provider_post_seen,
            "response_witness_seen": self.response_witness_seen,
            "source_or_score_seen": self.source_or_score_seen,
            "state": self.state.value,
            "unit_retired": self.unit_retired,
        }


MAX_PROVISIONING_ATTEMPTS: Final = 2
MAX_CONCURRENT_EPISODES: Final = 1


def _require_pre_post(state: AttemptState, event: str) -> None:
    if state.provider_post_seen or state.response_witness_seen or state.source_or_score_seen:
        raise DraftPolicyError(
            f"{event} is forbidden after provider dispatch, response, source, or score"
        )


def transition(state: AttemptState, event: DraftEvent) -> AttemptState:
    """Apply one fail-closed resource/attempt event.

    A provider POST is the campaign-attempt boundary.  Once it occurs the unit
    is retired, including when the response is lost.  Only an infrastructure
    failure before that boundary may lead to the single second provisioning
    attempt.
    """

    if event is DraftEvent.PROVISION_ATTEMPT:
        if state.state is not DraftState.UNLAUNCHED:
            raise DraftPolicyError(
                "the first provisioning attempt must start from draft_unlaunched"
            )
        return replace(
            state,
            state=DraftState.PROVISIONING,
            provisioning_attempts=state.provisioning_attempts + 1,
        )
    if event is DraftEvent.PROVISION_READY:
        if state.state is not DraftState.PROVISIONING:
            raise DraftPolicyError("provision_ready requires an in-flight provisioning attempt")
        return replace(state, state=DraftState.READY_PRE_POST)
    if event is DraftEvent.PROVISION_FAILED:
        if state.state is not DraftState.PROVISIONING:
            raise DraftPolicyError("provision_failed requires an in-flight provisioning attempt")
        _require_pre_post(state, "provision_failed")
        return replace(state, state=DraftState.REDEPLOYABLE_PRE_POST)
    if event is DraftEvent.REDEPLOY:
        _require_pre_post(state, "redeploy")
        if state.state is not DraftState.REDEPLOYABLE_PRE_POST:
            raise DraftPolicyError("redeploy requires a demonstrated pre-POST provisioning failure")
        if state.provisioning_attempts >= MAX_PROVISIONING_ATTEMPTS:
            raise DraftPolicyError("maximum provisioning attempts exceeded")
        return replace(
            state,
            state=DraftState.PROVISIONING,
            provisioning_attempts=state.provisioning_attempts + 1,
        )
    if event is DraftEvent.PROVIDER_POST:
        if state.state is not DraftState.READY_PRE_POST:
            raise DraftPolicyError("provider_post requires a ready pre-POST deployment")
        return replace(
            state,
            state=DraftState.PROVIDER_DISPATCHED,
            campaign_attempt_consumed=True,
            provider_post_seen=True,
            unit_retired=True,
        )
    if event is DraftEvent.RESPONSE_BYTES:
        if not state.provider_post_seen:
            raise DraftPolicyError("response bytes require a provider POST")
        return replace(
            state,
            state=DraftState.RESPONSE_WITNESSED,
            response_witness_seen=True,
            unit_retired=True,
        )
    if event in (DraftEvent.SOURCE_OUTPUT, DraftEvent.SCORE_OUTPUT):
        if not state.provider_post_seen:
            raise DraftPolicyError("source or score output requires a provider POST")
        return replace(
            state,
            state=DraftState.ARTIFACT_OBSERVED,
            source_or_score_seen=True,
            unit_retired=True,
        )
    if event is DraftEvent.ABORT:
        return replace(
            state,
            state=DraftState.TERMINAL_INCOMPLETE,
            unit_retired=state.unit_retired or state.provider_post_seen,
        )
    if event is DraftEvent.RECOVER_ARTIFACTS:
        if state.state is not DraftState.TERMINAL_INCOMPLETE:
            raise DraftPolicyError("artifact recovery is terminal-only")
        return state
    raise DraftPolicyError(f"unsupported draft event: {event}")


def reject_change_as_redeployment(change_class: str) -> None:
    """Require a new hashed amendment for non-resource changes."""

    allowed = {"capacity_resource", "ssh_endpoint"}
    if change_class not in allowed:
        raise DraftPolicyError(
            f"{change_class} is a protocol/code/dependency change; it requires a new amendment"
        )


def state_machine_contract() -> dict[str, object]:
    """Return the machine-readable contract frozen by this draft."""

    return {
        "campaign_max_concurrent": MAX_CONCURRENT_EPISODES,
        "maximum_provisioning_attempts": MAX_PROVISIONING_ATTEMPTS,
        "provider_dispatch_consumes_campaign_attempt": True,
        "provider_dispatch_retires_unit": True,
        "response_or_raw_witness_forbids_redeployment": True,
        "source_or_score_forbids_redeployment": True,
        "code_dependency_comparator_renderer_protocol_change_requires_amendment": True,
        "allowed_pre_post_redeployment_classes": [
            "capacity_resource",
            "ssh_endpoint",
        ],
        "states": [state.value for state in DraftState],
        "events": [event.value for event in DraftEvent],
    }


def canonical_json_bytes(value: Any) -> bytes:
    """Encode canonical JSON bytes with no trailing newline."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if encoded.endswith(b"\n"):
        raise AssertionError("canonical JSON unexpectedly ended in a newline")
    return encoded


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def fresh_identities(
    *,
    scientific_namespace: str,
    draft_domain: str,
    repair_commit: str,
    administrative_inputs: Mapping[str, str],
) -> dict[str, str]:
    """Derive administrative identities without outcome-bearing inputs.

    Archive, evaluator, trace, score, and observation material is deliberately
    rejected here.  The reserve candidate identity/seed/address is not derived
    by this helper; those remain unresolved until authenticated source-order
    selection succeeds.
    """

    forbidden_fragments = (
        "archive",
        "evaluator",
        "trace",
        "score",
        "observ",
        "historical",
        "terminal",
        "rollout",
    )
    if any(
        any(fragment in key.lower() for fragment in forbidden_fragments)
        for key in administrative_inputs
    ):
        raise DraftPolicyError("outcome-bearing input cannot derive administrative identities")
    material = {
        "scientific_namespace": scientific_namespace,
        "draft_domain": draft_domain,
        "repair_commit": repair_commit,
        "administrative_inputs": dict(sorted(administrative_inputs.items())),
    }
    root = sha256_json(material)
    return {
        "campaign_id": f"stage-d1-support-v13-campaign-{root[:24]}",
        "run_id": f"stage-d1-support-v13-run-{sha256_bytes((root + '|run').encode())[:24]}",
        "genesis_id": (
            f"stage-d1-support-v13-genesis-{sha256_bytes((root + '|genesis').encode())[:24]}"
        ),
        "ledger_id": (
            f"stage-d1-support-v13-ledger-{sha256_bytes((root + '|ledger').encode())[:24]}"
        ),
        "output_root_id": (
            f"stage-d1-support-v13-output-{sha256_bytes((root + '|output').encode())[:24]}"
        ),
        "campaign_seed": str(
            int.from_bytes(hashlib.sha256((root + "|seed").encode()).digest()[:8], "big")
            & ((1 << 63) - 1)
        ),
    }


def affordability_ledger(
    *,
    campaign_cap_usd: float,
    historical_wallet_usd: float,
    reserve_usd: float,
    max_provider_posts: int,
    max_completion_tokens: int,
    max_wall_hours: float,
    max_hourly_usd: float,
) -> dict[str, object]:
    """Build a conservative launch ledger without checking the live wallet."""

    wall_cap = max_wall_hours * max_hourly_usd
    return {
        "status": "fail_closed_until_launch_time_pricing_and_wallet_check",
        "wallet_checked_now": False,
        "historical_wallet_usd": historical_wallet_usd,
        "historical_reserve_usd": reserve_usd,
        "wallet_at_launch_usd": None,
        "campaign_cap_usd": campaign_cap_usd,
        "allowed_spend_rule": "min(campaign_cap_usd, wallet_at_launch_usd - reserve_usd)",
        "max_provider_posts": max_provider_posts,
        "max_completion_tokens": max_completion_tokens,
        "max_wall_hours": max_wall_hours,
        "max_hourly_usd": max_hourly_usd,
        "worst_case_wall_cost_usd": wall_cap,
        "token_price_usd_per_1k": None,
        "worst_case_token_cost_usd": None,
        "launch_gate": (
            "fail closed when wallet_at_launch_usd is unavailable, pricing is unavailable, "
            "or worst-case spend exceeds allowed spend"
        ),
    }


def nonoverlap_digest(
    values: Mapping[str, list[str]],
    candidate_values: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Return deterministic within- and cross-class collision checks.

    Candidate checks are explicitly unresolved until a canonical reserve row
    exists; they are never represented as passing booleans in that state.
    """

    checks: dict[str, bool] = {}
    digests: dict[str, str] = {}
    normalized = {name: list(items) for name, items in sorted(values.items())}
    for name, items in normalized.items():
        checks[f"{name}_unique"] = len(items) == len(set(items))
        digests[name] = sha256_json({"values": items})
    names = sorted(normalized)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            checks[f"{left_name}_disjoint_from_{right_name}"] = not (
                set(normalized[left_name]) & set(normalized[right_name])
            )
    if candidate_values is None:
        candidate_checks: dict[str, bool | None] = {
            "candidate_ids_disjoint_from_forbidden": None,
            "candidate_seeds_disjoint_from_forbidden": None,
            "candidate_references_disjoint_from_forbidden": None,
            "candidate_rendered_papers_disjoint_from_forbidden": None,
        }
        candidate_status = "unresolved_candidate_absent"
    else:
        forbidden = {item for items in normalized.values() for item in items}
        candidate_checks = {
            "candidate_ids_disjoint_from_forbidden": not (
                set(candidate_values.get("candidate_ids", ())) & forbidden
            ),
            "candidate_seeds_disjoint_from_forbidden": not (
                set(candidate_values.get("candidate_seeds", ())) & forbidden
            ),
            "candidate_references_disjoint_from_forbidden": not (
                set(candidate_values.get("candidate_references", ())) & forbidden
            ),
            "candidate_rendered_papers_disjoint_from_forbidden": not (
                set(candidate_values.get("candidate_rendered_papers", ())) & forbidden
            ),
        }
        candidate_status = "candidate_authenticated"
    return {
        "checks": checks,
        "digests": digests,
        "all_unique": all(checks.values()),
        "candidate_checks": candidate_checks,
        "candidate_status": candidate_status,
    }


__all__ = [
    "MAX_CONCURRENT_EPISODES",
    "MAX_PROVISIONING_ATTEMPTS",
    "AttemptState",
    "DraftEvent",
    "DraftPolicyError",
    "DraftState",
    "affordability_ledger",
    "canonical_json_bytes",
    "fresh_identities",
    "nonoverlap_digest",
    "reject_change_as_redeployment",
    "sha256_bytes",
    "sha256_json",
    "state_machine_contract",
    "transition",
]
