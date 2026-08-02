from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ENV_ROOT = Path(__file__).parents[1] / "environments" / "redco_evidence_selection_v2"
sys.path.insert(0, str(ENV_ROOT))

from test_stage_d_exact_action import _action, _prepared_key  # noqa: E402

from redco.analysis.stage_d_scientific_branch_group import (  # noqa: E402
    PreActionTargetCommitment,
)
from redco.analysis.stage_d_scientific_campaign import (  # noqa: E402
    runtime_snapshot_from_pre_action_evidence,
)
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress  # noqa: E402
from redco.contracts import canonical_json  # noqa: E402


def _fixture() -> tuple[bytes, PreActionTargetCommitment, object, bytes]:
    action = _action(key=_prepared_key())
    trace_id = "rollout-driver"
    address = PolicyEventAddress(0, "root", 0, 0)
    runtime = canonical_json(
        {
            "domain": "redco-stage-d-runtime-snapshot-v1",
            "schema_version": 1,
            "container": "frozen",
        }
    )
    snapshot = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-pre-action-prepared-snapshot-v1",
            "trace_id": trace_id,
            "event_address": address.as_payload(),
            "application_request": json.loads(action.key.request),
            "engine_endpoint": "http://engine/inference/v1/generate",
            "engine_request": json.loads(action.key.prepared_engine_request or b"null"),
            "engine_headers": {"X-Session-ID": trace_id},
            "observer_context": {
                "trace_id": trace_id,
                "rlm": {
                    "provenance_version": 2,
                    "depth": 0,
                    "session_id": "root-session",
                    "turn": 0,
                    "call_kind": "policy",
                    "lineage": "root",
                    "session_call_ordinal": 0,
                    "completed_episode_spawn_ordinals": [],
                },
            },
            "frozen_runtime_snapshot": json.loads(runtime),
        }
    )
    commitment = PreActionTargetCommitment(
        receipt=b"fixture",
        receipt_sha256="a" * 64,
        ledger_id="ledger",
        ledger_offset=1,
        prior_chain_sha256="b" * 64,
        group_id="group",
        rollout_id=trace_id,
        target_roster=("target",),
        target_ordinal=0,
        target_id="target",
        target_address=address,
        pre_action_snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),
        behavior_law_sha256="c" * 64,
        recorded_action_seed=action.key.sampler.seed,
        branch_count=2,
        continuation_replicates=1,
        failure_reward=-1.0,
        master_seed_sha256="d" * 64,
        commitment_sequence=0,
        action_reservation_sequence=1,
    )
    return snapshot, commitment, action, runtime


def test_runtime_snapshot_extracts_only_bound_nested_runtime() -> None:
    snapshot, commitment, action, runtime = _fixture()

    assert (
        runtime_snapshot_from_pre_action_evidence(
            snapshot,
            commitment=commitment,
            recorded_action=action,
        )
        == runtime
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("trace_id", "different-rollout"),
        ("engine_endpoint", "http://engine/v1/chat/completions"),
        ("frozen_runtime_snapshot", {"different": True}),
    ],
)
def test_runtime_snapshot_rejects_tampered_envelope(
    field: str,
    replacement: object,
) -> None:
    snapshot, commitment, action, _runtime = _fixture()
    payload = json.loads(snapshot)
    payload[field] = replacement

    with pytest.raises(ValueError, match="pre-action snapshot"):
        runtime_snapshot_from_pre_action_evidence(
            canonical_json(payload),
            commitment=commitment,
            recorded_action=action,
        )
