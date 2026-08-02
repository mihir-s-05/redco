"""Exact-roster, collection-only lifecycle for Stage-D source rollouts."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_source_contracts import SourceRollout
from redco.contracts import canonical_json


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def derive_source_episode_seed_and_salt(
    *,
    master_seed: str,
    scientific_group_id: str,
    rollout_slot: int,
) -> tuple[int, str]:
    """Derive one reproducible source draw without coupling rollout replicas."""
    if not master_seed or not scientific_group_id:
        raise ValueError("episode seed identities must be nonempty")
    if type(rollout_slot) is not int or rollout_slot < 0:
        raise ValueError("rollout slot must be a nonnegative integer")
    address = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-source-episode-seed-v1",
            "scientific_group_id": scientific_group_id,
            "rollout_slot": rollout_slot,
        }
    )
    digest = hmac.new(master_seed.encode("utf-8"), address, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") % (2**31), (f"stage-d-source-{digest.hex()}")


@dataclass(frozen=True, slots=True)
class SourceCollectionSlot:
    slot_id: str
    scientific_group_id: str
    example_id: str
    rollout_slot: int
    seed: int
    cache_salt: str

    @classmethod
    def build(
        cls,
        data: Mapping[str, Any],
        *,
        master_seed: str,
    ) -> SourceCollectionSlot:
        group_id = data.get("scientific_group_id")
        example_id = data.get("example_id")
        rollout_slot = data.get("rollout_slot")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("planned source slot lacks a scientific group")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("planned source slot lacks an example ID")
        if type(rollout_slot) is not int or rollout_slot < 0:
            raise ValueError("planned source slot lacks a nonnegative rollout slot")
        seed, cache_salt = derive_source_episode_seed_and_salt(
            master_seed=master_seed,
            scientific_group_id=group_id,
            rollout_slot=rollout_slot,
        )
        return cls(
            _source_slot_id(group_id, example_id, rollout_slot),
            group_id,
            example_id,
            rollout_slot,
            seed,
            cache_salt,
        )

    def to_payload(self) -> dict[str, str | int]:
        return {
            "slot_id": self.slot_id,
            "scientific_group_id": self.scientific_group_id,
            "example_id": self.example_id,
            "rollout_slot": self.rollout_slot,
            "seed": self.seed,
            "cache_salt": self.cache_salt,
        }


@dataclass(frozen=True, slots=True)
class StageDCollectionPlan:
    slots: tuple[SourceCollectionSlot, ...]
    plan_sha256: str

    @classmethod
    def build(
        cls,
        task_data: Sequence[Mapping[str, Any]],
        *,
        master_seed: str,
    ) -> StageDCollectionPlan:
        slots = tuple(
            SourceCollectionSlot.build(data, master_seed=master_seed) for data in task_data
        )
        if not slots or len({slot.slot_id for slot in slots}) != len(slots):
            raise ValueError("source collection plan must be nonempty and unique")
        group_members: dict[str, int] = {}
        for slot in slots:
            group_members[slot.scientific_group_id] = (
                group_members.get(slot.scientific_group_id, 0) + 1
            )
        if any(count < 2 for count in group_members.values()):
            raise ValueError("every source collection group requires at least two slots")
        payload = {
            "schema_version": 1,
            "domain": "redco-stage-d-source-collection-plan-v1",
            "slots": [slot.to_payload() for slot in slots],
        }
        return cls(slots, _sha256(canonical_json(payload)))

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-source-collection-plan-v1",
                "slots": [slot.to_payload() for slot in self.slots],
            }
        )


def verify_direct_collection_config(
    config: Any,
    *,
    planned_slot_count: int,
    require_new_output: bool = True,
) -> None:
    """Reject any outer lifecycle that can duplicate, resume, or omit a slot."""
    requirements = {
        "server": False,
        "num_rollouts": 1,
        "num_tasks": planned_slot_count,
        "shuffle": False,
        "resume": None,
        "push": False,
        "rich": False,
        "max_concurrent": 1,
    }
    mismatches = {
        name: (getattr(config, name, None), expected)
        for name, expected in requirements.items()
        if getattr(config, name, None) != expected
    }
    if mismatches:
        raise ValueError(f"source collection lifecycle is not exact: {mismatches}")
    output_dir = getattr(config, "output_dir", None)
    if not isinstance(output_dir, Path) or (
        require_new_output and output_dir.exists()
    ):
        raise ValueError("source collection requires a new explicit output directory")
    env = getattr(config, "env", None)
    if env is None or getattr(env, "max_concurrent", None) != 1:
        raise ValueError("source collection environment must be serial")
    if getattr(env.retries, "max_retries", None) != 0:
        raise ValueError("source collection forbids environment retries")
    if getattr(env.agent.retries, "max_retries", None) != 0:
        raise ValueError("source collection forbids agent retries")


async def run_exact_source_collection(
    config: Any,
    *,
    preregistered_plan_sha256: str,
    run_eval: Callable[[Any, Any], Awaitable[Sequence[Any]]],
    load_environment: Callable[[Any], Any],
    load_verified_sources: Callable[[], Sequence[SourceRollout]],
    persist_plan: Callable[[bytes], None],
) -> tuple[StageDCollectionPlan, tuple[Any, ...], bytes]:
    """Execute exactly one direct, serial, no-resume source collection campaign."""
    env = load_environment(config.env)
    tasks = env.taskset.select(config.num_tasks, config.shuffle)
    plan = StageDCollectionPlan.build(
        [task.data.model_dump(mode="json") for task in tasks],
        master_seed=config.env.master_seed,
    )
    verify_direct_collection_config(config, planned_slot_count=len(plan.slots))
    if plan.plan_sha256 != preregistered_plan_sha256:
        raise ValueError("materialized source collection plan differs from preregistration")
    persist_plan(plan.to_bytes())
    episodes = tuple(await run_eval(env, config))
    sources = tuple(load_verified_sources())
    receipt = verify_collection_outcomes(plan, episodes, sources)
    return plan, episodes, receipt


def verify_collection_outcomes(
    plan: StageDCollectionPlan,
    episodes: Sequence[Any],
    sources: Sequence[SourceRollout],
) -> bytes:
    """Bind every planned slot to one terminal source, including ineligible ones."""
    if len(episodes) != len(plan.slots):
        raise ValueError("source collection stopped before its complete planned roster")
    if len(sources) != len(plan.slots):
        raise ValueError("source artifact count differs from the planned denominator")
    if any(type(source) is not SourceRollout for source in sources):
        raise TypeError("collection outcomes require already-verified source contracts")
    by_rollout = {source.rollout_id: source for source in sources}
    if len(by_rollout) != len(sources):
        raise ValueError("source collection contains duplicate rollout IDs")
    dispositions: list[dict[str, Any]] = []
    used_rollouts: set[str] = set()
    for slot, episode in zip(plan.slots, episodes, strict=True):
        if getattr(episode, "ok", None) is not True:
            raise ValueError("planned source episode did not terminate successfully")
        traces = getattr(episode, "traces", None)
        if not isinstance(traces, list) or len(traces) != 1:
            raise ValueError("planned source episode did not produce exactly one trace")
        trace = traces[0]
        task_payload = trace.task.data.model_dump(mode="json")
        observed_group = task_payload.get("scientific_group_id")
        observed_example = task_payload.get("example_id")
        observed_slot = task_payload.get("rollout_slot")
        if (
            not isinstance(observed_group, str)
            or not isinstance(observed_example, str)
            or type(observed_slot) is not int
            or _source_slot_id(observed_group, observed_example, observed_slot)
            != slot.slot_id
            or observed_group != slot.scientific_group_id
            or observed_example != slot.example_id
            or observed_slot != slot.rollout_slot
        ):
            raise ValueError("source episode order or logical slot changed")
        source = by_rollout.get(trace.id)
        if source is None or source.group_id != slot.scientific_group_id:
            raise ValueError("planned source slot lacks its exact source artifact")
        used_rollouts.add(source.rollout_id)
        dispositions.append(
            {
                "slot_id": slot.slot_id,
                "example_id": slot.example_id,
                "rollout_slot": slot.rollout_slot,
                "seed": slot.seed,
                "cache_salt": slot.cache_salt,
                "rollout_id": source.rollout_id,
                "source_sha256": source.source_sha256,
                "branch_eligible": source.branch_eligible,
                "disposition": (
                    "eligible" if source.branch_eligible else "natural_topology_ineligible"
                ),
                "ineligibility_reason": source.ineligibility_reason,
            }
        )
    if used_rollouts != set(by_rollout):
        raise ValueError("unplanned source artifacts were present")
    return canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-source-collection-receipt-v1",
            "plan_sha256": plan.plan_sha256,
            "planned_slot_count": len(plan.slots),
            "terminal_slot_count": len(dispositions),
            "dispositions": dispositions,
        }
    )


def verify_collection_receipt(
    plan: StageDCollectionPlan,
    sources: Sequence[SourceRollout],
    receipt_bytes: bytes,
    *,
    evidence_loader: Callable[[str], bytes],
    allow_fixture_only: bool = False,
) -> str:
    """Verify a runner receipt before binding it into trainer authorization."""
    try:
        payload = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("collection receipt must be canonical JSON") from error
    if not isinstance(payload, dict) or canonical_json(payload) != receipt_bytes:
        raise ValueError("collection receipt must be canonical JSON")
    if set(payload) != {
        "schema_version",
        "domain",
        "plan_sha256",
        "planned_slot_count",
        "terminal_slot_count",
        "dispositions",
    }:
        raise ValueError("collection receipt fields differ")
    dispositions = payload.get("dispositions")
    if (
        payload.get("schema_version") != 1
        or payload.get("domain") != "redco-stage-d-source-collection-receipt-v1"
        or payload.get("plan_sha256") != plan.plan_sha256
        or payload.get("planned_slot_count") != len(plan.slots)
        or payload.get("terminal_slot_count") != len(plan.slots)
        or not isinstance(dispositions, list)
        or len(dispositions) != len(plan.slots)
    ):
        raise ValueError("collection receipt does not cover the frozen denominator")
    source_by_sha = {source.source_sha256: source for source in sources}
    if len(source_by_sha) != len(sources):
        raise ValueError("collection source roster is not unique")
    used: set[str] = set()
    for slot, disposition in zip(plan.slots, dispositions, strict=True):
        if not isinstance(disposition, dict) or set(disposition) != {
            "slot_id",
            "example_id",
            "rollout_slot",
            "seed",
            "cache_salt",
            "rollout_id",
            "source_sha256",
            "branch_eligible",
            "disposition",
            "ineligibility_reason",
        }:
            raise ValueError("collection disposition fields differ")
        source_sha256 = disposition.get("source_sha256")
        source = source_by_sha.get(source_sha256) if isinstance(source_sha256, str) else None
        expected_disposition = (
            "eligible"
            if source is not None and source.branch_eligible
            else "natural_topology_ineligible"
        )
        source_task = (
            {} if allow_fixture_only else _source_task_data(source, evidence_loader=evidence_loader)
        )
        source_sampling_matches = allow_fixture_only or (
            source is not None
            and all(
                _action_uses_slot_sampling(decision.action.key.request, slot)
                for decision in source.decisions
            )
        )
        if (
            disposition.get("slot_id") != slot.slot_id
            or disposition.get("example_id") != slot.example_id
            or disposition.get("rollout_slot") != slot.rollout_slot
            or disposition.get("seed") != slot.seed
            or disposition.get("cache_salt") != slot.cache_salt
            or source is None
            or disposition.get("rollout_id") != source.rollout_id
            or source.group_id != slot.scientific_group_id
            or (
                not allow_fixture_only
                and (
                    source_task.get("scientific_group_id") != slot.scientific_group_id
                    or source_task.get("example_id") != slot.example_id
                    or source_task.get("rollout_slot") != slot.rollout_slot
                )
            )
            or not source_sampling_matches
            or disposition.get("branch_eligible") is not source.branch_eligible
            or disposition.get("disposition") != expected_disposition
            or disposition.get("ineligibility_reason") != source.ineligibility_reason
            or source.source_sha256 in used
        ):
            raise ValueError("collection disposition differs from its exact source")
        used.add(source.source_sha256)
    if used != set(source_by_sha):
        raise ValueError("collection receipt omitted or added source artifacts")
    return _sha256(receipt_bytes)


def _source_task_data(
    source: SourceRollout | None,
    *,
    evidence_loader: Callable[[str], bytes],
) -> Mapping[str, Any]:
    if source is None:
        return {}
    try:
        episode = json.loads(evidence_loader(source.trace_sha256))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source trace evidence is not JSON") from error
    traces = episode.get("traces") if isinstance(episode, dict) else None
    if not isinstance(traces, list) or len(traces) != 1 or not isinstance(traces[0], dict):
        raise ValueError("source trace evidence has no unique trace")
    task = traces[0].get("task")
    data = task.get("data") if isinstance(task, dict) else None
    if not isinstance(data, dict):
        raise ValueError("source trace evidence lacks task data")
    return data


def _source_slot_id(
    scientific_group_id: str,
    example_id: str,
    rollout_slot: int,
) -> str:
    identity = canonical_json(
        {
            "domain": "redco-stage-d-source-slot-v1",
            "scientific_group_id": scientific_group_id,
            "example_id": example_id,
            "rollout_slot": rollout_slot,
        }
    )
    return f"source-slot-{_sha256(identity)[:24]}"


def _action_uses_slot_sampling(
    request_bytes: bytes,
    slot: SourceCollectionSlot,
) -> bool:
    try:
        request = json.loads(request_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(request, dict):
        return False
    extra_body = request.get("extra_body")
    return (
        request.get("seed") == slot.seed
        and isinstance(extra_body, dict)
        and extra_body.get("cache_salt") == slot.cache_salt
    )
