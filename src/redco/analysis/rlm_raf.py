"""RLM-shaped CPU proxy for dynamic-topology replay cost and soundness."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from math import fsum
from pathlib import Path

from redco.contracts import canonical_json
from redco.env.commands import CommandKind, JsonValue, TypedCommand
from redco.env.dynamic_replay import (
    BranchGuard,
    DynamicCommand,
    DynamicProgram,
    DynamicReplayEngine,
    EventRole,
    PolicyInvocation,
    WorkEstimate,
)
from redco.env.replay import ReplayMode

type WorkPayload = dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class RlmRafFailure:
    program_index: int
    alternative_index: int
    original_route: str
    replacement_route: str
    full_state_sha256: str
    sliced_state_sha256: str


@dataclass(frozen=True, slots=True)
class RlmRafReport:
    schema_version: int
    generated_at_utc: str
    campaign_seed: int
    programs: int
    alternatives_per_program: int
    paired_branches: int
    deterministic_failures: int
    topology_divergences: int
    added_events: int
    removed_events: int
    full_suffix_events: int
    sliced_suffix_events: int
    mean_sliced_event_fraction: float
    full_work_by_role: WorkPayload
    sliced_work_by_role: WorkPayload
    baseline_generated_tokens: int
    alternative_action_generated_tokens: int
    full_downstream_generated_tokens: int
    sliced_downstream_generated_tokens: int
    modeled_full_policy_token_raf: float
    modeled_sliced_policy_token_raf: float
    empirical_real_trace_status: str
    failures: tuple[RlmRafFailure, ...]
    passed_rlm_shaped_cpu_proxy: bool
    report_sha256: str = ""

    def unsigned_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("report_sha256")
        return payload

    def signed_dict(self) -> dict[str, object]:
        payload = self.unsigned_dict()
        payload["report_sha256"] = hashlib.sha256(
            canonical_json(payload)
        ).hexdigest()
        return payload


def run_rlm_raf_campaign(
    *,
    seed: int,
    programs: int,
    alternatives_per_program: int = 3,
    minimum_child_calls: int = 2,
    maximum_child_calls: int = 6,
) -> RlmRafReport:
    """Run exact paired replay over an RLM-shaped dynamic workload model."""
    if programs < 1 or alternatives_per_program < 1:
        raise ValueError("campaign sizes must be positive")
    if minimum_child_calls < 1 or maximum_child_calls < minimum_child_calls:
        raise ValueError("invalid child-call range")

    rng = random.Random(seed)
    failures: list[RlmRafFailure] = []
    topology_divergences = 0
    added_events = 0
    removed_events = 0
    full_event_counts: list[int] = []
    sliced_event_counts: list[int] = []
    full_work = _empty_work_payload()
    sliced_work = _empty_work_payload()
    baseline_generated_tokens = 0
    alternative_action_tokens = 0

    for program_index in range(programs):
        original_route = rng.choice(("direct", "recurse"))
        child_calls = rng.randint(minimum_child_calls, maximum_child_calls)
        program = _rlm_program(
            rng,
            program_index=program_index,
            original_route=original_route,
            child_calls=child_calls,
        )
        engine = DynamicReplayEngine(program, sampler=_deterministic_sampler)
        baseline_generated_tokens += _generated_tokens(
            program,
            engine.original.active_event_ids,
        )
        target_tokens = program.command("target-turn").work.generated_tokens

        for alternative_index in range(alternatives_per_program):
            replacement_label, replacement = _replacement_action(
                original_route,
                alternative_index,
            )
            full = engine.replay(
                target_event_id="target-turn",
                replacement=replacement,
                mode=ReplayMode.FULL_SUFFIX,
            )
            sliced = engine.replay(
                target_event_id="target-turn",
                replacement=replacement,
                mode=ReplayMode.SLICED,
            )
            if full.state_bytes != sliced.state_bytes:
                failures.append(
                    RlmRafFailure(
                        program_index=program_index,
                        alternative_index=alternative_index,
                        original_route=original_route,
                        replacement_route=replacement_label,
                        full_state_sha256=hashlib.sha256(
                            full.state_bytes
                        ).hexdigest(),
                        sliced_state_sha256=hashlib.sha256(
                            sliced.state_bytes
                        ).hexdigest(),
                    )
                )
            if sliced.topology.diverged:
                topology_divergences += 1
                added_events += len(sliced.topology.added_node_ids)
                removed_events += len(sliced.topology.removed_node_ids)

            full_ids = full.reexecuted_suffix_event_ids
            sliced_ids = sliced.reexecuted_suffix_event_ids
            full_event_counts.append(len(full_ids))
            sliced_event_counts.append(len(sliced_ids))
            _accumulate_work(
                full_work,
                program,
                full_ids,
                set(full.regenerated_policy_event_ids),
            )
            _accumulate_work(
                sliced_work,
                program,
                sliced_ids,
                set(sliced.regenerated_policy_event_ids),
            )
            alternative_action_tokens += target_tokens

    full_suffix_events = sum(full_event_counts)
    sliced_suffix_events = sum(sliced_event_counts)
    fractions = [
        sliced / full
        for sliced, full in zip(
            sliced_event_counts,
            full_event_counts,
            strict=True,
        )
        if full
    ]
    mean_fraction = fsum(fractions) / len(fractions) if fractions else 0.0
    full_downstream_tokens = _total_metric(full_work, "generated_tokens")
    sliced_downstream_tokens = _total_metric(sliced_work, "generated_tokens")
    full_raf = (
        (
            baseline_generated_tokens
            + alternative_action_tokens
            + full_downstream_tokens
        )
        / baseline_generated_tokens
    )
    sliced_raf = (
        (
            baseline_generated_tokens
            + alternative_action_tokens
            + sliced_downstream_tokens
        )
        / baseline_generated_tokens
    )
    passed = (
        not failures
        and topology_divergences > 0
        and added_events > 0
        and removed_events > 0
        and sliced_suffix_events < full_suffix_events
    )
    return RlmRafReport(
        schema_version=1,
        generated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        campaign_seed=seed,
        programs=programs,
        alternatives_per_program=alternatives_per_program,
        paired_branches=programs * alternatives_per_program,
        deterministic_failures=len(failures),
        topology_divergences=topology_divergences,
        added_events=added_events,
        removed_events=removed_events,
        full_suffix_events=full_suffix_events,
        sliced_suffix_events=sliced_suffix_events,
        mean_sliced_event_fraction=mean_fraction,
        full_work_by_role=full_work,
        sliced_work_by_role=sliced_work,
        baseline_generated_tokens=baseline_generated_tokens,
        alternative_action_generated_tokens=alternative_action_tokens,
        full_downstream_generated_tokens=full_downstream_tokens,
        sliced_downstream_generated_tokens=sliced_downstream_tokens,
        modeled_full_policy_token_raf=full_raf,
        modeled_sliced_policy_token_raf=sliced_raf,
        empirical_real_trace_status="pending_recorded_or_live_rlm_traces",
        failures=tuple(failures),
        passed_rlm_shaped_cpu_proxy=passed,
    )


def _rlm_program(
    rng: random.Random,
    *,
    program_index: int,
    original_route: str,
    child_calls: int,
) -> DynamicProgram:
    deep = BranchGuard("is_recursive^0", True)
    direct = BranchGuard("is_recursive^0", False)
    initial_state: dict[str, JsonValue] = {
        "route_prompt": (
            [9, program_index % 101]
            if original_route == "recurse"
            else [0, 100 + program_index % 101]
        ),
        "task_text": f"task-{program_index}-" + ("context " * 12),
        "base_prompt": [1, 20, program_index % 101],
    }
    for index in range(child_calls):
        initial_state[f"child_prompt_{index}"] = [
            1,
            30 + index,
            program_index % 101,
        ]

    templates: list[DynamicCommand] = [
        DynamicCommand(
            TypedCommand(
                "target-turn",
                CommandKind.IDENTITY,
                ("route_prompt",),
                "route^0",
            ),
            role=EventRole.ROOT_POLICY,
            work=WorkEstimate(generated_tokens=rng.randint(24, 64)),
            policy=PolicyInvocation(
                prompt_artifact="route_prompt",
                checkpoint_id="theta-rlm-proxy",
                decoding_config_hash="sample-proxy",
                event_seed=10_000 + program_index,
            ),
        ),
        DynamicCommand(
            TypedCommand(
                "route-check",
                CommandKind.CONTAINS,
                ("route^0",),
                "is_recursive^0",
                (("needle", 9),),
            ),
            work=WorkEstimate(cpu_units=1),
        ),
        DynamicCommand(
            TypedCommand(
                "partition-context",
                CommandKind.PARTITION,
                ("task_text",),
                "chunks^0",
                (("chunk_size", rng.randint(12, 24)),),
            ),
            work=WorkEstimate(cpu_units=3),
        ),
        DynamicCommand(
            TypedCommand(
                "inspect-context",
                CommandKind.SELECT,
                ("chunks^0",),
                "first_chunk^0",
                (("index", 0),),
            ),
            work=WorkEstimate(cpu_units=1),
        ),
        DynamicCommand(
            TypedCommand(
                "direct-root-turn",
                CommandKind.IDENTITY,
                ("base_prompt",),
                "answer^0",
                observation_dependencies=("inspect-context",),
            ),
            guard=direct,
            role=EventRole.ROOT_POLICY,
            work=WorkEstimate(generated_tokens=rng.randint(80, 160)),
            policy=PolicyInvocation(
                prompt_artifact="base_prompt",
                checkpoint_id="theta-rlm-proxy",
                decoding_config_hash="sample-proxy",
                event_seed=20_000 + program_index,
            ),
        ),
    ]
    normalized_outputs: list[str] = []
    for index in range(child_calls):
        child_output = f"child_{index}^0"
        normalized_output = f"normalized_{index}^0"
        normalized_outputs.append(normalized_output)
        templates.extend(
            (
                DynamicCommand(
                    TypedCommand(
                        f"child-call-{index}",
                        CommandKind.IDENTITY,
                        (f"child_prompt_{index}",),
                        child_output,
                        observation_dependencies=("inspect-context",),
                    ),
                    guard=deep,
                    role=EventRole.SUBCALL_POLICY,
                    work=WorkEstimate(
                        generated_tokens=rng.randint(64, 192)
                    ),
                    policy=PolicyInvocation(
                        prompt_artifact=f"child_prompt_{index}",
                        checkpoint_id="theta-rlm-proxy",
                        decoding_config_hash="sample-proxy",
                        event_seed=30_000 + program_index * 10 + index,
                    ),
                ),
                DynamicCommand(
                    TypedCommand(
                        f"normalize-child-{index}",
                        CommandKind.IDENTITY,
                        (child_output,),
                        normalized_output,
                    ),
                    guard=deep,
                    work=WorkEstimate(cpu_units=2),
                ),
            )
        )
    templates.extend(
        (
            DynamicCommand(
                TypedCommand(
                    "aggregate-children",
                    CommandKind.LIST_CONCAT,
                    tuple(normalized_outputs),
                    "aggregate^0",
                ),
                guard=deep,
                work=WorkEstimate(cpu_units=max(1, child_calls)),
            ),
            DynamicCommand(
                TypedCommand(
                    "recursive-root-turn",
                    CommandKind.IDENTITY,
                    ("aggregate^0",),
                    "answer^0",
                ),
                guard=deep,
                role=EventRole.ROOT_POLICY,
                work=WorkEstimate(generated_tokens=rng.randint(80, 160)),
                policy=PolicyInvocation(
                    prompt_artifact="aggregate^0",
                    checkpoint_id="theta-rlm-proxy",
                    decoding_config_hash="sample-proxy",
                    event_seed=40_000 + program_index,
                ),
            ),
            DynamicCommand(
                TypedCommand(
                    "judge",
                    CommandKind.CONTAINS,
                    ("answer^0",),
                    "reward^0",
                    (("needle", 1),),
                ),
                role=EventRole.JUDGE,
                work=WorkEstimate(
                    generated_tokens=rng.randint(32, 80),
                    judge_calls=1,
                ),
            ),
            DynamicCommand(
                TypedCommand(
                    "finish",
                    CommandKind.FINAL,
                    ("reward^0",),
                    "final^0",
                ),
                work=WorkEstimate(cpu_units=1),
            ),
        )
    )
    return DynamicProgram(
        initial_state=initial_state,
        commands=tuple(templates),
        terminal_output="final^0",
    )


def _replacement_action(
    original_route: str,
    alternative_index: int,
) -> tuple[str, list[JsonValue]]:
    if alternative_index % 2 == 0:
        if original_route == "direct":
            return "recurse-alternative", [9, 200 + alternative_index]
        return "direct-alternative", [0, 200 + alternative_index]
    token = 9 if original_route == "recurse" else 0
    return (
        f"{original_route}-alternative-{alternative_index}",
        [token, 200 + alternative_index],
    )


def _empty_work_payload() -> WorkPayload:
    return {
        role.value: {
            "events": 0,
            "cpu_units": 0,
            "generated_tokens": 0,
            "judge_calls": 0,
        }
        for role in EventRole
    }


def _accumulate_work(
    payload: WorkPayload,
    program: DynamicProgram,
    event_ids: tuple[str, ...],
    regenerated_policy_events: set[str],
) -> None:
    for event_id in event_ids:
        template = program.command(event_id)
        role = payload[template.role.value]
        role["events"] += 1
        role["cpu_units"] += template.work.cpu_units
        if (
            template.policy is None
            or event_id in regenerated_policy_events
        ):
            role["generated_tokens"] += template.work.generated_tokens
        role["judge_calls"] += template.work.judge_calls


def _generated_tokens(
    program: DynamicProgram,
    event_ids: tuple[str, ...],
) -> int:
    return sum(program.command(event_id).work.generated_tokens for event_id in event_ids)


def _total_metric(payload: WorkPayload, metric: str) -> int:
    return sum(role[metric] for role in payload.values())


def _deterministic_sampler(
    prompt_token_ids: tuple[int, ...],
    event_seed: int,
) -> tuple[int, ...]:
    del event_seed
    return (*prompt_token_ids, 1, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--programs", type=int, default=1_000)
    parser.add_argument("--alternatives", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/stage-b/rlm-raf-cpu/report.json"),
    )
    args = parser.parse_args()
    report = run_rlm_raf_campaign(
        seed=args.seed,
        programs=args.programs,
        alternatives_per_program=args.alternatives,
    )
    payload = report.signed_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(payload) + b"\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0 if report.passed_rlm_shaped_cpu_proxy else 1


if __name__ == "__main__":
    raise SystemExit(main())
