"""Randomized deterministic equivalence campaign for the Tier-0 replay engine."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from math import fsum

from redco.env.commands import CommandKind, JsonValue, TypedCommand
from redco.env.replay import Program, ReplayEngine, ReplayMode


@dataclass(frozen=True, slots=True)
class EquivalenceFailure:
    program_index: int
    target_event_id: str
    replacement: JsonValue
    full_terminal: JsonValue
    sliced_terminal: JsonValue


@dataclass(frozen=True, slots=True)
class EquivalenceReport:
    seed: int
    programs: int
    interventions: int
    failures: tuple[EquivalenceFailure, ...]
    full_suffix_events: int
    sliced_suffix_events: int
    mean_sliced_work_fraction: float
    dependency_edge_counts: dict[str, int]

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def run_randomized_equivalence(
    *,
    seed: int,
    program_count: int,
    interventions_per_program: int,
    events_per_program: int = 12,
) -> EquivalenceReport:
    """Run paired interventions with byte-equivalent JSON-domain values."""
    if program_count < 1 or interventions_per_program < 1 or events_per_program < 2:
        raise ValueError("campaign sizes must be positive and programs need two events")
    rng = random.Random(seed)
    failures: list[EquivalenceFailure] = []
    full_work: list[int] = []
    sliced_work: list[int] = []
    dependency_edge_counts = {
        "dataflow": 0,
        "control": 0,
        "call": 0,
        "side_effect": 0,
        "observation": 0,
        "resource": 0,
    }

    for program_index in range(program_count):
        program = _random_program(rng, events_per_program)
        for command in program.commands:
            dependency_edge_counts["dataflow"] += sum(
                input_name not in program.initial_state for input_name in command.inputs
            )
            dependency_edge_counts["control"] += len(command.control_dependencies)
            dependency_edge_counts["call"] += len(command.call_dependencies)
            dependency_edge_counts["side_effect"] += len(command.ordering_dependencies)
            dependency_edge_counts["observation"] += len(
                command.observation_dependencies
            )
            dependency_edge_counts["resource"] += len(command.resource_dependencies)
        engine = ReplayEngine(program)
        for _ in range(interventions_per_program):
            target = rng.choice(program.commands)
            replacement = rng.randint(-100, 100)
            full = engine.replay(
                target_event_id=target.event_id,
                replacement=replacement,
                mode=ReplayMode.FULL_SUFFIX,
            )
            sliced = engine.replay(
                target_event_id=target.event_id,
                replacement=replacement,
                mode=ReplayMode.SLICED,
            )
            full_terminal = full.terminal(program.terminal_output)
            sliced_terminal = sliced.terminal(program.terminal_output)
            if full.state != sliced.state:
                failures.append(
                    EquivalenceFailure(
                        program_index=program_index,
                        target_event_id=target.event_id,
                        replacement=replacement,
                        full_terminal=full_terminal,
                        sliced_terminal=sliced_terminal,
                    )
                )
            full_work.append(len(full.reexecuted_suffix_event_ids))
            sliced_work.append(len(sliced.reexecuted_suffix_event_ids))

    full_total = sum(full_work)
    sliced_total = sum(sliced_work)
    fractions = [
        sliced / full
        for sliced, full in zip(sliced_work, full_work, strict=True)
        if full > 0
    ]
    mean_fraction = fsum(fractions) / len(fractions) if fractions else 0.0
    return EquivalenceReport(
        seed=seed,
        programs=program_count,
        interventions=program_count * interventions_per_program,
        failures=tuple(failures),
        full_suffix_events=full_total,
        sliced_suffix_events=sliced_total,
        mean_sliced_work_fraction=mean_fraction,
        dependency_edge_counts=dependency_edge_counts,
    )


def _random_program(rng: random.Random, event_count: int) -> Program:
    initial_state: dict[str, JsonValue] = {
        "input^0": rng.randint(-10, 10),
        "bias^0": rng.randint(-10, 10),
        "constant^0": rng.randint(-10, 10),
    }
    available = list(initial_state)
    commands: list[TypedCommand] = []
    event_ids: list[str] = []
    for index in range(event_count):
        input_count = 1 if rng.random() < 0.4 else 2
        inputs = tuple(rng.sample(available, k=input_count))
        output = f"value^{index}"
        explicit: list[tuple[str, ...]] = [(), (), (), (), ()]
        if event_ids and rng.random() < 0.75:
            explicit[rng.randrange(len(explicit))] = (rng.choice(event_ids),)
        commands.append(
            TypedCommand(
                event_id=f"event-{index}",
                kind=CommandKind.ADD,
                inputs=inputs,
                output=output,
                control_dependencies=explicit[0],
                call_dependencies=explicit[1],
                ordering_dependencies=explicit[2],
                observation_dependencies=explicit[3],
                resource_dependencies=explicit[4],
            )
        )
        available.append(output)
        event_ids.append(f"event-{index}")
    return Program(
        initial_state=initial_state,
        commands=tuple(commands),
        terminal_output=commands[-1].output,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--programs", type=int, default=100)
    parser.add_argument("--interventions", type=int, default=20)
    parser.add_argument("--events", type=int, default=12)
    args = parser.parse_args()
    report = run_randomized_equivalence(
        seed=args.seed,
        program_count=args.programs,
        interventions_per_program=args.interventions,
        events_per_program=args.events,
    )
    print(json.dumps(report.as_dict(), sort_keys=True, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
