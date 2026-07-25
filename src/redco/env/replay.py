"""Deterministic prefix oracle plus full-suffix and sliced replay."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from redco.env.commands import JsonValue, TypedCommand, execute_command


class ReplayMode(StrEnum):
    FULL_SUFFIX = "full_suffix"
    SLICED = "sliced"


@dataclass(frozen=True, slots=True)
class Program:
    initial_state: dict[str, JsonValue]
    commands: tuple[TypedCommand, ...]
    terminal_output: str

    def __post_init__(self) -> None:
        available = set(self.initial_state)
        event_ids: set[str] = set()
        for command in self.commands:
            if command.event_id in event_ids:
                raise ValueError(f"duplicate event_id: {command.event_id}")
            missing_inputs = set(command.inputs) - available
            if missing_inputs:
                raise ValueError(
                    f"{command.event_id} reads unavailable inputs: {sorted(missing_inputs)}"
                )
            unknown_dependencies = set(command.explicit_event_dependencies) - event_ids
            if unknown_dependencies:
                raise ValueError(
                    f"{command.event_id} has unavailable event dependencies: "
                    f"{sorted(unknown_dependencies)}"
                )
            if command.output in available:
                raise ValueError(
                    f"SSA output already exists: {command.output}; create a new version"
                )
            event_ids.add(command.event_id)
            available.add(command.output)
        if self.terminal_output not in available:
            raise ValueError("terminal_output is not produced by the program")


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    state: dict[str, JsonValue]
    restored_prefix_event_ids: tuple[str, ...]
    reexecuted_suffix_event_ids: tuple[str, ...]

    def terminal(self, output_name: str) -> JsonValue:
        return self.state[output_name]


class DeterministicExecutor:
    """Execute a complete restricted program from immutable initial inputs."""

    def execute(self, program: Program) -> ExecutionResult:
        state = dict(program.initial_state)
        executed: list[str] = []
        for command in program.commands:
            state[command.output] = execute_command(command, state)
            executed.append(command.event_id)
        return ExecutionResult(state, (), tuple(executed))


class ReplayEngine:
    """Paired replay modes sharing one deterministic execution semantics."""

    def __init__(self, program: Program) -> None:
        self.program = program
        self.original = DeterministicExecutor().execute(program)
        self._event_index = {
            command.event_id: index for index, command in enumerate(program.commands)
        }
        self._descendants = _event_descendants(program)

    def replay(
        self,
        *,
        target_event_id: str,
        replacement: JsonValue,
        mode: ReplayMode,
    ) -> ExecutionResult:
        try:
            target_index = self._event_index[target_event_id]
        except KeyError as error:
            raise KeyError(f"unknown target event: {target_event_id}") from error

        prefix_state = dict(self.program.initial_state)
        restored_prefix: list[str] = []
        for command in self.program.commands[:target_index]:
            prefix_state[command.output] = execute_command(command, prefix_state)
            restored_prefix.append(command.event_id)

        target = self.program.commands[target_index]
        prefix_state[target.output] = replacement

        if mode is ReplayMode.FULL_SUFFIX:
            state = prefix_state
            selected_events = {
                command.event_id for command in self.program.commands[target_index + 1 :]
            }
        elif mode is ReplayMode.SLICED:
            state = dict(self.original.state)
            state.update(prefix_state)
            selected_events = set(self._descendants[target_event_id])
        else:
            raise ValueError(f"unsupported replay mode: {mode}")

        reexecuted_suffix: list[str] = []
        for command in self.program.commands[target_index + 1 :]:
            if command.event_id not in selected_events:
                continue
            state[command.output] = execute_command(command, state)
            reexecuted_suffix.append(command.event_id)

        return ExecutionResult(
            state=state,
            restored_prefix_event_ids=tuple(restored_prefix),
            reexecuted_suffix_event_ids=tuple(reexecuted_suffix),
        )


def _event_descendants(program: Program) -> dict[str, frozenset[str]]:
    producer_by_artifact = {
        command.output: command.event_id for command in program.commands
    }
    outgoing: dict[str, set[str]] = {
        command.event_id: set() for command in program.commands
    }
    for command in program.commands:
        dependencies = set(command.explicit_event_dependencies)
        dependencies.update(
            producer_by_artifact[input_name]
            for input_name in command.inputs
            if input_name in producer_by_artifact
        )
        for dependency in dependencies:
            outgoing[dependency].add(command.event_id)

    closures: dict[str, frozenset[str]] = {}
    for event_id in outgoing:
        seen: set[str] = set()
        queue = deque([event_id])
        while queue:
            current = queue.popleft()
            for descendant in outgoing[current]:
                if descendant not in seen:
                    seen.add(descendant)
                    queue.append(descendant)
        closures[event_id] = frozenset(seen)
    return closures
