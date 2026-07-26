"""Restricted, deterministic typed-command substrate for Tier 0."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None


class CommandKind(StrEnum):
    IDENTITY = "identity"
    PARTITION = "partition"
    SELECT = "select"
    CONCAT = "concat"
    LIST_CONCAT = "list_concat"
    ADD = "add"
    CONTAINS = "contains"
    IF_ELSE = "if_else"
    VERIFY = "verify"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class TypedCommand:
    event_id: str
    kind: CommandKind
    inputs: tuple[str, ...]
    output: str
    params: tuple[tuple[str, JsonValue], ...] = ()
    control_dependencies: tuple[str, ...] = ()
    call_dependencies: tuple[str, ...] = ()
    ordering_dependencies: tuple[str, ...] = ()
    observation_dependencies: tuple[str, ...] = ()
    resource_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.event_id or not self.output:
            raise ValueError("event_id and output must be non-empty")
        if tuple(sorted(self.params, key=lambda item: item[0])) != self.params:
            raise ValueError("params must be sorted by key")
        if len({key for key, _ in self.params}) != len(self.params):
            raise ValueError("parameter keys must be unique")

    def parameter(self, key: str, default: JsonValue = None) -> JsonValue:
        return dict(self.params).get(key, default)

    @property
    def explicit_event_dependencies(self) -> tuple[str, ...]:
        return (
            self.control_dependencies
            + self.call_dependencies
            + self.ordering_dependencies
            + self.observation_dependencies
            + self.resource_dependencies
        )


def execute_command(command: TypedCommand, state: Mapping[str, JsonValue]) -> JsonValue:
    """Execute one total, deterministic command against immutable-style inputs."""
    values = tuple(_lookup(state, name) for name in command.inputs)

    match command.kind:
        case CommandKind.IDENTITY | CommandKind.FINAL:
            _require_arity(command, values, 1)
            return values[0]
        case CommandKind.PARTITION:
            _require_arity(command, values, 1)
            text = _require_string(values[0], "partition input")
            chunk_size = _require_int(command.parameter("chunk_size"), "chunk_size")
            if chunk_size < 1:
                raise ValueError("chunk_size must be positive")
            return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
        case CommandKind.SELECT:
            _require_arity(command, values, 1)
            sequence = _require_list(values[0], "select input")
            index = _require_int(command.parameter("index"), "index")
            try:
                return sequence[index]
            except IndexError as error:
                raise ValueError(f"select index out of range: {index}") from error
        case CommandKind.CONCAT:
            separator = _require_string(command.parameter("separator", ""), "separator")
            return separator.join(_require_string(value, "concat input") for value in values)
        case CommandKind.LIST_CONCAT:
            flattened: list[JsonValue] = []
            for value in values:
                flattened.extend(_require_list(value, "list_concat input"))
            return flattened
        case CommandKind.ADD:
            return sum(_require_number(value, "add input") for value in values)
        case CommandKind.CONTAINS:
            _require_arity(command, values, 1)
            needle = command.parameter("needle")
            haystack = values[0]
            if isinstance(haystack, str):
                return _require_string(needle, "needle") in haystack
            if isinstance(haystack, list):
                return needle in haystack
            raise TypeError("contains input must be a string or list")
        case CommandKind.IF_ELSE:
            _require_arity(command, values, 3)
            predicate = values[0]
            if not isinstance(predicate, bool):
                raise TypeError("if_else predicate must be bool")
            return values[1] if predicate else values[2]
        case CommandKind.VERIFY:
            _require_arity(command, values, 1)
            return values[0] == command.parameter("expected")


def _lookup(state: Mapping[str, JsonValue], name: str) -> JsonValue:
    try:
        return state[name]
    except KeyError as error:
        raise KeyError(f"missing command input: {name}") from error


def _require_arity(
    command: TypedCommand,
    values: tuple[JsonValue, ...],
    expected: int,
) -> None:
    if len(values) != expected:
        raise ValueError(f"{command.kind} requires {expected} inputs, got {len(values)}")


def _require_string(value: JsonValue, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str")
    return value


def _require_int(value: JsonValue, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be int")
    return value


def _require_number(value: JsonValue, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be int or float")
    return value


def _require_list(value: JsonValue, label: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be list")
    return value
