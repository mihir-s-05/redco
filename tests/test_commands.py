from __future__ import annotations

import pytest

from redco.env.commands import CommandKind, JsonValue, TypedCommand, execute_command


def test_typed_partition_and_select() -> None:
    state: dict[str, JsonValue] = {"context": "abcdefgh"}
    partition = TypedCommand(
        "partition",
        CommandKind.PARTITION,
        ("context",),
        "chunks",
        (("chunk_size", 3),),
    )
    state["chunks"] = execute_command(partition, state)
    select = TypedCommand(
        "select",
        CommandKind.SELECT,
        ("chunks",),
        "selected",
        (("index", 1),),
    )

    assert execute_command(select, state) == "def"


def test_typed_commands_fail_closed_on_bad_types() -> None:
    command = TypedCommand(
        "partition",
        CommandKind.PARTITION,
        ("context",),
        "chunks",
        (("chunk_size", "three"),),
    )

    with pytest.raises(TypeError, match="chunk_size"):
        execute_command(command, {"context": "abc"})


def test_list_concat_preserves_order() -> None:
    command = TypedCommand(
        "merge",
        CommandKind.LIST_CONCAT,
        ("left", "right"),
        "merged^0",
    )

    assert execute_command(
        command,
        {"left": [1, 2], "right": [3, {"value": True}]},
    ) == [1, 2, 3, {"value": True}]
