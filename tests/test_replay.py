from __future__ import annotations

from redco.env.commands import CommandKind, TypedCommand
from redco.env.replay import Program, ReplayEngine, ReplayMode


def replay_program() -> Program:
    return Program(
        initial_state={"candidate": "wrong", "constant": "stable"},
        commands=(
            TypedCommand("target", CommandKind.IDENTITY, ("candidate",), "answer^0"),
            TypedCommand("unrelated", CommandKind.IDENTITY, ("constant",), "constant^0"),
            TypedCommand(
                "combine",
                CommandKind.CONCAT,
                ("answer^0", "constant^0"),
                "combined^0",
                (("separator", "|"),),
            ),
            TypedCommand("finish", CommandKind.FINAL, ("combined^0",), "final^0"),
        ),
        terminal_output="final^0",
    )


def test_sliced_matches_full_and_skips_unaffected_suffix_work() -> None:
    engine = ReplayEngine(replay_program())

    full = engine.replay(
        target_event_id="target",
        replacement="correct",
        mode=ReplayMode.FULL_SUFFIX,
    )
    sliced = engine.replay(
        target_event_id="target",
        replacement="correct",
        mode=ReplayMode.SLICED,
    )

    assert full.terminal("final^0") == sliced.terminal("final^0") == "correct|stable"
    assert full.reexecuted_suffix_event_ids == ("unrelated", "combine", "finish")
    assert sliced.reexecuted_suffix_event_ids == ("combine", "finish")


def test_no_effect_intervention_reuses_original_terminal_artifact() -> None:
    engine = ReplayEngine(replay_program())

    sliced = engine.replay(
        target_event_id="unrelated",
        replacement="changed",
        mode=ReplayMode.SLICED,
    )

    assert sliced.terminal("final^0") == "wrong|changed"
    assert sliced.reexecuted_suffix_event_ids == ("combine", "finish")

