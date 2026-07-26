from __future__ import annotations

import pytest

from redco.env.commands import CommandKind, TypedCommand
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


def rlm_branch_program(route: str) -> DynamicProgram:
    deep = BranchGuard("is_deep^0", True)
    shallow = BranchGuard("is_deep^0", False)
    return DynamicProgram(
        initial_state={
            "route_source": route,
            "constant": 4,
            "short_answer": "short-ok",
            "child_answer": "deep-ok",
            "suffix": "!",
        },
        commands=(
            DynamicCommand(
                TypedCommand(
                    "target-turn",
                    CommandKind.IDENTITY,
                    ("route_source",),
                    "route^0",
                ),
                role=EventRole.ROOT_POLICY,
                work=WorkEstimate(generated_tokens=32),
            ),
            DynamicCommand(
                TypedCommand(
                    "route-check",
                    CommandKind.CONTAINS,
                    ("route^0",),
                    "is_deep^0",
                    (("needle", "deep"),),
                )
            ),
            DynamicCommand(
                TypedCommand(
                    "telemetry",
                    CommandKind.ADD,
                    ("constant",),
                    "telemetry^0",
                )
            ),
            DynamicCommand(
                TypedCommand(
                    "short-root-turn",
                    CommandKind.IDENTITY,
                    ("short_answer",),
                    "answer^0",
                ),
                guard=shallow,
                role=EventRole.ROOT_POLICY,
                work=WorkEstimate(generated_tokens=48),
            ),
            DynamicCommand(
                TypedCommand(
                    "deep-child-call",
                    CommandKind.IDENTITY,
                    ("child_answer",),
                    "child^0",
                    call_dependencies=("route-check",),
                ),
                guard=deep,
                role=EventRole.SUBCALL_POLICY,
                work=WorkEstimate(generated_tokens=96),
            ),
            DynamicCommand(
                TypedCommand(
                    "deep-artifact",
                    CommandKind.CONCAT,
                    ("child^0", "suffix"),
                    "composed^0",
                ),
                guard=deep,
            ),
            DynamicCommand(
                TypedCommand(
                    "deep-root-turn",
                    CommandKind.IDENTITY,
                    ("composed^0",),
                    "answer^0",
                    observation_dependencies=("deep-artifact",),
                ),
                guard=deep,
                role=EventRole.ROOT_POLICY,
                work=WorkEstimate(generated_tokens=64),
            ),
            DynamicCommand(
                TypedCommand(
                    "judge",
                    CommandKind.VERIFY,
                    ("answer^0",),
                    "reward^0",
                    (("expected", "deep-ok!"),),
                ),
                role=EventRole.JUDGE,
                work=WorkEstimate(generated_tokens=32, judge_calls=1),
            ),
            DynamicCommand(
                TypedCommand(
                    "finish",
                    CommandKind.FINAL,
                    ("reward^0",),
                    "final^0",
                )
            ),
        ),
        terminal_output="final^0",
    )


@pytest.mark.parametrize(
    ("original_route", "replacement", "added", "removed", "terminal"),
    [
        (
            "short",
            "deep",
            ("deep-child-call", "deep-artifact", "deep-root-turn"),
            ("short-root-turn",),
            True,
        ),
        (
            "deep",
            "short",
            ("short-root-turn",),
            ("deep-child-call", "deep-artifact", "deep-root-turn"),
            False,
        ),
    ],
)
def test_dynamic_topology_full_and_sliced_are_exact(
    original_route: str,
    replacement: str,
    added: tuple[str, ...],
    removed: tuple[str, ...],
    terminal: bool,
) -> None:
    engine = DynamicReplayEngine(rlm_branch_program(original_route))

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

    assert full.state_bytes == sliced.state_bytes
    assert full.terminal("final^0") is terminal
    assert sliced.topology.diverged
    assert sliced.topology.added_node_ids == added
    assert sliced.topology.removed_node_ids == removed
    assert "telemetry" in full.reexecuted_suffix_event_ids
    assert "telemetry" in sliced.reused_suffix_event_ids
    assert set(sliced.state) == set(full.state)


def test_dynamic_replay_rejects_target_missing_from_original_branch() -> None:
    engine = DynamicReplayEngine(rlm_branch_program("short"))

    with pytest.raises(
        ValueError,
        match="intervention target is absent from the original branch",
    ):
        engine.replay(
            target_event_id="deep-child-call",
            replacement="different",
            mode=ReplayMode.SLICED,
        )


def test_dynamic_policy_nodes_share_exact_key_reuse_semantics() -> None:
    calls: list[tuple[tuple[int, ...], int]] = []

    def sampler(prompt: tuple[int, ...], seed: int) -> tuple[int, ...]:
        calls.append((prompt, seed))
        return (*prompt, 1)

    deep = BranchGuard("is_deep^0", True)
    direct = BranchGuard("is_deep^0", False)
    program = DynamicProgram(
        initial_state={"route_prompt": [0], "stable_prompt": [7]},
        commands=(
            DynamicCommand(
                TypedCommand(
                    "target-policy",
                    CommandKind.IDENTITY,
                    ("route_prompt",),
                    "route^0",
                ),
                role=EventRole.ROOT_POLICY,
                policy=PolicyInvocation(
                    "route_prompt",
                    "theta-test",
                    "decode-test",
                    1,
                ),
            ),
            DynamicCommand(
                TypedCommand(
                    "route-check",
                    CommandKind.CONTAINS,
                    ("route^0",),
                    "is_deep^0",
                    (("needle", 9),),
                )
            ),
            DynamicCommand(
                TypedCommand(
                    "unrelated-policy",
                    CommandKind.IDENTITY,
                    ("stable_prompt",),
                    "unrelated^0",
                ),
                role=EventRole.SUBCALL_POLICY,
                policy=PolicyInvocation(
                    "stable_prompt",
                    "theta-test",
                    "decode-test",
                    2,
                ),
            ),
            DynamicCommand(
                TypedCommand(
                    "direct-policy",
                    CommandKind.IDENTITY,
                    ("stable_prompt",),
                    "answer^0",
                ),
                guard=direct,
                role=EventRole.ROOT_POLICY,
                policy=PolicyInvocation(
                    "stable_prompt",
                    "theta-test",
                    "decode-test",
                    3,
                ),
            ),
            DynamicCommand(
                TypedCommand(
                    "deep-policy",
                    CommandKind.IDENTITY,
                    ("route^0",),
                    "answer^0",
                ),
                guard=deep,
                role=EventRole.ROOT_POLICY,
                policy=PolicyInvocation(
                    "route^0",
                    "theta-test",
                    "decode-test",
                    4,
                ),
            ),
        ),
        terminal_output="answer^0",
    )
    engine = DynamicReplayEngine(program, sampler=sampler)

    full = engine.replay(
        target_event_id="target-policy",
        replacement=[9],
        mode=ReplayMode.FULL_SUFFIX,
    )
    sliced = engine.replay(
        target_event_id="target-policy",
        replacement=[9],
        mode=ReplayMode.SLICED,
    )

    assert full.state_bytes == sliced.state_bytes
    assert full.regenerated_policy_event_ids == ("deep-policy",)
    assert sliced.regenerated_policy_event_ids == ("deep-policy",)
    assert "unrelated-policy" in full.reused_policy_event_ids
    assert "unrelated-policy" in sliced.reused_policy_event_ids
    assert calls.count(((9,), 4)) == 2
