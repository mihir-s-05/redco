from redco_credit_v1.taskset import (
    RedcoCreditEnvConfig,
    RedcoCreditTaskset,
    RedcoCreditTasksetConfig,
    parse_action,
    parse_route,
)


def test_taskset_covers_every_probe_at_each_seed() -> None:
    taskset = RedcoCreditTaskset(
        RedcoCreditTasksetConfig(repeats_per_probe=2, exogenous_seed_offset=10)
    )

    tasks = taskset.load()

    assert len(tasks) == 16
    assert {task.data.exogenous_seed for task in tasks} == {10, 11}
    assert all("<route>VALUE</route>" in task.data.prompt for task in tasks)
    assert all(
        canonical not in task.data.actions
        for task in tasks
        for _, canonical in task.data.action_map
    )
    assert tasks[0].data.actions == (
        "p0_a0",
        "p0_a1",
        "p0_a2",
        "p0_a3",
        "p0_a4",
        "p0_a5",
        "p0_a6",
        "p0_a7",
    )


def test_action_parser_never_repairs_or_accepts_invalid_values() -> None:
    actions = ("left", "right")

    assert parse_action("reasoning\n<action>left</action>", actions) == "left"
    assert parse_action("<action>invented</action>", actions) is None
    assert parse_action("left", actions) is None
    assert parse_route("<route>gamma</route>") == "gamma"
    assert parse_route("<route>invented</route>") is None


def test_env_config_freezes_branch_count_and_replay_mode() -> None:
    config = RedcoCreditEnvConfig(
        taskset={"id": "redco-credit-v1"},
        replay_mode="full_suffix",
    )

    assert config.branching_enabled is True
    assert config.replay_mode == "full_suffix"
