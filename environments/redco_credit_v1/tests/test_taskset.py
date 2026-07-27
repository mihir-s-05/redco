import verifiers.v1 as vf
from redco_credit_v1.taskset import (
    RedcoCreditEnvConfig,
    RedcoCreditTaskset,
    RedcoCreditTasksetConfig,
    branch_sampling,
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
    assert parse_action("reasoning\n<left></left>", actions) == "left"
    assert parse_action("<action>left</action><left></left>", actions) == "left"
    assert parse_action("<action>left</action><right></right>", actions) is None
    assert parse_action("<action>invented</action>", actions) is None
    assert parse_action("<action>invented</action><left></left>", actions) is None
    assert parse_action("left", actions) is None
    assert parse_route("<route>gamma</route>") == "gamma"
    assert parse_route("<route>invented</route>") is None


def test_branch_sampling_derives_distinct_auditable_requests() -> None:
    base = vf.SamplingConfig(
        temperature=0.7,
        max_tokens=64,
        extra_body={"cache_salt": "snapshot-0", "top_k": 20},
    )

    first = branch_sampling(base, seed=101, cache_salt_suffix="episode:original")
    second = branch_sampling(
        base,
        seed=202,
        cache_salt_suffix="episode:alternative_1",
    )

    assert first.temperature == second.temperature == 0.7
    assert first.max_tokens == second.max_tokens == 64
    assert first.model_dump()["seed"] == 101
    assert second.model_dump()["seed"] == 202
    assert first.model_dump()["extra_body"] == {
        "cache_salt": "snapshot-0:redco:episode:original",
        "top_k": 20,
    }
    assert second.model_dump()["extra_body"]["cache_salt"].endswith(
        "episode:alternative_1"
    )
    assert base.model_dump()["extra_body"]["cache_salt"] == "snapshot-0"


def test_env_config_freezes_branch_count_and_replay_mode() -> None:
    config = RedcoCreditEnvConfig(
        taskset={"id": "redco-credit-v1"},
        replay_mode="full_suffix",
    )

    assert config.branching_enabled is True
    assert config.replay_mode == "full_suffix"
