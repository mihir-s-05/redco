import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import verifiers.v1 as vf
from redco_credit_v1.taskset import (
    RedcoCreditEnv,
    RedcoCreditEnvConfig,
    RedcoCreditTaskset,
    RedcoCreditTasksetConfig,
    _context_seed,
    _forced_smoke_choices,
    branch_sampling,
    confusion_reward,
    constrain_root_routes,
    context_sampling,
    parse_action,
    parse_route,
)

from redco.algo.branching import trajectory_rloo


@dataclass(frozen=True)
class _MockContext:
    sampling: vf.SamplingConfig


class _MockAgent:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.ctx = _MockContext(
            vf.SamplingConfig(
                temperature=0.7,
                max_tokens=64,
                extra_body={"cache_salt": "mock"},
            )
        )
        self.tasks: list[vf.Task] = []
        self.traces: list[SimpleNamespace] = []
        self.trainable = True

    async def run(self, task: vf.Task) -> SimpleNamespace:
        self.tasks.append(task)
        metrics: dict[str, float] = {}
        trace = SimpleNamespace(
            id=f"mock-trace-{len(self.traces)}",
            last_reply=self.reply,
            metrics=metrics,
            record_metric=metrics.__setitem__,
        )
        self.traces.append(trace)
        return trace


def test_taskset_covers_every_probe_at_each_seed() -> None:
    taskset = RedcoCreditTaskset(
        RedcoCreditTasksetConfig(repeats_per_probe=2, exogenous_seed_offset=10)
    )

    tasks = taskset.load()

    assert len(tasks) == 16
    assert {task.data.exogenous_seed for task in tasks} == {10, 11}
    assert all("<route>VALUE</route>" in task.data.prompt for task in tasks)
    assert all(
        tuple(alias for alias, _ in task.data.action_map) == task.data.actions
        for task in tasks
    )
    assert all(
        len({canonical for _, canonical in task.data.action_map})
        == len(task.data.action_map)
        for task in tasks
    )
    assert tasks[0].data.actions == (
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
    )


def test_taskset_can_select_a_preregistered_probe_subset() -> None:
    taskset = RedcoCreditTaskset(
        RedcoCreditTasksetConfig(
            repeats_per_probe=3,
            exogenous_seed_offset=20,
            probe_names=("integration_planted_needle",),
        )
    )

    tasks = taskset.load()

    assert len(tasks) == 3
    assert {task.data.probe_name for task in tasks} == {
        "integration_planted_needle"
    }
    assert {task.data.exogenous_seed for task in tasks} == {20, 21, 22}
    assert all(task.data.actions == tuple(str(index) for index in range(8)) for task in tasks)


def test_action_parser_never_repairs_or_accepts_invalid_values() -> None:
    actions = ("left", "right")

    assert parse_action("reasoning\n<action>left</action>", actions) == "left"
    assert parse_action("reasoning\n<left></left>", actions) == "left"
    assert parse_action(" left \n", actions) == "left"
    assert parse_action("<action>left</action><left></left>", actions) == "left"
    assert parse_action("<action>left</action><right></right>", actions) is None
    assert parse_action("<action>invented</action>", actions) is None
    assert parse_action("<action>invented</action><left></left>", actions) is None
    assert parse_action("answer: left", actions) is None
    assert parse_route("<route>gamma</route>") == "gamma"
    assert parse_route("<route>invented</route>") is None


def test_branch_sampling_derives_distinct_auditable_requests() -> None:
    base = vf.SamplingConfig(
        temperature=0.7,
        max_tokens=64,
        extra_body={"cache_salt": "snapshot-0", "top_k": 20},
    )

    first = branch_sampling(
        base,
        seed=101,
        cache_salt_suffix="episode:original",
        temperature=2.0,
    )
    second = branch_sampling(
        base,
        seed=202,
        cache_salt_suffix="episode:alternative_1",
        temperature=2.0,
    )

    assert first.temperature == second.temperature == 2.0
    assert first.max_tokens == second.max_tokens == 1
    assert first.model_dump()["seed"] == 101
    assert second.model_dump()["seed"] == 202
    assert first.model_dump().get("allowed_token_ids") is None
    assert first.model_dump()["extra_body"] == {
        "cache_salt": "snapshot-0:redco:episode:original",
        "top_k": 20,
    }
    assert second.model_dump()["extra_body"]["cache_salt"].endswith(
        "episode:alternative_1"
    )
    assert base.model_dump()["extra_body"]["cache_salt"] == "snapshot-0"


def test_context_sampling_preserves_multitoken_route_budget() -> None:
    base = vf.SamplingConfig(
        temperature=0.7,
        max_tokens=64,
        extra_body={"cache_salt": "snapshot-0", "top_k": 20},
    )

    result = context_sampling(
        base,
        seed=303,
        cache_salt_suffix="episode:context",
        temperature=2.0,
    )

    assert result.temperature == 2.0
    assert result.max_tokens == 64
    assert result.model_dump()["seed"] == 303
    assert result.model_dump()["extra_body"] == {
        "cache_salt": "snapshot-0:redco:episode:context",
        "top_k": 20,
    }
    assert parse_route("<route>delta</route>") == "delta"


def test_constrain_root_routes_preserves_sampling_and_defines_four_choices() -> None:
    base = vf.SamplingConfig(
        temperature=2.0,
        max_tokens=64,
        seed=123,
        extra_body={"cache_salt": "snapshot-0", "top_k": 20},
    )

    result = constrain_root_routes(base)

    assert result.temperature == 2.0
    assert result.max_tokens == 64
    assert result.seed == 123
    assert result.extra_body == {
        "cache_salt": "snapshot-0",
        "top_k": 20,
        "structured_outputs": {
            "choice": [
                "<route>alpha</route>",
                "<route>beta</route>",
                "<route>gamma</route>",
                "<route>delta</route>",
            ]
        },
    }


def test_context_seed_is_stable_per_episode_and_distinct_across_episodes() -> None:
    first = _context_seed("confusion_redundant-0241", "0:episode:0")
    repeated = _context_seed("confusion_redundant-0241", "0:episode:0")
    second = _context_seed("confusion_redundant-0241", "0:episode:1")

    assert first == repeated
    assert first != second


def test_forced_smoke_covers_all_reward_regions_without_sampling() -> None:
    assert _forced_smoke_choices("0:episode:0") == (
        "<route>gamma</route>",
        "5",
    )
    assert _forced_smoke_choices("0:episode:1") == (
        "<route>delta</route>",
        "1",
    )
    assert _forced_smoke_choices("0:episode:2") == (
        "<route>gamma</route>",
        "0",
    )
    assert {
        _forced_smoke_choices(f"0:episode:{index}")[0]
        for index in range(8)
    } == {
        "<route>alpha</route>",
        "<route>beta</route>",
        "<route>gamma</route>",
        "<route>delta</route>",
    }
    assert {
        _forced_smoke_choices(f"0:episode:{index}")[1]
        for index in range(8)
    } == set("01234567")


def test_forced_smoke_sets_single_choice_on_root_and_target() -> None:
    env = RedcoCreditEnv(
        RedcoCreditEnvConfig(
            taskset={
                "id": "redco-credit-v1",
                "repeats_per_probe": 1,
                "probe_names": ["confusion_redundant"],
            },
            branching_enabled=False,
            forced_integration_smoke=True,
        )
    )
    task = env.taskset.load()[0]
    context = _MockAgent("<route>gamma</route>")
    original = _MockAgent("5")
    context.ctx = _MockContext(
        context.ctx.sampling.model_copy(
            update={"extra_body": {"cache_salt": "0:episode:0"}}
        )
    )
    agents = SimpleNamespace(context=context, original=original)

    asyncio.run(env.run(task, agents))

    assert context.ctx.sampling.extra_body["structured_outputs"] == {
        "choice": ["<route>gamma</route>"]
    }
    assert original.ctx.sampling.extra_body["structured_outputs"] == {
        "choice": ["5"]
    }


def test_constrained_root_rollout_uses_four_choices() -> None:
    env = RedcoCreditEnv(
        RedcoCreditEnvConfig(
            taskset={
                "id": "redco-credit-v1",
                "repeats_per_probe": 1,
                "probe_names": ["confusion_redundant"],
            },
            branching_enabled=False,
            constrained_root_routes=True,
        )
    )
    task = env.taskset.load()[0]
    context = _MockAgent("<route>gamma</route>")
    original = _MockAgent("5")
    context.ctx = _MockContext(
        context.ctx.sampling.model_copy(
            update={"extra_body": {"cache_salt": "0:episode:0"}}
        )
    )
    agents = SimpleNamespace(context=context, original=original)

    asyncio.run(env.run(task, agents))

    assert context.ctx.sampling.extra_body["structured_outputs"] == {
        "choice": [
            "<route>alpha</route>",
            "<route>beta</route>",
            "<route>gamma</route>",
            "<route>delta</route>",
        ]
    }


def test_mock_model_rollout_parses_root_and_emits_trainable_credit() -> None:
    env = RedcoCreditEnv(
        RedcoCreditEnvConfig(
            taskset={
                "id": "redco-credit-v1",
                "repeats_per_probe": 1,
                "probe_names": ["confusion_redundant"],
            },
            branch_group_size=4,
            context_temperature=2.0,
            branch_temperature=2.0,
        )
    )
    task = env.taskset.load()[0]
    context = _MockAgent("<route>beta</route>")
    original = _MockAgent("0")
    alternative_1 = _MockAgent("5")
    alternative_2 = _MockAgent("0")
    alternative_3 = _MockAgent("5")
    agents = SimpleNamespace(
        context=context,
        original=original,
        alternative_1=alternative_1,
        alternative_2=alternative_2,
        alternative_3=alternative_3,
    )

    asyncio.run(env.run(task, agents))

    assert context.ctx.sampling.max_tokens == 64
    assert len(context.tasks) == 1
    assert parse_route(context.traces[0].last_reply) == "beta"
    branch_agents = (original, alternative_1, alternative_2, alternative_3)
    assert all(agent.tasks[0].data.context_route == "beta" for agent in branch_agents)
    rewards = tuple(
        asyncio.run(agent.tasks[0].deterministic_reward(agent.traces[0]))
        for agent in branch_agents
    )
    advantages = trajectory_rloo(rewards)
    trainable_fraction = sum(value != 0.0 for value in advantages) / len(advantages)

    assert rewards == (0.0, 1.0, 0.0, 1.0)
    assert trainable_fraction > 0.0


def test_env_config_freezes_branch_count_and_replay_mode() -> None:
    config = RedcoCreditEnvConfig(
        taskset={"id": "redco-credit-v1"},
        replay_mode="full_suffix",
        branch_temperature=2.0,
    )

    assert config.branching_enabled is True
    assert config.replay_mode == "full_suffix"
    assert config.branch_temperature == 2.0


def test_env_config_rejects_unsupported_branch_temperature() -> None:
    try:
        RedcoCreditEnvConfig(
            taskset={"id": "redco-credit-v1"},
            branch_temperature=2.01,
        )
    except ValueError as error:
        assert "branch_temperature" in str(error)
        assert "less than or equal to 2" in str(error)
    else:
        raise AssertionError("unsupported vLLM temperature must fail locally")


def test_env_config_accepts_greedy_evaluation_temperature() -> None:
    config = RedcoCreditEnvConfig(
        taskset={"id": "redco-credit-v1"},
        branch_temperature=0.0,
    )

    assert config.branch_temperature == 0.0


def test_confusion_rewards_have_the_preregistered_causal_structure() -> None:
    for action in ("0", "5"):
        assert confusion_reward(
            "confusion_irrelevant",
            canonical_action=action,
            context_route="delta",
            episode_luck=1,
        ) == 0.5
    assert confusion_reward(
        "confusion_redundant",
        canonical_action="0",
        context_route="delta",
        episode_luck=-1,
    ) == 1.0
    assert confusion_reward(
        "confusion_redundant",
        canonical_action="5",
        context_route="alpha",
        episode_luck=1,
    ) == 1.0
    assert confusion_reward(
        "confusion_redundant",
        canonical_action="0",
        context_route="alpha",
        episode_luck=1,
    ) == 0.0
    assert confusion_reward(
        "confusion_lucky",
        canonical_action="5",
        context_route="gamma",
        episode_luck=1,
    ) == 2.25
    assert confusion_reward(
        "confusion_lucky",
        canonical_action="0",
        context_route="gamma",
        episode_luck=-1,
    ) == -0.75


def test_confusion_probe_subset_uses_the_same_octet_action_space() -> None:
    taskset = RedcoCreditTaskset(
        RedcoCreditTasksetConfig(
            repeats_per_probe=2,
            probe_names=(
                "confusion_irrelevant",
                "confusion_redundant",
                "confusion_lucky",
            ),
        )
    )

    tasks = taskset.load()

    assert len(tasks) == 6
    assert all(task.data.actions == tuple(str(index) for index in range(8)) for task in tasks)
