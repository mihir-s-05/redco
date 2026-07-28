"""Finite synthetic probes whose Q-values are known by construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import fsum
from typing import Literal

RewardFunction = Callable[[str, int], float]


@dataclass(frozen=True, slots=True)
class FiniteCreditProbe:
    name: str
    actions: tuple[str, ...]
    reward_function: RewardFunction

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("probe name must be non-empty")
        if len(self.actions) < 2 or len(set(self.actions)) != len(self.actions):
            raise ValueError("probe actions must contain at least two unique values")

    def q_values(self, exogenous_seeds: Iterable[int]) -> dict[str, float]:
        seeds = tuple(exogenous_seeds)
        if not seeds:
            raise ValueError("at least one exogenous seed is required")
        return {
            action: fsum(self.reward_function(action, seed) for seed in seeds) / len(seeds)
            for action in self.actions
        }

    def advantages(self, exogenous_seeds: Iterable[int]) -> dict[str, float]:
        q_values = self.q_values(exogenous_seeds)
        value = fsum(q_values.values()) / len(q_values)
        return {action: q_value - value for action, q_value in q_values.items()}

    def replay_reward(
        self,
        action: str | None,
        exogenous_seed: int,
        *,
        mode: Literal["full_suffix", "sliced"],
    ) -> float:
        """Execute one depth-one intervention under the Stage-C replay contract.

        These restricted probes have exactly one action-dependent descendant:
        the deterministic reward node. Full-suffix replay visits that complete
        suffix while sliced replay visits its dynamic descendant closure; the two
        sets coincide here. Invalid actions execute as failures with reward zero.
        """
        if mode not in {"full_suffix", "sliced"}:
            raise ValueError(f"unsupported replay mode: {mode}")
        if action not in self.actions:
            return 0.0
        return float(self.reward_function(action, exogenous_seed))


def credit_probe_by_name(name: str) -> FiniteCreditProbe:
    try:
        probes = (
            *standard_credit_probes(),
            integration_planted_needle(),
            *credit_confusion_probes(),
        )
        return {probe.name: probe for probe in probes}[name]
    except KeyError as error:
        raise ValueError(f"unknown credit probe: {name}") from error


def planted_needle(*, chunk_count: int, needle_chunk: int) -> FiniteCreditProbe:
    if chunk_count < 2:
        raise ValueError("chunk_count must be at least two")
    if needle_chunk not in range(chunk_count):
        raise ValueError("needle_chunk must identify an existing chunk")
    actions = tuple(str(index) for index in range(chunk_count))

    def reward(action: str, _: int) -> float:
        return float(int(action) == needle_chunk)

    return FiniteCreditProbe("planted_needle", actions, reward)


def integration_planted_needle() -> FiniteCreditProbe:
    """Return the signal-rich, non-learning fixture used by the live smoke."""
    probe = planted_needle(chunk_count=8, needle_chunk=1)
    return FiniteCreditProbe(
        "integration_planted_needle",
        probe.actions,
        probe.reward_function,
    )


def credit_confusion_probes() -> tuple[FiniteCreditProbe, ...]:
    """Return octet action spaces for the live multi-decision battery.

    Their complete rewards also depend on the already-sampled root context
    and, for the lucky probe, episode-level exogenous noise.  The environment
    composes those terms.  These local reward functions retain the target
    action's causal component so the action map remains explicit and finite.
    """
    actions = tuple(str(index) for index in range(8))

    def irrelevant(_: str, __: int) -> float:
        return 0.0

    def causal(action: str, _: int) -> float:
        return float(action == "5")

    return (
        FiniteCreditProbe("confusion_irrelevant", actions, irrelevant),
        FiniteCreditProbe("confusion_redundant", actions, causal),
        FiniteCreditProbe("confusion_lucky", actions, causal),
    )


def redundancy() -> FiniteCreditProbe:
    actions = ("none", "left", "right", "both")

    def reward(action: str, _: int) -> float:
        return float(action != "none")

    return FiniteCreditProbe("redundancy", actions, reward)


def spurious_correlation() -> FiniteCreditProbe:
    actions = ("spurious_absent", "spurious_present")

    def reward(_: str, exogenous_seed: int) -> float:
        return float(exogenous_seed % 2 == 0)

    return FiniteCreditProbe("spurious_correlation", actions, reward)


def control_flow_trap() -> FiniteCreditProbe:
    """Only the action that opens the planted branch can reach reward."""
    actions = ("skip_branch", "open_branch")

    def reward(action: str, exogenous_seed: int) -> float:
        planted_branch_is_live = exogenous_seed % 3 != 0
        return float(action == "open_branch" and planted_branch_is_live)

    return FiniteCreditProbe("control_flow_trap", actions, reward)


def aliasing_trap() -> FiniteCreditProbe:
    """Reward immutable copying while a hidden-alias mutation corrupts evidence."""
    actions = ("mutate_alias", "copy_then_mutate")

    def reward(action: str, _: int) -> float:
        return float(action == "copy_then_mutate")

    return FiniteCreditProbe("aliasing_trap", actions, reward)


def observation_trap() -> FiniteCreditProbe:
    """Changing a prompt-visible artifact must change the downstream policy state."""
    actions = ("reuse_stale_prompt", "render_changed_prompt")

    def reward(action: str, _: int) -> float:
        return float(action == "render_changed_prompt")

    return FiniteCreditProbe("observation_trap", actions, reward)


def side_effect_ordering_trap() -> FiniteCreditProbe:
    """The write must be committed before the dependent read."""
    actions = ("read_before_write", "write_before_read")

    def reward(action: str, _: int) -> float:
        return float(action == "write_before_read")

    return FiniteCreditProbe("side_effect_ordering_trap", actions, reward)


def resource_dependency_trap() -> FiniteCreditProbe:
    """A declared resource version is part of the replay dependency closure."""
    actions = ("stale_resource", "versioned_resource")

    def reward(action: str, exogenous_seed: int) -> float:
        resource_available = exogenous_seed % 5 != 0
        return float(action == "versioned_resource" and resource_available)

    return FiniteCreditProbe("resource_dependency_trap", actions, reward)


def standard_credit_probes() -> tuple[FiniteCreditProbe, ...]:
    """Return the fixed CPU probe suite used before any model training."""
    return (
        planted_needle(chunk_count=8, needle_chunk=5),
        redundancy(),
        spurious_correlation(),
        control_flow_trap(),
        aliasing_trap(),
        observation_trap(),
        side_effect_ordering_trap(),
        resource_dependency_trap(),
    )
