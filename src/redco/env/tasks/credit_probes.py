"""Finite synthetic probes whose Q-values are known by construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import fsum

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


def planted_needle(*, chunk_count: int, needle_chunk: int) -> FiniteCreditProbe:
    if chunk_count < 2:
        raise ValueError("chunk_count must be at least two")
    if needle_chunk not in range(chunk_count):
        raise ValueError("needle_chunk must identify an existing chunk")
    actions = tuple(str(index) for index in range(chunk_count))

    def reward(action: str, _: int) -> float:
        return float(int(action) == needle_chunk)

    return FiniteCreditProbe("planted_needle", actions, reward)


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

