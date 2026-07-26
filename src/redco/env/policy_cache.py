"""Exact-key policy-action reuse and branch-topology divergence records."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

from redco.contracts import canonical_json

Sampler = Callable[[tuple[int, ...], int], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class PolicyCallKey:
    prompt_token_ids_hash: str
    checkpoint_id: str
    decoding_config_hash: str
    event_seed: int

    def __post_init__(self) -> None:
        if (
            not self.prompt_token_ids_hash
            or not self.checkpoint_id
            or not self.decoding_config_hash
        ):
            raise ValueError("policy call key fields must be non-empty")
        if self.event_seed < 0:
            raise ValueError("event_seed must be non-negative")

    @classmethod
    def from_call(
        cls,
        prompt_token_ids: tuple[int, ...],
        *,
        checkpoint_id: str,
        decoding_config_hash: str,
        event_seed: int,
    ) -> PolicyCallKey:
        prompt_hash = hashlib.sha256(canonical_json(prompt_token_ids)).hexdigest()
        return cls(prompt_hash, checkpoint_id, decoding_config_hash, event_seed)


@dataclass(frozen=True, slots=True)
class CachedPolicyAction:
    key: PolicyCallKey
    action_token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CachedActionDecision:
    key: PolicyCallKey
    action_token_ids: tuple[int, ...]
    reused: bool
    reason: str


@dataclass(frozen=True, slots=True)
class TopologyDivergence:
    original_node_ids: tuple[str, ...]
    branch_node_ids: tuple[str, ...]

    @property
    def added_node_ids(self) -> tuple[str, ...]:
        original = set(self.original_node_ids)
        return tuple(node for node in self.branch_node_ids if node not in original)

    @property
    def removed_node_ids(self) -> tuple[str, ...]:
        branch = set(self.branch_node_ids)
        return tuple(node for node in self.original_node_ids if node not in branch)

    @property
    def topology_delta(self) -> int:
        return len(self.branch_node_ids) - len(self.original_node_ids)

    @property
    def diverged(self) -> bool:
        return self.original_node_ids != self.branch_node_ids


class PolicyActionCache:
    """Reuse only when prompt, checkpoint, decoding config, and seed all match."""

    def __init__(self) -> None:
        self._actions: dict[PolicyCallKey, tuple[int, ...]] = {}

    def fork(self) -> PolicyActionCache:
        """Copy the frozen action table for an isolated replay branch."""
        branch = PolicyActionCache()
        branch._actions = dict(self._actions)
        return branch

    def record(self, action: CachedPolicyAction) -> None:
        existing = self._actions.get(action.key)
        if existing is not None and existing != action.action_token_ids:
            raise RuntimeError("conflicting action for exact policy call key")
        self._actions[action.key] = action.action_token_ids

    def resolve(
        self,
        prompt_token_ids: tuple[int, ...],
        *,
        checkpoint_id: str,
        decoding_config_hash: str,
        event_seed: int,
        sampler: Sampler,
    ) -> CachedActionDecision:
        key = PolicyCallKey.from_call(
            prompt_token_ids,
            checkpoint_id=checkpoint_id,
            decoding_config_hash=decoding_config_hash,
            event_seed=event_seed,
        )
        cached = self._actions.get(key)
        if cached is not None:
            return CachedActionDecision(key, cached, True, "exact_key_match")
        sampled = sampler(prompt_token_ids, event_seed)
        self.record(CachedPolicyAction(key, sampled))
        return CachedActionDecision(key, sampled, False, "key_changed_or_missing")
