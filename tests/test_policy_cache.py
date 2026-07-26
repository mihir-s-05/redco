from __future__ import annotations

from redco.env.policy_cache import (
    CachedPolicyAction,
    PolicyActionCache,
    PolicyCallKey,
    TopologyDivergence,
)


def test_policy_action_reuse_requires_the_exact_key() -> None:
    cache = PolicyActionCache()
    prompt = (1, 2, 3)
    key = PolicyCallKey.from_call(
        prompt,
        checkpoint_id="theta-0",
        decoding_config_hash="decode-0",
        event_seed=11,
    )
    cache.record(CachedPolicyAction(key, (4, 5)))
    calls: list[tuple[tuple[int, ...], int]] = []

    def sampler(tokens: tuple[int, ...], seed: int) -> tuple[int, ...]:
        calls.append((tokens, seed))
        return (seed,)

    exact = cache.resolve(
        prompt,
        checkpoint_id="theta-0",
        decoding_config_hash="decode-0",
        event_seed=11,
        sampler=sampler,
    )
    changed = cache.resolve(
        (1, 2, 9),
        checkpoint_id="theta-0",
        decoding_config_hash="decode-0",
        event_seed=11,
        sampler=sampler,
    )

    assert exact.reused
    assert not changed.reused
    assert calls == [((1, 2, 9), 11)]


def test_topology_divergence_records_added_and_removed_nodes() -> None:
    divergence = TopologyDivergence(
        original_node_ids=("root", "call-1", "call-2"),
        branch_node_ids=("root", "call-1", "call-3", "call-4"),
    )

    assert divergence.added_node_ids == ("call-3", "call-4")
    assert divergence.removed_node_ids == ("call-2",)
    assert divergence.topology_delta == 1
