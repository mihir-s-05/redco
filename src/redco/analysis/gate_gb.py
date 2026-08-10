"""Execute the CPU portion of Gate GB and persist auditable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from redco.analysis.replay_equivalence import run_randomized_equivalence
from redco.contracts import SeedNamespace, canonical_json
from redco.env.artifacts import ArtifactStore
from redco.env.commands import JsonValue
from redco.env.policy_cache import CachedPolicyAction, PolicyActionCache, PolicyCallKey
from redco.env.tracer import EdgeKind, EventEdge, EventGraph, EventNode, EventNodeKind


@dataclass(frozen=True, slots=True)
class GateGbCpuReport:
    schema_version: int
    generated_at_utc: str
    campaign_seed: int
    randomized_programs: int
    interventions: int
    deterministic_failures: int
    deterministic_bitwise_equal: bool
    dependency_edge_counts: dict[str, int]
    all_six_dependency_kinds_exercised: bool
    full_suffix_events: int
    sliced_suffix_events: int
    mean_sliced_work_fraction: float
    event_raf: float
    snapshot_roundtrip_exact: bool
    exact_key_cache_reuse: bool
    changed_prompt_regenerated: bool
    changed_seed_regenerated: bool
    structural_seed_stable: bool
    hand_audit_passed: bool
    stochastic_model_audit_status: str
    passed_cpu_gate: bool
    report_sha256: str = ""

    def unsigned_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("report_sha256")
        return payload

    def signed_dict(self) -> dict[str, object]:
        payload = self.unsigned_dict()
        payload["report_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
        return payload


def run_gate(
    *,
    seed: int,
    programs: int,
    interventions_per_program: int,
    events: int,
) -> GateGbCpuReport:
    randomized = run_randomized_equivalence(
        seed=seed,
        program_count=programs,
        interventions_per_program=interventions_per_program,
        events_per_program=events,
    )
    hand_audit_passed = _audit_six_edge_closure()
    snapshot_roundtrip_exact = _audit_snapshot_roundtrip()
    cache_reuse, prompt_regenerated, seed_regenerated = _audit_policy_cache()
    namespace = SeedNamespace("gate-gb", "rollout-0", "target-0", 1)
    structural_seed_stable = namespace.action_seed(1) == namespace.action_seed(1)
    all_six = all(count > 0 for count in randomized.dependency_edge_counts.values())
    full = randomized.full_suffix_events
    sliced = randomized.sliced_suffix_events
    event_raf = (full / sliced) if sliced else float("inf")
    passed = (
        randomized.passed
        and all_six
        and hand_audit_passed
        and snapshot_roundtrip_exact
        and cache_reuse
        and prompt_regenerated
        and seed_regenerated
        and structural_seed_stable
    )
    return GateGbCpuReport(
        schema_version=1,
        generated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        campaign_seed=seed,
        randomized_programs=programs,
        interventions=randomized.interventions,
        deterministic_failures=len(randomized.failures),
        deterministic_bitwise_equal=randomized.passed,
        dependency_edge_counts=randomized.dependency_edge_counts,
        all_six_dependency_kinds_exercised=all_six,
        full_suffix_events=full,
        sliced_suffix_events=sliced,
        mean_sliced_work_fraction=randomized.mean_sliced_work_fraction,
        event_raf=event_raf,
        snapshot_roundtrip_exact=snapshot_roundtrip_exact,
        exact_key_cache_reuse=cache_reuse,
        changed_prompt_regenerated=prompt_regenerated,
        changed_seed_regenerated=seed_regenerated,
        structural_seed_stable=structural_seed_stable,
        hand_audit_passed=hand_audit_passed,
        stochastic_model_audit_status="pending_external_model_audit",
        passed_cpu_gate=passed,
    )


def _audit_six_edge_closure() -> bool:
    graph = EventGraph()
    kinds = tuple(EdgeKind)
    for index in range(len(kinds) + 1):
        graph.add_node(EventNode(f"node-{index}", EventNodeKind.OPERATION))
    for index, kind in enumerate(kinds):
        graph.add_edge(EventEdge(f"node-{index}", f"node-{index + 1}", kind))
    expected = frozenset(f"node-{index}" for index in range(1, len(kinds) + 1))
    return graph.descendants("node-0") == expected


def _audit_snapshot_roundtrip() -> bool:
    with tempfile.TemporaryDirectory() as directory:
        store = ArtifactStore(Path(directory))
        original: dict[str, JsonValue] = {
            "nested": {"answer": 42, "valid": True},
            "sequence": [1, "two", None],
        }
        state_ref = store.put_json(original)
        return store.get_bytes(state_ref) == canonical_json(original)


def _write_gate_report(root: Path, name: str, payload: object) -> Path:
    resolved_root = root.resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    data = canonical_json(payload) + b"\n"
    if not name or Path(name).name != name:
        raise ValueError("trace file name must be a single path component")
    path = resolved_root / name
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return path


def _audit_policy_cache() -> tuple[bool, bool, bool]:
    cache = PolicyActionCache()
    original_prompt = (1, 2, 3)
    original_key = PolicyCallKey.from_call(
        original_prompt,
        checkpoint_id="theta-gb",
        decoding_config_hash="decode-greedy",
        event_seed=7,
    )
    cache.record(CachedPolicyAction(original_key, (9, 10)))
    sampler_calls: list[tuple[tuple[int, ...], int]] = []

    def sampler(prompt: tuple[int, ...], seed: int) -> tuple[int, ...]:
        sampler_calls.append((prompt, seed))
        return (seed % 101, len(prompt))

    reused = cache.resolve(
        original_prompt,
        checkpoint_id="theta-gb",
        decoding_config_hash="decode-greedy",
        event_seed=7,
        sampler=sampler,
    )
    exact_reuse_without_sampling = reused.reused and not sampler_calls
    changed_prompt = cache.resolve(
        (1, 2, 4),
        checkpoint_id="theta-gb",
        decoding_config_hash="decode-greedy",
        event_seed=7,
        sampler=sampler,
    )
    changed_seed = cache.resolve(
        original_prompt,
        checkpoint_id="theta-gb",
        decoding_config_hash="decode-greedy",
        event_seed=8,
        sampler=sampler,
    )
    return (
        exact_reuse_without_sampling,
        not changed_prompt.reused and ((1, 2, 4), 7) in sampler_calls,
        not changed_seed.reused and (original_prompt, 8) in sampler_calls,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--programs", type=int, default=500)
    parser.add_argument("--interventions", type=int, default=20)
    parser.add_argument("--events", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/stage-b/gate-gb-cpu/report.json"),
    )
    args = parser.parse_args()
    report = run_gate(
        seed=args.seed,
        programs=args.programs,
        interventions_per_program=args.interventions,
        events=args.events,
    )
    path = _write_gate_report(
        args.output.parent,
        args.output.name,
        report.signed_dict(),
    )
    print(json.dumps(report.signed_dict(), indent=2, sort_keys=True))
    print(f"wrote {path}")
    return 0 if report.passed_cpu_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
