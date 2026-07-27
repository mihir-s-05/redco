"""Seeded stochastic-reward equivalence over dynamic full and sliced replay."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from redco.analysis.rlm_raf import _replacement_action, _rlm_program
from redco.contracts import canonical_json
from redco.env.dynamic_replay import DynamicReplayEngine
from redco.env.replay import ReplayMode


@dataclass(frozen=True, slots=True)
class EquivalenceInterval:
    samples_per_arm: int
    confidence: float
    margin: float
    full_mean: float
    sliced_mean: float
    mean_difference: float
    half_width: float
    lower: float
    upper: float
    passed: bool


@dataclass(frozen=True, slots=True)
class StochasticReplayReport:
    schema_version: int
    generated_at_utc: str
    master_seed: str
    program_seed: int
    programs: int
    alternatives_per_program: int
    paired_branches: int
    topology_divergences: int
    deterministic_state_mismatches: int
    full_positive_rewards: int
    sliced_positive_rewards: int
    overall_equivalence: EquivalenceInterval
    direct_route_equivalence: EquivalenceInterval
    recursive_route_equivalence: EquivalenceInterval
    passed_stochastic_gate: bool
    report_sha256: str = ""

    def unsigned_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("report_sha256")
        return payload

    def signed_dict(self) -> dict[str, object]:
        payload = self.unsigned_dict()
        payload["report_sha256"] = hashlib.sha256(
            canonical_json(payload)
        ).hexdigest()
        return payload


def _equivalence_interval(
    full: list[int],
    sliced: list[int],
    *,
    confidence: float,
    margin: float,
) -> EquivalenceInterval:
    if not full or len(full) != len(sliced):
        raise ValueError("reward arms must have the same positive sample count")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if not 0 < margin < 1:
        raise ValueError("margin must be in (0, 1)")
    if any(value not in {0, 1} for value in [*full, *sliced]):
        raise ValueError("rewards must be Bernoulli values")
    count = len(full)
    alpha = 1.0 - confidence
    # Union-bound two Hoeffding intervals, one for each Bernoulli arm.
    arm_half_width = math.sqrt(math.log(4.0 / alpha) / (2.0 * count))
    half_width = 2.0 * arm_half_width
    full_mean = math.fsum(full) / count
    sliced_mean = math.fsum(sliced) / count
    difference = full_mean - sliced_mean
    lower = difference - half_width
    upper = difference + half_width
    return EquivalenceInterval(
        samples_per_arm=count,
        confidence=confidence,
        margin=margin,
        full_mean=full_mean,
        sliced_mean=sliced_mean,
        mean_difference=difference,
        half_width=half_width,
        lower=lower,
        upper=upper,
        passed=lower > -margin and upper < margin,
    )


def _reward_probability(terminal: object) -> float:
    digest = hashlib.sha256(canonical_json(terminal)).digest()
    unit = int.from_bytes(digest[:8], "big") / ((1 << 64) - 1)
    return 0.2 + 0.6 * unit


def _bernoulli_reward(
    *,
    master_seed: str,
    program_index: int,
    alternative_index: int,
    arm: str,
    probability: float,
) -> int:
    payload = canonical_json(
        {
            "program_index": program_index,
            "alternative_index": alternative_index,
            "arm": arm,
            "purpose": "stochastic_terminal_reward",
        }
    )
    digest = hmac.new(
        master_seed.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / ((1 << 64) - 1)
    return int(unit < probability)


def run_stochastic_replay_equivalence(
    *,
    master_seed: str,
    program_seed: int,
    programs: int,
    alternatives_per_program: int,
    confidence: float,
    overall_margin: float,
    route_margin: float,
) -> StochasticReplayReport:
    if not master_seed:
        raise ValueError("master_seed must be non-empty")
    if programs < 2 or alternatives_per_program < 1:
        raise ValueError("campaign sizes must be positive and need two programs")
    rng = random.Random(program_seed)
    full_rewards: list[int] = []
    sliced_rewards: list[int] = []
    rewards_by_route: dict[str, tuple[list[int], list[int]]] = {
        "direct": ([], []),
        "recurse": ([], []),
    }
    topology_divergences = 0
    deterministic_mismatches = 0

    for program_index in range(programs):
        # Balance the two strata by construction so each frozen campaign has
        # enough observations for both preregistered route-level intervals.
        original_route = ("direct", "recurse")[program_index % 2]
        child_calls = rng.randint(2, 6)
        program = _rlm_program(
            rng,
            program_index=program_index,
            original_route=original_route,
            child_calls=child_calls,
        )
        engine = DynamicReplayEngine(program, sampler=lambda *_args: (1,))
        route_full, route_sliced = rewards_by_route[original_route]
        for alternative_index in range(alternatives_per_program):
            _, replacement = _replacement_action(
                original_route,
                alternative_index,
            )
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
            if full.state_bytes != sliced.state_bytes:
                deterministic_mismatches += 1
            if sliced.topology.diverged:
                topology_divergences += 1
            probability = _reward_probability(
                full.terminal(program.terminal_output)
            )
            full_reward = _bernoulli_reward(
                master_seed=master_seed,
                program_index=program_index,
                alternative_index=alternative_index,
                arm="full_suffix",
                probability=probability,
            )
            sliced_reward = _bernoulli_reward(
                master_seed=master_seed,
                program_index=program_index,
                alternative_index=alternative_index,
                arm="sliced",
                probability=probability,
            )
            full_rewards.append(full_reward)
            sliced_rewards.append(sliced_reward)
            route_full.append(full_reward)
            route_sliced.append(sliced_reward)

    overall = _equivalence_interval(
        full_rewards,
        sliced_rewards,
        confidence=confidence,
        margin=overall_margin,
    )
    direct = _equivalence_interval(
        *rewards_by_route["direct"],
        confidence=confidence,
        margin=route_margin,
    )
    recursive = _equivalence_interval(
        *rewards_by_route["recurse"],
        confidence=confidence,
        margin=route_margin,
    )
    passed = (
        deterministic_mismatches == 0
        and topology_divergences > 0
        and overall.passed
        and direct.passed
        and recursive.passed
    )
    return StochasticReplayReport(
        schema_version=1,
        generated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        master_seed=master_seed,
        program_seed=program_seed,
        programs=programs,
        alternatives_per_program=alternatives_per_program,
        paired_branches=len(full_rewards),
        topology_divergences=topology_divergences,
        deterministic_state_mismatches=deterministic_mismatches,
        full_positive_rewards=sum(full_rewards),
        sliced_positive_rewards=sum(sliced_rewards),
        overall_equivalence=overall,
        direct_route_equivalence=direct,
        recursive_route_equivalence=recursive,
        passed_stochastic_gate=passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master-seed", required=True)
    parser.add_argument("--program-seed", type=int, required=True)
    parser.add_argument("--programs", type=int, required=True)
    parser.add_argument("--alternatives-per-program", type=int, required=True)
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--overall-margin", type=float, required=True)
    parser.add_argument("--route-margin", type=float, required=True)
    args = parser.parse_args()
    report = run_stochastic_replay_equivalence(
        master_seed=args.master_seed,
        program_seed=args.program_seed,
        programs=args.programs,
        alternatives_per_program=args.alternatives_per_program,
        confidence=args.confidence,
        overall_margin=args.overall_margin,
        route_margin=args.route_margin,
    )
    payload = report.signed_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(payload) + b"\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report.passed_stochastic_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
