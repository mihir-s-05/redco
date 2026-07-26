"""Project replay work from exact policy dependencies in a recorded RLM trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from redco.contracts import PolicyNodeKind, canonical_json
from redco.env.tracer import EventNode, EventNodeKind
from redco.integrations.verifiers_provenance import import_trace_file


@dataclass(frozen=True, slots=True)
class RecordedTargetProjection:
    trace_id: str
    target_node_id: str
    target_call_index: int
    target_depth: int
    full_suffix_policy_events: int
    sliced_affected_policy_events: int
    exact_key_reusable_policy_events: int
    conservative_no_cache_full_prompt_tokens: int
    affected_prompt_tokens: int
    conservative_no_cache_full_generated_tokens: int
    affected_generated_tokens: int
    conservative_no_cache_generated_token_work_fraction: float
    conservative_no_cache_total_token_work_fraction: float
    modeled_no_cache_full_policy_token_raf: float
    modeled_exact_key_full_policy_token_raf: float
    modeled_sliced_policy_token_raf: float


@dataclass(frozen=True, slots=True)
class RecordedRafProjection:
    schema_version: int
    generated_at_utc: str
    source_sha256: str
    alternatives_per_target: int
    trace_count: int
    target_count: int
    exact_prompt_provenance_coverage: float
    structural_model_call_coverage: float
    exact_recursive_parent_coverage: float
    cross_component_fallbacks: int
    mean_conservative_no_cache_generated_token_work_fraction: float
    minimum_conservative_no_cache_generated_token_work_fraction: float
    requires_broader_trace: bool
    empirical_branch_replay_status: str
    targets: tuple[RecordedTargetProjection, ...]
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


def build_recorded_raf_projection(
    path: Path,
    *,
    alternatives_per_target: int = 3,
) -> RecordedRafProjection:
    """Compute a preregistration aid, not an empirical replay measurement."""
    if alternatives_per_target < 1:
        raise ValueError("alternatives_per_target must be positive")
    imported = import_trace_file(path)
    if not imported.ready_for_representative_raf:
        raise ValueError("trace must have exact prompt and recursive provenance")

    targets: list[RecordedTargetProjection] = []
    for trace in imported.traces:
        graph = trace.graph
        policy_nodes = [
            node
            for node in graph.nodes.values()
            if node.kind is EventNodeKind.POLICY
        ]
        baseline_generated_tokens = sum(
            _metric(node, "completion_tokens") for node in policy_nodes
        )
        for target in policy_nodes:
            if (
                target.metadata.get("policy_kind")
                != PolicyNodeKind.SUBCALL_OUTPUT.value
                or target.metadata.get("call_kind") != "policy"
            ):
                continue
            target_index = _required_integer(target, "call_index")
            target_depth = _required_integer(target, "agent_depth")
            target_action_tokens = _metric(target, "completion_tokens")
            full = [
                node
                for node in policy_nodes
                if _required_integer(node, "call_index") > target_index
            ]
            descendants = graph.descendants(target.node_id)
            sliced = [node for node in full if node.node_id in descendants]
            full_prompt = sum(_metric(node, "prompt_tokens") for node in full)
            sliced_prompt = sum(_metric(node, "prompt_tokens") for node in sliced)
            full_generated = sum(
                _metric(node, "completion_tokens") for node in full
            )
            sliced_generated = sum(
                _metric(node, "completion_tokens") for node in sliced
            )
            full_total = full_prompt + full_generated
            sliced_total = sliced_prompt + sliced_generated
            no_cache_full_raf = _modeled_raf(
                baseline_generated_tokens,
                alternatives_per_target,
                target_action_tokens,
                full_generated,
            )
            exact_key_raf = _modeled_raf(
                baseline_generated_tokens,
                alternatives_per_target,
                target_action_tokens,
                sliced_generated,
            )
            targets.append(
                RecordedTargetProjection(
                    trace_id=trace.trace_id,
                    target_node_id=target.node_id,
                    target_call_index=target_index,
                    target_depth=target_depth,
                    full_suffix_policy_events=len(full),
                    sliced_affected_policy_events=len(sliced),
                    exact_key_reusable_policy_events=len(full) - len(sliced),
                    conservative_no_cache_full_prompt_tokens=full_prompt,
                    affected_prompt_tokens=sliced_prompt,
                    conservative_no_cache_full_generated_tokens=(
                        full_generated
                    ),
                    affected_generated_tokens=sliced_generated,
                    conservative_no_cache_generated_token_work_fraction=_fraction(
                        sliced_generated,
                        full_generated,
                    ),
                    conservative_no_cache_total_token_work_fraction=_fraction(
                        sliced_total,
                        full_total,
                    ),
                    modeled_no_cache_full_policy_token_raf=no_cache_full_raf,
                    modeled_exact_key_full_policy_token_raf=exact_key_raf,
                    modeled_sliced_policy_token_raf=exact_key_raf,
                )
            )

    generated_fractions = [
        target.conservative_no_cache_generated_token_work_fraction
        for target in targets
    ]
    mean_fraction = (
        sum(generated_fractions) / len(generated_fractions)
        if generated_fractions
        else 0.0
    )
    minimum_fraction = min(generated_fractions, default=0.0)
    requires_broader = (
        len(targets) < 2
        or not generated_fractions
        or minimum_fraction >= 0.9
    )
    return RecordedRafProjection(
        schema_version=1,
        generated_at_utc=time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        source_sha256=imported.source_sha256,
        alternatives_per_target=alternatives_per_target,
        trace_count=len(imported.traces),
        target_count=len(targets),
        exact_prompt_provenance_coverage=(
            imported.exact_prompt_provenance_coverage
        ),
        structural_model_call_coverage=(
            imported.structural_model_call_coverage
        ),
        exact_recursive_parent_coverage=(
            imported.exact_recursive_parent_coverage
        ),
        cross_component_fallbacks=imported.cross_component_fallbacks,
        mean_conservative_no_cache_generated_token_work_fraction=(
            mean_fraction
        ),
        minimum_conservative_no_cache_generated_token_work_fraction=(
            minimum_fraction
        ),
        requires_broader_trace=requires_broader,
        empirical_branch_replay_status=(
            "not_run_projection_only"
        ),
        targets=tuple(targets),
    )


def _metric(node: EventNode, name: str) -> int:
    value = node.metadata.get(name)
    return value if type(value) is int and value >= 0 else 0


def _required_integer(node: EventNode, name: str) -> int:
    value = node.metadata.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"{node.node_id} lacks nonnegative {name}")
    return value


def _fraction(numerator: int, denominator: int) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


def _modeled_raf(
    baseline_generated_tokens: int,
    alternatives: int,
    target_action_tokens: int,
    downstream_generated_tokens: int,
) -> float:
    if not baseline_generated_tokens:
        return float("inf")
    return (
        baseline_generated_tokens
        + alternatives * (target_action_tokens + downstream_generated_tokens)
    ) / baseline_generated_tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alternatives-per-target", type=int, default=3)
    parser.add_argument(
        "--require-broader-trace",
        action="store_true",
        help="Exit nonzero when the trace lacks multiple targets or a savings candidate.",
    )
    args = parser.parse_args()
    report = build_recorded_raf_projection(
        args.input,
        alternatives_per_target=args.alternatives_per_target,
    )
    payload = report.signed_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(payload) + b"\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return int(args.require_broader_trace and report.requires_broader_trace)


if __name__ == "__main__":
    raise SystemExit(main())
