"""Empirical full-suffix versus sliced replay on an exactly recorded RLM trace."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from redco.contracts import (
    ActualEvaluationCost,
    EventAddress,
    SeedNamespace,
    canonical_json,
)
from redco.env.policy_cache import (
    CachedPolicyAction,
    PolicyActionCache,
    PolicyCallKey,
)
from redco.env.replay import ReplayMode
from redco.env.tracer import EventNodeKind
from redco.integrations.verifiers_provenance import import_trace_file
from redco.integrations.verifiers_trace import (
    RecordedPolicyCall,
    audit_trace_file,
    build_policy_cache,
    load_trace_records,
    path_to_node,
)


@dataclass(frozen=True, slots=True)
class GeneratedAction:
    token_ids: tuple[int, ...]
    prompt_tokens: int
    generated_tokens: int
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class ReplayArmResult:
    mode: str
    visited_call_indices: tuple[int, ...]
    exact_key_reused_call_indices: tuple[int, ...]
    terminal_action_sha256: str
    reward: float
    actual_cost: ActualEvaluationCost


@dataclass(frozen=True, slots=True)
class EmpiricalBranchPair:
    target_node_id: str
    target_call_index: int
    target_agent_depth: int
    alternative_index: int
    action_seed: int
    continuation_seed: int
    candidate_action_sha256: str
    branch_prompt_sha256: str
    candidate_action_generation: GeneratedAction
    downstream_generation: GeneratedAction
    full_suffix: ReplayArmResult
    sliced: ReplayArmResult
    terminal_artifacts_exact: bool
    rewards_exact: bool
    cached_actions_exact: bool


@dataclass(frozen=True, slots=True)
class EmpiricalOriginalArm:
    target_node_id: str
    target_call_index: int
    target_agent_depth: int
    continuation_seed: int
    prompt_sha256: str
    downstream_generation: GeneratedAction
    full_suffix: ReplayArmResult
    sliced: ReplayArmResult
    terminal_artifacts_exact: bool
    rewards_exact: bool
    cached_actions_exact: bool


@dataclass(frozen=True, slots=True)
class ReproducibilityAudit:
    seed: int
    first_action_sha256: str
    second_action_sha256: str
    exact: bool
    first_generation: GeneratedAction
    second_generation: GeneratedAction


@dataclass(frozen=True, slots=True)
class LosslessRenderBoundary:
    recorded_static_prefix_tokens: int
    canonical_suffix_start_tokens: int
    exact_common_suffix_tokens: int


@dataclass(frozen=True, slots=True)
class EmpiricalTargetMetrics:
    target_node_id: str
    target_call_index: int
    target_agent_depth: int
    paired_branches: int
    alternative_action_generated_tokens: int
    downstream_generated_tokens: int
    generation_prompt_tokens: int
    model_request_wall_seconds: float
    full_suffix_policy_events_visited: int
    sliced_policy_events_visited: int
    sliced_policy_event_fraction: float
    full_arm_cost: ActualEvaluationCost
    sliced_arm_cost: ActualEvaluationCost


@dataclass(frozen=True, slots=True)
class EmpiricalReplayReport:
    schema_version: int
    generated_at_utc: str
    source_sha256: str
    trace_id: str
    checkpoint_id: str
    branch_decoding_config_hash: str
    alternatives_per_target: int
    target_count: int
    paired_branches: int
    canonical_renderer_prompt_exact: bool
    lossless_hybrid_preflight_exact: bool
    render_boundary: LosslessRenderBoundary
    distinct_candidate_actions_per_target: bool
    minimum_distinct_candidate_fraction: float
    distinct_candidate_fraction_by_target: dict[str, float]
    distinct_candidate_gate_passed: bool
    deterministic_terminal_mismatches: int
    reward_mismatches: int
    cached_action_mismatches: int
    full_suffix_policy_events_visited: int
    sliced_policy_events_visited: int
    sliced_policy_event_fraction: float
    alternative_action_generated_tokens: int
    downstream_generated_tokens: int
    generation_prompt_tokens: int
    model_request_wall_seconds: float
    exclusive_gpu_service_wall_seconds_proxy: float
    full_arm_cost: ActualEvaluationCost
    sliced_arm_cost: ActualEvaluationCost
    baseline_generated_tokens: int
    empirical_full_policy_token_raf: float
    empirical_sliced_policy_token_raf: float
    same_prompt_same_seed_reproducibility: ReproducibilityAudit
    target_metrics: tuple[EmpiricalTargetMetrics, ...]
    regenerated_originals: tuple[EmpiricalOriginalArm, ...]
    pairs: tuple[EmpiricalBranchPair, ...]
    passed_representative_micro_gate: bool
    gate_gb_cleared: bool
    limitations: tuple[str, ...]
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


class TokenInferenceClient:
    """Small dependency-free client for pinned vLLM token endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def tokenize(self, prompt: str) -> tuple[int, ...]:
        payload = self._post(
            "/tokenize",
            {
                "model": self.model,
                "prompt": prompt,
                "add_special_tokens": False,
            },
        )
        return _integer_tuple(payload.get("tokens"), "tokenize.tokens")

    def detokenize(self, token_ids: tuple[int, ...]) -> str:
        payload = self._post(
            "/detokenize",
            {"model": self.model, "tokens": list(token_ids)},
        )
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise TypeError("detokenize.prompt must be a string")
        return prompt

    def render_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
    ) -> tuple[int, ...]:
        payload = self._post(
            "/v1/chat/completions/render",
            {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "parallel_tool_calls": False,
                "seed": seed,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        return _integer_tuple(payload.get("token_ids"), "render.token_ids")

    def generate(
        self,
        prompt_token_ids: tuple[int, ...],
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
    ) -> GeneratedAction:
        started = time.perf_counter()
        payload = self._post(
            "/inference/v1/generate",
            {
                "model": self.model,
                "token_ids": list(prompt_token_ids),
                "sampling_params": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "seed": seed,
                },
            },
        )
        elapsed = time.perf_counter() - started
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("generate response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise TypeError("generate choice must be an object")
        token_ids = _integer_tuple(choice.get("token_ids"), "choice.token_ids")
        if not token_ids:
            raise ValueError("generate returned an empty action")
        usage = payload.get("usage")
        usage_payload = usage if isinstance(usage, dict) else {}
        prompt_tokens = _nonnegative_integer(
            usage_payload.get("prompt_tokens"),
            fallback=len(prompt_token_ids),
        )
        generated_tokens = _nonnegative_integer(
            usage_payload.get("completion_tokens"),
            fallback=len(token_ids),
        )
        return GeneratedAction(
            token_ids=token_ids,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            wall_seconds=elapsed,
        )

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, Any]:
        body = _request_json(payload)
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={
                "Authorization": "Bearer EMPTY",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                response_body = response.read()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise InferenceTransportError(
                f"POST {path} failed with HTTP {exc.code}: {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise InferenceTransportError(
                f"POST {path} failed before receiving a response: {exc}"
            ) from exc
        decoded = json.loads(response_body)
        if not isinstance(decoded, dict):
            raise TypeError(f"{path} response must be an object")
        return decoded


class InferenceTransportError(RuntimeError):
    """An outcome-independent inference-service or network failure."""


class DeterministicReplayIneligibility(ValueError):
    """A declared fixed-topology or trace-eligibility failure."""


def _request_json(payload: dict[str, object]) -> bytes:
    """Serialize an API request without reordering prompt-bearing objects."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def build_replay_indices(
    *,
    target_call_index: int,
    target_node_id: str,
    policy_node_ids_by_call: dict[int, str],
    descendants: frozenset[str],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return full and sliced suffix policy-call indices."""
    full = tuple(
        call_index
        for call_index in sorted(policy_node_ids_by_call)
        if call_index > target_call_index
    )
    sliced = tuple(
        call_index
        for call_index in full
        if policy_node_ids_by_call[call_index] in descendants
    )
    if not sliced:
        raise DeterministicReplayIneligibility(
            "target has no affected suffix policy calls"
        )
    return full, sliced


def derive_branch_group_seeds(
    *,
    master_seed: str,
    rollout_id: str,
    target_node_id: str,
    final_turn_index: int,
    alternatives: int,
) -> tuple[int, tuple[int, ...]]:
    """Return one CRN continuation seed and unique alternative action seeds."""
    group = SeedNamespace(
        master_seed=master_seed,
        rollout_id=rollout_id,
        target_id=target_node_id,
        replicate=1,
    )
    continuation_seed = group.derive(
        EventAddress(
            parent_node_id=target_node_id,
            turn_index=final_turn_index,
            call_slot_index=0,
        )
    )
    action_seeds = tuple(
        SeedNamespace(
            master_seed=master_seed,
            rollout_id=rollout_id,
            target_id=target_node_id,
            replicate=index,
        ).action_seed(index)
        for index in range(1, alternatives + 1)
    )
    if len(set(action_seeds)) != alternatives:
        raise RuntimeError("alternative action seeds collided")
    return continuation_seed, action_seeds


def execute_cached_arm(
    *,
    mode: ReplayMode,
    calls_by_index: dict[int, RecordedPolicyCall],
    visited_call_indices: tuple[int, ...],
    final_call_index: int,
    branch_final_prompt: tuple[int, ...],
    branch_final_seed: int,
    branch_final_decoding_config_hash: str,
    branch_final_action: tuple[int, ...],
    cache: PolicyActionCache,
    reward: float,
) -> ReplayArmResult:
    """Execute one arm; every policy decision must already be exact-key cached."""
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    reused: list[int] = []
    terminal: tuple[int, ...] | None = None

    def unexpected_sampler(
        prompt_token_ids: tuple[int, ...],
        seed: int,
    ) -> tuple[int, ...]:
        raise RuntimeError(
            f"replay attempted an unpaired model generation for seed {seed} "
            f"and {len(prompt_token_ids)} prompt tokens"
        )

    for call_index in visited_call_indices:
        call = calls_by_index[call_index]
        if call.event_seed is None:
            raise ValueError(f"call {call_index} has no event seed")
        prompt = (
            branch_final_prompt
            if call_index == final_call_index
            else call.prompt_token_ids
        )
        event_seed = (
            branch_final_seed
            if call_index == final_call_index
            else call.event_seed
        )
        decision = cache.resolve(
            prompt,
            checkpoint_id=call.checkpoint_id,
            decoding_config_hash=(
                branch_final_decoding_config_hash
                if call_index == final_call_index
                else call.decoding_config_hash
            ),
            event_seed=event_seed,
            sampler=unexpected_sampler,
        )
        if not decision.reused:
            raise RuntimeError("paired replay did not reuse an exact-key action")
        reused.append(call_index)
        if call_index == final_call_index:
            terminal = decision.action_token_ids

    if terminal != branch_final_action:
        raise RuntimeError("replay did not reach the paired terminal action")
    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    serialized = canonical_json(
        {
            "mode": mode.value,
            "visited_call_indices": visited_call_indices,
            "reused_call_indices": reused,
            "terminal_action": terminal,
            "reward": reward,
        }
    )
    return ReplayArmResult(
        mode=mode.value,
        visited_call_indices=visited_call_indices,
        exact_key_reused_call_indices=tuple(reused),
        terminal_action_sha256=_token_hash(terminal),
        reward=reward,
        actual_cost=ActualEvaluationCost(
            cpu_seconds=cpu_seconds,
            wall_seconds=wall_seconds,
            storage_bytes=len(serialized),
        ),
    )


def replace_unique(text: str, old: str, new: str) -> str:
    if not old:
        raise DeterministicReplayIneligibility(
            "replacement source must be non-empty"
        )
    count = text.count(old)
    if count != 1:
        raise DeterministicReplayIneligibility(
            f"replacement source must occur exactly once, found {count}"
        )
    return text.replace(old, new, 1)


def derive_lossless_render_boundary(
    *,
    recorded_prompt: tuple[int, ...],
    recorded_static_prefix: tuple[int, ...],
    canonical_render: tuple[int, ...],
) -> LosslessRenderBoundary:
    if not recorded_static_prefix:
        raise DeterministicReplayIneligibility(
            "recorded static prefix must be non-empty"
        )
    if recorded_prompt[: len(recorded_static_prefix)] != recorded_static_prefix:
        raise DeterministicReplayIneligibility(
            "recorded static prefix does not prefix recorded prompt"
        )
    exact_suffix = recorded_prompt[len(recorded_static_prefix) :]
    if not exact_suffix:
        raise DeterministicReplayIneligibility(
            "recorded affected suffix must be non-empty"
        )
    canonical_suffix_start = len(canonical_render) - len(exact_suffix)
    if (
        canonical_suffix_start < 0
        or canonical_render[canonical_suffix_start:] != exact_suffix
    ):
        raise DeterministicReplayIneligibility(
            "canonical render does not preserve the exact recorded affected suffix"
        )
    reconstructed = (
        recorded_static_prefix + canonical_render[canonical_suffix_start:]
    )
    if reconstructed != recorded_prompt:
        raise DeterministicReplayIneligibility(
            "lossless hybrid did not reconstruct recorded prompt"
        )
    return LosslessRenderBoundary(
        recorded_static_prefix_tokens=len(recorded_static_prefix),
        canonical_suffix_start_tokens=canonical_suffix_start,
        exact_common_suffix_tokens=len(exact_suffix),
    )


def splice_lossless_rendered_suffix(
    *,
    recorded_static_prefix: tuple[int, ...],
    canonical_original: tuple[int, ...],
    canonical_branch: tuple[int, ...],
    boundary: LosslessRenderBoundary,
) -> tuple[int, ...]:
    split = boundary.canonical_suffix_start_tokens
    if len(recorded_static_prefix) != boundary.recorded_static_prefix_tokens:
        raise DeterministicReplayIneligibility(
            "recorded static prefix length changed"
        )
    if canonical_branch[:split] != canonical_original[:split]:
        raise DeterministicReplayIneligibility(
            "branch changed tokens before the affected suffix"
        )
    return recorded_static_prefix + canonical_branch[split:]


def run_empirical_replay(
    *,
    trace_path: Path,
    client: TokenInferenceClient,
    alternatives_per_target: int,
    master_seed: str,
    temperature: float,
    candidate_max_tokens: int,
    continuation_max_tokens: int,
    minimum_distinct_candidate_fraction: float = 1.0,
    maximum_targets: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EmpiricalReplayReport:
    if alternatives_per_target < 1:
        raise ValueError("alternatives_per_target must be positive")
    if not 0 < minimum_distinct_candidate_fraction <= 1:
        raise ValueError(
            "minimum_distinct_candidate_fraction must be in (0, 1]"
        )
    if maximum_targets is not None and maximum_targets < 1:
        raise ValueError("maximum_targets must be positive when provided")
    source_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    audit = audit_trace_file(trace_path)
    provenance = import_trace_file(trace_path)
    native_traces = load_trace_records(trace_path)
    if (
        len(native_traces) != 1
        or len(provenance.traces) != 1
        or not audit.ready_for_exact_key_replay
        or not provenance.ready_for_representative_raf
    ):
        raise DeterministicReplayIneligibility(
            "campaign requires one exact-key, representative trace"
        )
    trace = provenance.traces[0]
    calls = tuple(sorted(audit.calls, key=lambda item: item.call_index))
    calls_by_index = {call.call_index: call for call in calls}
    final_candidates = [
        call
        for call in calls
        if call.agent_depth == 0 and call.turn_index is not None
    ]
    if not final_candidates:
        raise DeterministicReplayIneligibility(
            "trace has no root policy calls"
        )
    final_call = max(final_candidates, key=lambda item: item.turn_index or 0)
    target_calls = [call for call in calls if call.agent_depth == 1]
    if not target_calls:
        raise DeterministicReplayIneligibility(
            "trace has no recursive subcall targets"
        )
    if maximum_targets is not None:
        # Structural, pre-action selector: the first depth-one call in the
        # native call table. No action tokens, rewards, or content are read.
        target_calls = target_calls[:maximum_targets]

    raw_trace = native_traces[0]
    nodes = _object_list(raw_trace.get("nodes"), "trace.nodes")
    expected_terminal = _message_content(nodes[final_call.node_index])
    message_path = path_to_node(nodes, final_call.node_index)[:-1]
    root_messages = _openai_messages([
        copy.deepcopy(_message(nodes[node_index]))
        for node_index in message_path
    ])
    recorded_static_prefix = tuple(
        token_id
        for node_index in message_path[:-1]
        for token_id in _integer_tuple(
            nodes[node_index].get("token_ids"),
            f"trace.nodes[{node_index}].token_ids",
        )
    )
    tools = _chat_tools(raw_trace.get("tools"))
    if final_call.event_seed is None:
        raise DeterministicReplayIneligibility(
            "final call has no event seed"
        )
    rendered_original = client.render_chat(
        root_messages,
        tools,
        seed=final_call.event_seed,
        temperature=temperature,
        max_tokens=768,
    )
    render_boundary = derive_lossless_render_boundary(
        recorded_prompt=final_call.prompt_token_ids,
        recorded_static_prefix=recorded_static_prefix,
        canonical_render=rendered_original,
    )

    policy_node_ids_by_call: dict[int, str] = {}
    for node in trace.graph.nodes.values():
        if node.kind is not EventNodeKind.POLICY:
            continue
        call_index = node.metadata.get("call_index")
        if type(call_index) is int:
            policy_node_ids_by_call[call_index] = node.node_id
    if len(policy_node_ids_by_call) != len(calls):
        raise DeterministicReplayIneligibility(
            "policy graph and native call table do not align"
        )

    baseline_cache = build_policy_cache(calls)
    branch_decoding_config_hash = hashlib.sha256(
        canonical_json(
            {
                "endpoint": "/inference/v1/generate",
                "temperature": temperature,
                "max_tokens": continuation_max_tokens,
            }
        )
    ).hexdigest()
    pairs: list[EmpiricalBranchPair] = []
    regenerated_originals: list[EmpiricalOriginalArm] = []
    candidate_hashes_by_target: dict[str, set[str]] = {}
    for target in target_calls:
        if target.agent_depth is None:
            raise DeterministicReplayIneligibility(
                "eligible target call has no agent depth"
            )
        original_child_text = _message_content(nodes[target.node_index])
        target_node_id = policy_node_ids_by_call[target.call_index]
        full_indices, sliced_indices = build_replay_indices(
            target_call_index=target.call_index,
            target_node_id=target_node_id,
            policy_node_ids_by_call=policy_node_ids_by_call,
            descendants=trace.graph.descendants(target_node_id),
        )
        original_child_text = _message_content(nodes[target.node_index])
        common_continuation_seed, action_seeds = derive_branch_group_seeds(
            master_seed=master_seed,
            rollout_id=trace.trace_id,
            target_node_id=target_node_id,
            final_turn_index=final_call.turn_index or 0,
            alternatives=alternatives_per_target,
        )
        regenerated_original = client.generate(
            final_call.prompt_token_ids,
            seed=common_continuation_seed,
            temperature=temperature,
            max_tokens=continuation_max_tokens,
        )
        regenerated_original_text = _clean_action_text(
            client.detokenize(regenerated_original.token_ids)
        )
        regenerated_original_reward = float(
            regenerated_original_text.strip() == expected_terminal.strip()
        )
        original_cache = baseline_cache.fork()
        original_final_key = PolicyCallKey.from_call(
            final_call.prompt_token_ids,
            checkpoint_id=final_call.checkpoint_id,
            decoding_config_hash=branch_decoding_config_hash,
            event_seed=common_continuation_seed,
        )
        original_cache.record(
            CachedPolicyAction(
                original_final_key,
                regenerated_original.token_ids,
            )
        )
        original_full = execute_cached_arm(
            mode=ReplayMode.FULL_SUFFIX,
            calls_by_index=calls_by_index,
            visited_call_indices=full_indices,
            final_call_index=final_call.call_index,
            branch_final_prompt=final_call.prompt_token_ids,
            branch_final_seed=common_continuation_seed,
            branch_final_decoding_config_hash=branch_decoding_config_hash,
            branch_final_action=regenerated_original.token_ids,
            cache=original_cache.fork(),
            reward=regenerated_original_reward,
        )
        original_sliced = execute_cached_arm(
            mode=ReplayMode.SLICED,
            calls_by_index=calls_by_index,
            visited_call_indices=sliced_indices,
            final_call_index=final_call.call_index,
            branch_final_prompt=final_call.prompt_token_ids,
            branch_final_seed=common_continuation_seed,
            branch_final_decoding_config_hash=branch_decoding_config_hash,
            branch_final_action=regenerated_original.token_ids,
            cache=original_cache.fork(),
            reward=regenerated_original_reward,
        )
        regenerated_originals.append(
            EmpiricalOriginalArm(
                target_node_id=target_node_id,
                target_call_index=target.call_index,
                target_agent_depth=target.agent_depth,
                continuation_seed=common_continuation_seed,
                prompt_sha256=_token_hash(final_call.prompt_token_ids),
                downstream_generation=regenerated_original,
                full_suffix=original_full,
                sliced=original_sliced,
                terminal_artifacts_exact=(
                    original_full.terminal_action_sha256
                    == original_sliced.terminal_action_sha256
                ),
                rewards_exact=(
                    original_full.reward == original_sliced.reward
                ),
                cached_actions_exact=(
                    set(original_sliced.exact_key_reused_call_indices)
                    <= set(original_full.exact_key_reused_call_indices)
                ),
            )
        )
        for alternative_index in range(1, alternatives_per_target + 1):
            action_seed = action_seeds[alternative_index - 1]
            continuation_seed = common_continuation_seed
            candidate = client.generate(
                target.prompt_token_ids,
                seed=action_seed,
                temperature=temperature,
                max_tokens=candidate_max_tokens,
            )
            candidate_text = _clean_action_text(
                client.detokenize(candidate.token_ids)
            )
            branch_messages = copy.deepcopy(root_messages)
            tool_message = next(
                (
                    message
                    for message in reversed(branch_messages)
                    if message.get("role") == "tool"
                ),
                None,
            )
            if tool_message is None:
                raise DeterministicReplayIneligibility(
                    "root conversation has no tool response"
                )
            content = tool_message.get("content")
            if not isinstance(content, str):
                raise DeterministicReplayIneligibility(
                    "root tool response content must be a string"
                )
            tool_message["content"] = replace_unique(
                content,
                original_child_text,
                candidate_text,
            )
            canonical_branch_prompt = client.render_chat(
                branch_messages,
                tools,
                seed=continuation_seed,
                temperature=temperature,
                max_tokens=768,
            )
            branch_prompt = splice_lossless_rendered_suffix(
                recorded_static_prefix=recorded_static_prefix,
                canonical_original=rendered_original,
                canonical_branch=canonical_branch_prompt,
                boundary=render_boundary,
            )
            downstream = client.generate(
                branch_prompt,
                seed=continuation_seed,
                temperature=temperature,
                max_tokens=continuation_max_tokens,
            )
            downstream_text = _clean_action_text(
                client.detokenize(downstream.token_ids)
            )
            reward = float(downstream_text.strip() == expected_terminal.strip())

            paired_cache = baseline_cache.fork()
            if final_call.event_seed is None:
                raise DeterministicReplayIneligibility(
                    "final call has no event seed"
                )
            branch_final_key = PolicyCallKey.from_call(
                branch_prompt,
                checkpoint_id=final_call.checkpoint_id,
                decoding_config_hash=branch_decoding_config_hash,
                event_seed=continuation_seed,
            )
            paired_cache.record(
                CachedPolicyAction(branch_final_key, downstream.token_ids)
            )
            full = execute_cached_arm(
                mode=ReplayMode.FULL_SUFFIX,
                calls_by_index=calls_by_index,
                visited_call_indices=full_indices,
                final_call_index=final_call.call_index,
                branch_final_prompt=branch_prompt,
                branch_final_seed=continuation_seed,
                branch_final_decoding_config_hash=(
                    branch_decoding_config_hash
                ),
                branch_final_action=downstream.token_ids,
                cache=paired_cache.fork(),
                reward=reward,
            )
            sliced = execute_cached_arm(
                mode=ReplayMode.SLICED,
                calls_by_index=calls_by_index,
                visited_call_indices=sliced_indices,
                final_call_index=final_call.call_index,
                branch_final_prompt=branch_prompt,
                branch_final_seed=continuation_seed,
                branch_final_decoding_config_hash=(
                    branch_decoding_config_hash
                ),
                branch_final_action=downstream.token_ids,
                cache=paired_cache.fork(),
                reward=reward,
            )
            candidate_sha = _token_hash(candidate.token_ids)
            candidate_hashes_by_target.setdefault(target_node_id, set()).add(
                candidate_sha
            )
            pairs.append(
                EmpiricalBranchPair(
                    target_node_id=target_node_id,
                    target_call_index=target.call_index,
                    target_agent_depth=target.agent_depth,
                    alternative_index=alternative_index,
                    action_seed=action_seed,
                    continuation_seed=continuation_seed,
                    candidate_action_sha256=candidate_sha,
                    branch_prompt_sha256=_token_hash(branch_prompt),
                    candidate_action_generation=candidate,
                    downstream_generation=downstream,
                    full_suffix=full,
                    sliced=sliced,
                    terminal_artifacts_exact=(
                        full.terminal_action_sha256
                        == sliced.terminal_action_sha256
                    ),
                    rewards_exact=full.reward == sliced.reward,
                    cached_actions_exact=(
                        set(sliced.exact_key_reused_call_indices)
                        <= set(full.exact_key_reused_call_indices)
                    ),
                )
            )
            if progress_callback is not None:
                progress_callback(
                    len(pairs),
                    len(target_calls) * alternatives_per_target,
                )

    if final_call.event_seed is None:
        raise DeterministicReplayIneligibility(
            "final call has no event seed"
        )
    repro_first = client.generate(
        final_call.prompt_token_ids,
        seed=final_call.event_seed,
        temperature=temperature,
        max_tokens=continuation_max_tokens,
    )
    repro_second = client.generate(
        final_call.prompt_token_ids,
        seed=final_call.event_seed,
        temperature=temperature,
        max_tokens=continuation_max_tokens,
    )
    reproducibility = ReproducibilityAudit(
        seed=final_call.event_seed,
        first_action_sha256=_token_hash(repro_first.token_ids),
        second_action_sha256=_token_hash(repro_second.token_ids),
        exact=repro_first.token_ids == repro_second.token_ids,
        first_generation=repro_first,
        second_generation=repro_second,
    )

    distinct_candidate_fractions = _distinct_candidate_fractions(
        candidate_hashes_by_target,
        alternatives_per_target=alternatives_per_target,
    )
    distinct_candidates = all(
        fraction == 1.0
        for fraction in distinct_candidate_fractions.values()
    )
    distinct_candidate_gate_passed = all(
        fraction >= minimum_distinct_candidate_fraction
        for fraction in distinct_candidate_fractions.values()
    )
    terminal_mismatches = sum(not pair.terminal_artifacts_exact for pair in pairs)
    reward_mismatches = sum(not pair.rewards_exact for pair in pairs)
    cache_mismatches = sum(not pair.cached_actions_exact for pair in pairs)
    original_terminal_mismatches = sum(
        not arm.terminal_artifacts_exact for arm in regenerated_originals
    )
    original_reward_mismatches = sum(
        not arm.rewards_exact for arm in regenerated_originals
    )
    original_cache_mismatches = sum(
        not arm.cached_actions_exact for arm in regenerated_originals
    )
    full_visits = sum(
        len(pair.full_suffix.visited_call_indices) for pair in pairs
    ) + sum(
        len(arm.full_suffix.visited_call_indices)
        for arm in regenerated_originals
    )
    sliced_visits = sum(
        len(pair.sliced.visited_call_indices) for pair in pairs
    ) + sum(
        len(arm.sliced.visited_call_indices)
        for arm in regenerated_originals
    )
    candidate_tokens = sum(
        pair.candidate_action_generation.generated_tokens for pair in pairs
    )
    downstream_tokens = sum(
        pair.downstream_generation.generated_tokens for pair in pairs
    ) + sum(
        arm.downstream_generation.generated_tokens
        for arm in regenerated_originals
    )
    generation_prompt_tokens = sum(
        pair.candidate_action_generation.prompt_tokens
        + pair.downstream_generation.prompt_tokens
        for pair in pairs
    ) + sum(
        arm.downstream_generation.prompt_tokens
        for arm in regenerated_originals
    )
    model_wall = sum(
        pair.candidate_action_generation.wall_seconds
        + pair.downstream_generation.wall_seconds
        for pair in pairs
    ) + sum(
        arm.downstream_generation.wall_seconds
        for arm in regenerated_originals
    ) + repro_first.wall_seconds + repro_second.wall_seconds
    full_cost = _sum_arm_costs(
        [
            *(pair.full_suffix for pair in pairs),
            *(arm.full_suffix for arm in regenerated_originals),
        ]
    )
    sliced_cost = _sum_arm_costs(
        [
            *(pair.sliced for pair in pairs),
            *(arm.sliced for arm in regenerated_originals),
        ]
    )
    baseline_generated = sum(
        call.completion_tokens_reported or len(call.action_token_ids)
        for call in calls
    )
    policy_token_raf = (
        (
            baseline_generated
            + candidate_tokens
            + downstream_tokens
        )
        / baseline_generated
    )
    sliced_fraction = sliced_visits / full_visits if full_visits else 0.0
    passed = (
        distinct_candidate_gate_passed
        and len(pairs) == len(target_calls) * alternatives_per_target
        and terminal_mismatches == 0
        and reward_mismatches == 0
        and cache_mismatches == 0
        and original_terminal_mismatches == 0
        and original_reward_mismatches == 0
        and original_cache_mismatches == 0
        and len(regenerated_originals) == len(target_calls)
        and sliced_visits < full_visits
        and sliced_fraction < 0.9
    )
    return EmpiricalReplayReport(
        schema_version=1,
        generated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_sha256=source_sha,
        trace_id=trace.trace_id,
        checkpoint_id=final_call.checkpoint_id,
        branch_decoding_config_hash=branch_decoding_config_hash,
        alternatives_per_target=alternatives_per_target,
        target_count=len(target_calls),
        paired_branches=len(pairs),
        canonical_renderer_prompt_exact=(
            rendered_original == final_call.prompt_token_ids
        ),
        lossless_hybrid_preflight_exact=True,
        render_boundary=render_boundary,
        distinct_candidate_actions_per_target=distinct_candidates,
        minimum_distinct_candidate_fraction=(
            minimum_distinct_candidate_fraction
        ),
        distinct_candidate_fraction_by_target=distinct_candidate_fractions,
        distinct_candidate_gate_passed=distinct_candidate_gate_passed,
        deterministic_terminal_mismatches=terminal_mismatches,
        reward_mismatches=reward_mismatches,
        cached_action_mismatches=cache_mismatches,
        full_suffix_policy_events_visited=full_visits,
        sliced_policy_events_visited=sliced_visits,
        sliced_policy_event_fraction=sliced_fraction,
        alternative_action_generated_tokens=candidate_tokens,
        downstream_generated_tokens=downstream_tokens,
        generation_prompt_tokens=generation_prompt_tokens,
        model_request_wall_seconds=model_wall,
        exclusive_gpu_service_wall_seconds_proxy=model_wall,
        full_arm_cost=full_cost,
        sliced_arm_cost=sliced_cost,
        baseline_generated_tokens=baseline_generated,
        empirical_full_policy_token_raf=policy_token_raf,
        empirical_sliced_policy_token_raf=policy_token_raf,
        same_prompt_same_seed_reproducibility=reproducibility,
        target_metrics=_target_metrics(pairs),
        regenerated_originals=tuple(regenerated_originals),
        pairs=tuple(pairs),
        passed_representative_micro_gate=passed,
        gate_gb_cleared=False,
        limitations=(
            (
                "One recorded four-child trace and a micro campaign do not "
                "satisfy Gate GB's thousands-of-interventions requirement."
                if len(pairs) < 1_000
                else "Thousands of interventions still cover only four "
                "depth-one states from one recorded fixed-topology trace."
            ),
            "Exclusive GPU service wall time is reported as a proxy; kernel-level "
            "GPU seconds are not available from the pinned endpoint.",
            "The captured protocol has fixed topology under child-output "
            "replacement, so this campaign does not exercise new live topology.",
            "Exact paired downstream actions isolate slicing; same-prompt "
            "reproducibility is reported separately and is not a slicing gate.",
        ),
    )


def _distinct_candidate_fractions(
    hashes_by_target: dict[str, set[str]],
    *,
    alternatives_per_target: int,
) -> dict[str, float]:
    if alternatives_per_target < 1:
        raise ValueError("alternatives_per_target must be positive")
    return {
        target_id: len(hashes) / alternatives_per_target
        for target_id, hashes in sorted(hashes_by_target.items())
    }


def _target_metrics(
    pairs: list[EmpiricalBranchPair],
) -> tuple[EmpiricalTargetMetrics, ...]:
    grouped: dict[str, list[EmpiricalBranchPair]] = {}
    for pair in pairs:
        grouped.setdefault(pair.target_node_id, []).append(pair)
    metrics: list[EmpiricalTargetMetrics] = []
    for target_id, target_pairs in sorted(grouped.items()):
        first = target_pairs[0]
        full_visits = sum(
            len(pair.full_suffix.visited_call_indices) for pair in target_pairs
        )
        sliced_visits = sum(
            len(pair.sliced.visited_call_indices) for pair in target_pairs
        )
        metrics.append(
            EmpiricalTargetMetrics(
                target_node_id=target_id,
                target_call_index=first.target_call_index,
                target_agent_depth=first.target_agent_depth,
                paired_branches=len(target_pairs),
                alternative_action_generated_tokens=sum(
                    pair.candidate_action_generation.generated_tokens
                    for pair in target_pairs
                ),
                downstream_generated_tokens=sum(
                    pair.downstream_generation.generated_tokens
                    for pair in target_pairs
                ),
                generation_prompt_tokens=sum(
                    pair.candidate_action_generation.prompt_tokens
                    + pair.downstream_generation.prompt_tokens
                    for pair in target_pairs
                ),
                model_request_wall_seconds=math.fsum(
                    pair.candidate_action_generation.wall_seconds
                    + pair.downstream_generation.wall_seconds
                    for pair in target_pairs
                ),
                full_suffix_policy_events_visited=full_visits,
                sliced_policy_events_visited=sliced_visits,
                sliced_policy_event_fraction=(
                    sliced_visits / full_visits if full_visits else 0.0
                ),
                full_arm_cost=_sum_arm_costs(
                    pair.full_suffix for pair in target_pairs
                ),
                sliced_arm_cost=_sum_arm_costs(
                    pair.sliced for pair in target_pairs
                ),
            )
        )
    return tuple(metrics)


def _sum_arm_costs(arms: Any) -> ActualEvaluationCost:
    items = tuple(arms)
    return ActualEvaluationCost(
        generated_tokens=sum(item.actual_cost.generated_tokens for item in items),
        judge_calls=sum(item.actual_cost.judge_calls for item in items),
        cpu_seconds=sum(item.actual_cost.cpu_seconds for item in items),
        gpu_seconds=sum(item.actual_cost.gpu_seconds for item in items),
        wall_seconds=sum(item.actual_cost.wall_seconds for item in items),
        storage_bytes=sum(item.actual_cost.storage_bytes for item in items),
    )


def _message_content(node: dict[str, Any]) -> str:
    message = _message(node)
    content = message.get("content")
    if not isinstance(content, str):
        raise TypeError("trace node message content must be a string")
    return content


def _message(node: dict[str, Any]) -> dict[str, Any]:
    message = node.get("message")
    if not isinstance(message, dict):
        raise TypeError("trace node message must be an object")
    return message


def _chat_tools(value: Any) -> list[dict[str, Any]]:
    raw_tools = (
        [value]
        if isinstance(value, dict)
        else _object_list(value, "trace.tools")
    )
    if not raw_tools:
        raise ValueError("trace.tools must be non-empty")
    normalized: list[dict[str, Any]] = []
    for tool in raw_tools:
        if tool.get("type") == "function" and isinstance(
            tool.get("function"),
            dict,
        ):
            normalized.append(copy.deepcopy(tool))
        else:
            normalized.append(
                {
                    "type": "function",
                    "function": copy.deepcopy(tool),
                }
            )
    return normalized


def _openai_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            continue
        calls = _object_list(raw_calls, "message.tool_calls")
        openai_calls: list[dict[str, Any]] = []
        for call in calls:
            function = call.get("function")
            if isinstance(function, dict):
                openai_calls.append(call)
                continue
            call_id = call.get("id")
            name = call.get("name")
            arguments = call.get("arguments")
            if (
                not isinstance(call_id, str)
                or not isinstance(name, str)
                or not isinstance(arguments, str)
            ):
                raise TypeError(
                    "normalized tool calls require string id, name, and arguments"
                )
            openai_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )
        message["tool_calls"] = openai_calls
    return normalized


def _clean_action_text(text: str) -> str:
    text = text.rstrip()
    for suffix in ("<|im_end|>", "<|endoftext|>"):
        if text.endswith(suffix):
            text = text.removesuffix(suffix).rstrip()
    return text.rstrip()


def _object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(
        not isinstance(item, dict) for item in value
    ):
        raise TypeError(f"{label} must be a list of objects")
    return value


def _integer_tuple(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise TypeError(f"{label} must be a list of integers")
    return tuple(value)


def _nonnegative_integer(value: Any, *, fallback: int) -> int:
    return value if type(value) is int and value >= 0 else fallback


def _token_hash(token_ids: tuple[int, ...]) -> str:
    return hashlib.sha256(canonical_json(token_ids)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--alternatives-per-target", type=int, default=3)
    parser.add_argument(
        "--maximum-targets",
        type=int,
        help=(
            "Structurally select at most this many depth-one targets in native "
            "call order. Stage D uses 1."
        ),
    )
    parser.add_argument("--master-seed", default="redco-stage-b-replay-v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--candidate-max-tokens", type=int, default=192)
    parser.add_argument("--continuation-max-tokens", type=int, default=96)
    parser.add_argument(
        "--minimum-distinct-candidate-fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    source_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if source_sha != args.expected_source_sha256:
        raise ValueError(
            f"source hash mismatch: expected {args.expected_source_sha256}, "
            f"observed {source_sha}"
        )
    audit = audit_trace_file(args.input)
    if not audit.calls:
        raise ValueError("input contains no policy calls")
    client = TokenInferenceClient(
        base_url=args.base_url,
        model=audit.calls[0].checkpoint_id,
        timeout_seconds=args.timeout_seconds,
    )
    if args.progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    def progress(completed: int, total: int) -> None:
        if args.progress_every and (
            completed % args.progress_every == 0 or completed == total
        ):
            print(
                f"REDCO_REPLAY_PROGRESS={completed}/{total}",
                file=sys.stderr,
                flush=True,
            )

    report = run_empirical_replay(
        trace_path=args.input,
        client=client,
        alternatives_per_target=args.alternatives_per_target,
        master_seed=args.master_seed,
        temperature=args.temperature,
        candidate_max_tokens=args.candidate_max_tokens,
        continuation_max_tokens=args.continuation_max_tokens,
        minimum_distinct_candidate_fraction=(
            args.minimum_distinct_candidate_fraction
        ),
        maximum_targets=args.maximum_targets,
        progress_callback=progress,
    )
    payload = report.signed_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(payload) + b"\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report.passed_representative_micro_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
