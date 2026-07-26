"""Branch-specific dynamic topology for deterministic Tier-0 replay."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from redco.contracts import canonical_json
from redco.env.commands import JsonValue, TypedCommand, execute_command
from redco.env.policy_cache import (
    PolicyActionCache,
    Sampler,
    TopologyDivergence,
)
from redco.env.replay import ReplayMode


class EventRole(StrEnum):
    """RLM-shaped work categories used by replay accounting."""

    ENVIRONMENT = "environment"
    ROOT_POLICY = "root_policy"
    SUBCALL_POLICY = "subcall_policy"
    JUDGE = "judge"


@dataclass(frozen=True, slots=True)
class WorkEstimate:
    """Logical work attached to one event, without pretending it is runtime."""

    cpu_units: int = 0
    generated_tokens: int = 0
    judge_calls: int = 0

    def __post_init__(self) -> None:
        if min(self.cpu_units, self.generated_tokens, self.judge_calls) < 0:
            raise ValueError("work estimates must be non-negative")


@dataclass(frozen=True, slots=True)
class BranchGuard:
    """Activate a command only when an earlier artifact has an exact JSON value."""

    artifact: str
    equals: JsonValue = True

    def matches(self, state: dict[str, JsonValue]) -> bool:
        try:
            actual = state[self.artifact]
        except KeyError as error:
            raise KeyError(f"guard reads missing artifact: {self.artifact}") from error
        return canonical_json(actual) == canonical_json(self.equals)


@dataclass(frozen=True, slots=True)
class PolicyInvocation:
    """Exact-key inputs for one policy event."""

    prompt_artifact: str
    checkpoint_id: str
    decoding_config_hash: str
    event_seed: int

    def __post_init__(self) -> None:
        if not self.prompt_artifact:
            raise ValueError("prompt_artifact must be non-empty")
        if not self.checkpoint_id or not self.decoding_config_hash:
            raise ValueError("policy identifiers must be non-empty")
        if self.event_seed < 0:
            raise ValueError("event_seed must be non-negative")


@dataclass(frozen=True, slots=True)
class DynamicCommand:
    """A typed command that may exist only in one materialized branch."""

    command: TypedCommand
    guard: BranchGuard | None = None
    role: EventRole = EventRole.ENVIRONMENT
    work: WorkEstimate = WorkEstimate(cpu_units=1)
    policy: PolicyInvocation | None = None

    def __post_init__(self) -> None:
        if (
            self.policy is not None
            and self.policy.prompt_artifact not in self.command.inputs
        ):
            raise ValueError("policy prompt artifact must be a command input")
        if self.policy is not None and self.role not in {
            EventRole.ROOT_POLICY,
            EventRole.SUBCALL_POLICY,
        }:
            raise ValueError("policy invocation requires a policy event role")


@dataclass(frozen=True, slots=True)
class DynamicProgram:
    """Ordered command templates whose active subset is branch-specific."""

    initial_state: dict[str, JsonValue]
    commands: tuple[DynamicCommand, ...]
    terminal_output: str

    def __post_init__(self) -> None:
        available = set(self.initial_state)
        event_ids: set[str] = set()
        for template in self.commands:
            command = template.command
            if command.event_id in event_ids:
                raise ValueError(f"duplicate event_id: {command.event_id}")
            if template.guard is not None and template.guard.artifact not in available:
                raise ValueError(
                    f"{command.event_id} guard reads unavailable artifact: "
                    f"{template.guard.artifact}"
                )
            missing_inputs = set(command.inputs) - available
            if missing_inputs:
                raise ValueError(
                    f"{command.event_id} reads unavailable inputs: "
                    f"{sorted(missing_inputs)}"
                )
            unknown_dependencies = set(command.explicit_event_dependencies) - event_ids
            if unknown_dependencies:
                raise ValueError(
                    f"{command.event_id} has unavailable event dependencies: "
                    f"{sorted(unknown_dependencies)}"
                )
            if command.output in self.initial_state:
                raise ValueError(
                    f"command output shadows initial artifact: {command.output}"
                )
            event_ids.add(command.event_id)
            # Multiple mutually exclusive templates may intentionally produce
            # the same branch-join artifact. Runtime execution rejects a pair
            # if both templates are active in one materialized branch.
            available.add(command.output)
        if self.terminal_output not in available:
            raise ValueError("terminal_output is not produced by the program")

    def command(self, event_id: str) -> DynamicCommand:
        for template in self.commands:
            if template.command.event_id == event_id:
                return template
        raise KeyError(f"unknown event: {event_id}")


@dataclass(frozen=True, slots=True)
class DynamicExecutionResult:
    state: dict[str, JsonValue]
    active_event_ids: tuple[str, ...]
    restored_prefix_event_ids: tuple[str, ...]
    reexecuted_suffix_event_ids: tuple[str, ...]
    reused_suffix_event_ids: tuple[str, ...]
    topology: TopologyDivergence
    policy_decisions: tuple[PolicyEventDecision, ...] = ()

    def terminal(self, output_name: str) -> JsonValue:
        return self.state[output_name]

    @property
    def state_bytes(self) -> bytes:
        return canonical_json(self.state)

    @property
    def regenerated_policy_event_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.event_id
            for decision in self.policy_decisions
            if decision.generated
        )

    @property
    def reused_policy_event_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.event_id
            for decision in self.policy_decisions
            if decision.reused
        )


@dataclass(frozen=True, slots=True)
class PolicyEventDecision:
    event_id: str
    reused: bool
    generated: bool
    reason: str


@dataclass(slots=True)
class _MaterializedTrace:
    state: dict[str, JsonValue]
    producer_by_artifact: dict[str, str | None]
    active_event_ids: list[str]
    policy_cache: PolicyActionCache
    policy_decisions: list[PolicyEventDecision]


class DynamicReplayEngine:
    """Replay branches that may add or remove commands and artifacts."""

    def __init__(
        self,
        program: DynamicProgram,
        *,
        policy_cache: PolicyActionCache | None = None,
        sampler: Sampler | None = None,
    ) -> None:
        self.program = program
        self._sampler = sampler
        if (
            any(template.policy is not None for template in program.commands)
            and sampler is None
        ):
            raise ValueError("a sampler is required for policy commands")
        self._event_index = {
            template.command.event_id: index
            for index, template in enumerate(program.commands)
        }
        initial_cache = policy_cache or PolicyActionCache()
        self._original = self._execute_original(initial_cache.fork())
        self._base_policy_cache = self._original.policy_cache.fork()

    @property
    def original(self) -> DynamicExecutionResult:
        active = tuple(self._original.active_event_ids)
        return DynamicExecutionResult(
            state=dict(self._original.state),
            active_event_ids=active,
            restored_prefix_event_ids=(),
            reexecuted_suffix_event_ids=active,
            reused_suffix_event_ids=(),
            topology=TopologyDivergence(active, active),
            policy_decisions=tuple(self._original.policy_decisions),
        )

    def replay(
        self,
        *,
        target_event_id: str,
        replacement: JsonValue,
        mode: ReplayMode,
    ) -> DynamicExecutionResult:
        try:
            target_index = self._event_index[target_event_id]
        except KeyError as error:
            raise KeyError(f"unknown target event: {target_event_id}") from error
        if target_event_id not in self._original.active_event_ids:
            raise ValueError("intervention target is absent from the original branch")

        prefix = self._execute_prefix(target_index)
        target = self.program.commands[target_index]
        if target.guard is not None and not target.guard.matches(prefix.state):
            raise RuntimeError("original intervention target became inactive")
        prefix.state[target.command.output] = replacement
        prefix.producer_by_artifact[target.command.output] = target_event_id
        prefix.active_event_ids.append(target_event_id)
        if target.policy is not None:
            prefix.policy_decisions.append(
                PolicyEventDecision(
                    target_event_id,
                    reused=False,
                    generated=False,
                    reason="intervention_action",
                )
            )

        if mode is ReplayMode.FULL_SUFFIX:
            return self._full_suffix(prefix, target_index)
        if mode is ReplayMode.SLICED:
            return self._sliced(prefix, target_index)
        raise ValueError(f"unsupported replay mode: {mode}")

    def _execute_original(
        self,
        policy_cache: PolicyActionCache,
    ) -> _MaterializedTrace:
        trace = _MaterializedTrace(
            state=dict(self.program.initial_state),
            producer_by_artifact={
                artifact: None for artifact in self.program.initial_state
            },
            active_event_ids=[],
            policy_cache=policy_cache,
            policy_decisions=[],
        )
        for template in self.program.commands:
            if self._is_active(template, trace.state):
                self._execute_active(template, trace)
        if self.program.terminal_output not in trace.state:
            raise RuntimeError("materialized branch did not produce terminal output")
        return trace

    def _execute_prefix(self, target_index: int) -> _MaterializedTrace:
        trace = _MaterializedTrace(
            state=dict(self.program.initial_state),
            producer_by_artifact={
                artifact: None for artifact in self.program.initial_state
            },
            active_event_ids=[],
            policy_cache=self._base_policy_cache.fork(),
            policy_decisions=[],
        )
        for template in self.program.commands[:target_index]:
            if self._is_active(template, trace.state):
                self._execute_active(template, trace)
        return trace

    def _full_suffix(
        self,
        prefix: _MaterializedTrace,
        target_index: int,
    ) -> DynamicExecutionResult:
        reexecuted: list[str] = []
        for template in self.program.commands[target_index + 1 :]:
            if self._is_active(template, prefix.state):
                self._execute_active(template, prefix)
                reexecuted.append(template.command.event_id)
        return self._result(prefix, target_index, reexecuted, [])

    def _sliced(
        self,
        prefix: _MaterializedTrace,
        target_index: int,
    ) -> DynamicExecutionResult:
        state = dict(self._original.state)
        state.update(prefix.state)
        producers = dict(self._original.producer_by_artifact)
        producers.update(prefix.producer_by_artifact)
        trace = _MaterializedTrace(
            state,
            producers,
            list(prefix.active_event_ids),
            prefix.policy_cache,
            list(prefix.policy_decisions),
        )
        original_events = set(self._original.active_event_ids)
        target_output = self.program.commands[target_index].command.output
        original_target_value = self._original.state[target_output]
        dirty_artifacts = (
            {target_output}
            if canonical_json(prefix.state[target_output])
            != canonical_json(original_target_value)
            else set()
        )
        dirty_events = {
            self.program.commands[target_index].command.event_id
        }
        reexecuted: list[str] = []
        reused: list[str] = []

        for template in self.program.commands[target_index + 1 :]:
            command = template.command
            if not self._is_active(template, trace.state):
                if (
                    trace.producer_by_artifact.get(command.output)
                    == command.event_id
                ):
                    trace.state.pop(command.output, None)
                    trace.producer_by_artifact.pop(command.output, None)
                continue

            trace.active_event_ids.append(command.event_id)
            guard_dirty = (
                template.guard is not None
                and template.guard.artifact in dirty_artifacts
            )
            data_dirty = any(name in dirty_artifacts for name in command.inputs)
            dependency_dirty = any(
                dependency in dirty_events
                for dependency in command.explicit_event_dependencies
            )
            producer_changed = (
                trace.producer_by_artifact.get(command.output)
                != command.event_id
            )
            should_execute = (
                command.event_id not in original_events
                or guard_dirty
                or data_dirty
                or dependency_dirty
                or producer_changed
            )
            if should_execute:
                current_producer = trace.producer_by_artifact.get(command.output)
                if (
                    current_producer != command.event_id
                    and current_producer in trace.active_event_ids
                ):
                    raise RuntimeError(
                        f"multiple active producers for {command.output}: "
                        f"{current_producer}, {command.event_id}"
                    )
                previous_value = trace.state.get(command.output)
                output_value = self._execute_value(template, trace)
                output_changed = (
                    current_producer != command.event_id
                    or command.output not in trace.state
                    or canonical_json(previous_value) != canonical_json(output_value)
                )
                trace.state[command.output] = output_value
                trace.producer_by_artifact[command.output] = command.event_id
                if output_changed:
                    dirty_artifacts.add(command.output)
                dirty_events.add(command.event_id)
                reexecuted.append(command.event_id)
            else:
                reused.append(command.event_id)
                if template.policy is not None:
                    trace.policy_decisions.append(
                        PolicyEventDecision(
                            command.event_id,
                            reused=True,
                            generated=False,
                            reason="unchanged_event_reuse",
                        )
                    )

        return self._result(trace, target_index, reexecuted, reused)

    def _result(
        self,
        trace: _MaterializedTrace,
        target_index: int,
        reexecuted: list[str],
        reused: list[str],
    ) -> DynamicExecutionResult:
        if self.program.terminal_output not in trace.state:
            raise RuntimeError("materialized branch did not produce terminal output")
        original_events = tuple(self._original.active_event_ids)
        branch_events = tuple(trace.active_event_ids)
        restored = tuple(
            event_id
            for event_id in trace.active_event_ids
            if self._event_index[event_id] < target_index
        )
        return DynamicExecutionResult(
            state=trace.state,
            active_event_ids=branch_events,
            restored_prefix_event_ids=restored,
            reexecuted_suffix_event_ids=tuple(reexecuted),
            reused_suffix_event_ids=tuple(reused),
            topology=TopologyDivergence(original_events, branch_events),
            policy_decisions=tuple(trace.policy_decisions),
        )

    @staticmethod
    def _is_active(
        template: DynamicCommand,
        state: dict[str, JsonValue],
    ) -> bool:
        return template.guard is None or template.guard.matches(state)

    def _execute_active(
        self,
        template: DynamicCommand,
        trace: _MaterializedTrace,
    ) -> None:
        command = template.command
        existing = trace.producer_by_artifact.get(command.output)
        if command.output in trace.state:
            raise RuntimeError(
                f"multiple active producers for {command.output}: "
                f"{existing}, {command.event_id}"
            )
        trace.state[command.output] = self._execute_value(template, trace)
        trace.producer_by_artifact[command.output] = command.event_id
        trace.active_event_ids.append(command.event_id)

    def _execute_value(
        self,
        template: DynamicCommand,
        trace: _MaterializedTrace,
    ) -> JsonValue:
        if template.policy is None:
            return execute_command(template.command, trace.state)
        prompt_value = trace.state[template.policy.prompt_artifact]
        if not isinstance(prompt_value, list) or any(
            type(token) is not int for token in prompt_value
        ):
            raise TypeError("policy prompt artifact must be a list of integer tokens")
        prompt_tokens = tuple(cast(list[int], prompt_value))
        if self._sampler is None:
            raise RuntimeError("policy sampler is unavailable")
        decision = trace.policy_cache.resolve(
            prompt_tokens,
            checkpoint_id=template.policy.checkpoint_id,
            decoding_config_hash=template.policy.decoding_config_hash,
            event_seed=template.policy.event_seed,
            sampler=self._sampler,
        )
        trace.policy_decisions.append(
            PolicyEventDecision(
                template.command.event_id,
                reused=decision.reused,
                generated=not decision.reused,
                reason=decision.reason,
            )
        )
        return list(decision.action_token_ids)
