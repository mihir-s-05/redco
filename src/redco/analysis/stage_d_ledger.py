from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

TokenRole = Literal[
    "root",
    "child",
    "branch_continuation",
    "evaluation",
    "judge",
]


@dataclass(frozen=True)
class UsageBucket:
    calls: int = 0
    prompt_tokens: int = 0
    cached_input_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    def plus(self, usage: Mapping[str, Any]) -> UsageBucket:
        prompt = _nonnegative_int(usage, "prompt_tokens")
        completion = _nonnegative_int(usage, "completion_tokens")
        cached = _optional_nonnegative_int(usage, "cached_input_tokens")
        reasoning = _optional_nonnegative_int(usage, "reasoning_tokens")
        return UsageBucket(
            calls=self.calls + 1,
            prompt_tokens=self.prompt_tokens + prompt,
            cached_input_tokens=self.cached_input_tokens + cached,
            completion_tokens=self.completion_tokens + completion,
            reasoning_tokens=self.reasoning_tokens + reasoning,
        )


@dataclass(frozen=True)
class ResourceMeters:
    optimizer_updates: int
    service_seconds: float
    wall_seconds: float
    gpu_seconds: float
    storage_bytes: int

    def __post_init__(self) -> None:
        if (
            self.optimizer_updates < 0
            or self.service_seconds < 0
            or self.wall_seconds < 0
            or self.gpu_seconds < 0
            or self.storage_bytes < 0
        ):
            raise ValueError("Stage D resource meters must be nonnegative")


@dataclass(frozen=True)
class StageDLedger:
    usage: dict[TokenRole, UsageBucket]
    resources: ResourceMeters

    @property
    def training_generated_tokens(self) -> int:
        return sum(
            self.usage[role].completion_tokens
            for role in ("root", "child", "branch_continuation")
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": 1,
            "usage": {role: asdict(bucket) for role, bucket in self.usage.items()},
            "resources": asdict(self.resources),
            "primary": {
                "training_generated_tokens": self.training_generated_tokens,
                "gpu_hours": self.resources.gpu_seconds / 3600,
                "wall_hours": self.resources.wall_seconds / 3600,
            },
        }
        return result


def _nonnegative_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"usage.{key} must be a nonnegative integer")
    return value


def _optional_nonnegative_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if value is None:
        return 0
    return _nonnegative_int(payload, key)


def _trace_role(trace: Mapping[str, Any], run_kind: str, call: Mapping[str, Any]) -> TokenRole:
    if run_kind == "eval":
        return "evaluation"
    info = trace.get("info") or {}
    if info.get("redco_record_kind") == "branch" or info.get("record_kind") == "branch":
        return "branch_continuation"
    structure = call.get("rlm") or {}
    depth = structure.get("depth")
    if isinstance(depth, int) and depth > 0:
        return "child"
    return "root"


def iter_traces(records: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for record in records:
        traces = record.get("traces")
        if traces is None:
            yield record
            continue
        if not isinstance(traces, list):
            raise ValueError("episode.traces must be a list")
        for trace in traces:
            if not isinstance(trace, Mapping):
                raise ValueError("episode trace must be an object")
            yield trace


def build_stage_d_ledger(
    train_records: Iterable[Mapping[str, Any]],
    eval_records: Iterable[Mapping[str, Any]],
    resources: ResourceMeters,
) -> StageDLedger:
    usage: dict[TokenRole, UsageBucket] = {
        role: UsageBucket()
        for role in (
            "root",
            "child",
            "branch_continuation",
            "evaluation",
            "judge",
        )
    }
    for run_kind, records in (("train", train_records), ("eval", eval_records)):
        for trace in iter_traces(records):
            for call in trace.get("calls") or []:
                if not isinstance(call, Mapping):
                    raise ValueError("trace call must be an object")
                call_usage = call.get("usage")
                if not isinstance(call_usage, Mapping):
                    raise ValueError("every Stage D policy call must record usage")
                role = _trace_role(trace, run_kind, call)
                usage[role] = usage[role].plus(call_usage)
            for judge_usage in trace.get("extra_usage") or []:
                if not isinstance(judge_usage, Mapping):
                    raise ValueError("trace extra_usage entry must be an object")
                usage["judge"] = usage["judge"].plus(judge_usage)
    return StageDLedger(usage=usage, resources=resources)


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records
