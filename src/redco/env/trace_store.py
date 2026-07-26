"""Stable local persistence for event graphs, branch tuples, and gate evidence."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json
from redco.env.tracer import EventGraph


@dataclass(frozen=True, slots=True)
class BranchTuple:
    task_id: str
    rollout_id: str
    node_id: str
    selection_rule: str
    action_source: str
    action_ref: str
    seed_bundle_id: str
    replay_mode: str
    reward: float
    logical_deployment_cost: dict[str, int | float]
    actual_eval_cost_meters: dict[str, int | float]
    decision_unit_weight: float
    divergence_markers: dict[str, str | int | None]
    checkpoint_id: str

    def __post_init__(self) -> None:
        required = (
            self.task_id,
            self.rollout_id,
            self.node_id,
            self.selection_rule,
            self.action_source,
            self.action_ref,
            self.seed_bundle_id,
            self.replay_mode,
            self.checkpoint_id,
        )
        if any(not value for value in required):
            raise ValueError("branch tuple identifiers must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceStore:
    """Atomic, canonical JSON/JSONL writer rooted in a local run directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write_graph(self, name: str, graph: EventGraph) -> Path:
        return self.write_json(f"{name}.event-graph.json", graph.as_dict())

    def write_branches(self, name: str, branches: tuple[BranchTuple, ...]) -> Path:
        payload = b"".join(canonical_json(branch.as_dict()) + b"\n" for branch in branches)
        return self._atomic_write(f"{name}.branches.jsonl", payload)

    def write_json(self, name: str, value: Any) -> Path:
        return self._atomic_write(name, canonical_json(value) + b"\n")

    def _atomic_write(self, name: str, data: bytes) -> Path:
        if not name or Path(name).name != name:
            raise ValueError("trace file name must be a single path component")
        path = self.root / name
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        return path
