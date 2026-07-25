# ReDCO

Reference implementation of ReDCO-Lite: behavior-policy counterfactual branching
with dependency-sound replay for restricted recursive dataflow agents.

The normative research and implementation specification is
[`redco-implementation-plan.md`](redco-implementation-plan.md).

## Current milestone

Tier 0 is CPU-only and intentionally independent of prime-rl:

- estimator and seed-addressing contracts;
- immutable, content-addressed artifacts;
- event-DAG tracing;
- deterministic full-suffix and graph-sliced replay;
- enumerable synthetic credit probes.

Prime-rl integration and paid GPU execution require an explicit manual checkpoint.

## Development

This repository uses `uv` exclusively.

```console
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy src tests
uv run python -m redco.analysis.replay_equivalence
```
