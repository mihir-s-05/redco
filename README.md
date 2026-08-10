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

The deterministic Gate GB campaign is a scientific execution path. Run it only
under its separately authorized operating procedure:

```console
uv run --offline --frozen python -m redco.analysis.gate_gb
```

It writes a machine-readable report under `runs/stage-b/`. Prime integration,
providers, models, paid GPU work, training, and scientific execution all require
an explicit checkpoint; repository verification authorizes none of them.

## Development

Redco uses `uv` exclusively. Start with the guides to
[architecture](docs/architecture.md),
[development and verification](docs/development.md), and
[provenance and retention](docs/provenance.md).

The [development guide](docs/development.md) owns the exact self-contained
verification commands and named profiles. The verifier owns import paths, test
classification, frozen Ruff exceptions, strict affected-module typing, and
cache-free compilation.
