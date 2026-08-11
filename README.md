# ReDCO

ReDCO is a small research prototype for behavior-policy counterfactual branching
and dependency-sound replay in recursive dataflow agents. The active checkout is
intentionally optimized for human review, experimentation, and change—not for
preserving every historical campaign as executable production machinery.

## Active research surface

The maintained implementation contains:

- immutable event, seed, and branch contracts;
- content-addressed artifacts and exact policy-call caching;
- event-DAG tracing and dependency-sliced replay;
- deterministic and stochastic replay-equivalence checks;
- synthetic credit probes and estimator diagnostics;
- the CPU-only Gate GB research gate.

Start in these files:

- `src/redco/contracts.py` — shared research values and canonical JSON;
- `src/redco/algo/` — branching and training primitives;
- `src/redco/env/` — artifacts, tracing, policy caching, and replay;
- `src/redco/analysis/gate_gb.py` — the end-to-end CPU research gate;
- `tests/` — the focused executable specification.

## Run it

Redco uses `uv`, never `pip`.

```console
uv sync --frozen
uv run --frozen pytest
uv run --frozen ruff check src tests
uv run --frozen mypy
```

Run the cheap CPU gate with:

```console
uv run --offline --frozen python -m redco.analysis.gate_gb
```

It writes under the ignored `runs/` tree. Network, provider, model, GPU,
training, or scientific campaigns require separate explicit authorization.

## Historical work

Old Stage C/D campaigns, launch protocols, provider integrations, reports,
configs, datasets, patches, and bespoke verification harnesses were retired from
the active checkout. They remain exactly recoverable from Git commit
`53a7c67c9cb6df39e44454f364aaf3c9ca352966`.

[`provenance/history-v1.jsonl`](provenance/history-v1.jsonl) is the normalized
recovery index: it records every pre-cleanup file's Git blob, raw SHA-256, byte
count, role, format, and safe schema metadata without rewriting the original
bytes. See [provenance](docs/provenance.md) for recovery instructions.

The concise guides are [architecture](docs/architecture.md),
[development](docs/development.md), and [provenance](docs/provenance.md).
