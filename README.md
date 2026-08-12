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
- decision-normalized ReDCO credit assignment and loss evaluation;
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
training, and scientific runs are launched manually with an experiment-specific
time and cost bound; no background capacity monitor is part of the active core.

## Algorithm in brief

ReDCO targets one policy decision before its action is observed, restores the
state immediately before that decision, and evaluates the original action
beside sampled alternatives. Rewards are converted into leave-one-out branch
advantages. The targeted action's ordinary trajectory credit is removed, then
replaced by the branch comparison; all other decisions keep their trajectory
credit. Each decision occupies one unit in the outer normalization regardless
of its token span, while its action log-probability remains the sum over its
selected tokens. The framework-neutral objective in `redco.algo.training` also
supports an optional squared log-probability drift penalty against the behavior
policy that produced the replay.

### Controlled learning result

On 2026-08-12, the dependency-free credit-confusion experiment ran 1,000
seeded trials per method and task. Both methods used 16 policy calls per update
and 1,152 calls per trial. ReDCO reduced mean drift on an irrelevant decision
from `0.0748` to exactly `0`. On the noisy lucky task it reached the `0.8`
policy threshold in 425 calls on average, versus 434 for trajectory LOO. On the
redundant task the final policies were effectively tied (`0.8814` versus
`0.8820`), while ReDCO's estimator variance was 1.54 times higher—a useful
counterexample to the claim that branching always reduces variance.

The ignored local report is reproducible with:

```console
uv run --offline --frozen python -m redco.analysis.credit_confusion \
  --output runs/credit-confusion/report.json
```

Its canonical payload SHA-256 is
`a8ed7400ade493fc7c7808c28f0bba61431f8b209e4514135d006c4112bcfa2e`.

### QASPER model pilot

The first model-scale experiment is a two-decision QASPER evidence-retrieval
task. The policy first chooses one of four paper paragraphs, then chooses the
complete evidence span from that paragraph. The 32-task dataset is rebuilt from
the pre-cleanup Git archive into a 24/8 train/evaluation split. Each training
update gives both arms exactly ten policy calls: trajectory LOO samples five
complete episodes, while ReDCO branches two paragraph decisions and replays
seven continuations from one committed paragraph. Both actions are constrained
to one label token, and the objective is normalized by policy decisions.

Validate the frozen dataset and pilot configuration without importing Torch:

```console
uv run --offline --frozen python scripts/build_qasper_evidence_pilot.py --check
uv run --offline --frozen python scripts/run_qasper_evidence_pilot.py --check
```

The GPU configuration pins Qwen3-4B-Instruct-2507, rank-8 LoRA, 24 updates per
arm, a three-hour absolute runtime ceiling, and a $6 total cost ceiling. The
live run uses one ephemeral Prime GPU and always downloads the compact report
and adapters before terminating the pod.

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
