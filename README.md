# ReDCO-Lite

ReDCO-Lite is a small research prototype for behavior-policy counterfactual
branching and dependency-sound replay in recursive dataflow agents. It is the
maintained baseline for the next research question, not a claim to have originated
intermediate counterfactual branching or graph-aware credit assignment. The active
checkout is intentionally optimized for human review, experimentation, and
change—not for preserving every historical campaign as executable production
machinery.

## Active research surface

The maintained implementation contains:

- immutable event, seed, and branch contracts;
- content-addressed artifacts and exact policy-call caching;
- event-DAG tracing and dependency-sliced replay;
- deterministic and stochastic replay-equivalence checks;
- decision-normalized ReDCO-Lite credit assignment and loss evaluation;
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

## ReDCO-Lite algorithm in brief

ReDCO-Lite targets one policy decision before its action is observed, restores the
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
and 1,152 calls per trial. ReDCO-Lite reduced mean drift on an irrelevant decision
from `0.0748` to exactly `0`. On the noisy lucky task it reached the `0.8`
policy threshold in 425 calls on average, versus 434 for trajectory LOO. On the
redundant task the final policies were effectively tied (`0.8814` versus
`0.8820`), while ReDCO-Lite's estimator variance was 1.54 times higher—a useful
counterexample to the claim that branching always reduces variance.

The ignored local report is reproducible with:

```console
uv run --offline --frozen python -m redco.analysis.credit_confusion \
  --output runs/credit-confusion/report.json
```

Its canonical payload SHA-256 is
`a8ed7400ade493fc7c7808c28f0bba61431f8b209e4514135d006c4112bcfa2e`.

### Completed model experiments

The retired QASPER campaigns tested a shallow two-decision evidence-retrieval
task with Qwen3-4B-Instruct-2507 and rank-8 LoRA. A one-seed pilot established the
model-scale path but not an accuracy advantage. A five-seed matched matrix found
the same mean exact-evidence score for trajectory LOO and the tested ReDCO-Lite
allocation. The final allocation sweep exposed the more useful result: spending
more calls on conditioned span continuations improved conditional span accuracy
while reducing upstream paragraph accuracy. This is an allocation frontier, not
algorithmic superiority. See the compact [pilot](results/qasper-evidence-pilot-v1.json),
[matrix](results/qasper-evidence-matrix-v1.json), and
[allocation-sweep](results/qasper-allocation-sweep-v1.json) records.

The completed MuSiQue gates established that the tested models could not reliably
produce ordered four-hop support paths, so no credit-learning comparison was
warranted. See the compact [capability](results/musique-ans-capability-gate-v1.json),
[candidate-scoring](results/musique-ans-candidate-scoring-v1.json), and
[Qwen3.5 matrix](results/musique-ans-qwen35-matrix-v1.json) records.

The campaign-specific launchers, frozen task snapshots, configs, model adapters,
and one-shot tests have been retired from the active checkout. Their exact source
remains recoverable from Git; the normalized compact results above remain as the
reviewable scientific record.

## Completed routing investigation

**Status: closed after a negative cost-matched CPU gate.** The MuSiQue campaign is
retired from the active tree, and no sequential, LLM, or Prime routing experiment
is pending.

Typed interchange successfully measures how declared artifacts and ambient context
contribute to current reward. It does not predict which exchangeable redundant
channel will survive an unseen asymmetric failure. In the fragile shortcut mode,
the normal table `(0, 1, 1, 1)` is invariant to swapping artifact and context, so
every channel-equivariant objective must value them equally. The resulting typed
policy approaches the analytical held-out value `7/8`; ordinary route LOO
approaches `5/6`, and route-independent scalar controls approach `7/12`.

The earlier condition called typed interchange directly evaluated the held-out
failure and was therefore an oracle shift penalty. At equal reward-call budgets it
ties uniform corruption. The corrected typed objective reaches `0.874997`, below
uniform corruption at `0.999927` despite using more evaluations. This closes the
current routing objective, not the broader study of information flow.

ReDCO-Lite remains a baseline, dependency-sliced replay remains a systems substrate,
and typed interchange plus information provenance remain auditing instruments.
See the [completed investigation](docs/research-direction.md), compact
[`result`](results/routing-controls-v2.json), and [pivot brief](docs/pivot-brief.md).

## Historical work

Old Stage C/D campaigns, launch protocols, provider integrations, reports,
configs, datasets, patches, and bespoke verification harnesses were retired from
the active checkout. They remain exactly recoverable from Git commit
`53a7c67c9cb6df39e44454f364aaf3c9ca352966`.

[`provenance/history-v1.jsonl`](provenance/history-v1.jsonl) is the normalized
recovery index: it records every pre-cleanup file's Git blob, raw SHA-256, byte
count, role, format, and safe schema metadata without rewriting the original
bytes. See [provenance](docs/provenance.md) for recovery instructions.

The retired `redco-implementation-plan.md` is explicitly superseded as of August
2026: later literature review invalidated its broad novelty positioning, and its
context-routing implication failed the corrected cost-matched CPU gate. The plan's
exact historical bytes remain indexed in `provenance/history-v1.jsonl` at Git blob
`6a4521bb093482899f8879289e8223a3a17bd275`; it is project history, not active
guidance.

The concise guides are [architecture](docs/architecture.md),
[development](docs/development.md), [research direction](docs/research-direction.md),
[pivot brief](docs/pivot-brief.md), and [provenance](docs/provenance.md).
