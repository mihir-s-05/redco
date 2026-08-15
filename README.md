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
complete episodes, while the tested ReDCO allocation samples two complete
episodes and six additional span continuations from one committed paragraph.
ReDCO therefore estimates paragraph credit from only two complete rewards and
span credit from seven alternatives including the original span. Both actions
are constrained to one label token, and the objective is normalized by policy
decisions.

Validate the frozen dataset and pilot configuration without importing Torch:

```console
uv run --offline --frozen python scripts/build_qasper_evidence_pilot.py --check
uv run --offline --frozen python scripts/run_qasper_evidence_pilot.py --check
```

The GPU configuration pins Qwen3-4B-Instruct-2507, rank-8 LoRA, 24 updates per
arm, a three-hour absolute runtime ceiling, and a $6 total cost ceiling. The
live run uses one ephemeral Prime GPU and always downloads the compact report
and adapters before terminating the pod.

The bounded Prime run completed on 2026-08-12 for $0.7788. Both arms improved
exact evidence selection from 4/8 to 5/8 evaluation tasks, so this seed does not
show a ReDCO accuracy advantage. ReDCO had lower mean and maximum gradient norms
but also lower sampled training reward. This is a useful pilot result: the
model-scale path works, while a convincing algorithm comparison now needs more
evaluation tasks and multiple matched seeds rather than a larger model. The
compact normalized result is
[`results/qasper-evidence-pilot-v1.json`](results/qasper-evidence-pilot-v1.json).

The matched follow-up used five seeds and expanded evaluation to 24
paper-disjoint tasks. Both arms averaged 12.8/24 exact evidence matches after
training from the common 11/24 baseline. Per-seed ReDCO-minus-trajectory counts
were `[0, 2, -1, 1, -2]`, with mean and median zero. This is a null for the
shallow task and the tested ReDCO `2+6` allocation, not for ReDCO generally.

The more informative decomposition is that conditional span accuracy given a
correct paragraph increased from 55/75 (73.3%) to 64/69 (92.8%) for trajectory
LOO and 64/70 (91.4%) for ReDCO, while paragraph accuracy fell in both arms.
The run therefore measured strong span learning alongside upstream paragraph
drift. It cost $0.4425 on one A100, retained no adapters, and left zero Prime
pods. See
[`results/qasper-evidence-matrix-v1.json`](results/qasper-evidence-matrix-v1.json).

The allocation follow-up kept the ten-call update budget fixed and compared
trajectory LOO against three branch-credit splits: `4+2`, `3+4`, and `2+6`
complete-root/conditioned-span calls. It used the same five matched seeds but
expanded evaluation to 96 paper-disjoint tasks per seed. Among the branch arms,
more span continuations produced a clear descriptive frontier: mean paragraph
counts fell from 52.4 to 51.8 to 50.6, while conditional span accuracy rose
from 79.2% to 89.0% to 94.5% and exact-evidence counts rose from 41.4 to 46.2
to 47.8. The `2+6` arm nearly tied trajectory LOO on exact evidence (47.8
versus 47.4 of 96), with 3.3 percentage points higher mean conditional span
accuracy and 1.4 fewer correct paragraphs. With five seeds this is evidence of
an allocation trade-off, not superiority. The one-A100 run cost $0.6819 and
left zero Prime pods. See
[`results/qasper-allocation-sweep-v1.json`](results/qasper-allocation-sweep-v1.json).

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
