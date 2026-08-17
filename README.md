# ReDCO-Lite

ReDCO-Lite is a small research implementation of local counterfactual credit
assignment for multi-step agents. It records an execution as an event graph,
branches from a selected policy decision, replays only the affected suffix, and
uses the resulting rewards to update that decision.

The repository is intentionally dependency-light and optimized for readable,
CPU-testable research code.

## Setup

Requirements: Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/).

```console
uv sync --frozen
uv run --frozen pytest
```

Run the static checks with:

```console
uv run --frozen ruff check src tests
uv run --frozen mypy
git diff --check
```

Run the end-to-end CPU gate with:

```console
uv run --offline --frozen python -m redco.analysis.gate_gb
```

## Algorithm

Given a completed trajectory, ReDCO-Lite:

1. Chooses a policy decision before observing its action.
2. Restores the state immediately before that decision.
3. Replays the original action and sampled alternatives through the affected
   downstream events.
4. Converts the branch rewards into leave-one-out advantages.
5. Replaces that decision's trajectory-level credit with its local branch credit.
6. Leaves every untargeted decision's ordinary trajectory credit unchanged.

For branch reward `R_i` among `K` alternatives, the local advantage is:

```text
A_i = R_i - sum(R_j for j != i) / (K - 1)
```

The training objective sums token log-probabilities within each action, then
normalizes across policy decisions rather than tokens. This prevents long actions
from receiving extra weight merely because they contain more tokens. An optional
squared log-probability penalty constrains drift from the behavior policy that
generated the replay.

ReDCO-Lite is most useful as a controlled baseline for studying where credit is
assigned. It does not assume that local branching always improves learning or
reduces variance.

## Repository map

| Path | Purpose |
|---|---|
| `src/redco/contracts.py` | Immutable event, seed, branch, and policy-call values |
| `src/redco/algo/` | Branch construction and training calculations |
| `src/redco/env/` | Tracing, artifacts, caching, commands, and replay |
| `src/redco/analysis/` | Small CPU research gates and diagnostics |
| `tests/` | Executable specification of maintained behavior |
| `results/` | Compact records from completed experiments |
| `provenance/` | Git recovery index for retired files |

See [architecture](docs/architecture.md), [development](docs/development.md), and
[provenance](docs/provenance.md) for the small amount of additional documentation.

## Research workflow

- Start with an analytical or CPU test that could falsify the hypothesis.
- Keep exploratory outputs under ignored `runs/` or `.artifacts/` directories.
- Promote only compact results needed for interpretation or reproduction.
- Give model or provider runs explicit time and cost limits.
- Delete obsolete campaign code after its result is preserved; Git is the archive.

## Timeline

- **2026-08-12:** Synthetic credit probes validated local branching and exposed a
  counterexample to the claim that it always lowers variance.
- **2026-08-12 to 2026-08-13:** QASPER experiments found a shallow-task null and
  an upstream-versus-downstream allocation trade-off.
- **2026-08-14:** MuSiQue capability gates did not support a deeper credit-learning
  comparison with the tested models.
- **2026-08-16:** Typed channel interchange was retained as an auditing tool; its
  routing objective failed a cost-matched CPU gate.
- **2026-08-17:** Campaign-specific code was retired, leaving the current lean
  baseline, replay system, tests, and compact results.
