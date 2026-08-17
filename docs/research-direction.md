# Research direction

## Baseline

The current algorithm is **ReDCO-Lite**: restore the state before one policy
decision, compare its original action with behavior-policy alternatives, replay
the affected suffix, and replace that decision's trajectory credit with a local
leave-one-out contrast. It remains a useful baseline and implementation substrate.
It is not the project's primary novelty claim.

## Question

The active question is whether a language-agent policy can learn when to rely on:

- a **declared channel**, where information has schema and provenance and becomes
  visible only after an explicit read, query, projection, or serialization; or
- an **ambient channel**, where information automatically enters a later model
  observation through conversation history, stdout, or inherited context.

The intended deployment property is selective routing: prefer declared interfaces
when ambient context is fragile or redundant, while retaining ambient context when
it has genuine incremental value.

## Measurement

For original and alternative declared-artifact values `A` and ambient-context
values `C`, evaluate:

```text
r00 = R(A_alt,  C_alt)
r10 = R(A_orig, C_alt)
r01 = R(A_alt,  C_orig)
r11 = R(A_orig, C_orig)
```

Report the total contrast `T = r11 - r00`, interaction
`I = r11 - r10 - r01 + r00`, and symmetric allocations:

```text
phi_A = 0.5 * ((r10 - r00) + (r11 - r01))
phi_C = 0.5 * ((r01 - r00) + (r11 - r10))
```

`phi_A + phi_C` must equal `T`. Interaction is reported separately and is not
added again. These are typed interchange effects under a declared replay protocol,
not unrestricted natural path-specific causal effects.

## Sequence

1. Audit existing reports without model or provider calls.
2. Run a deterministic four-cell kill test on artifact-only, context-only,
   redundant, and synergistic planted cases.
3. Stop if interventions are invalid, leak across channels, fail to recover the
   planted effects, or change under semantics-preserving representation changes.
4. If the gate passes, introduce one explicit policy action:
   `ARTIFACT_ONLY`, `CONTEXT_ONLY`, or `BOTH`.
5. Compare typed interchange credit against ordinary route-action learning,
   ReDCO-Lite, context dropout, and fixed routing.
6. Use an LLM or Prime only after cheaper policies show an improved
   robustness–utility frontier under a predeclared ambient-channel failure.

Full graph-structure/resource/payload factorization and recursive flow accounting
remain later hypotheses, not part of the first experiment.

## Current status

The retrospective audit inspected six completed QASPER and MuSiQue reports. None
recorded both an independently swappable declared-artifact channel and exact
ambient-context provenance, so no historical channel effect was reconstructed.
The compact negative result is
[`results/declared-ambient-channel-audit-v1.json`](../results/declared-ambient-channel-audit-v1.json).

The deterministic four-cell measurement gate recovered every planted effect,
interaction sign, and exact conservation law under two different raw encodings.
This validates the measurement contract only. Its compact result is
[`results/channel-interchange-kill-test-v1.json`](../results/channel-interchange-kill-test-v1.json).

The first dependency-free route-learning probe compared trajectory routing,
single-decision ReDCO-Lite, context dropout, context corruption, typed interchange,
and fixed artifact-only routing over 64 matched seeds. Typed interchange reached
`0.98613` mean ambient-failure reward versus `0.96941` for context corruption, but
its paired gain of `0.01672` (95% normal interval `[0.01574, 0.01770]`) missed the
predeclared `0.02` threshold and required twice as many logical reward evaluations.
ReDCO-Lite and trajectory routing were exactly equivalent in the one-decision
environment. The current decision is therefore **no LLM/Prime experiment** from
this probe. See [`results/routing-probe-v1.json`](../results/routing-probe-v1.json).
