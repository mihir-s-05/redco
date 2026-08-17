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
   ReDCO-Lite, uniform and state-based corruption, identical-cell objectives,
   shuffled credit, and privileged oracle controls.
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
single-decision ReDCO-Lite, context dropout, context corruption, an intervention
penalty then called typed interchange, and fixed artifact-only routing over 64
matched seeds. The intervention penalty reached `0.98613` mean ambient-failure
reward versus `0.96941` for context corruption, but required twice as many reward
evaluations. ReDCO-Lite and trajectory routing were exactly equivalent. See the
historical [`results/routing-probe-v1.json`](../results/routing-probe-v1.json).

### Objective correction and cost controls

The historical `typed_interchange` condition did **not** train from
`phi_A`, `phi_C`, or `I`. It directly evaluated the held-out ambient-failure
condition and penalized the normal-to-failure reward loss. It is now named
`oracle_heldout_shift_penalty`. This was a privileged robustness objective, not
evidence that the decomposition discovered a fragile channel.

The corrected CPU control enumerates all three route actions exactly. Its typed
route objective replaces ordinary task advantage with:

```text
U(ARTIFACT_ONLY) = phi_A
U(CONTEXT_ONLY)  = phi_C
U(BOTH)          = T + I
```

The last term deliberately reuses the interaction sign to penalize redundant
duplication and favor true synergy. It is a modified robustness objective, not an
unbiased estimator of ordinary expected reward. Four cells are evaluated once per
update and reused across all route actions; leave-one-out normalization then acts
on the three enumerated route utilities.

At 1,920 reward calls, the historical oracle penalty exceeded uniform context
corruption by only `0.0000223`. At 3,840 calls, their mean difference was
`0.0000000013`. The compute-efficiency advantage therefore disappears under
matched reward-evaluation budgets.

The genuine typed objective reached `0.874997` held-out reward. Uniform corruption
reached `0.999927`, and a noisy state-risk corruption baseline reached
`0.999999991`, both with 1,920 calls versus 2,560 for typed allocation. Typed
allocation strongly beat a shuffled-credit placebo, but it could not choose
between artifact and context in the shortcut mode because the normal four cells
were exactly symmetric redundancy: `(0, 1, 1, 1)`. The data identify redundancy,
not which redundant channel will survive an unseen failure.

The sequential benchmark and all LLM/Prime routing work remain paused. The current
result is a useful instrumentation result and a negative algorithmic gate, not a
reason to scale the routing objective. See
[`results/routing-controls-v2.json`](../results/routing-controls-v2.json).
