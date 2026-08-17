# Completed routing investigation

> **Status: closed after a negative cost-matched CPU gate (August 2026).**
> ReDCO-Lite remains a baseline and replay substrate. Typed interchange remains
> an auditing instrument. No sequential, LLM, or Prime routing experiment is
> justified by this investigation.

## Components and status

| Component | Status | Interpretation |
|---|---|---|
| ReDCO-Lite local branch credit | Preserved baseline | Valid local branching and replay, with substantial prior-work overlap |
| Current typed-routing objective | Closed | Current channel contribution does not identify unseen asymmetric reliability |
| Typed interchange audit | Preserved instrument | Measures present dependence, redundancy, synergy, and ambient leakage |
| Information provenance tracing | Preserved infrastructure | Records declared and ambient access paths in rendered observations |
| Reliability-aware information routing | Open future problem | Requires a new hypothesis with explicit reliability information |

## Measurement instrument

For original and alternative declared-artifact values `A` and ambient-context
values `C`, typed interchange evaluates:

```text
r00 = R(A_alt,  C_alt)
r10 = R(A_orig, C_alt)
r01 = R(A_alt,  C_orig)
r11 = R(A_orig, C_orig)
```

It reports total contrast `T`, interaction `I`, and symmetric allocations:

```text
T     = r11 - r00
I     = r11 - r10 - r01 + r00
phi_A = 0.5 * ((r10 - r00) + (r11 - r01))
phi_C = 0.5 * ((r01 - r00) + (r11 - r10))
```

`phi_A + phi_C = T`. These quantities describe contribution under the declared
intervention protocol. They do not, by themselves, predict channel persistence or
reliability after a distribution shift.

The deterministic kill test recovered planted artifact-only, context-only,
redundant, and synergistic effects under two equivalent encodings. This validates
the measurement contract in a symbolic setting; it is not evidence that an LLM
learns robust routing. See
[`results/channel-interchange-kill-test-v1.json`](../results/channel-interchange-kill-test-v1.json).

## Attribution–reliability separation

Let `D` denote all information available to a learner, and let `tau(D)` exchange
the artifact and context channel labels. If the observations are exchangeable,

```text
tau(D) = D,
```

and a route objective `F` is channel-equivariant,

```text
F(tau(D)) = tau(F(D)),
```

then its artifact and context utilities must be equal:

```text
F_A(D) = F_C(D).
```

Otherwise swapping the labels would both change and not change the same output.
Initialization noise, architectural asymmetry, or a hard-coded preference can
break the tie, but that choice is not an informed prediction of future reliability.

The fragile shortcut mode has the symmetric training table:

```text
(r00, r10, r01, r11) = (0, 1, 1, 1)
phi_A = phi_C = 0.5
I = -1
```

The observed training information identifies redundancy. It contains no reason to
prefer the artifact over context before an unseen failure affects only context.
The central conclusion is therefore:

> Current-distribution contribution is identified; future channel reliability is
> not. Causal attribution does not automatically transport across distribution
> shifts.

## Corrected objective and analytical outcomes

The tested typed route objective deliberately replaced ordinary route advantage:

```text
U(ARTIFACT_ONLY) = phi_A
U(CONTEXT_ONLY)  = phi_C
U(BOTH)          = T + I
```

`U(BOTH) = T + I` is a design choice, not a canonical consequence of factorial
attribution. It penalizes current redundancy and amplifies current synergy.
Negative interaction does not imply that redundant publication lacks insurance
value under future independent channel failures.

Because the deterministic policy parameters are separate for each planted mode,
the asymptotic held-out results follow directly:

```text
R_typed      = (1/2 + 1 + 1 + 1) / 4 = 7/8
R_trajectory = (1/3 + 1 + 1 + 1) / 4 = 5/6
R_scalar     = (1/3 + 2/3 + 1 + 1/3) / 4 = 7/12
```

The executable values `0.874997`, `0.833333`, and `0.583333` are finite-update
softmax confirmations of these analytical results, not discoveries from 64
independent stochastic replications.

## Cost correction

The first route probe incorrectly called one condition `typed_interchange`. It did
not use `phi_A`, `phi_C`, or `I`; it directly evaluated the held-out ambient failure
and penalized the normal-to-failure reward loss. It is now classified as
`oracle_heldout_shift_penalty`.

At 1,920 reward calls, the oracle penalty exceeded uniform context corruption by
only `0.0000223`. At 3,840 calls, their mean difference was `0.0000000013`. The
earlier apparent advantage was additional direct supervision about the exact test
shift, not more efficient channel attribution.

The genuine typed objective reached `0.874997` held-out reward using 2,560 calls.
Uniform corruption reached `0.999927` with 1,920 calls. A high-SNR risk-conditioned
corruption baseline reached `0.999999991`; its Beta-distributed proxy has `93.75%`
threshold classification accuracy in either planted class. The latter is
non-oracle only in the narrow sense that it receives a highly informative proxy
rather than the mode label directly. The kill decision does not depend on it,
because uniform corruption already dominates typed allocation.

See [`results/routing-controls-v2.json`](../results/routing-controls-v2.json).

## Reporting scope

- Each planted task mode has independent policy parameters. The experiment tests
  objective information content, not representation learning or mode inference.
- The deterministic typed and scalar arms repeat the same computation across
  seeds. Seeds are meaningful for stochastic augmentation schedules only.
- The scalar mean, minimum, and soft-minimum arms intentionally assign the same
  utility to every route. They are no-route-signal controls, not competitive
  action-specific robust optimizers.
- The shuffled arm is an action-assignment polarity check. Its degradation shows
  that misassigned route utility is harmful, not that the chosen allocation is
  uniquely correct.
- The historical result
  [`results/routing-probe-v1.json`](../results/routing-probe-v1.json) is preserved
  rather than rewritten; this document supplies its corrected interpretation.

## Retrospective audit

Six completed QASPER and MuSiQue reports lacked independently swappable declared
and ambient channel values plus exact pre-consumption replay state. No historical
four-cell attribution was reconstructed post hoc. See
[`results/declared-ambient-channel-audit-v1.json`](../results/declared-ambient-channel-audit-v1.json).

## Closure and future use

Do not extend this formulation with more seeds, a neural policy, alternative
`BOTH` formulas, a sequential rescue benchmark, or model-scale runs. A future
information-routing project is legitimate only if a natural agent problem exposes
observable reliability, persistence, cost, visibility, or failure-history signals
and explains what explicit interchange contributes beyond direct state-conditioned
RL, supervised risk prediction, corruption, or domain randomization.

Typed interchange remains useful for auditing whether a declared artifact is used,
whether a consumer relies on ambient leakage, whether a schema discards information,
whether channels are redundant or synergistic, and whether an expensive artifact
is causally irrelevant. Those diagnostic uses do not require reviving the closed
route objective.

The next research question should be selected independently of the engineering
already present here. See [pivot brief](pivot-brief.md).
