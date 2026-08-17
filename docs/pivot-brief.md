# Pivot brief

## Repository state

There is no active model-scale experiment. The typed-routing investigation is
complete and closed. The MuSiQue campaign is retired from the active tree, and no
Prime run is pending.

The repository now contains separable research assets rather than one monolithic
algorithm claim:

| Asset | Keep for |
|---|---|
| ReDCO-Lite | Controlled local-credit baseline |
| Dependency-sliced replay | Cheap, causally scoped counterfactual execution |
| Information provenance tracing | Declared-versus-ambient observation auditing |
| Typed interchange | Measuring current channel dependence and interaction |
| Compact negative results | Preventing repeated work on falsified hypotheses |

## Closed claims

Do not treat the following as active hypotheses:

- graph-local branching is broadly novel;
- ReDCO-Lite is generally superior to trajectory LOO;
- branching necessarily lowers variance;
- QASPER demonstrates a learning advantage;
- current four-cell contribution identifies unseen channel reliability;
- the historical oracle shift penalty demonstrates typed-credit efficiency;
- a larger model or deeper task is the missing validation for the closed routing
  objective.

## What a genuine pivot requires

A new project should begin with a new estimand, not another modification of local
or channel credit. Before implementation, write down:

1. The failure or capability that matters in a natural agent workflow.
2. The information available to the learner when it must act.
3. The quantity the proposed method estimates.
4. Why terminal reward, ordinary state-conditioned RL, supervised prediction,
   corruption, or domain randomization does not supply the same information as
   cheaply.
5. The cheapest analytical or CPU test that could falsify the idea.
6. A predeclared rule for whether any model or Prime experiment is warranted.

The new hypothesis should survive these questions without relying on ReDCO naming,
existing replay machinery, or sunk implementation effort. Existing components may
be reused only after the scientific need is established.

## Promising problem classes

These are search areas, not approved experiments:

- auditing undeclared ambient dependencies in real agent traces;
- diagnosing whether failures originate in a producer, interface, route, or
  consumer when terminal reward is sparse;
- comparing dependency-sliced replay cost with full perturbation or domain
  randomization;
- studying decisions where persistence, provenance, visibility, or context-window
  risk is observable before routing;
- measuring whether explicit interfaces omit information that remains available
  in prose or conversation history.

Any future reliability-aware routing method must represent contribution and
reliability separately. A useful conceptual factorization is:

```text
route_score(channel, state, history)
    = current_contribution
    * estimated_future_reliability
    - channel_cost
    + multi_channel_interaction
```

Typed interchange can inform the first and last terms. It does not infer the
second from exchangeable observations.

## Operational rule

No background capacity monitoring or speculative GPU work should begin during the
pivot. Prime remains available for bounded experiments after a new hypothesis
passes its analytical and CPU gates. Research scripts that exist only for a
one-time audit should be deleted after their compact result and interpretation are
preserved.

## Historical plan

The retired `redco-implementation-plan.md` is superseded as of August 2026.
Subsequent literature review invalidated its broad novelty positioning, and the
declared-versus-ambient routing objective later failed its cost-matched CPU gate.
The exact historical bytes remain recoverable through
`provenance/history-v1.jsonl`:

```text
Git blob:    6a4521bb093482899f8879289e8223a3a17bd275
Raw SHA-256: 915f8347d554a17c75c3f4438e84354c227fea1d9c210be2f78c83fbefcd7ab3
Bytes:       96,318
```

It is preserved as project history, not normative contributor guidance.
