# ReDCO evidence selection v1

This is the Stage D0 single-paper evidence-selection port. It deliberately
separates three concerns:

- the task and local paper snapshot;
- deterministic diagnostic scoring of exact evidence spans;
- the scientific reward judge, which must be supplied as a separately pinned
  `verifiers.v1` judge configuration.

The built-in exact-span F1 signal has reward weight zero. It is a diagnostic,
not a replacement for the frozen local judge. A live RL configuration without
one positive-weight judge must fail the Stage D0 preregistration audit.

The task uses the pinned built-in RLM harness. The paper is written to
`/workspace/evidence_context.txt`, and the policy uses IPython to search it.
Single-paper evidence selection is only an incumbent learning-path test; it is
not the Stage D credit-assignment experiment.
