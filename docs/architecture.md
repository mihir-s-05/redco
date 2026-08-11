# Architecture

The active repository has three layers:

```text
analysis / research gates
          |
          v
replay environment and algorithmic primitives
          |
          v
dependency-free contracts and integrity helpers
```

## Ownership

| Area | Owner | Responsibility |
|---|---|---|
| Contracts | `redco.contracts` | Canonical JSON, event addresses, seeds, and research values |
| Integrity | `redco.integrity` | Raw SHA-256 and strict digest validation |
| Algorithms | `redco.algo` | Branching and training calculations |
| Environment | `redco.env` | Artifacts, commands, tracing, caching, and replay |
| Analyses | `redco.analysis` | Small research questions and Gate GB |

Dependencies point inward. Core packages do not import analyses. Importing any
module must be free of network, provider, model, GPU, publication, or filesystem
side effects.

## Design rules

1. Keep one behavior owner.
2. Prefer a direct function and typed value over a framework or registry.
3. Keep files below 1,000 lines; split by scientific responsibility before then.
4. Parse and validate before mutation.
5. Tests explain active scientific behavior, not retired campaign history.
6. Experimental outputs go under ignored paths and are promoted deliberately.
7. Git history is the archive; old protocols do not stay live merely because
   they once mattered.

Historical Stage C/D modules and their exact recovery identities are listed in
`provenance/history-v1.jsonl`.
